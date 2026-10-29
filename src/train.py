import argparse
import glob
import json
import math
import os
import re
import signal
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Optimize PyTorch CUDA caching allocator to use virtual memory expandable segments (prevents fragmentation OOMs)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn.functional as F
try:
    import torch._dynamo
except ImportError:
    torch._dynamo = None
from transformers import AutoTokenizer

from src.model import PulsarConfig, PulsarForCausalLM
from src.optim import PulsarDualOptimizer
from src.packing import PackedBatchLoader, build_doc_index, plan_epoch


def format_duration(seconds: float) -> str:
    """Format seconds into readable hh:mm:ss or mm:ss."""
    if seconds <= 0 or math.isinf(seconds) or math.isnan(seconds):
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


class DualLogger:
    """Tee logger mirroring stdout to both the console and a persistent log file."""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        self.log_file = open(filepath, "a", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def reopen(self):
        try:
            self.log_file.close()
        except Exception:
            pass
        self.log_file = open(self.filepath, "a", encoding="utf-8")


def rollback_telemetry(
    checkpoint_dir: str,
    max_step: int,
    tokens_seen: int | None = None,
    best_val_loss: float | None = None,
) -> dict[str, int]:
    """Roll back telemetry files (metrics.jsonl, metrics.csv, live_status.json, train.log)
    to a specific max_step, pruning any uncommitted records from crashed or aborted runs.
    """
    stats = {
        "pruned_jsonl": 0,
        "kept_jsonl": 0,
        "pruned_csv": 0,
        "kept_csv": 0,
        "pruned_log_lines": 0,
        "rolled_back_live_status": 0,
    }

    # 1. Roll back metrics.jsonl
    jsonl_path = os.path.join(checkpoint_dir, "metrics.jsonl")
    last_kept_metric = None
    if os.path.exists(jsonl_path):
        kept_lines = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    record = json.loads(line_str)
                    if record.get("step", 0) <= max_step:
                        kept_lines.append(line_str + "\n")
                        last_kept_metric = record
                    else:
                        stats["pruned_jsonl"] += 1
                except Exception:
                    kept_lines.append(line_str + "\n")

        stats["kept_jsonl"] = len(kept_lines)
        if stats["pruned_jsonl"] > 0:
            tmp_path = jsonl_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.writelines(kept_lines)
            os.replace(tmp_path, jsonl_path)

    # 2. Roll back metrics.csv
    csv_path = os.path.join(checkpoint_dir, "metrics.csv")
    if os.path.exists(csv_path):
        kept_csv_lines = []
        with open(csv_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx == 0:
                    kept_csv_lines.append(line)
                    continue
                line_str = line.strip()
                if not line_str:
                    continue
                parts = line_str.split(",")
                try:
                    step_val = int(parts[0])
                    if step_val <= max_step:
                        kept_csv_lines.append(line)
                    else:
                        stats["pruned_csv"] += 1
                except Exception:
                    kept_csv_lines.append(line)

        stats["kept_csv"] = len(kept_csv_lines) - 1 if len(kept_csv_lines) > 0 else 0
        if stats["pruned_csv"] > 0:
            tmp_path = csv_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.writelines(kept_csv_lines)
            os.replace(tmp_path, csv_path)

    # 3. Roll back live_status.json
    live_path = os.path.join(checkpoint_dir, "live_status.json")
    if os.path.exists(live_path):
        try:
            with open(live_path, "r", encoding="utf-8") as f:
                live_data = json.load(f)
        except Exception:
            live_data = {}

        if live_data.get("step", 0) > max_step:
            stats["rolled_back_live_status"] = 1
            if last_kept_metric is not None:
                new_live = dict(last_kept_metric)
            else:
                new_live = dict(live_data)
                new_live["step"] = max_step
                if tokens_seen is not None:
                    new_live["tokens_seen"] = tokens_seen
                if best_val_loss is not None and best_val_loss != float("inf"):
                    new_live["best_val_loss"] = round(best_val_loss, 4)

            new_live["status"] = "resumed"
            tmp_path = live_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(new_live, f, indent=2)
            os.replace(tmp_path, live_path)

    # 4. Roll back train.log
    log_path = os.path.join(checkpoint_dir, "train.log")
    if os.path.exists(log_path):
        if isinstance(sys.stdout, DualLogger):
            sys.stdout.flush()
            try:
                sys.stdout.log_file.close()
            except Exception:
                pass

        step_pattern = re.compile(r"^\[Step\s+(\d+)/")
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            log_lines = f.readlines()

        last_valid_step_idx = -1
        for i, line in enumerate(log_lines):
            m = step_pattern.match(line)
            if m and int(m.group(1)) <= max_step:
                last_valid_step_idx = i

        cut_idx = len(log_lines)
        if max_step == 0:
            for i, line in enumerate(log_lines):
                m = step_pattern.match(line)
                if m and int(m.group(1)) > 0:
                    cut_idx = i
                    break
        elif last_valid_step_idx != -1:
            for i in range(last_valid_step_idx + 1, len(log_lines)):
                m = step_pattern.match(log_lines[i])
                if m and int(m.group(1)) > max_step:
                    cut_idx = i
                    break

        if cut_idx < len(log_lines):
            stats["pruned_log_lines"] = len(log_lines) - cut_idx
            tmp_path = log_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.writelines(log_lines[:cut_idx])
            os.replace(tmp_path, log_path)

        if isinstance(sys.stdout, DualLogger):
            sys.stdout.reopen()

    if any(stats[k] > 0 for k in ("pruned_jsonl", "pruned_csv", "pruned_log_lines", "rolled_back_live_status")):
        print(
            f"[Pulsar] Telemetry rollback: pruned uncommitted steps > {max_step} "
            f"(metrics.jsonl: {stats['pruned_jsonl']} entries, "
            f"metrics.csv: {stats['pruned_csv']} rows, "
            f"train.log: {stats['pruned_log_lines']} lines). "
            f"live_status synchronized to step {max_step}."
        )

    return stats


def compute_param_group_grad_norm(optimizer_sub) -> float:
    """Compute L2 gradient norm over parameters managed by a sub-optimizer."""
    if optimizer_sub is None:
        return 0.0
    total_sq = 0.0
    for group in optimizer_sub.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                total_sq += p.grad.detach().float().pow(2).sum().item()
    return math.sqrt(total_sq)


def get_lr_factor(
    step: int,
    warmup_steps: int,
    max_steps: int,
    schedule: str = "wsd",
    cooldown_steps: int = 0,
    min_ratio: float = 0.01,
) -> float:
    """Calculate learning rate multiplier for WSD (Warmup-Stable-Decay) or Cosine schedule."""
    if warmup_steps > 0 and step <= warmup_steps:
        return float(step) / float(max(1, warmup_steps))

    if schedule == "cosine":
        if step >= max_steps:
            return min_ratio
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * coeff

    # Default: WSD schedule
    if cooldown_steps > 0:
        cooldown_start = max(warmup_steps, max_steps - cooldown_steps)
        if step >= cooldown_start:
            progress = float(step - cooldown_start) / float(max(1, max_steps - cooldown_start))
            progress = min(1.0, max(0.0, progress))
            coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * coeff

    # Stable phase
    return 1.0


def update_optimizer_lrs(
    optimizer: PulsarDualOptimizer,
    factor: float,
    base_muon: float,
    base_adam: float,
):
    """Scale group learning rates across all three sub-optimizers.

    opt_adam_sr holds tok_emb/lm_head in bf16 (~17% of the model). It was omitted here,
    so those weights ran at 100% peak LR from step 1 with no warmup and never decayed
    during cooldown.
    """
    for g in optimizer.opt_muon.param_groups:
        g["lr"] = base_muon * factor
    for g in optimizer.opt_adam_sr.param_groups:
        g["lr"] = base_adam * factor
    for g in optimizer.opt_adam.param_groups:
        # The router group carries an explicit reduced multiplier so router drift stays
        # slower than the balancing bias can correct.
        mult = g.get("lr_mult", 1.0)
        g["lr"] = base_adam * mult * factor


class StreamingPinnedBatchLoader:
    """Preallocates tiny pinned CPU staging buffers for non-blocking asynchronous DMA transfers."""
    def __init__(
        self,
        data: np.memmap | np.ndarray | list[np.memmap | np.ndarray],
        micro_batch_size: int,
        seq_len: int,
        device: str = "cuda",
    ):
        if isinstance(data, (list, tuple)):
            self.shards = list(data)
        else:
            self.shards = [data]
        self.micro_batch_size = micro_batch_size
        self.seq_len = seq_len
        self.device = device
        self.total_tokens = sum(len(s) for s in self.shards)
        self.x_buf = torch.empty((micro_batch_size, seq_len), dtype=torch.long, pin_memory=(device == "cuda"))
        self.y_buf = torch.empty((micro_batch_size, seq_len), dtype=torch.long, pin_memory=(device == "cuda"))

    def get_batch(self, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.RandomState(seed)
        for i in range(self.micro_batch_size):
            shard_idx = rng.randint(0, len(self.shards))
            shard = self.shards[shard_idx]
            max_idx = len(shard) - self.seq_len - 1
            idx = rng.randint(0, max_idx)
            chunk = shard[idx : idx + self.seq_len + 1].astype(np.int64)
            self.x_buf[i].copy_(torch.from_numpy(chunk[:-1]))
            self.y_buf[i].copy_(torch.from_numpy(chunk[1:]))

        if self.device == "cuda":
            return (
                self.x_buf.to(self.device, non_blocking=True),
                self.y_buf.to(self.device, non_blocking=True),
            )
        return self.x_buf.clone(), self.y_buf.clone()


def generate_canary(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    device: str = "cuda",
    repetition_penalty: float = 1.15,
) -> str:
    """Autoregressively decode tokens using full recurrent depth until EOS or end token."""
    was_training = model.training
    model.eval()
    raw_model = getattr(model, "_orig_mod", model)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Dynamic end-of-sequence termination tokens
    stop_token_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_token_ids.add(tokenizer.eos_token_id)
    for tok_str in ["<|im_end|>", "<|endoftext|>", "</s>"]:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok_str)
            if tid is not None and tid != getattr(tokenizer, "unk_token_id", None):
                stop_token_ids.add(tid)
        except Exception:
            pass

    compiled_blocks = list(raw_model.core)
    for i, blk in enumerate(compiled_blocks):
        if hasattr(blk, "_orig_mod"):
            raw_model.core[i] = blk._orig_mod

    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                context_window = input_ids[:, -1024:]
                logits, _, _ = raw_model(context_window, labels=None)
                next_token_logits = logits[:, -1, :].clone()

                if repetition_penalty != 1.0:
                    seen_tokens = set(input_ids[0].tolist())
                    for tok_id in seen_tokens:
                        val = next_token_logits[0, tok_id]
                        if val > 0:
                            next_token_logits[0, tok_id] = val / repetition_penalty
                        else:
                            next_token_logits[0, tok_id] = val * repetition_penalty

                scaled_logits = next_token_logits / max(1e-4, temperature)
                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)

                if next_token.item() in stop_token_ids:
                    break
    finally:
        for i, blk in enumerate(compiled_blocks):
            raw_model.core[i] = blk

    if was_training:
        model.train()
    return tokenizer.decode(input_ids[0], skip_special_tokens=False)


def evaluate_loss(
    model: torch.nn.Module,
    val_loader: StreamingPinnedBatchLoader,
    eval_iters: int,
    num_pulses: int,
) -> dict:
    """Validate the model, returning the FINAL-pulse CE and the per-pulse breakdown.

    The old version returned the model's total training loss: a weighted sum of three
    pulse cross-entropies (0.15/0.25/0.60) plus z-loss, with intermediate pulses scored
    on a 25% token subsample at different offsets per pulse. That number is not the
    final model's loss and per-pulse values were computed on different tokens.

    Now every pulse is scored on 100% of the tokens (``eval_full_pulses``), and
    ``val_ce_final`` - the pulse-3 CE - is what checkpoint selection keys on.
    """
    was_training = model.training
    model.eval()
    # PackedBatchLoader walks a sequential cursor and ignores the seed, so without a
    # reset every eval scored a DIFFERENT slice of windows and val_ce_final was not
    # comparable across steps (or with best.pt's stored value).
    if hasattr(val_loader, "reset"):
        val_loader.reset()
    totals = []
    pulse_accum = [[] for _ in range(num_pulses)]
    with torch.no_grad():
        for i in range(eval_iters):
            seed_val = 999983 + i * 17
            x, y = val_loader.get_batch(seed_val)
            # Padded positions must not enter routing statistics, same as in training.
            rmask = val_loader.routing_mask(x) if hasattr(val_loader, "routing_mask") else None
            _, _, metrics = model(
                x, labels=y, num_pulses=num_pulses,
                subsample_offset=0, eval_full_pulses=True, routing_mask=rmask,
            )
            # Per-pulse CEs are the mean CE over all tokens for each pulse.
            step_pulses = metrics.get("pulse_losses", [])
            vals = [float(v) if not isinstance(v, torch.Tensor) else v.item() for v in step_pulses]
            for p_idx in range(min(num_pulses, len(vals))):
                pulse_accum[p_idx].append(vals[p_idx])
            if vals:
                totals.append(sum(vals) / len(vals))  # unweighted mean for reference

    if was_training:
        model.train()

    per_pulse = [float(np.mean(v)) if v else float("nan") for v in pulse_accum]
    val_ce_final = per_pulse[-1] if per_pulse else float("nan")
    return {
        "val_ce_final": val_ce_final,
        "val_ppl": math.exp(min(val_ce_final, 20.0)) if val_ce_final == val_ce_final else float("nan"),
        "per_pulse": per_pulse,
        # Positive means later pulses help: the owner's main signal on whether looping pays.
        "latent_gain": (per_pulse[0] - per_pulse[-1]) if len(per_pulse) > 1 else 0.0,
        "mean_pulse_ce": float(np.mean(per_pulse)) if per_pulse else float("nan"),
    }


def save_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: PulsarDualOptimizer,
    cfg: PulsarConfig,
    global_step: int,
    tokens_seen: int,
    best_val_loss: float,
    num_pulses: int = 3,
):
    """Save full model and optimizer states atomically with model summary."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    raw_model = getattr(model, "_orig_mod", model)
    model_summary = raw_model.get_model_summary(num_pulses) if hasattr(raw_model, "get_model_summary") else {}
    opt_state = optimizer.state_dict() if hasattr(optimizer, "state_dict") else {}
    ckpt = {
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val_loss,
        "config": cfg.to_dict() if hasattr(cfg, "to_dict") else cfg,
        "model_summary": model_summary,
        "model": raw_model.state_dict(),
        # Computed once: the three sub-optimizer dicts below are taken FROM it rather
        # than re-derived, which previously cloned every _fp32_state block a second
        # time and doubled that part of the checkpoint.
        "optimizer": opt_state,
        "opt_muon": opt_state.get("opt_muon"),
        "opt_adam": opt_state.get("opt_adam"),
        "opt_adam_sr": opt_state.get("opt_adam_sr"),
    }
    tmp_path = filepath + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, filepath)

    # Persist config.json in checkpoint directory for web monitor and external tools
    cfg_dir = os.path.dirname(filepath) or "."
    cfg_file = os.path.join(cfg_dir, "config.json")
    if not os.path.exists(cfg_file):
        try:
            cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else cfg
            with open(cfg_file + ".tmp", "w", encoding="utf-8") as f:
                json.dump(cfg_dict, f, indent=2)
            os.replace(cfg_file + ".tmp", cfg_file)
        except Exception:
            pass


def train():
    found_train = sorted(glob.glob("data/train_shard*.bin"))
    found_val = sorted(glob.glob("data/val_shard*.bin"))
    default_train_str = ",".join(found_train) if found_train else "data/train_shard0.bin"
    default_val_str = ",".join(found_val) if found_val else "data/val_shard0.bin"

    parser = argparse.ArgumentParser(description="Pulsar-MoE Recurrent Pretraining Engine")
    parser.add_argument("--train-bin", type=str, default=default_train_str, help="Path to train shard(s), comma-separated")
    parser.add_argument("--val-bin", type=str, default=default_val_str, help="Path to val shard(s), comma-separated")
    parser.add_argument("--tokenizer", type=str, default="data/pulsar_tokenizer", help="Tokenizer model path or HF name")
    parser.add_argument("--global-batch-tokens", type=int, default=131_072, help="Fixed global batch tokens (64 seqs x 2048)")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence context length")
    parser.add_argument("--micro-batch-size", type=int, default=3, help="Micro batch size per forward pass (MBS=3: 6.30 GB peak VRAM on 8GB GPU)")
    parser.add_argument("--max-steps", type=int, default=10000, help="Total curriculum training steps")
    parser.add_argument("--target-step", type=int, default=None, help="Target step to pause this run session")
    parser.add_argument("--lr-schedule", type=str, choices=["wsd", "cosine"], default="wsd", help="Learning rate schedule")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Linear warmup steps")
    parser.add_argument("--cooldown-steps", type=int, default=0, help="Cooldown steps for WSD decay")
    parser.add_argument("--min-lr-ratio", type=float, default=0.01, help="Minimum LR ratio during cooldown (docs specify 1 percent)")
    parser.add_argument("--eval-every", type=int, default=25, help="Steps between validation and canary generation")
    parser.add_argument("--eval-iters", type=int, default=10, help="Batches evaluated during validation")
    parser.add_argument("--canary-tokens", type=int, default=64, help="Maximum new tokens generated per canary prompt during evaluation")
    parser.add_argument(
        "--router-diag", action=argparse.BooleanOptionalAction, default=True,
        help="Sample router diagnostics (shared-mean fraction, bias override rate) on one "
             "micro-batch per log window. No behaviour change, no extra steady-state sync.",
    )
    parser.add_argument("--save-every", type=int, default=25, help="Steps between saving latest.pt")
    parser.add_argument("--no-save", action="store_true",
                        help="Never write checkpoints (smoke tests). A full checkpoint is "
                             "~1.3 GB; writing one per short test run fills small volumes.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--output-ckpt", type=str, default=None, help="Explicit filepath to save output checkpoints")
    parser.add_argument("--resume", nargs="?", const="latest", default=None, help="Resume training from checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--num-pulses", type=int, default=3, help="Number of recurrent pulses K")
    parser.add_argument("--muon-lr", type=float, default=0.02, help="Muon optimizer base LR for 2D matrices")
    parser.add_argument("--adam-lr", type=float, default=6e-4, help="AdamW base LR for 1D norms and embeddings")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Muon weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient norm clip value")
    parser.add_argument("--bias-lr", type=float, default=0.01,
                        help="MoE balancing-bias step. 0.001 was 10x too slow to track the "
                             "routing drift seen at step 55+.")
    parser.add_argument("--per-pulse-bias", action=argparse.BooleanOptionalAction, default=None,
                        help="Per-pulse MoE balancing bias (default ON). One global bias is diluted "
                             "by whichever pulse needs no correction. Use --no-per-pulse-bias to "
                             "disable. Old [E] checkpoints migrate by broadcasting.")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Recompute core blocks in backward instead of retaining activations. "
                             "Needed for long context (T >= 8k); trades ~30% speed for large savings.")
    parser.add_argument("--sampler", type=str, choices=["flat", "bestfit"], default="flat",
                        help="flat: fixed-stride windows (Stage 1). bestfit: EOS-aligned packing, "
                             "so documents that fit are presented intact (Stage 2+).")
    parser.add_argument("--moe-capacity-factor", type=float, default=None,
                        help="Override MoE capacity headroom (default 1.12). Raising it mid-run is "
                             "safe: capacity changes no checkpoint format.")
    parser.add_argument("--drop-warn-pct", type=float, default=2.0,
                        help="Warn when overall MoE drop rate exceeds this percentage")
    parser.add_argument("--no-stochastic-rounding", action="store_true",
                        help="Disable stochastic rounding for bf16 weights (default: on). "
                             "Without SR, sub-ulp updates and weight decay round away.")
    parser.add_argument("--muon-momentum-fp32", action="store_true",
                        help="Keep Muon momentum in FP32 (default: bf16, matching the weight)")
    parser.add_argument("--compile", action="store_true", default=False, help="Enable torch.compile")
    parser.add_argument("--thermal-cooldown-every", type=int, default=0, help="Cooldown pause frequency in steps")
    parser.add_argument("--thermal-cooldown-seconds", type=int, default=100, help="Duration of thermal cooldown pause")
    parser.add_argument("--reset-dataset", action="store_true", default=False,
                        help="Reset dataset window cursor to 0 upon resume (crucial when switching shards or transitioning curriculum stages, while preserving model weights, step count, and optimizer moments).")
    parser.add_argument("--rollback-telemetry-to-step", type=int, default=None,
                        help="Prune telemetry files (metrics.jsonl, metrics.csv, live_status.json, train.log) "
                             "back to a specific step without running training.")
    args = parser.parse_args()

    if args.rollback_telemetry_to_step is not None:
        stats = rollback_telemetry(args.checkpoint_dir, args.rollback_telemetry_to_step)
        print(f"[Pulsar] Standalone rollback to step {args.rollback_telemetry_to_step} completed: {stats}")
        return

    # Logging setup
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(args.checkpoint_dir, "train.log")
    sys.stdout = DualLogger(log_path)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print("=======================================================")
    print("      PULSAR-MoE: RECURRENT SPARSE MIXTURE-OF-EXPERTS  ")
    print("=======================================================")
    print(f"Device:              {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # Parse explicit training and validation paths (comma-separated or single file)
    train_paths = [p.strip() for p in args.train_bin.split(",") if p.strip()]
    for p in train_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Training shard not found: {p}")

    val_paths = [p.strip() for p in args.val_bin.split(",") if p.strip()]
    for p in val_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Validation shard not found: {p}")

    train_shards = []
    for p in train_paths:
        try:
            train_shards.append(np.memmap(p, dtype=np.uint16, mode="r"))
        except Exception as e:
            train_shards.append(np.fromfile(p, dtype=np.uint16))

    val_shards = []
    for p in val_paths:
        try:
            val_shards.append(np.memmap(p, dtype=np.uint16, mode="r"))
        except Exception as e:
            val_shards.append(np.fromfile(p, dtype=np.uint16))

    if args.sampler == "bestfit":
        # EOS-aligned packing: whole documents fill each window; leftover space comes
        # only from documents longer than seq_len (which can never be intact anyway).
        # See docs/CURRICULUM.md for the measured intactness and the filler-share caveat.
        _t0 = time.time()
        train_index = build_doc_index(train_shards)
        train_plan = plan_epoch(
            train_index.length.astype(np.int64), args.seq_len,
            seed=args.seed, epoch=0, strategy="bestfit",
        )
        train_loader = PackedBatchLoader(
            train_shards, train_index, train_plan,
            micro_batch_size=args.micro_batch_size, device=device,
        )
        pad_frac = train_plan.pad_tokens / max(1, train_plan.capacity_tokens)
        print(f"[Pulsar] Sampler: bestfit | {len(train_plan):,} windows | "
              f"padding {pad_frac*100:.3f}% | built in {time.time()-_t0:.1f}s")
        val_index = build_doc_index(val_shards)
        val_plan = plan_epoch(
            val_index.length.astype(np.int64), args.seq_len,
            seed=args.seed + 1, epoch=0, strategy="bestfit",
        )
        val_loader = PackedBatchLoader(
            val_shards, val_index, val_plan,
            micro_batch_size=args.micro_batch_size, device=device,
        )
        train_loader.total_tokens = int(train_index.length.sum())
        val_loader.total_tokens = int(val_index.length.sum())
    else:
        train_loader = StreamingPinnedBatchLoader(train_shards, args.micro_batch_size, args.seq_len, device)
        val_loader = StreamingPinnedBatchLoader(val_shards, args.micro_batch_size, args.seq_len, device)
    print(f"[Pulsar] Total Train Dataset: {train_loader.total_tokens:,} tokens across {len(train_shards)} shards")
    print(f"[Pulsar] Total Val Dataset:   {val_loader.total_tokens:,} tokens across {len(val_shards)} shards")

    # Calculate invariant hardware-agnostic batch dynamics
    tokens_per_micro = args.micro_batch_size * args.seq_len
    grad_accum = max(1, (args.global_batch_tokens + tokens_per_micro - 1) // tokens_per_micro)
    effective_global_batch_tokens = grad_accum * tokens_per_micro

    print(f"\n--- Batch Dynamics & Invariant Geometry ---")
    print(f"Global Batch Target: {args.global_batch_tokens:,} tokens")
    print(f"Micro Batch Size:    {args.micro_batch_size} seqs x {args.seq_len} tokens = {tokens_per_micro:,} tokens")
    print(f"Grad Accum Steps:    {grad_accum}")
    print(f"Effective Step:      {effective_global_batch_tokens:,} tokens / optimizer step")
    print(f"Recurrent Pulses:    K = {args.num_pulses}")

    # Initialize Tokenizer for canary prompt decoding
    print(f"[Pulsar] Loading tokenizer '{args.tokenizer}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    assert len(tokenizer) == 32768, f"Expected 32768 vocab, got {len(tokenizer)}"
    print(f"[Pulsar] Verified tokenizer vocabulary size: {len(tokenizer):,} tokens")

    # Initialize Model
    cfg = PulsarConfig(
        max_seq_len=32768,
        base_seq_len=1024,
        default_pulses=args.num_pulses,
    )
    if args.moe_capacity_factor is not None:
        cfg.moe_capacity_factor = args.moe_capacity_factor
    # None means "use the config default" (True); the flag only overrides when given.
    if args.per_pulse_bias is not None:
        cfg.per_pulse_bias = args.per_pulse_bias
    print(f"[Pulsar] Per-pulse balancing bias: {'ON' if cfg.per_pulse_bias else 'OFF'}")
    if args.no_save:
        print("[Pulsar] Checkpoint saving: DISABLED (--no-save)")
    print(f"[Pulsar] MoE capacity factor: {cfg.moe_capacity_factor} "
          f"(align {cfg.moe_capacity_align}) | drop warning above {args.drop_warn_pct}%")
    model = PulsarForCausalLM(cfg).to(device=device)
    # Single source of truth for mixed precision (WI-01). The previous implementation
    # cast the whole model to bf16 and then re-cast a name-based subset to FP32, which
    # missed pulse_res_scale / routed_scale / attn_gate / pulse_step_emb (left frozen in
    # bf16) and cast every buffer, including the MoE expert counters and RoPE caches.
    model.to_training_precision(torch.bfloat16)
    if args.grad_checkpoint:
        model.grad_checkpoint = True
        print("[Pulsar] Gradient checkpointing: ON")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Pulsar] Pulsar-MoE Model Initialized: {total_params:,} parameters ({total_params/1e6:.2f}M)")

    # Atomically write config.json so the web server and tools have authoritative config from step 0
    init_cfg_file = os.path.join(args.checkpoint_dir, "config.json")
    try:
        with open(init_cfg_file + ".tmp", "w", encoding="utf-8") as f:
            json.dump(cfg.to_dict(), f, indent=2)
        os.replace(init_cfg_file + ".tmp", init_cfg_file)
    except Exception:
        pass

    # Initialize Optimizer
    optimizer = PulsarDualOptimizer(
        model,
        muon_lr=args.muon_lr,
        adam_lr=args.adam_lr,
        weight_decay=args.weight_decay,
        stochastic_rounding=not args.no_stochastic_rounding,
        seed=args.seed,
        momentum_fp32=args.muon_momentum_fp32,
    )
    print(f"[Pulsar] Stochastic rounding: {'ON' if not args.no_stochastic_rounding else 'OFF'}"
          f" | Muon momentum: {'FP32' if args.muon_momentum_fp32 else 'bf16'}")

    # Setup Checkpointing and Resumption
    start_step = 0
    tokens_seen = 0
    best_val_loss = float("inf")
    latest_ckpt_path = os.path.join(args.checkpoint_dir, "latest.pt")
    if args.output_ckpt is not None:
        latest_ckpt_path = args.output_ckpt

    if args.resume is not None:
        resume_file = (
            latest_ckpt_path
            if args.resume == "latest"
            else args.resume
        )
        if os.path.exists(resume_file):
            print(f"\n[Pulsar] Resuming from checkpoint: {resume_file}")
            # Load to CPU first to prevent a 3.4 GB VRAM allocation spike on consumer GPUs
            ckpt = torch.load(resume_file, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"], strict=False)

            if "opt_muon" in ckpt and ckpt["opt_muon"] is not None:
                optimizer.opt_muon.load_state_dict(ckpt["opt_muon"])
            if "opt_adam" in ckpt and ckpt["opt_adam"] is not None:
                optimizer.opt_adam.load_state_dict(ckpt["opt_adam"])
            if "opt_adam_sr" in ckpt and ckpt["opt_adam_sr"] is not None:
                optimizer.opt_adam_sr.load_state_dict(ckpt["opt_adam_sr"])

            start_step = ckpt.get("global_step", 0)
            tokens_seen = ckpt.get("tokens_seen", 0)
            best_val_loss = ckpt.get("best_val_loss", float("inf"))

            # Free checkpoint CPU memory and flush any lingering GPU caches
            del ckpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Roll back any uncommitted telemetry lines (from a prior crashed or preempted run)
            # that were logged past start_step, ensuring strict monotonic alignment.
            rollback_telemetry(
                args.checkpoint_dir,
                start_step,
                tokens_seen=tokens_seen,
                best_val_loss=best_val_loss,
            )

            # A packed loaders' position is its cursor, not the RNG seed: without this
            # a resumed run restarts at window 0 and retrains the opening data.
            if isinstance(train_loader, PackedBatchLoader):
                if args.reset_dataset:
                    train_loader._cursor = 0
                    print("[Pulsar] Packed sampler cursor reset to window 0 (new curriculum stage/shards).")
                else:
                    consumed = start_step * grad_accum * args.micro_batch_size
                    train_loader._cursor = consumed
                    print(f"[Pulsar] Packed sampler resumed at window {consumed:,} "
                          f"of {len(train_loader.plan.windows):,}")
            print(f"[Pulsar] Resumed successfully at global_step {start_step} ({tokens_seen:,} tokens, best_val_loss={best_val_loss:.4f})")
        else:
            print(f"[Pulsar] Notice: Checkpoint '{resume_file}' not found. Starting from Step 0.")

    if args.compile and hasattr(torch, "compile"):
        print("[Pulsar] Compiling 8 physical recurrent core blocks with torch.compile...")
        if torch._dynamo is not None:
            torch._dynamo.config.recompile_limit = 64
            torch._dynamo.config.suppress_errors = True
            if hasattr(torch._dynamo.config, "allow_unspec_int_on_nn_module"):
                torch._dynamo.config.allow_unspec_int_on_nn_module = True
        raw_model = getattr(model, "_orig_mod", model)
        for i in range(len(raw_model.core)):
            raw_model.core[i] = torch.compile(raw_model.core[i], dynamic=False)

    # Graceful shutdown handler
    interrupted = False
    def signal_handler(sig, frame):
        nonlocal interrupted
        print("\n[Pulsar] Signal received. Finishing current step and saving checkpoint...")
        interrupted = True
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    canary_prompts = [
        {
            "id": 1,
            "tag": "Python / Algorithms",
            "category": "Python",
            "prompt": "def quicksort(arr: list[int]) -> list[int]:\n    if len(arr) <= 1:\n        return arr\n    ",
        },
        {
            "id": 2,
            "tag": "Agentic MCP Tool Call",
            "category": "Tool Calling",
            "prompt": "<|im_start|>user\nRename scan_01.pdf to 2026_Invoice.pdf in /home/user/docs<|im_end|>\n<|im_start|>assistant\n<|tool_call|>",
        },
        {
            "id": 3,
            "tag": "Math / Latent CoT",
            "category": "Math",
            "prompt": "<|im_start|>user\nIf a train travels 120 km in 2 hours, what is its speed in km/h?<|im_end|>\n<|im_start|>assistant\n<|think|>\n",
        },
        {
            "id": 4,
            "tag": "Bash / Linux CLI",
            "category": "CLI",
            "prompt": "<|im_start|>user\nWrite a bash command to find all .pdf files in ~/Downloads modified in the last 7 days.<|im_end|>\n<|im_start|>assistant\n",
        },
        {
            "id": 5,
            "tag": "STEM / World Knowledge",
            "category": "STEM",
            "prompt": "The primary function of mitochondria in a biological cell is to",
        },
    ]

    print(f"\n[Pulsar] Starting training: Steps {start_step + 1} -> {args.max_steps}...")
    model.train()
    t_start = time.time()
    t_step_prev = time.time()
    prev_train_loss = None
    last_val_loss = best_val_loss if best_val_loss != float("inf") else None
    last_val_ppl = math.exp(min(last_val_loss, 20.0)) if last_val_loss is not None else None
    last_val_pulse_losses = None
    last_live_canaries = []
    last_latent_gain = None
    last_val_pulse_ppls = None
    latest_canaries = []

    stop_step = args.target_step if args.target_step is not None else args.max_steps
    print(f"\n[Pulsar] Starting training: Steps {start_step + 1} -> {stop_step} (Curriculum Horizon: {args.max_steps})...\n")

    # Authoritative Run Configuration (SSOT for telemetry monitor)
    session_config = {
        "start_step": start_step,
        "target_step": stop_step,
        "target_steps": stop_step,
        "max_steps": args.max_steps,
        "curriculum_steps": args.max_steps,
        "warmup_steps": args.warmup_steps,
        "cooldown_steps": args.cooldown_steps,
        "global_batch_tokens": effective_global_batch_tokens,
        "seq_len": args.seq_len,
        "num_pulses": args.num_pulses,
        "lr_schedule": args.lr_schedule,
        "train_shards": [os.path.basename(p) for p in train_paths],
        "val_shards": [os.path.basename(p) for p in val_paths],
        "num_train_shards": len(train_shards),
        "train_dataset_tokens": train_loader.total_tokens,
    }
    run_cfg_path = os.path.join(args.checkpoint_dir, "run_config.json")
    try:
        with open(run_cfg_path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(session_config, f, indent=2)
        os.replace(run_cfg_path + ".tmp", run_cfg_path)
    except Exception as e:
        print(f"[Pulsar] Warning: could not write {run_cfg_path}: {e}")

    for step in range(start_step + 1, stop_step + 1):
        # Update learning rates with WSD or Cosine schedule
        lr_factor = get_lr_factor(
            step,
            args.warmup_steps,
            args.max_steps,
            schedule=args.lr_schedule,
            cooldown_steps=args.cooldown_steps,
            min_ratio=args.min_lr_ratio,
        )
        update_optimizer_lrs(optimizer, lr_factor, args.muon_lr, args.adam_lr)
        current_muon_lr = args.muon_lr * lr_factor
        current_adam_lr = args.adam_lr * lr_factor

        optimizer.zero_grad(set_to_none=True)
        step_metrics = {}
        t_fwd_total = 0.0
        t_bwd_total = 0.0

        # CUDA event timing for zero-overhead nanosecond hardware profiling
        use_cuda_events = torch.cuda.is_available()
        if use_cuda_events:
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_fwd = torch.cuda.Event(enable_timing=True)
            ev_bwd = torch.cuda.Event(enable_timing=True)
            ev_opt = torch.cuda.Event(enable_timing=True)
            ev_start.record()

        # Accumulate gradients across micro-batches
        accum_loss_tensor = torch.zeros((), device=device, dtype=torch.float32)
        metric_accum: dict = {}
        # Router diagnostics are sampled, not continuous: one micro-batch out of the
        # accumulation group, every 10 steps. There is no --log-every flag (an earlier
        # version referenced args.log_every, which does not exist and crashed the trainer
        # on step 2); metrics.jsonl is written every step.
        is_log_step = (step == 1) or (step % 10 == 0)
        for micro_step in range(grad_accum):
            seed_micro = args.seed + step * 1000 + micro_step
            x, y = train_loader.get_batch(seed_micro)

            t0 = time.time()
            # Packed windows may contain padding; exclude those positions from routing
            # statistics and z-loss. Flat windows are always full, so the mask is None.
            rmask = train_loader.routing_mask(x) if hasattr(train_loader, "routing_mask") else None
            _, loss, metrics = model(
                x, labels=y, num_pulses=args.num_pulses,
                subsample_offset=(step * grad_accum + micro_step), routing_mask=rmask,
                # Sampled, not continuous: one micro-batch per log window keeps the cost
                # at a no-grad matmul per (block, pulse) instead of every step.
                router_diag=(args.router_diag and is_log_step and micro_step == 0),
            )
            t1 = time.time()
            if use_cuda_events and micro_step == grad_accum - 1:
                ev_fwd.record()

            scaled_loss = loss / grad_accum
            scaled_loss.backward()
            t2 = time.time()
            if use_cuda_events and micro_step == grad_accum - 1:
                ev_bwd.record()

            t_fwd_total += (t1 - t0)
            t_bwd_total += (t2 - t1)
            accum_loss_tensor += loss.detach() / grad_accum
            # Average the reported per-pulse metrics across micro-batches. Keeping only
            # the last one made `pulse_losses` describe 1 of 43 micro-batches while
            # train_loss averaged all of them, so the two disagreed on every step
            # (measured mean gap 2.57 over a historical run).
            def _avg(prev, cur, n):
                """Average per-micro-batch values, including lists of scalars."""
                if cur is None:
                    return prev
                if isinstance(cur, (list, tuple)):
                    if prev is None:
                        return [x / n if x is not None else None for x in cur]
                    return [
                        (a + (b / n)) if (a is not None and b is not None) else (a if b is None else b)
                        for a, b in zip(prev, cur)
                    ]
                if prev is None:
                    return cur / n
                return prev + cur / n

            for _k, _v in metrics.items():
                if _k.startswith("router_diag_"):
                    # Only present on micro_step 0, so they must NOT be divided by
                    # grad_accum - that shrank every diagnostic by the accumulation
                    # factor (max_top1_share has an upper bound of 1 and came out 43x
                    # too small).
                    metric_accum[_k] = _v
                else:
                    metric_accum[_k] = _avg(metric_accum.get(_k), _v, grad_accum)
            step_metrics = metric_accum

        tokens_seen += effective_global_batch_tokens

        # Sub-optimizer gradient norms before clipping (fused foreach kernels on GPU without CPU stall)
        m_grads = [p.grad for group in optimizer.opt_muon.param_groups for p in group["params"] if p.grad is not None]
        # Both AdamW sub-optimizers: opt_adam_sr holds the bf16 embedding (~17% of
        # parameters), so omitting it hid most of the Adam gradient norm.
        a_grads = [
            p.grad
            for sub in (optimizer.opt_adam, optimizer.opt_adam_sr)
            for group in sub.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        muon_grad_norm_t = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(m_grads))) if m_grads else None
        adam_grad_norm_t = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(a_grads))) if a_grads else None

        # Global gradient clipping (fused C++ foreach multi-tensor)
        all_trainable = [p for p in model.parameters() if p.grad is not None]
        total_grad_norm_t = torch.nn.utils.clip_grad_norm_(all_trainable, args.grad_clip, foreach=True)

        # Optimizer step
        t_opt_start = time.time()
        optimizer.step()
        raw_model = getattr(model, "_orig_mod", model)
        raw_model.update_expert_biases(bias_lr=args.bias_lr, num_pulses=args.num_pulses)
        t_opt_total = time.time() - t_opt_start
        if use_cuda_events:
            ev_opt.record()
            ev_opt.synchronize()
            gpu_step_ms = ev_start.elapsed_time(ev_opt)
            gpu_fwd_ms = ev_start.elapsed_time(ev_fwd)
            gpu_bwd_ms = ev_fwd.elapsed_time(ev_bwd)
            gpu_opt_ms = ev_bwd.elapsed_time(ev_opt)
        else:
            gpu_step_ms = (t_fwd_total + t_bwd_total + t_opt_total) * 1000.0
            gpu_fwd_ms = t_fwd_total * 1000.0
            gpu_bwd_ms = t_bwd_total * 1000.0
            gpu_opt_ms = t_opt_total * 1000.0

        # Extract scalar metrics after stream synchronization to eliminate CPU-GPU pipeline bubbles
        accum_loss = float(accum_loss_tensor.item())
        muon_grad_norm = float(muon_grad_norm_t.item()) if muon_grad_norm_t is not None else 0.0
        adam_grad_norm = float(adam_grad_norm_t.item()) if adam_grad_norm_t is not None else 0.0
        total_grad_norm = float(total_grad_norm_t.item()) if isinstance(total_grad_norm_t, torch.Tensor) else float(total_grad_norm_t)

        time_per_token_us = (gpu_step_ms * 1000.0) / max(1, effective_global_batch_tokens)
        fwd_pct = (gpu_fwd_ms / max(1e-3, gpu_step_ms)) * 100.0
        bwd_pct = (gpu_bwd_ms / max(1e-3, gpu_step_ms)) * 100.0
        opt_pct = (gpu_opt_ms / max(1e-3, gpu_step_ms)) * 100.0

        # Step timing, speed and hardware utilization
        t_step_now = time.time()
        step_dt = t_step_now - t_step_prev
        t_step_prev = t_step_now
        tok_per_sec = effective_global_batch_tokens / max(1e-4, step_dt)

        # MFU estimation: FLOPs/tok scales with the number of pulses K.
        # The K=3 baseline is 2.05 GFLOPs/tok (6 * 97.15M non-embed active params * 3,
        # rounded to include attention). Scale linearly from that baseline so a 1- or
        # 2-pulse run is not reported as if it did K=3 work.
        flops_per_tok = 2.05e-3 * (args.num_pulses / 3.0)   # GFLOPs per token
        achieved_tflops = tok_per_sec * flops_per_tok

        # Dynamic GPU detection for peak dense BF16 TFLOPS and physical VRAM
        dev_name = torch.cuda.get_device_name(0).upper() if torch.cuda.is_available() else ""
        if "B300" in dev_name or "B200" in dev_name:
            peak_tflops = 2250.0
        elif "H200" in dev_name or "H100" in dev_name:
            peak_tflops = 989.0
        elif "A100" in dev_name:
            peak_tflops = 312.0
        elif "4060" in dev_name:
            peak_tflops = 60.0
        else:
            peak_tflops = 100.0
        mfu_pct = (achieved_tflops / peak_tflops) * 100.0

        # Loss delta, elapsed time & ETA calculation
        loss_delta = (accum_loss - prev_train_loss) if prev_train_loss is not None else 0.0
        prev_train_loss = accum_loss
        train_ppl = math.exp(min(accum_loss, 20.0))

        vram_reserved = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0
        vram_peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0.0

        def to_val(v, default=0.0):
            if v is None:
                return default
            if hasattr(v, "item"):
                return v.item()
            return float(v)

        worst_expert_share_pct = to_val(step_metrics.get("worst_expert_share_pct"))
        mean_block_spread_pct = to_val(step_metrics.get("mean_block_spread_pct"))
        max_block_spread_pct = to_val(step_metrics.get("max_block_spread_pct"))
        step_pulse_losses = [to_val(x) for x in step_metrics.get("pulse_losses", [])]
        if not step_pulse_losses:
            step_pulse_losses = [to_val(step_metrics.get(f"loss_pulse_{i+1}")) for i in range(args.num_pulses)]
        step_pulse_deltas = [to_val(x) for x in step_metrics.get("pulse_deltas", [])]
        if not step_pulse_deltas:
            step_pulse_deltas = [to_val(step_metrics.get(f"delta_pulse_{i+1}")) for i in range(args.num_pulses)]

        pulse_delta = to_val(step_metrics.get("pulse_final_delta"))
        conf = to_val(step_metrics.get("router_confidence"))
        max_bias = to_val(step_metrics.get("router_max_bias"))
        mix_lambda = to_val(step_metrics.get("input_mix_lambda"))
        e_pcts = [to_val(x) for x in step_metrics.get("expert_pcts", [])]
        if not e_pcts:
            e_pcts = [to_val(step_metrics.get(f"expert_pct_{i}")) for i in range(cfg.n_routed_experts)]

        # Telemetry stdout printout
        pct_sign = "+" if loss_delta > 0 else ""
        step_target_label = f"{step:4d}/{args.max_steps}" if args.target_step is None else f"{step:4d}/{stop_step} (Horizon: {args.max_steps})"
        print(
            f"\n[Step {step_target_label}] Loss: {accum_loss:.4f} ({pct_sign}{loss_delta:.4f}) | "
            f"Train PPL: {train_ppl:5.2f} | Tokens: {tokens_seen:,}"
        )
        moe_str = " | ".join(f"E{i}: {pct:4.1f}%" for i, pct in enumerate(e_pcts)) if e_pcts else "Dense FFN"
        pulses_str = " | ".join(
            f"L{i+1}: {step_pulse_losses[i]:.4f} (d{i+1}: {step_pulse_deltas[i]:.2f})"
            for i in range(min(len(step_pulse_losses), len(step_pulse_deltas)))
        )
        # Drop statistics are per-pulse on purpose: drops concentrate on whichever pulse
        # routes most unevenly, so a single global rate would hide the case that matters.
        moe_stats = raw_model.pop_moe_stats() if hasattr(raw_model, "pop_moe_stats") else {}
        # Learned-scale diagnostics (one sync per log step; cheap and worth seeing).
        with torch.no_grad():
            # Only the pulses this run actually uses; slots beyond K stay at init and
            # would otherwise clutter the log with meaningless values.
            K = args.num_pulses
            pulse_res_scale_mean = (
                raw_model.pulse_res_scale.detach().float().mean(dim=1)[:K].tolist()
            )
            routed_scale_mean = torch.stack([
                blk.mlp.routed_scale.detach().float().mean(dim=1)[:K]
                for blk in raw_model.core
            ]).mean(dim=0).tolist()
        drop_str = ""
        if moe_stats.get("per_pulse"):
            rates = [f"P{d['pulse']}:{d['drop_rate']*100:.2f}%" for d in moe_stats["per_pulse"] if d["assigned"] > 0]
            if rates:
                drop_str = f" | Drops: [{' '.join(rates)}] cap {int(moe_stats['capacity'])}"
                overall = moe_stats["drop_rate"] * 100.0
                if overall > args.drop_warn_pct:
                    drop_str += f" [WARN] overall {overall:.2f}% > {args.drop_warn_pct}% -> consider --moe-capacity-factor 1.25"
        print(f"  ↳ MoE Experts: [{moe_str}] | Conf: {conf:4.2f} | MaxBias: {max_bias:5.3f}{drop_str}")
        print(f"  ↳ Pulses: {pulses_str} | λ: {mix_lambda:.3f}")
        print(
            f"  ↳ Gradients: Total: {total_grad_norm:5.2f} (Muon: {muon_grad_norm:4.2f}, Adam: {adam_grad_norm:4.2f}) | LR: [Muon: {current_muon_lr:7.5f}, Adam: {current_adam_lr:.1e}]\n"
            f"  ↳ Profiling: GPU: {gpu_step_ms:5.1f}ms ({tok_per_sec/1e3:4.1f}k tok/s | {time_per_token_us:4.2f}µs/tok) [Fwd: {gpu_fwd_ms:5.1f}ms ({fwd_pct:2.0f}%), Bwd: {gpu_bwd_ms:5.1f}ms ({bwd_pct:2.0f}%), Opt: {gpu_opt_ms:4.1f}ms ({opt_pct:2.0f}%)]\n"
            f"  ↳ Hardware:  VRAM: {vram_peak:4.1f}GB / {total_vram_gb:4.1f}GB ({vram_peak/max(1e-2, total_vram_gb)*100:4.1f}% util, Reserved: {vram_reserved:4.1f}GB) | Compute: {achieved_tflops:5.1f} TF ({mfu_pct:4.1f}% MFU / {peak_tflops:.0f} TF peak)"
        )

        val_loss = None
        val_ppl = None
        val_pulse_losses = None
        val_pulse_ppls = None
        latent_gain = None   # computed only on eval steps; must be reset every step
        # Periodic Evaluation and Canary Generation
        if step % args.eval_every == 0 or step == stop_step or interrupted:
            eval_metrics = evaluate_loss(
                model, val_loader, args.eval_iters, args.num_pulses
            )
            val_loss = eval_metrics["val_ce_final"]      # checkpoint selection metric
            val_ppl = eval_metrics["val_ppl"]
            val_pulse_losses = eval_metrics["per_pulse"]
            val_pulse_ppls = [math.exp(min(pl, 20.0)) for pl in val_pulse_losses]
            latent_gain = eval_metrics["latent_gain"]
            last_val_loss = val_loss
            last_val_ppl = val_ppl
            last_val_pulse_losses = val_pulse_losses
            last_val_pulse_ppls = val_pulse_ppls
            # best.pt keys on the FINAL pulse CE on all tokens, not the weighted sum.
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            print(f"\n--- [Eval @ Step {step}] val_ce_final: {val_loss:.4f} (PPL: {val_ppl:.1f}) | Best: {best_val_loss:.4f} ---")
            for p_idx, pl in enumerate(val_pulse_losses):
                eff_l = (p_idx + 1) * cfg.n_core
                pl_ppl = math.exp(min(pl, 20.0))
                refine_str = f" | latent_gain {latent_gain:+.4f}" if (p_idx == len(val_pulse_losses) - 1 and len(val_pulse_losses) > 1) else ""
                print(f"  ↳ Pulse {p_idx + 1} [{eff_l:2d} eff layers]: CE = {pl:.4f} (PPL: {pl_ppl:.1f}, 100% tokens){refine_str}")
            print("--- Canary Generations Live ---")
            latest_canaries = []
            _canaries_generated_this_step = True
            for canary_meta in canary_prompts:
                canary_idx = canary_meta["id"]
                prompt = canary_meta["prompt"]
                tag = canary_meta["tag"]
                category = canary_meta["category"]
                try:
                    generation = generate_canary(
                        model, tokenizer, prompt, max_new_tokens=args.canary_tokens, temperature=0.7, device=device
                    )
                    generation_clean = generation.strip()
                    print(f"  [{tag}] {generation_clean.replace(chr(10), ' ')[:140]}...")
                    latest_canaries.append({
                        "id": canary_idx,
                        "tag": tag,
                        "category": category,
                        "prompt": prompt,
                        "completion": generation_clean,
                        # Which step produced this sample. Without it the fallback
                        # repeated the previous text silently and a single generation
                        # looked like N consecutive identical results.
                        "generated_at_step": step,
                    })
                except Exception as e:
                    print(f"  [{tag}] Canary error: {e}")
            print("-----------------------------------------------------------------")

            if latest_canaries:
                last_live_canaries = latest_canaries

            if is_best and not args.no_save:
                best_ckpt_path = os.path.join(args.checkpoint_dir, "best.pt")
                if args.output_ckpt is not None:
                    best_ckpt_path = args.output_ckpt
                save_checkpoint(best_ckpt_path, model, optimizer, cfg, step, tokens_seen, best_val_loss, num_pulses=args.num_pulses)
                print(f"[Pulsar] Saved best checkpoint (val_loss={best_val_loss:.4f}) -> {best_ckpt_path}")

        # On non-eval steps these are None; fall back to the last eval so the JSONL
        # record keeps the most recent validation numbers instead of nulls.
        if latent_gain is not None:
            last_latent_gain = round(latent_gain, 4)
        cur_latent_gain = last_latent_gain
        cur_vpl = val_pulse_losses if val_pulse_losses is not None else last_val_pulse_losses
        cur_vppl = val_pulse_ppls if val_pulse_ppls is not None else last_val_pulse_ppls

        raw_model = getattr(model, "_orig_mod", model)
        model_summary = raw_model.get_model_summary(args.num_pulses) if hasattr(raw_model, "get_model_summary") else {}

        metric_record = {
            "step": step,
            "max_steps": args.max_steps,
            "target_step": args.target_step if args.target_step is not None else args.max_steps,
            "target_steps": args.target_step if args.target_step is not None else args.max_steps,
            "tokens_seen": tokens_seen,
            "train_dataset_tokens": train_loader.total_tokens,
            "num_train_shards": len(train_shards),
            "train_shards": [os.path.basename(p) for p in train_paths],
            "global_batch_tokens": effective_global_batch_tokens,
            "micro_batch_size": args.micro_batch_size,
            "seq_len": args.seq_len,
            "num_pulses": args.num_pulses,
            # Schedule shape for the dashboard's phase label. Without these it falls
            # back to hardcoded guesses (a 100-step warmup), which mislabels the phase
            # for any run whose warmup is not 100.
            "grad_clip": args.grad_clip,
            "warmup_steps": args.warmup_steps,
            "cooldown_steps": args.cooldown_steps,
            "lr_schedule": args.lr_schedule,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "model_config": model_summary,
            "train_loss": round(accum_loss, 4),
            "loss_delta": round(loss_delta, 4),
            "train_ppl": round(train_ppl, 2),
            "val_ce_final": round(val_loss, 4) if val_loss is not None else (
                round(last_val_loss, 4) if last_val_loss is not None else None),
            "val_loss": round(val_loss, 4) if val_loss is not None else None,
            "val_ppl": round(val_ppl if val_ppl is not None else last_val_ppl, 2)
            if (val_ppl is not None or last_val_ppl is not None) else None,
            "val_pulse_losses": [round(x, 4) for x in cur_vpl] if cur_vpl is not None else None,
            "val_pulse_ppls": [round(x, 2) for x in cur_vppl] if cur_vppl is not None else None,
            "best_val_loss": round(best_val_loss, 4) if best_val_loss != float("inf") else None,
            "pulse_losses": [round(x, 4) for x in step_pulse_losses],
            "pulse_deltas": [round(x, 4) for x in step_pulse_deltas],
            "pulse_final_delta": round(pulse_delta, 4),
            "grad_norm_total": round(total_grad_norm, 4),
            "grad_norm_muon": round(muon_grad_norm, 4),
            "grad_norm_adam": round(adam_grad_norm, 4),
            "lr_muon": round(current_muon_lr, 6),
            "lr_adam": round(current_adam_lr, 6),
            "tok_per_sec": round(tok_per_sec, 1),
            "mfu_pct": round(mfu_pct, 1),
            "achieved_tflops": round(achieved_tflops, 1),
            "gpu_step_ms": round(gpu_step_ms, 2),
            "gpu_fwd_ms": round(gpu_fwd_ms, 2),
            "gpu_bwd_ms": round(gpu_bwd_ms, 2),
            "gpu_opt_ms": round(gpu_opt_ms, 2),
            "time_per_token_us": round(time_per_token_us, 2),
            "vram_peak": round(vram_peak, 2),
            "vram_reserved": round(vram_reserved, 2),
            "total_vram_gb": round(total_vram_gb, 2),
            "router_confidence": round(conf, 4),
            "router_max_bias": round(max_bias, 4),
            "expert_pcts": [round(x, 2) for x in e_pcts],
            "mix_lambda": round(mix_lambda, 4),
            # MoE drop accounting, per pulse. Drops concentrate on whichever pulse routes
            # most unevenly, so a single global rate hides the case that matters.
            "moe_capacity": int(moe_stats["capacity"]) if moe_stats else None,
            "moe_drop_rate": round(moe_stats["drop_rate"], 6) if moe_stats else None,
            "moe_drop_per_pulse": [
                {"pulse": d["pulse"], "assigned": int(d["assigned"]), "drop_rate": round(d["drop_rate"], 6)}
                for d in moe_stats.get("per_pulse", []) if d["assigned"] > 0
            ] if moe_stats else None,
            # Latent gain: p1 - pK over the whole validation set. Positive means the
            # recurrent pulses help; this is the signal on whether looping pays off.
            "latent_gain": cur_latent_gain,
            # Per-pulse expert load: the aggregate is what hid the imbalance.
            # R-01 step 1: router diagnostics. Absent on non-log steps, which is why
            # they are optional here rather than required keys.
            **{
                _k: round(float(_v), 5)
                for _k, _v in step_metrics.items()
                if _k.startswith("router_diag_")
            },
            "worst_expert_share_pct": round(float(worst_expert_share_pct), 2) if worst_expert_share_pct is not None else None,
            "mean_block_spread_pct": round(float(mean_block_spread_pct), 2) if mean_block_spread_pct is not None else None,
            "max_block_spread_pct": round(float(max_block_spread_pct), 2) if max_block_spread_pct is not None else None,
            "expert_pcts_per_pulse": (
                [[round(float(v), 2) for v in row] for row in step_metrics["expert_pcts_per_pulse"].tolist()]
                if step_metrics.get("expert_pcts_per_pulse") is not None else None
            ),
            # Learned scales - these were provably frozen before the precision fix, so
            # logging them makes a regression immediately visible.
            "pulse_res_scale_mean": [round(float(v), 5) for v in pulse_res_scale_mean],
            "routed_scale_mean": [round(float(v), 5) for v in routed_scale_mean],
            "canaries": latest_canaries if latest_canaries else last_live_canaries,
        }

        # Write to JSONL
        jsonl_file = os.path.join(args.checkpoint_dir, "metrics.jsonl")
        with open(jsonl_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(metric_record) + "\n")

        # Write to CSV
        csv_file = os.path.join(args.checkpoint_dir, "metrics.csv")
        csv_exists = os.path.exists(csv_file)
        e_cols = [f"e{i}" for i in range(cfg.n_routed_experts)]
        e_vals = [round(e_pcts[i], 2) if i < len(e_pcts) else 0.0 for i in range(cfg.n_routed_experts)]
        with open(csv_file, "a", encoding="utf-8") as f:
            if not csv_exists:
                header = [
                    "step", "tokens_seen", "train_loss", "loss_delta", "train_ppl", "val_loss",
                    "pulse_l1", "pulse_l2", "pulse_l3", "delta_h1", "delta_h2", "delta_h3",
                    *e_cols, "router_conf", "max_bias", "mix_lambda",
                    "grad_total", "grad_muon", "grad_adam",
                    "lr_muon", "lr_adam", "tok_per_sec", "mfu_pct", "vram_peak",
                ]
                f.write(",".join(header) + "\n")
            l_row = [round(step_pulse_losses[i], 4) if i < len(step_pulse_losses) else "" for i in range(3)]
            d_row = [round(step_pulse_deltas[i], 4) if i < len(step_pulse_deltas) else "" for i in range(3)]
            row = [
                step, tokens_seen, round(accum_loss, 4), round(loss_delta, 4), round(train_ppl, 2),
                round(val_loss, 4) if val_loss is not None else "",
                *l_row,
                *d_row,
                *e_vals,
                round(conf, 4), round(max_bias, 4), round(mix_lambda, 4),
                round(total_grad_norm, 2), round(muon_grad_norm, 2), round(adam_grad_norm, 2),
                round(current_muon_lr, 6), round(current_adam_lr, 6),
                round(tok_per_sec, 1), round(mfu_pct, 1), round(vram_peak, 2),
            ]
            f.write(",".join(str(x) for x in row) + "\n")

        # Write latest live status
        live_file = os.path.join(args.checkpoint_dir, "live_status.json")
        live_tmp = live_file + ".tmp"
        live_payload = {
            **metric_record,
            "val_loss": round(val_loss if val_loss is not None else last_val_loss, 4) if (val_loss is not None or last_val_loss is not None) else None,
            "val_ppl": round(val_ppl if val_ppl is not None else last_val_ppl, 2) if (val_ppl is not None or last_val_ppl is not None) else None,
            "val_pulse_losses": [round(x, 4) for x in cur_vpl] if cur_vpl is not None else None,
            "val_pulse_ppls": [round(x, 2) for x in cur_vppl] if cur_vppl is not None else None,
            # Fall back to the last generated set: on non-eval steps latest_canaries is
            # empty, which made the canary panel blank out between evals even though the
            # last generation was still the current state of the model.
            "canaries": latest_canaries if latest_canaries else last_live_canaries,
        }
        with open(live_tmp, "w", encoding="utf-8") as f:
            json.dump(live_payload, f, indent=2)
        os.replace(live_tmp, live_file)

        # Periodic Checkpoint Saving
        if not args.no_save and (step % args.save_every == 0 or step == stop_step or interrupted):
            save_checkpoint(latest_ckpt_path, model, optimizer, cfg, step, tokens_seen, best_val_loss, num_pulses=args.num_pulses)
            print(f"[Pulsar] Saved checkpoint -> {latest_ckpt_path} (step {step})")

        # Session Target or Milestone Checkpoint Saving
        if not args.no_save and ((args.target_step is not None and step == args.target_step) or (step % 1000 == 0)):
            milestone_path = os.path.join(args.checkpoint_dir, f"milestone_step_{step}.pt")
            save_checkpoint(milestone_path, model, optimizer, cfg, step, tokens_seen, best_val_loss, num_pulses=args.num_pulses)
            print(f"[Pulsar] Milestone Checkpoint Saved -> {milestone_path} (step {step})")

        # Thermal Cooldown
        if (
            args.thermal_cooldown_every > 0
            and step % args.thermal_cooldown_every == 0
            and step < stop_step
            and not interrupted
        ):
            print(f"\n[Hardware Protection] Pausing for {args.thermal_cooldown_seconds}s thermal cooldown...")
            torch.cuda.empty_cache()
            for sec_left in range(args.thermal_cooldown_seconds, 0, -1):
                live_payload["status"] = "cooling"
                live_payload["cooling_remaining_s"] = sec_left
                live_payload["cooling_total_s"] = args.thermal_cooldown_seconds
                with open(live_tmp, "w", encoding="utf-8") as f:
                    json.dump(live_payload, f, indent=2)
                os.replace(live_tmp, live_file)
                time.sleep(1)
            live_payload["status"] = "training"
            live_payload["cooling_remaining_s"] = 0
            with open(live_tmp, "w", encoding="utf-8") as f:
                json.dump(live_payload, f, indent=2)
            os.replace(live_tmp, live_file)
            t_step_prev = time.time()
            print("[Hardware Protection] Cooldown complete. Resuming training.\n")

        if interrupted:
            break

    total_time = time.time() - t_start
    final_status = "interrupted" if interrupted else "completed"
    if 'live_payload' in locals() and os.path.exists(live_file):
        try:
            live_payload["status"] = final_status
            with open(live_tmp, "w", encoding="utf-8") as f:
                json.dump(live_payload, f, indent=2)
            os.replace(live_tmp, live_file)
        except Exception:
            pass

    print(f"\n=======================================================")
    print(f"Training finished in {total_time/60:.2f} mins (Status: {final_status}).")
    print(f"Tokens seen: {tokens_seen:,} | Best Val Loss: {best_val_loss:.4f}")
    print(f"Latest checkpoint: {latest_ckpt_path}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    train()
