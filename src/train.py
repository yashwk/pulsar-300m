import argparse
import glob
import json
import math
import os
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
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        self.log_file = open(filepath, "a", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


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
    min_ratio: float = 0.05,
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
    """Scale group learning rates across Muon and AdamW."""
    for g in optimizer.opt_muon.param_groups:
        g["lr"] = base_muon * factor
    for g in optimizer.opt_adam.param_groups:
        g["lr"] = base_adam * factor


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
    max_new_tokens: int = 30,
    temperature: float = 0.7,
    device: str = "cuda",
    repetition_penalty: float = 1.15,
) -> str:
    """Autoregressively decode tokens using full recurrent depth."""
    was_training = model.training
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context_window = input_ids[:, -1024:]
            logits, _, _ = model(context_window, labels=None)
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

            if next_token.item() == tokenizer.eos_token_id:
                break

    if was_training:
        model.train()
    return tokenizer.decode(input_ids[0], skip_special_tokens=False)


def evaluate_loss(
    model: torch.nn.Module,
    val_loader: StreamingPinnedBatchLoader,
    eval_iters: int,
    num_pulses: int,
) -> tuple[float, list[float]]:
    """Evaluate mean cross-entropy loss and per-pulse breakdown on validation data dynamically for any K."""
    was_training = model.training
    model.eval()
    losses = []
    pulse_loss_accum = [[] for _ in range(num_pulses)]
    with torch.no_grad():
        for i in range(eval_iters):
            seed_val = 999983 + i * 17
            x, y = val_loader.get_batch(seed_val)
            _, loss, metrics = model(x, labels=y, num_pulses=num_pulses, subsample_offset=i)
            losses.append(loss.item())
            step_pulses = metrics.get("pulse_losses", [])
            for p_idx in range(min(num_pulses, len(step_pulses))):
                val = step_pulses[p_idx]
                pulse_loss_accum[p_idx].append(val if isinstance(val, (int, float)) else val.item())

    if was_training:
        model.train()

    mean_loss = float(np.mean(losses)) if losses else 0.0
    mean_pulses = [float(np.mean(pl)) if pl else mean_loss for pl in pulse_loss_accum]
    return mean_loss, mean_pulses


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
    ckpt = {
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val_loss,
        "config": cfg.to_dict() if hasattr(cfg, "to_dict") else cfg,
        "model_summary": model_summary,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict() if hasattr(optimizer, "state_dict") else None,
        "opt_muon": optimizer.opt_muon.state_dict(),
        "opt_adam": optimizer.opt_adam.state_dict(),
        "opt_adam_sr": optimizer.opt_adam_sr.state_dict(),
    }
    tmp_path = filepath + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, filepath)


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
    parser.add_argument("--min-lr-ratio", type=float, default=0.05, help="Minimum LR ratio during cooldown")
    parser.add_argument("--eval-every", type=int, default=25, help="Steps between validation and canary generation")
    parser.add_argument("--eval-iters", type=int, default=10, help="Batches evaluated during validation")
    parser.add_argument("--save-every", type=int, default=25, help="Steps between saving latest.pt")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--output-ckpt", type=str, default=None, help="Explicit filepath to save output checkpoints")
    parser.add_argument("--resume", nargs="?", const="latest", default=None, help="Resume training from checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--num-pulses", type=int, default=3, help="Number of recurrent pulses K")
    parser.add_argument("--muon-lr", type=float, default=0.02, help="Muon optimizer base LR for 2D matrices")
    parser.add_argument("--adam-lr", type=float, default=6e-4, help="AdamW base LR for 1D norms and embeddings")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Muon weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient norm clip value")
    parser.add_argument("--no-stochastic-rounding", action="store_true",
                        help="Disable stochastic rounding for bf16 weights (default: on). "
                             "Without SR, sub-ulp updates and weight decay round away.")
    parser.add_argument("--muon-momentum-fp32", action="store_true",
                        help="Keep Muon momentum in FP32 (default: bf16, matching the weight)")
    parser.add_argument("--compile", action="store_true", default=False, help="Enable torch.compile")
    parser.add_argument("--thermal-cooldown-every", type=int, default=0, help="Cooldown pause frequency in steps")
    parser.add_argument("--thermal-cooldown-seconds", type=int, default=100, help="Duration of thermal cooldown pause")
    args = parser.parse_args()

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
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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

    # Initialize Model
    cfg = PulsarConfig(
        max_seq_len=32768,
        base_seq_len=1024,
        default_pulses=args.num_pulses,
    )
    model = PulsarForCausalLM(cfg).to(device=device)
    # Single source of truth for mixed precision (WI-01). The previous implementation
    # cast the whole model to bf16 and then re-cast a name-based subset to FP32, which
    # missed pulse_res_scale / routed_scale / attn_gate / pulse_step_emb (left frozen in
    # bf16) and cast every buffer, including the MoE expert counters and RoPE caches.
    model.to_training_precision(torch.bfloat16)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Pulsar] Pulsar-MoE Model Initialized: {total_params:,} parameters ({total_params/1e6:.2f}M)")

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
            ckpt = torch.load(resume_file, map_location=device, weights_only=False)
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
            raw_model.core[i] = torch.compile(raw_model.core[i], dynamic=True)

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
    last_val_pulse_ppls = None
    latest_canaries = []

    stop_step = args.target_step if args.target_step is not None else args.max_steps
    print(f"\n[Pulsar] Starting training: Steps {start_step + 1} -> {stop_step} (Curriculum Horizon: {args.max_steps})...\n")

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

        # Accumulate gradients across micro-batches
        accum_loss_tensor = torch.zeros((), device=device, dtype=torch.float32)
        for micro_step in range(grad_accum):
            seed_micro = args.seed + step * 1000 + micro_step
            x, y = train_loader.get_batch(seed_micro)

            t0 = time.time()
            _, loss, metrics = model(x, labels=y, num_pulses=args.num_pulses, subsample_offset=micro_step)
            t1 = time.time()
            scaled_loss = loss / grad_accum
            scaled_loss.backward()
            t2 = time.time()

            t_fwd_total += (t1 - t0)
            t_bwd_total += (t2 - t1)
            accum_loss_tensor += loss.detach() / grad_accum
            step_metrics = metrics

        accum_loss = accum_loss_tensor.item()
        tokens_seen += effective_global_batch_tokens

        # Sub-optimizer gradient norms before clipping (fused foreach kernels)
        m_grads = [p.grad for group in optimizer.opt_muon.param_groups for p in group["params"] if p.grad is not None]
        a_grads = [p.grad for group in optimizer.opt_adam.param_groups for p in group["params"] if p.grad is not None]
        muon_grad_norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(m_grads))).item() if m_grads else 0.0
        adam_grad_norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(a_grads))).item() if a_grads else 0.0

        # Global gradient clipping (fused C++ foreach multi-tensor)
        all_trainable = [p for p in model.parameters() if p.grad is not None]
        total_grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable, args.grad_clip, foreach=True).item()

        # Optimizer step
        t_opt_start = time.time()
        optimizer.step()
        raw_model = getattr(model, "_orig_mod", model)
        raw_model.update_expert_biases(bias_lr=0.001)
        t_opt_total = time.time() - t_opt_start

        # Step timing, speed and hardware utilization
        t_step_now = time.time()
        step_dt = t_step_now - t_step_prev
        t_step_prev = t_step_now
        tok_per_sec = effective_global_batch_tokens / max(1e-4, step_dt)

        # MFU estimation: FLOPs/tok ≈ 6 * non_embed_active_params * K = 6 * 97.15M * 3 ≈ 2.05 GFLOPs/tok (Upgrade B)
        achieved_tflops = tok_per_sec * 2.05e-3
        mfu_pct = (achieved_tflops / 60.0) * 100.0  # 60 TFLOPs dense BF16 baseline (RTX 4060 Laptop)

        # Loss delta, elapsed time & ETA calculation
        loss_delta = (accum_loss - prev_train_loss) if prev_train_loss is not None else 0.0
        prev_train_loss = accum_loss
        train_ppl = math.exp(min(accum_loss, 20.0))

        vram_reserved = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0
        vram_peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

        def to_val(v, default=0.0):
            if v is None:
                return default
            if hasattr(v, "item"):
                return v.item()
            return float(v)

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
        print(
            f"\n[Step {step:4d}/{args.max_steps}] Loss: {accum_loss:.4f} ({pct_sign}{loss_delta:.4f}) | "
            f"Train PPL: {train_ppl:5.2f} | Tokens: {tokens_seen:,}"
        )
        moe_str = " | ".join(f"E{i}: {pct:4.1f}%" for i, pct in enumerate(e_pcts)) if e_pcts else "Dense FFN"
        pulses_str = " | ".join(
            f"L{i+1}: {step_pulse_losses[i]:.4f} (d{i+1}: {step_pulse_deltas[i]:.2f})"
            for i in range(min(len(step_pulse_losses), len(step_pulse_deltas)))
        )
        print(f"  ↳ MoE Experts: [{moe_str}] | Conf: {conf:4.2f} | MaxBias: {max_bias:5.3f}")
        print(f"  ↳ Pulses: {pulses_str} | λ: {mix_lambda:.3f}")
        print(
            f"  ↳ Gradients: Total: {total_grad_norm:5.2f} (Muon: {muon_grad_norm:4.2f}, Adam: {adam_grad_norm:4.2f})\n"
            f"  ↳ Learning Rates: [Muon: {current_muon_lr:7.5f}, Adam: {current_adam_lr:.1e}]\n"
            f"  ↳ Performance: {tok_per_sec/1e3:4.1f}k tok/s ({mfu_pct:4.1f}% MFU) | Step: {step_dt:4.2f}s (fwd: {t_fwd_total:4.2f}s, bwd: {t_bwd_total:4.2f}s, opt: {t_opt_total:4.2f}s) | VRAM: {vram_peak:3.1f}GB / 8.0GB"
        )

        val_loss = None
        val_ppl = None
        val_pulse_losses = None
        val_pulse_ppls = None
        # Periodic Evaluation and Canary Generation
        if step % args.eval_every == 0 or step == args.max_steps or interrupted:
            val_loss, val_pulse_losses = evaluate_loss(
                model, val_loader, args.eval_iters, args.num_pulses
            )
            val_ppl = math.exp(min(val_loss, 20.0))
            val_pulse_ppls = [math.exp(min(pl, 20.0)) for pl in val_pulse_losses]
            last_val_loss = val_loss
            last_val_ppl = val_ppl
            last_val_pulse_losses = val_pulse_losses
            last_val_pulse_ppls = val_pulse_ppls
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            latent_gain = (val_pulse_losses[-1] - val_pulse_losses[0]) if len(val_pulse_losses) > 1 else 0.0
            print(f"\n--- [Eval @ Step {step}] Overall Val Loss: {val_loss:.4f} (PPL: {val_ppl:.1f}) | Best: {best_val_loss:.4f} ---")
            for p_idx, pl in enumerate(val_pulse_losses):
                eff_l = (p_idx + 1) * cfg.n_core
                pl_ppl = math.exp(min(pl, 20.0))
                refine_str = f" | Latent Refinement: {latent_gain:+.4f}" if (p_idx == len(val_pulse_losses) - 1 and len(val_pulse_losses) > 1) else ""
                print(f"  ↳ Pulse {p_idx + 1} [{eff_l:2d} eff layers]: Val Loss = {pl:.4f} (PPL: {pl_ppl:.1f}){refine_str}")
            print("--- Canary Generations Live ---")
            latest_canaries = []
            for canary_meta in canary_prompts:
                canary_idx = canary_meta["id"]
                prompt = canary_meta["prompt"]
                tag = canary_meta["tag"]
                category = canary_meta["category"]
                try:
                    generation = generate_canary(
                        model, tokenizer, prompt, max_new_tokens=35, temperature=0.7, device=device
                    )
                    generation_clean = generation.replace("\n", " ").strip()
                    print(f"  [{tag}] {generation_clean}")
                    latest_canaries.append({
                        "id": canary_idx,
                        "tag": tag,
                        "category": category,
                        "prompt": prompt,
                        "completion": generation_clean
                    })
                except Exception as e:
                    print(f"  [{tag}] Canary error: {e}")
            print("-----------------------------------------------------------------")

            if is_best:
                best_ckpt_path = os.path.join(args.checkpoint_dir, "best.pt")
                if args.output_ckpt is not None:
                    best_ckpt_path = args.output_ckpt
                save_checkpoint(best_ckpt_path, model, optimizer, cfg, step, tokens_seen, best_val_loss, num_pulses=args.num_pulses)
                print(f"[Pulsar] Saved best checkpoint (val_loss={best_val_loss:.4f}) -> {best_ckpt_path}")

        cur_vpl = val_pulse_losses
        cur_vppl = val_pulse_ppls

        raw_model = getattr(model, "_orig_mod", model)
        model_summary = raw_model.get_model_summary(args.num_pulses) if hasattr(raw_model, "get_model_summary") else {}

        metric_record = {
            "step": step,
            "max_steps": args.max_steps,
            "tokens_seen": tokens_seen,
            "train_dataset_tokens": train_loader.total_tokens,
            "num_train_shards": len(train_shards),
            "global_batch_tokens": effective_global_batch_tokens,
            "micro_batch_size": args.micro_batch_size,
            "seq_len": args.seq_len,
            "num_pulses": args.num_pulses,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "model_config": model_summary,
            "train_loss": round(accum_loss, 4),
            "loss_delta": round(loss_delta, 4),
            "train_ppl": round(train_ppl, 2),
            "val_loss": round(val_loss, 4) if val_loss is not None else None,
            "val_ppl": round(val_ppl, 2) if val_ppl is not None else None,
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
            "vram_peak": round(vram_peak, 2),
            "router_confidence": round(conf, 4),
            "router_max_bias": round(max_bias, 4),
            "expert_pcts": [round(x, 2) for x in e_pcts],
            "mix_lambda": round(mix_lambda, 4),
            "canaries": latest_canaries if latest_canaries else (last_live_canaries if 'last_live_canaries' in locals() else []),
        }

        # Write to JSONL
        jsonl_file = os.path.join(args.checkpoint_dir, "metrics.jsonl")
        with open(jsonl_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(metric_record) + "\n")

        # Write to CSV
        csv_file = os.path.join(args.checkpoint_dir, "metrics.csv")
        csv_exists = os.path.exists(csv_file)
        with open(csv_file, "a", encoding="utf-8") as f:
            if not csv_exists:
                header = [
                    "step", "tokens_seen", "train_loss", "loss_delta", "train_ppl", "val_loss",
                    "pulse_l1", "pulse_l2", "pulse_l3", "delta_h1", "delta_h2", "delta_h3",
                    "e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7", "router_conf", "max_bias", "mix_lambda",
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
                round(e_pcts[0], 2) if len(e_pcts) > 0 else 0.0,
                round(e_pcts[1], 2) if len(e_pcts) > 1 else 0.0,
                round(e_pcts[2], 2) if len(e_pcts) > 2 else 0.0,
                round(e_pcts[3], 2) if len(e_pcts) > 3 else 0.0,
                round(e_pcts[4], 2) if len(e_pcts) > 4 else 0.0,
                round(e_pcts[5], 2) if len(e_pcts) > 5 else 0.0,
                round(e_pcts[6], 2) if len(e_pcts) > 6 else 0.0,
                round(e_pcts[7], 2) if len(e_pcts) > 7 else 0.0,
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
            "canaries": latest_canaries,
        }
        with open(live_tmp, "w", encoding="utf-8") as f:
            json.dump(live_payload, f, indent=2)
        os.replace(live_tmp, live_file)

        # Periodic Checkpoint Saving
        if step % args.save_every == 0 or step == stop_step or interrupted:
            save_checkpoint(latest_ckpt_path, model, optimizer, cfg, step, tokens_seen, best_val_loss, num_pulses=args.num_pulses)
            print(f"[Pulsar] Saved checkpoint -> {latest_ckpt_path} (step {step})")

        # Session Target or Milestone Checkpoint Saving
        if (args.target_step is not None and step == args.target_step) or (step % 1000 == 0):
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
    print(f"\n=======================================================")
    print(f"Training finished in {total_time/60:.2f} mins.")
    print(f"Tokens seen: {tokens_seen:,} | Best Val Loss: {best_val_loss:.4f}")
    print(f"Latest checkpoint: {latest_ckpt_path}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    train()
