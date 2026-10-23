#!/usr/bin/env python3
"""Branch a cosine-decayed checkpoint from a Stage-2 run, without touching the main run.

The curriculum decays LR only in Stage 3, so Stage 2 ends at peak LR and there is no
annealed short-context checkpoint to evaluate. This script takes a Stage-2 checkpoint,
runs a short cosine cooldown from its LR down to the floor in a SEPARATE output
directory, and writes ``branch_decay.pt``. The source checkpoint and its optimizer state
are never modified, so the main run can continue afterwards.

Usage:
    python scripts/branch_decay.py --checkpoint checkpoints/stage2/latest.pt \
        --steps 300 --out-dir checkpoints/stage2_decay
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.model import PulsarConfig, PulsarForCausalLM  # noqa: E402
from src.optim import PulsarDualOptimizer  # noqa: E402
from src.train import StreamingPinnedBatchLoader  # noqa: E402


def load_shards(paths):
    out = []
    for p in paths:
        out.append(np.memmap(p, dtype=np.uint16, mode="r"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train-bin", default="data/train_shard1.bin,data/train_shard2.bin")
    ap.add_argument("--out-dir", default=None,
                    help="Default: <checkpoint dir>/decay_<timestamp> - never the source dir")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--micro-batch-size", type=int, default=3)
    ap.add_argument("--global-batch-tokens", type=int, default=131_072)
    ap.add_argument("--min-lr-ratio", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-every", type=int, default=100)
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"checkpoint not found: {args.checkpoint}")
        return 1

    src_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    out_dir = args.out_dir or os.path.join(src_dir, f"decay_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir = os.path.abspath(out_dir)
    if out_dir == src_dir:
        raise SystemExit(
            "Refusing to write into the source checkpoint directory: this branch must not "
            "disturb the main run's state. Pass --out-dir."
        )
    os.makedirs(out_dir, exist_ok=True)

    # Record the source files so we can prove they were not touched.
    src_stat = {f: os.stat(os.path.join(src_dir, f)).st_mtime_ns
                for f in os.listdir(src_dir) if f.endswith(".pt")}

    print(f"[branch] source: {args.checkpoint}")
    print(f"[branch] output: {out_dir}")

    ck = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    cfg = PulsarConfig.from_dict(ck.get("config") or {}) if ck.get("config") else PulsarConfig()
    cfg.max_seq_len = 32768
    cfg.base_seq_len = 1024

    model = PulsarForCausalLM(cfg).cuda()
    model.to_training_precision(torch.bfloat16)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    print(f"[branch] loaded step {ck.get('global_step')} "
          f"({len(missing)} missing, {len(unexpected)} unexpected)")

    optimizer = PulsarDualOptimizer(model, muon_lr=0.02, adam_lr=6e-4, weight_decay=0.1,
                                    seed=args.seed)
    # Deliberately do NOT restore optimizer state: this is a fresh decay branch, and
    # copying momentum would let it influence the main run's trajectory if reused.
    lr_factor0 = ck.get("lr_factor", 1.0)

    shards = load_shards([p.strip() for p in args.train_bin.split(",") if p.strip()])
    loader = StreamingPinnedBatchLoader(shards, args.micro_batch_size, args.seq_len, "cuda")
    tokens_per_micro = args.micro_batch_size * args.seq_len
    grad_accum = max(1, (args.global_batch_tokens + tokens_per_micro - 1) // tokens_per_micro)

    base_muon, base_adam = 0.02, 6e-4
    model.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        # Cosine from the checkpoint's LR factor down to the floor.
        progress = (step - 1) / max(1, args.steps - 1)
        factor = args.min_lr_ratio + (lr_factor0 - args.min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
        for g in optimizer.opt_muon.param_groups:
            g["lr"] = base_muon * factor
        for sub in (optimizer.opt_adam_sr, optimizer.opt_adam):
            for g in sub.param_groups:
                g["lr"] = base_adam * factor

        optimizer.zero_grad(set_to_none=True)
        accum = 0.0
        for micro in range(grad_accum):
            x, y = loader.get_batch(args.seed + step * 1000 + micro)
            _, loss, _ = model(x, labels=y, num_pulses=cfg.default_pulses)
            (loss / grad_accum).backward()
            accum += loss.item() / grad_accum

        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None], 1.0)
        optimizer.step()
        model.update_expert_biases(bias_lr=0.001, num_pulses=cfg.default_pulses)

        if step % max(1, args.steps // 10) == 0 or step == args.steps:
            print(f"  [{step:4d}/{args.steps}] loss {accum:.4f} | lr factor {factor:.4f}")

        if step % args.save_every == 0 or step == args.steps:
            raw = getattr(model, "_orig_mod", model)
            torch.save(
                {
                    "global_step": step,
                    "branch_of": os.path.abspath(args.checkpoint),
                    "source_step": ck.get("global_step"),
                    "lr_factor": factor,
                    "config": cfg.to_dict(),
                    "model": raw.state_dict(),
                },
                os.path.join(out_dir, "branch_decay.pt.tmp"),
            )
            os.replace(os.path.join(out_dir, "branch_decay.pt.tmp"),
                       os.path.join(out_dir, "branch_decay.pt"))

    # Prove the source checkpoint was not modified.
    changed = [
        f for f, m in src_stat.items()
        if os.stat(os.path.join(src_dir, f)).st_mtime_ns != m
    ]
    print(f"\n[branch] wrote {os.path.join(out_dir, 'branch_decay.pt')}")
    print(f"[branch] source checkpoints modified: {changed if changed else 'none (correct)'}")
    print(f"[branch] elapsed {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
