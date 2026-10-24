#!/usr/bin/env python3
"""Re-score a checkpoint with the corrected eval metric (WI-06).

The old `evaluate_loss` returned the weighted sum of three pulse CEs plus z-loss, with
the intermediate pulses scored on a 25% token subsample at different offsets. That is
not the final model's loss. This loads a checkpoint and reports, on 100% of tokens:

    val_ce_final  (pulse 3 - the model's real loss)
    val_ce_p1..pK
    latent_gain   = p1 - pK   (positive means looping helps)

Usage:
    python scripts/rescore.py --checkpoint checkpoints/baseline/best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.model import PulsarForCausalLM  # noqa: E402
from src.packing import PackedBatchLoader, build_doc_index, plan_epoch  # noqa: E402
from src.train import StreamingPinnedBatchLoader, evaluate_loss  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/baseline/best.pt")
    ap.add_argument("--val-bin", default="data/val_shard1.bin")
    ap.add_argument("--sampler", choices=["flat", "bestfit"], default="flat", help="Sampling strategy (flat or bestfit packing)")
    ap.add_argument("--eval-iters", type=int, default=50)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--micro-batch-size", type=int, default=3)
    ap.add_argument("--num-pulses", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"checkpoint not found: {args.checkpoint}")
        return 1

    ck = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    cfg_dict = ck.get("config") or {}
    from src.model import PulsarConfig
    cfg = PulsarConfig.from_dict(cfg_dict) if cfg_dict else PulsarConfig()
    cfg.max_seq_len = 32768
    cfg.base_seq_len = 1024
    cfg.default_pulses = args.num_pulses

    model = PulsarForCausalLM(cfg).cuda()
    model.to_training_precision(torch.bfloat16)

    sd = ck.get("model", ck)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] {args.checkpoint}: step {ck.get('global_step')} "
          f"best_val_loss {ck.get('best_val_loss')}")
    print(f"[load] {len(missing)} missing, {len(unexpected)} unexpected keys")

    val = np.memmap(args.val_bin, dtype=np.uint16, mode="r")
    if args.sampler == "bestfit":
        val_index = build_doc_index([val])
        val_plan = plan_epoch(
            val_index.length.astype(np.int64), args.seq_len,
            seed=42, epoch=0, strategy="bestfit",
        )
        loader = PackedBatchLoader(
            [val], val_index, val_plan,
            micro_batch_size=args.micro_batch_size, device="cuda",
        )
    else:
        loader = StreamingPinnedBatchLoader(
            [val], micro_batch_size=args.micro_batch_size, seq_len=args.seq_len, device="cuda"
        )
    res = evaluate_loss(model, loader, args.eval_iters, args.num_pulses)

    print(f"\n=== Re-scored with corrected metric (100% of tokens) ===")
    print(f"  val_ce_final : {res['val_ce_final']:.4f}   (PPL {res['val_ppl']:.1f})")
    for i, ce in enumerate(res["per_pulse"], start=1):
        print(f"  val_ce_p{i}    : {ce:.4f}   (PPL {math.exp(min(ce, 20.0)):.1f})")
    print(f"  latent_gain  : {res['latent_gain']:+.4f}  "
          f"({'looping helps' if res['latent_gain'] > 0 else 'looping HURTS'})")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
