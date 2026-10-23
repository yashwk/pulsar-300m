#!/usr/bin/env python3
"""Report per-pulse ‖Δh‖ / ‖h_in‖ at step 0 and after a short run.

This is the evidence for the harmonic-damping decision: pulse_res_scale is initialised
to 1/(k+1) specifically to counter the "recurrent update dwarfs the residual state"
pathology. With the routed-expert init now fixed, the damping may be redundant.

Decision rule (owner):
    raw ratio < ~0.5  -> damping protects nothing; consider initialising scales at 1.0
    raw ratio > ~1    -> keep 1/(k+1)

"Raw" means before the per-pulse scale; "post-scale" means after it. Both are reported,
per pulse, as mean and p95 over the residual stream.

Usage:
    python scripts/dh_report.py --steps 0
    python scripts/dh_report.py --steps 20 --checkpoint checkpoints/smoke/latest.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.model import PulsarConfig, PulsarForCausalLM  # noqa: E402


@torch.no_grad()
def measure(model: PulsarForCausalLM, x: torch.Tensor, num_pulses: int) -> list[dict]:
    """Replay the pulse loop, capturing ‖h_in‖, ‖h_curr - h_in‖ and the scaled ‖Δh‖."""
    cfg = model.cfg
    h_0 = model.tok_emb(x)
    h_0_norm = model.anchor_norm(h_0)
    h_pulse = h_0
    v_first = None
    rows = []
    for k in range(num_pulses):
        step_emb = model.pulse_step_emb.weight[k].to(h_0_norm.dtype).view(1, 1, -1)
        if k == 0:
            h_in = h_0_norm + step_emb
        else:
            mix_k = torch.sigmoid(model.input_mix_logit[k]).to(h_pulse.dtype)
            h_in = (1.0 - mix_k) * model.pulse_state_norm(h_pulse) + mix_k * h_0_norm + step_emb

        h_curr = h_in
        for block in model.core:
            cos = model.rope_local(h_in, h_in.size(1))[0] if block.attn.is_local else model.rope_global(h_in, h_in.size(1))[0]
            sin = model.rope_local(h_in, h_in.size(1))[1] if block.attn.is_local else model.rope_global(h_in, h_in.size(1))[1]
            h_curr, v_first, _, _, _ = block(h_curr, cos, sin, pulse_idx=k, v_first=v_first)

        raw_delta = (h_curr - h_in)
        r_scale = model.pulse_res_scale[k].to(h_in.dtype)
        scaled_delta = raw_delta * r_scale
        h_pulse = h_in + scaled_delta

        def norms(t):
            v = t.float().norm(dim=-1).reshape(-1)
            return v

        hin_n = norms(h_in)
        raw_n = norms(raw_delta)
        sc_n = norms(scaled_delta)
        rows.append({
            "pulse": k + 1,
            "scale_init": float(1.0 / (k + 1.0)),
            "scale_learned_mean": float(r_scale.float().mean().item()),
            "hin_mean": float(hin_n.mean().item()),
            "raw_delta_mean": float(raw_n.mean().item()),
            "scaled_delta_mean": float(sc_n.mean().item()),
            "raw_ratio_mean": float((raw_n / hin_n.clamp_min(1e-6)).mean().item()),
            "raw_ratio_p95": float(np.percentile((raw_n / hin_n.clamp_min(1e-6)).cpu().numpy(), 95)),
            "scaled_ratio_mean": float((sc_n / hin_n.clamp_min(1e-6)).mean().item()),
            "scaled_ratio_p95": float(np.percentile((sc_n / hin_n.clamp_min(1e-6)).cpu().numpy(), 95)),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None, help="Optional .pt to load (else fresh init)")
    ap.add_argument("--steps", type=int, default=0, help="Train N steps before measuring (0 = step 0)")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--num-pulses", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = PulsarConfig(max_seq_len=32768, base_seq_len=1024, default_pulses=args.num_pulses)
    model = PulsarForCausalLM(cfg).cuda()
    model.to_training_precision(torch.bfloat16)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ck = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
        sd = ck.get("model", ck)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[load] {args.checkpoint}: {len(missing)} missing, {len(unexpected)} unexpected")
        model.to_training_precision(torch.bfloat16)

    if args.steps > 0:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model.train()
        for i in range(args.steps):
            x = torch.randint(0, cfg.vocab_size, (args.batch, args.seq_len), device="cuda")
            y = torch.randint(0, cfg.vocab_size, (args.batch, args.seq_len), device="cuda")
            opt.zero_grad()
            _, loss, _ = model(x, labels=y, num_pulses=args.num_pulses)
            loss.backward()
            opt.step()
        print(f"[train] {args.steps} steps, final loss {loss.item():.4f}")

    model.eval()
    x = torch.randint(0, cfg.vocab_size, (args.batch, args.seq_len), device="cuda")
    rows = measure(model, x, args.num_pulses)

    label = f"after {args.steps} steps" if args.steps else "at step 0"
    print(f"\n=== ‖Δh‖ / ‖h_in‖ {label} ===")
    print(f"{'pulse':>5} {'scale_init':>10} {'scale_learn':>11} {'‖h_in‖':>9} {'‖Δh‖raw':>9} "
          f"{'raw/‖h_in‖':>11} {'p95':>7} {'post-scale':>11} {'verdict':>22}")
    for r in rows:
        v = "damping protects" if r["raw_ratio_mean"] > 1.0 else (
            "damping redundant?" if r["raw_ratio_mean"] < 0.5 else "borderline")
        print(f"{r['pulse']:>5} {r['scale_init']:>10.3f} {r['scale_learned_mean']:>11.4f} "
              f"{r['hin_mean']:>9.2f} {r['raw_delta_mean']:>9.2f} {r['raw_ratio_mean']:>11.3f} "
              f"{r['raw_ratio_p95']:>7.3f} {r['scaled_ratio_mean']:>11.3f} {v:>22}")

    print("\nDecision rule: raw < 0.5 => consider init 1.0; raw > 1 => keep 1/(k+1).")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"steps": args.steps, "rows": rows}, f, indent=2)
        print(f"[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
