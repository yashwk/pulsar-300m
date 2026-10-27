"""Synthetic Fixture Generator for Pulsar Monitor V2.

Generates a realistic 45,000-step metrics.jsonl file with:
- Restarts (step backward jumps).
- Missing/older schema fields.
- Occasional NaN/Inf values.
- Thermal pause intervals.
- Validation evaluations every 50 steps.
- Incomplete/half-written line at EOF to test tail-reader robustness.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random


def generate_fixture(filepath: str, total_steps: int = 45000):
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    print(f"[Fixture] Generating {total_steps:,} synthetic steps to {filepath}...")
    with open(filepath, "w", encoding="utf-8") as f:
        step = 1
        loss = 4.2
        val_best = 4.0

        while step <= total_steps:
            # Thermal pauses: step stays constant or skips time
            is_cooling = (step % 1000 == 0) and (step != total_steps)

            # Restarts: at step 10,000 and 25,000, simulate a crash and rewind by 150 steps
            if step in (10150, 25150):
                rewind_step = step - 150
                print(f"[Fixture] Simulating restart at step {step} -> rewinding to {rewind_step}")
                step = rewind_step

            # Loss trajectory: logarithmic decay + noise
            loss = 1.95 + 2.2 * math.exp(-step / 8000.0) + random.gauss(0, 0.02)
            if step == 32000:
                # Loss spike anomaly
                loss += 1.5

            p1_loss = loss + 0.12 + random.gauss(0, 0.005)
            p2_loss = loss + 0.03 + random.gauss(0, 0.005)
            p3_loss = loss

            is_eval = (step % 50 == 0)
            val_loss = None
            if is_eval:
                val_loss = p3_loss + random.gauss(0, 0.01)
                if val_loss < val_best:
                    val_best = val_loss

            # MoE expert shares (mostly balanced around 12.5%, with occasional burst)
            shares = [12.5 + random.gauss(0, 0.4) for _ in range(8)]
            tot = sum(shares)
            shares = [round((s / tot) * 100.0, 2) for s in shares]

            rec = {
                "step": step,
                "tokens_seen": step * 131072,
                "train_dataset_tokens": 1427463964,
                "num_train_shards": 3,
                "global_batch_tokens": 131072,
                "train_loss": round(p1_loss * 0.25 + p2_loss * 0.25 + p3_loss * 0.5, 4),
                "pulse_losses": [round(p1_loss, 4), round(p2_loss, 4), round(p3_loss, 4)],
                "val_ce_final": round(val_loss, 4) if val_loss is not None else None,
                "val_loss": round(val_loss, 4) if val_loss is not None else None,
                "best_val_loss": round(val_best, 4),
                "tok_per_sec": round(9200.0 + random.gauss(0, 150), 1),
                "mfu_pct": round(31.8 + random.gauss(0, 0.5), 1),
                "vram_peak": round(6.8 + random.gauss(0, 0.05), 2),
                "grad_norm_total": round(0.25 + random.gauss(0, 0.03), 4),
                "lr_muon": round(0.02 * min(1.0, step / 300.0), 6),
                "lr_adam": round(0.0006 * min(1.0, step / 300.0), 6),
                "moe_drop_rate": round(0.015 + random.gauss(0, 0.003), 5),
                "expert_pcts": shares,
                "router_confidence": round(0.54 + random.gauss(0, 0.01), 3),
                "router_max_bias": round(min(1.0, 0.0003 * step), 3),
                "latent_gain": round(p1_loss - p3_loss, 4),
            }

            # Inject NaN anomaly at step 22,000 to test alert system
            if step == 22000:
                rec["train_loss"] = float("nan")

            f.write(json.dumps(rec) + "\n")
            step += 1

        # Simulate half-written line at EOF (no newline)
        f.write('{"step": 45001, "train_loss": 1.94')

    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"[Fixture] Done. Created {filepath} ({file_size_mb:.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="monitor/tools/fixture_45k_metrics.jsonl")
    parser.add_argument("--steps", type=int, default=45000)
    args = parser.parse_args()
    generate_fixture(args.output, total_steps=args.steps)
