#!/usr/bin/env python3
"""Measure EOS-aligned packing (C') against the flat sampler.

Reports, per context length:
  * intactness among documents that *can* fit, count- and token-weighted
  * intactness over all documents (including those longer than T, which no scheme
    can present intact)
  * padding waste, filler share
  * whether the over-long-document supply covers the leftover space
  * loader throughput (tokens/s and windows/s)

Usage
-----
    # packed vs flat on the real shards at T = 2048, 8192, 32768
    python scripts/measure_packing.py --shards data/train_shard1.bin --seq-lens 2048 8192 32768

    # synthetic, for a quick check
    python scripts/measure_packing.py --synthetic --seq-lens 2048 8192
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.packing import (  # noqa: E402
    PackedBatchLoader,
    build_doc_index,
    flat_window_ids,
    intact_fractions,
    plan_epoch,
    waste_fractions,
)


def synthetic_lengths(n: int, seed: int = 0) -> np.ndarray:
    """CoT-like length distribution: lognormal around the measured shard median."""
    rng = np.random.default_rng(seed)
    return np.clip(rng.lognormal(np.log(1278), 0.35, n), 100, 20000).astype(np.int64)


def measure(lengths: np.ndarray, shards, index, seq_len: int, micro_batch_size: int = 4) -> dict:
    # ---- flat ----
    first, last = flat_window_ids(index.global_start, lengths, seq_len)
    flat_intact = first == last
    fittable = lengths <= seq_len
    flat = {
        "count_among_fittable": float(flat_intact[fittable].mean()) if fittable.any() else 0.0,
        "tokens_among_fittable": float(
            lengths[flat_intact & fittable].sum() / max(1, int(lengths[fittable].sum()))
        ) if fittable.any() else 0.0,
        "count_all": float(flat_intact.mean()),
        "tokens_all": float(lengths[flat_intact].sum() / max(1, int(lengths.sum()))),
    }

    # ---- packed ----
    t0 = time.time()
    plan = plan_epoch(lengths, seq_len, seed=0, epoch=0, strategy="bestfit", lookahead=64)
    plan_secs = time.time() - t0

    pk = intact_fractions(lengths, plan=plan, seq_len=seq_len)
    waste = waste_fractions(plan)

    # ---- filler supply check ----
    # Filler may only come from documents longer than seq_len, which can never be intact
    # under any scheme. Every over-long token is eventually emitted, so the question is
    # not "is supply > demand" but "did demand get filled": leftover space that no whole
    # document could occupy is filler_used + padding. If padding is negligible, the
    # over-long pool covered the leftover. `unused_over_long` shows surplus.
    over_long_tokens = int(lengths[~fittable].sum())
    whole_able_tokens = int(lengths[fittable].sum())
    filler_used = int(waste["filler_tokens"])
    leftover_space = filler_used + int(waste["padding_tokens"])
    unused_over_long = max(0, over_long_tokens - filler_used)
    supply_ratio = (over_long_tokens / leftover_space) if leftover_space else float("inf")

    # ---- throughput ----
    n_windows = len(plan)
    loader = PackedBatchLoader(shards, index, plan, micro_batch_size=micro_batch_size, device="cpu")
    t0 = time.time()
    windows_read = 0
    batch = 0
    while True:
        try:
            x, _ = loader.get_batch([batch * micro_batch_size + k for k in range(micro_batch_size)])
        except IndexError:
            break
        windows_read += x.shape[0]
        batch += 1
        if windows_read >= min(n_windows, 2000):
            break
    elapsed = time.time() - t0
    tok_per_s = (windows_read * seq_len) / max(elapsed, 1e-9)
    win_per_s = windows_read / max(elapsed, 1e-9)

    return {
        "seq_len": seq_len,
        "documents": int(lengths.size),
        "fittable_documents": int(fittable.sum()),
        "over_long_documents": int((~fittable).sum()),
        "flat": flat,
        "packed": {
            "count_among_fittable": pk["count_among_fittable"],
            "tokens_among_fittable": pk["tokens_among_fittable"],
            "count_all": pk["count"],
            "tokens_all": pk["tokens"],
        },
        "waste": waste,
        "filler_supply": {
            "over_long_tokens": over_long_tokens,
            "leftover_space_tokens": leftover_space,
            "filler_tokens_used": filler_used,
            "supply_ratio": supply_ratio,
            "unused_over_long_tokens": unused_over_long,
            "supply_covered": bool(waste["padding_frac"] < 0.001),
            "whole_able_tokens": whole_able_tokens,
        },
        "plan_build_secs": round(plan_secs, 2),
        "loader": {
            "windows": n_windows,
            "tokens_per_s": round(tok_per_s, 1),
            "windows_per_s": round(win_per_s, 1),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", nargs="*", default=[])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n-synthetic", type=int, default=4000)
    ap.add_argument("--max-tokens", type=int, default=60_000_000, help="cap tokens scanned from real shards")
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[2048, 8192, 32768])
    ap.add_argument("--micro-batch-size", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.synthetic:
        lengths = synthetic_lengths(args.n_synthetic)
        # synthetic shard for the loader
        out = []
        tok = 1
        for ln in lengths:
            out.extend([(tok + i) % 60000 + 1 for i in range(int(ln) - 1)])
            out.append(0)
            tok += int(ln)
        shard = np.asarray(out, dtype=np.uint16)
        shards = [shard]
        index = build_doc_index(shards)
        source = "synthetic"
    else:
        if not args.shards:
            ap.error("provide --shards or --synthetic")
        shards = []
        budget = args.max_tokens
        for p in args.shards:
            arr = np.memmap(p, dtype=np.uint16, mode="r")
            if len(arr) > budget:
                arr = arr[:budget]
                budget = 0
            else:
                budget -= len(arr)
            shards.append(arr)
            if budget <= 0:
                break
        index = build_doc_index(shards)
        lengths = index.length.astype(np.int64)
        source = ", ".join(args.shards)

    print(f"source: {source}")
    print(f"documents: {len(lengths):,} | tokens: {int(lengths.sum()):,}")
    print(f"doc length: p50={int(np.median(lengths)):,} p90={int(np.percentile(lengths, 90)):,} "
          f"max={int(lengths.max()):,}")

    results = []
    for T in args.seq_lens:
        r = measure(lengths, shards, index, T, args.micro_batch_size)
        results.append(r)
        f, p, w, s = r["flat"], r["packed"], r["waste"], r["filler_supply"]
        print(f"\n=== T = {T:,} ===")
        print(f"  intact among fittable : flat {f['count_among_fittable']:.1%} -> packed {p['count_among_fittable']:.1%}"
              f"   (token-weighted {f['tokens_among_fittable']:.1%} -> {p['tokens_among_fittable']:.1%})")
        print(f"  intact over all docs  : flat {f['count_all']:.1%} -> packed {p['count_all']:.1%}"
              f"   (token-weighted {f['tokens_all']:.1%} -> {p['tokens_all']:.1%})")
        print(f"  padding waste         : {w['padding_frac']:.3%} of capacity ({w['padding_tokens']:,} tokens)")
        print(f"  filler share          : {w['filler_share_of_content']:.1%} of content tokens")
        print(f"  filler supply         : leftover {s['leftover_space_tokens']:,} tok, over-long pool "
              f"{s['over_long_tokens']:,} tok, unused {s['unused_over_long_tokens']:,}  -> "
              f"{'COVERED' if s['supply_covered'] else 'SHORT (padding rose)'}")
        print(f"  loader                : {r['loader']['windows']:,} windows, "
              f"{r['loader']['tokens_per_s']:,.0f} tok/s, {r['loader']['windows_per_s']:.0f} win/s")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
