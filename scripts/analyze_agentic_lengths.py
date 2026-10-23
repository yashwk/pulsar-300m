#!/usr/bin/env python3
"""WI-13 analysis: measure the cost of un-eliding reasoning traces.

Two independent modes:

  --mode shards   Decode reasoning documents already stored in the ``.bin`` shards
                  (via the tokenizer) and report length distributions, retention at
                  several context lengths, and multi-call share. No network access.

  --mode dry-run  Stream the HF sources for a given stage/shard using the *real*
                  builders, measure both the new (un-elided) and legacy (mid-elided)
                  output, and print the diversity trade-off: documents and tokens per
                  stream, un-elided token share, and document counts vs the elided build.

Nothing here writes shards. The dry-run is read-only.

Examples
--------
    python scripts/analyze_agentic_lengths.py --mode shards --limit-tokens 2000000
    python scripts/analyze_agentic_lengths.py --mode dry-run --stage 1 --shard-idx 1 --docs-per-stream 120
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_builder():
    """Import data/prepare_shards.py as a module (it lives outside a package)."""
    path = os.path.join(ROOT, "data", "prepare_shards.py")
    spec = importlib.util.spec_from_file_location("prepare_shards", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REASONING_STREAMS = ("1B", "2B", "3A")


# ---------------------------------------------------------------------------
# Mode 1: audit the shards already on disk
# ---------------------------------------------------------------------------

def mode_shards(args) -> int:
    tok = _tokenizer(args.tokenizer)
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0

    report = {}
    for path in args.shards:
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue
        arr = np.memmap(path, dtype=np.uint16, mode="r")
        n = len(arr)
        take = min(n, args.limit_tokens)
        sub = np.asarray(arr[:take])

        eos_pos = np.flatnonzero(sub == eos)
        if eos_pos.size == 0:
            print(f"[warn] {path}: no EOS found")
            continue
        bounds = np.concatenate(([0], eos_pos + 1))
        doc_lens = np.diff(bounds)
        doc_lens = doc_lens[doc_lens > args.min_doc_tokens]

        elision_marker = tok.encode("[...]", add_special_tokens=False)
        marker_hits = _count_subsequence(sub, elision_marker)

        stats = {
            "tokens_scanned": int(take),
            "documents": int(doc_lens.size),
            "mean_doc_tokens": float(doc_lens.mean()) if doc_lens.size else 0.0,
            "p50_doc_tokens": int(np.median(doc_lens)) if doc_lens.size else 0,
            "p90_doc_tokens": int(np.percentile(doc_lens, 90)) if doc_lens.size else 0,
            "max_doc_tokens": int(doc_lens.max()) if doc_lens.size else 0,
            "elision_markers": int(marker_hits),
            "elision_marker_rate_per_1k_docs": round(1000.0 * marker_hits / max(1, doc_lens.size), 2),
            "retention_pct": {
                f"T{T}": round(100.0 * float((doc_lens <= T).mean()), 2)
                for T in (1024, 2048, 4096, 8192, 16384)
            } if doc_lens.size else {},
        }
        report[path] = stats
        print(f"\n=== {path} ===")
        print(json.dumps(stats, indent=2))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n[written] {args.out}")
    return 0


def _count_subsequence(arr: np.ndarray, pat: list[int]) -> int:
    """Count non-overlapping occurrences of a token subsequence."""
    if not pat:
        return 0
    hits = 0
    i = 0
    np_pat = np.asarray(pat, dtype=arr.dtype)
    m = len(np_pat)
    while i <= len(arr) - m:
        window = arr[i:i + m]
        if np.array_equal(window, np_pat):
            hits += 1
            i += m
        else:
            i += 1
    return hits


def _tokenizer(path: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path)


# ---------------------------------------------------------------------------
# Mode 2: dry-run the builders (no shards written)
# ---------------------------------------------------------------------------

def mode_dry_run(args) -> int:
    ps = load_builder()
    tok = _tokenizer(args.tokenizer)

    cfg = ps.substream_config_for_stage(args.stage)
    keys = list(cfg.keys())
    quota = {s: int(round(cfg[s]["weight"] / 100.0 * args.total_tokens)) for s in keys}
    quota[keys[0]] += args.total_tokens - sum(quota.values())

    print("=" * 78)
    print(f"DRY RUN (no shards written) | stage={args.stage} shard={args.shard_idx} "
          f"total={args.total_tokens:,}")
    print(f"mix={'long-context reasoning-heavy' if args.stage >= 3 else 'Nemotron 80/20'}")
    print(f"trace policy: un-elided, cap {ps.MAX_TRACE_CHARS:,} chars")
    print("=" * 78)
    print(f"\n{'stream':6s} {'quota_tok':>12s} {'docs_new':>9s} {'docs_old':>9s} "
          f"{'ratio':>7s} {'tok/doc_new':>12s} {'tok/doc_old':>12s} {'x':>6s}")
    print("-" * 92)

    summary = {}
    for s in keys:
        new_tok, new_docs = 0, 0
        old_tok, old_docs = 0, 0
        gen_new = ps.make_substream_stream(s, num_shards=args.num_shards, shard_idx=args.shard_idx,
                                           batch_size=args.batch_size, base_seed=args.seed)
        gen_old = ps.make_substream_stream(s, num_shards=args.num_shards, shard_idx=args.shard_idx,
                                           batch_size=args.batch_size, base_seed=args.seed)
        seen = 0
        # Walk both streams together: same document sequence, two formatting policies.
        while new_tok < quota[s] and seen < args.max_docs_per_stream:
            try:
                txt = next(gen_new)
            except StopIteration:
                break
            seen += 1
            enc = tok.encode(txt, add_special_tokens=False)
            new_tok += len(enc) + 1
            new_docs += 1

            old_doc = _legacy_render(ps, s, txt)
            old_tok += len(tok.encode(old_doc, add_special_tokens=False)) + 1
            old_docs += 1

        # Documents the *old* policy would have needed to fill the same token quota.
        docs_old_needed = (old_docs * quota[s] / old_tok) if old_tok else 0.0
        tpd_new = new_tok / new_docs if new_docs else 0.0
        tpd_old = old_tok / old_docs if old_docs else 0.0
        ratio = (new_docs / docs_old_needed) if docs_old_needed else 0.0

        print(f"{s:6s} {quota[s]:12,d} {new_docs:9,d} {docs_old_needed:9,.0f} "
              f"{ratio:7.2f} {tpd_new:12,.0f} {tpd_old:12,.0f} {tpd_new / tpd_old if tpd_old else 0:6.2f}")
        summary[s] = {
            "quota_tokens": quota[s],
            "docs_new": new_docs,
            "docs_new_est_for_quota": int(round(new_docs * quota[s] / new_tok)) if new_tok else 0,
            "docs_old_est_for_quota": int(round(docs_old_needed)),
            "doc_retention_ratio_new_vs_old": round(ratio, 3),
            "tokens_per_doc_new": int(round(tpd_new)),
            "tokens_per_doc_old": int(round(tpd_old)),
            "token_inflation_x": round(tpd_new / tpd_old, 2) if tpd_old else None,
        }

    reasoning_quota = sum(quota[s] for s in REASONING_STREAMS if s in quota)
    print("-" * 92)
    print(f"reasoning streams (1B+2B+3A) quota: {reasoning_quota:,} tokens "
          f"({100.0 * reasoning_quota / args.total_tokens:.1f}% of shard)")
    print("un-elided token share = same quotas; document counts drop in proportion to "
          "token inflation (that is the diversity trade).")

    out = {
        "mode": "dry-run",
        "stage": args.stage,
        "shard_idx": args.shard_idx,
        "total_tokens": args.total_tokens,
        "seed": args.seed,
        "trace_policy": "un-elided",
        "max_trace_chars": ps.MAX_TRACE_CHARS,
        "streams": summary,
        "reasoning_quota_tokens": reasoning_quota,
        "reasoning_share_pct": round(100.0 * reasoning_quota / args.total_tokens, 2),
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\n[written] {args.out}")
    return 0


def _legacy_render(ps, stream: str, text: str) -> str:
    """Re-render a document the way the pre-WI-13 pipeline would have.

    The builders return already-formatted strings, so for 1B/2B we re-apply the
    legacy mid-elision to the think block; every other stream is unchanged.
    """
    if stream not in ("1B", "2B"):
        return text
    if "<|think|>" not in text or "<|/think|>" not in text:
        return text
    pre, rest = text.split("<|think|>", 1)
    think, post = rest.split("<|/think|>", 1)
    return f"{pre}<|think|>\n{ps.legacy_clean_think_trace(think.strip())}\n<|/think|>{post}"


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["shards", "dry-run"], default="shards")
    ap.add_argument("--tokenizer", default="data/pulsar_tokenizer")
    ap.add_argument("--out", default=None, help="Optional JSON report path")

    g = ap.add_argument_group("mode=shards")
    g.add_argument("--shards", nargs="*", default=["data/train_shard1.bin"])
    g.add_argument("--limit-tokens", type=int, default=2_000_000)
    g.add_argument("--min-doc-tokens", type=int, default=2)

    d = ap.add_argument_group("mode=dry-run")
    d.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    d.add_argument("--shard-idx", type=int, default=1)
    d.add_argument("--num-shards", type=int, default=12)
    d.add_argument("--total-tokens", type=int, default=500_000_000)
    d.add_argument("--batch-size", type=int, default=512)
    d.add_argument("--max-docs-per-stream", type=int, default=400)
    d.add_argument("--seed", type=int, default=20261023)

    args = ap.parse_args()
    if args.mode == "shards":
        return mode_shards(args)
    return mode_dry_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
