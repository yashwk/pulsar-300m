#!/usr/bin/env python3
"""Pulsar Concrete Step & Token Calculator.

Computes exact step counts, token counts, gradient accumulation steps (GAS),
warmup/cooldown horizons, and estimated runtimes from high-level token budgets
and shard definitions.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys


def parse_tokens(val: str | int | float) -> int:
    """Parse human token strings like '3.8B', '4M', '131k', '131072' into integer token count."""
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val).strip().upper()
    if s.endswith("B"):
        return int(float(s[:-1]) * 1_000_000_000)
    elif s.endswith("M"):
        return int(float(s[:-1]) * 1_000_000)
    elif s.endswith("K"):
        return int(float(s[:-1]) * 1_000)
    return int(float(s))


def format_tokens(n: int) -> str:
    """Format token count into human-readable string (e.g. 3.80B, 4.00M, 131.07K)."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B ({n:,})"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M ({n:,})"
    elif n >= 1_000:
        return f"{n / 1_000:.2f}K ({n:,})"
    return f"{n:,}"


def format_time(seconds: float) -> str:
    """Format duration in seconds into human-readable hours, minutes, seconds."""
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    elif s < 3600:
        m, sec = divmod(s, 60)
        return f"{m}m {sec:02d}s"
    else:
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        return f"{h}h {m:02d}m"


def count_shard_tokens(shard_paths: list[str]) -> tuple[int, dict[str, int]]:
    """Calculate token count of binary shards (uint16 format = 2 bytes per token)."""
    total = 0
    per_shard = {}
    for p in shard_paths:
        if os.path.exists(p):
            size = os.path.getsize(p)
            toks = size // 2
            per_shard[os.path.basename(p)] = toks
            total += toks
    return total, per_shard


def calculate_dynamics(
    total_tokens: int,
    global_batch_tokens: int = 131_072,
    micro_batch_size: int = 128,
    seq_len: int = 1024,
    target_tokens: int | None = None,
    target_step: int | None = None,
    warmup_ratio: float = 0.01,
    warmup_tokens: int | None = None,
) -> dict:
    tokens_per_micro = micro_batch_size * seq_len
    grad_accum = max(1, (global_batch_tokens + tokens_per_micro - 1) // tokens_per_micro)
    effective_gbt = grad_accum * tokens_per_micro

    # Total curriculum steps
    max_steps = max(1, total_tokens // effective_gbt)

    # Target steps
    if target_tokens is not None:
        effective_target_step = max(1, (target_tokens + effective_gbt - 1) // effective_gbt)
    elif target_step is not None:
        effective_target_step = target_step
    else:
        effective_target_step = max_steps

    target_tokens_seen = effective_target_step * effective_gbt

    # Warmup steps
    if warmup_tokens is not None:
        warmup_steps = max(1, warmup_tokens // effective_gbt)
    else:
        warmup_steps = max(1, int(round(max_steps * warmup_ratio)))

    return {
        "total_tokens": total_tokens,
        "global_batch_tokens": global_batch_tokens,
        "micro_batch_size": micro_batch_size,
        "seq_len": seq_len,
        "tokens_per_micro": tokens_per_micro,
        "grad_accum": grad_accum,
        "effective_global_batch_tokens": effective_gbt,
        "max_steps": max_steps,
        "curriculum_tokens_covered": max_steps * effective_gbt,
        "target_step": effective_target_step,
        "target_tokens": target_tokens_seen,
        "warmup_steps": warmup_steps,
        "warmup_tokens": warmup_steps * effective_gbt,
    }


def print_report(dyn: dict, speeds: list[int] = None):
    if speeds is None:
        speeds = [40_000, 60_000, 80_000, 100_000]

    print("=" * 72)
    print(" PULSAR CONCRETE STEP & TOKEN CALCULATOR")
    print("=" * 72)
    print(f"Total Curriculum Tokens:   {format_tokens(dyn['total_tokens'])}")
    print(f"Sequence Context Length:   {dyn['seq_len']:,} tokens")
    print(f"Micro-Batch Size (MBS):    {dyn['micro_batch_size']}")
    print(f"Tokens per Micro-Batch:    {dyn['tokens_per_micro']:,}")
    print(f"Gradient Accumulation Steps: GAS = {dyn['grad_accum']}")
    print(f"Global Batch Tokens (GBT): {dyn['effective_global_batch_tokens']:,} tokens/step")
    print("-" * 72)
    print(f"Total Curriculum Steps:    {dyn['max_steps']:,} steps")
    print(f"Curriculum Tokens Trained: {format_tokens(dyn['curriculum_tokens_covered'])}")
    print(f"Warmup Schedule:           {dyn['warmup_steps']:,} steps ({format_tokens(dyn['warmup_tokens'])})")
    print("-" * 72)
    print(f"Active Session Target:     Step {dyn['target_step']:,} ({format_tokens(dyn['target_tokens'])})")
    print("=" * 72)
    print(" ESTIMATED RUNTIMES")
    print("-" * 72)
    print(f"{'Throughput':<16} | {'Target Step (' + str(dyn['target_step']) + ')':<20} | {'Full Run (' + str(dyn['max_steps']) + ' steps)':<20}")
    print("-" * 72)
    for tps in speeds:
        target_sec = dyn["target_tokens"] / tps
        full_sec = dyn["curriculum_tokens_covered"] / tps
        print(f"{tps:>7,d} tok/s    | {format_time(target_sec):<20} | {format_time(full_sec):<20}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Pulsar Token & Step Calculator")
    parser.add_argument("--tokens", type=str, default="3.8B", help="Total curriculum tokens (e.g. 3.8B, 4000M)")
    parser.add_argument("--shards", type=str, default=None, help="Glob pattern or comma-separated list of .bin shard files")
    parser.add_argument("--global-batch-tokens", type=str, default="131072", help="Global batch tokens per step")
    parser.add_argument("--mbs", type=int, default=128, help="Micro batch size")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--target-tokens", type=str, default=None, help="Target tokens to pause (e.g. 4M, 100M)")
    parser.add_argument("--target-step", type=int, default=30, help="Target step to pause (e.g. 30)")
    parser.add_argument("--warmup-ratio", type=float, default=0.01, help="Warmup ratio of total curriculum (default 0.01 = 1%%)")
    parser.add_argument("--warmup-tokens", type=str, default=None, help="Explicit warmup token budget (e.g. 38M)")

    args = parser.parse_args()

    gbt = parse_tokens(args.global_batch_tokens)
    target_tok = parse_tokens(args.target_tokens) if args.target_tokens is not None else None
    warmup_tok = parse_tokens(args.warmup_tokens) if args.warmup_tokens is not None else None

    if args.shards:
        paths = []
        for p in args.shards.split(","):
            expanded = glob.glob(p.strip())
            paths.extend(expanded if expanded else [p.strip()])
        total_tokens, per_shard = count_shard_tokens(paths)
        if total_tokens > 0:
            print(f"[Shards] Found {len(per_shard)} shards totaling {format_tokens(total_tokens)}")
        else:
            total_tokens = parse_tokens(args.tokens)
    else:
        total_tokens = parse_tokens(args.tokens)

    dyn = calculate_dynamics(
        total_tokens=total_tokens,
        global_batch_tokens=gbt,
        micro_batch_size=args.mbs,
        seq_len=args.seq_len,
        target_tokens=target_tok,
        target_step=args.target_step if target_tok is None else None,
        warmup_ratio=args.warmup_ratio,
        warmup_tokens=warmup_tok,
    )

    print_report(dyn)


if __name__ == "__main__":
    main()
