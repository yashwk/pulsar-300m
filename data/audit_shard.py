"""Automated Audit Script for Pulsar Curriculum Shards.

Verifies:
  1. Special Token IDs 0..8 (<|endoftext|>, <|im_start|>, <|im_end|>, <|think|>, <|/think|>,
     <|tool_call|>, <|/tool_call|>, <|tool_response|>, <|/tool_response|>) frequencies (> 1,000 required).
  2. Decoded samples for visual verification (Direct Code 1A, Agentic Tool Call 3A, FineWiki 4A).
"""

import argparse
import numpy as np
from transformers import AutoTokenizer

SPECIAL_TOKEN_NAMES = {
    0: "<|endoftext|>",
    1: "<|im_start|>",
    2: "<|im_end|>",
    3: "<|think|>",
    4: "<|/think|>",
    5: "<|tool_call|>",
    6: "<|/tool_call|>",
    7: "<|tool_response|>",
    8: "<|/tool_response|>",
    9: "<|filler_pad|>",
    10: "<|latent_cot_start|>",
    11: "<|latent_cot_end|>",
}


def audit_shard(shard_path: str, tokenizer_path: str = "data/pulsar_tokenizer", min_count: int = 1000):
    print("==================================================================")
    print(f"   AUDIT OF SHARD: {shard_path}   ")
    print("==================================================================")

    tok = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load shard
    data = np.fromfile(shard_path, dtype=np.uint16)
    total_tokens = len(data)
    print(f"Total tokens in shard: {total_tokens:,}")

    # Count special tokens
    print("\n--- Special Token Frequencies ---")
    print(f"{'Token ID':<10s} | {'Name':<22s} | {'Count':<10s} | {'Frequency':<12s} | {'Gate Status'}")
    print("-" * 72)

    all_passed = True
    for tid in range(9):
        cnt = int(np.count_nonzero(data == tid))
        name = SPECIAL_TOKEN_NAMES.get(tid, f"ID {tid}")
        freq_str = f"1 in {total_tokens // max(1, cnt):,}" if cnt > 0 else "0 (NONE!)"
        passed = cnt >= min_count
        status_str = f"PASS ({cnt:,} >= {min_count:,})" if passed else f"FAIL ({cnt:,} < {min_count:,})"
        if not passed:
            all_passed = False
        print(f"{tid:<10d} | {name:<22s} | {cnt:<10,d} | {freq_str:<12s} | {status_str}")

    print("-" * 72)
    gate_result = "PASS - ALL SPECIAL TOKENS ACTIVE" if all_passed else "FAIL - SPECIAL TOKENS DEFICIENT"
    print(f"PRE-FLIGHT GATE RESULT: {gate_result}")
    print("==================================================================")

    # Visual Inspection: Find occurrences of 1A (code), 3A (tool_call), and 4A (wiki)
    print("\n--- Visual Inspection: Decoded Samples from Shard ---")

    # Sample 1: Tool Call (Token 5)
    tool_indices = np.where(data == 5)[0]
    if len(tool_indices) > 0:
        idx = tool_indices[0]
        start_idx = max(0, idx - 30)
        end_idx = min(len(data), idx + 100)
        decoded = tok.decode(data[start_idx:end_idx].tolist())
        print("\n[Sample: Agentic Tool Call with <|tool_call|>]")
        print("--------------------------------------------------")
        print(decoded)
        print("--------------------------------------------------")

    # Sample 2: CoT Reasoning (Token 3)
    think_indices = np.where(data == 3)[0]
    if len(think_indices) > 0:
        idx = think_indices[0]
        start_idx = max(0, idx - 20)
        end_idx = min(len(data), idx + 80)
        decoded = tok.decode(data[start_idx:end_idx].tolist())
        print("\n[Sample: CoT Reasoning with <|think|>]")
        print("--------------------------------------------------")
        print(decoded)
        print("--------------------------------------------------")

    # Sample 3: Code / Wiki (Start of file sample)
    decoded_head = tok.decode(data[:120].tolist())
    print("\n[Sample: Shard Header Sample (First 120 Tokens)]")
    print("--------------------------------------------------")
    print(decoded_head)
    print("--------------------------------------------------")

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-path", type=str, default="data/test_preflight_10m.bin")
    parser.add_argument("--tokenizer", type=str, default="data/pulsar_tokenizer")
    parser.add_argument("--min-count", type=int, default=1000)
    args = parser.parse_args()

    audit_shard(args.shard_path, args.tokenizer, args.min_count)
