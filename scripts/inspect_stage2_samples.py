#!/usr/bin/env python3
"""Inspect real formatted Stage 2 samples across all 8 substreams.

Pulls 2 real formatted documents per substream (1B, 1A, 2B, 2A, 3A, 3B, 4B, 5B)
with --max-doc-tokens 4096, tokenizes with data/pulsar_tokenizer, runs
_apply_prompt_loss_mask, and reports exact formatted text, token counts,
and unmasked loss percentages for visual inspection.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from transformers import AutoTokenizer

from data.prepare_shards import make_substream_stream
from src.packing import (
    IGNORE_INDEX,
    _apply_prompt_loss_mask,
)

SUBSTREAMS = [
    ("1B", "nvidia/OpenCodeReasoning (Code CoT)"),
    ("1A", "codeparrot/github-code-clean (Big-5 Replay)"),
    ("2B", "nvidia/OpenMathInstruct-2 (Math CoT)"),
    ("2A", "HuggingFaceTB/finemath-4plus (Math Replay)"),
    ("3A", "Agent-Ark/Toucan-1.5M (Agentic Tool-Use)"),
    ("3B", "open-thoughts/TaskTrove (Agentic Env Tasks)"),
    ("4B", "HuggingFaceFW/fineweb-edu (STEM Knowledge Replay)"),
    ("5B", "open-thoughts/OpenThoughts-114k (Deep Dialogue CoT)"),
]

def main():
    tokenizer_path = "data/pulsar_tokenizer"
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    eos_id = tokenizer.eos_token_id or 0

    assistant_role_ids = {
        tokenizer.encode("assistant", add_special_tokens=False)[0],
        tokenizer.encode(" assistant", add_special_tokens=False)[0],
    }
    newline_ids = {tokenizer.encode("\n", add_special_tokens=False)[0]}

    max_doc_tokens = 4096
    num_samples_per_substream = 2

    report_lines = []
    report_lines.append("# Stage 2 Formatted Substream Sample Inspection Report")
    report_lines.append(f"- Tokenizer: `{tokenizer_path}`")
    report_lines.append(f"- Max Doc Tokens: `{max_doc_tokens}`")
    report_lines.append(f"- Assistant Role IDs: `{sorted(list(assistant_role_ids))}`")
    report_lines.append(f"- Newline IDs: `{sorted(list(newline_ids))}`\n")

    for sid, desc in SUBSTREAMS:
        print(f"\nProcessing Substream [{sid}] {desc}...", flush=True)
        report_lines.append(f"## Substream {sid}: {desc}\n")

        gen = make_substream_stream(sid, num_shards=12, shard_idx=0, stage=2)
        valid_samples = []

        while len(valid_samples) < num_samples_per_substream:
            try:
                doc = next(gen)
            except StopIteration:
                break
            except Exception as e:
                print(f"Error fetching from {sid}: {e}", flush=True)
                break

            doc = doc.strip()
            if not doc:
                continue

            tok_ids = tokenizer.encode(doc, add_special_tokens=False) + [eos_id]
            if len(tok_ids) > max_doc_tokens:
                continue
            if len(tok_ids) < 10:
                continue

            toks = np.array(tok_ids, dtype=np.int64)
            L = len(toks)

            # Build causal targets
            y = np.full(L, IGNORE_INDEX, dtype=np.int64)
            if L > 1:
                y[: L - 1] = toks[1:L]

            # Apply prompt loss mask
            y_masked = _apply_prompt_loss_mask(
                toks,
                y,
                L,
                assistant_role_ids=assistant_role_ids,
                newline_ids=newline_ids,
                eos_id=eos_id,
            )

            unmasked_count = int(np.sum(y_masked != IGNORE_INDEX))
            unmasked_pct = (unmasked_count / max(1, L - 1)) * 100.0

            valid_samples.append({
                "text": doc,
                "tokens": L,
                "unmasked_count": unmasked_count,
                "unmasked_pct": unmasked_pct,
                "has_chatml": ("<|im_start|>" in doc),
            })
            print(f"  Sample {len(valid_samples)}: {L} tokens, {unmasked_pct:.1f}% unmasked", flush=True)

        for i, s in enumerate(valid_samples, 1):
            report_lines.append(f"### Sample {i} (Length: {s['tokens']} tokens, Unmasked: {s['unmasked_count']}/{s['tokens'] - 1} = {s['unmasked_pct']:.1f}%, ChatML: {s['has_chatml']})\n")
            report_lines.append("```text")
            # If sample text is very long, show complete prompt and start/end of thought and solution
            text = s["text"]
            report_lines.append(text)
            report_lines.append("```\n")

    out_path = "data/stage2_samples_inspection.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\nInspection complete! Written to {out_path}")

if __name__ == "__main__":
    main()
