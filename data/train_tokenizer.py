"""Train custom Pulsar-Tokenizer (49,152 vocab) on the 5-Pillar NVIDIA Nemotron Stack.

Features:
- Pure Power-of-2 Vocab: 49,152 tokens
- Byte-Level BPE (ByteLevel pre-tokenizer & decoder)
- GPT-4 / Llama-3 regex pattern with strict individual digit splitting (0-9)
- Preserves multi-space indentation runs
- 12 Reserved Special Tokens:
    0: <|endoftext|>
    1: <|im_start|>
    2: <|im_end|>
    3: <|think|>
    4: <|/think|>
    5: <|tool_call|>
    6: <|/tool_call|>
    7: <|tool_response|>
    8: <|/tool_response|>
    9: <|system|>
    10: <|user|>
    11: <|assistant|>
- 350MB balanced corpus across 5 pillars (30% Code, 25% Math, 15% Agentic, 22% STEM, 8% Chat)
- Streaming disk-buffered architecture (RAM footprint < 500 MB)
- Saves to data/pulsar_tokenizer/ and evaluates empirical bytes per token.
"""

import gc
import json
import os
import random
import time
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, Regex
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast, AutoTokenizer


SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|think|>",
    "<|/think|>",
    "<|tool_call|>",
    "<|/tool_call|>",
    "<|tool_response|>",
    "<|/tool_response|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
]

# GPT-4 / Llama-3 regex pattern with strict individual digit splitting (\p{N}{1})
# and multi-space indentation preservation
REGEX_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}]+|\p{N}{1}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


def format_row_to_text(row: dict) -> str:
    """Universal Schema Formatter mapping heterogeneous multi-turn/reasoning schemas into unified tokens."""
    # 1. Direct text fields
    if "text" in row and isinstance(row["text"], str) and row["text"].strip():
        return row["text"].strip()

    if "improved_text" in row and isinstance(row["improved_text"], str) and row["improved_text"].strip():
        return row["improved_text"].strip()
    if "improved_code" in row and isinstance(row["improved_code"], str) and row["improved_code"].strip():
        return row["improved_code"].strip()

    # 2. Problem / prompt + generated_solution / output (OpenCodeReasoning & OpenMathReasoning)
    if "input" in row and "output" in row:
        inp = str(row["input"]).strip()
        out = str(row["output"]).strip()
        return f"<|im_start|>user\n{inp}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>"

    if "problem_prompt" in row and "generated_solution" in row:
        p = str(row["problem_prompt"]).strip()
        s = str(row["generated_solution"]).strip()
        return f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n{s}<|im_end|>"

    if "problem" in row and "generated_solution" in row:
        p = str(row["problem"]).strip()
        s = str(row["generated_solution"]).strip()
        return f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n{s}<|im_end|>"

    if "problem" in row and "solution" in row:
        p = str(row["problem"]).strip()
        s = str(row["solution"]).strip()
        return f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n{s}<|im_end|>"

    if "question" in row and "cot" in row:
        q = str(row["question"]).strip()
        cot = str(row["cot"]).strip()
        ans = str(row.get("answer", "")).strip()
        return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<|think|>\n{cot}\n<|/think|>\n{ans}<|im_end|>"

    if "instruction" in row and "output" in row:
        inst = str(row["instruction"]).strip()
        inp = f"\n{row['input']}" if row.get("input") else ""
        out = str(row["output"]).strip()
        return f"<|im_start|>user\n{inst}{inp}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>"

    # 3. Messages / Conversations format (Nemotron-Post-Training & Toucan-1.5M)
    msgs = row.get("messages") or row.get("conversations")
    if msgs is not None:
        if isinstance(msgs, str):
            try:
                msgs = json.loads(msgs)
            except Exception:
                msgs = None
        if isinstance(msgs, (list, np.ndarray)):
            turns = []
            for turn in msgs:
                if not isinstance(turn, dict):
                    continue
                role = str(turn.get("role") or turn.get("from") or "user").lower()
                if role in ("human", "user"):
                    r_tag = "user"
                elif role in ("gpt", "assistant", "model"):
                    r_tag = "assistant"
                elif role in ("system",):
                    r_tag = "system"
                elif role in ("tool", "tool_response"):
                    r_tag = "tool_response"
                else:
                    r_tag = role

                content = str(turn.get("content") or turn.get("value") or "").strip()
                if "tool_calls" in turn and turn["tool_calls"]:
                    tc = turn["tool_calls"]
                    tc_str = json.dumps(tc, ensure_ascii=False) if isinstance(tc, (list, dict)) else str(tc)
                    content = f"<|tool_call|>\n{tc_str}\n<|/tool_call|>\n{content}".strip()

                turns.append(f"<|im_start|>{r_tag}\n{content}<|im_end|>")
            if turns:
                return "\n".join(turns)

    # 4. Fallback string extraction
    candidates = [str(v).strip() for v in row.values() if isinstance(v, (str, list, dict))]
    candidates = [c for c in candidates if len(c) > 20]
    if candidates:
        return max(candidates, key=len)
    return str(row)


def find_snapshot_file(repo_pattern: str, file_pattern: str) -> str:
    """Find a cached file in ~/.cache/huggingface/hub/."""
    hub_dir = os.path.expanduser("~/.cache/huggingface/hub/")
    for root, dirs, files in os.walk(hub_dir):
        if repo_pattern.replace("/", "--") in root:
            for f in files:
                if file_pattern in f:
                    return os.path.join(root, f)
    raise FileNotFoundError(f"Could not find cached file matching {repo_pattern} -> {file_pattern}")


def stream_parquet_to_file(fpath: str, out_file, max_bytes: int, batch_size: int = 500) -> tuple[int, int]:
    """Stream formatted text documents from a parquet file directly into out_file using tiny batches."""
    pf = pq.ParquetFile(fpath)
    written_bytes = 0
    doc_count = 0
    for batch in pf.iter_batches(batch_size=batch_size):
        pydict = batch.to_pydict()
        del batch
        keys = list(pydict.keys())
        num_rows = len(pydict[keys[0]])
        for i in range(num_rows):
            row = {k: pydict[k][i] for k in keys}
            text = format_row_to_text(row)
            if text and len(text) > 10:
                t_bytes = len(text.encode("utf-8"))
                out_file.write(text + "\n<|endoftext|>\n")
                written_bytes += t_bytes
                doc_count += 1
                if written_bytes >= max_bytes:
                    del pydict
                    gc.collect()
                    return written_bytes, doc_count
        del pydict
        gc.collect()
    return written_bytes, doc_count


def prepare_balanced_corpus(corpus_path: str, total_target_mb: float = 350.0):
    """Write balanced corpus to disk without blowing up RAM."""
    os.makedirs(os.path.dirname(corpus_path) or ".", exist_ok=True)
    p1_mb = total_target_mb * 0.30  # 105 MB Code
    p2_mb = total_target_mb * 0.25  # 87.5 MB Math
    p3_mb = total_target_mb * 0.15  # 52.5 MB Agentic
    p4_mb = total_target_mb * 0.22  # 77.0 MB STEM
    p5_mb = total_target_mb * 0.08  # 28.0 MB Chat

    print(f"[Tokenizer] Writing {total_target_mb:.1f} MB Corpus to '{corpus_path}'...")
    print(f"  ↳ Pillar 1 (Code 30%):        {p1_mb:.1f} MB")
    print(f"  ↳ Pillar 2 (Math 25%):        {p2_mb:.1f} MB")
    print(f"  ↳ Pillar 3 (Agentic MCP 15%): {p3_mb:.1f} MB")
    print(f"  ↳ Pillar 4 (STEM 22%):        {p4_mb:.1f} MB")
    print(f"  ↳ Pillar 5 (Reasoning 8%):    {p5_mb:.1f} MB")

    total_docs = 0
    total_bytes = 0

    with open(corpus_path, "w", encoding="utf-8") as out:
        # Pillar 1: OpenCodeReasoning
        f_p1 = find_snapshot_file("OpenCodeReasoning", "train-00000-of-00030.parquet")
        print(f"\n[Pillar 1 - Code] Streaming from {os.path.basename(f_p1)}...")
        b, d = stream_parquet_to_file(f_p1, out, max_bytes=int(p1_mb * 1024 * 1024))
        print(f"  Streamed {d:,} docs ({b / 1e6:.1f} MB)")
        total_docs += d
        total_bytes += b
        gc.collect()

        # Pillar 2: Nemotron-CC-Math-v1 (50%) + OpenMathReasoning (50%)
        f_p2_cc = find_snapshot_file("Nemotron-CC-Math-v1", "part_000000.parquet")
        f_p2_om = find_snapshot_file("OpenMathReasoning", "cot-00000-of-00144.parquet")
        print(f"\n[Pillar 2 - Math] Streaming from CC-Math and OpenMathReasoning...")
        b1, d1 = stream_parquet_to_file(f_p2_cc, out, max_bytes=int(p2_mb * 0.5 * 1024 * 1024))
        b2, d2 = stream_parquet_to_file(f_p2_om, out, max_bytes=int(p2_mb * 0.5 * 1024 * 1024))
        print(f"  Streamed {d1 + d2:,} docs ({(b1 + b2) / 1e6:.1f} MB)")
        total_docs += (d1 + d2)
        total_bytes += (b1 + b2)
        gc.collect()

        # Pillar 3: Toucan-1.5M (Agentic MCP)
        f_p3 = find_snapshot_file("Toucan-1.5M", "train-00000-of-00040.parquet")
        print(f"\n[Pillar 3 - Agentic] Streaming from {os.path.basename(f_p3)}...")
        b, d = stream_parquet_to_file(f_p3, out, max_bytes=int(p3_mb * 1024 * 1024))
        print(f"  Streamed {d:,} docs ({b / 1e6:.1f} MB)")
        total_docs += d
        total_bytes += b
        gc.collect()

        # Pillar 4: Nemotron-Pretraining-Specialized-v1 (STEM & Knowledge)
        f_p4 = find_snapshot_file("Nemotron-Pretraining-Specialized-v1", "part_000000.parquet")
        print(f"\n[Pillar 4 - STEM] Streaming from {os.path.basename(f_p4)}...")
        b, d = stream_parquet_to_file(f_p4, out, max_bytes=int(p4_mb * 1024 * 1024))
        print(f"  Streamed {d:,} docs ({b / 1e6:.1f} MB)")
        total_docs += d
        total_bytes += b
        gc.collect()

        # Pillar 5: Nemotron-Post-Training-Dataset-v2 (Chat & Reasoning)
        f_p5 = find_snapshot_file("Nemotron-Post-Training-Dataset-v2", "chat-00000-of-00012.parquet")
        print(f"\n[Pillar 5 - Chat] Streaming from {os.path.basename(f_p5)}...")
        b, d = stream_parquet_to_file(f_p5, out, max_bytes=int(p5_mb * 1024 * 1024))
        print(f"  Streamed {d:,} docs ({b / 1e6:.1f} MB)")
        total_docs += d
        total_bytes += b
        gc.collect()

    print(f"\n[Corpus Complete] {total_docs:,} Documents | {total_bytes / 1e6:.1f} MB on disk.")


def train_pulsar_tokenizer(save_dir: str = "data/pulsar_tokenizer"):
    corpus_file = "data/raw/tokenizer_corpus.txt"
    t0 = time.time()
    prepare_balanced_corpus(corpus_file, total_target_mb=350.0)

    print("\n[Pulsar-Tokenizer] Initializing Byte-Level BPE with Digit-Splitting Regex...")
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=Regex(REGEX_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=32768,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"[Pulsar-Tokenizer] Training Rust Byte-Level BPE from '{corpus_file}' (Vocab: 32,768)...")
    tok.train(files=[corpus_file], trainer=trainer)
    t_train = time.time() - t0
    print(f"[Pulsar-Tokenizer] Training completed in {t_train:.1f}s!")

    # Clean up temporary corpus file
    if os.path.exists(corpus_file):
        os.remove(corpus_file)
        print(f"[Pulsar-Tokenizer] Cleaned up temporary corpus file '{corpus_file}'")

    os.makedirs(save_dir, exist_ok=True)
    # Wrap with PreTrainedTokenizerFast for seamless Hugging Face compatibility
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        clean_up_tokenization_spaces=False,
    )
    fast_tok.save_pretrained(save_dir)
    print(f"[Pulsar-Tokenizer] Saved Hugging Face PreTrainedTokenizerFast to '{save_dir}'")

    # Save raw vocab.txt for fallback tooling
    vocab = tok.get_vocab()
    vocab_txt_path = os.path.join("data", "vocab.txt")
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    with open(vocab_txt_path, "w", encoding="utf-8") as f:
        for token, idx in sorted_vocab:
            f.write(f"{token}\n")
    print(f"[Pulsar-Tokenizer] Updated 'data/vocab.txt' ({len(sorted_vocab):,} tokens)")

    # Verification & Empirical Byte/Token Compression Evaluation
    print("\n=======================================================")
    print("      PULSAR-TOKENIZER VERIFICATION & BENCHMARK        ")
    print("=======================================================")
    loaded_tok = AutoTokenizer.from_pretrained(save_dir)
    print(f"Loaded Tokenizer:    {type(loaded_tok).__name__}")
    print(f"Total Vocab Size:    {len(loaded_tok):,} tokens")
    print(f"Special Tokens ({len(SPECIAL_TOKENS)}):")
    for st in SPECIAL_TOKENS:
        tid = loaded_tok.convert_tokens_to_ids(st)
        print(f"   {st:<20} -> ID: {tid}")

    # Digit splitting verification
    digit_test = "2026 = 1984 + 42 * 1"
    digit_tokens = [loaded_tok.decode([i]) for i in loaded_tok.encode(digit_test)]
    print(f"\nDigit Splitting Test: '{digit_test}'")
    print(f"  Tokens: {digit_tokens}")
    has_single_digits = all(len(t.strip()) <= 1 for t in digit_tokens if t.strip().isdigit())
    print(f"  Strict Single Digit Splitting Verified: {has_single_digits}")

    # Multi-space indentation verification
    indent_test = "def function():\n    x = 1\n        y = 2"
    enc_indent = loaded_tok.encode(indent_test)
    dec_indent = loaded_tok.decode(enc_indent)
    print(f"\nIndentation Roundtrip Test: {dec_indent == indent_test}")

    # Empirical bytes per token (b_bar) evaluation
    test_benchmarks = {
        "Python Code": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
        "Math & LaTeX": r"\text{For } \alpha, \beta > 0, \text{ the Beta function satisfies } B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha + \beta)} = \int_0^1 t^{\alpha - 1} (1-t)^{\beta - 1} dt.",
        "Agentic MCP / JSON": '{"tool_call": {"name": "browser_navigate", "arguments": {"url": "https://huggingface.co/nvidia", "timeout": 30000}}, "status": "pending"}',
        "STEM Science": "Photosystem II absorbs light energy to split water molecules into oxygen, protons, and electrons through photolysis within the thylakoid membrane.",
        "Assistant Reasoning": "<|im_start|>user\nSolve the equation 3x^2 - 12x + 9 = 0.<|im_end|>\n<|im_start|>assistant\n<|think|>\nDivide through by 3: x^2 - 4x + 3 = 0. Factor as (x - 1)(x - 3) = 0. Therefore x = 1 or x = 3.\n<|/think|>\nThe roots are x = 1 and x = 3.<|im_end|>",
    }

    print("\n--- Empirical Tokenization Efficiency (Bytes / Token: b̄) ---")
    tot_bytes = 0
    tot_tokens = 0
    for domain, text in test_benchmarks.items():
        raw_b = len(text.encode("utf-8"))
        toks = len(loaded_tok.encode(text))
        b_bar = raw_b / max(1, toks)
        tot_bytes += raw_b
        tot_tokens += toks
        print(f"  {domain:<20} {raw_b:4d} bytes | {toks:3d} tokens | {b_bar:4.2f} bytes/token")

    overall_b_bar = tot_bytes / max(1, tot_tokens)
    print(f"\n  Overall Empirical Compression: b̄ = {overall_b_bar:.2f} bytes / token")
    print("=======================================================\n")


if __name__ == "__main__":
    train_pulsar_tokenizer()
