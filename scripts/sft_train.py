#!/usr/bin/env python3
"""Pulsar Supervised Fine-Tuning (SFT) Cold-Start Alignment Harness.

Stage 4 of the Pulsar Curriculum:
Trains the model on curated instruction-response pairs (CoT reasoning, code, agentic tool calls)
with strict prompt-loss masking so gradients are backpropagated solely on assistant tokens.

Eliminates the "zero-advantage trap" in downstream RLVR / GRPO policy optimization.

Usage:
    python scripts/sft_train.py \
      --checkpoint checkpoints/milestone_step_5000.pt \
      --output-checkpoint checkpoints/pulsar_sft.pt \
      --lr 2e-5 \
      --epochs 3 \
      --batch-size 4 \
      --grad-accum 4 \
      --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.model import PulsarConfig, PulsarForCausalLM


# ---------------------------------------------------------------------------
# Default Curated SFT Demonstration Corpus (Cold-Start Proof of Concept)
# ---------------------------------------------------------------------------

DEFAULT_SFT_EXAMPLES: List[Dict[str, str]] = [
    {
        "system": "You are Pulsar, a helpful and precise reasoning AI assistant.",
        "user": "What is the derivative of f(x) = 3*x^2 + 5*x - 7 with respect to x?",
        "assistant": "<|think|>\nThe function is f(x) = 3*x^2 + 5*x - 7.\nWe apply the power rule of differentiation d/dx(x^n) = n*x^(n-1):\n1. d/dx(3*x^2) = 3 * 2 * x = 6*x\n2. d/dx(5*x) = 5 * 1 = 5\n3. d/dx(-7) = 0\nSumming these terms: f'(x) = 6*x + 5.\n<|/think|>\nThe derivative of f(x) = 3x^2 + 5x - 7 with respect to x is 6x + 5.",
    },
    {
        "system": "You are Pulsar, an expert software engineer specializing in Python.",
        "user": "Write a Python function `is_palindrome(s: str) -> bool` that checks if a string is a palindrome, ignoring non-alphanumeric characters and casing.",
        "assistant": "<|think|>\nWe need to determine if a string is a palindrome while ignoring non-alphanumeric characters and case.\n1. Filter the string: keep only c.isalnum() for c in s, and convert each to lowercase.\n2. Compare the cleaned string with its reverse: cleaned == cleaned[::-1].\n3. Handle edge cases: empty strings or single characters are valid palindromes.\n<|/think|>\n```python\ndef is_palindrome(s: str) -> bool:\n    cleaned = [c.lower() for c in s if c.isalnum()]\n    return cleaned == cleaned[::-1]\n```",
    },
    {
        "system": "You are an autonomous agent with tool execution capabilities. Always invoke tools using exact JSON schema syntax.",
        "user": "Check current disk usage and free memory on the host machine.",
        "assistant": "<|tool_call|>\n{\"name\": \"system_diagnostics\", \"arguments\": {\"check_disk\": true, \"check_memory\": true}}\n<|/tool_call|>",
    },
    {
        "system": "You are Pulsar, a helpful and precise reasoning AI assistant.",
        "user": "Solve for x in the equation: 2*x + 10 = 26.",
        "assistant": "<|think|>\nWe have the equation: 2*x + 10 = 26.\n1. Subtract 10 from both sides: 2*x = 26 - 10 = 16.\n2. Divide both sides by 2: x = 16 / 2 = 8.\nLet's verify: 2*(8) + 10 = 16 + 10 = 26. Correct.\n<|/think|>\nSubtracting 10 from both sides gives 2x = 16. Dividing by 2 yields x = 8.",
    },
    {
        "system": "You are an autonomous agent with tool execution capabilities. Always invoke tools using exact JSON schema syntax.",
        "user": "Search for all python files in the src/ directory containing 'PulsarConfig'.",
        "assistant": "<|tool_call|>\n{\"name\": \"search_code\", \"arguments\": {\"path\": \"src/\", \"pattern\": \"PulsarConfig\", \"file_extension\": \".py\"}}\n<|/tool_call|>",
    },
    {
        "system": "You are Pulsar, an expert software engineer specializing in Python.",
        "user": "Write a Python function `factorial(n: int) -> int` that computes the factorial of non-negative integer n using recursion.",
        "assistant": "<|think|>\nFactorial of non-negative integer n is defined as:\n- 0! = 1\n- n! = n * (n - 1)! for n > 0\nRaise ValueError if n < 0.\n<|/think|>\n```python\ndef factorial(n: int) -> int:\n    if n < 0:\n        raise ValueError(\"n must be non-negative\")\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n```",
    },
    {
        "system": "You are Pulsar, a helpful and precise reasoning AI assistant.",
        "user": "If a train travels 150 miles in 2.5 hours, what is its average speed in miles per hour?",
        "assistant": "<|think|>\nAverage speed is total distance divided by total time.\nDistance = 150 miles.\nTime = 2.5 hours.\nSpeed = 150 / 2.5 = 60 mph.\n<|/think|>\nThe average speed of the train is 60 miles per hour.",
    },
    {
        "system": "You are an autonomous agent with tool execution capabilities. Always invoke tools using exact JSON schema syntax.",
        "user": "List all branches in the git repository including remote tracking branches.",
        "assistant": "<|tool_call|>\n{\"name\": \"git_branch\", \"arguments\": {\"all\": true}}\n<|/tool_call|>",
    },
]


# ---------------------------------------------------------------------------
# SFT Dataset with Prompt Loss Masking
# ---------------------------------------------------------------------------

class SFTDataset(Dataset):
    """Tokenizes ChatML dialogues and creates target label masks (loss only on assistant tokens)."""

    def __init__(
        self,
        examples: List[Dict[str, str]],
        tokenizer: Any,
        max_seq_len: int = 1024,
    ):
        self.samples: List[Dict[str, torch.Tensor]] = []

        for ex in examples:
            sys_text = f"<|im_start|>system\n{ex['system']}<|im_end|>\n"
            user_text = f"<|im_start|>user\n{ex['user']}<|im_end|>\n"
            asst_prefix = "<|im_start|>assistant\n"
            asst_body = f"{ex['assistant']}<|im_end|>"

            prompt_text = sys_text + user_text + asst_prefix
            full_text = prompt_text + asst_body

            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)

            if len(full_ids) > max_seq_len:
                full_ids = full_ids[:max_seq_len]

            input_ids = torch.tensor(full_ids, dtype=torch.long)
            labels = torch.full_like(input_ids, -100)

            # Mask out prompt: only compute loss on assistant tokens
            prompt_len = len(prompt_ids)
            if prompt_len < len(input_ids):
                labels[prompt_len:] = input_ids[prompt_len:]

            self.samples.append({
                "input_ids": input_ids,
                "labels": labels,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def sft_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad batch sequences with EOS token id 0."""
    max_len = max(len(item["input_ids"]) for item in batch)
    b_input_ids = []
    b_labels = []

    for item in batch:
        seq_len = len(item["input_ids"])
        pad_len = max_len - seq_len

        if pad_len > 0:
            padded_in = torch.cat([item["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
            padded_lbl = torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        else:
            padded_in = item["input_ids"]
            padded_lbl = item["labels"]

        b_input_ids.append(padded_in)
        b_labels.append(padded_lbl)

    return {
        "input_ids": torch.stack(b_input_ids),
        "labels": torch.stack(b_labels),
    }


# ---------------------------------------------------------------------------
# Atomic File Writers
# ---------------------------------------------------------------------------

def write_atomic_json(filepath: str, data: dict):
    """Write dictionary to JSON file atomically."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    tmp_path = filepath + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, filepath)


# ---------------------------------------------------------------------------
# Main SFT Routine
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pulsar Supervised Fine-Tuning (SFT) Alignment Harness")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/milestone_step_5000.pt", help="Base pretraining checkpoint")
    parser.add_argument("--output-checkpoint", type=str, default="checkpoints/pulsar_sft.pt", help="Path to save aligned SFT weights")
    parser.add_argument("--tokenizer-path", type=str, default="data/pulsar_tokenizer", help="Path to tokenizer")
    parser.add_argument("--data-file", type=str, default=None, help="Optional JSONL file containing curated SFT dialogues")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for AdamW")
    parser.add_argument("--epochs", type=int, default=3, help="Number of SFT training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Micro-batch size per optimization step")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--seq-len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    parser.add_argument("--status-file", type=str, default="checkpoints/sft_status.json", help="Live status telemetry file")
    parser.add_argument("--metrics-file", type=str, default="checkpoints/sft_metrics.jsonl", help="Metrics log file")
    args = parser.parse_args()

    print("==================================================================")
    print("      PULSAR SUPERVISED FINE-TUNING (SFT) ALIGNMENT HARNESS       ")
    print(f"      Base Checkpoint:   {args.checkpoint}")
    print(f"      Output Target:     {args.output_checkpoint}")
    print(f"      Device:            {args.device}")
    print(f"      Epochs:            {args.epochs} | LR: {args.lr}")
    print("==================================================================")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # 2. Load Checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"[Error] Checkpoint '{args.checkpoint}' not found.")
        sys.exit(1)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = ck.get("config") or {}
    cfg = PulsarConfig.from_dict(cfg_dict) if cfg_dict else PulsarConfig()

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = PulsarForCausalLM(cfg).to(args.device, dtype=dtype)
    model.load_state_dict(ck.get("model", ck), strict=False)
    model.train()
    print(f"[Model] Pulsar model loaded successfully ({sum(p.numel() for p in model.parameters()):,} params).")

    # 3. Load Dataset
    examples = DEFAULT_SFT_EXAMPLES
    if args.data_file and os.path.exists(args.data_file):
        with open(args.data_file, "r", encoding="utf-8") as f:
            examples = [json.loads(line) for line in f if line.strip()]
        print(f"[Data] Loaded {len(examples)} custom examples from {args.data_file}")
    else:
        print(f"[Data] Using built-in curated demonstration set ({len(examples)} examples).")

    dataset = SFTDataset(examples, tokenizer, max_seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=sft_collate_fn)

    # 4. Setup Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    # 5. Training Loop
    total_steps = len(dataloader) * args.epochs
    global_step = 0
    t0 = time.time()

    write_atomic_json(args.status_file, {
        "status": "running",
        "phase": "sft_cold_start",
        "base_checkpoint": args.checkpoint,
        "step": 0,
        "total_steps": total_steps,
        "loss": 0.0,
        "last_updated": time.time(),
    })

    print(f"\n[SFT] Starting training across {args.epochs} epochs ({total_steps} total batches)...")
    optimizer.zero_grad()

    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        batch_count = 0

        for step, batch in enumerate(dataloader, start=1):
            global_step += 1
            input_ids = batch["input_ids"].to(args.device)
            labels = batch["labels"].to(args.device)

            with torch.autocast(device_type="cuda" if "cuda" in args.device else "cpu", dtype=dtype):
                outputs = model(input_ids)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
                # Shift for causal next-token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss_scaled = loss / args.grad_accum

            loss_scaled.backward()
            running_loss += loss.item()
            batch_count += 1

            if global_step % args.grad_accum == 0 or step == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if global_step % 2 == 0 or global_step == total_steps:
                avg_l = running_loss / max(1, batch_count)
                print(f"[Epoch {epoch}/{args.epochs} | Step {global_step}/{total_steps}] SFT Loss: {avg_l:.4f}")
                write_atomic_json(args.status_file, {
                    "status": "running",
                    "phase": "sft_cold_start",
                    "epoch": epoch,
                    "step": global_step,
                    "total_steps": total_steps,
                    "loss": round(avg_l, 4),
                    "last_updated": time.time(),
                })
                with open(args.metrics_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "epoch": epoch,
                        "step": global_step,
                        "loss": round(avg_l, 4),
                        "timestamp": time.time(),
                    }) + "\n")

    # 6. Save Fine-Tuned Checkpoint
    os.makedirs(os.path.dirname(os.path.abspath(args.output_checkpoint)), exist_ok=True)
    torch.save(
        {
            "global_step": ck.get("global_step", 0),
            "sft_steps": global_step,
            "base_checkpoint": args.checkpoint,
            "model": model.state_dict(),
            "config": cfg.to_dict(),
            "loss": avg_l,
        },
        args.output_checkpoint,
    )

    write_atomic_json(args.status_file, {
        "status": "completed",
        "phase": "sft_cold_start",
        "base_checkpoint": args.checkpoint,
        "output_checkpoint": args.output_checkpoint,
        "final_loss": round(avg_l, 4),
        "total_steps": global_step,
        "duration_seconds": round(time.time() - t0, 2),
        "last_updated": time.time(),
    })

    print(f"\n[SFT] Fine-tuning completed in {time.time() - t0:.1f}s. Saved SFT checkpoint -> {args.output_checkpoint}")


if __name__ == "__main__":
    main()
