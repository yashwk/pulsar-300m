#!/usr/bin/env python3
"""Pulsar Group Relative Policy Optimization (GRPO) Alignment Harness.

DeepSeek-R1-Zero style rule-based reinforcement learning on verifiable rewards (RLVR).
Eliminates the separate Critic/Value network to operate within tight VRAM bounds (< 7.0 GB).

Theoretical foundation:
- DeepSeekMath (Shao et al., 2024) / DeepSeek-R1 (2025)
- Normalized intra-group relative advantage: A_i = (R_i - mean(R)) / (std(R) + eps)
- Verifiable reward functions: Math accuracy, Python AST validity, and CoT format tags.

Usage:
    python scripts/grpo_train.py \
      --checkpoint checkpoints/best.pt \
      --group-size 4 \
      --lr 1e-5 \
      --max-steps 200 \
      --device cuda
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.model import PulsarConfig, PulsarForCausalLM
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Verifiable Reward Functions (RLVR)
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> str:
    """Extract boxed or labeled answer from text."""
    # 1. LaTeX boxed format
    m = re.findall(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m[-1].strip()
    # 2. Answer label format
    m = re.findall(r"(?:The\s+answer\s+is|Answer:)\s*([^\n\.]+)", text, re.IGNORECASE)
    if m:
        return m[-1].strip()
    # 3. Numeric final occurrence
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].strip()
    return ""


def reward_math(completion: str, ground_truth: str) -> float:
    """Score math reasoning accuracy and correctness."""
    pred = extract_answer(completion)
    gt = str(ground_truth).strip()
    if not pred or not gt:
        return 0.0

    # Exact match
    if pred.lower() == gt.lower():
        return 1.0

    # Numeric equivalence (e.g. 60 vs 60.0)
    try:
        if abs(float(pred) - float(gt)) < 1e-4:
            return 1.0
    except ValueError:
        pass

    return 0.0


def reward_format(completion: str) -> float:
    """Reward proper thinking structure (<|think|> ... <|/think|>)."""
    score = 0.0
    has_think_start = "<|think|>" in completion
    has_think_end = "<|/think|>" in completion or "</think>" in completion

    if has_think_start:
        score += 0.1
    if has_think_start and has_think_end:
        score += 0.2
    if completion.endswith("<|im_end|>") or "<|im_end|>" in completion:
        score += 0.1

    return score


def reward_python_ast(completion: str) -> float:
    """Reward valid executable Python syntax."""
    # Extract code between triple backticks if present
    code = completion
    if "```python" in completion:
        parts = completion.split("```python")
        if len(parts) > 1:
            code = parts[1].split("```")[0]
    elif "```" in completion:
        parts = completion.split("```")
        if len(parts) > 1:
            code = parts[1].split("```")[0]

    try:
        ast.parse(code)
        return 1.0
    except SyntaxError:
        return 0.0


def evaluate_rewards(completion: str, target: str, task_type: str = "math") -> float:
    """Aggregate rule-based rewards."""
    fmt_score = reward_format(completion)
    if task_type == "math":
        acc_score = reward_math(completion, target)
        return acc_score * 0.8 + fmt_score * 0.2
    elif task_type == "python":
        ast_score = reward_python_ast(completion)
        return ast_score * 0.8 + fmt_score * 0.2
    return fmt_score


# ---------------------------------------------------------------------------
# Default Curated RL Prompts
# ---------------------------------------------------------------------------

DEFAULT_RL_TASKS = [
    {
        "type": "math",
        "prompt": "<|im_start|>user\nIf a train travels 120 km in 2 hours, what is its average speed in km/h?<|im_end|>\n<|im_start|>assistant\n<|think|>\n",
        "target": "60"
    },
    {
        "type": "math",
        "prompt": "<|im_start|>user\nSolve for x: 3*x + 15 = 45.<|im_end|>\n<|im_start|>assistant\n<|think|>\n",
        "target": "10"
    },
    {
        "type": "math",
        "prompt": "<|im_start|>user\nWhat is 15 percent of 80?<|im_end|>\n<|im_start|>assistant\n<|think|>\n",
        "target": "12"
    },
    {
        "type": "math",
        "prompt": "<|im_start|>user\nIn a right-angled triangle with legs 3 and 4, what is the length of the hypotenuse?<|im_end|>\n<|im_start|>assistant\n<|think|>\n",
        "target": "5"
    },
    {
        "type": "math",
        "prompt": "<|im_start|>user\nWhat is the next number in the geometric progression: 2, 6, 18, 54, ...?<|im_end|>\n<|im_start|>assistant\n<|think|>\n",
        "target": "162"
    },
    {
        "type": "python",
        "prompt": "<|im_start|>user\nWrite a python function `is_even(n: int) -> bool` that returns True if n is even.<|im_end|>\n<|im_start|>assistant\n```python\n",
        "target": "def is_even(n: int) -> bool:\n    return n % 2 == 0\n"
    },
]


# ---------------------------------------------------------------------------
# GRPO Trainer
# ---------------------------------------------------------------------------

class GRPOTrainer:
    def __init__(
        self,
        model: PulsarForCausalLM,
        tokenizer: Any,
        device: str = "cuda",
        group_size: int = 4,
        lr: float = 1e-5,
        clip_epsilon: float = 0.2,
        beta_kl: float = 0.04,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.group_size = group_size
        self.clip_epsilon = clip_epsilon
        self.beta_kl = beta_kl

        # Freeze reference model weights by keeping a clone of initial parameters or checkpoint
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.01,
        )

    def sample_group(
        self, prompt_text: str, max_new_tokens: int = 128, temperature: float = 0.8
    ) -> List[Tuple[torch.Tensor, str]]:
        """Sample a group of G rollouts sequentially to bound memory."""
        self.model.eval()
        prompt_ids = self.tokenizer.encode(prompt_text, return_tensors="pt").to(self.device)
        group = []

        with torch.no_grad():
            for _ in range(self.group_size):
                gen_ids = self.model.generate(
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    repetition_penalty=1.1,
                )
                completion = self.tokenizer.decode(
                    gen_ids[0][prompt_ids.size(1):], skip_special_tokens=False
                )
                group.append((gen_ids, completion))

        return group

    def compute_grpo_step(self, task: Dict[str, Any]) -> Dict[str, float]:
        """Execute one complete GRPO policy optimization iteration."""
        prompt = task["prompt"]
        target = task["target"]
        task_type = task.get("type", "math")

        # 1. Rollout group of G completions
        group = self.sample_group(prompt, max_new_tokens=96, temperature=0.8)

        # 2. Score rewards
        raw_rewards = []
        for _, comp in group:
            r = evaluate_rewards(comp, target, task_type=task_type)
            raw_rewards.append(r)

        rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float32, device=self.device)
        mean_r = rewards_tensor.mean()
        std_r = rewards_tensor.std().clamp_min(1e-6)

        # 3. Relative normalized advantage
        advantages = (rewards_tensor - mean_r) / std_r

        # 4. Policy optimization step
        self.model.train()
        self.optimizer.zero_grad()

        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = prompt_ids.size(1)

        total_loss = 0.0
        for i, (gen_ids, _) in enumerate(group):
            adv = advantages[i]
            # Autoregressive next-token target shift:
            # x is gen_ids[:, :-1], targets y are gen_ids[:, 1:].
            seq = gen_ids[:, :256]
            if seq.size(1) <= prompt_len:
                continue

            x = seq[:, :-1]
            y = seq[:, 1:].clone()
            # Mask out prompt token targets so only completion tokens receive advantage gradients
            y[:, :prompt_len - 1] = -100

            _, loss, _ = self.model(x, labels=y, num_pulses=self.model.cfg.default_pulses, eval_full_pulses=True)
            if loss is not None:
                # Advantage-weighted surrogate objective
                step_loss = -adv * loss
                step_loss.backward()
                total_loss += step_loss.item()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "mean_reward": mean_r.item(),
            "max_reward": rewards_tensor.max().item(),
            "loss": total_loss / self.group_size,
            "best_completion": group[rewards_tensor.argmax().item()][1][:80].replace("\n", " "),
        }


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pulsar GRPO Reinforcement Learning Harness")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Base checkpoint path")
    parser.add_argument("--group-size", type=int, default=4, help="Rollout group size (G)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=100, help="Total GRPO optimization steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=======================================================")
    print("  PULSAR GRPO REASONING ALIGNMENT HARNESS")
    print(f"  Base Model:   {args.checkpoint}")
    print(f"  Group Size G: {args.group_size} rollouts / prompt")
    print(f"  Device:       {args.device} | LR: {args.lr}")
    print("=======================================================\n")

    tokenizer = AutoTokenizer.from_pretrained("data/pulsar_tokenizer")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = ck.get("config") or {}
    cfg = PulsarConfig.from_dict(cfg_dict) if cfg_dict else PulsarConfig()

    model = PulsarForCausalLM(cfg).to(args.device, dtype=torch.bfloat16)
    model.load_state_dict(ck.get("model", ck))
    print("[Pulsar] Model weights loaded successfully.")

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        group_size=args.group_size,
        lr=args.lr,
    )

    print(f"\n[GRPO] Beginning training across {len(DEFAULT_RL_TASKS)} verifiable tasks...")
    for step in range(1, args.max_steps + 1):
        task = DEFAULT_RL_TASKS[(step - 1) % len(DEFAULT_RL_TASKS)]
        metrics = trainer.compute_grpo_step(task)

        if step % 5 == 0 or step == 1:
            print(
                f"[GRPO Step {step:3d}/{args.max_steps}] "
                f"Mean Reward: {metrics['mean_reward']:.2f} | "
                f"Max: {metrics['max_reward']:.2f} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"Top: {metrics['best_completion']}"
            )

    output_path = "checkpoints/pulsar_grpo_math.pt"
    torch.save(
        {
            "global_step": args.max_steps,
            "model": model.state_dict(),
            "config": cfg.to_dict(),
        },
        output_path,
    )
    print(f"\n[GRPO] Training completed. Saved aligned weights -> {output_path}")


if __name__ == "__main__":
    main()
