#!/usr/bin/env python3
"""Pulsar Group Relative Policy Optimization (GRPO) Alignment Harness.

DeepSeek-R1-Zero style rule-based reinforcement learning on verifiable rewards (RLVR).
Eliminates the separate Critic/Value network to operate within tight VRAM bounds (< 7.0 GB).

Theoretical foundation:
- DeepSeekMath (Shao et al., 2024) / DeepSeek-R1 (2025)
- Normalized intra-group relative advantage: A_i = (R_i - mean(R)) / (std(R) + eps)
- Verifiable reward functions: Math accuracy, Python AST validity, MCP tool schema, and CoT format tags.

Safety and Invariance:
- Strictly treats pretraining checkpoints and logs (latest.pt, best.pt, metrics.jsonl, live_status.json)
  as read-only and immutable.
- Telemetry state is written to distinct files: checkpoints/grpo_status.json and checkpoints/grpo_metrics.jsonl.
- Aligned weights are saved to checkpoints/pulsar_grpo_hybrid.pt.

Usage:
    python scripts/grpo_train.py \
      --checkpoint checkpoints/milestone_step_4000.pt \
      --group-size 4 \
      --lr 1e-5 \
      --max-steps 50 \
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


def reward_tool_call(completion: str, expected_tool: str) -> float:
    """Reward valid MCP agentic tool call syntax and schema."""
    m = re.findall(r"<\|tool_call\|>(.*?)<\|/tool_call\|>", completion, re.DOTALL)
    if not m:
        m = re.findall(r"<\|tool_call\|>(.*)", completion, re.DOTALL)
    if not m:
        return 0.0

    raw = m[0].strip()
    try:
        data = json.loads(raw)
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            if isinstance(first, dict) and first.get("name") == expected_tool:
                return 1.0
            elif isinstance(first, dict) and "name" in first:
                return 0.5
        elif isinstance(data, dict):
            if data.get("name") == expected_tool:
                return 1.0
            elif "name" in data:
                return 0.5
    except Exception:
        pass
    return 0.0


def evaluate_rewards(completion: str, target: str, task_type: str = "math") -> Tuple[float, Dict[str, float]]:
    """Aggregate rule-based rewards with category breakdown."""
    fmt_score = reward_format(completion)
    acc_score = 0.0

    if task_type == "math":
        acc_score = reward_math(completion, target)
        total = acc_score * 0.8 + fmt_score * 0.2
    elif task_type == "python":
        acc_score = reward_python_ast(completion)
        total = acc_score * 0.8 + fmt_score * 0.2
    elif task_type == "tool":
        acc_score = reward_tool_call(completion, target)
        total = acc_score * 0.8 + fmt_score * 0.2
    else:
        total = fmt_score

    breakdown = {
        "total": float(total),
        "accuracy": float(acc_score),
        "format": float(fmt_score),
        "task_type": task_type,
    }
    return float(total), breakdown


# ---------------------------------------------------------------------------
# Default Curated Multi-Task RL Prompts
# ---------------------------------------------------------------------------

DEFAULT_RL_TASKS = [
    # --- Math Tasks (Verifiable Numeric / Algebraic Solutions) ---
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
    # --- Python Programming Tasks (Verifiable AST Syntax) ---
    {
        "type": "python",
        "prompt": "<|im_start|>user\nWrite a python function `is_even(n: int) -> bool` that returns True if n is even.<|im_end|>\n<|im_start|>assistant\n```python\n",
        "target": "def is_even(n: int) -> bool:\n    return n % 2 == 0\n"
    },
    {
        "type": "python",
        "prompt": "<|im_start|>user\nWrite a python function `factorial(n: int) -> int` to compute n factorial iteratively.<|im_end|>\n<|im_start|>assistant\n```python\n",
        "target": "def factorial(n: int) -> int:\n    res = 1\n    for i in range(1, n + 1):\n        res *= i\n    return res\n"
    },
    {
        "type": "python",
        "prompt": "<|im_start|>user\nWrite a python function `reverse_list(lst: list) -> list` that returns a reversed list.<|im_end|>\n<|im_start|>assistant\n```python\n",
        "target": "def reverse_list(lst: list) -> list:\n    return lst[::-1]\n"
    },
    # --- Agentic Tool Calling Tasks (Verifiable JSON MCP Schema) ---
    {
        "type": "tool",
        "prompt": "<|im_start|>user\nRename scan_01.pdf to 2026_Invoice.pdf in /home/user/docs<|im_end|>\n<|im_start|>assistant\n<|tool_call|>",
        "target": "rename_file"
    },
    {
        "type": "tool",
        "prompt": "<|im_start|>user\nCheck git commit history on the current repository<|im_end|>\n<|im_start|>assistant\n<|tool_call|>",
        "target": "git_log"
    },
    {
        "type": "tool",
        "prompt": "<|im_start|>user\nFind all occurrences of 'TODO' in src/train.py<|im_end|>\n<|im_start|>assistant\n<|tool_call|>",
        "target": "search_code"
    },
]


# ---------------------------------------------------------------------------
# Atomic File Writers
# ---------------------------------------------------------------------------

def write_atomic_json(filepath: str, data: dict):
    """Write dictionary to JSON file atomically to prevent partial reads by the web server."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    tmp_path = filepath + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, filepath)


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
        """Sample a group of G rollouts sequentially to bound GPU memory."""
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

    def compute_grpo_step(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one complete GRPO policy optimization iteration."""
        prompt = task["prompt"]
        target = task["target"]
        task_type = task.get("type", "math")

        # 1. Rollout group of G completions
        group = self.sample_group(prompt, max_new_tokens=96, temperature=0.8)

        # 2. Score rewards
        raw_rewards = []
        breakdowns = []
        for _, comp in group:
            r, bd = evaluate_rewards(comp, target, task_type=task_type)
            raw_rewards.append(r)
            breakdowns.append(bd)

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

        # Extract summaries for telemetry inspector
        rollouts_summary = []
        for j, (_, comp) in enumerate(group):
            rollouts_summary.append({
                "id": j + 1,
                "completion": comp[:160].replace("\n", " "),
                "reward": round(raw_rewards[j], 3),
                "accuracy": round(breakdowns[j]["accuracy"], 2),
                "format": round(breakdowns[j]["format"], 2),
                "advantage": round(advantages[j].item(), 3),
            })

        best_idx = int(rewards_tensor.argmax().item())
        mean_acc = sum(b["accuracy"] for b in breakdowns) / len(breakdowns)
        mean_fmt = sum(b["format"] for b in breakdowns) / len(breakdowns)

        return {
            "mean_reward": mean_r.item(),
            "max_reward": rewards_tensor.max().item(),
            "loss": total_loss / self.group_size,
            "accuracy": mean_acc,
            "format": mean_fmt,
            "task_type": task_type,
            "target": target,
            "best_completion": group[best_idx][1][:80].replace("\n", " "),
            "rollouts_summary": rollouts_summary,
        }


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pulsar GRPO Reinforcement Learning Harness")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/milestone_step_4000.pt", help="Base snapshot checkpoint path")
    parser.add_argument("--output-checkpoint", type=str, default="checkpoints/pulsar_grpo_hybrid.pt", help="Target path to save aligned weights")
    parser.add_argument("--status-file", type=str, default="checkpoints/grpo_status.json", help="Path for live telemetry state")
    parser.add_argument("--metrics-file", type=str, default="checkpoints/grpo_metrics.jsonl", help="Path for metrics history")
    parser.add_argument("--group-size", type=int, default=4, help="Rollout group size (G)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=50, help="Total GRPO optimization steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Pretraining trunk immutability check
    FORBIDDEN_FILES = {
        "latest.pt",
        "best.pt",
        "metrics.jsonl",
        "live_status.json",
        "run_config.json",
        "train.log",
        "metrics.csv",
    }
    for path, name in [
        (args.output_checkpoint, "output_checkpoint"),
        (args.status_file, "status_file"),
        (args.metrics_file, "metrics_file"),
    ]:
        base = os.path.basename(path)
        if base in FORBIDDEN_FILES or base.startswith("milestone_step_"):
            raise ValueError(
                f"Safety violation: {name} cannot point to pretraining trunk artifact '{base}'. "
                f"GRPO branch outputs must remain strictly isolated."
            )

    print("=======================================================")
    print("  PULSAR GRPO HYBRID REASONING ALIGNMENT HARNESS")
    print(f"  Base Model:       {args.checkpoint}")
    print(f"  Target Output:    {args.output_checkpoint}")
    print(f"  Group Size G:     {args.group_size} rollouts / prompt")
    print(f"  Device:           {args.device} | LR: {args.lr}")
    print(f"  Status Telemetry: {args.status_file}")
    print("=======================================================\n")

    # Fallback to best.pt or latest.pt if the milestone checkpoint does not exist yet
    if not os.path.exists(args.checkpoint):
        fallback = "checkpoints/best.pt" if os.path.exists("checkpoints/best.pt") else "checkpoints/latest.pt"
        print(f"[Warning] Specified checkpoint '{args.checkpoint}' not found. Falling back to '{fallback}'.")
        args.checkpoint = fallback

    tokenizer = AutoTokenizer.from_pretrained("data/pulsar_tokenizer")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = ck.get("config") or {}
    cfg = PulsarConfig.from_dict(cfg_dict) if cfg_dict else PulsarConfig()

    model = PulsarForCausalLM(cfg).to(args.device, dtype=torch.bfloat16)
    model.load_state_dict(ck.get("model", ck))
    print("[Pulsar] Base model weights loaded successfully in read-only mode.")

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        group_size=args.group_size,
        lr=args.lr,
    )

    # Initial status update
    write_atomic_json(args.status_file, {
        "status": "running",
        "mode": "grpo_rlvr",
        "base_checkpoint": args.checkpoint,
        "output_checkpoint": args.output_checkpoint,
        "step": 0,
        "max_steps": args.max_steps,
        "group_size": args.group_size,
        "mean_reward": 0.0,
        "max_reward": 0.0,
        "policy_loss": 0.0,
        "accuracy": 0.0,
        "format": 0.0,
        "recent_task": "",
        "recent_completion": "",
        "rollouts": [],
        "last_updated": time.time(),
    })

    print(f"\n[GRPO] Beginning training across {len(DEFAULT_RL_TASKS)} verifiable hybrid tasks...")
    for step in range(1, args.max_steps + 1):
        task = DEFAULT_RL_TASKS[(step - 1) % len(DEFAULT_RL_TASKS)]
        metrics = trainer.compute_grpo_step(task)

        # Write atomic telemetry
        write_atomic_json(args.status_file, {
            "status": "running" if step < args.max_steps else "completed",
            "mode": "grpo_rlvr",
            "base_checkpoint": args.checkpoint,
            "output_checkpoint": args.output_checkpoint,
            "step": step,
            "max_steps": args.max_steps,
            "group_size": args.group_size,
            "mean_reward": round(metrics["mean_reward"], 4),
            "max_reward": round(metrics["max_reward"], 4),
            "policy_loss": round(metrics["loss"], 4),
            "accuracy": round(metrics["accuracy"], 3),
            "format": round(metrics["format"], 3),
            "recent_task": f"[{metrics['task_type'].upper()}] -> {metrics['target']}",
            "recent_completion": metrics["best_completion"],
            "rollouts": metrics["rollouts_summary"],
            "last_updated": time.time(),
        })

        # Append to metrics history
        with open(args.metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "step": step,
                "mean_reward": round(metrics["mean_reward"], 4),
                "max_reward": round(metrics["max_reward"], 4),
                "policy_loss": round(metrics["loss"], 4),
                "accuracy": round(metrics["accuracy"], 3),
                "format": round(metrics["format"], 3),
                "task_type": metrics["task_type"],
                "timestamp": time.time(),
            }) + "\n")

        if step % 5 == 0 or step == 1:
            print(
                f"[GRPO Step {step:3d}/{args.max_steps}] "
                f"Mean Reward: {metrics['mean_reward']:.2f} | "
                f"Max: {metrics['max_reward']:.2f} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"Top: {metrics['best_completion']}"
            )

    # Save aligned weights
    torch.save(
        {
            "global_step": args.max_steps,
            "base_checkpoint": args.checkpoint,
            "model": model.state_dict(),
            "config": cfg.to_dict(),
        },
        args.output_checkpoint,
    )
    print(f"\n[GRPO] Training completed. Saved aligned weights -> {args.output_checkpoint}")


if __name__ == "__main__":
    main()
