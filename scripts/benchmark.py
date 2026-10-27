#!/usr/bin/env python3
"""Pulsar Automated Milestone Benchmark & Capability Evaluation Suite.

Evaluates checkpoints against 4 standardized capability benchmarks:
1. Python Code Syntactic Validity:
   Tests algorithmic code prompts and validates AST parsing via `ast.parse()`.
2. Agentic MCP Tool-Calling Schema Compliance:
   Tests tool-calling prompts and validates JSON formatting:
   `<|tool_call|>{"name": ..., "arguments": ...}<|/tool_call|>`.
3. Mathematical Reasoning & CoT Structure:
   Tests elementary & discrete math prompts for `<|think|>` and `<|/think|>`
   block closure and final answer emission.
4. KV-Cache & Inference Engine Metrics:
   Measures prefill latency, decode tokens/sec, and unified KV memory footprint.

Usage:
    # Run lightweight evaluation on CPU (safe to run concurrently with GPU training):
    python scripts/benchmark.py --checkpoint checkpoints/best.pt --device cpu

    # Run full evaluation on CUDA (when GPU is free):
    python scripts/benchmark.py --checkpoint checkpoints/best.pt --device cuda
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys
import time
from typing import Any, Dict, List, Tuple

import torch

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.infer import UnifiedKVCache, PulsarInferenceEngine, load_inference_model


# ==============================================================================
# Benchmark Test Suites
# ==============================================================================

PYTHON_BENCHMARKS = [
    {
        "id": "py_01",
        "name": "Fibonacci Function",
        "prompt": "def fibonacci(n: int) -> int:\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    },
    {
        "id": "py_02",
        "name": "Is Palindrome",
        "prompt": "def is_palindrome(s: str) -> bool:\n    \"\"\"Check if string is a palindrome.\"\"\"\n",
    },
    {
        "id": "py_03",
        "name": "Binary Search",
        "prompt": "def binary_search(arr: list[int], target: int) -> int:\n    \"\"\"Return index of target or -1.\"\"\"\n",
    },
    {
        "id": "py_04",
        "name": "Merge Sorted Lists",
        "prompt": "def merge_sorted(a: list[int], b: list[int]) -> list[int]:\n    \"\"\"Merge two sorted lists.\"\"\"\n",
    },
    {
        "id": "py_05",
        "name": "Quicksort Implementation",
        "prompt": "def quicksort(arr: list[int]) -> list[int]:\n    if len(arr) <= 1:\n        return arr\n",
    },
]

TOOL_BENCHMARKS = [
    {
        "id": "tool_01",
        "name": "File Rename Tool",
        "prompt": (
            "<|im_start|>system\nYou are a helpful AI assistant equipped with tools. "
            "Emit tool calls formatted as <|tool_call|>{\"name\": ..., \"arguments\": ...}<|/tool_call|>.<|im_end|>\n"
            "<|im_start|>user\nRename document scan_01.pdf to 2026_Invoice.pdf in /home/user/docs.<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
    },
    {
        "id": "tool_02",
        "name": "Current Weather Tool",
        "prompt": (
            "<|im_start|>system\nYou are a helpful AI assistant equipped with tools. "
            "Emit tool calls formatted as <|tool_call|>{\"name\": ..., \"arguments\": ...}<|/tool_call|>.<|im_end|>\n"
            "<|im_start|>user\nWhat is the current temperature in San Francisco?<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
    },
    {
        "id": "tool_03",
        "name": "Database Query Tool",
        "prompt": (
            "<|im_start|>system\nYou are a helpful AI assistant equipped with tools. "
            "Emit tool calls formatted as <|tool_call|>{\"name\": ..., \"arguments\": ...}<|/tool_call|>.<|im_end|>\n"
            "<|im_start|>user\nFind the user with email 'alice@example.com' from the users table.<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
    },
    {
        "id": "tool_04",
        "name": "Bash Command Tool",
        "prompt": (
            "<|im_start|>system\nYou are a helpful AI assistant equipped with tools. "
            "Emit tool calls formatted as <|tool_call|>{\"name\": ..., \"arguments\": ...}<|/tool_call|>.<|im_end|>\n"
            "<|im_start|>user\nList all listening TCP ports on this machine.<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
    },
    {
        "id": "tool_05",
        "name": "Calculator Tool",
        "prompt": (
            "<|im_start|>system\nYou are a helpful AI assistant equipped with tools. "
            "Emit tool calls formatted as <|tool_call|>{\"name\": ..., \"arguments\": ...}<|/tool_call|>.<|im_end|>\n"
            "<|im_start|>user\nCalculate the square root of 50176.<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
    },
]

MATH_BENCHMARKS = [
    {
        "id": "math_01",
        "name": "Train Speed Word Problem",
        "prompt": (
            "<|im_start|>user\nIf a train travels 120 km in 2 hours, what is its average speed in km/h?<|im_end|>\n"
            "<|im_start|>assistant\n<|think|>\n"
        ),
        "target": "60",
    },
    {
        "id": "math_02",
        "name": "Linear Equation",
        "prompt": (
            "<|im_start|>user\nSolve for x: 3*x + 15 = 45.<|im_end|>\n"
            "<|im_start|>assistant\n<|think|>\n"
        ),
        "target": "10",
    },
    {
        "id": "math_03",
        "name": "Percentage Calculation",
        "prompt": (
            "<|im_start|>user\nWhat is 15 percent of 80?<|im_end|>\n"
            "<|im_start|>assistant\n<|think|>\n"
        ),
        "target": "12",
    },
    {
        "id": "math_04",
        "name": "Right Triangle Hypotenuse",
        "prompt": (
            "<|im_start|>user\nIn a right-angled triangle with legs of length 3 and 4, what is the hypotenuse?<|im_end|>\n"
            "<|im_start|>assistant\n<|think|>\n"
        ),
        "target": "5",
    },
    {
        "id": "math_05",
        "name": "Geometric Progression",
        "prompt": (
            "<|im_start|>user\nWhat is the next term in the sequence: 2, 6, 18, 54, ...?<|im_end|>\n"
            "<|im_start|>assistant\n<|think|>\n"
        ),
        "target": "162",
    },
]


# ==============================================================================
# Evaluators
# ==============================================================================

def extract_python_code(text: str) -> str:
    """Extract code from markdown code fences or raw function string."""
    code_block = re.search(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL)
    if code_block:
        return code_block.group(1).strip()
    return text.strip()


def evaluate_python_ast(code: str) -> Tuple[bool, str]:
    """Test if generated code parses as valid Python AST."""
    clean_code = extract_python_code(code)
    try:
        ast.parse(clean_code)
        return True, "Valid AST"
    except SyntaxError as e:
        # Try evaluating just the first complete function
        lines = clean_code.split("\n")
        func_lines = []
        for line in lines:
            if line.startswith("<|"):
                break
            func_lines.append(line)
        try:
            ast.parse("\n".join(func_lines))
            return True, "Valid AST (truncated at stop token)"
        except SyntaxError:
            return False, f"SyntaxError: {e.msg} at line {e.lineno}"


def evaluate_tool_schema(text: str) -> Tuple[bool, str, Dict[str, Any] | None]:
    """Test if text contains a valid <|tool_call|> with JSON schema."""
    # Matches <|tool_call|> {json} ...
    pattern = r"<\|tool_call\|>\s*(\{.*?\})(?:\s*<\|/tool_call\|>|\s*<\|im_end\|>|\n|$)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        # Check if list of tools was generated
        list_pattern = r"<\|tool_call\|>\s*(\[\s*\{.*?\}\s*\])"
        match = re.search(list_pattern, text, re.DOTALL)
        if not match:
            return False, "Missing <|tool_call|> JSON pattern", None

    json_str = match.group(1).strip()
    try:
        data = json.loads(json_str)
        if isinstance(data, list) and len(data) > 0:
            data = data[0]
        if isinstance(data, dict):
            has_name = "name" in data or "function" in data or "tool" in data
            has_args = "arguments" in data or "parameters" in data or "input" in data
            if has_name and has_args:
                return True, "Valid tool schema", data
            return False, f"JSON valid but missing schema keys: {list(data.keys())}", data
        return False, "JSON is not an object or list of objects", None
    except json.JSONDecodeError as e:
        return False, f"JSONDecodeError: {e}", None


def evaluate_math_cot(text: str, target: str) -> Tuple[bool, bool, str]:
    """Test for <|think|> block closure and presence of target answer."""
    has_think_start = "<|think|>" in text
    has_think_end = "<|/think|>" in text or "</think>" in text
    cot_structured = has_think_start and has_think_end

    # Look for target number in the text following </think> or throughout
    answer_text = text.split("<|/think|>")[-1] if has_think_end else text
    found_target = bool(re.search(r"\b" + re.escape(target) + r"\b", answer_text))

    return cot_structured, found_target, answer_text.strip()[:100]


# ==============================================================================
# Runner
# ==============================================================================

def run_benchmark(
    checkpoint_path: str,
    device: str = "cpu",
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    output_json: str | None = None,
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"  PULSAR CAPABILITY BENCHMARK HARNESS")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {device} | Temp: {temperature} | Top-P: {top_p}")
    print(f"=======================================================\n")

    engine, tokenizer = load_inference_model(checkpoint_path, device=device)

    results: Dict[str, Any] = {
        "checkpoint": checkpoint_path,
        "device": device,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "suites": {},
    }

    # 1. Python Syntax Suite
    print("\n--- [Suite 1: Python AST Syntax Validity] ---")
    py_passes = 0
    py_results = []
    for item in PYTHON_BENCHMARKS:
        prompt = item["prompt"]
        gen = engine.generate(
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stream=False,
        )
        passed, msg = evaluate_python_ast(gen)
        if passed:
            py_passes += 1
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {item['name']:<25} -> {msg}")
        py_results.append({
            "id": item["id"],
            "name": item["name"],
            "passed": passed,
            "message": msg,
            "sample": gen[:120].replace("\n", " "),
        })
    py_rate = (py_passes / len(PYTHON_BENCHMARKS)) * 100
    print(f"  Python AST Pass Rate: {py_passes}/{len(PYTHON_BENCHMARKS)} ({py_rate:.1f}%)")
    results["suites"]["python"] = {"pass_rate": py_rate, "details": py_results}

    # 2. Agentic Tool Calling Suite
    print("\n--- [Suite 2: Agentic MCP Tool Schema Compliance] ---")
    tool_passes = 0
    tool_results = []
    for item in TOOL_BENCHMARKS:
        prompt = item["prompt"]
        gen = engine.generate(
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stream=False,
        )
        passed, msg, payload = evaluate_tool_schema(gen)
        if passed:
            tool_passes += 1
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {item['name']:<25} -> {msg}")
        tool_results.append({
            "id": item["id"],
            "name": item["name"],
            "passed": passed,
            "message": msg,
            "payload": payload,
            "sample": gen[:120].replace("\n", " "),
        })
    tool_rate = (tool_passes / len(TOOL_BENCHMARKS)) * 100
    print(f"  Tool Schema Pass Rate: {tool_passes}/{len(TOOL_BENCHMARKS)} ({tool_rate:.1f}%)")
    results["suites"]["tool_calling"] = {"pass_rate": tool_rate, "details": tool_results}

    # 3. Math CoT Suite
    print("\n--- [Suite 3: Mathematical Reasoning & CoT Structure] ---")
    cot_passes = 0
    target_passes = 0
    math_results = []
    for item in MATH_BENCHMARKS:
        prompt = item["prompt"]
        gen = engine.generate(
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stream=False,
        )
        cot_ok, target_ok, sample = evaluate_math_cot(gen, item["target"])
        if cot_ok:
            cot_passes += 1
        if target_ok:
            target_passes += 1
        cot_status = "OK" if cot_ok else "NO_COT"
        tgt_status = "CORRECT" if target_ok else "INCORRECT"
        print(f"  [{cot_status}|{tgt_status}] {item['name']:<25} (target: {item['target']}) -> {sample}")
        math_results.append({
            "id": item["id"],
            "name": item["name"],
            "cot_structured": cot_ok,
            "target_correct": target_ok,
            "target": item["target"],
            "sample": sample,
        })
    cot_rate = (cot_passes / len(MATH_BENCHMARKS)) * 100
    tgt_rate = (target_passes / len(MATH_BENCHMARKS)) * 100
    print(f"  CoT Structure Rate: {cot_passes}/{len(MATH_BENCHMARKS)} ({cot_rate:.1f}%) | Target Accuracy: {target_passes}/{len(MATH_BENCHMARKS)} ({tgt_rate:.1f}%)")
    results["suites"]["math"] = {
        "cot_structure_rate": cot_rate,
        "target_accuracy": tgt_rate,
        "details": math_results,
    }

    # 4. Latency & Memory Footprint
    print("\n--- [Suite 4: Inference Latency & KV Cache Memory] ---")
    test_prompt = "The quick brown fox jumps over the lazy dog. " * 8  # ~72 tokens
    t0 = time.perf_counter()
    gen = engine.generate(
        tokenizer,
        test_prompt,
        max_new_tokens=64,
        temperature=0.0,
        stream=False,
    )
    t1 = time.perf_counter()
    tokens_generated = 64
    elapsed = max(t1 - t0, 1e-4)
    tok_per_sec = tokens_generated / elapsed
    print(f"  Decode Throughput: {tok_per_sec:.1f} tok/s on {device}")

    # Footprint
    cache = UnifiedKVCache(engine.cfg, batch_size=1, device=device, dtype=engine.dtype)
    k_dummy = torch.randn(1, engine.cfg.n_kv_heads, 1024, engine.cfg.head_dim, device=device)
    v_dummy = torch.randn(1, engine.cfg.n_kv_heads, 1024, engine.cfg.head_dim, device=device)
    for i in range(engine.cfg.n_core):
        cache.update(i, k_dummy, v_dummy, pulse_idx=0, is_prefill=True)
    mem_1k_mb = cache.memory_footprint_bytes() / (1024 * 1024)
    print(f"  Unified KV Memory (1,024 context): {mem_1k_mb:.2f} MB")

    results["suites"]["performance"] = {
        "decode_tok_per_sec": round(tok_per_sec, 2),
        "kv_cache_1024_mb": round(mem_1k_mb, 2),
    }

    print("\n=======================================================")
    print("  BENCHMARK SUMMARY")
    print(f"  Python AST Valid:     {py_rate:.1f}%")
    print(f"  Tool Schema Valid:    {tool_rate:.1f}%")
    print(f"  Math CoT Structured:  {cot_rate:.1f}%")
    print(f"  Decode Speed:         {tok_per_sec:.1f} tok/s ({device})")
    print("=======================================================\n")

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[Benchmark] Saved full results to: {output_json}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Pulsar Capability Benchmark Harness")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Path to checkpoint file")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device (cpu or cuda)")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling threshold")
    parser.add_argument("--repetition-penalty", type=float, default=1.15, help="Repetition penalty")
    parser.add_argument("--greedy", action="store_true", help="Force greedy argmax sampling (temp=0)")
    parser.add_argument("--output-json", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    temp = 0.0 if args.greedy else args.temperature

    run_benchmark(
        checkpoint_path=args.checkpoint,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=temp,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
