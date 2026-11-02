import json
import re
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch

import pytest

from data.prepare_shards import (
    FOREIGN_SCRIPT_RE,
    BANNED_RAMBLE_RE,
    JEV_EVAL_SUBSTREAMS,
    JevQualityGate,
    distill_code_cot,
    distill_math_cot,
    is_strict_english_technical,
    strip_chatbot_fluff,
)


def test_is_strict_english_technical_rejection():
    """Verify fast rejection of non-English scripts (Cyrillic, CJK, Arabic, Devanagari)."""
    assert not is_strict_english_technical("Привет, как дела?")
    assert not is_strict_english_technical("这是关于快速排序的代码实现。")
    assert not is_strict_english_technical("مرحبا بالعالم - خوارزمية البحث")
    assert not is_strict_english_technical("नमस्ते दुनिया, सॉर्टिंग एल्गोरिदम")
    assert not is_strict_english_technical("")


def test_is_strict_english_technical_acceptance():
    """Verify acceptance of technical English with standard math and box-drawing symbols."""
    code_snippet = (
        "def quicksort(arr: list[int]) -> list[int]:\n"
        "    # Partition array around pivot: O(N log N) average complexity\n"
        "    if len(arr) <= 1:\n"
        "        return arr\n"
        "    pivot = arr[len(arr) // 2]\n"
        "    left = [x for x in arr if x < pivot]\n"
        "    mid = [x for x in arr if x == pivot]\n"
        "    right = [x for x in arr if x > pivot]\n"
        "    return quicksort(left) + mid + quicksort(right)\n"
    )
    assert is_strict_english_technical(code_snippet)

    math_snippet = (
        "Let x ≥ 0 and y ≥ 0. By Cauchy-Schwarz inequality:\n"
        "∑(a_i * b_i) ≤ √(∑ a_i²) * √(∑ b_i²)\n"
        "Therefore, the bound holds for all real vectors with error ε ≈ 0."
    )
    assert is_strict_english_technical(math_snippet)

    box_snippet = (
        "┌───────────────┬───────────────┐\n"
        "│ Matrix Row 1  │ Matrix Row 2  │\n"
        "├───────────────┼───────────────┤\n"
        "│ Value = 42    │ Value = 84    │\n"
        "└───────────────┴───────────────┘\n"
    )
    assert is_strict_english_technical(box_snippet)


def test_strip_chatbot_fluff():
    """Verify stripping of sycophantic greetings and assistant boilerplate openings."""
    t1 = "Certainly! Here is the Python solution you requested:\n\ndef add(a, b):\n    return a + b\n"
    assert strip_chatbot_fluff(t1).startswith("def add(a, b):")

    t2 = "Sure, I'd be happy to help with that problem!\n\nSELECT * FROM users WHERE active = 1;"
    assert strip_chatbot_fluff(t2).startswith("SELECT * FROM users")

    t3 = "As an AI language model, I recommend calculating the eigenvalues as follows:\n\nAv = λv"
    assert strip_chatbot_fluff(t3).startswith("I recommend calculating")

    clean = "To solve this problem, we observe that the recurrence relation simplifies to T(n) = 2T(n/2) + O(n)."
    assert strip_chatbot_fluff(clean) == clean


def test_distill_code_cot_case1_concise_first_try():
    """Verify that a concise, direct, first-try <think> block is preserved."""
    problem = "Write a function to compute the Fibonacci sequence iteratively in Python."
    solution = (
        "<think>\n"
        "We need an iterative function that tracks two variables a and b up to n. "
        "Time complexity will be O(n) and space complexity will be O(1).\n"
        "</think>\n"
        "```python\n"
        "def fib(n: int) -> int:\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n):\n"
        "        a, b = b, a + b\n"
        "    return a\n"
        "```"
    )
    res = distill_code_cot(problem, solution)
    assert res is not None
    assert "<|think|>" in res
    assert "<|/think|>" in res
    assert "We need an iterative function" in res
    assert "def fib(n: int) -> int:" in res


def test_distill_code_cot_case2_rambling_replacement():
    """Verify that rambling trial-and-error <think> is discarded in favor of post-think explanation."""
    problem = "Implement binary search in Python."
    solution = (
        "<think>\n"
        "Wait, let me reconsider what happens with the mid pointer. "
        "Hmm, actually, no, maybe the loop condition should be <= or <? "
        "Let's reconsider the corner cases when array is empty. Oops, wait a second...\n"
        "</think>\n"
        "Binary search operates by maintaining low and high bounds, inspecting the middle element, "
        "and halving the search space at each iteration. This achieves O(log n) time complexity.\n\n"
        "```python\n"
        "def binary_search(arr: list[int], target: int) -> int:\n"
        "    lo, hi = 0, len(arr) - 1\n"
        "    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if arr[mid] == target:\n"
        "            return mid\n"
        "        elif arr[mid] < target:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return -1\n"
        "```"
    )
    res = distill_code_cot(problem, solution)
    assert res is not None
    assert "<|think|>" in res
    assert "<|/think|>" in res
    # Rambling phrases must not appear
    assert "Wait, let me reconsider" not in res
    assert "Oops, wait a second" not in res
    # Clean explanation must be inside think block
    assert "Binary search operates by maintaining low and high bounds" in res
    assert "def binary_search" in res


def test_distill_code_cot_invalid_python_syntax():
    """Verify rejection of code blocks with Python syntax errors."""
    problem = "Write an invalid python function."
    solution = (
        "<think>\n"
        "Constructing a direct implementation of the function.\n"
        "</think>\n"
        "```python\n"
        "def broken(\n"
        "    return 42 +++\n"
        "```"
    )
    res = distill_code_cot(problem, solution)
    assert res is None


def test_distill_math_cot():
    """Verify distillation of math CoT with boxed answers and rejection of non-computational proofs."""
    # Proof essay without boxed answer -> rejected
    proof_problem = "Prove that for all topological spaces X, every compact subset of a Hausdorff space is closed."
    proof_solution = (
        "Let K be a compact subset of a Hausdorff space X. Let x ∈ X \\ K. "
        "For each y ∈ K, since X is Hausdorff, there exist disjoint open sets U_y and V_y containing x and y respectively."
    )
    assert distill_math_cot(proof_problem, proof_solution) is None

    # Applied computational problem with boxed answer -> accepted
    math_problem = "Find the sum of all integers from 1 to 50."
    math_solution = (
        "The sum of the first n positive integers is given by S = n(n + 1) / 2. "
        "Substituting n = 50 yields S = 50 * 51 / 2 = 25 * 51 = 1275. "
        "Therefore, the result is \\boxed{1275}."
    )
    res = distill_math_cot(math_problem, math_solution, expected_answer="1275")
    assert res is not None
    assert "<|think|>" in res
    assert "S = n(n + 1) / 2" in res
    assert "\\boxed{1275}" in res


def test_jev_quality_gate_substream_bypass():
    """Verify that deterministic substreams (1A, 3A, 3B) bypass Jev without API calls."""
    gate = JevQualityGate(api_key="mock_key", enabled=True)
    gate.evaluate_one = MagicMock(return_value=True)

    candidates = [
        ("1A", "def foo(): pass"),
        ("3A", "<|tool_call|>{...}<|/tool_call|>"),
        ("3B", "```toml\nkey = 'val'\n```"),
    ]

    filtered = gate.filter_batch(candidates)
    assert len(filtered) == 3
    # evaluate_one should not be called because all candidates are in bypass streams
    assert gate.evaluate_one.call_count == 0


def test_jev_quality_gate_decision_and_budget_circuit_breaker():
    """Verify Jev decision thresholds, cost accumulation, and circuit breaker trip."""
    gate = JevQualityGate(
        api_key="mock_key",
        api_url="https://mock.api/v1/systemone",
        max_usd=0.001,  # Tiny budget cap to test circuit breaker
        enabled=True,
    )

    # Mock successful HTTP response passing criteria
    pass_resp_data = {
        "answers": {
            "is_on_domain": {"type": "noul", "noul": 0.92},
            "is_coherent_and_direct": {"type": "noul", "noul": 0.85},
            "quality_tier": {"type": "score", "score": 2.2},
        },
        "usage": {"input_tokens": 500, "cost": 0.0006},
    }

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(pass_resp_data).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        # Call 1: within budget, passing criteria
        ok1 = gate.evaluate_one("Technical explanation of binary search trees.")
        assert ok1 is True
        assert gate.total_cost == 0.0006
        assert gate.circuit_broken is False

        # Call 2: incurs another 0.0006 -> total 0.0012 >= max_usd (0.001) -> trips circuit breaker
        ok2 = gate.evaluate_one("Another technical explanation.")
        assert ok2 is True
        assert gate.total_cost >= 0.001

        # Call 3: circuit breaker is tripped, returns True without making any HTTP request
        gate.evaluate_one = MagicMock(wraps=gate.evaluate_one)
        ok3 = gate.evaluate_one("Third document after circuit break.")
        assert ok3 is True
        assert gate.circuit_broken is True
