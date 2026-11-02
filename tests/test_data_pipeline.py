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
    clean_doc_text,
    distill_code_cot,
    distill_math_cot,
    is_auto_accepted,
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
    gate.evaluate_batch = MagicMock(return_value=[True])

    candidates = [
        ("1A", "def foo(): pass"),
        ("3A", "<|tool_call|>{...}<|/tool_call|>"),
        ("3B", "```toml\nkey = 'val'\n```"),
    ]

    filtered = gate.filter_batch(candidates)
    assert len(filtered) == 3
    # evaluate_batch should not be called because all candidates are in deterministic bypass streams
    assert gate.evaluate_batch.call_count == 0


def test_jev_quality_gate_decision_and_budget_circuit_breaker():
    """Verify Jev decision thresholds (score >= 1.65), cost accumulation, and circuit breaker trip."""
    gate = JevQualityGate(
        api_key="mock_key",
        api_url="https://mock.api/v1/systemone",
        max_usd=0.001,  # Tiny budget cap to test circuit breaker
        enabled=True,
    )

    pass_resp_data = {
        "answers": {
            "doc_0": {"type": "score", "score": 2.2},
            "doc_1": {"type": "score", "score": 1.2},  # Rejection: < 1.65
        },
        "usage": {"input_tokens": 500, "cost": 0.0006},
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = pass_resp_data

    # Mock post response
    patch_target = "requests.Session.post" if gate._session is not None else "urllib.request.urlopen"
    with patch(patch_target, return_value=mock_resp):
        # Call 1: doc_0 passes (2.2 >= 1.65), doc_1 fails (1.2 < 1.65)
        res = gate.evaluate_batch(["Technical explanation of binary search trees.", "Low-quality trivia snippet."])
        assert res == [True, False]
        assert gate.total_cost == 0.0006
        assert gate.circuit_broken is False

        # Call 2: incurs another 0.0006 -> total 0.0012 >= max_usd (0.001) -> trips circuit breaker
        res2 = gate.evaluate_batch(["Another technical explanation."])
        assert res2 == [True]
        assert gate.total_cost >= 0.001
        assert gate.circuit_broken is True

        # Call 3: circuit breaker is tripped, returns [True] immediately without calling post
        with patch(patch_target, side_effect=Exception("Should not be called")):
            res3 = gate.evaluate_batch(["Third document after circuit break.", "Fourth document."])
            assert res3 == [True, True]


def test_jev_quality_gate_multi_doc_batching():
    """Verify packing of up to 32 candidate document digests into a single non-causal isolated request."""
    gate = JevQualityGate(
        api_key="mock_key",
        api_url="https://mock.api/v1/systemone",
        model="jev-1.14.0",
        batch_size=32,
        enabled=True,
    )

    captured_payloads = []

    def mock_post_with_retry(payload):
        captured_payloads.append(payload)
        n = len(payload["questions"])
        return {
            "answers": {f"doc_{i}": {"type": "score", "score": 2.0} for i in range(n)},
            "usage": {"input_tokens": 100 * n, "cost": 0.0001 * n},
        }

    gate._post_with_retry = MagicMock(side_effect=mock_post_with_retry)

    # 35 candidates should be evaluated in 2 batches (32 and 3)
    candidates = [(f"Candidate document {i} with technical CS explanation. " * 30) for i in range(35)]
    results = gate.evaluate_batch(candidates)

    assert len(results) == 35
    assert all(results)
    assert gate._post_with_retry.call_count == 2

    first_payload = captured_payloads[0]
    assert first_payload["model"] == "jev-1.14.0"
    assert "strict pretraining data quality filter" in first_payload["state"]
    assert len(first_payload["questions"]) == 32
    assert "doc_0" in first_payload["questions"]
    assert "doc_31" in first_payload["questions"]

    # Check question isolation format and digest capping
    q0 = first_payload["questions"]["doc_0"]
    assert q0["type"] == "score"
    assert len(q0["instructions"]) <= 700
    assert any("reject:" in c for c in q0["criteria"])
    assert any("excellent:" in c for c in q0["criteria"])

    second_payload = captured_payloads[1]
    assert len(second_payload["questions"]) == 3


def test_jev_quality_gate_429_retry_and_circuit_breaker():
    """Verify HTTP 429 backoff retry, 401/402/403 circuit tripping, and 5-failure threshold."""
    gate = JevQualityGate(api_key="mock_key", enabled=True)

    # 1. Test 429 retry handling: 2 rate limits then success
    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.headers = {"Retry-After": "0.1"}

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = {
        "answers": {"doc_0": {"type": "score", "score": 2.0}},
        "usage": {"cost": 0.0001},
    }

    if gate._session is not None:
        with patch.object(gate._session, "post", side_effect=[resp_429, resp_429, resp_200]) as mock_post:
            with patch("time.sleep") as mock_sleep:
                res = gate._post_with_retry({"model": "jev-1.14.0", "questions": {}})
                assert res is not None
                assert mock_post.call_count == 3
                assert mock_sleep.call_count == 2
                assert gate.circuit_broken is False
                assert gate.consecutive_failures == 0

    # 2. Test 401 auth / credit error: trips circuit breaker immediately
    gate = JevQualityGate(api_key="mock_key", enabled=True)
    resp_401 = MagicMock()
    resp_401.status_code = 401
    if gate._session is not None:
        with patch.object(gate._session, "post", return_value=resp_401):
            res = gate._post_with_retry({"model": "jev-1.14.0", "questions": {}})
            assert res is None
            assert gate.circuit_broken is True

    # 3. Test 5 consecutive failures trips circuit breaker
    gate = JevQualityGate(api_key="mock_key", enabled=True)
    resp_500 = MagicMock()
    resp_500.status_code = 500
    resp_500.headers = {}
    if gate._session is not None:
        with patch.object(gate._session, "post", return_value=resp_500):
            with patch("time.sleep"):
                for _ in range(4):
                    gate._post_with_retry({"model": "jev-1.14.0", "questions": {}})
                    assert gate.circuit_broken is False
                gate._post_with_retry({"model": "jev-1.14.0", "questions": {}})
                assert gate.circuit_broken is True


def test_two_tier_pre_filter_auto_accept():
    """Verify Tier-1 $0.00 auto-accept rules for 1B, 2B, 4B and routing of ambiguous streams."""
    # Test clean_doc_text stripping fineweb_score header
    raw_4b = "<!-- fineweb_score: 4.35 -->\nThis is educational STEM content on linear algebra."
    assert clean_doc_text(raw_4b) == "This is educational STEM content on linear algebra."
    assert clean_doc_text("Plain text") == "Plain text"

    # 1B: Valid Python with non-rambling think trace -> auto-accept
    valid_1b = (
        "<|think|>\nLet's implement a clean partition function for quicksort.\n<|/think|>\n"
        "```python\ndef partition(arr, lo, hi):\n    pivot = arr[hi]\n    i = lo\n    return i\n```"
    )
    assert is_auto_accepted("1B", valid_1b) is True

    # 1B: Invalid Python syntax -> not auto-accepted (sent to Jev)
    invalid_1b = (
        "<|think|>\nSome thought.\n<|/think|>\n"
        "```python\ndef broken(:\n    pass\n```"
    )
    assert is_auto_accepted("1B", invalid_1b) is False

    # 1B: Rambling think trace -> not auto-accepted
    ramble_1b = (
        "<|think|>\nWait, let me rethink this. Actually, no, let me reconsider what happens. Wait, let me rethink...\n<|/think|>\n"
        "```python\ndef ok(): pass\n```"
    )
    assert is_auto_accepted("1B", ramble_1b) is False

    # 2B: Concise think block (<= 1400 chars) with \\boxed{} -> auto-accept
    valid_2b = (
        "<|think|>\nThe perimeter is 2 * (w + h) = 2 * (5 + 10) = 30.\n<|/think|>\n"
        "The perimeter is \\boxed{30}."
    )
    assert is_auto_accepted("2B", valid_2b) is True

    # 2B: Unboxed answer -> not auto-accepted
    unboxed_2b = (
        "<|think|>\nCompute the perimeter.\n<|/think|>\nThe perimeter is 30."
    )
    assert is_auto_accepted("2B", unboxed_2b) is False

    # 2B: Overly long think block (> 1400 chars) -> not auto-accepted
    long_think_2b = (
        "<|think|>\n" + ("Long derivation step. " * 100) + "\n<|/think|>\n"
        "The result is \\boxed{42}."
    )
    assert is_auto_accepted("2B", long_think_2b) is False

    # 4B: Score >= 4.2 with >= 4 STEM keywords -> auto-accept
    valid_4b = (
        "<!-- fineweb_score: 4.40 -->\n"
        "In linear algebra, a matrix eigenvalue problem satisfies Av = lambda v. "
        "The algorithm uses vector projection and calculus to compute the theorem."
    )
    assert is_auto_accepted("4B", valid_4b) is True

    # 4B: Borderline score (3.8) -> not auto-accepted (sent to Jev)
    borderline_4b = (
        "<!-- fineweb_score: 3.80 -->\n"
        "In linear algebra, a matrix eigenvalue problem satisfies Av = lambda v. "
        "The algorithm uses vector projection and calculus to compute the theorem."
    )
    assert is_auto_accepted("4B", borderline_4b) is False

    # 4A and 5A: always not auto-accepted (always routed to Jev)
    assert is_auto_accepted("4A", "STEM encyclopedia article on entropy.") is False
    assert is_auto_accepted("5A", "<|im_start|>user\nExplain TCP.<|im_end|>") is False

    # Integration test with filter_batch:
    gate = JevQualityGate(api_key="mock_key", enabled=True)

    def mock_eval_batch(batch_texts):
        return ["Pass" in t for t in batch_texts]

    gate.evaluate_batch = MagicMock(side_effect=mock_eval_batch)

    candidates = [
        ("1B", valid_1b),  # auto-accepted ($0.00)
        ("4A", "Candidate 4A Pass"),  # ambiguous -> evaluated and passes
        ("5A", "Candidate 5A Fail"),  # ambiguous -> evaluated and fails
    ]

    filtered = gate.filter_batch(candidates)
    assert len(filtered) == 2
    substreams = [s for s, txt in filtered]
    assert "1B" in substreams
    assert "4A" in substreams
    assert "5A" not in substreams
