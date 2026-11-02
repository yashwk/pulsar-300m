import numpy as np
import pytest

from src.packing import (
    BESTFIT,
    PackPlan,
    build_doc_index,
    plan_epoch,
)


def test_plan_epoch_global_window_shuffling():
    """Verify that global window shuffling uniformly distributes whole documents and filler."""
    seq_len = 1024
    rng = np.random.default_rng(42)

    # Construct an asymmetric corpus:
    # 100 short documents (<= 1024 tokens) total ~50k tokens (~10% of corpus)
    # 50 long documents (> 1024 tokens, e.g. 8000-12000 tokens) total ~450k tokens (~90% of corpus)
    short_lens = rng.integers(200, 800, size=100)
    long_lens = rng.integers(6000, 12000, size=50)

    all_lens = np.concatenate([short_lens, long_lens])
    total_tokens = int(all_lens.sum())

    plan = plan_epoch(all_lens, seq_len=seq_len, seed=123, strategy=BESTFIT)
    n_windows = len(plan.windows)
    assert n_windows > 10, "Plan should contain sufficient windows for distribution testing"

    # All tokens accounted for
    emitted_tokens = sum(w.tokens for w in plan.windows)
    assert plan.dropped_tail_tokens == 0 or plan.dropped_tail_tokens < seq_len
    assert total_tokens - emitted_tokens == plan.dropped_tail_tokens

    # Check window distribution in first half vs second half
    mid = n_windows // 2
    first_half = plan.windows[:mid]
    second_half = plan.windows[mid:]

    # A whole-document-heavy window has filler_tokens < seq_len // 2
    first_whole_wins = sum(1 for w in first_half if w.filler_tokens < seq_len // 2)
    second_whole_wins = sum(1 for w in second_half if w.filler_tokens < seq_len // 2)

    total_whole_wins = first_whole_wins + second_whole_wins
    assert total_whole_wins > 0, "There should be whole-document windows"

    # In an unshuffled plan, ALL whole windows would be in the first half (ratio ~100% vs 0%).
    # With global window shuffling, both halves should receive a substantial share (e.g. at least 25% each).
    first_ratio = first_whole_wins / total_whole_wins
    second_ratio = second_whole_wins / total_whole_wins

    assert 0.25 <= first_ratio <= 0.75, (
        f"Whole document windows not uniformly distributed: first half has {first_ratio:.2%}, "
        f"second half has {second_ratio:.2%}"
    )
    assert 0.25 <= second_ratio <= 0.75

    # Check average filler fraction across both halves
    first_mean_filler = np.mean([w.filler_tokens for w in first_half])
    second_mean_filler = np.mean([w.filler_tokens for w in second_half])

    # The mean filler tokens per window should be within 25% of each other
    filler_diff_ratio = abs(first_mean_filler - second_mean_filler) / max(first_mean_filler, second_mean_filler)
    assert filler_diff_ratio < 0.25, f"Filler distribution between halves differs by {filler_diff_ratio:.2%}"


def test_plan_epoch_determinism():
    """Verify that same seed and epoch produces identical window order."""
    lens = np.array([300, 1500, 400, 2500, 600, 800, 3000, 500], dtype=np.int64)
    plan1 = plan_epoch(lens, seq_len=1024, seed=999, epoch=0)
    plan2 = plan_epoch(lens, seq_len=1024, seed=999, epoch=0)

    assert len(plan1.windows) == len(plan2.windows)
    for w1, w2 in zip(plan1.windows, plan2.windows):
        np.testing.assert_array_equal(w1.docs, w2.docs)
        np.testing.assert_array_equal(w1.offs, w2.offs)
        np.testing.assert_array_equal(w1.lens, w2.lens)
        assert w1.filler_tokens == w2.filler_tokens
        assert w1.pad == w2.pad
