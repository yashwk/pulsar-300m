import numpy as np
import pytest

from src.packing import (
    BESTFIT,
    IGNORE_INDEX,
    DocIndex,
    PackedBatchLoader,
    PackPlan,
    Window,
    build_chatml_labels,
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


def test_build_chatml_labels_assistant_masking():
    """Verify assistant-only loss masking correctly masks prompt turns while preserving assistant tokens."""
    # Token IDs: im_start=1, im_end=2, think=3, /think=4, tool_resp=7, /tool_resp=8, assistant=957, user=1457, system=2240, \n=210
    part = np.array([
        # System turn (indices 0..5)
        1, 2240, 210, 101, 102, 2,
        # User turn (indices 6..10)
        1, 1457, 210, 201, 2,
        # Assistant turn with reasoning and response (indices 11..18)
        # <im_start>(11) assistant(12) \n(13) <think>(14) 301(15) </think>(16) 302(17) <im_end>(18)
        1, 957, 210, 3, 301, 4, 302, 2,
    ], dtype=np.int64)

    y = build_chatml_labels(part, ignore_index=IGNORE_INDEX)

    # All tokens in system turn (0..5) should not be predicted: y[0..5] == -100
    assert (y[:6] == IGNORE_INDEX).all(), "System turn targets must be masked (-100)"

    # All tokens in user turn (6..10) should not be predicted: y[6..10] == -100
    assert (y[6:11] == IGNORE_INDEX).all(), "User turn targets must be masked (-100)"

    # Assistant header (<im_start> assistant) at indices 11, 12 should not be predicted
    assert y[11] == IGNORE_INDEX, "im_start token target must be masked"
    assert y[12] == IGNORE_INDEX, "assistant role token target must be masked"

    # Active assistant predictions:
    # At index 13 (\n): predicts token 14 (<think>=3)
    assert y[13] == 3, f"Expected <think> (3) target at index 13, got {y[13]}"
    # At index 14 (<think>=3): predicts token 15 (301)
    assert y[14] == 301, f"Expected 301 target at index 14, got {y[14]}"
    # At index 15 (301): predicts token 16 (</think>=4)
    assert y[15] == 4, f"Expected </think> (4) target at index 15, got {y[15]}"
    # At index 16 (</think>=4): predicts token 17 (302)
    assert y[16] == 302, f"Expected 302 target at index 16, got {y[16]}"
    # At index 17 (302): predicts token 18 (<im_end>=2)
    assert y[17] == 2, f"Expected <im_end> (2) target at index 17, got {y[17]}"

    # At index 18 (<im_end>=2): turn is completed, next token must be masked
    assert y[18] == IGNORE_INDEX, "Tokens after assistant turn must be masked (-100)"


def test_packed_batch_loader_mask_prompt_loss():
    """Verify PackedBatchLoader masks prompt loss on ChatML and falls back on plain docs."""
    # Document 1: Plain code document (no <im_start>)
    plain_doc = np.array([500, 501, 502, 503, 504, 0], dtype=np.uint16)

    # Document 2: ChatML conversation
    # system (6 tokens) + user (5 tokens) + assistant (8 tokens) = 19 tokens + 1 EOS = 20 tokens
    chatml_doc = np.array([
        1, 2240, 210, 101, 102, 2,
        1, 1457, 210, 201, 2,
        1, 957, 210, 3, 301, 4, 302, 2,
        0, # EOS
    ], dtype=np.uint16)

    shard = np.concatenate([plain_doc, chatml_doc])
    index = build_doc_index([shard])
    plan = plan_epoch(index.length.astype(np.int64), seq_len=32, seed=42, strategy=BESTFIT)

    # 1. Loader with mask_prompt_loss=False (Stage 1 behavior)
    loader_unmasked = PackedBatchLoader(
        [shard], index, plan, micro_batch_size=1, device="cpu", mask_prompt_loss=False,
    )
    x_unmasked, y_unmasked = loader_unmasked.get_batch()
    # Unmasked: both plain and ChatML prompt tokens are predicted
    # In unmasked loader, find ChatML user token position and check it has valid target
    chatml_start = int(np.where(x_unmasked[0].numpy() == 1)[0][0])
    # Next token after im_start in unmasked loader is system (2240)
    assert y_unmasked[0, chatml_start].item() == 2240, "Unmasked loader must predict prompt tokens"

    # 2. Loader with mask_prompt_loss=True (Stage 2 behavior)
    loader_masked = PackedBatchLoader(
        [shard], index, plan, micro_batch_size=1, device="cpu", mask_prompt_loss=True,
    )
    x_masked, y_masked = loader_masked.get_batch()
    x_list = x_masked[0].tolist()
    y_list = y_masked[0].tolist()

    # Plain document segment: no <im_start>, should have standard next-token targets
    plain_pos = x_list.index(500)
    assert y_list[plain_pos] == 501, "Plain document must keep causal next-token targets"
    assert y_list[plain_pos + 1] == 502

    # ChatML segment: system and user tokens must have target -100
    sys_pos = x_list.index(2240)
    user_pos = x_list.index(1457)
    assert y_list[sys_pos] == IGNORE_INDEX, "System token target must be -100 when masked"
    assert y_list[user_pos] == IGNORE_INDEX, "User token target must be -100 when masked"

    # Assistant targets must be preserved
    # Find position of assistant newline (210 immediately after 957)
    asst_pos = x_list.index(957)
    asst_nl_pos = asst_pos + 1
    assert x_list[asst_nl_pos] == 210
    assert y_list[asst_nl_pos] == 3, "Assistant <think> target must be preserved"
    assert y_list[asst_nl_pos + 1] == 301, "Assistant body target must be preserved"
    assert y_list[asst_nl_pos + 2] == 4, "Assistant </think> target must be preserved"
    assert y_list[asst_nl_pos + 3] == 302, "Assistant answer target must be preserved"
    assert y_list[asst_nl_pos + 4] == 2, "Assistant closing <im_end> target must be preserved"
    assert y_list[asst_nl_pos + 5] == IGNORE_INDEX, "Post-turn token must be -100"
