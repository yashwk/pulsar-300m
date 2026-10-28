"""Tests for EOS-aligned document packing and packed loader contracts.

Covers:
- Document index boundary detection and token preservation.
- Best-fit plan_epoch bin packing without duplicate or dropped tokens.
- Next-token target construction and cross-document EOS masking.
- Intact document preservation and over-long chunking.
- Memory boundedness in PackedBatchLoader._last_n_real.
- Routing mask creation for padded sequences.
- Deterministic sequence reset and evaluation reproducibility.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import make_cfg
from src.model import PulsarForCausalLM
from src.packing import (
    DocIndex,
    PackedBatchLoader,
    build_doc_index,
    plan_epoch,
)
from src.train import evaluate_loss

EOS = 0


def _shard_from_docs(doc_lengths, eos=EOS, start_token=1):
    out = []
    tok = start_token
    for ln in doc_lengths:
        ln = int(ln)
        assert ln >= 1
        body = [(tok + i) % 60000 + 1 for i in range(ln - 1)]
        out.extend(body)
        out.append(eos)
        tok += ln
    return np.asarray(out, dtype=np.uint16)


def test_build_doc_index_finds_boundaries():
    """DocIndex identifies exact boundary positions and lengths of all documents."""
    lengths = [5, 12, 3]
    shard = _shard_from_docs(lengths)
    idx = build_doc_index([shard])
    assert len(idx) == 3
    assert idx.length.tolist() == lengths
    assert idx.start.tolist() == [0, 5, 17]
    for s, ln in zip(idx.start, idx.length):
        assert int(shard[s + ln - 1]) == EOS


def test_no_tokens_lost():
    """plan_epoch accounts for 100% of tokens across packed windows and dropped tails."""
    rng = np.random.default_rng(4)
    lengths = rng.integers(1, 1500, size=300).astype(np.int64)
    plan = plan_epoch(lengths, 1024, seed=0, epoch=0)
    emitted = sum(w.tokens for w in plan.windows)
    assert emitted + plan.dropped_tail_tokens == int(lengths.sum())


def test_loader_shapes_and_cross_doc_masking():
    """Loader outputs valid tensor shapes and masks cross-document targets at EOS boundaries."""
    rng = np.random.default_rng(5)
    lengths = rng.integers(100, 400, size=80).astype(np.int64)
    shard = _shard_from_docs(lengths)
    idx = build_doc_index([shard])
    plan = plan_epoch(lengths, 256, seed=0, epoch=0)
    loader = PackedBatchLoader([shard], idx, plan, micro_batch_size=2, device="cpu")

    x, y = loader.get_batch()
    assert x.shape == (2, 256) and y.shape == (2, 256)
    assert x.dtype == torch.int64

    is_eos = x[:, :-1] == 0
    eq = x[:, 1:] == y[:, :-1]
    assert bool(eq[~is_eos].all()), "non-boundary targets must be next-token"
    assert bool((y[:, :-1][is_eos] == -100).all()), "cross-document targets must be masked with -100"


def test_filler_only_from_over_long_documents():
    """Only documents longer than seq_len are partitioned into filler slices."""
    rng = np.random.default_rng(11)
    lengths = np.concatenate([
        rng.integers(20, 300, size=100),
        rng.integers(600, 1500, size=40),
    ]).astype(np.int64)
    T = 512
    plan = plan_epoch(lengths, T, seed=0, epoch=0)

    for w in plan.windows:
        for d, off, ln in zip(w.docs, w.offs, w.lens):
            d, off, ln = int(d), int(off), int(ln)
            if off == 0 and ln == int(lengths[d]):
                continue
            assert lengths[d] > T


def test_whole_documents_are_never_split():
    """Fitting documents are never split across multiple windows."""
    rng = np.random.default_rng(12)
    lengths = rng.integers(20, 400, size=250).astype(np.int64)
    plan = plan_epoch(lengths, 1024, seed=1, epoch=0)

    seen_whole, seen_partial = set(), set()
    for w in plan.windows:
        for d, off, ln in zip(w.docs, w.offs, w.lens):
            d, off, ln = int(d), int(off), int(ln)
            if off == 0 and ln == int(lengths[d]):
                seen_whole.add(d)
            else:
                seen_partial.add(d)
    assert not (seen_whole & seen_partial)


@pytest.mark.parametrize("T", [512, 1024, 2048])
def test_each_token_covered_exactly_once_per_epoch(T):
    """Every token in the shard is sampled exactly once per training epoch."""
    rng = np.random.default_rng(9)
    lens = np.concatenate([
        rng.integers(20, 400, 300),
        rng.integers(500, 2500, 150),
    ]).astype(np.int64)

    plan = plan_epoch(lens, T, seed=0, epoch=0)
    counts = np.zeros(int(lens.sum()), dtype=np.int32)
    base = np.concatenate(([0], np.cumsum(lens)[:-1]))
    for w in plan.windows:
        for d, off, ln in zip(w.docs, w.offs, w.lens):
            d, off, ln = int(d), int(off), int(ln)
            counts[base[d] + off: base[d] + off + ln] += 1
    assert counts.min() == 1 and counts.max() == 1


def test_loader_last_n_real_bounded_and_routing_mask(synthetic_shards):
    """PackedBatchLoader cleans up last_n_real per batch and produces valid routing masks."""
    train_path, _ = synthetic_shards
    shard = np.fromfile(train_path, dtype=np.uint16)
    idx = build_doc_index([shard])
    lengths = idx.length.astype(np.int64)
    plan = plan_epoch(lengths, 64, seed=0, epoch=0)
    loader = PackedBatchLoader([shard], idx, plan, micro_batch_size=2, device="cpu")

    for _ in range(5):
        loader.get_batch()
    assert len(loader._last_n_real) <= loader.micro_batch_size

    pad_idx = next((i for i, w in enumerate(loader.plan.windows) if w.pad > 0), None)
    if pad_idx is not None:
        x, _ = loader.get_batch([pad_idx])
        mask = loader.routing_mask(x)
        assert mask is not None
        assert mask.shape == x.shape
        assert mask.dtype == torch.bool


def test_evaluate_loss_uses_identical_windows_on_consecutive_calls(synthetic_shards):
    """evaluate_loss resets the loader cursor so validation loss is strictly deterministic."""
    _, val_path = synthetic_shards
    shard = np.fromfile(val_path, dtype=np.uint16)
    index = build_doc_index([shard])
    lengths = index.length.astype(np.int64)
    plan = plan_epoch(lengths, 64, seed=0, epoch=0)
    loader = PackedBatchLoader([shard], index, plan, micro_batch_size=2, device="cpu")

    cfg = make_cfg(n_core=2, d_model=64, vocab_size=256, top_k=2, local_attn_impl="sdpa")
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg).eval()

    first = evaluate_loss(model, loader, eval_iters=2, num_pulses=2)
    cursor_after_first = loader._cursor
    second = evaluate_loss(model, loader, eval_iters=2, num_pulses=2)

    assert first["val_ce_final"] == pytest.approx(second["val_ce_final"], rel=1e-6)
    assert loader._cursor == cursor_after_first
