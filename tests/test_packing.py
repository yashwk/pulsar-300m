"""Tests for EOS-aligned document packing (WI-07/WI-10 loader groundwork).

The important behaviours:

* packing never pads, and every window is filled exactly when enough documents exist;
* a document that fits a window is never split across two windows if avoidable;
* documents longer than the window are chunked across consecutive windows;
* the plan is a pure function of (seed, epoch) so resuming is exact;
* and - the measurement the owner asked for - intact-document fraction rises from the
  flat sampler to packed packing at T=2048.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.packing import (
    DocIndex,
    PackedBatchLoader,
    build_doc_index,
    flat_window_ids,
    intact_fractions,
    plan_epoch,
    waste_fractions,
)

EOS = 0


def _shard_from_docs(doc_lengths, eos=EOS, start_token=1):
    """Build a flat shard: each document is `len` tokens, the last one being EOS.

    Body tokens avoid ``eos`` (0) and stay inside the uint16 vocab, so document
    boundaries in the fixture are exactly the ones we intend to create.
    """
    out = []
    tok = start_token
    for ln in doc_lengths:
        ln = int(ln)
        assert ln >= 1
        body = [(tok + i) % 60000 + 1 for i in range(ln - 1)]  # +1 keeps 0 out
        out.extend(body)
        out.append(eos)
        tok += ln
    return np.asarray(out, dtype=np.uint16)


# ---------------------------------------------------------------------------
# Document index
# ---------------------------------------------------------------------------

def test_build_doc_index_finds_boundaries():
    lengths = [5, 12, 3]
    shard = _shard_from_docs(lengths)
    idx = build_doc_index([shard])
    assert len(idx) == 3
    assert idx.length.tolist() == lengths
    assert idx.start.tolist() == [0, 5, 17]
    # every document ends with EOS
    for s, ln in zip(idx.start, idx.length):
        assert int(shard[s + ln - 1]) == EOS


def test_no_tokens_lost():
    rng = np.random.default_rng(4)
    lengths = rng.integers(1, 1500, size=300).astype(np.int64)
    plan = plan_epoch(lengths, 1024, seed=0, epoch=0)
    emitted = sum(w.tokens for w in plan.windows)
    assert emitted + plan.dropped_tail_tokens == int(lengths.sum())


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def test_loader_shapes_and_targets():
    rng = np.random.default_rng(5)
    lengths = rng.integers(100, 400, size=80).astype(np.int64)
    shard = _shard_from_docs(lengths)
    idx = build_doc_index([shard])
    plan = plan_epoch(lengths, 256, seed=0, epoch=0)
    loader = PackedBatchLoader([shard], idx, plan, micro_batch_size=2, device="cpu")

    x, y = loader.get_batch()
    assert x.shape == (2, 256) and y.shape == (2, 256)
    assert x.dtype == torch.int64
    assert torch.equal(x[:, 1:], y[:, :-1]), "targets must be next-token"


def test_filler_only_from_over_long_documents():
    """Every segment with offs > 0 must belong to a document longer than seq_len."""
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
                continue  # a whole document
            # anything not whole must be a cut of an over-long document
            assert lengths[d] > T, (
                f"document {d} (len={lengths[d]}) was cut but could have been intact at T={T}"
            )


def test_whole_documents_are_never_split():
    """A document that fits must never appear as two partial segments."""
    rng = np.random.default_rng(12)
    lengths = rng.integers(20, 400, size=250).astype(np.int64)
    T = 1024
    plan = plan_epoch(lengths, T, seed=1, epoch=0)

    seen_whole, seen_partial = set(), set()
    for w in plan.windows:
        for d, off, ln in zip(w.docs, w.offs, w.lens):
            d, off, ln = int(d), int(off), int(ln)
            if off == 0 and ln == int(lengths[d]):
                seen_whole.add(d)
            else:
                seen_partial.add(d)
    assert not (seen_whole & seen_partial), "a document was both whole and split"


def test_each_token_covered_exactly_once_per_epoch():
    """No token is dropped and none is trained on twice (padding aside)."""
    rng = np.random.default_rng(9)
    lens = np.concatenate([
        rng.integers(20, 400, 300),
        rng.integers(500, 2500, 150),
    ]).astype(np.int64)

    for T in (512, 1024, 2048):
        plan = plan_epoch(lens, T, seed=0, epoch=0)
        counts = np.zeros(int(lens.sum()), dtype=np.int32)
        base = np.concatenate(([0], np.cumsum(lens)[:-1]))
        for w in plan.windows:
            for d, off, ln in zip(w.docs, w.offs, w.lens):
                d, off, ln = int(d), int(off), int(ln)
                counts[base[d] + off: base[d] + off + ln] += 1
        assert counts.min() == 1 and counts.max() == 1, (
            f"T={T}: expected every token exactly once, got min={counts.min()} max={counts.max()}"
        )
