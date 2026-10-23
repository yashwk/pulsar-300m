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


def test_build_doc_index_multi_shard():
    s1 = _shard_from_docs([4, 6])
    s2 = _shard_from_docs([8])
    idx = build_doc_index([s1, s2])
    assert len(idx) == 3
    assert idx.shard.tolist() == [0, 0, 1]
    assert idx.total_tokens == len(s1) + len(s2)
    # global starts are in the concatenated stream
    g = idx.global_start
    assert g[0] == 0 and g[1] == 4 and g[2] == len(s1)


# ---------------------------------------------------------------------------
# Flat vs packed: intactness
# ---------------------------------------------------------------------------

def test_flat_sampler_cuts_many_documents():
    """A median-length document at T=2048 is usually cut by the flat stride."""
    rng = np.random.default_rng(0)
    lengths = rng.integers(200, 2200, size=400).astype(np.int64)
    shard = _shard_from_docs(lengths)
    idx = build_doc_index([shard])
    gs = idx.global_start

    res = intact_fractions(lengths, global_start=gs, seq_len=2048)
    # Expect well below 1.0: the stride cuts documents that individually "fit".
    assert res["count"] < 0.75, f"flat sampler unexpectedly intact: {res}"
    assert res["documents"] == 400


def test_packing_keeps_most_fitting_documents_intact():
    """Packing should beat the flat sampler substantially at T=2048."""
    rng = np.random.default_rng(1)
    lengths = rng.integers(200, 2200, size=400).astype(np.int64)
    shard = _shard_from_docs(lengths)
    idx = build_doc_index([shard])

    flat = intact_fractions(lengths, global_start=idx.global_start, seq_len=2048)
    plan = plan_epoch(lengths, 2048, seed=0, epoch=0, strategy="bestfit")
    packed = intact_fractions(lengths, plan=plan, seq_len=2048)

    assert packed["count"] > flat["count"], (
        f"packing must not reduce intactness: flat={flat['count']:.3f} packed={packed['count']:.3f}"
    )
    assert packed["count"] > 0.8, f"packed intact fraction too low: {packed}"


def test_packing_windows_are_full_when_filler_available():
    """With over-long documents available as filler, windows should be exactly full.

    Padding is only permitted when neither pool can supply tokens, so a corpus with
    plenty of over-long documents should produce essentially zero waste.
    """
    rng = np.random.default_rng(2)
    # 40% of documents exceed the window, guaranteeing filler supply
    lengths = np.concatenate([
        rng.integers(50, 900, size=120),
        rng.integers(1100, 4000, size=80),
    ]).astype(np.int64)
    plan = plan_epoch(lengths, 1024, seed=3, epoch=0)
    assert len(plan) > 0

    waste = waste_fractions(plan)
    assert waste["padding_frac"] < 0.02, f"padding should be ~0 with filler supply: {waste}"
    assert waste["filler_tokens"] > 0, "filler phase should have been used"
    # all windows but possibly the terminal one are exactly full
    full = sum(1 for w in plan.windows if w.tokens + w.pad == plan.seq_len)
    assert full >= len(plan) - 1, f"only {full}/{len(plan)} windows are full"


def test_padding_only_at_epoch_end():
    """Windows are padded only when no document or filler is left."""
    # Only documents much smaller than the window, count not a multiple of capacity:
    # the terminal window cannot be filled.
    lengths = np.full(7, 100, dtype=np.int64)
    plan = plan_epoch(lengths, 1024, seed=0, epoch=0)
    assert len(plan) == 1
    w = plan.windows[0]
    assert w.tokens == 700
    assert w.pad == 1024 - 700
    assert plan.pad_tokens == 324


def test_documents_longer_than_window_are_chunked():
    lengths = np.array([5000], dtype=np.int64)
    plan = plan_epoch(lengths, 1024, seed=0, epoch=0)
    total = sum(w.tokens for w in plan.windows)
    assert total == 5000, "all tokens of a long document must be emitted"
    # each window holds a chunk of the same document with increasing offset
    offs = [int(w.offs[0]) for w in plan.windows]
    assert offs == sorted(offs) and offs[0] == 0


def test_packing_is_deterministic_and_epoch_dependent():
    rng = np.random.default_rng(3)
    lengths = rng.integers(100, 900, size=150).astype(np.int64)
    p1 = plan_epoch(lengths, 512, seed=7, epoch=0)
    p2 = plan_epoch(lengths, 512, seed=7, epoch=0)
    p3 = plan_epoch(lengths, 512, seed=7, epoch=1)

    def sig(p):
        return [(w.docs.tolist(), w.offs.tolist(), w.lens.tolist()) for w in p.windows]

    assert sig(p1) == sig(p2), "same (seed, epoch) must reproduce the same plan"
    assert sig(p1) != sig(p3), "different epoch should reshuffle"


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


def test_loader_starts_windows_at_document_boundaries():
    """The first token of each window is the first token of some document."""
    rng = np.random.default_rng(6)
    lengths = rng.integers(50, 300, size=60).astype(np.int64)
    shard = _shard_from_docs(lengths)
    idx = build_doc_index([shard])
    plan = plan_epoch(lengths, 256, seed=1, epoch=0)
    loader = PackedBatchLoader([shard], idx, plan, micro_batch_size=1, device="cpu")

    doc_first_tokens = {int(shard[s]) for s in idx.start}
    for wid in range(min(20, len(plan))):
        x, _ = loader.get_batch(window_ids=[wid])
        assert int(x[0, 0]) in doc_first_tokens, f"window {wid} does not start a document"


# ---------------------------------------------------------------------------
# C' contract: filler only from documents that can never be intact
# ---------------------------------------------------------------------------

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


def test_intact_among_fittable_is_high_at_2048():
    """The headline property: documents that can fit, do."""
    rng = np.random.default_rng(13)
    lengths = np.clip(rng.lognormal(np.log(1278), 0.35, 500), 100, 8000).astype(np.int64)
    T = 2048
    plan = plan_epoch(lengths, T, seed=0, epoch=0)
    res = intact_fractions(lengths, plan=plan, seq_len=T)
    assert res["count_among_fittable"] > 0.95, res


def test_padding_labels_are_ignored():
    """Loader must mark label positions that predict padding as ignore_index."""
    lengths = np.full(5, 100, dtype=np.int64)
    shard = _shard_from_docs(lengths)
    idx = build_doc_index([shard])
    plan = plan_epoch(lengths, 512, seed=0, epoch=0)
    loader = PackedBatchLoader([shard], idx, plan, micro_batch_size=1, device="cpu")
    x, y = loader.get_batch()
    # `x == 0` is not a padding indicator: every document ends with its own EOS token
    # (id 0). The plan's segment lengths give the real content length.
    n_real = plan.windows[0].tokens
    n_pad = 512 - n_real
    assert n_pad == plan.windows[0].pad > 0

    # exactly the padding positions plus the final real position are ignored: the last
    # real token has no successor inside the window
    assert int((y == -100).sum()) == n_pad + 1, "unexpected label masking"
    assert torch.all(y[0, n_real - 1:] == -100)
    # labels within the real region are exactly the next tokens
    assert torch.equal(y[0, : n_real - 1], x[0, 1:n_real])
    # padding positions in x are the trailing run after the content
    assert torch.all(x[0, n_real:] == 0)


def test_routing_mask_excludes_padding():
    lengths = np.full(5, 100, dtype=np.int64)
    shard = _shard_from_docs(lengths)
    idx = build_doc_index([shard])
    plan = plan_epoch(lengths, 512, seed=0, epoch=0)
    loader = PackedBatchLoader([shard], idx, plan, micro_batch_size=1, device="cpu")
    x, _ = loader.get_batch()
    mask = loader.routing_mask(x)
    n_real = plan.windows[0].tokens
    assert mask is not None
    # routing must cover exactly the real content positions, never the padding tail
    assert int(mask.sum()) == n_real
    assert bool(mask[0, :n_real].all()) and not bool(mask[0, n_real:].any())
