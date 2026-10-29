"""EOS-aligned document packing for the training sampler.

Why this module exists
----------------------
The flat sampler cuts fixed-stride windows out of a concatenated token stream, so a
document of length ``L`` straddles a window boundary with probability about ``L / T``.
For a median CoT document (1,278 tokens) at ``T = 2,048`` that is roughly 62%: most
documents that *fit* in a window are still cut. At ``T = 1,024`` a median CoT document
cannot fit at all. "Fraction of documents that fit" is therefore only an upper bound on
the fraction actually presented intact, and the token-weighted figure is lower again
because long documents carry more tokens.

Packing strategy (C')
---------------------
Each window is filled in two phases:

1. **Whole documents.** Every document of at most ``T`` tokens can be intact, so those
   are placed whole, best-fit with a bounded lookahead (candidate scan) rather than
   first-fit.
2. **Filler from documents longer than ``T``.** A document longer than the window can
   never be intact under any scheme, so cutting it costs nothing that was not already
   lost. The leftover space is filled with the next sequential chunk of such a
   document. This drives waste toward zero while leaving intactness untouched.

If both pools are exhausted mid-epoch, the window is completed with a pad token. That
is the *only* path that emits padding, and it sets the corresponding label positions to
``IGNORE_INDEX`` so they are excluded from the loss. Padding is a run artifact, not a
training signal: routing padding through the MoE would concentrate it on a few experts
and distort the balancing bias.

Cross-document attention masking is deliberately out of scope. Windows may contain
several documents and the model attends across boundaries. Adding masking is cheap once
``flex_attention`` (WI-09) is in place, but it is a separate behavioural decision.

Determinism
-----------
A plan is a pure function of ``(seed, epoch, seq_len, strategy)``: the document order is
a seeded permutation and packing is a deterministic greedy pass. Window ``k`` never
depends on how many windows were consumed before it, so training is exactly resumable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import torch

EOS_TOKEN_ID = 0
IGNORE_INDEX = -100

# Strategies
FLAT = "flat"
BESTFIT = "bestfit"
FIRSTFIT = "firstfit"


# ---------------------------------------------------------------------------
# Document index
# ---------------------------------------------------------------------------

@dataclass
class DocIndex:
    """Flat document index over one or more shards.

    ``shard`` / ``start`` / ``length`` are parallel arrays; ``length`` includes the
    terminating EOS token, because that token is part of what the model trains on.
    """

    shard: np.ndarray          # int32 - which shard the document lives in
    start: np.ndarray          # int64 - offset within that shard
    length: np.ndarray         # int32 - token count including EOS
    shard_offsets: np.ndarray  # int64 - cumulative token offset of each shard
    total_tokens: int = 0

    def __len__(self) -> int:
        return int(self.length.size)

    @property
    def global_start(self) -> np.ndarray:
        """Position of each document in the virtual concatenated stream."""
        return self.shard_offsets[self.shard] + self.start


def build_doc_index(shards: Sequence[np.ndarray], eos_id: int = EOS_TOKEN_ID) -> DocIndex:
    """Scan shards for EOS boundaries and build a document index.

    A document is ``[start, eos]`` inclusive. A trailing unterminated run is kept as a
    final document so no tokens are silently dropped.
    """
    shard_idx: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    lengths: list[np.ndarray] = []
    offsets = []
    running = 0

    for si, arr in enumerate(shards):
        n = len(arr)
        offsets.append(running)
        running += n
        if n == 0:
            continue
        eos = np.flatnonzero(np.asarray(arr) == eos_id).astype(np.int64)
        if eos.size == 0:
            s = np.array([0], dtype=np.int64)
            e = np.array([n - 1], dtype=np.int64)
        else:
            starts_in = np.concatenate(([0], eos[:-1] + 1))
            ends_in = eos
            if eos[-1] + 1 < n:
                starts_in = np.concatenate((starts_in, [eos[-1] + 1]))
                ends_in = np.concatenate((ends_in, [n - 1]))
            s, e = starts_in, ends_in
        keep = e >= s
        s, e = s[keep], e[keep]
        shard_idx.append(np.full(s.size, si, dtype=np.int32))
        starts.append(s)
        lengths.append((e - s + 1).astype(np.int32))

    if not shard_idx:
        empty = np.zeros(0, dtype=np.int64)
        return DocIndex(
            shard=np.zeros(0, dtype=np.int32), start=empty, length=np.zeros(0, dtype=np.int32),
            shard_offsets=np.asarray(offsets, dtype=np.int64), total_tokens=0,
        )

    return DocIndex(
        shard=np.concatenate(shard_idx),
        start=np.concatenate(starts).astype(np.int64),
        length=np.concatenate(lengths).astype(np.int32),
        shard_offsets=np.asarray(offsets, dtype=np.int64),
        total_tokens=int(running),
    )


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------

@dataclass
class Window:
    """One packed window: segments ``(doc, offset, length)`` plus optional padding."""

    docs: np.ndarray
    offs: np.ndarray
    lens: np.ndarray
    pad: int = 0          # pad token count at the end of this window
    filler_tokens: int = 0  # tokens contributed by documents longer than seq_len

    @property
    def tokens(self) -> int:
        return int(self.lens.sum())

    def __len__(self) -> int:
        return int(self.docs.size)


@dataclass
class PackPlan:
    windows: list[Window]
    seq_len: int
    strategy: str
    seed: int
    epoch: int
    dropped_tail_tokens: int = 0
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.windows)

    # -- accounting -------------------------------------------------------
    @property
    def pad_tokens(self) -> int:
        return sum(w.pad for w in self.windows)

    @property
    def filler_tokens(self) -> int:
        return sum(w.filler_tokens for w in self.windows)

    @property
    def capacity_tokens(self) -> int:
        return self.seq_len * len(self.windows)

    @property
    def content_tokens(self) -> int:
        return sum(w.tokens for w in self.windows)


def plan_epoch(
    doc_lengths: np.ndarray,
    seq_len: int,
    *,
    seed: int = 0,
    epoch: int = 0,
    strategy: str = BESTFIT,
    lookahead: int = 64,
    max_windows: int | None = None,
) -> PackPlan:
    """Pack whole documents, fill leftovers from over-long documents, pad only at the end.

    ``strategy='firstfit'`` places documents in encounter order instead of searching for
    the best fit; both keep whole documents whole. ``strategy='flat'`` is rejected here
    (the flat sampler is a separate code path) and exists only to be named.
    """
    if strategy == FLAT:
        raise ValueError("flat sampling is not a packed plan; use the flat loader")
    if strategy not in (BESTFIT, FIRSTFIT):
        raise ValueError(f"unknown strategy {strategy!r}")

    lengths = np.asarray(doc_lengths, dtype=np.int64)
    n = int(lengths.size)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch)]))

    # Partition the corpus once. Whole-able documents are placed whole; over-long
    # documents are consumed sequentially as filler, because no scheme can make them
    # intact, so their cut points are free.
    fits = np.flatnonzero(lengths <= seq_len)
    longs = np.flatnonzero(lengths > seq_len)
    order = rng.permutation(fits)

    # Long documents: streamed in a seeded order, as a queue of (doc, remaining).
    long_order = longs[rng.permutation(longs.size)] if longs.size else longs

    windows: list[Window] = []
    i = 0                      # cursor into `order`
    li = 0                     # cursor into `long_order`
    long_off: dict[int, int] = {}

    def take_filler(space: int) -> tuple[list[int], list[int], list[int]]:
        """Consume up to `space` tokens from over-long documents."""
        nonlocal li
        wd: list[int] = []
        wo: list[int] = []
        wl: list[int] = []
        while space > 0 and li < long_order.size:
            d = int(long_order[li])
            off = long_off.get(d, 0)
            rem = int(lengths[d]) - off
            if rem <= 0:
                long_off.pop(d, None)
                li += 1
                continue
            take = min(space, rem)
            wd.append(d)
            wo.append(off)
            wl.append(take)
            long_off[d] = off + take
            space -= take
            if long_off[d] >= int(lengths[d]):
                long_off.pop(d, None)
                li += 1
        return wd, wo, wl

    while i < order.size or li < long_order.size:
        if max_windows is not None and len(windows) >= max_windows:
            break

        space = seq_len
        wd: list[int] = []
        wo: list[int] = []
        wl: list[int] = []
        filler = 0

        # ---- phase 1: whole documents, best-fit within a bounded lookahead ----
        while space > 0 and i < order.size:
            d = int(order[i])
            ln = int(lengths[d])
            if ln <= space:
                wd.append(d); wo.append(0); wl.append(ln)
                space -= ln
                i += 1
                continue

            # Best fit: scan a bounded window of upcoming documents for the largest
            # one that still fits. first-fit keeps encounter order.
            best_j, best_len = -1, 0
            if strategy == BESTFIT:
                hi = min(i + 1 + lookahead, order.size)
                cand = order[i + 1:hi]
                cand_lens = lengths[cand]
                ok = np.flatnonzero(cand_lens <= space)
                if ok.size:
                    k = ok[np.argmax(cand_lens[ok])]
                    best_j, best_len = i + 1 + int(k), int(cand_lens[k])
            if best_j >= 0:
                order[i], order[best_j] = order[best_j], order[i]
                continue
            break  # nothing whole fits in the remainder

        # ---- phase 2: filler from documents longer than the window ----
        if space > 0:
            fd, fo, fl = take_filler(space)
            wd.extend(fd); wo.extend(fo); wl.extend(fl)
            filler = sum(fl)
            space -= filler

        # ---- phase 3: pad only when both pools are exhausted ----
        # No whole document fit and no over-long document could supply filler, yet
        # there is still a document cursor left: this window simply cannot be filled.
        # Pad the remainder (labels masked, routing excluded) and never drop the
        # document - leaving it unfilled would silently lose tokens.
        pad = space

        if wd or pad:
            windows.append(Window(
                docs=np.asarray(wd, dtype=np.int32),
                offs=np.asarray(wo, dtype=np.int64),
                lens=np.asarray(wl, dtype=np.int32),
                pad=int(pad),
                filler_tokens=int(filler),
            ))
        else:
            break

    # Documents longer than seq_len are emitted in full by the filler phase; whole-able
    # documents are all emitted too. Track the accounting so callers can assert that no
    # token was silently dropped.
    emitted = sum(int(w.tokens) for w in windows)
    total_docs_tokens = int(lengths.sum())

    return PackPlan(
        windows=windows,
        seq_len=seq_len,
        strategy=strategy,
        seed=int(seed),
        epoch=int(epoch),
        dropped_tail_tokens=max(0, total_docs_tokens - emitted),
        meta={
            "documents": n,
            "whole_able": int(fits.size),
            "over_long": int(longs.size),
            "lookahead": lookahead,
        },
    )


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def flat_window_ids(global_start: np.ndarray, lengths: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Window index of a document's first and last token under flat stride sampling."""
    first = global_start // seq_len
    last = (global_start + lengths - 1) // seq_len
    return first, last


def intact_fractions(
    lengths: np.ndarray,
    *,
    plan: PackPlan | None = None,
    global_start: np.ndarray | None = None,
    seq_len: int = 2048,
    is_cot: np.ndarray | None = None,
) -> dict:
    """Fraction of documents presented intact, count- and token-weighted.

    Reports overall figures and, when ``is_cot`` is given, the same restricted to CoT
    documents. ``*_among_fittable`` excludes documents longer than ``seq_len``, which no
    packing scheme can present intact - that is the number packing actually controls.
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    n = int(lengths.size)
    if n == 0:
        return {"count": 0.0, "tokens": 0.0, "documents": 0}

    if plan is not None:
        intact = np.zeros(n, dtype=bool)
        for w in plan.windows:
            whole = w.offs == 0
            if not whole.any():
                continue
            d = w.docs[whole].astype(np.int64)
            ln = w.lens[whole].astype(np.int64)
            ok = ln == lengths[d]
            intact[d[ok]] = True
    else:
        assert global_start is not None
        first, last = flat_window_ids(global_start, lengths, seq_len)
        intact = first == last

    fittable = lengths <= seq_len
    tot_tokens = int(lengths.sum())
    res = {
        "count": float(intact.mean()),
        "tokens": float(lengths[intact].sum() / max(1, tot_tokens)),
        "documents": n,
        "intact_documents": int(intact.sum()),
        "count_among_fittable": float(intact[fittable].mean()) if fittable.any() else 0.0,
        "tokens_among_fittable": float(
            lengths[intact & fittable].sum() / max(1, int(lengths[fittable].sum()))
        ) if fittable.any() else 0.0,
        "fittable_documents": int(fittable.sum()),
    }
    if is_cot is not None and np.asarray(is_cot).any():
        cot = np.asarray(is_cot, dtype=bool)
        res.update({
            "cot_count": float(intact[cot].mean()),
            "cot_tokens": float(lengths[intact & cot].sum() / max(1, int(lengths[cot].sum()))),
            "cot_documents": int(cot.sum()),
            "cot_count_among_fittable": float(
                intact[cot & fittable].mean()
            ) if (cot & fittable).any() else 0.0,
        })
    return res


def waste_fractions(plan: PackPlan) -> dict:
    """Padding / filler / capacity accounting for a packed plan."""
    if not plan.windows:
        return {"padding_tokens": 0, "padding_frac": 0.0, "filler_frac": 0.0}
    cap = plan.capacity_tokens
    pad = plan.pad_tokens
    filler = plan.filler_tokens
    content = plan.content_tokens
    return {
        "windows": len(plan.windows),
        "capacity_tokens": int(cap),
        "content_tokens": int(content),
        "padding_tokens": int(pad),
        "padding_frac": float(pad / max(1, cap)),
        "filler_tokens": int(filler),
        "filler_frac": float(filler / max(1, cap)),
        "filler_share_of_content": float(filler / max(1, content)),
        "waste_frac": float(pad / max(1, cap)),
    }


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class PackedBatchLoader:
    """Serve ``(x, y)`` batches from a packed plan, reading per-window from shards."""

    def __init__(
        self,
        shards: Sequence[np.ndarray],
        index: DocIndex,
        plan: PackPlan,
        *,
        micro_batch_size: int,
        device: str = "cpu",
        eos_id: int = EOS_TOKEN_ID,
        ignore_index: int = IGNORE_INDEX,
    ):
        self.shards = list(shards)
        self.index = index
        self.plan = plan
        self.micro_batch_size = micro_batch_size
        self.seq_len = plan.seq_len
        self.device = device
        self.eos_id = eos_id
        self.ignore_index = ignore_index
        self._cursor = 0
        self._last_n_real: list[int] = []

        is_cuda = (
            device == "cuda"
            or (isinstance(device, str) and device.startswith("cuda"))
            or (isinstance(device, torch.device) and device.type == "cuda")
        )
        self.is_cuda = is_cuda
        can_pin = is_cuda and torch.cuda.is_available()
        self.x_buf = torch.empty((micro_batch_size, self.seq_len), dtype=torch.long, pin_memory=can_pin)
        self.y_buf = torch.empty((micro_batch_size, self.seq_len), dtype=torch.long, pin_memory=can_pin)

    def reset(self) -> None:
        self._cursor = 0
        self._last_n_real = []

    def _read(self, d: int, off: int, length: int) -> np.ndarray:
        si = int(self.index.shard[d])
        s = int(self.index.start[d]) + off
        return np.asarray(self.shards[si][s:s + length], dtype=np.int64)

    def _build_window(self, w: Window) -> tuple[np.ndarray, int]:
        parts = [self._read(int(d), int(o), int(l)) for d, o, l in zip(w.docs, w.offs, w.lens)]
        toks = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
        n_real = int(toks.size)
        if n_real < self.seq_len:
            toks = np.concatenate([toks, np.full(self.seq_len - n_real, self.eos_id, dtype=np.int64)])
        return toks[: self.seq_len], n_real

    def get_batch(self, window_ids: Iterable[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x, y)`` with next-token targets.

        Label rule: ``y[i] = x[i + 1]`` for every position whose next token exists and
        is real. The final real position has no successor inside the window (the next
        document begins in the following window), and every padding position has no
        successor at all, so both get ``ignore_index``. Document boundaries *inside* a
        window need no masking: each document ends with its own EOS token, which is a
        real training target.
        """
        if window_ids is None:
            ids = [self._cursor + k for k in range(self.micro_batch_size)]
            self._cursor += self.micro_batch_size
        elif isinstance(window_ids, int):
            # The trainer's loop calls get_batch(seed); for a packed plan the seed is
            # unused - the plan already fixes the order - so advance sequentially and
            # keep the call site identical to the flat loader's.
            ids = [self._cursor + k for k in range(self.micro_batch_size)]
            self._cursor += self.micro_batch_size
        else:
            ids = list(window_ids)
        n_req = len(ids)
        if self.x_buf.shape[0] < n_req:
            can_pin = self.is_cuda and torch.cuda.is_available()
            self.x_buf = torch.empty((n_req, self.seq_len), dtype=torch.long, pin_memory=can_pin)
            self.y_buf = torch.empty((n_req, self.seq_len), dtype=torch.long, pin_memory=can_pin)

        # Cleared per call: this was appended to forever, leaking memory over training.
        self._last_n_real.clear()
        n_windows = len(self.plan.windows)
        for i, wid in enumerate(ids):
            if n_windows == 0:
                break
            wid = wid % n_windows   # next epoch: reuse the plan cyclically
            if wid >= n_windows:
                break
            toks, n_real = self._build_window(self.plan.windows[wid])
            y = np.full(self.seq_len, self.ignore_index, dtype=np.int64)
            if n_real > 1:
                y[: n_real - 1] = toks[1:n_real]
            # Cross-document transitions inside a packed window: at a document's
            # terminating EOS the next token belongs to a DIFFERENT document, so
            # predicting it from that EOS teaches an association that does not exist.
            # The EOS itself remains a valid target (it is predicted from real text).
            if n_real > 1:
                eos_at = np.flatnonzero(toks[: n_real - 1] == self.eos_id)
                y[eos_at] = self.ignore_index
            self.x_buf[i].copy_(torch.from_numpy(toks))
            self.y_buf[i].copy_(torch.from_numpy(y))
            self._last_n_real.append(int(n_real))
        if not self._last_n_real:
            raise IndexError("no windows left in plan")
        count = len(self._last_n_real)
        if self.is_cuda:
            return (
                self.x_buf[:count].to(self.device, non_blocking=True),
                self.y_buf[:count].to(self.device, non_blocking=True),
            )
        return self.x_buf[:count].clone(), self.y_buf[:count].clone()

    def routing_mask(self, x: torch.Tensor) -> torch.Tensor | None:
        """Boolean mask of positions that may enter MoE routing and expert counts.

        Padding must be excluded: a run of identical pad tokens would all route to the
        same few experts, inflating overflow drops and distorting the balancing bias.
        Token values cannot identify padding - EOS is a real document separator with the
        same id - so the mask comes from the plan's content lengths recorded by the last
        :meth:`get_batch` call.
        """
        rows = x.shape[0]
        if not self._last_n_real or all(n == self.seq_len for n in self._last_n_real[-rows:]):
            if torch.compiler.is_compiling():
                return torch.ones_like(x, dtype=torch.bool)
            return None
        n_real = self._last_n_real[-rows:]
        lengths = torch.as_tensor(n_real, device=x.device, dtype=torch.int64).unsqueeze(1)
        col_idx = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        return col_idx < lengths
