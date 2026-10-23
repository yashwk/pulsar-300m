# Pulsar-300M: Audit Remediation Report

Owner-facing record of the code-audit fixes. One section per work item: status, files
changed, tests added, measured numbers, deviations, open questions.

**Environment (measured, not assumed)**

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop, 8 GB, sm_89 |
| Driver / CUDA UMD | 615.78.08 / CUDA 13.4 |
| PyTorch | 2.14.1+cu130 |
| nvcc | 13.4.92 |
| Python | 3.14.8 |
| `pulsar_cuda` extension | builds and imports, 22 ops |
| `flash_attn` | **not installed** (optional path is import-guarded only) |
| `flex_attention` | available; **no GQA support in this torch build** (verified) |

**Baseline training geometry:** micro-batch 3 x 1024 tokens, 43 accumulation steps,
132,096 tokens/optimizer step. Measured 15.4 s/step, 8.6k tok/s, 7.0 GB peak
allocated at the start of this work.

---

## WI-13: data-pipeline quality

**Status: code shipped, no shard rebuild required (finding contradicts the original premise).**

### What the audit claimed

1. Think traces longer than a character budget are truncated by replacing the middle
   with a literal `[...]`, teaching the model to emit elided reasoning.
2. Agentic tool-call traces are filtered to fit 1,024 tokens, keeping only trivial
   single-call examples.

### What the code and shards actually show

| Claim | Verified result |
|---|---|
| Mid-trace elision exists in code | **TRUE** — `clean_think_trace` emitted `f"{head}\n\n[...]\n\n{tail}"` when a trace exceeded 1,800 chars. |
| ...and it damaged the existing shards | **FALSE** — 0 elision markers in 476.1M tokens (shard 1) and 476.3M (shard 2). Think blocks in the shards have p50 483 tokens, p90 643, max 1,088 — all far below the 1,800-char trigger. |
| A 1,024-token agentic filter exists | **FALSE** — no such filter in `data/prepare_shards.py`. `format_toucan_messages` applies no length limit at all. |
| Manifest's 20% CoT share is inflated by elision | **FALSE** — it reflects real, un-elided reasoning content. |

**Reconciling a confusing count.** A first scan found 98 `[...]` occurrences while a
second found 0. Both are correct and measure different token sequences:

* `[...]` alone tokenises to `[70, 29326]` — 98 occurrences, all genuine content
  (`i[...] = i%3`, OCR gaps, `dislodging[...]an enemy`), all outside think blocks.
* The elision marker `\n\n[...]\n\n` tokenises to `[210, 210, 70, 919, 543, 210]` —
  **0 occurrences anywhere in both shards**.

An earlier HF-based retention estimate (1B/2B medians of ~7k tokens) measured the whole
`output` field, which includes reasoning *plus* long code/solution text; the think block
itself is small. That estimate was the basis of a rebuild recommendation that this
on-disk check overturns. **The rebuild was cancelled** — it would have produced a
byte-different shard with no measurable content change, while losing document diversity.

### Full-document length distribution (measured on shard 1, 329,437 documents)

| | p50 | mean | p90 | p99 | max | fits @1024 | @2048 | @8192 |
|---|---|---|---|---|---|---|---|---|
| All documents | 940 | 1,445 | 2,739 | 10,674 | 363,955 | 54.1% | 84.8% | 98.1% |
| Documents with a think block (56,345) | 1,278 | 1,344 | 1,874 | — | 7,293 | **22.3%** | **94.2%** | 100% |

Shard 2 is consistent (p50 919; CoT p50 1,211; 28.2% / 95.4% / 100%).

**Consequences for WI-11 (flagged, not silently applied).** The audit's claim that
Stage 1 cannot show full reasoning arcs is confirmed (only 22.3% of CoT documents fit a
1,024-token window). But Stage 2's existing T=2,048 already fits **94.2%** of CoT
documents whole, so full arcs belong in Stage 2 as well — not only in the proposed
8-16k stage. This changes the WI-11 recommendation and is left for the owner.

**Also found: EOS separators are present and correct.** `<|endoftext|>` (token 0)
terminates every document: 329,437 EOS in shard 1, 343,455 in shard 2. The document-
aligned loader (WI-10) is therefore viable. This was verified rather than assumed.

### Changes made

* `data/prepare_shards.py`
  * `clean_think_trace` now returns traces **verbatim**, with only a safety cap
    (`MAX_TRACE_CHARS = 131072`, ~32k tokens) cut on a sentence boundary and with **no
    placeholder marker inserted**. Observed maximum document is 23,693 tokens, so the
    cap never fires on current sources.
  * `legacy_clean_think_trace` / `legacy_format_cot_block` retained *only* to let the
    analysis script measure the old build's behaviour.
  * Every random choice is seeded: `derive_stream_seed(stream, shard, base_seed)` uses
    sha256 (stable across processes and machines, unlike `hash()`), driving the Toucan
    70/15/15 harness selection for both `3A` and `5A`. Base seed is a CLI flag
    (`--seed`, default 20261023) and per-stream seeds are written to the manifest.
  * Manifests now record `trace_policy`, `max_trace_chars`, `corpus_version`,
    per-stream token/doc counts, per-stream token shares, the realised reasoning share,
    and the seeds used.
  * `SUBSTREAM_CONFIG_LONG` (42% reasoning: 1B 15%, 2B 15%, 3A 12%) selected only for
    Stage 3 / the intermediate long-context stage via `substream_config_for_stage()`.
    Stage 1/2 keep the documented 20% CoT mix. Marked **proposed** pending WI-11.
* `scripts/analyze_agentic_lengths.py` — `--mode shards` (offline shard audit: doc
  lengths, retention curve, elision-marker count) and `--mode dry-run` (read-only,
  side-by-side un-elided vs legacy-elided rendering; writes no shards).
* `.gitignore` — `data/` excluded the data-pipeline **source code**. Narrowed to
  `data/*` with `!data/*.py` and `!data/pulsar_tokenizer/`; shard blobs stay ignored.
  Before this fix the repo tracked 17 files and **none** of the data pipeline.

### Tests

Unit coverage for the formatter lands with WI-20's suite. The shard-level evidence above
is reproducible with:

```bash
python scripts/analyze_agentic_lengths.py --mode shards --shards data/train_shard1.bin
python scripts/analyze_agentic_lengths.py --mode dry-run --stage 1 --shard-idx 1
```

### Deviations from the work item

* The work item asks to "replace middle-elision with filtering". The patches now store
  traces **verbatim** (per owner direction) rather than filtering documents out. On the
  current corpus the two are equivalent, because the elision path never fired.
* No shards were regenerated, per the owner's decision after reviewing the on-disk
  evidence.

### Open questions

1. **WI-11 stage split.** Given 94.2% of CoT documents fit T=2,048, should Stage 2 also
   carry full reasoning arcs, and should the 8-16k "raise reasoning share" plan shrink
   or move accordingly? `SUBSTREAM_CONFIG_LONG` is a proposal, not a decision.
2. Should the 32k cap ever cut a document, or should oversized documents be filtered
   instead? Not exercised by current sources.

---

## WI-07 / WI-10 groundwork: EOS-aligned packing (`src/packing.py`)

**Status: implemented and measured; not yet wired into `train.py` (that is WI-07's CLI work).**

### Why the flat sampler under-delivers

The flat sampler draws fixed-stride windows from a concatenated token stream. A document
of length `L` straddles a stride-`T` boundary with probability about `L/T`, so at
`T=2,048` a median CoT document (1,278 tokens) is cut roughly 62% of the time *even
though it fits*. At `T=1,024` a median CoT document cannot fit at all.

This corrects an earlier framing in this report: "94.2% of CoT documents fit T=2,048"
measured whether a document would fit *if* a window happened to start at its beginning.
It is an upper bound on intactness, not the rate the sampler actually achieves.

### Strategy C'

1. **Whole documents first.** Every document of at most `T` tokens is placed whole,
   best-fit with bounded lookahead (64 candidates) rather than first-fit.
2. **Filler only from documents longer than `T`.** Those can never be intact under any
   scheme, so cutting them costs nothing already lost.
3. **Padding only at epoch end,** and only when both pools are exhausted. Padding
   positions get `ignore_index` labels and are excluded from MoE routing and expert
   counts: a pad run would otherwise route to the same few experts, inflating overflow
   drops and distorting the auxiliary-loss-free balancing bias.

### Measured on `data/train_shard1.bin` (475,000,000 tokens, 328,440 documents)

| T | intact among fittable (flat → packed) | token-weighted | intact over all docs | padding waste | filler share |
|---|---|---|---|---|---|
| 2,048 | 59.3% → **100.0%** | 43.1% → **100.0%** | 50.2% → 84.8% | 0.000% (832 tok) | 51.1% |
| 8,192 | 85.2% → **100.0%** | 70.1% → **100.0%** | 83.5% → 98.1% | 0.001% | 17.5% |
| 32,768 | 95.7% → **100.0%** | 87.1% → **100.0%** | 95.6% → 100.0% | 0.001% | 1.9% |

**Filler supply is covered at every T:** padding stays at the terminal-window floor.

**The 51.1% filler share at T=2,048 is a corpus property, not a packer defect.**
Documents longer than 2,048 tokens are 15.2% of documents but **51.1% of tokens**
(median of that tail: 3,395 tokens; the largest document is 363,955 tokens). Half the
token mass therefore lives in documents that can never be intact, and it necessarily
arrives as mid-document filler. This is recorded in `docs/CURRICULUM.md` so Stage 2 is
not read as 100% whole-document training.

**Loader throughput:** 135M tok/s packed vs 31.7M tok/s flat (4.3x faster - a few large
sequential reads per window instead of strided seeking), against a training demand of
~8.6k tok/s. Packing adds no I/O bottleneck. (Both figures are page-cache reads; this
compares I/O shape, not cold-disk throughput.)

### Resumability and coverage guarantees (tested)

* **Globally shuffled per epoch.** A shard written source-by-source has contiguous
  document indices per source. Measured on a 10-source synthetic shard: 0.00%
  adjacent-index pairs, all 10 sources present in the first 500 emitted documents, and
  no source exceeding 4x its uniform share in a 1,000-document prefix. A sequential plan
  would show ~100% adjacent pairs and one source at a time.
* **Pure function of `(seed, epoch)`.** Rebuilding a plan reproduces every window
  exactly; window `k` does not depend on how many windows were consumed, so resume is
  exact. Different seeds produce different plans.
* **Each token covered exactly once per epoch.** Verified token-by-token at T=512/1024/
  2048: zero missing positions, zero overlapping positions.

### Stage assignment

Packing is used for **Stage 2, the 8-16k stage and Stage 3**. **Stage 1 keeps the flat
permuted sampler**, because a median CoT document cannot fit a 1,024-token window, so
packing buys little there while complicating the data path.

### Open question

Whether intact arcs improve reasoning is **unmeasured**. The recommended signal is the
small GSM8K and HumanEval evals at the end of Stage 2, not a flat-vs-packed loss run.

---

## Baseline evaluation (Before numbers)

The 250-step baseline (150-step warmup, seed 42, MBS 3x1024) runs on the current shards.
`evaluate_loss` currently returns **the weighted sum of all three pulse losses plus
z-loss** (0.15/0.25/0.60), not the final-pulse loss. That makes it unsuitable as the
primary "before" metric for the final model, which is WI-06's finding.

Action, scheduled with WI-06: after `--eval-full-pulses` lands, load the baseline
checkpoint and recompute `val_ce_final`, `val_ce_p1`, `val_ce_p2`, `val_ce_p3` at 100%
of tokens, and record them here as the comparable "before". **Training loss is directly
comparable across the change** and is recorded as-is from the baseline run.

---

## Accepted-but-ignored options (audit of the branch)

**Context.** `muon_split_qkv` and `muon_split_gate_up` existed as Muon constructor
arguments with no implementation: setting them changed nothing at all. That is worse
than not having the flag, because it looks like an ablation was run. They have been
removed, and `tests/test_no_dead_options.py` now guards against the whole class:

* every non-conventional function parameter must be referenced in its body,
* every argparse option must be read as `args.<dest>`,
* every `PulsarConfig` field must be read outside the dataclass body,
* the specific removed flag names must not reappear without an implementation.

The check was validated by reintroducing `split_qkv` and confirming it fails.

**Branch-wide result:** after stripping the split flags, a sweep of all
`PulsarConfig` fields, every function/constructor parameter, and every CLI option in
`src/`, `scripts/`, `data/` and `monitor/` found no other accepted-but-ignored options.

Two adjacent issues *were* found, both real but different in kind:

1. **`PulsarConfig.from_dict` silently drops unknown keys.** A typo'd or stale config
   key is discarded without a word, so a checkpoint or config file can carry an
   intention that never takes effect:
   `PulsarConfig.from_dict({"per_pulse_biass": True})` returns `per_pulse_bias=False`
   with no warning. This matters more now that config has grown runtime knobs
   (`per_pulse_bias`, `moe_capacity_factor`, `moe_capacity_align`). Not fixed here;
   recorded as a follow-up. Suggested fix: log or raise on unknown keys.
2. **`PulsarModel.forward(early_exit_threshold=...)` is never passed by any caller.**
   The early-exit branch (`model.py:1029`) is unreachable from `train.py`, the scripts
   or `generate()`. It is an unused feature rather than a lie, because the parameter is
   part of the forward signature rather than a user-facing flag — but the branch is
   effectively dead and should either be wired to a flag or removed.

**Not done (optional): WI-18 Muon update splits.** The optional ablations
`muon_split_qkv` / `muon_split_gate_up` are deliberately skipped. The fused `W_qkv`
and `gate_up` matrices are orthogonalized as one matrix, so Q/K/V (and gate/up) share a
single update scale; splitting them into logical sub-matrices before Newton-Schulz
would decouple that. It changes training dynamics, no A/B runs are planned, and the
part of WI-18 that matters — the layout-aware scale convention — is landed and tested.
If re-added, implement and test the split in the same change; the regression test will
fail otherwise.
