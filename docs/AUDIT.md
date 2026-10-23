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
