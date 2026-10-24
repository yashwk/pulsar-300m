# Pulsar-300M: Audit Notes

Technical reference for the audit fixes: constraints the code now enforces, properties of
the corpus and sampler that are easy to get wrong, and the configuration knobs added.
Run outcomes and measurements live in commit messages.

---

## Environment constraints

| Component | Value |
|---|---|
| GPU | RTX 4060 Laptop, 8 GB, sm_89 |
| PyTorch / CUDA | 2.14.1+cu130 / CUDA 13.4 |
| `flash_attn` | not installed; the optional path is import-guarded only |
| `flex_attention` | available, but **no GQA support** in this torch build — K/V must be expanded with `repeat_interleave` |

`qk_rmsnorm` and several other fused kernels assume `head_dim == 64`; see below.

---

## Kernel contracts

### `qk_rmsnorm` requires `head_dim == 64`

The kernel gives each of 32 lanes one (even, odd) element pair, i.e. exactly 64 elements
per vector, and does **not** derive its footprint from `head_dim`. The original wrapper
computed `total_vectors` from the actual `head_dim` and validated nothing, so any other
value made the upper lanes read and write past the end of each vector. Symptom:
intermittent NaNs (~4% of runs) that vanish when anything perturbs allocator timing,
because the corruption depends on neighbouring memory contents.

The production model uses `head_dim == 64` and was never affected. Two guards now exist:

* the C++ wrapper raises `qk_rmsnorm kernel requires head_dim == 64, got N`;
* `PulsarAttention` only takes the fused path when `self.head_dim == 64`, falling back to
  the equivalent PyTorch implementation otherwise.

When adding fused kernels, validate shape assumptions in the wrapper rather than
trusting the caller.

---

## Corpus facts (measured on the shipped shards)

These contradicted the original audit and are worth not re-deriving:

| Question | Answer |
|---|---|
| Is mid-trace `[...]` elision in the shards? | **No.** 0 elision markers in 952M tokens. Think blocks are p50 483 tokens, far below the 1,800-char trigger, so the path never fired. |
| Is there a 1,024-token agentic filter? | **No.** `format_toucan_messages` applies no length limit. |
| Is the manifest's 20% CoT share inflated? | **No.** It reflects real, un-elided reasoning content. |
| Are documents EOS-separated? | **Yes.** `<|endoftext|>` (token 0) terminates every document; 329k EOS in shard 1, 343k in shard 2. |

The 98-vs-0 marker confusion is a tokenisation trap: `[...]` alone is `[70, 29326]`
(98 occurrences, all genuine content such as `i[...] = i%3`), while the elision form
`\n\n[...]\n\n` is `[210, 210, 70, 919, 543, 210]` — 0 occurrences. Match the full
marker, not the bracket pair.

**Document length, shard 1 (329,437 documents):**

| | p50 | p90 | max | fits @1024 | @2048 | @8192 |
|---|---|---|---|---|---|---|
| All documents | 940 | 2,739 | 363,955 | 54.1% | 84.8% | 98.1% |
| Documents with a think block (56,345) | 1,278 | 1,874 | 7,293 | **22.3%** | **94.2%** | 100% |

Consequence: a median CoT document cannot fit a 1,024-token window, but almost all fit
2,048. Stage 1 is therefore unsuitable for whole reasoning arcs; Stage 2 is where they
first become possible.

## Sampling

**"Fits" is not "is shown intact".** A flat sampler cuts fixed-stride windows, so a
document of length `L` straddles a boundary with probability ≈ `L/T`. At `T=2,048` a
median CoT document (1,278 tokens) is cut ~62% of the time even though it fits
comfortably. Any "fraction of documents that fit" figure is an upper bound on intactness.

`src/packing.py` implements best-fit packing (strategy C′) for Stage 2, the 8–16k stage
and Stage 3; Stage 1 keeps the flat permuted sampler.

1. **Whole documents first**, best-fit with bounded lookahead over documents of at most `T`.
2. **Filler only from documents longer than `T`** — those can never be intact under any
   sampler, so cutting them costs nothing already lost.
3. **Padding only when both pools are exhausted**, with `ignore_index` labels and routing
   exclusion. Padding must never enter routing: a run of identical pad tokens concentrates
   on the same few experts, inflating overflow drops and distorting the balancing bias.

Padding is identified from the plan's content lengths, **not** from token values — EOS is a
real document separator with the same id. A plan is a pure function of
`(seed, epoch, seq_len, strategy)`, so resume is exact.

Measured intactness for documents that can fit: 100% at T=2,048/8,192/32,768, with
padding ≤0.001%. At T=2,048 roughly half of content is necessarily mid-document filler,
because documents longer than 2,048 tokens are 15.2% of documents but 51.1% of tokens.
`CURRICULUM.md` states this so Stage 2 is not read as 100% whole-document training.

---

## Configuration added

| Knob | Default | Notes |
|---|---|---|
| `moe_capacity_factor` / `--moe-capacity-factor` | 1.12 | Capacity headroom; 8.3% at the old 1.08 was too tight and drops were invisible. Runtime-only: raising it mid-run is safe, it changes no checkpoint format. |
| `moe_capacity_align` | 32 | Capacity rounded up to this multiple (864 slots at N=3072, vs 960 at 1.25). |
| `--drop-warn-pct` | 2.0 | Warns when the overall MoE drop rate exceeds this. Per-pulse rates are logged every step; drops concentrate on late pulses. |
| `per_pulse_bias` / `--per-pulse-bias` | off | `[max_pulses, E]` balancing bias, each pulse updated from its own counts. Changes training dynamics. Old `[E]` checkpoints migrate by broadcast. |
| `--no-stochastic-rounding` | SR on | Disables SR for bf16 weights. Without SR, sub-ulp updates and decay round away. |
| `--muon-momentum-fp32` | off | Muon momentum in FP32 instead of bf16. |
| `--seed` (data pipeline) | 20261023 | Per-(stream, shard) seeds derived by sha256 and recorded in the manifest, so shard builds are reproducible. |

**Muon scale convention.** Updates are scaled by `max(1, out/in)**0.5`. `nn.Linear` stores
`[out, in]`, but MoE expert tensors are stored `[E, in, out]` for the batched GEMM, so
reading `shape[-2]/shape[-1]` blindly inverts the ratio for experts: `gate_up`
`[E, 1024, 2048]` received 1.0 instead of 1.414 and trained ~0.7x as fast as the
equivalent dense matrix. The rule is now layout-aware.

---

## Accepted-but-ignored options

A flag accepted and then never read is worse than no flag: it looks like an ablation was
run. `muon_split_qkv` / `muon_split_gate_up` were exactly that and have been removed.
`tests/test_no_dead_options.py` enforces:

* every non-conventional function parameter is referenced in its body;
* every argparse option is read as `args.<dest>`;
* every `PulsarConfig` field is read outside the dataclass body;
* the removed flag names cannot reappear without an implementation.

---

## Fixed issues

1. **`PulsarConfig.from_dict` silently dropped unknown keys.** A typo'd or stale key
   carried an intention that never took effect — the same failure mode as a flag accepted
   and ignored. Unknown keys now raise a `RuntimeWarning` (or `ValueError` with
   `strict=True`); the default stays non-fatal so older checkpoints still load.
2. **`forward(early_exit_threshold=...)` was dead.** No caller passed it, so the branch
   was unreachable from `train.py`, the scripts and `generate()`. Parameter and branch
   removed rather than wired to a new flag.

## Not done (optional)

**Muon update splits.** The fused `W_qkv` and `gate_up` matrices are orthogonalized as one
matrix, so Q/K/V (and gate/up) share a single update scale; splitting them into logical
sub-matrices before Newton-Schulz would decouple that. Deliberately skipped — it changes
training dynamics and no A/B runs are planned. If re-added, implement and test the split in
the same change.

---

## Round 2 (branch `fix/round2`)

Every claim below was verified against the code and by experiment before editing. Status
per item, with the measurement that decided it.

### R-06 — FIXED: activation-checkpoint context order was inverted

`torch.utils.checkpoint` documents `context_fn` as `(forward_ctx, recompute_ctx)`. The call
site passed `(_set_recomputing(True), _set_recomputing(False))`, so the original forward
suppressed the MoE counters and the recomputation applied them.

Measured before: with `grad_checkpoint=True`, counters after forward = **0**, after
backward = 512. After the fix: forward = 512, backward = 512, so counts still land exactly
once. Any forward not followed by a backward previously counted nothing.

### R-04 — FIXED: RoPE `inv_freq` up-cast instead of rebuilt

`to_training_precision` called `self.to(bf16)` (rounding the buffer) and then
`inv_freq.data.float()`, which recovers the rounded values. Measured error **3.65e-3
relative (0.37%)** — up to ~0.5 rad of angle error by position 1024, tens of radians at
32k. Now rebuilt from `(theta, dim)`; error is exactly **0.0**.

**Checkpoint consequence:** checkpoints trained before this encode rounded frequencies.
Do not resume one with the corrected code. The 107-step checkpoint is discardable.

### R-02 — FIXED: drop telemetry now comes from the slot assignment

Drops were `sum(max(bincount - capacity, 0))`, which equals the real count only while slot
logic is correct. The earlier fast-path bug discarded ~85% of assignments while this metric
read 0%.

Drops are now `(token_to_slot < 0).sum()`; the old value is kept as
`demand_over_capacity` and a warning fires when the two disagree. Measured with a
deliberately broken slot function: slot-derived reports **670**, the bincount heuristic
still reports **0**.

Side effect: slot assignment is now computed once per forward instead of once per dispatch
branch.

### R-03 — FIXED: host syncs removed from the hot path

Three sites: `int(counts.max())` (24x/forward), `valid_mask.sum().item()` +
`grad_output.item()` in CE, and `bool(valid.all())` in the MoE counter path. The CE scale
is now passed by device pointer (new `fused_softcap_ce_backward_dev`); the slot function
uses one sort path; the counter path is branch-free.

A `set_sync_debug_mode("error")` test caught a **fourth** site during development (the
`sum` reduction path) that the report did not list.

**The 5% throughput target was wrong and is withdrawn.** Measured cost of these syncs:
`compute_token_slots` 0.100 ms/call x 24 = 2.4 ms/forward = **0.10 s of a 17.5 s step
(0.6%)**. That cannot explain the gap the report attributed to it.

### R-07 — FIXED (low priority): unused global counter grew forever

In per-pulse mode `update_biases` reads `expert_pulse_counts`, while `expert_counts_accum`
accumulated and was never read or zeroed. Measured: 200 steps grew it to 25,600, i.e.
**~131,000 steps of headroom** before FP32 stops resolving +1 — a real defect, not a live
hazard. Now skipped in per-pulse mode; global mode unchanged (separate test pins it).

Added `_migrate_counters()` for round-1 checkpoints missing the persistent counter buffers.
The migration must consult `_non_persistent_buffers_set`: injecting a non-persistent buffer
makes it an unexpected key on reload (found by the resume test failing).

### R-01 step 1 — DONE: router diagnostics shipped, measurement only

`--router-diag` (default on), sampled on the first micro-batch of each log window. Adds
`router_diag_shared_mean_frac`, `_bias_override_rate`, `_top1_entropy`, `_max_top1_share`,
`_bias_abs_p50`, `_bias_abs_p99`.

Validated rather than merely plausible: synthetic input with a dominant shared mean scores
**~1.0**, pure noise **~0.004**. A second bug was found while wiring it: per-micro-batch
averaging divided every diagnostic by `grad_accum`, but diagnostics exist only on
micro-step 0 — `max_top1_share` was reported **43x too small (0.0026)**. A test now pins
the `[1/E, 1]` bound.

**Not done, awaiting owner:** R-01 step 3 (router-input centering) depends on the measured
`shared_mean_frac` from a smoke run; R-05, R-08 to R-13 are explicitly deferred.

### Corrected claims

- The report called logit centering "mathematically invariant." It is not: routing uses
  `sigmoid(logits) + bias`, and sigmoid is not shift-invariant. Top-k *ranking* is
  preserved because sigmoid is monotone, so it is safe, but gating weights and bias
  leverage do change. It also gives z-loss an irreducible floor of `ln(E) = 2.079`, so
  z-loss now penalises logit *spread* rather than magnitude. Verified numerically.
- The 8,593 tok/s baseline and 8,163 tok/s floor appear nowhere in this repository and are
  unsourced. Replaced by `docs/THROUGHPUT_BASELINE.md`, measured on current code:
  MBS 3 / cap 1.12 = **7,916 tok/s**; MBS 2 / cap 1.375 = **7,400**; MBS 2 / cap 1.12 =
  **7,000**. Note MBS 2 at capacity 1.12 is *slower* than at 1.375 despite less memory.

### Smoke run results (300 steps, current code)

Config: MBS 2 x 1024, capacity 1.375 (704 slots), warmup 150, bias_lr 0.01, seed 42.
Completed 300/300, exit 0, 91.5 min.

| step | val CE |
|---|---|
| 50 | 8.468 |
| 100 | 5.683 |
| 150 | 4.666 |
| 200 | 4.105 |
| 250 | 3.786 |
| 300 | 3.428 |

Final train loss 3.521, no NaN, peak VRAM 6.22 GB, mean 7,348 tok/s.

Muon gradient norm reproduced the earlier run's trajectory **exactly** at every matched
checkpoint (0.40 / 0.42 / 0.64 / 0.85 / 1.12 at steps 1/10/19/25/28), peaked at **4.56 on
step 47**, and fell back to **0.25** (clip factor 1.00) — below its step-1 value. The spike
is reproducible, self-correcting, and unaffected by anything in round 2, which is expected:
R-02/R-03/R-06/R-07/R-01 change no gradients.

### R-01 step 1 result: `shared_mean_frac` is a transient, not a steady state

Threshold agreed with the owner: "above about 0.3 in some (block, pulse)".

| step | shared_mean_frac |
|---|---|
| 1 | 0.007 |
| 30 | 0.030 |
| 40 | 0.104 |
| **50** | **0.446** |
| **60** | **0.643** |
| **70** | **0.453** |
| 80 | 0.267 |
| 100 | 0.109 |
| 200 | 0.072 |
| 300 | ~0.09 |

It crosses 0.3 for exactly one window (steps 50-70) and then decays ~6x; steps 100+ average
**0.083**. Read literally the step-3 trigger fires, but the condition it was written to
detect (a persistent token-independent component dominating later pulses) is not what the
data shows. Router-input centering would target a transient that self-resolves inside ~20
steps, in the same window as the Muon spike and the routing divergence.

Other diagnostics over the run: `bias_override_rate` rises to 0.61 at step 50 and decays
to 0.19 by step 240; `bias_abs_p99` peaks at 0.177, far below the +-1.0 clamp;
`max_top1_share` stays 0.17-0.24 against a uniform 0.125.

**Recommendation: do not implement step 3.** Decision is the owner's.
