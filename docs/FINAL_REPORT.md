# Audit Remediation — Final Report

**Branch:** `fix/audit-fixes` (11 commits, nothing pushed)
**Tests:** 64 passed, 1 xfailed (`pytest tests/`)
**Scope completed:** 8 of 20 work items, plus 1 pre-existing bug found along the way.

---

## 1. Before / After

| Metric | Baseline | After fixes | Verdict |
|---|---|---|---|
| Peak VRAM | 6.98 GB | **7.36 GB** | ✅ within 7.5 GB budget |
| Throughput | 8,593 tok/s | **7,908 tok/s** | ❌ **−8.0%** (floor 8,163) |
| NaNs | none | none | ✅ |
| `pulse_res_scale` distinct values | **8** (frozen) | **3,012** | ✅ learned |
| `routed_scale` distinct values | **1** (frozen) | **2,937** | ✅ learned |
| Tests | 0 | 64 | ✅ |
| Loss (20 steps) | — | 17.4 → 7.1, monotone | ✅ |

The throughput miss is a measured consequence of the capacity-factor fix, explained in §5.

---

## 2. What was fixed

### Precision (WI-01) — `b4c4860`
`train.py` cast the model to bf16 then re-cast a name-based subset to FP32, missing
`pulse_res_scale`, `routed_scale`, `attn_gate`, `pulse_step_emb`. bf16 half-spacing near
1.0 is 3.9e-3 against a 6e-4 peak Adam step, so **those parameters could not move at all**.

Proven on the baseline checkpoint after 150 optimizer steps:
`pulse_res_scale[0]` = 1.000000 with **one unique value**; `routed_scale` = 1.000000
**everywhere**. Muon-updated matrices moved normally.

The same cast hit **buffers**: MoE expert counters (accumulate into the hundreds; bf16
saturates) and RoPE caches. Now FP32 via a single `to_training_precision()`, plus a fix
for the `pulse_step_emb` site that promoted the whole bf16 residual stream to FP32.

### Routed-expert init — `7233e28`
Gate/up and down used `1/sqrt(d)` ≈ 0.031 against the shared expert's residual-output
rule ≈ 0.0058. **Measured 229× too large**, not the ~10× the audit estimated. Now
matched to the shared expert; measured ratio **0.726** (target: within 2×).

### z-loss gradient — `7233e28`
The kernel baked in `2·w/N·lse` assuming an upstream gradient of 1 and discarded the
incoming `_d_z_loss`. With the 1/24 pulse-block average and 1/43 accumulation, the
effective weight was **1,032× the configured 1e-4**. Now computed in PyTorch from the
router logits so autograd applies the correct chain rule; verified against the analytic
formula to 5e-12. Configured weight stays 1e-4 per owner decision.

### Capacity and deterministic dispatch — `2cae367`
Slots were assigned with `atomicAdd`, so overflow victims depended on block scheduling
(nondeterministic, biased toward later tokens) and drops were never logged. Now
`compute_token_slots()` assigns deterministically, dropping only the **lowest-weight**
assignments; the kernel only copies. Capacity factor is configurable
(`moe_capacity_factor`, 1.08 → **1.25**). Drop statistics accumulate in GPU buffers and
are drained by `pop_moe_stats()` — no `.item()` in the hot path. The PyTorch fallback
applies the same rule, so CPU evaluation now matches training.

### Stochastic rounding (WI-02) — `bda3e41`
bf16 matrices round-to-nearest, so ~5e-4 updates against 1.2e-4 spacing vanished as LR
decayed, and weight decay (0.998) rounded away at any LR. `stochastic_round_()` adds
uniform 16-bit noise to the FP32 pattern before truncating, making many small updates
accumulate. Applied to Muon and to bf16 AdamW (`AdamWBF16SR`); FP32 params keep fused
AdamW. Seeded generator per (seed, step) keeps runs reproducible and resumable.

Two implementation traps found and fixed: promoting gradients to FP32 before the batched
Newton-Schulz **OOM'd** and cost **2.06 s/step** (fixed: keep bf16 into NS); and
`torch.bincount` in the hot path forced a device sync (replaced with `index_add_`).

### Pre-existing NaN bug — `2cae367` (not in the audit)
`qk_rmsnorm_forward_kernel` assumes `head_dim == 64` (32 lanes × 2 elements) but never
validated it. With any other head_dim the upper lanes read and write past each vector,
corrupting memory. Symptom: **intermittent NaNs in ~4% of runs**, reproducing on
unmodified code, vanishing when forward hooks were attached.

Diagnosis: poisoned the CUDA allocator with NaN buffers, then wrapped each
`autograd.Function` to report the first producer. Fixed with a `TORCH_CHECK` plus a
model-side guard that falls back to PyTorch. Two regression tests pin it, including one
under allocator poisoning. The real model uses head_dim 64, so training was unaffected.

---

## 3. Data pipeline (WI-13)

**The audit's premise was partly wrong, and the on-disk evidence corrected it.**

| Claim | Reality |
|---|---|
| Mid-trace `[...]` elision damaged the shards | Code exists, but **0 elision markers** in 952M tokens; think blocks are p50 483 tokens, under the 1,800-char trigger |
| Agentic traces filtered to 1,024 tokens | **No such filter exists** in the pipeline |
| Manifest's 20% CoT share is inflated | It reflects **real** reasoning content |

Traces are now stored verbatim with a 32k-token safety cap and no placeholder marker; all
randomness is seeded per (stream, shard) and recorded in the manifest; `.gitignore` had
been excluding the data pipeline's own source, now narrowed. No rebuild was needed — the
approved rebuild was cancelled after this check, saving ~1 hour and ~2 GB.

## 4. Packing (WI-07/WI-10 groundwork)

The flat sampler cuts a document of length L at a stride-T boundary with probability
~L/T, so at T=2,048 a median CoT document is cut ~62% of the time even though it fits.

Strategy C′ measured on shard 1 (475M tokens):

| T | intact among fittable (flat → packed) | padding waste | filler share |
|---|---|---|---|
| 2,048 | 59.3% → **100%** | 0.000% | 51.1% |
| 8,192 | 85.2% → **100%** | 0.001% | 17.5% |
| 32,768 | 95.7% → **100%** | 0.001% | 1.9% |

Filler comes only from documents longer than T (which can never be intact); padding is
emitted only at epoch end, with labels ignored and routing excluded. All three WI-07
guarantees verified: globally shuffled order, pure function of (seed, epoch), and every
token covered exactly once per epoch. At T=2,048 half of content is necessarily
mid-document filler — recorded in `CURRICULUM.md` so Stage 2 isn't misread.

---

## 5. The throughput miss — analysis and decision needed

The −8% is **not** a leak or an accident. Isolated per micro-batch:

| capacity factor | capacity | ms/micro-batch |
|---|---|---|
| 1.08 (old) | 832 | 347 |
| 1.12 | 896 | 354 |
| 1.15 | 896 | 353 |
| 1.25 (new) | 960 | **362** |

Expert GEMMs are ~67% of GPU time and scale with capacity, so 1.08 → 1.25 costs ~4.3%
per micro-batch (~7% per step at 43 accumulation steps).

This is a correctness/throughput trade-off, not a bug: 1.08 left only 8.3% headroom, so
drops were frequent and entirely invisible. **Options:** keep 1.25 and accept −8%; or set
1.12 (896 slots) which recovers most of the loss while still widening headroom.

---

## 6. Not done (12 of 20 work items)

WI-06 (eval metric), WI-07 wiring, WI-08 (logging), WI-09 (flex_attention), WI-10
(activation checkpointing, doc loader), WI-11 (curriculum), WI-12 (prefix-invariant
q-scaling), WI-14 (FP32 router), WI-15 (kernel hygiene — partially done),
WI-16 (dead code), WI-17 (perf).

WI-18 is **complete for the part that matters**: the layout-aware Muon scale convention
is landed and tested. Its optional `muon_split_qkv` / `muon_split_gate_up` ablations
were removed rather than left as accepted-but-ignored flags - see
`docs/AUDIT.md` -> "Accepted-but-ignored options".

WI-19 (per-pulse bias) is implemented behind `--per-pulse-bias`, default off.

Also outstanding: the baseline's val loss is the pulse-weighted sum plus z-loss, not
`val_ce_final`; the step-150 checkpoint is retained to re-score once WI-06 lands.
**Do not resume the real run from that checkpoint** — frozen scales, corrupted counters,
old init.

## 7. Demo caveat

The smoke run used a **20-minute subset**, not the full 250-step run the audit asked for.
164 baseline steps is 2.3% of one shard epoch with 60% warmup, so loss comparisons at
this scale are noise. Per owner agreement the After validation is: budgets + no NaN +
scales moving + tests, not a loss delta. On that checklist everything passes except
throughput (§5).

## 8. Open decisions

1. **Capacity factor 1.25 vs 1.12** — the only failing budget item.
2. **WI-04 damping** — with init fixed, re-measure raw ‖Δh‖/‖h_in‖ per pulse; if < 0.5 the
   `1/(k+1)` init protects nothing and 1.0 may be better.
3. **WI-11 stage split** — 94.2% of CoT documents fit T=2,048, so Stage 2 may deserve the
   reasoning-heavy mix, not just the 8-16k stage.
4. **z-loss weight** — 1e-4 retained; revisit once router-logit RMS is logged (WI-08).
