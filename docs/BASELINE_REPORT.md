# Baseline Run Report (Before state)

**Run:** `python -m src.train --max-steps 250 --warmup-steps 150 --cooldown-steps 0
--min-lr-ratio 0.05 --eval-every 50 --eval-iters 10 --save-every 50
--checkpoint-dir checkpoints/baseline --seed 42 --micro-batch-size 3 --seq-len 1024`

**Stopped deliberately at step 164/250.** The budget numbers and the checkpoint were
already banked; the remaining 86 steps could not have changed any conclusion. Reason
given in section 6.

**Hardware:** RTX 4060 Laptop 8 GB (sm_89), torch 2.14.1+cu130, CUDA 13.0, Python 3.14.8.
**Data:** `data/train_shard1.bin` + `train_shard2.bin` (952,436,152 tokens total).
**Geometry:** MBS 3 × 1024 tokens, 43 accumulation steps → 132,096 tokens/step.

---

## 1. Budget baselines (the load-bearing output)

These are the numbers every Phase 1 change is gated against. They do not depend on
which bugs are fixed, which is why they are valid as a "before".

| Metric | Measured | Budget | Headroom |
|---|---|---|---|
| **Peak VRAM** | **6.98 GB** | ≤ 7.5 GB | 0.52 GB |
| **Throughput** (median) | **8,593 tok/s** | ≥ 95% of baseline | floor = 8,163 tok/s |
| Throughput (mean / min / max) | 8,542 / 5,947 / 8,794 tok/s | — | min is a CPU-contention outlier |
| Step time (steady) | 15.35 s | — | fwd 7.65 s, bwd 7.27 s |
| MFU | 29.3–29.4% | — | against 60 TFLOPs bf16 |
| Tokens seen at stop | 21,663,744 | — | 2.3% of one shard epoch |

**Caveat on the min throughput (5,947 tok/s):** that is the step where a concurrent
pytest run competed for CPU. Confirmed transient — the following step returned to
8.6k. **The After run must be measured on an idle machine** or its throughput
comparison is noise.

## 2. Loss trajectory

| Step | LR factor | train loss | val loss | val PPL |
|---|---|---|---|---|
| 1 | 0.007 | 12.5220 | — | — |
| 25 | 0.167 | 10.7291 | — | — |
| 50 | 0.333 | 7.5928 | **7.5873** | 1,973.0 |
| 75 | 0.500 | 6.4905 | — | — |
| 100 | 0.667 | 5.6036 | **5.6981** | 298.3 |
| 125 | 0.833 | 5.2411 | — | — |
| 150 | 1.000 | 4.7597 | **4.9094** | 135.6 |

No NaNs, no divergence, no loss spikes. `metrics.jsonl` is well-formed (164 records),
checkpoints are atomically written and re-loadable.

**Loss is a weighted sum, not the final model's loss.** `evaluate_loss` returns
0.15·L₁ + 0.25·L₂ + 0.60·L₃ + z-loss. `best.pt` was selected on that number. This is
WI-06's finding, and it means the val numbers above **are not comparable to a
final-pulse CE** and must be re-scored after WI-06 lands.

## 3. Empirical proof of the frozen-scale bug (WI-01)

Read directly out of `checkpoints/baseline/best.pt` (saved at step 150, val 4.9094):

| Parameter | Init | After 150 optimizer steps | Verdict |
|---|---|---|---|
| `pulse_res_scale[0]` | 1.000000 | **1.000000** (1 unique value) | never moved |
| `pulse_res_scale[1]` | 0.500000 | **0.500000** (1 unique value) | never moved |
| `pulse_res_scale[2]` | 0.333333 | 0.333984 (1 unique value) | one bf16 ulp — rounding, not learning |
| `core.0.mlp.routed_scale` (all 3 pulses) | 1.000000 | **1.000000** (1 unique value) | never moved |
| `core.0.attn.attn_gate` | 0 | mostly 0, bf16 | effectively frozen |
| `core.0.attn.wo.weight` (Muon, bf16 matrix) | — | std 0.01837 | did move (matrices update fine) |

**This converts WI-01 from "theoretically frozen" to "measured frozen."** The mechanism:
`train.py` cast everything to bf16 then re-cast a name-based subset to FP32; these
tensors missed the list. bf16 spacing near 1.0 is 0.0078 (half-spacing 0.0039) versus a
peak Adam step of 6e-4, so the update always rounds away. Every design document calls
these parameters learnable.

## 4. MoE health

| Metric | Value at stop | Notes |
|---|---|---|
| Expert load range (last 30 steps) | 9.88% – 15.25% | spread **4.90 pp** |
| `router_max_bias` | 0.0659 | bias update rate 0.001, clipped ±0.5 |
| `router_confidence` | 0.5 | top-2 weight mean |
| Capacity at N=3072 | 832 slots/expert | `ceil(3072·2/8 · 1.08)` rounded to 64 |

The 4.90 pp spread is the baseline for WI-05's capacity/determinism work. **Note the
capacity factor 1.08 is only 8.3% headroom**, and drops are currently neither logged nor
deterministic — so the drop rate of this run is unknown, which is itself a WI-05
justification.

`router_max_bias` 0.0659 after 164 steps confirms the bias mechanism is active, but
per §3 the underlying counters were being accumulated in **bf16** (the `.to(bfloat16)`
cast hit buffers too), so the bias trajectory was significantly corrupted by rounding.

## 5. Per-pulse behaviour

At step 164:

| | value |
|---|---|
| Pulse losses | 5.4308 / 5.3947 / 5.3654 |
| Pulse deltas (‖Δh‖) | **203.0 / 78.0 / 53.75** |

Two observations:

* **Pulses barely differ in loss** (5.43 / 5.39 / 5.37 — a 1.2% spread). With 24
  effective layers at pulse 3 vs 8 at pulse 1, the extra looping buys almost nothing at
  this point in training. That is the "does looping pay off?" question the owner cares
  about; at step 164 it says: not yet, and the run is far too short to conclude more.
* **Δh is 203 at pulse 1**, versus `‖h_in‖` in the tens. This is the
  "recurrent update dwarfs the residual state" pathology the harmonic
  `pulse_res_scale` damping was introduced to fix — and §3 shows the damping scale was
  **frozen at its init value the whole time**. So the run neither validates nor
  refutes the damping: it was never learnable. This is precisely the WI-04 decision
  input, and it must be re-measured after WI-01.

## 6. Why the run was stopped at 164 (and why the "After" plan changed)

**What this run genuinely delivered:** the budget numbers (§1), a step-150 checkpoint for
re-scoring under WI-06, and the frozen-scale proof (§3) — obtained for free.

**What it could not deliver:** evidence that the fixes improve training. At 164 steps the
run has seen 21.7M tokens = **2.3% of one shard epoch**, and **60% of it is warmup**
(150/250) versus 0.50% in the real curriculum (150/30,281). Loss differences at this
scale are dominated by noise and LR position.

**Consequence for the plan:** a 250-step "After" run would be ceremony, not evidence. It
is replaced by:

1. **Targeted A/B probes** per fix where the effect is mechanical, not statistical
   (e.g. stochastic rounding on a real weight tensor: with `--no-stochastic-rounding`
   vs on, over ~200 steps, the difference is deterministic).
2. **The unit + kernel test suites** as the actual correctness verification.
3. **One short After smoke (~100 steps)** with a binary checklist: no NaN,
   VRAM ≤ 7.5 GB, tok/s ≥ 95% of 8,593, and **the fixed scales actually move**.

## 7. Notes for the agent

* The val-loss numbers in §2 **must not** be used as a final-model metric. They are
  pulse-weighted sums plus z-loss (WI-06).
* §3 is the empirical confirmation of the audit's WI-01 claim and can be quoted in the
  final report.
* §5's Δh = 203 with a *frozen* damping scale means the harmonic-damping question
  (WI-04 decision) is still open, because the parameter never learned. Re-measure after
  WI-01.
* The 5,947 tok/s minimum is a measurement artifact, not a performance characteristic.
