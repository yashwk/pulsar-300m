# Throughput baseline

Measured on the current code at `fix/round2`, so these numbers are a valid reference for
later comparisons (R-03 delta, R-11 re-tune gates).

**These replace the 8,593 tok/s figure and 8,163 tok/s floor used earlier.** Those values
do not appear anywhere in this repository (`grep` over `docs/`, source and commit history)
and should be treated as unsourced. Everything below was measured here.

## Method

- `--max-steps 20 --eval-every 1000 --eval-iters 1 --no-save`, seed 42
- 1,024 token sequences, bestfit sampler, K=3, warmup 1 (so LR is nominal, not ramping)
- Mean of the last 12 reported `tok/s` values per run (steps 9-20), after warmup and
  compile-free steady state
- Single run per config; run-to-run spread observed on this machine is roughly +-3%

## Hardware

RTX 4060 Laptop, 8 GB (8,188 MiB), sm_89. PyTorch 2.14.1+cu130, CUDA 13.4.
The training session had exclusive use of the GPU (no other process >100 MiB).

## Results

| Config | Tokens/step | tok/s | VRAM peak | Step time |
|---|---|---|---|---|
| MBS 3, capacity 1.12 | 197,632 | **7,916** | 7.9 GB | ~25 s |
| **MBS 2, capacity 1.375** | 132,096 | **7,400** | 6.4 GB | ~18 s |
| MBS 2, capacity 1.12 | 132,096 | **7,000** | 6.2 GB | ~19 s |

Notes:

- MBS 3 moves more tokens per step but at 7.9 GB it leaves ~300 MiB of headroom on an
  8 GB card. It is the fastest in raw tok/s and the least safe.
- MBS 2 at capacity 1.375 is the configuration the 107-step run used. Its VRAM margin
  (~1.6 GB) is what makes it the practical choice.
- Capacity 1.12 at MBS 2 is *slower* than 1.375 despite allocating less: the smaller
  expert buffer overflows and the dispatch path does more wasted work. This is the
  strongest argument against reverting capacity to 1.12 without also changing something
  else.

## Usage

- R-03 deltas are measured against the MBS 2 / capacity 1.375 row (same config).
- R-11 re-tune gates: a candidate configuration must reach **7,400 tok/s or better** at
  MBS 2 / 1.375 to be considered non-regressive, or state why throughput was traded.
