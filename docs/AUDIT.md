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
