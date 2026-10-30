# Pulsar

Pulsar is a 283M-parameter, high-efficiency language model pairing recurrent compute efficiency with sparse Mixture-of-Experts (MoE), 32k hybrid local/global attention, harmonic residual refinement, and deep supervision.

Engineered for extreme parameter efficiency and throughput, Pulsar achieves the reasoning depth of a 24-layer Transformer while maintaining the active parameter and cache footprint of an 8-layer physical model. Validated locally and trained at scale on NVIDIA Blackwell and Hopper cloud GPUs (B300 SXM6, B200, H200, H100) via Modal.

---

## Key Architectural Specifications

| Specification | Value | Technical Details |
| :--- | :--- | :--- |
| **Model Name** | **Pulsar** | Native reasoning, agentic tool-calling, and code synthesis engine |
| **Total Parameters** | **283.42M** (`283,422,784`) | Continuous neural weights (zero static hash tables) |
| **Active Parameters** | **130.02M** (`130,023,424`) | 96.47M non-embedding active parameters per token (54.1% computational sparsity) |
| **Physical Core** | **8 Layers** | Recurrently looped K = 3 times |
| **Effective Reasoning Depth** | **24 Layers** | Equivalent compute depth to a 24-layer Transformer |
| **Hidden Dimension ($d_{\text{model}}$)** | **1024** | 16 Query Heads (GQA: 4 KV Heads, $d_{\text{head}} = 64$) |
| **Attention Output Gating** | **16-Head Sigmoid Gate** | Fused pulse-indexed per-head output gate with zero FlashAttention latency penalty |
| **Recurrent MoE MLP** | **1 Shared + 16 Routed** | Top-4 routed active ($d_{\text{expert}} = 512$, $d_{\text{shared}} = 1024$, active MLP width = 3072) |
| **Residual State Dynamics** | **Harmonic Refinement** | Learnable per-pulse channel scale $\mathbf{s}_k \sim 1/(k+1)$ (monotonic delta contraction) |
| **Context Length** | **32,768 Tokens** | 6 Local (window 1024, $\theta=10\text{k}$) + 2 Global (full causal 32k, $\theta=1\text{M}$) |
| **Vocabulary Size** | **32,768 Tokens** | Unified vocabulary matching standard open weights |
| **Optimizer** | **Muon + AdamW** | 2D weight matrices in Muon; embeddings, norms, gates & scales in AdamW |
| **Training Budget** | **3.80 Billion Tokens (Stage 1)** | 8 disjoint binary shards, 28,991 optimizer steps @ 131,072 tokens/step |
| **Cloud Throughput** | **260,000+ tok/s (24% MFU)** | NVIDIA B300 SXM6 Blackwell Ultra on Modal |

---

## Core Innovations

### 1. Fused Pulse-Indexed 16-Head Attention Output Gate
Evaluated immediately after FlashAttention (`scaled_dot_product_attention`) and prior to $W_o$:
$$\mathbf{g}_k = \sigma(\mathbf{x}\mathbf{W}_{\text{gate}}[k]^\top + \mathbf{b}_{\text{gate}}[k]) \in \mathbb{R}^{B \times T \times 16}$$
Operates via a custom fused CUDA kernel (`fused_attn_gate_forward_cuda` / `fused_attn_gate_backward_cuda`) that merges elementwise sigmoid multiplication and memory layout transposition into a single pass. Heads dynamically suppress attention-sink tokens and mute irrelevant heads without forcing non-contiguous gradient allocations in backward passes.

### 2. Recurrent MoE with Top-4 Routing and Shared Expert
Each physical layer incorporates an always-on shared expert ($d_{\text{shared}} = 1024$) and 16 specialized routed experts ($d_{\text{expert}} = 512$), selecting the Top-4 experts per token:
- Active FFN width: $1024 + 4 \times 512 = 3072$.
- Total FFN width across all 16 experts: $1024 + 16 \times 512 = 9216$.
- By linear algebra associativity, the routed output scale $\mathbf{r}_k$ is fused into the expert down-projection weights prior to `torch.bmm`, unlocking direct accumulation of `shared_out` inside the fused CUDA combine kernel and eliminating 6.4 GB/step of intermediate HBM allocations.

### 3. Harmonic Residual Refinement Scaling
Resolves the representation overwrite dilemma where recurrent updates previously dwarfed the base state:
$$\Delta \mathbf{h}_k = (\mathbf{h}_{\text{curr}} - \mathbf{h}_{\text{in}}) \odot \mathbf{s}_k, \quad \mathbf{s}_k \sim \frac{1}{k+1}$$
Harmonically scales updates ($1.00 \to 0.50 \to 0.33$), yielding strictly monotonic contraction across recursive passes ($\|\Delta \mathbf{h}_1\| = 40.0 > \|\Delta \mathbf{h}_2\| = 18.6 > \|\Delta \mathbf{h}_3\| = 12.4$).

### 4. 32k Hybrid Local/Global Attention
- **6 Local Layers** ($\theta = 10,000$, sliding window 1024): Provide high-resolution local positional representation for dense syntax, code structures, and JSON tool schemas.
- **2 Global Anchor Layers** (Layers 3 & 7, $\theta = 1,000,000$, full causal 32k span with YaRN temperature scaling): Execute full causal sequence sweeps. Looped K = 3 times, this executes 6 global context sweeps across sequence history.

### 5. Multi-Step Deep Supervision with Subsampled Auxiliary Loss
Intermediate recurrent pulses are guided by a 25% subsampled auxiliary cross-entropy loss with rotating disjoint offsets to prevent intermediate representation collapse:
$$\mathcal{L}_{\text{total}} = 0.15 \cdot \mathcal{L}_1 + 0.25 \cdot \mathcal{L}_2 + 0.60 \cdot \mathcal{L}_3 + \mathcal{L}_{\text{router\_z}}$$

### 6. High-Performance Fused CUDA Kernels (`pulsar_cuda`)
Hand-optimized C++/CUDA kernels in `csrc/kernels.cu` eliminate Python runtime overhead and HBM bandwidth roundtrips:
- **`qk_rmsnorm_rope`**: Fuses QK-RMSNorm and Rotary Positional Embeddings in registers, eliminating 32 GB/step of memory roundtrips.
- **`fused_attn_gate`**: Fuses 16-head sigmoid gating and memory transpose into a single vectorized pass.
- **`moe_dispatch` & `moe_combine`**: 128-bit vectorized `uint4` token dispatch and combine with register-level scaling and early-exit on dropped tokens.
- **`moe_slots_from_order`**: C++ post-argsort slot assignment and expert histogram reduction.
- **`fused_sr_update`**: Fuses Muon momentum weight updates with on-chip deterministic PRNG (SplitMix32) stochastic rounding in registers.
- **`fused_softcap_ce`**: Computes online logsumexp and fused backward in registers, cutting peak memory from 800 MB down to 8 KB per micro-batch.

---

## Directory Layout

```
pulsar-300m/
├── Makefile                # Unified developer automation targets
├── csrc/                   # Native C++ and CUDA kernel implementations
│   ├── kernels.cu          # Fused CUDA kernels (QK-RoPE, AttnGate, MoE, SR, Softcap CE)
│   └── pulsar_cuda.cpp     # PyTorch C++ bindings
├── src/                    # Core Python model and training engine
│   ├── model.py            # PulsarConfig, PulsarAttention, PulsarMoEMLP, PulsarForCausalLM
│   ├── optim.py            # PulsarDualOptimizer (Muon + AdamW with stochastic rounding)
│   ├── train.py            # Main training loop with automated telemetry rollback
│   ├── packing.py          # Best-fit document packing loader
│   └── infer.py            # Native KV-cache inference engine with sliding-window ring buffer
├── scripts/                # Cloud orchestration and utility scripts
│   ├── modal_train.py      # Modal cloud pretraining harness (B300, B200, H200, H100)
│   ├── sync_cloud_monitor.py # Background telemetry sync daemon from Modal volume
│   ├── calc_steps.py       # Exact curriculum and schedule calculator
│   ├── inspect_checkpoint.py # Comprehensive checkpoint inspector and tensor validator
│   └── benchmark.py        # Latency, memory, and throughput benchmarking suite
├── monitor/                # Real-time web telemetry dashboard
│   ├── web_server.py       # Lightweight HTTP server for live training dashboard
│   ├── index.html          # Modern dark-mode web monitoring UI
│   └── thresholds.json     # Automated health checks and warning thresholds
├── checkpoints/            # Single flat directory for checkpoints and telemetry
│   ├── latest.pt           # Atomic latest checkpoint
│   ├── best.pt             # Lowest validation loss checkpoint
│   ├── metrics.jsonl       # Per-step JSONL telemetry
│   ├── metrics.csv         # Tabular CSV training metrics
│   ├── live_status.json    # Instantaneous snapshot for web monitor
│   └── train.log           # Full stdout training logs
├── tests/                  # Unit tests and gradient parity assertions
│   ├── test_kernels.py     # Forward/backward parity checks for all fused CUDA kernels
│   ├── test_model.py       # Architectural constraints, parameter counts, memory benchmarks
│   ├── test_optim.py       # Muon and AdamW optimizer assertions
│   └── test_rollback.py    # Automated telemetry rollback validation
└── docs/                   # Detailed architectural and curriculum specifications
    ├── ARCHITECTURE.md     # In-depth architectural equations and tensor diagrams
    ├── CURRICULUM.md       # Pretraining curriculum and shard specifications
    ├── MONITOR.md          # Real-time telemetry dashboard architecture
    └── CLOUD_TRAINING.md   # Cloud training and operations runbook
```

---

## Quickstart

### 1. Installation
Clone the repository and set up a virtual environment with PyTorch and dependencies:
```bash
git clone https://github.com/yashwk/pulsar-300m.git
cd pulsar-300m
uv sync
```

### 2. Build Native CUDA Extension
Compile the custom CUDA acceleration library:
```bash
pip install -e . --no-build-isolation
```

Verify that the extension compiles and all tests pass:
```bash
uv run pytest tests/
```

### 3. Local Smoke Test Run
Run a quick 2-step verification run on CPU or GPU without writing checkpoints:
```bash
make smoke
# Or run manually:
python -m src.train --max-steps 2 --warmup-steps 1 --no-save \
    --micro-batch-size 2 --seq-len 1024 --train-bin data/smoke_train.bin \
    --val-bin data/smoke_val.bin --tokenizer data/pulsar_tokenizer
```

---

## Cloud Training on Modal

Pulsar is optimized for remote training on high-end cloud accelerators via Modal.

### 1. Launch Production Pretraining (Detached Mode)
Run unattended pretraining on an NVIDIA B300 SXM6 (Blackwell Ultra). The `--detach` flag launches the job in the cloud and exits immediately, ensuring that local Wi-Fi dropouts or laptop sleep will not interrupt training:
```bash
modal run scripts/modal_train.py \
    --gpu B300 \
    --detach \
    --target-step 28991 \
    --mbs 128 \
    --global-batch-tokens 131072 \
    --seq-len 1024 \
    --resume latest
```

### 2. Launch Local Real-Time Dashboard
Monitor training live at `http://localhost:7860`:
```bash
# Terminal 1: Background cloud telemetry sync
python scripts/sync_cloud_monitor.py --checkpoint-dir checkpoints

# Terminal 2: Web monitor server
python monitor/web_server.py --port 7860 --checkpoint-dir checkpoints
```

### 3. Automated Telemetry Rollback
If a cloud run is interrupted or preempted between checkpoint intervals, resuming with `--resume latest` automatically prunes orphaned records from `metrics.jsonl`, `metrics.csv`, and `train.log` back to the exact checkpoint step (`latest.pt`), guaranteeing clean, strictly monotonic telemetry curves on the dashboard.

---

## Technical Documentation
- [Architecture & Implementation Specification](docs/ARCHITECTURE.md)
- [Curriculum & Shard Specification](docs/CURRICULUM.md)
- [Monitoring & Dashboard Specification](docs/MONITOR.md)
- [Cloud Training & Operations Runbook](docs/CLOUD_TRAINING.md)

---


---

## Inference & Benchmarking

### 1. Interactive Inference
Generate completions from a trained checkpoint using the recurrent KV-cache engine:
```bash
python -m src.infer --checkpoint checkpoints/latest.pt --prompt "def fibonacci(n):"
```

### 2. Micro-Benchmark Suite
Benchmark execution speed, memory footprint, and MFU across model configurations:
```bash
python scripts/benchmark.py --seq-len 1024 --batch-size 4
```

## References & Acknowledgements
- **Unified Recurrent KV-Cache Dynamics:** Cross-pass recurrent state sharing inspired by UCSD Sandy Research / Fu et al. (*Be-Nice: Recurrent Transformers with Contractive State Attractors*).
- **DeepSeek Innovations:** Multi-head latent routing and auxiliary-loss-free balancing inspired by DeepSeek AI (*DeepSeek-V3 / DeepSeek-V4.1-Flash Technical Reports*).
- **Pipelined Self-Speculative Decoding:** Recurrent-depth draft extraction inspired by KAIST FLEXML Lab / Bae et al. (*LoopSpec: Pipelined Self-Speculative Decoding for Looped Transformers*).
- **Muon Optimizer:** High-efficiency matrix orthogonalization via Newton-Schulz iterations inspired by Jordan Keller / Moonshine AI (*Muon: An Optimizer for Hidden Layers*).
