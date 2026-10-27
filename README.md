# ⚡ Pulsar

**Pulsar** is a 300M-parameter, high-efficiency language model pairing **recurrent compute efficiency** with **sparse Mixture-of-Experts (MoE)**, **32k hybrid local/global attention**, **harmonic residual refinement**, and **deep supervision**.

Built from the ground up for consumer hardware (NVIDIA RTX 4060 8GB Laptop GPU) while seamlessly scaling to multi-GPU clusters via modular 500M-token streaming shards.

---

## 🌟 Key Architectural Highlights

| Specification | Value | Notes |
| :--- | :--- | :--- |
| **Model Name** | **Pulsar** | Native agentic, reasoning, and coding engine |
| **Total Parameters** | **299.68M** | Continuous neural weights (zero static hash tables) |
| **Active Parameters** | **147.92M** | 97.59M non-embedding active per token (50.6% computational sparsity) |
| **Physical Core** | **8 Layers** | Recurrently looped $K=3$ times |
| **Effective Depth** | **24 Layers** | Equivalent compute to a 24-layer Transformer |
| **Hidden Dimension ($d_{\text{model}}$)** | **1024** | 16 Attention Heads (GQA: 4 KV heads, $d_{\text{head}}=64$) |
| **Attention Output Gating** | **16-Head Sigmoid Gate** | Pulse-indexed per-head output gate (zero attention-sink noise, 0% FlashAttention penalty) |
| **Recurrent MoE MLP** | **1 Shared + 8 Routed** | Top-2 active ($d_{\text{expert}}=1024$, active MLP width = 3072) with per-pulse routed scaling |
| **Residual State Dynamics** | **Harmonic Refinement** | Learnable per-pulse channel scale $\mathbf{s}_k \sim 1/(k+1)$ (monotonic delta contraction) |
| **Context Length** | **32,768 Tokens** | 6 Local (window 1024, $\theta=10\text{k}$) + 2 Global (full causal, $\theta=1\text{M}$) |
| **Optimizer** | **Muon + AdamW** | 2D/3D weight matrices in Muon; embeddings, norms, gates & scales in AdamW |
| **Training Horizon** | **6.0 Billion Tokens** | 3-Stage Curriculum packed in modular 500M-token binary shards |
| **Hardware Throughput** | **~3,400–10,200 tok/s** | RTX 4060 Laptop GPU with fused CUDA MoE, Softcap CE & QK-RMSNorm kernels |

---

## 🔬 Core Innovations

1. **Pulse-Indexed 16-Head Attention Output Gate:**
   - Evaluated immediately after FlashAttention (`scaled_dot_product_attention`) and prior to $W_o$:
     $$\mathbf{g}_k = \sigma(\mathbf{x}\mathbf{W}_{\text{gate}}[k]^\top + \mathbf{b}_{\text{gate}}[k]) \in \mathbb{R}^{B \times T \times 16}$$
   - Initialized open at $\sigma(2.0) \approx 0.88$, allowing heads to dynamically throttle attention-sink noise and close irrelevant heads with zero FlashAttention performance penalty.

2. **Harmonic Per-Pulse Residual Refinement Scale:**
   - Overcomes the representation overwrite dilemma where recurrent updates previously dwarfed the base state ($\|\Delta \mathbf{h}\| \gg \|\mathbf{h}_{\text{in}}\|$):
     $$\Delta \mathbf{h}_k = (\mathbf{h}_{\text{curr}} - \mathbf{h}_{\text{in}}) \odot \mathbf{s}_k, \quad \mathbf{s}_k \sim \frac{1}{k+1}$$
   - Harmonically scales updates ($1.00 \to 0.50 \to 0.33$), yielding strictly monotonic contraction ($\|\Delta \mathbf{h}_1\| = 40.0 > \|\Delta \mathbf{h}_2\| = 18.6 > \|\Delta \mathbf{h}_3\| = 12.4$).

3. **Per-Pulse Routed-Expert Scaling:**
   - In `PulsarMoEMLP`, a learnable FP32 vector $\mathbf{r}_k$ scales the Top-2 routed expert contribution relative to the Always-On Shared Expert:
     $$\mathbf{y}_{\text{moe}} = \mathbf{y}_{\text{shared}} + \mathbf{y}_{\text{routed}} \odot \mathbf{r}_k$$
   - Balances foundational shared knowledge in early pulses with deep domain specialization in late pulses.

4. **Tool-Call Format:**
   - Agentic tool calls use a single, consistent format across the corpus:
     `<|tool_call|>{"name": ..., "arguments": ...}<|/tool_call|>`
   - Measured over the 952M-token corpus: 100,914 balanced open/close pairs, every
     payload using the `{"name", "arguments"}` schema. (An earlier version of this README
     described a 70/15/15 split across native, XML and Markdown-JSON formats; the corpus
     does not contain such a split.)

5. **Verbatim Reasoning Traces:**
   - `<|think|>` traces are stored intact. Measured over the corpus: 114,893 balanced
     open/close pairs and **zero** `\n\n[...]\n\n` splices. Every one of the 215
     `[...]` occurrences is genuine source content (NumPy ellipsis, quoted text, name
     elisions), not an inserted elision.
   - Training on ungrounded "skip the middle" bridges would teach the model to emit
     `[...]` and invent conclusions, so traces are kept verbatim instead.

6. **32k Hybrid Attention Pattern:**
   - 6 Local High-Precision Layers ($\theta=10,000$, sliding window 1024) provide 4x sharper positional resolution for code, math, and JSON.
   - 2 Global Anchor Layers (Layers 3 & 7, $\theta=1,000,000$) perform full causal 32k sweeps with YaRN query temperature scaling. Looped $K=3$ times, this executes 6 global context sweeps across sequence history.

7. **Multi-Step Deep Supervision with Subsampled Auxiliary Loss:**
   - Intermediate pulses use a 25% subsampled auxiliary cross-entropy loss with rotating disjoint offsets to prevent intermediate representation collapse:
     $$\mathcal{L}_{\text{total}} = 0.15 \cdot \mathcal{L}_1 + 0.25 \cdot \mathcal{L}_2 + 0.60 \cdot \mathcal{L}_3 + \mathcal{L}_{\text{router\_z}}$$

8. **Fused C++ CUDA Acceleration (`pulsar_cuda`):**
   - 128-bit vectorized (`uint4`) token dispatch & combine CUDA kernels eliminate Python loop and `torch.argsort` overhead.
   - Fused Logit Soft-Capped Cross-Entropy kernel (`fused_softcap_ce`) computes online logsumexp and fused backward in registers, cutting memory allocations from 800 MB down to 8 KB per micro-batch.
   - Fused QK-RMSNorm warp reduction kernel replaces 864 separate elementwise PyTorch kernels.

---

## 🚀 Quickstart

### 1. Build Native CUDA Extension
```bash
python setup.py build_ext --inplace
```

### 2. Check the Shards on Disk
The corpus is already built and tokenized (`data/*.bin`, uint16 token ids); there is no
shard-preparation script in this repository. See the state of what is present, including
document-length structure and what packing at a given context length would cost:
```bash
python scripts/measure_packing.py --seq-len 1024
python scripts/analyze_agentic_lengths.py --tokenizer data/pulsar_tokenizer
```

### 3. Sanity-Check the Trainer
```bash
# 2-step smoke run: --no-save avoids writing a 1.3 GB checkpoint
python -m src.train --max-steps 2 --warmup-steps 1 --no-save \
    --micro-batch-size 3 --seq-len 1024 --checkpoint-dir checkpoints/smoke
```
Omit `--no-save` for a real run; checkpoints go to `--checkpoint-dir` (default
`checkpoints/`), and `latest.pt` / `best.pt` are overwritten in place.

### 4. Launch Web Telemetry Dashboard
```bash
python monitor/web_server.py --port 7860
```
Open [http://localhost:7860](http://localhost:7860) to view real-time loss, per-pulse hidden state contraction ($\Delta h$), expert load distribution, and throughput gauges.

### 5. High-Efficiency Native Inference Engine
Pulsar implements a native unified KV-cache inference engine with sliding-window ring buffering for 11x KV cache compression (73.4 MB @ 32k context):
```bash
# Interactive REPL:
python src/infer.py --checkpoint checkpoints/best.pt --interactive

# Single prompt completion:
python src/infer.py --checkpoint checkpoints/best.pt --prompt "def quicksort(arr: list[int]) -> list[int]:"

# Memory footprint benchmark across context lengths:
python src/infer.py --benchmark
```

---

## 📚 Technical Documentation
- [500-Step Pretraining Milestone Report (CE: 2.6761, Latent Gain: +0.1465)](docs/TRAINING_RUN_500.md)
- [Architecture & Implementation Specification](docs/ARCHITECTURE.md)
- [Curriculum & Shard Specification](docs/CURRICULUM.md)
- [Engineering Audit & Telemetry Notes](docs/AUDIT.md)

---

## 📑 References & Acknowledgements
- **Unified Recurrent KV-Cache Dynamics:** Attractor dynamics and cross-pass recurrent state sharing inspired by UCSD Sandy Research / Fu et al. (*Be-Nice: Recurrent Transformers with Contractive State Attractors*).
- **Compressed Sparse & Sliding-Window Inference Bounds:** Multi-layer hybrid KV caching and context scaling inspired by DeepSeek CSA2 architecture (*DeepSeek-V4.1-Flash*).
