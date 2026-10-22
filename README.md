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

4. **Multi-Harness Tool-Call Randomization:**
   - Formats agentic tool calling probabilistically across 3 major agent ecosystems:
     - **70% Native Fast Format:** `<|tool_call|>{"name": ..., "arguments": ...}<|/tool_call|>`
     - **15% XML Format (Cline/Claude Code style):** `<|tool_call|><invoke name="..."><arguments>...</arguments></invoke><|/tool_call|>`
     - **15% Markdown JSON Format:** ````json\n{"tool": ..., "arguments": ...}\n````

5. **Complete-Thought Sentence Guard:**
   - Ensures `<|think|>` traces never cut mid-sentence or mid-word. Naturally concise traces ($\le 2,400$ chars) remain 100% intact, while longer traces preserve the opening problem setup, a clean `\n\n[...]\n\n` transition, and a complete concluding deduction terminating strictly with sentence punctuation.

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

### 2. Generate Shards (NVIDIA Nemotron 5-Pillar Mix)
```bash
python data/prepare_shards.py --stage 1 --shard-idx 1 --total-tokens 500000000
python data/prepare_shards.py --stage 1 --shard-idx 2 --total-tokens 500000000
```

### 3. Verify Shard Integrity
```bash
python data/audit_shard.py --shard-path data/train_shard1.bin --min-count 5000
```

### 4. Launch Web Telemetry Dashboard
```bash
python monitor/web_server.py --port 7860
```
Open [http://localhost:7860](http://localhost:7860) to view real-time loss, per-pulse hidden state contraction ($\Delta h$), expert load distribution, and throughput gauges.

---

## 📚 Technical Documentation
- [Architecture & Derivations](docs/ARCHITECTURE.md)
- [Model Implementation Blueprint](docs/model.md)
- [Curriculum & Shard Specification](docs/CURRICULUM.md)
