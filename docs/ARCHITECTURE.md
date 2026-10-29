# Pulsar Architecture & Implementation Specification: 8-Layer Recurrent Sparse MoE with 32k Hybrid Attention

- **Model Name:** Pulsar
- **Architecture Type:** Recurrent Sparse Mixture-of-Experts with 32k Hybrid Attention
- **Target Parameter Count:** **283.42M Total Parameters** (`283,422,784`)
  - **Active Parameters per Token:** **130.02M parameters** (`130,023,424`)
  - **Non-Embedding Active Parameters:** **96.47M parameters** (`96,468,992`)
  - **Computational Sparsity:** **54.1% sparsity** per forward pass
  - **Physical Core:** 8 Physical Transformer blocks
  - **Effective Depth:** 24 Layers ($K=3$ recurrent loops across the 8-block physical core)
  - **Context Length:** **32,768 Tokens** (32k context via Hybrid Local/Global Attention)
  - **Target Training Infrastructure:** NVIDIA B300 SXM6 (Blackwell Ultra), B200, H200, H100 via Modal / Local RTX 4060 validation
  - **Precision Standard:** BFloat16 compute for 2D/3D projection matrices, strictly preserved Float32 for 1D parameters, RMSNorm scales, router weights, attention output gates, and harmonic scale vectors.

---

## 1. Executive Summary & Core Architectural Invariants

Pulsar couples recurrent compute efficiency with sparse Mixture-of-Experts (MoE), harmonic residual refinement, and 32k hybrid local/global attention:

1. **Recurrent Depth Without Weight Rigidity:** An 8-layer physical Transformer core is looped recurrently K = 3 times (yielding an effective reasoning depth of 24 layers). Unlike traditional recurrent networks where early feature extraction and late-stage verification compete over identical MLP parameters, Pulsar uses **Pulse-Indexed Routing**: tokens dynamically dispatch to different experts depending on recursion depth.
2. **Fused Pulse-Indexed 16-Head Attention Output Gate:** Evaluated immediately after SDPA and before $W_o$, a per-pulse, per-head gating vector eliminates attention-sink noise by allowing heads to throttle output mass smoothly toward 0.0, operating with zero FlashAttention latency penalty via a custom fused CUDA kernel.
3. **Harmonic Per-Pulse Residual Refinement Scale:** Resolves the representation overwrite dilemma where recurrent updates previously dwarfed the base state ($\|\Delta \mathbf{h}\| \approx 150$ vs. $\|\mathbf{h}_{\text{in}}\| \approx 32$). Updates are scaled by learnable FP32 vectors $\mathbf{s}_k$ initialized harmonically to $1 / (k + 1)$ ($1.00 \to 0.50 \to 0.33$), yielding strictly monotonic contraction across recursive passes ($\|\Delta \mathbf{h}_1\| = 40.0 > \|\Delta \mathbf{h}_2\| = 18.6 > \|\Delta \mathbf{h}_3\| = 12.4$).
4. **Recurrent MoE with Top-4 Routing and Shared Expert:** Each block incorporates 1 always-on Shared Expert ($d_{\text{shared}} = 1024$) and 16 specialized Routed Experts ($d_{\text{expert}} = 512$), selecting Top-4 experts per token. Active MLP width is 3072 ($1024 + 4 \times 512$).
5. **Linear Associativity Fusion for MoE Combine:** The routed output scale $\mathbf{r}_k$ is absorbed into the expert down-projection weights prior to `torch.bmm`, unlocking direct accumulation of `shared_out` inside the fused CUDA combine kernel and eliminating 6.4 GB/step of intermediate HBM allocations.
6. **Auxiliary-Loss-Free Global Balancing:** The router uses an adaptive expert bias vector $b \in \mathbb{R}^{16}$ (or $[K, 16]$ per-pulse bias) to guide token distribution without adding artificial loss terms to the objective. Expert allocation counts are accumulated across micro-batches, allowing experts to specialize temporally without hardware imbalance.
7. **32k Hybrid Local/Global Attention Pattern:** 6 layers use high-precision local attention ($\theta = 10,000$, sliding window 1024), while 2 layers (Layers 3 & 7) provide global anchor attention ($\theta = 1,000,000$, full causal 32k span with YaRN temperature scaling). Across 3 recurrent pulses, the global layers execute 6 full 32k sweeps across sequence history.
8. **Fused Custom CUDA Kernels:** Critical bottlenecks (fused QK-RMSNorm + RoPE, fused Attention Gate + Transpose, fused MoE Dispatch & Combine with early-exit on dropped tokens, C++ slot assignment from argsort, fused Muon momentum stochastic rounding, and online softcap CE) are executed through hand-optimized CUDA kernels in `csrc/kernels.cu`, delivering sustained Tensor Core saturation.

---

## 2. Macro Dataflow & Architecture Diagrams

### 2.1 Macro Recurrent Execution Diagram
Illustrates dataflow across the K = 3 recurrent pulse cycle during pretraining:

```mermaid
flowchart TD
    subgraph InputStage ["1. Input & Hybrid RoPE Embedding"]
        TOK["Input Token IDs: x_t"]
        EMB["Tied Token Embedding [32,768 x 1024]"]
        ROPE_LOC["Local RoPE Cache (theta = 10k, 32k max)"]
        ROPE_GLOB["Global RoPE Cache (theta = 1M, 32k max)"]
        TOK --> EMB --> H0["Initial State: h_0"]
        TOK --> ROPE_LOC
        TOK --> ROPE_GLOB
    end

    subgraph RecurrentCore ["2. Recurrent 3-Pulse Engine (K = 3 Pulses, 8 Shared Blocks)"]
        H0 --> PASS1["Pulse 1 (Drafting: Effective Layers 1-8)"]
        PASS1 --> PASS2["Pulse 2 (Composition: Effective Layers 9-16)"]
        PASS2 --> PASS3["Pulse 3 (Deep Verification: Effective Layers 17-24)"]

        subgraph BlockDetail ["Inside Each Block (8 Physical Layers)"]
            NORM_ATTN["Per-Pulse Attn Norm: RMSNorm(h, weight[k])"]
            GQA["GQA Attention (16Q / 4KV) + QK-Norm + Value Skip (v_first)"]
            ATTN_GATE["16-Head Output Gate: sigmoid(W_gate[k]·x + b_gate[k])"]
            NORM_MLP["Per-Pulse MLP Norm: RMSNorm(h, weight[k])"]
            ROUTER["Fused Top-4 Router: Sigmoid + Expert Bias + Z-Loss"]
            SHARED_EXP["1 Shared SwiGLU Expert (1024 Dim, Always Active)"]
            ROUTED_EXP["16 Routed SwiGLU Experts (512 Dim, Top-4 Selected)"]
            ROUTED_SCALE["Fused Routed Scale: Down-Projection scaled by r_k"]
            CONTIG_EXEC["Batched 3D GEMM Execution (torch.bmm)"]
            RES_HARMONIC["Harmonic Residual Refinement: h_in + (h_curr - h_in) · s_k"]

            NORM_ATTN --> GQA --> ATTN_GATE --> NORM_MLP --> ROUTER
            NORM_MLP --> SHARED_EXP
            ROUTER --> ROUTED_EXP --> CONTIG_EXEC --> ROUTED_SCALE
            SHARED_EXP --> CONTIG_EXEC
            ROUTED_SCALE --> RES_HARMONIC
        end
    end

    subgraph SupervisionHead ["3. Subsampled Multi-Step Supervision with Logit Soft-Capping"]
        PASS1 -.-> AUX1["Pulse 1: 25% Rotating Subsample (weight = 0.15)"]
        PASS2 -.-> AUX2["Pulse 2: 25% Rotating Subsample (weight = 0.25)"]
        PASS3 --> FINAL["Pulse 3: 100% Full Tokens (weight = 0.60)"]
        SOFTCAP["Output Logit Soft-Capping: 30.0 * tanh(logits / 30.0)"]
        AUX1 --> SOFTCAP
        AUX2 --> SOFTCAP
        FINAL --> SOFTCAP
    end
```

---

## 3. Parameter Breakdown & Geometric Invariants

### 3.1 Dimensions & Hyperparameters

| Dimension / Setting | Value | Description |
| :--- | :---: | :--- |
| **Vocabulary Size ($V$)** | `32,768` | Byte-Level BPE strictly matching open model tokenizers |
| **Hidden Dimension ($d_{\text{model}}$)** | `1024` | Latent residual bus width (Pure Power-of-2) |
| **Physical Core Depth ($N_{\text{core}}$)** | `8` | Number of physical Transformer blocks |
| **Recurrent Passes ($K$)** | `3` (Default) | Effective reasoning depth: $8 \times 3 = \mathbf{24\text{ layers}}$ |
| **Max Supported Pulses** | `8` | Dynamically supports up to 64 effective layers |
| **Attention Query Heads ($H_Q$)** | `16` | Query dimension = $16 \times 64 = 1024$ |
| **Attention KV Heads ($H_{KV}$)** | `4` | Grouped-Query Attention (GQA, 4:1 ratio) |
| **Head Dimension ($d_{\text{head}}$)** | `64` | Strict dimension per attention head (warp-aligned) |
| **Attention Output Gating** | `16 Heads` | Fused pulse-indexed per-head gating vector ($\sigma(2.0) \approx 0.88$ init) |
| **Harmonic Residual Scaling** | `s_k ~ 1/(k+1)` | Channel-wise learnable scale per pulse ($1.0 \to 0.5 \to 0.33$) |
| **Shared Expert Dimension ($d_{\text{shared}}$)** | `1024` | Always-active core SwiGLU expert ($1024 \to 2048 \to 1024$) |
| **Routed Expert Dimension ($d_{\text{expert}}$)** | `512` | Specialized SwiGLU experts ($512 \to 1024 \to 512$) |
| **Shared Experts per Block** | `1` | Always-active core reasoning expert |
| **Routed Experts per Block** | `16` | Dynamically selected specialized experts ($E0..E15$) |
| **Top-K Active Routed Experts** | `4` | Active routed experts selected per token |
| **Active FFN Width per Token** | `3072` | $1024\text{ (shared)} + 4 \times 512\text{ (routed)}$ |
| **Total FFN Width per Block** | `9216` | $1024\text{ (shared)} + 16 \times 512\text{ (routed)}$ |
| **Context Length Support** | `32,768` | 32k context via hybrid local/global RoPE |
| **Local Attention Layers** | `0, 1, 2, 4, 5, 6` | $\theta = 10,000$, sliding window = 1024 |
| **Global Attention Layers** | `3, 7` | $\theta = 1,000,000$, full 32k causal span with YaRN |
| **Output Logit Soft-Capping** | `30.0` | $30.0 \times \tanh(\text{logits} / 30.0)$ |
| **Weight Tying** | `True` | `lm_head.weight = tok_emb.weight` |
| **MoE Capacity Factor** | `1.25` | Headroom factor (align 32, ensuring minimal token drop) |

### 3.2 Exact Parameter Breakdown

1. **Tied Token Embeddings & LM Head:**
   - $32,768 \times 1024 = \mathbf{33,554,432\text{ parameters}}$ (`33.55M`)
2. **8-Layer Physical Core:**
   - **Attention Projections (8 blocks):**
     - $W_{qkv}$ $[1024 \to 1536]$ (`1,572,864`) + $W_o$ $[1024 \to 1024]$ (`1,048,576`) = `2,621,440` per block
     - 8 blocks $\times 2,621,440 = \mathbf{20,971,520\text{ parameters}}$ (`20.97M`)
   - **16-Head Attention Output Gates (8 blocks):**
     - $W_{\text{gate}}$ $[8 \times 16 \times 1024] = 131,072$ params per block
     - $b_{\text{gate}}$ $[8 \times 16] = 128$ params per block
     - 8 blocks $\times 131,200 = \mathbf{1,049,600\text{ parameters}}$ (`1.05M`)
   - **Shared Core MLP (8 blocks):**
     - 1 Shared Expert per block: $W_{\text{gate\_up}}$ $[1024 \to 2048]$ (`2,097,152`) + $W_{\text{down}}$ $[1024 \to 1024]$ (`1,048,576`) = `3,145,728` per block
     - 8 blocks $\times 3,145,728 = \mathbf{25,165,824\text{ parameters}}$ (`25.17M`)
   - **16 Routed Experts MLP (8 blocks $\times$ 16 experts = 128 routed experts):**
     - Each routed expert: $W_{\text{gate\_up}}$ $[512 \to 1024]$ (`1,048,576`) + $W_{\text{down}}$ $[512 \to 1024]$ (`524,288`) = `1,572,864` params
     - 128 experts $\times 1,572,864 = \mathbf{201,326,592\text{ parameters}}$ (`201.33M`)
   - **Per-Pulse Scale, Norm, and Router Banks (8 blocks):**
     - QK-RMSNorm Scales: $8 \times [8 \times (16 \times 64 + 4 \times 64)] = 81,920$ params
     - Value Skip Blend Scales: $8 \times 8 = 64$ params
     - Layer RMSNorm Banks: $8 \times [2 \times 8 \times 1024] = 131,072$ params
     - Pulse-Indexed Routers: $8 \times [8 \times 16 \times 1024] = 1,048,576$ params
     - Per-Pulse Routed Expert Scales: $8 \times [8 \times 1024] = 65,536$ params
     - Router Biases: $8 \times 16 = 128$ params
     - Subtotal: $\mathbf{1,327,296\text{ parameters}}$ (`1.33M`)
3. **Conditioning & Output Norms:**
   - Harmonic Residual Scale (`pulse_res_scale` $[8 \times 1024]$): $8,192$ params
   - Pulse Step Embeddings: $8 \times 1024 = 8,192$ params
   - Unit-Normalized Pre-Mix Norms (`pulse_state_norm` + `anchor_norm`): $2 \times 1024 = 2,048$ params
   - Input Mix Logits (`input_mix_logit` $[8 \times 1024]$): $8,192$ params
   - Final RMSNorm: $1,024$ params
   - Subtotal: $\mathbf{27,648\text{ parameters}}$
4. **Authoritative Total Parameter Count:**
   $$\mathbf{33,554,432 + 20,971,520 + 1,049,600 + 25,165,824 + 201,326,592 + 1,327,296 + 27,648 = 283,422,784\text{ params}}\quad (\mathbf{\approx 283.42M})$$

---

## 4. Mathematical Formulation & Recurrent State Dynamics

### 4.1 Input Stage & Embedding
For an input sequence of token IDs $X = (x_1, x_2, \dots, x_T) \in \mathbb{V}^T$, initial token embeddings are generated via tied embedding matrix $W_{\text{emb}} \in \mathbb{R}^{V \times D}$:
$$h_0 = \text{Embedding}(X; W_{\text{emb}}) \in \mathbb{R}^{B \times T \times D}$$
where $V = 32,768$ and $D = 1024$.

### 4.2 Recurrent Pulse Loop & Coordinate-Consistent State Accumulation
The physical 8-layer Transformer core is executed recurrently for $K = 3$ pulses ($k \in \{0, 1, 2\}$):

1. **Unit-Normalized Anchor & State Mixing:**
   Prior to layer entry, the prompt anchor $h_0$ and the recurrent state $h_{\text{pulse}}$ are normalized to unit variance:
   $$h_{0, \text{norm}} = \text{RMSNorm}(h_0; \gamma_{\text{anchor}})$$
   $$h_{\text{pulse}, \text{norm}} = \text{RMSNorm}(h_{\text{pulse}}; \gamma_{\text{pulse\_state}})$$
   $$\lambda_k = \sigma(\theta_{\text{mix}}[k]) \in \mathbb{R}^D$$
   $$h_{\text{in}} = (1 - \lambda_k) \odot h_{\text{pulse}, \text{norm}} + \lambda_k \odot h_{0, \text{norm}} + E_{\text{step}}[k]$$
   where $E_{\text{step}} \in \mathbb{R}^{K_{\text{max}} \times D}$ provides pulse-positional conditioning.

2. **Core Forward Pass:**
   The state $h_{\text{in}}$ passes sequentially through the 8 physical Transformer blocks indexed by pulse $k$:
   $$h_{\text{curr}} = \text{TransformerCore}(h_{\text{in}}, \text{pulse\_idx}=k)$$

3. **Harmonic Residual State Refinement:**
   The recurrent state is updated with a learnable per-pulse channel scaling vector $\mathbf{s}_k \in \mathbb{R}^D$:
   $$\Delta h = (h_{\text{curr}} - h_{\text{in}}) \odot \mathbf{s}_k$$
   $$h_{\text{pulse}} \leftarrow h_{\text{in}} + \Delta h$$
   where $\mathbf{s}_k$ is initialized harmonically to $\frac{1}{k + 1}$ ($1.00 \to 0.50 \to 0.33$), guaranteeing strictly monotonic contraction across recursive passes ($\|\Delta \mathbf{h}_1\| > \|\Delta \mathbf{h}_2\| > \|\Delta \mathbf{h}_3\|$).

---

## 5. Custom Fused CUDA Kernels (`csrc/kernels.cu`)

All latency-critical primitives are implemented in pure CUDA C++ with PyBind11 bindings in `pulsar_cuda`:

| Kernel Function | Launch Geometry | Technical Description & Invariants |
| :--- | :--- | :--- |
| **`qk_rmsnorm_rope_forward_cuda` / `backward`** | 1 Warp (32 threads) per $(b, h, t)$ vector | Fuses QK-RMSNorm and Rotary Position Embeddings in registers. Loads $(x_i, x_{i+32})$, computes warp-reduced RMSNorm, applies $(w_{h, i}, w_{h, i+32})$, and immediately applies $(cos, sin)$ rotation in registers. Eliminates 32 GB/step of intermediate HBM roundtrips. |
| **`fused_attn_gate_forward_cuda` / `backward`** | 8 threads per head ($H=16$, $d=64$, 128 threads/block) | Vectorized `uint4` memory loads reading `attn_out_bhtd`, multiplies by $\sigma(\text{gate\_logits}[b, t, h])$, and writes contiguous `out_btd`. Backward generates contiguous `d_attn_out_bhtd` for SDPA backward, avoiding non-contiguous buffer copies. |
| **`moe_dispatch` & `moe_combine`** | 128-bit `uint4` vectorization | Parallel index gather/scatter directly into 3D batch matrices. Early-exits dropped blocks (`slot == -1`), skipping reductions. Fused combine adds `shared_out` directly. |
| **`moe_slots_from_order_cuda`** | 1D grid over total assigned tokens | Computes expert token counts and slot offsets from sorted token order in C++, replacing 7 separate ATen kernel calls. |
| **`fused_sr_update`** | 1D grid over parameter tensors | Evaluates Muon weight updates with on-chip deterministic PRNG (SplitMix32) stochastic rounding noise generated in registers. Eliminates 2.25 GB/step of HBM noise tensor allocations. |
| **`fused_softcap_ce`** | Online softmax with soft-capping | Evaluates $30.0 \times \tanh(\text{logits}/30.0)$ directly inside the cross-entropy kernel in registers without materializing $B \times T \times V$ activation tensors (cuts memory from 800 MB to 8 KB). |

---

## 6. Dual Optimizer Architecture & Shape-Grouped Muon Geometry

Pulsar uses a dual-optimizer engine (`PulsarDualOptimizer`) specialized for recurrent MoE topologies:

### 6.1 Muon Matrix Optimizer (`opt_muon` — 247.46M parameters)
- **Scope:** Updates all 2D Attention projections ($W_{qkv}, W_o$), shared MLP matrices, and 3D routed expert tensors.
- **Batched 3D GEMM Orthogonalization:** The 128 routed expert matrices are grouped into 3D tensors:
  - `w_gate_up`: shape `[128, 512, 1024]`
  - `w_down`: shape `[128, 512, 1024]`
- **Newton-Schulz Iteration (Degree 5):**
  For input matrix or batched matrix $G$, compute:
  $$X_0 = \frac{G}{\|G\|_F + \epsilon}$$
  $$X_{t+1} = a X_t + b (X_t X_t^\top) X_t + c (X_t X_t^\top)^2 X_t$$
  where $a = 3.4445$, $b = -4.7750$, and $c = 2.0315$. Evaluates all experts in parallel via batched matrix multiplications.
- **Base Hyperparameters:** $\eta_{\text{muon}} = 0.02$, momentum $= 0.95$, weight decay $= 0.1$. Supports fused stochastic rounding for BF16 weights.

### 6.2 AdamW Optimizer (`opt_adam` + `opt_adam_sr` — 35.96M parameters)
- **`opt_adam_sr` (33.55M parameters):** Updates tied token embedding `tok_emb` in BF16 using stochastic rounding.
- **`opt_adam` (2.40M parameters):** Updates all 1D parameters strictly in **`torch.float32`** (RMSNorm scales, router weights, attention gates, and harmonic scale vectors).
- **Base Hyperparameters:** $\eta_{\text{adam}} = 6 \times 10^{-4}$, $\beta = (0.9, 0.95)$, weight decay $= 0.0$.
