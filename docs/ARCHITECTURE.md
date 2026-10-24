# Pulsar Architecture & Implementation Specification: 8-Layer Recurrent Sparse MoE with 32k Hybrid Attention

- **Model Name:** Pulsar
- **Architecture Type:** Recurrent Sparse Mixture-of-Experts with 32k Hybrid Attention
- **Target Parameter Count:** **299.68M Total Parameters** (`299,675,776`)
  - **Active Parameters per Token:** **147.92M parameters** (`97.59M` non-embedding active)
  - **Shared Parameters:** **98.35M parameters** (Embeddings, Attention, Shared Experts, Norms, Output Gates)
  - **Routed Expert Parameters:** **201.33M parameters** ($8 \text{ blocks} \times 8 \text{ experts} \times 3.15\text{M}$)
  - **Tied Token Embeddings:** **50.33M parameters** ($49,152 \times 1024$)
  - **Sparsity Ratio:** **50.6% computational sparsity**
- **Effective Reasoning Depth:** 24 Layers ($K=3$ recurrent passes across an 8-layer physical core)
- **Context Length:** **32,768 Tokens** (32k context via Hybrid Local/Global Attention)
- **Target Deployment:** NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM) / Single-GPU and Distributed nodes
- **Precision:** Pure BFloat16 compute for 2D/3D projection matrices, strictly preserved Float32 for 1D parameters, RMSNorm scales, router weights, attention gates, and harmonic scale vectors.

---

## 1. Executive Summary & Core Architectural Invariants

Pulsar couples **recurrent compute efficiency** with **sparse Mixture-of-Experts (MoE)**, **harmonic residual refinement**, and **32k hybrid local/global attention**:

1. **Recurrent Depth Without Weight Rigidity:** An 8-layer physical Transformer core is looped recurrently $K=3$ times (yielding an effective reasoning depth of 24 layers). Unlike traditional recurrent networks where early feature extraction and late-stage verification compete over identical MLP parameters, Pulsar uses **Pulse-Indexed Routing**: tokens dynamically dispatch to different experts depending on recursion depth.
2. **Pulse-Indexed 16-Head Attention Output Gate:** Evaluated immediately after SDPA and before $W_o$, a per-pulse, per-head gating vector eliminates attention-sink noise by allowing heads to throttle output mass smoothly toward $0.0$, operating with **zero FlashAttention latency penalty**.
3. **Harmonic Per-Pulse Residual Refinement Scale:** Resolves the representation overwrite dilemma where recurrent updates previously dwarfed the base state ($\|\Delta \mathbf{h}\| \approx 150$ vs. $\|\mathbf{h}_{\text{in}}\| \approx 32$). Updates are scaled by learnable FP32 vectors $\mathbf{s}_k$ initialized harmonically to $1 / (k + 1)$ ($1.00 \to 0.50 \to 0.33$), yielding strictly monotonic contraction ($\|\Delta \mathbf{h}_1\| = 40.0 > \|\Delta \mathbf{h}_2\| = 18.6 > \|\Delta \mathbf{h}_3\| = 12.4$).
4. **Per-Pulse Routed-Expert Scale:** Scales the Top-2 routed expert contribution relative to the Always-On Shared Expert per pulse and channel, dynamically balancing general vs. specialized knowledge.
5. **DeepSeek-V3 Auxiliary-Loss-Free Global Balancing:** The router uses an expert bias vector $b \in \mathbb{R}^8$ (or $[K, 8]$ per-pulse bias) to guide token distribution without adding artificial loss terms to the objective. Expert allocation counts are accumulated across micro-batches, allowing experts to specialize temporally (e.g. drafting vs. verification) while maintaining uniform hardware utilization.
6. **32k Hybrid Local/Global Attention Pattern:** 6 layers use high-precision local attention ($\theta = 10,000$, sliding window 1024), while 2 layers (Layers 3 & 7) provide global anchor attention ($\theta = 1,000,000$, full causal 32k span with YaRN temperature scaling). Across 3 recurrent pulses, the global layers execute 6 full 32k sweeps across sequence history.
7. **Fused Custom CUDA Kernels:** Critical bottlenecks (4-warp RMSNorm, QK-RMSNorm with grid-stride register accumulation, fused MoE Top-2 router + z-loss, zero-copy SwiGLU, and online softcap CE) are executed through hand-optimized CUDA kernels in `csrc/kernels.cu`, delivering sustained Tensor Core saturation.

---

## 2. Architecture Diagrams

### 2.1 Macro Recurrent Execution Diagram
Illustrates dataflow across the $K=3$ recurrent pulse cycle during pretraining:

```mermaid
flowchart TD
    subgraph InputStage ["1. Input & Hybrid RoPE Embedding"]
        TOK["Input Token IDs: x_t"]
        EMB["Tied Token Embedding [49,152 x 1024]"]
        ROPE_LOC["Local RoPE Cache (theta = 10k, 32k max)"]
        ROPE_GLOB["Global RoPE Cache (theta = 1M, 32k max)"]
        TOK --> EMB --> H0["Initial State: h_0"]
        TOK --> ROPE_LOC
        TOK --> ROPE_GLOB
    end

    subgraph RecurrentCore ["2. Recurrent 3-Pulse Engine (K = 3 Pulses, 8 Shared Blocks)"]
        H0 --> PASS1["Pulse 1 (Drafting: Layers 1-8)"]
        PASS1 --> PASS2["Pulse 2 (Relational Composition: Layers 9-16)"]
        PASS2 --> PASS3["Pulse 3 (Deep Verification: Layers 17-24)"]

        subgraph BlockDetail ["Inside Each Block (8 Physical Layers)"]
            NORM_ATTN["Per-Pulse Attn Norm: RMSNorm(h, weight[k])"]
            GQA["GQA Attention (16Q / 4KV) + QK-Norm + Value Skip (v_first)"]
            ATTN_GATE["16-Head Output Gate: sigmoid(W_gate[k]·x + b_gate[k])"]
            NORM_MLP["Per-Pulse MLP Norm: RMSNorm(h, weight[k])"]
            ROUTER["Fused Top-2 Router: Sigmoid + Expert Bias + Z-Loss"]
            SHARED_EXP["1 Shared SwiGLU Expert (1024 Dim, Always Active)"]
            ROUTED_EXP["8 Routed SwiGLU Experts (1024 Dim, Top-2 Selected)"]
            ROUTED_SCALE["Per-Pulse Routed Scale: routed_out · r_k"]
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

### 2.2 Micro Tensor Flow & Block Implementation Diagram
Illustrates low-level coordinate transformations, unit-normalized state mixing, slot dispatch, batched 3D GEMM, and loss head connections:

```mermaid
flowchart TD
    subgraph RecurrentStateEngine ["Recurrent State Loop (K = 3 Pulses)"]
        H0["Tied Embedding State: h_0"]
        HPULSE["Recurrent State: h_pulse"]
        
        NORM_A["RMSNorm(h_0; γ_anchor)"]
        NORM_P["RMSNorm(h_pulse; γ_pulse_state)"]
        MIX["Input State Mixing: (1 - λ_k)·h_pulse_norm + λ_k·h_0_norm + E_step[k]"]
        
        H0 --> NORM_A --> MIX
        HPULSE --> NORM_P --> MIX
        MIX --> BLOCK_ENTRY["Transformer Block Entry (h_in)"]
    end

    subgraph TransformerBlock ["Physical Transformer Block (Conditioned on Pulse Index k)"]
        BLOCK_ENTRY --> NORM_ATTN["Pre-Attn RMSNorm(h; γ_attn[k])"]
        
        subgraph GQASublayer ["1. Grouped-Query Attention (GQA)"]
            NORM_ATTN --> PROJ_QKV["Fused Linear: W_qkv [1024 -> 1536]"]
            PROJ_QKV --> SPLIT["Q [16 heads x 64], K [4 heads x 64], V [4 heads x 64]"]
            SPLIT --> QK_NORM["Per-Pulse, Per-Head QK-RMSNorm"]
            QK_NORM --> ROPE_HYBRID["Hybrid RoPE (Local theta=10k / Global theta=1M YaRN)"]
            SPLIT --> V_RES["First-Pulse Value Blend: (1 - λ_v)·V + λ_v·V_first"]
            ROPE_HYBRID --> FLASH_SDPA["Flash Causal SDPA (Scaled Dot-Product)"]
            V_RES --> FLASH_SDPA
            FLASH_SDPA --> ATTN_GATE["Pulse-Indexed 16-Head Output Gate: sigmoid(W_gate[k]·x + b_gate[k])"]
            ATTN_GATE --> PROJ_O["Output Linear: W_o [1024 -> 1024]"]
        end

        PROJ_O --> RES_ATTN["Attention Residual: h + Attn_Out"]
        RES_ATTN --> NORM_MLP["Pre-MLP RMSNorm(h; γ_mlp[k])"]

        subgraph MoESublayer ["2. Recurrent Sparse MoE Engine"]
            NORM_MLP --> SHARED["Shared Expert (Always Active, 1024 Dim SwiGLU)"]
            NORM_MLP --> ROUTER["Pulse-Indexed Router: Sigmoid(W_router[k]·x) + Expert Bias b"]
            ROUTER --> TOP2["Aux-Loss-Free Top-2 Selection & Unbiased Gating"]
            TOP2 --> DISPATCH["Fused Dispatch to Aligned Slots (C = 864)"]
            DISPATCH --> BATCH_GEMM1["Batched 3D GEMM Gate-Up: torch.bmm(E_in, W_gate_up)"]
            BATCH_GEMM1 --> FUSED_SWIGLU["Zero-Copy Fused SwiGLU Kernel"]
            FUSED_SWIGLU --> BATCH_GEMM2["Batched 3D GEMM Down: torch.bmm(Act, W_down)"]
            BATCH_GEMM2 --> COMBINE["Fused Combine & Accumulate Top-2"]
            COMBINE --> ROUTED_SCALE["Per-Pulse Routed Scale: routed_out · r_k"]
            SHARED --> MOE_OUT["Combined MLP Output"]
            ROUTED_SCALE --> MOE_OUT
        end

        MOE_OUT --> RES_MLP["Block Output: h_curr"]
    end

    RES_MLP --> RES_HARMONIC["Harmonic Residual Refinement: Δh = (h_curr - h_in) · s_k"]
    RES_HARMONIC --> STATE_ACCUM["h_pulse <- h_in + Δh"]
    STATE_ACCUM -.->|"Next Pulse (k+1)"| HPULSE

    STATE_ACCUM --> SUPERVISION["Subsampled Multi-Step Supervision"]
    subgraph MultiStepLoss ["Supervision & Soft-Capping Head"]
        SUPERVISION --> PULSE1_HEAD["Pulse 1: 25% Subsampled CE (weight = 0.15)"]
        SUPERVISION --> PULSE2_HEAD["Pulse 2: 25% Subsampled CE (weight = 0.25)"]
        SUPERVISION --> PULSE3_HEAD["Pulse 3: 100% Full Tokens CE (weight = 0.60)"]
        PULSE1_HEAD --> SOFTCAP["Fused Online Soft-Capped CE (30.0 * tanh)"]
        PULSE2_HEAD --> SOFTCAP
        PULSE3_HEAD --> SOFTCAP
    end
```

---

## 3. Parameter & Geometry Specification

### 3.1 Dimensions & Hyperparameters

| Dimension / Setting | Value | Description |
| :--- | :---: | :--- |
| **Vocabulary Size ($V$)** | `49,152` | Byte-Level BPE with strict single-digit splitting (`0-9`) |
| **Hidden Dimension ($d_{\text{model}}$)** | `1024` | Latent residual bus width (Pure Power-of-2) |
| **Physical Core Depth ($N_{\text{core}}$)** | `8` | Number of physical Transformer blocks |
| **Recurrent Passes ($K$)** | `3` (Default) | Effective reasoning depth: $8 \times 3 = \mathbf{24\text{ layers}}$ |
| **Max Supported Pulses** | `8` | Dynamically supports up to 64 effective layers |
| **Attention Query Heads ($H_Q$)** | `16` | Query dimension = $16 \times 64 = 1024$ |
| **Attention KV Heads ($H_{KV}$)** | `4` | Grouped-Query Attention (GQA, 4:1 ratio) |
| **Head Dimension ($d_{\text{head}}$)** | `64` | Strict dimension per attention head (warp-aligned) |
| **Attention Output Gating** | `16 Heads` | Pulse-indexed per-head gating vector ($\sigma(2.0) \approx 0.88$ init) |
| **Harmonic Residual Scaling** | `s_k ~ 1/(k+1)` | Channel-wise learnable scale per pulse ($1.0 \to 0.5 \to 0.33$) |
| **Expert FFN Dimension ($d_{\text{expert}}$)** | `1024` | SwiGLU expansion width ($1024 \to 2048 \to 1024$) |
| **Shared Experts per Block** | `1` | Always-active core reasoning expert |
| **Routed Experts per Block** | `8` | Dynamically selected specialized experts ($E0..E7$) |
| **Top-K Active Routed Experts** | `2` | Active routed experts selected per token |
| **Active FFN Width per Token** | `3072` | $1024\text{ (shared)} + 2 \times 1024\text{ (routed)}$ |
| **Context Length Support** | `32,768` | 32k context via hybrid local/global RoPE |
| **Local Attention Layers** | `0, 1, 2, 4, 5, 6` | $\theta = 10,000$, sliding window = 1024 |
| **Global Attention Layers** | `3, 7` | $\theta = 1,000,000$, full 32k causal span with YaRN |
| **Output Logit Soft-Capping** | `30.0` | $30.0 \times \tanh(\text{logits} / 30.0)$ |
| **Weight Tying** | `True` | `lm_head.weight = tok_emb.weight` |
| **MoE Capacity Factor** | `1.12` | Headroom factor (align 32, yielding $C=864$ slots at MBS=3) |

### 3.2 Exact Parameter Breakdown

1. **Tied Token Embeddings & LM Head:**
   - $49,152 \times 1024 = \mathbf{50,331,648\text{ parameters}}$ (`50.33M`)
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
   - **8 Routed Experts MLP (8 blocks $\times$ 8 experts = 64 routed experts):**
     - Each routed expert: $W_{\text{gate\_up}}$ $[1024 \to 2048]$ + $W_{\text{down}}$ $[1024 \to 1024]$ = `3,145,728` params
     - 64 experts $\times 3,145,728 = \mathbf{201,326,592\text{ parameters}}$ (`201.33M`)
   - **Per-Pulse Scale, Norm, and Router Banks (8 blocks):**
     - QK-RMSNorm Scales: $8 \times [8 \times (16 \times 64 + 4 \times 64)] = 81,920$ params
     - Value Skip Blend Scales: $8 \times 8 = 64$ params
     - Layer RMSNorm Banks: $8 \times [2 \times 8 \times 1024] = 131,072$ params
     - Pulse-Indexed Routers: $8 \times [8 \times 8 \times 1024] = 524,288$ params
     - Per-Pulse Routed Expert Scales: $8 \times [8 \times 1024] = 65,536$ params
     - Router Biases: $8 \times 8 = 64$ params
     - Subtotal: $\mathbf{802,944\text{ parameters}}$ (`~0.80M`)
3. **Conditioning & Output Norms:**
   - Harmonic Residual Scale (`pulse_res_scale` $[8 \times 1024]$): $8,192$ params
   - Pulse Step Embeddings: $8 \times 1024 = 8,192$ params
   - Unit-Normalized Pre-Mix Norms (`pulse_state_norm` + `anchor_norm`): $2 \times 1024 = 2,048$ params
   - Input Mix Logits (`input_mix_logit` $[8 \times 1024]$): $8,192$ params
   - Final RMSNorm: $1024$ params
   - Subtotal: $\mathbf{27,648\text{ parameters}}$
4. **Grand Total Parameters:**
   $$\mathbf{50,331,648 + 20,971,520 + 1,049,600 + 25,165,824 + 201,326,592 + 802,944 + 27,648 = 299,675,776\text{ params}}\quad (\mathbf{\approx 299.68M})$$

---

## 4. Mathematical Formulation & Recurrent State Dynamics

### 4.1 Input Stage & Embedding
For an input sequence of token IDs $X = (x_1, x_2, \dots, x_T) \in \mathbb{V}^T$, initial token embeddings are generated via tied embedding matrix $W_{\text{emb}} \in \mathbb{R}^{V \times D}$:
$$h_0 = \text{Embedding}(X; W_{\text{emb}}) \in \mathbb{R}^{B \times T \times D}$$
where $V = 49,152$ and $D = 1024$.

### 4.2 Recurrent Pulse Loop & Coordinate-Consistent State Accumulation
The physical 8-layer Transformer core is executed recurrently for $K = 3$ pulses ($k \in \{0, 1, 2\}$).

1. **Unit-Normalized Anchor & State Mixing:**
   Prior to layer entry, the prompt anchor $h_0$ and the recurrent state $h_{\text{pulse}}$ are normalized to unit variance to prevent coordinate scale mismatch:
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
   where $\mathbf{s}_k$ is initialized harmonically to $\frac{1}{k + 1}$ ($1.00 \to 0.50 \to 0.33$).
   *(Initial unscaled updates produced $\|\Delta \mathbf{h}\| \approx 150$ versus $\|\mathbf{h}_{\text{in}}\| \approx 32$, overwriting early representations. Harmonic scaling strictly contracts update norms across recursion: $\|\Delta \mathbf{h}_1\| = 40.0 \to \|\Delta \mathbf{h}_2\| = 18.6 \to \|\Delta \mathbf{h}_3\| = 12.4$, guaranteeing stable multi-pulse refinement).*

---

## 5. Block Sublayer Formulations

Each of the 8 physical blocks contains a Grouped-Query Attention (GQA) sublayer and a Sparse Mixture-of-Experts (MoE) sublayer, conditioned on pulse index $k$.

### 5.1 Grouped-Query Attention (GQA) with QK-Norm & Value Residual

1. **Pre-Attention Normalization:**
   $$x_{\text{attn}} = \text{RMSNorm}(h; \gamma_{\text{attn\_norm}}[k])$$

2. **Fused Projection ($W_{qkv} \in \mathbb{R}^{D \times (D_Q + 2 D_{KV})}$):**
   $$Q, K_{\text{raw}}, V_{\text{raw}} = x_{\text{attn}} W_{qkv}$$
   where $D_Q = 16 \times 64 = 1024$ and $D_{KV} = 4 \times 64 = 256$ ($4:1$ GQA ratio).

3. **Per-Pulse, Per-Head QK-RMSNorm:**
   Attention logits are bounded natively via Cauchy-Schwarz before rotary embeddings:
   $$Q_{\text{norm}} = \text{RMSNorm}(Q) \odot \gamma_Q[k, h_Q]$$
   $$K_{\text{norm}} = \text{RMSNorm}(K_{\text{raw}}) \odot \gamma_K[k, h_{KV}]$$
   where $\gamma_Q \in \mathbb{R}^{8 \times 16 \times 64}$ and $\gamma_K \in \mathbb{R}^{8 \times 4 \times 64}$ are FP32 trainable scale factors initialized to `1.0`. The CUDA kernel strictly enforces `head_dim == 64` for 32-lane warp reduction alignment.

4. **32k Hybrid Local / Global RoPE:**
   - **Local Layers ($l \in \{0, 1, 2, 4, 5, 6\}$):** Apply RoPE with base $\theta_{\text{local}} = 10,000.0$ evaluated within a sliding window of $1024$ tokens.
   - **Global Anchor Layers ($l \in \{3, 7\}$):** Apply RoPE with ultra-high wavelength base $\theta_{\text{global}} = 1,000,000.0$ ($1\text{M}$), attending over the full causal sequence span ($T \le 32,768$) with YaRN temperature scaling:
     $$Q_{\text{scaled}} = Q_{\text{norm}} \cdot \left(1.0 + 0.1 \cdot \ln\frac{T}{1024}\right) \quad \text{for } T > 1024$$
   - Across $K=3$ recurrent pulses, Layers 3 and 7 execute **6 full global causal sweeps** across sequence history.
   - Rotary inverse frequencies (`inv_freq`) are strictly computed and held in FP32 to prevent phase-drift artifacts.

5. **First-Pulse Value-Residual Skip ($v_{\text{first}}$):**
   - At Layer 0, Pulse 0: $V_{\text{first}} = V_{\text{raw}} = W_v \cdot x_{\text{attn}}$.
   - For all subsequent layers and pulses:
     $$\lambda_v = \sigma(\theta_v[k])$$
     $$V = (1 - \lambda_v) V_{\text{raw}} + \lambda_v V_{\text{first}}$$
     where $\theta_v$ is initialized to `-2.2` ($\lambda_v \approx 0.10$).

6. **Flash Causal Attention & Pulse-Indexed 16-Head Output Gate:**
   $$O = \text{SDPA}(Q, K, V, \text{is\_causal}=\text{True}) \in \mathbb{R}^{B \times H_Q \times T \times d_{\text{head}}}$$
   After computing attention via SDPA, each of the 16 attention heads is dynamically gated per pulse:
   $$\mathbf{g}_k = \sigma(x_{\text{attn}} W_{\text{gate}}[k]^\top + b_{\text{gate}}[k]) \in \mathbb{R}^{B \times T \times 16}$$
   where $W_{\text{gate}} \in \mathbb{R}^{K_{\text{max}} \times 16 \times D}$ and $b_{\text{gate}} \in \mathbb{R}^{K_{\text{max}} \times 16}$ ($b_{\text{gate}}$ initialized to `2.0` so $\sigma(2.0) \approx 0.88$, starting all heads open).
   $$O_{\text{gated}}[:, h, :, :] = O[:, h, :, :] \cdot \mathbf{g}_k[:, :, h, \text{None}]$$
   $$h \leftarrow h + \text{reshape}(O_{\text{gated}}) W_o \quad (W_o \in \mathbb{R}^{D \times D})$$
   *(Eliminates Softmax attention-sink noise by allowing heads to throttle output mass smoothly toward $0.0$, operating with zero FlashAttention kernel latency penalty).*

---

### 5.2 Sparse Mixture-of-Experts (MoE) Sublayer

1. **Pre-MLP Normalization:**
   $$x_{\text{mlp}} = \text{RMSNorm}(h; \gamma_{\text{mlp\_norm}}[k])$$

2. **Pulse-Indexed Router & DeepSeek-V3 Auxiliary-Loss-Free Selection:**
   $$\text{logits} = x_{\text{mlp}} W_{\text{router}}[k] \in \mathbb{R}^{N \times 8}$$
   $$s = \sigma(\text{logits}) \in \mathbb{R}^{N \times 8}$$
   $$\text{Top-}2 = \operatorname{argtop2}(s + b)$$
   where $b \in \mathbb{R}^8$ is the adaptive bias vector updated via:
   $$b_i \leftarrow b_i + 0.001 \cdot \operatorname{sign}(\bar{C} - C_i), \quad b_i \in [-0.5, 0.5]$$
   Unbiased weights are normalized across the selected Top-2:
   $$w_i = \frac{s_i}{\sum_{j \in \text{Top-}2} s_j}, \quad i \in \text{Top-}2$$
   Expert assignments are accumulated globally across all micro-batches and pulses (or per-pulse via `per_pulse_bias`), allowing temporal specialization (e.g., drafting vs. verification) without hardware imbalance.

3. **1 Shared Expert (Always Active):**
   $$\text{shared\_out} = \text{SwiGLU}(x_{\text{mlp}} W_{\text{sh\_gu}}) W_{\text{sh\_down}}$$
   where $W_{\text{sh\_gu}} \in \mathbb{R}^{1024 \times 2048}$ and $W_{\text{sh\_down}} \in \mathbb{R}^{1024 \times 1024}$.

4. **8 Routed Experts via Batched 3D GEMM:**
   Tokens are dispatched into expert slots with aligned capacity ($C = 864$ slots at $MBS=3$, based on `moe_capacity_factor=1.12` and `moe_capacity_align=32`):
   $$\text{gu} = \text{torch.bmm}(\text{expert\_inputs}, W_{\text{routed\_gu}}) \in \mathbb{R}^{8 \times C \times 2048}$$
   $$\text{act} = \text{FusedSwiGLU}(\text{gu}, d_{\text{ff}}=1024) \in \mathbb{R}^{8 \times C \times 1024}$$
   $$\text{exp\_out} = \text{torch.bmm}(\text{act}, W_{\text{routed\_down}}) \in \mathbb{R}^{8 \times C \times 1024}$$

5. **Per-Pulse Routed Expert Scaling & Fused Output Combination:**
   Routed expert outputs are scaled channel-wise by a learnable per-pulse scale vector $\mathbf{r}_k \in \mathbb{R}^D$ (initialized to `1.0`), then combined with the shared expert residual:
   $$y = \text{SharedExpert}(x_{\text{mlp}}) + \left(\sum_{j \in \text{Top-}2} w_j \cdot \text{exp\_out}_j\right) \odot \mathbf{r}_k$$
   $$h \leftarrow h + y$$

---

## 6. Custom Fused CUDA Kernels (`csrc/kernels.cu`)

All latency-critical primitives are implemented in pure CUDA C++ with PyBind11 bindings in `pulsar_cuda`:

| Kernel Function | Launch Geometry | Optimizations & Invariants |
| :--- | :--- | :--- |
| **`rmsnorm_forward_vec8_kernel`** | `threads = 128`, `blocks = N` | Exact 4-warp alignment ($1024 / 8 = 128$). Vectorized 128-bit `uint4` loads. |
| **`rmsnorm_backward_vec8_kernel`** | `threads = 128`, `blocks = N` | Thread-register accumulation across rows, single atomic write at exit. |
| **`qk_rmsnorm_forward_kernel`** | `threads = 64`, `blocks = (N, H)` | Warp-specialized for $d_{\text{head}} = 64$ ($2 \times 32$ threads). Zero idle lanes. Enforces strict `head_dim == 64` contract. |
| **`qk_rmsnorm_backward_kernel`** | Grid-stride 2D loop | Thread registers accumulate $dw_Q$ and $dw_K$ across all tokens. Eliminates atomic contention (**10.5x speedup**). |
| **`moe_router_topk_kernel`** | `threads = 128`, `blocks = (N + 127)/128` | Fused evaluation of `sigmoid(logits)`, bias addition, Top-2 selection, unbiased normalization, and row `logsumexp` z-loss in registers. |
| **`fused_swiglu_forward/bwd`** | `threads = 256`, 1D grid | Direct strided reads/writes on contiguous `[..., 2048]` tensors. Zero `.chunk(2)` and zero `torch.cat`. |
| **`moe_dispatch / combine`** | Aligned capacity kernels | Parallel index gather/scatter directly into 3D batch matrices. Derives drop telemetry from true unassigned slots (`slot == -1`). |
| **`fused_softcap_ce`** | Online softmax with soft-capping | Evaluates $30.0 \times \tanh(\text{logits}/30.0)$ directly inside the cross-entropy kernel without allocating huge $B \times T \times V$ FP32 activation tensors. |

---

## 7. Multi-Step Deep Supervision with Rotating Subsampling

Pulsar employs deep supervision across all $K=3$ recurrent passes during pretraining:

$$\mathcal{L}_{\text{total}} = 0.15 \cdot \mathcal{L}_{\text{pulse1}} + 0.25 \cdot \mathcal{L}_{\text{pulse2}} + 0.60 \cdot \mathcal{L}_{\text{pulse3}} + \mathcal{L}_z$$

- **Intermediate Pulses ($k \in \{0, 1\}$):** Subsamples 25% of tokens with rotating disjoint offsets:
  $$\text{offset} = (\text{subsample\_offset} + k) \pmod 4$$
  $$h_{\text{sub}} = h_{\text{pulse}}[:, \text{offset}::4, :]$$
  Reduces vocabulary projection GEMMs ($[N, 1024] \times [1024, 49152]$) and cross-entropy allocations by **75%** on Pulses 1 & 2 while preventing intermediate representation collapse.
- **Final Pulse ($k = 2$):** Evaluates 100% of tokens to ensure global pretraining loss calibration.
- **Logit Soft-Capping:** All logits are bounded to $[-30.0, 30.0]$ via $30.0 \times \tanh(\text{logits} / 30.0)$, preventing entropy collapse and logit drifting.

---

## 8. Dual Optimizer Architecture & Shape-Grouped Muon Geometry

Pulsar uses a dual-optimizer engine (`PulsarDualOptimizer`) specialized for recurrent MoE topologies:

### 8.1 Muon Matrix Optimizer (`opt_muon` — 247.46M parameters)
- **Scope:** Updates all 2D Attention projections ($W_{qkv}, W_o$), shared MLP matrices, and routed expert matrices.
- **Batched 3D GEMM Orthogonalization:** The 64 routed expert matrices are grouped into two 3D tensors:
  - `w_gate_up`: shape `[64, 1024, 2048]`
  - `w_down`: shape `[64, 1024, 1024]`
- **Newton-Schulz Iteration (Degree 5):**
  For input matrix or batched matrix $G$, compute:
  $$X_0 = \frac{G}{\|G\|_F + \epsilon}$$
  $$X_{t+1} = a X_t + b (X_t X_t^\top) X_t + c (X_t X_t^\top)^2 X_t$$
  where $a = 3.4445$, $b = -4.7750$, and $c = 2.0315$. Evaluates all 64 experts in parallel via batched matrix multiplications, eliminating 64 individual kernel launch overheads.
- **Layout-Aware Scale Convention:** Updates are scaled by $\max(1, \text{rows}/\text{cols})^{0.5}$. The optimizer accounts for the `[E, in, out]` storage layout of expert 3D tensors, ensuring `w_gate_up` receives the correct $\sqrt{2048/1024} = 1.414$ scaling.
- **Base Hyperparameters:** $\eta_{\text{muon}} = 0.02$, momentum $= 0.95$, weight decay $= 0.1$. Supports stochastic rounding for BF16 weights and optional FP32 momentum buffers.

### 8.2 AdamW Optimizer (`opt_adam` — 52.21M parameters)
- **Scope:** Updates `tok_emb` (50.33M), `pulse_step_emb`, `router_weight` (0.52M), `input_mix_logit`, `attn_gate` (1.05M), `attn_gate_bias`, `pulse_res_scale`, `routed_scale`, and all RMSNorm scale parameters.
- **Strict FP32 Preservation:** All 1D/norm parameters, router weights, attention gates/biases, and per-pulse scales are strictly preserved in **`torch.float32`** throughout forward, backward, and optimizer updates to guarantee numerical stability.
- **Base Hyperparameters:** $\eta_{\text{adam}} = 6 \times 10^{-4}$, $\beta = (0.9, 0.95)$, weight decay $= 0.0$, fused CUDA kernel enabled.
