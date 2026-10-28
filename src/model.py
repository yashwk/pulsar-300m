import json
import math
import os
import threading
import warnings
from dataclasses import asdict, dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PulsarConfig:
    vocab_size: int = 32768
    d_model: int = 1024
    n_heads: int = 16
    n_kv_heads: int = 4
    head_dim: int = 64
    d_expert: int = 512
    d_shared: int | None = None
    n_core: int = 8  # 8 core layers recurrently looped K times
    max_pulses: int = 8
    default_pulses: int = 3
    base_seq_len: int = 1024  # Base pretraining context window
    max_seq_len: int = 32768  # 32k context support
    rope_theta_local: float = 10000.0  # 10k theta for 6 local high-precision layers
    rope_theta_global: float = 1000000.0  # 1M theta for 2 global 32k anchor layers
    sliding_window: int = 1024  # Local sliding window size
    # Local-attention implementation: "auto" picks the best available (flex_attention
    # with a cached BlockMask, else sdpa_mask). "sdpa_mask" builds a dense boolean mask
    # and is only sane for small T - it is kept as the reference implementation for
    # tests, not for production long context.
    local_attn_impl: str = "auto"
    n_routed_experts: int = 16
    top_k: int = 4
    eos_token_id: int = 0
    rms_eps: float = 1e-6
    ce_chunk_size: int = 4096
    # Divided by n_core*K (=24) when applied, so the effective weight is 4.2e-5.
    # Raised from 1e-4 because centring the logits and slowing the router reduced the
    # only remaining regulariser on router magnitude.
    router_z_loss_weight: float = 1e-3
    aux_loss_weight_1: float = 0.15
    aux_loss_weight_2: float = 0.25
    final_loss_weight: float = 0.60
    logit_softcap: float = 30.0
    # Expert capacity headroom: capacity = ceil(avg_load * factor), rounded up to
    # moe_capacity_align. 1.08 (the old hardcoded value) left only 8.3% headroom, so
    # drops were common and invisible. 1.12 with 32-alignment gives 864 slots at
    # N=3072 (vs 832 at 1.08/64). Capacity is a runtime setting: it changes no
    # checkpoint format, so it can be raised mid-run if the logged drop rate is high.
    moe_capacity_factor: float = 1.25
    moe_capacity_align: int = 32
    # Per-pulse balancing bias. Drops are a per-pulse phenomenon (early pulses route
    # differently from late ones), so a single across-pulse bias can leave one pulse
    # systematically over-loaded. Default off: it changes training dynamics, and old
    # checkpoints carry a [E] bias which is migrated by broadcasting.
    # Per-pulse balancing bias. The three pulses route differently (measured at
    # step 80: P1 dropped 5.4% of assignments while P2/P3 dropped 23.6%/23.7%), so a
    # single global bias vector is diluted by whichever pulse needs no correction.
    # Migration for old [E] checkpoints is handled in _migrate_checkpoint_bias().
    per_pulse_bias: bool = True

    def __post_init__(self):
        if self.d_shared is None:
            self.d_shared = 1024 if self.d_model >= 1024 else self.d_expert

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict, *, strict: bool = False) -> "PulsarConfig":
        """Build a config from a dict, reporting keys this class does not define.

        Silently dropping unknown keys let a typo'd or stale key carry an intention
        that never took effect - the same failure mode as a flag that is accepted and
        ignored. Unknown keys are now always reported; ``strict=True`` raises, and the
        default warns and continues so loading older checkpoints stays possible.
        """
        valid_keys = set(cls.__dataclass_fields__.keys())
        unknown = sorted(set(d) - valid_keys)
        if unknown:
            msg = (
                f"unknown PulsarConfig key(s): {', '.join(unknown)}. "
                f"These were previously ignored silently; the value is NOT in effect."
            )
            if strict:
                raise ValueError(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "PulsarConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


try:
    import pulsar_cuda
    HAS_PULSAR_CUDA = True
except ImportError:
    HAS_PULSAR_CUDA = False


class _RMSNormCUDAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        weight_f = weight.float().contiguous()
        y, rsqrt = pulsar_cuda.rmsnorm_forward(x.contiguous(), weight_f, eps)
        ctx.save_for_backward(x, weight, rsqrt)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, rsqrt = ctx.saved_tensors
        weight_f = weight.float().contiguous()
        dx, dweight = pulsar_cuda.rmsnorm_backward(dy.contiguous(), x.contiguous(), weight_f, rsqrt)
        return dx, dweight.to(weight.dtype), None


class _FusedQK_RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, T: int, eps: float):
        weight_f = weight.float().contiguous()
        y, rsqrt = pulsar_cuda.qk_rmsnorm_forward(x.contiguous(), weight_f, T, eps)
        ctx.save_for_backward(x, weight, rsqrt)
        ctx.T = T
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, weight, rsqrt = ctx.saved_tensors
        weight_f = weight.float().contiguous()
        dx, dweight = pulsar_cuda.qk_rmsnorm_backward(dy.contiguous(), x.contiguous(), weight_f, rsqrt, ctx.T)
        return dx, dweight.to(weight.dtype), None, None


class _FusedSoftcapCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        targets: torch.Tensor,
        softcap: float = 30.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        N = logits.size(0)
        losses, lse = pulsar_cuda.fused_softcap_ce_forward(logits.contiguous(), targets.contiguous(), softcap, ignore_index)
        ctx.save_for_backward(logits, targets, lse)
        ctx.softcap = softcap
        ctx.reduction = reduction
        ctx.ignore_index = ignore_index
        ctx.N = N

        if reduction == "mean":
            # Keep n_valid on the device. .item() here forced a host sync on every
            # micro-batch (43 per optimizer step).
            n_valid = (targets != ignore_index).sum().clamp_min(1).to(torch.float32)
            ctx.n_valid = n_valid
            return losses.sum() / n_valid
        elif reduction == "sum":
            ctx.n_valid = None
            return losses.sum()
        ctx.n_valid = None
        return losses

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        logits, targets, lse = ctx.saved_tensors
        # grad_output / n_valid stays a device tensor so the kernel reads the scale
        # without a host round-trip; the scalar variant is kept for the sum path.
        if ctx.n_valid is not None:
            scale_dev = (grad_output.reshape(()) / ctx.n_valid).to(torch.float32).contiguous()
            d_logits = pulsar_cuda.fused_softcap_ce_backward_dev(
                logits.contiguous(), targets.contiguous(), lse.contiguous(),
                ctx.softcap, scale_dev, ctx.ignore_index,
            )
        else:
            # "sum" reduction: a grad_output of 1.0 is the norm, but pass it as a device
            # tensor anyway so a non-trivial upstream gradient cannot force a sync.
            scale_dev = grad_output.reshape(()).to(torch.float32).contiguous()
            d_logits = pulsar_cuda.fused_softcap_ce_backward_dev(
                logits.contiguous(), targets.contiguous(), lse.contiguous(),
                ctx.softcap, scale_dev, ctx.ignore_index,
            )
        return d_logits, None, None, None, None


def _softcap_cross_entropy(
    raw_logits: torch.Tensor,
    targets: torch.Tensor,
    softcap: float,
    reduction: str = "sum",
    ignore_index: int = -100,
) -> torch.Tensor:
    """Fused softcap + cross-entropy helper.

    When compiled by TorchInductor, fuses softcap tanh and log_softmax into a single Triton
    kernel, avoiding materialization of [N, V] softcapped logits in HBM.
    """
    sc = softcap * torch.tanh(raw_logits.float() / softcap)
    return F.cross_entropy(sc, targets, ignore_index=ignore_index, reduction=reduction)


try:
    _compiled_softcap_ce = torch.compile(_softcap_cross_entropy)
except Exception:
    _compiled_softcap_ce = _softcap_cross_entropy


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, max_pulses: int | None = None):
        super().__init__()
        self.eps = eps
        self.max_pulses = max_pulses
        if max_pulses is not None and max_pulses > 1:
            # Per-pulse FP32 scale weights
            self.weight = nn.Parameter(torch.ones(max_pulses, dim, dtype=torch.float32))
        else:
            self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor, pulse_idx: int = 0) -> torch.Tensor:
        w = self.weight[pulse_idx] if (self.max_pulses is not None and self.max_pulses > 1) else self.weight
        if HAS_PULSAR_CUDA and x.is_cuda and x.dtype == torch.bfloat16 and not torch.compiler.is_compiling():
            return _RMSNormCUDAFunction.apply(x, w, self.eps)
        norm = x.float().pow(2).mean(-1, keepdim=True)
        x_normed = x.float() * torch.rsqrt(norm + self.eps)
        return (w.float() * x_normed).to(x.dtype)


# Set while torch.utils.checkpoint is recomputing a block during backward. The MoE
# forward mutates counters (expert_counts_accum, drop stats); a recomputation must not
# apply those side effects twice.
class _RecomputeState(threading.local):
    active = False


_RECOMPUTING = _RecomputeState()


class _RecomputeContext:
    """Context manager that sets the recomputation flag for its duration."""

    def __init__(self, flag: bool):
        self.flag = flag
        self.prev = False

    def __enter__(self):
        self.prev = _RECOMPUTING.active
        _RECOMPUTING.active = self.flag
        return self

    def __exit__(self, *exc):
        _RECOMPUTING.active = self.prev
        return False


def _set_recomputing(flag: bool) -> _RecomputeContext:
    return _RecomputeContext(flag)


# ---------------------------------------------------------------------------
# Local (sliding-window) attention implementations
# ---------------------------------------------------------------------------

# Cached BlockMask per (B, H, T, window, device) - building one costs ~100 ms.
_BLOCK_MASK_CACHE: dict[tuple, object] = {}
_FLEX_ATTENTION = None
# Disable flex_attention globally (e.g. for the reference comparison in tests).
flex_enabled = True


def _flex_attention_available() -> bool:
    global _FLEX_ATTENTION
    if _FLEX_ATTENTION is not None:
        return _FLEX_ATTENTION
    try:
        from torch.nn.attention.flex_attention import flex_attention  # noqa: F401
        _FLEX_ATTENTION = True
    except Exception:
        _FLEX_ATTENTION = False
    return _FLEX_ATTENTION


def _get_local_block_mask(B: int, H: int, T: int, window: int, device) -> object:
    """Cached BlockMask for the causal sliding-window pattern.

    The mask depends only on (q_idx, kv_idx), so it is built with B=None, H=None:
    passing real B and H replicates identical metadata B*H times (48x for B=3, H=16).
    """
    key = (T, window, str(device))
    cached = _BLOCK_MASK_CACHE.get(key)
    if cached is not None:
        return cached

    from torch.nn.attention.flex_attention import create_block_mask

    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & ((q_idx - kv_idx) < window)

    # _compile=True is REQUIRED, not an optimisation. With the default (_compile=False)
    # this torch version builds a full T x T mask tensor internally just to derive the
    # block metadata - 8 GB at T=32768 - which defeats the entire point of using
    # flex_attention. The compiled path computes block counts directly.
    block_mask = create_block_mask(mask_mod, None, None, T, T, device=device, _compile=True)

    # Bound the cache: T can vary per stage, and an unbounded dict grows forever.
    if len(_BLOCK_MASK_CACHE) >= 8:
        _BLOCK_MASK_CACHE.pop(next(iter(_BLOCK_MASK_CACHE)))
    _BLOCK_MASK_CACHE[key] = block_mask
    return block_mask


_FLEX_COMPILED = None


def _compiled_flex_attention():
    """flex_attention wrapped in torch.compile.

    Uncompiled flex_attention warns that it "materialises the full scores matrix", which
    defeats the purpose of using it. Compiled, it generates a fused kernel and never
    allocates T x T.
    """
    global _FLEX_COMPILED
    if _FLEX_COMPILED is None:
        from torch.nn.attention.flex_attention import flex_attention

        try:
            _FLEX_COMPILED = torch.compile(flex_attention, dynamic=False)
        except Exception:  # pragma: no cover - compile unavailable
            _FLEX_COMPILED = flex_attention
    return _FLEX_COMPILED


def _local_attn_flex(q, k, v, window: int, enable_gqa: bool):
    """Sliding-window attention via flex_attention (no dense mask, no T x T allocation).

    This torch build's flex_attention does not support GQA, so K/V are expanded with
    repeat_interleave when head counts differ. K/V are the small side (n_kv_heads), so
    the expansion is cheap relative to the query side.
    """
    B, H, T, D = q.shape
    KVH = k.shape[1]
    if enable_gqa and KVH != H:
        rep = H // KVH
        k = torch.repeat_interleave(k, rep, dim=1)
        v = torch.repeat_interleave(v, rep, dim=1)

    block_mask = _get_local_block_mask(B, H, T, window, q.device)
    return _compiled_flex_attention()(q, k, v, block_mask=block_mask)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 32768, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_seq_len)

    def _inv_freq_fp32(self, device=None) -> torch.Tensor:
        """Recompute inv_freq in FP32 from (theta, dim).

        The registered buffer is not authoritative after a bf16 cast: rounding it then
        up-casting does not recover the original frequencies.
        """
        dev = device if device is not None else self.inv_freq.device
        return 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).to(dev).float() / self.dim))

    def _set_cos_sin_cache(self, seq_len: int, device=None, dtype=None):
        self.max_seq_len = seq_len
        dev = device if device is not None else self.inv_freq.device
        dt = dtype if dtype is not None else torch.float32
        t = torch.arange(seq_len, dtype=torch.float32, device=dev)
        freqs = torch.outer(t, self.inv_freq.to(dev))
        self.register_buffer("cos_cached", freqs.cos().to(dt), persistent=False)
        self.register_buffer("sin_cached", freqs.sin().to(dt), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len, device=x.device, dtype=torch.float32)
        target_dtype = torch.float32 if HAS_PULSAR_CUDA else x.dtype
        return (
            self.cos_cached[:seq_len].to(device=x.device, dtype=target_dtype),
            self.sin_cached[:seq_len].to(device=x.device, dtype=target_dtype),
        )


class _RoPECUDAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin):
        ctx.save_for_backward(cos, sin)
        return pulsar_cuda.rope_forward(x.contiguous(), cos, sin)

    @staticmethod
    def backward(ctx, dy):
        cos, sin = ctx.saved_tensors
        # Orthogonal transpose rotation: R^T(theta) = R(-theta)
        dx = pulsar_cuda.rope_forward(dy.contiguous(), cos, -sin)
        return dx, None, None


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if HAS_PULSAR_CUDA and x.is_cuda and x.dtype == torch.bfloat16 and not torch.compiler.is_compiling():
        return _RoPECUDAFunction.apply(x, cos, sin)
    d = x.shape[-1] // 2
    # FP32 intermediates: rope_forward_kernel rotates in FP32 with FP32 cos/sin tables
    # and casts to bf16 at the end. Downcasting the tables to bf16 first made the
    # fallback (and every torch.compile / CPU / FP32 run) drift from the kernel.
    cos_f = cos.unsqueeze(0).unsqueeze(0).float()
    sin_f = sin.unsqueeze(0).unsqueeze(0).float()
    xf = x.float()
    x1, x2 = xf[..., :d], xf[..., d:]
    return torch.cat([x1 * cos_f - x2 * sin_f, x2 * cos_f + x1 * sin_f], dim=-1).to(x.dtype)


class PulsarAttention(nn.Module):
    def __init__(self, cfg: PulsarConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.num_queries_per_kv = cfg.n_heads // cfg.n_kv_heads
        self.is_local = (layer_idx in {0, 1, 2, 4, 5, 6})

        # Fused wqkv: [1024 -> 1536] (1024 Q + 256 K + 256 V)
        q_dim = cfg.n_heads * cfg.head_dim
        kv_dim = cfg.n_kv_heads * cfg.head_dim
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.wqkv = nn.Linear(cfg.d_model, q_dim + 2 * kv_dim, bias=False)
        self.wo = nn.Linear(q_dim, cfg.d_model, bias=False)
        self.wo.is_residual_out = True

        # Per-Pulse, Per-Head QK-RMSNorm weights in FP32
        self.q_norm_weight = nn.Parameter(torch.ones(cfg.max_pulses, cfg.n_heads, cfg.head_dim, dtype=torch.float32))
        self.k_norm_weight = nn.Parameter(torch.ones(cfg.max_pulses, cfg.n_kv_heads, cfg.head_dim, dtype=torch.float32))

        # Value-residual blending factor initialized to -2.2 (sigmoid(-2.2) ≈ 0.10)
        self.v_resid_lambda = nn.Parameter(torch.full((cfg.max_pulses,), -2.2, dtype=torch.float32))

        # Pulse-Indexed 16-Head Attention Output Gate (starts open at sigmoid(2.0) ≈ 0.88)
        self.attn_gate = nn.Parameter(torch.zeros(cfg.max_pulses, cfg.n_heads, cfg.d_model, dtype=torch.float32))
        self.attn_gate_bias = nn.Parameter(torch.full((cfg.max_pulses, cfg.n_heads), 2.0, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        pulse_idx: int = 0,
        v_first: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T, _ = x.shape
        qkv = self.wqkv(x)
        q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Value-Residual skip connection
        if self.layer_idx == 0 and pulse_idx == 0:
            v_first = v
        elif v_first is not None:
            lv = torch.sigmoid(self.v_resid_lambda[pulse_idx]).to(v.dtype)
            v = (1.0 - lv) * v + lv * v_first

        # Per-Pulse, Per-Head QK-RMSNorm (Fused single-warp reduction CUDA kernel).
        # The kernel is written for head_dim == 64: each of the 32 lanes handles one
        # (even, odd) pair, i.e. exactly 64 elements. It does NOT validate head_dim, so
        # any other value makes the upper lanes read and write past the end of the
        # vector (intermittent NaNs from garbage entering the warp reduction). Guard
        # here and fall back to the PyTorch path for other head dims.
        if (HAS_PULSAR_CUDA and self.head_dim == 64 and q.is_cuda
                and q.dtype == torch.bfloat16 and not torch.compiler.is_compiling()
                and hasattr(pulsar_cuda, "qk_rmsnorm_forward")):
            q = _FusedQK_RMSNormFunction.apply(q, self.q_norm_weight[pulse_idx], T, self.cfg.rms_eps)
            k = _FusedQK_RMSNormFunction.apply(k, self.k_norm_weight[pulse_idx], T, self.cfg.rms_eps)
        else:
            q_rms = torch.rsqrt(q.float().pow(2).mean(-1, keepdim=True) + self.cfg.rms_eps)
            scale_q = self.q_norm_weight[pulse_idx].view(1, self.n_heads, 1, self.head_dim)
            q = (scale_q.float() * (q.float() * q_rms)).to(q.dtype)

            k_rms = torch.rsqrt(k.float().pow(2).mean(-1, keepdim=True) + self.cfg.rms_eps)
            scale_k = self.k_norm_weight[pulse_idx].view(1, self.n_kv_heads, 1, self.head_dim)
            k = (scale_k.float() * (k.float() * k_rms)).to(k.dtype)

        # Apply RoPE
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Global layers (3 & 7): YaRN-style temperature scaling. The scale is a function
        # of each QUERY POSITION, not of the call's total length:
        #     scale[t] = 1 + 0.1 * ln(max(t + 1, base) / base)   (1.0 for t < base)
        # Using T made a query's scaling depend on how many tokens followed it, so the
        # model was not prefix-invariant and a KV cache would have been incorrect.
        if not self.is_local and T > self.cfg.base_seq_len:
            base = self.cfg.base_seq_len
            pos = torch.arange(1, T + 1, device=q.device, dtype=torch.float32)
            scale = 1.0 + 0.1 * torch.log(pos.clamp_min(base) / base)
            q = q * scale.view(1, 1, T, 1).to(q.dtype)

        enable_gqa = (self.num_queries_per_kv > 1)
        if self.is_local and T > self.cfg.sliding_window:
            # Sliding window. The old implementation materialised T x T int64 index
            # tensors (8.6 GB at T=32768) and a boolean mask (1.07 GB) just to express
            # "causal, within `window`" - impossible on the target 8 GB card, and it
            # forced SDPA onto the math backend.
            impl = self.cfg.local_attn_impl
            if impl == "auto":
                impl = "flex" if (flex_enabled and _flex_attention_available()) else "sdpa_mask"
            if impl == "flex":
                out = _local_attn_flex(q, k, v, self.cfg.sliding_window, enable_gqa)
            elif impl == "flash_attn":
                raise RuntimeError("flash_attn backend requested but not installed")
            else:
                # Reference path: correct but O(T^2) memory. Small T / tests only.
                col_idx = torch.arange(T, device=x.device).unsqueeze(0)
                row_idx = torch.arange(T, device=x.device).unsqueeze(1)
                diff = row_idx - col_idx
                mask = (diff >= 0) & (diff < self.cfg.sliding_window)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

        # Pulse-Indexed 16-Head Output Gate
        gate_w = self.attn_gate[pulse_idx].to(x.dtype)
        gate_b = self.attn_gate_bias[pulse_idx].to(x.dtype)
        g = torch.sigmoid(F.linear(x, gate_w, gate_b))  # [B, T, n_heads]
        out = out * g.transpose(1, 2).unsqueeze(-1)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out), v_first


class _FusedSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gu, d_ff):
        ctx.save_for_backward(gu)
        ctx.d_ff = d_ff
        if HAS_PULSAR_CUDA and gu.is_cuda and gu.dtype == torch.bfloat16 and not torch.compiler.is_compiling() and hasattr(pulsar_cuda, "fused_swiglu_forward"):
            return pulsar_cuda.fused_swiglu_forward(gu.contiguous(), d_ff)
        gate = gu[..., :d_ff]
        up = gu[..., d_ff:]
        # FP32 intermediates, matching fused_swiglu_forward_kernel and the backward
        # fallback below. Computing in bf16 here made the reference path drift from the
        # kernel it is supposed to validate (and from its own gradient).
        return (F.silu(gate.float()) * up.float()).to(gu.dtype)

    @staticmethod
    def backward(ctx, dout):
        gu, = ctx.saved_tensors
        d_ff = ctx.d_ff
        if HAS_PULSAR_CUDA and dout.is_cuda and dout.dtype == torch.bfloat16 and not torch.compiler.is_compiling() and hasattr(pulsar_cuda, "fused_swiglu_backward"):
            return pulsar_cuda.fused_swiglu_backward(dout.contiguous(), gu.contiguous(), d_ff), None
        gate = gu[..., :d_ff]
        up = gu[..., d_ff:]
        sig = torch.sigmoid(gate.float())
        silu = gate.float() * sig
        dgate = dout.float() * up.float() * (silu + sig * (1.0 - silu))
        dup = dout.float() * silu
        return torch.cat([dgate.to(gu.dtype), dup.to(gu.dtype)], dim=-1), None


def compute_token_slots(
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    capacity: int,
) -> torch.Tensor:
    """Assign each (token, k) assignment to a slot, deterministically.

    The kernel used to assign slots with ``atomicAdd``, so which assignments overflowed
    depended on block scheduling: nondeterministic run to run, and biased toward tokens
    scheduled later. Slots are assigned here instead, deterministically:

    * Fast path (no overflow - the normal case with capacity factor 1.25): slots are
      assigned in token order per expert via a cumulative offset. No sorting.
    * Slow path (an expert overflows capacity): assignments are ordered by descending
      routing weight, so only the **lowest-weight** ones are dropped, identically on
      every run.

    This runs once per MoE layer per pulse (24x per forward), so the hot path avoids a
    sort entirely.

    Returns int32 ``[N, K]`` with the slot index, or -1 for a dropped assignment.
    """
    N, K = topk_idx.shape
    device = topk_idx.device
    flat_e = topk_idx.reshape(-1).long()
    M = flat_e.numel()
    if M == 0:
        return torch.empty((N, K), dtype=torch.int32, device=device)

    with torch.no_grad():
        counts = torch.zeros(num_experts, device=device, dtype=torch.long)
        counts.index_add_(0, flat_e, torch.ones(M, device=device, dtype=torch.long))
        start_off = counts.cumsum(0) - counts            # [E] group start per expert

        # Single path, no branch and no host sync.
        #
        # The rank must be the assignment's ORDINAL WITHIN ITS EXPERT, not its position
        # in the flattened token order, so assignments are ordered by (expert asc,
        # weight desc). That one ordering does both jobs: it establishes the expert
        # grouping the rank needs, and it puts the highest-weight assignments first so
        # only the lowest-weight ones fall past capacity.
        #
        # The previous version branched on `int(counts.max()) <= capacity` to skip the
        # sort, but that .item() forced a device sync 24x per forward. Measured cost of
        # the branch itself was 0.10 ms/call, so the sync was the only reason to have it.
        flat_w = topk_weights.reshape(-1).float()
        # Single composite key sort: flat_e in [0, E), flat_w >= 0.
        # (expert asc, weight desc) is mapped to ascending order of sort_key in one 32-bit sort:
        # w / (1.0 + w) strictly preserves monotonic ordering for any w in [0, inf) and maps into [0, 1):
        w_norm = flat_w.clamp_min(0.0) / (1.0 + flat_w.clamp_min(0.0))
        sort_key = flat_e.to(torch.float32) - w_norm * 0.5
        order = torch.argsort(sort_key, stable=True)
        e_sorted = flat_e[order]
        counts_sorted = torch.zeros(num_experts, device=device, dtype=torch.long)
        counts_sorted.index_add_(0, e_sorted, torch.ones(M, device=device, dtype=torch.long))
        start_sorted = counts_sorted.cumsum(0) - counts_sorted
        rank_sorted = torch.arange(M, device=device) - start_sorted[e_sorted]
        slot_sorted = torch.where(rank_sorted < capacity, rank_sorted, torch.full_like(rank_sorted, -1))
        slots = torch.empty(M, dtype=torch.int32, device=device)
        slots[order] = slot_sorted.to(torch.int32)
        return slots.view(N, K)


class _FusedMoEDispatchFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        token_to_slot: torch.Tensor,
        num_experts: int,
        capacity: int,
    ):
        (expert_inputs,) = pulsar_cuda.moe_dispatch_forward(
            x.contiguous(), topk_idx.contiguous(), token_to_slot.contiguous(), num_experts, capacity
        )
        ctx.save_for_backward(topk_idx, token_to_slot)
        ctx.N = x.size(0)
        ctx.D = x.size(1)
        return expert_inputs

    @staticmethod
    def backward(ctx, d_expert_inputs: torch.Tensor):
        topk_idx, token_to_slot = ctx.saved_tensors
        dx = pulsar_cuda.moe_dispatch_backward(
            d_expert_inputs.contiguous(), topk_idx, token_to_slot, ctx.N, ctx.D
        )
        return dx, None, None, None, None


class _FusedMoECombineFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        expert_outputs: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        token_to_slot: torch.Tensor,
        shared_out: torch.Tensor | None = None,
    ):
        sh = shared_out.contiguous() if shared_out is not None else None
        routed_out = pulsar_cuda.moe_combine_forward(
            expert_outputs.contiguous(),
            topk_idx.contiguous(),
            topk_weights.contiguous(),
            token_to_slot.contiguous(),
            sh,
        )
        ctx.save_for_backward(expert_outputs, topk_idx, topk_weights, token_to_slot)
        ctx.has_shared = shared_out is not None
        return routed_out

    @staticmethod
    def backward(ctx, d_routed_out: torch.Tensor):
        expert_outputs, topk_idx, topk_weights, token_to_slot = ctx.saved_tensors
        d_expert_outputs, d_topk_weights = pulsar_cuda.moe_combine_backward(
            d_routed_out.contiguous(), expert_outputs, topk_idx, topk_weights, token_to_slot
        )
        d_shared_out = d_routed_out if ctx.has_shared else None
        return d_expert_outputs, None, d_topk_weights, None, d_shared_out


class _FusedMoERouterFunction(torch.autograd.Function):
    """Fused router top-2 selection and weight normalisation.

    Z-loss is deliberately NOT computed here. The kernel's z-term assumed an upstream
    gradient of 1, but the real scale is 1/24 (averaging over 8 blocks x 3 pulses) times
    1/grad_accum, which made the effective z-loss weight ~1000x its configured value.
    The backward also ignored the incoming _d_z_loss entirely. Z-loss is now computed
    in PyTorch from the router logits so autograd applies the correct chain rule.

    The kernel is called with z_loss_weight=0, so it only produces the top-2 weight
    normalisation gradient.
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, expert_bias: torch.Tensor):
        topk_idx, topk_weights, _z_loss_unused = pulsar_cuda.moe_router_topk_forward(
            logits.contiguous(), expert_bias.float().contiguous(), 0.0
        )
        ctx.save_for_backward(logits, topk_idx)
        return topk_idx, topk_weights

    @staticmethod
    def backward(ctx, _d_topk_idx, d_topk_weights: torch.Tensor):
        logits, topk_idx = ctx.saved_tensors
        d_logits = pulsar_cuda.moe_router_topk_backward(
            d_topk_weights.contiguous(), logits.contiguous(), topk_idx, 0.0
        )
        return d_logits, None


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_ff = d_ff
        self.w_gate_up = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.w_down.is_residual_out = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gu = self.w_gate_up(x)
        act = _FusedSwiGLUFunction.apply(gu, self.d_ff)
        return self.w_down(act)


class PulsarMoEMLP(nn.Module):
    def __init__(self, cfg: PulsarConfig):
        super().__init__()
        self.cfg = cfg
        d_shared = cfg.d_shared if cfg.d_shared is not None else cfg.d_expert
        self.shared_expert = SwiGLUMLP(cfg.d_model, d_shared)

        # 3D Contiguous Expert Parameters: [n_routed_experts, d_in, d_out]
        # Eliminates torch.stack overhead entirely (saving ~36 ms per micro-batch)
        #
        # Init must match the shared expert, which follows the residual-output rule
        # std = 0.02 / sqrt(2 * n_core * default_pulses) for its down projection and
        # 0.02 for upstream weights (see _init_weights). The old values
        # (1/sqrt(d_model) ~ 0.031 for gate_up and 1/sqrt(d_expert) ~ 0.031 for down)
        # made the routed path ~10x the shared expert's contribution at init, which
        # dominates the MLP update and is the likely cause of the |dh| >> |h_in|
        # blow-up that harmonic damping was added to mask.
        residual_std = 0.02 / math.sqrt(2 * cfg.n_core * cfg.default_pulses)
        # LeCun fan-in + top-k variance matching:
        # Matches routed combined output variance to shared expert down-projection output variance.
        routed_down_std = residual_std * math.sqrt((cfg.d_shared / cfg.d_expert) * cfg.top_k)
        self.routed_gate_up = nn.Parameter(torch.empty(cfg.n_routed_experts, cfg.d_model, 2 * cfg.d_expert))
        self.routed_down = nn.Parameter(torch.empty(cfg.n_routed_experts, cfg.d_expert, cfg.d_model))
        nn.init.normal_(self.routed_gate_up, std=0.02)
        nn.init.normal_(self.routed_down, std=routed_down_std)

        # Pulse-Indexed Router [max_pulses, n_routed_experts, d_model]
        # std=0.002 yields initial logits std ≈ 0.064 across 16 experts for near-uniform initial exploration
        self.router_weight = nn.Parameter(torch.empty(cfg.max_pulses, cfg.n_routed_experts, cfg.d_model, dtype=torch.float32))
        nn.init.normal_(self.router_weight, mean=0.0, std=0.002)

        # Per-Pulse Routed-Expert Scale [max_pulses, d_model]
        self.routed_scale = nn.Parameter(torch.ones(cfg.max_pulses, cfg.d_model, dtype=torch.float32))

        # Global Across-Pulse Auxiliary-Loss-Free Bias
        if cfg.per_pulse_bias:
            self.register_buffer(
                "expert_bias",
                torch.zeros(cfg.max_pulses, cfg.n_routed_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.register_buffer("expert_bias", torch.zeros(cfg.n_routed_experts, dtype=torch.float32), persistent=True)
        # persistent=True: update_biases() reads these, so a checkpoint taken mid-step
        # must carry them or the resumed run applies a different bias update. They are
        # small (E and K*E floats), so the checkpoint cost is negligible.
        self.register_buffer("expert_counts_accum", torch.zeros(cfg.n_routed_experts, dtype=torch.float32), persistent=True)
        self.register_buffer("expert_pulse_counts", torch.zeros(cfg.max_pulses, cfg.n_routed_experts, dtype=torch.float32), persistent=True)
        # Drop statistics, accumulated on the GPU and drained by pop_moe_stats().
        # No .item() in the hot path: syncing per micro-batch would serialise the stream.
        self.register_buffer("drop_counts_accum", torch.zeros((), dtype=torch.float32), persistent=False)
        # What the OLD metric reported: sum(max(count - capacity, 0)) from bincounts.
        # Kept only for cross-checking against the slot-derived value.
        self.register_buffer(
            "demand_over_capacity_accum", torch.zeros((), dtype=torch.float32), persistent=False
        )
        self.register_buffer("assigned_counts_accum", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("capacity_last", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("pulse_drop_accum", torch.zeros(cfg.max_pulses, dtype=torch.float32), persistent=False)
        self.register_buffer("pulse_assigned_accum", torch.zeros(cfg.max_pulses, dtype=torch.float32), persistent=False)

    def forward(
        self, x: torch.Tensor, pulse_idx: int = 0, routing_mask: torch.Tensor | None = None,
        router_diag: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        N = B * T
        x_flat = x.view(N, D)
        router_diag = getattr(self, "_diag_capture", None) is not None

        # 1. Pulse-indexed router logits
        # Router logits in FP32. bf16 spacing near |logit|=1 is ~0.008, comparable to
        # the 0.001 expert-bias step, so routing decisions near ties were quantised.
        # The transients are small (N x E), unlike an FP32 copy of x.
        router_w = self.router_weight[pulse_idx]
        logits = F.linear(
            x_flat.float(), router_w.float() if router_w.dtype != torch.float32 else router_w
        )  # [N, E] fp32
        # Centre the logits across experts. Top-k selection and the bias addition are
        # both invariant to a per-token constant, so this changes nothing mathematically
        # but keeps the absolute scale from drifting with the shared batch-mean component
        # of the hidden state (strong by pulse 3), which saturated sigmoid and left the
        # [-0.5, 0.5] bias without leverage.
        logits = logits - logits.mean(dim=-1, keepdim=True)

        # Padded positions must not influence routing statistics or z-loss: a run of
        # identical pad tokens would all choose the same experts, inflating drop counts
        # and distorting the balancing bias.
        valid = None
        if routing_mask is not None:
            valid = routing_mask.reshape(-1).bool()

        # 2. Router selection: fused top-2 + sigmoid + weight normalisation (CUDA),
        #    with z-loss computed in PyTorch so autograd scales it correctly. The z
        #    tensor is [N, E] (a few hundred KB), negligible next to the router matmul.
        lse_sq = torch.logsumexp(logits.float(), dim=-1).pow(2)
        if valid is not None:
            valid_f = valid.to(torch.float32)
            z_loss = self.cfg.router_z_loss_weight * ((lse_sq * valid_f).sum() / valid_f.sum().clamp_min(1.0))
        else:
            z_loss = self.cfg.router_z_loss_weight * lse_sq.mean()
        if (HAS_PULSAR_CUDA and logits.is_cuda
                and not torch.compiler.is_compiling()
                and hasattr(pulsar_cuda, "moe_router_topk_forward")
                and self.cfg.n_routed_experts == 8
                and self.cfg.top_k == 2):
            bias = self.expert_bias[pulse_idx] if self.cfg.per_pulse_bias else self.expert_bias
            topk_idx, topk_weights = _FusedMoERouterFunction.apply(logits, bias)
        else:
            scores = torch.sigmoid(logits.float())
            bias = self.expert_bias[pulse_idx] if self.cfg.per_pulse_bias else self.expert_bias
            biased_scores = scores + bias
            _, topk_idx = torch.topk(biased_scores, k=self.cfg.top_k, dim=-1)  # [N, K]
            topk_raw = torch.gather(scores, -1, topk_idx)
            topk_weights = (topk_raw / (topk_raw.sum(dim=-1, keepdim=True) + 1e-6)).to(x.dtype)

        # Invalidate padded positions so they do not compete for expert capacity slots
        if valid is not None:
            topk_weights = topk_weights * valid.unsqueeze(-1).to(topk_weights.dtype)

        # 3. Shared expert projection
        shared_out = self.shared_expert(x_flat)

        # 4. Fused MoE Dispatch & Batched 3D GEMM Execution (with 64-aligned cuBLAS tensor core tiling)
        # Capacity: ceil(avg_load * factor) rounded up to a multiple of 64. The factor is
        # configurable (default 1.25); the old hardcoded 1.08 gave only 8.3% headroom at
        # N=3072, so drops were common but never measured.
        avg_load = N * self.cfg.top_k / self.cfg.n_routed_experts
        raw_cap = int(math.ceil(avg_load * self.cfg.moe_capacity_factor))
        align = max(1, int(self.cfg.moe_capacity_align))
        capacity = max(align, ((raw_cap + align - 1) // align) * align)

        # Pre-drop counts, per expert. Computed from topk_idx (not from surviving slots)
        # so that drop statistics reflect demand rather than capacity.
        # Pre-drop counts per expert, kept fully asynchronous. torch.bincount must
        # size its output from the data, which forces a device sync; index_add_ on a
        # known-size buffer does not. This runs 24x per micro-batch (8 blocks x 3
        # pulses), so the sync was measurable.
        # Slot assignment is computed once, before the counters, because the REAL number
        # of dropped assignments is the count of negative slots. Deriving drops from
        # bincounts only equals that while the slot logic is correct - the fast-path bug
        # discarded ~85% of assignments while this metric read 0%.
        token_to_slot = compute_token_slots(
            topk_idx, topk_weights, self.cfg.n_routed_experts, capacity
        )
        if valid is not None:
            token_to_slot = torch.where(valid.unsqueeze(-1), token_to_slot, torch.full_like(token_to_slot, -1))

        with torch.no_grad():
            bincounts = torch.zeros(self.cfg.n_routed_experts, device=x.device, dtype=torch.float32)
            count_idx = topk_idx.reshape(-1)
            # Branch-free, so no host sync on `valid.all()`. Padded assignments are
            # remapped onto expert 0 and weighted 0, which index_add_ discards; the
            # unmasked case is simply the same expression with an all-ones weight.
            if valid is not None:
                assign_valid = valid.repeat_interleave(self.cfg.top_k)
                weights = assign_valid.to(torch.float32)
                count_idx = torch.where(assign_valid, count_idx, torch.zeros_like(count_idx))
            else:
                weights = torch.ones(topk_idx.numel(), device=x.device, dtype=torch.float32)
            bincounts.index_add_(0, count_idx, weights)
            # Counters feed the auxiliary-loss-free bias update, so only TRAINING
            # forwards may move them. Validation and generation were accumulating too:
            # one eval forward added 768 assignments and shifted the routing bias before
            # the next optimizer step. Suppressed during checkpoint recompute as well,
            # otherwise every counter would increment twice.
            if self.training and (not _RECOMPUTING.active):
                # In per-pulse mode the bias update reads expert_pulse_counts only, so
                # expert_counts_accum would grow forever without being read or zeroed
                # (FP32 stops resolving +1 at 2^24). Skip it there.
                if not self.cfg.per_pulse_bias:
                    self.expert_counts_accum.add_(bincounts)
                self.expert_pulse_counts[pulse_idx].add_(bincounts)
                # Drop accounting stays on the GPU; drained by pop_moe_stats().
                # Tracked per pulse as well as in total: drops are a per-pulse
                # phenomenon, so a single global rate would hide the case that matters.
                # Drops measured from the ACTUAL slot assignment (only counting valid tokens).
                if valid is not None:
                    dropped = ((token_to_slot < 0) & valid.unsqueeze(-1)).sum().to(torch.float32)
                    assigned = (valid.sum() * self.cfg.top_k).to(torch.float32)
                else:
                    dropped = (token_to_slot < 0).sum().to(torch.float32)
                    assigned = torch.full_like(dropped, float(token_to_slot.numel()))
                # demand_over_capacity is kept for comparison: it is what the old code
                # reported, and the two disagree whenever the slot logic is wrong.
                self.demand_over_capacity_accum.add_(
                    (bincounts - capacity).clamp(min=0.0).sum()
                )
                self.drop_counts_accum.add_(dropped)
                self.assigned_counts_accum.add_(assigned)
                self.pulse_drop_accum[pulse_idx].add_(dropped)
                self.pulse_assigned_accum[pulse_idx].add_(assigned)
                self.capacity_last.fill_(float(capacity))

        if (HAS_PULSAR_CUDA and x_flat.is_cuda and x_flat.dtype == torch.bfloat16
                and not torch.compiler.is_compiling()
                and hasattr(pulsar_cuda, "moe_dispatch_forward")):
            expert_inputs = _FusedMoEDispatchFunction.apply(
                x_flat, topk_idx, token_to_slot, self.cfg.n_routed_experts, capacity
            )

            # Batched expert projections with ZERO weight stacking
            gu = torch.bmm(expert_inputs, self.routed_gate_up)              # [E, C, 2048]
            act = _FusedSwiGLUFunction.apply(gu, self.cfg.d_expert)          # [E, C, 1024]
            expert_out = torch.bmm(act, self.routed_down)                   # [E, C, 1024]

            # Fused combine with shared expert residual initialization (zero extra memory roundtrips)
            r_scale = self.routed_scale[pulse_idx].to(x.dtype)
            routed_only = _FusedMoECombineFunction.apply(
                expert_out, topk_idx, topk_weights, token_to_slot, None
            )
            out = (shared_out + routed_only * r_scale).view(B, T, D)
        else:
            # Fallback for CPU, non-bf16, torch.compile, or environments without custom CUDA extension.
            # Vectorized PyTorch MoE engine: zero Python loops over experts, zero dynamic shape
            # graph breaks, and zero NaN poisoning (strictly initialized with torch.zeros).
            flat_token_idx = torch.arange(N, device=x.device).unsqueeze(1).expand(N, self.cfg.top_k).reshape(-1)
            flat_expert_ids = topk_idx.reshape(-1)
            flat_weights = topk_weights.reshape(-1).to(x.dtype)
            flat_slots = token_to_slot.reshape(-1)
            E = self.cfg.n_routed_experts
            C = capacity

            valid_mask = (flat_slots >= 0) & (flat_expert_ids >= 0) & (flat_expert_ids < E)
            clamped_slot = flat_slots.clamp(min=0).long()
            dest_idx = flat_expert_ids * C + clamped_slot

            valid_tokens = x_flat[flat_token_idx] * valid_mask.unsqueeze(-1).to(x.dtype)
            expert_inputs_flat = torch.zeros(E * C, D, device=x.device, dtype=x.dtype)
            expert_inputs_flat.index_add_(0, dest_idx, valid_tokens)
            expert_inputs = expert_inputs_flat.view(E, C, D).to(self.routed_gate_up.dtype)

            # Batched 3D GEMMs with ZERO weight stacking
            gu = torch.bmm(expert_inputs, self.routed_gate_up)
            act = _FusedSwiGLUFunction.apply(gu, self.cfg.d_expert)
            expert_out = torch.bmm(act, self.routed_down.to(act.dtype))

            # Vectorized gather
            expert_out_flat = expert_out.view(E * C, D)
            w = (flat_weights * valid_mask.to(flat_weights.dtype)).unsqueeze(-1)
            contrib = (w * expert_out_flat[dest_idx]).to(x.dtype)

            routed_out = torch.zeros(N, D, device=x.device, dtype=x.dtype)
            routed_out.index_add_(0, flat_token_idx, contrib)

            r_scale = self.routed_scale[pulse_idx].to(x.dtype)
            out = (shared_out + routed_out * r_scale).view(B, T, D)
        # Router confidence = mean max gating weight. topk_weights are normalised to sum
        # to 1 per token, so their mean over all (token, k) entries is exactly 0.5 no
        # matter what the router does - the old value was pinned at 0.5 by construction.
        # The per-token max is the quantity that actually moves as routing sharpens.
        mean_max_weight = topk_weights.detach().max(dim=-1).values.mean() if topk_weights.ndim == 2 else topk_weights.detach().mean()
        if router_diag:
            self._diag_capture.append(
                (pulse_idx, x_flat.detach(), logits.detach(), topk_idx.detach())
            )
        return out, z_loss, mean_max_weight, bincounts

    def router_diagnostics(self, x_flat: torch.Tensor, logits: torch.Tensor,
                           topk_idx: torch.Tensor, pulse_idx: int) -> dict:
        """Measure whether the router is being driven by a token-independent term.

        The hypothesis under test is that by pulse 3 the hidden state carries a strong
        batch-mean component, so a router update along +/-x_bar moves the logits of all
        tokens together and saturates sigmoid. shared_mean_frac is the share of
        logit variance explained by that shared term:

            Var_e(w_e . mu) / (Var_e(w_e . mu) + mean_tokens Var_e(w_e . (x - mu)))

        Also reports how often the balancing bias overrides the content-based top-2.
        Runs under no_grad on the caller's sample; no host syncs here.
        """
        with torch.no_grad():
            xf = x_flat.float()
            mu = xf.mean(dim=0, keepdim=True)                       # [1, D]
            centred = xf - mu
            w = self.router_weight[pulse_idx].float()

            shared = mu @ w.t()                                     # [1, E]
            diff = centred @ w.t()                                  # [N, E]
            var_shared = shared.var(dim=-1, unbiased=False).mean()
            var_diff = diff.var(dim=-1, unbiased=False).mean()
            denom = (var_shared + var_diff).clamp_min(1e-12)
            shared_mean_frac = (var_shared / denom).reshape(())

            # bias_override_rate: top-2 with bias differs from top-2 without it.
            bias = (self.expert_bias[pulse_idx] if self.cfg.per_pulse_bias
                    else self.expert_bias).float()
            k = self.cfg.top_k
            scores = torch.sigmoid(logits.float())
            with_bias = torch.topk(scores + bias, k=k, dim=-1).indices
            without = torch.topk(scores, k=k, dim=-1).indices
            # Compare as sets: the same two experts in a different order is not an override.
            same = (with_bias.unsqueeze(-1) == without.unsqueeze(1)).any(-1).sum(-1) == k
            bias_override_rate = (~same).float().mean()

            # Top-1 share per expert and router entropy (bits of the top-1 histogram).
            counts = torch.bincount(topk_idx[:, 0].reshape(-1),
                                    minlength=self.cfg.n_routed_experts).float()
            shares = counts / counts.sum().clamp_min(1.0)
            ent = -(shares * (shares + 1e-9).log()).sum()

            abs_bias = bias.abs()

        return {
            "shared_mean_frac": shared_mean_frac.detach(),
            "bias_override_rate": bias_override_rate.detach(),
            "top1_entropy": ent.detach(),
            "max_top1_share": shares.max().detach(),
            "bias_abs_p50": abs_bias.median().detach() if abs_bias.numel() else torch.zeros(()),
            "bias_abs_p99": (
                torch.quantile(abs_bias.float(), 0.99) if abs_bias.numel() else torch.zeros(())
            ),
        }

    def update_biases(self, bias_lr: float = 0.001, num_pulses: int | None = None):
        """Auxiliary-loss-free bias update.

        With per_pulse_bias enabled each pulse's bias is updated from that pulse's own
        counts, because load imbalance is a per-pulse phenomenon.
        """
        if self.cfg.per_pulse_bias:
            k = num_pulses or self.cfg.default_pulses
            counts = self.expert_pulse_counts[:k]
            avg = counts.mean(dim=1, keepdim=True)
            self.expert_bias[:k].add_(bias_lr * torch.sign(avg - counts))
            # Zero-center across expert dimension to prevent DC drift saturation
            self.expert_bias[:k].sub_(self.expert_bias[:k].mean(dim=1, keepdim=True))
            self.expert_bias[:k].clamp_(-1.0, 1.0)
            self.expert_pulse_counts[:k].zero_()
            # Reset even though it is not accumulated in this mode, so a model that
            # switches modes (or loads an old checkpoint) cannot inherit a stale sum.
            self.expert_counts_accum.zero_()
        else:
            avg_count = self.expert_counts_accum.mean()
            self.expert_bias.add_(bias_lr * torch.sign(avg_count - self.expert_counts_accum))
            self.expert_bias.sub_(self.expert_bias.mean())
            self.expert_bias.clamp_(-1.0, 1.0)
            self.expert_counts_accum.zero_()


class PulsarBlock(nn.Module):
    def __init__(self, cfg: PulsarConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_eps, max_pulses=cfg.max_pulses)
        self.attn = PulsarAttention(cfg, layer_idx)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.rms_eps, max_pulses=cfg.max_pulses)
        self.mlp = PulsarMoEMLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        pulse_idx: int = 0,
        v_first: torch.Tensor | None = None,
        routing_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_in = self.attn_norm(x, pulse_idx=pulse_idx)
        attn_out, v_first = self.attn(attn_in, cos, sin, pulse_idx=pulse_idx, v_first=v_first)
        x = x + attn_out

        mlp_in = self.mlp_norm(x, pulse_idx=pulse_idx)
        mlp_out, z_loss, gate_weight_mean, bincounts = self.mlp(
            mlp_in, pulse_idx=pulse_idx, routing_mask=routing_mask
        )
        x = x + mlp_out
        return x, v_first, z_loss, gate_weight_mean, bincounts


class PulsarForCausalLM(nn.Module):
    # Parameters that must stay FP32 even when the model is cast to bf16 for training.
    #
    # These are small, non-matrix tensors whose optimizer steps are tiny compared with
    # bf16 spacing. Near 1.0 the bf16 spacing is 0.0078 and half-spacing 0.0039; an
    # AdamW step at peak LR is 6e-4, and a Muon step on a decayed LR is smaller still.
    # A bf16 copy therefore cannot move at all, silently freezing what the docs describe
    # as learnable. Matrix weights are deliberately NOT in this table: they are updated
    # through stochastic rounding (WI-02) rather than kept in FP32, to bound VRAM.
    FP32_PARAM_SUBSTRINGS = (
        "norm",
        "bias",
        "router",
        "mix_logit",
        "res_scale",
        "routed_scale",
        "attn_gate",
        "pulse_step_emb",
        "v_resid_lambda",
        "q_norm_weight",
        "k_norm_weight",
    )

    # Buffers that must stay FP32 regardless of the model dtype.
    FP32_BUFFER_SUBSTRINGS = (
        "expert_bias",
        "expert_counts_accum",
        "expert_pulse_counts",
        "inv_freq",
        "cos_cached",
        "sin_cached",
    )

    def to_training_precision(self, dtype: torch.dtype = torch.bfloat16) -> "PulsarForCausalLM":
        """Cast the model to training precision - the single source of truth.

        Replaces the ad-hoc cast-then-re-cast block in ``train.py``, which matched on
        ``p.ndim == 1 or "norm" ...`` and therefore missed every small tensor whose name
        does not contain those substrings (``pulse_res_scale``, ``routed_scale``,
        ``attn_gate``, ``pulse_step_emb.weight``), leaving them bf16 and frozen. It also
        cast *buffers* to bf16, including the MoE expert counters (which accumulate
        counts in the hundreds and saturate) and the RoPE caches.

        After the cast the RoPE caches are rebuilt in FP32, because a bf16 cos/sin table
        quantises the rotation angles.
        """
        self.to(dtype=dtype)

        for name, param in self.named_parameters():
            if self._param_stays_fp32(name, param):
                param.data = param.data.float()

        for name, buf in self.named_buffers():
            if any(s in name for s in self.FP32_BUFFER_SUBSTRINGS):
                buf.data = buf.data.float()

        # RoPE frequency tables must be FP32 and correctly sized for the configured
        # context length after a device/dtype move.
        for rope in (getattr(self, "rope_local", None), getattr(self, "rope_global", None)):
            if rope is not None:
                # REBUILD from (theta, dim) in FP32. Calling .float() on the buffer only
                # up-casts values that self.to(bf16) already rounded: measured error is
                # 3.65e-3 relative (0.37%), i.e. up to ~0.5 rad of angle error by
                # position 1024 and tens of radians at 32k. Training stayed
                # self-consistent only because every path used the same rounded table;
                # any FP32 eval or export would place tokens differently.
                rope.inv_freq.data = rope._inv_freq_fp32(device=rope.inv_freq.device)
                rope._set_cos_sin_cache(rope.max_seq_len, device=rope.inv_freq.device, dtype=torch.float32)

        return self

    def _param_stays_fp32(self, name: str, param: torch.Tensor) -> bool:
        return param.ndim == 1 or any(s in name for s in self.FP32_PARAM_SUBSTRINGS)

    def __init__(self, cfg: PulsarConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope_local = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta_local)
        self.rope_global = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta_global)

        # 8-core recurrent backbone (evaluated K times)
        self.core = nn.ModuleList([PulsarBlock(cfg, layer_idx=i) for i in range(cfg.n_core)])
        self.pulse_step_emb = nn.Embedding(cfg.max_pulses, cfg.d_model)

        # Unit-Normalized Per-Pulse Prompt Anchor
        self.pulse_state_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.anchor_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        # Gradient checkpointing for long context: recompute each core block in
        # backward instead of retaining activations. Off by default.
        self.grad_checkpoint = False
        # Initialized to -1.3863 so sigmoid(-1.3863) = 0.200 exact
        self.input_mix_logit = nn.Parameter(torch.full((cfg.max_pulses, cfg.d_model), -1.3863, dtype=torch.float32))

        # Harmonic Per-Pulse Learnable Residual Scale [max_pulses, d_model]
        # Initialized harmonically: 1.0 / (k + 1.0) -> Pulse 1: 1.0, Pulse 2: 0.5, Pulse 3: 0.33
        init_res_scales = torch.empty(cfg.max_pulses, cfg.d_model, dtype=torch.float32)
        for k in range(cfg.max_pulses):
            init_res_scales[k].fill_(1.0 / (k + 1.0))
        self.pulse_res_scale = nn.Parameter(init_res_scales)

        # Output projection head (tied with token embedding)
        self.final_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        nn.init.zeros_(self.pulse_step_emb.weight)
        self.lm_head.weight = self.tok_emb.weight  # Weight tying

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "is_residual_out", False):
                total_effective_layers = self.cfg.n_core * self.cfg.default_pulses
                std = 0.02 / math.sqrt(2 * total_effective_layers)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # std=0.006 eliminates tied-embedding self-logit boost (+20.48 spike at std=0.02),
            # stabilizing initial Step 0 loss near ln(vocab_size)
            nn.init.normal_(module.weight, mean=0.0, std=0.006)

    def update_expert_biases(self, bias_lr: float = 0.001, num_pulses: int | None = None):
        """Update auxiliary-loss-free routing biases globally across all blocks."""
        for block in self.core:
            block.mlp.update_biases(bias_lr=bias_lr, num_pulses=num_pulses)

    def _migrate_checkpoint_bias(self, state_dict: dict) -> dict:
        """Broadcast an old [E] expert_bias across pulses for per_pulse_bias models.

        Old checkpoints store a single across-pulse bias. Enabling per-pulse bias would
        otherwise fail to load it; broadcasting is the neutral migration (every pulse
        starts from the shared value).
        """
        if not self.cfg.per_pulse_bias:
            return state_dict
        migrated = dict(state_dict)
        default_bias = torch.zeros(self.cfg.n_routed_experts, dtype=torch.float32)
        for name, buf in self.named_buffers():
            if not name.endswith("expert_bias"):
                continue
            cur = migrated.get(name)
            if cur is None:
                migrated[name] = (
                    default_bias.unsqueeze(0).expand(self.cfg.max_pulses, -1).clone()
                )
                continue
            if cur.shape == (self.cfg.n_routed_experts,):
                cur = cur.to(torch.float32).unsqueeze(0).expand(self.cfg.max_pulses, -1).clone()
                print(f"[migrate] {name}: [E] -> [max_pulses, E] by broadcast")
            # Zero-center across expert dimension to remove historical DC drift
            b = cur.float()
            migrated[name] = (b - b.mean(dim=-1, keepdim=True)).to(cur.dtype)
        return migrated

    def _migrate_counters(self, state_dict: dict) -> dict:
        """Fill persistent MoE counter buffers that round-1 checkpoints lack.

        expert_counts_accum / expert_pulse_counts became persistent so resume would be
        exact, but checkpoints written before that have no such keys; strict loading then
        fails on a missing key rather than a shape mismatch, which reads like corruption.
        """
        migrated = dict(state_dict)
        # Only PERSISTENT buffers belong in a state_dict. Adding a non-persistent one
        # makes it an unexpected key on the way back in.
        for mod_name, module in self.named_modules():
            for buf_name, buf in module.named_buffers(recurse=False):
                if buf_name in module._non_persistent_buffers_set:
                    continue
                full = f"{mod_name}.{buf_name}" if mod_name else buf_name
                if full in migrated:
                    continue
                if "counts_accum" in buf_name or "pulse_counts" in buf_name:
                    migrated[full] = torch.zeros_like(buf)
                    print(f"[migrate] {full}: missing in checkpoint, filled with zeros")
        return migrated

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        migrated = self._migrate_checkpoint_bias(state_dict)
        migrated = self._migrate_counters(migrated)
        return super().load_state_dict(migrated, strict=strict, assign=assign)

    @torch.no_grad()
    def pop_moe_stats(self) -> dict:
        """Drain accumulated MoE drop statistics. Call once per log window.

        Counters live on the GPU and are only synchronised here, so the training step
        itself performs no host sync. Returns totals since the last call and resets.
        """
        dropped = 0.0
        assigned = 0.0
        capacity = 0.0
        demand_over = 0.0
        per_block_drop = []
        pulse_drop = torch.zeros(self.cfg.max_pulses, device=self.core[0].mlp.drop_counts_accum.device)
        pulse_assigned = torch.zeros_like(pulse_drop)
        for block in self.core:
            mlp = block.mlp
            d = float(mlp.drop_counts_accum.item())
            a = float(mlp.assigned_counts_accum.item())
            c = float(mlp.capacity_last.item())
            dropped += d
            assigned += a
            capacity = c or capacity
            per_block_drop.append(d)
            pulse_drop += mlp.pulse_drop_accum.float()
            pulse_assigned += mlp.pulse_assigned_accum.float()
            demand_over += float(mlp.demand_over_capacity_accum.item())
            mlp.demand_over_capacity_accum.zero_()
            mlp.drop_counts_accum.zero_()
            mlp.assigned_counts_accum.zero_()
            mlp.pulse_drop_accum.zero_()
            mlp.pulse_assigned_accum.zero_()
        per_pulse = [
            {
                "pulse": k + 1,
                "dropped": float(pulse_drop[k].item()),
                "assigned": float(pulse_assigned[k].item()),
                "drop_rate": float((pulse_drop[k] / pulse_assigned[k]).item())
                if float(pulse_assigned[k].item()) > 0 else 0.0,
            }
            for k in range(self.cfg.max_pulses)
        ]
        # Cross-check: the slot-derived count is authoritative, the bincount heuristic is
        # only a proxy that silently disagrees when slot assignment is broken.
        if abs(demand_over - dropped) > max(1.0, 0.01 * max(1.0, dropped)):
            print(
                f"[Pulsar][WARN] MoE drop accounting mismatch: "
                f"slot-derived {dropped:.0f} vs demand-over-capacity {demand_over:.0f}"
            )
        return {
            "dropped": dropped,
            "assigned": assigned,
            "capacity": capacity,
            "drop_rate": (dropped / assigned) if assigned > 0 else 0.0,
            "demand_over_capacity": demand_over,
            "per_block_dropped": per_block_drop,
            "per_pulse": per_pulse,
        }

    def _chunked_ce(self, h: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Mean soft-capped cross-entropy over all tokens, chunked to bound VRAM."""
        h_flat = h.reshape(-1, self.cfg.d_model)
        tgt_flat = labels.reshape(-1)
        chunk = getattr(self.cfg, "ce_chunk_size", 4096)
        total = torch.zeros((), device=h_flat.device, dtype=torch.float32)
        for i in range(0, h_flat.size(0), chunk):
            hc = h_flat[i:i + chunk]
            tc = tgt_flat[i:i + chunk]
            raw = F.linear(hc, self.lm_head.weight)
            if (HAS_PULSAR_CUDA and raw.is_cuda and raw.dtype == torch.bfloat16
                    and not torch.compiler.is_compiling()
                    and hasattr(pulsar_cuda, "fused_softcap_ce_forward")):
                part = _FusedSoftcapCrossEntropyFunction.apply(raw, tc, self.cfg.logit_softcap, "sum", -100)
            elif raw.is_cuda:
                part = _compiled_softcap_ce(raw, tc, self.cfg.logit_softcap, "sum", -100)
            else:
                part = _softcap_cross_entropy(raw, tc, self.cfg.logit_softcap, "sum", -100)
            total = total + part
        # Divide by VALID tokens, not the total: ignored (-100) positions contribute
        # zero to the sum, so dividing by all tokens deflates the loss (and its
        # gradients). Any batch with padding or ignore_index was affected.
        n_valid = (tgt_flat != -100).sum().clamp_min(1)
        return total / n_valid

    def _compute_pulse_weights(self, K: int) -> list[float]:
        w_first = self.cfg.aux_loss_weight_1
        w_mid = self.cfg.aux_loss_weight_2
        w_final = self.cfg.final_loss_weight
        if K == 1:
            raw = [w_final]
        else:
            raw = [w_first] + [w_mid] * (K - 2) + [w_final]
        w_sum = sum(raw)
        return [w / w_sum for w in raw]

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        num_pulses: int | None = None,
        subsample_offset: int = 0,
        eval_full_pulses: bool = False,
        routing_mask: torch.Tensor | None = None,
        router_diag: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, float]]:
        B, T = input_ids.shape
        K = num_pulses if num_pulses is not None else self.cfg.default_pulses
        if K < 1 or K > self.cfg.max_pulses:
            raise ValueError(f"num_pulses must be in [1, {self.cfg.max_pulses}], got {K}")

        # Token embedding & RoPE frequencies (local 10k and global 1M)
        h_0 = self.tok_emb(input_ids)
        cos_local, sin_local = self.rope_local(h_0, T)
        cos_global, sin_global = self.rope_global(h_0, T)

        # Router diagnostics are sampled, not continuous: capture is enabled for this
        # forward only, and the trainer calls it on at most one micro-batch per log
        # window. Cost is a no-grad matmul per (block, pulse) on that one batch.
        if router_diag:
            for block in self.core:
                block.mlp._diag_capture = []

        h_0_norm = self.anchor_norm(h_0)
        h_pulse = h_0
        v_first = None
        total_loss = torch.zeros((), device=input_ids.device, dtype=torch.float32) if labels is not None else None
        total_z_loss = torch.zeros((), device=input_ids.device, dtype=torch.float32)

        # Deep-supervision weights, normalised over the K ACTIVE pulses so they always
        # sum to 1.0.
        pulse_weights = self._compute_pulse_weights(K)

        pulse_losses: list[torch.Tensor] = []
        pulse_deltas: list[torch.Tensor] = []
        gate_confidence_accum = torch.zeros((), device=input_ids.device, dtype=torch.float32)
        global_expert_counts = torch.zeros(self.cfg.n_routed_experts, device=input_ids.device, dtype=torch.float32)
        pulse_expert_counts = torch.zeros(K, self.cfg.n_routed_experts, device=input_ids.device, dtype=torch.float32)
        # Per-(block, pulse) expert counts. Summing across blocks by expert INDEX mixes
        # unrelated routers: Expert 0 of block 0 has nothing to do with Expert 0 of
        # block 1, so averaging them cancelled out per-block collapses and reported a
        # ~7pp spread while the implied true worst expert was ~38%.
        block_pulse_counts = torch.zeros(
            self.cfg.n_core, K, self.cfg.n_routed_experts,
            device=input_ids.device, dtype=torch.float32,
        )
        logits = None

        for k in range(K):
            # 1. Unit-Normalized Prompt Anchor Mixing (h_0_norm cached across pulses)
            # step_emb is a FP32 parameter (kept FP32 by to_training_precision because a
            # bf16 copy would be frozen by rounding). Cast it to the activation dtype
            # here, or the addition would silently promote the entire residual stream
            # and every downstream matmul to FP32.
            step_emb = self.pulse_step_emb.weight[k].to(h_0_norm.dtype).view(1, 1, -1)
            if k == 0:
                h_in = h_0_norm + step_emb
            else:
                mix_k = torch.sigmoid(self.input_mix_logit[k]).to(h_pulse.dtype)
                h_in = (1.0 - mix_k) * self.pulse_state_norm(h_pulse) + mix_k * h_0_norm + step_emb

            # 2. Core 8-block recurrent pass
            h_curr = h_in
            for i, block in enumerate(self.core):
                cos = cos_local if block.attn.is_local else cos_global
                sin = sin_local if block.attn.is_local else sin_global
                if self.grad_checkpoint and self.training:
                    from torch.utils.checkpoint import checkpoint

                    # Blocks mutate MoE counters as a side effect. Recompute during
                    # backward would double-count them, so counters are frozen around
                    # context_fn takes (forward_ctx, recompute_ctx) in that order, per
                    # the torch.utils.checkpoint docs: "The function and its recomputation
                    # will be run under the first and second context managers
                    # respectively." This was inverted, so the ORIGINAL forward skipped
                    # the MoE counters and the recomputation applied them - the counts
                    # still landed exactly once, but only during backward, so a forward
                    # that is never backpropagated never counted at all.
                    h_curr, v_first, z_loss, gate_weight_mean, bincounts = checkpoint(
                        block, h_curr, cos, sin,
                        pulse_idx=k, v_first=v_first, routing_mask=routing_mask,
                        use_reentrant=False,
                        context_fn=lambda: (_set_recomputing(False), _set_recomputing(True)),
                    )
                else:
                    h_curr, v_first, z_loss, gate_weight_mean, bincounts = block(
                        h_curr, cos, sin, pulse_idx=k, v_first=v_first,
                        routing_mask=routing_mask,
                    )
                total_z_loss = total_z_loss + z_loss
                gate_confidence_accum = gate_confidence_accum + gate_weight_mean
                global_expert_counts = global_expert_counts + bincounts
                pulse_expert_counts[k] = pulse_expert_counts[k] + bincounts
                block_pulse_counts[i, k] = block_pulse_counts[i, k] + bincounts

            # 3. Residual state update with harmonic contraction factor
            delta_h = (h_curr - h_in) * self.pulse_res_scale[k].to(h_in.dtype)
            h_pulse = h_in + delta_h

            delta_val = delta_h.detach().norm(dim=-1).mean()
            pulse_deltas.append(delta_val)

            # 4. Multi-Step Deep Supervision with Output Logit Soft-Capping
            if labels is not None:
                if k < K - 1:
                    if eval_full_pulses:
                        # Eval path: score EVERY pulse on 100% of tokens (chunked, same
                        # as the final pulse) so per-pulse CEs are comparable and
                        # val_ce_final is the model's real loss. Training keeps the
                        # cheaper 25% subsample.
                        loss_sub = self._chunked_ce(self.final_norm(h_pulse), labels)
                    else:
                        # Training: 25% subsampling with rotating disjoint offsets
                        offset = (subsample_offset + k) % 4
                        h_sub = h_pulse[:, offset::4, :].contiguous()
                        lbl_sub = labels[:, offset::4].contiguous()
                        raw_logits_sub = F.linear(self.final_norm(h_sub), self.lm_head.weight)
                        if (HAS_PULSAR_CUDA and raw_logits_sub.is_cuda
                                and raw_logits_sub.dtype == torch.bfloat16
                                and not torch.compiler.is_compiling()
                                and hasattr(pulsar_cuda, "fused_softcap_ce_forward")):
                            loss_sub = _FusedSoftcapCrossEntropyFunction.apply(
                                raw_logits_sub.view(-1, self.cfg.vocab_size),
                                lbl_sub.view(-1),
                                self.cfg.logit_softcap,
                                "mean",
                                -100,
                            )
                        elif raw_logits_sub.is_cuda:
                            loss_sub = _compiled_softcap_ce(
                                raw_logits_sub.view(-1, self.cfg.vocab_size),
                                lbl_sub.view(-1),
                                self.cfg.logit_softcap,
                                "mean",
                                -100,
                            )
                        else:
                            loss_sub = _softcap_cross_entropy(
                                raw_logits_sub.view(-1, self.cfg.vocab_size),
                                lbl_sub.view(-1),
                                self.cfg.logit_softcap,
                                "mean",
                                -100,
                            )

                    total_loss = total_loss + (pulse_weights[k] * loss_sub)
                    pulse_losses.append(loss_sub)
                else:
                    # Final pulse: always 100% of tokens (chunked to bound VRAM)
                    full_loss = self._chunked_ce(self.final_norm(h_pulse), labels)
                    total_loss = total_loss + (pulse_weights[k] * full_loss)
                    pulse_losses.append(full_loss)
        if labels is not None:
            # Add average router z-loss
            avg_z_loss = total_z_loss / (self.cfg.n_core * K)
            total_loss = total_loss + avg_z_loss

        if labels is None and logits is None:
            raw_logits = self.lm_head(self.final_norm(h_pulse))
            logits = self.cfg.logit_softcap * torch.tanh(raw_logits / self.cfg.logit_softcap)

        # Metrics stay as GPU tensors so that returning them costs no host sync. The
        # forward runs once per micro-batch (43x per optimizer step), and the previous
        # implementation called .item() ~20 times per forward - roughly 860 device
        # syncs per step, each one draining the pipeline. train.py materialises them
        # once per log window instead.
        max_bias = torch.stack([b.mlp.expert_bias.float().abs().max() for b in self.core]).max()
        tot_counts = global_expert_counts.sum().clamp_min(1.0)
        expert_pct = (global_expert_counts.float() / tot_counts) * 100.0

        # Per-pulse load. The aggregate above is what hid the imbalance: P1 routing
        # nearly uniformly while P2/P3 skewed looked balanced once averaged, and a
        # single global bias cannot correct for pulses that need different nudges.
        pulse_stacked = pulse_expert_counts                      # [K, E]
        pulse_tot = pulse_stacked.sum(dim=1, keepdim=True).clamp_min(1.0)
        expert_pct_per_pulse = (pulse_stacked.float() / pulse_tot) * 100.0

        # True per-(block, pulse) load. `expert_pcts` averages Expert e across blocks,
        # but those are separate routers, so a block collapsing onto one expert is
        # hidden by blocks that prefer another. Report the worst single-expert share and
        # the mean per-block spread instead.
        bp_tot = block_pulse_counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
        bp_pct = (block_pulse_counts.float() / bp_tot) * 100.0     # [n_core, K, E]
        worst_expert_share = bp_pct.max()                          # scalar percent
        per_block_spread = (bp_pct.max(dim=-1).values - bp_pct.min(dim=-1).values)  # [n_core, K]
        mean_block_spread = per_block_spread.mean()
        max_block_spread = per_block_spread.max()

        def as_tensor(value):
            return value if isinstance(value, torch.Tensor) else torch.tensor(float(value), device=input_ids.device)

        pulse_loss_t = [as_tensor(pl) for pl in pulse_losses]
        pulse_delta_t = [as_tensor(pd) for pd in pulse_deltas]
        zero = torch.zeros((), device=input_ids.device)

        metrics = {
            "num_pulses": K,
            # GPU tensors; train.py converts once per log window.
            "pulse_losses": pulse_loss_t,
            "pulse_deltas": pulse_delta_t,
            "pulse_final_delta": (self.pulse_state_norm(h_pulse) - h_0_norm).detach().norm(dim=-1).mean() if h_pulse is not None else zero,
            "router_confidence": gate_confidence_accum / (self.cfg.n_core * max(1, K)),
            "router_max_bias": max_bias,
            "expert_pcts": expert_pct,
            "expert_pcts_per_pulse": expert_pct_per_pulse,
            "worst_expert_share_pct": worst_expert_share,
            "mean_block_spread_pct": mean_block_spread,
            "max_block_spread_pct": max_block_spread,
            "input_mix_lambda": torch.sigmoid(self.input_mix_logit[min(1, K - 1)]).detach().mean(),
            # Legacy scalar accessors
            "loss_pulse_1": pulse_loss_t[0] if pulse_loss_t else zero,
            "loss_pulse_2": pulse_loss_t[1] if len(pulse_loss_t) >= 2 else (pulse_loss_t[0] if pulse_loss_t else zero),
            "loss_pulse_3": pulse_loss_t[-1] if pulse_loss_t else zero,
            "delta_pulse_1": pulse_delta_t[0] if pulse_delta_t else zero,
            "delta_pulse_2": pulse_delta_t[1] if len(pulse_delta_t) >= 2 else (pulse_delta_t[0] if pulse_delta_t else zero),
            "delta_pulse_3": pulse_delta_t[-1] if pulse_delta_t else zero,
        }
        # Router diagnostics (R-01 step 1): aggregated over (block, pulse) for this
        # forward only. Empty unless router_diag was requested.
        router_diag_metrics = {}
        if router_diag:
            captured = [
                entry
                for block in self.core
                for entry in (getattr(block.mlp, "_diag_capture", None) or [])
            ]
            for block in self.core:
                block.mlp._diag_capture = None
            if captured:
                modules = [b.mlp for b in self.core for _ in range(K)]
                per = [
                    mod.router_diagnostics(xf, lg, ti, pi)
                    for (pi, xf, lg, ti), mod in zip(captured, modules)
                ]
                router_diag_metrics = {
                    f"router_diag_{key}": torch.stack(
                        [d[key].reshape(()) for d in per]
                    ).mean()
                    for key in per[0]
                }
                metrics.update(router_diag_metrics)

        return logits, total_loss, metrics

    def get_model_summary(self, num_pulses: int | None = None) -> dict:
        K = num_pulses if num_pulses is not None else self.cfg.default_pulses
        total_params = sum(p.numel() for p in self.parameters())
        per_block_attn = (
            (self.cfg.n_heads + 2 * self.cfg.n_kv_heads) * self.cfg.head_dim * self.cfg.d_model
            + self.cfg.d_model * self.cfg.d_model
        )
        d_shared = self.cfg.d_shared if self.cfg.d_shared is not None else self.cfg.d_expert
        shared_mlp = 3 * self.cfg.d_model * d_shared
        routed_mlp = self.cfg.top_k * (3 * self.cfg.d_model * self.cfg.d_expert)
        non_embed_active = self.cfg.n_core * (per_block_attn + shared_mlp + routed_mlp)
        active_params = self.cfg.vocab_size * self.cfg.d_model + non_embed_active

        return {
            "model_name": "Pulsar",
            "total_params": total_params,
            "active_params": active_params,
            "non_embed_active_params": non_embed_active,
            "n_core": self.cfg.n_core,
            "num_pulses": K,
            "effective_layers": self.cfg.n_core * K,
            "d_model": self.cfg.d_model,
            "n_heads": self.cfg.n_heads,
            "n_kv_heads": self.cfg.n_kv_heads,
            "head_dim": self.cfg.head_dim,
            "d_expert": self.cfg.d_expert,
            "d_shared": d_shared,
            "n_routed_experts": self.cfg.n_routed_experts,
            "n_shared_experts": 1,
            "top_k": self.cfg.top_k,
            "active_ffn_dim": d_shared + self.cfg.top_k * self.cfg.d_expert,
            "vocab_size": self.cfg.vocab_size,
            "max_seq_len": self.cfg.max_seq_len,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        num_pulses: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively decode tokens using full recurrent depth (supports batched RL rollouts)."""
        was_training = self.training
        self.eval()
        device = input_ids.device
        eos_id = eos_token_id if eos_token_id is not None else self.cfg.eos_token_id

        curr_ids = input_ids.clone()
        # Per-row completion tracking. Previously only B==1 stopped at EOS, so batched
        # rollouts kept generating past the end of finished sequences.
        finished = torch.zeros(curr_ids.size(0), dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            # Use the full configured context, not just the base pretraining length:
            # capping at base_seq_len meant generate() could never exercise 32k.
            context_window = curr_ids[:, -self.cfg.max_seq_len:]
            logits, _, _ = self(context_window, labels=None, num_pulses=num_pulses)
            next_logits = logits[:, -1, :].clone()

            if repetition_penalty != 1.0:
                for b in range(curr_ids.size(0)):
                    seen = set(curr_ids[b].tolist())
                    for tok in seen:
                        if next_logits[b, tok] > 0:
                            next_logits[b, tok] /= repetition_penalty
                        else:
                            next_logits[b, tok] *= repetition_penalty

            if temperature > 0:
                next_logits = next_logits / temperature

                if top_k > 0:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = -float("Inf")

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    for b in range(curr_ids.size(0)):
                        indices_to_remove = sorted_indices[b, sorted_indices_to_remove[b]]
                        next_logits[b, indices_to_remove] = -float("Inf")

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            # Finished rows emit EOS rather than continuing to sample.
            next_token = torch.where(
                finished.unsqueeze(-1), torch.full_like(next_token, eos_id), next_token
            )
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            finished |= (next_token.squeeze(-1) == eos_id)
            if bool(finished.all()):
                break

        if was_training:
            self.train()
        return curr_ids

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "config.json")
        self.cfg.to_json(config_path)
        weights_path = os.path.join(save_directory, "model.pt")
        raw_model = getattr(self, "_orig_mod", self)
        torch.save(raw_model.state_dict(), weights_path)

    @classmethod
    def from_pretrained(cls, load_directory: str, device: str = "cpu", dtype: torch.dtype = torch.bfloat16) -> "PulsarForCausalLM":
        config_path = os.path.join(load_directory, "config.json")
        weights_path = os.path.join(load_directory, "model.pt")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found at: {config_path}")
        cfg = PulsarConfig.from_json(config_path)
        model = cls(cfg).to(device=device)
        model.to_training_precision(dtype)
        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=device, weights_only=True)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(
                    f"[Pulsar] from_pretrained: {len(missing)} missing, "
                    f"{len(unexpected)} unexpected keys"
                )
        return model
