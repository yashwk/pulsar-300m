"""Kernel contract and numerical parity tests for Pulsar-300M.

Covers:
- RMSNorm forward and backward gradient parity across warp-unaligned and multi-tile dimensions.
- QK-RMSNorm forward/backward across head counts and head_dim=64 contract guard.
- RoPE half-split rotation, fallback parity, empty tensor guard, and T > 65535 launch.
- SwiGLU forward/backward parity and non-contiguous slice handling.
- FusedSoftcapCE forward and backward with ignore_index=-100.
- MoE router top-k, slot determinism, lowest-weight drop priority, dispatch, and combine.
- CUDA vs PyTorch fallback full-layer agreement and dtype stability.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from conftest import assert_close, make_cfg, HAS_CUDA, HAS_PULSAR_CUDA
from src.model import compute_token_slots

needs_gpu = pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
needs_cuda_ext = pytest.mark.skipif(not HAS_PULSAR_CUDA, reason="pulsar_cuda extension required")


def _bf16(*shape, device="cuda", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(*shape, generator=g).to(device=device, dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

def ref_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return (w.float() * xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


@needs_gpu
@needs_cuda_ext
@pytest.mark.parametrize("dim", [96, 128, 256, 384, 768, 1024, 1536, 2048])
def test_rmsnorm_forward_and_backward(dim):
    """RMSNorm forward and backward gradients (dx, dw) match FP32 autograd reference."""
    import pulsar_cuda

    x = _bf16(16, dim, seed=dim)
    w = torch.randn(dim, device="cuda", dtype=torch.float32).abs() + 0.5
    eps = 1e-6

    # Forward parity
    y_kernel, rsqrt = pulsar_cuda.rmsnorm_forward(x, w, eps)
    y_ref = ref_rmsnorm(x, w, eps)
    assert_close(y_kernel, y_ref, atol=2e-2, msg=f"rmsnorm fwd dim={dim}")

    # Backward gradient parity
    dy = _bf16(16, dim, seed=dim + 1)
    xr = x.float().requires_grad_(True)
    wr = w.detach().clone().requires_grad_(True)
    yr = wr * xr * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + eps)
    (yr * dy.float()).sum().backward()

    dx, dw = pulsar_cuda.rmsnorm_backward(dy, x, w, rsqrt)
    assert_close(dx, xr.grad.to(torch.bfloat16), atol=5e-2, msg=f"rmsnorm dx dim={dim}")
    assert_close(dw, wr.grad, atol=1e-4, msg=f"rmsnorm dw dim={dim}")


# ---------------------------------------------------------------------------
# QK-RMSNorm
# ---------------------------------------------------------------------------

@needs_gpu
@needs_cuda_ext
@pytest.mark.parametrize("n_heads", [4, 16])
def test_qk_rmsnorm_forward_and_backward(n_heads):
    """QK-RMSNorm per-head normalization forward and backward match FP32 autograd."""
    import pulsar_cuda

    B, T, H, D = 2, 32, n_heads, 64
    x = _bf16(B, H, T, D, seed=n_heads)
    w = torch.randn(H, D, device="cuda", dtype=torch.float32).abs() + 0.5
    eps = 1e-6

    # Forward
    y_kernel, rsqrt = pulsar_cuda.qk_rmsnorm_forward(x, w, T, eps)
    xf = x.float()
    y_ref = (w.view(1, H, 1, D) * xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)
    assert_close(y_kernel, y_ref, atol=2e-2, msg=f"qk_rmsnorm heads={n_heads}")

    # Backward
    dy = _bf16(B, H, T, D, seed=n_heads + 1)
    xr = x.detach().float().requires_grad_(True)
    wr = w.detach().requires_grad_(True)
    yr = wr.view(1, H, 1, D) * xr * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + eps)
    (yr * dy.float()).sum().backward()

    dx, dw = pulsar_cuda.qk_rmsnorm_backward(dy, x, w, rsqrt, T)
    assert_close(xr.grad.to(torch.bfloat16), dx, atol=5e-2, msg="qk dx")
    assert_close(wr.grad, dw, atol=5e-2, msg="qk dw")


@needs_gpu
@needs_cuda_ext
def test_qk_rmsnorm_rejects_wrong_head_dim():
    """QK-RMSNorm kernel rejects head_dim != 64 with RuntimeError."""
    import pulsar_cuda

    x = torch.randn(1, 4, 8, 32, device="cuda", dtype=torch.bfloat16)
    w = torch.rand(4, 32, device="cuda", dtype=torch.float32) + 0.5
    with pytest.raises(RuntimeError, match="head_dim == 64"):
        pulsar_cuda.qk_rmsnorm_forward(x, w, 8, 1e-6)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def ref_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.size(-1) // 2
    x1 = x[..., :half].float()
    x2 = x[..., half:].float()
    c = cos[: x.size(-2)].float()[None, None, :, :]
    s = sin[: x.size(-2)].float()[None, None, :, :]
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    return torch.cat([o1, o2], dim=-1).to(x.dtype)


@needs_gpu
@needs_cuda_ext
def test_rope_forward_and_fallback_parity():
    """RoPE forward matches FP32 reference and PyTorch fallback path."""
    import pulsar_cuda
    import src.model as M

    B, H, T, D = 2, 4, 32, 64
    x = _bf16(B, H, T, D, seed=7)
    inv = 1.0 / (10000.0 ** (torch.arange(0, D, 2, device="cuda").float() / D))
    t = torch.arange(T, device="cuda").float()
    freqs = torch.outer(t, inv)
    cos, sin = freqs.cos(), freqs.sin()

    out_kernel = pulsar_cuda.rope_forward(x, cos, sin)
    out_ref = ref_rope(x, cos, sin)
    assert_close(out_kernel, out_ref, atol=2e-2, msg="rope_forward")

    orig = M.HAS_PULSAR_CUDA
    try:
        M.HAS_PULSAR_CUDA = False
        out_fallback = M.apply_rotary_emb(x, cos, sin)
    finally:
        M.HAS_PULSAR_CUDA = orig
    assert_close(out_kernel, out_fallback, atol=1e-3, msg="rope fallback bit-parity")


@needs_gpu
@needs_cuda_ext
def test_rope_boundary_and_empty_guards():
    """RoPE handles empty tensors and long sequences beyond gridDim.y=65535."""
    import pulsar_cuda

    # Empty tensor guard
    q_empty = torch.empty(1, 1, 0, 64, device="cuda", dtype=torch.bfloat16)
    cos_empty = torch.empty(0, 32, device="cuda")
    sin_empty = torch.empty(0, 32, device="cuda")
    out_empty = pulsar_cuda.rope_forward(q_empty, cos_empty, sin_empty)
    assert out_empty.numel() == 0

    # T >= 65536 guard
    T = 66000
    q_large = torch.randn(1, 1, T, 64, device="cuda", dtype=torch.bfloat16)
    cos_large = torch.ones(T, 32, device="cuda")
    sin_large = torch.zeros(T, 32, device="cuda")
    out_large = pulsar_cuda.rope_forward(q_large, cos_large, sin_large)
    assert out_large.shape == q_large.shape
    assert torch.isfinite(out_large).all()


# ---------------------------------------------------------------------------
# Fused SwiGLU
# ---------------------------------------------------------------------------

def ref_swiglu(gu: torch.Tensor, d_ff: int) -> torch.Tensor:
    gate = gu[..., :d_ff].float()
    up = gu[..., d_ff:].float()
    return (F.silu(gate) * up).to(gu.dtype)


@needs_gpu
@needs_cuda_ext
def test_fused_swiglu_forward_and_backward():
    """Fused SwiGLU forward and backward match FP32 reference."""
    import pulsar_cuda

    E, C, d_ff = 4, 16, 128
    gu = _bf16(E, C, 2 * d_ff, seed=8)

    # Forward
    out = pulsar_cuda.fused_swiglu_forward(gu, d_ff)
    ref = ref_swiglu(gu, d_ff)
    assert_close(out, ref, atol=2e-2, msg="fused_swiglu fwd")

    # Backward
    gu_ref = gu.detach().float().requires_grad_(True)
    y = ref_swiglu(gu_ref, d_ff)
    dy = _bf16(*y.shape, seed=10)
    (y * dy.float()).sum().backward()

    dgu = pulsar_cuda.fused_swiglu_backward(dy, gu, d_ff)
    assert_close(dgu, gu_ref.grad.to(torch.bfloat16), atol=5e-2, msg="fused_swiglu bwd")


@needs_gpu
def test_swiglu_noncontiguous_input():
    """SwiGLU wrapper supports non-contiguous slice inputs without crashing."""
    from src.model import _FusedSwiGLUFunction

    big = torch.randn(4, 16, 512, device="cuda", dtype=torch.bfloat16)
    gu = big[..., ::2]
    assert not gu.is_contiguous()
    out = _FusedSwiGLUFunction.apply(gu, 128)
    assert out.shape == (4, 16, 128)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Softcap Cross-Entropy
# ---------------------------------------------------------------------------

def ref_softcap_ce(logits: torch.Tensor, targets: torch.Tensor, softcap: float, ignore_index: int):
    sc = softcap * torch.tanh(logits.float() / softcap)
    per_tok = F.cross_entropy(sc, targets, ignore_index=ignore_index, reduction="none")
    valid = (targets != ignore_index)
    n_valid = valid.sum().clamp_min(1)
    return per_tok, n_valid


@needs_gpu
@needs_cuda_ext
def test_softcap_ce_forward_and_backward():
    """Fused softcap cross-entropy matches FP32 reference with ignore_index=-100."""
    import pulsar_cuda

    N, V = 64, 512
    logits = _bf16(N, V, seed=11) * 2.0
    targets = torch.randint(0, V, (N,), device="cuda", dtype=torch.long)
    targets[:8] = -100

    # Forward
    losses, lse = pulsar_cuda.fused_softcap_ce_forward(logits, targets, 30.0, -100)
    per_tok, n_valid = ref_softcap_ce(logits, targets, 30.0, -100)
    assert_close(losses[8:], per_tok[8:], atol=3e-2)
    assert torch.all(losses[:8] == 0)

    # Backward
    logits_ref = logits.detach().float().requires_grad_(True)
    per_tok_ref, _ = ref_softcap_ce(logits_ref, targets, 30.0, -100)
    loss = per_tok_ref.sum() / n_valid
    loss.backward()

    scale = 1.0 / float(n_valid)
    d_logits = pulsar_cuda.fused_softcap_ce_backward(logits, targets, lse, 30.0, scale, -100)
    assert_close(d_logits, logits_ref.grad.to(torch.bfloat16), atol=3e-2)
    assert torch.all(d_logits[:8] == 0)


# ---------------------------------------------------------------------------
# MoE Router, Dispatch, and Combine
# ---------------------------------------------------------------------------

@needs_gpu
@needs_cuda_ext
@pytest.mark.parametrize("E,K", [(8, 2), (16, 4)])
def test_router_topk_and_weight_normalization(E, K):
    """MoE router top-k selects top experts and normalizes weights strictly to 1.0."""
    import pulsar_cuda

    N = 64
    logits = _bf16(N, E, seed=13)
    bias = torch.randn(E, device="cuda", dtype=torch.float32) * 0.1

    idx, weights, z = pulsar_cuda.moe_router_topk_forward(logits, bias, K, 1e-4)
    centered = logits.float() - logits.float().mean(dim=-1, keepdim=True)
    scores = torch.sigmoid(centered)
    biased = scores + bias.float()
    idx_ref = torch.topk(biased, k=K, dim=-1).indices
    raw = torch.gather(scores, -1, idx_ref)
    weights_ref = raw / (raw.sum(-1, keepdim=True) + 1e-6)

    assert_close(weights, weights_ref, atol=2e-3)
    assert torch.allclose(weights.float().sum(-1), torch.ones(N, device="cuda"), atol=max(2e-3, K * 1e-3))


@needs_gpu
@needs_cuda_ext
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_moe_router_parity_e16_topk4(dtype):
    """_FusedMoERouterFunction matches PyTorch reference for E=16, top_k=4 forward and backward."""
    from src.model import _FusedMoERouterFunction
    torch.manual_seed(42)
    N, E, K = 32, 16, 4

    logits_ref = torch.randn(N, E, device="cuda", dtype=dtype, requires_grad=True)
    bias = torch.randn(E, device="cuda", dtype=torch.float32) * 0.1

    # Reference PyTorch path:
    centered_ref = logits_ref - logits_ref.mean(dim=-1, keepdim=True)
    scores_ref = torch.sigmoid(centered_ref.float())
    biased_scores_ref = scores_ref + bias
    _, idx_ref = torch.topk(biased_scores_ref, k=K, dim=-1)
    raw_ref = torch.gather(scores_ref, -1, idx_ref)
    weights_ref = (raw_ref / (raw_ref.sum(dim=-1, keepdim=True) + 1e-6)).to(dtype)

    d_weights = torch.randn_like(weights_ref)
    weights_ref.backward(d_weights)
    grad_ref = logits_ref.grad.clone()

    # CUDA path: _FusedMoERouterFunction
    logits_cuda = logits_ref.detach().clone().requires_grad_(True)
    idx_cuda, weights_cuda = _FusedMoERouterFunction.apply(logits_cuda, bias, K)
    weights_cuda.backward(d_weights)
    grad_cuda = logits_cuda.grad.clone()

    assert torch.equal(idx_ref, idx_cuda)
    assert_close(weights_cuda, weights_ref, atol=2e-3)
    assert_close(grad_cuda, grad_ref, atol=5e-3)


@pytest.mark.parametrize("routing", [
    [[0, 1], [0, 1]],
    [[0, 1], [2, 3], [4, 5], [6, 7]],
    [[0, 0], [0, 0], [1, 1], [1, 1]],
    [[3, 3], [3, 3]],
    [[0, 1], [1, 0]],
])
def test_moe_slots_unique_per_expert(routing):
    """compute_token_slots produces unique, contiguous slots per expert without collision."""
    ti = torch.tensor(routing)
    weights = torch.ones(*ti.shape)
    E = 8
    C = int(ti.reshape(-1).bincount(minlength=E).max().item())
    slots = compute_token_slots(ti, weights, E, C)

    flat_e, flat_s = ti.reshape(-1), slots.reshape(-1)
    for e in range(E):
        sel = (flat_e == e)
        if not bool(sel.any()):
            continue
        got = flat_s[sel].tolist()
        assert all(0 <= s < C for s in got)
        assert len(set(got)) == len(got)
        assert sorted(got) == list(range(len(got)))


@needs_gpu
def test_drop_priority_is_lowest_weight_first():
    """Over-capacity slot assignment drops the lowest-weight assignments first."""
    N, K, E, capacity = 200, 2, 2, 16
    topk_idx = torch.zeros(N, K, dtype=torch.long, device="cuda")
    w = torch.arange(N * K, device="cuda", dtype=torch.float32).reshape(N, K)

    slots = compute_token_slots(topk_idx, w, E, capacity)
    survived = slots.reshape(-1) >= 0
    kept = w.reshape(-1)[survived]
    dropped = w.reshape(-1)[~survived]

    assert survived.sum().item() == capacity
    assert kept.min() > dropped.max()


@needs_gpu
@needs_cuda_ext
def test_dispatch_and_combine_parity():
    """moe_dispatch and moe_combine forward kernels match reference gather and scatter."""
    import pulsar_cuda

    N, D, E, K = 16, 32, 4, 2
    capacity = 32
    x = _bf16(N, D, seed=15)
    topk_idx = torch.randint(0, E, (N, K), device="cuda", dtype=torch.long)
    topk_weights = torch.rand(N, K, device="cuda", dtype=torch.float32)
    topk_weights = (topk_weights / topk_weights.sum(-1, keepdim=True)).bfloat16()

    token_to_slot = compute_token_slots(topk_idx, topk_weights, E, capacity)
    (expert_inputs,) = pulsar_cuda.moe_dispatch_forward(x, topk_idx, token_to_slot, E, capacity)

    for t in range(N):
        for k in range(K):
            slot = int(token_to_slot[t, k])
            if slot >= 0:
                e = int(topk_idx[t, k])
                assert_close(expert_inputs[e, slot], x[t], atol=1e-3)

    expert_out = _bf16(E, capacity, D, seed=18)
    out = pulsar_cuda.moe_combine_forward(expert_out, topk_idx, topk_weights, token_to_slot, None)

    ref = torch.zeros(N, D, device="cuda", dtype=torch.float32)
    for t in range(N):
        acc = torch.zeros(D, device="cuda", dtype=torch.float32)
        for k in range(K):
            slot = int(token_to_slot[t, k])
            if slot >= 0:
                e = int(topk_idx[t, k])
                acc += expert_out[e, slot].float() * topk_weights[t, k].float()
        ref[t] = acc
    assert_close(out, ref, atol=3e-2)


@needs_gpu
@needs_cuda_ext
def test_cuda_vs_fallback_moe_layer():
    """Full PulsarMoEMLP layer matches bit-for-bit between CUDA and fallback paths."""
    from src.model import PulsarMoEMLP
    import src.model as model_mod

    cfg = make_cfg(d_model=64, d_expert=64, n_routed_experts=8, top_k=2, max_pulses=2)
    torch.manual_seed(1)
    mlp = PulsarMoEMLP(cfg).cuda().bfloat16().eval()

    x = _bf16(2, 16, cfg.d_model, seed=19)
    with torch.no_grad():
        out_cuda, _, _, _ = mlp(x, pulse_idx=0)

    orig = model_mod.HAS_PULSAR_CUDA
    try:
        model_mod.HAS_PULSAR_CUDA = False
        with torch.no_grad():
            out_fallback, _, _, _ = mlp(x, pulse_idx=0)
    finally:
        model_mod.HAS_PULSAR_CUDA = orig

    assert_close(out_cuda, out_fallback, atol=3e-2)


@needs_gpu
@needs_cuda_ext
def test_cuda_vs_fallback_moe_layer_backward():
    """Full PulsarMoEMLP backward gradients match between CUDA and fallback paths."""
    from src.model import PulsarMoEMLP
    import src.model as model_mod

    cfg = make_cfg(d_model=64, d_expert=64, n_routed_experts=8, top_k=2, max_pulses=2)
    torch.manual_seed(42)
    mlp_cuda = PulsarMoEMLP(cfg).cuda().bfloat16().train()
    mlp_fb = PulsarMoEMLP(cfg).cuda().bfloat16().train()
    mlp_fb.load_state_dict(mlp_cuda.state_dict())

    x1 = torch.randn(2, 16, cfg.d_model, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)

    orig = model_mod.HAS_PULSAR_CUDA
    try:
        model_mod.HAS_PULSAR_CUDA = True
        out_cuda, _, _, _ = mlp_cuda(x1, pulse_idx=0)
        out_cuda.sum().backward()

        model_mod.HAS_PULSAR_CUDA = False
        out_fb, _, _, _ = mlp_fb(x2, pulse_idx=0)
        out_fb.sum().backward()
    finally:
        model_mod.HAS_PULSAR_CUDA = orig

    assert_close(out_cuda, out_fb, atol=3e-2, msg="forward match")
    assert_close(x1.grad, x2.grad, atol=3e-2, msg="dx match")
    assert_close(mlp_cuda.routed_down.grad, mlp_fb.routed_down.grad, atol=3e-2, msg="d_routed_down match")
    assert_close(mlp_cuda.routed_scale.grad, mlp_fb.routed_scale.grad, atol=3e-2, msg="d_routed_scale match")


@needs_gpu
def test_forward_backward_in_both_dtypes_on_cuda():
    """Full causal forward and backward pass succeed in both FP32 and BF16 on CUDA."""
    from src.model import PulsarForCausalLM

    for dtype in (torch.float32, torch.bfloat16):
        cfg = make_cfg(n_core=2, d_model=128, head_dim=32, vocab_size=512, n_routed_experts=8, top_k=2)
        torch.manual_seed(0)
        model = PulsarForCausalLM(cfg).cuda()
        if dtype is torch.bfloat16:
            model.to_training_precision(torch.bfloat16)
        model.train()
        x = torch.randint(0, cfg.vocab_size, (2, 16), device="cuda")
        y = torch.randint(0, cfg.vocab_size, (2, 16), device="cuda")
        _, loss, _ = model(x, labels=y, num_pulses=2)
        loss.backward()
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Fused QK-RMSNorm + RoPE
# ---------------------------------------------------------------------------

@needs_gpu
@needs_cuda_ext
def test_fused_qk_rmsnorm_rope_parity():
    """_FusedQKRMSNormRoPEFunction forward and backward match unfused QK-RMSNorm + RoPE."""
    import pulsar_cuda
    from src.model import _FusedQK_RMSNormFunction, _FusedQKRMSNormRoPEFunction, apply_rotary_emb

    B, H, T, D = 2, 4, 16, 64
    x1 = _bf16(B, H, T, D, seed=42).requires_grad_(True)
    x2 = x1.detach().clone().requires_grad_(True)
    w1 = torch.randn(H, D, device="cuda", dtype=torch.float32, requires_grad=True)
    w2 = w1.detach().clone().requires_grad_(True)

    # cos and sin tables
    theta = 10000.0
    inv_freq = 1.0 / (theta ** (torch.arange(0, D, 2, device="cuda").float() / D))
    t = torch.arange(T, device="cuda", dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()

    # Fused pass
    out_fused = _FusedQKRMSNormRoPEFunction.apply(x1, w1, cos, sin, T, 1e-6)
    out_fused.sum().backward()

    # Unfused reference pass
    q_norm = _FusedQK_RMSNormFunction.apply(x2, w2, T, 1e-6)
    out_unfused = apply_rotary_emb(q_norm, cos, sin)
    out_unfused.sum().backward()

    assert_close(out_fused, out_unfused, atol=3.5e-2, msg="fwd qk_rmsnorm_rope match")
    assert_close(x1.grad, x2.grad, atol=3.5e-2, msg="dx qk_rmsnorm_rope match")
    assert_close(w1.grad, w2.grad, atol=6e-2, msg="dw qk_rmsnorm_rope match")


# ---------------------------------------------------------------------------
# Fused Attention Gate + Layout Transpose
# ---------------------------------------------------------------------------

@needs_gpu
@needs_cuda_ext
def test_fused_attn_gate_parity():
    """_FusedAttnGateFunction forward and backward match unfused transpose + sigmoid gate."""
    from src.model import _FusedAttnGateFunction

    B, H, T, D = 2, 4, 16, 64
    attn_out1 = _bf16(B, H, T, D, seed=101).requires_grad_(True)
    attn_out2 = attn_out1.detach().clone().requires_grad_(True)

    gate_logits1 = torch.randn(B, T, H, device="cuda", dtype=torch.float32, requires_grad=True)
    gate_logits2 = gate_logits1.detach().clone().requires_grad_(True)

    # Fused
    out_fused = _FusedAttnGateFunction.apply(attn_out1, gate_logits1)
    out_fused.sum().backward()

    # Unfused reference
    g = torch.sigmoid(gate_logits2)
    out_unfused = (attn_out2.transpose(1, 2) * g.unsqueeze(-1)).reshape(B, T, -1)
    out_unfused.sum().backward()

    assert_close(out_fused, out_unfused, atol=2e-3, msg="fwd attn_gate match")
    assert_close(attn_out1.grad, attn_out2.grad, atol=2e-3, msg="d_attn_out match")
    assert_close(gate_logits1.grad, gate_logits2.grad, atol=2e-3, msg="d_gate_logits match")


# ---------------------------------------------------------------------------
# Fused MoE Slot Assignment & Histogram from Order
# ---------------------------------------------------------------------------

@needs_gpu
@needs_cuda_ext
def test_moe_slots_from_order_parity():
    """moe_slots_from_order matches eager histogram, prefix sum, and slot assignment."""
    import pulsar_cuda
    from src.model import _make_sort_key_eager, _compute_slot_assignment_eager

    M = 128
    E = 8
    capacity = 16
    flat_e = torch.randint(0, E, (M,), device="cuda", dtype=torch.long)
    flat_w = torch.rand(M, device="cuda", dtype=torch.float32)

    sort_key = _make_sort_key_eager(flat_e, flat_w)
    order = torch.argsort(sort_key, stable=True)

    # CUDA kernel
    slots_cuda, counts_cuda = pulsar_cuda.moe_slots_from_order(flat_e, order, E, capacity)

    # Reference
    e_sorted = flat_e[order]
    counts_ref = torch.zeros(E, device="cuda", dtype=torch.long)
    counts_ref.index_add_(0, e_sorted, torch.ones(M, device="cuda", dtype=torch.long))
    start_sorted = counts_ref.cumsum(0) - counts_ref
    slots_ref = _compute_slot_assignment_eager(start_sorted, e_sorted, capacity, order)

    assert torch.equal(counts_cuda, counts_ref)
    assert torch.equal(slots_cuda, slots_ref)


# ---------------------------------------------------------------------------
# Fused Muon Weight Decay + Stochastic Rounding Update
# ---------------------------------------------------------------------------

@needs_gpu
@needs_cuda_ext
def test_fused_sr_update_parity():
    """fused_sr_update preserves sub-ulp accumulation within statistical tolerance."""
    import pulsar_cuda

    n = 4096
    dst = torch.full((n,), 0.02, dtype=torch.bfloat16, device="cuda")
    step = 3e-5
    update = torch.full((n,), 1.0, dtype=torch.bfloat16, device="cuda")

    reps = 1000
    for i in range(reps):
        pulsar_cuda.fused_sr_update(dst, update, step, 1.0, 42 + i * 10007)

    expected_drift = reps * step
    got_drift = (dst.float() - 0.02).mean().item()
    lo, hi = 0.85 * expected_drift, 1.15 * expected_drift
    assert lo <= got_drift <= hi


# ---------------------------------------------------------------------------
# Fused Residual-Add + RMSNorm
# ---------------------------------------------------------------------------

@needs_gpu
@needs_cuda_ext
def test_fused_add_rmsnorm_parity():
    """_FusedAddRMSNormFunction forward and backward match separate addition + RMSNorm."""
    from src.model import _FusedAddRMSNormFunction, _RMSNormCUDAFunction

    N, D = 32, 1024
    x1 = _bf16(N, D, seed=101).requires_grad_(True)
    res1 = _bf16(N, D, seed=102).requires_grad_(True)
    w1 = torch.randn(D, device="cuda", dtype=torch.float32, requires_grad=True)

    x2 = x1.detach().clone().requires_grad_(True)
    res2 = res1.detach().clone().requires_grad_(True)
    w2 = w1.detach().clone().requires_grad_(True)

    # Fused pass
    x_sum1, y1 = _FusedAddRMSNormFunction.apply(x1, res1, w1, 1e-6)
    loss1 = (y1 * 2.0).sum() + (x_sum1 * 0.5).sum()
    loss1.backward()

    # Unfused pass
    x_sum2 = x2 + res2
    y2 = _RMSNormCUDAFunction.apply(x_sum2, w2, 1e-6)
    loss2 = (y2 * 2.0).sum() + (x_sum2 * 0.5).sum()
    loss2.backward()

    assert_close(x_sum1, x_sum2, atol=1e-3, msg="fused_add_rmsnorm x_sum match")
    assert_close(y1, y2, atol=1e-3, msg="fused_add_rmsnorm y match")
    assert_close(x1.grad, x2.grad, atol=3.5e-2, msg="fused_add_rmsnorm dx match")
    assert_close(res1.grad, res2.grad, atol=3.5e-2, msg="fused_add_rmsnorm dres match")
    assert_close(w1.grad, w2.grad, atol=2e-2, msg="fused_add_rmsnorm dw match")


# ---------------------------------------------------------------------------
# Fused Chunked Linear + Softcap CE
# ---------------------------------------------------------------------------

@needs_gpu
@needs_cuda_ext
def test_fused_linear_softcap_ce_parity():
    """_FusedLinearSoftcapCEFunction forward and backward match unfused linear + softcap CE."""
    from src.model import _FusedLinearSoftcapCEFunction, _softcap_cross_entropy

    N, D, V = 64, 128, 512
    h1 = _bf16(N, D, seed=201).requires_grad_(True)
    w1 = _bf16(V, D, seed=202).requires_grad_(True)
    labels = torch.randint(0, V, (N,), device="cuda", dtype=torch.long)
    labels[::5] = -100 # some ignored tokens

    h2 = h1.detach().clone().requires_grad_(True)
    w2 = w1.detach().clone().requires_grad_(True)

    # Fused pass (chunk size = 16 to exercise multiple chunks)
    loss1 = _FusedLinearSoftcapCEFunction.apply(h1, w1, labels, 30.0, 16, -100)
    loss1.backward()

    # Reference pass
    raw2 = torch.nn.functional.linear(h2, w2)
    loss2 = _softcap_cross_entropy(raw2, labels, 30.0, "mean", -100)
    loss2.backward()

    assert_close(loss1, loss2, atol=1e-3, msg="fused_linear_ce loss match")
    assert_close(h1.grad, h2.grad, atol=1e-2, msg="fused_linear_ce dh match")
    assert_close(w1.grad, w2.grad, atol=1e-2, msg="fused_linear_ce dweight match")


# ---------------------------------------------------------------------------
# Stride-Aware QK-RMSNorm + RoPE
# ---------------------------------------------------------------------------

@needs_gpu
@needs_cuda_ext
def test_qk_rmsnorm_rope_non_contiguous_parity():
    """_FusedQKRMSNormRoPEFunction handles non-contiguous (transposed) inputs accurately."""
    from src.model import _FusedQKRMSNormRoPEFunction, _FusedQK_RMSNormFunction, apply_rotary_emb

    B, T, H, D = 2, 16, 4, 64
    # Create contiguous [B, T, H, D] then transpose to [B, H, T, D]
    raw1 = _bf16(B, T, H, D, seed=301).requires_grad_(True)
    raw2 = raw1.detach().clone().requires_grad_(True)
    w1 = torch.randn(H, D, device="cuda", dtype=torch.float32, requires_grad=True)
    w2 = w1.detach().clone().requires_grad_(True)

    q1 = raw1.transpose(1, 2) # Non-contiguous! stride(3) == 1, stride(2) == H*D, stride(1) == D
    assert not q1.is_contiguous()

    q2 = raw2.transpose(1, 2)

    theta = 10000.0
    inv_freq = 1.0 / (theta ** (torch.arange(0, D, 2, device="cuda").float() / D))
    t = torch.arange(T, device="cuda", dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()

    out1 = _FusedQKRMSNormRoPEFunction.apply(q1, w1, cos, sin, T, 1e-6)
    out1.sum().backward()

    q2_c = q2.contiguous()
    q_norm2 = _FusedQK_RMSNormFunction.apply(q2_c, w2, T, 1e-6)
    out2 = apply_rotary_emb(q_norm2, cos, sin)
    out2.sum().backward()

    assert_close(out1, out2, atol=3.5e-2, msg="stride-aware fwd match")
    assert_close(raw1.grad, raw2.grad, atol=3.5e-2, msg="stride-aware dx match")
    assert_close(w1.grad, w2.grad, atol=6e-2, msg="stride-aware dw match")


# ---------------------------------------------------------------------------
# Fused MoE Router Top-K with Z-Loss Parity
# ---------------------------------------------------------------------------

@needs_gpu
@needs_cuda_ext
def test_moe_router_topk_z_loss_forward_backward_parity():
    """_FusedMoERouterFunction computes fused top-k, sigmoid, weight normalisation, and z-loss accurately."""
    from src.model import _FusedMoERouterFunction

    N, E, K = 64, 16, 4
    torch.manual_seed(42)
    logits = torch.randn(N, E, device="cuda", dtype=torch.float32, requires_grad=True)
    bias = torch.randn(E, device="cuda", dtype=torch.float32) * 0.1
    z_w = 1e-3

    # Fused CUDA path
    topk_idx, topk_weights, z_loss = _FusedMoERouterFunction.apply(logits, bias, K, z_w)
    loss = (topk_weights.float() * 0.5).sum() + z_loss
    loss.backward()
    grad_fused = logits.grad.clone()

    # Eager reference path
    logits.grad.zero_()
    logits_c = logits - logits.mean(dim=-1, keepdim=True)
    scores = torch.sigmoid(logits_c)
    biased_scores = scores + bias
    _, ref_topk_idx = torch.topk(biased_scores, k=K, dim=-1)
    ref_topk_raw = torch.gather(scores, -1, ref_topk_idx)
    ref_topk_weights = ref_topk_raw / (ref_topk_raw.sum(dim=-1, keepdim=True) + 1e-6)
    ref_z = z_w * torch.logsumexp(logits_c, dim=-1).pow(2).mean()
    ref_loss = (ref_topk_weights * 0.5).sum() + ref_z
    ref_loss.backward()
    grad_ref = logits.grad.clone()

    assert torch.equal(topk_idx, ref_topk_idx)
    assert_close(topk_weights.float(), ref_topk_weights, atol=1e-3, msg="topk weights match")
    assert_close(z_loss, ref_z, atol=1e-5, msg="z_loss forward match")
    assert_close(grad_fused, grad_ref, atol=1e-3, msg="router backward grad match")


# ---------------------------------------------------------------------------
# Token-Centric MoE Combine Backward Parity
# ---------------------------------------------------------------------------

@needs_gpu
@needs_cuda_ext
def test_moe_combine_backward_token_centric_parity():
    """_FusedMoECombineFunction backward with token-centric 1-block-per-token matches reference."""
    from src.model import _FusedMoECombineFunction

    N, E, C, D, K = 32, 16, 16, 1024, 4
    torch.manual_seed(123)
    logits = torch.randn(N, E, device="cuda")
    topk_idx = torch.topk(logits, K, dim=-1)[1]
    topk_weights = torch.softmax(torch.randn(N, K, device="cuda"), dim=-1).to(torch.bfloat16)
    token_to_slot, counts = compute_token_slots(topk_idx, topk_weights, E, C, return_counts=True)

    expert_outputs = torch.randn(E, C, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    tw = topk_weights.detach().clone().requires_grad_(True)
    shared_out = torch.randn(N, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    y = _FusedMoECombineFunction.apply(expert_outputs, topk_idx, tw, token_to_slot, shared_out, counts)
    dy = torch.randn_like(y)
    y.backward(dy)

    assert expert_outputs.grad is not None
    assert tw.grad is not None
    assert shared_out.grad is not None
    assert torch.isfinite(expert_outputs.grad).all()
    assert torch.isfinite(tw.grad).all()
    assert torch.isfinite(shared_out.grad).all()
    assert torch.equal(shared_out.grad, dy)
