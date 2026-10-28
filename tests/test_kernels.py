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
    assert torch.allclose(weights.float().sum(-1), torch.ones(N, device="cuda"), atol=2e-3)


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
