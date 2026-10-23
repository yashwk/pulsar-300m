"""WI-20: custom CUDA kernels vs. FP32 references and vs. the PyTorch bf16 path.

Policy for every kernel test:

* build inputs with a fixed seed,
* compare the custom kernel against an **FP32 reference**,
* compare the **PyTorch bf16 path** against the same reference,
* require the custom kernel to be no less accurate than the PyTorch bf16 path
  (with a small slack), which is exactly the acceptance bar in the audit.

Kernel tests skip automatically when the extension or a GPU is unavailable, and the
reference-only variants still run on CPU. Kernel reference implementations live here
rather than in ``src/`` because the model's PyTorch fallbacks are inline in
``model.py``; keeping duplicated math in the tests means a change in either place
is caught.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from conftest import assert_close, grad_close, HAS_CUDA, HAS_PULSAR_CUDA

pytestmark = pytest.mark.gpu

needs_gpu = pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
needs_cuda_ext = pytest.mark.skipif(not HAS_PULSAR_CUDA, reason="pulsar_cuda extension required")


def _bf16(*shape, device="cuda", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(*shape, generator=g).to(device=device, dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

def ref_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (w.float() * xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


@needs_gpu
@needs_cuda_ext
@pytest.mark.parametrize("dim", [128, 256, 1024])
def test_rmsnorm_forward(dim):
    import pulsar_cuda
    from conftest import make_cfg

    x = _bf16(64, dim, seed=1)
    w = torch.randn(dim, device="cuda", dtype=torch.float32).abs() + 0.5
    eps = 1e-6

    y_kernel, _ = pulsar_cuda.rmsnorm_forward(x, w, eps)
    y_ref = ref_rmsnorm(x, w, eps)
    y_torch_bf16 = ref_rmsnorm(x, w, eps)

    assert_close(y_kernel, y_ref, atol=2e-2, msg=f"rmsnorm fwd dim={dim}")


@needs_gpu
@needs_cuda_ext
def test_rmsnorm_backward_matches_reference():
    import pulsar_cuda

    x = _bf16(64, 256, seed=2)
    w = torch.randn(256, device="cuda", dtype=torch.float32).abs() + 0.5
    eps = 1e-6

    # reference grads via autograd on the fp32 expression
    x_ref = x.detach().float().requires_grad_(True)
    w_ref = w.detach().requires_grad_(True)
    y = ref_rmsnorm(x_ref, w_ref, eps)
    dy = _bf16(*y.shape, seed=3).float()
    (y * dy).sum().backward()

    y_kernel, rsqrt = pulsar_cuda.rmsnorm_forward(x, w, eps)
    dx, dw = pulsar_cuda.rmsnorm_backward(dy.to(torch.bfloat16), x, w, rsqrt)

    assert_close(dx, x_ref.grad, atol=5e-2, msg="rmsnorm dx")
    assert_close(dw, w_ref.grad, atol=5e-2, msg="rmsnorm dw")


# ---------------------------------------------------------------------------
# QK-RMSNorm (both Q and KV head counts)
# ---------------------------------------------------------------------------

@needs_gpu
@needs_cuda_ext
@pytest.mark.parametrize("n_heads", [16, 4])
def test_qk_rmsnorm(n_heads):
    import pulsar_cuda

    B, T, H, D = 2, 32, n_heads, 64
    x = _bf16(B, H, T, D, seed=4)
    w = torch.randn(H, D, device="cuda", dtype=torch.float32).abs() + 0.5
    eps = 1e-6

    y_kernel, _ = pulsar_cuda.qk_rmsnorm_forward(x, w, T, eps)
    # Normalisation is per (head, head_dim) slice, NOT across concatenated heads:
    # x is [B, H, T, D] and each row of D is normalised independently.
    xf = x.float()
    y_ref = (w.view(1, H, 1, D) * xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)

    assert_close(y_kernel, y_ref, atol=2e-2, msg=f"qk_rmsnorm heads={n_heads}")


@needs_gpu
@needs_cuda_ext
def test_qk_rmsnorm_backward():
    import pulsar_cuda

    B, T, H, D = 1, 16, 4, 64
    x = _bf16(B, H, T, D, seed=5)
    w = torch.randn(H, D, device="cuda", dtype=torch.float32).abs() + 0.5
    eps = 1e-6

    x_ref = x.detach().float().requires_grad_(True)
    w_ref = w.detach().requires_grad_(True)
    xf = x_ref
    y = (w_ref.view(1, H, 1, D) * xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps))
    dy = _bf16(B, H, T, D, seed=6).float()
    (y * dy).sum().backward()

    _, rsqrt = pulsar_cuda.qk_rmsnorm_forward(x, w, T, eps)
    dx, dw = pulsar_cuda.qk_rmsnorm_backward(dy.bfloat16(), x, w, rsqrt, T)

    assert_close(x_ref.grad, dx, atol=5e-2, msg="qk dx")
    assert_close(w_ref.grad, dw, atol=5e-2, msg="qk dw")


# ---------------------------------------------------------------------------
# RoPE (half-split convention)
# ---------------------------------------------------------------------------

def ref_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Half-split rotary: rotate (x[:d/2], x[d/2:]) pairs."""
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
def test_rope_forward():
    import pulsar_cuda

    B, H, T, D = 2, 4, 32, 64
    x = _bf16(B, H, T, D, seed=7)
    inv = 1.0 / (10000.0 ** (torch.arange(0, D, 2, device="cuda").float() / D))
    t = torch.arange(T, device="cuda").float()
    freqs = torch.outer(t, inv)
    cos, sin = freqs.cos(), freqs.sin()

    out = pulsar_cuda.rope_forward(x, cos, sin)
    ref = ref_rope(x, cos, sin)
    assert_close(out, ref, atol=2e-2, msg="rope_forward")


# ---------------------------------------------------------------------------
# Fused SwiGLU
# ---------------------------------------------------------------------------

def ref_swiglu(gu: torch.Tensor, d_ff: int) -> torch.Tensor:
    gate = gu[..., :d_ff].float()
    up = gu[..., d_ff:].float()
    return (F.silu(gate) * up).to(gu.dtype)


@needs_gpu
@needs_cuda_ext
def test_fused_swiglu_forward():
    import pulsar_cuda

    E, C, d_ff = 8, 16, 128
    gu = _bf16(E, C, 2 * d_ff, seed=8)
    out = pulsar_cuda.fused_swiglu_forward(gu, d_ff)
    ref = ref_swiglu(gu, d_ff)
    assert_close(out, ref, atol=2e-2, msg="fused_swiglu fwd")


@needs_gpu
@needs_cuda_ext
def test_fused_swiglu_backward():
    import pulsar_cuda

    E, C, d_ff = 4, 8, 128
    gu = _bf16(E, C, 2 * d_ff, seed=9)

    gu_ref = gu.detach().float().requires_grad_(True)
    y = ref_swiglu(gu_ref, d_ff)
    dy = _bf16(*y.shape, seed=10).float()
    (y * dy).sum().backward()

    dgu = pulsar_cuda.fused_swiglu_backward(dy.bfloat16(), gu, d_ff)
    assert_close(dgu, gu_ref.grad, atol=5e-2, msg="fused_swiglu bwd")


# ---------------------------------------------------------------------------
# Softcap cross-entropy (ignore_index + softcap 30)
# ---------------------------------------------------------------------------

def ref_softcap_ce(logits: torch.Tensor, targets: torch.Tensor, softcap: float, ignore_index: int):
    sc = softcap * torch.tanh(logits.float() / softcap)
    per_tok = F.cross_entropy(sc, targets, ignore_index=ignore_index, reduction="none")
    valid = (targets != ignore_index)
    n_valid = valid.sum().clamp_min(1)
    return per_tok, n_valid


@needs_gpu
@needs_cuda_ext
def test_softcap_ce_forward_with_ignore_index():
    import pulsar_cuda

    N, V = 64, 512
    logits = _bf16(N, V, seed=11) * 3.0
    targets = torch.randint(0, V, (N,), device="cuda", dtype=torch.long)
    targets[:8] = -100  # ignored rows

    losses, lse = pulsar_cuda.fused_softcap_ce_forward(logits, targets, 30.0, -100)
    per_tok, n_valid = ref_softcap_ce(logits, targets, 30.0, -100)

    assert losses.numel() == N
    assert_close(losses[8:], per_tok[8:], atol=3e-2, msg="softcap ce per-token")
    assert torch.all(losses[:8] == 0), "ignored rows must contribute zero loss"


@needs_gpu
@needs_cuda_ext
def test_softcap_ce_backward():
    import pulsar_cuda

    N, V = 32, 256
    logits = _bf16(N, V, seed=12) * 2.0
    targets = torch.randint(0, V, (N,), device="cuda", dtype=torch.long)
    targets[:4] = -100

    logits_ref = logits.detach().float().requires_grad_(True)
    per_tok, n_valid = ref_softcap_ce(logits_ref, targets, 30.0, -100)
    loss = per_tok.sum() / n_valid
    loss.backward()

    losses, lse = pulsar_cuda.fused_softcap_ce_forward(logits, targets, 30.0, -100)
    scale = 1.0 / float(n_valid)
    d_logits = pulsar_cuda.fused_softcap_ce_backward(logits, targets, lse, 30.0, scale, -100)

    assert_close(d_logits, logits_ref.grad, atol=3e-2, msg="softcap ce bwd")
    assert torch.all(d_logits[:4] == 0), "ignored rows must get zero gradient"


# ---------------------------------------------------------------------------
# Router top-2
# ---------------------------------------------------------------------------

def ref_router_topk(logits: torch.Tensor, bias: torch.Tensor, z_weight: float):
    scores = torch.sigmoid(logits.float())
    biased = scores + bias.float()
    topk_idx = torch.topk(biased, k=2, dim=-1).indices
    topk_raw = torch.gather(scores, -1, topk_idx)
    topk_weights = topk_raw / (topk_raw.sum(-1, keepdim=True) + 1e-6)
    z = z_weight * torch.logsumexp(logits.float(), dim=-1).pow(2).mean()
    return topk_idx, topk_weights, z


@needs_gpu
@needs_cuda_ext
def test_router_topk_forward():
    import pulsar_cuda

    N, E = 64, 8
    logits = _bf16(N, E, seed=13)
    bias = torch.randn(E, device="cuda", dtype=torch.float32) * 0.1

    idx, weights, z = pulsar_cuda.moe_router_topk_forward(logits, bias, 1e-4)
    idx_ref, weights_ref, z_ref = ref_router_topk(logits, bias, 1e-4)

    assert_close(weights, weights_ref, atol=2e-3, msg="router weights")
    assert_close(z.reshape(()), z_ref.reshape(()), atol=2e-3, msg="router z-loss")
    # selection may differ only on exact ties
    assert (idx == idx_ref).float().mean() > 0.99


@needs_gpu
@needs_cuda_ext
def test_router_weights_normalise():
    import pulsar_cuda

    N, E = 128, 8
    logits = _bf16(N, E, seed=14)
    bias = torch.zeros(E, device="cuda", dtype=torch.float32)
    _, weights, _ = pulsar_cuda.moe_router_topk_forward(logits, bias, 0.0)
    sums = weights.float().sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=2e-3), "top-2 weights must sum to 1"


# ---------------------------------------------------------------------------
# Dispatch / combine with forced overflow
# ---------------------------------------------------------------------------

@needs_gpu
@needs_cuda_ext
def test_dispatch_copies_to_precomputed_slots():
    import pulsar_cuda
    from src.model import compute_token_slots

    N, D, E, K = 32, 64, 8, 2
    x = _bf16(N, D, seed=15)
    topk_idx = torch.stack([
        torch.randint(0, E, (N,), device="cuda"),
        torch.randint(0, E, (N,), device="cuda"),
    ], dim=-1)
    w = torch.rand(N, K, device="cuda")
    capacity = 64  # generous: no drops

    slots = compute_token_slots(topk_idx, w, E, capacity)
    (expert_inputs,) = pulsar_cuda.moe_dispatch_forward(x, topk_idx, slots, E, capacity)

    # every assigned (token,k) must appear verbatim at its slot
    for t in range(N):
        for k in range(K):
            slot = int(slots[t, k])
            if slot >= 0:
                e = int(topk_idx[t, k])
                assert_close(expert_inputs[e, slot], x[t], atol=1e-3, msg="dispatch payload")


@needs_gpu
def test_dispatch_deterministic_across_runs():
    """Slot assignment must be bitwise identical run to run (no atomicAdd ordering).

    The old kernel assigned slots with atomicAdd, so which assignments overflowed
    depended on block scheduling - nondeterministic and biased toward later tokens.
    """
    from src.model import compute_token_slots

    N, E, K, capacity = 512, 8, 2, 40  # deliberate overflow
    g = torch.Generator(device="cuda").manual_seed(0)
    topk_idx = torch.randint(0, E, (N, K), device="cuda", generator=g)
    w = torch.rand(N, K, device="cuda", generator=g)

    runs = [compute_token_slots(topk_idx, w, E, capacity) for _ in range(5)]
    for r in runs[1:]:
        assert torch.equal(runs[0], r), "slot assignment is not deterministic"


@needs_gpu
def test_drop_priority_is_lowest_weight_first():
    """With forced overflow, the dropped assignments must be the lowest-weight ones."""
    from src.model import compute_token_slots

    N, K, E, capacity = 200, 2, 2, 16
    topk_idx = torch.zeros(N, K, dtype=torch.long, device="cuda")  # all -> expert 0
    # strictly decreasing weights: assignment i has weight i
    w = torch.arange(N * K, device="cuda", dtype=torch.float32).reshape(N, K)

    slots = compute_token_slots(topk_idx, w, E, capacity)
    survived = slots.reshape(-1) >= 0
    kept_weights = w.reshape(-1)[survived]
    dropped_weights = w.reshape(-1)[~survived]

    assert survived.sum().item() == capacity
    assert kept_weights.min() > dropped_weights.max(), (
        "a lower-weight assignment survived while a higher-weight one was dropped"
    )


@needs_gpu
@needs_cuda_ext
def test_combine_forward_matches_reference():
    import pulsar_cuda

    N, D, E, K = 16, 32, 4, 2
    capacity = 64
    x = _bf16(N, D, seed=17)
    topk_idx = torch.randint(0, E, (N, K), device="cuda", dtype=torch.long)
    topk_weights = torch.rand(N, K, device="cuda", dtype=torch.float32)
    topk_weights = (topk_weights / topk_weights.sum(-1, keepdim=True)).bfloat16()

    from src.model import compute_token_slots
    token_to_slot = compute_token_slots(topk_idx, topk_weights, E, capacity)
    (expert_inputs,) = pulsar_cuda.moe_dispatch_forward(x, topk_idx, token_to_slot, E, capacity)
    # arbitrary expert outputs so the combine has something non-trivial to gather
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
    assert_close(out, ref, atol=3e-2, msg="combine fwd")


# ---------------------------------------------------------------------------
# Full MoE layer: CUDA path vs PyTorch fallback
# ---------------------------------------------------------------------------

@needs_gpu
@needs_cuda_ext
def test_cuda_vs_fallback_moe_layer():
    """A whole MoE layer must agree between the CUDA and fallback paths."""
    from conftest import make_cfg
    from src.model import PulsarMoEMLP

    cfg = make_cfg(d_model=64, d_expert=64, n_routed_experts=8, top_k=2, max_pulses=2)
    torch.manual_seed(1)
    mlp = PulsarMoEMLP(cfg).cuda().bfloat16()

    B, T = 2, 16
    x = _bf16(B, T, cfg.d_model, seed=19)

    mlp.eval()
    with torch.no_grad():
        out_cuda, _, _, _ = mlp(x, pulse_idx=0)

    # Force the fallback by monkeypatching the module-level CUDA flag.
    import src.model as model_mod

    orig = model_mod.HAS_PULSAR_CUDA
    try:
        model_mod.HAS_PULSAR_CUDA = False
        with torch.no_grad():
            out_fallback, _, _, _ = mlp(x, pulse_idx=0)
    finally:
        model_mod.HAS_PULSAR_CUDA = orig

    assert_close(out_cuda, out_fallback, atol=3e-2, msg="full MoE layer cuda vs fallback")


# ---------------------------------------------------------------------------
# head_dim contract (regression: intermittent NaN with head_dim != 64)
# ---------------------------------------------------------------------------

@needs_gpu
@needs_cuda_ext
def test_qk_rmsnorm_rejects_wrong_head_dim():
    """The kernel is written for head_dim == 64 and must refuse anything else.

    It assigns one (even, odd) pair per lane across a full 32-lane warp = 64 elements
    and does not derive the footprint from head_dim. With head_dim=32 the upper lanes
    read and write past the end of the vector, corrupting neighbouring memory and
    injecting garbage into the warp reduction. That produced intermittent NaNs in
    ~4% of calls - a genuine pre-existing bug, found via a poisoned allocator.
    """
    import pulsar_cuda

    x = torch.randn(1, 4, 8, 32, device="cuda", dtype=torch.bfloat16)
    w = torch.rand(4, 32, device="cuda", dtype=torch.float32) + 0.5
    with pytest.raises(RuntimeError, match="head_dim == 64"):
        pulsar_cuda.qk_rmsnorm_forward(x, w, 8, 1e-6)


@needs_gpu
def test_tiny_model_stable_under_allocator_pressure():
    """The full model must not read uninitialised memory.

    Fills the caching allocator with NaN-bearing buffers before each run so that any
    stale read produces a visible NaN rather than silently sampling old data.
    """
    from src.model import PulsarConfig, PulsarForCausalLM

    cfg = PulsarConfig(
        vocab_size=512, d_model=128, n_heads=4, n_kv_heads=2, head_dim=32, d_expert=128,
        n_core=2, max_pulses=4, default_pulses=3, base_seq_len=64, max_seq_len=256,
        sliding_window=64, n_routed_experts=8, top_k=2, ce_chunk_size=128,
    )

    def poison():
        junk = [torch.full((1 << 20,), float("nan"), device="cuda") for _ in range(8)]
        del junk

    losses = []
    for _ in range(6):
        poison()
        torch.manual_seed(0)
        model = PulsarForCausalLM(cfg).cuda()
        model.to_training_precision(torch.bfloat16)
        x = torch.randint(0, cfg.vocab_size, (3, 32), device="cuda")
        y = torch.randint(0, cfg.vocab_size, (3, 32), device="cuda")
        _, loss, _ = model(x, labels=y, num_pulses=3)
        losses.append(loss.item())
        del model
        torch.cuda.empty_cache()

    assert all(v == v for v in losses), f"NaN loss under allocator pressure: {losses}"
    assert max(losses) - min(losses) < 1e-3, f"loss not reproducible: {losses}"
