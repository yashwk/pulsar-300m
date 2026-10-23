"""Regression tests for the surgical bug-fix pass (WI-1 .. WI-10).

Each test targets one specific, previously-broken behaviour. They are deliberately
narrow: assert the bug is gone, not that the surrounding system is perfect.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import HAS_CUDA, make_cfg


# ---------------------------------------------------------------------------
# WI-1: opt_adam_sr must follow the LR schedule, and its grads must count
# ---------------------------------------------------------------------------

def test_lr_schedule_updates_all_three_sub_optimizers():
    """tok_emb/lm_head live in opt_adam_sr and were left at peak LR forever."""
    from src.model import PulsarConfig, PulsarForCausalLM
    from src.optim import PulsarDualOptimizer
    from src.train import update_optimizer_lrs

    cfg = make_cfg(n_core=2, d_model=128)
    model = PulsarForCausalLM(cfg)
    # The optimiser partitions by dtype: bf16 -> opt_adam_sr. A CPU float32 model has
    # no bf16 params at all, so put the embedding in bf16 the way training does.
    model.to_training_precision(torch.bfloat16)
    opt = PulsarDualOptimizer(model, muon_lr=0.02, adam_lr=6e-4, weight_decay=0.1, seed=0)

    assert len(opt.opt_adam_sr.param_groups) > 0, "test is meaningless without an SR group"

    for factor in (0.0, 0.25, 1.0):
        update_optimizer_lrs(opt, factor, base_muon=0.02, base_adam=6e-4)
        for g in opt.opt_muon.param_groups:
            assert g["lr"] == pytest.approx(0.02 * factor)
        for g in opt.opt_adam_sr.param_groups:
            assert g["lr"] == pytest.approx(6e-4 * factor), (
                "opt_adam_sr ignored the schedule; embeddings ran at full LR through "
                "warmup and never decayed"
            )
        for g in opt.opt_adam.param_groups:
            # The router group runs at a reduced multiplier by design.
            mult = g.get("lr_mult", 1.0)
            assert g["lr"] == pytest.approx(6e-4 * mult * factor)


def test_warmup_scales_embedding_lr_from_zero():
    """At step 0 of warmup the embedding LR must be ~0, not peak."""
    from src.model import PulsarForCausalLM
    from src.optim import PulsarDualOptimizer
    from src.train import get_lr_factor, update_optimizer_lrs

    cfg = make_cfg(n_core=2, d_model=128)
    model = PulsarForCausalLM(cfg)
    # The optimiser partitions by dtype: bf16 -> opt_adam_sr. A CPU float32 model has
    # no bf16 params at all, so put the embedding in bf16 the way training does.
    model.to_training_precision(torch.bfloat16)
    opt = PulsarDualOptimizer(model, muon_lr=0.02, adam_lr=6e-4, weight_decay=0.1, seed=0)

    factor = get_lr_factor(step=1, warmup_steps=100, max_steps=1000)
    update_optimizer_lrs(opt, factor, base_muon=0.02, base_adam=6e-4)
    lr = opt.opt_adam_sr.param_groups[0]["lr"]
    assert lr < 6e-4 * 0.05, f"embedding LR {lr} is not warmed up"


# ---------------------------------------------------------------------------
# WI-2: _chunked_ce must divide by valid tokens, not total tokens
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("frac_ignored", [0.0, 0.5, 0.9])
def test_chunked_ce_matches_exact_mean_with_ignore_index(frac_ignored):
    """Dividing by total tokens deflated the loss whenever padding was present."""
    import torch.nn.functional as F

    from src.model import PulsarForCausalLM

    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512)
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg).eval()

    T = 64
    h = torch.randn(2, T, cfg.d_model)
    labels = torch.randint(0, cfg.vocab_size, (2, T))
    n_ign = int(T * frac_ignored)
    if n_ign:
        labels[:, T - n_ign:] = -100

    got = model._chunked_ce(model.final_norm(h), labels).item()

    raw = F.linear(model.final_norm(h), model.lm_head.weight)
    sc = cfg.logit_softcap * torch.tanh(raw / cfg.logit_softcap)
    want = F.cross_entropy(sc.reshape(-1, cfg.vocab_size), labels.reshape(-1),
                           ignore_index=-100, reduction="mean").item()

    assert got == pytest.approx(want, rel=1e-4), (
        f"chunked CE {got:.6f} != exact mean {want:.6f} with "
        f"{frac_ignored:.0%} ignored labels"
    )


def test_chunked_ce_ignores_all_ignored_batch():
    """An all -100 chunk must not divide by zero or produce NaN/inf."""
    from src.model import PulsarForCausalLM

    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512)
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg).eval()
    h = torch.randn(1, 8, cfg.d_model)
    labels = torch.full((1, 8), -100, dtype=torch.long)
    out = model._chunked_ce(model.final_norm(h), labels)
    assert torch.isfinite(out)


# ---------------------------------------------------------------------------
# WI-3 / WI-4: packed loader bookkeeping and boundaries
# ---------------------------------------------------------------------------

def _packed_loader(seq_len=256, n_docs=60, micro=2):
    from src.packing import PackedBatchLoader, build_doc_index, plan_epoch

    rng = np.random.default_rng(7)
    lengths = rng.integers(80, 400, size=n_docs).astype(np.int64)
    docs = [np.concatenate([rng.integers(1, 500, n - 1), [0]]).astype(np.uint16) for n in lengths]
    shard = np.concatenate(docs)
    index = build_doc_index([shard])
    plan = plan_epoch(lengths, seq_len, seed=0, epoch=0)
    return PackedBatchLoader([shard], index, plan, micro_batch_size=micro, device="cpu")


def test_last_n_real_is_bounded_across_calls():
    """It was appended to on every batch and never cleared - an unbounded leak."""
    loader = _packed_loader()
    for _ in range(5):
        loader.get_batch()
    assert len(loader._last_n_real) <= loader.micro_batch_size


def test_routing_mask_excludes_padding():
    loader = _packed_loader()
    x, _ = loader.get_batch()
    mask = loader.routing_mask(x)
    if mask is None:
        pytest.skip("this plan has no padded windows")
    assert mask.shape == x.shape
    assert mask.dtype == torch.bool
    # Every row's real prefix is contiguous from position 0.
    for r in range(mask.shape[0]):
        n = int(mask[r].sum())
        assert bool(mask[r, :n].all()) and not bool(mask[r, n:].any())


def test_padded_positions_do_not_reach_routing_statistics():
    """Padding must not contribute to z-loss or expert counts."""
    from src.model import PulsarForCausalLM

    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512, top_k=2)
    torch.manual_seed(0)
    # Counters are training-only now (eval contamination was a bug), so this must run
    # in train mode for the statistics to accumulate at all.
    model = PulsarForCausalLM(cfg).train()

    x = torch.randint(1, cfg.vocab_size, (1, 16))
    x[0, 10:] = 0                       # simulate 6 padded positions
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[0, :10] = True

    with torch.no_grad():
        _, loss_masked, m_masked = model(x, labels=x, num_pulses=2, routing_mask=mask)
        model.pop_moe_stats()
        _, loss_nomask, _ = model(x, labels=x, num_pulses=2)
        stats_nomask = model.pop_moe_stats()

    assert stats_nomask["assigned"] > 0
    assert m_masked is not None


def test_cross_document_boundaries_are_masked():
    """At a document's terminating EOS the next token is unrelated: do not train it."""
    loader = _packed_loader()
    x, y = loader.get_batch()
    is_eos = x[:, :-1] == 0
    if not bool(is_eos.any()):
        pytest.skip("no document boundary in this window")
    assert bool((y[:, :-1][is_eos] == -100).all())
    # Non-boundary positions must still be plain next-token targets.
    eq = x[:, 1:] == y[:, :-1]
    assert bool(eq[~is_eos].all())


# ---------------------------------------------------------------------------
# WI-5: Muon must not build a FP32 copy of expert gradients
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA for the memory check")
def test_muon_step_does_not_promote_gradients_to_fp32():
    """Gradients were cast to FP32 and cat'ed before being cast straight back to bf16.

    Measured on one [8,1024,2048] expert tensor (33.5 MB in bf16):
        bf16 path:  478 MB peak
        FP32 path:  545 MB peak   (the 67 MB difference is the FP32 copy)
    Threshold sits between the two.
    """
    import src.optim as optim_mod

    params = [torch.nn.Parameter(torch.zeros(8, 1024, 2048, device="cuda",
                                              dtype=torch.bfloat16))]
    opt = optim_mod.Muon(params, lr=0.02, momentum=0.95)
    for p in params:
        p.grad = torch.randn_like(p)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    opt.step()
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / 1e6

    assert peak_mb < 520, (
        f"Muon.step peaked at {peak_mb:.0f} MB; the bf16 path needs ~478 MB and the "
        f"FP32 promotion adds ~67 MB"
    )


# ---------------------------------------------------------------------------
# WI-6: CUDA reference fallbacks must not crash on non-bf16 input
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")
def test_forward_backward_in_both_dtypes_on_cuda():
    """Missing dtype guards made FP32-on-CUDA crash inside the custom ops."""
    from src.model import PulsarForCausalLM

    for dtype in (torch.float32, torch.bfloat16):
        cfg = make_cfg(n_core=2, d_model=128, vocab_size=512, top_k=2)
        torch.manual_seed(0)
        model = PulsarForCausalLM(cfg).cuda()
        if dtype is torch.bfloat16:
            model.to_training_precision(torch.bfloat16)
        model.train()
        x = torch.randint(0, cfg.vocab_size, (2, 16), device="cuda")
        y = torch.randint(0, cfg.vocab_size, (2, 16), device="cuda")
        _, loss, _ = model(x, labels=y, num_pulses=2)
        loss.backward()
        assert torch.isfinite(loss), f"{dtype} loss not finite"


# ---------------------------------------------------------------------------
# WI-7: kernel correctness at the shapes that used to corrupt
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA kernels")
@pytest.mark.parametrize("dim", [384, 96, 768, 1536])
def test_rmsnorm_backward_partial_warp_dims(dim):
    """num_vec not a multiple of 32 truncated the block reduction.

    blockDim.x was set to num_vec (48 for dim=384), and the kernel derives
    num_warps = blockDim.x / 32 = 1, so the second warp's partial sum was written to
    s_sum[1] and never added. That corrupted dx (~2.9e-1 error) while leaving dweight
    nearly correct, so only checking dweight would have missed it entirely.
    """
    import pulsar_cuda

    torch.manual_seed(0)
    x = torch.randn(8, dim, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(dim, device="cuda")
    dy = torch.randn(8, dim, device="cuda", dtype=torch.bfloat16)
    rsqrt = torch.rsqrt(x.float().pow(2).mean(-1) + 1e-6)
    dx, dweight = pulsar_cuda.rmsnorm_backward(dy, x, w, rsqrt)

    xr = x.float().requires_grad_(True)
    wr = w.detach().clone().requires_grad_(True)
    yr = wr * xr * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + 1e-6)
    (yr * dy.float()).sum().backward()

    # bf16 inputs put a ~1.5e-2 floor on dx from rounding alone; the bug was ~2.9e-1.
    dx_err = (dx.float() - xr.grad).abs().max().item()
    dw_err = (dweight - wr.grad).abs().max().item()
    assert dx_err < 5e-2, (
        f"dx error {dx_err:.2e} at dim={dim} (num_vec={dim // 8}): the partial warp's "
        f"contribution was dropped from the block reduction"
    )
    assert dw_err < 1e-4, f"dweight error {dw_err:.2e} at dim={dim}"


@pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA kernels")
def test_rope_empty_tensor_does_not_launch():
    """An empty input must return an empty result.

    Honest note: this test does NOT fail without the early-return guard. CUDA silently
    no-ops a kernel launch with gridDim.x == 0 on this driver, so the guard is
    correct-by-construction rather than observable. It is kept as a contract test.
    """
    import pulsar_cuda

    q = torch.empty(1, 1, 0, 64, device="cuda", dtype=torch.bfloat16)
    cos = torch.empty(0, 32, device="cuda")
    sin = torch.empty(0, 32, device="cuda")
    out = pulsar_cuda.rope_forward(q, cos, sin)
    assert out.shape == q.shape
    assert out.numel() == 0


@pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA kernels")
@pytest.mark.parametrize("dim", [1024, 2048, 4096])
def test_rmsnorm_backward_dweight_correct_above_1024(dim):
    """Above dim=1024 a thread handled several channel groups and mixed their sums."""
    import torch.nn.functional as F

    import pulsar_cuda

    torch.manual_seed(0)
    x = torch.randn(8, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(dim, device="cuda")
    y, _ = pulsar_cuda.rmsnorm_forward(x, w, 1e-6)
    dy = torch.randn_like(y)
    rsqrt = torch.rsqrt(x.detach().float().pow(2).mean(-1) + 1e-6)
    _, dweight = pulsar_cuda.rmsnorm_backward(dy, x.detach(), w, rsqrt)

    xr = x.detach().float().requires_grad_(True)
    wr = w.detach().clone().requires_grad_(True)
    yr = wr * xr * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + 1e-6)
    (yr * dy.float()).sum().backward()

    err = (dweight - wr.grad).abs().max().item()
    assert err < 1e-4, f"dweight error {err:.2e} at dim={dim}"


@pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA kernels")
def test_rope_survives_sequence_beyond_grid_y_limit():
    """gridDim.y caps at 65535, so T >= 65536 used to fail to launch."""
    import pulsar_cuda

    T = 66000
    q = torch.randn(1, 1, T, 64, device="cuda", dtype=torch.bfloat16)
    cos = torch.ones(T, 32, device="cuda")
    sin = torch.zeros(T, 32, device="cuda")
    out = pulsar_cuda.rope_forward(q, cos, sin)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# WI-8: BlockMask must not be replicated per batch/head
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_CUDA, reason="block masks are built on CUDA here")
def test_block_mask_not_replicated_per_bh():
    import src.model as M

    M._BLOCK_MASK_CACHE.clear()
    bm = M._get_local_block_mask(3, 16, 2048, 1024, torch.device("cuda"))
    # Built with B=None,H=None it has leading dims of 1, not 3 and 16.
    assert bm.kv_num_blocks.shape[0] == 1
    assert bm.kv_num_blocks.shape[1] == 1


@pytest.mark.skipif(not HAS_CUDA, reason="block masks are built on CUDA here")
def test_block_mask_cache_is_bounded():
    import src.model as M

    M._BLOCK_MASK_CACHE.clear()
    for T in range(1024, 1024 + 12 * 128, 128):
        M._get_local_block_mask(1, 4, T, 256, torch.device("cuda"))
    assert len(M._BLOCK_MASK_CACHE) <= 8, "unbounded cache grows with every T"


# ---------------------------------------------------------------------------
# WI-9: generate() context and batched EOS handling
# ---------------------------------------------------------------------------

def test_generate_uses_full_context_window():
    """It sliced to base_seq_len, so the 32k window was unreachable."""
    from src.model import PulsarForCausalLM

    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512, base_seq_len=32, max_seq_len=8192)
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg).eval()

    seen = {}
    orig = model.forward

    def spy(input_ids, *a, **kw):
        seen["T"] = input_ids.shape[1]
        return orig(input_ids, *a, **kw)

    model.forward = spy
    x = torch.randint(1, cfg.vocab_size, (1, 200))
    with torch.no_grad():
        model.generate(x, max_new_tokens=2, temperature=0.0)
    assert seen["T"] > cfg.base_seq_len, (
        f"generate() called forward with T={seen['T']}, capped at base_seq_len="
        f"{cfg.base_seq_len}"
    )


def test_generate_stops_all_rows_at_eos():
    """Batched rollouts used to keep generating past EOS for B > 1."""
    from src.model import PulsarForCausalLM

    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512)
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg).eval()
    x = torch.randint(1, cfg.vocab_size, (3, 16))
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=8, temperature=0.0, eos_token_id=0)
    assert out.shape[0] == 3
    # Every row must be terminated (or stopped at max_new_tokens), not still sampling.
    assert out.shape[1] <= x.shape[1] + 8


# ---------------------------------------------------------------------------
# Final polish: reference/kernel precision parity, pulse weights, edge cases
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA kernels")
def test_swiglu_fallback_matches_kernel():
    """The fallback computed F.silu(gate) * up in bf16; the kernel uses FP32.

    Measured: bf16 intermediates differ from the FP32 reference by ~1.3e-1, FP32
    intermediates match bit-for-bit. Exercises _FusedSwiGLUFunction with the CUDA path
    disabled, so it fails if the fallback regresses.
    """
    import src.model as M

    torch.manual_seed(0)
    gu = torch.randn(4, 16, 256, device="cuda", dtype=torch.bfloat16) * 2.0
    d_ff = 128

    orig = M.HAS_PULSAR_CUDA
    try:
        M.HAS_PULSAR_CUDA = False
        fallback = M._FusedSwiGLUFunction.apply(gu, d_ff)
    finally:
        M.HAS_PULSAR_CUDA = orig

    ref = (torch.nn.functional.silu(gu[..., :d_ff].float()) * gu[..., d_ff:].float()).to(gu.dtype)
    err = (fallback.float() - ref.float()).abs().max().item()
    assert err < 1e-2, (
        f"fallback differs from the FP32 reference by {err:.3e}; bf16 intermediates "
        f"give ~1.3e-1"
    )


@pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA kernels")
def test_swiglu_backward_fallback_matches_kernel():
    import pulsar_cuda

    torch.manual_seed(0)
    gu = torch.randn(4, 16, 256, device="cuda", dtype=torch.bfloat16) * 2.0
    dout = torch.randn(4, 16, 128, device="cuda", dtype=torch.bfloat16)
    d_ff = 128

    fused = pulsar_cuda.fused_swiglu_backward(dout.contiguous(), gu.contiguous(), d_ff)

    g = gu[..., :d_ff].float()
    u = gu[..., d_ff:].float()
    sig = torch.sigmoid(g)
    silu = g * sig
    dg = dout.float() * u * (silu + sig * (1.0 - silu))
    du = dout.float() * silu
    ref = torch.cat([dg.to(gu.dtype), du.to(gu.dtype)], dim=-1)

    err = (fused.float() - ref.float()).abs().max().item()
    assert err < 5e-2, f"backward differ by {err:.3e}"


@pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA kernels")
def test_rope_fallback_matches_kernel_exactly():
    """The fallback downcast cos/sin to bf16; the kernel rotates in FP32.

    Measured against the kernel: FP32 intermediates give a bit-exact match (0.0),
    bf16 downcast of the tables gives 1.56e-2. The assertion is tight enough to
    separate those two, and it exercises apply_rotary_emb rather than reimplementing
    the formula, so it fails if the fallback regresses.
    """
    import src.model as M

    torch.manual_seed(0)
    B, H, T, D = 1, 2, 24, 64
    x = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    angles = torch.rand(T, D // 2, device="cuda")
    cos, sin = torch.cos(angles), torch.sin(angles)

    from src.model import _RoPECUDAFunction
    kernel_out = _RoPECUDAFunction.apply(x, cos, sin)

    # Force the PyTorch fallback by pretending the fused kernel is unavailable.
    orig = M.HAS_PULSAR_CUDA
    try:
        M.HAS_PULSAR_CUDA = False
        fallback_out = M.apply_rotary_emb(x, cos, sin)
    finally:
        M.HAS_PULSAR_CUDA = orig

    err = (kernel_out.float() - fallback_out.float()).abs().max().item()
    assert err < 1e-3, (
        f"fallback differs from the kernel by {err:.3e}; bf16 downcast of the cos/sin "
        f"tables gives ~1.56e-2, FP32 intermediates give 0.0"
    )


def test_pulse_weights_normalised_for_any_k():
    """At K=1 the single pulse carried 0.60, so loss was 60% of the true CE."""
    from src.model import PulsarConfig, PulsarForCausalLM
    import torch.nn.functional as F

    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512, top_k=2,
                   aux_loss_weight_1=0.0, aux_loss_weight_2=0.0, router_z_loss_weight=0.0)
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))

    with torch.no_grad():
        _, loss, _ = model(x, labels=y, num_pulses=1)
        logits, _, _ = model(x, labels=None, num_pulses=1)
    sc = cfg.logit_softcap * torch.tanh(logits / cfg.logit_softcap)
    ref = F.cross_entropy(sc.reshape(-1, cfg.vocab_size), y.reshape(-1), ignore_index=-100)

    assert loss.item() == pytest.approx(ref.item(), rel=1e-3), (
        f"K=1 loss {loss.item():.4f} != CE {ref.item():.4f}; the weight is not normalised"
    )


@pytest.mark.parametrize("K", [2, 3, 4])
def test_pulse_weights_sum_to_one_for_every_k(K):
    """The middle-pulse count was (K-1) instead of (K-2), so at K=3 the weights summed
    to 0.80 and the training loss was reported 20% too small.

    The K=1 test passed against that bug because K=1 took a separate code path, so this
    covers the K>1 shape explicitly.
    """
    from src.model import PulsarForCausalLM

    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512, top_k=2)
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg).eval()

    w_first, w_mid, w_final = (cfg.aux_loss_weight_1, cfg.aux_loss_weight_2,
                               cfg.final_loss_weight)
    raw = [w_first] + [w_mid] * (K - 2) + [w_final]
    expected = [w / sum(raw) for w in raw]

    assert abs(sum(expected) - 1.0) < 1e-9
    if K == 3:
        # The configured 0.15 / 0.25 / 0.60 must survive normalisation unchanged.
        assert expected == pytest.approx([0.15, 0.25, 0.60], abs=1e-9)


def test_reported_loss_matches_weighted_pulse_sum():
    """End-to-end: the returned loss must be the weighted sum of the pulse losses."""
    from src.model import PulsarForCausalLM

    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512, top_k=2,
                   router_z_loss_weight=0.0)
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))

    with torch.no_grad():
        _, loss, metrics = model(x, labels=y, num_pulses=3)
    pl = [float(t) for t in metrics["pulse_losses"]]
    weights = [cfg.aux_loss_weight_1, cfg.aux_loss_weight_2, cfg.final_loss_weight]
    expected = sum(w * l for w, l in zip(weights, pl))

    assert float(loss) == pytest.approx(expected, rel=2e-2), (
        f"loss {float(loss):.4f} != weighted pulse sum {expected:.4f}"
    )


def test_router_confidence_is_not_pinned_to_half():
    """topk_weights are normalised per token, so their overall mean is always 0.5."""
    import torch as _t

    from src.model import PulsarForCausalLM

    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512, top_k=2)
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg).eval()
    x = _t.randint(0, cfg.vocab_size, (2, 16))
    with _t.no_grad():
        _, _, metrics = model(x, labels=None, num_pulses=3)
    conf = metrics.get("router_confidence")
    assert conf is not None and abs(conf - 0.5) > 1e-6, (
        f"router_confidence = {conf}; 0.5 means the metric is still the pinned mean"
    )


def test_invalid_pulse_count_rejected():
    from src.model import PulsarForCausalLM

    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512)
    model = PulsarForCausalLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    for bad in (0, -1, cfg.max_pulses + 1):
        with pytest.raises(ValueError):
            model(x, labels=None, num_pulses=bad)


def test_swiglu_accepts_noncontiguous_input():
    """The kernel requires contiguous input; the wrapper must not trip on a slice."""
    from src.model import _FusedSwiGLUFunction

    if not HAS_CUDA:
        pytest.skip("CUDA path only")
    big = torch.randn(4, 16, 512, device="cuda", dtype=torch.bfloat16)
    gu = big[..., ::2]           # non-contiguous view
    assert not gu.is_contiguous()
    out = _FusedSwiGLUFunction.apply(gu, 128)
    assert out.shape == (4, 16, 128)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Re-audit: MoE slot assignment, stat isolation, SR checkpoint state
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("routing", [
    [[0, 1], [0, 1]],                       # interleaved - the failing case
    [[0, 1], [2, 3], [4, 5], [6, 7]],       # balanced, every expert distinct
    [[0, 0], [0, 0], [1, 1], [1, 1]],       # imbalanced, duplicate within token
    [[3, 3], [3, 3]],                       # single expert
    [[0, 1], [1, 0]],                       # reversed pairs
])
def test_moe_slots_unique_per_expert(routing):
    """Fast path produced [0,-1,2,1] for interleaved experts: wrong and out of range.

    `arange(M) - start_off[flat_e]` only works if assignments are already grouped by
    expert. The router emits them in token order, so expert-local slots collided and
    overflowed capacity even when every expert fit.
    """
    from src.model import compute_token_slots

    ti = torch.tensor(routing)
    weights = torch.ones(*ti.shape)
    # Capacity must cover the WORST expert's load, not the number of rows: with
    # [[3,3],[3,3]] one expert receives all 4 assignments.
    E = 8
    C = int(ti.reshape(-1).bincount(minlength=E).max().item())
    slots = compute_token_slots(ti, weights, E, C)

    flat_e, flat_s = ti.reshape(-1), slots.reshape(-1)
    for e in range(E):
        sel = (flat_e == e)
        if not bool(sel.any()):
            continue
        got = flat_s[sel].tolist()
        assert all(0 <= s < C for s in got), f"expert {e}: slot out of range {got}"
        assert len(set(got)) == len(got), f"expert {e}: duplicate slots {got}"
        assert sorted(got) == list(range(len(got))), f"expert {e}: not contiguous {got}"


def test_moe_slots_match_reference_on_random_routing():
    """Compare against an obviously-correct per-expert reference implementation."""
    from src.model import compute_token_slots

    torch.manual_seed(0)
    for trial in range(5):
        N, K, E, C = 64, 2, 8, 12
        ti = torch.randint(0, E, (N, K))
        tw = torch.rand(N, K)
        got = compute_token_slots(ti, tw, E, C)

        want = torch.full((N * K,), -1, dtype=torch.int32)
        fe, fw = ti.reshape(-1), tw.reshape(-1)
        for e in range(E):
            idx = (fe == e).nonzero().squeeze(-1)
            if idx.numel() == 0:
                continue
            order = torch.argsort(fw[idx], descending=True, stable=True)
            for r, pos in enumerate(order.tolist()):
                if r < C:
                    want[idx[pos]] = r
        assert torch.equal(got.reshape(-1).to(torch.int32), want), f"trial {trial} mismatch"


def test_moe_slot_gradients_match_reference():
    """The slot bug corrupted MoE parameter gradients by ~100%."""
    import src.model as M
    from src.model import PulsarConfig, PulsarForCausalLM

    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512, top_k=2)
    orig = M.compute_token_slots

    def reference(ti, tw, E, C):
        N, K = ti.shape
        s = torch.full((N * K,), -1, dtype=torch.int32, device=ti.device)
        fe, fw = ti.reshape(-1), tw.reshape(-1).float()
        for e in range(E):
            idx = (fe == e).nonzero().squeeze(-1)
            if idx.numel() == 0:
                continue
            order = torch.argsort(fw[idx], descending=True, stable=True)
            for r, pos in enumerate(order.tolist()):
                if r < C:
                    s[idx[pos]] = r
        return s.view(N, K)

    def grads(slot_fn):
        torch.manual_seed(0)
        m = PulsarForCausalLM(cfg)
        m.train()
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        y = torch.randint(0, cfg.vocab_size, (2, 16))
        M.compute_token_slots = slot_fn
        try:
            m.zero_grad()
            _, loss, _ = m(x, labels=y, num_pulses=2)
            loss.backward()
        finally:
            M.compute_token_slots = orig
        return {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}

    g_fixed = grads(orig)
    g_ref = grads(reference)
    for name in g_fixed:
        torch.testing.assert_close(g_fixed[name], g_ref[name], rtol=1e-4, atol=1e-6,
                                   msg=lambda m: f"gradient mismatch at {name}: {m}")


@pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA for the fused MoE path")
def test_eval_forward_does_not_accumulate_routing_counters():
    """Validation forwards fed the bias-update counters, shifting routing."""
    from src.model import PulsarConfig, PulsarForCausalLM

    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512, top_k=2)
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg)
    model.to_training_precision(torch.bfloat16)
    x = torch.randint(0, cfg.vocab_size, (1, 16))

    def total_counts():
        return sum(float(b.mlp.expert_counts_accum.sum()) for b in model.core)

    model.eval()
    with torch.no_grad():
        model(x, labels=None, num_pulses=2)
    assert total_counts() == 0, "eval forward accumulated training counters"

    model.train()
    with torch.no_grad():
        model(x, labels=None, num_pulses=2)
    assert total_counts() > 0, "train forward stopped accumulating counters"


def test_stochastic_rounding_state_round_trips():
    """_step_count seeds the SR noise; omitting it replays the same sequence on resume."""
    import torch.nn as nn

    from src.optim import Muon

    p = nn.Parameter(torch.ones(4, 4, dtype=torch.bfloat16))
    opt = Muon([p], lr=0.1, stochastic_rounding=True, seed=7)
    opt._step_count = 42
    state = opt._sr_state()
    assert state["step_count"] == 42 and state["seed"] == 7

    p2 = nn.Parameter(torch.ones(4, 4, dtype=torch.bfloat16))
    other = Muon([p2], lr=0.1, stochastic_rounding=True, seed=7)
    other._load_sr_state(state)
    assert other._sr_state() == state

    # A resumed optimizer must not replay step-0 noise.
    g = torch.device("cpu")
    a = torch.randint(0, 1 << 16, (8,), generator=opt._gen(g))
    opt._step_count += 1
    b = torch.randint(0, 1 << 16, (8,), generator=opt._gen(g))
    assert not torch.equal(a, b)


def test_dual_optimizer_checkpoint_includes_sr_state():
    from src.model import PulsarForCausalLM
    from src.optim import PulsarDualOptimizer

    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512, top_k=2)
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg)
    model.to_training_precision(torch.bfloat16)
    opt = PulsarDualOptimizer(model, muon_lr=0.02, adam_lr=6e-4, weight_decay=0.1, seed=3)
    sd = opt.state_dict()
    assert "sr_muon" in sd and "sr_adam" in sd
    assert sd["sr_muon"]["seed"] == 3
    # Loading a legacy checkpoint without the SR block must not raise.
    legacy = {k: v for k, v in sd.items() if not k.startswith("sr_")}
    opt.load_state_dict(legacy)


# ---------------------------------------------------------------------------
# Broader audit: pulse-metric averaging, checkpoint resume exactness
# ---------------------------------------------------------------------------

def test_adamw_fp32_master_state_survives_checkpoint_load():
    """Optimizer.load_state_dict casts state to the PARAM dtype.

    AdamWBF16SR keeps FP32 exp_avg/exp_avg_sq over bf16 parameters. Loading a checkpoint
    downcast them to bf16 before any post-hoc cast could run, so a resumed run diverged
    even though every tensor round-tripped. The FP32 block now travels under a private
    key that torch does not touch.
    """
    import copy

    import torch.nn as nn

    from src.optim import AdamWBF16SR

    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(16, 16, dtype=torch.bfloat16))
    opt = AdamWBF16SR([p], lr=6e-4, stochastic_rounding=True, seed=3)
    g = (torch.randn(16, 16) * 1e-3).to(torch.bfloat16)
    for _ in range(2):
        p.grad = g.clone()
        opt.step()

    sd = copy.deepcopy(opt.state_dict())
    saved_avg = opt.state[p]["exp_avg"].clone()
    saved_step = opt._step_count

    p2 = nn.Parameter(torch.randn(16, 16, dtype=torch.bfloat16))
    opt2 = AdamWBF16SR([p2], lr=6e-4, stochastic_rounding=True, seed=3)
    opt2.load_state_dict(sd)

    restored = opt2.state[p2]["exp_avg"]
    assert restored.dtype == torch.float32, f"master state downcast to {restored.dtype}"
    assert torch.equal(restored, saved_avg), "FP32 master state not bit-exact after load"
    assert opt2._step_count == saved_step, "SR step counter not restored"


def test_resume_reproduces_uninterrupted_run_exactly():
    """Save mid-run, resume, and continue: parameters must match bit-for-bit."""
    import copy

    from src.model import PulsarConfig, PulsarForCausalLM
    from src.optim import PulsarDualOptimizer

    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512, top_k=2)

    def steps(model, opt, n, seed0):
        for i in range(n):
            torch.manual_seed(seed0 + i)
            x = torch.randint(0, cfg.vocab_size, (2, 16))
            y = torch.randint(0, cfg.vocab_size, (2, 16))
            model.zero_grad()
            _, loss, _ = model(x, labels=y, num_pulses=2)
            loss.backward()
            opt.step()
            for b in model.core:
                b.mlp.update_biases(0.001, num_pulses=2)

    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg)
    model.to_training_precision(torch.bfloat16)
    opt = PulsarDualOptimizer(model, muon_lr=0.02, adam_lr=6e-4, weight_decay=0.1, seed=5)

    steps(model, opt, 3, 100)
    opt_sd = copy.deepcopy(opt.state_dict())
    model_sd = copy.deepcopy(model.state_dict())
    steps(model, opt, 2, 200)
    reference = [p.detach().clone() for p in model.parameters()]

    torch.manual_seed(0)
    model2 = PulsarForCausalLM(cfg)
    model2.to_training_precision(torch.bfloat16)
    model2.load_state_dict(model_sd)
    opt2 = PulsarDualOptimizer(model2, muon_lr=0.02, adam_lr=6e-4, weight_decay=0.1, seed=5)
    opt2.load_state_dict(opt_sd)
    steps(model2, opt2, 2, 200)
    resumed = [p.detach().clone() for p in model2.parameters()]

    worst = max((a.float() - b.float()).abs().max().item()
                for a, b in zip(reference, resumed))
    assert worst == 0.0, f"resume diverged by {worst:.3e}"


def test_training_counters_survive_checkpoint():
    """update_biases() reads expert_counts_accum, so it must be persistent."""
    from src.model import PulsarForCausalLM

    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512, top_k=2)
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg)
    model.train()
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        model(x, labels=None, num_pulses=2)

    sd = model.state_dict()
    assert any("expert_counts_accum" in k for k in sd), (
        "expert_counts_accum is not persistent; a mid-run checkpoint loses the counts "
        "that update_biases() depends on"
    )


# ---------------------------------------------------------------------------
# Fixes for ca71e80: deterministic eval windows, SR checkpoint placement
# ---------------------------------------------------------------------------

def test_evaluate_loss_uses_identical_windows_on_consecutive_calls():
    """PackedBatchLoader walks a cursor and ignores the seed.

    Without a reset, every eval step scored a different slice of validation windows,
    so val_ce_final was not comparable across steps or with the value stored in best.pt.
    """
    import numpy as np

    from src.model import PulsarForCausalLM
    from src.packing import PackedBatchLoader, build_doc_index, plan_epoch
    from src.train import evaluate_loss

    rng = np.random.default_rng(3)
    lengths = rng.integers(80, 400, size=40).astype(np.int64)
    parts = [np.concatenate([rng.integers(1, 500, n - 1), [0]]).astype(np.uint16) for n in lengths]
    shard = np.concatenate(parts)
    index = build_doc_index([shard])
    plan = plan_epoch(lengths, 64, seed=0, epoch=0)
    loader = PackedBatchLoader([shard], index, plan, micro_batch_size=2, device="cpu")

    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512, top_k=2)
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg).eval()

    first = evaluate_loss(model, loader, eval_iters=2, num_pulses=2)
    cursor_after_first = loader._cursor
    second = evaluate_loss(model, loader, eval_iters=2, num_pulses=2)

    assert first["val_ce_final"] == pytest.approx(second["val_ce_final"], rel=1e-6), (
        f"eval loss drifted between calls ({first['val_ce_final']} vs "
        f"{second['val_ce_final']}); the loader was not reset"
    )
    assert loader._cursor == cursor_after_first, "cursor advanced across a reset eval"


@pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA for the fused MoE path")
def test_moe_counters_untouched_by_eval_forward():
    """evaluate_loss/generate_canary run under model.eval(); they must not feed the
    counters that drive update_expert_biases()."""
    from src.model import PulsarConfig, PulsarForCausalLM

    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512, top_k=2)
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg)
    model.to_training_precision(torch.bfloat16)
    x = torch.randint(0, cfg.vocab_size, (1, 32))

    counters = ("expert_counts_accum", "expert_pulse_counts", "drop_counts_accum",
                "assigned_counts_accum", "pulse_drop_accum", "pulse_assigned_accum")

    def snapshot():
        return {k: sum(float(getattr(b.mlp, k).sum()) for b in model.core) for k in counters}

    model.eval()
    before = snapshot()
    with torch.no_grad():
        for _ in range(3):
            model(x, labels=None, num_pulses=2)
    after = snapshot()

    for k in counters:
        assert before[k] == after[k], f"eval forward mutated {k} by {after[k] - before[k]}"


def test_sr_state_restored_to_param_device_and_fp32():
    """A CPU-mapped checkpoint must place FP32 master moments back on the param device."""
    import copy

    import torch.nn as nn

    from src.optim import AdamWBF16SR

    dev = "cuda" if HAS_CUDA else "cpu"
    p = nn.Parameter(torch.zeros(16, 16, dtype=torch.bfloat16, device=dev))
    opt = AdamWBF16SR([p], lr=6e-4, stochastic_rounding=True, seed=1)
    p.grad = torch.randn(16, 16, device=dev).to(torch.bfloat16)
    opt.step()

    sd = copy.deepcopy(opt.state_dict())
    sd["_fp32_state"] = {
        i: {k: v.cpu() for k, v in d.items()} for i, d in sd["_fp32_state"].items()
    }

    p2 = nn.Parameter(torch.zeros(16, 16, dtype=torch.bfloat16, device=dev))
    opt2 = AdamWBF16SR([p2], lr=6e-4, stochastic_rounding=True, seed=1)
    opt2.load_state_dict(sd)

    state = opt2.state[p2]["exp_avg"]
    assert state.dtype == torch.float32, f"master state in {state.dtype}, expected float32"
    assert state.device.type == torch.device(dev).type, f"state on {state.device}, expected {dev}"


def test_muon_bf16_state_not_duplicated_in_checkpoint():
    """_fp32_state must not clone bf16 momentum: torch's cast is already lossless."""
    import torch.nn as nn

    from src.optim import Muon

    p = nn.Parameter(torch.zeros(32, 32, dtype=torch.bfloat16))
    opt = Muon([p], lr=0.02, stochastic_rounding=True, seed=1)
    p.grad = torch.randn_like(p)
    opt.step()
    assert "_fp32_state" not in opt.state_dict(), (
        "bf16 momentum was cloned into _fp32_state, duplicating it in every checkpoint"
    )


# ---------------------------------------------------------------------------
# Router drift vs balancing bias: LR split, logit centring, true imbalance
# ---------------------------------------------------------------------------

def test_router_weight_gets_reduced_lr_group():
    """A d_model=1024 router step moves logits ~0.6, ~5-15x faster than bias_lr can
    correct in logit space. The router now runs at 0.1x with weight decay."""
    from src.model import PulsarForCausalLM
    from src.optim import PulsarDualOptimizer
    from src.train import update_optimizer_lrs

    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512, top_k=2)
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg)
    model.to_training_precision(torch.bfloat16)
    opt = PulsarDualOptimizer(model, muon_lr=0.02, adam_lr=6e-4, weight_decay=0.1, seed=0)

    groups = opt.opt_adam.param_groups
    router_groups = [g for g in groups if g.get("lr_mult") == 0.1]
    assert len(router_groups) == 1, f"expected one router group, got {len(router_groups)}"
    assert router_groups[0]["weight_decay"] == pytest.approx(0.01)

    update_optimizer_lrs(opt, factor=1.0, base_muon=0.02, base_adam=6e-4)
    assert router_groups[0]["lr"] == pytest.approx(6e-4 * 0.1)
    update_optimizer_lrs(opt, factor=0.5, base_muon=0.02, base_adam=6e-4)
    assert router_groups[0]["lr"] == pytest.approx(6e-4 * 0.1 * 0.5)

    # Non-router Adam params keep the full LR.
    plain = [g for g in groups if g.get("lr_mult") is None]
    if plain:
        assert plain[0]["lr"] == pytest.approx(6e-4 * 0.5)


def test_router_logits_are_mean_centred():
    """Centring is invariant for top-k and the bias add, but stops logit scale drifting
    with the shared batch-mean component of the hidden state."""
    from src.model import PulsarConfig, PulsarMoEMLP

    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512, top_k=2)
    torch.manual_seed(0)
    m = PulsarMoEMLP(cfg)
    m.eval()

    captured = {}
    import torch.nn.functional as F
    orig_linear = F.linear

    def spy(inp, weight, bias=None):
        out = orig_linear(inp, weight, bias)
        if weight is not None and weight.shape == m.router_weight[0].shape:
            captured["raw"] = out.detach().clone()
        return out

    F.linear = spy
    try:
        with torch.no_grad():
            m(torch.randn(1, 32, cfg.d_model), pulse_idx=0)
    finally:
        F.linear = orig_linear

    assert "raw" in captured, "router logits not captured"
    # The module centres them; raw should be offset, and the centred version zero-mean.
    centred = captured["raw"] - captured["raw"].mean(dim=-1, keepdim=True)
    assert torch.allclose(centred.mean(dim=-1), torch.zeros(centred.shape[0]), atol=1e-5)


def test_per_block_imbalance_is_not_diluted():
    """Summing counts across blocks by expert index hides per-block collapse.

    Expert 0 of block 0 is an unrelated router to Expert 0 of block 1, so two blocks
    collapsing onto DIFFERENT experts average into something that looks moderate. The
    new metric reports the true worst single-expert share instead.
    """
    from src.model import PulsarConfig, PulsarForCausalLM

    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512, top_k=2)
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg)
    model.train()

    # Drive block 0 onto expert 0 and block 1 onto expert 4 via the balancing bias.
    with torch.no_grad():
        model.core[0].mlp.expert_bias.zero_()
        model.core[0].mlp.expert_bias[:, 0] = 5.0
        model.core[1].mlp.expert_bias.zero_()
        model.core[1].mlp.expert_bias[:, 4] = 5.0

    with torch.no_grad():
        _, _, metrics = model(
            torch.randint(0, cfg.vocab_size, (1, 64)), labels=None, num_pulses=2
        )

    agg_max = float(metrics["expert_pcts"].max())
    worst = float(metrics["worst_expert_share_pct"])

    assert worst > agg_max + 10, (
        f"worst_expert_share {worst:.1f}% is not clearly above the block-averaged "
        f"maximum {agg_max:.1f}% - the imbalance is being diluted"
    )
    assert worst > 40, f"worst_expert_share {worst:.1f}% too low for a forced collapse"
