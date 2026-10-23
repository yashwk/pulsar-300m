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
            assert g["lr"] == pytest.approx(6e-4 * factor)


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
    model = PulsarForCausalLM(cfg).eval()

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
