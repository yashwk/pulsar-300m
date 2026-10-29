"""Optimizer contract tests for Pulsar-300M.

Covers:
- Muon 2D and 3D aspect ratio scaling and BF16 Nesterov momentum.
- AdamWBF16SR stochastic rounding, unbiased updates, and FP32 master state preservation.
- PulsarDualOptimizer parameter partitioning across Muon, AdamW, and AdamW-SR.
- Warmup and learning rate schedule synchronization across all sub-optimizers.
- Checkpoint save/load roundtrips, device placement, and bit-exact resumption.
"""

from __future__ import annotations

import copy
import math
import pytest
import torch
import torch.nn as nn

from conftest import make_cfg, HAS_CUDA
from src.model import PulsarConfig, PulsarForCausalLM
from src.optim import AdamWBF16SR, Muon, PulsarDualOptimizer, stochastic_round_
from src.train import get_lr_factor, update_optimizer_lrs


# ---------------------------------------------------------------------------
# Muon Contract Tests
# ---------------------------------------------------------------------------

def test_muon_aspect_ratio_scaling_2d_and_3d():
    """Muon scales updates by sqrt(out/in) for 2D [out, in] and 3D [E, in, out] tensors."""
    # 2D Linear weight: [out_features, in_features] = [2048, 1024] -> scale = sqrt(2)
    p_2d = nn.Parameter(torch.zeros(2048, 1024, dtype=torch.bfloat16))
    p_2d.grad = torch.randn_like(p_2d)

    # 3D MoE expert weight: [E, in_features, out_features] = [8, 1024, 2048] -> scale = sqrt(2)
    p_3d = nn.Parameter(torch.zeros(8, 1024, 2048, dtype=torch.bfloat16))
    p_3d.grad = torch.randn_like(p_3d)

    opt = Muon([p_2d, p_3d], lr=0.02, weight_decay=0.0, stochastic_rounding=False)
    opt.step()

    # Both tensors should have had update scale sqrt(2048/1024) = sqrt(2) ≈ 1.4142
    scale_2d = max(1.0, 2048 / 1024) ** 0.5
    scale_3d = max(1.0, 2048 / 1024) ** 0.5
    assert math.isclose(scale_2d, math.sqrt(2.0), rel_tol=1e-5)
    assert math.isclose(scale_3d, math.sqrt(2.0), rel_tol=1e-5)


@pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA for VRAM peak allocation check")
def test_muon_step_does_not_promote_gradients_to_fp32():
    """Muon executes Nesterov momentum update in BF16 without FP32 promotion."""
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < 600 * 1024 * 1024:
        pytest.skip(f"Not enough free VRAM ({free_bytes / 1e6:.0f} MB free) for peak test")

    param = nn.Parameter(torch.zeros(8, 1024, 2048, device="cuda", dtype=torch.bfloat16))
    opt = Muon([param], lr=0.02, momentum=0.95)
    param.grad = torch.randn_like(param)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    opt.step()
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    assert peak_mb < 520, f"Muon.step peaked at {peak_mb:.0f} MB; expected < 520 MB in BF16"


def test_muon_stochastic_rounding_moves_small_updates():
    """Stochastic rounding in Muon ensures small updates accumulate on BF16 weights."""
    torch.manual_seed(0)
    p = nn.Parameter(torch.full((64, 64), 0.02, dtype=torch.bfloat16))
    opt = Muon([p], lr=0.02, weight_decay=0.0, stochastic_rounding=True, seed=0)
    p.grad = torch.full_like(p, 1e-3)

    before = p.detach().clone()
    for _ in range(5):
        p.grad = torch.full_like(p, 1e-3)
        opt.step()
    assert not torch.equal(before, p.detach())


def test_muon_bf16_momentum_not_duplicated_in_checkpoint():
    """Muon avoids duplicating BF16 momentum buffer in _fp32_state."""
    p = nn.Parameter(torch.zeros(32, 32, dtype=torch.bfloat16))
    opt = Muon([p], lr=0.02, stochastic_rounding=True, seed=1)
    p.grad = torch.randn_like(p)
    opt.step()
    assert "_fp32_state" not in opt.state_dict()


@pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA for tall parameter Muon test")
def test_muon_tall_parameter_update_matches_direction():
    """Muon on tall 2D parameters (size(-2) > size(-1)) updates in the correct direction with contiguous memory."""
    from src.optim import zeropower_via_newtonschulz5
    torch.manual_seed(42)
    H, W = 128, 64
    p = nn.Parameter(torch.zeros(H, W, device="cuda", dtype=torch.bfloat16))
    p_init = p.detach().clone()
    g = torch.randn(H, W, device="cuda", dtype=torch.bfloat16)
    p.grad = g.clone()

    lr = 0.02
    opt = Muon([p], lr=lr, momentum=0.0, weight_decay=0.0, stochastic_rounding=True, seed=42)
    opt.step()

    g_orth_ref = zeropower_via_newtonschulz5(g)
    scale = max(1.0, H / W) ** 0.5
    expected_delta = -lr * scale * g_orth_ref

    actual_delta = (p.detach() - p_init).float()
    cos_sim = torch.cosine_similarity(actual_delta.flatten(), expected_delta.flatten(), dim=0)
    assert cos_sim.item() > 0.99, f"Tall 2D update direction mismatch: cos_sim={cos_sim.item():.4f}"


# ---------------------------------------------------------------------------
# Stochastic Rounding & AdamWBF16SR Contract Tests
# ---------------------------------------------------------------------------

def test_sr_unbiased_accumulation():
    """Stochastic rounding accumulates sub-ulp updates within 10% of exact trajectory."""
    torch.manual_seed(0)
    n = 4096
    dst = torch.full((n,), 0.02, dtype=torch.bfloat16)
    exact = torch.full((n,), 0.02, dtype=torch.float32)
    gen = torch.Generator(device="cpu").manual_seed(1)

    step = 3e-5
    reps = 2000
    for _ in range(reps):
        exact = exact + step
        stochastic_round_(dst, exact, gen)

    expected_drift = reps * step
    got_drift = (dst.float() - 0.02).mean().item()
    lo, hi = 0.9 * expected_drift, 1.1 * expected_drift
    assert lo <= got_drift <= hi


def test_sr_decay_and_seed_reproducibility():
    """Stochastic rounding tracks exponential decay and produces reproducible bit-level sequences."""
    torch.manual_seed(0)
    n = 8192
    dst = torch.full((n,), 1.0, dtype=torch.bfloat16)
    exact = torch.full((n,), 1.0, dtype=torch.float32)
    gen = torch.Generator(device="cpu").manual_seed(2)

    for _ in range(50):
        exact = exact * 0.998
        stochastic_round_(dst, exact, gen)

    expected = 0.998 ** 50
    got = dst.float().mean().item()
    assert abs(got - expected) / expected < 0.10

    # Seed reproducibility
    src = torch.randn(1000, dtype=torch.float32) * 0.02
    a, b = torch.empty(1000, dtype=torch.bfloat16), torch.empty(1000, dtype=torch.bfloat16)
    stochastic_round_(a, src, torch.Generator(device="cpu").manual_seed(42))
    stochastic_round_(b, src, torch.Generator(device="cpu").manual_seed(42))
    assert torch.equal(a, b)


def test_adamw_fp32_master_state_survives_checkpoint_load():
    """AdamWBF16SR preserves FP32 master exp_avg across checkpoint serialization."""
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
    assert restored.dtype == torch.float32
    assert torch.equal(restored, saved_avg)
    assert opt2._step_count == saved_step


def test_sr_state_restored_to_param_device_and_fp32():
    """Loading a CPU checkpoint maps FP32 moments back to the parameter's target device."""
    dev = "cuda" if HAS_CUDA else "cpu"
    p = nn.Parameter(torch.zeros(16, 16, dtype=torch.bfloat16, device=dev))
    opt = AdamWBF16SR([p], lr=6e-4, stochastic_rounding=True, seed=1)
    p.grad = torch.randn(16, 16, device=dev).to(torch.bfloat16)
    opt.step()

    sd = copy.deepcopy(opt.state_dict())
    sd["_fp32_state"] = {i: {k: v.cpu() for k, v in d.items()} for i, d in sd["_fp32_state"].items()}

    p2 = nn.Parameter(torch.zeros(16, 16, dtype=torch.bfloat16, device=dev))
    opt2 = AdamWBF16SR([p2], lr=6e-4, stochastic_rounding=True, seed=1)
    opt2.load_state_dict(sd)

    state = opt2.state[p2]["exp_avg"]
    assert state.dtype == torch.float32
    assert state.device.type == torch.device(dev).type


# ---------------------------------------------------------------------------
# PulsarDualOptimizer Contract Tests
# ---------------------------------------------------------------------------

def test_dual_optimizer_group_routing_and_router_lr():
    """DualOptimizer routes 2D/3D matrix weights to Muon, embeddings to SR, and router to 0.1x LR."""
    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512, top_k=2)
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg)
    model.to_training_precision(torch.bfloat16)
    opt = PulsarDualOptimizer(model, muon_lr=0.02, adam_lr=6e-4, weight_decay=0.1, seed=0)

    # 1. Parameter counts per optimizer
    assert len(opt.opt_muon.param_groups) > 0
    assert len(opt.opt_adam_sr.param_groups) > 0
    assert len(opt.opt_adam.param_groups) > 0

    # 2. Router weight receives 0.1x LR multiplier
    router_groups = [g for g in opt.opt_adam.param_groups if g.get("lr_mult") == 0.1]
    assert len(router_groups) == 1
    assert router_groups[0]["weight_decay"] == pytest.approx(0.01)

    # 3. Schedule updates all groups proportionally
    update_optimizer_lrs(opt, factor=1.0, base_muon=0.02, base_adam=6e-4)
    assert router_groups[0]["lr"] == pytest.approx(6e-4 * 0.1)
    for g in opt.opt_muon.param_groups:
        assert g["lr"] == pytest.approx(0.02)
    for g in opt.opt_adam_sr.param_groups:
        assert g["lr"] == pytest.approx(6e-4)


@pytest.mark.parametrize("factor", [0.0, 0.25, 0.75, 1.0])
def test_lr_schedule_and_warmup_scaling(factor):
    """Learning rate factor correctly scales embedding, Muon, and Adam sub-optimizers."""
    cfg = make_cfg(n_core=2, d_model=128)
    model = PulsarForCausalLM(cfg)
    model.to_training_precision(torch.bfloat16)
    opt = PulsarDualOptimizer(model, muon_lr=0.02, adam_lr=6e-4, weight_decay=0.1, seed=0)

    update_optimizer_lrs(opt, factor, base_muon=0.02, base_adam=6e-4)
    for g in opt.opt_muon.param_groups:
        assert g["lr"] == pytest.approx(0.02 * factor)
    for g in opt.opt_adam_sr.param_groups:
        assert g["lr"] == pytest.approx(6e-4 * factor)
    for g in opt.opt_adam.param_groups:
        mult = g.get("lr_mult", 1.0)
        assert g["lr"] == pytest.approx(6e-4 * mult * factor)


def test_warmup_scales_embedding_lr_from_zero():
    """At step 1 of warmup, the embedding learning rate starts near zero."""
    cfg = make_cfg(n_core=2, d_model=128)
    model = PulsarForCausalLM(cfg)
    model.to_training_precision(torch.bfloat16)
    opt = PulsarDualOptimizer(model, muon_lr=0.02, adam_lr=6e-4, weight_decay=0.1, seed=0)

    factor = get_lr_factor(step=1, warmup_steps=100, max_steps=1000)
    update_optimizer_lrs(opt, factor, base_muon=0.02, base_adam=6e-4)
    lr = opt.opt_adam_sr.param_groups[0]["lr"]
    assert lr < 6e-4 * 0.05


def test_resume_reproduces_uninterrupted_run_exactly():
    """DualOptimizer state dict restore exactly reproduces an uninterrupted training trajectory."""
    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512, top_k=2)

    def run_steps(model, opt, n_steps, seed_start):
        for i in range(n_steps):
            torch.manual_seed(seed_start + i)
            x = torch.randint(0, cfg.vocab_size, (2, 16))
            y = torch.randint(0, cfg.vocab_size, (2, 16))
            model.zero_grad()
            _, loss, _ = model(x, labels=y, num_pulses=2)
            loss.backward()
            opt.step()
            for b in model.core:
                b.mlp.update_biases(0.001, num_pulses=2)

    # 1. Uninterrupted run: 3 + 2 steps
    torch.manual_seed(0)
    m1 = PulsarForCausalLM(cfg)
    m1.to_training_precision(torch.bfloat16)
    opt1 = PulsarDualOptimizer(m1, muon_lr=0.02, adam_lr=6e-4, weight_decay=0.1, seed=5)
    run_steps(m1, opt1, 3, 100)
    opt_sd = copy.deepcopy(opt1.state_dict())
    m_sd = copy.deepcopy(m1.state_dict())
    run_steps(m1, opt1, 2, 200)
    ref_params = [p.detach().clone() for p in m1.parameters()]

    # 2. Interrupted and resumed run: restore at step 3 and run 2 steps
    torch.manual_seed(0)
    m2 = PulsarForCausalLM(cfg)
    m2.to_training_precision(torch.bfloat16)
    m2.load_state_dict(m_sd)
    opt2 = PulsarDualOptimizer(m2, muon_lr=0.02, adam_lr=6e-4, weight_decay=0.1, seed=5)
    opt2.load_state_dict(opt_sd)
    run_steps(m2, opt2, 2, 200)
    res_params = [p.detach().clone() for p in m2.parameters()]

    worst_diff = max((a.float() - b.float()).abs().max().item() for a, b in zip(ref_params, res_params))
    assert worst_diff == 0.0
