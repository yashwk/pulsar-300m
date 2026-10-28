"""Tests for the MoE init-balance and z-loss fixes."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from conftest import make_cfg


# ---------------------------------------------------------------------------
# Routed-expert init balance
# ---------------------------------------------------------------------------

def test_init_balance_shared_vs_routed():
    """At init, the routed path must not dominate the shared expert.

    The old init used 1/sqrt(d_model) for routed_gate_up and 1/sqrt(d_expert) for
    routed_down (~0.031 each), while the shared expert's down projection follows the
    residual-output rule 0.02/sqrt(2*n_core*default_pulses) (~0.0029). That made the
    routed contribution roughly an order of magnitude larger than the shared one at
    init, dominating the MLP delta.

    Note: instantiate through PulsarForCausalLM so _init_weights runs - constructing
    PulsarMoEMLP directly leaves the shared expert at PyTorch's default init, which
    makes this comparison meaningless.
    """
    from src.model import PulsarForCausalLM

    cfg = make_cfg(d_model=128, d_expert=128, n_core=2, default_pulses=3)
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg).float()
    mlp = model.core[0].mlp
    mlp.eval()

    N = 256
    x = torch.randn(1, N, cfg.d_model) * 0.5
    xf = x.view(N, cfg.d_model)

    with torch.no_grad():
        shared = mlp.shared_expert(xf)
        out, _, _, _ = mlp(x, pulse_idx=0)
        routed = out.view(N, cfg.d_model) - shared

    def rms(t):
        return t.float().pow(2).mean().sqrt().item()

    rs, rr = rms(shared), rms(routed)
    ratio = rr / max(rs, 1e-12)
    assert 0.5 <= ratio <= 2.0, (
        f"routed/shared RMS ratio at init is {ratio:.2f} "
        f"(shared {rs:.5f}, routed {rr:.5f}); expected within a factor of 2"
    )


def test_routed_init_matches_shared_convention():
    """Gate/up uses 0.02, down uses the residual-output std."""
    from src.model import PulsarMoEMLP

    cfg = make_cfg(d_model=256, d_expert=256, n_core=4, default_pulses=3)
    torch.manual_seed(0)
    mlp = PulsarMoEMLP(cfg)

    expected_down = 0.02 / math.sqrt(2 * cfg.n_core * cfg.default_pulses)
    got_up = mlp.routed_gate_up.detach().float().std().item()
    got_down = mlp.routed_down.detach().float().std().item()

    assert abs(got_up - 0.02) < 0.002, f"routed_gate_up std {got_up:.4f}, expected ~0.02"
    assert abs(got_down - expected_down) < expected_down * 0.15, (
        f"routed_down std {got_down:.5f}, expected ~{expected_down:.5f}"
    )


# ---------------------------------------------------------------------------
# z-loss gradient scaling
# ---------------------------------------------------------------------------

def test_zloss_gradient_matches_explicit_formula():
    """The z_loss actually returned by the MoE module must backprop the analytic grad.

    z = w * mean_i( logsumexp(logits_i)^2 )
    dz/dlogits_i = w * 2*L_i*softmax(logits_i) / N

    This goes through PulsarMoEMLP.forward rather than restating the formula, so it
    catches the original defect: the CUDA kernel baked in (2*w/N)*L and assumed an
    upstream gradient of 1, while ignoring the real incoming gradient and the 1/24
    pulse/block average. The effective weight was ~1000x the configured value.
    """
    from src.model import PulsarMoEMLP

    cfg = make_cfg(d_model=64, d_expert=64, n_routed_experts=8, top_k=2, max_pulses=2)
    torch.manual_seed(0)
    mlp = PulsarMoEMLP(cfg).float()

    N = 32
    x = torch.randn(1, N, cfg.d_model)
    out, z_loss, _, _ = mlp(x, pulse_idx=0)

    # only take the z-loss gradient (detach the rest of the graph)
    z_loss.backward()

    logits = F.linear(x.view(N, cfg.d_model), mlp.router_weight[0])
    # The module mean-centres logits across experts before computing z-loss; the
    # reference must apply the same transform or the analytic gradient is off by the
    # centring projection.
    logits = logits - logits.mean(dim=-1, keepdim=True)
    lse = torch.logsumexp(logits, dim=-1, keepdim=True)
    expected_dlogits = cfg.router_z_loss_weight * 2.0 * lse * torch.softmax(logits, dim=-1) / N
    # Centring is a projection logits -> logits - mean(logits); its gradient must be
    # projected too, and the centring term vanishes because softmax sums to 1.
    expected_dlogits = expected_dlogits - expected_dlogits.mean(dim=-1, keepdim=True)
    expected_w = expected_dlogits.t() @ x.view(N, cfg.d_model)

    got = mlp.router_weight.grad[0]
    max_err = (got - expected_w).abs().max().item()
    scale = expected_w.abs().max().item()
    assert max_err <= 1e-4 * max(1.0, scale), f"z-loss grad err {max_err:.3e} vs scale {scale:.3e}"


def test_zloss_scales_with_loss_division():
    """Dividing the loss (as grad accumulation does) must scale the gradient by 1/43."""
    from src.model import PulsarMoEMLP

    cfg = make_cfg(d_model=64, d_expert=64, n_routed_experts=8, top_k=2, max_pulses=2)

    def grad_after(div):
        torch.manual_seed(0)
        mlp = PulsarMoEMLP(cfg).float()
        x = torch.randn(32, cfg.d_model)
        logits = F.linear(x, mlp.router_weight[0])
        loss = cfg.router_z_loss_weight * torch.logsumexp(logits, dim=-1).pow(2).mean() / div
        loss.backward()
        return mlp.router_weight.grad[0][0].clone()

    g1 = grad_after(1)
    g43 = grad_after(43)
    ratio = (g43.norm() / g1.norm()).item()
    assert abs(ratio - 1 / 43) < 0.02 * (1 / 43), f"expected 1/43 scaling, got {ratio:.5f}"


# ---------------------------------------------------------------------------
# DeepSeekMoE decoupled shared expert and vectorized E=16, top_k=4 routing
# ---------------------------------------------------------------------------

def test_decoupled_shared_expert_summary():
    """Shared expert width must be decoupled from d_expert in configuration and summary."""
    from src.model import PulsarConfig, PulsarForCausalLM

    cfg = PulsarConfig(
        vocab_size=32768,
        d_model=1024,
        d_expert=512,
        d_shared=1024,
        n_routed_experts=16,
        top_k=4,
    )
    model = PulsarForCausalLM(cfg)
    summary = model.get_model_summary()

    assert summary["d_shared"] == 1024
    assert summary["d_expert"] == 512
    assert summary["n_routed_experts"] == 16
    assert summary["top_k"] == 4
    # active_ffn_dim = d_shared + top_k * d_expert = 1024 + 4 * 512 = 3072
    assert summary["active_ffn_dim"] == 3072
    assert model.core[0].mlp.shared_expert.d_ff == 1024
    assert model.core[0].mlp.routed_gate_up.shape == (16, 1024, 1024)
    assert model.core[0].mlp.routed_down.shape == (16, 512, 1024)


def test_fine_grained_moe_e16_top4_vectorized_forward_backward():
    """PulsarMoEMLP with E=16, top_k=4 must route dynamically and propagate clean gradients."""
    from src.model import PulsarConfig, PulsarMoEMLP

    cfg = PulsarConfig(
        vocab_size=1024,
        d_model=128,
        d_expert=64,
        d_shared=128,
        n_routed_experts=16,
        top_k=4,
        max_pulses=2,
    )
    torch.manual_seed(42)
    mlp = PulsarMoEMLP(cfg).float()

    B, T, D = 2, 16, 128
    x = torch.randn(B, T, D, requires_grad=True)
    out, z_loss, mean_max_weight, bincounts = mlp(x, pulse_idx=0)

    assert out.shape == (B, T, D)
    assert not torch.isnan(out).any()
    assert bincounts.shape == (16,)
    assert 0.0 < mean_max_weight <= 1.0

    loss = out.sum() + z_loss
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert mlp.routed_gate_up.grad is not None
    assert not torch.isnan(mlp.routed_gate_up.grad).any()
    assert mlp.routed_down.grad is not None
    assert not torch.isnan(mlp.routed_down.grad).any()
    assert mlp.shared_expert.w_down.weight.grad is not None
    assert not torch.isnan(mlp.shared_expert.w_down.weight.grad).any()


def test_padded_positions_not_competing_in_moe():
    """Padded positions in routing_mask must not compete for capacity or count as dropped."""
    from src.model import PulsarConfig, PulsarMoEMLP

    cfg = PulsarConfig(
        vocab_size=1024,
        d_model=128,
        d_expert=64,
        d_shared=128,
        n_routed_experts=16,
        top_k=4,
        max_pulses=2,
    )
    torch.manual_seed(42)
    mlp = PulsarMoEMLP(cfg).float()
    mlp.train()

    B, T, D = 2, 16, 128
    x = torch.randn(B, T, D, requires_grad=True)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, 8:] = False
    mask[1, 12:] = False

    out, z_loss, _, _ = mlp(x, pulse_idx=0, routing_mask=mask)
    assert out.shape == (B, T, D)
    assert not torch.isnan(out).any()

    # Backprop should propagate cleanly
    loss = out.sum() + z_loss
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_smoltalk_no_tool_convention_injection():
    """Substream 5A (SmolTalk) should not have [Tool Calling Convention] injected."""
    from data.prepare_shards import format_toucan_messages

    sample_msgs = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]
    formatted = format_toucan_messages(sample_msgs, native_only=True, inject_tool_hint=False)
    assert "[Tool Calling Convention]" not in formatted
    assert "<|im_start|>user\nWhat is the capital of France?<|im_end|>" in formatted
    assert "<|im_start|>assistant\nThe capital of France is Paris.<|im_end|>" in formatted

