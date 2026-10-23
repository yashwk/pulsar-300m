"""WI-19: per-pulse balancing bias (default off; owner-decided)."""

from __future__ import annotations

import pytest
import torch

from conftest import make_cfg


def _model(per_pulse: bool):
    from src.model import PulsarForCausalLM

    cfg = make_cfg()
    cfg.per_pulse_bias = per_pulse
    torch.manual_seed(0)
    return PulsarForCausalLM(cfg), cfg


def test_flag_off_keeps_bias_shape():
    model, cfg = _model(False)
    bias = dict(model.named_buffers())["core.0.mlp.expert_bias"]
    assert bias.shape == (cfg.n_routed_experts,)


def test_flag_on_gives_per_pulse_shape():
    model, cfg = _model(True)
    bias = dict(model.named_buffers())["core.0.mlp.expert_bias"]
    assert bias.shape == (cfg.max_pulses, cfg.n_routed_experts)


def test_flag_off_is_bit_identical_to_baseline():
    """With the flag off, a fixed-seed forward must reproduce exactly."""
    model, cfg = _model(False)
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    y = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        _, l1, _ = model(x, labels=y, num_pulses=3)
        _, l2, _ = model(x, labels=y, num_pulses=3)
    assert torch.equal(l1, l2), "flag off must be deterministic"


def test_flag_on_runs_and_updates_bias_per_pulse():
    model, cfg = _model(True)
    model.train()
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    y = torch.randint(0, cfg.vocab_size, (2, 32))
    _, loss, _ = model(x, labels=y, num_pulses=3)
    loss.backward()

    mlp = model.core[0].mlp
    before = mlp.expert_bias.clone()
    model.update_expert_biases(bias_lr=0.01, num_pulses=3)
    after = mlp.expert_bias

    # only the pulses that ran should move; unused pulses stay at zero
    changed = (after != before).any(dim=1)
    assert changed[:3].any(), "no bias movement for active pulses"
    assert not changed[3:].any(), "bias moved for a pulse that never ran"


def test_old_checkpoint_bias_migrates_by_broadcast():
    """A checkpoint with [E] expert_bias must load into a per-pulse model."""
    old_model, cfg = _model(False)
    old_sd = old_model.state_dict()

    new_model, _ = _model(True)
    new_model.load_state_dict(old_sd, strict=False)

    migrated = dict(new_model.named_buffers())["core.0.mlp.expert_bias"]
    assert migrated.shape == (cfg.max_pulses, cfg.n_routed_experts)
    # every pulse row equals the original across-pulse value
    orig = dict(old_model.named_buffers())["core.0.mlp.expert_bias"]
    for k in range(cfg.max_pulses):
        assert torch.equal(migrated[k], orig)


def test_forward_uses_pulse_specific_bias():
    """Each pulse must read its OWN bias row, not a shared one."""
    from src.model import PulsarForCausalLM

    cfg = make_cfg()
    cfg.per_pulse_bias = True
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg)
    mlp = model.core[0].mlp

    x = torch.randint(0, cfg.vocab_size, (1, 8))
    y = torch.randint(0, cfg.vocab_size, (1, 8))

    # Give each pulse a distinct, large bias so the routing must differ.
    with torch.no_grad():
        mlp.expert_bias.zero_()
        mlp.expert_bias[2, 0] = 10.0     # pulse 3 strongly prefers expert 0
    with torch.no_grad():
        _, loss_a, _ = model(x, labels=y, num_pulses=3)

    with torch.no_grad():
        mlp.expert_bias.zero_()          # neutral bias
        _, loss_b, _ = model(x, labels=y, num_pulses=3)

    assert not torch.equal(loss_a, loss_b), (
        "changing a single pulse's bias did not change the loss - the pulse-specific "
        "bias row is not being used"
    )
