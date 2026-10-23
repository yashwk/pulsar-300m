"""CPU-runnable tests: model wiring, loss, schedules, data loader, checkpoints.

These run without a GPU so the suite is usable on any machine. Anything that
requires CUDA lives in ``test_kernels.py`` / ``test_gpu_*.py`` and is marked ``gpu``.
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest
import torch

from conftest import make_cfg


# ---------------------------------------------------------------------------
# Model wiring and loss
# ---------------------------------------------------------------------------

def test_model_constructs_and_counts_params(tiny_cfg):
    from src.model import PulsarForCausalLM

    model = PulsarForCausalLM(tiny_cfg)
    n = sum(p.numel() for p in model.parameters())
    assert n > 0
    summary = model.get_model_summary(3)
    assert summary["total_params"] == n
    assert summary["effective_layers"] == tiny_cfg.n_core * 3


def test_forward_and_backward_cpu(tiny_cfg):
    from src.model import PulsarForCausalLM

    torch.manual_seed(0)
    model = PulsarForCausalLM(tiny_cfg)
    x = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
    y = torch.randint(0, tiny_cfg.vocab_size, (2, 16))

    logits, loss, metrics = model(x, labels=y, num_pulses=3)
    assert loss is not None and torch.isfinite(loss), "loss must be finite"
    assert logits is None  # training forward returns metrics, not logits

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads), "no gradients flowed"

    # every block's router must have produced a z-loss
    assert "pulse_losses" in metrics
    assert len(metrics["pulse_losses"]) == 3


def test_inference_forward_returns_logits(tiny_cfg):
    from src.model import PulsarForCausalLM

    torch.manual_seed(0)
    model = PulsarForCausalLM(tiny_cfg).eval()
    x = torch.randint(0, tiny_cfg.vocab_size, (1, 8))
    with torch.no_grad():
        logits, loss, _ = model(x, labels=None)
    assert logits.shape == (1, 8, tiny_cfg.vocab_size)
    assert loss is None


def test_loss_decreases_on_tiny_overfit(tiny_cfg):
    """A few Adam steps on one batch must reduce the loss."""
    from src.model import PulsarForCausalLM

    torch.manual_seed(0)
    model = PulsarForCausalLM(tiny_cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
    y = torch.randint(0, tiny_cfg.vocab_size, (2, 16))

    losses = []
    for _ in range(6):
        opt.zero_grad()
        _, loss, _ = model(x, labels=y, num_pulses=3)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert all(math.isfinite(v) for v in losses)
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


# ---------------------------------------------------------------------------
# LR schedule (WI-11)
# ---------------------------------------------------------------------------

def test_lr_schedule_warmup_plateau_cooldown():
    from src.train import get_lr_factor

    warmup, total, cooldown = 150, 1000, 200
    # warmup is linear and starts near zero
    assert get_lr_factor(1, warmup, total) == pytest.approx(1 / warmup, rel=1e-6)
    assert get_lr_factor(warmup, warmup, total) == pytest.approx(1.0, rel=1e-6)
    # plateau
    assert get_lr_factor(warmup + 10, warmup, total) == pytest.approx(1.0, rel=1e-6)
    # cooldown is monotone non-increasing and reaches the floor
    vals = [get_lr_factor(s, warmup, total, cooldown_steps=cooldown, min_ratio=0.01)
            for s in range(total - cooldown, total + 1)]
    assert all(b <= a + 1e-9 for a, b in zip(vals, vals[1:])), "cooldown must not increase LR"
    assert vals[-1] == pytest.approx(0.01, rel=1e-3)


@pytest.mark.xfail(
    reason="WI-11 not yet implemented: --min-lr-ratio defaults to 0.05 while "
           "docs/CURRICULUM.md documents a 1% floor. Remove this marker when WI-11 lands.",
    strict=True,
)
def test_min_lr_ratio_default_is_one_percent():
    """The CLI default must match docs/CURRICULUM.md after WI-11."""
    import inspect
    import re

    from src import train as train_mod

    src = inspect.getsource(train_mod.train)
    m = re.search(r'"--min-lr-ratio",\s*type=float,\s*default=([0-9.e-]+)', src)
    if m is None:
        pytest.skip("min-lr-ratio CLI definition not found (refactored?)")
    assert float(m.group(1)) == pytest.approx(0.01), "min-lr-ratio default should be 0.01"


# ---------------------------------------------------------------------------
# Batch loader (WI-07 groundwork)
# ---------------------------------------------------------------------------

def _write_fake_shard(path: str, tokens: np.ndarray) -> None:
    tokens.astype(np.uint16).tofile(path)


def test_loader_shapes_and_determinism(tmp_path):
    from src.train import StreamingPinnedBatchLoader

    tok = np.arange(10_000, dtype=np.uint16)
    p = tmp_path / "s.bin"
    _write_fake_shard(str(p), tok)
    shard = np.memmap(str(p), dtype=np.uint16, mode="r")

    loader = StreamingPinnedBatchLoader([shard], micro_batch_size=2, seq_len=32, device="cpu")
    x1, y1 = loader.get_batch(123)
    x2, y2 = loader.get_batch(123)
    assert x1.shape == (2, 32) and y1.shape == (2, 32)
    assert torch.equal(x1, x2) and torch.equal(y1, y2), "same seed must give the same batch"
    # y is x shifted by one within each window
    assert torch.equal(x1[:, 1:], y1[:, :-1])


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip_strict(tiny_cfg, tmp_path):
    from src.model import PulsarForCausalLM

    torch.manual_seed(0)
    model = PulsarForCausalLM(tiny_cfg)
    sd = model.state_dict()
    path = tmp_path / "m.pt"
    torch.save(sd, path)

    loaded = torch.load(path, map_location="cpu", weights_only=True)
    torch.manual_seed(1)
    model2 = PulsarForCausalLM(tiny_cfg)
    missing, unexpected = model2.load_state_dict(loaded, strict=False)
    assert not missing, f"unexpectedly missing keys: {missing}"
    assert not unexpected, f"unexpectedly unexpected keys: {unexpected}"


# ---------------------------------------------------------------------------
# WI-01: precision policy
# ---------------------------------------------------------------------------

import torch.nn as nn  # noqa: E402

from src.model import PulsarForCausalLM  # noqa: E402


def _tiny_model_for_precision():
    from conftest import make_cfg

    cfg = make_cfg(d_model=64, d_expert=64, n_heads=4, n_kv_heads=2, head_dim=16, n_core=2)
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg)
    model.to_training_precision(torch.float32)  # CPU: keep a single dtype end-to-end
    return model, cfg


def test_dtypes_learnable_scales_are_fp32():
    """The tensors the audit found silently frozen must be FP32 after the cast.

    Previously train.py cast everything to bf16 and re-cast a name-based subset, which
    missed pulse_res_scale, routed_scale, attn_gate and pulse_step_emb. bf16 half-spacing
    near 1.0 is 3.9e-3 while the peak Adam step is 6e-4, so those tensors could not move
    at all.
    """
    model, _ = _tiny_model_for_precision()
    params = dict(model.named_parameters())

    must_be_fp32 = [
        "pulse_res_scale",
        "pulse_step_emb.weight",
        "input_mix_logit",
        "core.0.mlp.routed_scale",
        "core.0.attn.attn_gate",
        "core.0.attn.attn_gate_bias",
        "core.0.attn.v_resid_lambda",
        "core.0.attn.q_norm_weight",
        "core.0.mlp.router_weight",
        "core.0.attn_norm.weight",
    ]
    for name in must_be_fp32:
        assert name in params, f"missing parameter {name}"
        assert params[name].dtype == torch.float32, f"{name} is {params[name].dtype}, expected float32"


def test_dtypes_buffers_are_fp32():
    """MoE counters and RoPE tables must not be cast to bf16.

    Expert counts accumulate into the hundreds; bf16 saturates them, so the
    auxiliary-loss-free bias update degenerates into rounding noise.
    """
    model, _ = _tiny_model_for_precision()
    buffers = dict(model.named_buffers())
    for name, buf in buffers.items():
        if any(s in name for s in ("expert_bias", "expert_counts_accum", "expert_pulse_counts",
                                   "inv_freq", "cos_cached", "sin_cached")):
            assert buf.dtype == torch.float32, f"buffer {name} is {buf.dtype}, expected float32"


def test_to_training_precision_is_idempotent_and_returns_self():
    model, _ = _tiny_model_for_precision()
    out = model.to_training_precision(torch.float32)
    assert out is model
    before = {n: p.dtype for n, p in model.named_parameters()}
    model.to_training_precision(torch.float32)
    after = {n: p.dtype for n, p in model.named_parameters()}
    assert before == after


def test_no_fp32_promotion_in_forward():
    """The residual stream and every block output must stay in the activation dtype.

    The audit's specific hazard: pulse_step_emb is FP32, and adding it to a bf16
    residual promotes everything downstream to FP32. On CPU we assert against the
    float32 model that all intermediates are float32 (i.e. no float64/implicit
    promotion), which catches accidental upcasting to double.
    """
    model, cfg = _tiny_model_for_precision()
    model.eval()
    seen = []

    def hook(name):
        def fn(module, inputs, output):
            t = output[0] if isinstance(output, tuple) else output
            if isinstance(t, torch.Tensor):
                seen.append((name, t.dtype))
        return fn

    handles = []
    for i, blk in enumerate(model.core):
        handles.append(blk.register_forward_hook(hook(f"block{i}")))
    handles.append(model.pulse_step_emb.register_forward_hook(hook("step_emb")))

    x = torch.randint(0, cfg.vocab_size, (1, 8))
    y = torch.randint(0, cfg.vocab_size, (1, 8))
    _ = model(x, labels=y, num_pulses=3)
    for h in handles:
        h.remove()

    assert seen, "no forward hooks fired"
    bad = [(n, d) for n, d in seen if d != torch.float32]
    assert not bad, f"unexpected dtypes in forward: {bad}"


def test_scales_actually_learn():
    """Regression test for the frozen-scale bug.

    With the old precision handling these parameters were bf16 and could not move at
    all. Assert they change measurably over a few optimizer steps.
    """
    model, cfg = _tiny_model_for_precision()
    model.train()

    names = ["pulse_res_scale", "core.0.mlp.routed_scale"]
    params = dict(model.named_parameters())
    start = {n: params[n].detach().clone() for n in names}

    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randint(0, cfg.vocab_size, (2, 12))
    y = torch.randint(0, cfg.vocab_size, (2, 12))

    for _ in range(8):
        opt.zero_grad()
        _, loss, _ = model(x, labels=y, num_pulses=3)
        loss.backward()
        opt.step()

    for n in names:
        delta = (params[n].detach() - start[n]).abs().max().item()
        assert delta > 1e-6, f"{n} did not learn (max delta {delta:.2e})"


@pytest.mark.gpu
def test_bf16_precision_policy_on_gpu():
    """On GPU, verify the real training precision policy in bf16.

    This is the configuration where the bug bit: matrices bf16, scale/norm/router
    tensors FP32, and no FP32 promotion of the bf16 residual stream.
    """
    from conftest import make_cfg

    cfg = make_cfg(d_model=64, d_expert=64, n_heads=4, n_kv_heads=2, head_dim=16, n_core=2)
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg).cuda()
    model.to_training_precision(torch.bfloat16)

    params = dict(model.named_parameters())
    # matrix weights are bf16 (updated via stochastic rounding in WI-02)
    assert params["core.0.attn.wqkv.weight"].dtype == torch.bfloat16
    # small learnable tensors stay FP32
    assert params["pulse_res_scale"].dtype == torch.float32
    assert params["core.0.mlp.routed_scale"].dtype == torch.float32
    assert params["pulse_step_emb.weight"].dtype == torch.float32

    # residual stream must remain bf16 through every block
    seen = []
    handles = [
        blk.register_forward_hook(
            lambda m, i, o, _n=f"b{idx}": seen.append((_n, (o[0] if isinstance(o, tuple) else o).dtype))
        )
        for idx, blk in enumerate(model.core)
    ]
    x = torch.randint(0, cfg.vocab_size, (1, 8), device="cuda")
    model.eval()
    with torch.no_grad():
        model(x, labels=None, num_pulses=3)
    for h in handles:
        h.remove()

    bad = [(n, d) for n, d in seen if d != torch.bfloat16]
    assert not bad, f"FP32 promotion into the residual stream: {bad}"
