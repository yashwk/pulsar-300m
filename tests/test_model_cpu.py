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
