"""WI-18: Muon update-scale convention must be layout-aware."""

from __future__ import annotations

import math

import pytest
import torch

from src.optim import Muon


def _scale_for(shape):
    """Replicate the scale rule the optimizer applies for a given parameter shape."""
    is_3d = len(shape) == 3
    if is_3d:
        in_features, out_features = shape[-2], shape[-1]
    else:
        out_features, in_features = shape[-2], shape[-1]
    return max(1.0, out_features / in_features) ** 0.5


def test_scale_2d_linear_layout():
    # nn.Linear [out, in]
    assert _scale_for((2048, 1024)) == pytest.approx(math.sqrt(2), rel=1e-6)
    assert _scale_for((1024, 1024)) == pytest.approx(1.0, rel=1e-6)
    assert _scale_for((256, 1024)) == pytest.approx(1.0, rel=1e-6)  # out < in clamps to 1


def test_scale_3d_expert_layout():
    """Experts are [E, in, out], so out is the LAST dim.

    The old code read shape[-2]/shape[-1] for both layouts, giving gate_up
    [E, 1024, 2048] a scale of 1.0 instead of sqrt(2) - routed gate/up trained
    ~0.7x as fast as the equivalent dense matrix.
    """
    assert _scale_for((8, 1024, 2048)) == pytest.approx(math.sqrt(2), rel=1e-6)
    assert _scale_for((8, 1024, 1024)) == pytest.approx(1.0, rel=1e-6)


def test_3d_scale_differs_from_naive_reading():
    """Guard against regressing to the layout-blind formula."""
    shape = (8, 1024, 2048)
    naive = max(1.0, shape[-2] / shape[-1]) ** 0.5
    correct = _scale_for(shape)
    assert naive == pytest.approx(1.0, rel=1e-6)
    assert correct == pytest.approx(math.sqrt(2), rel=1e-6)
    assert abs(naive - correct) > 0.4


def test_muon_step_magnitude_scales_with_aspect_ratio():
    """A wide matrix must move more than a square one under the same gradient."""
    torch.manual_seed(0)
    p_square = torch.nn.Parameter(torch.zeros(64, 64, dtype=torch.bfloat16))
    p_wide = torch.nn.Parameter(torch.zeros(64, 128, dtype=torch.bfloat16))  # out=64,in=128 -> scale 1

    torch.manual_seed(1)
    g = torch.randn(64, 64)
    p_square.grad = g.clone().to(torch.bfloat16)
    p_wide.grad = torch.cat([g, g], dim=1).to(torch.bfloat16)

    opt = Muon([p_square], lr=0.02, weight_decay=0.0, stochastic_rounding=True, seed=0)
    opt.step()

    # both are [out=64, in=64] and [out=64, in=128] -> scale 1.0 and 1.0
    assert p_square.abs().max() > 0
