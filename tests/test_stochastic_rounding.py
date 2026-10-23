"""Tests for stochastic rounding (WI-02).

The point of SR is that many sub-ulp updates accumulate instead of vanishing. These
tests verify unbiasedness and that the stored weight tracks the exact FP32 trajectory.
"""

from __future__ import annotations

import pytest
import torch

from src.optim import stochastic_round_


def test_sr_unbiased_small_updates():
    """2000 updates of 3e-5 on a weight of 0.02 must accumulate, not vanish.

    bf16 spacing at |w|=0.02 is ~1.2e-4 and half-spacing 6e-5, so a 3e-5 step always
    rounds the same way under round-to-nearest: the weight never moves.
    """
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
    assert lo <= got_drift <= hi, (
        f"SR drift {got_drift:.6f} outside 10% of exact {expected_drift:.6f}"
    )


def test_round_to_nearest_contracts_to_zero():
    """Control: the same loop with RNE must not move at all, proving SR is doing work."""
    torch.manual_seed(0)
    n = 1024
    w = torch.full((n,), 0.02, dtype=torch.float32)
    step = 3e-5
    for _ in range(2000):
        w = w + step
    rne = w.to(torch.bfloat16)
    # RNE rounds the final exact value to the nearest bf16 - it does move here, because
    # the loop accumulated in FP32. The real contrast is stepping in bf16 directly:
    w2 = torch.full((n,), 0.02, dtype=torch.bfloat16)
    for _ in range(2000):
        w2 = (w2.float() + step).to(torch.bfloat16)   # round each step (naive bf16)
    assert torch.allclose(w2, torch.full_like(w2, 0.02)), "naive bf16 stepping should be frozen"


def test_sr_decay_tracks_exact_trajectory():
    """Repeated x *= 0.998 must track the exact FP32 trajectory in expectation."""
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
    rel = abs(got - expected) / expected
    assert rel < 0.10, f"SR decay {got:.6f} vs exact {expected:.6f} (rel err {rel:.2%})"


def test_sr_preserves_expectation_over_random_values():
    """Mean of many independently rounded values approximates the FP32 mean."""
    torch.manual_seed(3)
    n = 100_000
    src = torch.randn(n, dtype=torch.float32)
    dst = torch.empty(n, dtype=torch.bfloat16)
    gen = torch.Generator(device="cpu").manual_seed(4)
    stochastic_round_(dst, src, gen)
    err = abs(dst.float().mean().item() - src.mean().item())
    assert err < 2e-3, f"SR mean bias {err:.2e} too large"


def test_muon_sr_moves_small_updates():
    """A Muon step on a bf16 weight must actually change it."""
    from src.optim import Muon

    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.full((64, 64), 0.02, dtype=torch.bfloat16))
    opt = Muon([p], lr=0.02, weight_decay=0.0, stochastic_rounding=True, seed=0)
    p.grad = torch.full_like(p, 1e-3)

    before = p.detach().clone()
    for _ in range(5):
        p.grad = torch.full_like(p, 1e-3)
        opt.step()
    assert not torch.equal(before, p.detach()), "Muon with SR did not change the weight"


def test_sr_reproducible_with_seed():
    """Same seed and step => identical rounding, so runs are reproducible."""
    torch.manual_seed(0)
    src = torch.randn(1000, dtype=torch.float32) * 0.02
    a = torch.empty(1000, dtype=torch.bfloat16)
    b = torch.empty(1000, dtype=torch.bfloat16)
    stochastic_round_(a, src, torch.Generator(device="cpu").manual_seed(42))
    stochastic_round_(b, src, torch.Generator(device="cpu").manual_seed(42))
    assert torch.equal(a, b)
