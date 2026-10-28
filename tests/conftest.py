"""Shared fixtures and helpers for the Pulsar test suite.

Two model scales are available:

* ``tiny_cfg``   - the default; small enough that the full kernel/reference suite runs
                  on CPU in seconds. Every structural feature of the real model is
                  preserved (recurrent core, per-pulse params, GQA, MoE, deep
                  supervision).
* ``gpu_cfg``    - the real training geometry, used only by GPU tests.

Numerical-comparison helpers live here so every kernel test uses the same tolerance
policy: custom CUDA kernels are compared *both* against an FP32 reference (absolute
tolerance) *and* against the pure-PyTorch bf16 path (relative tolerance), which is
what the audit asks for - a fast kernel should be no less accurate than the
straightforward PyTorch implementation it replaces.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import PulsarConfig, PulsarForCausalLM  # noqa: E402

HAS_CUDA = torch.cuda.is_available()

try:  # pragma: no cover - trivial
    import pulsar_cuda  # noqa: F401
    HAS_PULSAR_CUDA = HAS_CUDA
except ImportError:  # pragma: no cover
    HAS_PULSAR_CUDA = False


def make_cfg(**overrides) -> PulsarConfig:
    """Tiny config: every architectural feature, small enough for CPU."""
    base = dict(
        vocab_size=512,
        d_model=128,
        n_heads=4,
        n_kv_heads=2,
        head_dim=32,
        d_expert=128,
        d_shared=128,
        n_core=2,
        max_pulses=4,
        default_pulses=3,
        base_seq_len=64,
        max_seq_len=512,
        sliding_window=64,
        n_routed_experts=8,
        top_k=2,
        ce_chunk_size=128,
    )
    base.update(overrides)
    return PulsarConfig(**base)


def make_gpu_cfg(**overrides) -> PulsarConfig:
    """The real training geometry (sm_89, 8 GB)."""
    base = dict(max_seq_len=32768, base_seq_len=1024, default_pulses=3)
    base.update(overrides)
    return PulsarConfig(**base)


@pytest.fixture
def tiny_cfg():
    return make_cfg()


@pytest.fixture
def tiny_model(tiny_cfg):
    torch.manual_seed(0)
    model = PulsarForCausalLM(tiny_cfg)
    model.eval()
    return model


@pytest.fixture
def gpu():
    if not HAS_CUDA:
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def cuda_ext():
    if not HAS_PULSAR_CUDA:
        pytest.skip("pulsar_cuda extension not built/available")
    return True


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 2e-2,
    rtol: float = 2e-2,
    msg: str = "",
) -> None:
    """Assert closeness on the caller's tolerance, failing with a useful message."""
    a = actual.detach().float()
    e = expected.detach().float()
    if a.shape != e.shape:
        raise AssertionError(f"{msg}: shape mismatch {tuple(a.shape)} vs {tuple(e.shape)}")
    max_abs = (a - e).abs().max().item() if a.numel() else 0.0
    denom = e.abs().clamp_min(1e-6)
    max_rel = ((a - e).abs() / denom).max().item() if a.numel() else 0.0
    if not (math.isfinite(max_abs) and math.isfinite(max_rel)):
        raise AssertionError(f"{msg}: non-finite values (max_abs={max_abs}, max_rel={max_rel})")
    if max_abs > atol and max_rel > rtol:
        raise AssertionError(
            f"{msg}: max_abs={max_abs:.3e} (atol {atol:.1e}), "
            f"max_rel={max_rel:.3e} (rtol {rtol:.1e})"
        )


def grad_close(a: torch.Tensor | None, b: torch.Tensor | None, *, atol: float = 2e-2, rtol: float = 2e-2, msg: str = "") -> None:
    assert a is not None and b is not None, f"{msg}: missing gradient"
    assert_close(a, b, atol=atol, rtol=rtol, msg=msg)


def fp32_reference_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (w.float() * xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def peak_memory_gb(device: torch.device | None = None) -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1e9


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
