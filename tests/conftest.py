"""Shared fixtures and test helpers for the Pulsar test suite."""

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

try:
    import pulsar_cuda  # noqa: F401
    HAS_PULSAR_CUDA = HAS_CUDA
except ImportError:
    HAS_PULSAR_CUDA = False

needs_gpu = pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU not available")
needs_cuda_ext = pytest.mark.skipif(not HAS_PULSAR_CUDA, reason="pulsar_cuda extension not available")


def make_cfg(**overrides) -> PulsarConfig:
    """Fast, lightweight configuration for CPU and behavioral unit tests."""
    base = dict(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_expert=32,
        d_shared=64,
        n_core=2,
        n_routed_experts=4,
        top_k=2,
        max_pulses=4,
        default_pulses=3,
        base_seq_len=32,
        max_seq_len=128,
        sliding_window=16,
        ce_chunk_size=64,
    )
    base.update(overrides)
    return PulsarConfig(**base)


@pytest.fixture
def tiny_cfg() -> PulsarConfig:
    return make_cfg()


@pytest.fixture
def tiny_model_cpu(tiny_cfg) -> PulsarForCausalLM:
    torch.manual_seed(0)
    m = PulsarForCausalLM(tiny_cfg)
    m.eval()
    return m


@pytest.fixture
def tiny_model(tiny_model_cpu) -> PulsarForCausalLM:
    """Backward-compatible alias for tiny_model_cpu."""
    return tiny_model_cpu


@pytest.fixture
def tiny_model_cuda(tiny_cfg) -> PulsarForCausalLM:
    if not HAS_CUDA:
        pytest.skip("CUDA not available")
    torch.manual_seed(0)
    m = PulsarForCausalLM(tiny_cfg).to("cuda")
    m.eval()
    return m


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


@pytest.fixture
def synthetic_shards(tmp_path):
    """Generate tiny deterministic binary training and validation shards for loader tests."""
    import numpy as np
    rng = np.random.RandomState(42)
    train_file = tmp_path / "train_shard0.bin"
    val_file = tmp_path / "val_shard0.bin"

    def make_shard(path, num_tokens=2000):
        tokens = rng.randint(1, 256, size=num_tokens, dtype=np.uint16)
        # Place document boundary (token 0) every 40 tokens
        for idx in range(40, num_tokens, 40):
            tokens[idx] = 0
        tokens.tofile(path)
        return str(path)

    t_path = make_shard(train_file, 2000)
    v_path = make_shard(val_file, 500)
    return t_path, v_path


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 2e-2,
    rtol: float = 2e-2,
    msg: str = "",
) -> None:
    """Assert closeness on the caller's tolerance with detailed diagnostics."""
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
            f"{msg}: max_abs={max_abs:.3e} (atol {atol:.1e}), max_rel={max_rel:.3e} (rtol {rtol:.1e})"
        )


def grad_close(
    a: torch.Tensor | None,
    b: torch.Tensor | None,
    *,
    atol: float = 2e-2,
    rtol: float = 2e-2,
    msg: str = "",
) -> None:
    assert a is not None and b is not None, f"{msg}: missing gradient"
    assert_close(a, b, atol=atol, rtol=rtol, msg=msg)


def fp32_reference_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (w.float() * xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)
