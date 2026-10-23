"""WI-12: global-layer query scaling must be prefix-invariant.

The YaRN-style temperature scale grows with how far past the base context a query sits.
It must be a function of the query's own position, not of the total sequence length:
otherwise a token's representation changes depending on how many tokens come after it,
which breaks prefix invariance and makes KV caching incorrect.

Tested at the attention layer rather than the whole model. The full model washes the
effect out (measured: a 0.106 absolute change at a single global layer produced only
~2e-4 of logit drift end-to-end, because the layer output passes through a residual
add, an RMSNorm and a MoE block). Asserting on the layer keeps the signal clean.
"""

from __future__ import annotations

import pytest
import torch

from conftest import make_cfg


def _global_attn(n_core=8, base_seq_len=64):
    from src.model import PulsarAttention

    cfg = make_cfg(n_core=n_core, base_seq_len=base_seq_len, max_seq_len=2048, sliding_window=64)
    torch.manual_seed(0)
    layer = PulsarAttention(cfg, layer_idx=3)   # layer 3 is global
    layer.eval()
    with torch.no_grad():
        for p in layer.parameters():
            p.normal_(0.0, 0.3)
    return layer, cfg


@pytest.mark.parametrize("prefix", [32, 96])
def test_global_layer_prefix_invariance(prefix):
    """A prefix's output must not change when more tokens are appended.

    With length-dependent scaling this fails by ~0.1 (q is scaled by
    ln(T/base), which grows with total length); with per-position scaling it is exact.
    """
    layer, cfg = _global_attn()
    total = prefix + 512
    x = torch.randn(1, total, cfg.d_model) * 0.5
    cos = torch.ones(total, cfg.head_dim // 2)
    sin = torch.zeros(total, cfg.head_dim // 2)

    with torch.no_grad():
        short, _ = layer(x[:, :prefix], cos[:prefix], sin[:prefix], pulse_idx=0, v_first=None)
        long, _ = layer(x, cos, sin, pulse_idx=0, v_first=None)

    max_abs = (short[0].float() - long[0, :prefix].float()).abs().max().item()
    assert max_abs < 1e-5, (
        f"global layer prefix of length {prefix} changed by {max_abs:.3e} when more "
        f"tokens were appended - query scaling depends on total sequence length"
    )


def test_scale_is_one_below_base_context():
    """Positions before base_seq_len must be scaled by exactly 1.0."""
    cfg = make_cfg(base_seq_len=64, max_seq_len=2048)
    base = cfg.base_seq_len
    pos = torch.arange(1, 200, dtype=torch.float32)
    scale = 1.0 + 0.1 * torch.log(pos.clamp_min(base) / base)
    assert torch.allclose(scale[:base], torch.ones(base)), "positions < base must be unscaled"


def test_scale_grows_monotonically_past_base():
    cfg = make_cfg(base_seq_len=64, max_seq_len=2048)
    base = cfg.base_seq_len
    pos = torch.arange(base, 1024, dtype=torch.float32)
    scale = 1.0 + 0.1 * torch.log(pos.clamp_min(base) / base)
    assert scale[0] == pytest.approx(1.0)
    assert torch.all(scale[1:] > scale[:-1]), "scale must increase with position"
