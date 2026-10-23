"""WI-09: local attention must not build a dense T x T mask, and must match it numerically.

The old implementation allocated T x T int64 index tensors (8.6 GB at T=32768) plus a
boolean mask (1.07 GB) purely to express "causal and within `window`". On the target
8 GB card that is impossible at long context, and passing a mask to SDPA forces the math
backend. flex_attention expresses the same pattern as a BlockMask: O(T) metadata, fused
kernel.
"""

from __future__ import annotations

import pytest
import torch

from conftest import HAS_CUDA, make_cfg


def _layer(impl, base_seq_len=64, window=64, n_core=8):
    from src.model import PulsarAttention

    cfg = make_cfg(n_core=n_core, base_seq_len=base_seq_len, max_seq_len=4096, sliding_window=window)
    cfg.local_attn_impl = impl
    torch.manual_seed(0)
    layer = PulsarAttention(cfg, layer_idx=0)   # layer 0 is local
    if HAS_CUDA:
        layer = layer.cuda()
    layer.eval()
    with torch.no_grad():
        for p in layer.parameters():
            p.normal_(0.0, 0.2)
    return layer, cfg


@pytest.mark.skipif(not HAS_CUDA, reason="flex_attention requires CUDA here")
def test_flex_matches_sdpa_mask_forward():
    """The fused path must reproduce the dense-mask reference."""
    T = 256
    flex_layer, cfg = _layer("flex")
    ref_layer, _ = _layer("sdpa_mask")
    ref_layer.load_state_dict(flex_layer.state_dict())   # identical weights

    x = torch.randn(1, T, cfg.d_model, device="cuda") * 0.5
    cos = torch.ones(T, cfg.head_dim // 2, device="cuda")
    sin = torch.zeros(T, cfg.head_dim // 2, device="cuda")

    with torch.no_grad():
        a, _ = flex_layer(x, cos, sin, pulse_idx=0, v_first=None)
        b, _ = ref_layer(x, cos, sin, pulse_idx=0, v_first=None)

    max_abs = (a.float() - b.float()).abs().max().item()
    assert max_abs < 2e-2, f"flex vs dense mask differ by {max_abs:.3e}"


@pytest.mark.skipif(not HAS_CUDA, reason="flex_attention requires CUDA here")
def test_flex_backward_matches_reference():
    T = 128
    flex_layer, cfg = _layer("flex")
    ref_layer, _ = _layer("sdpa_mask")
    ref_layer.load_state_dict(flex_layer.state_dict())

    x1 = (torch.randn(1, T, cfg.d_model, device="cuda") * 0.5).requires_grad_(True)
    x2 = x1.detach().clone().requires_grad_(True)
    cos = torch.ones(T, cfg.head_dim // 2, device="cuda")
    sin = torch.zeros(T, cfg.head_dim // 2, device="cuda")

    a, _ = flex_layer(x1, cos, sin, pulse_idx=0, v_first=None)
    a.sum().backward()
    b, _ = ref_layer(x2, cos, sin, pulse_idx=0, v_first=None)
    b.sum().backward()

    gmax = (x1.grad.float() - x2.grad.float()).abs().max().item()
    assert gmax < 5e-2, f"backward differ by {gmax:.3e}"


def test_no_dense_mask_allocation():
    """The reference path allocates T x T; the flex path must not.

    On CPU we can only assert the shape of what the reference builds and that the flex
    path stays linear in the block-mask metadata it caches.
    """
    from src.model import _get_local_block_mask

    if not HAS_CUDA:
        pytest.skip("block masks are built on CUDA in this environment")
    T, window = 4096, 1024
    bm = _get_local_block_mask(1, 8, T, window, torch.device("cuda"))
    # BlockMask stores per-block metadata, not a T x T tensor.
    dense_bytes = T * T
    meta_bytes = sum(
        getattr(bm, name).numel() * getattr(bm, name).element_size()
        for name in ("kv_num_blocks", "kv_indices")
        if hasattr(bm, name) and isinstance(getattr(bm, name), torch.Tensor)
    )
    assert meta_bytes < dense_bytes / 8, (
        f"BlockMask metadata {meta_bytes} bytes is not clearly smaller than a dense "
        f"bool mask ({dense_bytes} bytes)"
    )


def test_auto_falls_back_to_reference_without_flex():
    """With flex disabled, 'auto' must still work via the dense reference path."""
    import src.model as M

    layer, cfg = _layer("auto")
    dev = next(layer.parameters()).device
    x = torch.randn(1, 256, cfg.d_model, device=dev) * 0.5
    cos = torch.ones(256, cfg.head_dim // 2, device=dev)
    sin = torch.zeros(256, cfg.head_dim // 2, device=dev)

    orig = M.flex_enabled
    try:
        M.flex_enabled = False
        with torch.no_grad():
            out, _ = layer(x, cos, sin, pulse_idx=0, v_first=None)
    finally:
        M.flex_enabled = orig
    assert out.shape == (1, 256, cfg.d_model)
    assert torch.isfinite(out).all()
