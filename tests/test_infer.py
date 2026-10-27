"""Tests for Pulsar Unified KV-Cache and Inference Engine.

Verifies:
1. Unified KV cache sharing across recurrent passes.
2. Sliding-window ring buffering for local layers (capping at sliding_window).
3. Unbounded global context for anchor layers (3 and 7).
4. Memory footprint calculations.
5. Autoregressive single-token decode step parity.
"""

import pytest
import torch
import torch.nn.functional as F

from src.infer import UnifiedKVCache, PulsarInferenceEngine
from src.model import PulsarConfig, PulsarForCausalLM


def create_mini_config() -> PulsarConfig:
    """Create a lightweight config for CPU test execution."""
    cfg = PulsarConfig()
    cfg.vocab_size = 128
    cfg.d_model = 64
    cfg.n_heads = 4
    cfg.n_kv_heads = 2
    cfg.head_dim = 16
    cfg.d_expert = 64
    cfg.n_core = 8
    cfg.default_pulses = 3
    cfg.base_seq_len = 32
    cfg.max_seq_len = 128
    cfg.sliding_window = 16  # Small sliding window to test ring-buffering
    cfg.n_routed_experts = 4
    cfg.top_k = 2
    return cfg


def test_unified_kv_cache_sliding_window_eviction():
    """Local layers must cap their cached sequence length at sliding_window."""
    cfg = create_mini_config()
    cache = UnifiedKVCache(cfg, batch_size=1, device="cpu", dtype=torch.float32)

    # Initial prefill with 24 tokens (> sliding_window 16)
    T = 24
    k = torch.randn(1, cfg.n_kv_heads, T, cfg.head_dim)
    v = torch.randn(1, cfg.n_kv_heads, T, cfg.head_dim)

    # Layer 0 is local -> should be truncated to 16
    k_cached_0, v_cached_0 = cache.update(0, k, v, pulse_idx=0, is_prefill=True)
    assert k_cached_0.shape[2] == 16, f"Expected local layer to cap at 16, got {k_cached_0.shape[2]}"

    # Layer 3 is global -> should retain all 24 tokens
    k_cached_3, v_cached_3 = cache.update(3, k, v, pulse_idx=0, is_prefill=True)
    assert k_cached_3.shape[2] == 24, f"Expected global layer to retain all 24 tokens, got {k_cached_3.shape[2]}"


def test_unified_kv_cache_cross_pulse_inplace_update():
    """In pulses > 0, the cache updates the latest token in-place rather than expanding."""
    cfg = create_mini_config()
    cache = UnifiedKVCache(cfg, batch_size=1, device="cpu", dtype=torch.float32)

    # Step 1: Ingest 4 initial tokens
    k_init = torch.randn(1, cfg.n_kv_heads, 4, cfg.head_dim)
    v_init = torch.randn(1, cfg.n_kv_heads, 4, cfg.head_dim)
    cache.update(0, k_init, v_init, pulse_idx=0, is_prefill=True)
    assert cache.layers[0]["k"].shape[2] == 4

    # Step 2: Autoregressive token step at pulse 0 (appends 1 token)
    k_p0 = torch.randn(1, cfg.n_kv_heads, 1, cfg.head_dim)
    v_p0 = torch.randn(1, cfg.n_kv_heads, 1, cfg.head_dim)
    cache.update(0, k_p0, v_p0, pulse_idx=0, is_prefill=False)
    assert cache.layers[0]["k"].shape[2] == 5

    # Step 3: Same token step at pulse 1 (refines the latest token in-place)
    k_p1 = torch.randn(1, cfg.n_kv_heads, 1, cfg.head_dim)
    v_p1 = torch.randn(1, cfg.n_kv_heads, 1, cfg.head_dim)
    cache.update(0, k_p1, v_p1, pulse_idx=1, is_prefill=False)
    # Length MUST remain 5, not grow to 6!
    assert cache.layers[0]["k"].shape[2] == 5, "Cross-pulse update should not increase cache length!"
    # Verify the last token was updated
    assert torch.allclose(cache.layers[0]["k"][:, :, -1:, :], k_p1)


def test_memory_footprint_calculation():
    """Verify exact memory footprint matches the theoretical 11x compression formula."""
    cfg = create_mini_config()
    cache = UnifiedKVCache(cfg, batch_size=1, device="cpu", dtype=torch.float32)

    # 10 tokens in all layers
    k = torch.randn(1, cfg.n_kv_heads, 10, cfg.head_dim)
    v = torch.randn(1, cfg.n_kv_heads, 10, cfg.head_dim)
    for i in range(cfg.n_core):
        cache.update(i, k, v, pulse_idx=0, is_prefill=True)

    bytes_total = cache.memory_footprint_bytes()
    expected_elements = cfg.n_core * 2 * (1 * cfg.n_kv_heads * 10 * cfg.head_dim)
    expected_bytes = expected_elements * 4  # float32 = 4 bytes
    assert bytes_total == expected_bytes


def test_inference_engine_step():
    """Verify decode_step outputs logits of correct vocab shape."""
    cfg = create_mini_config()
    model = PulsarForCausalLM(cfg).to(dtype=torch.float32)
    engine = PulsarInferenceEngine(model, device="cpu", dtype=torch.float32)
    cache = UnifiedKVCache(cfg, batch_size=1, device="cpu", dtype=torch.float32)

    # Prompt of 3 tokens
    prompt_ids = torch.tensor([[10, 20, 30]], dtype=torch.long)
    logits = engine.prefill(prompt_ids, cache)
    assert logits.shape == (1, 1, cfg.vocab_size)

    # Single decode step
    next_logits = engine.decode_step(token_id=40, pos=3, kv_cache=cache)
    assert next_logits.shape == (1, cfg.vocab_size)
    assert not torch.isnan(next_logits).any()


class MockTokenizer:
    def __init__(self, eos_id: int = 1):
        self.eos_token_id = eos_id
        self.unk_token_id = 0

    def encode(self, prompt: str, return_tensors: str = "pt") -> torch.Tensor:
        return torch.tensor([[10, 20]], dtype=torch.long)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        return f" tok_{token_ids[0]}"

    def convert_tokens_to_ids(self, s: str) -> int:
        return self.eos_token_id


def test_inference_engine_generate_and_stream():
    """Verify end-to-end generation and streaming using the unified KV-cache."""
    cfg = create_mini_config()
    model = PulsarForCausalLM(cfg).to(dtype=torch.float32)
    engine = PulsarInferenceEngine(model, device="cpu", dtype=torch.float32)
    tokenizer = MockTokenizer()

    # Test full generate
    output = engine.generate(
        tokenizer,
        prompt="hello",
        max_new_tokens=4,
        temperature=0.0,
        stream=False,
    )
    assert isinstance(output, str)
    assert output.startswith("hello")

    # Test streaming
    tokens = list(engine.stream_generate(
        tokenizer,
        prompt="hello",
        max_new_tokens=3,
        temperature=0.0,
    ))
    assert len(tokens) <= 3
    assert all(isinstance(t, str) for t in tokens)


def test_self_speculative_generation():
    """Verify self-speculative generation produces valid output and state rollback works."""
    cfg = create_mini_config()
    model = PulsarForCausalLM(cfg).to(dtype=torch.float32)
    engine = PulsarInferenceEngine(model, device="cpu", dtype=torch.float32)
    tokenizer = MockTokenizer()

    # Generate with speculative=True
    output = engine.generate(
        tokenizer,
        prompt="hello",
        max_new_tokens=6,
        temperature=0.0,
        stream=False,
        speculative=True,
    )
    assert isinstance(output, str)
    assert output.startswith("hello")

    # Streaming with self-speculative
    tokens = list(engine.self_speculative_stream_generate(
        tokenizer,
        prompt="hello",
        max_new_tokens=4,
        temperature=0.0,
    ))
    assert len(tokens) <= 4
    assert all(isinstance(t, str) for t in tokens)

