"""Architecture and model contract tests for Pulsar-300M.

Covers:
- Meta-device parameter counts and decoupled Shared Expert summary.
- LeCun fan-in variance matching and Step 0 loss baseline.
- Mixed-precision retention of FP32 scales, counters, and normalization buffers.
- Correct normalization of _chunked_ce with ignored tokens and all-ignored batches.
- Pulse weight normalization and loss weighting across recurrent depths K.
- MoE counter freezing under eval() and checkpoint recomputation.
- Vectorized MoE composite argsort, drop priority, and padding exclusion.
- Static-compilation unwrapping during canary generation.
- Router z-loss gradients and loss scaling.
- RoPE inv_freq exactness after bf16 precision setup.
- Generation context window sizing and batched EOS termination.
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn.functional as F

from conftest import make_cfg, assert_close, needs_gpu, HAS_CUDA
from src.model import PulsarConfig, PulsarForCausalLM, PulsarMoEMLP, compute_token_slots
import src.model as M
from src.train import generate_canary
from data.prepare_shards import format_toucan_messages


def test_model_meta_summary_and_parameter_counts():
    """Production 300M parameter count and summary on meta device without allocating RAM."""
    with torch.device("meta"):
        cfg = PulsarConfig(
            vocab_size=32768,
            d_model=1024,
            d_expert=512,
            d_shared=1024,
            n_routed_experts=16,
            top_k=4,
            n_core=8,
            default_pulses=3,
        )
        model = PulsarForCausalLM(cfg)
        summary = model.get_model_summary(num_pulses=3)

    assert summary["vocab_size"] == 32768
    assert summary["d_shared"] == 1024
    assert summary["d_expert"] == 512
    assert summary["active_ffn_dim"] == 3072  # 1024 shared + 4 * 512 routed
    assert 280_000_000 <= summary["total_params"] <= 285_000_000
    assert 95_000_000 <= summary["non_embed_active_params"] <= 105_000_000


def test_init_variance_matching_and_step0_loss():
    """LeCun fan-in matches routed/shared output variance within 2x and Step 0 loss is near ln(V)."""
    cfg = make_cfg(d_model=128, d_expert=64, d_shared=128, n_routed_experts=8, top_k=2)
    torch.manual_seed(42)
    model = PulsarForCausalLM(cfg).float()
    model.eval()

    # Step 0 loss on random tokens near ln(vocab_size)
    B, T = 2, 32
    x = torch.randint(0, cfg.vocab_size, (B, T))
    y = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        _, loss, _ = model(x, labels=y)
    ln_v = math.log(cfg.vocab_size)
    assert loss.item() < ln_v + 1.2, f"Loss {loss.item():.2f} exceeded baseline ln(V)={ln_v:.2f}"

    # Output variance matching between shared expert and routed MoE branch
    mlp = model.core[0].mlp
    N = 128
    x_test = torch.randn(1, N, cfg.d_model) * 0.5
    xf = x_test.view(N, cfg.d_model)

    with torch.no_grad():
        shared = mlp.shared_expert(xf)
        out, _, _, _ = mlp(x_test, pulse_idx=0)
        routed = out.view(N, cfg.d_model) - shared

    def rms(t):
        return t.float().pow(2).mean().sqrt().item()

    rs, rr = rms(shared), rms(routed)
    ratio = rr / max(rs, 1e-12)
    assert 0.5 <= ratio <= 2.0, f"routed/shared RMS ratio {ratio:.2f} outside [0.5, 2.0]"


def test_to_training_precision_retains_fp32():
    """BF16 casting retains sensitive scales, router logits, and normalization in FP32."""
    cfg = make_cfg()
    model = PulsarForCausalLM(cfg)
    model.to_training_precision(torch.bfloat16)

    assert model.tok_emb.weight.dtype == torch.bfloat16
    assert model.core[0].attn.wqkv.weight.dtype == torch.bfloat16

    assert model.pulse_res_scale.dtype == torch.float32
    assert model.core[0].mlp.routed_scale.dtype == torch.float32
    assert model.core[0].mlp.router_weight.dtype == torch.float32
    assert model.core[0].attn_norm.weight.dtype == torch.float32


@pytest.mark.parametrize("frac_ignored", [0.0, 0.25, 0.5, 0.75])
def test_chunked_ce_divides_by_valid_tokens_only(frac_ignored):
    """_chunked_ce normalizes strictly by valid tokens, not total sequence length."""
    cfg = make_cfg(vocab_size=64, d_model=32, ce_chunk_size=16)
    model = PulsarForCausalLM(cfg).float()

    N = 64
    h = torch.randn(1, N, cfg.d_model)
    targets = torch.randint(0, cfg.vocab_size, (1, N))
    if frac_ignored > 0:
        n_mask = int(N * frac_ignored)
        targets[0, :n_mask] = -100

    loss = model._chunked_ce(h, targets)
    logits = model.lm_head(h)
    expected = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1), ignore_index=-100)
    assert_close(loss, expected, atol=1e-4, rtol=1e-4)


def test_chunked_ce_ignores_all_ignored_batch():
    """An all -100 chunk returns finite loss without division by zero."""
    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512)
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg).eval()
    h = torch.randn(1, 8, cfg.d_model)
    labels = torch.full((1, 8), -100, dtype=torch.long)
    out = model._chunked_ce(model.final_norm(h), labels)
    assert torch.isfinite(out)


@pytest.mark.parametrize("K", [1, 2, 3, 4])
def test_pulse_weights_sum_to_one(K):
    """Deep supervision pulse loss weights sum strictly to 1.0 for any pulse count."""
    cfg = make_cfg(max_pulses=4)
    model = PulsarForCausalLM(cfg)
    w = model._compute_pulse_weights(K)
    assert len(w) == K
    assert abs(sum(w) - 1.0) < 1e-6


def test_reported_loss_matches_weighted_pulse_sum():
    """Returned total loss matches the weighted sum of pulse losses."""
    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512, top_k=2, router_z_loss_weight=0.0)
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))

    with torch.no_grad():
        _, loss, metrics = model(x, labels=y, num_pulses=3)
    pl = [float(t) for t in metrics["pulse_losses"]]
    weights = [cfg.aux_loss_weight_1, cfg.aux_loss_weight_2, cfg.final_loss_weight]
    expected = sum(w * l for w, l in zip(weights, pl))
    assert float(loss) == pytest.approx(expected, rel=2e-2)


def test_invalid_pulse_count_rejected():
    """Out-of-range pulse counts raise ValueError."""
    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512)
    model = PulsarForCausalLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    for bad in (0, -1, cfg.max_pulses + 1):
        with pytest.raises(ValueError):
            model(x, labels=None, num_pulses=bad)


def test_eval_forward_counters_frozen():
    """model.eval() never mutates expert usage counters or drop statistics."""
    cfg = make_cfg(max_pulses=2, default_pulses=2)
    model = PulsarForCausalLM(cfg)
    model.eval()

    mlp = model.core[0].mlp
    initial_accum = mlp.expert_counts_accum.clone()
    initial_pulse = mlp.expert_pulse_counts.clone()

    x = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        model(x)

    assert torch.equal(mlp.expert_counts_accum, initial_accum)
    assert torch.equal(mlp.expert_pulse_counts, initial_pulse)


def test_router_confidence_metric():
    """router_confidence tracks the mean router top-k weight without pinned artifact."""
    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512, top_k=2)
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        _, _, metrics = model(x, labels=None, num_pulses=3)
    conf = metrics.get("router_confidence")
    assert conf is not None and abs(conf - 0.5) > 1e-6


def test_router_logits_are_mean_centred():
    """Router logits are mean-centered per token across experts."""
    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512, top_k=2)
    torch.manual_seed(0)
    mlp = PulsarMoEMLP(cfg).eval()

    captured = {}
    orig_linear = F.linear

    def spy(inp, weight, bias=None):
        out = orig_linear(inp, weight, bias)
        if weight is not None and weight.shape == mlp.router_weight[0].shape:
            captured["raw"] = out.detach().clone()
        return out

    F.linear = spy
    try:
        with torch.no_grad():
            mlp(torch.randn(1, 32, cfg.d_model), pulse_idx=0)
    finally:
        F.linear = orig_linear

    assert "raw" in captured
    centred = captured["raw"] - captured["raw"].mean(dim=-1, keepdim=True)
    assert torch.allclose(centred.mean(dim=-1), torch.zeros(centred.shape[0]), atol=1e-5)


def test_per_block_imbalance_is_not_diluted():
    """Worst expert share metric reports true worst single-expert share without cross-block dilution."""
    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512, n_routed_experts=8, top_k=2)
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg).train()

    with torch.no_grad():
        model.core[0].mlp.expert_bias.zero_()
        model.core[0].mlp.expert_bias[:, 0] = 5.0
        model.core[1].mlp.expert_bias.zero_()
        model.core[1].mlp.expert_bias[:, 4] = 5.0

    with torch.no_grad():
        _, _, metrics = model(torch.randint(0, cfg.vocab_size, (1, 64)), labels=None, num_pulses=2)

    agg_max = float(metrics["expert_pcts"].max())
    worst = float(metrics["worst_expert_share_pct"])
    assert worst > agg_max + 10
    assert worst > 40


def test_checkpointed_forward_accumulates_counters_during_forward():
    """Activation checkpointing accumulates MoE counters during forward without double-counting on recompute."""
    device = "cuda" if HAS_CUDA else "cpu"
    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512, top_k=2, local_attn_impl="sdpa")
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg).to(device)
    model.to_training_precision(torch.bfloat16)
    model.train()
    model.grad_checkpoint = True

    x = torch.randint(0, cfg.vocab_size, (2, 32), device=device)
    y = torch.randint(0, cfg.vocab_size, (2, 32), device=device)

    def counts():
        return sum(float(b.mlp.expert_pulse_counts.sum()) for b in model.core)

    _, loss, _ = model(x, labels=y, num_pulses=2)
    after_forward = counts()
    loss.backward()
    after_backward = counts()

    assert after_forward > 0
    assert after_backward == after_forward


def test_inv_freq_is_exact_after_training_precision():
    """RoPE inv_freq maintains exact FP32 precision after to_training_precision."""
    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512, top_k=2)
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg)
    model.to_training_precision(torch.bfloat16)

    rope = [m for _, m in model.named_modules() if hasattr(m, "inv_freq")][0]
    expected = 1.0 / (rope.theta ** (torch.arange(0, rope.dim, 2).float() / rope.dim))
    assert rope.inv_freq.dtype == torch.float32
    max_err = (rope.inv_freq.float() - expected).abs().max().item()
    assert max_err == 0.0


def test_drops_measured_from_slots_not_bincounts():
    """Drop metric reflects actual negative slot assignments rather than bincount heuristics."""
    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512, top_k=2)
    cfg.moe_capacity_factor = 10.0
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg)
    model.to_training_precision(torch.bfloat16)
    model.train()

    x = torch.randint(0, cfg.vocab_size, (2, 64))
    with torch.no_grad():
        model(x, labels=None, num_pulses=2)
    good = model.pop_moe_stats()
    assert good["dropped"] == 0
    assert good["demand_over_capacity"] == 0

    orig = M.compute_token_slots

    def broken(ti, tw, E, C):
        s = orig(ti, tw, E, C)
        return torch.where(s % 3 == 0, s, torch.full_like(s, -1))

    M.compute_token_slots = broken
    try:
        with torch.no_grad():
            model(x, labels=None, num_pulses=2)
    finally:
        M.compute_token_slots = orig
    bad = model.pop_moe_stats()
    assert bad["dropped"] > 0
    assert bad["demand_over_capacity"] == 0


@pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA to detect host syncs")
def test_training_step_has_no_host_sync():
    """Training forward and backward pass execute asynchronously without device synchronizations."""
    cfg = make_cfg(n_core=2, d_model=128, head_dim=32, vocab_size=512, n_routed_experts=8, top_k=2)
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg).cuda()
    model.to_training_precision(torch.bfloat16)
    model.train()

    x = torch.randint(0, cfg.vocab_size, (2, 32), device="cuda")
    y = torch.randint(0, cfg.vocab_size, (2, 32), device="cuda")

    # Warmup lazy allocations
    try:
        _, loss, _ = model(x, labels=y, num_pulses=2)
        loss.backward()
    finally:
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:, 24:] = False

    torch.cuda.set_sync_debug_mode("error")
    try:
        _, loss, _ = model(x, labels=y, num_pulses=2, routing_mask=mask)
        loss.backward()
    finally:
        torch.cuda.set_sync_debug_mode("default")


def test_per_pulse_and_global_counter_modes():
    """per_pulse_bias toggles accumulation of global vs per-pulse usage counters."""
    cfg_pulse = make_cfg(n_core=2, d_model=128, vocab_size=512, top_k=2, per_pulse_bias=True)
    m_pulse = PulsarForCausalLM(cfg_pulse).train()
    x = torch.randint(0, cfg_pulse.vocab_size, (1, 16))

    with torch.no_grad():
        m_pulse(x, labels=None, num_pulses=2)
        for block in m_pulse.core:
            block.mlp.update_biases(0.01, num_pulses=2)
    assert sum(float(b.mlp.expert_counts_accum.sum()) for b in m_pulse.core) == 0

    cfg_global = make_cfg(n_core=2, d_model=128, vocab_size=512, top_k=2, per_pulse_bias=False)
    m_global = PulsarForCausalLM(cfg_global).train()
    with torch.no_grad():
        m_global(x, labels=None, num_pulses=2)
    assert sum(float(b.mlp.expert_counts_accum.sum()) for b in m_global.core) > 0


def test_shared_mean_frac_and_router_diag():
    """Router diagnostic isolates shared batch-mean input components and remains inert by default."""
    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512, n_routed_experts=8, top_k=2, local_attn_impl="sdpa")
    mlp = PulsarMoEMLP(cfg)
    N, D = 256, cfg.d_model

    def measure(x):
        logits = x @ mlp.router_weight[0].float().t()
        idx = torch.topk(logits, cfg.top_k, dim=-1).indices
        return float(mlp.router_diagnostics(x, logits, idx, 0)["shared_mean_frac"])

    noise_only = torch.randn(N, D)
    with_shared = torch.randn(N, D) * 0.1 + torch.randn(1, D) * 5.0
    assert measure(noise_only) < 0.2
    assert measure(with_shared) > 0.8

    model = PulsarForCausalLM(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    _, _, plain = model(x, labels=x, num_pulses=2)
    assert not any(k.startswith("router_diag_") for k in plain)

    _, _, with_diag = model(x, labels=x, num_pulses=2, router_diag=True)
    assert any(k.startswith("router_diag_") for k in with_diag)
    share = float(with_diag["router_diag_max_top1_share"])
    assert 1.0 / cfg.n_routed_experts - 1e-6 <= share <= 1.0


def test_compute_token_slots_composite_sort():
    """Single composite sort matches reference and prioritizes highest weights."""
    def reference_two_sort_slots(topk_idx, topk_weights, num_experts, capacity):
        N, K = topk_idx.shape
        device = topk_idx.device
        flat_e = topk_idx.reshape(-1).long()
        flat_w = topk_weights.reshape(-1).float()
        M = flat_e.numel()
        order = torch.argsort(flat_w, descending=True, stable=True)
        order = order[torch.argsort(flat_e[order], stable=True)]
        e_sorted = flat_e[order]
        counts_sorted = torch.zeros(num_experts, device=device, dtype=torch.long)
        counts_sorted.index_add_(0, e_sorted, torch.ones(M, device=device, dtype=torch.long))
        start_sorted = counts_sorted.cumsum(0) - counts_sorted
        rank_sorted = torch.arange(M, device=device) - start_sorted[e_sorted]
        slot_sorted = torch.where(rank_sorted < capacity, rank_sorted, torch.full_like(rank_sorted, -1))
        slots = torch.empty(M, dtype=torch.int32, device=device)
        slots[order] = slot_sorted.to(torch.int32)
        return slots.view(N, K)

    torch.manual_seed(123)
    N, K, E, C = 256, 4, 16, 32
    topk_idx = torch.randint(0, E, (N, K))
    topk_weights = torch.rand(N, K)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    got = compute_token_slots(topk_idx, topk_weights, E, C)
    expected = reference_two_sort_slots(topk_idx, topk_weights, E, C)
    assert torch.equal(got, expected)


def test_moe_padding_excluded_from_capacity():
    """Padded positions in routing_mask must not compete for capacity or count as dropped."""
    cfg = make_cfg(max_pulses=2)
    mlp = PulsarMoEMLP(cfg).float().train()

    B, T, D = 2, 16, cfg.d_model
    x = torch.randn(B, T, D, requires_grad=True)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, 8:] = False
    mask[1, 12:] = False

    out, z_loss, _, _ = mlp(x, pulse_idx=0, routing_mask=mask)
    assert out.shape == (B, T, D)
    loss = out.sum() + z_loss
    loss.backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()


def test_zloss_gradient_and_scaling():
    """MoE router z-loss computes exact analytic gradients and scales with gradient accumulation."""
    cfg = make_cfg(d_model=64, d_expert=64, n_routed_experts=8, top_k=2, max_pulses=2)
    torch.manual_seed(0)
    mlp = PulsarMoEMLP(cfg).float()

    N = 32
    x = torch.randn(1, N, cfg.d_model)
    out, z_loss, _, _ = mlp(x, pulse_idx=0)
    z_loss.backward()

    logits = F.linear(x.view(N, cfg.d_model), mlp.router_weight[0])
    logits = logits - logits.mean(dim=-1, keepdim=True)
    lse = torch.logsumexp(logits, dim=-1, keepdim=True)
    expected_dlogits = cfg.router_z_loss_weight * 2.0 * lse * torch.softmax(logits, dim=-1) / N
    expected_dlogits = expected_dlogits - expected_dlogits.mean(dim=-1, keepdim=True)
    expected_w = expected_dlogits.t() @ x.view(N, cfg.d_model)

    got = mlp.router_weight.grad[0]
    max_err = (got - expected_w).abs().max().item()
    scale = expected_w.abs().max().item()
    assert max_err <= 1e-4 * max(1.0, scale)


def test_smoltalk_no_tool_convention_injection():
    """SmolTalk substream 5A omits tool calling convention when inject_tool_hint=False."""
    sample_msgs = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]
    formatted = format_toucan_messages(sample_msgs, native_only=True, inject_tool_hint=False)
    assert "[Tool Calling Convention]" not in formatted
    assert "<|im_start|>user\nWhat is the capital of France?<|im_end|>" in formatted


def test_generate_full_context_and_eos_termination():
    """model.generate handles full context window and terminates all batched sequences on EOS."""
    cfg = make_cfg(n_core=2, d_model=128, vocab_size=512, base_seq_len=32, max_seq_len=1024)
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg).eval()

    seen = {}
    orig = model.forward

    def spy(input_ids, *a, **kw):
        seen["T"] = input_ids.shape[1]
        return orig(input_ids, *a, **kw)

    model.forward = spy
    x = torch.randint(1, cfg.vocab_size, (1, 64))
    with torch.no_grad():
        model.generate(x, max_new_tokens=2, temperature=0.0)
    assert seen["T"] > cfg.base_seq_len

    model.forward = orig
    x_batch = torch.randint(1, cfg.vocab_size, (3, 16))
    with torch.no_grad():
        out = model.generate(x_batch, max_new_tokens=8, temperature=0.0, eos_token_id=0)
    assert out.shape[0] == 3
    assert out.shape[1] <= x_batch.shape[1] + 8


def test_generate_canary_compilation_unwrap():
    """generate_canary unwraps compiled core blocks during generation and restores them afterward."""
    cfg = make_cfg(n_core=2, default_pulses=2)
    model = PulsarForCausalLM(cfg).float().eval()

    if hasattr(torch, "compile"):
        for i in range(len(model.core)):
            model.core[i] = torch.compile(model.core[i], dynamic=False)

    class MockTokenizer:
        eos_token_id = 0
        unk_token_id = 1
        def encode(self, prompt, return_tensors="pt"):
            return torch.tensor([[5, 10, 15, 20]], dtype=torch.long)
        def decode(self, ids, skip_special_tokens=False):
            return "canary test prompt"
        def convert_tokens_to_ids(self, s):
            return 0

    tok = MockTokenizer()
    result = generate_canary(model, tok, "canary prompt", max_new_tokens=4, device="cpu")
    assert isinstance(result, str)

    if hasattr(torch, "compile"):
        for blk in model.core:
            assert hasattr(blk, "_orig_mod"), "Compiled core block was not restored"
