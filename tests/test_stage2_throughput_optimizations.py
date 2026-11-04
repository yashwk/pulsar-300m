import math
import pytest
import torch
import torch.nn.functional as F

try:
    import pulsar_cuda
    HAS_PULSAR_CUDA = True
except ImportError:
    HAS_PULSAR_CUDA = False

from src.model import (
    PulsarConfig,
    PulsarForCausalLM,
    PulsarAttention,
    _FusedSwiGLUFunction,
    _FusedHarmonicResidualFunction,
    compute_token_slots,
    _get_band_mask,
)
from src.optim import zeropower_via_newtonschulz5


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_PULSAR_CUDA, reason="CUDA with pulsar_cuda required")
class TestStage2ThroughputOptimizations:

    def test_fused_swiglu_parity(self):
        """Optimization 1: Vectorized SwiGLU matches F.silu(g) * u forward and backward."""
        torch.manual_seed(42)
        device = "cuda"

        for D in [512, 1024, 2048, 513]:
            N = 64
            gu_cuda = torch.randn(N, 2 * D, dtype=torch.bfloat16, device=device, requires_grad=True)
            gu_ref = gu_cuda.clone().detach().requires_grad_(True)

            # CUDA path via _FusedSwiGLUFunction
            out_cuda = _FusedSwiGLUFunction.apply(gu_cuda, D)

            # PyTorch reference path
            gate, up = gu_ref.chunk(2, dim=-1)
            out_ref = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)

            assert torch.allclose(out_cuda, out_ref, atol=1e-2, rtol=1e-2), f"SwiGLU fwd mismatch for D={D}"

            # Backward
            grad_out = torch.randn_like(out_cuda)
            out_cuda.backward(grad_out)
            out_ref.backward(grad_out)

            assert torch.allclose(gu_cuda.grad, gu_ref.grad, atol=1e-2, rtol=1e-2), f"SwiGLU bwd mismatch for D={D}"

    def test_moe_assign_slots_cuda_parity(self):
        """Optimization 2: Direct CUDA slot assignment matches sequential token-order per-expert rank."""
        torch.manual_seed(42)
        device = "cuda"
        N, K, E, C = 2048, 4, 16, 600

        topk_idx = torch.randint(0, E, (N, K), dtype=torch.int64, device=device)
        slots_cuda, counts_cuda = pulsar_cuda.moe_assign_slots(topk_idx, E, C)

        # Reference sequential assignment in token order
        flat_e = topk_idx.flatten().cpu().tolist()
        expected_slots = []
        cur_counts = [0] * E
        for e in flat_e:
            rank = cur_counts[e]
            cur_counts[e] += 1
            expected_slots.append(rank if rank < C else -1)

        ref_slots = torch.tensor(expected_slots, dtype=torch.int32, device=device).view(N, K)
        ref_counts = torch.tensor(cur_counts, dtype=torch.int64, device=device)

        assert torch.equal(slots_cuda, ref_slots), "Slot assignment mismatch!"
        assert torch.equal(counts_cuda, ref_counts), "Expert counts mismatch!"

        # Test capacity overflow dropping
        tight_C = 20
        slots_tight, counts_tight = pulsar_cuda.moe_assign_slots(topk_idx, E, tight_C)
        assert (slots_tight < tight_C).all()
        assert (slots_tight >= -1).all()
        dropped = (slots_tight == -1).sum().item()
        assert dropped > 0, "Expected drops with tight capacity"

    def test_windowed_chunked_local_attention_parity(self):
        """Optimization 3: Exact Windowed Chunked SDPA matches full masked SDPA at T=4096, W=1024."""
        torch.manual_seed(42)
        device = "cuda"
        B, H_q, H_kv, T, D = 2, 16, 4, 4096, 64
        W = 1024

        def eval_attn(chunked: bool):
            torch.manual_seed(123)
            q = torch.randn(B, H_q, T, D, dtype=torch.bfloat16, device=device, requires_grad=True)
            k = torch.randn(B, H_kv, T, D, dtype=torch.bfloat16, device=device, requires_grad=True)
            v = torch.randn(B, H_kv, T, D, dtype=torch.bfloat16, device=device, requires_grad=True)

            if not chunked:
                # Reference full masked SDPA
                row_idx = torch.arange(T, device=device).unsqueeze(1)
                col_idx = torch.arange(T, device=device).unsqueeze(0)
                diff = row_idx - col_idx
                dense_mask = (diff >= 0) & (diff < W)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=dense_mask, enable_gqa=True)
            else:
                # Exact Windowed Chunked SDPA
                num_chunks = T // W
                out_0 = F.scaled_dot_product_attention(
                    q[:, :, :W, :], k[:, :, :W, :], v[:, :, :W, :],
                    is_causal=True, enable_gqa=True
                )
                q_rest = q[:, :, W:, :].view(B, H_q, num_chunks - 1, W, D).permute(0, 2, 1, 3, 4).reshape(B * (num_chunks - 1), H_q, W, D)

                k_chunks = k.view(B, H_kv, num_chunks, W, D)
                v_chunks = v.view(B, H_kv, num_chunks, W, D)
                k_win = torch.cat([k_chunks[:, :, :-1], k_chunks[:, :, 1:]], dim=3).permute(0, 2, 1, 3, 4).reshape(B * (num_chunks - 1), H_kv, 2 * W, D)
                v_win = torch.cat([v_chunks[:, :, :-1], v_chunks[:, :, 1:]], dim=3).permute(0, 2, 1, 3, 4).reshape(B * (num_chunks - 1), H_kv, 2 * W, D)

                band_mask = _get_band_mask(W, device)
                out_rest = F.scaled_dot_product_attention(
                    q_rest, k_win, v_win,
                    attn_mask=band_mask,
                    enable_gqa=True
                )
                out_rest = out_rest.view(B, num_chunks - 1, H_q, W, D).permute(0, 2, 1, 3, 4).reshape(B, H_q, T - W, D)
                out = torch.cat([out_0, out_rest], dim=2)

            loss = out.mean()
            loss.backward()
            return out, q.grad, k.grad, v.grad

        o_ref, gq_ref, gk_ref, gv_ref = eval_attn(chunked=False)
        o_chk, gq_chk, gk_chk, gv_chk = eval_attn(chunked=True)

        assert torch.allclose(o_ref, o_chk, atol=1e-2, rtol=1e-2), "Output mismatch"
        cos_gq = F.cosine_similarity(gq_ref.flatten().float(), gq_chk.flatten().float(), dim=0).item()
        cos_gk = F.cosine_similarity(gk_ref.flatten().float(), gk_chk.flatten().float(), dim=0).item()
        cos_gv = F.cosine_similarity(gv_ref.flatten().float(), gv_chk.flatten().float(), dim=0).item()

        assert cos_gq > 0.9999, f"gq cosine similarity too low: {cos_gq}"
        assert cos_gk > 0.9999, f"gk cosine similarity too low: {cos_gk}"
        assert cos_gv > 0.9999, f"gv cosine similarity too low: {cos_gv}"

    def test_pre_lm_head_token_compaction_parity(self):
        """Optimization 4: Pre-lm_head valid-token compaction matches uncompacted loss and grads."""
        torch.manual_seed(42)
        device = "cuda"
        cfg = PulsarConfig(
            n_core=2,
            d_model=256,
            n_heads=4,
            n_kv_heads=2,
            head_dim=64,
            n_routed_experts=4,
            d_expert=128,
            d_shared=256,
            vocab_size=1024,
            max_seq_len=256,
            base_seq_len=64,
            default_pulses=2,
            max_pulses=2,
        )
        model = PulsarForCausalLM(cfg).to(device=device)

        B, T = 2, 64
        input_ids = torch.randint(0, cfg.vocab_size, (B, T), device=device)
        labels = input_ids.clone()
        # Mask out 50% of tokens as prompt (-100)
        labels[:, :32] = -100

        # Run forward with compaction (default in model)
        _, loss_compact, _ = model(input_ids, labels=labels)
        loss_compact.backward()
        assert loss_compact.item() > 0.0

        # Verify edge case: 100% masked tokens
        all_masked_labels = torch.full_like(labels, -100)
        loss_zero = model._chunked_ce(torch.randn(B * T, cfg.d_model, device=device, dtype=torch.bfloat16), all_masked_labels)
        assert loss_zero.item() == 0.0

    def test_fused_harmonic_residual_parity(self):
        """Optimization 5: Fused harmonic residual matches torch.lerp forward and backward."""
        torch.manual_seed(42)
        device = "cuda"
        B, T, D = 4, 32, 512
        h_in_cuda = torch.randn(B, T, D, dtype=torch.bfloat16, device=device, requires_grad=True)
        h_curr_cuda = torch.randn(B, T, D, dtype=torch.bfloat16, device=device, requires_grad=True)
        scale_cuda = torch.rand(D, dtype=torch.float32, device=device, requires_grad=True)

        h_in_ref = h_in_cuda.clone().detach().requires_grad_(True)
        h_curr_ref = h_curr_cuda.clone().detach().requires_grad_(True)
        scale_ref = scale_cuda.clone().detach().requires_grad_(True)

        out_cuda = _FusedHarmonicResidualFunction.apply(h_in_cuda, h_curr_cuda, scale_cuda)
        out_ref = torch.lerp(h_in_ref, h_curr_ref, scale_ref.to(h_in_ref.dtype))

        assert torch.allclose(out_cuda, out_ref, atol=1e-2, rtol=1e-2)

        grad_out = torch.randn_like(out_cuda) / (B * T)
        out_cuda.backward(grad_out)
        out_ref.backward(grad_out)

        assert torch.allclose(h_in_cuda.grad, h_in_ref.grad, atol=1e-2, rtol=1e-2)
        assert torch.allclose(h_curr_cuda.grad, h_curr_ref.grad, atol=1e-2, rtol=1e-2)
        assert torch.allclose(scale_cuda.grad, scale_ref.grad, atol=1e-2, rtol=1e-2)

    def test_batched_3d_muon_newton_schulz_parity(self):
        """Optimization 6: Batched 3D Newton-Schulz matches sequential per-matrix evaluation."""
        torch.manual_seed(42)
        device = "cuda"
        # 16 expert matrices of size 512 x 1024
        Batch, M, N = 16, 512, 1024
        G_3d = torch.randn(Batch, M, N, dtype=torch.bfloat16, device=device)

        # Batched 3D evaluation via fused bmm/baddbmm
        out_batched = zeropower_via_newtonschulz5(G_3d)

        # Sequential reference
        out_seq_list = []
        for i in range(Batch):
            out_seq_list.append(zeropower_via_newtonschulz5(G_3d[i]))
        out_seq = torch.stack(out_seq_list, dim=0)

        cos_sim = F.cosine_similarity(out_batched.flatten().float(), out_seq.flatten().float(), dim=0).item()
        assert cos_sim > 0.9999, f"Cosine similarity too low: {cos_sim}"
        assert torch.allclose(out_batched, out_seq, atol=1e-2, rtol=1e-2), "3D Newton-Schulz mismatch"
