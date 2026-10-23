"""WI-06: the validation metric must be the final model's loss on all tokens."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from conftest import make_cfg


def test_val_ce_final_matches_manual_cross_entropy():
    """val_ce_final must equal a manual CE of the final-pulse logits over all tokens."""
    from src.model import PulsarForCausalLM

    cfg = make_cfg()
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg).eval()

    x = torch.randint(0, cfg.vocab_size, (1, 24))
    y = torch.randint(0, cfg.vocab_size, (1, 24))

    with torch.no_grad():
        _, _, metrics = model(x, labels=y, num_pulses=3, eval_full_pulses=True)
        got = float(metrics["pulse_losses"][-1])

        # Manual: recompute the final pulse path independently
        h = model.tok_emb(x)
        h0n = model.anchor_norm(h)
        h_pulse = h
        v_first = None
        for k in range(3):
            step = model.pulse_step_emb.weight[k].to(h0n.dtype).view(1, 1, -1)
            if k == 0:
                h_in = h0n + step
            else:
                mix = torch.sigmoid(model.input_mix_logit[k]).to(h_pulse.dtype)
                h_in = (1 - mix) * model.pulse_state_norm(h_pulse) + mix * h0n + step
            hc = h_in
            for blk in model.core:
                cos, sin = (model.rope_local if blk.attn.is_local else model.rope_global)(h_in, h_in.size(1))
                hc, v_first, _, _, _ = blk(hc, cos, sin, pulse_idx=k, v_first=v_first)
            h_pulse = h_in + (hc - h_in) * model.pulse_res_scale[k].to(h_in.dtype)

        out = model.final_norm(h_pulse)
        raw = F.linear(out.view(-1, cfg.d_model), model.lm_head.weight)
        sc = cfg.logit_softcap * torch.tanh(raw / cfg.logit_softcap)
        expected = F.cross_entropy(sc, y.view(-1))

    assert abs(got - expected.item()) < 1e-3, f"val_ce_final {got:.6f} vs manual {expected.item():.6f}"


def test_eval_full_pulses_scores_all_tokens():
    """Intermediate pulses must be scored on 100% of tokens in eval mode.

    Training uses a 25% subsample; eval must not, or per-pulse CEs are computed on
    different tokens and cannot be compared.
    """
    from src.model import PulsarForCausalLM

    cfg = make_cfg()
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 32))
    y = torch.randint(0, cfg.vocab_size, (1, 32))

    with torch.no_grad():
        _, _, m_full = model(x, labels=y, num_pulses=3, eval_full_pulses=True)
        _, _, m_sub = model(x, labels=y, num_pulses=3, eval_full_pulses=False)

    # With full tokens the intermediate-pulse CE differs from the 25% subsample value.
    full_p1 = float(m_full["pulse_losses"][0])
    sub_p1 = float(m_sub["pulse_losses"][0])
    assert full_p1 != pytest.approx(sub_p1, abs=1e-6), "eval_full_pulses had no effect"
    # The final pulse is always full-token, so it must agree.
    assert float(m_full["pulse_losses"][-1]) == pytest.approx(float(m_sub["pulse_losses"][-1]), rel=1e-4)


def test_evaluate_loss_returns_final_pulse_contract():
    """evaluate_loss must expose val_ce_final, per-pulse CEs and latent_gain."""
    import numpy as np

    from src.model import PulsarForCausalLM
    from src.train import StreamingPinnedBatchLoader, evaluate_loss

    cfg = make_cfg()
    torch.manual_seed(0)
    model = PulsarForCausalLM(cfg).eval()

    tokens = np.random.randint(1, cfg.vocab_size, 20000).astype(np.uint16)
    loader = StreamingPinnedBatchLoader([tokens], micro_batch_size=1, seq_len=32, device="cpu")

    out = evaluate_loss(model, loader, eval_iters=2, num_pulses=3)
    for key in ("val_ce_final", "val_ppl", "per_pulse", "latent_gain", "mean_pulse_ce"):
        assert key in out, f"missing {key}"
    assert len(out["per_pulse"]) == 3
    assert out["val_ce_final"] == pytest.approx(out["per_pulse"][-1], rel=1e-6)
    assert all(math.isfinite(v) for v in out["per_pulse"])
