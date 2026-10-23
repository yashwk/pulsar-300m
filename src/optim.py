"""Optimizers with stochastic rounding for bf16 weights.

Why stochastic rounding
-----------------------
Matrices and ``tok_emb`` are stored in bf16. A Muon update is about ``lr * 0.025 *
scale`` ~ 5e-4 per element at peak LR, while the bf16 spacing at |w| = 0.02 is 1.2e-4
(half-spacing 6e-5). Updates round-to-nearest, so once the LR factor drops they vanish;
decoupled weight decay (``1 - lr*wd`` = 0.998) mostly rounds away at any LR. The cosine
cooldown - where most of WSD's benefit comes from - progressively stalls.

Stochastic rounding writes ``w + delta`` in FP32, then rounds each element up or down
with probability proportional to the fractional part. The expected value of the stored
weight equals the exact FP32 result, so many small updates accumulate correctly instead
of being discarded.

Implementation notes
--------------------
* The FP32 working copy is per-parameter. The largest tensor is ``tok_emb``
  (49,152 x 1024 = 16.8M elements, 67 MB), so this never materialises the whole model
  in FP32.
* A dedicated ``torch.Generator`` seeded from ``(seed, step)`` makes runs reproducible
  and resumable, and keeps SR noise out of the global RNG stream.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer as _Optimizer
from torch.optim import Optimizer


@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    assert G.ndim in (2, 3), f"Expected 2D matrix or 3D batched matrix, got {G.shape}"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    norm = X.norm(dim=(-2, -1), keepdim=True) + eps
    X = X / norm
    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.transpose(-2, -1)
        transposed = True
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.transpose(-2, -1)
    return X


def stochastic_round_(dst: torch.Tensor, src_fp32: torch.Tensor, generator: torch.Generator | None = None) -> None:
    """Write ``src_fp32`` into bf16 ``dst`` using stochastic rounding.

    TRUNCATION IS SIMPLE, so add uniform noise first and then truncate. Concretely:
    form the FP32 value, add a random integer in [0, 2**16) to the low 16 bits of its
    bit pattern, then mask those bits off. Each element therefore rounds up with
    probability equal to its fractional position between neighbouring bf16 values, so
    the expected stored value equals the exact FP32 result.
    """
    src = src_fp32.contiguous()
    if dst.dtype == torch.bfloat16:
        noise = torch.randint(
            0, 1 << 16, src.shape, dtype=torch.int32, device=src.device, generator=generator
        )
        bits = src.view(torch.int32)
        rounded = ((bits + noise) & ~0xFFFF).view(torch.float32)
        dst.copy_(rounded)
    else:
        dst.copy_(src)


class _SRMixin:
    """Shared generator/step bookkeeping for stochastic rounding."""

    stochastic_rounding: bool
    seed: int
    _step_count: int
    _generator: torch.Generator | None

    def _gen(self, device: torch.device, offset: int = 0) -> torch.Generator | None:
        if not self.stochastic_rounding:
            return None
        if self._generator is None or self._generator.device != device:
            self._generator = torch.Generator(device=device)
        # Reproducible per (seed, step) without touching the global RNG stream.
        self._generator.manual_seed(
            (self.seed * 1_000_003 + self._step_count + offset) % (2**63 - 1)
        )
        return self._generator

    def _sr_state(self) -> dict:
        return {
            "seed": self.seed,
            "step_count": self._step_count,
            "stochastic_rounding": self.stochastic_rounding,
        }

    def _load_sr_state(self, state: dict | None):
        # Backward compatible: older checkpoints have no SR block, so the step counter
        # simply restarts. Callers that care about exact resume use the new format.
        if not state:
            return
        self.seed = int(state.get("seed", self.seed))
        self._step_count = int(state.get("step_count", 0))
        if "stochastic_rounding" in state:
            self.stochastic_rounding = bool(state["stochastic_rounding"])


class _SROptimizerMixin(_SRMixin):
    """Adds stochastic-rounding bookkeeping to the optimizer's own state_dict.

    The generator is seeded from (seed, _step_count), so a checkpoint that omits the
    counter replays the step-0 rounding noise on resume - the run diverges even though
    every tensor round-trips. Keeping it inside state_dict means any caller that saves
    optimizer state gets it for free, rather than depending on the trainer remembering.
    """

    def state_dict(self):
        sd = super().state_dict()
        sd["_sr"] = self._sr_state()
        # Preserve FP32 master state exactly; see load_state_dict. Only FP32 tensors
        # over a non-FP32 parameter need this: Muon's momentum buffer is bf16 and the
        # parameter is bf16, so torch's cast is lossless and cloning it here would
        # duplicate every expert momentum buffer into the checkpoint.
        live = [p for g in self.param_groups for p in g["params"]]
        fp32_block = {}
        for idx, param in enumerate(live):
            pstate = self.state.get(param)
            if not pstate or param.dtype == torch.float32:
                continue
            keep = {
                k: v.detach().clone()
                for k, v in pstate.items()
                if torch.is_tensor(v) and v.dtype == torch.float32
            }
            if keep:
                fp32_block[idx] = keep
        if fp32_block:
            sd["_fp32_state"] = fp32_block
        return sd

    def load_state_dict(self, state_dict):
        sd = dict(state_dict)
        sr = sd.pop("_sr", None)
        # Optimizer.load_state_dict casts every state tensor to the PARAMETER's dtype,
        # which silently downcasts AdamWBF16SR's FP32 master moments (exp_avg /
        # exp_avg_sq) to bf16 on the way in - precision is already gone by the time we
        # could cast back. So the FP32 moments travel under a private key that torch
        # does not touch, and are restored verbatim here.
        fp32_state = sd.pop("_fp32_state", None)
        super().load_state_dict(sd)
        self._load_sr_state(sr)
        if fp32_state:
            for idx, param in enumerate(p for g in self.param_groups for p in g["params"]):
                saved = fp32_state.get(idx)
                pstate = self.state.get(param)
                if not saved or pstate is None:
                    continue
                for key, val in saved.items():
                    # A checkpoint is often mapped to CPU before load; place the state
                    # back on the parameter's device and in FP32.
                    pstate[key] = val.to(device=param.device, dtype=torch.float32).clone()


class Muon(_SROptimizerMixin, Optimizer):
    """Muon with FP32 update accumulation and stochastic rounding on write-back."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        *,
        stochastic_rounding: bool = True,
        seed: int = 42,
        momentum_fp32: bool = False,
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.stochastic_rounding = stochastic_rounding
        self.seed = seed
        self.momentum_fp32 = momentum_fp32
        self._step_count = 0
        self._generator = None

    @torch.no_grad()
    def step(self):
        self._step_count += 1
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]

            # Group parameters of identical shape to batch Newton-Schulz orthogonalization
            shape_groups = {}
            for p in group["params"]:
                if p.grad is None:
                    continue
                shape_groups.setdefault(p.shape, []).append(p)

            for shape, plist in shape_groups.items():
                g_updates = []
                for p in plist:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        buf_dtype = torch.float32 if self.momentum_fp32 else g.dtype
                        state["momentum_buffer"] = torch.zeros_like(g, dtype=buf_dtype)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g.to(buf.dtype))
                    # Keep the Nesterov combination in bf16 all the way through
                    # Newton-Schulz. Promoting to FP32 here allocated an extra copy of
                    # one [8, 1024, 2048] expert tensor (~67 MB, and the following cat
                    # doubles it) only to be cast straight back to bf16.
                    # Measured on that tensor: bf16 path 478 MB peak, FP32 path 545 MB.
                    g_updates.append(g.to(torch.bfloat16).add(buf.to(torch.bfloat16), alpha=momentum))

                # Batch along dim 0: 2D matrices stack to [N, H, W]; 3D expert tensors cat to [N*E, H, W]
                is_3d = (len(shape) == 3)
                if is_3d:
                    G_batched = torch.cat(g_updates, dim=0)
                else:
                    G_batched = torch.stack(g_updates, dim=0)

                G_orth_batched = zeropower_via_newtonschulz5(G_batched)

                # Muon's convention is scale = max(1, out_features/in_features) ** 0.5,
                # which equalises the update size for matrices with unequal aspect ratios.
                #
                # nn.Linear stores [out, in], but the MoE expert tensors are stored
                # [E, in, out] so the fusable 3-D batched GEMM works directly. Reading
                # shape[-2]/shape[-1] blindly therefore inverts the ratio for the experts:
                # gate_up [E, 1024, 2048] got 1.0 instead of sqrt(2), making routed
                # gate/up train ~0.7x as fast as the shared expert's equivalent.
                if is_3d:
                    in_features, out_features = shape[-2], shape[-1]
                else:
                    out_features, in_features = shape[-2], shape[-1]
                scale = max(1.0, out_features / in_features) ** 0.5

                if is_3d:
                    orth_chunks = torch.chunk(G_orth_batched, len(plist), dim=0)
                else:
                    orth_chunks = torch.unbind(G_orth_batched, dim=0)

                gen = self._gen(plist[0].device)
                for p, g_orth in zip(plist, orth_chunks):
                    # Apply decay and update in FP32 against the exact stored value, write
                    # back with stochastic rounding. Uniform rounding of the decay keeps
                    # `p *= 1 - lr*wd` unbiased instead of vanishing entirely.
                    w32 = p.float()
                    if wd > 0:
                        w32 = w32 * (1.0 - lr * wd)
                    w32 = w32.add(g_orth.float(), alpha=-lr * scale)
                    if self.stochastic_rounding and p.dtype == torch.bfloat16:
                        stochastic_round_(p.data, w32, gen)
                    else:
                        p.copy_(w32)


class AdamWBF16SR(_SROptimizerMixin, _Optimizer):
    """AdamW for bf16 parameters with FP32 master state and stochastic rounding."""

    def __init__(
        self,
        params,
        lr: float = 6e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        stochastic_rounding: bool = True,
        seed: int = 42,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.stochastic_rounding = stochastic_rounding
        self.seed = seed
        self._step_count = 0
        self._generator = None

    @torch.no_grad()
    def step(self):
        self._step_count += 1
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(g, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(g, dtype=torch.float32)
                state["step"] += 1
                step = state["step"]
                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                g_f = g.float()
                m.mul_(beta1).add_(g_f, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g_f, g_f, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denom = (v / bias_correction2).sqrt_().add_(eps)

                w32 = p.float()
                if wd > 0:
                    w32 = w32.mul_(1.0 - lr * wd)
                w32 = w32.addcdiv_(m / bias_correction1, denom, value=-lr)

                if self.stochastic_rounding and p.dtype == torch.bfloat16:
                    stochastic_round_(p.data, w32, self._gen(p.device, offset=7))
                else:
                    p.copy_(w32)


class PulsarDualOptimizer:
    def __init__(
        self,
        model: torch.nn.Module,
        muon_lr: float = 0.02,
        adam_lr: float = 6e-4,
        weight_decay: float = 0.1,
        *,
        stochastic_rounding: bool = True,
        seed: int = 42,
        momentum_fp32: bool = False,
    ):
        muon_params = []
        adam_bf16_params = []
        adam_fp32_params = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # 2D weight matrices in Attention and 3D batched MoE matrices go to Muon
            is_matrix = (p.ndim in (2, 3))
            is_excluded = (
                "tok_emb" in name
                or "lm_head" in name
                or "pulse_step_emb" in name
                or "input_mix_logit" in name
                or "norm" in name
                or "router" in name
                or "v_resid_lambda" in name
                or "attn_gate" in name
                or "pulse_res_scale" in name
                or "routed_scale" in name
            )
            if is_matrix and not is_excluded:
                muon_params.append(p)
            elif p.dtype == torch.bfloat16:
                adam_bf16_params.append(p)
            else:
                adam_fp32_params.append(p)

        self.stochastic_rounding = stochastic_rounding
        self.opt_muon = Muon(
            muon_params, lr=muon_lr, weight_decay=weight_decay,
            stochastic_rounding=stochastic_rounding, seed=seed, momentum_fp32=momentum_fp32,
        )
        # bf16 params (only tok_emb after the precision fix) need SR; FP32 params use
        # PyTorch's fused AdamW, which is already exact at FP32.
        self.opt_adam_sr = AdamWBF16SR(
            adam_bf16_params, lr=adam_lr, betas=(0.9, 0.95), weight_decay=0.0,
            stochastic_rounding=stochastic_rounding, seed=seed,
        )
        use_fused = torch.cuda.is_available() and len(adam_fp32_params) > 0 and adam_fp32_params[0].is_cuda
        self.opt_adam = torch.optim.AdamW(
            adam_fp32_params, lr=adam_lr, betas=(0.9, 0.95), weight_decay=0.0, fused=use_fused
        )

    def zero_grad(self, set_to_none: bool = True):
        self.opt_muon.zero_grad(set_to_none=set_to_none)
        self.opt_adam_sr.zero_grad(set_to_none=set_to_none)
        self.opt_adam.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.opt_muon.step()
        self.opt_adam_sr.step()
        self.opt_adam.step()

    def state_dict(self) -> dict:
        # The stochastic-rounding sequence is seeded from (seed, step_count), so both
        # must travel with the optimizer state. Omitting them made a resumed run replay
        # the SAME rounding noise from step 0 instead of continuing the sequence.
        return {
            "opt_muon": self.opt_muon.state_dict(),
            "opt_adam_sr": self.opt_adam_sr.state_dict(),
            "opt_adam": self.opt_adam.state_dict(),
            "sr_muon": self.opt_muon._sr_state(),
            "sr_adam": self.opt_adam_sr._sr_state(),
        }

    def load_state_dict(self, state_dict: dict):
        if state_dict.get("opt_muon") is not None:
            self.opt_muon.load_state_dict(state_dict["opt_muon"])
        if state_dict.get("opt_adam_sr") is not None:
            self.opt_adam_sr.load_state_dict(state_dict["opt_adam_sr"])
        if state_dict.get("opt_adam") is not None:
            self.opt_adam.load_state_dict(state_dict["opt_adam"])
        # Absent in checkpoints written before this fix; ignored gracefully.
        self.opt_muon._load_sr_state(state_dict.get("sr_muon"))
        self.opt_adam_sr._load_sr_state(state_dict.get("sr_adam"))
