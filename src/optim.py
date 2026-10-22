import torch
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


class Muon(Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, weight_decay: float = 0.1):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
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
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    g_updates.append(g.add(buf, alpha=momentum))

                # Batch along dim 0: 2D matrices stack to [N, H, W]; 3D expert tensors cat to [N*E, H, W]
                is_3d = (len(shape) == 3)
                if is_3d:
                    G_batched = torch.cat(g_updates, dim=0)
                else:
                    G_batched = torch.stack(g_updates, dim=0)

                # Single batched invocation of Newton-Schulz5 orthogonalization per unique shape group
                G_orth_batched = zeropower_via_newtonschulz5(G_batched)

                rows = shape[-2]
                cols = shape[-1]
                scale = max(1.0, rows / cols) ** 0.5

                if is_3d:
                    orth_chunks = torch.chunk(G_orth_batched, len(plist), dim=0)
                else:
                    orth_chunks = torch.unbind(G_orth_batched, dim=0)

                for p, g_orth in zip(plist, orth_chunks):
                    if wd > 0:
                        p.mul_(1.0 - lr * wd)
                    p.add_(g_orth.to(p.dtype), alpha=-lr * scale)


class PulsarDualOptimizer:
    def __init__(
        self,
        model: torch.nn.Module,
        muon_lr: float = 0.02,
        adam_lr: float = 6e-4,
        weight_decay: float = 0.1,
    ):
        muon_params = []
        adam_params = []

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
            else:
                adam_params.append(p)

        self.opt_muon = Muon(muon_params, lr=muon_lr, weight_decay=weight_decay)
        use_fused = torch.cuda.is_available() and len(adam_params) > 0 and adam_params[0].is_cuda
        self.opt_adam = torch.optim.AdamW(adam_params, lr=adam_lr, betas=(0.9, 0.95), weight_decay=0.0, fused=use_fused)

    def zero_grad(self, set_to_none: bool = True):
        self.opt_muon.zero_grad(set_to_none=set_to_none)
        self.opt_adam.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.opt_muon.step()
        self.opt_adam.step()

    def state_dict(self) -> dict:
        return {
            "opt_muon": self.opt_muon.state_dict(),
            "opt_adam": self.opt_adam.state_dict(),
        }

    def load_state_dict(self, state_dict: dict):
        if "opt_muon" in state_dict and state_dict["opt_muon"] is not None:
            self.opt_muon.load_state_dict(state_dict["opt_muon"])
        if "opt_adam" in state_dict and state_dict["opt_adam"] is not None:
            self.opt_adam.load_state_dict(state_dict["opt_adam"])
