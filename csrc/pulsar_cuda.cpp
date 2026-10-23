#include <torch/extension.h>
#include <vector>
#include <optional>

std::vector<torch::Tensor> rmsnorm_forward_cuda(torch::Tensor x, torch::Tensor weight, float eps);
std::vector<torch::Tensor> rmsnorm_backward_cuda(torch::Tensor dy, torch::Tensor x, torch::Tensor weight, torch::Tensor rsqrt);
std::vector<torch::Tensor> fused_add_rmsnorm_cuda(torch::Tensor x, torch::Tensor residual, torch::Tensor weight, float eps);

torch::Tensor swiglu_forward_cuda(torch::Tensor gate, torch::Tensor up);
std::vector<torch::Tensor> swiglu_backward_cuda(torch::Tensor dout, torch::Tensor gate, torch::Tensor up);
torch::Tensor fused_swiglu_forward_cuda(torch::Tensor gu, int64_t d_ff);
torch::Tensor fused_swiglu_backward_cuda(torch::Tensor dout, torch::Tensor gu, int64_t d_ff);

torch::Tensor rope_forward_cuda(torch::Tensor qk, torch::Tensor cos, torch::Tensor sin);

torch::Tensor engram_hash_gather_cuda(
    torch::Tensor input_ids,
    torch::Tensor canonical_map,
    torch::Tensor table,
    int engram_heads,
    int slots_per_head,
    int head_dim,
    int eos_id
);

torch::Tensor engram_hash_cuda(
    torch::Tensor input_ids,
    torch::Tensor canonical_map,
    int engram_heads,
    int slots_per_head,
    int eos_id
);

void muon_step_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor momentum_buffer,
    float lr,
    float momentum,
    float weight_decay
);

void adamw_step_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    int step
);

std::vector<torch::Tensor> moe_dispatch_forward_cuda(
    torch::Tensor x,
    torch::Tensor topk_idx,
    torch::Tensor token_to_slot,
    int64_t num_experts,
    int64_t capacity
);

torch::Tensor moe_dispatch_backward_cuda(
    torch::Tensor d_expert_inputs,
    torch::Tensor topk_idx,
    torch::Tensor token_to_slot,
    int64_t N,
    int64_t D
);

torch::Tensor moe_combine_forward_cuda(
    torch::Tensor expert_outputs,
    torch::Tensor topk_idx,
    torch::Tensor topk_weights,
    torch::Tensor token_to_slot,
    std::optional<torch::Tensor> shared_out = std::nullopt
);

std::vector<torch::Tensor> moe_combine_backward_cuda(
    torch::Tensor d_routed_out,
    torch::Tensor expert_outputs,
    torch::Tensor topk_idx,
    torch::Tensor topk_weights,
    torch::Tensor token_to_slot
);

std::vector<torch::Tensor> qk_rmsnorm_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    int64_t T,
    float eps
);

std::vector<torch::Tensor> qk_rmsnorm_backward_cuda(
    torch::Tensor dy,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor rsqrt_in,
    int64_t T
);

std::vector<torch::Tensor> fused_softcap_ce_forward_cuda(
    torch::Tensor logits,
    torch::Tensor targets,
    float softcap,
    int64_t ignore_index
);

torch::Tensor fused_softcap_ce_backward_cuda(
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor lse,
    float softcap,
    float grad_scale,
    int64_t ignore_index
);

std::vector<torch::Tensor> moe_router_topk_forward_cuda(
    torch::Tensor logits,
    torch::Tensor expert_bias,
    float z_loss_weight
);

torch::Tensor moe_router_topk_backward_cuda(
    torch::Tensor d_topk_weights,
    torch::Tensor logits,
    torch::Tensor topk_idx,
    float z_loss_weight
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_forward", &rmsnorm_forward_cuda, "RMSNorm Forward (CUDA)");
    m.def("rmsnorm_backward", &rmsnorm_backward_cuda, "RMSNorm Backward (CUDA)");
    m.def("fused_add_rmsnorm", &fused_add_rmsnorm_cuda, "Fused Add + RMSNorm Forward (CUDA)");
    m.def("swiglu_forward", &swiglu_forward_cuda, "SwiGLU Forward (CUDA)");
    m.def("swiglu_backward", &swiglu_backward_cuda, "SwiGLU Backward (CUDA)");
    m.def("fused_swiglu_forward", &fused_swiglu_forward_cuda, "Fused SwiGLU Forward (CUDA)");
    m.def("fused_swiglu_backward", &fused_swiglu_backward_cuda, "Fused SwiGLU Backward (CUDA)");
    m.def("rope_forward", &rope_forward_cuda, "RoPE Forward (CUDA)");
    m.def("engram_hash_gather", &engram_hash_gather_cuda, "Engram Hash and Gather (CUDA)");
    m.def("engram_hash", &engram_hash_cuda, "Engram Hash Calculation (CUDA)");
    m.def("muon_step", &muon_step_cuda, "Muon Optimizer Step (CUDA)");
    m.def("adamw_step", &adamw_step_cuda, "AdamW Optimizer Step (CUDA)");
    m.def("moe_dispatch_forward", &moe_dispatch_forward_cuda, "Fused MoE Dispatch Forward (CUDA)");
    m.def("moe_dispatch_backward", &moe_dispatch_backward_cuda, "Fused MoE Dispatch Backward (CUDA)");
    m.def("moe_combine_forward", &moe_combine_forward_cuda, "Fused MoE Combine Forward (CUDA)",
          py::arg("expert_outputs"), py::arg("topk_idx"), py::arg("topk_weights"),
          py::arg("token_to_slot"), py::arg("shared_out") = py::none());
    m.def("moe_combine_backward", &moe_combine_backward_cuda, "Fused MoE Combine Backward (CUDA)");
    m.def("qk_rmsnorm_forward", &qk_rmsnorm_forward_cuda, "Fused QK-RMSNorm Forward (CUDA)");
    m.def("qk_rmsnorm_backward", &qk_rmsnorm_backward_cuda, "Fused QK-RMSNorm Backward (CUDA)");
    m.def("fused_softcap_ce_forward", &fused_softcap_ce_forward_cuda, "Fused Softcap Cross-Entropy Forward (CUDA)",
          py::arg("logits"), py::arg("targets"), py::arg("softcap") = 30.0f, py::arg("ignore_index") = -100);
    m.def("fused_softcap_ce_backward", &fused_softcap_ce_backward_cuda, "Fused Softcap Cross-Entropy Backward (CUDA)",
          py::arg("logits"), py::arg("targets"), py::arg("lse"), py::arg("softcap") = 30.0f, py::arg("grad_scale") = 1.0f, py::arg("ignore_index") = -100);
    m.def("moe_router_topk_forward", &moe_router_topk_forward_cuda, "Fused MoE Router Top-2 Forward (CUDA)",
          py::arg("logits"), py::arg("expert_bias"), py::arg("z_loss_weight") = 1e-4f);
    m.def("moe_router_topk_backward", &moe_router_topk_backward_cuda, "Fused MoE Router Top-2 Backward (CUDA)",
          py::arg("d_topk_weights"), py::arg("logits"), py::arg("topk_idx"), py::arg("z_loss_weight") = 1e-4f);
}
