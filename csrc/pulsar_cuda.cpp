#include <torch/extension.h>
#include <vector>
#include <optional>

std::vector<torch::Tensor> rmsnorm_forward_cuda(torch::Tensor x, torch::Tensor weight, float eps);
std::vector<torch::Tensor> rmsnorm_backward_cuda(torch::Tensor dy, torch::Tensor x, torch::Tensor weight, torch::Tensor rsqrt);

std::vector<torch::Tensor> fused_add_rmsnorm_forward_cuda(torch::Tensor x, torch::Tensor residual, torch::Tensor weight, float eps);
std::vector<torch::Tensor> fused_add_rmsnorm_backward_cuda(torch::Tensor dy_normed, torch::Tensor dx_sum_upstream, torch::Tensor x_sum, torch::Tensor weight, torch::Tensor rsqrt);



torch::Tensor fused_swiglu_forward_cuda(torch::Tensor gu, int64_t d_ff);
torch::Tensor fused_swiglu_backward_cuda(torch::Tensor dout, torch::Tensor gu, int64_t d_ff);

torch::Tensor rope_forward_cuda(torch::Tensor qk, torch::Tensor cos, torch::Tensor sin);





std::vector<torch::Tensor> moe_dispatch_forward_cuda(
    torch::Tensor x,
    torch::Tensor topk_idx,
    torch::Tensor token_to_slot,
    int64_t num_experts,
    int64_t capacity,
    std::optional<torch::Tensor> expert_counts = std::nullopt
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
    torch::Tensor token_to_slot,
    std::optional<torch::Tensor> expert_counts = std::nullopt
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
torch::Tensor fused_softcap_ce_backward_dev_cuda(
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor lse,
    float softcap,
    torch::Tensor grad_scale_dev,
    int64_t ignore_index
);

std::vector<torch::Tensor> moe_router_topk_forward_cuda(
    torch::Tensor logits,
    torch::Tensor expert_bias,
    int64_t top_k = 2,
    float z_loss_weight = 1e-4f
);

torch::Tensor moe_router_topk_backward_cuda(
    torch::Tensor d_topk_weights,
    torch::Tensor logits,
    torch::Tensor topk_idx,
    float z_loss_weight = 1e-4f
);

std::vector<torch::Tensor> qk_rmsnorm_rope_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor cos,
    torch::Tensor sin,
    int64_t T,
    float eps
);

std::vector<torch::Tensor> qk_rmsnorm_rope_backward_cuda(
    torch::Tensor dy,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor rsqrt_in,
    torch::Tensor cos,
    torch::Tensor sin,
    int64_t T
);

torch::Tensor fused_attn_gate_forward_cuda(
    torch::Tensor attn_out,
    torch::Tensor gate_logits
);

std::vector<torch::Tensor> fused_attn_gate_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor attn_out,
    torch::Tensor gate_logits
);

std::vector<torch::Tensor> moe_slots_from_order_cuda(
    torch::Tensor flat_e,
    torch::Tensor order,
    int64_t num_experts,
    int64_t capacity
);

void fused_sr_update_cuda(
    torch::Tensor dst,
    torch::Tensor update,
    float lr_scale,
    float wd_factor,
    int64_t seed
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_forward", &rmsnorm_forward_cuda, "RMSNorm Forward (CUDA)");
    m.def("rmsnorm_backward", &rmsnorm_backward_cuda, "RMSNorm Backward (CUDA)");
    m.def("fused_add_rmsnorm_forward", &fused_add_rmsnorm_forward_cuda, "Fused Add + RMSNorm Forward (CUDA)",
          py::arg("x"), py::arg("residual"), py::arg("weight"), py::arg("eps") = 1e-6f);
    m.def("fused_add_rmsnorm_backward", &fused_add_rmsnorm_backward_cuda, "Fused Add + RMSNorm Backward (CUDA)",
          py::arg("dy_normed"), py::arg("dx_sum_upstream"), py::arg("x_sum"), py::arg("weight"), py::arg("rsqrt"));

    m.def("fused_swiglu_forward", &fused_swiglu_forward_cuda, "Fused SwiGLU Forward (CUDA)");
    m.def("fused_swiglu_backward", &fused_swiglu_backward_cuda, "Fused SwiGLU Backward (CUDA)");
    m.def("rope_forward", &rope_forward_cuda, "RoPE Forward (CUDA)");

    m.def("moe_dispatch_forward", &moe_dispatch_forward_cuda, "Fused MoE Dispatch Forward (CUDA)",
          py::arg("x"), py::arg("topk_idx"), py::arg("token_to_slot"),
          py::arg("num_experts"), py::arg("capacity"), py::arg("expert_counts") = py::none());
    m.def("moe_dispatch_backward", &moe_dispatch_backward_cuda, "Fused MoE Dispatch Backward (CUDA)");
    m.def("moe_combine_forward", &moe_combine_forward_cuda, "Fused MoE Combine Forward (CUDA)",
          py::arg("expert_outputs"), py::arg("topk_idx"), py::arg("topk_weights"),
          py::arg("token_to_slot"), py::arg("shared_out") = py::none());
    m.def("moe_combine_backward", &moe_combine_backward_cuda, "Fused MoE Combine Backward (CUDA)",
          py::arg("d_routed_out"), py::arg("expert_outputs"), py::arg("topk_idx"),
          py::arg("topk_weights"), py::arg("token_to_slot"), py::arg("expert_counts") = py::none());
    m.def("qk_rmsnorm_forward", &qk_rmsnorm_forward_cuda, "Fused QK-RMSNorm Forward (CUDA)");
    m.def("qk_rmsnorm_backward", &qk_rmsnorm_backward_cuda, "Fused QK-RMSNorm Backward (CUDA)");
    m.def("qk_rmsnorm_rope_forward", &qk_rmsnorm_rope_forward_cuda, "Fused QK-RMSNorm + RoPE Forward (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("cos"), py::arg("sin"), py::arg("T"), py::arg("eps") = 1e-6f);
    m.def("qk_rmsnorm_rope_backward", &qk_rmsnorm_rope_backward_cuda, "Fused QK-RMSNorm + RoPE Backward (CUDA)",
          py::arg("dy"), py::arg("x"), py::arg("weight"), py::arg("rsqrt_in"), py::arg("cos"), py::arg("sin"), py::arg("T"));

    m.def("fused_attn_gate_forward", &fused_attn_gate_forward_cuda, "Fused Attention Gate + Transpose Forward (CUDA)",
          py::arg("attn_out"), py::arg("gate_logits"));
    m.def("fused_attn_gate_backward", &fused_attn_gate_backward_cuda, "Fused Attention Gate + Transpose Backward (CUDA)",
          py::arg("grad_out"), py::arg("attn_out"), py::arg("gate_logits"));

    m.def("moe_slots_from_order", &moe_slots_from_order_cuda, "Fused MoE Slots from Order (CUDA)",
          py::arg("flat_e"), py::arg("order"), py::arg("num_experts"), py::arg("capacity"));

    m.def("fused_sr_update", &fused_sr_update_cuda, "Fused Muon Stochastic Rounding Update (CUDA)",
          py::arg("dst"), py::arg("update"), py::arg("lr_scale"), py::arg("wd_factor"), py::arg("seed"));

    m.def("fused_softcap_ce_forward", &fused_softcap_ce_forward_cuda, "Fused Softcap Cross-Entropy Forward (CUDA)",
          py::arg("logits"), py::arg("targets"), py::arg("softcap") = 30.0f, py::arg("ignore_index") = -100);
    m.def("fused_softcap_ce_backward", &fused_softcap_ce_backward_cuda, "Fused Softcap Cross-Entropy Backward (CUDA)",
          py::arg("logits"), py::arg("targets"), py::arg("lse"), py::arg("softcap") = 30.0f, py::arg("grad_scale") = 1.0f, py::arg("ignore_index") = -100);
    m.def("fused_softcap_ce_backward_dev", &fused_softcap_ce_backward_dev_cuda,
          "Fused Softcap Cross-Entropy Backward with a device-side grad scale (no host sync)",
          py::arg("logits"), py::arg("targets"), py::arg("lse"), py::arg("softcap") = 30.0f,
          py::arg("grad_scale_dev"), py::arg("ignore_index") = -100);
    m.def("moe_router_topk_forward",
          [](torch::Tensor logits, torch::Tensor expert_bias, int64_t top_k, float z_loss_weight) {
              return moe_router_topk_forward_cuda(logits, expert_bias, top_k, z_loss_weight);
          },
          "Fused MoE Router Top-K Forward (CUDA)",
          py::arg("logits"), py::arg("expert_bias"), py::arg("top_k") = 2, py::arg("z_loss_weight") = 1e-4f);
    m.def("moe_router_topk_forward",
          [](torch::Tensor logits, torch::Tensor expert_bias, float z_loss_weight) {
              return moe_router_topk_forward_cuda(logits, expert_bias, 2, z_loss_weight);
          },
          "Fused MoE Router Top-K Forward (CUDA) with default top_k=2",
          py::arg("logits"), py::arg("expert_bias"), py::arg("z_loss_weight") = 1e-4f);
    m.def("moe_router_topk_backward", &moe_router_topk_backward_cuda, "Fused MoE Router Top-K Backward (CUDA)",
          py::arg("d_topk_weights"), py::arg("logits"), py::arg("topk_idx"), py::arg("z_loss_weight") = 1e-4f);
}
