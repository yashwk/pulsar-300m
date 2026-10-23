#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <optional>

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float silu_fast(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float dsilu_fast(float x) {
    float s = 1.0f / (1.0f + __expf(-x));
    return s * (1.0f + x * (1.0f - s));
}

// ------------------------------------------------------------------------
// 1. Vectorized RMSNorm (128-bit loads/stores: 8 BF16 elements per thread)
// ------------------------------------------------------------------------
__global__ void rmsnorm_forward_vec8_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ weight,
    __nv_bfloat16* __restrict__ y,
    float* __restrict__ rsqrt_out,
    int total_tokens, int dim, float eps
) {
    int tok = blockIdx.x;
    if (tok >= total_tokens) return;

    const __nv_bfloat16* x_tok = x + tok * dim;
    __nv_bfloat16* y_tok = y + tok * dim;
    int num_vec = dim / 8;

    float sum_sq = 0.0f;
    for (int v = threadIdx.x; v < num_vec; v += blockDim.x) {
        uint4 in_u4 = *reinterpret_cast<const uint4*>(x_tok + v * 8);
        const __nv_bfloat16* in_bf = reinterpret_cast<const __nv_bfloat16*>(&in_u4);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = __bfloat162float(in_bf[k]);
            sum_sq += val * val;
        }
    }
    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float s_sum[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    if (lane == 0) s_sum[wid] = sum_sq;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        int num_warps = blockDim.x / WARP_SIZE;
        for (int w = 0; w < num_warps; ++w) total += s_sum[w];
        float inv_std = rsqrtf(total / dim + eps);
        s_sum[0] = inv_std;
        if (rsqrt_out) rsqrt_out[tok] = inv_std;
    }
    __syncthreads();

    float rsqrt_val = s_sum[0];
    for (int v = threadIdx.x; v < num_vec; v += blockDim.x) {
        uint4 in_u4 = *reinterpret_cast<const uint4*>(x_tok + v * 8);
        const __nv_bfloat16* in_bf = reinterpret_cast<const __nv_bfloat16*>(&in_u4);
        __nv_bfloat16 out_bf[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = __bfloat162float(in_bf[k]);
            float w = weight[v * 8 + k];
            out_bf[k] = __float2bfloat16(val * rsqrt_val * w);
        }
        *reinterpret_cast<uint4*>(y_tok + v * 8) = *reinterpret_cast<uint4*>(out_bf);
    }
}

__global__ void rmsnorm_forward_scalar_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ weight,
    __nv_bfloat16* __restrict__ y,
    float* __restrict__ rsqrt_out,
    int total_tokens, int dim, float eps
) {
    int tok = blockIdx.x;
    if (tok >= total_tokens) return;

    const __nv_bfloat16* x_tok = x + tok * dim;
    __nv_bfloat16* y_tok = y + tok * dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = __bfloat162float(x_tok[i]);
        sum_sq += val * val;
    }
    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float s_sum[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    if (lane == 0) s_sum[wid] = sum_sq;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        int num_warps = blockDim.x / WARP_SIZE;
        for (int w = 0; w < num_warps; ++w) total += s_sum[w];
        float inv_std = rsqrtf(total / dim + eps);
        s_sum[0] = inv_std;
        if (rsqrt_out) rsqrt_out[tok] = inv_std;
    }
    __syncthreads();

    float rsqrt_val = s_sum[0];
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = __bfloat162float(x_tok[i]);
        float w = weight[i];
        y_tok[i] = __float2bfloat16(val * rsqrt_val * w);
    }
}

__global__ void rmsnorm_backward_vec8_kernel(
    const __nv_bfloat16* __restrict__ dy,
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ rsqrt_in,
    __nv_bfloat16* __restrict__ dx,
    float* __restrict__ dweight,
    int total_tokens, int dim
) {
    int num_vec = dim / 8;
    float inv_dim = 1.0f / dim;
    float dw_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int tok = blockIdx.x; tok < total_tokens; tok += gridDim.x) {
        const __nv_bfloat16* dy_tok = dy + tok * dim;
        const __nv_bfloat16* x_tok = x + tok * dim;
        __nv_bfloat16* dx_tok = dx + tok * dim;
        float rsqrt_val = rsqrt_in[tok];

        float sum_dy_x_w = 0.0f;
        for (int v = threadIdx.x; v < num_vec; v += blockDim.x) {
            uint4 dy_u4 = *reinterpret_cast<const uint4*>(dy_tok + v * 8);
            uint4 x_u4 = *reinterpret_cast<const uint4*>(x_tok + v * 8);
            const __nv_bfloat16* dy_bf = reinterpret_cast<const __nv_bfloat16*>(&dy_u4);
            const __nv_bfloat16* x_bf = reinterpret_cast<const __nv_bfloat16*>(&x_u4);

            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float dy_val = __bfloat162float(dy_bf[k]);
                float x_val = __bfloat162float(x_bf[k]);
                float w_val = weight[v * 8 + k];
                sum_dy_x_w += dy_val * x_val * w_val;
                dw_acc[k] += dy_val * x_val * rsqrt_val;
            }
        }
        sum_dy_x_w = warp_reduce_sum(sum_dy_x_w);

        __shared__ float s_sum[32];
        int lane = threadIdx.x % WARP_SIZE;
        int wid = threadIdx.x / WARP_SIZE;
        if (lane == 0) s_sum[wid] = sum_dy_x_w;
        __syncthreads();

        if (threadIdx.x == 0) {
            float total = 0.0f;
            int num_warps = blockDim.x / WARP_SIZE;
            for (int w = 0; w < num_warps; ++w) total += s_sum[w];
            s_sum[0] = total;
        }
        __syncthreads();

        float total_dy_x_w = s_sum[0];

        for (int v = threadIdx.x; v < num_vec; v += blockDim.x) {
            uint4 dy_u4 = *reinterpret_cast<const uint4*>(dy_tok + v * 8);
            uint4 x_u4 = *reinterpret_cast<const uint4*>(x_tok + v * 8);
            const __nv_bfloat16* dy_bf = reinterpret_cast<const __nv_bfloat16*>(&dy_u4);
            const __nv_bfloat16* x_bf = reinterpret_cast<const __nv_bfloat16*>(&x_u4);
            __nv_bfloat16 dx_bf[8];

            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float dy_val = __bfloat162float(dy_bf[k]);
                float x_val = __bfloat162float(x_bf[k]);
                float w_val = weight[v * 8 + k];
                float term1 = w_val * dy_val * rsqrt_val;
                float term2 = x_val * (rsqrt_val * rsqrt_val * rsqrt_val) * total_dy_x_w * inv_dim;
                dx_bf[k] = __float2bfloat16(term1 - term2);
            }
            *reinterpret_cast<uint4*>(dx_tok + v * 8) = *reinterpret_cast<uint4*>(dx_bf);
        }
        __syncthreads();
    }

    // Accumulate dweight once per thread after finishing all tokens assigned to this block
    for (int v = threadIdx.x; v < num_vec; v += blockDim.x) {
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            atomicAdd(&dweight[v * 8 + k], dw_acc[k]);
        }
    }
}

std::vector<torch::Tensor> rmsnorm_forward_cuda(torch::Tensor x, torch::Tensor weight, float eps) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be a contiguous CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
    auto orig_shape = x.sizes();
    int dim = orig_shape.back();
    int total_tokens = x.numel() / dim;

    if (total_tokens == 0) {
        return {torch::empty_like(x), torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()))};
    }

    auto y = torch::empty_like(x);
    auto rsqrt_out = torch::empty({total_tokens}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));

    int threads = (dim >= 256) ? 128 : 32;
    int blocks = total_tokens;

    if (dim % 8 == 0) {
        rmsnorm_forward_vec8_kernel<<<blocks, threads>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<c10::BFloat16>()),
            weight.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(y.data_ptr<c10::BFloat16>()),
            rsqrt_out.data_ptr<float>(),
            total_tokens, dim, eps
        );
    } else {
        rmsnorm_forward_scalar_kernel<<<blocks, threads>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<c10::BFloat16>()),
            weight.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(y.data_ptr<c10::BFloat16>()),
            rsqrt_out.data_ptr<float>(),
            total_tokens, dim, eps
        );
    }
    return {y, rsqrt_out};
}

std::vector<torch::Tensor> rmsnorm_backward_cuda(torch::Tensor dy, torch::Tensor x, torch::Tensor weight, torch::Tensor rsqrt) {
    int dim = x.sizes().back();
    int total_tokens = x.numel() / dim;

    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be a contiguous CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(dy.scalar_type() == torch::kBFloat16, "dy must be bfloat16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");

    if (total_tokens == 0) {
        return {torch::empty_like(x), torch::zeros_like(weight)};
    }

    // The vec8 kernel accumulates per-thread dweight for the channel group at index
    // threadIdx.x. If num_vec > blockDim.x a thread handles several groups and the
    // partial sums for different channels get mixed. Keep one thread per group, and
    // cap at the hardware limit.
    TORCH_CHECK(dim % 8 == 0, "rmsnorm backward requires dim % 8 == 0, got ", dim);
    int num_vec = dim / 8;
    TORCH_CHECK(num_vec <= 1024, "rmsnorm backward supports num_vec <= 1024 (dim <= 8192), got dim ", dim);

    auto dx = torch::empty_like(x);
    auto dweight = torch::zeros_like(weight);

    // blockDim.x must be a whole number of warps: the kernel computes
    // num_warps = blockDim.x / WARP_SIZE, which truncates (48/32 = 1), dropping the
    // partial warp's contribution to total_dy_x_w (corrupting dx) and running
    // __shfl_down_sync with a full mask over a partially active warp. Rounding up
    // keeps every warp full; the surplus threads simply find v >= num_vec and add 0.
    //   dim=384 -> num_vec=48 -> threads=64 (was 48: 2.9e-1 dx error)
    int threads = std::min(1024, ((std::max(32, num_vec) + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE);
    int blocks = std::min(total_tokens, 256);
    rmsnorm_backward_vec8_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(dy.data_ptr<c10::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<c10::BFloat16>()),
        weight.data_ptr<float>(),
        rsqrt.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(dx.data_ptr<c10::BFloat16>()),
        dweight.data_ptr<float>(),
        total_tokens, dim
    );
    return {dx, dweight};
}

// ------------------------------------------------------------------------
// 2. Fused Residual Addition + RMSNorm (Forward)
// ------------------------------------------------------------------------
__global__ void fused_add_rmsnorm_forward_vec8_kernel(
    __nv_bfloat16* __restrict__ x, // in-out residual
    const __nv_bfloat16* __restrict__ res,
    const float* __restrict__ weight,
    __nv_bfloat16* __restrict__ y, // normalized output
    float* __restrict__ rsqrt_out,
    int total_tokens, int dim, float eps
) {
    int tok = blockIdx.x;
    if (tok >= total_tokens) return;

    __nv_bfloat16* x_tok = x + tok * dim;
    const __nv_bfloat16* res_tok = res + tok * dim;
    __nv_bfloat16* y_tok = y + tok * dim;
    int num_vec = dim / 8;

    float sum_sq = 0.0f;
    for (int v = threadIdx.x; v < num_vec; v += blockDim.x) {
        uint4 x_u4 = *reinterpret_cast<const uint4*>(x_tok + v * 8);
        uint4 res_u4 = *reinterpret_cast<const uint4*>(res_tok + v * 8);
        const __nv_bfloat16* x_bf = reinterpret_cast<const __nv_bfloat16*>(&x_u4);
        const __nv_bfloat16* res_bf = reinterpret_cast<const __nv_bfloat16*>(&res_u4);
        __nv_bfloat16 sum_bf[8];

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = __bfloat162float(x_bf[k]) + __bfloat162float(res_bf[k]);
            sum_bf[k] = __float2bfloat16(val);
            float stored = __bfloat162float(sum_bf[k]);
            sum_sq += stored * stored;
        }
        *reinterpret_cast<uint4*>(x_tok + v * 8) = *reinterpret_cast<uint4*>(sum_bf);
    }
    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float s_sum[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    if (lane == 0) s_sum[wid] = sum_sq;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        int num_warps = blockDim.x / WARP_SIZE;
        for (int w = 0; w < num_warps; ++w) total += s_sum[w];
        float inv_std = rsqrtf(total / dim + eps);
        s_sum[0] = inv_std;
        if (rsqrt_out) rsqrt_out[tok] = inv_std;
    }
    __syncthreads();

    float rsqrt_val = s_sum[0];
    for (int v = threadIdx.x; v < num_vec; v += blockDim.x) {
        uint4 x_u4 = *reinterpret_cast<const uint4*>(x_tok + v * 8);
        const __nv_bfloat16* x_bf = reinterpret_cast<const __nv_bfloat16*>(&x_u4);
        __nv_bfloat16 y_bf[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = __bfloat162float(x_bf[k]);
            float w = weight[v * 8 + k];
            y_bf[k] = __float2bfloat16(val * rsqrt_val * w);
        }
        *reinterpret_cast<uint4*>(y_tok + v * 8) = *reinterpret_cast<uint4*>(y_bf);
    }
}

std::vector<torch::Tensor> fused_add_rmsnorm_cuda(torch::Tensor x, torch::Tensor residual, torch::Tensor weight, float eps) {
    int dim = x.sizes().back();
    int total_tokens = x.numel() / dim;

    if (total_tokens == 0) {
        return {x, torch::empty_like(x), torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()))};
    }

    auto y = torch::empty_like(x);
    auto rsqrt_out = torch::empty({total_tokens}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));

    int threads = (dim >= 256) ? 128 : 32;
    int blocks = total_tokens;
    fused_add_rmsnorm_forward_vec8_kernel<<<blocks, threads>>>(
        reinterpret_cast<__nv_bfloat16*>(x.data_ptr<c10::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<c10::BFloat16>()),
        weight.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr<c10::BFloat16>()),
        rsqrt_out.data_ptr<float>(),
        total_tokens, dim, eps
    );
    return {x, y, rsqrt_out};
}

// ------------------------------------------------------------------------
// 3. Vectorized SwiGLU (Forward & Backward: 128-bit uint4)
// ------------------------------------------------------------------------
__global__ void swiglu_forward_vec_kernel(
    const __nv_bfloat16* __restrict__ g,
    const __nv_bfloat16* __restrict__ u,
    __nv_bfloat16* __restrict__ out,
    int n
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (idx + 8 <= n) {
        uint4 g_u4 = *reinterpret_cast<const uint4*>(g + idx);
        uint4 u_u4 = *reinterpret_cast<const uint4*>(u + idx);
        const __nv_bfloat16* g_bf = reinterpret_cast<const __nv_bfloat16*>(&g_u4);
        const __nv_bfloat16* u_bf = reinterpret_cast<const __nv_bfloat16*>(&u_u4);
        __nv_bfloat16 out_bf[8];

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float g_val = __bfloat162float(g_bf[k]);
            float u_val = __bfloat162float(u_bf[k]);
            out_bf[k] = __float2bfloat16(silu_fast(g_val) * u_val);
        }
        *reinterpret_cast<uint4*>(out + idx) = *reinterpret_cast<uint4*>(out_bf);
    } else {
        for (int i = idx; i < n; ++i) {
            float g_val = __bfloat162float(g[i]);
            float u_val = __bfloat162float(u[i]);
            out[i] = __float2bfloat16(silu_fast(g_val) * u_val);
        }
    }
}

__global__ void swiglu_backward_vec_kernel(
    const __nv_bfloat16* __restrict__ dout,
    const __nv_bfloat16* __restrict__ g,
    const __nv_bfloat16* __restrict__ u,
    __nv_bfloat16* __restrict__ dg,
    __nv_bfloat16* __restrict__ du,
    int n
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (idx + 8 <= n) {
        uint4 dout_u4 = *reinterpret_cast<const uint4*>(dout + idx);
        uint4 g_u4 = *reinterpret_cast<const uint4*>(g + idx);
        uint4 u_u4 = *reinterpret_cast<const uint4*>(u + idx);

        const __nv_bfloat16* do_bf = reinterpret_cast<const __nv_bfloat16*>(&dout_u4);
        const __nv_bfloat16* g_bf = reinterpret_cast<const __nv_bfloat16*>(&g_u4);
        const __nv_bfloat16* u_bf = reinterpret_cast<const __nv_bfloat16*>(&u_u4);

        __nv_bfloat16 dg_bf[8];
        __nv_bfloat16 du_bf[8];

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float do_val = __bfloat162float(do_bf[k]);
            float g_val = __bfloat162float(g_bf[k]);
            float u_val = __bfloat162float(u_bf[k]);

            du_bf[k] = __float2bfloat16(do_val * silu_fast(g_val));
            dg_bf[k] = __float2bfloat16(do_val * u_val * dsilu_fast(g_val));
        }
        *reinterpret_cast<uint4*>(dg + idx) = *reinterpret_cast<uint4*>(dg_bf);
        *reinterpret_cast<uint4*>(du + idx) = *reinterpret_cast<uint4*>(du_bf);
    } else {
        for (int i = idx; i < n; ++i) {
            float do_val = __bfloat162float(dout[i]);
            float g_val = __bfloat162float(g[i]);
            float u_val = __bfloat162float(u[i]);
            du[i] = __float2bfloat16(do_val * silu_fast(g_val));
            dg[i] = __float2bfloat16(do_val * u_val * dsilu_fast(g_val));
        }
    }
}

torch::Tensor swiglu_forward_cuda(torch::Tensor gate, torch::Tensor up) {
    auto out = torch::empty_like(gate);
    int n = gate.numel();
    int n_vec = (n + 7) / 8;
    swiglu_forward_vec_kernel<<<(n_vec + 255) / 256, 256>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr<c10::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(up.data_ptr<c10::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<c10::BFloat16>()),
        n
    );
    return out;
}

std::vector<torch::Tensor> swiglu_backward_cuda(torch::Tensor dout, torch::Tensor gate, torch::Tensor up) {
    auto dgate = torch::empty_like(gate);
    auto dup = torch::empty_like(up);
    int n = gate.numel();
    int n_vec = (n + 7) / 8;
    swiglu_backward_vec_kernel<<<(n_vec + 255) / 256, 256>>>(
        reinterpret_cast<const __nv_bfloat16*>(dout.data_ptr<c10::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr<c10::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(up.data_ptr<c10::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(dgate.data_ptr<c10::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(dup.data_ptr<c10::BFloat16>()),
        n
    );
    return {dgate, dup};
}

// ------------------------------------------------------------------------
// 3b. Fused SwiGLU: direct [..., 2*d_ff] -> [..., d_ff] (zero .chunk/.contiguous overhead)
// ------------------------------------------------------------------------
__global__ void fused_swiglu_forward_kernel(
    const __nv_bfloat16* __restrict__ gu,
    __nv_bfloat16* __restrict__ out,
    int N,
    int d_ff
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * d_ff;
    if (idx < total) {
        int r = idx / d_ff;
        int c = idx % d_ff;
        int row_offset = r * (2 * d_ff);
        float g_val = __bfloat162float(gu[row_offset + c]);
        float u_val = __bfloat162float(gu[row_offset + d_ff + c]);
        out[idx] = __float2bfloat16(silu_fast(g_val) * u_val);
    }
}

__global__ void fused_swiglu_backward_kernel(
    const __nv_bfloat16* __restrict__ dout,
    const __nv_bfloat16* __restrict__ gu,
    __nv_bfloat16* __restrict__ dgu,
    int N,
    int d_ff
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * d_ff;
    if (idx < total) {
        int r = idx / d_ff;
        int c = idx % d_ff;
        int row_offset = r * (2 * d_ff);
        float do_val = __bfloat162float(dout[idx]);
        float g_val = __bfloat162float(gu[row_offset + c]);
        float u_val = __bfloat162float(gu[row_offset + d_ff + c]);
        float du = do_val * silu_fast(g_val);
        float dg = do_val * u_val * dsilu_fast(g_val);
        dgu[row_offset + c] = __float2bfloat16(dg);
        dgu[row_offset + d_ff + c] = __float2bfloat16(du);
    }
}

torch::Tensor fused_swiglu_forward_cuda(torch::Tensor gu, int64_t d_ff) {
    TORCH_CHECK(gu.is_cuda() && gu.is_contiguous(), "gu must be a contiguous CUDA tensor");
    TORCH_CHECK(gu.scalar_type() == torch::kBFloat16, "gu must be bfloat16");
    int total_elements = gu.numel();
    if (total_elements == 0) {
        auto sizes = gu.sizes().vec();
        sizes.back() = d_ff;
        return torch::empty(sizes, gu.options());
    }
    int N = total_elements / (2 * d_ff);
    auto sizes = gu.sizes().vec();
    sizes.back() = d_ff;
    auto out = torch::empty(sizes, gu.options());
    int total = N * d_ff;
    fused_swiglu_forward_kernel<<<(total + 255) / 256, 256>>>(
        reinterpret_cast<const __nv_bfloat16*>(gu.data_ptr<c10::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<c10::BFloat16>()),
        N, (int)d_ff
    );
    return out;
}

torch::Tensor fused_swiglu_backward_cuda(torch::Tensor dout, torch::Tensor gu, int64_t d_ff) {
    TORCH_CHECK(gu.is_cuda() && gu.is_contiguous(), "gu must be a contiguous CUDA tensor");
    TORCH_CHECK(gu.scalar_type() == torch::kBFloat16, "gu must be bfloat16");
    TORCH_CHECK(dout.is_contiguous(), "dout must be contiguous");
    TORCH_CHECK(dout.scalar_type() == torch::kBFloat16, "dout must be bfloat16");
    auto dgu = torch::empty_like(gu);
    int total_elements = gu.numel();
    if (total_elements == 0) {
        return dgu;
    }
    int N = total_elements / (2 * d_ff);
    int total = N * d_ff;
    fused_swiglu_backward_kernel<<<(total + 255) / 256, 256>>>(
        reinterpret_cast<const __nv_bfloat16*>(dout.data_ptr<c10::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(gu.data_ptr<c10::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(dgu.data_ptr<c10::BFloat16>()),
        N, (int)d_ff
    );
    return dgu;
}

// ------------------------------------------------------------------------
// 4. RoPE (Optimized without redundant qk.clone())
// ------------------------------------------------------------------------
__global__ void rope_forward_kernel(
    const __nv_bfloat16* __restrict__ qk_in,
    __nv_bfloat16* __restrict__ qk_out,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    int B, int num_heads, int T, int head_dim
) {
    int half_dim = head_dim / 2;
    // Flatten (b, h, t) onto blockIdx.x. The previous layout put T on gridDim.y, whose
    // hardware limit is 65535, so T >= 65536 failed to launch silently.
    int row = blockIdx.x;
    int t = row % T;
    int h = (row / T) % num_heads;
    int b = row / (T * num_heads);
    int i = threadIdx.x;

    if (i < half_dim && t < T && h < num_heads && b < B) {
        int idx1 = b * (num_heads * T * head_dim) + h * (T * head_dim) + t * head_dim + i;
        int idx2 = idx1 + half_dim;

        float x1 = __bfloat162float(qk_in[idx1]);
        float x2 = __bfloat162float(qk_in[idx2]);
        float cos_val = cos_cache[t * half_dim + i];
        float sin_val = sin_cache[t * half_dim + i];

        qk_out[idx1] = __float2bfloat16(x1 * cos_val - x2 * sin_val);
        qk_out[idx2] = __float2bfloat16(x2 * cos_val + x1 * sin_val);
    }
}

torch::Tensor rope_forward_cuda(torch::Tensor qk, torch::Tensor cos, torch::Tensor sin) {
    int B = qk.size(0);
    int num_heads = qk.size(1);
    int T = qk.size(2);
    int head_dim = qk.size(3);

    TORCH_CHECK(qk.is_cuda() && qk.is_contiguous(), "qk must be a contiguous CUDA tensor");
    TORCH_CHECK(qk.scalar_type() == torch::kBFloat16, "qk must be bfloat16");

    auto out = torch::empty_like(qk);
    // A zero-sized tensor would produce grid.x == 0, which is an invalid launch.
    if (qk.numel() == 0) {
        return out;
    }
    // One block per (b, h, t) flattened onto x, which supports 2^31-1 blocks.
    dim3 grid(B * T * num_heads);
    dim3 block(head_dim / 2);

    auto cos_f = (cos.scalar_type() == torch::kFloat32) ? cos.contiguous() : cos.to(torch::kFloat32).contiguous();
    auto sin_f = (sin.scalar_type() == torch::kFloat32) ? sin.contiguous() : sin.to(torch::kFloat32).contiguous();

    rope_forward_kernel<<<grid, block>>>(
        reinterpret_cast<const __nv_bfloat16*>(qk.data_ptr<c10::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<c10::BFloat16>()),
        cos_f.data_ptr<float>(),
        sin_f.data_ptr<float>(),
        B, num_heads, T, head_dim
    );
    return out;
}

// ------------------------------------------------------------------------
// 5. Engram Hash & 12-Head Vectorized Table Gather
// ------------------------------------------------------------------------
__global__ void engram_gather_kernel(
    const int64_t* __restrict__ input_ids,
    const int64_t* __restrict__ canonical_map,
    const __nv_bfloat16* __restrict__ table,
    __nv_bfloat16* __restrict__ out,
    int B, int T, int total_tokens,
    int engram_heads, int slots_per_head, int head_dim,
    int eos_id
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_tokens) return;

    int b = idx / T;
    int t = idx % T;

    int64_t c0 = canonical_map[input_ids[idx]];
    int64_t c1 = 0, c2 = 0, c3 = 0;

    if (t >= 1) {
        int64_t id1 = input_ids[b * T + (t - 1)];
        if (id1 != eos_id) c1 = canonical_map[id1];
    }
    if (t >= 2) {
        int64_t id1 = input_ids[b * T + (t - 1)];
        int64_t id2 = input_ids[b * T + (t - 2)];
        if (id1 != eos_id && id2 != eos_id) c2 = canonical_map[id2];
    }
    if (t >= 3) {
        int64_t id1 = input_ids[b * T + (t - 1)];
        int64_t id2 = input_ids[b * T + (t - 2)];
        int64_t id3 = input_ids[b * T + (t - 3)];
        if (id1 != eos_id && id2 != eos_id && id3 != eos_id) c3 = canonical_map[id3];
    }

    const int64_t primes[12][4] = {
        // 4 Heads for 2-grams (Heads 0..3)
        {1000003, 1000033, 1, 1},
        {1000037, 1000039, 1, 1},
        {1000081, 1000099, 1, 1},
        {1000117, 1000121, 1, 1},
        // 4 Heads for 3-grams (Heads 4..7)
        {1000133, 1000151, 1000159, 1},
        {1000171, 1000183, 1000187, 1},
        {1000193, 1000199, 1000211, 1},
        {1000213, 1000231, 1000249, 1},
        // 4 Heads for 4-grams (Heads 8..11)
        {1000253, 1000273, 1000289, 1000291},
        {1000303, 1000313, 1000333, 1000357},
        {1000367, 1000393, 1000397, 1000403},
        {1000409, 1000411, 1000423, 1000427}
    };

    __nv_bfloat16* out_tok = out + idx * (engram_heads * head_dim);

    for (int h = 0; h < engram_heads; ++h) {
        int64_t h0 = c0 * primes[h][0];
        int64_t h1 = c1 * primes[h][1];
        int64_t h2 = (primes[h][2] > 1) ? (c2 * primes[h][2]) : 0;
        int64_t h3 = (primes[h][3] > 1) ? (c3 * primes[h][3]) : 0;

        int64_t raw_hash = (slots_per_head & (slots_per_head - 1)) == 0
            ? ((h0 ^ h1 ^ h2 ^ h3) & (slots_per_head - 1))
            : ((h0 ^ h1 ^ h2 ^ h3) % slots_per_head);
        int slot_idx = (int)raw_hash + h * slots_per_head;

        const __nv_bfloat16* src = table + slot_idx * head_dim;
        __nv_bfloat16* dst = out_tok + h * head_dim;

        // 128-bit vectorized gather (8 uint4 transactions for 64 head_dim)
        const uint4* src_vec = reinterpret_cast<const uint4*>(src);
        uint4* dst_vec = reinterpret_cast<uint4*>(dst);
        #pragma unroll
        for (int d = 0; d < 8; ++d) {
            dst_vec[d] = src_vec[d];
        }
    }
}

torch::Tensor engram_hash_gather_cuda(
    torch::Tensor input_ids,
    torch::Tensor canonical_map,
    torch::Tensor table,
    int engram_heads,
    int slots_per_head,
    int head_dim,
    int eos_id
) {
    int B = input_ids.size(0);
    int T = input_ids.size(1);
    int total_tokens = B * T;

    auto out = torch::empty({B, T, engram_heads * head_dim}, table.options());

    int threads = 128;
    int blocks = (total_tokens + threads - 1) / threads;

    engram_gather_kernel<<<blocks, threads>>>(
        input_ids.data_ptr<int64_t>(),
        canonical_map.data_ptr<int64_t>(),
        reinterpret_cast<const __nv_bfloat16*>(table.data_ptr<c10::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<c10::BFloat16>()),
        B, T, total_tokens,
        engram_heads, slots_per_head, head_dim,
        eos_id
    );
    return out;
}

__global__ void engram_hash_kernel(
    const int64_t* __restrict__ input_ids,
    const int64_t* __restrict__ canonical_map,
    int64_t* __restrict__ indices_out,
    int B, int T, int total_tokens,
    int engram_heads, int slots_per_head,
    int eos_id
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_tokens) return;

    int b = idx / T;
    int t = idx % T;

    int64_t c0 = canonical_map[input_ids[idx]];
    int64_t c1 = 0, c2 = 0, c3 = 0;

    if (t >= 1) {
        int64_t id1 = input_ids[b * T + (t - 1)];
        if (id1 != eos_id) c1 = canonical_map[id1];
    }
    if (t >= 2) {
        int64_t id1 = input_ids[b * T + (t - 1)];
        int64_t id2 = input_ids[b * T + (t - 2)];
        if (id1 != eos_id && id2 != eos_id) c2 = canonical_map[id2];
    }
    if (t >= 3) {
        int64_t id1 = input_ids[b * T + (t - 1)];
        int64_t id2 = input_ids[b * T + (t - 2)];
        int64_t id3 = input_ids[b * T + (t - 3)];
        if (id1 != eos_id && id2 != eos_id && id3 != eos_id) c3 = canonical_map[id3];
    }

    const int64_t primes[12][4] = {
        {1000003, 1000033, 1, 1},
        {1000037, 1000039, 1, 1},
        {1000081, 1000099, 1, 1},
        {1000117, 1000121, 1, 1},
        {1000133, 1000151, 1000159, 1},
        {1000171, 1000183, 1000187, 1},
        {1000193, 1000199, 1000211, 1},
        {1000213, 1000231, 1000249, 1},
        {1000253, 1000273, 1000289, 1000291},
        {1000303, 1000313, 1000333, 1000357},
        {1000367, 1000393, 1000397, 1000403},
        {1000409, 1000411, 1000423, 1000427}
    };

    int64_t* out_tok = indices_out + idx * engram_heads;

    #pragma unroll
    for (int h = 0; h < engram_heads; ++h) {
        int64_t h0 = c0 * primes[h][0];
        int64_t h1 = c1 * primes[h][1];
        int64_t h2 = (primes[h][2] > 1) ? (c2 * primes[h][2]) : 0;
        int64_t h3 = (primes[h][3] > 1) ? (c3 * primes[h][3]) : 0;

        int64_t raw_hash = (slots_per_head & (slots_per_head - 1)) == 0
            ? ((h0 ^ h1 ^ h2 ^ h3) & (slots_per_head - 1))
            : ((h0 ^ h1 ^ h2 ^ h3) % slots_per_head);
        out_tok[h] = raw_hash + (int64_t)h * slots_per_head;
    }
}

torch::Tensor engram_hash_cuda(
    torch::Tensor input_ids,
    torch::Tensor canonical_map,
    int engram_heads,
    int slots_per_head,
    int eos_id
) {
    int B = input_ids.size(0);
    int T = input_ids.size(1);
    int total_tokens = B * T;

    auto indices = torch::empty({B, T, engram_heads}, input_ids.options());

    int threads = 128;
    int blocks = (total_tokens + threads - 1) / threads;

    engram_hash_kernel<<<blocks, threads>>>(
        input_ids.data_ptr<int64_t>(),
        canonical_map.data_ptr<int64_t>(),
        indices.data_ptr<int64_t>(),
        B, T, total_tokens,
        engram_heads, slots_per_head,
        eos_id
    );
    return indices;
}

// ------------------------------------------------------------------------
// 6. Muon Newton-Schulz 5 Update
// ------------------------------------------------------------------------
torch::Tensor zeropower_via_newtonschulz5_cuda(torch::Tensor G, int steps = 5, float eps = 1e-7) {
    float a = 3.4445f, b = -4.7750f, c = 2.0315f;
    auto X = G.to(torch::kBFloat16);
    float norm = X.norm().item<float>() + eps;
    X = X / norm;

    bool transposed = false;
    if (X.size(0) > X.size(1)) {
        X = X.t().contiguous();
        transposed = true;
    }

    for (int s = 0; s < steps; ++s) {
        auto A = torch::matmul(X, X.t());
        auto B = b * A + c * torch::matmul(A, A);
        X = a * X + torch::matmul(B, X);
    }

    if (transposed) {
        X = X.t().contiguous();
    }
    return X;
}

void muon_step_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor momentum_buffer,
    float lr,
    float momentum,
    float weight_decay
) {
    momentum_buffer.mul_(momentum).add_(grad);
    auto g_orth = zeropower_via_newtonschulz5_cuda(grad.add(momentum_buffer, momentum));
    float scale = sqrtf(fmaxf(1.0f, (float)param.size(0) / (float)param.size(1)));

    if (weight_decay > 0.0f) {
        param.mul_(1.0f - lr * weight_decay);
    }
    param.add_(g_orth.to(param.dtype()), -lr * scale);
}

// ------------------------------------------------------------------------
// 7. In-place AdamW Step
// ------------------------------------------------------------------------
__global__ void adamw_step_kernel(
    float* __restrict__ p,
    const float* __restrict__ g,
    float* __restrict__ m,
    float* __restrict__ v,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bias_correction1, float bias_correction2,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float grad = g[idx];
        if (weight_decay > 0.0f) {
            p[idx] -= lr * weight_decay * p[idx];
        }
        float m_val = beta1 * m[idx] + (1.0f - beta1) * grad;
        float v_val = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
        m[idx] = m_val;
        v[idx] = v_val;

        float m_hat = m_val / bias_correction1;
        float v_hat = v_val / bias_correction2;
        p[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

void adamw_step_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    int step
) {
    int n = param.numel();
    float bc1 = 1.0f - powf(beta1, (float)step);
    float bc2 = 1.0f - powf(beta2, (float)step);

    adamw_step_kernel<<<(n + 255) / 256, 256>>>(
        param.data_ptr<float>(),
        grad.data_ptr<float>(),
        exp_avg.data_ptr<float>(),
        exp_avg_sq.data_ptr<float>(),
        lr, beta1, beta2, eps, weight_decay,
        bc1, bc2, n
    );
}

// ------------------------------------------------------------------------
// 8. Fused MoE Dispatch & Combine Kernels
// ------------------------------------------------------------------------

__global__ void moe_dispatch_forward_kernel(
    const __nv_bfloat16* __restrict__ x,           // [N, D]
    const int64_t* __restrict__ topk_idx,          // [N, K]
    const int32_t* __restrict__ token_to_slot,     // [N, K] precomputed, -1 = dropped
    __nv_bfloat16* __restrict__ expert_inputs,     // [E, C, D]
    int N, int D, int K, int E, int C
) {
    int idx = blockIdx.x; // Block per (token, k): 0 <= idx < N * K
    if (idx >= N * K) return;

    // Slot assignment is decided outside the kernel so that which assignments are
    // dropped is deterministic: the caller keeps the highest routing weights and
    // drops the rest, independent of block scheduling. This kernel only copies.
    int slot = token_to_slot[idx];
    if (slot < 0 || slot >= C) return;

    int expert_id = (int)topk_idx[idx];
    if (expert_id < 0 || expert_id >= E) return;

    int token_idx = idx / K;
    const __nv_bfloat16* src = x + token_idx * D;
    __nv_bfloat16* dst = expert_inputs + (expert_id * C + slot) * D;

    if (D % 8 == 0) {
        int num_vec = D / 8;
        for (int v = threadIdx.x; v < num_vec; v += blockDim.x) {
            *reinterpret_cast<uint4*>(dst + v * 8) = *reinterpret_cast<const uint4*>(src + v * 8);
        }
    } else {
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            dst[d] = src[d];
        }
    }
}

__global__ void moe_dispatch_backward_kernel(
    const __nv_bfloat16* __restrict__ d_expert_inputs, // [E, C, D]
    const int64_t* __restrict__ topk_idx,              // [N, K]
    const int32_t* __restrict__ token_to_slot,         // [N, K]
    __nv_bfloat16* __restrict__ dx,                    // [N, D]
    int N, int D, int K, int E, int C
) {
    int token_idx = blockIdx.x; // Block per token: 0 <= token_idx < N
    if (token_idx >= N) return;

    __nv_bfloat16* dx_ptr = dx + token_idx * D;
    int num_vec = D / 8;

    for (int v = threadIdx.x; v < num_vec; v += blockDim.x) {
        float acc[8] = {0.0f};

        for (int k = 0; k < K; ++k) {
            int flat_idx = token_idx * K + k;
            int expert_id = (int)topk_idx[flat_idx];
            int slot = token_to_slot[flat_idx];

            if (slot >= 0 && slot < C && expert_id >= 0 && expert_id < E) {
                uint4 in_u4 = *reinterpret_cast<const uint4*>(
                    d_expert_inputs + ((expert_id * C + slot) * D + v * 8)
                );
                const __nv_bfloat16* in_bf = reinterpret_cast<const __nv_bfloat16*>(&in_u4);
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc[i] += __bfloat162float(in_bf[i]);
                }
            }
        }

        __nv_bfloat16 out_bf[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            out_bf[i] = __float2bfloat16(acc[i]);
        }
        *reinterpret_cast<uint4*>(dx_ptr + v * 8) = *reinterpret_cast<uint4*>(out_bf);
    }
}

__global__ void moe_combine_forward_kernel(
    const __nv_bfloat16* __restrict__ expert_outputs, // [E, C, D]
    const int64_t* __restrict__ topk_idx,             // [N, K]
    const __nv_bfloat16* __restrict__ topk_weights,   // [N, K]
    const int32_t* __restrict__ token_to_slot,        // [N, K]
    const __nv_bfloat16* __restrict__ shared_out,     // [N, D] (optional fused addition)
    __nv_bfloat16* __restrict__ routed_out,           // [N, D]
    int N, int D, int K, int E, int C
) {
    int token_idx = blockIdx.x; // Block per token: 0 <= token_idx < N
    if (token_idx >= N) return;

    __nv_bfloat16* out_ptr = routed_out + token_idx * D;
    int num_vec = D / 8;

    for (int v = threadIdx.x; v < num_vec; v += blockDim.x) {
        float acc[8] = {0.0f};

        // Fused shared expert residual initialization (zero extra memory roundtrips)
        if (shared_out != nullptr) {
            uint4 sh_u4 = *reinterpret_cast<const uint4*>(shared_out + token_idx * D + v * 8);
            const __nv_bfloat16* sh_bf = reinterpret_cast<const __nv_bfloat16*>(&sh_u4);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[i] = __bfloat162float(sh_bf[i]);
            }
        }

        for (int k = 0; k < K; ++k) {
            int flat_idx = token_idx * K + k;
            int expert_id = (int)topk_idx[flat_idx];
            int slot = token_to_slot[flat_idx];
            float w = __bfloat162float(topk_weights[flat_idx]);

            if (slot >= 0 && slot < C && expert_id >= 0 && expert_id < E) {
                uint4 in_u4 = *reinterpret_cast<const uint4*>(
                    expert_outputs + ((expert_id * C + slot) * D + v * 8)
                );
                const __nv_bfloat16* in_bf = reinterpret_cast<const __nv_bfloat16*>(&in_u4);
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc[i] += w * __bfloat162float(in_bf[i]);
                }
            }
        }

        __nv_bfloat16 out_bf[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            out_bf[i] = __float2bfloat16(acc[i]);
        }
        *reinterpret_cast<uint4*>(out_ptr + v * 8) = *reinterpret_cast<uint4*>(out_bf);
    }
}

__global__ void moe_combine_backward_kernel(
    const __nv_bfloat16* __restrict__ d_routed_out,   // [N, D]
    const __nv_bfloat16* __restrict__ expert_outputs, // [E, C, D]
    const int64_t* __restrict__ topk_idx,             // [N, K]
    const __nv_bfloat16* __restrict__ topk_weights,   // [N, K]
    const int32_t* __restrict__ token_to_slot,        // [N, K]
    __nv_bfloat16* __restrict__ d_expert_outputs,     // [E, C, D]
    __nv_bfloat16* __restrict__ d_topk_weights,       // [N, K]
    int N, int D, int K, int E, int C
) {
    int idx = blockIdx.x; // Block per (token, k): 0 <= idx < N * K
    if (idx >= N * K) return;

    int token_idx = idx / K;
    int expert_id = (int)topk_idx[idx];
    int slot = token_to_slot[idx];
    float w = __bfloat162float(topk_weights[idx]);

    float dot_val = 0.0f;

    if (slot >= 0 && slot < C && expert_id >= 0 && expert_id < E) {
        const __nv_bfloat16* dy_ptr = d_routed_out + token_idx * D;
        const __nv_bfloat16* e_out_ptr = expert_outputs + (expert_id * C + slot) * D;
        __nv_bfloat16* de_out_ptr = d_expert_outputs + (expert_id * C + slot) * D;

        int num_vec = D / 8;
        for (int v = threadIdx.x; v < num_vec; v += blockDim.x) {
            uint4 dy_u4 = *reinterpret_cast<const uint4*>(dy_ptr + v * 8);
            uint4 eo_u4 = *reinterpret_cast<const uint4*>(e_out_ptr + v * 8);
            const __nv_bfloat16* dy_bf = reinterpret_cast<const __nv_bfloat16*>(&dy_u4);
            const __nv_bfloat16* eo_bf = reinterpret_cast<const __nv_bfloat16*>(&eo_u4);

            __nv_bfloat16 de_bf[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float dy_f = __bfloat162float(dy_bf[i]);
                float eo_f = __bfloat162float(eo_bf[i]);
                dot_val += dy_f * eo_f;
                de_bf[i] = __float2bfloat16(w * dy_f);
            }
            *reinterpret_cast<uint4*>(de_out_ptr + v * 8) = *reinterpret_cast<uint4*>(de_bf);
        }
    }

    dot_val = warp_reduce_sum(dot_val);
    __shared__ float s_dot[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    if (lane == 0) s_dot[wid] = dot_val;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total_dot = 0.0f;
        int num_warps = blockDim.x / WARP_SIZE;
        for (int i = 0; i < num_warps; ++i) total_dot += s_dot[i];
        d_topk_weights[idx] = __float2bfloat16(total_dot);
    }
}

std::vector<torch::Tensor> moe_dispatch_forward_cuda(
    torch::Tensor x,
    torch::Tensor topk_idx,
    torch::Tensor token_to_slot,
    int64_t num_experts,
    int64_t capacity
) {
    int N = x.size(0);
    int D = x.size(1);
    int K = topk_idx.size(1);
    int E = (int)num_experts;
    int C = (int)capacity;

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(topk_idx.is_cuda(), "topk_idx must be a CUDA tensor");
    TORCH_CHECK(token_to_slot.is_cuda(), "token_to_slot must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(topk_idx.is_contiguous(), "topk_idx must be contiguous");
    TORCH_CHECK(token_to_slot.is_contiguous(), "token_to_slot must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(topk_idx.scalar_type() == torch::kInt64, "topk_idx must be int64");
    TORCH_CHECK(token_to_slot.scalar_type() == torch::kInt32, "token_to_slot must be int32");
    TORCH_CHECK(token_to_slot.size(0) == N && token_to_slot.size(1) == K,
                "token_to_slot must be [N, K]");
    TORCH_CHECK(D % 8 == 0, "D must be a multiple of 8 for vectorised dispatch, got ", D);

    auto expert_inputs = torch::zeros({E, C, D}, x.options());

    if (N > 0 && K > 0) {
        int threads = (D == 768) ? 96 : 128;
        int blocks = N * K;
        moe_dispatch_forward_kernel<<<blocks, threads>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<c10::BFloat16>()),
            topk_idx.data_ptr<int64_t>(),
            token_to_slot.data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(expert_inputs.data_ptr<c10::BFloat16>()),
            N, D, K, E, C
        );
    }
    return {expert_inputs};
}

torch::Tensor moe_dispatch_backward_cuda(
    torch::Tensor d_expert_inputs,
    torch::Tensor topk_idx,
    torch::Tensor token_to_slot,
    int64_t N,
    int64_t D
) {
    int E = d_expert_inputs.size(0);
    int C = d_expert_inputs.size(1);
    int K = topk_idx.size(1);

    auto dx = torch::empty({N, D}, d_expert_inputs.options());
    if (N > 0) {
        int threads = (D == 768) ? 96 : 128;
        int blocks = (int)N;
        moe_dispatch_backward_kernel<<<blocks, threads>>>(
            reinterpret_cast<const __nv_bfloat16*>(d_expert_inputs.data_ptr<c10::BFloat16>()),
            topk_idx.data_ptr<int64_t>(),
            token_to_slot.data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(dx.data_ptr<c10::BFloat16>()),
            (int)N, (int)D, K, E, C
        );
    }
    return dx;
}

torch::Tensor moe_combine_forward_cuda(
    torch::Tensor expert_outputs,
    torch::Tensor topk_idx,
    torch::Tensor topk_weights,
    torch::Tensor token_to_slot,
    std::optional<torch::Tensor> shared_out = std::nullopt
) {
    int N = topk_idx.size(0);
    int K = topk_idx.size(1);
    int E = expert_outputs.size(0);
    int C = expert_outputs.size(1);
    int D = expert_outputs.size(2);

    auto routed_out = torch::empty({N, D}, expert_outputs.options());
    if (N > 0) {
        int threads = (D == 768) ? 96 : 128;
        int blocks = N;
        const __nv_bfloat16* sh_ptr = (shared_out.has_value() && shared_out->defined() && shared_out->numel() > 0)
            ? reinterpret_cast<const __nv_bfloat16*>(shared_out->data_ptr<c10::BFloat16>())
            : nullptr;
        moe_combine_forward_kernel<<<blocks, threads>>>(
            reinterpret_cast<const __nv_bfloat16*>(expert_outputs.data_ptr<c10::BFloat16>()),
            topk_idx.data_ptr<int64_t>(),
            reinterpret_cast<const __nv_bfloat16*>(topk_weights.data_ptr<c10::BFloat16>()),
            token_to_slot.data_ptr<int32_t>(),
            sh_ptr,
            reinterpret_cast<__nv_bfloat16*>(routed_out.data_ptr<c10::BFloat16>()),
            N, D, K, E, C
        );
    }
    return routed_out;
}

std::vector<torch::Tensor> moe_combine_backward_cuda(
    torch::Tensor d_routed_out,
    torch::Tensor expert_outputs,
    torch::Tensor topk_idx,
    torch::Tensor topk_weights,
    torch::Tensor token_to_slot
) {
    int N = topk_idx.size(0);
    int K = topk_idx.size(1);
    int E = expert_outputs.size(0);
    int C = expert_outputs.size(1);
    int D = expert_outputs.size(2);

    auto d_expert_outputs = torch::zeros_like(expert_outputs);
    auto d_topk_weights = torch::empty_like(topk_weights);

    if (N > 0 && K > 0) {
        int threads = (D == 768) ? 96 : 128;
        int blocks = N * K;
        moe_combine_backward_kernel<<<blocks, threads>>>(
            reinterpret_cast<const __nv_bfloat16*>(d_routed_out.data_ptr<c10::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(expert_outputs.data_ptr<c10::BFloat16>()),
            topk_idx.data_ptr<int64_t>(),
            reinterpret_cast<const __nv_bfloat16*>(topk_weights.data_ptr<c10::BFloat16>()),
            token_to_slot.data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(d_expert_outputs.data_ptr<c10::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(d_topk_weights.data_ptr<c10::BFloat16>()),
            N, D, K, E, C
        );
    }
    return {d_expert_outputs, d_topk_weights};
}

// ------------------------------------------------------------------------
// 9. Fused QK-RMSNorm (Warp-Specialized for head_dim = 64)
// ------------------------------------------------------------------------

__global__ void qk_rmsnorm_forward_kernel(
    const __nv_bfloat16* __restrict__ x,       // [total_vectors, 64]
    const float* __restrict__ weight,          // [num_heads, 64]
    __nv_bfloat16* __restrict__ y,             // [total_vectors, 64]
    float* __restrict__ rsqrt_out,             // [total_vectors]
    int total_vectors,
    int num_heads,
    int T,
    int head_dim,
    float eps
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= total_vectors) return;

    int lane = threadIdx.x % WARP_SIZE;
    int h = (warp_id / T) % num_heads;

    const __nv_bfloat16* x_vec = x + warp_id * head_dim;
    __nv_bfloat16* y_vec = y + warp_id * head_dim;
    const float* w_vec = weight + h * head_dim;

    float v0 = __bfloat162float(x_vec[lane * 2]);
    float v1 = __bfloat162float(x_vec[lane * 2 + 1]);

    float sum_sq = v0 * v0 + v1 * v1;
    sum_sq = warp_reduce_sum(sum_sq);

    float inv_std = rsqrtf(sum_sq / (float)head_dim + eps);
    inv_std = __shfl_sync(0xffffffff, inv_std, 0);

    if (lane == 0 && rsqrt_out != nullptr) {
        rsqrt_out[warp_id] = inv_std;
    }

    float w0 = w_vec[lane * 2];
    float w1 = w_vec[lane * 2 + 1];

    y_vec[lane * 2] = __float2bfloat16(v0 * inv_std * w0);
    y_vec[lane * 2 + 1] = __float2bfloat16(v1 * inv_std * w1);
}

__global__ void qk_rmsnorm_backward_kernel(
    const __nv_bfloat16* __restrict__ dy,      // [total_vectors, 64]
    const __nv_bfloat16* __restrict__ x,       // [total_vectors, 64]
    const float* __restrict__ weight,          // [num_heads, 64]
    const float* __restrict__ rsqrt_in,        // [total_vectors]
    __nv_bfloat16* __restrict__ dx,            // [total_vectors, 64]
    float* __restrict__ dweight,               // [num_heads, 64]
    int total_vectors,
    int num_heads,
    int T,
    int head_dim
) {
    int lane = threadIdx.x % WARP_SIZE;
    int warp_in_block = threadIdx.x / WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;
    int h = blockIdx.y;
    if (h >= num_heads) return;

    int vectors_per_head = total_vectors / num_heads; // B * T
    int total_grid_warps = gridDim.x * warps_per_block;
    int global_warp_idx = blockIdx.x * warps_per_block + warp_in_block;

    const float* w_vec = weight + h * head_dim;
    float w0 = w_vec[lane * 2];
    float w1 = w_vec[lane * 2 + 1];

    float dw0_acc = 0.0f;
    float dw1_acc = 0.0f;

    for (int tok_idx = global_warp_idx; tok_idx < vectors_per_head; tok_idx += total_grid_warps) {
        int b = tok_idx / T;
        int t = tok_idx % T;
        int vec_idx = (b * num_heads + h) * T + t;

        const __nv_bfloat16* dy_vec = dy + vec_idx * head_dim;
        const __nv_bfloat16* x_vec = x + vec_idx * head_dim;
        __nv_bfloat16* dx_vec = dx + vec_idx * head_dim;
        float rsqrt_val = rsqrt_in[vec_idx];

        float x0 = __bfloat162float(x_vec[lane * 2]);
        float x1 = __bfloat162float(x_vec[lane * 2 + 1]);
        float dy0 = __bfloat162float(dy_vec[lane * 2]);
        float dy1 = __bfloat162float(dy_vec[lane * 2 + 1]);

        dw0_acc += dy0 * (x0 * rsqrt_val);
        dw1_acc += dy1 * (x1 * rsqrt_val);

        float dot = (dy0 * w0 * x0) + (dy1 * w1 * x1);
        dot = warp_reduce_sum(dot);
        dot = __shfl_sync(0xffffffff, dot, 0);

        float rsqrt_cube_dot = (rsqrt_val * rsqrt_val * dot) / (float)head_dim;

        float dx0 = rsqrt_val * (dy0 * w0 - x0 * rsqrt_cube_dot);
        float dx1 = rsqrt_val * (dy1 * w1 - x1 * rsqrt_cube_dot);

        dx_vec[lane * 2] = __float2bfloat16(dx0);
        dx_vec[lane * 2 + 1] = __float2bfloat16(dx1);
    }

    float* dw_vec = dweight + h * head_dim;
    atomicAdd(&dw_vec[lane * 2], dw0_acc);
    atomicAdd(&dw_vec[lane * 2 + 1], dw1_acc);
}

std::vector<torch::Tensor> qk_rmsnorm_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    int64_t T,
    float eps
) {
    int head_dim = x.size(-1);
    int total_vectors = x.numel() / head_dim;
    int num_heads = weight.size(0);

    // The kernel assumes a full warp per vector: 32 lanes x 2 elements = 64. With any
    // other head_dim the upper lanes read and write past the end of the vector, which
    // corrupts neighbouring memory and injects garbage (intermittent NaNs) into the
    // warp reduction. Fail loudly instead of computing silently wrong results.
    TORCH_CHECK(head_dim == 64,
                "qk_rmsnorm kernel requires head_dim == 64, got ", head_dim,
                " (use the PyTorch fallback for other head dims)");
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be a contiguous CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    auto y = torch::empty_like(x);
    auto rsqrt_out = torch::empty({total_vectors}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));

    if (total_vectors > 0) {
        int threads_per_block = 128;
        int warps_per_block = threads_per_block / WARP_SIZE;
        int blocks = (total_vectors + warps_per_block - 1) / warps_per_block;

        qk_rmsnorm_forward_kernel<<<blocks, threads_per_block>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<c10::BFloat16>()),
            weight.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(y.data_ptr<c10::BFloat16>()),
            rsqrt_out.data_ptr<float>(),
            total_vectors,
            num_heads,
            (int)T,
            head_dim,
            eps
        );
    }
    return {y, rsqrt_out};
}

std::vector<torch::Tensor> qk_rmsnorm_backward_cuda(
    torch::Tensor dy,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor rsqrt_in,
    int64_t T
) {
    int head_dim = x.size(-1);
    int total_vectors = x.numel() / head_dim;
    int num_heads = weight.size(0);

    auto dx = torch::empty_like(x);
    auto dweight = torch::zeros_like(weight);

    if (total_vectors > 0) {
        int threads_per_block = 128;
        int warps_per_block = threads_per_block / WARP_SIZE;
        int vectors_per_head = total_vectors / num_heads;
        int blocks_x = std::min((vectors_per_head + warps_per_block - 1) / warps_per_block, 32);
        dim3 grid(blocks_x, num_heads);

        qk_rmsnorm_backward_kernel<<<grid, threads_per_block>>>(
            reinterpret_cast<const __nv_bfloat16*>(dy.data_ptr<c10::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<c10::BFloat16>()),
            weight.data_ptr<float>(),
            rsqrt_in.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(dx.data_ptr<c10::BFloat16>()),
            dweight.data_ptr<float>(),
            total_vectors,
            num_heads,
            (int)T,
            head_dim
        );
    }
    return {dx, dweight};
}

// ------------------------------------------------------------------------
// 10. Fused Vectorized Logit Soft-Capped Cross-Entropy Kernel
// ------------------------------------------------------------------------

__global__ void fused_softcap_ce_forward_vec8_kernel(
    const __nv_bfloat16* __restrict__ logits, // [N, V]
    const int64_t* __restrict__ targets,     // [N]
    float* __restrict__ losses,              // [N]
    float* __restrict__ lse_out,             // [N]
    int N, int V, float softcap, int64_t ignore_index
) {
    int token_idx = blockIdx.x;
    if (token_idx >= N) return;

    int64_t target = targets[token_idx];
    if (target == ignore_index) {
        if (threadIdx.x == 0) {
            losses[token_idx] = 0.0f;
            lse_out[token_idx] = 0.0f;
        }
        return;
    }

    const __nv_bfloat16* x = logits + token_idx * V;
    float inv_softcap = 1.0f / softcap;
    int num_vec = V / 8;

    // Pass 1: Vectorized 128-bit load & max reduction
    float thread_max = -1e20f;
    for (int v = threadIdx.x; v < num_vec; v += blockDim.x) {
        uint4 in_u4 = *reinterpret_cast<const uint4*>(x + v * 8);
        const __nv_bfloat16* bf = reinterpret_cast<const __nv_bfloat16*>(&in_u4);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = __bfloat162float(bf[k]);
            float sc_val = softcap * tanhf(val * inv_softcap);
            thread_max = fmaxf(thread_max, sc_val);
        }
    }

    thread_max = warp_reduce_max(thread_max);
    __shared__ float s_max[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    if (lane == 0) s_max[wid] = thread_max;
    __syncthreads();

    if (wid == 0) {
        float m = (lane < blockDim.x / WARP_SIZE) ? s_max[lane] : -1e20f;
        m = warp_reduce_max(m);
        if (lane == 0) s_max[0] = m;
    }
    __syncthreads();
    float max_val = s_max[0];

    // Pass 2: Vectorized 128-bit load & sum exp
    float thread_sum = 0.0f;
    for (int v = threadIdx.x; v < num_vec; v += blockDim.x) {
        uint4 in_u4 = *reinterpret_cast<const uint4*>(x + v * 8);
        const __nv_bfloat16* bf = reinterpret_cast<const __nv_bfloat16*>(&in_u4);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = __bfloat162float(bf[k]);
            float sc_val = softcap * tanhf(val * inv_softcap);
            thread_sum += expf(sc_val - max_val);
        }
    }

    thread_sum = warp_reduce_sum(thread_sum);
    __shared__ float s_sum[32];
    if (lane == 0) s_sum[wid] = thread_sum;
    __syncthreads();

    if (wid == 0) {
        float s = (lane < blockDim.x / WARP_SIZE) ? s_sum[lane] : 0.0f;
        s = warp_reduce_sum(s);
        if (lane == 0) s_sum[0] = s;
    }
    __syncthreads();
    float sum_exp = s_sum[0];

    if (threadIdx.x == 0) {
        float lse = max_val + logf(fmaxf(sum_exp, 1e-20f));
        lse_out[token_idx] = lse;

        float target_loss = 0.0f;
        if (target >= 0 && target < V) {
            float t_val = __bfloat162float(x[target]);
            float sc_target = softcap * tanhf(t_val * inv_softcap);
            target_loss = lse - sc_target;
        }
        losses[token_idx] = target_loss;
    }
}

__global__ void fused_softcap_ce_backward_vec8_kernel(
    const __nv_bfloat16* __restrict__ logits, // [N, V]
    const int64_t* __restrict__ targets,     // [N]
    const float* __restrict__ lse,           // [N]
    __nv_bfloat16* __restrict__ d_logits,    // [N, V]
    int N, int V, float softcap, float grad_scale, int64_t ignore_index
) {
    int token_idx = blockIdx.x;
    if (token_idx >= N) return;

    int64_t target = targets[token_idx];
    const __nv_bfloat16* x = logits + token_idx * V;
    __nv_bfloat16* dx = d_logits + token_idx * V;
    int num_vec = V / 8;

    if (target == ignore_index) {
        for (int v = threadIdx.x; v < num_vec; v += blockDim.x) {
            *reinterpret_cast<uint4*>(dx + v * 8) = make_uint4(0, 0, 0, 0);
        }
        return;
    }

    float token_lse = lse[token_idx];
    float inv_softcap = 1.0f / softcap;

    for (int v = threadIdx.x; v < num_vec; v += blockDim.x) {
        uint4 in_u4 = *reinterpret_cast<const uint4*>(x + v * 8);
        const __nv_bfloat16* bf = reinterpret_cast<const __nv_bfloat16*>(&in_u4);
        __nv_bfloat16 out_bf[8];

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            int idx = v * 8 + k;
            float val = __bfloat162float(bf[k]);
            float tanh_val = tanhf(val * inv_softcap);
            float sc_val = softcap * tanh_val;
            float prob = expf(sc_val - token_lse);
            float indicator = (idx == target) ? 1.0f : 0.0f;

            float d_softcap = 1.0f - tanh_val * tanh_val;
            float grad = (prob - indicator) * d_softcap * grad_scale;
            out_bf[k] = __float2bfloat16(grad);
        }
        *reinterpret_cast<uint4*>(dx + v * 8) = *reinterpret_cast<const uint4*>(out_bf);
    }
}

std::vector<torch::Tensor> fused_softcap_ce_forward_cuda(
    torch::Tensor logits,
    torch::Tensor targets,
    float softcap,
    int64_t ignore_index
) {
    int N = logits.size(0);
    int V = logits.size(1);
    auto losses = torch::empty({N}, torch::TensorOptions().dtype(torch::kFloat32).device(logits.device()));
    auto lse = torch::empty({N}, torch::TensorOptions().dtype(torch::kFloat32).device(logits.device()));

    int threads = 512;
    int blocks = N;
    fused_softcap_ce_forward_vec8_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr<c10::BFloat16>()),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        lse.data_ptr<float>(),
        N, V, softcap, ignore_index
    );
    return {losses, lse};
}

torch::Tensor fused_softcap_ce_backward_cuda(
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor lse,
    float softcap,
    float grad_scale,
    int64_t ignore_index
) {
    int N = logits.size(0);
    int V = logits.size(1);
    auto d_logits = torch::empty_like(logits);

    int threads = 512;
    int blocks = N;
    fused_softcap_ce_backward_vec8_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr<c10::BFloat16>()),
        targets.data_ptr<int64_t>(),
        lse.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(d_logits.data_ptr<c10::BFloat16>()),
        N, V, softcap, grad_scale, ignore_index
    );
    return d_logits;
}

// ------------------------------------------------------------------------
// 11. Fused MoE Router Top-2 + Sigmoid + Z-Loss (E = 8, K = 2)
// ------------------------------------------------------------------------

__device__ __forceinline__ float load_logit(const __nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float load_logit(const float v) { return v; }

template <typename logit_t> __device__ __forceinline__ logit_t store_logit(float v);
template <> __device__ __forceinline__ __nv_bfloat16 store_logit<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
template <> __device__ __forceinline__ float store_logit<float>(float v) { return v; }

template <typename logit_t>
__global__ void moe_router_topk_forward_kernel(
    const logit_t* __restrict__ logits,            // [N, 8], fp32 or bf16
    const float* __restrict__ expert_bias,         // [8]
    int64_t* __restrict__ topk_idx,                // [N, 2]
    __nv_bfloat16* __restrict__ topk_weights,      // [N, 2]
    float* __restrict__ z_loss_out,                // [N]
    float z_loss_weight,
    int N
) {
    int tok = blockIdx.x * blockDim.x + threadIdx.x;
    if (tok >= N) return;

    const logit_t* l_tok = logits + tok * 8;
    float l[8];
    float s[8];
    float b[8];
    float max_l = -1e20f;

    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        float val = load_logit(l_tok[e]);
        l[e] = val;
        max_l = fmaxf(max_l, val);
        float sig = 1.0f / (1.0f + __expf(-val));
        s[e] = sig;
        b[e] = sig + expert_bias[e];
    }

    // Top-2 selection in registers
    int e0 = 0, e1 = 1;
    float max0 = b[0], max1 = b[1];
    if (max1 > max0) {
        max0 = b[1]; max1 = b[0];
        e0 = 1; e1 = 0;
    }
    #pragma unroll
    for (int e = 2; e < 8; ++e) {
        float val = b[e];
        if (val > max0) {
            max1 = max0;
            e1 = e0;
            max0 = val;
            e0 = e;
        } else if (val > max1) {
            max1 = val;
            e1 = e;
        }
    }

    // Unbiased normalized Top-2 gating weights
    float u0 = s[e0];
    float u1 = s[e1];
    float sum_u = u0 + u1 + 1e-6f;
    float w0 = u0 / sum_u;
    float w1 = u1 / sum_u;

    topk_idx[tok * 2] = e0;
    topk_idx[tok * 2 + 1] = e1;
    topk_weights[tok * 2] = __float2bfloat16(w0);
    topk_weights[tok * 2 + 1] = __float2bfloat16(w1);

    // Row LogSumExp for z-loss
    float sum_exp = 0.0f;
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        sum_exp += __expf(l[e] - max_l);
    }
    float lse = max_l + logf(fmaxf(sum_exp, 1e-20f));
    z_loss_out[tok] = z_loss_weight * (lse * lse);
}

template <typename logit_t>
__global__ void moe_router_topk_backward_kernel(
    const __nv_bfloat16* __restrict__ d_topk_weights, // [N, 2]
    const logit_t* __restrict__ logits,               // [N, 8]
    const int64_t* __restrict__ topk_idx,             // [N, 2]
    logit_t* __restrict__ d_logits,                   // [N, 8]
    float z_loss_weight,
    int N
) {
    int tok = blockIdx.x * blockDim.x + threadIdx.x;
    if (tok >= N) return;

    const logit_t* l_tok = logits + tok * 8;
    logit_t* dl_tok = d_logits + tok * 8;

    float l[8];
    float s[8];
    float max_l = -1e20f;
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        float val = load_logit(l_tok[e]);
        l[e] = val;
        max_l = fmaxf(max_l, val);
        s[e] = 1.0f / (1.0f + __expf(-val));
    }

    // 1. Z-loss gradient
    float sum_exp = 0.0f;
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        sum_exp += __expf(l[e] - max_l);
    }
    float inv_sum_exp = 1.0f / fmaxf(sum_exp, 1e-20f);
    float lse = max_l + logf(fmaxf(sum_exp, 1e-20f));
    float z_factor = (2.0f * z_loss_weight / (float)N) * lse;

    int e0 = (int)topk_idx[tok * 2];
    int e1 = (int)topk_idx[tok * 2 + 1];

    float u0 = s[e0];
    float u1 = s[e1];
    float D = u0 + u1 + 1e-6f;
    float inv_D2 = 1.0f / (D * D);

    float d0 = __bfloat162float(d_topk_weights[tok * 2]);
    float d1 = __bfloat162float(d_topk_weights[tok * 2 + 1]);

    float du0 = (d0 - d1) * u1 * inv_D2 * u0 * (1.0f - u0);
    float du1 = (d1 - d0) * u0 * inv_D2 * u1 * (1.0f - u1);

    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        float p_e = __expf(l[e] - max_l) * inv_sum_exp;
        float g = z_factor * p_e;
        if (e == e0) g += du0;
        if (e == e1) g += du1;
        dl_tok[e] = store_logit<logit_t>(g);
    }
}

std::vector<torch::Tensor> moe_router_topk_forward_cuda(
    torch::Tensor logits,
    torch::Tensor expert_bias,
    float z_loss_weight
) {
    int N = logits.size(0);
    auto topk_idx = torch::empty({N, 2}, torch::TensorOptions().dtype(torch::kInt64).device(logits.device()));
    // Weights are always bf16 for the combine kernel, regardless of logits dtype.
    auto topk_weights = torch::empty(
        {N, 2}, torch::TensorOptions().dtype(torch::kBFloat16).device(logits.device()));
    auto z_loss_per_token = torch::empty({N}, torch::TensorOptions().dtype(torch::kFloat32).device(logits.device()));

    if (N > 0) {
        int threads = 128;
        int blocks = (N + threads - 1) / threads;
        auto* tkw = reinterpret_cast<__nv_bfloat16*>(topk_weights.data_ptr<c10::BFloat16>());
        if (logits.scalar_type() == torch::kFloat32) {
            moe_router_topk_forward_kernel<float><<<blocks, threads>>>(
                logits.data_ptr<float>(), expert_bias.data_ptr<float>(),
                topk_idx.data_ptr<int64_t>(), tkw,
                z_loss_per_token.data_ptr<float>(), z_loss_weight, N);
        } else {
            moe_router_topk_forward_kernel<__nv_bfloat16><<<blocks, threads>>>(
                reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr<c10::BFloat16>()),
                expert_bias.data_ptr<float>(),
                topk_idx.data_ptr<int64_t>(), tkw,
                z_loss_per_token.data_ptr<float>(), z_loss_weight, N);
        }
    }
    auto z_loss = z_loss_per_token.sum() / (float)std::max(1, N);
    return {topk_idx, topk_weights, z_loss};
}

torch::Tensor moe_router_topk_backward_cuda(
    torch::Tensor d_topk_weights,
    torch::Tensor logits,
    torch::Tensor topk_idx,
    float z_loss_weight
) {
    int N = logits.size(0);
    auto d_logits = torch::empty_like(logits);
    if (N > 0) {
        int threads = 128;
        int blocks = (N + threads - 1) / threads;
        auto* dtw = reinterpret_cast<const __nv_bfloat16*>(d_topk_weights.data_ptr<c10::BFloat16>());
        if (logits.scalar_type() == torch::kFloat32) {
            moe_router_topk_backward_kernel<float><<<blocks, threads>>>(
                dtw, logits.data_ptr<float>(), topk_idx.data_ptr<int64_t>(),
                d_logits.data_ptr<float>(), z_loss_weight, N);
        } else {
            moe_router_topk_backward_kernel<__nv_bfloat16><<<blocks, threads>>>(
                dtw,
                reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr<c10::BFloat16>()),
                topk_idx.data_ptr<int64_t>(),
                reinterpret_cast<__nv_bfloat16*>(d_logits.data_ptr<c10::BFloat16>()),
                z_loss_weight, N);
        }
    }
    return d_logits;
}
