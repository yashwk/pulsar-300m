import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Read target architecture or default to local RTX 4060 (sm_89)
extra_nvcc_args = [
    "-O3",
    "-std=c++20",
    "--expt-relaxed-constexpr",
    "-use_fast_math",
    "-Xptxas=-O3",
    "--threads=1",
]
if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    arch = os.environ.get("PULSAR_CUDA_ARCH", "sm_89")
    extra_nvcc_args.append(f"-arch={arch}")
else:
    # Filter unsupported architectures to avoid ValueError: Unknown CUDA arch (10.0)
    import torch.utils.cpp_extension as _ext
    candidate_archs = [a.strip() for a in os.environ["TORCH_CUDA_ARCH_LIST"].split(";") if a.strip()]
    valid_archs = []
    for a in candidate_archs:
        base_a = a.split("+")[0]
        try:
            _ext._get_cuda_arch_flags([base_a])
            valid_archs.append(a)
        except Exception:
            # Fall back to 9.0+PTX so Blackwell JIT compiles from Hopper PTX
            if "9.0+PTX" not in valid_archs:
                valid_archs.append("9.0+PTX")
    if valid_archs:
        os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(valid_archs)

setup(
    name="pulsar_cuda",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="pulsar_cuda",
            sources=[
                "csrc/pulsar_cuda.cpp",
                "csrc/kernels.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20"],
                "nvcc": extra_nvcc_args,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

