import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Read target architecture or default to local RTX 4060 (sm_89)
arch = os.environ.get("PULSAR_CUDA_ARCH", "sm_89")

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
                "nvcc": [
                    "-O3",
                    "-std=c++20",
                    f"-arch={arch}",
                    "--expt-relaxed-constexpr",
                    "-use_fast_math",
                    "-Xptxas=-O3",
                    "--threads=1",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
