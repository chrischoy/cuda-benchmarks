from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="implicit_gemm_benchmark",
    ext_modules=[
        CUDAExtension(
            name="implicit_gemm_benchmark",
            sources=["matmul.cu", "matmul_wmma_sm80.cu"],
            include_dirs=[
                "../../include",
                "../../3rdparty/cccl",
            ],
            extra_compile_args={
                "nvcc": [
                    "-O2",
                    "--expt-relaxed-constexpr",
                    "--extended-lambda",
                    "-gencode=arch=compute_80,code=sm_80",
                ],
                "cxx": ["-O2"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
