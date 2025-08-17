from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="implicit_gemm_benchmark",
    ext_modules=[
        CUDAExtension(
            name="implicit_gemm_benchmark",
            sources=["matmul.cu"],
            include_dirs=["../../include"],
            extra_compile_args={
                "nvcc": ["-O2", "--expt-relaxed-constexpr", "--extended-lambda"],
                "cxx": ["-O2"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
