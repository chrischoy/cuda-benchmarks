from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="matrix_memory_benchmark",
    ext_modules=[
        CUDAExtension(
            name="matrix_load_benchmark",
            sources=["load.cu"],
            include_dirs=["../../include"],
            extra_compile_args={
                "nvcc": ["-O2", "--expt-relaxed-constexpr", "--extended-lambda"],
                "cxx": ["-O2"],
            },
        ),
        CUDAExtension(
            name="matrix_store_benchmark",
            sources=["store.cu"],
            include_dirs=["../../include"],
            extra_compile_args={
                "nvcc": ["-O2", "--expt-relaxed-constexpr", "--extended-lambda"],
                "cxx": ["-O2"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
