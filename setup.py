import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME

setup(
    name="supersonic",
    ext_modules=[
        CUDAExtension(
            name="supersonic",
            sources=[
                "src/Bindings.cpp",
                "src/Environment.cu",
            ],
            include_dirs=["include"],
            libraries=["cudart"],
            library_dirs=[os.path.join(CUDA_HOME, "lib64")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--use_fast_math", "-std=c++17"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
