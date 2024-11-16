from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = [
    CUDAExtension(
        "torch_ops_matmul", [
            "src/matmul_kernel.cu",
            "src/matmul.cpp"
        ],
        include_dirs=[
            '/root/torch_ops_matmul/include'
        ]
    ) 
]

setup(
    name="torch_ops_matmul",
    version="1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False
)
