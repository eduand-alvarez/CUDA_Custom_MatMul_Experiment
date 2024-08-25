# setup.py

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='matrix_mul_extension',
    ext_modules=[
        CUDAExtension('matrix_mul_extension', [
            'src/matrix_mul_extension.cpp',  # C++ code
            'src/matrix_mul.cu',  # CUDA code
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
