#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CUDA extension setup for KTransformersOps with SM_86 (Ampere) MXFP4 support.
"""
from setuptools import setup, Extension
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='KTransformersOps',
    ext_modules=[
        CUDAExtension(
            'KTransformersOps', [
                'custom_gguf/dequant.cu',
                'binding.cpp',
                'gptq_marlin/gptq_marlin.cu',
                'moe/moe_topk_softmax_kernels.cu',
                'mxfp4/mxfp4_dequant.cu',  # MXFP4 dequantization kernel for SM_86
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-Xcompiler', '-fPIC',
                    # SM_86 (Ampere) architecture flag - required for MXFP4 training kernels
                    '-gencode', 'arch=compute_86,code=sm_86',
                ]
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
