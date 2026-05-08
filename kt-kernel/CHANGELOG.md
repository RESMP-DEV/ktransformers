# Changelog

All notable changes to the kt-kernel project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to semantic versioning principles.

## [Unreleased]

### Added
- **MXFP4 CUDA Kernel for SM_86 (Ampere)**: Implemented micro-scale float4 (MXFP4) dequantization kernel targeting Ampere SM_86 architecture.
  - Added `cuda/mxfp4/mxfp4_dequant.cu` with FP32/FP16/BF16 output support
  - Added `cuda/mxfp4/ops.h` header with function declarations
  - Added Python binding `dequantize_mxfp4` in `cuda/binding.cpp`
  - Updated `cuda/setup.py` with SM_86 architecture flag (`-gencode arch=compute_86,code=sm_86`)
  - Added test module `test/test_mxfp4_dequant.py` with CUDA availability skip logic
  - MXFP4 format: 1 E8M0 scale byte + packed FP4 data (2 values per byte)
  - Dequantization formula: `value = fp4_decode(nibble) * scale`

### Changed
- `cuda/binding.cpp`: Added MXFP4 module include and binding
- AVX2 MXFP4 DQ scale conversion now uses small UE8M0 lookup tables for FP32
  and BF16 bit patterns instead of recomputing the bit shifts at each use.

### Build Configuration
- Default CUDA architectures include SM_86: `80;86;89;90` (CMakeLists.txt)
- NVCC flags: `-O3 --use_fast_math -Xcompiler -fPIC -gencode arch=compute_86,code=sm_86`

---
*Generated: 2026-05-07*
