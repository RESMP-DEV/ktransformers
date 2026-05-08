/**
 * @Description  : MXFP4 (Micro-scale Float 4-bit) kernel declarations for SM_86 Ampere.
 * @Author       : AlphaHENG
 * @Date         : 2026-05-07
 * @Copyright (c) 2026 by AlphaHENG, All Rights Reserved.
 **/
#pragma once

#include <torch/extension.h>
#include <torch/library.h>
#include <torch/torch.h>

/// Dequantize an MXFP4 tensor to the specified floating-point dtype.
///
/// Parameters
/// ----------
/// data          : int8_t host pointer to MXFP4 packed bytes
/// num_bytes     : total number of packed bytes (host-side)
/// blk_size      : number of bytes per quantized block (e.g. 17 = 1 scale byte + 16 data bytes)
/// ele_per_blk   : number of dequantized elements per block (e.g. 32, since each data byte holds 2 FP4 values)
/// device        : target torch::Device (must be CUDA)
/// target_dtype  : output dtype (kFloat16, kBFloat16, kFloat32)
///
/// Returns
/// -------
/// torch::Tensor of shape [num_blocks, ele_per_blk] in target_dtype on ``device``.
torch::Tensor dequantize_mxfp4(const int8_t* data, const int num_bytes, const int blk_size,
                               const int ele_per_blk, const torch::Device device,
                               const torch::Dtype target_dtype);
