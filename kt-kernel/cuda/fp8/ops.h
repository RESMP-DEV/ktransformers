/**
 * @Description  : FP8 E4M3 linear kernel declarations for SM_86 Ampere.
 * @Author       : AlphaHENG
 * @Date         : 2026-05-09
 * @Copyright (c) 2026 by AlphaHENG, All Rights Reserved.
 **/
#pragma once

#include <torch/extension.h>
#include <torch/library.h>
#include <torch/torch.h>

/// Decode packed FP8 E4M3FN weights with 128x128 block scales and multiply by
/// a CUDA activation matrix.
///
/// x            : CUDA tensor [M, K], dtype fp32/fp16/bf16
/// weight_bytes : CUDA uint8 tensor [N, K], one FP8 value per byte
/// scales       : CUDA float32 tensor [ceil(N/128), ceil(K/128)]
///
/// Returns a CUDA tensor [M, N] in x.dtype.
torch::Tensor fp8_linear(torch::Tensor x, torch::Tensor weight_bytes, torch::Tensor scales);
