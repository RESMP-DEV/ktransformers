/**
 * @Description  : W8A8 INT8 linear kernel declarations for SM_86 Ampere.
 * @Author       : AlphaHENG
 * @Date         : 2026-05-11
 * @Copyright (c) 2026 by AlphaHENG, All Rights Reserved.
 **/
#pragma once

#include <torch/extension.h>
#include <torch/library.h>
#include <torch/torch.h>

/// Quantize activations per row, multiply by pre-quantized INT8 weights, and
/// dequantize with per-row activation scales and per-output weight scales.
///
/// x            : CUDA tensor [M, K], dtype fp32/fp16/bf16
/// qweight      : CUDA int8 tensor [N, K]
/// weight_scale : CUDA float32 tensor [N]
/// smooth_scale : CUDA float32 tensor [K] or empty; x is divided by this before quantization
///
/// Returns a CUDA tensor [M, N] in x.dtype.
torch::Tensor int8_smoothquant_linear(torch::Tensor x, torch::Tensor qweight,
                                      torch::Tensor weight_scale, torch::Tensor smooth_scale);
