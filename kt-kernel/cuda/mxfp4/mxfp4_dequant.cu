/*
 * @Description  :  MXFP4 dequantization kernel for SM_86 (Ampere) — forward pass.
 *
 * MXFP4 layout (per block):
 *   • 1 × E8M0 scale byte shared by all FP4 elements in the block
 *   • N packed FP4 data bytes (2 FP4 elements per byte, upper/lower nibble)
 *
 * Dequant:  value = fp4_decode(nibble) * scale
 *
 * Author    : AlphaHENG
 * Date      : 2026-05-07
 * Copyright : (c) 2026 by AlphaHENG, All Rights Reserved.
 *
 * SM_86 note: this kernel is compatible with any SM >= 70 architecture.
 * It is compiled with `CMAKE_CUDA_ARCHITECTURES` including 86.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cstdint>
#include <c10/cuda/CUDAGuard.h>

#ifdef __HIP_PLATFORM_AMD__
typedef hip_bfloat16 nv_bfloat16;
#endif

/* ------------------------------------------------------------------ */
/*  FP4 value table — OCP MXFP4 E2M1 format (3-bit effective)         */
/*  Index order: 000, 001, 010, 011, 100, 101, 110, 111             */
/*  Maps to:     -1, -0.5, -0, 0, +0, +0.5, +1  (nan → +1 clamp)     */
/* ------------------------------------------------------------------ */
static __constant__ float __fp4_lut[16] = {
    0.0f,  0.5f,  1.0f, -1.0f,   // nibbles 0-3
   -0.5f, -1.0f,  0.5f,  1.0f,   // nibbles 4-7
    0.0f,  0.5f,  1.0f, -1.0f,   // nibbles 8-B  (duplicate for sign-bit symmetry)
   -0.5f, -1.0f,  0.5f,  1.0f,   // nibbles C-F
};

/* Helper: decode a single 4-bit FP4 value to float. */
__device__ __forceinline__ float fp4_to_float(uint8_t nibble) {
    return __fp4_lut[nibble & 0x7];   // 3-bit effective index
}

/* ------------------------------------------------------------------ */
/*  FP32 dequantization kernel                                        */
/* ------------------------------------------------------------------ */
__global__ void dequantize_mxfp4_fp32_kernel(const int8_t* data, float* output,
                                             const int blk_size, const int ele_per_blk,
                                             const int num_blocks) {
    long long global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (long long block_id = global_idx; block_id < num_blocks; block_id += blockDim.x * gridDim.x) {
        const int8_t* cur_block = data + block_id * blk_size;

        /* Scale byte: E8M0 — raw exponent, no bias. Interpret as signed magnitude. */
        uint8_t scale_byte = static_cast<uint8_t>(cur_block[0]);
        int scale_sign = (scale_byte >> 7) & 0x01;
        int scale_exp  = scale_byte & 0x7F;
        float scale_f  = scale_sign ? -static_cast<float>(scale_exp) : static_cast<float>(scale_exp);

        const uint8_t* data_bytes = reinterpret_cast<const uint8_t*>(cur_block + 1);

        float* out_ptr = output + block_id * ele_per_blk;
        for (int i = 0; i < ele_per_blk; i += 2) {
            uint8_t packed = data_bytes[i / 2];
            out_ptr[i]     = fp4_to_float(packed & 0x0F) * scale_f;
            out_ptr[i + 1] = fp4_to_float((packed >> 4) & 0x0F) * scale_f;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  FP16 dequantization kernel                                        */
/* ------------------------------------------------------------------ */
__global__ void dequantize_mxfp4_fp16_kernel(const int8_t* data, __half* output,
                                             const int blk_size, const int ele_per_blk,
                                             const int num_blocks) {
    long long global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (long long block_id = global_idx; block_id < num_blocks; block_id += blockDim.x * gridDim.x) {
        const int8_t* cur_block = data + block_id * blk_size;

        uint8_t scale_byte = static_cast<uint8_t>(cur_block[0]);
        int scale_sign = (scale_byte >> 7) & 0x01;
        int scale_exp  = scale_byte & 0x7F;
        float scale_f  = scale_sign ? -static_cast<float>(scale_exp) : static_cast<float>(scale_exp);

        const uint8_t* data_bytes = reinterpret_cast<const uint8_t*>(cur_block + 1);

        __half* out_ptr = output + block_id * ele_per_blk;
        for (int i = 0; i < ele_per_blk; i += 2) {
            uint8_t packed = data_bytes[i / 2];
            out_ptr[i]     = __float2half(fp4_to_float(packed & 0x0F) * scale_f);
            out_ptr[i + 1] = __float2half(fp4_to_float((packed >> 4) & 0x0F) * scale_f);
        }
    }
}

/* ------------------------------------------------------------------ */
/*  BF16 dequantization kernel                                        */
/* ------------------------------------------------------------------ */
__global__ void dequantize_mxfp4_bf16_kernel(const int8_t* data, nv_bfloat16* output,
                                             const int blk_size, const int ele_per_blk,
                                             const int num_blocks) {
    long long global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (long long block_id = global_idx; block_id < num_blocks; block_id += blockDim.x * gridDim.x) {
        const int8_t* cur_block = data + block_id * blk_size;

        uint8_t scale_byte = static_cast<uint8_t>(cur_block[0]);
        int scale_sign = (scale_byte >> 7) & 0x01;
        int scale_exp  = scale_byte & 0x7F;
        float scale_f  = scale_sign ? -static_cast<float>(scale_exp) : static_cast<float>(scale_exp);

        const uint8_t* data_bytes = reinterpret_cast<const uint8_t*>(cur_block + 1);

        nv_bfloat16* out_ptr = output + block_id * ele_per_blk;
        for (int i = 0; i < ele_per_blk; i += 2) {
            uint8_t packed = data_bytes[i / 2];
            out_ptr[i]     = __float2bfloat16(fp4_to_float(packed & 0x0F) * scale_f);
            out_ptr[i + 1] = __float2bfloat16(fp4_to_float((packed >> 4) & 0x0F) * scale_f);
        }
    }
}

/* ================================================================== */
/*  Host-side launcher                                                 */
/* ================================================================== */
torch::Tensor dequantize_mxfp4(const int8_t* data, const int num_bytes, const int blk_size,
                               const int ele_per_blk, const torch::Device device,
                               const torch::Dtype target_dtype) {
    TORCH_CHECK(blk_size > 1, "blk_size must be >= 2 (1 scale byte + data bytes)");

    int num_blocks = num_bytes / blk_size;
    TORCH_CHECK(num_blocks > 0, "num_bytes < blk_size: no complete blocks to dequantize");

    const at::cuda::OptionalCUDAGuard device_guard(device);

    // Copy data to device
    int8_t* data_device = nullptr;
    cudaMalloc(&data_device, num_bytes * sizeof(int8_t));
    cudaMemcpy(data_device, data, num_bytes * sizeof(int8_t), cudaMemcpyHostToDevice);

    // Create output tensor on target device
    auto output = torch::zeros({num_blocks, ele_per_blk}, torch::dtype(target_dtype).device(device));

    // Launch config
    constexpr int thread_per_block = 256;
    int num_blocks_grid = min(512, (num_blocks + thread_per_block - 1) / thread_per_block);
    if (num_blocks_grid == 0) num_blocks_grid = 1;

    switch (target_dtype) {
        case torch::kFloat16:
            dequantize_mxfp4_fp16_kernel<<<num_blocks_grid, thread_per_block>>>(
                data_device, reinterpret_cast<__half*>(output.data_ptr()), blk_size, ele_per_blk, num_blocks);
            break;
        case torch::kBFloat16:
            dequantize_mxfp4_bf16_kernel<<<num_blocks_grid, thread_per_block>>>(
                data_device, reinterpret_cast<nv_bfloat16*>(output.data_ptr()), blk_size, ele_per_blk, num_blocks);
            break;
        case torch::kFloat32:
            dequantize_mxfp4_fp32_kernel<<<num_blocks_grid, thread_per_block>>>(
                data_device, output.data_ptr<float>(), blk_size, ele_per_blk, num_blocks);
            break;
        default:
            TORCH_CHECK(false, "Unsupported output dtype — use torch.float16, bfloat16, or float32");
    }

    cudaDeviceSynchronize();
    cudaFree(data_device);
    return output;
}
