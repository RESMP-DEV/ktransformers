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
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cstdint>
#include <limits>
#include <c10/cuda/CUDAException.h>
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

static __constant__ float __fp4_e2m1_lut[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,
    2.0f,  3.0f,  4.0f,  6.0f,
    0.0f, -0.5f, -1.0f, -1.5f,
   -2.0f, -3.0f, -4.0f, -6.0f,
};

/* Helper: decode a single 4-bit FP4 value to float. */
__device__ __forceinline__ float fp4_to_float(uint8_t nibble) {
    return __fp4_lut[nibble & 0x7];   // 3-bit effective index
}

__device__ __forceinline__ float fp4_e2m1_to_float(uint8_t nibble) {
    return __fp4_e2m1_lut[nibble & 0x0F];
}

__device__ __forceinline__ float load_as_float(const float* ptr, int idx) {
    return ptr[idx];
}

__device__ __forceinline__ float load_as_float(const __half* ptr, int idx) {
    return __half2float(ptr[idx]);
}

__device__ __forceinline__ float load_as_float(const nv_bfloat16* ptr, int idx) {
    return __bfloat162float(ptr[idx]);
}

__device__ __forceinline__ void store_from_float(float* ptr, int idx, float value) {
    ptr[idx] = value;
}

__device__ __forceinline__ void store_from_float(__half* ptr, int idx, float value) {
    ptr[idx] = __float2half(value);
}

__device__ __forceinline__ void store_from_float(nv_bfloat16* ptr, int idx, float value) {
    ptr[idx] = __float2bfloat16(value);
}

template <typename scalar_t>
__global__ void mxfp4_linear_kernel(const scalar_t* __restrict__ x,
                                    const uint8_t* __restrict__ weight,
                                    const float* __restrict__ scales,
                                    scalar_t* __restrict__ out,
                                    int M,
                                    int N,
                                    int K,
                                    int packed_K,
                                    int scale_K) {
    extern __shared__ float partial[];
    const int output_idx = blockIdx.x;
    const int m = output_idx / N;
    const int n = output_idx - m * N;
    float acc = 0.0f;

    for (int packed_k = threadIdx.x; packed_k < packed_K; packed_k += blockDim.x) {
        const uint8_t packed = weight[n * packed_K + packed_k];
        const int k0 = packed_k * 2;
        const int scale_idx = n * scale_K + packed_k / 16;
        const float scale = scales[scale_idx];
        const float w0 = fp4_e2m1_to_float(packed & 0x0F) * scale;
        const float w1 = fp4_e2m1_to_float((packed >> 4) & 0x0F) * scale;
        acc += load_as_float(x, m * K + k0) * w0;
        if (k0 + 1 < K) {
            acc += load_as_float(x, m * K + k0 + 1) * w1;
        }
    }

    partial[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        store_from_float(out, output_idx, partial[0]);
    }
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

torch::Tensor mxfp4_linear(torch::Tensor x, torch::Tensor weight_bytes, torch::Tensor scales) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight_bytes.is_cuda(), "weight_bytes must be a CUDA tensor");
    TORCH_CHECK(scales.is_cuda(), "scales must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be [M, K]");
    TORCH_CHECK(weight_bytes.dim() == 2, "weight_bytes must be [N, K/2]");
    TORCH_CHECK(scales.dim() == 2, "scales must be [N, K/32]");
    TORCH_CHECK(weight_bytes.scalar_type() == torch::kUInt8, "weight_bytes must be torch.uint8");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must be torch.float32");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16 ||
                    x.scalar_type() == torch::kBFloat16,
                "x dtype must be float32, float16, or bfloat16");

    x = x.contiguous();
    weight_bytes = weight_bytes.contiguous();
    scales = scales.contiguous();

    const int64_t M64 = x.size(0);
    const int64_t K64 = x.size(1);
    const int64_t N64 = weight_bytes.size(0);
    const int64_t packed_K64 = weight_bytes.size(1);
    const int64_t scale_K64 = scales.size(1);
    TORCH_CHECK(K64 == packed_K64 * 2, "x K must equal weight_bytes.size(1) * 2");
    TORCH_CHECK(scales.size(0) == N64, "scales N must match weight N");
    TORCH_CHECK(scale_K64 == (K64 + 31) / 32, "scales K must equal ceil(K / 32)");
    TORCH_CHECK(M64 <= std::numeric_limits<int>::max(), "M too large");
    TORCH_CHECK(N64 <= std::numeric_limits<int>::max(), "N too large");
    TORCH_CHECK(K64 <= std::numeric_limits<int>::max(), "K too large");
    TORCH_CHECK(M64 * N64 <= std::numeric_limits<unsigned int>::max(), "M * N too large for CUDA grid");

    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    auto output = torch::empty({M64, N64}, x.options());
    if (M64 == 0 || N64 == 0) {
        return output;
    }

    constexpr int threads = 256;
    const dim3 grid(static_cast<unsigned int>(M64 * N64));
    const size_t smem = threads * sizeof(float);
    const int M = static_cast<int>(M64);
    const int N = static_cast<int>(N64);
    const int K = static_cast<int>(K64);
    const int packed_K = static_cast<int>(packed_K64);
    const int scale_K = static_cast<int>(scale_K64);
    const auto stream = at::cuda::getCurrentCUDAStream();
    const uint8_t* weight_ptr = weight_bytes.data_ptr<uint8_t>();
    const float* scale_ptr = scales.data_ptr<float>();

    if (x.scalar_type() == torch::kFloat32) {
        mxfp4_linear_kernel<float><<<grid, threads, smem, stream>>>(
            x.data_ptr<float>(), weight_ptr, scale_ptr, output.data_ptr<float>(), M, N, K, packed_K, scale_K);
    } else if (x.scalar_type() == torch::kFloat16) {
        mxfp4_linear_kernel<__half><<<grid, threads, smem, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr()),
            weight_ptr,
            scale_ptr,
            reinterpret_cast<__half*>(output.data_ptr()),
            M,
            N,
            K,
            packed_K,
            scale_K);
    } else {
        mxfp4_linear_kernel<nv_bfloat16><<<grid, threads, smem, stream>>>(
            reinterpret_cast<const nv_bfloat16*>(x.data_ptr()),
            weight_ptr,
            scale_ptr,
            reinterpret_cast<nv_bfloat16*>(output.data_ptr()),
            M,
            N,
            K,
            packed_K,
            scale_K);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
