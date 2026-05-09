/*
 * @Description  :  FP8 E4M3FN linear kernel for SM_86 decode paths.
 *
 * DeepSeek V4 Flash FP8 weights use torch.float8_e4m3fn with
 * torch.float8_e8m0fnu block scales. The runner converts scales to fp32 once,
 * so this kernel consumes raw FP8 bytes and fp32 [ceil(N/128), ceil(K/128)]
 * scales directly.
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>
#include <torch/extension.h>
#include <torch/library.h>
#include <torch/torch.h>

#ifdef __HIP_PLATFORM_AMD__
typedef hip_bfloat16 nv_bfloat16;
#endif

__device__ __forceinline__ float fp8_e4m3fn_to_float(uint8_t value) {
    const int sign = (value & 0x80) ? -1 : 1;
    const int exp = (value >> 3) & 0x0F;
    const int mant = value & 0x07;
    if (exp == 0) {
        if (mant == 0) {
            return sign < 0 ? -0.0f : 0.0f;
        }
        return sign * ldexpf(static_cast<float>(mant) / 8.0f, -6);
    }
    if (exp == 0x0F && mant == 0x07) {
        return 0.0f;
    }
    return sign * ldexpf(1.0f + static_cast<float>(mant) / 8.0f, exp - 7);
}

__device__ __forceinline__ float fp8_load_as_float(const float* ptr, int idx) {
    return ptr[idx];
}

__device__ __forceinline__ float fp8_load_as_float(const __half* ptr, int idx) {
    return __half2float(ptr[idx]);
}

__device__ __forceinline__ float fp8_load_as_float(const nv_bfloat16* ptr, int idx) {
    return __bfloat162float(ptr[idx]);
}

__device__ __forceinline__ void fp8_store_from_float(float* ptr, int idx, float value) {
    ptr[idx] = value;
}

__device__ __forceinline__ void fp8_store_from_float(__half* ptr, int idx, float value) {
    ptr[idx] = __float2half(value);
}

__device__ __forceinline__ void fp8_store_from_float(nv_bfloat16* ptr, int idx, float value) {
    ptr[idx] = __float2bfloat16(value);
}

template <typename scalar_t>
__global__ void fp8_linear_kernel(const scalar_t* __restrict__ x,
                                  const uint8_t* __restrict__ weight,
                                  const float* __restrict__ scales,
                                  scalar_t* __restrict__ out,
                                  int M,
                                  int N,
                                  int K,
                                  int scale_N,
                                  int scale_K) {
    extern __shared__ float partial[];
    const int output_idx = blockIdx.x;
    const int m = output_idx / N;
    const int n = output_idx - m * N;
    float acc = 0.0f;
    const int scale_row = min(n / 128, scale_N - 1);

    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        const uint8_t encoded = weight[n * K + k];
        const int scale_col = min(k / 128, scale_K - 1);
        const float w = fp8_e4m3fn_to_float(encoded) * scales[scale_row * scale_K + scale_col];
        acc += fp8_load_as_float(x, m * K + k) * w;
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
        fp8_store_from_float(out, output_idx, partial[0]);
    }
}

torch::Tensor fp8_linear(torch::Tensor x, torch::Tensor weight_bytes, torch::Tensor scales) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight_bytes.is_cuda(), "weight_bytes must be a CUDA tensor");
    TORCH_CHECK(scales.is_cuda(), "scales must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be [M, K]");
    TORCH_CHECK(weight_bytes.dim() == 2, "weight_bytes must be [N, K]");
    TORCH_CHECK(scales.dim() == 2, "scales must be [ceil(N/128), ceil(K/128)]");
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
    TORCH_CHECK(weight_bytes.size(1) == K64, "weight K must match x K");
    TORCH_CHECK(scales.size(0) == (N64 + 127) / 128, "scales N must equal ceil(weight N / 128)");
    TORCH_CHECK(scales.size(1) == (K64 + 127) / 128, "scales K must equal ceil(weight K / 128)");
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
    const int scale_N = static_cast<int>(scales.size(0));
    const int scale_K = static_cast<int>(scales.size(1));
    const auto stream = at::cuda::getCurrentCUDAStream();
    const uint8_t* weight_ptr = weight_bytes.data_ptr<uint8_t>();
    const float* scale_ptr = scales.data_ptr<float>();

    if (x.scalar_type() == torch::kFloat32) {
        fp8_linear_kernel<float><<<grid, threads, smem, stream>>>(
            x.data_ptr<float>(), weight_ptr, scale_ptr, output.data_ptr<float>(), M, N, K, scale_N, scale_K);
    } else if (x.scalar_type() == torch::kFloat16) {
        fp8_linear_kernel<__half><<<grid, threads, smem, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr()),
            weight_ptr,
            scale_ptr,
            reinterpret_cast<__half*>(output.data_ptr()),
            M,
            N,
            K,
            scale_N,
            scale_K);
    } else {
        fp8_linear_kernel<nv_bfloat16><<<grid, threads, smem, stream>>>(
            reinterpret_cast<const nv_bfloat16*>(x.data_ptr()),
            weight_ptr,
            scale_ptr,
            reinterpret_cast<nv_bfloat16*>(output.data_ptr()),
            M,
            N,
            K,
            scale_N,
            scale_K);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
