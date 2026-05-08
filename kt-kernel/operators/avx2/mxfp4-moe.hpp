/**
 * @Description  : AVX2 MXFP4 MoE operator
 * @Author       : Codex
 * @Date         : 2026-05-08
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * MXFP4 layout:
 *   weights: row-major [N, K / 2], two E2M1 FP4 values per byte
 *   scales:  row-major [N, K / group_size], bf16/ue8m0 at source, float in BufferB
 *
 * Compute path:
 *   BF16 activation -> FP32
 *   E2M1 nibbles -> FP32 via AVX2/PSHUFB
 *   group scale is applied inside the dot-product loop
 **/
#ifndef CPUINFER_OPERATOR_AVX2_MXFP4_MOE_H
#define CPUINFER_OPERATOR_AVX2_MXFP4_MOE_H

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "avx2_bf16_gemm.hpp"
#include "avx2_bf16_utils.hpp"
#include "moe_base.hpp"
#include "mxfp4_dequant.hpp"

namespace avx2 {

struct GemmKernelAVX2MXFP4 {
  using dt = uint8_t;
  using output_t = float;
  using source_scale_t = ggml_bf16_t;
  static constexpr int M_STEP = 1;
  static constexpr int N_STEP = 4;
  static constexpr int K_STEP = 16;
  static constexpr int N_BLOCK = 256;
  static constexpr int K_BLOCK = 32;
  static constexpr double ELEMENT_SIZE = 0.5;

  static void config() {}

  static int recommended_nth(int n) { return std::max(1, div_up(n, N_BLOCK)); }

  static std::pair<int, int> split_range_n(int n, int ith, int nth) { return split_range(n, ith, nth); }

  struct BufferA {
    ggml_bf16_t* data = nullptr;
    size_t max_m = 0;
    size_t k = 0;

    BufferA() = default;
    BufferA(size_t m, size_t k_, void* ptr) : data((ggml_bf16_t*)ptr), max_m(m), k(k_) {}

    static size_t required_size(size_t m, size_t k) { return m * k * sizeof(ggml_bf16_t); }

    void set_data(void* ptr) { data = (ggml_bf16_t*)ptr; }

    void from_mat(int m, const ggml_bf16_t* src, int ith, int nth) {
      if (ith == 0 && nth == 1) {
        std::memcpy(data, src, (size_t)m * k * sizeof(ggml_bf16_t));
      } else {
        auto [m_start, m_end] = split_range(m, ith, nth);
        std::memcpy(data + (size_t)m_start * k, src + (size_t)m_start * k,
                    (size_t)(m_end - m_start) * k * sizeof(ggml_bf16_t));
      }
    }
  };

  struct BufferB {
    uint8_t* b = nullptr;
    float* d = nullptr;
    int n = 0;
    int k = 0;
    int row_bytes = 0;
    int group_size = 32;
    int num_groups = 0;

    BufferB() = default;
    BufferB(size_t n_, size_t k_, int gs, void* ptr) : n((int)n_), k((int)k_), group_size(gs) {
      if (group_size <= 0 || (group_size % 16) != 0) {
        throw std::runtime_error("AVX2 MXFP4 requires group_size to be a positive multiple of 16");
      }
      if ((k % 2) != 0 || (k % group_size) != 0) {
        throw std::runtime_error("AVX2 MXFP4 requires K to be divisible by 2 and group_size");
      }
      row_bytes = k / 2;
      num_groups = k / group_size;
      b = (uint8_t*)ptr;
      d = (float*)((uint8_t*)ptr + (size_t)n * row_bytes);
    }

    static size_t required_size(size_t n, size_t k, int gs) {
      if (gs <= 0) return 0;
      return n * (k / 2) + n * (k / gs) * sizeof(float);
    }

    void from_mat(const uint8_t* src_weights, const ggml_bf16_t* src_scales, int ith, int nth) {
      auto [n_start, n_end] = split_range(n, ith, nth);
      if (n_start >= n_end) return;
      std::memcpy(b + (size_t)n_start * row_bytes, src_weights + (size_t)n_start * row_bytes,
                  (size_t)(n_end - n_start) * row_bytes);
      convert_or_copy(d + (size_t)n_start * num_groups, src_scales + (size_t)n_start * num_groups,
                      (size_t)(n_end - n_start) * num_groups);
    }
  };

  struct BufferC {
    float* data = nullptr;
    size_t max_m = 0;
    size_t n = 0;

    BufferC() = default;
    BufferC(size_t m, size_t n_, void* ptr) : data((float*)ptr), max_m(m), n(n_) {}

    static size_t required_size(size_t m, size_t n) { return m * n * sizeof(float); }

    void set_data(void* ptr) { data = (float*)ptr; }

    void to_mat(int m, ggml_bf16_t* dst, int ith, int nth) {
      auto [n_start, n_end] = split_range((int)n, ith, nth);
      for (int mi = 0; mi < m; mi++) {
        float* src_row = data + (size_t)mi * n;
        ggml_bf16_t* dst_row = dst + (size_t)mi * n;
        int j = n_start;
        for (; j + 8 <= n_end; j += 8) {
          store_fp32_to_bf16(dst_row + j, _mm256_loadu_ps(src_row + j));
        }
        for (; j < n_end; j++) {
          dst_row[j] = GGML_FP32_TO_BF16(src_row[j]);
        }
      }
    }
  };
};

struct GemmKernelAVX2MXFP4DQ {
  using dt = uint8_t;
  using output_t = float;
  using source_scale_t = uint8_t;
  using BufferA = GemmKernelAVX2MXFP4::BufferA;
  using BufferC = GemmKernelAVX2MXFP4::BufferC;
  static constexpr int M_STEP = GemmKernelAVX2MXFP4::M_STEP;
  static constexpr int N_STEP = GemmKernelAVX2MXFP4::N_STEP;
  static constexpr int K_STEP = GemmKernelAVX2MXFP4::K_STEP;
  static constexpr int N_BLOCK = GemmKernelAVX2MXFP4::N_BLOCK;
  static constexpr int K_BLOCK = GemmKernelAVX2MXFP4::K_BLOCK;
  static constexpr double ELEMENT_SIZE = 0.5;

  static void config() {}

  static int recommended_nth(int n) { return GemmKernelAVX2MXFP4::recommended_nth(n); }

  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    return GemmKernelAVX2MXFP4::split_range_n(n, ith, nth);
  }

  struct BufferB {
    uint8_t* b = nullptr;
    uint8_t* d = nullptr;
    int n = 0;
    int k = 0;
    int row_bytes = 0;
    int group_size = 32;
    int num_groups = 0;

    BufferB() = default;
    BufferB(size_t n_, size_t k_, int gs, void* ptr) : n((int)n_), k((int)k_), group_size(gs) {
      if (group_size <= 0 || (group_size % 16) != 0) {
        throw std::runtime_error("AVX2 MXFP4 DQ requires group_size to be a positive multiple of 16");
      }
      if ((k % 2) != 0 || (k % group_size) != 0) {
        throw std::runtime_error("AVX2 MXFP4 DQ requires K to be divisible by 2 and group_size");
      }
      row_bytes = k / 2;
      num_groups = k / group_size;
      b = (uint8_t*)ptr;
      d = (uint8_t*)ptr + (size_t)n * row_bytes;
    }

    static size_t required_size(size_t n, size_t k, int gs) {
      if (gs <= 0) return 0;
      return n * (k / 2) + n * (k / gs);
    }

    void from_mat(const uint8_t* src_weights, const uint8_t* src_scales, int ith, int nth) {
      auto [n_start, n_end] = split_range(n, ith, nth);
      if (n_start >= n_end) return;
      std::memcpy(b + (size_t)n_start * row_bytes, src_weights + (size_t)n_start * row_bytes,
                  (size_t)(n_end - n_start) * row_bytes);
      std::memcpy(d + (size_t)n_start * num_groups, src_scales + (size_t)n_start * num_groups,
                  (size_t)(n_end - n_start) * num_groups);
    }
  };
};

static inline void mxfp4_fma_16(const ggml_bf16_t* a, const uint8_t* w, __m256 scale, __m256* acc) {
  __m256 w0, w1;
  mxfp4x16_to_2xfp32(w, &w0, &w1);
  w0 = _mm256_mul_ps(w0, scale);
  w1 = _mm256_mul_ps(w1, scale);
  *acc = _mm256_fmadd_ps(load_bf16_to_fp32(a), w0, *acc);
  *acc = _mm256_fmadd_ps(load_bf16_to_fp32(a + 8), w1, *acc);
}

static inline float mxfp4_dot_scaled(const ggml_bf16_t* a_row, const uint8_t* b_row, const float* scales,
                                     int group_size, int num_groups) {
  __m256 acc = _mm256_setzero_ps();
  for (int g = 0; g < num_groups; ++g) {
    const int k_base = g * group_size;
    const __m256 scale = _mm256_set1_ps(scales[g]);
    for (int kk = 0; kk < group_size; kk += 16) {
      mxfp4_fma_16(a_row + k_base + kk, b_row + (k_base + kk) / 2, scale, &acc);
    }
  }
  return hsum_avx2(acc);
}

inline const std::array<uint32_t, 256> UE8M0_F32_BITS = [] {
  std::array<uint32_t, 256> table{};
  for (int i = 1; i < 255; ++i) table[(size_t)i] = (uint32_t)i << 23;
  return table;
}();

inline const std::array<uint16_t, 256> UE8M0_BF16_BITS = [] {
  std::array<uint16_t, 256> table{};
  for (int i = 1; i < 255; ++i) table[(size_t)i] = (uint16_t)i << 7;
  return table;
}();

static inline __m256 ue8m0_scale_to_m256(uint8_t raw) {
  return _mm256_castsi256_ps(_mm256_set1_epi32((int)UE8M0_F32_BITS[(size_t)raw]));
}

static inline ggml_bf16_t ue8m0_scale_to_bf16(uint8_t raw) {
  ggml_bf16_t out;
  out.bits = UE8M0_BF16_BITS[(size_t)raw];
  return out;
}

static inline float mxfp4_dot_scaled_ue8m0(const ggml_bf16_t* a_row, const uint8_t* b_row, const uint8_t* scales,
                                           int group_size, int num_groups) {
  __m256 acc = _mm256_setzero_ps();
  for (int g = 0; g < num_groups; ++g) {
    const int k_base = g * group_size;
    const __m256 scale = ue8m0_scale_to_m256(scales[g]);
    for (int kk = 0; kk < group_size; kk += 16) {
      mxfp4_fma_16(a_row + k_base + kk, b_row + (k_base + kk) / 2, scale, &acc);
    }
  }
  return hsum_avx2(acc);
}

static inline void gemm_mxfp4(int m, int n, int k, GemmKernelAVX2MXFP4::BufferA& a,
                              GemmKernelAVX2MXFP4::BufferB& b, GemmKernelAVX2MXFP4::BufferC& c, int ith, int nth) {
  (void)k;
  auto [n_start, n_end] = split_range(n, ith, nth);
  const int group_size = b.group_size;
  const int num_groups = b.num_groups;

  for (int mi = 0; mi < m; ++mi) {
    const ggml_bf16_t* a_row = a.data + (size_t)mi * a.k;
    float* c_row = c.data + (size_t)mi * n;
    int ni = n_start;

    for (; ni + 4 <= n_end; ni += 4) {
      const uint8_t* b0 = b.b + (size_t)(ni + 0) * b.row_bytes;
      const uint8_t* b1 = b.b + (size_t)(ni + 1) * b.row_bytes;
      const uint8_t* b2 = b.b + (size_t)(ni + 2) * b.row_bytes;
      const uint8_t* b3 = b.b + (size_t)(ni + 3) * b.row_bytes;
      const float* s0 = b.d + (size_t)(ni + 0) * num_groups;
      const float* s1 = b.d + (size_t)(ni + 1) * num_groups;
      const float* s2 = b.d + (size_t)(ni + 2) * num_groups;
      const float* s3 = b.d + (size_t)(ni + 3) * num_groups;

      __m256 acc0 = _mm256_setzero_ps();
      __m256 acc1 = _mm256_setzero_ps();
      __m256 acc2 = _mm256_setzero_ps();
      __m256 acc3 = _mm256_setzero_ps();

      for (int g = 0; g < num_groups; ++g) {
        const int k_base = g * group_size;
        const __m256 scale0 = _mm256_set1_ps(s0[g]);
        const __m256 scale1 = _mm256_set1_ps(s1[g]);
        const __m256 scale2 = _mm256_set1_ps(s2[g]);
        const __m256 scale3 = _mm256_set1_ps(s3[g]);
        for (int kk = 0; kk < group_size; kk += 16) {
          const int k_off = k_base + kk;
          mxfp4_fma_16(a_row + k_off, b0 + k_off / 2, scale0, &acc0);
          mxfp4_fma_16(a_row + k_off, b1 + k_off / 2, scale1, &acc1);
          mxfp4_fma_16(a_row + k_off, b2 + k_off / 2, scale2, &acc2);
          mxfp4_fma_16(a_row + k_off, b3 + k_off / 2, scale3, &acc3);
        }
      }

      c_row[ni + 0] = hsum_avx2(acc0);
      c_row[ni + 1] = hsum_avx2(acc1);
      c_row[ni + 2] = hsum_avx2(acc2);
      c_row[ni + 3] = hsum_avx2(acc3);
    }

    for (; ni < n_end; ++ni) {
      const uint8_t* b_row = b.b + (size_t)ni * b.row_bytes;
      const float* scales = b.d + (size_t)ni * num_groups;
      c_row[ni] = mxfp4_dot_scaled(a_row, b_row, scales, group_size, num_groups);
    }
  }
}

static inline void gemm_mxfp4(int m, int n, int k, GemmKernelAVX2MXFP4DQ::BufferA& a,
                              GemmKernelAVX2MXFP4DQ::BufferB& b, GemmKernelAVX2MXFP4DQ::BufferC& c, int ith,
                              int nth) {
  (void)k;
  auto [n_start, n_end] = split_range(n, ith, nth);
  const int group_size = b.group_size;
  const int num_groups = b.num_groups;

  for (int mi = 0; mi < m; ++mi) {
    const ggml_bf16_t* a_row = a.data + (size_t)mi * a.k;
    float* c_row = c.data + (size_t)mi * n;
    int ni = n_start;

    for (; ni + 4 <= n_end; ni += 4) {
      const uint8_t* b0 = b.b + (size_t)(ni + 0) * b.row_bytes;
      const uint8_t* b1 = b.b + (size_t)(ni + 1) * b.row_bytes;
      const uint8_t* b2 = b.b + (size_t)(ni + 2) * b.row_bytes;
      const uint8_t* b3 = b.b + (size_t)(ni + 3) * b.row_bytes;
      const uint8_t* s0 = b.d + (size_t)(ni + 0) * num_groups;
      const uint8_t* s1 = b.d + (size_t)(ni + 1) * num_groups;
      const uint8_t* s2 = b.d + (size_t)(ni + 2) * num_groups;
      const uint8_t* s3 = b.d + (size_t)(ni + 3) * num_groups;

      __m256 acc0 = _mm256_setzero_ps();
      __m256 acc1 = _mm256_setzero_ps();
      __m256 acc2 = _mm256_setzero_ps();
      __m256 acc3 = _mm256_setzero_ps();

      for (int g = 0; g < num_groups; ++g) {
        const int k_base = g * group_size;
        const __m256 scale0 = ue8m0_scale_to_m256(s0[g]);
        const __m256 scale1 = ue8m0_scale_to_m256(s1[g]);
        const __m256 scale2 = ue8m0_scale_to_m256(s2[g]);
        const __m256 scale3 = ue8m0_scale_to_m256(s3[g]);
        for (int kk = 0; kk < group_size; kk += 16) {
          const int k_off = k_base + kk;
          mxfp4_fma_16(a_row + k_off, b0 + k_off / 2, scale0, &acc0);
          mxfp4_fma_16(a_row + k_off, b1 + k_off / 2, scale1, &acc1);
          mxfp4_fma_16(a_row + k_off, b2 + k_off / 2, scale2, &acc2);
          mxfp4_fma_16(a_row + k_off, b3 + k_off / 2, scale3, &acc3);
        }
      }

      c_row[ni + 0] = hsum_avx2(acc0);
      c_row[ni + 1] = hsum_avx2(acc1);
      c_row[ni + 2] = hsum_avx2(acc2);
      c_row[ni + 3] = hsum_avx2(acc3);
    }

    for (; ni < n_end; ++ni) {
      const uint8_t* b_row = b.b + (size_t)ni * b.row_bytes;
      const uint8_t* scales = b.d + (size_t)ni * num_groups;
      c_row[ni] = mxfp4_dot_scaled_ue8m0(a_row, b_row, scales, group_size, num_groups);
    }
  }
}

}  // namespace avx2

template <class T = avx2::GemmKernelAVX2MXFP4>
class AVX2_MXFP4_MOE_TP : public AVX2_MOE_BASE<T, AVX2_MXFP4_MOE_TP<T>> {
  using Base = AVX2_MOE_BASE<T, AVX2_MXFP4_MOE_TP<T>>;
  using Base::config_;
  using Base::down_ba_;
  using Base::down_bb_;
  using Base::down_bc_;
  using Base::gate_bb_;
  using Base::gate_bc_;
  using Base::gate_up_ba_;
  using Base::m_local_num_;
  using Base::tp_part_idx;
  using Base::up_bb_;
  using Base::up_bc_;

 public:
  using typename Base::input_t;
  using typename Base::output_t;

  AVX2_MXFP4_MOE_TP() = default;
  AVX2_MXFP4_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {}

  void derived_init() {
    auto& quant_config = config_.quant_config;
    if (quant_config.group_size <= 0 || (quant_config.group_size % 16) != 0 || quant_config.zero_point) {
      throw std::runtime_error("AVX2 MXFP4 MoE only supports weight-only K-group FP4 with group_size % 16 == 0");
    }
    printf("Created AVX2_MXFP4_MOE_TP %d at numa %d\n", tp_part_idx, numa_node_of_cpu(sched_getcpu()));
  }

  ~AVX2_MXFP4_MOE_TP() = default;

  size_t buffer_a_required_size_impl(size_t m, size_t k) const { return T::BufferA::required_size(m, k); }
  size_t buffer_b_required_size_impl(size_t n, size_t k) const {
    return T::BufferB::required_size(n, k, config_.quant_config.group_size);
  }
  size_t buffer_c_required_size_impl(size_t m, size_t n) const { return T::BufferC::required_size(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a_impl(size_t m, size_t k, void* data) const {
    return std::make_shared<typename T::BufferA>(m, k, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b_impl(size_t n, size_t k, void* data) const {
    return std::make_shared<typename T::BufferB>(n, k, config_.quant_config.group_size, data);
  }
  std::shared_ptr<typename T::BufferC> make_buffer_c_impl(size_t m, size_t n, void* data) const {
    return std::make_shared<typename T::BufferC>(m, n, data);
  }

  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, int qlen) {
    (void)qlen;
    int m = m_local_num_[expert_idx];
    auto& ba = gate_up_ba_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];
    avx2::gemm_mxfp4(m, config_.intermediate_size, config_.hidden_size, *ba, *bb, *bc, ith, nth);
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int qlen) {
    (void)qlen;
    int m = m_local_num_[expert_idx];
    avx2::gemm_mxfp4(m, config_.hidden_size, config_.intermediate_size, *down_ba_[expert_idx],
                     *down_bb_[expert_idx], *down_bc_[expert_idx], ith, nth);
  }

  void load_weights() {
    using scale_t = typename T::source_scale_t;
    auto& quant_config = config_.quant_config;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    if (quant_config.group_size <= 0 || quant_config.zero_point) {
      throw std::runtime_error("AVX2 MXFP4 MoE only supports K-group FP4 without zero point");
    }
    if (config_.gate_proj == nullptr || config_.up_proj == nullptr || config_.down_proj == nullptr ||
        config_.gate_scale == nullptr || config_.up_scale == nullptr || config_.down_scale == nullptr) {
      throw std::runtime_error("AVX2 MXFP4 MoE requires native packed weights and scale pointers");
    }

    const size_t weight_elems = (size_t)config_.intermediate_size * config_.hidden_size;
    const size_t weight_bytes = weight_elems / 2;
    const size_t scale_elems = weight_elems / quant_config.group_size;

    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map, weight_bytes, scale_elems](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;

          gate_bb_[expert_idx]->from_mat((uint8_t*)config_.gate_proj + logical_expert_id * weight_bytes,
                                         (scale_t*)config_.gate_scale + logical_expert_id * scale_elems, ith, nth);
          up_bb_[expert_idx]->from_mat((uint8_t*)config_.up_proj + logical_expert_id * weight_bytes,
                                       (scale_t*)config_.up_scale + logical_expert_id * scale_elems, ith, nth);
        },
        nullptr);

    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map, weight_bytes, scale_elems](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;

          down_bb_[expert_idx]->from_mat((uint8_t*)config_.down_proj + logical_expert_id * weight_bytes,
                                         (scale_t*)config_.down_scale + logical_expert_id * scale_elems, ith, nth);
        },
        nullptr);
  }

  static inline void fast_memcpy(void* __restrict dst, const void* __restrict src, size_t bytes) {
    uint8_t* d = (uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    size_t i = 0;
    for (; i + 32 <= bytes; i += 32) {
      __m256i v = _mm256_loadu_si256((const __m256i*)(s + i));
      _mm256_storeu_si256((__m256i*)(d + i), v);
    }
    if (i < bytes) std::memcpy(d + i, s + i, bytes - i);
  }

  static inline void fast_fp32_to_bf16(ggml_bf16_t* __restrict dst, const float* __restrict src, size_t count) {
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
      avx2::store_fp32_to_bf16(dst + i, _mm256_loadu_ps(src + i));
    }
    for (; i < count; ++i) dst[i] = GGML_FP32_TO_BF16(src[i]);
  }

  static inline void fast_scale_to_bf16(ggml_bf16_t* __restrict dst, const float* __restrict src, size_t count) {
    fast_fp32_to_bf16(dst, src, count);
  }

  static inline void fast_scale_to_bf16(ggml_bf16_t* __restrict dst, const uint8_t* __restrict src, size_t count) {
    size_t i = 0;
    for (; i < count; ++i) dst[i] = avx2::ue8m0_scale_to_bf16(src[i]);
  }

  void write_weights_to_buffer(int gpu_tp_count, int cpu_tp_count, int expert_id, const GeneralMOEConfig& full_config,
                               const std::vector<uintptr_t>& w13_weight_ptrs,
                               const std::vector<uintptr_t>& w13_scale_ptrs,
                               const std::vector<uintptr_t>& w2_weight_ptrs,
                               const std::vector<uintptr_t>& w2_scale_ptrs) const {
    const int group_size = config_.quant_config.group_size;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    const size_t cpu_tp_weight_elem_count = (size_t)config_.intermediate_size * config_.hidden_size;
    const size_t cpu_tp_weight_bytes = cpu_tp_weight_elem_count / 2;
    const size_t cpu_tp_scale_elem_count = cpu_tp_weight_elem_count / group_size;
    const size_t gpu_tp_weight_elem_count =
        (size_t)full_config.intermediate_size * full_config.hidden_size / gpu_tp_count;
    const size_t gpu_tp_weight_bytes = gpu_tp_weight_elem_count / 2;
    const size_t gpu_tp_scale_elem_count = gpu_tp_weight_elem_count / group_size;

    if (cpu_tp_count >= gpu_tp_count) {
      int target_gpu_tp = tp_part_idx / (cpu_tp_count / gpu_tp_count);
      int local_idx = tp_part_idx % (cpu_tp_count / gpu_tp_count);

      uint8_t* w13_weight_dst = (uint8_t*)w13_weight_ptrs[target_gpu_tp];
      ggml_bf16_t* w13_scale_dst = (ggml_bf16_t*)w13_scale_ptrs[target_gpu_tp];
      uint8_t* w2_weight_dst = (uint8_t*)w2_weight_ptrs[target_gpu_tp];
      ggml_bf16_t* w2_scale_dst = (ggml_bf16_t*)w2_scale_ptrs[target_gpu_tp];

      const size_t offset_in_gpu_weight = local_idx * cpu_tp_weight_bytes;
      const size_t offset_in_gpu_scale = local_idx * cpu_tp_scale_elem_count;

      constexpr int NUM_WEIGHT_TASKS = 8;
      constexpr int MIN_COLS_PER_TASK = 128;
      int num_down_tasks = std::max(1, config_.hidden_size / MIN_COLS_PER_TASK);
      num_down_tasks = std::min(num_down_tasks, 32);
      int total_tasks = NUM_WEIGHT_TASKS * 2 + num_down_tasks + 2;
      size_t weight_chunk_size = (cpu_tp_weight_bytes + NUM_WEIGHT_TASKS - 1) / NUM_WEIGHT_TASKS;
      weight_chunk_size = (weight_chunk_size + 31) & ~31ULL;

      pool->do_work_stealing_job(
          total_tasks, nullptr,
          [&, this, num_down_tasks, expert_id, weight_chunk_size, offset_in_gpu_weight, offset_in_gpu_scale,
           gpu_tp_weight_bytes, gpu_tp_scale_elem_count, w13_weight_dst, w13_scale_dst, w2_weight_dst, w2_scale_dst,
           group_size](int task_id) {
            if (task_id < NUM_WEIGHT_TASKS) {
              size_t start = (size_t)task_id * weight_chunk_size;
              size_t end = std::min(start + weight_chunk_size, cpu_tp_weight_bytes);
              if (start < end) fast_memcpy(w13_weight_dst + offset_in_gpu_weight + start, gate_bb_[expert_id]->b + start,
                                           end - start);
            } else if (task_id < NUM_WEIGHT_TASKS * 2) {
              int chunk_idx = task_id - NUM_WEIGHT_TASKS;
              size_t start = (size_t)chunk_idx * weight_chunk_size;
              size_t end = std::min(start + weight_chunk_size, cpu_tp_weight_bytes);
              if (start < end) fast_memcpy(w13_weight_dst + offset_in_gpu_weight + gpu_tp_weight_bytes + start,
                                           up_bb_[expert_id]->b + start, end - start);
            } else if (task_id < NUM_WEIGHT_TASKS * 2 + num_down_tasks) {
              int chunk_idx = task_id - NUM_WEIGHT_TASKS * 2;
              size_t cols_per_chunk = (config_.hidden_size + num_down_tasks - 1) / num_down_tasks;
              size_t col_start = (size_t)chunk_idx * cols_per_chunk;
              size_t col_end = std::min(col_start + cols_per_chunk, (size_t)config_.hidden_size);
              size_t weight_per_col = (size_t)config_.intermediate_size / 2;
              size_t scale_per_col = (size_t)config_.intermediate_size / group_size;
              size_t gpu_weight_stride = (size_t)(full_config.intermediate_size / gpu_tp_count) / 2;
              size_t gpu_scale_stride = (size_t)(full_config.intermediate_size / gpu_tp_count) / group_size;
              size_t gpu_weight_slice_offset = local_idx * weight_per_col;
              size_t gpu_scale_slice_offset = local_idx * scale_per_col;

              for (size_t col = col_start; col < col_end; ++col) {
                fast_memcpy(w2_weight_dst + col * gpu_weight_stride + gpu_weight_slice_offset,
                            down_bb_[expert_id]->b + col * weight_per_col, weight_per_col);
                fast_scale_to_bf16(w2_scale_dst + col * gpu_scale_stride + gpu_scale_slice_offset,
                                   down_bb_[expert_id]->d + col * scale_per_col, scale_per_col);
              }
            } else if (task_id == NUM_WEIGHT_TASKS * 2 + num_down_tasks) {
              fast_scale_to_bf16(w13_scale_dst + offset_in_gpu_scale, gate_bb_[expert_id]->d, cpu_tp_scale_elem_count);
            } else {
              fast_scale_to_bf16(w13_scale_dst + offset_in_gpu_scale + gpu_tp_scale_elem_count, up_bb_[expert_id]->d,
                                 cpu_tp_scale_elem_count);
            }
          },
          nullptr);
    } else {
      int gpu_tps_per_cpu_tp = gpu_tp_count / cpu_tp_count;
      int start_gpu_tp = tp_part_idx * gpu_tps_per_cpu_tp;
      size_t data_per_gpu_tp_weight = cpu_tp_weight_bytes / gpu_tps_per_cpu_tp;
      size_t data_per_gpu_tp_scale = cpu_tp_scale_elem_count / gpu_tps_per_cpu_tp;

      constexpr int NUM_WEIGHT_TASKS = 8;
      constexpr int MIN_COLS_PER_TASK = 128;
      int num_down_tasks = std::max(1, config_.hidden_size / MIN_COLS_PER_TASK);
      num_down_tasks = std::min(num_down_tasks, 32);
      int tasks_per_gpu_tp = NUM_WEIGHT_TASKS * 2 + num_down_tasks + 2;
      int total_tasks = tasks_per_gpu_tp * gpu_tps_per_cpu_tp;
      size_t weight_chunk_size = (data_per_gpu_tp_weight + NUM_WEIGHT_TASKS - 1) / NUM_WEIGHT_TASKS;
      weight_chunk_size = (weight_chunk_size + 31) & ~31ULL;

      pool->do_work_stealing_job(
          total_tasks, nullptr,
          [&, this, gpu_tps_per_cpu_tp, start_gpu_tp, data_per_gpu_tp_weight, data_per_gpu_tp_scale, num_down_tasks,
           tasks_per_gpu_tp, expert_id, weight_chunk_size, gpu_tp_weight_bytes, gpu_tp_scale_elem_count,
           group_size](int task_id) {
            int local_gpu_idx = task_id / tasks_per_gpu_tp;
            int task_type = task_id % tasks_per_gpu_tp;
            int gpu_tp_idx = start_gpu_tp + local_gpu_idx;
            uint8_t* w13_weight_dst = (uint8_t*)w13_weight_ptrs[gpu_tp_idx];
            ggml_bf16_t* w13_scale_dst = (ggml_bf16_t*)w13_scale_ptrs[gpu_tp_idx];
            uint8_t* w2_weight_dst = (uint8_t*)w2_weight_ptrs[gpu_tp_idx];
            ggml_bf16_t* w2_scale_dst = (ggml_bf16_t*)w2_scale_ptrs[gpu_tp_idx];
            size_t cpu_offset_weight = (size_t)local_gpu_idx * data_per_gpu_tp_weight;
            size_t cpu_offset_scale = (size_t)local_gpu_idx * data_per_gpu_tp_scale;

            if (task_type < NUM_WEIGHT_TASKS) {
              size_t start = (size_t)task_type * weight_chunk_size;
              size_t end = std::min(start + weight_chunk_size, data_per_gpu_tp_weight);
              if (start < end) fast_memcpy(w13_weight_dst + start, gate_bb_[expert_id]->b + cpu_offset_weight + start,
                                           end - start);
            } else if (task_type < NUM_WEIGHT_TASKS * 2) {
              int chunk_idx = task_type - NUM_WEIGHT_TASKS;
              size_t start = (size_t)chunk_idx * weight_chunk_size;
              size_t end = std::min(start + weight_chunk_size, data_per_gpu_tp_weight);
              if (start < end) fast_memcpy(w13_weight_dst + gpu_tp_weight_bytes + start,
                                           up_bb_[expert_id]->b + cpu_offset_weight + start, end - start);
            } else if (task_type < NUM_WEIGHT_TASKS * 2 + num_down_tasks) {
              int chunk_idx = task_type - NUM_WEIGHT_TASKS * 2;
              size_t cols_per_chunk = (config_.hidden_size + num_down_tasks - 1) / num_down_tasks;
              size_t col_start = (size_t)chunk_idx * cols_per_chunk;
              size_t col_end = std::min(col_start + cols_per_chunk, (size_t)config_.hidden_size);
              size_t weight_per_gpu_col = (size_t)(config_.intermediate_size / gpu_tps_per_cpu_tp) / 2;
              size_t scale_per_gpu_col = (size_t)(config_.intermediate_size / gpu_tps_per_cpu_tp) / group_size;

              for (size_t col = col_start; col < col_end; ++col) {
                size_t col_offset_weight = col * ((size_t)config_.intermediate_size / 2) +
                                           (size_t)local_gpu_idx * data_per_gpu_tp_weight / config_.hidden_size;
                size_t col_offset_scale = col * ((size_t)config_.intermediate_size / group_size) +
                                          (size_t)local_gpu_idx * data_per_gpu_tp_scale / config_.hidden_size;
                fast_memcpy(w2_weight_dst + col * weight_per_gpu_col, down_bb_[expert_id]->b + col_offset_weight,
                            weight_per_gpu_col);
                fast_scale_to_bf16(w2_scale_dst + col * scale_per_gpu_col, down_bb_[expert_id]->d + col_offset_scale,
                                   scale_per_gpu_col);
              }
            } else if (task_type == NUM_WEIGHT_TASKS * 2 + num_down_tasks) {
              fast_scale_to_bf16(w13_scale_dst, gate_bb_[expert_id]->d + cpu_offset_scale, data_per_gpu_tp_scale);
            } else {
              fast_scale_to_bf16(w13_scale_dst + gpu_tp_scale_elem_count, up_bb_[expert_id]->d + cpu_offset_scale,
                                 data_per_gpu_tp_scale);
            }
          },
          nullptr);
    }
  }
};

template <typename K>
class TP_MOE<AVX2_MXFP4_MOE_TP<K>> : public TP_MOE<AVX2_MOE_BASE<K, AVX2_MXFP4_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<AVX2_MOE_BASE<K, AVX2_MXFP4_MOE_TP<K>>>;
  using Base::Base;

  void load_weights() override {
    using scale_t = typename K::source_scale_t;
    auto& config = this->config;
    auto& tps = this->tps;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;
    const bool use_per_expert_ptrs = !config.gate_projs.empty();
    const int group_size = config.quant_config.group_size;

    if (group_size <= 0 || (group_size % 16) != 0 || config.quant_config.zero_point) {
      throw std::runtime_error("AVX2 MXFP4 requires group_size % 16 == 0 and zero_point=false");
    }
    if (use_per_expert_ptrs) {
      if (config.up_projs.empty() || config.down_projs.empty() || config.gate_scales.empty() ||
          config.up_scales.empty() || config.down_scales.empty()) {
        throw std::runtime_error("AVX2 MXFP4 requires packed per-expert weights plus per-expert scale tensors");
      }
    } else if (config.gate_proj == nullptr || config.up_proj == nullptr || config.down_proj == nullptr ||
               config.gate_scale == nullptr || config.up_scale == nullptr || config.down_scale == nullptr) {
      throw std::runtime_error("AVX2 MXFP4 requires packed weights plus scale tensors");
    }

    const size_t full_weight_elems = (size_t)config.intermediate_size * config.hidden_size;
    const size_t full_scale_elems = full_weight_elems / group_size;

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      if (tpc.intermediate_size % group_size != 0) {
        throw std::runtime_error("AVX2 MXFP4 TP intermediate_size must be divisible by group_size");
      }
      const size_t tp_weight_elems = (size_t)tpc.intermediate_size * tpc.hidden_size;
      const size_t tp_scale_elems = tp_weight_elems / group_size;

      tpc.gate_proj = new uint8_t[(tpc.expert_num * tp_weight_elems) / 2];
      tpc.up_proj = new uint8_t[(tpc.expert_num * tp_weight_elems) / 2];
      tpc.down_proj = new uint8_t[(tpc.expert_num * tp_weight_elems) / 2];
      tpc.gate_scale = new scale_t[tpc.expert_num * tp_scale_elems];
      tpc.up_scale = new scale_t[tpc.expert_num * tp_scale_elems];
      tpc.down_scale = new scale_t[tpc.expert_num * tp_scale_elems];

      const size_t gate_up_weight_src_offset = ((size_t)i * tp_weight_elems) / 2;
      const size_t gate_up_scale_src_offset = (size_t)i * tp_scale_elems;
      const size_t down_weight_src_col_offset = (size_t)i * tpc.intermediate_size;
      const size_t down_scale_src_block_k_offset = down_weight_src_col_offset / group_size;

      pool->get_subpool(i)->do_work_stealing_job(
          tpc.expert_num, nullptr,
          [&](int expert_id_) {
            const size_t expert_id = expert_map(physical_to_logical_map, expert_id_);
            uint8_t* gate_dst = (uint8_t*)tpc.gate_proj + (expert_id * tp_weight_elems) / 2;
            uint8_t* up_dst = (uint8_t*)tpc.up_proj + (expert_id * tp_weight_elems) / 2;
            uint8_t* down_dst = (uint8_t*)tpc.down_proj + (expert_id * tp_weight_elems) / 2;
            scale_t* gate_scale_dst = (scale_t*)tpc.gate_scale + expert_id * tp_scale_elems;
            scale_t* up_scale_dst = (scale_t*)tpc.up_scale + expert_id * tp_scale_elems;
            scale_t* down_scale_dst = (scale_t*)tpc.down_scale + expert_id * tp_scale_elems;

            const uint8_t* gate_src;
            const uint8_t* up_src;
            const uint8_t* down_src;
            const scale_t* gate_scale_src;
            const scale_t* up_scale_src;
            const scale_t* down_scale_src;

            if (use_per_expert_ptrs) {
              gate_src = (const uint8_t*)config.gate_projs[0][expert_id] + gate_up_weight_src_offset;
              up_src = (const uint8_t*)config.up_projs[0][expert_id] + gate_up_weight_src_offset;
              down_src = (const uint8_t*)config.down_projs[0][expert_id];
              gate_scale_src = (const scale_t*)config.gate_scales[0][expert_id] + gate_up_scale_src_offset;
              up_scale_src = (const scale_t*)config.up_scales[0][expert_id] + gate_up_scale_src_offset;
              down_scale_src = (const scale_t*)config.down_scales[0][expert_id];
            } else {
              gate_src = (const uint8_t*)config.gate_proj + (expert_id * full_weight_elems) / 2 +
                         gate_up_weight_src_offset;
              up_src = (const uint8_t*)config.up_proj + (expert_id * full_weight_elems) / 2 +
                       gate_up_weight_src_offset;
              down_src = (const uint8_t*)config.down_proj + (expert_id * full_weight_elems) / 2;
              gate_scale_src = (const scale_t*)config.gate_scale + expert_id * full_scale_elems +
                               gate_up_scale_src_offset;
              up_scale_src = (const scale_t*)config.up_scale + expert_id * full_scale_elems + gate_up_scale_src_offset;
              down_scale_src = (const scale_t*)config.down_scale + expert_id * full_scale_elems;
            }

            std::memcpy(gate_dst, gate_src, tp_weight_elems / 2);
            std::memcpy(up_dst, up_src, tp_weight_elems / 2);
            std::memcpy(gate_scale_dst, gate_scale_src, sizeof(scale_t) * tp_scale_elems);
            std::memcpy(up_scale_dst, up_scale_src, sizeof(scale_t) * tp_scale_elems);

            const size_t full_down_row_bytes = (size_t)config.intermediate_size / 2;
            const size_t tp_down_row_bytes = (size_t)tpc.intermediate_size / 2;
            const size_t full_down_scale_row = (size_t)config.intermediate_size / group_size;
            const size_t tp_down_scale_row = (size_t)tpc.intermediate_size / group_size;
            for (int row = 0; row < config.hidden_size; ++row) {
              std::memcpy(down_dst + (size_t)row * tp_down_row_bytes,
                          down_src + (size_t)row * full_down_row_bytes + down_weight_src_col_offset / 2,
                          tp_down_row_bytes);
              std::memcpy(down_scale_dst + (size_t)row * tp_down_scale_row,
                          down_scale_src + (size_t)row * full_down_scale_row + down_scale_src_block_k_offset,
                          sizeof(scale_t) * tp_down_scale_row);
            }
          },
          nullptr);
    });

    pool->dispense_backend()->do_numa_job([&, this](int i) { tps[i]->load_weights(); });

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      delete[] (uint8_t*)tpc.gate_proj;
      delete[] (uint8_t*)tpc.up_proj;
      delete[] (uint8_t*)tpc.down_proj;
      delete[] (scale_t*)tpc.gate_scale;
      delete[] (scale_t*)tpc.up_scale;
      delete[] (scale_t*)tpc.down_scale;
    });

    this->weights_loaded = true;
  }

  void write_weight_scale_to_buffer(int gpu_tp_count, int expert_id, const std::vector<uintptr_t>& w13_weight_ptrs,
                                    const std::vector<uintptr_t>& w13_scale_ptrs,
                                    const std::vector<uintptr_t>& w2_weight_ptrs,
                                    const std::vector<uintptr_t>& w2_scale_ptrs) {
    if (this->weights_loaded == false) throw std::runtime_error("Not Loaded");
    if (this->tps.empty()) throw std::runtime_error("No TP parts initialized");
    if ((int)w13_weight_ptrs.size() != gpu_tp_count || (int)w13_scale_ptrs.size() != gpu_tp_count ||
        (int)w2_weight_ptrs.size() != gpu_tp_count || (int)w2_scale_ptrs.size() != gpu_tp_count) {
      throw std::runtime_error("Pointer arrays size must match gpu_tp_count");
    }

    this->config.pool->dispense_backend()->do_numa_job([&, this](int i) {
      this->tps[i]->write_weights_to_buffer(gpu_tp_count, this->tp_count, expert_id, this->config, w13_weight_ptrs,
                                            w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs);
    });
  }
};

#endif  // CPUINFER_OPERATOR_AVX2_MXFP4_MOE_H
