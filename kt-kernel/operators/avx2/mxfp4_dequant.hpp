/**
 * @Description  : MXFP4 E2M1 dequantization helpers for AVX2
 * @Author       : Codex
 * @Date         : 2026-05-08
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * MXFP4 stores two E2M1 FP4 values per byte:
 *   low nibble  = even K element
 *   high nibble = odd K element
 *
 * This header decodes packed E2M1 nibbles to FP32 vectors. Block scales are
 * applied by the GEMM kernel so the decoded vectors can be reused across M rows.
 **/
#ifndef CPUINFER_OPERATOR_AVX2_MXFP4_DEQUANT_H
#define CPUINFER_OPERATOR_AVX2_MXFP4_DEQUANT_H

#include <immintrin.h>

#include <cstdint>

namespace avx2 {

// E2M1 values as BF16 bit patterns:
// {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
// -0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0}
alignas(16) static constexpr uint8_t mxfp4_bf16_lo[16] = {
    0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0,
    0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0};
alignas(16) static constexpr uint8_t mxfp4_bf16_hi[16] = {
    0x00, 0x3F, 0x3F, 0x3F, 0x40, 0x40, 0x40, 0x40,
    0x80, 0xBF, 0xBF, 0xBF, 0xC0, 0xC0, 0xC0, 0xC0};

static inline __m256 bf16x8_bits_to_fp32(__m128i bf16_bits) {
  __m256i i32 = _mm256_cvtepu16_epi32(bf16_bits);
  return _mm256_castsi256_ps(_mm256_slli_epi32(i32, 16));
}

// Decode 8 packed bytes into 16 FP32 values in K order.
static inline void mxfp4x16_to_2xfp32(const uint8_t* src, __m256* out0, __m256* out1) {
  const __m128i packed = _mm_loadl_epi64((const __m128i*)src);
  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m128i lo_nibbles = _mm_and_si128(packed, mask);
  const __m128i hi_nibbles = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);

  const __m128i lut_lo = _mm_load_si128((const __m128i*)mxfp4_bf16_lo);
  const __m128i lut_hi = _mm_load_si128((const __m128i*)mxfp4_bf16_hi);

  const __m128i lo_l = _mm_shuffle_epi8(lut_lo, lo_nibbles);
  const __m128i lo_h = _mm_shuffle_epi8(lut_hi, lo_nibbles);
  const __m128i hi_l = _mm_shuffle_epi8(lut_lo, hi_nibbles);
  const __m128i hi_h = _mm_shuffle_epi8(lut_hi, hi_nibbles);

  const __m128i lo_bf16 = _mm_unpacklo_epi8(lo_l, lo_h);
  const __m128i hi_bf16 = _mm_unpacklo_epi8(hi_l, hi_h);

  const __m128i first8 = _mm_unpacklo_epi16(lo_bf16, hi_bf16);
  const __m128i second8 = _mm_unpackhi_epi16(lo_bf16, hi_bf16);

  *out0 = bf16x8_bits_to_fp32(first8);
  *out1 = bf16x8_bits_to_fp32(second8);
}

static inline float mxfp4_to_fp32_scalar(uint8_t packed, int k_in_byte) {
  const uint8_t nibble = (packed >> (k_in_byte * 4)) & 0x0F;
  const uint16_t bits = (uint16_t)mxfp4_bf16_lo[nibble] | ((uint16_t)mxfp4_bf16_hi[nibble] << 8);
  union {
    uint32_t u;
    float f;
  } v;
  v.u = (uint32_t)bits << 16;
  return v.f;
}

}  // namespace avx2

#endif  // CPUINFER_OPERATOR_AVX2_MXFP4_DEQUANT_H
