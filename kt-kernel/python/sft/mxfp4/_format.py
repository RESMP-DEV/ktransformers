# MXFP4 scalar format reference — E2M1 FP4 + E8M0 block scale
# SPDX-License-Identifier: Apache-2.0

"""
MXFP4 (Microscaling FP4) scalar reference implementation.

Format
------
Element format  : E2M1  – 2 exponent bits, 1 mantissa bit, sign bit.
                   Values: ±0, ±2, ±3, ±4, ±6, ±8, ±12, ±24 (inf/NaN omitted for
                   simplicity; ±24 is the largest finite magnitude.)
Block scale     : E8M0  – 8-bit unsigned shared exponent, bias 127.
                   Decode: ``2 ** (raw - 127)``.
                   ``raw == 0`` encodes a zero block (scale = 0).
                   ``raw == 255`` is reserved.
Block size      : 32 elements share one E8M0 scale (default).

Packing contract
----------------
* ``packed`` (uint8 array, length = N // 2):
  low  nibble (bits 0-3)  = element at index  2k
  high nibble (bits 4-7)  = element at index  2k+1

* ``block_scales`` (uint8 array, length = ceil(N / block_size)):
  One E8M0 byte per block of ``block_size`` elements.

Dequantized value
-----------------
    value[i] = decode_e2m1(nibble[i]) * decode_e8m0(block_scale[i // block_size])
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default microscaling block size (32 elements share one E8M0 scale).
MXFP4_BLOCK_SIZE: int = 32

#: E2M1 maximum finite magnitude (±6 * 2**2 = ±24; but actual max in
#: E2M1 encoding is 24.0 with exp=3, mantissa=1).
MXFP4_E2M1_MAX: float = 6.0  # max *fractional* magnitude per scale unit

#: E8M0 bias (same as IEEE-32 bias).
MXFP4_E8M0_BIAS: int = 127

# Pre-built E2M1 decode table (index = nibble 0..15)
# Layout: sign(1) | exp(2) | mantissa(1)
#   exp 00, m 0 -> 0    exp 00, m 1 -> 2
#   exp 01, m 0 -> 4    exp 01, m 1 -> 6
#   exp 10, m 0 -> 8    exp 10, m 1 -> 12
#   exp 11, m 0 -> Inf  exp 11, m 1 -> NaN
# We map both Inf and NaN to 0.0 to keep the table finite.
# Positive values (nibble & ~8):
#   0b0000 -> 0.0   0b0001 -> 2.0   0b0010 -> 4.0   0b0011 -> 6.0
#   0b0100 -> 8.0   0b0101 -> 12.0  0b0110 -> 0.0   0b0111 -> 0.0
# (nibbles 6 and 7 are Inf/NaN -> treated as 0 for quantization purposes)
_E2M1_DECODE_TABLE: list[float] = [
    0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 0.0, 0.0,   # positive (0..7)
    0.0, -2.0, -4.0, -6.0, -8.0, -12.0, 0.0, 0.0,  # negative (8..15)
]

# Pre-built E2M1 *absolute* decode for quantization:
# Maps (exp_bits, mantissa_bit) -> positive value before sign
#   00, 0 -> 0    00, 1 -> 2
#   01, 0 -> 4    01, 1 -> 6
#   10, 0 -> 8    10, 1 -> 12
#   11, 0 -> Inf  11, 1 -> NaN   (both clamp to 12 for quantization)
_E2M1_ABS: list[float] = [0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 12.0, 12.0]

# ---------------------------------------------------------------------------
# Single-element encode / decode
# ---------------------------------------------------------------------------


def encode_e2m1(value: float) -> int:
    """Encode a single float to an E2M1 FP4 nibble (4-bit unsigned, 0..15).

    The sign bit is the MSB (bit 3).  The two exponent bits are bits 2-1.
    The mantissa bit is bit 0.

    Quantization uses nearest-neighbor rounding across the 6 non-zero
    positive absolute magnitudes: {2, 4, 6, 8, 12}.  Zero maps to 0.
    Values exceeding ±12 clamp to ±12 (the largest finite E2M1 magnitude).

    Returns:
        int in [0, 15].
    """
    if value == 0.0:
        return 0

    sign = 0
    mag = value
    if mag < 0:
        sign = 8
        mag = -mag

    # Find nearest representable magnitude
    best_nibble = 0
    best_dist = float("inf")
    for nib in range(6):  # 1..5 are the non-zero positive nibbles
        candidate = _E2M1_ABS[nib]
        dist = abs(mag - candidate)
        if dist < best_dist:
            best_dist = dist
            best_nibble = nib

    return sign | best_nibble


def decode_e2m1(nibble: int) -> float:
    """Decode a single E2M1 FP4 nibble to a float.

    Nibbles representing Inf (0b0110 / 0b1110) or NaN (0b0111 / 0b1111)
    decode to 0.0 in this reference (consistent with quantization clamp).

    Args:
        nibble: int in [0, 15].

    Returns:
        float.
    """
    return _E2M1_DECODE_TABLE[nibble & 0xF]


def encode_e8m0(scale: float) -> int:
    """Encode a positive float block-scale to an E8M0 uint8.

    E8M0 is an 8-bit unsigned shared exponent with bias 127:
        decoded = 2 ** (raw - 127)

    Encoding finds the exponent *e* such that ``2**e <= scale < 2**(e+1)``,
    then stores ``e + 127``.  Zero scale encodes as raw=0.
    Scale <= 0 (including negative) encodes as raw=0 (zero block).

    Args:
        scale: positive float (block absolute-max / E2M1_MAX).

    Returns:
        int in [0, 255].
    """
    if scale <= 0.0:
        return 0
    e = int(math.floor(math.log2(scale)))
    # Clamp to representable range [1, 254]
    raw = e + MXFP4_E8M0_BIAS
    if raw < 1:
        return 1  # smallest positive exponent
    if raw > 254:
        return 254  # largest non-reserved exponent
    return raw


def decode_e8m0(raw: int) -> float:
    """Decode an E8M0 uint8 to a float block-scale.

    ``raw == 0`` → scale = 0.0 (zero block).
    ``raw == 255`` is reserved; returns 0.0.

    Args:
        raw: int in [0, 255].

    Returns:
        float.
    """
    if raw == 0 or raw == 255:
        return 0.0
    return 2.0 ** (raw - MXFP4_E8M0_BIAS)


# ---------------------------------------------------------------------------
# Nibble pack / unpack
# ---------------------------------------------------------------------------


def pack_nibbles(nibbles: Tensor) -> Tensor:
    """Pack a 1-D uint8 nibble tensor (values 0..15) into packed uint8 bytes.

    Each output byte holds two nibbles:
        low  nibble = input[2k]
        high nibble = input[2k+1]

    Args:
        nibbles: 1-D uint8 tensor of even length, values in [0, 15].

    Returns:
        1-D uint8 tensor of length ``nibbles.numel() // 2``.
    """
    assert nibbles.ndim == 1, f"Expected 1-D tensor, got {nibbles.ndim}-D"
    assert nibbles.numel() % 2 == 0, "Nibble count must be even"
    n = nibbles.numel()
    lo = nibbles[0::2] & 0xF
    hi = nibbles[1::2] & 0xF
    return (hi << 4) | lo


def unpack_nibbles(packed: Tensor) -> Tensor:
    """Unpack a 1-D uint8 packed-nibble tensor into individual nibbles.

    Inverse of :func:`pack_nibbles`.

    Args:
        packed: 1-D uint8 tensor.

    Returns:
        1-D uint8 tensor of length ``2 * packed.numel()``, values in [0, 15].
    """
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    # Interleave: output[2k] = lo[k], output[2k+1] = hi[k]
    out = torch.stack([lo, hi], dim=1).reshape(-1)
    return out


# ---------------------------------------------------------------------------
# Full pack / unpack / dequant
# ---------------------------------------------------------------------------


def pack_fp32_to_mxfp4(
    values: Tensor,
    block_size: int = MXFP4_BLOCK_SIZE,
) -> Tuple[Tensor, Tensor]:
    """Quantize an FP32 tensor to MXFP4 and return packed weights + block scales.

    Args:
        values: 1-D or 2-D FP32 tensor.  2-D is treated as rows of elements;
            each row is quantized independently (block scales are per-row).
        block_size: number of consecutive elements sharing one E8M0 scale.

    Returns:
        (packed, block_scales) where:
          - ``packed`` is a uint8 tensor with the same leading dims as input,
            last dim halved (two nibbles per byte).
          - ``block_scales`` is a uint8 tensor of E8M0 exponents, shape
            ``(*leading, n_cols // block_size)`` for 2-D or ``(N // block_size,)``
            for 1-D.
    """
    if values.ndim == 1:
        return _pack_1d(values, block_size)
    elif values.ndim == 2:
        return _pack_2d(values, block_size)
    else:
        raise ValueError(f"Expected 1-D or 2-D tensor, got {values.ndim}-D")


def _pack_1d(values: Tensor, block_size: int) -> Tuple[Tensor, Tensor]:
    """Pack a 1-D FP32 tensor to MXFP4."""
    N = values.numel()
    assert N % block_size == 0, (
        f"Vector length {N} is not a multiple of block_size {block_size}"
    )
    n_blocks = N // block_size

    # Reshape into blocks [n_blocks, block_size]
    vals = values.reshape(n_blocks, block_size)

    # Compute per-block absolute max
    amax = vals.abs().max(dim=1).values  # [n_blocks]

    # Block scale = amax / E2M1_MAX  (so max element maps to ±12)
    block_scale_val = amax / MXFP4_E2M1_MAX
    # Avoid division by zero for zero blocks
    block_scale_val = torch.where(amax == 0, torch.zeros_like(block_scale_val), block_scale_val)

    # Encode block scales to E8M0
    block_scales_raw = torch.tensor(
        [encode_e8m0(s.item()) for s in block_scale_val],
        dtype=torch.uint8,
    )

    # Decode block scales back to float for quantization
    block_scale_decoded = torch.tensor(
        [decode_e8m0(r.item()) for r in block_scales_raw],
        dtype=torch.float32,
    )

    # Normalize values by block scale
    # For zero blocks (scale=0), all nibbles should be 0
    safe_scale = torch.where(block_scale_decoded == 0, torch.ones_like(block_scale_decoded), block_scale_decoded)
    normalized = vals / safe_scale.unsqueeze(1)

    # Quantize each element to E2M1
    nibbles = torch.tensor(
        [encode_e2m1(v.item()) for v in normalized.reshape(-1)],
        dtype=torch.uint8,
    )

    # Pack nibbles into bytes
    packed = pack_nibbles(nibbles)

    # Reshape packed to [n_blocks, block_size // 2]
    packed = packed.reshape(n_blocks, block_size // 2)

    # Flatten back
    packed = packed.reshape(-1)

    return packed, block_scales_raw


def _pack_2d(values: Tensor, block_size: int) -> Tuple[Tensor, Tensor]:
    """Pack a 2-D FP32 tensor to MXFP4 (per-row blocks along columns)."""
    n_rows, n_cols = values.shape
    assert n_cols % block_size == 0, (
        f"Column count {n_cols} is not a multiple of block_size {block_size}"
    )
    n_blocks_per_row = n_cols // block_size

    # Reshape: [n_rows, n_blocks_per_row, block_size]
    vals = values.reshape(n_rows, n_blocks_per_row, block_size)

    # Per-block absolute max: [n_rows, n_blocks_per_row]
    amax = vals.abs().max(dim=2).values

    # Block scale
    block_scale_val = amax / MXFP4_E2M1_MAX
    block_scale_val = torch.where(amax == 0, torch.zeros_like(block_scale_val), block_scale_val)

    # Encode to E8M0
    block_scales_raw = torch.tensor(
        [[encode_e8m0(s.item()) for s in row] for row in block_scale_val],
        dtype=torch.uint8,
    )

    # Decode back for quantization
    block_scale_decoded = torch.tensor(
        [[decode_e8m0(int(r)) for r in row] for row in block_scales_raw],
        dtype=torch.float32,
    )

    # Normalize
    safe_scale = torch.where(block_scale_decoded == 0, torch.ones_like(block_scale_decoded), block_scale_decoded)
    normalized = vals / safe_scale.unsqueeze(2)

    # Quantize to E2M1 nibbles
    nibbles = torch.tensor(
        [encode_e2m1(v.item()) for v in normalized.reshape(-1)],
        dtype=torch.uint8,
    ).reshape(n_rows, n_cols)

    # Pack each row
    packed_rows = []
    for r in range(n_rows):
        row_packed = pack_nibbles(nibbles[r])
        packed_rows.append(row_packed)

    packed = torch.stack(packed_rows, dim=0)  # [n_rows, n_cols // 2]
    return packed, block_scales_raw


def unpack_mxfp4_to_fp32(
    packed: Tensor,
    block_scales: Tensor,
    block_size: int = MXFP4_BLOCK_SIZE,
) -> Tensor:
    """Unpack MXFP4 packed data + E8M0 block scales back to FP32.

    Args:
        packed: uint8 tensor of packed nibbles.
            - 1-D: shape ``(N_packed,)`` where original length was ``2 * N_packed``.
            - 2-D: shape ``(n_rows, n_cols_packed)`` where original cols = ``2 * n_cols_packed``.
        block_scales: uint8 tensor of E8M0 exponents.
            - 1-D: shape ``(n_blocks,)``.
            - 2-D: shape ``(n_rows, n_blocks_per_row)``.
        block_size: number of elements per block scale.

    Returns:
        FP32 tensor with the same leading dims as ``packed``, last dim doubled.
    """
    if packed.ndim == 1:
        return _unpack_1d(packed, block_scales, block_size)
    elif packed.ndim == 2:
        return _unpack_2d(packed, block_scales, block_size)
    else:
        raise ValueError(f"Expected 1-D or 2-D packed tensor, got {packed.ndim}-D")


def _unpack_1d(packed: Tensor, block_scales: Tensor, block_size: int) -> Tensor:
    n_elements = packed.numel() * 2
    nibbles = unpack_nibbles(packed)

    n_blocks = block_scales.numel()
    # Decode each nibble to E2M1 float
    e2m1_vals = torch.tensor(
        [decode_e2m1(int(n)) for n in nibbles],
        dtype=torch.float32,
    )

    # Decode block scales
    scale_vals = torch.tensor(
        [decode_e8m0(int(s)) for s in block_scales],
        dtype=torch.float32,
    )

    # Broadcast: each block of block_size elements shares one scale
    scale_expanded = scale_vals.repeat_interleave(block_size)

    return e2m1_vals * scale_expanded


def _unpack_2d(packed: Tensor, block_scales: Tensor, block_size: int) -> Tensor:
    n_rows = packed.shape[0]
    n_cols_packed = packed.shape[1]
    n_cols = n_cols_packed * 2
    n_blocks_per_row = block_scales.shape[1]

    # Unpack each row
    row_outputs = []
    for r in range(n_rows):
        nibbles = unpack_nibbles(packed[r])

        # Decode E2M1
        e2m1_vals = torch.tensor(
            [decode_e2m1(int(n)) for n in nibbles],
            dtype=torch.float32,
        )

        # Decode block scales for this row
        scale_vals = torch.tensor(
            [decode_e8m0(int(s)) for s in block_scales[r]],
            dtype=torch.float32,
        )

        # Expand scales
        scale_expanded = scale_vals.repeat_interleave(block_size)

        row_outputs.append(e2m1_vals * scale_expanded)

    return torch.stack(row_outputs, dim=0)


def dequant_mxfp4(
    packed: Tensor,
    block_scales: Tensor,
    block_size: int = MXFP4_BLOCK_SIZE,
    *,
    additional_scale: Tensor | None = None,
) -> Tensor:
    """Dequantize MXFP4 to FP32 with optional additional per-tensor scale.

    This is the main entry point used by KT-SFT GPU kernels.  It performs
    ``unpack`` → ``E2M1 decode`` → ``E8M0 decode`` → ``optional scaling``.

    Args:
        packed: uint8 packed MXFP4 data (1-D or 2-D).
        block_scales: uint8 E8M0 exponents.
        block_size: elements per block scale (default 32).
        additional_scale: optional tensor multiplied element-wise after
            block-scale dequantization.  Used when the calling kernel needs
            to combine the MX scale with a second scale factor (e.g. a
            global scale or per-channel scale).

    Returns:
        FP32 tensor.
    """
    result = unpack_mxfp4_to_fp32(packed, block_scales, block_size)
    if additional_scale is not None:
        result = result * additional_scale
    return result
