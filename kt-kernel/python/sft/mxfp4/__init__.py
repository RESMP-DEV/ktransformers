# MXFP4 (Microscaling FP4) format reference for KT-SFT
# SPDX-License-Identifier: Apache-2.0

"""
MXFP4 scalar reference and packing contract for KT-SFT.

Implements E2M1 FP4 with E8M0 block-scale microscaling, following the
OCP MX specification.  Block size defaults to 32 elements sharing one
shared exponent.

Packing layout
--------------
packed : uint8 array
    Each byte holds two FP4 nibbles (low nibble = element *2k*,
    high nibble = element *2k+1*).
block_scales : uint8 array
    E8M0 shared exponents, one per block of ``block_size`` elements.
    Decode: ``2 ** (exp - 127)``;  ``exp == 0`` means the entire block
    is zero, ``exp == 255`` is reserved.

Dequantization
--------------
``value = fp4_e2m1_decode(nibble) * e8m0_decode(shared_exp)``

Public API
----------
- :func:`pack_fp32_to_mxfp4`  – quantize + pack
- :func:`unpack_mxfp4_to_fp32` – unpack + dequantize in one step
- :func:`dequant_mxfp4`        – dequantize with explicit block-scale args
- :func:`pack_nibbles`         – raw nibble packing
- :func:`unpack_nibbles`       – raw nibble unpacking
- :func:`encode_e2m1`         – single float → 4-bit E2M1 nibble
- :func:`decode_e2m1`         – single 4-bit E2M1 nibble → float
- :func:`encode_e8m0`         – block-scale float → E8M0 uint8
- :func:`decode_e8m0`         – E8M0 uint8 → block-scale float
"""

from ._format import (
    MXFP4_BLOCK_SIZE,
    MXFP4_E2M1_MAX,
    MXFP4_E8M0_BIAS,
    encode_e2m1,
    decode_e2m1,
    encode_e8m0,
    decode_e8m0,
    pack_nibbles,
    unpack_nibbles,
    pack_fp32_to_mxfp4,
    unpack_mxfp4_to_fp32,
    dequant_mxfp4,
)

__all__ = [
    "MXFP4_BLOCK_SIZE",
    "MXFP4_E2M1_MAX",
    "MXFP4_E8M0_BIAS",
    "encode_e2m1",
    "decode_e2m1",
    "encode_e8m0",
    "decode_e8m0",
    "pack_nibbles",
    "unpack_nibbles",
    "pack_fp32_to_mxfp4",
    "unpack_mxfp4_to_fp32",
    "dequant_mxfp4",
]
