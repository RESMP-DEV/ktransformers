"""Tests for the SM86 FP8 E4M3FN linear CUDA op."""

from __future__ import annotations

import sys

import pytest
import torch

try:
    ops = __import__("KTransformersOps")
    CUDA_AVAILABLE = torch.cuda.is_available()
except (ImportError, ModuleNotFoundError, OSError):
    CUDA_AVAILABLE = False
    ops = None


def _decode_e4m3fn_byte(value: int) -> float:
    sign = -1.0 if value & 0x80 else 1.0
    exp = (value >> 3) & 0x0F
    mant = value & 0x07
    if exp == 0:
        if mant == 0:
            return -0.0 if sign < 0 else 0.0
        return sign * (mant / 8.0) * (2.0**-6)
    if exp == 0x0F and mant == 0x07:
        return 0.0
    return sign * (1.0 + mant / 8.0) * (2.0 ** (exp - 7))


def _decode_fp8_weight_reference(weight_bytes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    packed = weight_bytes.cpu()
    table = torch.tensor([_decode_e4m3fn_byte(i) for i in range(256)], dtype=torch.float32)
    decoded = table[packed.long()]
    scale_f = scales.cpu().float().repeat_interleave(128, dim=0)
    scale_f = scale_f.repeat_interleave(128, dim=1)[: decoded.shape[0], : decoded.shape[1]]
    return decoded * scale_f


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA extension not available")
def test_fp8_linear_matches_reference_fp32():
    weight_bytes = torch.tensor(
        [
            [0x38, 0x40, 0xC0, 0x30, 0x20, 0xB8, 0x48, 0x10],
            [0x3C, 0xBC, 0x44, 0xC4, 0x28, 0xA8, 0x50, 0xD0],
            [0x00, 0x80, 0x08, 0x88, 0x7E, 0xFE, 0x70, 0xF0],
        ],
        dtype=torch.uint8,
        device="cuda",
    )
    scales = torch.tensor([[1.0]], dtype=torch.float32, device="cuda")
    x = torch.linspace(-1.25, 1.25, steps=16, dtype=torch.float32, device="cuda").reshape(2, 8)

    result = ops.fp8_linear(x, weight_bytes, scales)
    decoded_weight = _decode_fp8_weight_reference(weight_bytes, scales)
    expected = x.cpu().float() @ decoded_weight.T

    assert result.shape == (2, 3)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result.cpu(), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA extension not available")
def test_fp8_linear_matches_torch_float8_decode():
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        pytest.skip("torch.float8_e4m3fn is not available")

    base = torch.linspace(-2.0, 2.0, steps=256, dtype=torch.float32, device="cuda").reshape(4, 64)
    weight_fp8 = base.to(fp8_dtype)
    weight_bytes = weight_fp8.view(torch.uint8)
    scales = torch.tensor([[0.5]], dtype=torch.float32, device="cuda")
    x = torch.randn(3, 64, dtype=torch.float32, device="cuda")

    result = ops.fp8_linear(x, weight_bytes, scales)
    expected = x.float() @ (weight_fp8.float() * scales[0, 0]).T

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(CUDA_AVAILABLE, reason="Test runs only when CUDA is NOT available")
def test_skip_when_cuda_unavailable():
    assert not CUDA_AVAILABLE


def test_fp8_module_import():
    if CUDA_AVAILABLE:
        assert hasattr(ops, "fp8_linear")
        assert callable(ops.fp8_linear)


if __name__ == "__main__":
    if not CUDA_AVAILABLE:
        print("SKIP: CUDA not available - tests will be skipped")
        sys.exit(0)
    pytest.main([__file__, "-v"])
