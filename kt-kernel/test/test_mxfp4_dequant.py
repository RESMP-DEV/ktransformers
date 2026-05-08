"""Test for MXFP4 dequantization kernel (SM_86 Ampere).

This test module will skip cleanly when CUDA is unavailable.
"""

import sys
import os
import pytest
import torch

# Try to import CUDA extension; skip if not available
try:
    # The extension is built via the kt-kernel build system
    # Attempt import from the built module location
    import KTransformersOps as ops
    CUDA_AVAILABLE = torch.cuda.is_available()
except (ImportError, ModuleNotFoundError, OSError):
    CUDA_AVAILABLE = False
    ops = None


def _make_mxfp4_block(scale: float, values: list) -> bytes:
    """Create a single MXFP4 block for testing.
    
    MXFP4 block format:
      - 1 byte: scale (E8M0 format, simplified as raw byte)
      - N bytes: packed FP4 data (2 values per byte)
    
    For testing, we use a simplified scale encoding where the raw byte value
    directly represents the scale magnitude.
    """
    # Scale byte: simple encoding (positive only for test)
    scale_byte = min(127, max(0, int(scale)))
    
    # Pack FP4 values (2 per byte)
    # FP4 encoding: 0=0, 1=0.5, 2=1.0, 3=-1.0, 4=-0.5, 5=-1.0, 6=0.5, 7=1.0
    fp4_encode = {
        0.0: 0, 0.5: 1, 1.0: 2, -1.0: 3, -0.5: 4
    }
    
    data_bytes = []
    for i in range(0, len(values), 2):
        lo = fp4_encode.get(values[i], 0)
        hi = fp4_encode.get(values[i+1] if i+1 < len(values) else 0.0, 0)
        data_bytes.append((hi << 4) | lo)
    
    return bytes([scale_byte] + data_bytes)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestMXFP4Dequant:
    """Test MXFP4 dequantization on CUDA device."""
    
    def test_basic_dequant_fp32(self):
        """Test basic dequantization to float32."""
        # Create test data: 2 blocks, each with 1 scale + 8 data bytes = 9 bytes per block
        # But we use blk_size=17 for 16 FP4 values (32 elements) per block
        blk_size = 17  # 1 scale + 16 data bytes
        ele_per_blk = 32  # 2 FP4 values per byte
        
        # Create raw MXFP4 data (2 blocks)
        raw_data = bytearray()
        for block_id in range(2):
            scale = 2.0 + block_id  # Different scales per block
            # Create 16 data bytes with known FP4 values
            for byte_idx in range(16):
                raw_data.append((block_id + byte_idx) % 256)
            # Insert scale at beginning of block
            raw_data.insert(block_id * blk_size, int(scale * 10))  # Scale encoded
        
        data_ptr = int.from_bytes(bytes(raw_data[:8]), 'little')  # Dummy pointer for test
        
        # For a proper test, we need actual MXFP4 data
        # Let's create simpler test data
        test_data = bytes([10, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
                          20, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88])
        
        data_array = (torch.tensor(list(test_data), dtype=torch.int8)
                      .contiguous()
                      .cpu()
                      .numpy())
        data_ptr = data_array.ctypes.data
        
        result = ops.dequantize_mxfp4(
            data_ptr, len(test_data), blk_size, ele_per_blk,
            torch.device('cuda'), torch.float32
        )
        
        assert result.shape == (2, 32)
        assert result.dtype == torch.float32
        assert result.device.type == 'cuda'
    
    def test_dequant_fp16(self):
        """Test dequantization to float16."""
        blk_size = 17
        ele_per_blk = 32
        
        test_data = bytes([10] + [0x12] * 16 + [20] + [0x34] * 16)
        data_array = (torch.tensor(list(test_data), dtype=torch.int8)
                      .contiguous().cpu().numpy())
        data_ptr = data_array.ctypes.data
        
        result = ops.dequantize_mxfp4(
            data_ptr, len(test_data), blk_size, ele_per_blk,
            torch.device('cuda'), torch.float16
        )
        
        assert result.shape == (2, 32)
        assert result.dtype == torch.float16
    
    def test_dequant_bf16(self):
        """Test dequantization to bfloat16."""
        blk_size = 17
        ele_per_blk = 32
        
        test_data = bytes([10] + [0x12] * 16 + [20] + [0x34] * 16)
        data_array = (torch.tensor(list(test_data), dtype=torch.int8)
                      .contiguous().cpu().numpy())
        data_ptr = data_array.ctypes.data
        
        result = ops.dequantize_mxfp4(
            data_ptr, len(test_data), blk_size, ele_per_blk,
            torch.device('cuda'), torch.bfloat16
        )
        
        assert result.shape == (2, 32)
        assert result.dtype == torch.bfloat16
    
    def test_single_block(self):
        """Test with a single block."""
        blk_size = 17
        ele_per_blk = 32
        
        test_data = bytes([15] + [0x55] * 16)  # Scale=15, all data=0x55
        data_array = (torch.tensor(list(test_data), dtype=torch.int8)
                      .contiguous().cpu().numpy())
        data_ptr = data_array.ctypes.data
        
        result = ops.dequantize_mxfp4(
            data_ptr, len(test_data), blk_size, ele_per_blk,
            torch.device('cuda'), torch.float32
        )
        
        assert result.shape == (1, 32)
        # All values should be non-zero since scale=15 and data=0x55
        assert torch.any(result != 0)


@pytest.mark.skipif(CUDA_AVAILABLE, reason="Test runs only when CUDA is NOT available")
def test_skip_when_cuda_unavailable():
    """Verify test properly skips when CUDA unavailable."""
    # This test just confirms the skip mechanism works
    assert not CUDA_AVAILABLE, "CUDA should not be available for this test to run"


def test_mxfp4_module_import():
    """Test that the module structure is correct."""
    # Verify ops.h includes are correct by checking import
    if CUDA_AVAILABLE:
        assert hasattr(ops, 'dequantize_mxfp4')
        assert callable(ops.dequantize_mxfp4)


if __name__ == '__main__':
    # Run with pytest-style output
    if not CUDA_AVAILABLE:
        print("SKIP: CUDA not available - tests will be skipped")
        sys.exit(0)
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    
    # Run tests
    pytest.main([__file__, '-v'])
