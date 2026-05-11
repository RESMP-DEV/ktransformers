"""Coverage for SmoothQuant W8A8 artifact helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def _load_smoothquant_module():
    module_path = Path(__file__).resolve().parents[1] / "python" / "utils" / "smoothquant.py"
    spec = importlib.util.spec_from_file_location("kt_smoothquant_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["kt_smoothquant_test"] = module
    spec.loader.exec_module(module)
    return module


def test_calibrate_weight_int8_smoothquant_round_trips_formula():
    smoothquant = _load_smoothquant_module()
    weight = torch.tensor([[1.0, -2.0, 0.5], [-4.0, 2.0, 1.0]], dtype=torch.float32)
    activation_amax = torch.tensor([2.0, 8.0, 4.0], dtype=torch.float32)

    params = smoothquant.calibrate_weight_int8_smoothquant(
        weight,
        activation_amax,
        smoothquant.SmoothQuantConfig(alpha=0.5),
    )

    x = torch.tensor([[0.5, -1.0, 2.0]], dtype=torch.float32)
    actual = torch.nn.functional.linear(
        x / params.smooth_scale.reshape(1, -1),
        params.weight_q.float() * params.weight_scale.reshape(-1, 1),
    )
    expected = torch.nn.functional.linear(x, weight)

    assert params.weight_q.dtype == torch.int8
    assert params.weight_scale.shape == (2,)
    assert params.smooth_scale.shape == (3,)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)


def test_apply_block_hadamard_is_orthogonal_on_full_blocks():
    smoothquant = _load_smoothquant_module()
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float32)

    rotated = smoothquant.apply_block_hadamard(x, 4)
    restored = smoothquant.apply_block_hadamard(rotated, 4)

    torch.testing.assert_close(restored, x, rtol=1e-6, atol=1e-6)


def test_calibrate_weight_int8_mr_gptq_emits_group_scales_and_rotation_metadata():
    smoothquant = _load_smoothquant_module()
    weight = torch.tensor([[1.0, -2.0, 0.5, 3.0], [-4.0, 2.0, 1.0, -0.5]], dtype=torch.float32)
    activation_amax = torch.tensor([2.0, 8.0, 4.0, 1.0], dtype=torch.float32)

    params = smoothquant.calibrate_weight_int8_mr_gptq(
        weight,
        activation_amax,
        smoothquant.MRGPTQInt8Config(rotation_block_size=2, scale_grid_steps=1),
    )

    x = torch.tensor([[0.5, -1.0, 2.0, 3.0]], dtype=torch.float32)
    x_rot = smoothquant.apply_block_hadamard(x, 2)
    scale = params.weight_scale.repeat_interleave(2, dim=1)[:, : weight.shape[1]]
    actual = torch.nn.functional.linear(x_rot, params.weight_q.float() * scale)
    expected = torch.nn.functional.linear(x, weight)

    assert params.weight_q.dtype == torch.int8
    assert params.weight_scale.shape == (2, 2)
    assert params.rotation_block_size == 2
    assert params.gptq_error_propagation == "not_applied_activation_amax_only"
    assert params.activation_order.tolist() == [1, 2, 0, 3]
    torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)


def test_calibrate_weight_int8_mr_gptq_uses_hessian_when_supplied():
    smoothquant = _load_smoothquant_module()
    weight = torch.tensor([[1.0, -2.0, 0.5, 3.0], [-4.0, 2.0, 1.0, -0.5]], dtype=torch.float32)
    activation_amax = torch.tensor([2.0, 8.0, 4.0, 1.0], dtype=torch.float32)
    hessian = torch.tensor(
        [
            [3.0, 0.2, 0.0, 0.0],
            [0.2, 9.0, 0.1, 0.0],
            [0.0, 0.1, 4.0, 0.3],
            [0.0, 0.0, 0.3, 2.0],
        ],
        dtype=torch.float32,
    )

    params = smoothquant.calibrate_weight_int8_mr_gptq(
        weight,
        activation_amax,
        smoothquant.MRGPTQInt8Config(rotation_block_size=2, scale_grid_steps=1, gptq_block_size=2),
        hessian=hessian,
    )

    assert params.gptq_error_propagation == "applied"
    assert params.weight_q.dtype == torch.int8
    assert params.weight_scale.shape == (2, 2)
    assert params.activation_order.shape == (4,)
