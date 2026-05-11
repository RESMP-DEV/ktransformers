"""Static coverage for offline FP8 to SmoothQuant INT8 conversion helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_converter_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "convert_fp8_smoothquant_int8.py"
    spec = importlib.util.spec_from_file_location("convert_fp8_smoothquant_int8", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_block_scale_expansion_matches_deepseek_fp8_layout():
    converter = _load_converter_module()
    scale = torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float32)

    expanded = converter._expand_scale(scale, out_dim=2, in_dim=4)

    torch.testing.assert_close(
        expanded,
        torch.tensor(
            [
                [10.0, 10.0, 20.0, 20.0],
                [30.0, 30.0, 40.0, 40.0],
            ],
            dtype=torch.float32,
        ),
    )


def test_dequantize_weight_applies_scale_before_int8_calibration():
    converter = _load_converter_module()
    weight = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    scale = torch.tensor([[0.5, 2.0]], dtype=torch.float32)

    dequantized = converter._dequantize_weight_for_int8(weight, scale)

    torch.testing.assert_close(
        dequantized,
        torch.tensor([[0.5, 1.0, 6.0, 8.0]], dtype=torch.float32),
    )


def test_scale_key_candidates_default_to_deepseek_block_scale_first():
    converter = _load_converter_module()

    assert converter._scale_key_candidates("model.layers.0.foo.weight", None)[:2] == [
        "model.layers.0.foo.weight_scale_inv",
        "model.layers.0.foo.weight_scale",
    ]
    assert converter._scale_key_candidates("model.layers.0.foo.weight", "scale") == [
        "model.layers.0.foo.scale"
    ]


def test_artifact_keys_distinguish_smoothquant_and_mr_gptq():
    converter = _load_converter_module()

    smooth_keys = converter._artifact_keys("model.layers.0.foo.weight", "smoothquant")
    mr_keys = converter._artifact_keys("model.layers.0.foo.weight", "mr-gptq")

    assert smooth_keys == (
        "model.layers.0.foo.smooth_int8.weight",
        "model.layers.0.foo.smooth_int8.weight_scale",
        "model.layers.0.foo.smooth_int8.smooth_scale",
        None,
        None,
    )
    assert mr_keys == (
        "model.layers.0.foo.mr_gptq_int8.weight",
        "model.layers.0.foo.mr_gptq_int8.weight_scale",
        None,
        "model.layers.0.foo.mr_gptq_int8.activation_order",
        "model.layers.0.foo.mr_gptq_int8.hessian_diag",
    )


def test_optional_hessian_loader_accepts_json_gram(tmp_path):
    converter = _load_converter_module()
    stats = tmp_path / "hessian.json"
    stats.write_text(
        '{"tensors": {"model.layers.0.foo.weight": {"gram": [[1.0, 0.0], [0.0, 2.0]]}}}',
        encoding="utf-8",
    )

    hessians = converter._load_optional_hessian(stats)

    torch.testing.assert_close(
        hessians["model.layers.0.foo.weight"],
        torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32),
    )
