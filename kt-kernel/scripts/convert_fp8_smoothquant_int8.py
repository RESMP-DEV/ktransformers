#!/usr/bin/env python3
"""Convert FP8 safetensors weights into offline W8A8 INT8 artifacts.

The DeepSeek FP8 checkpoint format stores FP8 weight bytes and block scales as
separate tensors. This script dequantizes those weights once, applies either
SmoothQuant or an MR-GPTQ-inspired micro-rotated INT8 layout, and writes
reusable artifacts consumed by `moe_reap_calibration_multi_gpu.py`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from kt_kernel.utils.smoothquant import (
        MRGPTQInt8Config,
        SmoothQuantConfig,
        calibrate_weight_int8_mr_gptq,
        calibrate_weight_int8_smoothquant,
    )
except ModuleNotFoundError:
    _spec = importlib.util.spec_from_file_location(
        "kt_smoothquant_local",
        ROOT / "python" / "utils" / "smoothquant.py",
    )
    if _spec is None or _spec.loader is None:
        raise
    _smoothquant = importlib.util.module_from_spec(_spec)
    sys.modules["kt_smoothquant_local"] = _smoothquant
    _spec.loader.exec_module(_smoothquant)
    MRGPTQInt8Config = _smoothquant.MRGPTQInt8Config
    SmoothQuantConfig = _smoothquant.SmoothQuantConfig
    calibrate_weight_int8_mr_gptq = _smoothquant.calibrate_weight_int8_mr_gptq
    calibrate_weight_int8_smoothquant = _smoothquant.calibrate_weight_int8_smoothquant


MANIFEST_BY_METHOD = {
    "smoothquant": "smoothquant-int8-manifest.json",
    "mr-gptq": "mr-gptq-int8-manifest.json",
}

SCHEME_BY_METHOD = {
    "smoothquant": "w8a8_smoothquant",
    "mr-gptq": "w8a8_mr_gptq_int8",
}


def _iter_safetensors(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.safetensors"))


def _load_stats(path: Path) -> dict[str, torch.Tensor]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    tensors = raw.get("tensors", raw)
    stats: dict[str, torch.Tensor] = {}
    for name, value in tensors.items():
        amax = value.get("amax") if isinstance(value, dict) else value
        stats[name] = torch.tensor(amax, dtype=torch.float32)
    return stats


def _load_optional_hessian(path: Path | None) -> dict[str, torch.Tensor]:
    if path is None:
        return {}
    if path.suffix in {".pt", ".pth"}:
        raw = torch.load(path, map_location="cpu")
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
    tensors = raw.get("tensors", raw) if isinstance(raw, dict) else raw
    hessians: dict[str, torch.Tensor] = {}
    if not isinstance(tensors, dict):
        raise ValueError(f"hessian stats must be a dict-like file: {path}")
    for name, value in tensors.items():
        if isinstance(value, dict):
            value = value.get("hessian", value.get("gram"))
        if value is None:
            continue
        hessians[name] = torch.as_tensor(value, dtype=torch.float32)
    return hessians


def _stats_for_key(stats: dict[str, torch.Tensor], key: str) -> torch.Tensor | None:
    candidates = [key]
    if key.endswith(".weight"):
        candidates.append(key[:-7])
    if key.startswith("model."):
        candidates.append(key.removeprefix("model."))
        if key.endswith(".weight"):
            candidates.append(key.removeprefix("model.")[:-7])
    for candidate in candidates:
        if candidate in stats:
            return stats[candidate]
    return None


def _hessian_for_key(hessians: dict[str, torch.Tensor], key: str) -> torch.Tensor | None:
    return _stats_for_key(hessians, key)


def _artifact_keys(weight_key: str, method: str) -> tuple[str, str, str | None, str | None, str | None]:
    base = weight_key.removesuffix(".weight")
    if method == "smoothquant":
        return (
            f"{base}.smooth_int8.weight",
            f"{base}.smooth_int8.weight_scale",
            f"{base}.smooth_int8.smooth_scale",
            None,
            None,
        )
    if method == "mr-gptq":
        return (
            f"{base}.mr_gptq_int8.weight",
            f"{base}.mr_gptq_int8.weight_scale",
            None,
            f"{base}.mr_gptq_int8.activation_order",
            f"{base}.mr_gptq_int8.hessian_diag",
        )
    raise ValueError(f"unknown INT8 artifact method: {method}")


def _scale_key_candidates(weight_key: str, scale_suffix: str | None) -> list[str]:
    base = weight_key.removesuffix(".weight")
    suffixes = [scale_suffix] if scale_suffix else [
        "weight_scale_inv",
        "weight_scale",
        "scale",
        "scales",
    ]
    return [f"{base}.{suffix}" for suffix in suffixes if suffix]


def _is_fp8_dtype(dtype: torch.dtype) -> bool:
    return "float8" in str(dtype)


def _expand_scale(scale: torch.Tensor, out_dim: int, in_dim: int) -> torch.Tensor:
    scale_f = scale.float()
    if scale_f.ndim == 1:
        if scale_f.numel() == out_dim:
            return scale_f.reshape(out_dim, 1)
        if scale_f.numel() == in_dim:
            return scale_f.reshape(1, in_dim)
        raise ValueError(f"1D scale length must match out_dim or in_dim: {scale_f.numel()} vs {out_dim}/{in_dim}")
    if scale_f.ndim == 2 and scale_f.shape[1] == 1 and scale_f.shape[0] == out_dim:
        return scale_f.reshape(out_dim, 1)
    if scale_f.ndim != 2:
        raise ValueError(f"scale must be 1D or 2D, got shape={tuple(scale_f.shape)}")

    row_block = max(1, math.ceil(out_dim / scale_f.shape[0]))
    col_block = max(1, math.ceil(in_dim / scale_f.shape[1]))
    expanded = scale_f.repeat_interleave(row_block, dim=0)[:out_dim]
    expanded = expanded.repeat_interleave(col_block, dim=1)[:, :in_dim]
    return expanded


def _dequantize_weight_for_int8(weight: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    weight_f = weight.float()
    if scale is None:
        if _is_fp8_dtype(weight.dtype):
            raise ValueError("FP8 tensors require a scale tensor before INT8 conversion")
        return weight_f
    if weight.ndim != 2:
        raise ValueError(f"only 2D weights are supported, got shape={tuple(weight.shape)}")
    out_dim, in_dim = weight.shape
    return weight_f * _expand_scale(scale, out_dim, in_dim)


def _load_tensor(path: Path, key: str) -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as reader:
        return reader.get_tensor(key)


def _build_tensor_file_map(files: list[Path]) -> dict[str, Path]:
    from safetensors import safe_open

    tensor_file_map: dict[str, Path] = {}
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as reader:
            for key in reader.keys():
                tensor_file_map[key] = file
    return tensor_file_map


def build_artifacts(
    weights_path: Path,
    activation_stats: Path,
    output_dir: Path,
    include_regex: str,
    alpha: float,
    method: str,
    mr_rotation_block_size: int,
    mr_scale_grid_steps: int,
    mr_min_scale_ratio: float,
    hessian_stats: Path | None,
    max_tensors: int | None,
    dry_run: bool,
    scale_suffix: str | None = None,
    include_non_fp8: bool = False,
) -> dict[str, Any]:
    from safetensors.torch import save_file

    files = _iter_safetensors(weights_path)
    if not files:
        raise FileNotFoundError(f"No safetensors files found under {weights_path}")

    stats = _load_stats(activation_stats)
    hessians = _load_optional_hessian(hessian_stats)
    tensor_file_map = _build_tensor_file_map(files)
    include = re.compile(include_regex)
    smoothquant_config = SmoothQuantConfig(alpha=alpha)
    mr_config = MRGPTQInt8Config(
        rotation_block_size=mr_rotation_block_size,
        scale_grid_steps=mr_scale_grid_steps,
        min_scale_ratio=mr_min_scale_ratio,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "scheme": SCHEME_BY_METHOD[method],
        "method": method,
        "source_dtype": "fp8_dequantized_to_int8",
        "alpha": alpha,
        "mr_gptq": {
            "rotation": "block_hadamard" if method == "mr-gptq" else None,
            "rotation_block_size": mr_rotation_block_size if method == "mr-gptq" else None,
            "rotation_normalized": True if method == "mr-gptq" else None,
            "tail_policy": "identity" if method == "mr-gptq" else None,
            "scale_grid_steps": mr_scale_grid_steps if method == "mr-gptq" else None,
            "min_scale_ratio": mr_min_scale_ratio if method == "mr-gptq" else None,
            "gptq_error_propagation": (
                "applied_when_hessian_available" if method == "mr-gptq" and hessian_stats else (
                    "not_applied_activation_amax_only" if method == "mr-gptq" else None
                )
            ),
        },
        "source": str(weights_path),
        "activation_stats": str(activation_stats),
        "hessian_stats": str(hessian_stats) if hessian_stats else None,
        "scale_suffix": scale_suffix or "auto",
        "files": [],
        "skipped": [],
    }
    converted = 0
    out_by_source_file: dict[Path, tuple[dict[str, torch.Tensor], dict[str, Any]]] = {}

    for key in sorted(tensor_file_map):
        if max_tensors is not None and converted >= max_tensors:
            break
        if not key.endswith(".weight") or not include.search(key):
            continue

        src_file = tensor_file_map[key]
        weight = _load_tensor(src_file, key)
        if weight.ndim != 2:
            manifest["skipped"].append({"tensor": key, "reason": f"shape={list(weight.shape)}"})
            continue
        if not include_non_fp8 and not _is_fp8_dtype(weight.dtype):
            manifest["skipped"].append({"tensor": key, "reason": f"non_fp8_dtype={weight.dtype}"})
            continue

        activation_amax = _stats_for_key(stats, key)
        if activation_amax is None:
            manifest["skipped"].append({"tensor": key, "reason": "missing_activation_stats"})
            continue

        scale_key = next(
            (candidate for candidate in _scale_key_candidates(key, scale_suffix) if candidate in tensor_file_map),
            None,
        )
        scale = _load_tensor(tensor_file_map[scale_key], scale_key) if scale_key is not None else None
        if _is_fp8_dtype(weight.dtype) and scale is None:
            manifest["skipped"].append({"tensor": key, "reason": "missing_fp8_scale"})
            continue

        weight_f = _dequantize_weight_for_int8(weight, scale)
        q_key, w_scale_key, smooth_key, order_key, hessian_key = _artifact_keys(key, method)

        out_tensors, file_entry = out_by_source_file.setdefault(
            src_file,
            ({}, {"source": str(src_file), "tensors": []}),
        )
        tensor_entry = {
            "source": key,
            "source_dtype": str(weight.dtype),
            "source_scale": scale_key,
            "qweight": q_key,
            "weight_scale": w_scale_key,
            "shape": list(weight.shape),
        }
        if method == "smoothquant":
            params = calibrate_weight_int8_smoothquant(weight_f, activation_amax, smoothquant_config)
            out_tensors[q_key] = params.weight_q.cpu()
            out_tensors[w_scale_key] = params.weight_scale.cpu()
            assert smooth_key is not None
            out_tensors[smooth_key] = params.smooth_scale.cpu()
            tensor_entry["smooth_scale"] = smooth_key
        elif method == "mr-gptq":
            hessian = _hessian_for_key(hessians, key)
            params = calibrate_weight_int8_mr_gptq(weight_f, activation_amax, mr_config, hessian=hessian)
            out_tensors[q_key] = params.weight_q.cpu()
            out_tensors[w_scale_key] = params.weight_scale.cpu()
            assert order_key is not None and hessian_key is not None
            out_tensors[order_key] = params.activation_order.cpu()
            out_tensors[hessian_key] = params.hessian_diag.cpu()
            tensor_entry.update(
                {
                    "activation_order": order_key,
                    "hessian_diag": hessian_key,
                    "rotation": {
                        "type": "block_hadamard",
                        "block_size": params.rotation_block_size,
                        "normalized": True,
                        "tail_policy": "identity",
                    },
                    "scale_shape": list(params.weight_scale.shape),
                    "gptq_error_propagation": params.gptq_error_propagation,
                }
            )
        else:
            raise ValueError(f"unknown INT8 artifact method: {method}")
        file_entry["tensors"].append(tensor_entry)
        converted += 1

    for src_file, (out_tensors, file_entry) in out_by_source_file.items():
        suffix = "smooth-int8" if method == "smoothquant" else "mr-gptq-int8"
        out_file = output_dir / f"{src_file.stem}.{suffix}.safetensors"
        file_entry["artifact"] = str(out_file)
        if not dry_run:
            save_file(out_tensors, out_file)
        manifest["files"].append(file_entry)

    manifest["converted_tensors"] = converted
    manifest_path = output_dir / MANIFEST_BY_METHOD[method]
    if not dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="safetensors file or directory")
    parser.add_argument("--activation-stats", type=Path, required=True, help="per-weight activation amax JSON")
    parser.add_argument("--hessian-stats", type=Path, help="optional per-weight Hessian/Gram stats for real GPTQ")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=tuple(MANIFEST_BY_METHOD),
        default="smoothquant",
        help="INT8 artifact method. mr-gptq emits micro-rotated/group-scaled INT8 artifacts.",
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--mr-rotation-block-size",
        type=int,
        default=128,
        help="Hadamard block size for --method mr-gptq. Must be a power of two.",
    )
    parser.add_argument(
        "--mr-scale-grid-steps",
        type=int,
        default=16,
        help="Scale candidates per row/group for MR-GPTQ weighted MSE fitting. 1 uses absmax.",
    )
    parser.add_argument(
        "--mr-min-scale-ratio",
        type=float,
        default=0.5,
        help="Smallest candidate scale as a fraction of absmax scale for MR-GPTQ fitting.",
    )
    parser.add_argument(
        "--include-regex",
        default=r"(w1|w2|w3|gate|up|down|q_proj|k_proj|v_proj|o_proj).*weight$",
        help="only convert weight tensors whose key matches this regex",
    )
    parser.add_argument(
        "--scale-suffix",
        choices=("weight_scale_inv", "weight_scale", "scale", "scales"),
        help="explicit FP8 scale suffix; omit to auto-detect common suffixes",
    )
    parser.add_argument("--include-non-fp8", action="store_true", help="also quantize BF16/FP16/FP32 matched weights")
    parser.add_argument("--max-tensors", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_artifacts(
        weights_path=args.weights,
        activation_stats=args.activation_stats,
        output_dir=args.output_dir,
        include_regex=args.include_regex,
        alpha=args.alpha,
        method=args.method,
        mr_rotation_block_size=args.mr_rotation_block_size,
        mr_scale_grid_steps=args.mr_scale_grid_steps,
        mr_min_scale_ratio=args.mr_min_scale_ratio,
        hessian_stats=args.hessian_stats,
        max_tensors=args.max_tensors,
        dry_run=args.dry_run,
        scale_suffix=args.scale_suffix,
        include_non_fp8=args.include_non_fp8,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["converted_tensors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
