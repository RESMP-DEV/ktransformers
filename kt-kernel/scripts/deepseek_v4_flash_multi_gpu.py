#!/usr/bin/env python3
"""DeepSeek V4 Flash single-process multi-GPU/CPU-offload runner.

This runner imports the official DeepSeek V4 inference `model.py` and
`kernel.py`, but keeps host-specific placement policy in this repository. The
placement deliberately leaves configurable GPU headroom and spills more routed
experts to CPU instead of trying to fill every visible GPU.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import math
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional

FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)
FP8_SCALE_DTYPE = getattr(torch, "float8_e8m0fnu", None)
FP4_DTYPE = getattr(torch, "float4_e2m1fn_x2", None)
FP4_TABLE = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)

PLACEMENT_PROFILES: dict[str, dict[str, float | str]] = {
    "conservative": {
        "gpu_memory_fraction": 0.58,
        "gpu_headroom_gb": 10.0,
        "expert_placement_strategy": "front-loading",
    },
    "throughput": {
        "gpu_memory_fraction": 0.62,
        "gpu_headroom_gb": 7.0,
        "expert_placement_strategy": "front-loading",
    },
    "high-residency": {
        "gpu_memory_fraction": 0.84,
        "gpu_headroom_gb": 3.5,
        "expert_placement_strategy": "front-loading",
    },
}


def _install_hadamard_fallback() -> None:
    if "fast_hadamard_transform" in sys.modules:
        return
    try:
        __import__("fast_hadamard_transform")
        return
    except Exception:
        pass

    module = types.ModuleType("fast_hadamard_transform")
    module.__spec__ = importlib.machinery.ModuleSpec("fast_hadamard_transform", loader=None)

    def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        n = x.size(-1)
        if n <= 0 or n & (n - 1):
            raise ValueError(f"Hadamard dimension must be a power of two, got {n}")
        y = x.contiguous().clone()
        prefix = y.shape[:-1]
        step = 1
        while step < n:
            y = y.reshape(*prefix, -1, step * 2)
            left = y[..., :step].clone()
            right = y[..., step : step * 2].clone()
            y[..., :step] = left + right
            y[..., step : step * 2] = left - right
            y = y.reshape(*prefix, n)
            step *= 2
        return y * scale

    module.hadamard_transform = hadamard_transform  # type: ignore[attr-defined]
    sys.modules["fast_hadamard_transform"] = module


def _parse_gpu_ids(value: str) -> list[int]:
    if value == "auto":
        return list(range(torch.cuda.device_count()))
    ids = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not ids:
        raise ValueError("No GPU ids selected")
    return ids


def _auto_dense_gpu(gpu_ids: list[int]) -> int:
    best_gid = gpu_ids[0]
    best_free = -1
    for gid in gpu_ids:
        with torch.cuda.device(gid):
            free, _ = torch.cuda.mem_get_info()
        if free > best_free:
            best_gid = gid
            best_free = free
    return best_gid


def _pre_hopper_selected(gpu_ids: list[int]) -> bool:
    for gid in gpu_ids:
        major, _ = torch.cuda.get_device_capability(gid)
        if major < 9:
            return True
    return False


def _capability_has_int8_tensor_cores(capability: tuple[int, int]) -> bool:
    major, _minor = capability
    return major >= 8


def _gpu_has_int8_tensor_cores(gid: int) -> bool:
    return _capability_has_int8_tensor_cores(torch.cuda.get_device_capability(gid))


def _resolve_auto_mode(value: str, enabled_when_auto: bool) -> bool:
    if value == "auto":
        return enabled_when_auto
    return value == "on"


def _resolve_placement_policy(args: argparse.Namespace) -> tuple[float, float, str]:
    profile_name = "conservative" if args.placement_profile == "custom" else args.placement_profile
    profile = PLACEMENT_PROFILES[profile_name]
    gpu_memory_fraction = (
        float(args.gpu_memory_fraction)
        if args.gpu_memory_fraction is not None
        else float(profile["gpu_memory_fraction"])
    )
    gpu_headroom_gb = (
        float(args.gpu_headroom_gb)
        if args.gpu_headroom_gb is not None
        else float(profile["gpu_headroom_gb"])
    )
    expert_placement_strategy = (
        args.expert_placement_strategy
        if args.expert_placement_strategy is not None
        else str(profile["expert_placement_strategy"])
    )
    return gpu_memory_fraction, gpu_headroom_gb, expert_placement_strategy


def _parse_int_csv(value: str | None) -> list[int] | None:
    if value is None:
        return None
    parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    return parsed or None


def _physical_core_count() -> int:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        pairs: set[tuple[str, str]] = set()
        physical_id: str | None = None
        core_id: str | None = None
        for raw_line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line:
                if physical_id is not None and core_id is not None:
                    pairs.add((physical_id, core_id))
                physical_id = None
                core_id = None
                continue
            if line.startswith("physical id"):
                physical_id = line.split(":", 1)[1].strip()
            elif line.startswith("core id"):
                core_id = line.split(":", 1)[1].strip()
        if physical_id is not None and core_id is not None:
            pairs.add((physical_id, core_id))
        if pairs:
            return len(pairs)
    return max(1, (os.cpu_count() or 1) // 2)


def _resolve_cpuinfer_threads(requested_threads: int) -> int:
    return _physical_core_count() if requested_threads <= 0 else requested_threads


def _is_quantized_or_scale_dtype(dtype: torch.dtype) -> bool:
    return dtype in {FP8_DTYPE, FP4_DTYPE, FP8_SCALE_DTYPE}


def _expand_block_scale(
    scale: torch.Tensor,
    out_dim: int,
    in_dim: int,
    row_block: int,
    col_block: int,
    device: torch.device,
) -> torch.Tensor:
    scale_f = scale.to(device=device).float()
    if row_block > 1:
        scale_f = scale_f.repeat_interleave(row_block, dim=0)[:out_dim]
    if col_block > 1:
        scale_f = scale_f.repeat_interleave(col_block, dim=1)[:, :in_dim]
    return scale_f


def _dequant_fp8_weight(weight: torch.Tensor, device: torch.device) -> torch.Tensor:
    out_dim, in_dim = weight.shape
    scale = getattr(weight, "scale", None)
    if scale is None:
        raise RuntimeError("FP8 weight is missing its .scale tensor")
    weight_f = weight.to(device=device).float()
    scale_f = _expand_block_scale(scale, out_dim, in_dim, 128, 128, device)
    return weight_f * scale_f


def _dequant_fp4_weight(weight: torch.Tensor, device: torch.device) -> torch.Tensor:
    out_dim, packed_in_dim = weight.shape
    in_dim = packed_in_dim * 2
    scale = getattr(weight, "scale", None)
    if scale is None:
        raise RuntimeError("FP4 weight is missing its .scale tensor")
    packed = weight.view(torch.uint8).to(device=device)
    table = FP4_TABLE.to(device=device)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    values = torch.stack([table[low.long()], table[high.long()]], dim=-1).flatten(-2)
    scale_f = _expand_block_scale(scale, out_dim, in_dim, 1, 32, device)
    return values[:, :in_dim] * scale_f


def _torch_act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: str | None = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
) -> Any:
    """Compatibility fallback for QAT simulation on GPUs without native FP8 kernels."""
    del scale_fmt
    if inplace:
        return x
    n = x.size(-1)
    scale_shape = (*x.size()[:-1], math.ceil(n / block_size))
    scale = torch.ones(scale_shape, device=x.device, dtype=torch.float32)
    if scale_dtype != torch.float32:
        scale = scale.to(scale_dtype)
    return x.contiguous(), scale


def _torch_fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
) -> Any:
    del block_size
    if inplace:
        return x
    scale = torch.ones((*x.size()[:-1], math.ceil(x.size(-1) / 32)), device=x.device, dtype=torch.float32)
    return x.contiguous(), scale


def _patch_torch_quant_fallback(model_module: Any) -> None:
    original_linear = model_module.linear

    def fallback_linear(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert bias is None
        if weight.dtype == FP8_DTYPE:
            output_device = x.device
            output_dtype = x.dtype
            compute_device = weight.device
            weight_f = _dequant_fp8_weight(weight, compute_device)
            x_f = x.to(device=compute_device, dtype=torch.float32)
            y = functional.linear(x_f, weight_f)
            return y.to(device=output_device, dtype=output_dtype)
        if weight.dtype == FP4_DTYPE:
            output_device = x.device
            output_dtype = x.dtype
            compute_device = weight.device
            weight_f = _dequant_fp4_weight(weight, compute_device)
            x_f = x.to(device=compute_device, dtype=torch.float32)
            y = functional.linear(x_f, weight_f)
            return y.to(device=output_device, dtype=output_dtype)
        return original_linear(x, weight, bias)

    model_module.linear = fallback_linear
    model_module.act_quant = _torch_act_quant
    model_module.fp4_act_quant = _torch_fp4_act_quant
    try:
        import kernel as kernel_module
    except Exception:
        kernel_module = None
    if kernel_module is not None:
        kernel_module.act_quant = _torch_act_quant
        kernel_module.fp4_act_quant = _torch_fp4_act_quant


def _torch_sparse_attn_fallback(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Torch fallback for the DeepSeek sparse attention TileLang kernel.

    RTX 3090-class Ampere GPUs expose about 99 KiB of opt-in dynamic shared
    memory per block, while the upstream sparse-attention kernel for this shape
    requests more than that. This fallback keeps the same sink-token denominator
    behavior and favors compatibility over speed for smoke/debug runs.
    """
    bsz, seqlen, n_heads, _ = q.size()
    topk = topk_idxs.size(-1)
    if topk == 0:
        return torch.zeros_like(q)

    if kv.device != q.device:
        kv = kv.to(q.device)
    if topk_idxs.device != q.device:
        topk_idxs = topk_idxs.to(q.device)
    if attn_sink.device != q.device:
        attn_sink = attn_sink.to(q.device)

    valid = topk_idxs >= 0
    safe_idxs = topk_idxs.clamp_min(0)
    batch_idxs = torch.arange(bsz, device=q.device)[:, None, None]
    gathered = kv[batch_idxs, safe_idxs]

    scores = torch.einsum("bshd,bskd->bshk", q.float(), gathered.float()) * softmax_scale
    scores = scores.masked_fill(~valid[:, :, None, :], -torch.inf)
    sink_scores = attn_sink.float().view(1, 1, n_heads)
    scores_max = torch.maximum(scores.max(dim=-1).values, sink_scores)

    weights = torch.exp(scores - scores_max[..., None])
    weights = weights.masked_fill(~valid[:, :, None, :], 0.0)
    denom = weights.sum(dim=-1) + torch.exp(sink_scores - scores_max)
    probs = weights / denom.clamp_min(1e-20)[..., None]
    out = torch.einsum("bshk,bskd->bshd", probs, gathered.float())
    return out.to(dtype=q.dtype)


def _patch_torch_attention_fallback(model_module: Any) -> None:
    model_module.sparse_attn = _torch_sparse_attn_fallback
    try:
        import kernel as kernel_module
    except Exception:
        kernel_module = None
    if kernel_module is not None:
        kernel_module.sparse_attn = _torch_sparse_attn_fallback


def _active_expert_indices(
    counts: list[int],
    experts_start_idx: int,
    experts_end_idx: int,
) -> list[int]:
    return [
        expert_idx
        for expert_idx in range(experts_start_idx, experts_end_idx)
        if counts[expert_idx] > 0
    ]


def _has_active_cpu_expert(counts: list[int], gpu_mask: torch.Tensor) -> bool:
    for expert_idx, count in enumerate(counts):
        if count > 0 and not bool(gpu_mask[expert_idx]):
            return True
    return False


def _expert_budget(
    gid: int,
    gpu_memory_fraction: float,
    gpu_headroom_bytes: int,
) -> int:
    props = torch.cuda.get_device_properties(gid)
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(gid)
    except TypeError:
        with torch.cuda.device(gid):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
    except Exception:
        total_bytes = int(props.total_memory)
        free_bytes = max(0, total_bytes - torch.cuda.memory_allocated(gid))
    total_bytes = int(total_bytes)
    free_bytes = int(free_bytes)
    used_bytes = max(0, total_bytes - free_bytes)
    fraction_cap = int(total_bytes * gpu_memory_fraction)
    headroom_cap = max(0, total_bytes - gpu_headroom_bytes)
    return max(0, min(fraction_cap, headroom_cap) - used_bytes)


def _ordered_expert_keys(
    expert_sizes: dict[tuple[int, int], int],
    expert_placement_strategy: str,
) -> list[tuple[int, int]]:
    if expert_placement_strategy == "front-loading":
        return sorted(expert_sizes, key=lambda key: (-expert_sizes[key], key[0], key[1]))

    if expert_placement_strategy != "balanced":
        raise ValueError(f"Unsupported expert placement strategy: {expert_placement_strategy}")

    by_layer: dict[int, list[int]] = {}
    for layer_idx, expert_idx in expert_sizes:
        by_layer.setdefault(layer_idx, []).append(expert_idx)
    for experts in by_layer.values():
        experts.sort()

    ordered: list[tuple[int, int]] = []
    layers = sorted(by_layer)
    max_experts = max((len(experts) for experts in by_layer.values()), default=0)
    for offset in range(max_experts):
        for layer_idx in layers:
            experts = by_layer[layer_idx]
            if offset < len(experts):
                ordered.append((layer_idx, experts[offset]))
    return ordered


def _plan_expert_placement(
    expert_sizes: dict[tuple[int, int], int],
    gpu_budget: dict[int, int],
    expert_placement_strategy: str,
) -> tuple[dict[tuple[int, int], str], dict[int, int]]:
    placement_order = sorted(gpu_budget, key=lambda gid: gpu_budget[gid], reverse=True)
    remaining_budget = dict(gpu_budget)
    expert_map: dict[tuple[int, int], str] = {}

    for key in _ordered_expert_keys(expert_sizes, expert_placement_strategy):
        size = expert_sizes[key]
        candidates = [gid for gid in placement_order if remaining_budget[gid] >= size]
        if candidates:
            if expert_placement_strategy == "balanced":
                gid = max(candidates, key=lambda candidate: (remaining_budget[candidate], -placement_order.index(candidate)))
            else:
                gid = candidates[0]
            expert_map[key] = f"cuda:{gid}"
            remaining_budget[gid] -= size
        else:
            expert_map[key] = "cpu"

    return expert_map, remaining_budget


def distribute_experts(
    transformer: Any,
    gpu_ids: list[int],
    dense_gpu: int,
    gpu_memory_fraction: float,
    gpu_headroom_gb: float,
    expert_placement_strategy: str,
) -> dict[tuple[int, int], str]:
    """Distribute routed experts with a conservative GPU memory budget."""
    gpu_headroom_bytes = int(gpu_headroom_gb * 1024**3)
    gpu_budget = {
        gid: _expert_budget(gid, gpu_memory_fraction, gpu_headroom_bytes)
        for gid in gpu_ids
    }

    expert_sizes: dict[tuple[int, int], int] = {}
    total_experts = 0

    for layer_idx, layer in enumerate(transformer.layers):
        moe = layer.ffn
        for expert_idx, expert in enumerate(moe.experts):
            if expert is None:
                continue
            total_experts += 1
            size = sum(param.numel() * param.element_size() for param in expert.parameters())
            expert_sizes[(layer_idx, expert_idx)] = size

    expert_map, remaining_budget = _plan_expert_placement(
        expert_sizes,
        gpu_budget,
        expert_placement_strategy,
    )
    gpu_placed = sum(1 for device in expert_map.values() if device.startswith("cuda:"))
    cpu_placed = sum(1 for device in expert_map.values() if device == "cpu")

    for layer_idx, layer in enumerate(transformer.layers):
        moe = layer.ffn
        for expert_idx, expert in enumerate(moe.experts):
            if expert is None:
                continue
            expert.to(expert_map[(layer_idx, expert_idx)], non_blocking=True)

    for gid in gpu_ids:
        torch.cuda.synchronize(gid)

    print(
        f"  Placed {gpu_placed}/{total_experts} experts on GPU "
        f"({gpu_placed / total_experts * 100:.0f}%), {cpu_placed} on CPU"
    )
    layer_gpu_counts: dict[int, int] = {}
    for (layer_idx, _), device in expert_map.items():
        if device.startswith("cuda:"):
            layer_gpu_counts[layer_idx] = layer_gpu_counts.get(layer_idx, 0) + 1
    if layer_gpu_counts:
        total_layers = len(transformer.layers)
        counts = [layer_gpu_counts.get(layer_idx, 0) for layer_idx in range(total_layers)]
        print(
            "  Layer GPU experts: "
            f"min={min(counts)}, max={max(counts)}, nonzero_layers={sum(count > 0 for count in counts)}/{len(counts)}"
        )
    print(
        f"  Placement policy: gpu_memory_fraction={gpu_memory_fraction:.2f}, "
        f"gpu_headroom_gb={gpu_headroom_gb:.1f}, dense_gpu={dense_gpu}, "
        f"strategy={expert_placement_strategy}"
    )
    for gid in gpu_ids:
        used = torch.cuda.memory_allocated(gid) / 1024**3
        total = torch.cuda.get_device_properties(gid).total_memory / 1024**3
        remaining_gb = remaining_budget[gid] / 1024**3
        print(f"    GPU {gid}: {used:.1f} / {total:.1f} GB, remaining expert budget {remaining_gb:.1f} GB")

    return expert_map


def _move_dense_weights(
    model: Any,
    dense_device: torch.device,
    dense_cpu_offload: str,
) -> tuple[int, int]:
    gpu_bytes = 0
    cpu_bytes = 0
    for name, param in model.named_parameters():
        is_routed_expert = "experts." in name and "shared_experts" not in name
        if is_routed_expert:
            continue
        keep_on_cpu = dense_cpu_offload == "quantized" and _is_quantized_or_scale_dtype(param.dtype)
        if keep_on_cpu:
            cpu_bytes += param.numel() * param.element_size()
        else:
            param.data = param.data.to(dense_device, non_blocking=True)
            gpu_bytes += param.numel() * param.element_size()
    for _, buf in model.named_buffers():
        buf.data = buf.data.to(dense_device, non_blocking=True)
    return gpu_bytes, cpu_bytes


def patched_moe_forward(self_moe: Any, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """MoE forward for experts placed across GPU and CPU devices."""
    shape = x.size()
    dense_device = x.device
    x = x.view(-1, self_moe.dim)
    weights, indices = self_moe.gate(x, input_ids.flatten())
    y = torch.zeros_like(x, dtype=torch.float32)
    counts = torch.bincount(indices.flatten(), minlength=self_moe.n_routed_experts).tolist()
    active_experts = _active_expert_indices(
        counts,
        self_moe.experts_start_idx,
        self_moe.experts_end_idx,
    )
    kt_cpu_moe = getattr(self_moe, "_kt_cpu_moe", None)

    if kt_cpu_moe is not None:
        gpu_mask = getattr(self_moe, "_kt_gpu_experts_mask", None)
        if gpu_mask is None or _has_active_cpu_expert(counts, gpu_mask):
            if dense_device.type == "cuda":
                cuda_stream = torch.cuda.current_stream(dense_device).cuda_stream
            else:
                cuda_stream = 0
            cpu_out = kt_cpu_moe.forward(x, indices, weights, cuda_stream)
            y += cpu_out.to(device=dense_device, dtype=y.dtype)

    for expert_idx in active_experts:
        expert = self_moe.experts[expert_idx]
        if expert is None:
            continue
        expert_device = next(expert.parameters()).device
        if kt_cpu_moe is not None and expert_device.type == "cpu":
            continue
        idx, top = torch.where(indices == expert_idx)

        if expert_device == dense_device:
            out = expert(x[idx], weights[idx, top, None])
            y.index_add_(0, idx, out.to(device=dense_device, dtype=y.dtype))
        elif expert_device.type == "cuda":
            x_exp = x[idx].to(expert_device, non_blocking=True)
            w_exp = weights[idx, top, None].to(expert_device, non_blocking=True)
            out = expert(x_exp, w_exp)
            y.index_add_(
                0,
                idx,
                out.to(device=dense_device, dtype=y.dtype, non_blocking=True),
            )
        else:
            x_exp = x[idx].to("cpu")
            w_exp = weights[idx, top, None].to("cpu")
            out = expert(x_exp.float(), w_exp.float())
            y.index_add_(0, idx, out.to(device=dense_device, dtype=y.dtype))

    shared_device = next(self_moe.shared_experts.parameters()).device
    if shared_device == dense_device:
        y += self_moe.shared_experts(x)
    elif shared_device.type == "cuda":
        out = self_moe.shared_experts(x.to(shared_device, non_blocking=True))
        y += out.to(dense_device, non_blocking=True)
    else:
        out = self_moe.shared_experts(x.to("cpu"))
        y += out.to(dense_device)
    return y.type_as(x).view(shape)


def _patch_kt_cpuinfer_stream_api() -> None:
    """Adapt newer Python wrappers to CPUInfer builds without stream methods."""
    from kt_kernel import _kt_kernel_ext, experts_base

    if hasattr(_kt_kernel_ext.CPUInfer, "submit_with_cuda_stream"):
        return

    base_cls = experts_base.BaseMoEWrapper
    if getattr(base_cls, "_eva_cpuinfer_no_stream_patch", False):
        return

    def submit_forward(
        self: Any,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        cuda_stream: Any,
    ) -> None:
        del cuda_stream
        flat_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        (
            input_tensor_cpu,
            immediate_experts_ids_cpu,
            deferred_experts_ids_cpu,
            weights_cpu,
            output_cpu,
            bsz_tensor_cpu,
            _output_gpu,
        ) = experts_base.KExpertsCPUBuffer.get_buffer(flat_hidden_states, self.num_experts_per_tok)

        current_slot = self.layer_idx % experts_base.KExpertsCPUBuffer.buffer_depth
        next_slot = (current_slot + 1) % experts_base.KExpertsCPUBuffer.buffer_depth
        bsz_slot_tensor = bsz_tensor_cpu[current_slot]

        topk_ids_long = topk_ids.to(torch.long)
        if self.max_deferred_experts_per_token > 0:
            protected_k = self.num_experts_per_tok - self.max_deferred_experts_per_token
            immediate_ids, deferred_ids = self.select_deferred_experts(topk_ids_long, topk_weights, protected_k)
        else:
            immediate_ids = topk_ids_long
            deferred_ids = None

        input_tensor_cpu[current_slot].copy_(flat_hidden_states, non_blocking=True)
        weights_cpu[current_slot].copy_(topk_weights, non_blocking=True)
        immediate_experts_ids_cpu[current_slot].copy_(immediate_ids, non_blocking=True)
        if hidden_states.device.type == "cuda":
            torch.cuda.current_stream(hidden_states.device).synchronize()

        incremental = base_cls._layer_has_pending_deferred.get(self.layer_idx - 1, False)
        self.cpu_infer.submit(
            self.moe.forward_task(
                bsz_slot_tensor.data_ptr(),
                immediate_experts_ids_cpu[current_slot].size(-1),
                immediate_experts_ids_cpu[current_slot].data_ptr(),
                weights_cpu[current_slot].data_ptr(),
                input_tensor_cpu[current_slot].data_ptr(),
                output_cpu[current_slot].data_ptr(),
                incremental,
            )
        )

        base_cls._layer_has_pending_deferred[self.layer_idx] = False
        if deferred_ids is not None:
            deferred_experts_ids_cpu[current_slot].copy_(deferred_ids, non_blocking=True)
            if hidden_states.device.type == "cuda":
                torch.cuda.current_stream(hidden_states.device).synchronize()
            self.cpu_infer.submit(
                self.moe.forward_task(
                    bsz_slot_tensor.data_ptr(),
                    deferred_experts_ids_cpu[current_slot].size(-1),
                    deferred_experts_ids_cpu[current_slot].data_ptr(),
                    weights_cpu[current_slot].data_ptr(),
                    input_tensor_cpu[current_slot].data_ptr(),
                    output_cpu[next_slot].data_ptr(),
                    False,
                )
            )
            base_cls._layer_has_pending_deferred[self.layer_idx] = True

    def sync_forward(self: Any, hidden_states: torch.Tensor, cuda_stream: Any) -> torch.Tensor:
        del cuda_stream
        flat_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        (
            _input_tensor_cpu,
            _immediate_experts_ids_cpu,
            _deferred_experts_ids_cpu,
            _weights_cpu,
            output_cpu,
            _bsz_tensor_cpu,
            output_gpu,
        ) = experts_base.KExpertsCPUBuffer.get_buffer(flat_hidden_states, self.num_experts_per_tok)

        current_slot = self.layer_idx % experts_base.KExpertsCPUBuffer.buffer_depth
        allow_pending = 1 if base_cls._layer_has_pending_deferred.get(self.layer_idx, False) else 0
        self.cpu_infer.sync(allow_pending)
        output_gpu[current_slot].copy_(output_cpu[current_slot], non_blocking=True)
        return output_gpu[current_slot]

    base_cls.submit_forward = submit_forward
    base_cls.sync_forward = sync_forward
    base_cls._eva_cpuinfer_no_stream_patch = True
    print("[kt-mxfp4] patched CPUInfer submit/sync stream compatibility", flush=True)


def _attach_kt_mxfp4_cpu_moe(
    model: Any,
    ckpt_file: Path,
    expert_map: dict[tuple[int, int], str],
    cpuinfer_threads: int,
    threadpool_count: int,
    chunked_prefill_size: int,
    max_deferred_experts_per_token: int,
    cpu_moe_method: str,
    kt_capture_batch_sizes: list[int] | None,
    numa_nodes: list[int] | None,
) -> int:
    from kt_kernel import KTMoEWrapper

    _patch_kt_cpuinfer_stream_api()
    if kt_capture_batch_sizes:
        KTMoEWrapper.set_capture_batch_sizes(kt_capture_batch_sizes)
        print(f"  [kt-mxfp4] capture_batch_sizes={kt_capture_batch_sizes}", flush=True)
    print(
        "  [kt-mxfp4] "
        f"method={cpu_moe_method}, cpuinfer_threads={cpuinfer_threads}, "
        f"threadpool_count={threadpool_count}, numa_nodes={numa_nodes or 'default'}, "
        f"max_deferred_experts_per_token={max_deferred_experts_per_token}",
        flush=True,
    )

    attached = 0
    for layer_idx, layer in enumerate(model.layers):
        moe = layer.ffn
        gpu_mask = torch.tensor(
            [
                expert_map.get((layer_idx, expert_idx), "cpu").startswith("cuda:")
                for expert_idx in range(moe.n_routed_experts)
            ],
            dtype=torch.bool,
            device="cpu",
        )
        cpu_experts = int((~gpu_mask).sum().item())
        if cpu_experts == 0:
            continue

        wrapper = KTMoEWrapper(
            layer_idx=layer_idx,
            num_experts=moe.n_routed_experts,
            num_experts_per_tok=moe.n_activated_experts,
            hidden_size=moe.dim,
            moe_intermediate_size=moe.experts[moe.experts_start_idx].w1.out_features,
            gpu_experts_mask=gpu_mask,
            cpuinfer_threads=cpuinfer_threads,
            threadpool_count=threadpool_count,
            weight_path=str(ckpt_file),
            chunked_prefill_size=chunked_prefill_size,
            max_deferred_experts_per_token=max_deferred_experts_per_token,
            method=cpu_moe_method,
            numa_nodes=numa_nodes,
        )
        physical_to_logical = torch.arange(moe.n_routed_experts, dtype=torch.int64, device="cpu").contiguous()
        wrapper.load_weights(physical_to_logical)
        moe._kt_cpu_moe = wrapper
        moe._kt_gpu_experts_mask = gpu_mask

        for expert_idx in range(moe.experts_start_idx, moe.experts_end_idx):
            if not bool(gpu_mask[expert_idx]):
                moe.experts[expert_idx] = None
        attached += 1
        print(
            f"  [kt-mxfp4] layer={layer_idx} cpu_experts={cpu_experts} "
            f"gpu_experts={int(gpu_mask.sum().item())}"
        )

    return attached


def sample(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return probs.div_(torch.empty_like(probs).exponential_()).argmax(dim=-1)


def _special_token_content(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content
    return None


def _load_tokenizer(tokenizer_path: Path) -> Any:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as exc:
        print(f"[tokenizer] AutoTokenizer unavailable ({exc}); using tokenizers fallback", flush=True)

    from tokenizers import Tokenizer

    tokenizer_file = tokenizer_path / "tokenizer.json"
    config_file = tokenizer_path / "tokenizer_config.json"
    raw_tokenizer = Tokenizer.from_file(str(tokenizer_file))
    config = json.loads(config_file.read_text(encoding="utf-8")) if config_file.exists() else {}
    eos_token = _special_token_content(config.get("eos_token")) or "<｜end▁of▁sentence｜>"
    eos_token_id = raw_tokenizer.token_to_id(eos_token)
    if eos_token_id is None:
        raise RuntimeError(f"Could not resolve EOS token id for {eos_token!r}")

    class TokenizersCompat:
        def __init__(self, tokenizer: Tokenizer, eos_id: int) -> None:
            self._tokenizer = tokenizer
            self.eos_token_id = eos_id

        def encode(self, text: str) -> list[int]:
            return self._tokenizer.encode(text).ids

        def decode(self, token_ids: list[int]) -> str:
            return self._tokenizer.decode(list(token_ids), skip_special_tokens=False)

    return TokenizersCompat(raw_tokenizer, eos_token_id)


@torch.inference_mode()
def generate(
    model: Any,
    prompt_tokens: list[list[int]],
    max_new_tokens: int,
    eos_id: int,
    dense_device: torch.device,
    temperature: float = 1.0,
) -> list[list[int]]:
    prompt_lens = [len(tokens) for tokens in prompt_tokens]
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device=dense_device)
    for idx, prompt in enumerate(prompt_tokens):
        tokens[idx, : len(prompt)] = torch.tensor(prompt, dtype=torch.long, device=dense_device)
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device=dense_device)
    prompt_mask = tokens != -1

    started = time.time()
    decode_started: float | None = None
    generated = 0
    for cur_pos in range(min(prompt_lens), total_len):
        step = generated + 1
        input_tokens = cur_pos - prev_pos
        step_started = time.time()
        print(
            f"  [decode] step={step}/{max_new_tokens} cur_pos={cur_pos} "
            f"input_tokens={input_tokens}",
            flush=True,
        )
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        step_elapsed = time.time() - step_started
        print(f"  [decode] forward step={step} done in {step_elapsed:.1f}s", flush=True)
        if step == 1:
            print(
                "  [throughput] phase=prefill "
                f"tokens={input_tokens} seconds={step_elapsed:.3f} "
                f"tok_s={input_tokens / max(step_elapsed, 1e-6):.4f}",
                flush=True,
            )
            decode_started = time.time()
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        generated += 1

        elapsed = time.time() - started
        post_prefill_generated = max(0, generated - 1)
        decode_elapsed = 0.0 if decode_started is None else time.time() - decode_started
        print(
            f"  [decode] generated={generated}/{max_new_tokens}, "
            f"{generated / max(elapsed, 1e-6):.2f} tok/s",
            flush=True,
        )
        if post_prefill_generated:
            print(
                "  [throughput] phase=decode "
                f"tokens={post_prefill_generated} seconds={decode_elapsed:.3f} "
                f"tok_s={post_prefill_generated / max(decode_elapsed, 1e-6):.4f}",
                flush=True,
            )

        if bool(finished.all()):
            break

    elapsed = time.time() - started
    print(f"  Done: {generated} tokens in {elapsed:.1f}s ({generated / max(elapsed, 1e-6):.2f} tok/s)")

    completions = []
    for idx, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[idx] : prompt_lens[idx] + max_new_tokens]
        if eos_id in toks:
            toks = toks[: toks.index(eos_id)]
        toks.append(eos_id)
        completions.append(toks)
    return completions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--dense-gpu", default="auto", help="'auto' or a physical CUDA device id")
    parser.add_argument("--gpu-ids", default="auto", help="'auto' or comma-separated physical CUDA device ids")
    parser.add_argument(
        "--placement-profile",
        choices=("conservative", "throughput", "high-residency", "custom"),
        default="conservative",
        help="Preset expert placement policy. high-residency fills more whole layers on GPU.",
    )
    parser.add_argument("--gpu-memory-fraction", type=float)
    parser.add_argument("--gpu-headroom-gb", type=float)
    parser.add_argument(
        "--torch-quant-fallback",
        choices=("auto", "on", "off"),
        default="auto",
        help="Use torch dequantized FP8/FP4 fallbacks instead of TileLang quant GEMMs.",
    )
    parser.add_argument(
        "--dense-cpu-offload",
        choices=("auto", "none", "quantized"),
        default="auto",
        help="Keep selected dense weights on CPU. 'auto' keeps quantized dense weights on CPU when fallback is active.",
    )
    parser.add_argument(
        "--torch-attn-fallback",
        choices=("auto", "on", "off"),
        default="auto",
        help="Use torch sparse-attention fallback when TileLang shared-memory requirements exceed the selected GPU capability.",
    )
    parser.add_argument(
        "--expert-placement-strategy",
        choices=("balanced", "front-loading"),
        default=None,
        help="Choose which routed experts are moved to GPU. 'balanced' spreads capacity across layers; "
        "'front-loading' preserves the previous largest-first early-layer policy.",
    )
    parser.add_argument(
        "--cpu-moe-backend",
        choices=("torch", "kt-mxfp4"),
        default="torch",
        help="CPU routed-expert backend. kt-mxfp4 uses kt-kernel MXFP4 instead of torch FP4 dequant fallback.",
    )
    parser.add_argument(
        "--gpu-quant-backend",
        choices=("native", "int8-smoothquant"),
        default="native",
        help="GPU resident-expert backend. int8-smoothquant requires precomputed calibrated W8A8 artifacts.",
    )
    parser.add_argument(
        "--gpu-smoothquant-artifacts",
        type=Path,
        help="Directory containing smoothquant-int8-manifest.json and W8A8 safetensors artifacts.",
    )
    parser.add_argument(
        "--cpuinfer-threads",
        type=int,
        default=0,
        help="KT CPUInfer worker threads. 0 selects physical cores, which is usually best for AVX2.",
    )
    parser.add_argument("--threadpool-count", type=int, default=1)
    parser.add_argument(
        "--cpuinfer-numa-nodes",
        help="Comma-separated NUMA ids for KT subpools. Omit for KT defaults.",
    )
    parser.add_argument("--chunked-prefill-size", type=int, default=512)
    parser.add_argument(
        "--max-deferred-experts-per-token",
        type=int,
        default=0,
        help="Let KT defer this many low-weight CPU experts per token for cross-layer CPU overlap.",
    )
    parser.add_argument(
        "--cpu-moe-method",
        choices=("MXFP4", "MXFP4_DQ"),
        default="MXFP4",
        help="KT native CPU expert method to use when --cpu-moe-backend=kt-mxfp4.",
    )
    parser.add_argument(
        "--kt-capture-batch-sizes",
        help="Comma-separated KT CPU buffer batch sizes to pre-cache, e.g. 1,16.",
    )
    parser.add_argument("--torch-threads", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    _install_hadamard_fallback()
    sys.path.insert(0, str(args.inference_dir.resolve()))
    encoding_dir = args.inference_dir.resolve().parent / "encoding"
    sys.path.insert(0, str(encoding_dir))

    import model as model_module

    model_module.world_size = 1
    model_module.rank = 0
    from encoding_dsv4 import encode_messages, parse_message_from_completion_text
    from model import ModelArgs, MoE, Transformer

    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    dense_gpu = _auto_dense_gpu(gpu_ids) if args.dense_gpu == "auto" else int(args.dense_gpu)
    if dense_gpu not in gpu_ids:
        gpu_ids.append(dense_gpu)
    gpu_memory_fraction, gpu_headroom_gb, expert_placement_strategy = _resolve_placement_policy(args)
    cpuinfer_numa_nodes = _parse_int_csv(args.cpuinfer_numa_nodes)
    kt_capture_batch_sizes = _parse_int_csv(args.kt_capture_batch_sizes)
    cpuinfer_threads = _resolve_cpuinfer_threads(args.cpuinfer_threads)
    selected_pre_hopper = _pre_hopper_selected(gpu_ids)
    torch_quant_fallback = _resolve_auto_mode(args.torch_quant_fallback, selected_pre_hopper)
    torch_attn_fallback = _resolve_auto_mode(args.torch_attn_fallback, selected_pre_hopper)
    if args.dense_cpu_offload == "auto":
        dense_cpu_offload = "quantized" if torch_quant_fallback else "none"
    else:
        dense_cpu_offload = args.dense_cpu_offload
    if torch_quant_fallback:
        _patch_torch_quant_fallback(model_module)
    if torch_attn_fallback:
        _patch_torch_attention_fallback(model_module)

    print(f"GPUs selected: {gpu_ids}")
    int8_tc_gpus = []
    for gid in gpu_ids:
        props = torch.cuda.get_device_properties(gid)
        capability = torch.cuda.get_device_capability(gid)
        has_int8_tc = _capability_has_int8_tensor_cores(capability)
        if has_int8_tc:
            int8_tc_gpus.append(gid)
        with torch.cuda.device(gid):
            free, total = torch.cuda.mem_get_info()
        print(
            f"  GPU {gid}: {props.name}, sm={capability[0]}{capability[1]}, total={total / 1024**3:.1f} GB, "
            f"free={free / 1024**3:.1f} GB, int8_tensor_cores={'yes' if has_int8_tc else 'no'}"
        )
    print(
        "GPU quant policy: "
        f"backend={args.gpu_quant_backend}, "
        f"smoothquant_artifacts={args.gpu_smoothquant_artifacts or 'none'}, "
        f"int8_tensor_core_gpus={int8_tc_gpus}"
    )
    if args.gpu_quant_backend == "int8-smoothquant":
        unsupported = [gid for gid in gpu_ids if not _gpu_has_int8_tensor_cores(gid)]
        if unsupported:
            raise RuntimeError(f"int8-smoothquant selected but GPUs lack Ampere INT8 tensor cores: {unsupported}")
        if args.gpu_smoothquant_artifacts is None:
            raise RuntimeError(
                "int8-smoothquant requires --gpu-smoothquant-artifacts. "
                "Build calibrated W8A8 artifacts with kt-kernel/scripts/build_smoothquant_int8_artifacts.py."
            )
        manifest_path = args.gpu_smoothquant_artifacts / "smoothquant-int8-manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"SmoothQuant manifest not found: {manifest_path}")
        raise RuntimeError(
            "int8-smoothquant artifact validation passed, but the CUDA W8A8 replacement path is not wired into "
            "the DeepSeek V4 runner yet. Use --gpu-quant-backend native for execution until the tensor-core "
            "kernel adapter is implemented."
        )
    print(
        "Compatibility policy: "
        f"torch_quant_fallback={torch_quant_fallback}, "
        f"torch_attn_fallback={torch_attn_fallback}, "
        f"dense_cpu_offload={dense_cpu_offload}"
    )
    print(
        "Placement profile: "
        f"profile={args.placement_profile}, "
        f"gpu_memory_fraction={gpu_memory_fraction:.2f}, "
        f"gpu_headroom_gb={gpu_headroom_gb:.1f}, "
        f"strategy={expert_placement_strategy}"
    )
    print(
        "CPU policy: "
        f"torch_threads={args.torch_threads}, "
        f"cpuinfer_threads={cpuinfer_threads}, "
        f"threadpool_count={args.threadpool_count}, "
        f"cpuinfer_numa_nodes={cpuinfer_numa_nodes or 'default'}, "
        f"kt_capture_batch_sizes={kt_capture_batch_sizes or 'none'}"
    )

    dense_device = torch.device(f"cuda:{dense_gpu}")
    torch.cuda.set_device(dense_gpu)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(33377335)

    with args.config.open("r", encoding="utf-8") as f:
        model_args = ModelArgs(**json.load(f))
    model_args.max_batch_size = 1
    print(
        f"\nModel: {model_args.n_layers} layers, {model_args.n_routed_experts} experts, "
        f"dim={model_args.dim}, top-{model_args.n_activated_experts}"
    )

    print("\nAllocating model on CPU...")
    started = time.time()
    with torch.device("cpu"):
        model = Transformer(model_args)
    print(f"  Done in {time.time() - started:.1f}s")

    ckpt_file = args.ckpt_path / "model0-mp1.safetensors"
    print(f"\nLoading weights from {ckpt_file}...")
    started = time.time()
    from safetensors.torch import load_model

    load_model(model, str(ckpt_file), strict=False, device="cpu")
    print(f"  Done in {time.time() - started:.1f}s")

    print(f"\nPlacing dense weights for GPU {dense_gpu}...")
    started = time.time()
    dense_gpu_bytes, dense_cpu_bytes = _move_dense_weights(model, dense_device, dense_cpu_offload)
    torch.cuda.synchronize(dense_gpu)
    dense_gb = torch.cuda.memory_allocated(dense_gpu) / 1024**3
    print(
        f"  Dense parameters moved to GPU: {dense_gpu_bytes / 1024**3:.1f} GB; "
        f"kept on CPU: {dense_cpu_bytes / 1024**3:.1f} GB"
    )
    print(f"  CUDA allocated on GPU {dense_gpu}: {dense_gb:.1f} GB")
    print(f"  Done in {time.time() - started:.1f}s")

    print(f"\nDistributing experts across GPUs {gpu_ids}...")
    started = time.time()
    expert_map = distribute_experts(
        model,
        gpu_ids=gpu_ids,
        dense_gpu=dense_gpu,
        gpu_memory_fraction=gpu_memory_fraction,
        gpu_headroom_gb=gpu_headroom_gb,
        expert_placement_strategy=expert_placement_strategy,
    )
    print(f"  Done in {time.time() - started:.1f}s")

    if args.cpu_moe_backend == "kt-mxfp4":
        print("\nAttaching kt-kernel MXFP4 CPU experts...")
        started = time.time()
        attached = _attach_kt_mxfp4_cpu_moe(
            model,
            ckpt_file=ckpt_file,
            expert_map=expert_map,
            cpuinfer_threads=cpuinfer_threads,
            threadpool_count=args.threadpool_count,
            chunked_prefill_size=args.chunked_prefill_size,
            max_deferred_experts_per_token=args.max_deferred_experts_per_token,
            cpu_moe_method=args.cpu_moe_method,
            kt_capture_batch_sizes=kt_capture_batch_sizes,
            numa_nodes=cpuinfer_numa_nodes,
        )
        print(f"  Attached kt-kernel MXFP4 wrappers to {attached} layers in {time.time() - started:.1f}s")

    MoE.forward = patched_moe_forward
    torch.set_default_device(dense_device)

    tokenizer = _load_tokenizer(args.tokenizer_path)
    print("\nReady!")

    messages: list[dict[str, str]] = []
    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            break
        if prompt == "/exit":
            break
        if prompt == "/clear":
            messages.clear()
            continue
        if not prompt.strip():
            continue

        messages.append({"role": "user", "content": prompt})
        prompt_tokens = tokenizer.encode(encode_messages(messages, thinking_mode="chat"))
        print(f"  ({len(prompt_tokens)} prompt tokens)")

        completion_tokens = generate(
            model,
            [prompt_tokens],
            args.max_new_tokens,
            tokenizer.eos_token_id,
            dense_device,
            args.temperature,
        )
        completion = tokenizer.decode(completion_tokens[0])
        print(completion)
        messages.append(parse_message_from_completion_text(completion, thinking_mode="chat"))


if __name__ == "__main__":
    main()
