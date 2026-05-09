"""Static coverage for the DeepSeek V4 Flash debug runner helpers."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch


def _load_runner_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "deepseek_v4_flash_multi_gpu.py"
    spec = importlib.util.spec_from_file_location("deepseek_v4_flash_multi_gpu", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_throughput_profile_uses_measured_layer_fill_policy():
    runner = _load_runner_module()
    args = argparse.Namespace(
        placement_profile="throughput",
        gpu_memory_fraction=None,
        gpu_headroom_gb=None,
        expert_placement_strategy=None,
    )

    assert runner._resolve_placement_policy(args) == (0.62, 7.0, "front-loading")


def test_high_residency_profile_uses_layer_fill_aggressive_policy():
    runner = _load_runner_module()
    args = argparse.Namespace(
        placement_profile="high-residency",
        gpu_memory_fraction=None,
        gpu_headroom_gb=None,
        expert_placement_strategy=None,
    )

    assert runner._resolve_placement_policy(args) == (0.84, 3.5, "front-loading")


def test_custom_placement_overrides_profile_values():
    runner = _load_runner_module()
    args = argparse.Namespace(
        placement_profile="custom",
        gpu_memory_fraction=0.77,
        gpu_headroom_gb=5.0,
        expert_placement_strategy="balanced",
    )

    assert runner._resolve_placement_policy(args) == (0.77, 5.0, "balanced")


def test_active_cpu_expert_detection_respects_gpu_mask():
    runner = _load_runner_module()
    counts = [0, 2, 0, 1]
    gpu_mask = torch.tensor([False, True, False, True], dtype=torch.bool)

    assert runner._active_expert_indices(counts, 0, 4) == [1, 3]
    assert not runner._has_active_cpu_expert(counts, gpu_mask)

    gpu_mask[3] = False
    assert runner._has_active_cpu_expert(counts, gpu_mask)


def test_cpuinfer_thread_auto_uses_physical_core_count(monkeypatch):
    runner = _load_runner_module()
    monkeypatch.setattr(runner, "_physical_core_count", lambda: 24)

    assert runner._resolve_cpuinfer_threads(0) == 24
    assert runner._resolve_cpuinfer_threads(12) == 12
