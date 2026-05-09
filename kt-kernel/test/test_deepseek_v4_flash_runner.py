"""Static coverage for the DeepSeek V4 Flash debug runner helpers."""

from __future__ import annotations

import argparse
import importlib.util
import json
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


def test_activation_profile_loads_runtime_routing_counts(tmp_path):
    runner = _load_runner_module()
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "format": "deepseek_v4_flash_runtime_profile_v1",
                "routing": {
                    "layers": [
                        {"layer": 0, "expert_counts": [0, 3, 9]},
                        {"layer": 1, "expert_counts": [5, 0, 1]},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    assert runner._load_activation_scores(profile) == {
        (0, 0): 0.0,
        (0, 1): 3.0,
        (0, 2): 9.0,
        (1, 0): 5.0,
        (1, 1): 0.0,
        (1, 2): 1.0,
    }


def test_activation_aware_placement_prefers_hot_experts():
    runner = _load_runner_module()
    expert_sizes = {(0, 0): 10, (0, 1): 10, (1, 0): 10}
    scores = {(0, 0): 1.0, (0, 1): 100.0, (1, 0): 2.0}

    expert_map, remaining = runner._plan_expert_placement(
        expert_sizes,
        {0: 10},
        "activation-aware",
        scores,
    )

    assert expert_map[(0, 1)] == "cuda:0"
    assert expert_map[(0, 0)] == "cpu"
    assert expert_map[(1, 0)] == "cpu"
    assert remaining[0] == 0
