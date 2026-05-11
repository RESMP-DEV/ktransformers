"""Static coverage for the MoE REAP calibration runner helpers."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest
import torch


def _load_runner_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "moe_reap_calibration_multi_gpu.py"
    spec = importlib.util.spec_from_file_location("moe_reap_calibration_multi_gpu", script)
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


def test_hc_split_sinkhorn_torch_fallback_shapes_and_normalizes():
    runner = _load_runner_module()
    torch.manual_seed(0)
    mixes = torch.randn(2, 3, 24)
    scale = torch.tensor([0.5, 0.25, 0.75], dtype=torch.float32)
    base = torch.randn(24)

    pre, post, comb = runner._torch_hc_split_sinkhorn_fallback(
        mixes,
        scale,
        base,
        hc_mult=4,
        sinkhorn_iters=20,
        eps=1e-6,
    )

    assert pre.shape == (2, 3, 4)
    assert post.shape == (2, 3, 4)
    assert comb.shape == (2, 3, 4, 4)
    assert torch.all(pre > 0)
    assert torch.all(post >= 0)
    torch.testing.assert_close(comb.sum(dim=-1), torch.ones(2, 3, 4), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(comb.sum(dim=-2), torch.ones(2, 3, 4), rtol=1e-4, atol=1e-4)


def test_parse_prompt_record_accepts_domain_json():
    runner = _load_runner_module()

    text, domain = runner._parse_prompt_record('{"domain_tag":"code-python","text":"write code"}')

    assert text == "write code"
    assert domain == "code-python"
    assert runner._parse_prompt_record("plain prompt") == ("plain prompt", None)


def test_write_domain_routing_counts_outputs_eva_tensor(tmp_path):
    runner = _load_runner_module()
    runner.DOMAIN_ROUTING_COUNTS.clear()
    runner.DOMAIN_ROUTING_COUNTS["code-python"] = {0: [1, 2, 0], 1: [0, 1, 3]}
    runner.DOMAIN_ROUTING_COUNTS["instruction-following"] = {0: [4, 0, 1]}
    output = tmp_path / "counts.pt"

    runner._write_domain_routing_counts(
        output,
        num_layers=2,
        num_experts=3,
        prompt_source=None,
    )

    saved = torch.load(output, map_location="cpu", weights_only=False)
    counts = saved["counts"]
    code_idx = runner.EVA_DOMAIN_TAGS.index("code-python")
    inst_idx = runner.EVA_DOMAIN_TAGS.index("instruction-following")
    assert counts.shape == (2, len(runner.EVA_DOMAIN_TAGS), 3)
    assert counts[0, code_idx].tolist() == [1, 2, 0]
    assert counts[1, code_idx].tolist() == [0, 1, 3]
    assert counts[0, inst_idx].tolist() == [4, 0, 1]
    assert saved["metadata"]["total_routing_hits"] == 12


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
                "format": "moe_reap_runtime_profile_v1",
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


def test_gpu_fp4_backend_auto_prefers_sm86_extension(monkeypatch):
    runner = _load_runner_module()
    monkeypatch.setattr(runner, "_device_is_sm86_cuda", lambda device: True)
    monkeypatch.setattr(runner, "_load_sm86_mxfp4_ops", lambda: object())

    assert runner._select_gpu_fp4_backend("auto", torch.device("cuda:0"), torch.device("cuda:0")) == "sm86-mxfp4"
    assert runner._select_gpu_fp4_backend("torch", torch.device("cuda:0"), torch.device("cuda:0")) == "torch"
    assert runner._select_gpu_fp4_backend("auto", torch.device("cuda:0"), torch.device("cpu")) == "torch"


def test_gpu_fp4_backend_forced_mode_fails_without_extension(monkeypatch):
    runner = _load_runner_module()
    monkeypatch.setattr(runner, "_device_is_sm86_cuda", lambda device: True)
    monkeypatch.setattr(runner, "_load_sm86_mxfp4_ops", lambda: None)

    with pytest.raises(RuntimeError, match="KTransformersOps.mxfp4_linear"):
        runner._select_gpu_fp4_backend("sm86-mxfp4", torch.device("cuda:0"), torch.device("cuda:0"))


def test_gpu_fp8_backend_auto_prefers_sm86_extension(monkeypatch):
    runner = _load_runner_module()
    monkeypatch.setattr(runner, "_device_is_sm86_cuda", lambda device: True)
    monkeypatch.setattr(runner, "_load_sm86_fp8_ops", lambda: object())

    assert runner._select_gpu_fp8_backend("auto", torch.device("cuda:0"), torch.device("cuda:0")) == "sm86-fp8"
    assert runner._select_gpu_fp8_backend("torch", torch.device("cuda:0"), torch.device("cuda:0")) == "torch"
    assert runner._select_gpu_fp8_backend("auto", torch.device("cuda:0"), torch.device("cpu")) == "torch"


def test_gpu_fp8_backend_forced_mode_fails_without_extension(monkeypatch):
    runner = _load_runner_module()
    monkeypatch.setattr(runner, "_device_is_sm86_cuda", lambda device: True)
    monkeypatch.setattr(runner, "_load_sm86_fp8_ops", lambda: None)

    with pytest.raises(RuntimeError, match="KTransformersOps.fp8_linear"):
        runner._select_gpu_fp8_backend("sm86-fp8", torch.device("cuda:0"), torch.device("cuda:0"))


def test_smoothquant_artifact_store_attaches_and_runs_linear(tmp_path):
    runner = _load_runner_module()
    qweight = torch.tensor([[1, -2, 3], [4, 5, -6]], dtype=torch.int8)
    weight_scale = torch.tensor([0.25, 0.5], dtype=torch.float32)
    smooth_scale = torch.tensor([2.0, 4.0, 8.0], dtype=torch.float32)
    manifest = tmp_path / "smoothquant-int8-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "scheme": "w8a8_smoothquant",
                "files": [
                    {
                        "artifact": str(tmp_path / "weights.smooth-int8.safetensors"),
                        "tensors": [
                            {
                                "source": "model.proj.weight",
                                "qweight": "model.proj.smooth_int8.weight",
                                "weight_scale": "model.proj.smooth_int8.weight_scale",
                                "smooth_scale": "model.proj.smooth_int8.smooth_scale",
                                "shape": [2, 3],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    model = torch.nn.Sequential()
    model.add_module("proj", torch.nn.Linear(3, 2, bias=False))
    store = runner.SmoothQuantArtifactStore(manifest)
    stats = runner._attach_smoothquant_int8_artifacts(model, store)
    x = torch.tensor([[8.0, 4.0, 2.0]], dtype=torch.float32)

    class FakeStore:
        def tensors_for(self, entry, device):
            del entry
            return qweight.to(device), weight_scale.to(device), smooth_scale.to(device)

    model.proj.weight._eva_smoothquant_store = FakeStore()
    actual = runner._smoothquant_int8_linear(x, model.proj.weight, torch.float32)
    expected = torch.nn.functional.linear(
        x / smooth_scale.reshape(1, -1),
        qweight.float() * weight_scale.reshape(-1, 1),
    )

    assert stats["attached_tensors"] == 1
    torch.testing.assert_close(actual, expected)


def test_mr_gptq_artifact_store_applies_activation_rotation(tmp_path):
    runner = _load_runner_module()
    qweight = torch.tensor([[2, 0], [0, 2]], dtype=torch.int8)
    weight_scale = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
    manifest = tmp_path / "mr-gptq-int8-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "scheme": "w8a8_mr_gptq_int8",
                "files": [
                    {
                        "artifact": str(tmp_path / "weights.mr-gptq-int8.safetensors"),
                        "tensors": [
                            {
                                "source": "model.proj.weight",
                                "qweight": "model.proj.mr_gptq_int8.weight",
                                "weight_scale": "model.proj.mr_gptq_int8.weight_scale",
                                "rotation": {
                                    "type": "block_hadamard",
                                    "block_size": 2,
                                    "normalized": True,
                                    "tail_policy": "identity",
                                },
                                "shape": [2, 2],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    model = torch.nn.Sequential()
    model.add_module("proj", torch.nn.Linear(2, 2, bias=False))
    store = runner.SmoothQuantArtifactStore(manifest, expected_scheme="w8a8_mr_gptq_int8")
    stats = runner._attach_smoothquant_int8_artifacts(model, store)
    x = torch.tensor([[2.0**0.5, 2.0**0.5]], dtype=torch.float32)

    class FakeStore:
        scheme = "w8a8_mr_gptq_int8"

        def tensors_for(self, entry, device):
            del entry
            return qweight.to(device), weight_scale.to(device), torch.empty(0, device=device)

    model.proj.weight._eva_smoothquant_store = FakeStore()
    model.proj.weight._eva_smoothquant_entry = store.lookup("model.proj.weight")
    actual = runner._smoothquant_int8_linear(x, model.proj.weight, torch.float32)
    expected = torch.nn.functional.linear(runner._apply_block_hadamard(x, 2), qweight.float())

    assert stats["attached_tensors"] == 1
    assert runner._int8_artifact_profile_name(model.proj.weight) == "linear_int8_mr_gptq"
    torch.testing.assert_close(actual, expected)


def test_cpu_moe_phase_profile_accumulates_named_phases():
    runner = _load_runner_module()

    runner._record_cpu_moe_phase("cpuinfer_sync", 0.25)
    runner._record_cpu_moe_phase("cpuinfer_sync", 0.75, calls=3)

    assert runner.CPU_MOE_PROFILE["phase_seconds"]["cpuinfer_sync"] == 1.0
    assert runner.CPU_MOE_PROFILE["phase_calls"]["cpuinfer_sync"] == 4


def test_activation_amax_records_named_linear_inputs():
    runner = _load_runner_module()
    runner.ACTIVATION_STATS_ENABLED = True
    runner.ACTIVATION_AMAX.clear()
    runner.ACTIVATION_TOKENS.clear()
    weight = torch.nn.Parameter(torch.zeros(2, 3))
    weight._eva_param_name = "model.layers.0.self_attn.q_proj.weight"

    runner._record_activation_amax(
        weight,
        torch.tensor(
            [
                [1.0, -2.0, 3.0],
                [-4.0, 0.5, 2.0],
            ]
        ),
    )
    runner._record_activation_amax(weight, torch.tensor([[0.25, -6.0, 1.0]]))

    torch.testing.assert_close(
        runner.ACTIVATION_AMAX["model.layers.0.self_attn.q_proj.weight"],
        torch.tensor([4.0, 6.0, 3.0]),
    )
    assert runner.ACTIVATION_TOKENS["model.layers.0.self_attn.q_proj.weight"] == 3
