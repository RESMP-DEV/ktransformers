#!/usr/bin/env python
# coding=utf-8
"""AVX2 MXFP4 MoE accuracy tests for KT-Kernel."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from kt_kernel import kt_kernel_ext

torch.manual_seed(42)

expert_num = 4
hidden_size = 128
intermediate_size = 256
num_experts_per_tok = 2
max_len = 64
group_size = 32
validation_iter = 2
CPUINFER_PARAM = 8

E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def act_fn(x):
    return x / (1.0 + torch.exp(-x))


def mlp_torch(input, gate_proj, up_proj, down_proj):
    gate_buf = torch.mm(input, gate_proj.t())
    up_buf = torch.mm(input, up_proj.t())
    return torch.mm(act_fn(gate_buf) * up_buf, down_proj.t())


def moe_torch(input, expert_ids, weights, gate_proj, up_proj, down_proj):
    cnts = expert_ids.new_zeros((expert_ids.shape[0], expert_num))
    cnts.scatter_(1, expert_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = expert_ids.view(-1).argsort()
    sorted_tokens = input[idxs // expert_ids.shape[1]]

    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        outputs.append(mlp_torch(sorted_tokens[start_idx:end_idx], gate_proj[i], up_proj[i], down_proj[i]))
        start_idx = end_idx

    outs = torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)
    new_x = torch.empty_like(outs)
    new_x[idxs] = outs
    return new_x.view(*expert_ids.shape, -1).float().mul_(weights.unsqueeze(-1)).sum(1).to(new_x.dtype)


def quantize_mxfp4_tensor(weights: torch.Tensor):
    weights_f32 = weights.float()
    e, rows, cols = weights_f32.shape
    assert cols % group_size == 0

    reshaped = weights_f32.view(e, rows, cols // group_size, group_size)
    max_abs = torch.clamp(reshaped.abs().amax(dim=-1, keepdim=True), min=1e-8)
    scales = (max_abs / 6.0).squeeze(-1)
    normalized = reshaped / scales.unsqueeze(-1)

    distances = torch.abs(normalized.unsqueeze(-1) - E2M1_VALUES.view(1, 1, 1, 1, 16))
    closest = distances.argmin(dim=-1).to(torch.uint8)
    dequant = (E2M1_VALUES[closest.long()] * scales.unsqueeze(-1)).view(e, rows, cols)

    nibbles = closest.view(e, rows, cols // 2, 2)
    packed_bytes = (nibbles[..., 1] << 4) | nibbles[..., 0]
    bytes_view = packed_bytes.view(e, rows, cols // 8, 4)
    packed_i32 = (
        bytes_view[..., 0].to(torch.int32)
        | (bytes_view[..., 1].to(torch.int32) << 8)
        | (bytes_view[..., 2].to(torch.int32) << 16)
        | (bytes_view[..., 3].to(torch.int32) << 24)
    ).contiguous()

    return packed_i32, scales.to(torch.bfloat16).contiguous(), dequant.contiguous()


def test_avx2_mxfp4_accuracy(qlen, label):
    physical_to_logical_map = torch.tensor(range(expert_num), dtype=torch.int64).contiguous()
    CPUInfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)

    with torch.inference_mode():
        gate = (torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32) / 20.0).to(torch.bfloat16)
        up = (torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32) / 20.0).to(torch.bfloat16)
        down = (torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.float32) / 20.0).to(torch.bfloat16)

        gate_q, gate_s, gate_deq = quantize_mxfp4_tensor(gate)
        up_q, up_s, up_deq = quantize_mxfp4_tensor(up)
        down_q, down_s, down_deq = quantize_mxfp4_tensor(down)

        config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
        config.max_len = max_len
        config.gate_proj = gate_q.data_ptr()
        config.up_proj = up_q.data_ptr()
        config.down_proj = down_q.data_ptr()
        config.gate_scale = gate_s.data_ptr()
        config.up_scale = up_s.data_ptr()
        config.down_scale = down_s.data_ptr()
        config.quant_config.bits = 4
        config.quant_config.group_size = group_size
        config.quant_config.zero_point = False
        config.pool = CPUInfer.backend_

        moe = kt_kernel_ext.moe.AVX2MXFP4_MOE(config)
        CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
        CPUInfer.sync()

        print(f"\n--- {label} (qlen={qlen}) ---")
        for i in range(validation_iter):
            expert_ids = torch.stack([torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)]).contiguous()
            weights = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
            input_data = (torch.randn((qlen, hidden_size), dtype=torch.float32) / 100.0).to(torch.bfloat16).contiguous()
            output = torch.empty((qlen, hidden_size), dtype=torch.bfloat16).contiguous()
            bsz_tensor = torch.tensor([qlen], dtype=torch.int32)

            CPUInfer.submit(
                moe.forward_task(
                    bsz_tensor.data_ptr(),
                    num_experts_per_tok,
                    expert_ids.data_ptr(),
                    weights.data_ptr(),
                    input_data.data_ptr(),
                    output.data_ptr(),
                    False,
                )
            )
            CPUInfer.sync()

            ref = moe_torch(input_data.float(), expert_ids, weights, gate_deq, up_deq, down_deq).to(torch.bfloat16)
            diff = torch.mean(torch.abs(output.float() - ref.float())) / (torch.mean(torch.abs(ref.float())) + 1e-8)
            print(f"  Iteration {i}: diff = {diff:.6f}")
            assert diff < 0.08, f"MXFP4 accuracy test failed: diff={diff:.6f} >= 0.08"


if __name__ == "__main__":
    try:
        test_avx2_mxfp4_accuracy(qlen=1, label="Decode")
        test_avx2_mxfp4_accuracy(qlen=16, label="Prefill")
        print("ALL TESTS PASSED")
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
