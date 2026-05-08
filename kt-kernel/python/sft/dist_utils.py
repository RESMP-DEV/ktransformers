# Distributed and checkpoint utilities for SFT
# SPDX-License-Identifier: Apache-2.0

"""
Shared distributed communication and gradient-checkpoint detection helpers.

This is a leaf module — no imports from other sft/ submodules.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch


def _all_gather_qlens(local_qlen: int, device: torch.device, world_size: int) -> list[int]:
    import torch.distributed as dist

    local_qlen_t = torch.tensor([int(local_qlen)], device=device, dtype=torch.int64)
    gathered = [torch.empty(1, device=device, dtype=torch.int64) for _ in range(world_size)]
    dist.all_gather(gathered, local_qlen_t)
    return [int(t.item()) for t in gathered]


def _qlen_offsets(all_qlens: list[int]) -> list[int]:
    offsets = [0]
    for q in all_qlens:
        offsets.append(offsets[-1] + int(q))
    return offsets


def _dist_gather_varlen_to_rank0(
    local_tensor: torch.Tensor,
    *,
    all_qlens: list[int],
    rank: int,
    world_size: int,
) -> list[torch.Tensor] | None:
    gathered = _dist_gather_varlen_many_to_rank0(
        [local_tensor],
        all_qlens=all_qlens,
        rank=rank,
        world_size=world_size,
    )
    return None if gathered is None else gathered[0]


def _dist_gather_varlen_many_to_rank0(
    local_tensors: list[torch.Tensor],
    *,
    all_qlens: list[int],
    rank: int,
    world_size: int,
) -> list[list[torch.Tensor]] | None:
    import torch.distributed as dist

    local_tensors = [t.contiguous() for t in local_tensors]
    local_expected = int(all_qlens[rank])
    for idx, local_tensor in enumerate(local_tensors):
        if local_tensor.shape[0] != local_expected:
            raise RuntimeError(
                f"Local leading dim mismatch on rank {rank}, tensor {idx}: "
                f"got {local_tensor.shape[0]}, expected {local_expected}"
            )

    if rank == 0:
        gathered_many: list[list[torch.Tensor | None]] = []
        for local_tensor in local_tensors:
            gathered: list[torch.Tensor | None] = [None] * world_size
            gathered[0] = local_tensor
            gathered_many.append(gathered)

        ops: list[dist.P2POp] = []
        for src in range(1, world_size):
            qlen_src = int(all_qlens[src])
            for tensor_idx, local_tensor in enumerate(local_tensors):
                recv_shape = (qlen_src, *local_tensor.shape[1:])
                recv = torch.empty(recv_shape, device=local_tensor.device, dtype=local_tensor.dtype)
                gathered_many[tensor_idx][src] = recv
                if qlen_src > 0:
                    ops.append(dist.P2POp(dist.irecv, recv, src))
        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        out_many: list[list[torch.Tensor]] = []
        for tensor_idx, gathered in enumerate(gathered_many):
            out: list[torch.Tensor] = []
            for idx, t in enumerate(gathered):
                if t is None:
                    raise RuntimeError(f"Missing gathered tensor {tensor_idx} for rank {idx} on rank0.")
                out.append(t)
            out_many.append(out)
        return out_many

    if local_expected > 0:
        reqs = dist.batch_isend_irecv([dist.P2POp(dist.isend, local_tensor, 0) for local_tensor in local_tensors])
        for req in reqs:
            req.wait()
    return None


def _dist_scatter_varlen_from_rank0(
    *,
    rank0_chunks: list[torch.Tensor] | None,
    all_qlens: list[int],
    rank: int,
    world_size: int,
    feature_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    outs = _dist_scatter_varlen_many_from_rank0(
        rank0_chunks_per_tensor=None if rank0_chunks is None else [rank0_chunks],
        all_qlens=all_qlens,
        rank=rank,
        world_size=world_size,
        feature_shapes=[feature_shape],
        devices=[device],
        dtypes=[dtype],
    )
    return outs[0]


def _dist_scatter_varlen_many_from_rank0(
    *,
    rank0_chunks_per_tensor: list[list[torch.Tensor]] | None,
    all_qlens: list[int],
    rank: int,
    world_size: int,
    feature_shapes: list[tuple[int, ...]],
    devices: list[torch.device],
    dtypes: list[torch.dtype],
) -> list[torch.Tensor]:
    import torch.distributed as dist

    tensor_count = len(feature_shapes)
    if len(devices) != tensor_count or len(dtypes) != tensor_count:
        raise RuntimeError("feature_shapes, devices, and dtypes must have matching lengths.")

    local_qlen = int(all_qlens[rank])
    local_outs = [
        torch.empty((local_qlen, *feature_shapes[i]), device=devices[i], dtype=dtypes[i]) for i in range(tensor_count)
    ]

    if rank == 0:
        if rank0_chunks_per_tensor is None or len(rank0_chunks_per_tensor) != tensor_count:
            raise RuntimeError("rank0_chunks_per_tensor must contain one chunk list per tensor on rank0.")
        for tensor_idx, chunks in enumerate(rank0_chunks_per_tensor):
            if len(chunks) != world_size:
                raise RuntimeError(f"rank0 chunks for tensor {tensor_idx} must contain one chunk per rank.")
            if int(chunks[0].shape[0]) != local_qlen:
                raise RuntimeError(
                    f"Rank0 local chunk mismatch for tensor {tensor_idx}: "
                    f"got {chunks[0].shape[0]}, expected {local_qlen}"
                )
            if local_qlen > 0:
                local_outs[tensor_idx].copy_(chunks[0])
        ops: list[dist.P2POp] = []
        for dst in range(1, world_size):
            qlen_dst = int(all_qlens[dst])
            if qlen_dst <= 0:
                continue
            for tensor_idx, chunks in enumerate(rank0_chunks_per_tensor):
                chunk = chunks[dst].contiguous()
                if int(chunk.shape[0]) != qlen_dst:
                    raise RuntimeError(
                        f"Rank{dst} chunk mismatch for tensor {tensor_idx} on rank0: "
                        f"got {chunk.shape[0]}, expected {qlen_dst}"
                    )
                ops.append(dist.P2POp(dist.isend, chunk, dst))
        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()
        return local_outs

    if local_qlen > 0:
        reqs = dist.batch_isend_irecv([dist.P2POp(dist.irecv, local_out, 0) for local_out in local_outs])
        for req in reqs:
            req.wait()
    return local_outs



def _checkpoint_hook_mode() -> str:
    """Infer checkpoint phase from current saved_tensors_hooks top.

    Returns one of:
      - "first_forward": non-reentrant checkpoint's _checkpoint_hook
      - "recompute": non-reentrant checkpoint's _recomputation_hook
      - "none": no default saved_tensors_hooks on top
      - "other": unknown hook stack entry
      - "error": failed to query hook stack
    """
    try:
        top = torch._C._autograd._top_saved_tensors_default_hooks(False)
    except Exception:
        return "error"
    if top is None:
        return "none"
    try:
        pack_fn, _ = top
        mod = getattr(pack_fn, "__module__", "")
        qual = getattr(pack_fn, "__qualname__", getattr(pack_fn, "__name__", ""))
        tag = f"{mod}.{qual}"
    except Exception:
        return "other"
    if "_recomputation_hook.__init__.<locals>.pack_hook" in tag:
        return "recompute"
    if "_checkpoint_hook.__init__.<locals>.pack_hook" in tag:
        return "first_forward"
    return "other"


def _maybe_zero3_gathered_parameters(params: list[torch.nn.Parameter]):
    if not params:
        return nullcontext()
    try:
        from transformers.integrations import is_deepspeed_zero3_enabled
    except Exception:
        return nullcontext()
    if not is_deepspeed_zero3_enabled():
        return nullcontext()
    try:
        import deepspeed  # type: ignore
    except Exception:
        return nullcontext()
    return deepspeed.zero.GatheredParameters(params, modifier_rank=0)
