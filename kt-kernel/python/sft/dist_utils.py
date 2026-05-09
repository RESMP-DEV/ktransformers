# Distributed and checkpoint utilities for SFT
# SPDX-License-Identifier: Apache-2.0

"""
Shared distributed communication and gradient-checkpoint detection helpers.

This is a leaf module — no imports from other sft/ submodules.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Protocol

import torch


# ---------------------------------------------------------------------------
# Backend capability detection
# ---------------------------------------------------------------------------


def _supports_expert_parallel() -> bool:
    """Check if the current distributed backend supports point-to-point ops.

    Returns True for NCCL/GLOO backends that provide isend/irecv.
    Falls back to False for unsupported backends (e.g. some third-party backends).
    """
    try:
        import torch.distributed as dist

        if not dist.is_initialized():
            return True  # single-process, no backend to check

        backend = dist.get_backend()
        return backend in ("nccl", "gloo")
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Expert ownership helpers
# ---------------------------------------------------------------------------


def _expert_owner(
    expert_id: int,
    world_size: int,
    expert_to_owner: dict[int, int] | None = None,
) -> int:
    """Map an expert id to its owner rank.

    Args:
        expert_id: The expert index to look up.
        world_size: Number of ranks in the process group.
        expert_to_owner: Optional explicit mapping. When provided, this takes
            precedence over the modulo fallback.

    Returns:
        Owner rank in [0, world_size).
    """
    if expert_to_owner is not None:
        owner = expert_to_owner.get(expert_id)
        if owner is not None:
            return owner
    return expert_id % world_size


@dataclass(frozen=True, slots=True)
class ExpertEntry:
    """Metadata for one topk slot of one source token."""

    source_rank: int
    source_row: int  # row within that rank's local_qlen
    topk_slot: int  # 0..num_experts_per_tok-1
    expert_id: int
    owner_rank: int


@dataclass
class ExpertDispatchPlan:
    """Collective routing plan for expert-parallel forward."""

    # Per-source-rank dispatch lists: owner_rank -> list of ExpertEntry
    dispatch: list[dict[int, list[ExpertEntry]]]

    # Receive plan per rank: owner_rank -> list of ExpertEntry
    receive: dict[int, list[ExpertEntry]]

    # Total entries each rank will send to each owner
    send_counts: list[list[int]]  # [source_rank][owner_rank]

    # Total entries each rank will receive from each source
    recv_counts: list[list[int]]  # [owner_rank][source_rank]


def build_expert_dispatch_plan(
    all_qlens: list[int],
    expert_ids: torch.Tensor,
    num_experts_per_tok: int,
    world_size: int,
    expert_to_owner: dict[int, int] | None = None,
) -> ExpertDispatchPlan:
    """Build routing plan for expert-parallel dispatch.

    Args:
        all_qlens: Per-rank token counts.
        expert_ids: Flattened [total_qlen, num_experts_per_tok] expert indices.
        num_experts_per_tok: Number of topk experts per token.
        world_size: Number of ranks.
        expert_to_owner: Optional expert->rank mapping.

    Returns:
        ExpertDispatchPlan with routing metadata.
    """
    dispatch: list[dict[int, list[ExpertEntry]]] = [{} for _ in range(world_size)]
    send_counts = [[0] * world_size for _ in range(world_size)]

    current_row = 0
    for src_rank in range(world_size):
        qlen_src = all_qlens[src_rank]
        for row in range(qlen_src):
            for slot in range(num_experts_per_tok):
                eid = int(expert_ids[current_row, slot])
                owner = _expert_owner(eid, world_size, expert_to_owner)
                entry = ExpertEntry(
                    source_rank=src_rank,
                    source_row=row,
                    topk_slot=slot,
                    expert_id=eid,
                    owner_rank=owner,
                )
                if owner not in dispatch[src_rank]:
                    dispatch[src_rank][owner] = []
                dispatch[src_rank][owner].append(entry)
                send_counts[src_rank][owner] += 1
            current_row += 1

    receive: dict[int, list[ExpertEntry]] = {r: [] for r in range(world_size)}
    recv_counts = [[0] * world_size for _ in range(world_size)]
    for src_rank in range(world_size):
        for owner_rank, entries in dispatch[src_rank].items():
            receive[owner_rank].extend(entries)
            recv_counts[owner_rank][src_rank] = len(entries)

    return ExpertDispatchPlan(
        dispatch=dispatch,
        receive=receive,
        send_counts=send_counts,
        recv_counts=recv_counts,
    )


# ---------------------------------------------------------------------------
# Collective helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Expert-parallel dispatch helpers
# ---------------------------------------------------------------------------


def _dist_gather_varlen_to_expert_owners(
    local_tensors: list[torch.Tensor],
    dispatch_plan: ExpertDispatchPlan,
    rank: int,
    world_size: int,
) -> list[list[torch.Tensor]]:
    """Dispatch per-token variable-length tensors to expert owners.

    Uses point-to-point sends based on the dispatch plan. Each source rank
    sends to the owner ranks that hold its expert entries. Token ordering is
    preserved via ExpertEntry metadata.

    Args:
        local_tensors: Per-rank tensors to dispatch (each [local_qlen, *]).
        dispatch_plan: Precomputed routing plan from build_expert_dispatch_plan.
        rank: Current rank.
        world_size: Number of ranks.

    Returns:
        Per-owner-rank list of received tensor lists.
        Each entry is [num_entries_for_owner, *feature_dims].
    """
    import torch.distributed as dist

    local_dispatch = dispatch_plan.dispatch[rank]
    tensor_count = len(local_tensors)
    feature_shapes = [t.shape[1:] for t in local_tensors]
    devices = [t.device for t in local_tensors]
    dtypes = [t.dtype for t in local_tensors]

    # ---- Single rank: extract locally without communication ----
    if world_size == 1:
        received_per_owner: list[list[torch.Tensor]] = []
        for owner_rank in range(world_size):
            entries = local_dispatch.get(owner_rank, [])
            owner_tensors: list[torch.Tensor] = []
            if not entries:
                # Empty dispatch — return empty buffers matching feature shapes
                for t_idx in range(tensor_count):
                    owner_tensors.append(
                        torch.empty((0, *feature_shapes[t_idx]), device=devices[t_idx], dtype=dtypes[t_idx])
                    )
            else:
                # Build a per-entry index mapping for output reconstruction
                for t_idx in range(tensor_count):
                    local_tensor = local_tensors[t_idx]
                    slices = [local_tensor[e.source_row : e.source_row + 1] for e in entries]
                    if slices:
                        owner_tensors.append(torch.cat(slices, dim=0))
                    else:
                        owner_tensors.append(
                            torch.empty((0, *feature_shapes[t_idx]), device=devices[t_idx], dtype=dtypes[t_idx])
                        )
            received_per_owner.append(owner_tensors)
        return received_per_owner

    # ---- Multi-rank: allgather send counts first ----
    local_send_counts = [len(local_dispatch.get(dst, [])) for dst in range(world_size)]
    all_send_counts: list[list[int]] = [[0] * world_size for _ in range(world_size)]
    dist.all_gather_object(all_send_counts, local_send_counts)
    local_recv_counts = [sum(all_send_counts[src][rank] for src in range(world_size))]

    # ---- Phase 1: non-rank-0 sources send to destinations ----
    if rank != 0:
        for dst in range(world_size):
            entries = local_dispatch.get(dst, [])
            if not entries:
                continue
            # Pack each tensor with its entry count as a header
            for t_idx, local_tensor in enumerate(local_tensors):
                slices = [local_tensor[e.source_row : e.source_row + 1] for e in entries]
                if slices:
                    buf = torch.cat(slices, dim=0).contiguous()
                else:
                    buf = torch.empty((0, *feature_shapes[t_idx]), device=devices[t_idx], dtype=dtypes[t_idx])
                dist.send(buf, dst=dst)
        return []  # Non-rank-0 returns nothing; rank-0 collects everything

    # ---- Phase 1: rank 0 receives from all other ranks ----
    recv_buffers: list[list[torch.Tensor]] = [[] for _ in range(world_size)]
    for src in range(1, world_size):
        for dst in range(world_size):
            count = all_send_counts[src][dst]
            if count == 0:
                continue
            for t_idx in range(tensor_count):
                recv_shape = (count, *feature_shapes[t_idx])
                buf = torch.empty(recv_shape, device=devices[t_idx], dtype=dtypes[t_idx])
                dist.recv(buf, src=src)
                recv_buffers[dst].append(buf)

    # ---- Phase 2: pack rank 0's own local slices ----
    for dst in range(world_size):
        entries = local_dispatch.get(dst, [])
        if not entries:
            continue
        for t_idx, local_tensor in enumerate(local_tensors):
            slices = [local_tensor[e.source_row : e.source_row + 1] for e in entries]
            if slices:
                recv_buffers[dst].append(torch.cat(slices, dim=0))
            else:
                recv_buffers[dst].append(
                    torch.empty((0, *feature_shapes[t_idx]), device=devices[t_idx], dtype=dtypes[t_idx])
                )

    # ---- Consolidate per-owner buffers into canonical layout ----
    received_per_owner: list[list[torch.Tensor]] = []
    for owner_rank in range(world_size):
        owner_bufs = recv_buffers[owner_rank]
        if len(owner_bufs) == tensor_count:
            # Only rank 0 had entries for this owner — already in canonical form
            received_per_owner.append(owner_bufs)
        else:
            # Concatenate received slices per tensor index
            owner_tensors: list[torch.Tensor] = []
            for t_idx in range(tensor_count):
                t_slices = [owner_bufs[i] for i in range(t_idx, len(owner_bufs), tensor_count)]
                if t_slices:
                    # All received slices for this tensor index should have
                    # the same feature shape; concatenate along dim 0
                    owner_tensors.append(torch.cat(t_slices, dim=0))
                else:
                    owner_tensors.append(
                        torch.empty((0, *feature_shapes[t_idx]), device=devices[t_idx], dtype=dtypes[t_idx])
                    )
            received_per_owner.append(owner_tensors)

    return received_per_owner


def _dist_scatter_varlen_from_expert_owners(
    received_per_owner: list[list[torch.Tensor]],
    dispatch_plan: ExpertDispatchPlan,
    rank: int,
    world_size: int,
    output_feature_shapes: list[tuple[int, ...]],
    devices: list[torch.device],
    dtypes: list[torch.dtype],
) -> list[torch.Tensor]:
    """Scatter expert outputs back to source ranks preserving token order.

    Args:
        received_per_owner: Per-owner received tensors from
            _dist_gather_varlen_to_expert_owners. Only non-empty on rank 0.
        dispatch_plan: Routing plan used for dispatch.
        rank: Current rank.
        world_size: Number of ranks.
        output_feature_shapes: Feature dimensions for output tensors.
        devices: Output devices per tensor.
        dtypes: Output dtypes per tensor.

    Returns:
        List of output tensors in the same order as input local_tensors.
    """
    import torch.distributed as dist

    tensor_count = len(output_feature_shapes)
    receive = dispatch_plan.receive.get(rank, [])

    # ---- Single rank: restore order from in-memory receive buffers ----
    if world_size == 1:
        outputs: list[torch.Tensor] = [
            torch.empty((0, *output_feature_shapes[i]), device=devices[i], dtype=dtypes[i])
            for i in range(tensor_count)
        ]
        if not receive:
            return outputs

        # Determine local qlen from source entries that originate on this rank
        local_entries = [e for e in receive if e.source_rank == 0]
        if not local_entries:
            return outputs

        max_row = max(e.source_row for e in local_entries) + 1
        outputs = [
            torch.empty((max_row, *output_feature_shapes[i]), device=devices[i], dtype=dtypes[i])
            for i in range(tensor_count)
        ]

        # Map each (topk_slot, source_row) back to the received buffer position
        # received_per_owner[owner_rank][t_idx] is [num_owner_entries, *feat]
        # The entries in dispatch_plan.receive[rank] are in the same order as
        # the received slices, so we can index directly.
        pos = 0
        for entry in receive:
            if entry.source_rank != 0:
                continue
            for t_idx in range(tensor_count):
                src_buf = received_per_owner[entry.owner_rank][t_idx]
                # pos indexes into the concatenated owner buffer for this entry
                if pos < src_buf.shape[0]:
                    outputs[t_idx][entry.source_row : entry.source_row + 1] = src_buf[pos : pos + 1]
            pos += 1

        return outputs

    # ---- Multi-rank: rank 0 orchestrates the scatter ----
    if rank == 0:
        # Partition received_per_owner outputs into destination-rank chunks
        results_by_dst: dict[int, list[list[torch.Tensor]]] = {
            r: [[] for _ in range(tensor_count)] for r in range(world_size)
        }

        for owner_rank, owner_tensors in enumerate(received_per_owner):
            owner_receive = dispatch_plan.receive.get(owner_rank, [])
            # Group entries by source_rank, preserving the entry order within each group
            by_source: dict[int, list[tuple[int, ExpertEntry]]] = {}
            for e_idx, entry in enumerate(owner_receive):
                if entry.source_rank not in by_source:
                    by_source[entry.source_rank] = []
                by_source[entry.source_rank].append((e_idx, entry))

            for src_rank, src_entries in by_source.items():
                for e_idx, entry in src_entries:
                    for t_idx in range(tensor_count):
                        owner_buf = owner_tensors[t_idx]
                        if e_idx < owner_buf.shape[0]:
                            results_by_dst[src_rank][t_idx].append(owner_buf[e_idx : e_idx + 1])

        # Send chunks to each destination rank
        for dst in range(1, world_size):
            for t_idx in range(tensor_count):
                chunks = results_by_dst[dst][t_idx]
                if not chunks:
                    continue
                buf = torch.cat(chunks, dim=0).contiguous()
                dist.send(buf, dst=dst)

        # Return rank 0's own chunks in original token order
        outputs: list[torch.Tensor] = []
        for t_idx in range(tensor_count):
            chunks = results_by_dst[0][t_idx]
            if not chunks:
                outputs.append(
                    torch.empty((0, *output_feature_shapes[t_idx]), device=devices[t_idx], dtype=dtypes[t_idx])
                )
            else:
                outputs.append(torch.cat(chunks, dim=0))
        return outputs

    # ---- Non-rank-0: receive from rank 0 ----
    local_entries = [e for e in receive if e.source_rank != 0]
    outputs: list[torch.Tensor] = [
        torch.empty((0, *output_feature_shapes[i]), device=devices[i], dtype=dtypes[i])
        for i in range(tensor_count)
    ]
    if not local_entries:
        return outputs

    max_row = max(e.source_row for e in local_entries) + 1
    outputs = [
        torch.empty((max_row, *output_feature_shapes[i]), device=devices[i], dtype=dtypes[i])
        for i in range(tensor_count)
    ]

    for t_idx in range(tensor_count):
        recv_buf = torch.empty(
            (len(local_entries), *output_feature_shapes[t_idx]), device=devices[t_idx], dtype=dtypes[t_idx]
        )
        dist.recv(recv_buf, src=0)

        # Restore original token order using source_row from each entry
        for e_idx, entry in enumerate(local_entries):
            outputs[t_idx][entry.source_row : entry.source_row + 1] = recv_buf[e_idx : e_idx + 1]

    return outputs


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


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
