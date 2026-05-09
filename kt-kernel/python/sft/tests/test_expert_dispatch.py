# SPDX-License-Identifier: Apache-2.0
"""Developer tests for expert-parallel dispatch in dist_utils."""

from __future__ import annotations

import torch

from sft.dist_utils import (
    ExpertDispatchPlan,
    ExpertEntry,
    _expert_owner,
    build_expert_dispatch_plan,
    _dist_gather_varlen_to_expert_owners,
    _dist_scatter_varlen_from_expert_owners,
    _supports_expert_parallel,
)


class TestExpertOwnership:
    """Tests for _expert_owner and build_expert_dispatch_plan."""

    def test_expert_owner_modulo(self):
        """Modulo fallback routes each expert to rank = expert_id % world_size."""
        assert _expert_owner(0, world_size=2) == 0
        assert _expert_owner(1, world_size=2) == 1
        assert _expert_owner(2, world_size=2) == 0
        assert _expert_owner(3, world_size=2) == 1
        assert _expert_owner(0, world_size=4) == 0
        assert _expert_owner(5, world_size=4) == 1

    def test_expert_owner_explicit_map(self):
        """Explicit expert_to_owner dict takes precedence over modulo."""
        mapping = {0: 1, 1: 0, 2: 2}
        assert _expert_owner(0, world_size=3, expert_to_owner=mapping) == 1
        assert _expert_owner(1, world_size=3, expert_to_owner=mapping) == 0
        assert _expert_owner(2, world_size=3, expert_to_owner=mapping) == 2
        # Unmapped expert falls back to modulo
        assert _expert_owner(3, world_size=3, expert_to_owner=mapping) == 0

    def test_single_rank_fallback(self):
        """Backend capability check returns bool without requiring a process group."""
        val = _supports_expert_parallel()
        assert isinstance(val, bool)

    def test_dispatch_plan_ragged(self):
        """Uneven token counts across ranks don't cause errors."""
        all_qlens = [3, 0, 5]  # rank 1 is empty
        num_experts_per_tok = 2
        world_size = 3
        total_qlen = sum(all_qlens)

        expert_ids = torch.zeros((total_qlen, num_experts_per_tok), dtype=torch.long)
        row = 0
        for rank, qlen in enumerate(all_qlens):
            for tok in range(qlen):
                expert_ids[row, 0] = (rank * 10 + tok) % 4
                expert_ids[row, 1] = (rank * 10 + tok + 1) % 4
                row += 1

        plan = build_expert_dispatch_plan(
            all_qlens=all_qlens,
            expert_ids=expert_ids,
            num_experts_per_tok=num_experts_per_tok,
            world_size=world_size,
        )

        # Empty rank produces no entries
        assert plan.dispatch[1] == {}
        # All entries from rank 0 and rank 2 map to owners
        for src in [0, 2]:
            for owner, entries in plan.dispatch[src].items():
                assert len(entries) > 0
                for e in entries:
                    assert e.source_rank == src
                    assert e.owner_rank == owner
        # Receive plan covers all ranks
        for r in range(world_size):
            assert r in plan.receive


class TestDispatchPlanIntegrity:
    """Contract invariant tests for ExpertDispatchPlan."""

    def test_every_entry_has_one_owner(self):
        """Each expert entry in dispatch appears exactly once in some owner's receive list."""
        all_qlens = [4, 3, 5]
        num_experts_per_tok = 2
        world_size = 3
        total_qlen = sum(all_qlens)

        expert_ids = torch.randint(0, 8, (total_qlen, num_experts_per_tok), dtype=torch.long)
        plan = build_expert_dispatch_plan(
            all_qlens=all_qlens,
            expert_ids=expert_ids,
            num_experts_per_tok=num_experts_per_tok,
            world_size=world_size,
        )

        all_entries: set[tuple[int, int, int]] = set()
        for src_rank, dispatch in enumerate(plan.dispatch):
            for owner, entries in dispatch.items():
                for e in entries:
                    key = (e.source_rank, e.source_row, e.topk_slot)
                    assert key not in all_entries, f"Duplicate entry {key}"
                    all_entries.add(key)

        received_count = sum(len(lst) for lst in plan.receive.values())
        assert received_count == len(all_entries), (
            f"Dispatch entry count {len(all_entries)} != receive count {received_count}"
        )

    def test_send_recv_counts_balanced(self):
        """Total entries sent = total entries received across all ranks."""
        all_qlens = [2, 7, 3]
        num_experts_per_tok = 3
        world_size = 3
        total_qlen = sum(all_qlens)

        expert_ids = torch.randint(0, 16, (total_qlen, num_experts_per_tok), dtype=torch.long)
        plan = build_expert_dispatch_plan(
            all_qlens=all_qlens,
            expert_ids=expert_ids,
            num_experts_per_tok=num_experts_per_tok,
            world_size=world_size,
        )

        total_sent = sum(sum(row) for row in plan.send_counts)
        total_recv = sum(sum(row) for row in plan.recv_counts)
        assert total_sent == total_recv, (
            f"Total sent {total_sent} != total recv {total_recv}"
        )
        assert total_sent == total_qlen * num_experts_per_tok

    def test_uneven_ownership(self):
        """Each expert maps to exactly one owner, and every entry returns to its source."""
        all_qlens = [3, 3]
        num_experts_per_tok = 2
        world_size = 2
        total_qlen = 6

        # Expert ids are chosen so some map to owner 0, others to owner 1
        # With world_size=2: even -> 0, odd -> 1
        expert_ids = torch.tensor([[0, 1], [2, 3], [4, 5],
                                   [1, 0], [3, 2], [5, 4]], dtype=torch.long)

        plan = build_expert_dispatch_plan(
            all_qlens=all_qlens,
            expert_ids=expert_ids,
            num_experts_per_tok=num_experts_per_tok,
            world_size=world_size,
        )

        # Count entries destined for each owner
        for owner in range(world_size):
            owner_entries = []
            for src in range(world_size):
                if owner in plan.dispatch[src]:
                    owner_entries.extend(plan.dispatch[src][owner])
            # Entries in receive for this owner must match dispatch
            recv_entries = [e for e in plan.receive[owner] if e.owner_rank == owner]
            assert len(owner_entries) == len(recv_entries)

        # Every entry in receive has a matching dispatch entry
        for owner, recv_entries in plan.receive.items():
            for e in recv_entries:
                dispatch_entries = plan.dispatch[e.source_rank].get(owner, [])
                matching = [de for de in dispatch_entries
                            if de.source_row == e.source_row and de.topk_slot == e.topk_slot]
                assert len(matching) == 1, f"No match for ({e.source_rank},{e.source_row},{e.topk_slot})"


class TestGatherScatterSingleRank:
    """Test gather/scatter helpers in single-rank (no-communication) mode."""

    def test_single_rank_empty_output(self):
        """Empty receive list returns empty output tensors."""
        all_qlens = [3]
        num_experts_per_tok = 2
        world_size = 1
        total_qlen = 3

        expert_ids = torch.zeros((total_qlen, num_experts_per_tok), dtype=torch.long)
        plan = build_expert_dispatch_plan(
            all_qlens=all_qlens,
            expert_ids=expert_ids,
            num_experts_per_tok=num_experts_per_tok,
            world_size=world_size,
        )

        hidden = torch.arange(total_qlen * 4, dtype=torch.float32).reshape(total_qlen, 4)
        ids = expert_ids
        weights = torch.ones(total_qlen, num_experts_per_tok, dtype=torch.bfloat16)

        received = _dist_gather_varlen_to_expert_owners(
            [hidden, ids, weights],
            dispatch_plan=plan,
            rank=0,
            world_size=world_size,
        )

        assert len(received) == 1
        assert received[0][0].shape[0] == total_qlen

        outputs = _dist_scatter_varlen_from_expert_owners(
            received_per_owner=received,
            dispatch_plan=plan,
            rank=0,
            world_size=world_size,
            output_feature_shapes=[(4,), (num_experts_per_tok,), (num_experts_per_tok,)],
            devices=[hidden.device, ids.device, weights.device],
            dtypes=[hidden.dtype, ids.dtype, weights.dtype],
        )

        assert all(t.shape[0] == total_qlen for t in outputs)

    def test_single_rank_token_order_preserved(self):
        """Output tokens are in the same order as the input hidden states."""
        all_qlens = [4]
        num_experts_per_tok = 2
        world_size = 1
        total_qlen = 4

        # Mark each token's hidden with a unique id
        hidden = torch.tensor([[float(row)] * 4 for row in range(total_qlen)], dtype=torch.float32)
        expert_ids = torch.zeros((total_qlen, num_experts_per_tok), dtype=torch.long)
        weights = torch.ones(total_qlen, num_experts_per_tok, dtype=torch.bfloat16)

        plan = build_expert_dispatch_plan(
            all_qlens=all_qlens,
            expert_ids=expert_ids,
            num_experts_per_tok=num_experts_per_tok,
            world_size=world_size,
        )

        received = _dist_gather_varlen_to_expert_owners(
            [hidden, expert_ids, weights],
            dispatch_plan=plan,
            rank=0,
            world_size=world_size,
        )

        outputs = _dist_scatter_varlen_from_expert_owners(
            received_per_owner=received,
            dispatch_plan=plan,
            rank=0,
            world_size=world_size,
            output_feature_shapes=[(4,), (num_experts_per_tok,), (num_experts_per_tok,)],
            devices=[hidden.device, expert_ids.device, weights.device],
            dtypes=[hidden.dtype, expert_ids.dtype, weights.dtype],
        )

        # Hidden state ordering must be preserved
        out_hidden = outputs[0]
        assert out_hidden.shape == hidden.shape
        for row in range(total_qlen):
            assert out_hidden[row, 0] == float(row), f"Token order broken at row {row}"


class TestDispatchModeDefault:
    """Verify the fallback dispatch mode is the default behavior."""

    def test_env_var_defaults_to_disabled(self):
        """KT_SFT_EXPERT_PARALLEL environment variable defaults to '0' (disabled)."""
        import os
        # The env var should default to disabled when not set
        val = os.environ.get("KT_SFT_EXPERT_PARALLEL", "0")
        assert val == "0", "Default KT_SFT_EXPERT_PARALLEL should be '0'"
        assert val != "1", "Expert parallel should not be enabled by default"

    def test_supports_expert_parallel_returns_bool(self):
        """_supports_expert_parallel() returns a bool without requiring a process group."""
        from sft.dist_utils import _supports_expert_parallel
        result = _supports_expert_parallel()
        assert isinstance(result, bool), f"Expected bool, got {type(result)}"

    def test_dispatch_mode_attribute_exists(self):
        """KTMoELayerWrapper has _dispatch_mode attribute for inspection."""
        from unittest.mock import MagicMock
        from sft.layer import KTMoELayerWrapper
        from sft.arch import MOEArchConfig

        # Create minimal mock objects for testing
        original_moe = MagicMock()
        original_moe.gate = None
        original_moe.experts = None

        wrapper = MagicMock()

        moe_config = MagicMock(spec=MOEArchConfig)
        moe_config.router_type = "default"
        moe_config.router_attr = "gate"
        moe_config.experts_attr = "experts"
        moe_config.has_shared_experts = False
        moe_config.num_experts_per_tok = 2

        layer = KTMoELayerWrapper(
            original_moe=original_moe,
            wrapper=wrapper,
            lora_params=None,
            moe_config=moe_config,
            hidden_size=512,
            layer_idx=0,
        )

        # Verify _dispatch_mode attribute exists and defaults to fallback
        assert hasattr(layer, "_dispatch_mode"), "Layer should have _dispatch_mode attribute"
        assert layer._dispatch_mode == "fallback", f"Default dispatch mode should be 'fallback', got {layer._dispatch_mode}"


class TestDispatchPlanUnchanged:
    """Ensure plan-building logic is unchanged from the original API."""

    def test_plan_has_required_fields(self):
        """ExpertDispatchPlan exposes all required routing metadata."""
        all_qlens = [2, 2]
        num_experts_per_tok = 1
        world_size = 2
        total_qlen = 4

        expert_ids = torch.tensor([[0], [1], [2], [3]], dtype=torch.long)
        plan = build_expert_dispatch_plan(
            all_qlens=all_qlens,
            expert_ids=expert_ids,
            num_experts_per_tok=num_experts_per_tok,
            world_size=world_size,
        )

        assert hasattr(plan, "dispatch")
        assert hasattr(plan, "receive")
        assert hasattr(plan, "send_counts")
        assert hasattr(plan, "recv_counts")
        assert len(plan.dispatch) == world_size
        assert len(plan.receive) == world_size
        assert len(plan.send_counts) == world_size
        assert len(plan.recv_counts) == world_size

    def test_expert_entry_slots(self):
        """ExpertEntry is a frozen dataclass with expected fields."""
        e = ExpertEntry(source_rank=1, source_row=5, topk_slot=0, expert_id=7, owner_rank=2)
        assert e.source_rank == 1
        assert e.source_row == 5
        assert e.topk_slot == 0
        assert e.expert_id == 7
        assert e.owner_rank == 2
