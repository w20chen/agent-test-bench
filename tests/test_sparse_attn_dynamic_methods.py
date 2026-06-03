"""Focused tests for dynamic sparse attention methods."""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from serving.recording.attention_bus import AttentionBus
from serving.sparse_attention.base import SparseAttentionContext
from serving.sparse_attention.block_topk import BlockTopKSparseAttention
from serving.sparse_attention.heavy_hitter import HeavyHitterSparseAttention
from serving.sparse_attention.quest import QuestSparseAttention


class _ToyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_idx = 0
        self.head_dim = 4
        self.scaling = 1.0
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(4))
            self.k_proj.weight.copy_(torch.eye(4))


class _FakeCache:
    def __init__(self, key_states: torch.Tensor) -> None:
        self.key_states = key_states

    def __getitem__(self, layer_idx: int):
        if layer_idx != 0:
            raise KeyError(layer_idx)
        return self.key_states, None

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx != 0:
            raise KeyError(layer_idx)
        return int(self.key_states.shape[-2])


def _context(*, cached_keys: torch.Tensor, hidden: torch.Tensor) -> SparseAttentionContext:
    cos = torch.ones((1, hidden.shape[-2], hidden.shape[-1]), dtype=hidden.dtype)
    sin = torch.zeros_like(cos)
    return SparseAttentionContext(
        module=_ToyAttention(),
        hidden_states=hidden,
        position_embeddings=(cos, sin),
        past_key_values=_FakeCache(cached_keys),
        attention_mask=None,
    )


def _kept(mask: torch.Tensor) -> set[int]:
    return {int(i) for i in torch.nonzero(mask[0, 0, 0] == 0.0).flatten().tolist()}


def test_block_topk_selects_blocks_from_current_qk() -> None:
    cached = torch.zeros((1, 1, 8, 4), dtype=torch.float32)
    cached[0, 0, 3, 0] = 10.0
    cached[0, 0, 4, 0] = 8.0
    hidden = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    method = BlockTopKSparseAttention(
        budget=5,
        sink_size=1,
        recent_window=1,
        block_size=2,
    )

    mask = method.build_additive_mask(
        layer_idx=0,
        query_len=1,
        key_len=9,
        phase="decode",
        decode_step=0,
        device=hidden.device,
        dtype=hidden.dtype,
        context=_context(cached_keys=cached, hidden=hidden),
    )

    assert mask.shape == (1, 1, 1, 9)
    assert _kept(mask) == {0, 2, 3, 4, 8}
    meta = method.record_metadata(layer_idx=0, phase="decode", decode_step=0)
    assert meta["selected_blocks"][:2] == [1, 2]
    assert meta["selected_middle_indices"] == [2, 3, 4]
    assert method.kept_count(9) == 5


def test_quest_uses_page_minmax_envelope() -> None:
    cached = torch.zeros((1, 1, 8, 4), dtype=torch.float32)
    cached[0, 0, 5, 0] = 6.0
    cached[0, 0, 2, 0] = 3.0
    hidden = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    method = QuestSparseAttention(
        budget=4,
        sink_size=1,
        recent_window=1,
        block_size=2,
    )

    mask = method.build_additive_mask(
        layer_idx=0,
        query_len=1,
        key_len=9,
        phase="decode",
        decode_step=0,
        device=hidden.device,
        dtype=hidden.dtype,
        context=_context(cached_keys=cached, hidden=hidden),
    )

    assert _kept(mask) == {0, 4, 5, 8}
    meta = method.record_metadata(layer_idx=0, phase="decode", decode_step=0)
    assert meta["selected_pages"][0] == 2
    assert meta["selected_middle_indices"] == [4, 5]


def test_heavy_hitter_selects_previous_attention_scores() -> None:
    bus = AttentionBus()
    method = HeavyHitterSparseAttention(
        budget=4,
        sink_size=1,
        recent_window=1,
        block_size=2,
        num_layers=1,
        attention_bus=bus,
    )
    attn = torch.zeros((1, 1, 1, 8), dtype=torch.float32)
    attn[0, 0, 0, 4] = 0.7
    attn[0, 0, 0, 2] = 0.3
    bus.publish(
        layer=0,
        attn=attn,
        query_positions=torch.tensor([7]),
        key_len=8,
        phase="decode",
        suspended=False,
    )

    mask = method.build_additive_mask(
        layer_idx=0,
        query_len=1,
        key_len=8,
        phase="decode",
        decode_step=1,
        device=attn.device,
        dtype=attn.dtype,
        context=None,
    )

    assert _kept(mask) == {0, 2, 4, 7}
    meta = method.record_metadata(layer_idx=0, phase="decode", decode_step=1)
    assert meta["selection_reason"] == "selected"
    assert meta["selected_middle_indices"] == [4, 2]


def test_dynamic_methods_keep_prefill_dense_causal() -> None:
    method = BlockTopKSparseAttention(
        budget=4,
        sink_size=1,
        recent_window=1,
        block_size=2,
    )
    mask = method.build_additive_mask(
        layer_idx=0,
        query_len=4,
        key_len=4,
        phase="prefill",
        device=torch.device("cpu"),
        dtype=torch.float32,
        context=None,
    )
    neg_inf = torch.finfo(torch.float32).min
    for q in range(4):
        for k in range(4):
            expected = 0.0 if k <= q else pytest.approx(neg_inf)
            assert float(mask[0, 0, q, k]) == expected
    meta = json.dumps(method.record_metadata(layer_idx=0, phase="prefill", decode_step=-1))
    assert "phase_dense" in meta


def test_block_topk_decode_requires_context() -> None:
    method = BlockTopKSparseAttention(
        budget=4,
        sink_size=1,
        recent_window=1,
        block_size=2,
    )
    with pytest.raises(ValueError, match="position_embeddings"):
        method.build_additive_mask(
            layer_idx=0,
            query_len=1,
            key_len=8,
            phase="decode",
            device=torch.device("cpu"),
            dtype=torch.float32,
            context=None,
        )


def test_heavy_hitter_subscribes_to_full_prefill_bus() -> None:
    """Regression for round-2 review #1.

    Decode mask decisions for heavy_hitter rank middle keys by historical
    post-softmax attention. If the bus only delivers sampled prefill rows
    (max_prefill_queries=80 by default), the rank buffer is built from a
    biased subset and decode selections degenerate toward sink+recent
    fallback. The fix is the class-level `prefill_observe_mode = "full"`
    which opts heavy_hitter into the full-prefill bus path.

    A future refactor that flips it back to "sampled" silently corrupts
    every observe-only heavy_hitter trace; this test pins the contract.
    """
    from serving.sparse_attention.heavy_hitter import HeavyHitterSparseAttention

    assert HeavyHitterSparseAttention.prefill_observe_mode == "full", (
        "heavy_hitter must subscribe to full-prefill bus; sampled mode "
        "leaves the score buffer built from a 80-row subset and decode "
        "selections fall back to sink+recent."
    )


def test_heavy_hitter_reset_state_clears_scores() -> None:
    bus = AttentionBus()
    method = HeavyHitterSparseAttention(
        budget=4,
        sink_size=1,
        recent_window=1,
        block_size=2,
        num_layers=1,
        attention_bus=bus,
    )
    attn = torch.zeros((1, 1, 1, 8), dtype=torch.float32)
    attn[0, 0, 0, 4] = 1.0
    bus.publish(
        layer=0,
        attn=attn,
        query_positions=torch.tensor([7]),
        key_len=8,
        phase="decode",
        suspended=False,
    )
    method.reset_state()
    mask = method.build_additive_mask(
        layer_idx=0,
        query_len=1,
        key_len=8,
        phase="decode",
        device=torch.device("cpu"),
        dtype=torch.float32,
        context=None,
    )
    assert _kept(mask) == {0, 7}
    assert method.record_metadata(layer_idx=0, phase="decode", decode_step=0)[
        "selection_reason"
    ] == "sink_recent_no_scores"
