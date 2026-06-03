"""Observe-only must skip [1,1,Q,K] tensor allocation.

Regression for a 3.3x per-turn slowdown: in observe mode the pre-hook
was unconditionally calling build_additive_mask, which materialized a
full [1,1,Q,K] mask tensor (~5GB fp16 at Q=K~50K) per layer per call,
then immediately discarded it. The fix: each method short-circuits and
returns None when self.observe_only=True, AFTER updating record state.

This test pins two contracts:
1. build_additive_mask returns None for every method when observe_only=True
2. State that the recorder consumes (kept_count, record_metadata) is still
   correctly updated — observe-only semantics unchanged
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from serving.recording.attention_bus import AttentionBus
from serving.sparse_attention import build_sparse_attention
from serving.sparse_attention.base import (
    SparseAttentionConfig,
    SparseAttentionContext,
)


@pytest.mark.parametrize("method_name", ["sliding", "streaming"])
def test_sliding_observe_only_returns_none_prefill(method_name: str) -> None:
    config = SparseAttentionConfig(
        name=method_name,
        sink_size=4,
        recent_window=8,
        record=True,
        observe_only=True,
    )
    method = build_sparse_attention(config, num_layers=1)
    mask = method.build_additive_mask(
        layer_idx=0,
        query_len=16,
        key_len=16,
        phase="prefill",
        decode_step=-1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert mask is None, f"{method_name} observe-only prefill must return None"
    # kept_count is a pure function of key_len for sliding — no state to check.
    assert method.kept_count(16) > 0


@pytest.mark.parametrize("method_name", ["sliding", "streaming"])
def test_sliding_observe_only_returns_none_decode(method_name: str) -> None:
    config = SparseAttentionConfig(
        name=method_name,
        sink_size=4,
        recent_window=8,
        record=True,
        observe_only=True,
    )
    method = build_sparse_attention(config, num_layers=1)
    mask = method.build_additive_mask(
        layer_idx=0,
        query_len=1,
        key_len=32,
        phase="decode",
        decode_step=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert mask is None
    assert method.kept_count(32) > 0


@pytest.mark.parametrize("method_name", ["block_topk", "quest", "heavy_hitter"])
def test_dynamic_observe_only_returns_none_prefill_phase_dense(method_name: str) -> None:
    """Prefill / phase_dense early return: must skip mask but still _record state."""
    bus = AttentionBus()
    config = SparseAttentionConfig(
        name=method_name,
        sink_size=4,
        recent_window=8,
        budget=16,
        block_size=4,
        record=True,
        observe_only=True,
        phase_scope="decode_only",
    )
    method = build_sparse_attention(config, num_layers=1, attention_bus=bus)
    mask = method.build_additive_mask(
        layer_idx=0,
        query_len=16,
        key_len=16,
        phase="prefill",
        decode_step=-1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert mask is None, f"{method_name} observe-only prefill must return None"
    md = method.record_metadata(layer_idx=0, phase="prefill", decode_step=-1)
    assert md.get("selection_reason") in {"phase_dense", "prefill_dense"}, (
        f"{method_name} must record phase_dense/prefill_dense reason; got {md}"
    )
    # kept_count must reflect the dense fallback (all key positions kept).
    assert method.kept_count(16) == 16, (
        f"{method_name} dense fallback must record kept_count==key_len; "
        f"got {method.kept_count(16)}"
    )


@pytest.mark.parametrize("method_name", ["block_topk", "quest", "heavy_hitter"])
def test_dynamic_observe_only_returns_none_empty_key_len(method_name: str) -> None:
    """key_len <= 0 early return: must skip mask, still _record reason=empty."""
    bus = AttentionBus()
    config = SparseAttentionConfig(
        name=method_name,
        sink_size=4,
        recent_window=8,
        budget=16,
        block_size=4,
        record=True,
        observe_only=True,
        phase_scope="decode_only",
    )
    method = build_sparse_attention(config, num_layers=1, attention_bus=bus)
    mask = method.build_additive_mask(
        layer_idx=0,
        query_len=1,
        key_len=0,
        phase="decode",
        decode_step=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert mask is None, f"{method_name} observe-only key_len=0 must return None"
    md = method.record_metadata(layer_idx=0, phase="decode", decode_step=0)
    assert md.get("selection_reason") == "empty", (
        f"{method_name} key_len=0 must record reason=empty; got {md}"
    )


def test_enforce_mode_still_returns_tensor() -> None:
    """Regression: enforce mode (observe_only=False) must still return a Tensor."""
    config = SparseAttentionConfig(
        name="sliding",
        sink_size=4,
        recent_window=8,
        record=False,
        observe_only=False,
    )
    method = build_sparse_attention(config, num_layers=1)
    mask = method.build_additive_mask(
        layer_idx=0,
        query_len=8,
        key_len=16,
        phase="prefill",
        decode_step=-1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert mask is not None and mask.shape == (1, 1, 8, 16)


@pytest.mark.parametrize("method_name", ["block_topk", "quest", "heavy_hitter"])
def test_enforce_mode_dynamic_still_returns_tensor_prefill(method_name: str) -> None:
    """Dynamic methods in enforce mode keep the dense-fallback tensor for prefill."""
    bus = AttentionBus()
    config = SparseAttentionConfig(
        name=method_name,
        sink_size=4,
        recent_window=8,
        budget=16,
        block_size=4,
        record=True,
        observe_only=False,
        phase_scope="decode_only",
    )
    method = build_sparse_attention(config, num_layers=1, attention_bus=bus)
    mask = method.build_additive_mask(
        layer_idx=0,
        query_len=8,
        key_len=16,
        phase="prefill",
        decode_step=-1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert mask is not None, f"{method_name} enforce prefill must return a tensor"
    assert mask.shape == (1, 1, 8, 16), (
        f"{method_name} enforce prefill shape: got {tuple(mask.shape)}"
    )


# -- decode (Q=1) observe-only short-circuit tests ---------------------------
#
# Existing tests cover dynamic-method PREFILL (phase_dense early return) and
# key_len=0 (empty early return). The decode (Q=1) path is the interesting
# one: _rank_middle_positions -> cap_middle_selection -> keep.update(...) ->
# _record(...) all run BEFORE the observe-only short-circuit. A future
# refactor that moves `if self.observe_only: return None` ahead of `_record`
# would silently produce reason="selected" with zero recorded state. These
# tests fail loudly in that case.


class _ToyAttention(nn.Module):
    """Mirrors the fixture from tests/test_sparse_attn_dynamic_methods.py."""

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


def _decode_context(
    *, cached_keys: torch.Tensor, hidden: torch.Tensor
) -> SparseAttentionContext:
    cos = torch.ones((1, hidden.shape[-2], hidden.shape[-1]), dtype=hidden.dtype)
    sin = torch.zeros_like(cos)
    return SparseAttentionContext(
        module=_ToyAttention(),
        hidden_states=hidden,
        position_embeddings=(cos, sin),
        past_key_values=_FakeCache(cached_keys),
        attention_mask=None,
    )


@pytest.mark.parametrize("method_name", ["block_topk", "quest"])
def test_dynamic_observe_only_decode_short_circuit(method_name: str) -> None:
    """Decode (Q=1) observe-only: mask is None but rank->keep->record ran."""
    bus = AttentionBus()
    config = SparseAttentionConfig(
        name=method_name,
        sink_size=1,
        recent_window=1,
        budget=4,
        block_size=2,
        record=True,
        observe_only=True,
        phase_scope="decode_only",
    )
    method = build_sparse_attention(config, num_layers=1, attention_bus=bus)
    # key_len=9 > sink+recent so there is a non-empty middle band to rank.
    cached = torch.zeros((1, 1, 8, 4), dtype=torch.float32)
    cached[0, 0, 4, 0] = 9.0  # dominant middle key
    cached[0, 0, 3, 0] = 5.0
    hidden = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])

    mask = method.build_additive_mask(
        layer_idx=0,
        query_len=1,
        key_len=9,
        phase="decode",
        decode_step=0,
        device=hidden.device,
        dtype=hidden.dtype,
        context=_decode_context(cached_keys=cached, hidden=hidden),
    )

    assert mask is None, f"{method_name} observe-only decode must return None"
    md = method.record_metadata(layer_idx=0, phase="decode", decode_step=0)
    assert md["selection_reason"] == "selected", (
        f"{method_name} decode must record reason='selected' (proves the "
        f"rank->cap->keep->_record pipeline ran before short-circuit); got {md}"
    )
    # kept_count should be positive (sink + recent + at least one middle pick).
    assert method.kept_count(9) > 0, (
        f"{method_name} decode observe-only must update kept_count; "
        f"got {method.kept_count(9)}"
    )


def test_heavy_hitter_observe_only_decode_short_circuit() -> None:
    """Decode (Q=1) observe-only heavy_hitter: requires seeded scores.

    Without a prior `observe(...)` publish, `_rank_middle_positions` returns
    []  and the reason degenerates to 'sink_recent_no_scores'. We seed the
    score buffer the realistic way: publish a synthetic prefill attention
    tensor through the bus (option (a) in the task spec — kept under 10
    lines by mirroring test_heavy_hitter_selects_previous_attention_scores).
    """
    bus = AttentionBus()
    config = SparseAttentionConfig(
        name="heavy_hitter",
        sink_size=1,
        recent_window=1,
        budget=4,
        block_size=2,
        record=True,
        observe_only=True,
        phase_scope="decode_only",
    )
    method = build_sparse_attention(config, num_layers=1, attention_bus=bus)
    # Seed scores via the bus so _rank_middle_positions returns a real ranking.
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
        device=torch.device("cpu"),
        dtype=torch.float32,
        context=None,
    )

    assert mask is None, "heavy_hitter observe-only decode must return None"
    md = method.record_metadata(layer_idx=0, phase="decode", decode_step=1)
    assert md["selection_reason"] == "selected", (
        f"heavy_hitter decode must record reason='selected' (proves the "
        f"rank->cap->keep->_record pipeline ran before short-circuit); got {md}"
    )
    assert method.kept_count(8) > 0, (
        f"heavy_hitter decode observe-only must update kept_count; "
        f"got {method.kept_count(8)}"
    )
