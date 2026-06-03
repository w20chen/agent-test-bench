"""Unit tests for `H2OCache` (plan G16#2 + bus subscription contract).

Direct exercises of `observe()` and `_decide_evict()` without driving the
full `update()` path through transformers — that is covered by the smoke
script (`scripts/spikes/step6_h2o_smoke.py`).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from serving.kv_policies.base import EvictionPolicyConfig
from serving.kv_policies.h2o import H2OCache
from serving.kv_policies.recorder import KVEvictionRecorder
from serving.recording.attention_bus import AttentionBus


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


def _make_cache(
    *,
    budget: int = 4,
    sink_size: int = 1,
    recent_window: int = 1,
    num_layers: int = 2,
    max_position_embeddings: int = 64,
    bus: AttentionBus | None = None,
    prefill_mode: str = "full",
    recorder: KVEvictionRecorder | None = None,
) -> tuple[H2OCache, AttentionBus]:
    """Build an H2OCache + the bus it subscribes to."""
    config = EvictionPolicyConfig(
        name="h2o",
        budget=budget,
        sink_size=sink_size,
        recent_window=recent_window,
        prefill_mode=prefill_mode,  # type: ignore[arg-type]
    )
    bus = bus or AttentionBus()
    cache = H2OCache(
        config,
        num_layers=num_layers,
        recorder=recorder,
        attention_bus=bus,
        max_position_embeddings=max_position_embeddings,
    )
    return cache, bus


def _attn_with_peaks(
    *,
    key_len: int,
    peak_positions: list[int],
    n_heads: int = 2,
    n_query_rows: int = 1,
    peak_value: float = 10.0,
) -> torch.Tensor:
    """Construct a (B=1, H, Q, K) attn tensor with mass concentrated on `peak_positions`.

    Each query row puts `peak_value` on each peak position and 0 elsewhere.
    Not a real softmax distribution (doesn't sum to 1) — but `H2OCache.observe`
    only sums and head-means, so the absolute scale is what matters for the
    decision contract.
    """
    attn = torch.zeros(1, n_heads, n_query_rows, key_len, dtype=torch.float32)
    for pos in peak_positions:
        attn[..., pos] = peak_value
    return attn


# ---------------------------------------------------------------------------
# G16#2: heavy-hitter eviction picks the right positions.
# ---------------------------------------------------------------------------


def test_h2o_evicts_lowest_score() -> None:
    """Plan G16#2: positions {3, 7} score 10, others 0; budget=4 sink=1 recent=1.

    Expected keep set:
      sink   = {0}                         (position 0)
      recent = {key_len - 1}               (position 9)
      heavy  = top 2 of middle [1, 9) by score = {3, 7}

    -> keep = sorted({0, 3, 7, 9}) = [0, 3, 7, 9], evict = [1, 2, 4, 5, 6, 8].
    """
    cache, _bus = _make_cache(
        budget=4, sink_size=1, recent_window=1, max_position_embeddings=16
    )
    key_len = 10
    attn = _attn_with_peaks(key_len=key_len, peak_positions=[3, 7])
    cache.observe(
        layer=0,
        attn=attn,
        query_positions=torch.tensor([0], dtype=torch.long),
        key_len=key_len,
        phase="prefill",
    )

    decision = cache._decide_evict(layer_idx=0, key_len=key_len)
    assert decision.reason == "over_budget"
    # Heavy-hitter set must include the two peaks; sink + recent always kept.
    assert set(decision.keep_indices) >= {0, 3, 7, 9}
    assert decision.keep_indices == sorted(decision.keep_indices)
    # Post-eviction length == budget.
    assert len(decision.keep_indices) == 4
    # Partition coverage.
    assert sorted(decision.keep_indices + decision.evict_indices) == list(range(key_len))
    # policy_state carries the topk diagnostics for npz.
    state = decision.policy_state
    assert state is not None
    assert set(state["score_topk_index"]) == {3, 7}
    # Score values match the cumulative head-mean we accumulated.
    # Each peak got 10.0 across all heads -> mean = 10.0 (one query row).
    assert all(abs(v - 10.0) < 1e-5 for v in state["score_topk_value"])
    assert state["score_evicted_index"] == [1, 2, 4, 5, 6, 8]
    assert all(abs(v) < 1e-5 for v in state["score_evicted_value"])


def test_h2o_under_budget_no_eviction() -> None:
    cache, _bus = _make_cache(
        budget=4, sink_size=1, recent_window=1, max_position_embeddings=16
    )
    decision = cache._decide_evict(layer_idx=0, key_len=4)
    assert decision.reason == "none"
    assert decision.evict_indices == []
    assert decision.keep_indices == list(range(4))


# ---------------------------------------------------------------------------
# Bus + suspend interaction.
# ---------------------------------------------------------------------------


def test_h2o_subscribes_with_always_active() -> None:
    """always_active=True so suspend gating cannot starve the score buffer."""
    bus = AttentionBus()
    cache, _ = _make_cache(bus=bus, max_position_embeddings=16)
    assert bus.n_consumers() == 1
    assert getattr(cache, "always_active") is True

    key_len = 8
    attn = _attn_with_peaks(key_len=key_len, peak_positions=[2, 5])
    bus.publish(
        layer=0,
        attn=attn,
        query_positions=torch.tensor([0], dtype=torch.long),
        key_len=key_len,
        phase="decode",
        suspended=True,  # gating ON
    )

    # Score buffer must have absorbed the publish despite the gate.
    buffer = cache._scores[0]
    assert buffer is not None
    # Peaks at 2 and 5 should match peak_value (head-mean of constant peak).
    assert float(buffer[2]) > 0
    assert float(buffer[5]) > 0
    # Off-peak slot stayed zero.
    assert float(buffer[1]) == 0


def test_h2o_unsubscribe_works() -> None:
    """Manual unsubscribe stops further observations (provider drives this)."""
    bus = AttentionBus()
    cache, _ = _make_cache(bus=bus, max_position_embeddings=16)
    assert bus.n_consumers() == 1

    bus.unsubscribe(cache)
    assert bus.n_consumers() == 0

    key_len = 8
    attn = _attn_with_peaks(key_len=key_len, peak_positions=[2])
    bus.publish(
        layer=0,
        attn=attn,
        query_positions=torch.tensor([0], dtype=torch.long),
        key_len=key_len,
        phase="prefill",
        suspended=False,
    )
    # observe() never ran -> score buffer is still None.
    assert cache._scores[0] is None


# ---------------------------------------------------------------------------
# Score buffer compaction after a triggered eviction.
# ---------------------------------------------------------------------------


def test_h2o_post_evict_compacts_scores() -> None:
    """After eviction, score buffer must mirror the keep set.

    Drive: observe -> _decide_evict (returns keep + heavy + recent) ->
    `_post_evict_hook(decision)`. Verify buffer slots match the
    `buffer[keep_indices]` permutation, with the tail zeroed.
    """
    cache, _bus = _make_cache(
        budget=4, sink_size=1, recent_window=1, max_position_embeddings=16
    )
    key_len = 10
    attn = _attn_with_peaks(key_len=key_len, peak_positions=[3, 7])
    cache.observe(
        layer=0,
        attn=attn,
        query_positions=torch.tensor([0], dtype=torch.long),
        key_len=key_len,
        phase="prefill",
    )

    # Snapshot buffer + decision (without going through the K/V path).
    buffer_before = cache._scores[0].clone()
    decision = cache._decide_evict(layer_idx=0, key_len=key_len)
    cache._post_evict_hook(layer_idx=0, decision=decision)

    buffer_after = cache._scores[0]
    n_keep = len(decision.keep_indices)
    # Live prefix matches the gathered slots.
    expected = buffer_before[torch.as_tensor(decision.keep_indices, dtype=torch.long)]
    assert torch.allclose(buffer_after[:n_keep], expected)
    # Tail past the keep set is zero (so future observe() into a now-empty
    # slot writes the correct cumulative sum from 0).
    assert torch.all(buffer_after[n_keep:] == 0)
    assert cache._score_lengths[0] == n_keep


# ---------------------------------------------------------------------------
# Construction / config validation.
# ---------------------------------------------------------------------------


def test_h2o_requires_attention_bus() -> None:
    config = EvictionPolicyConfig(
        name="h2o", budget=4, sink_size=1, recent_window=1
    )
    with pytest.raises(ValueError, match="attention_bus"):
        H2OCache(
            config,
            num_layers=2,
            attention_bus=None,  # type: ignore[arg-type]
            max_position_embeddings=64,
        )


def test_h2o_requires_max_position_embeddings() -> None:
    config = EvictionPolicyConfig(
        name="h2o", budget=4, sink_size=1, recent_window=1
    )
    bus = AttentionBus()
    with pytest.raises(ValueError, match="max_position_embeddings"):
        H2OCache(
            config,
            num_layers=2,
            attention_bus=bus,
            max_position_embeddings=None,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="max_position_embeddings"):
        H2OCache(
            config,
            num_layers=2,
            attention_bus=bus,
            max_position_embeddings=0,
        )


def test_h2o_validates_budget_floor() -> None:
    bus = AttentionBus()
    with pytest.raises(ValueError, match="budget >= sink_size \\+ recent_window"):
        H2OCache(
            EvictionPolicyConfig(
                name="h2o", budget=4, sink_size=4, recent_window=4
            ),
            num_layers=2,
            attention_bus=bus,
            max_position_embeddings=64,
        )


def test_h2o_unknown_aggregate_raises() -> None:
    """`sum`, `mean`, `ema` are wired (step 8); anything else must raise."""
    bus = AttentionBus()
    with pytest.raises(ValueError, match="unsupported"):
        H2OCache(
            EvictionPolicyConfig(
                name="h2o",
                budget=4,
                sink_size=1,
                recent_window=1,
                aggregate="bogus",  # type: ignore[arg-type]
            ),
            num_layers=2,
            attention_bus=bus,
            max_position_embeddings=64,
        )


def test_h2o_ema_decay_must_be_in_range() -> None:
    """ema_decay <= 0 or >= 1 makes the EMA degenerate; reject loudly."""
    bus = AttentionBus()
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="ema_decay"):
            H2OCache(
                EvictionPolicyConfig(
                    name="h2o",
                    budget=4,
                    sink_size=1,
                    recent_window=1,
                    aggregate="ema",
                    ema_decay=bad,
                ),
                num_layers=2,
                attention_bus=bus,
                max_position_embeddings=64,
            )


def test_h2o_observe_validates_shape() -> None:
    cache, _bus = _make_cache(max_position_embeddings=16)
    # Wrong dim count.
    with pytest.raises(ValueError, match="4-D attn"):
        cache.observe(
            layer=0,
            attn=torch.zeros(2, 8),
            query_positions=torch.tensor([0], dtype=torch.long),
            key_len=8,
            phase="prefill",
        )
    # Mismatched key_len.
    with pytest.raises(ValueError, match="published key_len"):
        cache.observe(
            layer=0,
            attn=torch.zeros(1, 2, 1, 8),
            query_positions=torch.tensor([0], dtype=torch.long),
            key_len=10,
            phase="prefill",
        )
    # Out-of-range layer.
    with pytest.raises(ValueError, match="outside"):
        cache.observe(
            layer=99,
            attn=torch.zeros(1, 2, 1, 8),
            query_positions=torch.tensor([0], dtype=torch.long),
            key_len=8,
            phase="prefill",
        )


def test_h2o_observe_rejects_overlong_key() -> None:
    cache, _bus = _make_cache(max_position_embeddings=8)
    with pytest.raises(ValueError, match="exceeds"):
        cache.observe(
            layer=0,
            attn=torch.zeros(1, 2, 1, 16),
            query_positions=torch.tensor([0], dtype=torch.long),
            key_len=16,
            phase="prefill",
        )


# ---------------------------------------------------------------------------
# Decide-evict guards: never run with no observation.
# ---------------------------------------------------------------------------


def test_h2o_decide_evict_without_observe_falls_back() -> None:
    """Score buffer absent → streaming-style fallback (not raise).

    Prefill-first-evict case: first cache.update sees `key_len > budget` and
    the bus has not yet published. Falling back to sink + recent + uniform-
    middle preserves the H2O paper's keep-set shape and keeps the recording
    chain alive — raising here would kill the whole session via the
    `finally unsubscribe` chain in HFRecordingProvider.chat().
    """
    cache, _bus = _make_cache(
        budget=4, sink_size=1, recent_window=1, max_position_embeddings=16
    )
    decision = cache._decide_evict(layer_idx=0, key_len=10)
    assert decision.reason == "score_missing"
    # Sink (pos 0) + 2 uniformly-spaced middle (step = 8//2 = 4) +
    # recent (pos 9) → {0, 1, 5, 9}; evict the rest.
    assert decision.keep_indices == [0, 1, 5, 9]
    assert set(decision.evict_indices) == {2, 3, 4, 6, 7, 8}


def test_h2o_decide_evict_with_stale_middle_falls_back() -> None:
    """Score buffer not covering middle window → fallback (not raise).

    `score_lengths < middle_end` means observe() has not seen the heavy-
    hitter window yet (typically prefill mid-flight). Same fallback as the
    `buffer is None` case — paper-correct top-k can't be computed, so fall
    back to a streaming-style keep set marked with `reason="score_missing"`.
    """
    cache, _bus = _make_cache(
        budget=4, sink_size=1, recent_window=1, max_position_embeddings=64
    )
    # Observe at key_len=2 only — middle for key_len=10 is [1, 9), so we
    # need score length >= 9, but only see 2.
    cache.observe(
        layer=0,
        attn=torch.zeros(1, 2, 1, 2),
        query_positions=torch.tensor([0], dtype=torch.long),
        key_len=2,
        phase="prefill",
    )
    decision = cache._decide_evict(layer_idx=0, key_len=10)
    assert decision.reason == "score_missing"
    assert decision.keep_indices == [0, 1, 5, 9]
    assert set(decision.evict_indices) == {2, 3, 4, 6, 7, 8}


def test_h2o_update_defers_over_budget_prefill_until_observe() -> None:
    """The first over-budget prefill must be ranked by observed attention.

    `DynamicCache.update()` runs before the attention hook publishes scores, so
    H2O defers the prefill compaction. Once `observe()` sees the final prefill
    query row, the cache is compacted with a real top-k decision and the
    recorder row is written as `over_budget`, not `score_missing`.
    """
    recorder = KVEvictionRecorder(call_idx=0, policy_name="h2o")
    cache, _bus = _make_cache(
        budget=4,
        sink_size=1,
        recent_window=1,
        num_layers=1,
        max_position_embeddings=16,
        recorder=recorder,
        prefill_mode="full",
    )
    key_states = torch.zeros(1, 1, 10, 4)
    value_states = torch.zeros(1, 1, 10, 4)

    keys, _values = cache.update(key_states, value_states, layer_idx=0)
    assert int(keys.shape[-2]) == 10
    assert cache.get_seq_length(0) == 10
    assert cache._pending_prefill_evictions == {0: (-1, 10)}
    assert recorder.n_records() == 0

    cache.observe(
        layer=0,
        attn=_attn_with_peaks(key_len=10, peak_positions=[3, 7], n_query_rows=10),
        query_positions=torch.arange(10, dtype=torch.long),
        key_len=10,
        phase="prefill",
    )

    assert cache._pending_prefill_evictions == {}
    assert cache.get_seq_length(0) == 4
    assert recorder.n_records() == 1
    row = recorder._rows[0]
    assert row.phase == "prefill"
    assert row.evict_reason == "over_budget"
    assert row.pre_len == 10
    assert row.post_len == 4
    assert set(row.kept_indices) == {0, 3, 7, 9}
    assert row.score_topk_index is not None
    assert set(row.score_topk_index) == {3, 7}
    assert row.score_evicted_index == [1, 2, 4, 5, 6, 8]
    assert row.score_evicted_value is not None
    assert all(abs(v) < 1e-5 for v in row.score_evicted_value)


def test_h2o_full_prefill_waits_for_final_chunk_before_compacting() -> None:
    """Chunked full-prefill scoring must not compact after the first chunk."""
    cache, _bus = _make_cache(
        budget=4,
        sink_size=1,
        recent_window=1,
        num_layers=1,
        max_position_embeddings=16,
        prefill_mode="full",
    )
    cache.update(torch.zeros(1, 1, 10, 4), torch.zeros(1, 1, 10, 4), layer_idx=0)

    cache.observe(
        layer=0,
        attn=_attn_with_peaks(key_len=10, peak_positions=[3], n_query_rows=4),
        query_positions=torch.arange(4, dtype=torch.long),
        key_len=10,
        phase="prefill",
    )
    assert cache.get_seq_length(0) == 10
    assert cache._pending_prefill_evictions == {0: (-1, 10)}

    cache.observe(
        layer=0,
        attn=_attn_with_peaks(key_len=10, peak_positions=[7], n_query_rows=6),
        query_positions=torch.arange(4, 10, dtype=torch.long),
        key_len=10,
        phase="prefill",
    )
    assert cache.get_seq_length(0) == 4
    assert cache._pending_prefill_evictions == {}


# ---------------------------------------------------------------------------
# Step 8: aggregate=mean and aggregate=ema correctness.
# ---------------------------------------------------------------------------


def _make_cache_with_aggregate(
    aggregate: str,
    *,
    ema_decay: float = 0.9,
    budget: int = 4,
    sink_size: int = 1,
    recent_window: int = 1,
    max_position_embeddings: int = 32,
) -> tuple[H2OCache, AttentionBus]:
    bus = AttentionBus()
    cfg = EvictionPolicyConfig(
        name="h2o",
        budget=budget,
        sink_size=sink_size,
        recent_window=recent_window,
        aggregate=aggregate,
        ema_decay=ema_decay,
    )
    cache = H2OCache(
        cfg,
        num_layers=2,
        attention_bus=bus,
        max_position_embeddings=max_position_embeddings,
    )
    return cache, bus


def test_h2o_aggregate_mean_evicts_correctly() -> None:
    """mean normalises by observation count, so a position observed once at
    score=10 outranks a position observed twice at score=6 (mean 6.0).
    """
    cache, _bus = _make_cache_with_aggregate("mean", max_position_embeddings=16)
    key_len = 10

    # First observe: every middle slot sees one query row, peak at slot 7
    # (score 6) and slot 3 (score 6). Counts everywhere = 1.
    attn1 = _attn_with_peaks(
        key_len=key_len, peak_positions=[3, 7], peak_value=6.0, n_query_rows=1
    )
    cache.observe(
        layer=0,
        attn=attn1,
        query_positions=torch.tensor([0], dtype=torch.long),
        key_len=key_len,
        phase="prefill",
    )
    # Second observe: slot 3 only — sees 6 again. Now slot 3 has sum=12,
    # count=2 -> mean 6.0; slot 7 still sum=6, count=2 -> mean 3.0.
    # Slot 5 gets a fresh score 10 with count 2 -> mean 5.0.
    attn2 = _attn_with_peaks(
        key_len=key_len, peak_positions=[3, 5], peak_value=10.0, n_query_rows=1
    )
    # bring it to score 6 at slot 3, 10 at slot 5
    attn2[..., 3] = 6.0
    attn2[..., 5] = 10.0
    cache.observe(
        layer=0,
        attn=attn2,
        query_positions=torch.tensor([0], dtype=torch.long),
        key_len=key_len,
        phase="prefill",
    )

    # All middle slots seen twice (count = 2 across [0, key_len)).
    counts = cache._score_counts[0]
    assert counts is not None
    assert int(counts[3]) == 2
    assert int(counts[5]) == 2
    assert int(counts[7]) == 2

    # Decision: budget=4, sink=1, recent=1 -> n_heavy=2 over middle [1,9).
    # Means: slot 3 = (6+6)/2 = 6.0 ; slot 5 = (0+10)/2 = 5.0 ;
    # slot 7 = (6+0)/2 = 3.0 ; others = 0. Top-2 = {3, 5}.
    decision = cache._decide_evict(layer_idx=0, key_len=key_len)
    state = decision.policy_state
    assert state is not None
    assert set(state["score_topk_index"]) == {3, 5}, state
    # keep = sink {0} + recent {9} + heavy {3, 5}
    assert set(decision.keep_indices) == {0, 3, 5, 9}


def test_h2o_aggregate_ema_decays() -> None:
    """ema buffer applies decay on each observe; decision picks the most
    recently heavy positions, not the historically dominant ones."""
    cache, _bus = _make_cache_with_aggregate(
        "ema", ema_decay=0.5, max_position_embeddings=16
    )
    key_len = 10

    # Observe a tall historical peak at slot 3.
    attn1 = _attn_with_peaks(
        key_len=key_len, peak_positions=[3], peak_value=10.0, n_query_rows=1
    )
    cache.observe(
        layer=0,
        attn=attn1,
        query_positions=torch.tensor([0], dtype=torch.long),
        key_len=key_len,
        phase="prefill",
    )
    buffer_after_first = cache._scores[0].clone()
    assert float(buffer_after_first[3]) == pytest.approx(10.0, abs=1e-5)

    # Observe a fresh peak at slot 7. With decay=0.5, slot 3 should fall
    # to 5.0 and slot 7 should land at 10.0 (first-time write into a
    # previously-zero slot still applies the decay path because the layer
    # was already initialised; new addend wins).
    attn2 = _attn_with_peaks(
        key_len=key_len, peak_positions=[7], peak_value=10.0, n_query_rows=1
    )
    cache.observe(
        layer=0,
        attn=attn2,
        query_positions=torch.tensor([0], dtype=torch.long),
        key_len=key_len,
        phase="prefill",
    )
    buf = cache._scores[0]
    assert float(buf[3]) == pytest.approx(5.0, abs=1e-5)
    assert float(buf[7]) == pytest.approx(10.0, abs=1e-5)

    # Decision: budget=4, sink=1, recent=1 -> n_heavy=2; middle scores
    # rank slot 7 (10.0) > slot 3 (5.0) > all others.
    decision = cache._decide_evict(layer_idx=0, key_len=key_len)
    state = decision.policy_state
    assert state is not None
    assert state["score_topk_index"][0] == 7  # heaviest first
    assert set(state["score_topk_index"]) == {3, 7}


def test_h2o_aggregate_ema_first_observe_is_assignment() -> None:
    """First observe on a layer must skip the (1 - decay) scaling that a
    naive `mul_(decay).add_(per_key)` would apply to the freshly-zero buffer.
    Without the short-circuit the very first decision would be biased toward
    zero by exactly `1 - ema_decay`.
    """
    cache, _bus = _make_cache_with_aggregate(
        "ema", ema_decay=0.1, max_position_embeddings=8
    )
    attn = _attn_with_peaks(
        key_len=4, peak_positions=[2], peak_value=10.0, n_query_rows=1
    )
    cache.observe(
        layer=0,
        attn=attn,
        query_positions=torch.tensor([0], dtype=torch.long),
        key_len=4,
        phase="prefill",
    )
    # Should be the raw 10.0, NOT 0.1*0 + 10 = 10 (coincidence at zero start)
    # nor 10*0.1 = 1.0 — verify it's exactly the per-key value.
    assert float(cache._scores[0][2]) == pytest.approx(10.0, abs=1e-5)


# ---------------------------------------------------------------------------
# G16#5 echo: observe() must never call torch.softmax (consumer just reads).
# ---------------------------------------------------------------------------


def test_h2o_observe_does_not_resoftmax(monkeypatch) -> None:
    """H2O must consume the bus tensor as-is; any re-softmax breaks G16#5."""
    cache, _bus = _make_cache(max_position_embeddings=16)
    real = torch.softmax
    calls = {"n": 0}

    def counting_softmax(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(torch, "softmax", counting_softmax)

    cache.observe(
        layer=0,
        attn=torch.zeros(1, 2, 1, 8),
        query_positions=torch.tensor([0], dtype=torch.long),
        key_len=8,
        phase="decode",
    )
    assert calls["n"] == 0, (
        "H2OCache.observe must not call torch.softmax (G16#5); "
        f"saw {calls['n']} calls"
    )
