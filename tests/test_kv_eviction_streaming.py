"""Unit tests for `StreamingLLMCache` (plan G16#1 + decide-evict invariants).

Same shape as `test_kv_eviction_random.py`: exercise `_decide_evict` in
isolation; the full `update()` path that runs through transformers tensors
is covered by the smoke script + step-7 e2e suite.
"""

from __future__ import annotations

import pytest

from serving.kv_policies.base import EvictionPolicyConfig
from serving.kv_policies.streaming import StreamingLLMCache


def _make_cache(
    *,
    budget: int,
    sink_size: int,
    recent_window: int,
    num_layers: int = 4,
) -> StreamingLLMCache:
    config = EvictionPolicyConfig(
        name="streaming",
        budget=budget,
        sink_size=sink_size,
        recent_window=recent_window,
    )
    return StreamingLLMCache(config, num_layers=num_layers)


def test_streaming_keeps_sink_and_recent() -> None:
    """Plan G16#1: budget=8, sink=2, recent=4 over a sweep of key lengths.

    `key_len <= budget` -> no eviction. Once `key_len > budget`, we keep
    `[0, 1] ∪ [key_len-6 .. key_len-1]`, so post-eviction length stays at the
    fixed StreamingLLM capacity.
    """
    cache = _make_cache(budget=8, sink_size=2, recent_window=6)

    # Sweep key_len from 1 to 64 inclusive — pin every transition.
    for n in range(1, 65):
        decision = cache._decide_evict(layer_idx=0, key_len=n)
        if n <= 8:
            assert decision.evict_indices == [], f"n={n} unexpected eviction"
            assert decision.keep_indices == list(range(n)), f"n={n} keep mismatch"
            assert decision.reason == "none", f"n={n} reason"
        else:
            expected_keep = [0, 1] + list(range(n - 6, n))
            expected_evict = list(range(2, n - 6))
            assert decision.keep_indices == expected_keep, f"n={n} keep mismatch"
            assert decision.evict_indices == expected_evict, f"n={n} evict mismatch"
            assert decision.reason == "over_budget", f"n={n} reason"

    # Spot-check the headline values from the task brief.
    d12 = cache._decide_evict(layer_idx=0, key_len=12)
    assert d12.keep_indices == [0, 1, 6, 7, 8, 9, 10, 11]
    assert d12.evict_indices == [2, 3, 4, 5]

    d64 = cache._decide_evict(layer_idx=0, key_len=64)
    assert d64.keep_indices == [0, 1, 58, 59, 60, 61, 62, 63]


def test_streaming_config_validation() -> None:
    """sink + recent must equal budget; budget is the fixed cache capacity."""
    with pytest.raises(ValueError, match="budget == sink_size \\+ recent_window"):
        StreamingLLMCache(
            EvictionPolicyConfig(
                name="streaming",
                budget=8,
                sink_size=4,
                recent_window=8,  # 4 + 8 = 12 > 8
            ),
            num_layers=4,
        )
    with pytest.raises(ValueError, match="budget == sink_size \\+ recent_window"):
        StreamingLLMCache(
            EvictionPolicyConfig(
                name="streaming",
                budget=8,
                sink_size=2,
                recent_window=4,  # 2 + 4 = 6 < 8
            ),
            num_layers=4,
        )

    # Other defensive guards: positive budget, non-negative sink, positive recent.
    with pytest.raises(ValueError, match="positive config.budget"):
        StreamingLLMCache(
            EvictionPolicyConfig(name="streaming", budget=None, sink_size=2, recent_window=6),
            num_layers=4,
        )
    with pytest.raises(ValueError, match="recent_window > 0"):
        StreamingLLMCache(
            EvictionPolicyConfig(name="streaming", budget=8, sink_size=2, recent_window=0),
            num_layers=4,
        )


def test_streaming_layer_independence() -> None:
    """Same key_len -> identical decision on every layer.

    StreamingLLM is a pure function of (key_len, sink, recent). Pinning this
    behaviour explicitly so a future refactor doesn't accidentally introduce
    per-layer state (which would diverge from the paper's contract and break
    cross-policy comparisons against H2O in step 6).
    """
    cache = _make_cache(budget=8, sink_size=2, recent_window=6, num_layers=8)
    decisions = [
        cache._decide_evict(layer_idx=i, key_len=20) for i in range(8)
    ]
    first = decisions[0]
    for i, d in enumerate(decisions[1:], start=1):
        assert d.keep_indices == first.keep_indices, f"layer {i} keep diverged"
        assert d.evict_indices == first.evict_indices, f"layer {i} evict diverged"
        assert d.reason == first.reason, f"layer {i} reason diverged"
