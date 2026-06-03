"""Unit tests for `RandomEvictCache` (plan G16#3 + decide-evict invariants).

The cache's `_decide_evict` is exercised in isolation here; the full
`update()` path that touches transformers tensors is covered by the smoke
script + the step-7 e2e suite.
"""

from __future__ import annotations

import pytest

from serving.kv_policies.base import EvictionPolicyConfig
from serving.kv_policies.random_evict import RandomEvictCache


def _make_cache(*, budget: int, seed: int, num_layers: int = 4) -> RandomEvictCache:
    config = EvictionPolicyConfig(name="random", budget=budget, seed=seed)
    return RandomEvictCache(config, num_layers=num_layers)


def test_random_evict_seeded_determinism() -> None:
    """Same seed -> identical evict sets across instances (plan G16#3)."""
    cache_a = _make_cache(budget=16, seed=42)
    cache_b = _make_cache(budget=16, seed=42)
    cache_c = _make_cache(budget=16, seed=43)

    decisions_a, decisions_b, decisions_c = [], [], []
    for _ in range(5):
        # Single layer keeps the comparison apples-to-apples; per-layer
        # independence is exercised in test_random_evict_layer_independence.
        decisions_a.append(cache_a._decide_evict(layer_idx=0, key_len=64).evict_indices)
        decisions_b.append(cache_b._decide_evict(layer_idx=0, key_len=64).evict_indices)
        decisions_c.append(cache_c._decide_evict(layer_idx=0, key_len=64).evict_indices)

    assert decisions_a == decisions_b, "same seed produced different evict sets"
    # At least one of the 5 draws should differ between seed=42 and seed=43.
    assert decisions_a != decisions_c, "different seeds produced identical evict sets"


def test_random_evict_no_eviction_under_budget() -> None:
    cache = _make_cache(budget=16, seed=0)
    decision = cache._decide_evict(layer_idx=0, key_len=8)
    assert decision.evict_indices == []
    assert decision.keep_indices == list(range(8))
    assert decision.reason == "none"


def test_random_evict_over_budget_shape_and_invariants() -> None:
    cache = _make_cache(budget=16, seed=0)
    decision = cache._decide_evict(layer_idx=0, key_len=64)
    assert decision.reason == "over_budget"
    assert len(decision.keep_indices) == 16
    assert len(decision.evict_indices) == 64 - 16
    # Keep / evict partition [0, key_len).
    combined = sorted(decision.keep_indices + decision.evict_indices)
    assert combined == list(range(64))
    # Decision lists are sorted (eases downstream CSR storage / debugging).
    assert decision.keep_indices == sorted(decision.keep_indices)
    assert decision.evict_indices == sorted(decision.evict_indices)


def test_random_evict_layer_independence() -> None:
    """Different layers draw independent samples from the same RNG.

    (b) in random_evict.py: per-layer eviction is the policy contract; this
    test pins the behaviour so a future refactor doesn't accidentally couple
    layers.
    """
    cache = _make_cache(budget=16, seed=7, num_layers=4)
    decision_layer0 = cache._decide_evict(layer_idx=0, key_len=64)
    decision_layer1 = cache._decide_evict(layer_idx=1, key_len=64)
    # Same seeded RNG, sequential calls -> different draws.
    assert decision_layer0.evict_indices != decision_layer1.evict_indices


def test_random_evict_requires_positive_budget() -> None:
    with pytest.raises(ValueError, match="positive config.budget"):
        RandomEvictCache(
            EvictionPolicyConfig(name="random", budget=None),
            num_layers=4,
        )
    with pytest.raises(ValueError, match="positive config.budget"):
        RandomEvictCache(
            EvictionPolicyConfig(name="random", budget=0),
            num_layers=4,
        )
