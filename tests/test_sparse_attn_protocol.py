"""Protocol satisfaction tests for BaseSparseAttention."""

from __future__ import annotations

import pytest

from serving.sparse_attention.base import BaseSparseAttention, SparseAttentionConfig
from serving.sparse_attention import build_sparse_attention
from serving.recording.attention_bus import AttentionBus


# streaming is a CLI alias for sliding; the class name attribute stays "sliding"
_NAME_ALIASES = {"streaming": "sliding"}


@pytest.mark.parametrize("name,kwargs", [
    ("sliding", {"sink_size": 4, "recent_window": 8}),
    ("streaming", {"sink_size": 4, "recent_window": 8}),
    ("heavy_hitter", {"sink_size": 4, "recent_window": 8, "budget": 32}),
    ("block_topk", {"sink_size": 4, "recent_window": 8, "budget": 32}),
    ("quest", {"sink_size": 4, "recent_window": 8, "budget": 32}),
])
def test_method_satisfies_base_protocol(name: str, kwargs: dict) -> None:
    """Each registered sparse method must satisfy BaseSparseAttention."""
    cfg = SparseAttentionConfig(name=name, **kwargs)
    bus = AttentionBus() if name == "heavy_hitter" else None
    method = build_sparse_attention(cfg, num_layers=4, attention_bus=bus)
    assert isinstance(method, BaseSparseAttention), (
        f"{type(method).__name__} fails Protocol satisfaction; "
        "likely missing `name` or `observe_only` attribute."
    )
    expected_name = _NAME_ALIASES.get(name, name)
    assert isinstance(method.name, str) and method.name == expected_name
    assert isinstance(method.observe_only, bool)
