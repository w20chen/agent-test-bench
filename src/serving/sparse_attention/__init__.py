"""Sparse attention methods for the HF recording path.

First-release method: `sliding` (sink prefix + recent tail). Mirrors
`serving.kv_policies` layout one-for-one but addresses an orthogonal axis
of the design space — sparse attention masks key positions per query
instead of physically dropping K/V slots.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from serving.sparse_attention.base import (
    BaseSparseAttention,
    SparseAttentionConfig,
)
from serving.sparse_attention.recorder import SparseAttentionRecorder

if TYPE_CHECKING:
    from serving.recording.attention_bus import AttentionBus


def build_sparse_attention(
    config: SparseAttentionConfig,
    num_layers: int,
    recorder: SparseAttentionRecorder | None = None,
    *,
    attention_bus: "AttentionBus | None" = None,
) -> BaseSparseAttention:
    """Factory for the configured sparse attention method.

    Callers must gate on `config is not None` upstream — the `"none"`
    method is a CLI-layer concept (no method instance needed). The
    `recorder` is the per-call audit sink; the provider swaps it before
    each `recording_session()` and the pre-hook reads it via closure.
    `heavy_hitter` subscribes to `attention_bus` so it can use previously
    observed attention scores. Other methods ignore the bus.
    """
    del recorder  # recorder is swapped on LayerCapturer, not owned by methods
    name = config.name
    if name in {"sliding", "streaming"}:
        from serving.sparse_attention.sliding import SlidingWindowSparseAttention

        method = SlidingWindowSparseAttention.from_config(config)
        return method
    if name == "block_topk":
        from serving.sparse_attention.block_topk import BlockTopKSparseAttention

        return BlockTopKSparseAttention.from_config(config)
    if name == "quest":
        from serving.sparse_attention.quest import QuestSparseAttention

        return QuestSparseAttention.from_config(config)
    if name == "heavy_hitter":
        if attention_bus is None:
            raise ValueError(
                "build_sparse_attention(name='heavy_hitter') requires attention_bus"
            )
        from serving.sparse_attention.heavy_hitter import HeavyHitterSparseAttention

        return HeavyHitterSparseAttention.from_config(
            config,
            num_layers=num_layers,
            attention_bus=attention_bus,
        )
    if name == "none":
        raise ValueError(
            "build_sparse_attention should not be called when method is disabled "
            "(config.name == 'none'); gate on `sparse_attention_config is not None`"
        )
    raise NotImplementedError(f"sparse attention method {name!r} not registered")


__all__ = [
    "BaseSparseAttention",
    "SparseAttentionConfig",
    "SparseAttentionRecorder",
    "build_sparse_attention",
]
