"""KV cache eviction policies for the HF recording path.

Step 3-6 wire `random` + `streaming` + `h2o`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from serving.kv_policies.base import (
    BaseEvictionCache,
    EvictionDecision,
    EvictionPolicyConfig,
)
from serving.kv_policies.recorder import KVEvictionRecorder

if TYPE_CHECKING:
    from serving.recording.attention_bus import AttentionBus


def build_eviction_cache(
    config: EvictionPolicyConfig,
    num_layers: int,
    recorder: KVEvictionRecorder | None = None,
    *,
    attention_bus: "AttentionBus | None" = None,
    max_position_embeddings: int | None = None,
) -> BaseEvictionCache:
    """Factory for the configured eviction cache.

    Callers must gate on `config is not None` upstream — the `"none"` policy
    is a CLI-layer concept (no cache subclass needed) so reaching this factory
    with `name="none"` indicates a wiring bug, not a runtime fallback.

    `attention_bus` and `max_position_embeddings` are required for `h2o`
    (it subscribes to the bus at construction and pre-allocates a per-layer
    score buffer sized by `max_position_embeddings`); other policies ignore
    them.
    """
    name = config.name
    if name == "random":
        # Local import keeps `transformers.cache_utils` import out of module
        # load when no policy is in use.
        from serving.kv_policies.random_evict import RandomEvictCache

        return RandomEvictCache(config, num_layers, recorder=recorder)
    if name == "streaming":
        from serving.kv_policies.streaming import StreamingLLMCache

        return StreamingLLMCache(config, num_layers, recorder=recorder)
    if name == "h2o":
        if attention_bus is None:
            raise ValueError(
                "build_eviction_cache(name='h2o') requires attention_bus; "
                "H2O subscribes at construction to share the post-softmax "
                "tensor with LayerCapturer."
            )
        if max_position_embeddings is None:
            raise ValueError(
                "build_eviction_cache(name='h2o') requires max_position_embeddings "
                "(used to pre-allocate the per-layer score buffer)."
            )
        from serving.kv_policies.h2o import H2OCache

        return H2OCache(
            config,
            num_layers,
            recorder=recorder,
            attention_bus=attention_bus,
            max_position_embeddings=int(max_position_embeddings),
        )
    if name == "none":
        raise ValueError(
            "build_eviction_cache should not be called when policy is disabled "
            "(config.name == 'none'); gate on `eviction_config is not None`"
        )
    raise NotImplementedError(f"policy {name!r} not registered")


__all__ = [
    "BaseEvictionCache",
    "EvictionDecision",
    "EvictionPolicyConfig",
    "KVEvictionRecorder",
    "build_eviction_cache",
]
