"""StreamingLLM (Xiao et al. 2023) KV eviction policy.

Algorithm: keep the first `sink_size` "attention sink" tokens plus the most
recent `recent_window` tokens; drop the middle. The sink positions absorb
otherwise-orphaned attention mass that the model assigns to position 0
regardless of semantic content (paper §3); the recent window preserves local
context for next-token prediction.

Why this is the right baseline implementation
---------------------------------------------

This is the **naive** StreamingLLM variant: kept K/V slots retain their
original RoPE rotation and HF's `cache_position` continues to advance with the
absolute generation index. We deliberately do **not** re-rotate cached keys
to "shift" them as if they were freshly placed at positions
`[0..sink+recent-1]`. The paper compares both flavours; without re-rotation
the model still functions because attention sinks dominate the softmax tails
and recent-window tokens still see the correct relative offsets among
themselves. Going the full re-rotation route requires either patching every
attention layer's `position_embeddings` kwarg via a pre-forward hook or
re-running RoPE on the cached K tensor at every eviction — both are
non-trivial engineering and orthogonal to the question this step is trying to
answer (does our `BaseEvictionCache` plumbing carry a non-random policy
through HF generate cleanly?). If the smoke shows the naive variant produces
incoherent text, step 4 escalates to the pre-forward-hook approach (plan
note E11) — but that's a follow-up, not a defect of this step.

Reference: https://arxiv.org/abs/2309.17453, GitHub repo
mit-han-lab/streaming-llm (see the `enable_streaming_llm` patch for the
shifted-RoPE variant).

Budget semantics
----------------

`config.budget` is the fixed StreamingLLM cache capacity:

* `key_len <= budget` → no eviction.
* `key_len > budget` → evict middle, keep
  `[0..sink_size-1] ∪ [key_len-recent_window..key_len-1]`.

`__init__` enforces `budget == sink_size + recent_window`. Letting `budget`
float above the sink+recent capacity turns StreamingLLM into a custom
"large trigger, small post-eviction window" policy and makes budget-labeled
comparisons misleading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from serving.kv_policies.base import (
    BaseEvictionCache,
    EvictionDecision,
    EvictionPolicyConfig,
)

if TYPE_CHECKING:
    from serving.kv_policies.recorder import KVEvictionRecorder


class StreamingLLMCache(BaseEvictionCache):
    """`BaseEvictionCache` with sink-prefix + recent-window eviction.

    Layer-independent: `_decide_evict` is a pure function of `key_len` and
    the policy config; it does **not** read `layer_idx`. Same `key_len` →
    same decision on every layer. This is intentional and contrasts with
    `RandomEvictCache`, where each layer draws a fresh sample from the same
    RNG, and `H2OCache` (step 6), where each layer's decision depends on its
    own attention scores.
    """

    def __init__(
        self,
        config: EvictionPolicyConfig,
        num_layers: int,
        recorder: "KVEvictionRecorder | None" = None,
    ) -> None:
        super().__init__(config, num_layers, recorder=recorder)
        if config.budget is None or config.budget <= 0:
            raise ValueError(
                f"StreamingLLMCache requires positive config.budget; got {config.budget!r}"
            )
        if config.sink_size < 0:
            raise ValueError(
                f"StreamingLLMCache requires sink_size >= 0; got {config.sink_size!r}"
            )
        if config.recent_window <= 0:
            raise ValueError(
                f"StreamingLLMCache requires recent_window > 0; got {config.recent_window!r}"
            )
        floor = int(config.sink_size) + int(config.recent_window)
        if int(config.budget) != floor:
            raise ValueError(
                f"StreamingLLMCache requires budget == sink_size + recent_window "
                f"({config.budget!r} != {config.sink_size!r} + {config.recent_window!r} "
                f"= {floor}); budget is the fixed cache capacity"
            )

    def _decide_evict(self, layer_idx: int, key_len: int) -> EvictionDecision:
        budget = int(self.config.budget)  # type: ignore[arg-type]
        if key_len <= budget:
            return EvictionDecision(
                keep_indices=list(range(key_len)),
                evict_indices=[],
                reason="none",
            )
        sink = int(self.config.sink_size)
        recent = int(self.config.recent_window)
        # Defensive: if the prompt itself were already shorter than sink+recent
        # we'd be in the no-evict branch above (budget >= sink+recent). Once
        # we reach this branch the slicing is well-formed.
        recent_start = key_len - recent
        keep = list(range(0, sink)) + list(range(recent_start, key_len))
        evict = list(range(sink, recent_start))
        return EvictionDecision(
            keep_indices=keep,
            evict_indices=evict,
            reason="over_budget",
        )
