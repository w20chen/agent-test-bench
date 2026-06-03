"""Scaffolding for KV cache eviction policies.

Subclass `BaseEvictionCache` to implement a specific policy. The base provides
the recording hook and physical-drop plumbing; subclasses only fill
`_decide_evict()` (see plan step 3+).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from transformers.cache_utils import DynamicCache

if TYPE_CHECKING:
    import torch

    from serving.kv_policies.recorder import KVEvictionRecorder


@dataclass
class EvictionPolicyConfig:
    """User-facing config for a KV eviction policy.

    Fields mirror the plan's Key Interface Sketches table. Subclass-specific
    fields are tolerated as unused for policies that don't need them
    (e.g. `seed` is only consulted by random).
    """

    name: Literal["none", "streaming", "h2o", "random"]
    budget: int | None  # required when name != "none"
    sink_size: int = 4  # streaming + h2o
    recent_window: int = 256  # streaming + h2o
    heavy_ratio: float = 0.5  # h2o
    aggregate: Literal["sum", "mean", "ema"] = "sum"  # h2o
    ema_decay: float = 0.9  # h2o (when aggregate="ema")
    seed: int = 0  # random
    record: bool = True  # F15: instrument toggle
    prefill_mode: Literal["sampled", "full"] = "full"  # h2o prefill fidelity


@dataclass
class EvictionDecision:
    """One layer's keep/evict decision for one update() call.

    Uses plain Python lists so this module stays torch-free at the type level;
    concrete subclasses may carry tensors in `policy_state` for diagnostics.
    """

    keep_indices: list[int]
    evict_indices: list[int]
    reason: str
    policy_state: dict[str, Any] | None = None


class BaseEvictionCache(DynamicCache):
    """`DynamicCache` subclass that enforces a budget via `_decide_evict()`.

    Lifecycle per `update()` call:
      1. delegate to `super().update()` to append the new K/V slots
      2. call `_decide_evict(layer_idx, post_append_len)`
      3. give subclasses one chance to defer the decision when their score is
         only available after the current attention forward (H2O prefill)
      4. if any evictions, physically drop those positions from this layer's
         K/V tensors so `get_seq_length()` returns the post-eviction length
      5. if a recorder is attached and `config.record` is True, record the
         decision via `KVEvictionRecorder.append()`

    Step 2-only: `_decide_evict` is abstract. Step 3+ adds policy subclasses.
    """

    def __init__(
        self,
        config: EvictionPolicyConfig,
        num_layers: int,
        recorder: "KVEvictionRecorder | None" = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_layers = int(num_layers)
        self.recorder = recorder
        # Per-layer decode-step counter; prefill calls reset to -1.
        # We use a plain dict because the recording layer needs absolute step
        # ids that match attention.npz's record_decode_step semantics.
        self._step_counter: dict[int, int] = {}
        self._seen_prefill: set[int] = set()

    def requires_attention(self) -> bool:
        """True if this policy needs to subscribe to AttentionBus (H2O)."""
        return False

    def notify_new_call(self, call_idx: int) -> None:
        """Mark a fresh chat() boundary.

        Resets per-call step counters so the next forward maps to `(prefill, -1)`
        and decode counts from 0 again. Physical KV slots, score buffers, and
        bus subscription persist across calls — only the recording labels reset.
        """
        del call_idx
        self._step_counter.clear()
        self._seen_prefill.clear()

    def update(
        self,
        key_states: "torch.Tensor",
        value_states: "torch.Tensor",
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        keys, values = super().update(key_states, value_states, layer_idx, cache_kwargs)
        pre_len = int(keys.shape[-2])
        query_len = int(key_states.shape[-2])
        phase, step = self._advance_step(layer_idx, query_len=query_len)

        decision = self._decide_evict(layer_idx, pre_len)
        if self._defer_decision(
            layer_idx=layer_idx,
            phase=phase,
            step=step,
            pre_len=pre_len,
            decision=decision,
        ):
            return keys, values
        if decision.evict_indices:
            keys, values = self._physically_drop(layer_idx, decision.keep_indices)
            # Subclasses with per-key-position state (H2O score buffer) need to
            # compact that state in lockstep with the K/V drop. Default no-op
            # for stateless policies (streaming, random).
            self._post_evict_hook(layer_idx, decision)
        post_len = int(keys.shape[-2])

        self._record_decision(
            layer_idx=layer_idx,
            phase=phase,
            step=step,
            pre_len=pre_len,
            post_len=post_len,
            decision=decision,
        )
        return keys, values

    def _defer_decision(
        self,
        *,
        layer_idx: int,
        phase: str,
        step: int,
        pre_len: int,
        decision: EvictionDecision,
    ) -> bool:
        """Return True when a subclass will finish and record this decision later."""
        del layer_idx, phase, step, pre_len, decision
        return False

    def _record_decision(
        self,
        *,
        layer_idx: int,
        phase: str,
        step: int,
        pre_len: int,
        post_len: int,
        decision: EvictionDecision,
    ) -> None:
        if self.recorder is None or not self.config.record:
            return
        # `policy_state` is the carry slot for policy-specific diagnostics the
        # recorder writes to npz; H2O fills `score_topk_index/value`, other
        # policies leave it None and the recorder fills sentinels.
        policy_state = decision.policy_state or {}
        self.recorder.append(
            step=step,
            layer=int(layer_idx),
            phase=phase,
            pre_len=pre_len,
            post_len=post_len,
            budget=int(self.config.budget) if self.config.budget is not None else -1,
            kept_indices=list(decision.keep_indices),
            evicted_indices=list(decision.evict_indices),
            evict_reason=decision.reason,
            score_topk_index=policy_state.get("score_topk_index"),
            score_topk_value=policy_state.get("score_topk_value"),
            score_evicted_index=policy_state.get("score_evicted_index"),
            score_evicted_value=policy_state.get("score_evicted_value"),
        )

    def _post_evict_hook(self, layer_idx: int, decision: EvictionDecision) -> None:
        """Hook fired after `_physically_drop` succeeds.

        Default: no-op. Subclasses with per-key-position state (H2O's score
        buffer) override to compact that state by `decision.keep_indices` so
        subsequent `observe()` calls write into the right slots.
        """
        del layer_idx, decision
        return

    def _decide_evict(self, layer_idx: int, key_len: int) -> EvictionDecision:
        """Return keep/evict decision for `layer_idx` given current `key_len`.

        Abstract — step 3+ subclasses implement actual policy logic.
        """
        raise NotImplementedError(
            "BaseEvictionCache._decide_evict is abstract; step 3+ subclasses "
            "(streaming/h2o/random) must implement."
        )

    def _physically_drop(
        self,
        layer_idx: int,
        keep_indices: list[int],
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Replace this layer's K/V tensors with the subset at `keep_indices`.

        Probes both `layers[idx].keys/values` (4.57+ struct) and the legacy
        `key_cache/value_cache` attrs since DynamicCache internals shifted
        between minor versions (see `hooks.py:_cached_key_states` for the
        same defensive pattern).
        """
        import torch

        keys, values = self._get_layer_kv(layer_idx)
        index = torch.as_tensor(keep_indices, dtype=torch.long, device=keys.device)
        new_keys = keys.index_select(-2, index)
        new_values = values.index_select(-2, index)
        self._set_layer_kv(layer_idx, new_keys, new_values)
        return new_keys, new_values

    def _get_layer_kv(
        self, layer_idx: int
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        layers = getattr(self, "layers", None)
        if layers is not None:
            layer = layers[layer_idx]
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is not None and values is not None:
                return keys, values
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def _set_layer_kv(
        self,
        layer_idx: int,
        keys: "torch.Tensor",
        values: "torch.Tensor",
    ) -> None:
        layers = getattr(self, "layers", None)
        if layers is not None:
            layer = layers[layer_idx]
            if hasattr(layer, "keys") and hasattr(layer, "values"):
                layer.keys = keys
                layer.values = values
                return
        self.key_cache[layer_idx] = keys
        self.value_cache[layer_idx] = values

    def _advance_step(
        self, layer_idx: int, *, query_len: int
    ) -> tuple[str, int]:
        """Decide (phase, step) for this update() call.

        Phase rule: a multi-token query, or the first call on a layer, is
        prefill. All subsequent single-token queries on the same layer are
        decode steps numbered from 0. Matches LayerCapturer's decode_step
        semantics for cross-artifact alignment.
        """
        if query_len > 1 or layer_idx not in self._seen_prefill:
            self._seen_prefill.add(layer_idx)
            self._step_counter[layer_idx] = 0
            return "prefill", -1
        step = self._step_counter[layer_idx]
        self._step_counter[layer_idx] = step + 1
        return "decode", step
