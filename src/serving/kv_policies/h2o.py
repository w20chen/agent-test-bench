"""H2O (Heavy-Hitter Oracle) KV eviction policy.

Reference: Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative
Inference of Large Language Models" (NeurIPS 2023, https://arxiv.org/abs/2306.14048).

Algorithm sketch (paper §4)
---------------------------

For each layer, maintain a per-key-position cumulative attention score. When
`key_len > budget` triggers eviction, partition `[0, key_len)` into

    sink   := [0, sink_size)
    recent := [key_len - recent_window, key_len)
    middle := [sink_size, key_len - recent_window)

Always keep `sink ∪ recent` (matches StreamingLLM semantics for sink + local
context). From `middle`, keep the top `budget - sink_size - recent_window`
positions ranked by accumulated attention score; evict the rest. The
post-eviction layer length is exactly `budget`.

Design choices (locked here so reviewers can audit later H2O variants against
the same baseline contract).

A. **Score buffer shape — head-mean at `observe()` time, NOT per-head store.**
   The decision uses head-averaged scores (paper convention; cheap and
   well-behaved for GQA). We accumulate the head-mean directly in
   `observe()` instead of storing per-(layer, head, key_pos) and reducing at
   decision time. Algebraically equivalent because both sum and head-mean are
   linear; the per-head store would cost `num_kv_heads`x more memory (≈ 8x
   for Qwen3) for zero behavioural difference. The published `attn` from the
   bus carries the per-query-head dim already, so we mean over that axis.

B. **`config.aggregate` — `sum` (default), `mean`, `ema`.**
   * `sum` — paper default; cumulative score across observed queries, prefill
     + decode sharing one accumulator.
   * `mean` — `sum / observation_count` per key position. Maintains a parallel
     int counter per (layer, key_pos) incremented by `n_query_rows` on every
     observe(). Decision time divides the live prefix by the counter (clamped
     to 1 to avoid div-by-zero on never-observed slots; sink+recent are kept
     unconditionally so an unobserved slot is harmless). Algebraically this
     re-normalises the score so that older positions don't dominate purely
     because they have been exposed to more queries.
   * `ema` — exponential moving average with decay `ema_decay`. observe()
     applies `buffer = ema_decay * buffer + per_key` on the live prefix only,
     leaving the unallocated tail untouched. First observe on a layer
     short-circuits to a plain assignment so the EMA does not start from
     ambient zero (which would scale the first decision by `1 - ema_decay`).
     Score buffer compaction (`_post_evict_hook`) gathers the EMA prefix the
     same way as for sum, so the moving average survives eviction without a
     phase shift.

C. **Score buffer pre-allocation.**
   We pre-allocate a single `(max_position_embeddings,)` fp32 tensor per
   layer at first `observe()` (lazy on attn.device). Decode is hot — every
   step would otherwise grow the buffer; an upfront fp32 vector at 128k pos
   is ~512 KiB per layer (32 layers ≈ 16 MiB). If a downstream model exceeds
   `max_position_embeddings`, we raise — silent truncation would corrupt the
   eviction decisions invisibly.

D. **Device.**
   The score buffer lives on the same device as the published `attn` tensor.
   First `observe()` for a layer pins the device; subsequent calls assume it
   stays put. This avoids per-call `.to(...)` traffic on the decode path.

E. **Post-eviction score buffer compaction.**
   `_physically_drop` shrinks the K/V tensors by `keep_indices`; the score
   buffer must follow the same permutation or the next `observe()` will
   write into stale slots. We override `_post_evict_hook` (added to
   `BaseEvictionCache` for this purpose) and rebuild the prefix as
   `buffer[keep_indices]`. The base class fires the hook *after* the K/V
   drop succeeds.

Lifecycle invariants (plan §H4 + Top Risk #4)
---------------------------------------------

* `always_active = True` (class-level): even when `LayerCapturer.suspend_attention()`
  is gating the bus, H2O still observes. A silent skip would corrupt the
  score buffer relative to the cache state and the next eviction would pick
  stale heavy hitters.
* `requires_attention()` returns True so the provider knows it must wire the
  bus.
* The provider owns subscribe/unsubscribe lifecycle: H2O subscribes itself
  on construction, but the provider must call `attention_bus.unsubscribe(cache)`
  in a `finally` block around `model.generate(...)`. Letting subscriptions
  accumulate across calls would (a) leak references to dead caches and
  (b) cause double-observation if a cache from a prior call is still
  registered.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from serving.kv_policies.base import (
    BaseEvictionCache,
    EvictionDecision,
    EvictionPolicyConfig,
)

if TYPE_CHECKING:
    from serving.kv_policies.recorder import KVEvictionRecorder
    from serving.recording.attention_bus import AttentionBus


class H2OCache(BaseEvictionCache):
    """`BaseEvictionCache` with heavy-hitter (cumulative attention) eviction."""

    # Class-level: always observe even while LayerCapturer is suspended (plan
    # Top Risk #4). Per-instance overrides are not supported; H2O semantics
    # break if any decision is made on a partial accumulator.
    always_active: bool = True

    def __init__(
        self,
        config: EvictionPolicyConfig,
        num_layers: int,
        recorder: "KVEvictionRecorder | None" = None,
        *,
        attention_bus: "AttentionBus",
        max_position_embeddings: int,
    ) -> None:
        super().__init__(config, num_layers, recorder=recorder)
        if config.budget is None or config.budget <= 0:
            raise ValueError(
                f"H2OCache requires positive config.budget; got {config.budget!r}"
            )
        if config.sink_size < 0:
            raise ValueError(
                f"H2OCache requires sink_size >= 0; got {config.sink_size!r}"
            )
        if config.recent_window < 0:
            raise ValueError(
                f"H2OCache requires recent_window >= 0; got {config.recent_window!r}"
            )
        floor = int(config.sink_size) + int(config.recent_window)
        if int(config.budget) < floor:
            raise ValueError(
                f"H2OCache requires budget >= sink_size + recent_window "
                f"({config.budget!r} < {config.sink_size!r} + {config.recent_window!r} "
                f"= {floor})"
            )
        if config.aggregate not in {"sum", "mean", "ema"}:
            raise ValueError(
                f"aggregate={config.aggregate!r} unsupported; "
                "use one of {sum, mean, ema}"
            )
        if config.aggregate == "ema":
            if not (0.0 < float(config.ema_decay) < 1.0):
                raise ValueError(
                    f"aggregate='ema' requires 0 < ema_decay < 1; "
                    f"got {config.ema_decay!r}"
                )
        if config.prefill_mode not in {"sampled", "full"}:
            raise ValueError(
                f"prefill_mode={config.prefill_mode!r} unsupported; "
                "use one of {sampled, full}"
            )
        if attention_bus is None:
            raise ValueError("H2OCache requires an attention_bus to subscribe to")
        if max_position_embeddings is None or int(max_position_embeddings) <= 0:
            raise ValueError(
                "H2OCache requires positive max_position_embeddings; "
                f"got {max_position_embeddings!r}"
            )
        self._max_pos = int(max_position_embeddings)
        # Lazy per-layer score buffers: `_scores[layer]` is None until the
        # first observe() pins device + dtype.
        self._scores: list[torch.Tensor | None] = [None] * int(num_layers)
        self._score_lengths: list[int] = [0] * int(num_layers)
        # `aggregate="mean"` needs an observation count per (layer, key_pos)
        # so the decision-time divide is well-defined. Lazy alloc parallel to
        # `_scores` keeps the cost zero for sum/ema. We use int32 (n_query_rows
        # per observe is bounded by max_prefill_queries ≪ 2**31).
        self._aggregate = str(config.aggregate)
        self._ema_decay = float(config.ema_decay)
        self._score_counts: list[torch.Tensor | None] = [None] * int(num_layers)
        self.prefill_observe_mode = str(config.prefill_mode)
        self._pending_prefill_evictions: dict[int, tuple[int, int]] = {}
        # Subscribe AFTER all validation succeeds so a constructor failure
        # cannot leave a half-built cache wired to the bus.
        self._attention_bus: "AttentionBus" = attention_bus
        attention_bus.subscribe(self)

    def requires_attention(self) -> bool:
        return True

    # -- AttentionConsumer protocol -----------------------------------------

    def observe(
        self,
        *,
        layer: int,
        attn: torch.Tensor,
        query_positions: torch.Tensor,
        key_len: int,
        phase: str,
    ) -> None:
        """Accumulate head-mean attention into the per-layer score buffer.

        Shape contract from `AttentionBus`: `attn` is
        `(B, num_q_heads, n_query_rows, key_len)`. We sum across query rows
        and batch (each query contributes its full softmax row), then mean
        across heads to get a `(key_len,)` vector that we add into the
        running buffer at slots `[0:key_len]`.
        """
        if attn.ndim != 4:
            raise ValueError(
                f"H2OCache.observe expected 4-D attn (B,H,Q,K); got {tuple(attn.shape)}"
            )
        actual_key_len = int(attn.shape[-1])
        if actual_key_len != int(key_len):
            raise ValueError(
                "H2OCache.observe: attn key axis "
                f"({actual_key_len}) != published key_len ({key_len})"
            )
        if int(layer) < 0 or int(layer) >= self.num_layers:
            raise ValueError(
                f"H2OCache.observe: layer {layer} outside [0, {self.num_layers})"
            )
        if actual_key_len > self._max_pos:
            raise ValueError(
                f"H2OCache.observe: key_len {actual_key_len} exceeds "
                f"max_position_embeddings {self._max_pos}; bump the config "
                "or pick a smaller context"
            )

        # Sum across query-row axis (each query contributes its softmax row)
        # then sum across batch, then mean across heads. Order of reductions
        # is irrelevant (linear) but doing query-sum first matches the paper
        # cumulative-attention definition more naturally.
        per_head_per_key = attn.sum(dim=2).sum(dim=0)  # (H, K)
        per_key = per_head_per_key.mean(dim=0)  # (K,) — head-mean (choice A)
        # Number of query rows that contributed mass into this observation.
        # Each query row sums to ~1 across keys, so for `mean` we need the
        # divisor to scale with both the query axis and the (already summed)
        # batch axis.
        n_query_rows = int(attn.shape[0]) * int(attn.shape[2])

        buffer = self._scores[int(layer)]
        first_observe = buffer is None
        if buffer is None:
            # Choice C+D: pre-allocate fp32 buffer on the attn device on
            # first sight. fp32 keeps numerics stable across the cumulative
            # sum even when attn arrives in bf16/fp16.
            buffer = torch.zeros(self._max_pos, dtype=torch.float32, device=attn.device)
            self._scores[int(layer)] = buffer

        per_key_fp32 = per_key.to(dtype=buffer.dtype)
        if self._aggregate == "ema":
            if first_observe:
                # Avoid scaling the very first sample by (1 - ema_decay), which
                # would silently bias initial decisions toward zero.
                buffer[:actual_key_len] = per_key_fp32
            else:
                # EMA touches only the live prefix so the unallocated tail
                # never accumulates ambient zero updates.
                buffer[:actual_key_len].mul_(self._ema_decay).add_(per_key_fp32)
        else:
            # `sum` and `mean` both accumulate raw mass; `mean` divides at
            # decision time using the parallel count buffer below.
            buffer[:actual_key_len].add_(per_key_fp32)

        if self._aggregate == "mean":
            counts = self._score_counts[int(layer)]
            if counts is None:
                counts = torch.zeros(
                    self._max_pos, dtype=torch.int32, device=buffer.device
                )
                self._score_counts[int(layer)] = counts
            counts[:actual_key_len].add_(n_query_rows)

        if actual_key_len > self._score_lengths[int(layer)]:
            self._score_lengths[int(layer)] = actual_key_len
        if self._observed_last_prefill_query(
            query_positions=query_positions,
            key_len=actual_key_len,
            phase=phase,
        ):
            self._flush_pending_prefill_eviction(
                layer_idx=int(layer), key_len=actual_key_len
            )

    def _observed_last_prefill_query(
        self, *, query_positions: torch.Tensor, key_len: int, phase: str
    ) -> bool:
        if str(phase) != "prefill":
            return False
        if query_positions.numel() == 0:
            return False
        return int(query_positions.max().item()) == int(key_len) - 1

    def _defer_decision(
        self,
        *,
        layer_idx: int,
        phase: str,
        step: int,
        pre_len: int,
        decision: EvictionDecision,
    ) -> bool:
        """Defer first over-budget prefill until this layer's attention is observed.

        HF appends K/V before the attention hook publishes post-softmax rows.
        For the first over-budget prefill, evicting inside `update()` would use
        the `score_missing` fallback. H2O's paper contract requires ranking by
        observed attention, so we keep the full prefill K/V for the current
        layer forward and compact it in `observe()` immediately after scores
        are accumulated.
        """
        if phase == "prefill" and decision.reason == "score_missing":
            self._pending_prefill_evictions[int(layer_idx)] = (int(step), int(pre_len))
            return True
        return False

    def _flush_pending_prefill_eviction(self, *, layer_idx: int, key_len: int) -> None:
        pending = self._pending_prefill_evictions.pop(int(layer_idx), None)
        if pending is None:
            return
        step, pre_len = pending
        if int(pre_len) != int(key_len):
            raise RuntimeError(
                "H2OCache pending prefill eviction length mismatch: "
                f"pending pre_len={pre_len}, observed key_len={key_len}"
            )

        decision = self._decide_evict(layer_idx=int(layer_idx), key_len=int(key_len))
        if decision.reason == "score_missing":
            raise RuntimeError(
                "H2OCache deferred prefill eviction still has no score after observe(); "
                "the attention bus did not publish a usable prefill score."
            )
        keys, _values = self._get_layer_kv(int(layer_idx))
        if int(keys.shape[-2]) != int(key_len):
            raise RuntimeError(
                "H2OCache pending prefill eviction cache length mismatch: "
                f"cache_len={int(keys.shape[-2])}, observed key_len={key_len}"
            )
        if decision.evict_indices:
            keys, _values = self._physically_drop(int(layer_idx), decision.keep_indices)
            self._post_evict_hook(int(layer_idx), decision)
        post_len = int(keys.shape[-2])
        self._record_decision(
            layer_idx=int(layer_idx),
            phase="prefill",
            step=int(step),
            pre_len=int(pre_len),
            post_len=post_len,
            decision=decision,
        )

    # -- Eviction logic ------------------------------------------------------

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
        # Defensive: caller-side validation guarantees budget >= sink+recent,
        # so middle window is always non-empty when budget < key_len. But pin
        # the invariant explicitly so future config changes can't sneak past.
        if sink + recent > key_len:
            raise RuntimeError(
                f"H2OCache: sink({sink}) + recent({recent}) > key_len({key_len}); "
                "cannot partition without overlap"
            )
        n_heavy = budget - sink - recent
        middle_start = sink
        middle_end = key_len - recent
        middle_len = middle_end - middle_start
        if n_heavy < 0:
            raise RuntimeError(
                f"H2OCache: heavy_count budget({budget}) - sink({sink}) - recent({recent}) "
                f"< 0; constructor invariant breached"
            )

        buffer = self._scores[int(layer_idx)]
        # Order-of-operations note: in HF generate, the layer's
        # `past_key_values.update(...)` runs BEFORE softmax/capturer hook for
        # the same forward step. So at decision time the score buffer
        # reflects scores accumulated through the *previous* forward.
        #
        # Prefill case: the *first* forward pass has no "previous" forward;
        # if prefill_len > budget the very first cache.update triggers an
        # eviction call here with `buffer is None` (or score_lengths <
        # middle_end). Fail-fast in that branch is wrong — the H2O paper's
        # heavy-hitter ranking is undefined without observed attention, and
        # raising would silently kill the whole recording session via the
        # `finally unsubscribe` chain in HFRecordingProvider.chat().
        #
        # Fallback for direct/stale decision calls: use a streaming-style keep
        # set when score is missing — sink + recent + uniformly-spaced middle
        # positions filling the `n_heavy` slots. The normal `update()` path
        # defers over-budget prefill decisions until `observe()` has populated
        # scores, so paper-faithful H2O runs should not record this reason for
        # prefill.
        if buffer is None or self._score_lengths[int(layer_idx)] < middle_end:
            if n_heavy <= 0 or middle_len <= 0:
                keep_middle: list[int] = []
            else:
                # Evenly-spaced absolute positions in `[middle_start, middle_end)`.
                # Linear ramp guarantees `n_heavy_eff` distinct indices when
                # `middle_len >= n_heavy`; clamp to a unique-preserving cap.
                n_heavy_eff = min(n_heavy, middle_len)
                step = max(1, middle_len // n_heavy_eff)
                keep_middle = sorted(
                    {middle_start + i * step for i in range(n_heavy_eff)}
                    & set(range(middle_start, middle_end))
                )
            keep = list(range(0, sink)) + keep_middle + list(range(middle_end, key_len))
            evict = sorted(set(range(key_len)) - set(keep))
            return EvictionDecision(
                keep_indices=keep,
                evict_indices=evict,
                reason="score_missing",
            )

        # Pick top-`n_heavy` positions in the middle window by accumulated
        # score. `topk` returns descending; sort the absolute indices for a
        # stable keep set.
        middle_scores = buffer[middle_start:middle_end]
        if self._aggregate == "mean":
            counts = self._score_counts[int(layer_idx)]
            if counts is None:
                # mean was selected but observe() never ran for this layer;
                # _decide_evict's earlier `buffer is None` guard covers the
                # symmetric case for sum/ema, and the mean-specific count
                # missing here means observe() wired the buffer but skipped
                # the count branch — invariant breach, not a runtime fallback.
                raise RuntimeError(
                    f"H2OCache._decide_evict(mean): no count buffer for layer "
                    f"{layer_idx}; observe() failed to maintain the parallel "
                    "count tensor."
                )
            # Sink + recent are always kept regardless of score; the only
            # slots whose count actually matters are middle slots that have
            # been observed. Clamp to 1 so unobserved middle slots (count==0)
            # produce a finite (zero) mean rather than a NaN that would
            # propagate through topk.
            middle_counts = counts[middle_start:middle_end].to(dtype=middle_scores.dtype)
            middle_scores = middle_scores / middle_counts.clamp(min=1)
        # n_heavy may be 0 (budget == sink + recent); topk(0) is degenerate
        # but valid in torch.
        n_heavy_eff = min(n_heavy, middle_len)
        if n_heavy_eff > 0:
            top_values, top_local_idx = torch.topk(middle_scores, n_heavy_eff)
            heavy_indices_abs = (top_local_idx + middle_start).tolist()
            score_topk_value = [float(v) for v in top_values.detach().cpu().tolist()]
        else:
            heavy_indices_abs = []
            score_topk_value = []
        score_topk_index = list(heavy_indices_abs)

        keep_set = set(range(0, sink))
        keep_set.update(heavy_indices_abs)
        keep_set.update(range(middle_end, key_len))
        keep = sorted(keep_set)
        evict = sorted(set(range(key_len)) - keep_set)
        score_evicted_index = [
            int(idx) for idx in evict if middle_start <= int(idx) < middle_end
        ]
        if score_evicted_index:
            evicted_local_idx = torch.as_tensor(
                [idx - middle_start for idx in score_evicted_index],
                dtype=torch.long,
                device=middle_scores.device,
            )
            score_evicted_value = [
                float(v)
                for v in middle_scores.index_select(0, evicted_local_idx)
                .detach()
                .cpu()
                .tolist()
            ]
        else:
            score_evicted_value = []

        return EvictionDecision(
            keep_indices=keep,
            evict_indices=evict,
            reason="over_budget",
            policy_state={
                "score_topk_index": score_topk_index,
                "score_topk_value": score_topk_value,
                "score_evicted_index": score_evicted_index,
                "score_evicted_value": score_evicted_value,
            },
        )

    # -- Post-eviction state compaction (choice E) --------------------------

    def _post_evict_hook(self, layer_idx: int, decision: EvictionDecision) -> None:
        """Compact the score buffer (and mean-aggregate count) to mirror the K/V drop.

        Without this, the next `observe()` would `.add_()` into stale slots
        — old positions would remain in the buffer past the K/V boundary
        and the eviction decision after the next over-budget step would
        ghost-keep evicted positions.
        """
        buffer = self._scores[int(layer_idx)]
        if buffer is None:
            return
        keep = decision.keep_indices
        n_keep = len(keep)
        counts = self._score_counts[int(layer_idx)]
        if n_keep == 0:
            buffer.zero_()
            if counts is not None:
                counts.zero_()
            self._score_lengths[int(layer_idx)] = 0
            return
        index = torch.as_tensor(keep, dtype=torch.long, device=buffer.device)
        # Read the live prefix into a fresh permutation; `index_select` is
        # safe vs aliasing where in-place gather would not be. Then scatter
        # back into the head of the same buffer and zero the tail.
        compact = buffer.index_select(0, index)
        buffer.zero_()
        buffer[:n_keep] = compact
        if counts is not None:
            count_index = index.to(device=counts.device)
            compact_counts = counts.index_select(0, count_index)
            counts.zero_()
            counts[:n_keep] = compact_counts
        self._score_lengths[int(layer_idx)] = n_keep
