"""Historical heavy-hitter sparse attention mask."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from serving.sparse_attention.base import SparseAttentionConfig, SparseAttentionContext
from serving.sparse_attention.patterns import (
    additive_mask_from_keep,
    cap_middle_selection,
    dense_or_causal_mask,
    middle_indices,
    sink_recent_keep_indices,
    validate_sparse_budget,
)

if TYPE_CHECKING:
    from serving.recording.attention_bus import AttentionBus


class HeavyHitterSparseAttention:
    """H2O-style historical attention scoring applied as an attention mask."""

    name = "heavy_hitter"
    always_active = True
    prefill_observe_mode = "full"
    # Correctness depends on every prefill token's attention reaching the
    # AttentionBus so `_scores` accumulates per-key history. With session KV
    # delta-prefill the cached prefix is skipped → bus never sees those rows →
    # `_rank_middle_positions` returns [] → decode silently degrades to
    # streaming-LLM. The HF backend reads this flag and refuses to build a
    # session cache when it is True (see `_build_session_cache`).
    requires_full_prefill = True

    def __init__(
        self,
        *,
        budget: int,
        sink_size: int,
        recent_window: int,
        block_size: int,
        score_reduction: str = "max",
        phase_scope: str = "decode_only",
        observe_only: bool = False,
        num_layers: int,
        attention_bus: "AttentionBus",
    ) -> None:
        self.budget = validate_sparse_budget(
            method_name=self.name,
            budget=budget,
            sink_size=sink_size,
            recent_window=recent_window,
            block_size=block_size,
            score_reduction=score_reduction,
            phase_scope=phase_scope,
        )
        self.sink_size = int(sink_size)
        self.recent_window = int(recent_window)
        self.block_size = int(block_size)
        self.score_reduction = str(score_reduction)
        self.phase_scope = str(phase_scope)
        self.observe_only = bool(observe_only)
        self._scores: list[torch.Tensor | None] = [None] * int(num_layers)
        self._score_lengths: list[int] = [0] * int(num_layers)
        self._last_kept_count = 0
        self._last_metadata: dict[str, Any] = {}
        attention_bus.subscribe(self)

    def reset_state(self) -> None:
        """Clear historical scores at sparse-call boundaries.

        Heavy-hitter sets `requires_full_prefill = True`, which forces the HF
        backend to skip session KV cache delta-prefill — every chat() call
        re-prefills the full prompt and the AttentionBus sees every key
        position. Carrying scores across calls would therefore double-count
        replayed context, so we reset at each call boundary.
        """
        for idx in range(len(self._scores)):
            self._scores[idx] = None
            self._score_lengths[idx] = 0
        self._last_kept_count = 0
        self._last_metadata = {}

    @classmethod
    def from_config(
        cls,
        config: SparseAttentionConfig,
        *,
        num_layers: int,
        attention_bus: "AttentionBus",
    ) -> "HeavyHitterSparseAttention":
        if config.budget is None:
            raise ValueError(
                f"{cls.__name__} requires SparseAttentionConfig.budget to be set "
                f"(method={config.name!r})"
            )
        return cls(
            budget=int(config.budget),
            sink_size=config.sink_size,
            recent_window=config.recent_window,
            block_size=config.block_size,
            score_reduction=config.score_reduction,
            phase_scope=config.phase_scope,
            observe_only=config.observe_only,
            num_layers=num_layers,
            attention_bus=attention_bus,
        )

    def observe(
        self,
        *,
        layer: int,
        attn: torch.Tensor,
        query_positions: torch.Tensor,
        key_len: int,
        phase: str,
    ) -> None:
        del query_positions, phase
        if attn.ndim != 4:
            raise ValueError(f"heavy_hitter expected 4-D attn; got {tuple(attn.shape)}")
        if int(attn.shape[-1]) != int(key_len):
            raise ValueError(
                f"heavy_hitter attn key axis {int(attn.shape[-1])} != key_len {key_len}"
            )
        layer_idx = int(layer)
        if layer_idx < 0 or layer_idx >= len(self._scores):
            raise ValueError(f"heavy_hitter layer {layer_idx} outside configured range")
        per_head_per_key = attn.sum(dim=2).sum(dim=0)
        per_key = per_head_per_key.mean(dim=0).to(dtype=torch.float32)
        buffer = self._scores[layer_idx]
        if buffer is None or int(buffer.shape[0]) < int(key_len):
            new_len = int(key_len)
            new_buffer = torch.zeros(new_len, dtype=torch.float32, device=attn.device)
            if buffer is not None:
                old = min(int(buffer.shape[0]), new_len)
                new_buffer[:old] = buffer[:old].to(device=attn.device)
            buffer = new_buffer
            self._scores[layer_idx] = buffer
        buffer[: int(key_len)].add_(per_key)
        self._score_lengths[layer_idx] = max(self._score_lengths[layer_idx], int(key_len))

    def build_additive_mask(
        self,
        *,
        layer_idx: int,
        query_len: int,
        key_len: int,
        phase: str,
        decode_step: int = -1,
        device: "Any",
        dtype: "Any",
        context: SparseAttentionContext | None = None,
    ) -> "Any":
        del decode_step, context
        if key_len <= 0:
            self._record(kept=[], reason="empty")
            if self.observe_only:
                return None
            return dense_or_causal_mask(
                query_len=query_len, key_len=key_len, device=device, dtype=dtype
            )
        if self.phase_scope == "decode_only" and phase != "decode":
            self._record(kept=range(key_len), reason="phase_dense")
            if self.observe_only:
                return None
            return dense_or_causal_mask(
                query_len=query_len, key_len=key_len, device=device, dtype=dtype
            )
        if query_len != 1:
            self._record(kept=range(key_len), reason="prefill_dense")
            if self.observe_only:
                return None
            return dense_or_causal_mask(
                query_len=query_len, key_len=key_len, device=device, dtype=dtype
            )

        ranked_middle = self._rank_middle_positions(layer_idx=int(layer_idx), key_len=key_len)
        reason = "selected" if ranked_middle else "sink_recent_no_scores"
        selected_middle = cap_middle_selection(
            key_len=key_len,
            budget=self.budget,
            sink_size=self.sink_size,
            recent_window=self.recent_window,
            ranked_middle=ranked_middle,
        )
        keep = sink_recent_keep_indices(
            key_len=key_len,
            sink_size=self.sink_size,
            recent_window=self.recent_window,
        )
        keep.update(selected_middle)
        self._record(kept=keep, reason=reason, selected_middle=selected_middle)
        if self.observe_only:
            return None
        return additive_mask_from_keep(
            keep_indices=keep,
            query_len=query_len,
            key_len=key_len,
            device=device,
            dtype=dtype,
        )

    def _rank_middle_positions(self, *, layer_idx: int, key_len: int) -> list[int]:
        buffer = self._scores[int(layer_idx)]
        candidates = middle_indices(
            key_len=key_len, sink_size=self.sink_size, recent_window=self.recent_window
        )
        if buffer is None or not candidates:
            return []
        if self._score_lengths[int(layer_idx)] < max(candidates) + 1:
            return []
        idx = torch.as_tensor(candidates, dtype=torch.long, device=buffer.device)
        scores = buffer.index_select(0, idx)
        order = torch.argsort(scores, descending=True).detach().cpu().tolist()
        return [candidates[int(i)] for i in order]

    def _record(
        self,
        *,
        kept: Any,
        reason: str,
        selected_middle: list[int] | None = None,
    ) -> None:
        kept_list = [int(x) for x in kept]
        self._last_kept_count = len(set(kept_list))
        self._last_metadata = {
            "budget": self.budget,
            "phase_scope": self.phase_scope,
            "selection_reason": reason,
            "selected_middle_count": len(selected_middle or []),
            "selected_middle_indices": [int(x) for x in (selected_middle or [])],
        }

    def kept_count(self, key_len: int) -> int:
        del key_len
        return int(self._last_kept_count)

    def record_metadata(
        self,
        *,
        layer_idx: int,
        phase: str,
        decode_step: int,
    ) -> dict[str, Any]:
        del layer_idx, phase, decode_step
        return dict(self._last_metadata)


__all__ = ["HeavyHitterSparseAttention"]
