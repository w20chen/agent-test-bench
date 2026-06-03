"""Quest-style query-aware page sparse attention."""

from __future__ import annotations

from typing import Any

from serving.sparse_attention.base import SparseAttentionConfig, SparseAttentionContext
from serving.sparse_attention.patterns import (
    additive_mask_from_keep,
    block_id_for_position,
    cap_middle_selection,
    dense_or_causal_mask,
    middle_indices,
    sink_recent_keep_indices,
    validate_sparse_budget,
)
from serving.sparse_attention.state import (
    current_query_states,
    full_key_states_for_pre_hook,
)


class QuestSparseAttention:
    """Decode-time Quest-style page selector using key min/max envelopes."""

    name = "quest"
    observe_only: bool
    # Page envelopes (min/max) are computed at decode from the full cached + delta
    # K state, so cache reuse is safe — same K seen whether we re-prefilled or
    # resumed.
    requires_full_prefill = False

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
        self._last_kept_count = 0
        self._last_metadata: dict[str, Any] = {}

    @classmethod
    def from_config(cls, config: SparseAttentionConfig) -> "QuestSparseAttention":
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
        )

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
        del decode_step
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
        if context is None or context.position_embeddings is None:
            raise ValueError(
                "quest requires position_embeddings in decode; "
                "cannot silently fall back to dense attention"
            )

        ranked_middle, selected_pages = self._rank_middle_positions(
            context=context,
            layer_idx=layer_idx,
            key_len=key_len,
        )
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
        self._record(
            kept=keep,
            reason="selected",
            selected_pages=selected_pages,
            selected_middle=selected_middle,
        )
        if self.observe_only:
            return None
        return additive_mask_from_keep(
            keep_indices=keep,
            query_len=query_len,
            key_len=key_len,
            device=device,
            dtype=dtype,
        )

    def _rank_middle_positions(
        self, *, context: SparseAttentionContext, layer_idx: int, key_len: int
    ) -> tuple[list[int], list[int]]:
        import torch

        q_states = current_query_states(
            module=context.module,
            hidden_states=context.hidden_states,
            position_embeddings=context.position_embeddings,  # type: ignore[arg-type]
        )
        key_states = full_key_states_for_pre_hook(
            module=context.module,
            layer_idx=int(layer_idx),
            hidden_states=context.hidden_states,
            position_embeddings=context.position_embeddings,  # type: ignore[arg-type]
            past_key_values=context.past_key_values,
        )
        if int(key_states.shape[-2]) != int(key_len):
            raise ValueError(
                f"quest key_len mismatch: computed {int(key_states.shape[-2])}, "
                f"hook reported {key_len}"
            )
        candidates = middle_indices(
            key_len=key_len, sink_size=self.sink_size, recent_window=self.recent_window
        )
        if not candidates:
            return [], []

        page_to_positions: dict[int, list[int]] = {}
        for pos in candidates:
            page_to_positions.setdefault(
                block_id_for_position(pos, self.block_size), []
            ).append(pos)

        # Validate head-shape contract once before vectorization.
        n_kv_heads = int(key_states.shape[1])
        n_query_heads = int(q_states.shape[1])
        if n_query_heads % n_kv_heads != 0:
            raise ValueError(
                f"query heads must be divisible by kv heads: {n_query_heads} vs {n_kv_heads}"
            )

        # Vectorize: build per-candidate keys, scatter_reduce by page_id over the
        # position axis to get per-page min/max key envelopes; then einsum upper-bound
        # scores across all pages in one shot.
        candidates_t = torch.as_tensor(
            candidates, dtype=torch.long, device=key_states.device
        )
        page_ids_t = candidates_t // int(self.block_size)                  # [Nc]
        unique_pages, inverse = torch.unique(
            page_ids_t, return_inverse=True
        )                                                                  # [Np], [Nc]
        np_pages = int(unique_pages.shape[0])
        cand_keys = key_states.index_select(-2, candidates_t)  # [B, H_kv, Nc, D]
        B, _, _, head_dim = cand_keys.shape
        inv_exp = inverse.view(1, 1, -1, 1).expand(B, n_kv_heads, -1, head_dim)

        page_min = torch.full(
            (B, n_kv_heads, np_pages, head_dim), float("inf"),
            device=key_states.device, dtype=cand_keys.dtype,
        )
        page_min = page_min.scatter_reduce(
            2, inv_exp, cand_keys, reduce="amin", include_self=True
        )
        page_max = torch.full(
            (B, n_kv_heads, np_pages, head_dim), float("-inf"),
            device=key_states.device, dtype=cand_keys.dtype,
        )
        page_max = page_max.scatter_reduce(
            2, inv_exp, cand_keys, reduce="amax", include_self=True
        )

        q_for_kv = q_states.reshape(
            q_states.shape[0],
            n_kv_heads,
            n_query_heads // n_kv_heads,
            q_states.shape[-2],
            q_states.shape[-1],
        )  # [B, H_kv, G, T, D]
        positive = torch.clamp(q_for_kv, min=0)
        negative = torch.clamp(q_for_kv, max=0)
        # upper[b, p, h, g, t] = sum_d positive[b,h,g,t,d] * page_max[b,h,p,d]
        #                      + sum_d negative[b,h,g,t,d] * page_min[b,h,p,d]
        upper = torch.einsum("bhgtd,bhpd->bphgt", positive, page_max)
        upper = upper + torch.einsum("bhgtd,bhpd->bphgt", negative, page_min)
        # Reduce per page: collapse all other dims into a single scalar per page.
        if self.score_reduction == "mean":
            page_scores_t = upper.mean(dim=(0, 2, 3, 4))   # [Np]
        else:
            page_scores_t = upper.amax(dim=(0, 2, 3, 4))   # [Np]

        scores_cpu = page_scores_t.detach().cpu().tolist()
        pages_cpu = unique_pages.detach().cpu().tolist()
        ordered = sorted(zip(scores_cpu, pages_cpu), key=lambda item: (-item[0], item[1]))
        ranked_positions: list[int] = []
        selected_pages: list[int] = []
        for _score, page_id in ordered:
            selected_pages.append(int(page_id))
            ranked_positions.extend(page_to_positions[int(page_id)])
        return ranked_positions, selected_pages

    def _record(
        self,
        *,
        kept: Any,
        reason: str,
        selected_pages: list[int] | None = None,
        selected_middle: list[int] | None = None,
    ) -> None:
        kept_list = [int(x) for x in kept]
        self._last_kept_count = len(set(kept_list))
        self._last_metadata = {
            "budget": self.budget,
            "block_size": self.block_size,
            "score_reduction": self.score_reduction,
            "phase_scope": self.phase_scope,
            "selection_reason": reason,
            "selected_pages": list(selected_pages or []),
            "selected_middle_count": len(selected_middle or []),
            "selected_middle_indices": [int(x) for x in (selected_middle or [])],
        }

    def reset_state(self) -> None:
        """No-op: quest computes its keep set fresh each decode step."""

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


__all__ = ["QuestSparseAttention"]
