"""Sliding-window sparse attention (sink prefix + recent tail).

Per-row keep set (query row q at absolute position `cached_len + q =
key_len - query_len + q`):
  k in [0, sink_size)                       -> 0    (attend)
  k in [key_len - recent_window, key_len)   -> 0    (attend)
  otherwise                                  -> -inf (mask)
  AND additionally: k > (key_len - query_len) + q -> -inf (causal)

At prefill (query_len > 1) the additive mask we return is `[1,1,Q,K]` with
the upper triangle (k > absolute query position) forced to -inf. This is
load-bearing: the recording pre-hook sets HF SDPA's `attention_mask`
kwarg to a non-None tensor, which disables SDPA's implicit `is_causal`
shortcut. Without the causal upper triangle here, query row 0 would see
the recent-window tail (future tokens) — a hindsight leak.

At decode (query_len == 1) the single query row sits at position
`key_len - 1`, which is >= every k in `[0, key_len)`. Causality is
trivially satisfied, so we keep the cheaper `[1,1,1,K]` key-uniform
mask shape and let SDPA broadcast.

When `sink_size + recent_window >= key_len`, no sparsity positions are
masked at decode; causal still applies at prefill via the per-row
upper-triangular cut.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from serving.sparse_attention.base import SparseAttentionConfig, SparseAttentionContext

if TYPE_CHECKING:
    import torch


class SlidingWindowSparseAttention:
    """Sink-prefix + recent-tail sparse attention."""

    name = "sliding"
    # Stateless: keep set depends only on (sink_size, recent_window, key_len),
    # so session KV cache reuse is safe — no per-step accumulator to corrupt.
    requires_full_prefill = False

    def __init__(self, sink_size: int, recent_window: int, observe_only: bool = False) -> None:
        if sink_size < 0:
            raise ValueError(
                f"SlidingWindowSparseAttention requires sink_size >= 0; "
                f"got {sink_size!r}"
            )
        if recent_window < 0:
            raise ValueError(
                f"SlidingWindowSparseAttention requires recent_window >= 0; "
                f"got {recent_window!r}"
            )
        if sink_size + recent_window <= 0:
            raise ValueError(
                "SlidingWindowSparseAttention requires sink_size + recent_window > 0; "
                f"got sink_size={sink_size!r}, recent_window={recent_window!r}"
            )
        self.sink_size = int(sink_size)
        self.recent_window = int(recent_window)
        self.observe_only = bool(observe_only)

    @classmethod
    def from_config(cls, config: SparseAttentionConfig) -> "SlidingWindowSparseAttention":
        return cls(sink_size=config.sink_size, recent_window=config.recent_window, observe_only=config.observe_only)

    def build_additive_mask(
        self,
        *,
        layer_idx: int,
        query_len: int,
        key_len: int,
        phase: str,
        decode_step: int = -1,
        device: "torch.device",
        dtype: "torch.dtype",
        context: SparseAttentionContext | None = None,
    ) -> "torch.Tensor":
        del layer_idx, phase, decode_step, context  # method is head/layer-uniform
        import torch

        if query_len < 1:
            raise ValueError(f"query_len must be >= 1; got {query_len!r}")
        if key_len < 0:
            raise ValueError(f"key_len must be >= 0; got {key_len!r}")

        if self.observe_only:
            return None

        # Build a 1D key-mask first (same sparsity set for every query row).
        keep = torch.zeros(key_len, dtype=torch.bool, device=device)
        sink = min(self.sink_size, key_len)
        if sink > 0:
            keep[:sink] = True
        recent_start = max(0, key_len - self.recent_window)
        if self.recent_window > 0:
            keep[recent_start:] = True

        neg_inf = torch.finfo(dtype).min
        key_mask = torch.where(
            keep,
            torch.zeros((), dtype=dtype, device=device),
            torch.full((), neg_inf, dtype=dtype, device=device),
        )

        if query_len == 1:
            # Decode: the single query is at absolute position key_len-1,
            # which is >= every k in [0, key_len), so causal is trivially
            # satisfied. Keep the cheaper key-uniform shape and let SDPA
            # broadcast across the query dimension.
            return key_mask.view(1, 1, 1, key_len)

        # Prefill: per-row causal upper-triangular cut. The q-th query row's
        # absolute position is `cached_len + q = (key_len - query_len) + q`;
        # any k strictly greater is a future token and must be -inf,
        # because setting kwargs["attention_mask"] non-None disables HF
        # SDPA's implicit causal.
        offset = key_len - query_len
        q_idx = torch.arange(query_len, device=device).view(query_len, 1)
        k_idx = torch.arange(key_len, device=device).view(1, key_len)
        causal_keep = k_idx <= (offset + q_idx)  # [Q, K] bool
        mask_2d = torch.where(
            causal_keep,
            key_mask.view(1, key_len).expand(query_len, key_len),
            torch.full((), neg_inf, dtype=dtype, device=device),
        )
        return mask_2d.view(1, 1, query_len, key_len)

    def kept_count(self, key_len: int) -> int:
        """Number of unmasked key positions for the given `key_len`."""
        if key_len <= 0:
            return 0
        sink = min(self.sink_size, key_len)
        recent_start = max(0, key_len - self.recent_window)
        # Union of [0, sink) and [recent_start, key_len)
        if recent_start <= sink:
            return key_len
        return sink + (key_len - recent_start)

    def effective_kept_count_sum(self, *, query_len: int, key_len: int) -> int:
        """Number of sparse-and-causal visible `(query, key)` cells.

        `kept_count()` is a key-uniform summary of the sliding sparsity
        pattern. During prefill, causal masking additionally removes future
        keys per query row, so the effective count must sum across query rows.
        """
        if query_len < 1:
            raise ValueError(f"query_len must be >= 1; got {query_len!r}")
        if key_len <= 0:
            return 0

        sink = min(self.sink_size, key_len)
        recent_start = max(0, key_len - self.recent_window)
        offset = key_len - query_len
        if recent_start <= sink:
            return _causal_visible_interval_sum(
                start=0,
                end=key_len,
                offset=offset,
                query_len=query_len,
            )
        return _causal_visible_interval_sum(
            start=0,
            end=sink,
            offset=offset,
            query_len=query_len,
        ) + _causal_visible_interval_sum(
            start=recent_start,
            end=key_len,
            offset=offset,
            query_len=query_len,
        )

    def effective_density(self, *, query_len: int, key_len: int) -> float:
        """Fraction of sparse-and-causal visible `(query, key)` cells."""
        if query_len < 1:
            raise ValueError(f"query_len must be >= 1; got {query_len!r}")
        if key_len <= 0:
            return 0.0
        return float(
            self.effective_kept_count_sum(query_len=query_len, key_len=key_len)
        ) / float(query_len * key_len)

    def reset_state(self) -> None:
        """No-op: sliding window is stateless."""

    def record_metadata(
        self,
        *,
        layer_idx: int,
        phase: str,
        decode_step: int,
    ) -> dict[str, Any]:
        del layer_idx, phase, decode_step
        # sink_size / recent_window already live in attempt-level meta.json
        # (the `sparse_attention` block); duplicating them per row in
        # extras_json would be pure bloat. Return an empty dict so the
        # recorder serialises the literal "{}" string and the column stays
        # schema-stable.
        return {}


def _causal_visible_interval_sum(
    *,
    start: int,
    end: int,
    offset: int,
    query_len: int,
) -> int:
    """Sum visible keys in `[start, end)` over causal query rows.

    Query row `q` is at absolute position `offset + q`, so it can see keys
    `k <= offset + q`. This returns:
    `sum_q |{k in [start, end): k <= offset + q}|`.
    """
    if query_len <= 0 or end <= start:
        return 0
    length = end - start
    # Per-row visible count is clamp((offset + q + 1) - start, 0, length).
    base = offset + 1 - start
    partial_start = max(0, -base + 1)
    partial_end = min(query_len, length - base)
    total = 0
    if partial_start < partial_end:
        n_partial = partial_end - partial_start
        q_sum = (partial_start + partial_end - 1) * n_partial // 2
        total += base * n_partial + q_sum
    full_start = max(0, length - base)
    if full_start < query_len:
        total += length * (query_len - full_start)
    return int(total)
