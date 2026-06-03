"""Small tensor utilities for per-call internal recordings."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RecordingConfig:
    """Runtime knobs for bounded internal recording."""

    attention_top_k: int = 32
    decode_window: int = 64
    max_prefill_queries: int = 80
    model_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    generation_seed: int = 0
    # Layers for which per-head per-span attention stats are captured.
    # Empty tuple disables the feature. Must all be < num_hidden_layers.
    # Recommended for real models: (0, 6, 12, 18, 24, 30, 36, 47)
    # Layer 47 is the last attention layer in Qwen3-Coder-30B (retrieval/copy heads concentrate there).
    per_head_stats_layers: tuple[int, ...] = ()


def segment_role(message: dict[str, Any]) -> str:
    role = str(message.get("role") or "unknown")
    if role == "assistant" and message.get("tool_calls"):
        return "assistant_call"
    if role == "assistant":
        return "assistant_message"
    if role == "tool":
        return "tool_result"
    return role


def query_sampling_seed(base_seed: int, call_idx: int) -> str:
    """Stable per-call seed for bounded prefill query-row sampling."""
    return f"{int(base_seed)}:{int(call_idx)}"


def select_query_positions(
    query_len: int,
    max_queries: int,
    *,
    seed: int | str = 0,
) -> list[int]:
    if query_len <= 0:
        raise ValueError(f"query_len must be positive, got {query_len}")
    if max_queries <= 0:
        raise ValueError(f"max_queries must be positive, got {max_queries}")
    if query_len <= max_queries:
        return list(range(query_len))
    rng = random.Random(seed)
    positions: list[int] = []
    for idx in range(max_queries):
        start = (idx * query_len) // max_queries
        stop = ((idx + 1) * query_len) // max_queries
        positions.append(rng.randrange(start, stop))
    return positions


def token_segment_ids(
    total_tokens: int,
    segments: list[dict[str, Any]],
    *,
    generated_segment_id: int | None = None,
):
    import torch

    if total_tokens < 0:
        raise ValueError(f"total_tokens must be non-negative, got {total_tokens}")
    fill = -1 if generated_segment_id is None else generated_segment_id
    ids = torch.full((total_tokens,), fill, dtype=torch.long)
    for idx, segment in enumerate(segments):
        start = int(segment.get("token_start", segment.get("start", 0)))
        end = int(segment.get("token_end", segment.get("end", 0)))
        if start < 0 or end < start:
            raise ValueError(f"invalid segment bounds: {segment}")
        if start >= total_tokens:
            continue
        ids[start : min(end, total_tokens)] = idx
    return ids


def segment_bucket(attn_rows, token_ids, n_segments: int):
    import torch

    if n_segments <= 0:
        raise ValueError(f"n_segments must be positive, got {n_segments}")
    if attn_rows.ndim == 4:
        attn_rows = attn_rows[0].mean(dim=0)
    elif attn_rows.ndim == 3:
        attn_rows = attn_rows.mean(dim=0)
    if attn_rows.ndim != 2:
        raise ValueError(f"attention rows must be rank 2 after head mean: {attn_rows.shape}")

    rows = attn_rows.to(dtype=torch.float32)
    key_ids = token_ids[: rows.shape[1]].to(device=rows.device)
    out = torch.zeros(
        (rows.shape[0], n_segments), dtype=torch.float32, device=rows.device
    )
    valid = (key_ids >= 0) & (key_ids < n_segments)
    if bool(valid.any()):
        index = key_ids[valid].view(1, -1).expand(rows.shape[0], -1)
        out.scatter_add_(1, index, rows[:, valid])
    row_sums = out.sum(dim=1, keepdim=True)
    nonzero = row_sums > 0
    out = torch.where(nonzero, out / row_sums.clamp_min(torch.finfo(out.dtype).tiny), out)
    return out


def padded_top_k(rows, k: int):
    import torch

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if rows.ndim != 2:
        raise ValueError(f"rows must be rank 2, got {rows.shape}")
    effective_k = min(k, int(rows.shape[1]))
    weights, indices = torch.topk(rows, k=effective_k, dim=-1)
    weights = weights.to(dtype=torch.float32)
    if effective_k == k:
        return indices, weights
    pad_cols = k - effective_k
    index_pad = torch.full(
        (rows.shape[0], pad_cols), -1, dtype=indices.dtype, device=indices.device
    )
    weight_pad = torch.zeros(
        (rows.shape[0], pad_cols), dtype=weights.dtype, device=weights.device
    )
    return torch.cat([indices, index_pad], dim=1), torch.cat([weights, weight_pad], dim=1)


def heavy_hitter(rows, k: int):
    if rows.ndim != 2:
        raise ValueError(f"rows must be rank 2, got {rows.shape}")
    return padded_top_k(rows.mean(dim=0, keepdim=True, dtype=rows.dtype), k)


def expert_load_per_segment(
    router_logits,
    token_ids,
    *,
    n_segments: int,
    top_k_experts: int,
):
    import torch

    if router_logits.ndim < 2:
        raise ValueError(f"router_logits must end with expert dimension: {router_logits.shape}")
    logits = router_logits.reshape(-1, router_logits.shape[-1]).to(dtype=torch.float32)
    n_tokens, n_experts = int(logits.shape[0]), int(logits.shape[1])
    k = min(top_k_experts, n_experts)
    weights, choices = torch.topk(torch.softmax(logits, dim=-1), k=k, dim=-1)
    load = torch.zeros((n_segments, n_experts), dtype=torch.float32, device=logits.device)

    segment_ids = token_ids[:n_tokens].to(device=logits.device)
    valid = (segment_ids >= 0) & (segment_ids < n_segments)
    if bool(valid.any()):
        valid_segments = segment_ids[valid]
        for rank in range(k):
            load.index_put_(
                (valid_segments, choices[valid, rank]),
                weights[valid, rank],
                accumulate=True,
            )
    return choices, weights, load


class DecodeRingBuffer:
    """Keep only the most recent decode records."""

    def __init__(self, maxlen: int) -> None:
        if maxlen <= 0:
            raise ValueError(f"maxlen must be positive, got {maxlen}")
        self._items: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._dropped = 0

    def append(self, item: dict[str, Any]) -> None:
        if len(self._items) == self._items.maxlen:
            self._dropped += 1
        self._items.append(item)

    def clear(self) -> None:
        self._items.clear()
        self._dropped = 0

    def to_list(self) -> list[dict[str, Any]]:
        return list(self._items)

    def dropped_count(self) -> int:
        return int(self._dropped)

    def __len__(self) -> int:
        return len(self._items)
