"""Mask and keep-set utilities for sparse attention methods."""

from __future__ import annotations

from collections.abc import Iterable


def validate_sparse_budget(
    *,
    method_name: str,
    budget: int | None,
    sink_size: int,
    recent_window: int,
    block_size: int,
    score_reduction: str,
    phase_scope: str,
) -> int:
    """Validate common dynamic sparse-attention configuration."""
    if budget is None:
        raise ValueError(f"{method_name} requires a positive budget")
    budget_int = int(budget)
    if budget_int <= 0:
        raise ValueError(f"{method_name} requires budget > 0; got {budget!r}")
    if sink_size < 0:
        raise ValueError(f"{method_name} requires sink_size >= 0; got {sink_size!r}")
    if recent_window < 0:
        raise ValueError(
            f"{method_name} requires recent_window >= 0; got {recent_window!r}"
        )
    if budget_int < int(sink_size) + int(recent_window):
        raise ValueError(
            f"{method_name} requires budget >= sink_size + recent_window "
            f"({budget_int!r} < {sink_size!r} + {recent_window!r})"
        )
    if block_size <= 0:
        raise ValueError(f"{method_name} requires block_size > 0; got {block_size!r}")
    if score_reduction not in {"max", "mean"}:
        raise ValueError(
            f"{method_name} requires score_reduction in {{'max', 'mean'}}; "
            f"got {score_reduction!r}"
        )
    if phase_scope != "decode_only":
        raise ValueError(
            f"{method_name} currently supports only phase_scope='decode_only'; "
            f"got {phase_scope!r}"
        )
    return budget_int


def sink_recent_keep_indices(
    *, key_len: int, sink_size: int, recent_window: int
) -> set[int]:
    """Return the sink-prefix and recent-tail keep set."""
    keep: set[int] = set()
    if key_len <= 0:
        return keep
    sink = min(int(sink_size), int(key_len))
    if sink > 0:
        keep.update(range(0, sink))
    if int(recent_window) > 0:
        keep.update(range(max(0, int(key_len) - int(recent_window)), int(key_len)))
    return keep


def middle_indices(*, key_len: int, sink_size: int, recent_window: int) -> list[int]:
    """Return candidate middle positions not covered by sink/recent."""
    if key_len <= 0:
        return []
    start = min(max(0, int(sink_size)), int(key_len))
    end = max(start, int(key_len) - max(0, int(recent_window)))
    return list(range(start, end))


def cap_middle_selection(
    *,
    key_len: int,
    budget: int,
    sink_size: int,
    recent_window: int,
    ranked_middle: Iterable[int],
) -> list[int]:
    """Take ranked middle positions until the full keep set reaches budget."""
    base = sink_recent_keep_indices(
        key_len=key_len, sink_size=sink_size, recent_window=recent_window
    )
    slots = max(0, min(int(budget), int(key_len)) - len(base))
    if slots == 0:
        return []
    selected: list[int] = []
    seen: set[int] = set()
    candidates = set(middle_indices(key_len=key_len, sink_size=sink_size, recent_window=recent_window))
    for idx in ranked_middle:
        pos = int(idx)
        if pos in seen or pos not in candidates:
            continue
        selected.append(pos)
        seen.add(pos)
        if len(selected) >= slots:
            break
    return selected


def dense_or_causal_mask(
    *,
    query_len: int,
    key_len: int,
    device: object,
    dtype: object,
) -> object:
    """Return a dense additive mask with explicit causal cut for prefill."""
    import torch

    if query_len < 1:
        raise ValueError(f"query_len must be >= 1; got {query_len!r}")
    if key_len < 0:
        raise ValueError(f"key_len must be >= 0; got {key_len!r}")
    if query_len == 1:
        return torch.zeros((1, 1, 1, key_len), dtype=dtype, device=device)

    offset = key_len - query_len
    q_idx = torch.arange(query_len, device=device).view(query_len, 1)
    k_idx = torch.arange(key_len, device=device).view(1, key_len)
    keep = k_idx <= (offset + q_idx)
    neg_inf = torch.finfo(dtype).min
    mask = torch.where(
        keep,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), neg_inf, dtype=dtype, device=device),
    )
    return mask.view(1, 1, query_len, key_len)


def additive_mask_from_keep(
    *,
    keep_indices: Iterable[int],
    query_len: int,
    key_len: int,
    device: object,
    dtype: object,
) -> object:
    """Return a sparse additive mask with explicit causal cut for prefill."""
    import torch

    if query_len < 1:
        raise ValueError(f"query_len must be >= 1; got {query_len!r}")
    if key_len < 0:
        raise ValueError(f"key_len must be >= 0; got {key_len!r}")

    keep = torch.zeros(key_len, dtype=torch.bool, device=device)
    for idx in keep_indices:
        pos = int(idx)
        if 0 <= pos < key_len:
            keep[pos] = True
    neg_inf = torch.finfo(dtype).min
    key_mask = torch.where(
        keep,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), neg_inf, dtype=dtype, device=device),
    )
    if query_len == 1:
        return key_mask.view(1, 1, 1, key_len)

    causal = dense_or_causal_mask(
        query_len=query_len, key_len=key_len, device=device, dtype=dtype
    ).view(query_len, key_len)
    sparse = key_mask.view(1, key_len).expand(query_len, key_len)
    mask = torch.where(causal < 0, causal, sparse)
    return mask.view(1, 1, query_len, key_len)


def block_id_for_position(position: int, block_size: int) -> int:
    """Return the contiguous block id for an absolute key position."""
    return int(position) // int(block_size)


__all__ = [
    "additive_mask_from_keep",
    "block_id_for_position",
    "cap_middle_selection",
    "dense_or_causal_mask",
    "middle_indices",
    "sink_recent_keep_indices",
    "validate_sparse_budget",
]
