"""Shared sparse-attention keep-set reconstruction helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SparseParams:
    sink_size: int
    recent_window: int


def reconstruct_keep_set(
    *,
    method_name: str,
    method_params: SparseParams,
    key_len: int,
    extras: dict[str, object] | None = None,
) -> np.ndarray:
    """Return sorted unique key positions kept by `method_name`.

    Dynamic methods reconstruct their middle-token keep set from the runtime
    `extras_json["selected_middle_indices"]` emitted by `sparse_attention.npz`.
    Output dtype is `np.int32`.
    """
    if method_name not in {"sliding", "streaming", "heavy_hitter", "block_topk", "quest"}:
        raise NotImplementedError(
            f"keep-set reconstruction for method {method_name!r} is not implemented"
        )
    if key_len <= 0:
        return np.empty(0, dtype=np.int32)
    if method_name in {"heavy_hitter", "block_topk", "quest"}:
        extras = extras or {}
        reason = str(extras.get("selection_reason", ""))
        if reason in {"phase_dense", "prefill_dense"}:
            return np.arange(key_len, dtype=np.int32)
    sink = min(method_params.sink_size, key_len)
    recent_start = max(0, key_len - method_params.recent_window)
    keep = np.zeros(key_len, dtype=bool)
    if sink > 0:
        keep[:sink] = True
    if method_params.recent_window > 0:
        keep[recent_start:] = True
    if method_name in {"heavy_hitter", "block_topk", "quest"}:
        if extras is None:
            raise ValueError(
                f"method {method_name!r} requires extras_json selected_middle_indices"
            )
        raw_selected = extras.get("selected_middle_indices")
        if raw_selected is None:
            raise ValueError(
                f"method {method_name!r} extras_json missing selected_middle_indices"
            )
        if not isinstance(raw_selected, list):
            raise ValueError(
                f"method {method_name!r} selected_middle_indices must be a list"
            )
        for item in raw_selected:
            pos = int(item)
            if 0 <= pos < key_len:
                keep[pos] = True
    return np.nonzero(keep)[0].astype(np.int32)
