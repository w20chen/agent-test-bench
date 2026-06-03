"""Per-call audit recorder for KV eviction decisions.

One `KVEvictionRecorder` instance is created per `recording_session()` (i.e.
per LLM call) and written to `recordings/iter_<call_idx>/kv_eviction.npz` at
`LayerCapturer.flush()` time. Schema mirrors `attention.npz`'s CSR layout so
both can be joined on `(call_idx, layer, decode_step)`.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import numpy as np


# Sentinel values for absent h2o scores in non-h2o records.
_SCORE_INDEX_MISSING = -1
_SCORE_VALUE_MISSING = np.nan


@dataclass
class _Row:
    """One row of the kv_eviction.npz CSR table."""

    step: int
    layer: int
    phase: str
    pre_len: int
    post_len: int
    budget: int
    kept_indices: list[int]
    evicted_indices: list[int]
    evict_reason: str
    # h2o-only; None on non-h2o policies.
    score_topk_index: list[int] | None = None
    score_topk_value: list[float] | None = None
    score_evicted_index: list[int] | None = None
    score_evicted_value: list[float] | None = None


@dataclass
class KVEvictionRecorder:
    """Buffers per-update() decisions, flushed once to npz on session close."""

    call_idx: int
    policy_name: str
    _rows: list[_Row] = field(default_factory=list, init=False, repr=False)

    def append(
        self,
        *,
        step: int,
        layer: int,
        phase: str,
        pre_len: int,
        post_len: int,
        budget: int,
        kept_indices: list[int],
        evicted_indices: list[int],
        evict_reason: str,
        score_topk_index: list[int] | None = None,
        score_topk_value: list[float] | None = None,
        score_evicted_index: list[int] | None = None,
        score_evicted_value: list[float] | None = None,
    ) -> None:
        if pre_len - post_len != len(evicted_indices):
            raise ValueError(
                f"recorder invariant violated: pre_len={pre_len}, "
                f"post_len={post_len}, evicted={len(evicted_indices)}"
            )
        if score_evicted_index is not None and score_evicted_value is not None:
            if len(score_evicted_index) != len(score_evicted_value):
                raise ValueError(
                    "score_evicted_index/value length mismatch: "
                    f"{len(score_evicted_index)} vs {len(score_evicted_value)}"
                )
        self._rows.append(
            _Row(
                step=int(step),
                layer=int(layer),
                phase=str(phase),
                pre_len=int(pre_len),
                post_len=int(post_len),
                budget=int(budget),
                kept_indices=list(kept_indices),
                evicted_indices=list(evicted_indices),
                evict_reason=str(evict_reason),
                score_topk_index=(
                    list(score_topk_index) if score_topk_index is not None else None
                ),
                score_topk_value=(
                    list(score_topk_value) if score_topk_value is not None else None
                ),
                score_evicted_index=(
                    list(score_evicted_index)
                    if score_evicted_index is not None
                    else None
                ),
                score_evicted_value=(
                    list(score_evicted_value)
                    if score_evicted_value is not None
                    else None
                ),
            )
        )

    def n_records(self) -> int:
        return len(self._rows)

    def write(self, npz_path: pathlib.Path) -> None:
        """Serialise buffer to a single compressed npz.

        Empty buffers raise — an empty kv_eviction.npz would be silently
        misleading downstream; callers should gate on `n_records() > 0`.
        """
        if not self._rows:
            raise RuntimeError(
                f"KVEvictionRecorder(call_idx={self.call_idx}) has no records; "
                "gate the write on n_records() > 0"
            )

        n_rows = len(self._rows)
        record_step = np.fromiter((r.step for r in self._rows), dtype=np.int32, count=n_rows)
        record_layer = np.fromiter((r.layer for r in self._rows), dtype=np.int32, count=n_rows)
        # NumPy "U7" / "U16" preserve plan-spec fixed-width unicode dtype.
        record_phase = np.asarray([r.phase for r in self._rows], dtype="U7")
        pre_len = np.fromiter((r.pre_len for r in self._rows), dtype=np.int32, count=n_rows)
        post_len = np.fromiter((r.post_len for r in self._rows), dtype=np.int32, count=n_rows)
        budget = np.fromiter((r.budget for r in self._rows), dtype=np.int32, count=n_rows)
        evict_reason = np.asarray([r.evict_reason for r in self._rows], dtype="U16")

        kept_offsets, kept_indices = _build_csr(
            [r.kept_indices for r in self._rows]
        )
        evicted_offsets, evicted_indices = _build_csr(
            [r.evicted_indices for r in self._rows]
        )

        score_topk_index, score_topk_value = _build_score_topk(self._rows)
        (
            score_evicted_offsets,
            score_evicted_index,
            score_evicted_value,
        ) = _build_score_csr(
            self._rows,
            index_attr="score_evicted_index",
            value_attr="score_evicted_value",
        )

        np.savez_compressed(
            npz_path,
            call_idx=np.asarray(self.call_idx, dtype=np.int32),
            policy_name=np.asarray(self.policy_name, dtype="U16"),
            record_step=record_step,
            record_layer=record_layer,
            record_phase=record_phase,
            pre_len=pre_len,
            post_len=post_len,
            budget=budget,
            kept_offsets=kept_offsets,
            kept_indices=kept_indices,
            evicted_offsets=evicted_offsets,
            evicted_indices=evicted_indices,
            evict_reason=evict_reason,
            score_topk_index=score_topk_index,
            score_topk_value=score_topk_value,
            score_evicted_offsets=score_evicted_offsets,
            score_evicted_index=score_evicted_index,
            score_evicted_value=score_evicted_value,
        )


def _build_csr(per_row: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Build CSR (offsets, flat_indices) from a list of variable-length rows."""
    n_rows = len(per_row)
    offsets = np.zeros(n_rows + 1, dtype=np.int64)
    for i, row in enumerate(per_row):
        offsets[i + 1] = offsets[i] + len(row)
    total = int(offsets[-1])
    flat = np.empty(total, dtype=np.int32)
    cursor = 0
    for row in per_row:
        if not row:
            continue
        flat[cursor : cursor + len(row)] = np.asarray(row, dtype=np.int32)
        cursor += len(row)
    return offsets, flat


def _build_score_topk(rows: list[_Row]) -> tuple[np.ndarray, np.ndarray]:
    """Build a dense (R, k) score table; -1 / NaN for missing rows.

    `k` is the max non-None row length so non-h2o callers contribute a
    well-defined (R, 0) table.
    """
    n_rows = len(rows)
    k = 0
    for r in rows:
        if r.score_topk_index is not None:
            k = max(k, len(r.score_topk_index))
    index = np.full((n_rows, k), _SCORE_INDEX_MISSING, dtype=np.int32)
    value = np.full((n_rows, k), _SCORE_VALUE_MISSING, dtype=np.float32)
    for i, r in enumerate(rows):
        if r.score_topk_index is None:
            continue
        n = len(r.score_topk_index)
        index[i, :n] = np.asarray(r.score_topk_index, dtype=np.int32)
        if r.score_topk_value is not None:
            value[i, : len(r.score_topk_value)] = np.asarray(
                r.score_topk_value, dtype=np.float32
            )
    return index, value


def _build_score_csr(
    rows: list[_Row],
    *,
    index_attr: str,
    value_attr: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build CSR arrays for variable-width score diagnostics."""
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    per_row_index: list[list[int]] = []
    per_row_value: list[list[float]] = []
    for row in rows:
        indices = getattr(row, index_attr)
        values = getattr(row, value_attr)
        if indices is None:
            row_indices: list[int] = []
            row_values: list[float] = []
        else:
            row_indices = list(indices)
            if values is None:
                row_values = [float("nan")] * len(row_indices)
            else:
                row_values = list(values)
            if len(row_indices) != len(row_values):
                raise ValueError(
                    f"{index_attr}/{value_attr} length mismatch: "
                    f"{len(row_indices)} vs {len(row_values)}"
                )
        per_row_index.append(row_indices)
        per_row_value.append(row_values)
        offsets[len(per_row_index)] = offsets[len(per_row_index) - 1] + len(row_indices)

    total = int(offsets[-1])
    flat_index = np.empty(total, dtype=np.int32)
    flat_value = np.empty(total, dtype=np.float32)
    cursor = 0
    for row_indices, row_values in zip(per_row_index, per_row_value, strict=True):
        if not row_indices:
            continue
        n = len(row_indices)
        flat_index[cursor : cursor + n] = np.asarray(row_indices, dtype=np.int32)
        flat_value[cursor : cursor + n] = np.asarray(row_values, dtype=np.float32)
        cursor += n
    return offsets, flat_index, flat_value
