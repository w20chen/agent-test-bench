"""Per-call audit recorder for sparse attention decisions.

One `SparseAttentionRecorder` per `recording_session()` (per LLM call). Each
pre-hook invocation appends one row keyed by `(layer, phase, decode_step)`;
`write(npz_path)` flushes the accumulated rows to a compressed npz next to
`attention.npz` / `kv_eviction.npz`.

Method-specific extras are serialised to a single `extras_json` U-column
rather than minted into schema columns; this keeps the writer
method-agnostic at the cost of a JSON-decode step on the read path. The
artifact is small (one row per (layer, phase, step)) so CSR /
pre-allocation is not warranted.

`density` semantics: for key-uniform methods (sliding), density describes
the sparse key pattern before the causal cut. During prefill, the true
sparse-and-causal visible-cell fraction is query-row dependent and belongs
in `extras_json` (for sliding: `effective_kept_count_sum` and
`effective_density`). For future per-query methods (Quest, MInference, etc.)
`kept_count` should be the MEAN kept count across query rows, and `density`
accordingly the mean per-row keep fraction; per-row breakdowns belong in
`extras_json` so the top-level schema stays method-agnostic.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class _Row:
    """One row of the sparse_attention.npz table."""

    step: int
    layer: int
    phase: str
    decode_step: int
    query_len: int
    key_len: int
    kept_count: int
    density: float
    extras: dict[str, Any]


@dataclass
class SparseAttentionRecorder:
    """Buffers per-forward sparse decisions, flushed once to npz."""

    call_idx: int
    method_name: str
    _rows: list[_Row] = field(default_factory=list, init=False, repr=False)

    def append(
        self,
        *,
        step: int,
        layer: int,
        phase: str,
        decode_step: int,
        query_len: int,
        key_len: int,
        kept_count: int,
        extras: dict[str, Any] | None = None,
    ) -> None:
        if query_len < 1:
            raise ValueError(f"query_len must be >= 1; got {query_len!r}")
        if key_len < 0:
            raise ValueError(f"key_len must be >= 0; got {key_len!r}")
        if kept_count < 0 or kept_count > key_len:
            raise ValueError(
                f"kept_count must be in [0, key_len]; got kept_count={kept_count!r}, "
                f"key_len={key_len!r}"
            )
        density = float(kept_count) / float(key_len) if key_len > 0 else 0.0
        self._rows.append(
            _Row(
                step=int(step),
                layer=int(layer),
                phase=str(phase),
                decode_step=int(decode_step),
                query_len=int(query_len),
                key_len=int(key_len),
                kept_count=int(kept_count),
                density=density,
                extras=dict(extras) if extras is not None else {},
            )
        )

    def n_records(self) -> int:
        return len(self._rows)

    def write(self, npz_path: pathlib.Path) -> None:
        """Serialise buffer to a compressed npz.

        Empty buffers raise — a zero-row sparse_attention.npz is silently
        misleading downstream; callers should gate the write on
        `n_records() > 0` (mirrors `KVEvictionRecorder.write`).
        """
        if not self._rows:
            raise RuntimeError(
                f"SparseAttentionRecorder(call_idx={self.call_idx}) has no records; "
                "gate the write on n_records() > 0"
            )

        n_rows = len(self._rows)
        record_step = np.fromiter((r.step for r in self._rows), dtype=np.int32, count=n_rows)
        record_layer = np.fromiter(
            (r.layer for r in self._rows), dtype=np.int32, count=n_rows
        )
        record_phase = np.asarray([r.phase for r in self._rows], dtype="U7")
        record_decode_step = np.fromiter(
            (r.decode_step for r in self._rows), dtype=np.int32, count=n_rows
        )
        query_len = np.fromiter(
            (r.query_len for r in self._rows), dtype=np.int32, count=n_rows
        )
        key_len = np.fromiter(
            (r.key_len for r in self._rows), dtype=np.int32, count=n_rows
        )
        kept_count = np.fromiter(
            (r.kept_count for r in self._rows), dtype=np.int32, count=n_rows
        )
        density = np.asarray(
            [r.density for r in self._rows], dtype=np.float16
        )
        # extras as JSON strings; downstream loader json.loads() lazily.
        extras_json = np.asarray(
            [json.dumps(r.extras, sort_keys=True) for r in self._rows],
            dtype=object,
        )

        if len(self.method_name) > 16:
            raise ValueError(
                f"method_name {self.method_name!r} exceeds the 16-char npz column "
                "width. Either rename the method shorter or widen the U16 dtype "
                "(both writer here and loader in recording_loader.py)."
            )
        np.savez_compressed(
            npz_path,
            call_idx=np.asarray(self.call_idx, dtype=np.int32),
            method_name=np.asarray(self.method_name, dtype="U16"),
            record_step=record_step,
            record_layer=record_layer,
            record_phase=record_phase,
            record_decode_step=record_decode_step,
            query_len=query_len,
            key_len=key_len,
            kept_count=kept_count,
            density=density,
            extras_json=extras_json,
        )
