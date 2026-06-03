"""Tests for sparse-filtered segment attention grid construction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.recoding_figures.plot_sparse_segment_grid import (  # noqa: E402
    build_sparse_segment_grids,
    sparse_filtered_segment_rows,
)
from scripts.recoding_figures.recording_loader import load_iteration_records  # noqa: E402
from scripts.recoding_figures.sparse_keep_sets import SparseParams  # noqa: E402


def _write_attempt(
    tmp_path: Path,
    *,
    missing_sparse: bool = False,
    observe_only: bool = True,
    sparse_decode_steps: list[int] | None = None,
    sparse_kept_count: int = 3,
) -> Path:
    attempt_dir = tmp_path / "task-a" / "attempt_1"
    iter_dir = attempt_dir / "recordings" / "iter_0000"
    iter_dir.mkdir(parents=True)

    meta = {
        "sparse_attention": {
            "method": "block_topk",
            "sink_size": 1,
            "recent_window": 1,
            "budget": 3,
            "block_size": 2,
            "score_reduction": "max",
            "phase_scope": "decode_only",
            "record": True,
            "observe_only": observe_only,
        },
        "iters": [
            {
                "dir": "iter_0000",
                "call_idx": 0,
                "input_tokens": 8,
                "output_tokens": 1,
                "complete": True,
            }
        ],
    }
    (attempt_dir / "recordings" / "meta.json").write_text(
        json.dumps(meta),
        encoding="utf-8",
    )
    segments = {
        "call_idx": 0,
        "complete": True,
        "segments": [
            {
                "role": "system",
                "token_start": 0,
                "token_end": 2,
                "message_index": 0,
                "first_seen_call": 0,
            },
            {
                "role": "user",
                "token_start": 2,
                "token_end": 8,
                "message_index": 1,
                "first_seen_call": 0,
            },
        ],
    }
    (iter_dir / "segments.json").write_text(json.dumps(segments), encoding="utf-8")
    (iter_dir / ".done").write_text("", encoding="utf-8")

    np.savez_compressed(
        iter_dir / "attention.npz",
        call_idx=np.int32(0),
        record_layer=np.asarray([0], dtype=np.int32),
        record_phase=np.asarray(["decode"], dtype="U7"),
        record_decode_step=np.asarray([0], dtype=np.int32),
        query_row_offsets=np.asarray([0, 1], dtype=np.int64),
        query_positions=np.asarray([7], dtype=np.int32),
        query_heads=np.asarray([0], dtype=np.int32),
        topk_csr_offsets=np.asarray([0, 4], dtype=np.int64),
        topk_csr_indices=np.asarray([0, 3, 6, 7], dtype=np.int32),
        topk_csr_weights=np.asarray([0.2, 0.5, 0.25, 0.05], dtype=np.float32),
        top_k=np.int32(4),
    )
    np.savez_compressed(iter_dir / "routing.npz", call_idx=np.int32(0))
    if not missing_sparse:
        decode_steps = sparse_decode_steps or [0]
        extras = [
            json.dumps(
                {
                    "selection_reason": "selected",
                    "selected_middle_indices": [3],
                },
                sort_keys=True,
            )
            for _ in decode_steps
        ]
        np.savez_compressed(
            iter_dir / "sparse_attention.npz",
            call_idx=np.int32(0),
            method_name=np.asarray("block_topk", dtype="U16"),
            record_step=np.arange(len(decode_steps), dtype=np.int32),
            record_layer=np.zeros(len(decode_steps), dtype=np.int32),
            record_phase=np.asarray(["decode"] * len(decode_steps), dtype="U7"),
            record_decode_step=np.asarray(decode_steps, dtype=np.int32),
            query_len=np.ones(len(decode_steps), dtype=np.int32),
            key_len=np.full(len(decode_steps), 8, dtype=np.int32),
            kept_count=np.full(len(decode_steps), sparse_kept_count, dtype=np.int32),
            density=np.full(len(decode_steps), sparse_kept_count / 8, dtype=np.float16),
            extras_json=np.asarray(extras, dtype=object),
        )
    return attempt_dir


def test_sparse_filtered_segment_rows_use_runtime_keep_set(tmp_path: Path) -> None:
    attempt = _write_attempt(tmp_path)
    records = load_iteration_records([attempt])
    trajectory, by_layer = sparse_filtered_segment_rows(
        records,
        method_name="block_topk",
        method_params=SparseParams(sink_size=1, recent_window=1),
    )
    assert len(trajectory) == 2
    assert len(by_layer) == 2

    rows = {row["role"]: row for row in trajectory}
    assert rows["system"]["visible_attention_share_mean"] == pytest.approx(0.2)
    assert rows["user"]["visible_attention_share_mean"] == pytest.approx(0.55)
    assert rows["user"]["visible_attention_over_baseline"] == pytest.approx(0.55 / 0.75)


def test_sparse_filtered_segment_rows_raise_on_missing_sparse_row(tmp_path: Path) -> None:
    attempt = _write_attempt(tmp_path, missing_sparse=True)
    records = load_iteration_records([attempt])
    with pytest.raises(FileNotFoundError, match="sparse_attention.npz"):
        sparse_filtered_segment_rows(
            records,
            method_name="block_topk",
            method_params=SparseParams(sink_size=1, recent_window=1),
        )


def test_sparse_filtered_segment_rows_ignore_extra_sparse_rows(tmp_path: Path) -> None:
    attempt = _write_attempt(tmp_path, sparse_decode_steps=[0, 1])
    records = load_iteration_records([attempt])
    trajectory, by_layer = sparse_filtered_segment_rows(
        records,
        method_name="block_topk",
        method_params=SparseParams(sink_size=1, recent_window=1),
    )
    assert len(trajectory) == 2
    assert len(by_layer) == 2


def test_sparse_filtered_segment_rows_raise_on_kept_count_drift(tmp_path: Path) -> None:
    attempt = _write_attempt(tmp_path, sparse_kept_count=4)
    records = load_iteration_records([attempt])
    with pytest.raises(ValueError, match="reconstructed keep_set size"):
        sparse_filtered_segment_rows(
            records,
            method_name="block_topk",
            method_params=SparseParams(sink_size=1, recent_window=1),
        )


def test_build_sparse_segment_grids_rejects_non_observe_only(tmp_path: Path) -> None:
    attempt = _write_attempt(tmp_path, observe_only=False)
    with pytest.raises(ValueError, match="observe_only"):
        build_sparse_segment_grids(inputs=[attempt], output_dir=tmp_path / "out")


def test_build_sparse_segment_grids_writes_per_task_outputs(tmp_path: Path) -> None:
    attempt = _write_attempt(tmp_path)
    output_dir = tmp_path / "out"
    summary = build_sparse_segment_grids(inputs=[attempt], output_dir=output_dir)
    group = summary["groups"][0]
    assert group["label"] == "task-a"
    assert Path(group["plot"]["grid_png"]).is_file()
    assert (output_dir / "task-a" / "segment_attention_sparse_filtered_trajectory.csv").is_file()
