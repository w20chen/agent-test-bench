"""Unit tests for scripts/recoding_figures/kv_eviction_metrics.py.

Synthetic frames + tiny on-disk attention.npz fixtures exercise every
public function without needing real Modal traces.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "recoding_figures"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from kv_eviction_metrics import (  # noqa: E402
    aggregate_heavy_hitters_per_layer,
    aggregate_sink_recent_share_per_layer,
    compute_attention_js_per_layer,
    compute_eviction_profile_rows,
    compute_heavy_jaccard_per_layer,
    compute_phase_distribution,
    compute_role_survival_rows,
    jaccard,
    js_divergence,
    load_segments_by_iter_dir,
)
from recording_loader import (  # noqa: E402
    IterationRecord,
    KVEvictionFrame,
    _has_recording_files,
    load_kv_eviction,
)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def test_js_divergence_matching_dists_is_zero():
    p = np.array([0.4, 0.6])
    assert js_divergence(p, p) == pytest.approx(0.0, abs=1e-9)


def test_js_divergence_disjoint_dists_is_ln2():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert js_divergence(p, q) == pytest.approx(np.log(2.0), abs=1e-6)


def test_js_divergence_handles_zero_total():
    p = np.array([0.0, 0.0])
    q = np.array([0.5, 0.5])
    assert js_divergence(p, q) == 0.0


def test_jaccard_basic():
    assert jaccard({1, 2, 3}, {2, 3, 4}) == pytest.approx(2 / 4)
    assert jaccard(set(), set()) == 0.0
    assert jaccard({1, 2}, set()) == 0.0
    assert jaccard({1, 2}, {1, 2}) == 1.0


# ---------------------------------------------------------------------------
# Eviction profile
# ---------------------------------------------------------------------------


def _make_frame(
    *,
    n_rows: int,
    record_phase: list[str],
    pre_len: list[int],
    post_len: list[int],
    budget: int,
    evict_reason: list[str],
    kept_per_row: list[list[int]],
    record_layer: list[int] | None = None,
    record_step: list[int] | None = None,
    iter_dir: list[str] | None = None,
    task: list[str] | None = None,
    call_idx: list[int] | None = None,
) -> KVEvictionFrame:
    record_layer = record_layer or [0] * n_rows
    record_step = record_step or [-1] * n_rows
    iter_dir = iter_dir or ["/iter_0000"] * n_rows
    task = task or ["t"] * n_rows
    call_idx = call_idx or [0] * n_rows
    kept_per_row_arr = [np.asarray(k, dtype=np.int32) for k in kept_per_row]
    return KVEvictionFrame(
        n_rows=n_rows,
        task=np.asarray(task, dtype=object),
        call_idx=np.asarray(call_idx, dtype=np.int32),
        iter_dir=np.asarray(iter_dir, dtype=object),
        policy_name=np.asarray(["h2o"] * n_rows, dtype="U16"),
        record_step=np.asarray(record_step, dtype=np.int32),
        record_layer=np.asarray(record_layer, dtype=np.int32),
        record_phase=np.asarray(record_phase, dtype="U7"),
        pre_len=np.asarray(pre_len, dtype=np.int32),
        post_len=np.asarray(post_len, dtype=np.int32),
        budget=np.asarray([budget] * n_rows, dtype=np.int32),
        evict_reason=np.asarray(evict_reason, dtype="U16"),
        kept_offsets=np.zeros(n_rows + 1, dtype=np.int64),
        kept_indices=np.empty(0, dtype=np.int32),
        evicted_offsets=np.zeros(n_rows + 1, dtype=np.int64),
        evicted_indices=np.empty(0, dtype=np.int32),
        kept_per_row=kept_per_row_arr,
    )


def test_phase_distribution_counts_prefill_decode():
    frame = _make_frame(
        n_rows=4,
        record_phase=["prefill", "decode", "decode", "prefill"],
        pre_len=[1100, 1100, 1099, 200],
        post_len=[1024, 1023, 1099, 200],  # last row: no evict
        budget=1024,
        evict_reason=["over_budget", "over_budget", "none", "none"],
        kept_per_row=[[0], [0], [0], [0]],
    )
    rows = compute_phase_distribution(frame, run_label="h2o-b1024")
    by_phase = {r["phase"]: r for r in rows}
    assert by_phase["prefill"]["n_decisions"] == 2
    assert by_phase["prefill"]["n_decisions_with_evict"] == 1
    assert by_phase["prefill"]["n_evicted_total"] == 76
    assert by_phase["prefill"]["reasons"] == {"over_budget": 1, "none": 1}
    assert by_phase["decode"]["n_decisions"] == 2
    assert by_phase["decode"]["n_decisions_with_evict"] == 1


def test_eviction_profile_rows_round_trip():
    frame = _make_frame(
        n_rows=2,
        record_phase=["prefill", "decode"],
        pre_len=[2050, 1025],
        post_len=[1024, 1024],
        budget=1024,
        evict_reason=["over_budget", "over_budget"],
        kept_per_row=[[0, 1], [0, 1]],
        record_layer=[5, 7],
    )
    rows = compute_eviction_profile_rows(frame, run_label="b1024")
    assert len(rows) == 2
    assert rows[0]["n_evicted"] == 1026
    assert rows[1]["layer"] == 7


def test_phase_distribution_empty_frame():
    empty = KVEvictionFrame(
        n_rows=0,
        task=np.empty(0, dtype=object),
        call_idx=np.empty(0, dtype=np.int32),
        iter_dir=np.empty(0, dtype=object),
        policy_name=np.empty(0, dtype="U16"),
        record_step=np.empty(0, dtype=np.int32),
        record_layer=np.empty(0, dtype=np.int32),
        record_phase=np.empty(0, dtype="U7"),
        pre_len=np.empty(0, dtype=np.int32),
        post_len=np.empty(0, dtype=np.int32),
        budget=np.empty(0, dtype=np.int32),
        evict_reason=np.empty(0, dtype="U16"),
        kept_offsets=np.zeros(1, dtype=np.int64),
        kept_indices=np.empty(0, dtype=np.int32),
        evicted_offsets=np.zeros(1, dtype=np.int64),
        evicted_indices=np.empty(0, dtype=np.int32),
    )
    assert compute_phase_distribution(empty, run_label="none") == []
    assert compute_eviction_profile_rows(empty, run_label="none") == []


# ---------------------------------------------------------------------------
# Role survival
# ---------------------------------------------------------------------------


def test_role_survival_full_keep_all_roles_one_hundred_percent():
    segments = {
        "/iter_0": {
            "segments": [
                {"role": "system", "token_start": 0, "token_end": 4},
                {"role": "user", "token_start": 4, "token_end": 8},
                {"role": "assistant", "token_start": 8, "token_end": 12},
            ]
        }
    }
    frame = _make_frame(
        n_rows=1,
        record_phase=["prefill"],
        pre_len=[12],
        post_len=[12],
        budget=1024,
        evict_reason=["none"],
        kept_per_row=[list(range(12))],
        iter_dir=["/iter_0"],
    )
    rows = compute_role_survival_rows(
        frame,
        segments,
        role_labels=["system", "user", "assistant_message", "other"],
        run_label="x",
    )
    by_role = {r["role"]: r for r in rows}
    assert by_role["system"]["survival_rate"] == 1.0
    assert by_role["user"]["total_tokens"] == 4
    assert by_role["assistant_message"]["kept_tokens"] == 4


def test_role_survival_partial_keep_only_recent():
    # Same 12-token prompt, evict everything except last 4 (recent window).
    segments = {
        "/iter_0": {
            "segments": [
                {"role": "system", "token_start": 0, "token_end": 4},
                {"role": "tool", "token_start": 4, "token_end": 8},
                {"role": "user", "token_start": 8, "token_end": 12},
            ]
        }
    }
    frame = _make_frame(
        n_rows=1,
        record_phase=["prefill"],
        pre_len=[12],
        post_len=[4],
        budget=4,
        evict_reason=["over_budget"],
        kept_per_row=[[8, 9, 10, 11]],
        iter_dir=["/iter_0"],
    )
    rows = compute_role_survival_rows(
        frame, segments, role_labels=["system", "user", "tool_result", "other"], run_label="x"
    )
    by_role = {r["role"]: r for r in rows}
    assert by_role["system"]["survival_rate"] == 0.0
    assert by_role["tool_result"]["survival_rate"] == 0.0
    assert by_role["user"]["survival_rate"] == 1.0


# ---------------------------------------------------------------------------
# Attention loaders (synthetic .npz)
# ---------------------------------------------------------------------------


def _make_attention_npz(
    path: Path,
    *,
    record_layers: list[int],
    record_phases: list[str],
    query_row_offsets: list[int],
    query_positions: list[int],
    topk_indices: np.ndarray,
    topk_weights: np.ndarray,
    heavy_indices: np.ndarray,
    n_segments: int = 1,
) -> None:
    if topk_indices.ndim != 2 or topk_indices.shape != topk_weights.shape:
        raise ValueError("topk_indices/topk_weights must be aligned rank-2 arrays")
    width = int(topk_indices.shape[1])
    valid = topk_indices >= 0
    counts = valid.sum(axis=1, dtype=np.int64)
    csr_offsets = np.zeros(int(topk_indices.shape[0]) + 1, dtype=np.int64)
    csr_offsets[1:] = np.cumsum(counts, dtype=np.int64)
    csr_indices = topk_indices[valid].astype(np.int32)
    csr_weights = topk_weights[valid].astype(np.float16)
    np.savez_compressed(
        path,
        record_layer=np.asarray(record_layers, dtype=np.int32),
        record_phase=np.asarray(record_phases, dtype="U7"),
        query_row_offsets=np.asarray(query_row_offsets, dtype=np.int64),
        query_positions=np.asarray(query_positions, dtype=np.int64),
        top_k=np.asarray(width, dtype=np.int32),
        topk_csr_offsets=csr_offsets,
        topk_csr_indices=csr_indices,
        topk_csr_weights=csr_weights,
        heavy_indices=heavy_indices.astype(np.int32),
        n_segments=np.asarray(n_segments, dtype=np.int32),
    )


def _make_record(iter_dir: Path) -> IterationRecord:
    return IterationRecord(
        task="synthetic",
        attempt_dir=iter_dir.parent.parent,
        recordings_dir=iter_dir.parent,
        iter_dir=iter_dir,
        call_idx=0,
    )


def test_load_kv_eviction_decodes_evicted_h2o_scores(tmp_path):
    iter_dir = tmp_path / "attempt_1" / "recordings" / "iter_0000"
    iter_dir.mkdir(parents=True)
    np.savez_compressed(
        iter_dir / "kv_eviction.npz",
        call_idx=np.asarray(0, dtype=np.int32),
        policy_name=np.asarray("h2o", dtype="U16"),
        record_step=np.asarray([-1, 0], dtype=np.int32),
        record_layer=np.asarray([0, 0], dtype=np.int32),
        record_phase=np.asarray(["prefill", "decode"], dtype="U7"),
        pre_len=np.asarray([8, 5], dtype=np.int32),
        post_len=np.asarray([4, 4], dtype=np.int32),
        budget=np.asarray([4, 4], dtype=np.int32),
        kept_offsets=np.asarray([0, 4, 8], dtype=np.int64),
        kept_indices=np.asarray([0, 3, 6, 7, 0, 2, 3, 4], dtype=np.int32),
        evicted_offsets=np.asarray([0, 4, 5], dtype=np.int64),
        evicted_indices=np.asarray([1, 2, 4, 5, 1], dtype=np.int32),
        evict_reason=np.asarray(["over_budget", "over_budget"], dtype="U16"),
        score_topk_index=np.asarray([[3, 6], [2, -1]], dtype=np.int32),
        score_topk_value=np.asarray([[0.9, 0.8], [0.7, np.nan]], dtype=np.float32),
        score_evicted_offsets=np.asarray([0, 4, 5], dtype=np.int64),
        score_evicted_index=np.asarray([1, 2, 4, 5, 1], dtype=np.int32),
        score_evicted_value=np.asarray([0.1, 0.2, 0.3, 0.4, 0.05], dtype=np.float32),
    )

    frame = load_kv_eviction([_make_record(iter_dir)])

    assert frame.n_rows == 2
    assert frame.score_evicted_offsets.tolist() == [0, 4, 5]
    assert frame.score_evicted_index.tolist() == [1, 2, 4, 5, 1]
    assert frame.score_evicted_per_row[0].tolist() == [1, 2, 4, 5]
    assert frame.score_evicted_per_row[1].tolist() == [1]
    np.testing.assert_allclose(
        frame.score_evicted_value_per_row[0],
        np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
    )


def test_aggregate_heavy_hitters_unions_per_layer(tmp_path):
    iter_a = tmp_path / "attempt_1" / "recordings" / "iter_0000"
    iter_a.mkdir(parents=True)
    iter_b = tmp_path / "attempt_1" / "recordings" / "iter_0001"
    iter_b.mkdir(parents=True)

    _make_attention_npz(
        iter_a / "attention.npz",
        record_layers=[0, 1],
        record_phases=["prefill", "prefill"],
        query_row_offsets=[0, 1, 2],
        query_positions=[3, 3],
        topk_indices=np.array([[0, 1], [0, 2]]),
        topk_weights=np.array([[0.7, 0.3], [0.6, 0.4]]),
        heavy_indices=np.array([[0, 1], [2, 3]]),
    )
    _make_attention_npz(
        iter_b / "attention.npz",
        record_layers=[0],
        record_phases=["decode"],
        query_row_offsets=[0, 1],
        query_positions=[5],
        topk_indices=np.array([[5, 0]]),
        topk_weights=np.array([[0.9, 0.1]]),
        heavy_indices=np.array([[4, 5]]),
    )

    per_layer = aggregate_heavy_hitters_per_layer([_make_record(iter_a), _make_record(iter_b)])
    assert per_layer[0] == {0, 1, 4, 5}
    assert per_layer[1] == {2, 3}


def test_aggregate_sink_recent_share_classifies_topk_positions(tmp_path):
    # key_len=10, sink=2, recent=3 → sink=[0,1], recent=[7,8,9], middle=[2..6]
    iter_dir = tmp_path / "attempt_1" / "recordings" / "iter_0000"
    iter_dir.mkdir(parents=True)
    _make_attention_npz(
        iter_dir / "attention.npz",
        record_layers=[0],
        record_phases=["prefill"],
        query_row_offsets=[0, 1],
        query_positions=[9],
        topk_indices=np.array([[0, 4, 8]]),  # one each: sink / middle / recent
        topk_weights=np.array([[0.5, 0.2, 0.3]]),
        heavy_indices=np.array([[0]]),
    )
    out = aggregate_sink_recent_share_per_layer([_make_record(iter_dir)], sink=2, recent=3)
    layer = out[0]
    assert layer["sink_share"] == pytest.approx(0.5, abs=1e-3)
    assert layer["middle_share"] == pytest.approx(0.2, abs=1e-3)
    assert layer["recent_share"] == pytest.approx(0.3, abs=1e-3)


def test_aggregate_sink_recent_share_reads_csr_topk(tmp_path):
    iter_dir = tmp_path / "attempt_1" / "recordings" / "iter_0000"
    iter_dir.mkdir(parents=True)
    np.savez_compressed(
        iter_dir / "attention.npz",
        record_layer=np.asarray([0], dtype=np.int32),
        record_phase=np.asarray(["prefill"], dtype="U7"),
        query_row_offsets=np.asarray([0, 1], dtype=np.int64),
        query_positions=np.asarray([9], dtype=np.int64),
        top_k=np.asarray(3, dtype=np.int32),
        topk_csr_offsets=np.asarray([0, 3], dtype=np.int64),
        topk_csr_indices=np.asarray([0, 4, 8], dtype=np.int32),
        topk_csr_weights=np.asarray([0.5, 0.2, 0.3], dtype=np.float32),
        heavy_indices=np.asarray([[0]], dtype=np.int32),
        n_segments=np.asarray(1, dtype=np.int32),
    )

    out = aggregate_sink_recent_share_per_layer([_make_record(iter_dir)], sink=2, recent=3)

    assert out[0]["sink_share"] == pytest.approx(0.5, abs=1e-6)
    assert out[0]["middle_share"] == pytest.approx(0.2, abs=1e-6)
    assert out[0]["recent_share"] == pytest.approx(0.3, abs=1e-6)


def test_aggregate_sink_recent_share_rejects_malformed_topk(tmp_path):
    iter_dir = tmp_path / "attempt_1" / "recordings" / "iter_0000"
    iter_dir.mkdir(parents=True)
    np.savez_compressed(
        iter_dir / "attention.npz",
        record_layer=np.asarray([0], dtype=np.int32),
        record_phase=np.asarray(["prefill"], dtype="U7"),
        query_row_offsets=np.asarray([0, 1], dtype=np.int64),
        query_positions=np.asarray([9], dtype=np.int64),
        top_k=np.asarray(2, dtype=np.int32),
        topk_csr_offsets=np.asarray([0, 2], dtype=np.int64),
        topk_csr_indices=np.asarray([-1, 8], dtype=np.int32),
        topk_csr_weights=np.asarray([0.5, 0.3], dtype=np.float32),
        n_segments=np.asarray(1, dtype=np.int32),
    )

    with pytest.raises(ValueError, match="invalid attention top-k schema"):
        aggregate_sink_recent_share_per_layer([_make_record(iter_dir)], sink=2, recent=3)


# ---------------------------------------------------------------------------
# JS / Jaccard pairings
# ---------------------------------------------------------------------------


def test_attention_js_per_layer_and_heavy_jaccard():
    per_run = {
        "baseline": {0: np.array([0.5, 0.5]), 1: np.array([0.9, 0.1])},
        "variant": {0: np.array([0.5, 0.5]), 1: np.array([0.1, 0.9])},
    }
    js_rows = compute_attention_js_per_layer(per_run, baseline_label="baseline")
    by_layer = {r["layer"]: r for r in js_rows}
    assert by_layer[0]["js"] == pytest.approx(0.0, abs=1e-9)
    assert by_layer[1]["js"] > 0.3

    heavy = {
        "baseline": {0: {1, 2, 3}, 1: {1, 2, 3}},
        "variant": {0: {1, 2, 3}, 1: {4, 5, 6}},
    }
    jc = compute_heavy_jaccard_per_layer(heavy, baseline_label="baseline")
    by_layer_j = {r["layer"]: r for r in jc}
    assert by_layer_j[0]["jaccard"] == 1.0
    assert by_layer_j[1]["jaccard"] == 0.0


# ---------------------------------------------------------------------------
# segments preload
# ---------------------------------------------------------------------------


def test_load_segments_by_iter_dir(tmp_path):
    iter_dir = tmp_path / "iter_0000"
    iter_dir.mkdir(parents=True)
    payload = {"segments": [{"role": "user", "token_start": 0, "token_end": 5}]}
    (iter_dir / "segments.json").write_text(json.dumps(payload), encoding="utf-8")
    record = IterationRecord(
        task="t",
        attempt_dir=tmp_path,
        recordings_dir=tmp_path,
        iter_dir=iter_dir,
        call_idx=0,
    )
    out = load_segments_by_iter_dir([record])
    assert str(iter_dir) in out
    assert out[str(iter_dir)]["segments"][0]["role"] == "user"


def test_has_recording_files_requires_completion_sentinel(tmp_path):
    iter_dir = tmp_path / "iter_0000"
    iter_dir.mkdir(parents=True)
    (iter_dir / "attention.npz").write_bytes(b"not-used")
    (iter_dir / "routing.npz").write_bytes(b"not-used")
    (iter_dir / "segments.json").write_text(
        json.dumps({"complete": True}),
        encoding="utf-8",
    )

    assert not _has_recording_files(iter_dir)

    (iter_dir / ".done").write_text("complete\n", encoding="utf-8")
    assert _has_recording_files(iter_dir)

    (iter_dir / "segments.json").write_text(
        json.dumps({"complete": False}),
        encoding="utf-8",
    )
    assert not _has_recording_files(iter_dir)
