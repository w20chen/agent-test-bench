"""Unit tests for `scripts/recoding_figures/score_sparse_selection.py`.

Builds tiny hand-crafted attention.npz + sparse_attention.npz + meta.json
trees under tmp_path, then runs the scoring logic and asserts the numbers
match closed-form values.

Constructs three concrete cases:
- **Perfect coverage**: every true top-K key falls in sliding keep set →
  recall@k==1.0, mass_in_keep_set == sum(top-k weights).
- **Zero coverage**: every true top-K key falls in the masked middle band →
  recall@k==0.0, mass_in_keep_set==0.
- **Partial coverage**: ~half top-K in keep set, ~half outside; mass and
  recall computed analytically.

Also tests the row-alignment sanity check: a sparse_attention.npz row
missing for a (layer, phase, decode_step) that attention.npz has must raise
a clear ValueError, not silently produce wrong scores.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.recoding_figures.score_sparse_selection import (  # noqa: E402
    SparseParams,
    reconstruct_keep_set,
    score_attempts,
)


def _write_attention_npz(
    iter_dir: Path,
    *,
    records: list[dict],
) -> None:
    """records: each {layer, phase, decode_step, queries: [{position, head, topk_indices, topk_weights}]}."""
    record_layer = np.array([r["layer"] for r in records], dtype=np.int32)
    record_phase = np.array([r["phase"] for r in records], dtype="U7")
    record_decode_step = np.array(
        [r["decode_step"] for r in records], dtype=np.int32
    )
    q_offsets = [0]
    csr_offsets = [0]
    csr_indices: list[int] = []
    csr_weights: list[float] = []
    query_positions: list[int] = []
    query_heads: list[int] = []
    for r in records:
        for q in r["queries"]:
            query_positions.append(q["position"])
            query_heads.append(q["head"])
            csr_indices.extend(int(x) for x in q["topk_indices"])
            csr_weights.extend(float(x) for x in q["topk_weights"])
            csr_offsets.append(len(csr_indices))
        q_offsets.append(q_offsets[-1] + len(r["queries"]))

    payload = {
        "call_idx": np.int32(0),
        "record_layer": record_layer,
        "record_phase": record_phase,
        "record_decode_step": record_decode_step,
        "query_row_offsets": np.array(q_offsets, dtype=np.int64),
        "topk_csr_offsets": np.array(csr_offsets, dtype=np.int64),
        "topk_csr_indices": np.array(csr_indices, dtype=np.int32),
        "topk_csr_weights": np.array(csr_weights, dtype=np.float16),
        "query_positions": np.array(query_positions, dtype=np.int32),
        "query_heads": np.array(query_heads, dtype=np.int32),
        "top_k": np.int32(8),
    }
    np.savez_compressed(iter_dir / "attention.npz", **payload)


def _write_sparse_npz(
    iter_dir: Path,
    *,
    records: list[dict],
    method: str = "sliding",
) -> None:
    """records: each {layer, phase, decode_step, query_len, key_len, kept_count}."""
    n = len(records)
    payload = {
        "call_idx": np.int32(0),
        "method_name": np.array(method, dtype="U16"),
        "record_step": np.arange(n, dtype=np.int32),
        "record_layer": np.array([r["layer"] for r in records], dtype=np.int32),
        "record_phase": np.array([r["phase"] for r in records], dtype="U7"),
        "record_decode_step": np.array(
            [r["decode_step"] for r in records], dtype=np.int32
        ),
        "query_len": np.array([r["query_len"] for r in records], dtype=np.int32),
        "key_len": np.array([r["key_len"] for r in records], dtype=np.int32),
        "kept_count": np.array([r["kept_count"] for r in records], dtype=np.int32),
        "density": np.array(
            [r["kept_count"] / max(r["key_len"], 1) for r in records],
            dtype=np.float16,
        ),
        "extras_json": np.array(
            [json.dumps(r.get("extras", {}), sort_keys=True) for r in records],
            dtype=object,
        ),
    }
    np.savez_compressed(iter_dir / "sparse_attention.npz", **payload)


def _write_meta_and_segments(
    attempt_dir: Path,
    *,
    sink_size: int,
    recent_window: int,
    iter_dir: Path,
    method: str = "sliding",
) -> None:
    recordings_dir = attempt_dir / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": {"name": "toy"},
        "sparse_attention": {
            "method": method,
            "sink_size": sink_size,
            "recent_window": recent_window,
            "record": True,
            "observe_only": True,
        },
        "iters": [{"dir": iter_dir.name, "call_idx": 0, "complete": True}],
    }
    (recordings_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    segments = {
        "call_idx": 0,
        "input_tokens": 8,
        "output_tokens": 0,
        "total_tokens": 8,
        "complete": True,
        "segments": [
            {
                "role": "user",
                "token_start": 0,
                "token_end": 8,
                "message_index": 0,
                "tool_call_id": None,
                "name": None,
                "has_content": True,
                "has_tool_calls": False,
                "first_seen_call": 0,
            }
        ],
        "token_segment_id": [0] * 8,
    }
    (iter_dir / "segments.json").write_text(json.dumps(segments), encoding="utf-8")
    (iter_dir / ".done").write_text("", encoding="utf-8")


def _make_attempt(
    tmp_path: Path,
    *,
    name: str,
    sink_size: int,
    recent_window: int,
    attention_records: list[dict],
    sparse_records: list[dict],
    method: str = "sliding",
) -> Path:
    attempt_dir = tmp_path / name
    iter_dir = attempt_dir / "recordings" / "iter_0000"
    iter_dir.mkdir(parents=True, exist_ok=True)
    _write_meta_and_segments(
        attempt_dir,
        sink_size=sink_size,
        recent_window=recent_window,
        iter_dir=iter_dir,
        method=method,
    )
    _write_attention_npz(iter_dir, records=attention_records)
    _write_sparse_npz(iter_dir, records=sparse_records, method=method)
    # Empty routing.npz: loader's `_has_recording_files` gate requires the
    # file to exist for every iter, even when the model isn't MoE.
    np.savez_compressed(iter_dir / "routing.npz", call_idx=np.int32(0))
    return attempt_dir


# ---------------------------------------------------------------------------
# reconstruct_keep_set: closed-form sanity
# ---------------------------------------------------------------------------


def test_reconstruct_keep_set_sliding_basic() -> None:
    params = SparseParams(sink_size=2, recent_window=3)
    keep = reconstruct_keep_set(method_name="sliding", method_params=params, key_len=8)
    # sink {0,1} ∪ recent {5,6,7}
    assert set(keep.tolist()) == {0, 1, 5, 6, 7}


def test_reconstruct_keep_set_full_when_window_exceeds_key_len() -> None:
    params = SparseParams(sink_size=4, recent_window=10)
    keep = reconstruct_keep_set(method_name="sliding", method_params=params, key_len=8)
    assert set(keep.tolist()) == set(range(8))


def test_reconstruct_keep_set_rejects_unknown_method() -> None:
    params = SparseParams(sink_size=2, recent_window=2)
    with pytest.raises(NotImplementedError):
        reconstruct_keep_set(method_name="unknown", method_params=params, key_len=8)


def test_reconstruct_keep_set_dynamic_uses_selected_middle_indices() -> None:
    params = SparseParams(sink_size=1, recent_window=2)
    keep = reconstruct_keep_set(
        method_name="quest",
        method_params=params,
        key_len=8,
        extras={"selected_middle_indices": [3, 4]},
    )
    assert set(keep.tolist()) == {0, 3, 4, 6, 7}


def test_reconstruct_keep_set_dynamic_prefill_dense_reason() -> None:
    params = SparseParams(sink_size=1, recent_window=1)
    keep = reconstruct_keep_set(
        method_name="quest",
        method_params=params,
        key_len=8,
        extras={"selection_reason": "phase_dense", "phase_scope": "decode_only"},
    )
    assert set(keep.tolist()) == set(range(8))


# ---------------------------------------------------------------------------
# End-to-end scoring against hand-built npz: perfect / zero / partial
# ---------------------------------------------------------------------------


def test_scoring_perfect_coverage(tmp_path: Path) -> None:
    """Sliding(sink=2, recent=3, key_len=8) keeps {0,1,5,6,7}; top-K all in keep."""
    attention_records = [
        {
            "layer": 0,
            "phase": "prefill",
            "decode_step": -1,
            "queries": [
                {
                    "position": 7,
                    "head": 0,
                    "topk_indices": [7, 6, 1, 0, 5],
                    "topk_weights": [0.4, 0.3, 0.15, 0.1, 0.05],
                },
            ],
        }
    ]
    sparse_records = [
        {"layer": 0, "phase": "prefill", "decode_step": -1,
         "query_len": 8, "key_len": 8, "kept_count": 5},
    ]
    attempt = _make_attempt(
        tmp_path, name="perfect", sink_size=2, recent_window=3,
        attention_records=attention_records, sparse_records=sparse_records,
    )
    df = score_attempts(attempt_dirs=[attempt], recall_ks=(8,))
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["mass_in_keep_set"] == pytest.approx(1.0, abs=1e-3)
    assert row["mass_in_topk"] == pytest.approx(1.0, abs=1e-3)
    assert row["recall_at_8"] == pytest.approx(1.0)


def test_scoring_zero_coverage(tmp_path: Path) -> None:
    """Top-K all live in masked middle band {2,3,4}; recall and mass = 0."""
    attention_records = [
        {
            "layer": 0,
            "phase": "prefill",
            "decode_step": -1,
            "queries": [
                {
                    "position": 7,
                    "head": 0,
                    "topk_indices": [2, 3, 4],
                    "topk_weights": [0.5, 0.3, 0.2],
                },
            ],
        }
    ]
    sparse_records = [
        {"layer": 0, "phase": "prefill", "decode_step": -1,
         "query_len": 8, "key_len": 8, "kept_count": 5},
    ]
    attempt = _make_attempt(
        tmp_path, name="zero", sink_size=2, recent_window=3,
        attention_records=attention_records, sparse_records=sparse_records,
    )
    df = score_attempts(attempt_dirs=[attempt], recall_ks=(8,))
    row = df.row(0, named=True)
    assert row["mass_in_keep_set"] == pytest.approx(0.0)
    assert row["recall_at_8"] == pytest.approx(0.0)
    assert row["n_topk"] == 3
    assert row["keep_set_size"] == 5


def test_scoring_partial_coverage_recall_at_k(tmp_path: Path) -> None:
    """Top-K=4 with 2 in keep {0,7}, 2 out {3,4}. recall@2 = 0.5, recall@4 = 0.5."""
    attention_records = [
        {
            "layer": 0,
            "phase": "prefill",
            "decode_step": -1,
            "queries": [
                {
                    "position": 7,
                    "head": 0,
                    "topk_indices": [7, 3, 4, 0],  # sorted DESC by weight
                    "topk_weights": [0.4, 0.3, 0.2, 0.1],
                },
            ],
        }
    ]
    sparse_records = [
        {"layer": 0, "phase": "prefill", "decode_step": -1,
         "query_len": 8, "key_len": 8, "kept_count": 5},
    ]
    attempt = _make_attempt(
        tmp_path, name="partial", sink_size=2, recent_window=3,
        attention_records=attention_records, sparse_records=sparse_records,
    )
    df = score_attempts(attempt_dirs=[attempt], recall_ks=(2, 4))
    row = df.row(0, named=True)
    # In keep: index 7 (w=0.4), index 0 (w=0.1). Sum = 0.5.
    assert row["mass_in_keep_set"] == pytest.approx(0.5, abs=1e-3)
    # recall@2: of first 2 topk indices [7, 3], 1 in keep -> 0.5
    assert row["recall_at_2"] == pytest.approx(0.5)
    # recall@4: of first 4 [7,3,4,0], 2 in keep -> 0.5
    assert row["recall_at_4"] == pytest.approx(0.5)


def test_scoring_dynamic_prefill_dense_keeps_all_topk(tmp_path: Path) -> None:
    """Decode-only dynamic methods should score prefill dense fallback as dense."""
    attention_records = [
        {
            "layer": 0,
            "phase": "prefill",
            "decode_step": -1,
            "queries": [
                {
                    "position": 7,
                    "head": 0,
                    "topk_indices": [2, 3, 4],
                    "topk_weights": [0.5, 0.3, 0.2],
                },
            ],
        }
    ]
    sparse_records = [
        {
            "layer": 0,
            "phase": "prefill",
            "decode_step": -1,
            "query_len": 8,
            "key_len": 8,
            "kept_count": 8,
            "extras": {"selection_reason": "phase_dense", "phase_scope": "decode_only"},
        },
    ]
    attempt = _make_attempt(
        tmp_path,
        name="dynamic_prefill_dense",
        sink_size=1,
        recent_window=1,
        attention_records=attention_records,
        sparse_records=sparse_records,
        method="quest",
    )
    df = score_attempts(attempt_dirs=[attempt], recall_ks=(8,))
    row = df.row(0, named=True)
    assert row["mass_in_keep_set"] == pytest.approx(1.0, abs=1e-3)
    assert row["recall_at_8"] == pytest.approx(1.0)
    assert row["keep_set_size"] == 8


def test_scoring_multi_layer_multi_query(tmp_path: Path) -> None:
    """Two layers, two queries each: verify per-(layer,phase) grouping shape."""
    attention_records = [
        {
            "layer": 0, "phase": "prefill", "decode_step": -1,
            "queries": [
                {"position": 7, "head": 0, "topk_indices": [0, 7], "topk_weights": [0.6, 0.4]},
                {"position": 6, "head": 1, "topk_indices": [3, 6], "topk_weights": [0.5, 0.5]},
            ],
        },
        {
            "layer": 1, "phase": "decode", "decode_step": 0,
            "queries": [
                {"position": 8, "head": 0, "topk_indices": [7, 0], "topk_weights": [0.7, 0.3]},
            ],
        },
    ]
    sparse_records = [
        {"layer": 0, "phase": "prefill", "decode_step": -1,
         "query_len": 8, "key_len": 8, "kept_count": 5},
        {"layer": 1, "phase": "decode", "decode_step": 0,
         "query_len": 1, "key_len": 9, "kept_count": 5},
    ]
    attempt = _make_attempt(
        tmp_path, name="multi", sink_size=2, recent_window=3,
        attention_records=attention_records, sparse_records=sparse_records,
    )
    df = score_attempts(attempt_dirs=[attempt], recall_ks=(2,))
    assert df.height == 3
    # Layer 0, query 0: both indices [0,7] in keep {0,1,5,6,7} -> mass=1.0
    # Layer 0, query 1: indices [3,6]; 3 not in keep, 6 in keep -> mass=0.5
    # Layer 1, query 0: key_len=9, keep={0,1,6,7,8}; indices [7,0] both in -> mass=1.0
    rows = df.sort(["layer", "query_idx"]).to_dicts()
    assert rows[0]["mass_in_keep_set"] == pytest.approx(1.0, abs=1e-3)
    assert rows[1]["mass_in_keep_set"] == pytest.approx(0.5, abs=1e-3)
    assert rows[2]["mass_in_keep_set"] == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Row alignment sanity check
# ---------------------------------------------------------------------------


def test_scoring_raises_on_missing_sparse_row(tmp_path: Path) -> None:
    """attention.npz has (layer=1,...) but sparse_attention.npz only has layer=0 → ValueError."""
    attention_records = [
        {
            "layer": 0, "phase": "prefill", "decode_step": -1,
            "queries": [{"position": 7, "head": 0, "topk_indices": [0], "topk_weights": [1.0]}],
        },
        {
            "layer": 1, "phase": "prefill", "decode_step": -1,
            "queries": [{"position": 7, "head": 0, "topk_indices": [0], "topk_weights": [1.0]}],
        },
    ]
    sparse_records = [
        {"layer": 0, "phase": "prefill", "decode_step": -1,
         "query_len": 8, "key_len": 8, "kept_count": 5},
        # layer 1 intentionally missing
    ]
    attempt = _make_attempt(
        tmp_path, name="misaligned", sink_size=2, recent_window=3,
        attention_records=attention_records, sparse_records=sparse_records,
    )
    with pytest.raises(ValueError, match="no matching row in sparse_attention.npz"):
        score_attempts(attempt_dirs=[attempt], recall_ks=(8,))


def test_scoring_raises_on_kept_count_drift(tmp_path: Path) -> None:
    """If runtime-recorded kept_count disagrees with reconstruct_keep_set, refuse to score.

    Simulates meta.json sparse_attention block diverging from the actual
    method config used at runtime (e.g. somebody hand-edited meta.json, or
    the method changed sink_size between record and meta write). Scoring
    must raise rather than silently produce wrong mass/recall numbers.
    """
    attention_records = [
        {
            "layer": 0, "phase": "prefill", "decode_step": -1,
            "queries": [{"position": 7, "head": 0, "topk_indices": [0], "topk_weights": [1.0]}],
        },
    ]
    # meta.json (via _make_attempt) says sink=2, recent=3 -> reconstructed
    # keep_set has size 5 for key_len=8. We forge kept_count=99 in the npz
    # to force the disagreement.
    sparse_records = [
        {"layer": 0, "phase": "prefill", "decode_step": -1,
         "query_len": 8, "key_len": 8, "kept_count": 99},
    ]
    attempt = _make_attempt(
        tmp_path, name="kept_count_drift", sink_size=2, recent_window=3,
        attention_records=attention_records, sparse_records=sparse_records,
    )
    with pytest.raises(ValueError, match="reconstructed keep_set size"):
        score_attempts(attempt_dirs=[attempt], recall_ks=(8,))
