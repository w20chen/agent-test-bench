from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from scripts.recoding_figures.metrics import pairwise_js
from scripts.recoding_figures.followup_metrics import (
    alpha_blend_summary,
    context_role_cache_summary,
    decode_residual_closure_summary,
    distribution_component_leaderboard,
    load_attention_decode_step_distributions,
    load_attention_context_group_distributions,
    load_attention_segment_recency_distributions,
    residual_explanation_leaderboard,
    sliding_window_detection_summary,
)


@dataclass(frozen=True)
class _Record:
    task: str
    call_idx: int


def _dataset(records: list[_Record], rows: np.ndarray, *, labels: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        modality="unknown",
        records=records,
        layers=[0],
        axis_labels=labels,
        distributions={0: rows.astype(np.float64)},
        observation_counts={0: np.ones(rows.shape[0], dtype=np.float64)},
    )


def test_sliding_window_detector_hits_task_splice() -> None:
    records = [_Record("a", idx) for idx in range(4)] + [
        _Record("b", idx) for idx in range(4, 8)
    ]
    rows = np.asarray(
        [
            [0.9, 0.1],
            [0.88, 0.12],
            [0.91, 0.09],
            [0.87, 0.13],
            [0.1, 0.9],
            [0.12, 0.88],
            [0.09, 0.91],
            [0.11, 0.89],
        ],
        dtype=np.float64,
    )
    summary = sliding_window_detection_summary(
        _dataset(records, rows, labels=["x", "y"]),
        windows=(2,),
        tolerance=1,
    )

    row = summary["rows"][0]
    assert row["rank_detection_rate"] == 1.0
    assert 4 in row["rank_alerts"]


def test_context_role_cache_can_beat_layer_static() -> None:
    records = [_Record("a", 0), _Record("a", 1)]
    moe_rows = np.asarray(
        [
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.9],
        ],
        dtype=np.float64,
    )
    attention_rows = np.asarray(
        [
            [0.95, 0.05],
            [0.05, 0.95],
        ],
        dtype=np.float64,
    )
    moe = _dataset(records, moe_rows, labels=["0", "1", "2", "3"])
    attention = _dataset(records, attention_rows, labels=["system", "generation"])

    summary = context_role_cache_summary(moe, attention, ks=(1,))
    row = summary["rows"][0]

    assert row["dominant_context_role_coverage"] > row["layer_static_coverage"]
    assert row["attention_mixture_role_coverage"] > row["layer_static_coverage"]


def test_context_group_loader_assigns_recent_gen_boundary(tmp_path) -> None:
    iter_dir = tmp_path / "iter_0000"
    iter_dir.mkdir()
    segments = [
        {"role": "system", "token_start": 0, "token_end": 1},
        {"role": "tool_result", "token_start": 10, "token_end": 11},
        {"role": "generation", "token_start": 44, "token_end": 45},
        {"role": "generation", "token_start": 43, "token_end": 44},
        {"role": "user", "token_start": 1, "token_end": 301},
    ]
    token_segment_id = [4] * 301
    token_segment_id[0] = 0
    token_segment_id[10] = 1
    token_segment_id[44] = 2
    token_segment_id[43] = 3
    (iter_dir / "segments.json").write_text(
        json.dumps({"segments": segments, "token_segment_id": token_segment_id})
    )
    np.savez(
        iter_dir / "attention.npz",
        record_layer=np.asarray([0], dtype=np.int32),
        record_phase=np.asarray(["decode"]),
        query_row_offsets=np.asarray([0, 1], dtype=np.int64),
        query_positions=np.asarray([300], dtype=np.int64),
        top_k=np.asarray(4, dtype=np.int32),
        topk_csr_offsets=np.asarray([0, 4], dtype=np.int64),
        topk_csr_indices=np.asarray([0, 10, 44, 43], dtype=np.int32),
        topk_csr_weights=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float16),
    )
    record = SimpleNamespace(iter_dir=iter_dir)

    dataset = load_attention_context_group_distributions(
        [record],
        role_labels=["system", "tool_result", "generation", "user"],
        recent_token_window=256,
    )

    np.testing.assert_allclose(
        dataset.distributions[0][0],
        np.asarray([0.1, 0.2, 0.3, 0.4]),
        atol=1e-3,
    )
    assert dataset.observation_counts[0][0] == 1.0


def test_context_group_loader_marks_zero_mass_as_missing(tmp_path) -> None:
    iter_dir = tmp_path / "iter_0000"
    iter_dir.mkdir()
    (iter_dir / "segments.json").write_text(
        json.dumps(
            {
                "segments": [{"role": "generation", "token_start": 0, "token_end": 2}],
                "token_segment_id": [0, 0],
            }
        )
    )
    np.savez(
        iter_dir / "attention.npz",
        record_layer=np.asarray([0], dtype=np.int32),
        record_phase=np.asarray(["decode"]),
        query_row_offsets=np.asarray([0, 1], dtype=np.int64),
        query_positions=np.asarray([1], dtype=np.int64),
        top_k=np.asarray(1, dtype=np.int32),
        topk_csr_offsets=np.asarray([0, 1], dtype=np.int64),
        topk_csr_indices=np.asarray([0], dtype=np.int32),
        topk_csr_weights=np.asarray([0.0], dtype=np.float16),
    )
    record = SimpleNamespace(iter_dir=iter_dir)

    dataset = load_attention_context_group_distributions(
        [record],
        role_labels=["generation"],
    )

    np.testing.assert_allclose(dataset.distributions[0][0], np.zeros(4))
    assert dataset.observation_counts[0][0] == 0.0


def test_decode_step_loader_uses_generation_token_offsets(tmp_path) -> None:
    iter_dir = tmp_path / "iter_0000"
    iter_dir.mkdir()
    segments = [
        {"role": "user", "token_start": 0, "token_end": 3},
        {"role": "generation", "token_start": 3, "token_end": 8},
    ]
    (iter_dir / "segments.json").write_text(
        json.dumps(
            {
                "segments": segments,
                "token_segment_id": [0, 0, 0, 1, 1, 1, 1, 1],
            }
        )
    )
    np.savez(
        iter_dir / "attention.npz",
        record_layer=np.asarray([0], dtype=np.int32),
        record_phase=np.asarray(["decode"]),
        query_row_offsets=np.asarray([0, 4], dtype=np.int64),
        query_positions=np.asarray([3, 4, 7, 2], dtype=np.int64),
        segment_mass=np.asarray(
            [
                [0.5, 0.5],
                [0.4, 0.6],
                [0.3, 0.7],
                [0.9, 0.1],
            ],
            dtype=np.float64,
        ),
    )
    record = SimpleNamespace(iter_dir=iter_dir)

    dataset = load_attention_decode_step_distributions(
        [record],
        role_labels=["user", "generation"],
    )

    np.testing.assert_allclose(
        dataset.distributions[0][0],
        np.asarray([1 / 3, 2 / 3, 0.0, 0.0, 0.0]),
    )
    assert dataset.observation_counts[0][0] == 4.0


def test_segment_recency_loader_splits_latest_and_earlier(tmp_path) -> None:
    iter_dir = tmp_path / "iter_0000"
    iter_dir.mkdir()
    segments = [
        {"role": "user", "token_start": 0, "token_end": 1},
        {"role": "user", "token_start": 1, "token_end": 2},
        {"role": "tool", "token_start": 2, "token_end": 3},
        {"role": "generation", "token_start": 3, "token_end": 4},
    ]
    (iter_dir / "segments.json").write_text(
        json.dumps({"segments": segments, "token_segment_id": [0, 1, 2, 3]})
    )
    np.savez(
        iter_dir / "attention.npz",
        record_layer=np.asarray([0], dtype=np.int32),
        record_phase=np.asarray(["decode"]),
        query_row_offsets=np.asarray([0, 1], dtype=np.int64),
        query_positions=np.asarray([3], dtype=np.int64),
        segment_mass=np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float64),
    )
    record = SimpleNamespace(iter_dir=iter_dir)

    dataset = load_attention_segment_recency_distributions(
        [record],
        role_labels=["user", "tool_result", "generation"],
    )
    values = dict(zip(dataset.axis_labels, dataset.distributions[0][0], strict=True))

    assert values["earlier_user"] == 0.1
    assert values["latest_user"] == 0.2
    assert values["latest_tool_result"] == 0.3
    assert values["generation"] == 0.4


def test_alpha_blend_interpolates_static_and_dynamic() -> None:
    records = [_Record("a", 0), _Record("a", 1), _Record("a", 2)]
    rows = np.asarray(
        [
            [0.9, 0.1],
            [0.88, 0.12],
            [0.87, 0.13],
        ],
        dtype=np.float64,
    )
    summary = alpha_blend_summary(
        _dataset(records, rows, labels=["0", "1"]),
        alphas=(0.0, 1.0),
        ks=(1,),
    )

    by_alpha = {row["alpha"]: row for row in summary["rows"]}
    assert by_alpha[0.0]["same_task_coverage"] >= 0.87
    assert by_alpha[1.0]["same_task_coverage"] >= 0.87


def test_decode_residual_closure_reports_position_and_query_role() -> None:
    records = [_Record("a", idx) for idx in range(4)]
    attention = _dataset(
        records,
        np.asarray(
            [
                [0.9, 0.1],
                [0.7, 0.3],
                [0.3, 0.7],
                [0.1, 0.9],
            ],
            dtype=np.float64,
        ),
        labels=["system", "generation"],
    )
    key_role = _dataset(
        records,
        np.asarray([[0.5, 0.5]] * 4, dtype=np.float64),
        labels=["system", "generation"],
    )
    distance = _dataset(
        records,
        np.asarray(
            [
                [0.9, 0.1],
                [0.7, 0.3],
                [0.3, 0.7],
                [0.1, 0.9],
            ],
            dtype=np.float64,
        ),
        labels=["near", "far"],
    )
    query_role = _dataset(
        records,
        np.asarray([[0.0, 1.0]] * 4, dtype=np.float64),
        labels=["system", "generation"],
    )

    summary = decode_residual_closure_summary(
        attention,
        key_role,
        distance,
        query_role,
        max_lag=2,
    )

    assert summary["distance_decay"]["mean_corr_abs_residual_vs_distance_js"] > 0.0
    assert summary["query_token_semantic_type"]["dominant_role"] == "generation"
    assert summary["query_token_semantic_type"]["lexical_token_semantics_available"] is False


def test_residual_explanation_leaderboard_ranks_matching_feature() -> None:
    records = [_Record("a", idx) for idx in range(5)]
    attention = _dataset(
        records,
        np.asarray(
            [
                [0.95, 0.05],
                [0.90, 0.10],
                [0.50, 0.50],
                [0.10, 0.90],
                [0.05, 0.95],
            ],
            dtype=np.float64,
        ),
        labels=["system", "generation"],
    )
    key_role = _dataset(
        records,
        np.asarray([[0.5, 0.5]] * 5, dtype=np.float64),
        labels=["system", "generation"],
    )
    matching_feature = _dataset(
        records,
        np.asarray(
            [
                [0.95, 0.05],
                [0.90, 0.10],
                [0.50, 0.50],
                [0.10, 0.90],
                [0.05, 0.95],
            ],
            dtype=np.float64,
        ),
        labels=["x", "y"],
    )

    summary = residual_explanation_leaderboard(
        attention,
        key_role,
        distribution_features={"matching_distribution": matching_feature},
        scalar_features={"noise": np.asarray([0.0, 1.0, 0.0, 1.0, 0.0])},
        pair_features={
            "matching_residual": _expected_abs_constant_control_residual(
                attention.distributions[0]
            )
        },
    )

    by_feature = {row["feature"]: row for row in summary["rows"]}
    assert by_feature["pair:matching_residual"]["mean_abs_corr"] > 0.99


def test_distribution_component_leaderboard_scores_component() -> None:
    records = [_Record("a", idx) for idx in range(5)]
    attention = _dataset(
        records,
        np.asarray(
            [
                [0.95, 0.05],
                [0.90, 0.10],
                [0.50, 0.50],
                [0.10, 0.90],
                [0.05, 0.95],
            ],
            dtype=np.float64,
        ),
        labels=["system", "generation"],
    )
    key_role = _dataset(
        records,
        np.asarray([[0.5, 0.5]] * 5, dtype=np.float64),
        labels=["system", "generation"],
    )
    feature = _dataset(
        records,
        np.asarray(
            [
                [0.95, 0.05],
                [0.90, 0.10],
                [0.50, 0.50],
                [0.10, 0.90],
                [0.05, 0.95],
            ],
            dtype=np.float64,
        ),
        labels=["left", "right"],
    )

    summary = distribution_component_leaderboard(attention, key_role, feature)

    assert summary["rows"][0]["feature"].startswith("component:unknown:left")
    assert "layer_rows" not in summary["rows"][0]


def _expected_abs_constant_control_residual(rows: np.ndarray) -> np.ndarray:
    js = pairwise_js(rows)
    upper = np.triu_indices_from(js, k=1)
    residual = js - float(np.mean(js[upper]))
    return np.abs(residual)
