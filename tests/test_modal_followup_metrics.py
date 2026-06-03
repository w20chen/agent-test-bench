from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np

from scripts.recoding_figures.modal_followup_metrics import (
    bootstrap_mean_ci,
    fit_log_linear_half_life,
    layer_hotset_jaccard_summary,
    name_head_cluster,
    summarize_correlation_heterogeneity,
    tool_result_segment_ages,
)
from scripts.recoding_figures.recording_loader import LayerDistributionSet


def test_tool_result_segment_ages_use_first_seen_message_index(tmp_path) -> None:
    records = []
    for call_idx in range(3):
        iter_dir = tmp_path / f"iter_{call_idx:04d}"
        iter_dir.mkdir()
        segments = [{"role": "system", "message_index": 0}]
        if call_idx >= 1:
            segments.append({"role": "tool_result", "message_index": 4})
        (iter_dir / "segments.json").write_text(json.dumps({"segments": segments}))
        records.append(
            SimpleNamespace(
                attempt_dir=tmp_path / "attempt_1",
                task="task-a",
                call_idx=call_idx,
                iter_dir=iter_dir,
            )
        )

    ages, diagnostics = tool_result_segment_ages(records)

    assert ages[0] == [None]
    assert ages[1] == [None, 0]
    assert ages[2] == [None, 1]
    assert diagnostics["message_index_coverage"] == 1.0


def test_tool_result_segment_ages_are_attempt_scoped(tmp_path) -> None:
    records = []
    for attempt_name, call_idx in [("attempt_a", 2), ("attempt_b", 7)]:
        iter_dir = tmp_path / attempt_name / f"iter_{call_idx:04d}"
        iter_dir.mkdir(parents=True)
        (iter_dir / "segments.json").write_text(
            json.dumps(
                {
                    "segments": [
                        {"role": "system", "message_index": 0},
                        {"role": "tool_result", "message_index": 4},
                    ]
                }
            )
        )
        records.append(
            SimpleNamespace(
                attempt_dir=tmp_path / attempt_name,
                task="task-a",
                call_idx=call_idx,
                iter_dir=iter_dir,
            )
        )

    ages, _diagnostics = tool_result_segment_ages(records)

    assert ages[0] == [None, 0]
    assert ages[1] == [None, 0]


def test_fit_log_linear_half_life_recovers_two_iter_decay() -> None:
    rows = [
        {"age_in_iters": age, "mean_attention_mass": 0.8 * 0.5 ** (age / 2), "query_rows": 10}
        for age in range(6)
    ]

    fit = fit_log_linear_half_life(rows)

    assert fit["half_life_iters"] is not None
    assert math.isclose(float(fit["half_life_iters"]), 2.0, rel_tol=1e-6)
    assert float(fit["n_fit_points"]) == 6.0


def test_layer_hotset_jaccard_summary_computes_adjacent_structure() -> None:
    records = [SimpleNamespace(task="task-a")]
    dataset = LayerDistributionSet(
        modality="moe",
        records=records,  # type: ignore[arg-type]
        layers=[0, 1, 2],
        axis_labels=["0", "1", "2", "3"],
        distributions={
            0: np.asarray([[0.9, 0.8, 0.1, 0.0]], dtype=np.float64),
            1: np.asarray([[0.7, 0.6, 0.5, 0.0]], dtype=np.float64),
            2: np.asarray([[0.0, 0.1, 0.8, 0.9]], dtype=np.float64),
        },
        observation_counts={layer: np.asarray([1.0]) for layer in [0, 1, 2]},
    )

    summary = layer_hotset_jaccard_summary(dataset, ks=(2,))
    row = summary["rows"][0]

    assert row["k"] == 2.0
    assert math.isclose(row["mean_pairwise_jaccard"], 1 / 3)
    assert math.isclose(row["adjacent_layer_jaccard"], 0.5)
    assert summary["matrices"]["2"]["matrix"][0][0] == 1.0


def test_layer_hotset_jaccard_summary_deduplicates_effective_k() -> None:
    records = [SimpleNamespace(task="task-a")]
    dataset = LayerDistributionSet(
        modality="moe",
        records=records,  # type: ignore[arg-type]
        layers=[0],
        axis_labels=["0", "1", "2", "3"],
        distributions={0: np.asarray([[0.4, 0.3, 0.2, 0.1]], dtype=np.float64)},
        observation_counts={0: np.asarray([1.0])},
    )

    summary = layer_hotset_jaccard_summary(dataset, ks=(8, 16, 32, 64))

    assert [row["k"] for row in summary["rows"]] == [4.0]
    assert list(summary["matrices"]) == ["4"]




def test_summarize_correlation_heterogeneity_bootstraps_mean() -> None:
    rows = [
        {
            "phase": "all",
            "layer": idx,
            "corr_attention_residual_vs_moe_js": value,
            "corr_attention_js_vs_moe_js": 0.5,
        }
        for idx, value in enumerate([-0.4, -0.2, 0.1, 0.3])
    ]

    summary = summarize_correlation_heterogeneity(
        rows,
        phases=("all", "decode"),
        n_bootstrap=200,
        random_state=1,
    )

    assert summary["all"]["n_layers"] == 4.0
    assert summary["all"]["n_positive_residual_layers"] == 2.0
    assert summary["all"]["n_negative_residual_layers"] == 2.0
    assert summary["all"]["bootstrap_mean_residual_corr_ci95_low"] is not None
    assert summary["all"]["bootstrap_mean_residual_corr_ci95_high"] is not None
    assert summary["decode"]["n_layers"] == 0.0
    assert summary["decode"]["bootstrap_mean_residual_corr_ci95_low"] is None


def test_bootstrap_mean_ci_and_head_cluster_names() -> None:
    low, high = bootstrap_mean_ci([1.0, 2.0, 3.0], n_bootstrap=100, random_state=0)

    assert low is not None
    assert high is not None
    assert low <= 2.0 <= high
    assert name_head_cluster([0.1, 0.2, 0.7, 0.0]) == "tool-result reader head"
