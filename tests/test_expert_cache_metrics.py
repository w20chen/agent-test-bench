from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


RECODING_FIGURES = Path(__file__).resolve().parents[1] / "scripts" / "recoding_figures"
sys.path.insert(0, str(RECODING_FIGURES))

from expert_cache_metrics import expert_cache_coverage_summary  # noqa: E402
from moe_phase_audit import (  # noqa: E402
    _accumulate_phase_denominators,
    _decode_prefill_ratios,
    _empty_phase_totals,
)
from recording_loader import LayerDistributionSet  # noqa: E402


def test_adjacent_dynamic_coverage_splits_same_task_and_boundary() -> None:
    records = [
        SimpleNamespace(task="task-a"),
        SimpleNamespace(task="task-a"),
        SimpleNamespace(task="task-b"),
    ]
    dataset = LayerDistributionSet(
        modality="moe",
        records=records,  # type: ignore[arg-type]
        layers=[0],
        axis_labels=["0", "1", "2", "3"],
        distributions={
            0: np.asarray(
                [
                    [0.7, 0.2, 0.1, 0.0],
                    [0.6, 0.3, 0.1, 0.0],
                    [0.1, 0.8, 0.1, 0.0],
                ],
                dtype=np.float64,
            )
        },
        observation_counts={0: np.asarray([10.0, 20.0, 30.0], dtype=np.float64)},
    )

    summary = expert_cache_coverage_summary(dataset, ks=(1,))
    row = summary["coverage_rows"][0]

    assert row["adjacent_same_task_coverage"] == 0.6
    assert row["adjacent_cross_task_splice_coverage"] == 0.1
    assert row["adjacent_prev_iter_coverage"] == 0.3
    assert row["adjacent_same_task_equal_task_coverage"] == 0.6
    assert row["adjacent_cross_task_splice_equal_splice_coverage"] == 0.1
    assert row["n_same_task_layer_transitions"] == 1.0
    assert row["n_cross_task_splice_layer_transitions"] == 1.0
    assert row["per_task"][0]["task"] == "task-a"
    assert row["per_cross_task_splice"][0]["prev_task"] == "task-a"
    assert row["per_cross_task_splice"][0]["next_task"] == "task-b"
    assert math.isclose(row["static_layer_coverage"], 32.0 / 60.0)
    assert math.isclose(row["static_global_coverage"], 32.0 / 60.0)


def test_moe_phase_denominator_accumulates_token_rows_and_load() -> None:
    totals = _empty_phase_totals()
    expert_load = np.asarray(
        [
            [[1.0, 0.0], [0.5, 0.5]],
            [[2.0, 1.0], [1.0, 0.0]],
            [[0.0, 3.0], [2.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )

    _accumulate_phase_denominators(
        totals,
        record_phases=np.asarray(["prefill", "decode", "mixed", "unknown"]),
        token_row_offsets=np.asarray([0, 10, 11, 15, 15], dtype=np.int64),
        expert_load=expert_load,
        top_k=8,
    )

    assert totals["prefill"]["routing_records"] == 1.0
    assert totals["prefill"]["token_rows"] == 10.0
    assert totals["prefill"]["topk_assignments"] == 80.0
    assert totals["prefill"]["expert_load_sum"] == 2.0
    assert totals["decode"]["token_rows"] == 1.0
    assert totals["decode"]["topk_assignments"] == 8.0
    assert totals["decode"]["expert_load_sum"] == 4.0
    assert totals["mixed"]["token_rows"] == 4.0
    assert totals["unknown"]["token_rows"] == 0.0

    ratios = _decode_prefill_ratios(totals)
    assert ratios["token_rows"] == 0.1
    assert ratios["topk_assignments"] == 0.1
    assert ratios["expert_load_sum"] == 2.0
