"""Expert cache coverage metrics for MoE routing distributions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


def expert_cache_coverage_summary(
    dataset: Any,
    *,
    ks: Sequence[int] = (8, 16, 32, 64),
) -> dict[str, Any]:
    """Summarize static and adjacent previous-iteration expert coverage."""
    n_experts = len(dataset.axis_labels)
    rows: list[dict[str, Any]] = []
    global_dist = _weighted_distribution(dataset)

    for k_value in ks:
        k = min(int(k_value), n_experts)
        if k <= 0:
            raise ValueError("k must be positive")
        global_hotset = _top_indices(global_dist, k)
        static_global = _coverage_for_fixed_hotset(dataset, global_hotset)
        static_layer = _layer_static_coverage(dataset, k)
        adjacent = _adjacent_dynamic_coverage(dataset, k)
        rows.append(
            {
                "k": float(k),
                "static_global_coverage": static_global["coverage"],
                "static_layer_coverage": static_layer["coverage"],
                "static_layer_minus_global": static_layer["coverage"]
                - static_global["coverage"],
                "adjacent_prev_iter_coverage": adjacent["coverage"],
                "adjacent_same_task_coverage": adjacent["same_task_coverage"],
                "adjacent_cross_task_splice_coverage": adjacent[
                    "cross_task_splice_coverage"
                ],
                "adjacent_same_task_equal_task_coverage": adjacent[
                    "same_task_equal_task_coverage"
                ],
                "adjacent_cross_task_splice_equal_splice_coverage": adjacent[
                    "cross_task_splice_equal_splice_coverage"
                ],
                "adjacent_topk_jaccard": adjacent["topk_jaccard"],
                "adjacent_same_task_topk_jaccard": adjacent[
                    "same_task_topk_jaccard"
                ],
                "adjacent_cross_task_splice_topk_jaccard": adjacent[
                    "cross_task_splice_topk_jaccard"
                ],
                "n_adjacent_layer_transitions": adjacent[
                    "n_adjacent_layer_transitions"
                ],
                "n_same_task_layer_transitions": adjacent[
                    "n_same_task_layer_transitions"
                ],
                "n_cross_task_splice_layer_transitions": adjacent[
                    "n_cross_task_splice_layer_transitions"
                ],
                "per_task": adjacent["per_task"],
                "per_cross_task_splice": adjacent["per_cross_task_splice"],
                "layer_hotset_pairwise_jaccard": _layer_hotset_jaccard(dataset, k),
            }
        )

    return {
        "n_layers": float(len(dataset.layers)),
        "n_experts": float(n_experts),
        "coverage_rows": rows,
        "definition": (
            "For each layer and adjacent iteration pair, take the previous "
            "iteration's top-k expert distribution support and measure the "
            "current iteration's expert-load mass covered by that hotset. "
            "Observation-weighted summaries use the current row's MoE "
            "observation count, matching the original 0.6417 top-32 metric. "
            "Cross-task adjacent pairs are adjacent in the provided record order; "
            "with the curated-14 loader this is a task-sorted synthetic splice, "
            "not a chronological runtime task switch."
        ),
    }


def _adjacent_dynamic_coverage(dataset: Any, k: int) -> dict[str, Any]:
    overall = _WeightedAccumulator()
    same_task = _WeightedAccumulator()
    cross_task_splice = _WeightedAccumulator()
    per_task: dict[str, _WeightedAccumulator] = {}
    per_cross_task_splice: dict[tuple[int, str, str], _WeightedAccumulator] = {}
    jaccard_all: list[float] = []
    jaccard_same: list[float] = []
    jaccard_cross_task_splice: list[float] = []

    records = dataset.records
    for layer in dataset.layers:
        matrix = dataset.distributions[layer]
        obs = dataset.observation_counts[layer]
        for idx in range(1, len(records)):
            if obs[idx - 1] <= 0 or obs[idx] <= 0:
                continue
            prev_hotset = set(int(item) for item in _top_indices(matrix[idx - 1], k))
            curr_hotset = set(int(item) for item in _top_indices(matrix[idx], k))
            coverage = float(matrix[idx, list(prev_hotset)].sum())
            weight = float(obs[idx])
            jaccard = _jaccard(prev_hotset, curr_hotset)
            prev_task = str(records[idx - 1].task)
            curr_task = str(records[idx].task)

            overall.add(coverage, weight)
            jaccard_all.append(jaccard)
            if prev_task == curr_task:
                same_task.add(coverage, weight)
                per_task.setdefault(curr_task, _WeightedAccumulator()).add(
                    coverage,
                    weight,
                )
                jaccard_same.append(jaccard)
            else:
                cross_task_splice.add(coverage, weight)
                key = (idx, prev_task, curr_task)
                per_cross_task_splice.setdefault(key, _WeightedAccumulator()).add(
                    coverage,
                    weight,
                )
                jaccard_cross_task_splice.append(jaccard)

    per_task_rows = [
        {
            "task": task,
            "coverage": accumulator.mean(),
            "weight": accumulator.weight,
            "n_layer_transitions": float(accumulator.count),
        }
        for task, accumulator in sorted(per_task.items())
    ]
    per_cross_task_splice_rows = [
        {
            "splice_index": float(index),
            "prev_task": prev_task,
            "next_task": next_task,
            "coverage": accumulator.mean(),
            "weight": accumulator.weight,
            "n_layer_transitions": float(accumulator.count),
        }
        for (index, prev_task, next_task), accumulator in sorted(
            per_cross_task_splice.items()
        )
    ]

    return {
        "coverage": overall.mean(),
        "same_task_coverage": same_task.mean(),
        "cross_task_splice_coverage": cross_task_splice.mean(),
        "same_task_equal_task_coverage": _mean(
            row["coverage"] for row in per_task_rows
        ),
        "cross_task_splice_equal_splice_coverage": _mean(
            row["coverage"] for row in per_cross_task_splice_rows
        ),
        "topk_jaccard": _mean(jaccard_all),
        "same_task_topk_jaccard": _mean(jaccard_same),
        "cross_task_splice_topk_jaccard": _mean(jaccard_cross_task_splice),
        "n_adjacent_layer_transitions": float(overall.count),
        "n_same_task_layer_transitions": float(same_task.count),
        "n_cross_task_splice_layer_transitions": float(cross_task_splice.count),
        "per_task": per_task_rows,
        "per_cross_task_splice": per_cross_task_splice_rows,
    }


def _coverage_for_fixed_hotset(dataset: Any, hotset: Any) -> dict[str, float]:
    accumulator = _WeightedAccumulator()
    for layer in dataset.layers:
        matrix = dataset.distributions[layer]
        weights = dataset.observation_counts[layer].astype(np.float64)
        valid = weights > 0
        if not bool(valid.any()):
            continue
        for coverage, weight in zip(
            matrix[valid][:, hotset].sum(axis=1),
            weights[valid],
            strict=True,
        ):
            accumulator.add(float(coverage), float(weight))
    return {"coverage": accumulator.mean()}


def _layer_static_coverage(dataset: Any, k: int) -> dict[str, float]:
    accumulator = _WeightedAccumulator()
    for layer in dataset.layers:
        hotset = _top_indices(_weighted_distribution(dataset, layer), k)
        matrix = dataset.distributions[layer]
        weights = dataset.observation_counts[layer].astype(np.float64)
        valid = weights > 0
        if not bool(valid.any()):
            continue
        for coverage, weight in zip(
            matrix[valid][:, hotset].sum(axis=1),
            weights[valid],
            strict=True,
        ):
            accumulator.add(float(coverage), float(weight))
    return {"coverage": accumulator.mean()}


def _weighted_distribution(dataset: Any, layer: int | None = None) -> Any:
    total: Any | None = None
    total_weight = 0.0
    layers = [layer] if layer is not None else dataset.layers
    for selected_layer in layers:
        matrix = dataset.distributions[selected_layer]
        weights = dataset.observation_counts[selected_layer].astype(np.float64)
        valid = weights > 0
        if not bool(valid.any()):
            continue
        weighted = np.sum(matrix[valid] * weights[valid, None], axis=0)
        total = weighted if total is None else total + weighted
        total_weight += float(weights[valid].sum())
    if total is None or total_weight <= 0:
        return np.zeros(len(dataset.axis_labels), dtype=np.float64)
    return total / float(total.sum())


def _layer_hotset_jaccard(dataset: Any, k: int) -> float:
    hotsets = [
        set(int(item) for item in _top_indices(_weighted_distribution(dataset, layer), k))
        for layer in dataset.layers
    ]
    return _mean(
        _jaccard(hotsets[left], hotsets[right])
        for left in range(len(hotsets))
        for right in range(left + 1, len(hotsets))
    )


def _top_indices(values: Any, k: int) -> Any:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"expected rank-1 distribution, got {arr.shape}")
    if np.any(arr < 0) or np.any(~np.isfinite(arr)):
        raise ValueError("expert distribution contains negative or non-finite values")
    if k >= arr.size:
        return np.arange(arr.size)
    return np.argpartition(-arr, kth=k - 1)[:k]


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    if not union:
        return float("nan")
    return float(len(left & right) / len(union))


def _mean(values: Any) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


class _WeightedAccumulator:
    def __init__(self) -> None:
        self.weighted_sum = 0.0
        self.weight = 0.0
        self.count = 0

    def add(self, value: float, weight: float) -> None:
        if not math.isfinite(value) or not math.isfinite(weight) or weight <= 0:
            return
        self.weighted_sum += value * weight
        self.weight += weight
        self.count += 1

    def mean(self) -> float:
        if self.weight <= 0:
            return float("nan")
        return float(self.weighted_sum / self.weight)
