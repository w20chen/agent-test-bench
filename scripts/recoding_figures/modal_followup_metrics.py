"""Post-hoc follow-up metrics for agent attention/MoE recordings."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np


FOUR_WAY_HEAD_LABELS = (
    "system_mass",
    "latest_user_mass",
    "latest_tool_mass",
    "generation_mass",
)


def tool_result_segment_ages(
    records: Sequence[Any],
) -> tuple[dict[int, list[int | None]], dict[str, float]]:
    """Return per-record tool-result segment ages in same-task LLM iterations.

    Age zero means the tool result is visible for the first recorded call in
    that task. Segment identity uses `message_index` when available, which is
    stable as the prompt grows across calls. Segments without a stable message
    index are still assigned an age from a conservative fallback identity and
    counted in the diagnostics.
    """
    first_seen: dict[tuple[str, str, tuple[str, Any]], int] = {}
    by_record: dict[int, list[int | None]] = {}
    n_tool_segments = 0
    n_with_message_index = 0

    for record_index, record in enumerate(records):
        segments = _read_segments(record)
        ages: list[int | None] = [None for _segment in segments]
        call_idx = _call_idx(record, fallback=record_index)
        attempt_dir = str(getattr(record, "attempt_dir", ""))
        task = str(getattr(record, "task", ""))
        for segment_index, segment in enumerate(segments):
            if _normalize_role(segment) not in {"tool", "tool_result"}:
                continue
            n_tool_segments += 1
            identity = _tool_segment_identity(segment, segment_index)
            if identity[0] == "message_index":
                n_with_message_index += 1
            key = (attempt_dir, task, identity)
            if key not in first_seen:
                first_seen[key] = call_idx
            ages[segment_index] = max(0, call_idx - first_seen[key])
        by_record[record_index] = ages

    diagnostics = {
        "n_tool_result_segments": float(n_tool_segments),
        "n_tool_result_segments_with_message_index": float(n_with_message_index),
        "message_index_coverage": (
            float(n_with_message_index / n_tool_segments)
            if n_tool_segments > 0
            else float("nan")
        ),
    }
    return by_record, diagnostics


def fit_log_linear_half_life(
    rows: Sequence[dict[str, Any]],
    *,
    age_key: str = "age_in_iters",
    mass_key: str = "mean_attention_mass",
    weight_key: str = "query_rows",
) -> dict[str, float | None]:
    """Fit a weighted log-linear decay and return the implied half-life."""
    ages: list[float] = []
    masses: list[float] = []
    weights: list[float] = []
    for row in rows:
        age = _finite_float(row.get(age_key))
        mass = _finite_float(row.get(mass_key))
        weight = _finite_float(row.get(weight_key))
        if age is None or mass is None or weight is None:
            continue
        if age < 0 or mass <= 0 or weight <= 0:
            continue
        ages.append(age)
        masses.append(mass)
        weights.append(weight)

    unique_ages = sorted(set(ages))
    if len(unique_ages) < 2:
        return {
            "half_life_iters": None,
            "decay_lambda": None,
            "log_linear_r2": None,
            "n_fit_points": float(len(unique_ages)),
            "min_age": min(unique_ages) if unique_ages else None,
            "max_age": max(unique_ages) if unique_ages else None,
        }

    x = np.asarray(ages, dtype=np.float64)
    y = np.log(np.asarray(masses, dtype=np.float64))
    fit_weights = np.asarray(weights, dtype=np.float64)
    slope, intercept = np.polyfit(x, y, deg=1, w=np.sqrt(fit_weights))
    predicted = slope * x + intercept
    y_mean = float(np.average(y, weights=fit_weights))
    ss_res = float(np.sum(fit_weights * (y - predicted) ** 2))
    ss_tot = float(np.sum(fit_weights * (y - y_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    decay_lambda = float(-slope)
    half_life = (
        float(math.log(2.0) / decay_lambda)
        if decay_lambda > 0 and math.isfinite(decay_lambda)
        else None
    )
    return {
        "half_life_iters": half_life,
        "decay_lambda": decay_lambda,
        "log_linear_r2": r2 if math.isfinite(r2) else None,
        "n_fit_points": float(len(unique_ages)),
        "min_age": float(min(unique_ages)),
        "max_age": float(max(unique_ages)),
    }


def layer_hotset_jaccard_summary(
    dataset: Any,
    *,
    ks: Sequence[int] = (8, 16, 32, 64),
) -> dict[str, Any]:
    """Compute expert-identity Jaccard matrices between layer-static hotsets."""
    rows: list[dict[str, Any]] = []
    matrices: dict[str, Any] = {}
    seen_ks: set[int] = set()
    for k_value in ks:
        k = min(int(k_value), len(dataset.axis_labels))
        if k <= 0:
            raise ValueError("k must be positive")
        if k in seen_ks:
            continue
        seen_ks.add(k)
        layers = list(dataset.layers)
        hotsets = {
            int(layer): set(
                int(item) for item in _top_indices(_weighted_distribution(dataset, layer), k)
            )
            for layer in layers
        }
        matrix = np.full((len(layers), len(layers)), np.nan, dtype=np.float64)
        for left_idx, left_layer in enumerate(layers):
            for right_idx, right_layer in enumerate(layers):
                matrix[left_idx, right_idx] = _jaccard(
                    hotsets[int(left_layer)],
                    hotsets[int(right_layer)],
                )
        adjacent_values = [
            float(matrix[idx, idx + 1]) for idx in range(len(layers) - 1)
        ]
        non_adjacent_values = [
            float(matrix[left, right])
            for left in range(len(layers))
            for right in range(left + 1, len(layers))
            if right != left + 1
        ]
        pairwise_values = [
            float(matrix[left, right])
            for left in range(len(layers))
            for right in range(left + 1, len(layers))
        ]
        key = str(k)
        matrices[key] = {
            "layers": [float(layer) for layer in layers],
            "matrix": matrix.tolist(),
        }
        adjacent_mean = _mean(adjacent_values)
        non_adjacent_mean = _mean(non_adjacent_values)
        rows.append(
            {
                "k": float(k),
                "mean_pairwise_jaccard": _mean(pairwise_values),
                "adjacent_layer_jaccard": adjacent_mean,
                "non_adjacent_layer_jaccard": non_adjacent_mean,
                "adjacent_minus_non_adjacent": adjacent_mean - non_adjacent_mean,
                "min_pairwise_jaccard": _min(pairwise_values),
                "max_pairwise_jaccard": _max(pairwise_values),
            }
        )
    return {
        "n_layers": float(len(dataset.layers)),
        "n_experts": float(len(dataset.axis_labels)),
        "rows": rows,
        "matrices": matrices,
        "definition": (
            "For each layer, choose the top-k experts from the observation-"
            "weighted layer-static expert-load distribution, then compute "
            "expert-identity Jaccard overlap for every layer pair."
        ),
    }


def summarize_correlation_heterogeneity(
    rows: Sequence[dict[str, Any]],
    *,
    phases: Sequence[str] | None = None,
    random_state: int = 0,
    n_bootstrap: int = 5000,
) -> dict[str, Any]:
    """Summarize layer-wise attention/MoE correlation signs and bootstrap CIs."""
    phase_names = list(phases or sorted({str(row.get("phase")) for row in rows if row.get("phase")}))
    out: dict[str, Any] = {}
    for phase in phase_names:
        phase_rows = [row for row in rows if str(row.get("phase")) == phase]
        residual = np.asarray(
            [
                float(row["corr_attention_residual_vs_moe_js"])
                for row in phase_rows
                if _finite_float(row.get("corr_attention_residual_vs_moe_js")) is not None
            ],
            dtype=np.float64,
        )
        raw = np.asarray(
            [
                float(row["corr_attention_js_vs_moe_js"])
                for row in phase_rows
                if _finite_float(row.get("corr_attention_js_vs_moe_js")) is not None
            ],
            dtype=np.float64,
        )
        ci_low, ci_high = bootstrap_mean_ci(
            residual,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        sorted_residual_rows = sorted(
            phase_rows,
            key=lambda row: float(row.get("corr_attention_residual_vs_moe_js") or 0.0),
        )
        out[phase] = {
            "n_layers": float(residual.size),
            "mean_residual_corr": _mean(residual),
            "median_residual_corr": _median(residual),
            "bootstrap_mean_residual_corr_ci95_low": ci_low,
            "bootstrap_mean_residual_corr_ci95_high": ci_high,
            "n_positive_residual_layers": float(np.sum(residual > 0.0)),
            "n_negative_residual_layers": float(np.sum(residual < 0.0)),
            "mean_raw_corr": _mean(raw),
            "strongest_negative_residual_layers": sorted_residual_rows[:6],
            "strongest_positive_residual_layers": list(reversed(sorted_residual_rows[-6:])),
        }
    return out


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    n_bootstrap: int = 5000,
    random_state: int = 0,
) -> tuple[float | None, float | None]:
    """Return a percentile bootstrap 95% CI for the mean."""
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None
    rng = np.random.default_rng(random_state)
    draws = rng.choice(arr, size=(int(n_bootstrap), arr.size), replace=True)
    means = draws.mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def name_head_cluster(center: Sequence[float]) -> str:
    """Name a head cluster from its four-way specialization center."""
    values = np.asarray(center, dtype=np.float64)
    if values.shape != (len(FOUR_WAY_HEAD_LABELS),):
        raise ValueError(f"expected four center values, got {values.shape}")
    labels = {
        0: "system anchor head",
        1: "recency reader head",
        2: "tool-result reader head",
        3: "self-attender head",
    }
    return labels[int(np.argmax(values))]


def _read_segments(record: Any) -> list[dict[str, Any]]:
    iter_dir = Path(str(getattr(record, "iter_dir")))
    payload = json.loads((iter_dir / "segments.json").read_text(encoding="utf-8"))
    return list(payload.get("segments", []))


def _tool_segment_identity(
    segment: dict[str, Any],
    segment_index: int,
) -> tuple[str, Any]:
    message_index = segment.get("message_index")
    if message_index is not None:
        return ("message_index", int(message_index))
    if segment.get("char_start") is not None and segment.get("char_end") is not None:
        return ("char_span", (int(segment["char_start"]), int(segment["char_end"])))
    return (
        "segment_position",
        (
            int(segment_index),
            int(segment.get("token_start", 0) or 0),
            int(segment.get("token_end", 0) or 0),
        ),
    )


def _call_idx(record: Any, *, fallback: int) -> int:
    value = getattr(record, "call_idx", None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _normalize_role(segment: dict[str, Any]) -> str:
    role = str(segment.get("role", segment.get("segment_role", "other")) or "other")
    role = role.lower()
    if role == "assistant" and bool(segment.get("has_tool_calls")):
        return "assistant_call"
    if role == "assistant":
        return "assistant_message"
    if role in {
        "system",
        "user",
        "assistant_message",
        "assistant_call",
        "tool",
        "tool_result",
        "gen_prompt",
        "generation",
        "meta",
    }:
        return role
    return "other"


def _weighted_distribution(dataset: Any, layer: int) -> np.ndarray:
    matrix = np.asarray(dataset.distributions[layer], dtype=np.float64)
    weights = np.asarray(dataset.observation_counts[layer], dtype=np.float64)
    valid = np.isfinite(weights) & (weights > 0.0)
    if not bool(valid.any()):
        return np.zeros(len(dataset.axis_labels), dtype=np.float64)
    weighted = np.sum(matrix[valid] * weights[valid, None], axis=0)
    total = float(weighted.sum())
    if total <= 0 or not math.isfinite(total):
        return np.zeros(len(dataset.axis_labels), dtype=np.float64)
    return weighted / total


def _top_indices(values: np.ndarray, k: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"expected rank-1 distribution, got {arr.shape}")
    if np.any(arr < 0) or np.any(~np.isfinite(arr)):
        raise ValueError("expert distribution contains negative or non-finite values")
    return np.argsort(-arr, kind="stable")[:k]


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    if not union:
        return float("nan")
    return float(len(left & right) / len(union))


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _median(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _min(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.min(arr)) if arr.size else float("nan")


def _max(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if arr.size else float("nan")
