from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Any


METHODS = (
    ("last_run", "Last Run"),
    ("test_count", "Test Count"),
    ("per_test", "Per-Test Historical"),
    ("unknown_test_fallback", "Unknown-Test Fallback"),
    ("recommended", "Recommended"),
)
RELIABILITY_LEVELS = ("high", "medium", "low", "coldstart", "error", "unavailable")


def _iter_prediction_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("pytest_runtime/predictions.jsonl"))


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            row["_source"] = str(path)
            rows.append(row)
    return rows


def _available_rows(rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    key = f"prediction_{method}_s"
    return [row for row in rows if isinstance(row.get(key), (int, float))]


def _reliability_level(row: dict[str, Any]) -> str:
    reliability = row.get("prediction_reliability")
    if not isinstance(reliability, dict):
        return "unavailable"
    level = reliability.get("level")
    return str(level) if level in RELIABILITY_LEVELS else "unavailable"


def _mae(rows: list[dict[str, Any]], method: str) -> float | None:
    values = []
    for row in _available_rows(rows, method):
        errors = row.get("absolute_error") or {}
        value = errors.get(method)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return statistics.mean(values) if values else None


def _mape(rows: list[dict[str, Any]], method: str) -> float | None:
    values = []
    for row in _available_rows(rows, method):
        errors = row.get("relative_error") or {}
        value = errors.get(method)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return statistics.mean(values) if values else None


def _mean_actual(rows: list[dict[str, Any]]) -> float | None:
    values = [
        float(row["actual_duration_s"])
        for row in rows
        if isinstance(row.get("actual_duration_s"), (int, float))
    ]
    return statistics.mean(values) if values else None


def _mean_collect_only_overhead(rows: list[dict[str, Any]]) -> float | None:
    values = [
        float(row["collect_only_duration_s"])
        for row in rows
        if isinstance(row.get("collect_only_duration_s"), (int, float))
    ]
    return statistics.mean(values) if values else None


def _fmt_seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def _fmt_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source",
        "iteration",
        "command",
        "actual_duration_s",
        "collect_only_duration_s",
        "total_duration_with_prediction_overhead_s",
        "collected_count",
        "pre_execution_collected_count",
        "prediction_last_run_s",
        "prediction_test_count_s",
        "prediction_per_test_s",
        "prediction_per_test_without_overhead_s",
        "prediction_per_test_overhead_s",
        "prediction_unknown_test_fallback_s",
        "prediction_unknown_test_fallback_without_overhead_s",
        "prediction_unknown_test_fallback_overhead_s",
        "prediction_recommended_s",
        "prediction_recommended_method",
        "prediction_reliability_level",
        "known_node_ratio",
        "file_fallback_ratio",
        "project_fallback_ratio",
        "unknown_fallback_ratio",
        "collected_count_delta_ratio",
        "absolute_error_last_run",
        "absolute_error_test_count",
        "absolute_error_per_test",
        "absolute_error_unknown_test_fallback",
        "absolute_error_recommended",
        "relative_error_last_run",
        "relative_error_test_count",
        "relative_error_per_test",
        "relative_error_unknown_test_fallback",
        "relative_error_recommended",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            abs_err = row.get("absolute_error") or {}
            rel_err = row.get("relative_error") or {}
            reliability = row.get("prediction_reliability") or {}
            writer.writerow(
                {
                    "source": row.get("_source"),
                    "iteration": row.get("iteration"),
                    "command": row.get("command"),
                    "actual_duration_s": row.get("actual_duration_s"),
                    "collect_only_duration_s": row.get("collect_only_duration_s"),
                    "total_duration_with_prediction_overhead_s": row.get(
                        "total_duration_with_prediction_overhead_s"
                    ),
                    "collected_count": row.get("collected_count"),
                    "pre_execution_collected_count": row.get(
                        "pre_execution_collected_count"
                    ),
                    "prediction_last_run_s": row.get("prediction_last_run_s"),
                    "prediction_test_count_s": row.get("prediction_test_count_s"),
                    "prediction_per_test_s": row.get("prediction_per_test_s"),
                    "prediction_per_test_without_overhead_s": row.get(
                        "prediction_per_test_without_overhead_s"
                    ),
                    "prediction_per_test_overhead_s": row.get(
                        "prediction_per_test_overhead_s"
                    ),
                    "prediction_unknown_test_fallback_s": row.get(
                        "prediction_unknown_test_fallback_s"
                    ),
                    "prediction_unknown_test_fallback_without_overhead_s": row.get(
                        "prediction_unknown_test_fallback_without_overhead_s"
                    ),
                    "prediction_unknown_test_fallback_overhead_s": row.get(
                        "prediction_unknown_test_fallback_overhead_s"
                    ),
                    "prediction_recommended_s": row.get("prediction_recommended_s"),
                    "prediction_recommended_method": row.get(
                        "prediction_recommended_method"
                    ),
                    "prediction_reliability_level": reliability.get("level"),
                    "known_node_ratio": reliability.get("known_node_ratio"),
                    "file_fallback_ratio": reliability.get("file_fallback_ratio"),
                    "project_fallback_ratio": reliability.get(
                        "project_fallback_ratio"
                    ),
                    "unknown_fallback_ratio": reliability.get(
                        "unknown_fallback_ratio"
                    ),
                    "collected_count_delta_ratio": reliability.get(
                        "collected_count_delta_ratio"
                    ),
                    "absolute_error_last_run": abs_err.get("last_run"),
                    "absolute_error_test_count": abs_err.get("test_count"),
                    "absolute_error_per_test": abs_err.get("per_test"),
                    "absolute_error_unknown_test_fallback": abs_err.get(
                        "unknown_test_fallback"
                    ),
                    "absolute_error_recommended": abs_err.get("recommended"),
                    "relative_error_last_run": rel_err.get("last_run"),
                    "relative_error_test_count": rel_err.get("test_count"),
                    "relative_error_per_test": rel_err.get("per_test"),
                    "relative_error_unknown_test_fallback": rel_err.get(
                        "unknown_test_fallback"
                    ),
                    "relative_error_recommended": rel_err.get("recommended"),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize pytest runtime prediction artifacts.",
    )
    parser.add_argument("trace_or_result_dir", type=Path)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path for per-run rows.",
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in _iter_prediction_files(args.trace_or_result_dir):
        rows.extend(_load_rows(path))

    valid_rows = [
        row
        for row in rows
        if any(
            isinstance(row.get(f"prediction_{method}_s"), (int, float))
            for method, _ in METHODS
        )
    ]
    finalized_rows = [
        row
        for row in rows
        if isinstance(row.get("actual_duration_s"), (int, float))
        or isinstance(row.get("collect_only_duration_s"), (int, float))
    ]
    long_rows = [
        row
        for row in valid_rows
        if isinstance(row.get("actual_duration_s"), (int, float))
        and float(row["actual_duration_s"]) >= 30.0
    ]

    print(f"Valid runs: {len(valid_rows)}")
    print(f"Long pytest runs: {len(long_rows)}")
    mean_actual = _mean_actual(valid_rows)
    if mean_actual is not None:
        print(f"Average actual runtime: {mean_actual:.1f}s")
    mean_collect_only = _mean_collect_only_overhead(finalized_rows)
    if mean_collect_only is not None:
        print(f"Average collect-only overhead: {mean_collect_only:.1f}s")
    print()
    print(f"{'Method':<24} {'N':>6} {'MAE(s)':>10} {'MAPE':>10} {'Long MAPE':>12}")
    for method, label in METHODS:
        print(
            f"{label:<24} "
            f"{len(_available_rows(valid_rows, method)):>6} "
            f"{_fmt_seconds(_mae(valid_rows, method)):>10} "
            f"{_fmt_percent(_mape(valid_rows, method)):>10} "
            f"{_fmt_percent(_mape(long_rows, method)):>12}"
        )
    print()
    print("Reliability buckets:")
    for level in RELIABILITY_LEVELS:
        bucket = [
            row
            for row in finalized_rows
            if _reliability_level(row) == level
        ]
        print(
            f"  {level:<11} "
            f"runs={len(bucket):>4} "
            f"recommended_MAPE={_fmt_percent(_mape(bucket, 'recommended')):>8} "
            f"recommended_MAE={_fmt_seconds(_mae(bucket, 'recommended')):>8}"
        )

    if args.csv is not None:
        _write_csv(finalized_rows, args.csv)
        print(f"\nCSV written to: {args.csv}")


if __name__ == "__main__":
    main()
