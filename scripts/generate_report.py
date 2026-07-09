#!/usr/bin/env python3
"""Generate a Markdown VTune report from ``vtune_runs.csv``.

The input is the CSV emitted by ``scripts/analyze_vtune.py``. Every reported
number is computed from the CSV at runtime; the script does not embed result
values.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_METRICS: tuple[tuple[str, str, int], ...] = (
    ("Elapsed time (s)", "fine/Elapsed Time", 3),
    ("CPI rate", "fine/CPI Rate", 3),
    ("Derived IPC", "fine/IPC (derived)", 3),
    ("Retiring (%)", "fine/Retiring", 2),
    ("Front-end bound (%)", "fine/Front-End Bound", 2),
    ("Back-end bound (%)", "fine/Back-End Bound", 2),
    ("Bad speculation (%)", "fine/Bad Speculation", 2),
    ("Memory bound (%)", "fine/Memory Bound", 2),
    ("Branch mispredict (%)", "fine/Branch Mispredict", 2),
    ("ICache misses (%)", "fine/ICache Misses", 2),
    ("L3 bound (%)", "fine/L3 Bound", 2),
    ("Average CPU (%)", "coarse/cpu_percent.avg", 2),
    ("Peak memory (MB)", "coarse/memory_mb.max", 1),
    ("Disk read delta (MB)", "coarse/disk_read_mb.delta", 2),
    ("Disk write delta (MB)", "coarse/disk_write_mb.delta", 2),
    ("Context switches delta", "coarse/context_switches.delta", 0),
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"input CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"input CSV has no rows: {path}")
    if "tool" not in rows[0]:
        raise ValueError("input CSV must contain a 'tool' column")
    return rows


def _number(row: dict[str, str], column: str) -> float | None:
    value = row.get(column, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _values(rows: Iterable[dict[str, str]], tool: str, column: str) -> list[float]:
    return [
        value
        for row in rows
        if row.get("tool") == tool
        for value in [_number(row, column)]
        if value is not None
    ]


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "n": float(len(values)),
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def _format(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "NA"
    if abs(value) >= 1e6 or (value != 0 and abs(value) < 1e-3):
        return f"{value:.3g}"
    return f"{value:.{digits}f}"


def _is_tma_valid(row: dict[str, str]) -> bool:
    bad_spec = _number(row, "fine/Bad Speculation") or 0.0
    retiring = _number(row, "fine/Retiring") or 0.0
    front_end = _number(row, "fine/Front-End Bound") or 0.0
    back_end = _number(row, "fine/Back-End Bound") or 0.0
    return not (
        bad_spec >= 90.0
        and retiring == 0.0
        and front_end == 0.0
        and back_end == 0.0
    )


def _tool_order(rows: list[dict[str, str]], explicit: str | None) -> list[str]:
    if explicit:
        return [tool.strip() for tool in explicit.split(",") if tool.strip()]
    return sorted({row["tool"] for row in rows if row.get("tool")})


def build_report(
    rows: list[dict[str, str]],
    *,
    source_path: Path,
    tools: list[str],
) -> str:
    """Build a Markdown report from flattened VTune rows."""
    lines: list[str] = [
        "# VTune Profiling Report",
        "",
        f"- Source CSV: `{source_path}`",
        f"- Profiled runs: {len(rows)}",
        f"- Experiment groups: {len({row.get('experiment', '') for row in rows})}",
        "- Runs per tool: "
        + ", ".join(
            f"`{tool}`={count}"
            for tool, count in sorted(Counter(row["tool"] for row in rows).items())
        ),
        "",
        "## Data Quality",
        "",
        "| Tool | Runs | TMA valid | TMA invalid | Valid rate |",
        "|---|---:|---:|---:|---:|",
    ]

    valid_rows_by_tool: dict[str, list[dict[str, str]]] = defaultdict(list)
    for tool in tools:
        tool_rows = [row for row in rows if row.get("tool") == tool]
        valid = [row for row in tool_rows if _is_tma_valid(row)]
        valid_rows_by_tool[tool] = valid
        invalid_count = len(tool_rows) - len(valid)
        valid_rate = (len(valid) / len(tool_rows) * 100.0) if tool_rows else 0.0
        lines.append(
            f"| {tool} | {len(tool_rows)} | {len(valid)} | "
            f"{invalid_count} | {valid_rate:.0f}% |"
        )

    lines.extend(
        [
            "",
            "TMA-valid filtering only affects microarchitecture tables. Coarse "
            "resource metrics are reported from every row with numeric data.",
            "",
            "## Metrics",
            "",
            "| Metric | Tool | n | min | mean | median | max | std |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for label, column, digits in DEFAULT_METRICS:
        is_fine_metric = column.startswith("fine/")
        for tool in tools:
            metric_rows = valid_rows_by_tool[tool] if is_fine_metric else rows
            stats = _stats(_values(metric_rows, tool, column))
            if stats is None:
                continue
            lines.append(
                f"| {label} | {tool} | {int(stats['n'])} | "
                f"{_format(stats['min'], digits)} | "
                f"{_format(stats['mean'], digits)} | "
                f"{_format(stats['median'], digits)} | "
                f"{_format(stats['max'], digits)} | "
                f"{_format(stats['std'], digits)} |"
            )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_input = Path(__file__).with_name("vtune_runs.csv")
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="CSV produced by scripts/analyze_vtune.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("vtune_report.md"),
        help="Markdown report path.",
    )
    parser.add_argument(
        "--tools",
        help="Optional comma-separated tool order. Defaults to tools in CSV.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Write the report without printing it to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_rows(args.input)
    report = build_report(
        rows,
        source_path=args.input,
        tools=_tool_order(rows, args.tools),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    if not args.quiet:
        print(report, end="")
    print(f"[written] {args.output}")


if __name__ == "__main__":
    main()
