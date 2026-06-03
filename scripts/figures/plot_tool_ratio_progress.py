#!/usr/bin/env python3
"""Plot tool-time ratio distribution and tool-call progress over execution."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TOOL_ORDER = ["Bash", "Read", "Edit", "Grep", "Glob", "Write", "TodoWrite", "Other"]
TOOL_COLORS = {
    "Bash": "#5DA5DA",
    "Read": "#FAA43A",
    "Edit": "#60BD68",
    "Grep": "#F15854",
    "Glob": "#9C7AC7",
    "Write": "#B276B2",
    "TodoWrite": "#DE77AE",
    "Other": "#8C8C8C",
}


@dataclass(frozen=True)
class CohortSpec:
    label: str
    root: Path
    model_substring: str | None = None


@dataclass(frozen=True)
class AttemptData:
    task: str
    total_time_s: float
    active_time_s: float
    tool_time_s: float
    tool_ratio: float
    results_path: Path
    tool_calls_path: Path
    trace_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-task tool ratio and normalized tool usage progress.",
    )
    parser.add_argument(
        "--cohort",
        action="append",
        required=True,
        help="Cohort spec: LABEL=PATH or LABEL=PATH::MODEL_SUBSTRING",
    )
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of normalized execution bins for panel (b)",
    )
    parser.add_argument(
        "--title",
        default="Tool usage profile",
        help="Console title only",
    )
    return parser.parse_args()


def parse_cohort(raw: str) -> CohortSpec:
    if "=" not in raw:
        raise ValueError(f"Invalid cohort spec {raw!r}")
    label, remainder = raw.split("=", 1)
    if "::" in remainder:
        path_str, model_substring = remainder.split("::", 1)
    else:
        path_str, model_substring = remainder, None
    root = Path(path_str).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Cohort path does not exist: {root}")
    return CohortSpec(label=label, root=root, model_substring=model_substring or None)


def load_attempts(spec: CohortSpec) -> list[AttemptData]:
    attempts: list[AttemptData] = []
    for results_path in sorted(spec.root.rglob("results.json")):
        data = json.loads(results_path.read_text(encoding="utf-8"))
        model = str(data.get("model") or "")
        if spec.model_substring and spec.model_substring not in model:
            continue

        tool_calls_path = results_path.with_name("tool_calls.json")
        trace_path = results_path.with_name("trace.jsonl")
        if not tool_calls_path.exists() or not trace_path.exists():
            continue

        active_time = float(data.get("active_time") or 0.0)
        tool_time = float(data.get("tool_time") or 0.0)
        total_time = float(data.get("total_time") or 0.0)
        if active_time <= 0 or total_time <= 0:
            continue

        attempts.append(
            AttemptData(
                task=str(data.get("instance_id") or results_path.parent.parent.name),
                total_time_s=total_time,
                active_time_s=active_time,
                tool_time_s=tool_time,
                tool_ratio=tool_time / active_time,
                results_path=results_path,
                tool_calls_path=tool_calls_path,
                trace_path=trace_path,
            )
        )
    if not attempts:
        raise ValueError(f"No attempts found for cohort {spec.label!r}")
    return attempts


def parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def tool_bucket_name(raw_name: str) -> str:
    normalized = raw_name.strip().lower()
    if normalized in {"bash", "exec"}:
        return "Bash"
    if normalized in {"read", "read_file"}:
        return "Read"
    if normalized in {"edit", "edit_file"}:
        return "Edit"
    if normalized == "grep":
        return "Grep"
    if normalized == "glob":
        return "Glob"
    if normalized in {"write", "write_file"}:
        return "Write"
    if normalized in {"todowrite", "todo_write"}:
        return "TodoWrite"
    return "Other"


def timeline_bounds(trace_path: Path) -> tuple[float, float] | None:
    start: float | None = None
    end: float | None = None
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record_type = record.get("type")
            if record_type == "action":
                ts0 = record.get("ts_start")
                ts1 = record.get("ts_end", ts0)
                if isinstance(ts0, (int, float)):
                    start = ts0 if start is None else min(start, ts0)
                if isinstance(ts1, (int, float)):
                    end = ts1 if end is None else max(end, ts1)
            elif "timestamp" in record:
                ts = parse_iso(record.get("timestamp"))
                if ts is None:
                    continue
                value = ts.timestamp()
                start = value if start is None else min(start, value)
                end = value if end is None else max(end, value)
    if start is None or end is None or end <= start:
        return None
    return start, end


def accumulate_progress_counts(attempts: list[AttemptData], n_bins: int) -> dict[str, np.ndarray]:
    counts = {name: np.zeros(n_bins, dtype=float) for name in TOOL_ORDER}
    for attempt in attempts:
        bounds = timeline_bounds(attempt.trace_path)
        if bounds is None:
            continue
        start, end = bounds
        span = end - start
        if span <= 0:
            continue
        items = json.loads(attempt.tool_calls_path.read_text(encoding="utf-8"))
        for item in items:
            ts0 = parse_iso(item.get("timestamp"))
            ts1 = parse_iso(item.get("end_timestamp")) or ts0
            if ts0 is None or ts1 is None:
                continue
            mid = (ts0.timestamp() + ts1.timestamp()) / 2.0
            frac = (mid - start) / span
            frac = min(max(frac, 0.0), 0.999999)
            bucket = min(int(frac * n_bins), n_bins - 1)
            counts[tool_bucket_name(str(item.get("tool") or ""))][bucket] += 1
    return counts


def plot_figure(
    cohorts: list[tuple[CohortSpec, list[AttemptData]]],
    counts: dict[str, np.ndarray],
    output_path: Path,
    n_bins: int,
) -> None:
    fig, (ax_hist, ax_stack) = plt.subplots(1, 2, figsize=(12, 6))

    colors = ["#5DA5DA", "#60BD68", "#B276B2", "#F17CB0"]
    all_ratios = []
    max_ratio = 0.0
    for idx, (spec, attempts) in enumerate(cohorts):
        ratios = [attempt.tool_ratio * 100.0 for attempt in attempts]
        all_ratios.extend(ratios)
        max_ratio = max(max_ratio, max(ratios, default=0.0))
        ax_hist.hist(
            ratios,
            bins=np.arange(0, max(max_ratio + 10, 90), 5),
            alpha=0.55,
            color=colors[idx % len(colors)],
            label=f"{spec.label} (n={len(attempts)})",
        )

    mean_ratio = float(np.mean(all_ratios))
    median_ratio = float(np.median(all_ratios))
    ax_hist.axvline(mean_ratio, color="red", linestyle="--", linewidth=1.5, label=f"Mean ({mean_ratio:.1f}%)")
    ax_hist.axvline(median_ratio, color="black", linestyle=":", linewidth=1.2, label=f"Median ({median_ratio:.1f}%)")
    ax_hist.set_title("(a) Per-Task Tool Time Ratio")
    ax_hist.set_xlabel("Tool Time Ratio (%)")
    ax_hist.set_ylabel("Number of Tasks")
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.25)

    x = np.arange(n_bins)
    y = [counts[name] for name in TOOL_ORDER]
    labels = [f"{i*10}-{(i+1)*10}%" for i in range(n_bins)]
    ax_stack.stackplot(
        x,
        *y,
        labels=TOOL_ORDER,
        colors=[TOOL_COLORS[name] for name in TOOL_ORDER],
        alpha=0.95,
    )
    ax_stack.set_xticks(x)
    ax_stack.set_xticklabels(labels, rotation=35)
    ax_stack.set_title("(b) Tool Usage Over Execution Timeline")
    ax_stack.set_xlabel("Normalized Execution Time")
    ax_stack.set_ylabel("Tool Call Count")
    ax_stack.legend(loc="upper right")
    ax_stack.grid(True, alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_summary(title: str, cohorts: list[tuple[CohortSpec, list[AttemptData]]], output_path: Path) -> None:
    print(title)
    print("=" * len(title))
    for spec, attempts in cohorts:
        ratios = [attempt.tool_ratio * 100.0 for attempt in attempts]
        print(f"\n[{spec.label}] n={len(attempts)}")
        print(f"  mean ratio (%):   {np.mean(ratios):.2f}")
        print(f"  median ratio (%): {np.median(ratios):.2f}")
        for attempt in attempts:
            print(
                f"    - {attempt.task}: total={attempt.total_time_s:.1f}s "
                f"active={attempt.active_time_s:.1f}s tool={attempt.tool_time_s:.1f}s "
                f"ratio={attempt.tool_ratio*100:.2f}%"
            )
    print(f"\nSaved figure to: {output_path}")


def main() -> None:
    args = parse_args()
    cohort_specs = [parse_cohort(raw) for raw in args.cohort]
    cohorts = [(spec, load_attempts(spec)) for spec in cohort_specs]
    combined_attempts = [attempt for _, attempts in cohorts for attempt in attempts]
    counts = accumulate_progress_counts(combined_attempts, args.bins)
    output_path = Path(args.output).expanduser().resolve()
    plot_figure(cohorts, counts, output_path, args.bins)
    print_summary(args.title, cohorts, output_path)


if __name__ == "__main__":
    main()
