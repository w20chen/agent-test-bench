#!/usr/bin/env python3
"""Plot REAL-mode tool ratio distribution and tool progress over compacted execution."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _real_trace_metrics import (  # noqa: E402
    CohortSpec,
    RealTraceMetrics,
    load_real_trace_metrics,
    parse_cohort,
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot REAL-mode tool ratio and tool progress over compacted execution.",
    )
    parser.add_argument(
        "--cohort",
        action="append",
        required=True,
        help="Cohort spec in the form LABEL=PATH or LABEL=PATH::MODEL_SUBSTRING.",
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
        default="Tool usage profile (REAL mode)",
        help="Console title only",
    )
    return parser.parse_args()


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


def accumulate_progress_counts(
    attempts: list[RealTraceMetrics], n_bins: int
) -> dict[str, np.ndarray]:
    counts = {name: np.zeros(n_bins, dtype=float) for name in TOOL_ORDER}
    for attempt in attempts:
        for tool_span in attempt.tool_spans:
            bucket = min(int(tool_span.midpoint_frac * n_bins), n_bins - 1)
            counts[tool_bucket_name(tool_span.tool_name)][bucket] += 1
    return counts


def plot_figure(
    cohorts: list[tuple[CohortSpec, list[RealTraceMetrics]]],
    counts: dict[str, np.ndarray],
    output_path: Path,
    n_bins: int,
) -> None:
    fig, (ax_hist, ax_stack) = plt.subplots(1, 2, figsize=(12, 6))

    colors = ["#5DA5DA", "#60BD68", "#B276B2", "#F17CB0"]
    all_ratios: list[float] = []
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
    ax_hist.axvline(
        mean_ratio,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean ({mean_ratio:.1f}%)",
    )
    ax_hist.axvline(
        median_ratio,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label=f"Median ({median_ratio:.1f}%)",
    )
    ax_hist.set_title("(a) Per-Task Tool Ratio (REAL)")
    ax_hist.set_xlabel("Tool Ratio within Tool + LLM Time (%)")
    ax_hist.set_ylabel("Number of Tasks")
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.25)

    x = np.arange(n_bins)
    y = [counts[name] for name in TOOL_ORDER]
    labels = [
        f"{100.0 * i / n_bins:.0f}-{100.0 * (i + 1) / n_bins:.0f}%"
        for i in range(n_bins)
    ]
    ax_stack.stackplot(
        x,
        *y,
        labels=TOOL_ORDER,
        colors=[TOOL_COLORS[name] for name in TOOL_ORDER],
        alpha=0.95,
    )
    ax_stack.set_xticks(x)
    ax_stack.set_xticklabels(labels, rotation=35)
    ax_stack.set_title("(b) Tool Usage Over REAL Execution Timeline")
    ax_stack.set_xlabel("Normalized REAL Execution Time")
    ax_stack.set_ylabel("Tool Call Count")
    ax_stack.legend(loc="upper right")
    ax_stack.grid(True, alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_summary(
    title: str,
    cohorts: list[tuple[CohortSpec, list[RealTraceMetrics]]],
    output_path: Path,
) -> None:
    print(title)
    print("=" * len(title))
    for spec, attempts in cohorts:
        ratios = [attempt.tool_ratio * 100.0 for attempt in attempts]
        print(f"\n[{spec.label}] n={len(attempts)}")
        print(f"  mean ratio (%):   {np.mean(ratios):.2f}")
        print(f"  median ratio (%): {np.median(ratios):.2f}")
        for attempt in attempts:
            print(
                f"    - {attempt.task}: total_real={attempt.total_time_s:.1f}s "
                f"llm={attempt.llm_time_s:.1f}s tool={attempt.tool_time_s:.1f}s "
                f"ratio={attempt.tool_ratio * 100:.2f}%"
            )
    print(f"\nSaved figure to: {output_path}")


def main() -> None:
    args = parse_args()
    cohort_specs = [parse_cohort(raw) for raw in args.cohort]
    cohorts = [(spec, load_real_trace_metrics(spec)) for spec in cohort_specs]
    combined_attempts = [attempt for _, attempts in cohorts for attempt in attempts]
    counts = accumulate_progress_counts(combined_attempts, args.bins)
    output_path = Path(args.output).expanduser().resolve()
    plot_figure(cohorts, counts, output_path, args.bins)
    print_summary(args.title, cohorts, output_path)


if __name__ == "__main__":
    main()
