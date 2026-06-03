#!/usr/bin/env python3
"""Plot execution-time distribution and phase breakdown from results.json files.

Examples:
    conda run -n ML python scripts/plot_execution_profile.py \
      --cohort "Claude Code Haiku=traces/swe-rebench/claude-code-haiku::haiku" \
      --output output/figures/cc_haiku_execution_profile.png

    conda run -n ML python scripts/plot_execution_profile.py \
      --cohort "Claude Code Haiku=traces/swe-rebench/claude-code-haiku::haiku" \
      --cohort "OpenClaw Haiku=tmp/openclaw-two-models/traces/swe-rebench/anthropic-claude-haiku-4.5::anthropic/claude-haiku-4.5" \
      --output output/figures/cc_vs_openclaw_haiku_execution_profile.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class CohortSpec:
    label: str
    root: Path
    model_substring: str | None = None


@dataclass(frozen=True)
class TaskMetrics:
    task: str
    total_time_s: float
    active_time_s: float
    tool_time_s: float
    thinking_time_s: float
    tool_ratio_active: float
    model: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot execution time distribution and tool/thinking phase breakdown.",
    )
    parser.add_argument(
        "--cohort",
        action="append",
        required=True,
        help=(
            "Cohort spec in the form LABEL=PATH or LABEL=PATH::MODEL_SUBSTRING. "
            "The script recursively scans PATH for results.json files."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output PNG file.",
    )
    parser.add_argument(
        "--title",
        default="Execution profile",
        help="Figure title prefix (used only in console output, not rendered on plot).",
    )
    return parser.parse_args()


def parse_cohort(raw: str) -> CohortSpec:
    if "=" not in raw:
        raise ValueError(f"Invalid cohort spec {raw!r}; expected LABEL=PATH[::MODEL]")
    label, remainder = raw.split("=", 1)
    if "::" in remainder:
        path_str, model_substring = remainder.split("::", 1)
    else:
        path_str, model_substring = remainder, None
    root = Path(path_str).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Cohort path does not exist: {root}")
    return CohortSpec(label=label, root=root, model_substring=model_substring or None)


def load_cohort(spec: CohortSpec) -> list[TaskMetrics]:
    results_files = sorted(spec.root.rglob("results.json"))
    metrics: list[TaskMetrics] = []
    for path in results_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        model = data.get("model")
        if spec.model_substring and spec.model_substring not in str(model):
            continue
        total_time = float(data.get("total_time") or 0.0)
        active_time = float(data.get("active_time") or 0.0)
        tool_time = float(data.get("tool_time") or 0.0)
        if total_time <= 0 or active_time <= 0:
            continue
        task = str(data.get("instance_id") or path.parent.parent.name)
        thinking_time = max(0.0, active_time - tool_time)
        tool_ratio = tool_time / active_time if active_time > 0 else 0.0
        metrics.append(
            TaskMetrics(
                task=task,
                total_time_s=total_time,
                active_time_s=active_time,
                tool_time_s=tool_time,
                thinking_time_s=thinking_time,
                tool_ratio_active=tool_ratio,
                model=model,
            )
        )
    if not metrics:
        raise ValueError(f"No matching results.json files found for cohort {spec.label!r}")
    return metrics


def plot_figure(cohorts: list[tuple[CohortSpec, list[TaskMetrics]]], output_path: Path) -> None:
    plt.style.use("default")
    fig, (ax_hist, ax_bar) = plt.subplots(1, 2, figsize=(12, 6))

    colors = ["#5DA5DA", "#60BD68", "#F17CB0", "#B2912F"]
    bin_width_min = 2

    all_minutes = [
        metric.total_time_s / 60.0
        for _, metrics in cohorts
        for metric in metrics
    ]
    max_minutes = max(all_minutes)
    bins = np.arange(0, max_minutes + bin_width_min, bin_width_min)
    if len(bins) < 2:
        bins = np.array([0, 2])

    for idx, (spec, metrics) in enumerate(cohorts):
        minutes = [metric.total_time_s / 60.0 for metric in metrics]
        color = colors[idx % len(colors)]
        ax_hist.hist(minutes, bins=bins, alpha=0.55, color=color, label=f"{spec.label} (n={len(metrics)})")
        mean_min = float(np.mean(minutes))
        median_min = float(np.median(minutes))
        ax_hist.axvline(mean_min, color=color, linestyle="--", linewidth=1.5)
        ax_hist.axvline(median_min, color=color, linestyle=":", linewidth=1.5)

    ax_hist.set_title("(a) Task Execution Time")
    ax_hist.set_xlabel("Execution Time (minutes)")
    ax_hist.set_ylabel("Number of Tasks")
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.25)

    x = np.arange(len(cohorts))
    tool_pct = []
    thinking_pct = []
    tick_labels = []
    for spec, metrics in cohorts:
        total_active = sum(metric.active_time_s for metric in metrics)
        total_tool = sum(metric.tool_time_s for metric in metrics)
        tool_percent = (100.0 * total_tool / total_active) if total_active > 0 else 0.0
        thinking_percent = 100.0 - tool_percent
        tool_pct.append(tool_percent)
        thinking_pct.append(thinking_percent)
        tick_labels.append(f"{spec.label}\n" f"n={len(metrics)}")

    ax_bar.bar(x, tool_pct, width=0.5, color="#4C97D8", label="Tool Execution")
    ax_bar.bar(x, thinking_pct, width=0.5, bottom=tool_pct, color="#F5A623", label="LLM Thinking")
    for idx, (tool_percent, think_percent) in enumerate(zip(tool_pct, thinking_pct, strict=True)):
        ax_bar.text(idx, tool_percent / 2, f"{tool_percent:.1f}%", ha="center", va="center", color="white", fontsize=14, weight="bold")
        ax_bar.text(idx, tool_percent + think_percent / 2, f"{think_percent:.1f}%", ha="center", va="center", color="white", fontsize=14, weight="bold")

    ax_bar.set_title("(b) Execution Phase Breakdown")
    ax_bar.set_ylabel("Percentage of Execution Time (%)")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(tick_labels)
    ax_bar.set_ylim(0, 105)
    ax_bar.legend()
    ax_bar.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_summary(title: str, cohorts: list[tuple[CohortSpec, list[TaskMetrics]]], output_path: Path) -> None:
    print(title)
    print("=" * len(title))
    for spec, metrics in cohorts:
        minutes = [metric.total_time_s / 60.0 for metric in metrics]
        total_active = sum(metric.active_time_s for metric in metrics)
        total_tool = sum(metric.tool_time_s for metric in metrics)
        tool_percent = (100.0 * total_tool / total_active) if total_active > 0 else 0.0
        print(f"\n[{spec.label}] n={len(metrics)}")
        print(f"  mean(min):   {np.mean(minutes):.2f}")
        print(f"  median(min): {np.median(minutes):.2f}")
        print(f"  tool%:       {tool_percent:.2f}")
        print("  tasks:")
        for metric in metrics:
            print(
                f"    - {metric.task}: total={metric.total_time_s:.1f}s "
                f"active={metric.active_time_s:.1f}s tool={metric.tool_time_s:.1f}s "
                f"ratio={metric.tool_ratio_active*100:.2f}%"
            )
    print(f"\nSaved figure to: {output_path}")


def main() -> None:
    args = parse_args()
    cohort_specs = [parse_cohort(raw) for raw in args.cohort]
    cohorts = [(spec, load_cohort(spec)) for spec in cohort_specs]
    output_path = Path(args.output).expanduser().resolve()
    plot_figure(cohorts, output_path)
    print_summary(args.title, cohorts, output_path)


if __name__ == "__main__":
    main()
