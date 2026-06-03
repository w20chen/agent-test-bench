#!/usr/bin/env python3
"""Plot REAL-mode execution-time distribution and tool/LLM breakdown.

Examples:
    conda run -n ML python scripts/figures/plot_execution_profile_real.py \
      --cohort "OpenClaw GLM Arm=traces/swe-rebench/swe-rebench-arm-10-tasks-openclaw-glm-docker-100::z-ai/glm-5.1" \
      --output output/figures/openclaw_glm_arm_execution_profile_real.png
"""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot REAL-mode execution profile using compacted LLM latency.",
    )
    parser.add_argument(
        "--cohort",
        action="append",
        required=True,
        help="Cohort spec in the form LABEL=PATH or LABEL=PATH::MODEL_SUBSTRING.",
    )
    parser.add_argument("--output", required=True, help="Path to the output PNG file.")
    parser.add_argument(
        "--title",
        default="Execution profile (REAL mode)",
        help="Console title only.",
    )
    return parser.parse_args()


def plot_figure(
    cohorts: list[tuple[CohortSpec, list[RealTraceMetrics]]], output_path: Path
) -> None:
    plt.style.use("default")
    fig, (ax_hist, ax_bar) = plt.subplots(1, 2, figsize=(12, 6))

    colors = ["#5DA5DA", "#60BD68", "#F17CB0", "#B2912F"]
    bin_width_min = 1
    all_minutes = [
        metric.total_time_s / 60.0 for _, metrics in cohorts for metric in metrics
    ]
    max_minutes = max(all_minutes)
    bins = np.arange(0, max_minutes + bin_width_min, bin_width_min)
    if len(bins) < 2:
        bins = np.array([0, 1])

    for idx, (spec, metrics) in enumerate(cohorts):
        minutes = [metric.total_time_s / 60.0 for metric in metrics]
        color = colors[idx % len(colors)]
        ax_hist.hist(
            minutes,
            bins=bins,
            alpha=0.55,
            color=color,
            label=f"{spec.label} (n={len(metrics)})",
        )
        ax_hist.axvline(
            float(np.mean(minutes)), color=color, linestyle="--", linewidth=1.5
        )
        ax_hist.axvline(
            float(np.median(minutes)), color=color, linestyle=":", linewidth=1.5
        )

    ax_hist.set_title("(a) Task Execution Time (REAL)")
    ax_hist.set_xlabel("Execution Time (minutes)")
    ax_hist.set_ylabel("Number of Tasks")
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.25)

    x = np.arange(len(cohorts))
    tool_pct: list[float] = []
    llm_pct: list[float] = []
    tick_labels: list[str] = []
    for spec, metrics in cohorts:
        total_tool = sum(metric.tool_time_s for metric in metrics)
        total_llm = sum(metric.llm_time_s for metric in metrics)
        active_total = total_tool + total_llm
        tool_percent = (100.0 * total_tool / active_total) if active_total > 0 else 0.0
        llm_percent = (100.0 * total_llm / active_total) if active_total > 0 else 0.0
        tool_pct.append(tool_percent)
        llm_pct.append(llm_percent)
        tick_labels.append(f"{spec.label}\nn={len(metrics)}")

    ax_bar.bar(x, tool_pct, width=0.5, color="#4C97D8", label="Tool Execution")
    ax_bar.bar(
        x,
        llm_pct,
        width=0.5,
        bottom=tool_pct,
        color="#F5A623",
        label="LLM Call Latency",
    )
    for idx, (tool_percent, llm_percent) in enumerate(
        zip(tool_pct, llm_pct, strict=True)
    ):
        ax_bar.text(
            idx,
            tool_percent / 2,
            f"{tool_percent:.1f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=14,
            weight="bold",
        )
        ax_bar.text(
            idx,
            tool_percent + llm_percent / 2,
            f"{llm_percent:.1f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=14,
            weight="bold",
        )

    ax_bar.set_title("(b) Active Phase Breakdown (REAL)")
    ax_bar.set_ylabel("Percentage of Tool + LLM Time (%)")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(tick_labels)
    ax_bar.set_ylim(0, 105)
    ax_bar.legend()
    ax_bar.grid(True, axis="y", alpha=0.25)

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
    for spec, metrics in cohorts:
        minutes = [metric.total_time_s / 60.0 for metric in metrics]
        total_tool = sum(metric.tool_time_s for metric in metrics)
        total_llm = sum(metric.llm_time_s for metric in metrics)
        active_total = total_tool + total_llm
        tool_percent = (100.0 * total_tool / active_total) if active_total > 0 else 0.0
        print(f"\n[{spec.label}] n={len(metrics)}")
        print(f"  mean(min):   {np.mean(minutes):.2f}")
        print(f"  median(min): {np.median(minutes):.2f}")
        print(f"  tool%:       {tool_percent:.2f}")
        print("  tasks:")
        for metric in metrics:
            print(
                f"    - {metric.task}: total_real={metric.total_time_s:.1f}s "
                f"llm={metric.llm_time_s:.1f}s tool={metric.tool_time_s:.1f}s "
                f"other={metric.other_time_s:.1f}s ratio={metric.tool_ratio * 100:.2f}%"
            )
    print(f"\nSaved figure to: {output_path}")


def main() -> None:
    args = parse_args()
    cohort_specs = [parse_cohort(raw) for raw in args.cohort]
    cohorts = [(spec, load_real_trace_metrics(spec)) for spec in cohort_specs]
    output_path = Path(args.output).expanduser().resolve()
    plot_figure(cohorts, output_path)
    print_summary(args.title, cohorts, output_path)


if __name__ == "__main__":
    main()
