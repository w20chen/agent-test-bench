"""Generate tool-call duration boxplots from recorded trace summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = ROOT / "reports" / "trace_case_ppt" / "scratch" / "case_summary.json"
OUTPUT_PATH = (
    ROOT
    / "reports"
    / "trace_case_ppt"
    / "output"
    / "all_benchmarks_tool_call_duration_boxplots.png"
)

BENCHMARK_ORDER = [
    "SWE-bench Verified",
    "SWE-rebench",
    "Terminal-Bench",
    "DeepResearch-Bench",
]

DISPLAY_LABELS = {
    "SWE-bench Verified": "SWE-bench Verified",
    "SWE-rebench": "SWE-rebench",
    "Terminal-Bench": "Terminal Bench",
    "DeepResearch-Bench": "Deep Research",
}

ORDERED_COLORS = [
    "#a9cdf5",
    "#f5cf83",
    "#9be2ca",
    "#c7b5f2",
]

LOG_FLOOR_S = 1e-3


def load_tool_durations_by_benchmark(summary_path: Path) -> dict[str, list[float]]:
    """Return recorded tool-call durations grouped by benchmark."""
    with summary_path.open("r", encoding="utf-8") as f:
        summary: dict[str, Any] = json.load(f)

    durations_by_benchmark = {benchmark: [] for benchmark in BENCHMARK_ORDER}
    for case in summary.get("cases", []):
        benchmark = case.get("benchmark")
        if benchmark not in durations_by_benchmark:
            raise ValueError(f"Unexpected benchmark in summary: {benchmark!r}")
        for span in case.get("timeline", []):
            if span.get("kind") != "tool":
                continue
            duration_s = float(span.get("duration_s", 0.0))
            durations_by_benchmark[benchmark].append(max(duration_s, LOG_FLOOR_S))

    missing = [
        benchmark
        for benchmark, durations in durations_by_benchmark.items()
        if not durations
    ]
    if missing:
        raise ValueError(f"No tool-call durations found for: {missing}")
    return durations_by_benchmark


def plot_boxplots(durations_by_benchmark: dict[str, list[float]], output_path: Path) -> None:
    """Write the all-benchmark tool duration boxplot figure."""
    data = [durations_by_benchmark[benchmark] for benchmark in BENCHMARK_ORDER]
    labels = [DISPLAY_LABELS[benchmark] for benchmark in BENCHMARK_ORDER]

    fig, ax = plt.subplots(figsize=(12.49, 6.74), dpi=100)
    boxplot = ax.boxplot(
        data,
        patch_artist=True,
        tick_labels=labels,
        showfliers=False,
        widths=0.5,
        medianprops={"color": "#111827", "linewidth": 2.5},
        boxprops={"edgecolor": "#4b5563", "linewidth": 1.7},
        whiskerprops={"color": "#4b5563", "linewidth": 1.7},
        capprops={"color": "#4b5563", "linewidth": 1.7},
    )

    for patch, color in zip(boxplot["boxes"], ORDERED_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.set_yscale("log")
    ax.set_ylim(6e-4, 80)
    ax.set_title(
        "Tool-call duration distributions by benchmark",
        fontsize=29,
        fontweight="bold",
        color="#111827",
        pad=20,
    )
    ax.set_ylabel("Duration per tool call (seconds, log scale)", fontsize=22)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#111827")
    ax.spines["bottom"].set_color("#111827")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.grid(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    durations_by_benchmark = load_tool_durations_by_benchmark(SUMMARY_PATH)
    plot_boxplots(durations_by_benchmark, OUTPUT_PATH)


if __name__ == "__main__":
    main()
