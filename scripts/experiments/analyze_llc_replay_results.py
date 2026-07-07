#!/usr/bin/env python3
"""Analyze LLC replay experiment results and generate report-ready plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


LABELS = {
    "os_default": "OS default",
    "compact_llc": "Compact LLC",
    "spread_llc": "Spread LLC",
    "compact_cluster": "Compact inferred CCL cluster",
    "compact_clusters_same_llc": "Packed inferred CCL clusters, same Linux LLC",
    "spread_clusters_same_llc": "Inferred CCL cluster spread, same Linux LLC",
    "spread_clusters_all": "Inferred CCL cluster spread, all Linux LLCs",
    "near_numa_spread": "Near NUMA spread",
    "far_numa_spread": "Far NUMA spread",
}
COLORS = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
)


def _label(placement: str) -> str:
    return LABELS.get(placement, placement.replace("_", " ").title())


def _color(index: int) -> str:
    return COLORS[index % len(COLORS)]


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - idx) + ordered[upper] * (idx - lower)


def _mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _action_duration_s(action: dict[str, Any], key: str) -> float:
    data = action.get("data") or {}
    explicit = data.get(key)
    if explicit is not None:
        return float(explicit) / 1000.0
    start = float(action.get("ts_start") or 0.0)
    end = float(action.get("ts_end") or start)
    return max(0.0, end - start)


def _load_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "experiment_manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def discover_placements(root: Path) -> list[str]:
    """Return placement names from manifest when available, else directories."""
    manifest = _load_manifest(root)
    runs = manifest.get("runs") or []
    names = [
        str(run["placement"])
        for run in runs
        if isinstance(run, dict) and run.get("placement")
    ]
    if not names:
        names = [
            child.name
            for child in sorted(root.iterdir())
            if child.is_dir() and list(child.glob("*/attempt_1/trace.jsonl"))
        ]
    ordered: list[str] = []
    for name in names:
        if name not in ordered:
            ordered.append(name)
    return ordered


def _load_trace_row(placement: str, trace_path: Path) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "action":
                actions.append(record)

    timestamps = [
        float(value)
        for action in actions
        for value in (action.get("ts_start"), action.get("ts_end"))
        if isinstance(value, (int, float))
    ]
    elapsed_s = max(timestamps) - min(timestamps) if timestamps else 0.0

    tool_s = 0.0
    llm_s = 0.0
    exec_tool_s = 0.0
    pytest_s = 0.0
    n_tools = 0
    for action in actions:
        data = action.get("data") or {}
        if action.get("action_type") == "tool_exec":
            duration = _action_duration_s(action, "duration_ms")
            tool_s += duration
            n_tools += 1
            tool_name = str(data.get("tool_name") or "")
            if tool_name.startswith("exec-"):
                exec_tool_s += duration
            if tool_name == "exec-pytest":
                pytest_s += duration
        elif action.get("action_type") == "llm_call":
            llm_s += _action_duration_s(action, "llm_latency_ms")

    cpu_values: list[float] = []
    resources_path = trace_path.with_name("resources.json")
    if resources_path.exists():
        resources = json.loads(resources_path.read_text(encoding="utf-8"))
        for sample in resources.get("samples", []):
            try:
                cpu_values.append(float(str(sample.get("cpu_percent", "0")).rstrip("%")))
            except (TypeError, ValueError):
                continue

    return {
        "placement": placement,
        "agent": trace_path.parents[1].name,
        "attempt": trace_path.parent.name,
        "elapsed_s": elapsed_s,
        "tool_s": tool_s,
        "llm_s": llm_s,
        "exec_tool_s": exec_tool_s,
        "pytest_s": pytest_s,
        "n_tools": n_tools,
        "cpu_mean_pct": _mean_or_none(cpu_values),
        "cpu_p95_pct": _percentile(cpu_values, 0.95) if cpu_values else None,
    }


def load_rows(root: Path, placements: list[str] | None = None) -> list[dict[str, Any]]:
    selected = placements or discover_placements(root)
    rows: list[dict[str, Any]] = []
    for placement in selected:
        placement_dir = root / placement
        if not placement_dir.exists():
            continue
        for trace_path in sorted(placement_dir.glob("*/attempt_1/trace.jsonl")):
            rows.append(_load_trace_row(placement, trace_path))
    if not rows:
        raise ValueError(f"no trace.jsonl files found under {root}")
    return rows


def summarize(
    rows: list[dict[str, Any]],
    placements: list[str] | None = None,
) -> list[dict[str, Any]]:
    selected = placements or []
    if not selected:
        for row in rows:
            placement = str(row["placement"])
            if placement not in selected:
                selected.append(placement)

    summary: list[dict[str, Any]] = []
    for placement in selected:
        group = [row for row in rows if row["placement"] == placement]
        if not group:
            continue

        def values(key: str) -> list[float]:
            return [float(row[key]) for row in group if row.get(key) is not None]

        summary.append(
            {
                "placement": placement,
                "agent_count": len(group),
                "agent_elapsed_mean_s": statistics.mean(values("elapsed_s")),
                "agent_elapsed_p95_s": _percentile(values("elapsed_s"), 0.95),
                "tool_mean_s": statistics.mean(values("tool_s")),
                "tool_p95_agent_s": _percentile(values("tool_s"), 0.95),
                "llm_mean_s": statistics.mean(values("llm_s")),
                "pytest_mean_s": statistics.mean(values("pytest_s")),
                "cpu_mean_pct": _mean_or_none(values("cpu_mean_pct")),
                "cpu_p95_pct": _mean_or_none(values("cpu_p95_pct")),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summary_by_placement(summary: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["placement"]): row for row in summary}


def write_notes(root: Path, summary: list[dict[str, Any]]) -> None:
    manifest = _load_manifest(root)
    source_trace = manifest.get("source_trace", "unknown")
    num_agents = manifest.get("num_agents", "unknown")
    replay_speed = manifest.get("replay_speed", "unknown")
    cluster_size = manifest.get("cluster_size", "unknown")

    ordered = sorted(summary, key=lambda row: float(row["tool_mean_s"]))
    lines = [
        "# LLC Replay Result Notes",
        "",
        f"Source trace: `{source_trace}`",
        f"Agents per placement: `{num_agents}`",
        f"Replay speed: `{replay_speed}`",
        f"Cluster size: `{cluster_size}`",
        "",
        "This analysis is descriptive. It reports observed replay timings and does not infer causality by itself.",
        "Cluster placements use inferred 4-core CCL groups from topology ordering; they are not treated as verified hardware LLC slices unless a separate latency/counter probe confirms the mapping.",
        "",
        "| Placement | Agents | Mean tool s | p95 tool s | Mean LLM s | Mean CPU % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        cpu_mean = row["cpu_mean_pct"]
        cpu_text = "" if cpu_mean is None else f"{float(cpu_mean):.1f}"
        lines.append(
            "| "
            f"{row['placement']} | "
            f"{row['agent_count']} | "
            f"{float(row['tool_mean_s']):.3f} | "
            f"{float(row['tool_p95_agent_s']):.3f} | "
            f"{float(row['llm_mean_s']):.3f} | "
            f"{cpu_text} |"
        )

    if len(ordered) >= 2:
        fastest = ordered[0]
        slowest = ordered[-1]
        delta = float(slowest["tool_mean_s"]) - float(fastest["tool_mean_s"])
        lines.extend(
            [
                "",
                (
                    "Largest mean-tool gap: "
                    f"{slowest['placement']} minus {fastest['placement']} = "
                    f"{delta:.3f}s per replayed agent."
                ),
            ]
        )

    if any((run.get("agent_assignments") for run in manifest.get("runs", []) if isinstance(run, dict))):
        lines.extend(
            [
                "",
                "Per-agent CPU assignments are recorded in `experiment_manifest.json` and each placement `run_config.json`.",
            ]
        )

    (root / "analysis_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(root: Path, rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    placements = [str(row["placement"]) for row in summary]
    plot_placements = [name for name in placements if name != "os_default"] or placements
    by_placement = _summary_by_placement(summary)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(12.5, 1.8 * len(plot_placements)), 4.8),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )

    ax = axes[0]
    positions = np.arange(len(plot_placements))
    all_tool_values: list[float] = []
    for idx, placement in enumerate(plot_placements):
        values = [float(row["tool_s"]) for row in rows if row["placement"] == placement]
        if not values:
            continue
        all_tool_values.extend(values)
        jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else [0.0]
        ax.scatter(
            np.full(len(values), idx) + jitter,
            values,
            color=_color(idx),
            s=42,
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
        ax.boxplot(
            values,
            positions=[idx],
            widths=0.42,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": _color(idx), "alpha": 0.18, "edgecolor": _color(idx)},
            medianprops={"color": "black", "linewidth": 1.5},
            whiskerprops={"color": _color(idx)},
            capprops={"color": _color(idx)},
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([_label(p) for p in plot_placements], rotation=20, ha="right")
    ax.set_ylim(0, max(all_tool_values) * 1.18 if all_tool_values else 1.0)
    ax.set_ylabel("CPU tool time per replayed agent (s)")
    ax.set_title("Per-agent tool phase")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    metrics = [("tool_mean_s", "Mean"), ("tool_p95_agent_s", "p95")]
    width = min(0.8 / max(1, len(plot_placements)), 0.22)
    x = np.arange(len(metrics))
    max_bar = 0.0
    for idx, placement in enumerate(plot_placements):
        values = [float(by_placement[placement][key]) for key, _label_text in metrics]
        max_bar = max(max_bar, *values)
        offset = (idx - (len(plot_placements) - 1) / 2) * width
        ax.bar(
            x + offset,
            values,
            width,
            color=_color(idx),
            label=_label(placement),
            edgecolor="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _key, label in metrics])
    ax.set_ylim(0, max_bar * 1.16 if max_bar else 1.0)
    ax.set_ylabel("CPU tool time (s)")
    ax.set_title("Tool-phase summary")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("CPU Tool Phase by Placement", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        -0.02,
        "Only re-executed tool time is plotted; replayed LLM latency is reported separately in CSV.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(root / "analysis_llc_tool_phase.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument(
        "--placements",
        default=None,
        help="Optional comma-separated placement names. Defaults to manifest or result directories.",
    )
    args = parser.parse_args()

    root = args.result_dir.resolve()
    placements = (
        [item.strip() for item in args.placements.split(",") if item.strip()]
        if args.placements
        else discover_placements(root)
    )
    rows = load_rows(root, placements)
    summary = summarize(rows, placements)
    write_csv(root / "analysis_per_agent.csv", rows)
    write_csv(root / "analysis_summary.csv", summary)
    write_notes(root, summary)
    write_plots(root, rows, summary)

    for path in (
        root / "analysis_summary.csv",
        root / "analysis_per_agent.csv",
        root / "analysis_notes.md",
        root / "analysis_llc_tool_phase.png",
    ):
        print(path)


if __name__ == "__main__":
    main()
