#!/usr/bin/env python3
"""Plot resource utilization aligned with agent phases (tool_exec vs llm_call).

Reads a v5 simulate trace JSONL + per-task resources.json to produce
timeline charts with phase-colored backgrounds and resource metrics.

Examples:
    python scripts/figures/plot_resource_phase_alignment.py \
      --trace-dir traces/simulate/run_dir \
      --output output/figures/phase_resource

    python scripts/figures/plot_resource_phase_alignment.py \
      --trace-dir traces/simulate/run_dir \
      --metric memory \
      --output output/figures/phase_resource_mem
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Phase:
    action_type: str  # "tool_exec" or "llm_call"
    ts_start: float
    ts_end: float


@dataclass(frozen=True)
class ResourceSample:
    epoch: float
    cpu_percent: float
    memory_mb: float
    disk_read_bytes: float = 0.0
    disk_write_bytes: float = 0.0
    net_rx_bytes: float = 0.0
    net_tx_bytes: float = 0.0


@dataclass
class TaskData:
    agent_id: str
    phases: list[Phase] = field(default_factory=list)
    samples: list[ResourceSample] = field(default_factory=list)


PHASE_COLORS = {
    "tool_exec": "#FF6D0030",
    "llm_call": "#00E5FF20",
}
PHASE_LABELS = {
    "tool_exec": "Tool Exec",
    "llm_call": "LLM Call",
}

METRIC_EXTRACTORS: dict[str, tuple[str, str]] = {
    "cpu": ("CPU %", "cpu_percent"),
    "memory": ("Memory (MB)", "memory_mb"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase × resource alignment analysis.",
    )
    parser.add_argument(
        "--trace-dir",
        required=True,
        help="Directory containing simulate output (*.jsonl + {task}/attempt_1/resources.json).",
    )
    parser.add_argument(
        "--metric",
        default="cpu",
        choices=list(METRIC_EXTRACTORS.keys()),
        help="Resource metric to plot (default: cpu).",
    )
    parser.add_argument(
        "--output",
        default="output/figures/phase_resource",
        help="Output path prefix (per-task PNGs appended with task name).",
    )
    return parser.parse_args()


def find_trace_jsonl(trace_dir: Path) -> Path | None:
    """Find the simulate JSONL file in the trace directory."""
    candidates = sorted(trace_dir.glob("simulate_*.jsonl"))
    if candidates:
        return candidates[-1]
    candidates = sorted(trace_dir.glob("*.jsonl"))
    return candidates[-1] if candidates else None


def load_actions_by_agent(trace_jsonl: Path) -> dict[str, list[dict[str, Any]]]:
    """Parse v5 trace JSONL, group action records by agent_id."""
    agents: dict[str, list[dict[str, Any]]] = {}
    with trace_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") != "action":
                continue
            agent_id = record.get("agent_id", "")
            if agent_id:
                agents.setdefault(agent_id, []).append(record)
    for actions in agents.values():
        actions.sort(key=lambda a: a.get("ts_start", 0))
    return agents


def build_phases(actions: list[dict[str, Any]]) -> list[Phase]:
    """Extract L1 phases from sorted actions."""
    phases = []
    for act in actions:
        action_type = act.get("action_type")
        if action_type not in ("tool_exec", "llm_call"):
            continue
        ts_start = act.get("ts_start", 0)
        ts_end = act.get("ts_end", ts_start)
        if ts_end > ts_start:
            phases.append(Phase(action_type=action_type, ts_start=ts_start, ts_end=ts_end))
    return phases


def load_resource_samples(resources_path: Path) -> list[ResourceSample]:
    """Load samples from resources.json."""
    with resources_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    samples = []
    for s in data.get("samples", []):
        epoch = s.get("epoch")
        if epoch is None:
            continue
        cpu_raw = s.get("cpu_percent", "0")
        cpu = float(cpu_raw.rstrip("% ")) if isinstance(cpu_raw, str) else float(cpu_raw or 0)

        mem_raw = s.get("mem_usage", "0")
        if isinstance(mem_raw, str):
            left = mem_raw.split("/")[0].strip()
            # Extract numeric value (assume MB for simplicity)
            import re
            m = re.match(r"([\d.]+)\s*(B|[kKMGT]i?B)?", left)
            mem_mb = float(m.group(1)) if m else 0.0
            unit = m.group(2) if m else "MB"
            multipliers = {"B": 1e-6, "kB": 1e-3, "KB": 1e-3, "KiB": 1/1024,
                           "MB": 1, "MiB": 1, "GB": 1000, "GiB": 1024,
                           "TB": 1e6, "TiB": 1024**2}
            mem_mb *= multipliers.get(unit, 1)
        else:
            mem_mb = float(s.get("memory_mb", 0) or 0)

        samples.append(ResourceSample(
            epoch=float(epoch),
            cpu_percent=cpu,
            memory_mb=mem_mb,
            disk_read_bytes=float(s.get("disk_read_bytes", 0) or 0),
            disk_write_bytes=float(s.get("disk_write_bytes", 0) or 0),
            net_rx_bytes=float(s.get("net_rx_bytes", 0) or 0),
            net_tx_bytes=float(s.get("net_tx_bytes", 0) or 0),
        ))
    samples.sort(key=lambda s: s.epoch)
    return samples


def load_task_data(
    trace_dir: Path,
    trace_jsonl: Path,
) -> list[TaskData]:
    """Build per-task data from trace + resources."""
    agents = load_actions_by_agent(trace_jsonl)
    tasks = []
    for agent_id, actions in agents.items():
        phases = build_phases(actions)
        resources_path = trace_dir / agent_id / "attempt_1" / "resources.json"
        samples = load_resource_samples(resources_path) if resources_path.exists() else []
        tasks.append(TaskData(agent_id=agent_id, phases=phases, samples=samples))
    return tasks


def compute_phase_stats(
    task: TaskData,
    metric_attr: str,
) -> dict[str, dict[str, float]]:
    """Compute per-phase aggregate stats for a metric."""
    phase_values: dict[str, list[float]] = {"tool_exec": [], "llm_call": []}

    for sample in task.samples:
        val = getattr(sample, metric_attr)
        for phase in task.phases:
            if phase.ts_start <= sample.epoch <= phase.ts_end:
                phase_values[phase.action_type].append(val)
                break

    stats: dict[str, dict[str, float]] = {}
    for ptype, values in phase_values.items():
        if values:
            arr = np.array(values)
            stats[ptype] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "p95": float(np.percentile(arr, 95)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "n": len(values),
            }
        else:
            stats[ptype] = {"mean": 0, "median": 0, "p95": 0, "min": 0, "max": 0, "n": 0}
    return stats


def plot_task(
    task: TaskData,
    metric_name: str,
    metric_attr: str,
    output_path: Path,
) -> None:
    """Plot one task's resource timeline with phase background."""
    if not task.samples:
        print(f"  SKIP {task.agent_id}: no resource samples")
        return

    t0 = task.phases[0].ts_start if task.phases else task.samples[0].epoch
    times = [s.epoch - t0 for s in task.samples]
    values = [getattr(s, metric_attr) for s in task.samples]

    fig, ax = plt.subplots(figsize=(14, 3.5))

    # Phase backgrounds
    for phase in task.phases:
        color = PHASE_COLORS.get(phase.action_type, "#00000010")
        ax.axvspan(phase.ts_start - t0, phase.ts_end - t0,
                   alpha=1, color=color, zorder=0)

    # Resource line
    ax.plot(times, values, linewidth=0.8, color="#00E5FF", zorder=2)
    ax.fill_between(times, values, alpha=0.15, color="#00E5FF", zorder=1)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{task.agent_id} — {metric_name} × Phase", fontsize=10)
    ax.set_xlim(left=0)

    # Legend for phases
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PHASE_COLORS["tool_exec"], label="Tool Exec"),
        Patch(facecolor=PHASE_COLORS["llm_call"], label="LLM Call"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main() -> None:
    args = parse_args()
    trace_dir = Path(args.trace_dir)
    metric_name, metric_attr = METRIC_EXTRACTORS[args.metric]
    output_prefix = Path(args.output)

    trace_jsonl = find_trace_jsonl(trace_dir)
    if trace_jsonl is None:
        print(f"ERROR: No JSONL trace found in {trace_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Trace: {trace_jsonl}")
    print(f"Metric: {metric_name}")

    tasks = load_task_data(trace_dir, trace_jsonl)
    if not tasks:
        print("ERROR: No tasks found in trace", file=sys.stderr)
        sys.exit(1)

    print(f"Tasks: {len(tasks)}")
    print()

    # Per-task charts
    for task in tasks:
        safe_name = task.agent_id.replace("/", "_").replace(":", "_")
        plot_task(task, metric_name, metric_attr, output_prefix / f"{safe_name}.png")

    # Summary table
    print(f"\n{'='*80}")
    print(f"Phase × {metric_name} Summary")
    print(f"{'='*80}")
    print(f"{'Task':<45} {'Tool(mean)':>10} {'LLM(mean)':>10} {'Ratio':>8}")
    print(f"{'-'*45} {'-'*10} {'-'*10} {'-'*8}")

    ratios = []
    for task in tasks:
        stats = compute_phase_stats(task, metric_attr)
        tool_mean = stats["tool_exec"]["mean"]
        llm_mean = stats["llm_call"]["mean"]
        ratio = tool_mean / llm_mean if llm_mean > 0 else float("inf")
        ratios.append(ratio)
        short_id = task.agent_id[:44]
        print(f"{short_id:<45} {tool_mean:>10.2f} {llm_mean:>10.2f} {ratio:>8.2f}x")

    if ratios:
        finite_ratios = [r for r in ratios if r != float("inf")]
        if finite_ratios:
            print(f"\nAggregate peak/trough ratio: "
                  f"mean={np.mean(finite_ratios):.2f}x "
                  f"median={np.median(finite_ratios):.2f}x "
                  f"range=[{min(finite_ratios):.2f}x, {max(finite_ratios):.2f}x]")


if __name__ == "__main__":
    main()
