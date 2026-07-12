#!/usr/bin/env python3
"""Analyze topology-derived Kunpeng LLC scaling replay outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - idx) + ordered[upper] * (idx - lower)


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _read_trace(trace_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "summary":
                summaries.append(record)
            elif record.get("type") == "action":
                actions.append(record)
    return summaries, actions


def _resource_summary(resources_path: Path) -> dict[str, float | bool | str | None]:
    if not resources_path.exists():
        return {}
    payload = _read_json(resources_path)
    summary = payload.get("summary") or {}

    def nested(section: str, key: str) -> float | None:
        value = summary.get(section)
        if not isinstance(value, dict):
            return None
        item = value.get(key)
        return float(item) if isinstance(item, (int, float)) else None

    return {
        "resource_duration_s": (
            float(summary["duration_seconds"])
            if isinstance(summary.get("duration_seconds"), (int, float))
            else None
        ),
        "cpu_avg_pct": nested("cpu_percent", "avg"),
        "cpu_max_pct": nested("cpu_percent", "max"),
        "mem_avg_mb": nested("memory_mb", "avg"),
        "mem_max_mb": nested("memory_mb", "max"),
        "memory_bandwidth_available": bool(summary.get("memory_bandwidth_available")),
        "memory_bandwidth_reason": summary.get("memory_bandwidth_reason"),
    }


def _run_dirs(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.glob("n*/*"))
        if path.is_dir() and (path / "run_config.json").exists()
    ]


def analyze(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in _run_dirs(root):
        config = _read_json(run_dir / "run_config.json")
        agent_count = int(config.get("agent_count") or run_dir.parent.name.lstrip("n"))
        placement = str(config.get("placement") or run_dir.name)
        trace_paths = sorted(run_dir.glob("*/attempt_1/trace.jsonl"))
        resource_paths = sorted(run_dir.glob("*/attempt_1/resources.json"))

        summaries: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        for trace_path in trace_paths:
            trace_summaries, trace_actions = _read_trace(trace_path)
            summaries.extend(trace_summaries)
            actions.extend(trace_actions)

        elapsed_s: list[float] = []
        agent_exec_s: list[float] = []
        setup_s: list[float] = []
        total_tool_ms: list[float] = []
        total_llm_ms: list[float] = []
        failed_actions = 0
        succeeded_actions = 0
        for summary in summaries:
            if isinstance(summary.get("elapsed_s"), (int, float)):
                elapsed_s.append(float(summary["elapsed_s"]))
            timing = summary.get("timing") or {}
            if isinstance(timing.get("agent_exec_s"), (int, float)):
                agent_exec_s.append(float(timing["agent_exec_s"]))
            if isinstance(timing.get("container_setup_s"), (int, float)):
                setup_s.append(float(timing["container_setup_s"]))
            if isinstance(summary.get("total_tool_ms"), (int, float)):
                total_tool_ms.append(float(summary["total_tool_ms"]))
            if isinstance(summary.get("total_llm_ms"), (int, float)):
                total_llm_ms.append(float(summary["total_llm_ms"]))
            failed_actions += int(summary.get("failed_actions") or 0)
            succeeded_actions += int(summary.get("succeeded_actions") or 0)

        tool_actions = [a for a in actions if a.get("action_type") == "tool_exec"]
        llm_actions = [a for a in actions if a.get("action_type") == "llm_call"]
        tool_by_name: dict[str, list[float]] = {}
        for action in tool_actions:
            data = action.get("data") or {}
            name = str(data.get("tool_name") or "unknown")
            duration = float(data.get("duration_ms") or 0.0)
            tool_by_name.setdefault(name, []).append(duration)

        resource_rows = [_resource_summary(path) for path in resource_paths]

        def resource_values(key: str) -> list[float]:
            return [
                float(row[key])
                for row in resource_rows
                if isinstance(row.get(key), (int, float))
            ]

        memory_bandwidth_available = any(
            row.get("memory_bandwidth_available") is True for row in resource_rows
        )
        memory_bandwidth_reasons = sorted(
            {
                str(row["memory_bandwidth_reason"])
                for row in resource_rows
                if row.get("memory_bandwidth_reason")
            }
        )

        complete = (
            config.get("returncode") == 0
            and len(summaries) == agent_count
            and failed_actions == 0
        )
        top_tools = ";".join(
            f"{name}:{sum(values):.0f}ms/{len(values)}"
            for name, values in sorted(
                tool_by_name.items(),
                key=lambda item: sum(item[1]),
                reverse=True,
            )[:5]
        )

        rows.append(
            {
                "agent_count": agent_count,
                "placement": placement,
                "complete": complete,
                "returncode": config.get("returncode"),
                "agents_observed": len(summaries),
                "agents_expected": agent_count,
                "trace_files": len(trace_paths),
                "resource_files": len(resource_paths),
                "cpus": (
                    ",".join(str(cpu) for cpu in config.get("cpus") or [])
                    if config.get("cpus") is not None
                    else "os_default"
                ),
                "llc_ids": "|".join(str(llc) for llc in config.get("llc_ids") or []),
                "wall_s": max(elapsed_s) if elapsed_s else None,
                "agent_elapsed_mean_s": _mean(elapsed_s),
                "agent_elapsed_p95_s": _percentile(elapsed_s, 0.95),
                "agent_exec_mean_s": _mean(agent_exec_s),
                "container_setup_mean_s": _mean(setup_s),
                "tool_total_mean_ms": _mean(total_tool_ms),
                "tool_total_p95_ms": _percentile(total_tool_ms, 0.95),
                "llm_total_mean_ms": _mean(total_llm_ms),
                "failed_actions": failed_actions,
                "succeeded_actions": succeeded_actions,
                "tool_actions": len(tool_actions),
                "llm_actions": len(llm_actions),
                "cpu_avg_pct": _mean(resource_values("cpu_avg_pct")),
                "cpu_max_pct": _mean(resource_values("cpu_max_pct")),
                "mem_avg_mb": _mean(resource_values("mem_avg_mb")),
                "mem_max_mb": _mean(resource_values("mem_max_mb")),
                "memory_bandwidth_available": memory_bandwidth_available,
                "memory_bandwidth_reason": ";".join(memory_bandwidth_reasons),
                "top_tools": top_tools,
                "run_dir": str(run_dir),
            }
        )
    return rows


def _format_float(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_notes(path: Path, root: Path, rows: list[dict[str, Any]]) -> None:
    manifest_path = root / "experiment_manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    complete = [row for row in rows if row["complete"]]
    incomplete = [row for row in rows if not row["complete"]]

    lines = [
        "# LLC Scaling Analysis",
        "",
        f"Experiment root: `{root}`",
        f"Source trace: `{manifest.get('source_trace', 'unknown')}`",
        f"Agent counts: `{manifest.get('agent_counts', 'unknown')}`",
        f"Replay speed: `{manifest.get('replay_speed', 'unknown')}`",
        f"Cluster size: `{manifest.get('cluster_size', 'unknown')}`",
        "",
        f"Complete runs: **{len(complete)} / {len(rows)}**",
    ]
    if incomplete:
        lines.extend(["", "Incomplete runs:"])
        for row in incomplete:
            lines.append(
                f"- n={row['agent_count']} `{row['placement']}`: "
                f"returncode={row['returncode']}, "
                f"agents={row['agents_observed']}/{row['agents_expected']}"
            )

    lines.extend(
        [
            "",
            "## Timing Summary",
            "",
            "| n | placement | wall s | mean tool ms | mean LLM ms | failed |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(complete, key=lambda r: (r["agent_count"], str(r["placement"]))):
        lines.append(
            "| "
            f"{row['agent_count']} | "
            f"{row['placement']} | "
            f"{_format_float(row['wall_s'])} | "
            f"{_format_float(row['tool_total_mean_ms'], 1)} | "
            f"{_format_float(row['llm_total_mean_ms'], 1)} | "
            f"{row['failed_actions']} |"
        )

    lines.extend(["", "## Relative To OS Default", ""])
    lines.append("| n | placement | wall delta | tool delta |")
    lines.append("|---:|---|---:|---:|")
    by_key = {(row["agent_count"], row["placement"]): row for row in complete}
    for row in sorted(complete, key=lambda r: (r["agent_count"], str(r["placement"]))):
        if row["placement"] == "os_default":
            continue
        base = by_key.get((row["agent_count"], "os_default"))
        if not base or not base.get("wall_s") or not base.get("tool_total_mean_ms"):
            continue
        wall_delta = (float(row["wall_s"]) / float(base["wall_s"]) - 1.0) * 100.0
        tool_delta = (
            float(row["tool_total_mean_ms"])
            / float(base["tool_total_mean_ms"])
            - 1.0
        ) * 100.0
        lines.append(
            f"| {row['agent_count']} | {row['placement']} | "
            f"{wall_delta:+.2f}% | {tool_delta:+.2f}% |"
        )

    lines.extend(
        [
            "",
            "Memory-bandwidth counters are descriptive only. When "
            "`memory_bandwidth_available` is false, no bandwidth conclusion "
            "should be drawn from these traces.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze n*/placement Kunpeng LLC scaling replay outputs.",
    )
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs"),
        help="Directory for summary CSV and markdown notes.",
    )
    args = parser.parse_args()

    root = args.experiment_root
    rows = analyze(root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    label = root.name
    csv_path = args.output_dir / f"llc_scaling_summary_{label}.csv"
    notes_path = args.output_dir / f"llc_scaling_summary_{label}.md"
    write_csv(csv_path, rows)
    write_notes(notes_path, root, rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {notes_path}")


if __name__ == "__main__":
    main()
