#!/usr/bin/env python3
"""Summarize LLC placement experiment runs without modifying raw traces."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _trace_elapsed_s(trace_path: Path) -> float | None:
    timestamps: list[float] = []
    try:
        handle = trace_path.open("r", encoding="utf-8")
    except OSError:
        return None
    with handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in ("ts", "timestamp", "time"):
                value = record.get(key)
                if isinstance(value, (int, float)):
                    timestamps.append(float(value))
                    break
    if len(timestamps) < 2:
        return None
    return max(timestamps) - min(timestamps)


def _run_summary(run_dir: Path) -> dict[str, Any]:
    config = _read_json(run_dir / "run_config.json") or {}
    manifest = _read_json(run_dir / "run_manifest.json") or {}
    trace_paths = sorted(run_dir.glob("**/trace.jsonl"))
    elapsed = [
        value for value in (_trace_elapsed_s(path) for path in trace_paths)
        if value is not None
    ]
    return {
        "placement": config.get("placement", run_dir.name),
        "cpus": ",".join(str(cpu) for cpu in config.get("cpus") or []),
        "llc_ids": ",".join(str(llc) for llc in config.get("llc_ids") or []),
        "run_dir": str(run_dir),
        "returncode": config.get("returncode"),
        "trace_count": len(trace_paths),
        "agent_elapsed_mean_s": statistics.mean(elapsed) if elapsed else None,
        "agent_elapsed_p50_s": _percentile(elapsed, 0.50),
        "agent_elapsed_p95_s": _percentile(elapsed, 0.95),
        "manifest_status": manifest.get("status"),
    }


def summarize(root: Path) -> list[dict[str, Any]]:
    run_dirs = [
        path for path in sorted(root.iterdir())
        if path.is_dir() and (path / "run_config.json").exists()
    ]
    return [_run_summary(path) for path in run_dirs]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize LLC placement runs.")
    parser.add_argument("experiment_root", type=Path)
    args = parser.parse_args()

    rows = summarize(args.experiment_root)
    out_json = args.experiment_root / "summary.json"
    out_csv = args.experiment_root / "summary.csv"
    out_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    fieldnames = [
        "placement",
        "cpus",
        "llc_ids",
        "run_dir",
        "returncode",
        "trace_count",
        "agent_elapsed_mean_s",
        "agent_elapsed_p50_s",
        "agent_elapsed_p95_s",
        "manifest_status",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_json} and {out_csv}")


if __name__ == "__main__":
    main()
