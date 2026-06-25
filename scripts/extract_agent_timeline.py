#!/usr/bin/env python3
"""Extract agent lifecycle timeline from per-agent trace.jsonl files.

Scans a simulation output directory for ``*/attempt_*/trace.jsonl`` files,
reads each one, and extracts the wall-clock start/end time of every agent.

Usage::

    python scripts/extract_agent_timeline.py \\
        --input-dir traces/simulate/swe-rebench/sweep_320a_1cpu \\
        --output traces/simulate/swe-rebench/sweep_320a_1cpu/agent_timeline.jsonl

Output format (one JSON record per agent)::

    {
      "agent_id": "django__django-12345--a7",
      "start_ts": 1719000000.123,
      "end_ts": 1719000120.456,
      "elapsed_s": 120.333,
      "n_actions": 42,
      "n_llm_calls": 21,
      "n_tool_execs": 19,
      "n_other": 2,
      "source_trace": "traces/swe-rebench/.../django__django-12345/trace.jsonl",
      "source_agent_id": "django__django-12345"
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_agent_timeline(input_dir: Path) -> list[dict[str, object]]:
    """Scan *input_dir* for per-agent trace files and extract lifecycle records.

    Args:
        input_dir: Simulation output directory containing per-agent subdirs.

    Returns:
        List of agent lifecycle dicts, sorted by ``start_ts``.
    """
    trace_files = sorted(input_dir.glob("*/attempt_*/trace.jsonl"))
    if not trace_files:
        logger.warning("No trace.jsonl files found under %s", input_dir)
        return []

    records: list[dict[str, object]] = []

    for trace_path in trace_files:
        # Infer agent_id from the directory name: <agent_id>/attempt_N/trace.jsonl
        attempt_dir = trace_path.parent
        agent_dir = attempt_dir.parent
        agent_id = agent_dir.name

        try:
            actions = _read_actions(trace_path)
        except Exception:
            logger.exception("Failed to read %s", trace_path)
            continue

        if not actions:
            logger.warning("%s: no actions found, skipping", agent_id)
            records.append({
                "agent_id": agent_id,
                "start_ts": None,
                "end_ts": None,
                "elapsed_s": 0.0,
                "n_actions": 0,
                "n_llm_calls": 0,
                "n_tool_execs": 0,
                "n_other": 0,
                "source_trace": None,
                "source_agent_id": None,
            })
            continue

        start_ts = min(a["ts_start"] for a in actions)
        end_ts = max(a["ts_end"] for a in actions)
        n_llm = sum(1 for a in actions if a["action_type"] == "llm_call")
        n_tool = sum(1 for a in actions if a["action_type"] == "tool_exec")
        n_other = len(actions) - n_llm - n_tool

        # Extract source trace path and original agent_id from metadata
        metadata = _read_metadata(trace_path)
        source_trace = (metadata or {}).get("source_trace")
        source_agent_id = (metadata or {}).get("instance_id")

        records.append({
            "agent_id": agent_id,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "elapsed_s": round(max(0.0, end_ts - start_ts), 3),
            "n_actions": len(actions),
            "n_llm_calls": n_llm,
            "n_tool_execs": n_tool,
            "n_other": n_other,
            "source_trace": source_trace,
            "source_agent_id": source_agent_id,
        })

    records.sort(key=lambda r: float(r.get("start_ts") or 0.0))
    return records


def _read_metadata(trace_path: Path) -> dict[str, object] | None:
    """Read the ``trace_metadata`` record from a trace JSONL file."""
    try:
        with trace_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("type") == "trace_metadata":
                    return record
    except (OSError, json.JSONDecodeError):
        pass
    return None


def _read_actions(trace_path: Path) -> list[dict[str, object]]:
    """Read all ``action`` records from a trace JSONL file.

    Returns:
        List of dicts with keys ``ts_start``, ``ts_end``, ``action_type``.
    """
    actions: list[dict[str, object]] = []
    with trace_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") != "action":
                continue
            ts_start = record.get("ts_start")
            ts_end = record.get("ts_end")
            if ts_start is None or ts_end is None:
                continue
            actions.append({
                "ts_start": float(ts_start),
                "ts_end": float(ts_end),
                "action_type": record.get("action_type", "unknown"),
            })
    return actions


def print_summary(records: list[dict[str, object]]) -> None:
    """Print a human-readable summary of the agent timeline."""
    if not records:
        print("No agent records found.")
        return

    valid = [r for r in records if r["start_ts"] is not None]
    if not valid:
        print(f"{len(records)} agents, all with no actions.")
        return

    start_ts_all = min(float(r["start_ts"]) for r in valid)
    end_ts_all = max(float(r["end_ts"]) for r in valid)
    elapsed_s_all = end_ts_all - start_ts_all
    elapsed_list = [float(r["elapsed_s"]) for r in valid]

    print(f"Total agents:          {len(records)}")
    print(f"Agents with actions:   {len(valid)}")
    print(f"Experiment wall time:  {elapsed_s_all:.1f}s ({elapsed_s_all/60:.1f} min)")
    print(f"Agent elapsed (mean):  {sum(elapsed_list)/len(elapsed_list):.1f}s")
    print(f"Agent elapsed (min):   {min(elapsed_list):.1f}s")
    print(f"Agent elapsed (max):   {max(elapsed_list):.1f}s")
    print(f"Agent elapsed (p50):   {_percentile(elapsed_list, 50):.1f}s")
    print(f"Agent elapsed (p95):   {_percentile(elapsed_list, 95):.1f}s")
    print(f"Agent elapsed (p99):   {_percentile(elapsed_list, 99):.1f}s")

    total_actions = sum(int(r["n_actions"]) for r in valid)
    total_llm = sum(int(r["n_llm_calls"]) for r in valid)
    total_tool = sum(int(r["n_tool_execs"]) for r in valid)
    print(f"Total actions:         {total_actions} "
          f"(llm={total_llm}, tool={total_tool})")


def _percentile(data: list[float], p: float) -> float:
    """Compute the *p*-th percentile of *data* (linear interpolation)."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (p / 100.0) * (len(sorted_data) - 1)
    f = int(k)
    c = k - f
    if f + 1 < len(sorted_data):
        return sorted_data[f] + c * (sorted_data[f + 1] - sorted_data[f])
    return sorted_data[f]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract agent lifecycle timeline from per-agent trace files.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Simulation output directory containing per-agent subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the agent timeline JSONL file.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    if not args.input_dir.is_dir():
        print(f"ERROR: --input-dir does not exist: {args.input_dir}", file=sys.stderr)
        sys.exit(2)

    records = extract_agent_timeline(args.input_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Wrote %d agent timeline records to %s", len(records), args.output)
    print_summary(records)


if __name__ == "__main__":
    main()
