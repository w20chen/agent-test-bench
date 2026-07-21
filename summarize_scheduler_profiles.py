#!/usr/bin/env python3
"""Summarize tool scheduler JSONL output into a readable table.

Usage:
    python summarize_scheduler_profiles.py demo_scheduler_profiles.jsonl
"""

from __future__ import annotations

import json
import sys
from typing import Any


def _safe_get(d: dict, *keys: str, default: Any = "N/A") -> Any:
    """Safely traverse nested dict keys."""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d


def _format_val(val: Any) -> str:
    """Format a value for table display."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.2f}"
    if isinstance(val, bool):
        return str(val)
    return str(val)


def _mb(val: Any) -> str:
    """Format bytes as MiB."""
    if val is None or not isinstance(val, (int, float)):
        return "N/A"
    return f"{val / (1024 * 1024):.1f}"


def summarize(jsonl_path: str) -> None:
    """Read JSONL and print summary table."""
    records: list[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("No records found.")
        return

    # Header
    header = (
        f"{'command':<40} {'runtime':>8} {'short':>5} "
        f"{'med_cores':>10} {'p90_cores':>10} {'peak_cores':>10} "
        f"{'decisions':>9} {'moves':>6} {'mem_sens':>10} {'rss_mb':>8}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for rec in records:
        cmd = " ".join(rec.get("command", ["?"]))
        if len(cmd) > 38:
            cmd = cmd[:35] + "..."

        final = rec.get("final_profile", {})
        decisions = rec.get("decisions", [])
        n_moves = sum(1 for d in decisions if d.get("action") == "recommend_move")

        # Try to get memory sensitivity from first decision
        mem_sens = "unknown"
        for d in decisions:
            mem_sens = d.get("memory_sensitivity", "unknown")
            break

        row = (
            f"{cmd:<40} "
            f"{_format_val(rec.get('runtime_s')):>8} "
            f"{_format_val(final.get('short_tool')):>5} "
            f"{_format_val(final.get('median_effective_cores')):>10} "
            f"{_format_val(final.get('p90_effective_cores')):>10} "
            f"{_format_val(final.get('peak_effective_cores')):>10} "
            f"{_format_val(len(decisions)):>9} "
            f"{_format_val(n_moves):>6} "
            f"{_format_val(mem_sens):>10} "
            f"{_mb(final.get('rss_peak_bytes')):>8}"
        )
        print(row)

    print(sep)

    # Aggregate stats
    total_runtime = sum(
        rec.get("runtime_s", 0) for rec in records
    )
    total_decisions = sum(
        len(rec.get("decisions", [])) for rec in records
    )
    total_moves = sum(
        sum(1 for d in rec.get("decisions", [])
            if d.get("action") == "recommend_move")
        for rec in records
    )
    print(f"\nTotal records:      {len(records)}")
    print(f"Total runtime:      {total_runtime:.1f}s")
    print(f"Total decisions:    {total_decisions}")
    print(f"Total recommends:   {total_moves}")

    # Per-decision details
    if total_decisions > 0:
        print(f"\n--- Decision Details ---")
        for rec in records:
            cmd = " ".join(rec.get("command", ["?"]))
            if len(cmd) > 50:
                cmd = cmd[:47] + "..."
            for d in rec.get("decisions", []):
                if d.get("action") == "recommend_move":
                    print(
                        f"  [{cmd}] @ {d.get('elapsed_s', 0):.1f}s: "
                        f"pred={d.get('predicted_cores', 0):.1f} "
                        f"gain={d.get('gain', 0):.2f} "
                        f"-> {d.get('recommended_placement', '?')}"
                    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <profiles.jsonl>")
        sys.exit(1)
    summarize(sys.argv[1])
