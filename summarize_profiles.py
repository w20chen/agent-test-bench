#!/usr/bin/env python3
"""Summarize tool profiler JSONL output into a readable table.

Usage:
    python summarize_profiles.py demo_profiles.jsonl
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
    """Read JSONL and print summary table + aggregate comparison stats."""
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
        f"{'command':<35} {'wall':>6} {'short':>5} "
        f"{'early_beh':>14} {'final_beh':>14} "
        f"{'early_cores':>11} {'final_cores':>11} "
        f"{'peak_cores':>10} {'peak_proc':>9} {'peak_thr':>9} "
        f"{'rss_mb':>8} {'read_mb':>8} {'write_mb':>8} {'stable':>8}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for rec in records:
        cmd = rec.get("command_string", "?")
        if len(cmd) > 33:
            cmd = cmd[:30] + "..."

        final = rec.get("final_profile", {})
        early = rec.get("early_profile", {})

        row = (
            f"{cmd:<35} "
            f"{_format_val(final.get('total_wall_time_s')):>6} "
            f"{_format_val(final.get('short_tool')):>5} "
            f"{_safe_get(early, 'preliminary_behavior', default='--'):>14} "
            f"{_format_val(final.get('preliminary_behavior')):>14} "
            f"{_safe_get(early, 'avg_effective_cores', default='--'):>11} "
            f"{_format_val(final.get('avg_effective_cores')):>11} "
            f"{_format_val(final.get('peak_effective_cores')):>10} "
            f"{_format_val(final.get('peak_process_count')):>9} "
            f"{_format_val(final.get('peak_thread_count')):>9} "
            f"{_mb(final.get('rss_peak_bytes')):>8} "
            f"{_mb(final.get('total_read_bytes')):>8} "
            f"{_mb(final.get('total_write_bytes')):>8} "
            f"{_format_val(final.get('profile_stability')):>8}"
        )
        print(row)

    print(sep)

    # Aggregate comparison stats
    records_with_early = [
        r for r in records if r.get("early_profile", {}).get("available")
    ]

    print(f"\nAggregate comparison statistics:")
    print(f"  Total invocations:              {len(records)}")
    print(f"  With early profile:             {len(records_with_early)}")

    if records_with_early:
        # Behavior change
        changed = sum(
            1
            for r in records_with_early
            if r.get("early_final_comparison", {}).get("behavior_changed")
        )
        print(f"  Behavior changed (early→final): {changed}/{len(records_with_early)} "
              f"({changed / len(records_with_early) * 100:.1f}%)")

        # Relative error
        errors = [
            r.get("early_final_comparison", {}).get("effective_cores_relative_error", 0.0)
            or 0.0
            for r in records_with_early
        ]
        avg_err = sum(errors) / len(errors)
        print(f"  Avg effective cores rel error: {avg_err:.3f}")

        # Stable vs unstable
        stable_recs = [
            r
            for r in records_with_early
            if r.get("early_profile", {}).get("profile_stability") == "stable"
        ]
        unstable_recs = [
            r
            for r in records_with_early
            if r.get("early_profile", {}).get("profile_stability") == "unstable"
        ]

        if stable_recs:
            s_errs = [
                r.get("early_final_comparison", {}).get("effective_cores_relative_error", 0.0)
                or 0.0
                for r in stable_recs
            ]
            print(f"  Stable   calls: {len(stable_recs)}, avg rel error: {sum(s_errs) / len(s_errs):.3f}")

        if unstable_recs:
            u_errs = [
                r.get("early_final_comparison", {}).get("effective_cores_relative_error", 0.0)
                or 0.0
                for r in unstable_recs
            ]
            print(f"  Unstable calls: {len(unstable_recs)}, avg rel error: {sum(u_errs) / len(u_errs):.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <profiles.jsonl>", file=sys.stderr)
        sys.exit(1)
    summarize(sys.argv[1])
