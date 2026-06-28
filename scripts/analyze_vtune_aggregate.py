#!/usr/bin/env python3
"""Aggregate VTune profiling results across many pytest invocations.

Walks ``<base_dir>/**/vtune/pytest_*/``, reads ``summary.json`` /
``coarse.json`` / ``fine.json`` from each window, and produces:

1. ``vtune_aggregate.csv`` — one row per pytest invocation, flat columns.
2. ``vtune_aggregate_summary.txt`` — distribution statistics (p5/p50/p95/…)
   for every numeric metric, grouped by coarse / perf-PMU / VTune-TMA layers.
3. ``vtune_aggregate_dist.png`` — kernel density estimation (KDE) distribution
   plots for the most informative PMU and TMA metrics (optional, requires
   matplotlib + scipy).

Usage::

    # CSV + text summary only (no deps beyond stdlib)
    python scripts/analyze_vtune_aggregate.py --input traces/my_sweep/

    # Include distribution plots (needs matplotlib, scipy)
    python scripts/analyze_vtune_aggregate.py --input traces/my_sweep/ --plot

    # Custom output prefix
    python scripts/analyze_vtune_aggregate.py --input traces/my_sweep/ \\
        --output results/vtune_analysis

The output is designed to be directly consumed by pandas / jq / spreadsheet
tools for further slicing (by benchmark, instance, exit code, etc.).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants — VTune TMA Level-1 / Level-2 metric keys (Intel uarch-exploration)
# ---------------------------------------------------------------------------

# VTune ``-report summary`` CSV keys we explicitly recognise and type-cast.
# Keys NOT in this map are still collected as raw strings.
_VTUNE_TMA_NUMERIC_KEYS: set[str] = {
    # Timing / throughput
    "Elapsed Time",
    "CPI Rate",
    "CPI Rate (estimated)",
    "Instructions Retired",
    "Clockticks",
    "Clockticks per Instruction Retired",
    # TMA Level 1
    "Front-End Bound",
    "Back-End Bound",
    "Retiring",
    "Bad Speculation",
    # TMA Level 2 — back-end breakdown
    "Memory Bound",
    "Core Bound",
    # TMA Level 2 — front-end breakdown
    "Front-End Latency",
    "Front-End Bandwidth",
    # TMA Level 2 — bad speculation breakdown
    "Branch Mispredict",
    "Machine Clears",
    # Memory hierarchy
    "L1 Bound",
    "L2 Bound",
    "L3 Bound",
    "DRAM Bound",
    "Store Bound",
    # Cache
    "L1 Hit Rate",
    "L2 Hit Rate",
    "LLC Miss Rate",
    # Misc
    "Average CPU Frequency",
    "Total Thread Count",
    "Paused Time",
}

# Perf PMU keys from the ContainerStatsSampler (min/max/avg per window).
_PERF_KEYS = ("ipc", "l1i_hit_rate", "branch_miss_rate")
# Coarse system-metric keys from Docker stats + cgroup sampling.
_COARSE_KEYS = (
    "cpu_percent",
    "memory_mb",
    "disk_read_mb",
    "disk_write_mb",
    "net_rx_mb",
    "net_tx_mb",
    "context_switches",
)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _try_read_json(path: Path) -> dict[str, Any] | None:
    """Safely read a JSON file; return None on any error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _flatten_nested(
    d: dict[str, Any],
    prefix: str,
    *,
    keys: tuple[str, ...],
    default: Any = None,
) -> dict[str, Any]:
    """Flatten ``{k: {min/max/avg/delta}}`` → ``{prefix_k_min, ...}``.

    Example: ``{"cpu_percent": {"min": 10, "max": 95, "avg": 50}}``
    → ``{"coarse_cpu_percent_min": 10, "coarse_cpu_percent_max": 95, ...}``
    """
    out: dict[str, Any] = {}
    for k in keys:
        val = d.get(k, default)
        if isinstance(val, dict):
            for stat in ("min", "max", "avg", "delta"):
                v = val.get(stat)
                if v is not None:
                    out[f"{prefix}_{k}_{stat}"] = v
        elif val is not None:
            out[f"{prefix}_{k}"] = val
    return out


def _parse_vtune_tma(tma: dict[str, Any]) -> dict[str, Any]:
    """Convert raw VTune CSV values to typed numbers where possible.

    Percentages ("22.5%") → float; plain numbers → float (via guess);
    everything else stays as string.
    """
    out: dict[str, Any] = {}
    for raw_key, raw_val in tma.items():
        key = raw_key.strip()
        val_str = str(raw_val).strip()
        # Try percentage first
        if val_str.endswith("%"):
            try:
                out[f"tma_{key}"] = float(val_str[:-1])
            except ValueError:
                out[f"tma_{key}"] = val_str
            continue
        # Try numeric
        try:
            out[f"tma_{key}"] = float(val_str.replace(",", ""))
        except ValueError:
            out[f"tma_{key}"] = val_str
    return out


def _collect_all_windows(base_dir: Path) -> list[dict[str, Any]]:
    """Recursively find all pytest_* directories and collect their JSON."""
    rows: list[dict[str, Any]] = []
    n_skipped = 0
    n_coarse = 0
    n_fine = 0

    for window_path in sorted(base_dir.rglob("vtune/pytest_*/summary.json")):
        run_dir = window_path.parent
        rel = run_dir.relative_to(base_dir)

        summary = _try_read_json(window_path)
        if summary is None:
            n_skipped += 1
            continue

        row: dict[str, Any] = {
            "path": str(rel),
            "cmd": summary.get("cmd", ""),
            "ts_start": summary.get("ts_start"),
            "ts_end": summary.get("ts_end"),
            "duration_s": summary.get("duration_s"),
            "returncode": summary.get("returncode"),
            "n_samples": summary.get("n_samples"),
            "coarse_source": summary.get("coarse_source", ""),
        }

        # Derive hierarchy path parts for grouping
        # Path structure: .../<instance_id>/<attempt>/vtune/pytest_<ts>_<pid>/
        #   parts[-1] = pytest_<ts>_<pid>
        #   parts[-2] = vtune
        #   parts[-3] = attempt
        #   parts[-4] = instance_id
        parts = rel.parts
        if len(parts) >= 4:
            row["instance_id"] = parts[-4]
        if len(parts) >= 3:
            row["attempt"] = parts[-3]

        # --- coarse.json ---
        coarse = _try_read_json(run_dir / "coarse.json")
        if coarse:
            n_coarse += 1
            row.update(_flatten_nested(coarse, "coarse", keys=_COARSE_KEYS))

        # --- fine.json ---
        fine = _try_read_json(run_dir / "fine.json")
        if fine:
            n_fine += 1
            # perf PMU counters
            perf = fine.get("perf", {})
            row.update(_flatten_nested(perf, "perf", keys=_PERF_KEYS))
            # VTune TMA
            tma = fine.get("vtune", {})
            if isinstance(tma, dict) and "error" not in tma:
                row.update(_parse_vtune_tma(tma))

        rows.append(row)

    print(
        f"[collect] {len(rows)} pytest windows found"
        f" ({n_coarse} with coarse, {n_fine} with fine)"
        f", {n_skipped} skipped",
        file=sys.stderr,
    )
    return rows


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def _percentile(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolation percentile (matches numpy default)."""
    if not sorted_vals:
        return float("nan")
    n = len(sorted_vals)
    k = (n - 1) * p / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return sorted_vals[lo]
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _compute_distribution(
    values: list[float],
) -> dict[str, float]:
    """Compute distribution summary for a list of numeric values."""
    if not values:
        return {}
    n = len(values)
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return {"count": n, "valid": 0}
    clean.sort()
    return {
        "count": n,
        "valid": len(clean),
        "min": clean[0],
        "p5": _percentile(clean, 5),
        "p25": _percentile(clean, 25),
        "p50": _percentile(clean, 50),
        "mean": sum(clean) / len(clean),
        "p75": _percentile(clean, 75),
        "p95": _percentile(clean, 95),
        "max": clean[-1],
        "std": math.sqrt(sum((x - sum(clean) / len(clean)) ** 2 for x in clean) / len(clean)),
    }


def _metric_layer(key: str) -> str:
    """Classify a flattened column name into a metric layer."""
    if key.startswith("coarse_"):
        return "coarse (system)"
    if key.startswith("perf_"):
        return "perf (PMU)"
    if key.startswith("tma_"):
        return "vtune (TMA)"
    return "summary"


def _write_summary_txt(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write human-readable distribution statistics grouped by metric layer."""
    # Collect all numeric values per key
    key_values: dict[str, list[float]] = defaultdict(list)
    tma_keys_seen: set[str] = set()
    perf_keys_seen: set[str] = set()
    coarse_keys_seen: set[str] = set()

    for row in rows:
        for key, val in row.items():
            if not isinstance(val, (int, float)):
                continue
            if not math.isfinite(val):
                continue
            key_values[key].append(val)
            if key.startswith("tma_"):
                tma_keys_seen.add(key)
            elif key.startswith("perf_"):
                perf_keys_seen.add(key)
            elif key.startswith("coarse_"):
                coarse_keys_seen.add(key)

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("VTune Aggregate — Distribution Summary")
    lines.append("=" * 72)
    lines.append(f"Total pytest windows: {len(rows)}")
    lines.append(f"TMA metrics available:   {len(tma_keys_seen)} keys")
    lines.append(f"Perf PMU metrics:        {len(perf_keys_seen)} keys")
    lines.append(f"Coarse system metrics:   {len(coarse_keys_seen)} keys")
    lines.append("")
    lines.append("Columns: count valid min p5 p25 p50 mean p75 p95 max std")
    lines.append("")

    for layer, label in [
        ("tma_", "VTune Top-down Microarchitecture Analysis (TMA)"),
        ("perf_", "Perf PMU Counters (perf stat)"),
        ("coarse_", "Coarse System Metrics (Docker stats + cgroup)"),
        ("duration", "Summary"),
    ]:
        lines.append("-" * 72)
        lines.append(f"  {label}")
        lines.append("-" * 72)
        for key in sorted(key_values):
            if layer == "tma_" and not key.startswith("tma_"):
                continue
            if layer == "perf_" and not key.startswith("perf_"):
                continue
            if layer == "coarse_" and not key.startswith("coarse_"):
                continue
            if layer == "duration" and key != "duration_s":
                continue
            dist = _compute_distribution(key_values[key])
            if not dist:
                continue
            lines.append(
                f"  {key:<45s} "
                f"{dist.get('count', 0):>4d} "
                f"{dist.get('valid', 0):>4d} "
                f"{dist.get('min', 0):>10.4g} "
                f"{dist.get('p5', 0):>10.4g} "
                f"{dist.get('p25', 0):>10.4g} "
                f"{dist.get('p50', 0):>10.4g} "
                f"{dist.get('mean', 0):>10.4g} "
                f"{dist.get('p75', 0):>10.4g} "
                f"{dist.get('p95', 0):>10.4g} "
                f"{dist.get('max', 0):>10.4g} "
                f"{dist.get('std', 0):>10.4g}"
            )
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[summary] wrote {output_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write flat CSV with all columns from all rows."""
    if not rows:
        print("[csv] no data to write", file=sys.stderr)
        return
    # Collect union of all keys (in stable order)
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                all_keys.append(key)

    import csv

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[csv] wrote {len(rows)} rows × {len(all_keys)} cols → {output_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Distribution plots (optional: matplotlib + scipy)
# ---------------------------------------------------------------------------


def _make_dist_plots(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Generate KDE distribution plots for key metrics."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import stats as scipy_stats  # noqa: F401
    except ImportError:
        print(
            "[plot] matplotlib + scipy required for --plot; "
            "install with: pip install matplotlib scipy",
            file=sys.stderr,
        )
        return

    # Collect metrics of interest
    metrics: list[tuple[str, str, str]] = [
        # (column_prefix, display_name, unit)
        ("coarse_cpu_percent_avg", "CPU % (avg)", "%"),
        ("coarse_memory_mb_avg", "Memory (avg)", "MB"),
        ("coarse_disk_read_mb_delta", "Disk Read (Δ)", "MB"),
        ("coarse_disk_write_mb_delta", "Disk Write (Δ)", "MB"),
        ("coarse_context_switches_delta", "Context Switches (Δ)", "count"),
        ("perf_ipc_avg", "IPC (avg)", ""),
        ("perf_l1i_hit_rate_avg", "L1I Hit Rate (avg)", "%"),
        ("perf_branch_miss_rate_avg", "Branch Miss Rate (avg)", "%"),
        ("tma_Front-End Bound", "Front-End Bound", "% slots"),
        ("tma_Back-End Bound", "Back-End Bound", "% slots"),
        ("tma_Retiring", "Retiring", "% slots"),
        ("tma_Bad Speculation", "Bad Speculation", "% slots"),
        ("tma_Memory Bound", "Memory Bound", "% slots"),
        ("tma_Core Bound", "Core Bound", "% slots"),
        ("tma_CPI Rate", "CPI Rate", ""),
        ("tma_L1 Bound", "L1 Bound", "% slots"),
        ("tma_L2 Bound", "L2 Bound", "% slots"),
        ("tma_L3 Bound", "L3 Bound", "% slots"),
        ("tma_DRAM Bound", "DRAM Bound", "% slots"),
        ("duration_s", "Wall Duration", "s"),
    ]

    # Filter to metrics actually present
    present: list[tuple[str, str, str]] = []
    for col, name, unit in metrics:
        vals = [r.get(col) for r in rows]
        clean = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
        if len(clean) >= 3:
            present.append((col, name, unit))

    if not present:
        print("[plot] no metrics with ≥3 valid values — skipping plot", file=sys.stderr)
        return

    # Layout: ~3 columns
    n_cols = min(3, len(present))
    n_rows = int(math.ceil(len(present) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False,
    )
    fig.suptitle(
        "VTune Aggregate — PMU / TMA Distribution (KDE)",
        fontsize=14,
        fontweight="bold",
    )

    for idx, (col, name, unit) in enumerate(present):
        ax = axes[idx // n_cols][idx % n_cols]
        vals = [r.get(col) for r in rows]
        clean = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]

        # Histogram (normalized)
        ax.hist(clean, bins=min(30, len(clean) // 2), density=True,
                alpha=0.4, color="steelblue", edgecolor="white", linewidth=0.5)

        # KDE
        try:
            kde = scipy_stats.gaussian_kde(clean)
            x_range = max(clean) - min(clean)
            pad = x_range * 0.1 if x_range > 0 else 1
            xs = [min(clean) - pad + (max(clean) - min(clean) + 2 * pad) * i / 299
                  for i in range(300)]
            ys = kde(xs)
            ax.plot(xs, ys, color="darkorange", linewidth=2, alpha=0.9)
        except Exception:
            pass

        # Vertical lines for p50 / p95
        clean_sorted = sorted(clean)
        p50 = _percentile(clean_sorted, 50)
        p95 = _percentile(clean_sorted, 95)
        ax.axvline(p50, color="green", linestyle="--", linewidth=1.2,
                   alpha=0.7, label=f"p50={p50:.3g}")
        ax.axvline(p95, color="red", linestyle="--", linewidth=1.2,
                   alpha=0.7, label=f"p95={p95:.3g}")

        title = f"{name}"
        if unit:
            title += f" ({unit})"
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Density")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(present), n_rows * n_cols):
        ax = axes[idx // n_cols][idx % n_cols]
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {output_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Top-N slowest / hottest
# ---------------------------------------------------------------------------


def _write_top_n(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write top-20 tables: slowest by duration, highest CPU, lowest IPC."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("Top-N Reports")
    lines.append("=" * 72)
    lines.append("")

    # Top-20 by duration
    lines.append("── Top-20 longest pytest invocations ──")
    by_dur = sorted(
        [r for r in rows if r.get("duration_s") is not None],
        key=lambda r: r.get("duration_s", 0), reverse=True,
    )[:20]
    for i, r in enumerate(by_dur, 1):
        dur = r.get("duration_s", 0)
        cmd = (r.get("cmd") or "")[:80]
        lines.append(f"  {i:2d}. {dur:8.2f}s  {cmd}")

    # Top-20 by avg CPU
    lines.append("")
    lines.append("── Top-20 highest average CPU % ──")
    by_cpu = sorted(
        [r for r in rows if r.get("coarse_cpu_percent_avg") is not None],
        key=lambda r: r.get("coarse_cpu_percent_avg", 0), reverse=True,
    )[:20]
    for i, r in enumerate(by_cpu, 1):
        cpu = r.get("coarse_cpu_percent_avg", 0)
        cmd = (r.get("cmd") or "")[:80]
        lines.append(f"  {i:2d}. {cpu:7.1f}%  {cmd}")

    # Top-20 lowest IPC (most inefficient microarchitecturally)
    lines.append("")
    lines.append("── Top-20 lowest IPC (Instructions Per Cycle) ──")
    by_ipc = sorted(
        [r for r in rows if r.get("perf_ipc_avg") is not None],
        key=lambda r: r.get("perf_ipc_avg", 999),
    )[:20]
    for i, r in enumerate(by_ipc, 1):
        ipc = r.get("perf_ipc_avg", 0)
        cmd = (r.get("cmd") or "")[:80]
        lines.append(f"  {i:2d}. IPC={ipc:.4f}  {cmd}")

    # Top-20 highest Back-End Bound (memory/cache bound)
    lines.append("")
    lines.append("── Top-20 highest Back-End Bound % (TMA) ──")
    by_be = sorted(
        [r for r in rows if r.get("tma_Back-End Bound") is not None],
        key=lambda r: r.get("tma_Back-End Bound", 0), reverse=True,
    )[:20]
    for i, r in enumerate(by_be, 1):
        be = r.get("tma_Back-End Bound", 0)
        mb = r.get("tma_Memory Bound")
        cb = r.get("tma_Core Bound")
        extras = []
        if mb is not None:
            extras.append(f"Mem={mb:.1f}%")
        if cb is not None:
            extras.append(f"Core={cb:.1f}%")
        extra_str = f" ({', '.join(extras)})" if extras else ""
        cmd = (r.get("cmd") or "")[:70]
        lines.append(f"  {i:2d}. BE={be:.1f}%{extra_str}  {cmd}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[topn]  wrote {output_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate VTune profiling results across many pytest windows.",
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Base directory containing **/vtune/pytest_*/ directories.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output prefix (default: <input>/vtune_aggregate).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate distribution KDE plots (requires matplotlib + scipy).",
    )
    args = parser.parse_args()

    base = args.input.resolve()
    if not base.is_dir():
        print(f"ERROR: {base} is not a directory", file=sys.stderr)
        sys.exit(1)

    prefix = args.output or (base / "vtune_aggregate")

    # 1. Collect
    rows = _collect_all_windows(base)
    if not rows:
        print("No pytest windows found — nothing to do.", file=sys.stderr)
        sys.exit(0)

    # 2. CSV
    _write_csv(rows, Path(str(prefix) + ".csv"))

    # 3. Distribution summary
    _write_summary_txt(rows, Path(str(prefix) + "_summary.txt"))

    # 4. Top-N
    _write_top_n(rows, Path(str(prefix) + "_topn.txt"))

    # 5. Plots (optional)
    if args.plot:
        _make_dist_plots(rows, Path(str(prefix) + "_dist.png"))

    print(f"\nDone. Output files at: {prefix}*.csv/.txt/.png", file=sys.stderr)


if __name__ == "__main__":
    main()
