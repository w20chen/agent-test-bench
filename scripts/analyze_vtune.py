#!/usr/bin/env python3
"""
analyze_vtune.py
================
Automated aggregation & analysis of VTune profiling results across all
experiments in this directory.

Directory layout assumed (discovered, not hard-coded):

    <experiment>/<attempt>/vtune/exec-<tool>_<timestamp>_<pid>/
        coarse.json   -> coarse-grained resource profile (min/max/avg/delta)
        fine.json     -> fine-grained microarchitecture profile (VTune TMA + perf)

The folder name `exec-<tool>_...` encodes which command/tool was profiled
(e.g. grep, pytest, python, find).

Outputs (written next to this script):
    vtune_runs.csv          one row per profiled run (all coarse + key fine metrics)
    vtune_summary.json      per-tool aggregate stats (min/max/mean/std/count) + analysis

Usage:
    python3 analyze_vtune.py [--root DIR] [--outdir DIR]
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
from collections import defaultdict

# --------------------------------------------------------------------------- #
# Metric selection
# --------------------------------------------------------------------------- #

# Coarse metrics: every entry in coarse.json is included. Each entry is a dict
# with some subset of {min, max, avg, delta}. We flatten as "<metric>.<stat>".
COARSE_STATS = ("min", "max", "avg", "delta")

# Fine-grained metrics: a curated subset of the (157) VTune fields that matter
# most for systems-level analysis. Grouped by category for readability; the
# script only uses the flat name list when extracting.
FINE_METRICS = {
    "overview": [
        "Elapsed Time",            # seconds
        "Clockticks",
        "Instructions Retired",
        "CPI Rate",                # cycles per instruction (lower is better)
        "Average CPU Frequency",
        "Total Thread Count",
    ],
    # Top-down Microarchitecture Analysis (TMA) level-1/2 breakdown (% of slots)
    "topdown": [
        "Retiring",
        "Front-End Bound",
        "Back-End Bound",
        "Bad Speculation",
        "Core Bound",
        "Memory Bound",
        "Useful Work",
    ],
    # Branch prediction / speculation
    "branch": [
        "Mispredictions",
        "Branch Mispredict",
        "Other Mispredicts",
        "Branching Overhead",
        "Branch Resteers",
        "Mispredicts Resteers",
        "Machine Clears",
    ],
    # Instruction cache / front-end fetch
    "icache": [
        "ICache Misses",
        "Code L2 Hit",
        "Code L2 Miss",
        "Instruction Fetch Bandwidth",
        "Front-End Latency",
        "Front-End Bandwidth",
        "(Info) DSB Coverage",
    ],
    # Data cache hierarchy L1/L2/L3 + DRAM
    "dcache": [
        "L1 Bound",
        "L1 Latency Dependency",
        "L2 Bound",
        "L3 Bound",
        "DRAM Bound",
        "Cache Memory Bandwidth",
        "Cache Memory Latency",
        "Store Bound",
    ],
    # Translation lookaside buffers (TLB)
    "tlb": [
        "ITLB Misses",
        "DTLB Overhead",
        "Load STLB Hit",
        "Load STLB Miss",
        "DTLB Store Overhead",
        "Store STLB Hit",
        "Store STLB Miss",
        "Memory Data TLBs",
    ],
    # Compute / port utilization
    "compute": [
        "Compute Bound Estimation",
        "Port Utilization",
        "ALU Operation Utilization",
        "Load Operation Utilization",
        "Store Operation Utilization",
        "Divider",
        "Serializing Operations",
    ],
}

# Flattened ordered list of fine metric names actually extracted.
FINE_KEYS = [m for group in FINE_METRICS.values() for m in group]

# perf sub-section (live perf-counter sampling) — also surfaced.
PERF_KEYS = ["ipc", "l1i_hit_rate", "branch_miss_rate"]


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #

EXEC_RE = re.compile(r"exec-([A-Za-z0-9]+)_(\d+T\d+)_(\d+)_(\d+)")


def parse_exec_dirname(name):
    """Return (tool, timestamp, ...) from an exec-<tool>_... folder name."""
    m = EXEC_RE.match(name)
    if m:
        return m.group(1), m.group(2)
    # fallback: strip 'exec-' prefix and take leading token
    base = name[len("exec-"):] if name.startswith("exec-") else name
    return base.split("_")[0], ""


def to_number(value):
    """Best-effort convert a VTune field to float.

    Some fields are strings like '5.5% (3.505 out of 64)' or plain numbers.
    Returns float or None when not numeric.
    """
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # grab first number (handles '5.5% (...)' -> 5.5)
        m = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", value)
        if m:
            return float(m.group(0))
    return None


def load_json(path):
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #

def collect_runs(root):
    """Walk the tree and build one record per profiled exec dir."""
    runs = []
    pattern = os.path.join(root, "*", "attempt_*", "vtune", "exec-*")
    for exec_dir in sorted(glob.glob(pattern)):
        if not os.path.isdir(exec_dir):
            continue
        parts = exec_dir.split(os.sep)
        # .../<experiment>/<attempt>/vtune/<exec-dir>
        experiment = parts[-4]
        attempt = parts[-3]
        dirname = parts[-1]
        tool, timestamp = parse_exec_dirname(dirname)

        coarse = load_json(os.path.join(exec_dir, "coarse.json"))
        fine = load_json(os.path.join(exec_dir, "fine.json"))

        if coarse is None and fine is None:
            # nothing profiled here (failed/empty run) -> skip
            continue

        rec = {
            "experiment": experiment,
            "attempt": attempt,
            "tool": tool,
            "run_dir": dirname,
            "timestamp": timestamp,
            "has_coarse": coarse is not None,
            "has_fine": fine is not None,
            "coarse": {},
            "fine": {},
            "perf": {},
        }

        # ---- coarse: include everything ----
        if isinstance(coarse, dict):
            for metric, payload in coarse.items():
                if isinstance(payload, dict):
                    for stat in COARSE_STATS:
                        if stat in payload:
                            val = to_number(payload[stat])
                            if val is not None:
                                rec["coarse"][f"{metric}.{stat}"] = val
                else:
                    val = to_number(payload)
                    if val is not None:
                        rec["coarse"][metric] = val

        # ---- fine: curated VTune keys + perf ----
        if isinstance(fine, dict):
            vt = fine.get("vtune", {})
            if isinstance(vt, dict):
                for key in FINE_KEYS:
                    if key in vt:
                        val = to_number(vt[key])
                        if val is not None:
                            rec["fine"][key] = val
                # derived: IPC from clockticks / instructions if available
                ipc_num = to_number(vt.get("Instructions Retired"))
                ipc_den = to_number(vt.get("Clockticks"))
                if ipc_num and ipc_den:
                    rec["fine"]["IPC (derived)"] = ipc_num / ipc_den

            perf = fine.get("perf", {})
            if isinstance(perf, dict):
                for key in PERF_KEYS:
                    payload = perf.get(key)
                    if isinstance(payload, dict):
                        avg = to_number(payload.get("avg"))
                        if avg is not None:
                            rec["perf"][key] = avg

        runs.append(rec)
    return runs


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def aggregate(runs):
    """Per-tool aggregate stats for every numeric metric."""
    # tool -> metric_fullname -> list of values
    buckets = defaultdict(lambda: defaultdict(list))
    counts = defaultdict(int)

    for rec in runs:
        tool = rec["tool"]
        counts[tool] += 1
        for section in ("coarse", "fine", "perf"):
            prefix = section
            for metric, val in rec[section].items():
                buckets[tool][f"{prefix}/{metric}"].append(val)

    summary = {}
    for tool, metrics in buckets.items():
        tool_summary = {"num_runs": counts[tool], "metrics": {}}
        for metric, values in sorted(metrics.items()):
            if not values:
                continue
            tool_summary["metrics"][metric] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            }
        summary[tool] = tool_summary
    return summary


def build_analysis(summary):
    """Produce a compact human-readable regularity analysis per tool."""
    analysis = {}
    # metrics we call out explicitly in the narrative
    highlight = [
        ("fine/CPI Rate", "CPI", "lower is better"),
        ("fine/Retiring", "Retiring %", "higher is better"),
        ("fine/Front-End Bound", "Front-End Bound %", None),
        ("fine/Back-End Bound", "Back-End Bound %", None),
        ("fine/Bad Speculation", "Bad Speculation %", None),
        ("fine/Memory Bound", "Memory Bound %", None),
        ("fine/Branch Mispredict", "Branch Mispredict %", None),
        ("fine/ICache Misses", "ICache Misses %", None),
        ("fine/L3 Bound", "L3 Bound %", None),
        ("fine/Elapsed Time", "Elapsed Time (s)", None),
        ("coarse/cpu_percent.avg", "avg CPU %", None),
        ("coarse/memory_mb.max", "peak memory (MB)", None),
    ]
    for tool, ts in summary.items():
        lines = []
        m = ts["metrics"]
        lines.append(f"{tool}: {ts['num_runs']} profiled run(s).")
        # identify dominant top-down bottleneck on average
        td = {
            k: m.get(f"fine/{k}", {}).get("mean")
            for k in ("Retiring", "Front-End Bound", "Back-End Bound", "Bad Speculation")
        }
        td = {k: v for k, v in td.items() if v is not None}
        if td:
            dom = max(td, key=td.get)
            lines.append(f"  Dominant top-down category (avg): {dom} = {td[dom]:.1f}%.")
        for key, label, note in highlight:
            if key in m:
                s = m[key]
                suffix = f" ({note})" if note else ""
                lines.append(
                    f"  {label}: min={s['min']:.3g} mean={s['mean']:.3g} "
                    f"max={s['max']:.3g} std={s['std']:.3g}{suffix}"
                )
        analysis[tool] = "\n".join(lines)
    return analysis


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def write_csv(runs, path):
    # union of all metric columns, stable order: coarse, fine, perf
    coarse_cols, fine_cols, perf_cols = set(), set(), set()
    for r in runs:
        coarse_cols |= r["coarse"].keys()
        fine_cols |= r["fine"].keys()
        perf_cols |= r["perf"].keys()

    meta_cols = ["experiment", "attempt", "tool", "run_dir", "timestamp",
                 "has_coarse", "has_fine"]
    coarse_cols = ["coarse/" + c for c in sorted(coarse_cols)]
    # keep fine cols in curated order, then any extras (e.g. derived)
    ordered_fine = [k for k in FINE_KEYS if any(k in r["fine"] for r in runs)]
    extras = sorted({k for r in runs for k in r["fine"]} - set(ordered_fine))
    fine_cols = ["fine/" + c for c in ordered_fine + extras]
    perf_cols = ["perf/" + c for c in sorted(perf_cols)]

    header = meta_cols + coarse_cols + fine_cols + perf_cols
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in runs:
            row = [r[c] for c in meta_cols]
            for c in coarse_cols:
                row.append(r["coarse"].get(c[len("coarse/"):], ""))
            for c in fine_cols:
                row.append(r["fine"].get(c[len("fine/"):], ""))
            for c in perf_cols:
                row.append(r["perf"].get(c[len("perf/"):], ""))
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--root", default=here,
                    help="root directory containing experiment folders")
    ap.add_argument("--outdir", default=here, help="where to write outputs")
    args = ap.parse_args()

    runs = collect_runs(args.root)
    if not runs:
        print("No VTune runs found under", args.root)
        return

    summary = aggregate(runs)
    analysis = build_analysis(summary)

    csv_path = os.path.join(args.outdir, "vtune_runs.csv")
    json_path = os.path.join(args.outdir, "vtune_summary.json")

    write_csv(runs, csv_path)

    out = {
        "root": os.path.abspath(args.root),
        "num_runs": len(runs),
        "num_experiments": len({r["experiment"] for r in runs}),
        "tools": sorted({r["tool"] for r in runs}),
        "runs_per_tool": {t: summary[t]["num_runs"] for t in summary},
        "fine_metrics_extracted": FINE_KEYS,
        "per_tool_summary": summary,
        "analysis": analysis,
        # full per-run detail also embedded for downstream programmatic use
        "runs": runs,
    }
    with open(json_path, "w") as fh:
        json.dump(out, fh, indent=2)

    # ---- console report ----
    print("=" * 70)
    print("VTune profiling aggregation")
    print("=" * 70)
    print(f"root            : {out['root']}")
    print(f"experiments     : {out['num_experiments']}")
    print(f"profiled runs   : {out['num_runs']}")
    print(f"tools           : {', '.join(out['tools'])}")
    print(f"runs per tool   : {out['runs_per_tool']}")
    print("-" * 70)
    for tool in sorted(analysis):
        print(analysis[tool])
        print("-" * 70)
    print(f"\nWrote per-run table : {csv_path}")
    print(f"Wrote summary/json  : {json_path}")


if __name__ == "__main__":
    main()
