#!/usr/bin/env python3
"""Validate inferred Kunpeng sub-LLC clusters with pointer chasing.

The topology probe can infer candidate CPU clusters from Linux sysfs ordering,
but sysfs does not expose Kunpeng CCL/LLC-slice identifiers directly on every
kernel. This script runs a real pointer-chase workload under controlled CPU
affinity to test whether a candidate cluster size is supported by measured
interference.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.probe_llc_topology import (  # noqa: E402
    CpuTopology,
    parse_cpu_list,
    probe_topology,
)

CHASE_RE = re.compile(
    r"ns_per_access\s+(?P<ns>[0-9]+(?:\.[0-9]+)?)\s+accesses\s+(?P<accesses>[0-9]+)"
)


@dataclass(frozen=True, slots=True)
class CandidateCluster:
    candidate_cluster_size: int
    cluster_index: int
    cpus: list[int]
    numa_node: int | None
    llc_id: str
    cluster_id: str


@dataclass(frozen=True, slots=True)
class ChaseTrial:
    candidate_cluster_size: int
    mode: str
    run: int
    victim_cpu: int
    aggressor_cpus: list[int]
    victim_cluster_id: str
    aggressor_cluster_ids: list[str]
    skip_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ChaseMeasurement:
    candidate_cluster_size: int
    mode: str
    run: int
    victim_cpu: int
    aggressor_cpus: list[int]
    victim_cluster_id: str
    aggressor_cluster_ids: list[str]
    victim_command: list[str]
    aggressor_commands: list[list[str]]
    ns_per_access: float | None
    accesses: int | None
    stdout: str
    stderr: str
    returncode: int | None
    aggressor_returncodes: list[int | None]
    skipped: bool
    skip_reason: str | None


def parse_positive_int_list(value: str) -> list[int]:
    """Parse a comma-separated list of positive integers."""
    values: list[int] = []
    for part in value.split(","):
        text = part.strip()
        if not text:
            continue
        item = int(text)
        if item <= 0:
            raise ValueError("values must be positive")
        if item not in values:
            values.append(item)
    if not values:
        raise ValueError("at least one value is required")
    return values


def parse_chase_output(output: str) -> tuple[float, int]:
    """Extract ``ns_per_access`` and ``accesses`` from chase stdout."""
    match = CHASE_RE.search(output)
    if match is None:
        raise ValueError(f"could not parse chase output: {output!r}")
    return float(match.group("ns")), int(match.group("accesses"))


def _physical_core_representatives(
    topology: list[CpuTopology],
) -> dict[tuple[int | None, str], list[int]]:
    by_core: dict[tuple[int | None, str, int], int] = {}
    for rec in sorted(topology, key=lambda item: item.cpu):
        core_key = rec.core_id if rec.core_id is not None else rec.cpu
        key = (rec.numa_node, rec.llc_id, core_key)
        by_core[key] = min(rec.cpu, by_core.get(key, rec.cpu))

    by_domain: dict[tuple[int | None, str], list[int]] = {}
    for numa_node, llc_id, core_key in sorted(by_core):
        by_domain.setdefault((numa_node, llc_id), []).append(
            by_core[(numa_node, llc_id, core_key)]
        )
    return {domain: sorted(cpus) for domain, cpus in by_domain.items()}


def infer_candidate_clusters(
    topology: list[CpuTopology],
    *,
    candidate_cluster_size: int,
) -> list[CandidateCluster]:
    """Infer candidate clusters from physical-core CPU ordering per LLC domain."""
    if candidate_cluster_size <= 0:
        raise ValueError("candidate_cluster_size must be positive")

    clusters: list[CandidateCluster] = []
    for (numa_node, llc_id), cpus in sorted(
        _physical_core_representatives(topology).items(),
        key=lambda item: min(item[1]),
    ):
        for cluster_index, start in enumerate(range(0, len(cpus), candidate_cluster_size)):
            cluster_cpus = cpus[start : start + candidate_cluster_size]
            if not cluster_cpus:
                continue
            clusters.append(
                CandidateCluster(
                    candidate_cluster_size=candidate_cluster_size,
                    cluster_index=cluster_index,
                    cpus=cluster_cpus,
                    numa_node=numa_node,
                    llc_id=llc_id,
                    cluster_id=(
                        f"numa{numa_node}:llc{llc_id}:"
                        f"size{candidate_cluster_size}:cluster{cluster_index}"
                    ),
                )
            )
    return clusters


def _cluster_for_cpu(clusters: list[CandidateCluster], cpu: int) -> CandidateCluster:
    for cluster in clusters:
        if cpu in cluster.cpus:
            return cluster
    raise ValueError(f"victim CPU {cpu} is not in any candidate cluster")


def _cluster_ids_for_cpus(
    clusters: list[CandidateCluster],
    cpus: list[int],
) -> list[str]:
    return [_cluster_for_cpu(clusters, cpu).cluster_id for cpu in cpus]


def _take_aggressors(
    cpus: list[int],
    *,
    count: int,
    victim_cpu: int,
) -> list[int]:
    return [cpu for cpu in cpus if cpu != victim_cpu][:count]


def _chase_path_text(chase_bin: Path) -> str:
    if chase_bin.is_absolute():
        return str(chase_bin)
    if chase_bin.parent == Path("."):
        return f".{os.sep}{chase_bin.name}"
    return str(chase_bin)


def build_trials(
    topology: list[CpuTopology],
    *,
    candidate_cluster_sizes: list[int],
    runs: int,
    aggressors_per_run: int,
    victim_cpus: list[int] | None = None,
) -> list[ChaseTrial]:
    """Build baseline and interference trials for each candidate size."""
    if runs <= 0:
        raise ValueError("runs must be positive")
    if aggressors_per_run <= 0:
        raise ValueError("aggressors_per_run must be positive")

    trials: list[ChaseTrial] = []
    for candidate_size in candidate_cluster_sizes:
        clusters = infer_candidate_clusters(
            topology,
            candidate_cluster_size=candidate_size,
        )
        if not clusters:
            continue

        selected_victims = (
            victim_cpus
            if victim_cpus is not None
            else [cluster.cpus[0] for cluster in clusters]
        )
        for victim_cpu in selected_victims:
            victim_cluster = _cluster_for_cpu(clusters, victim_cpu)
            same_available = _take_aggressors(
                victim_cluster.cpus,
                count=len(victim_cluster.cpus),
                victim_cpu=victim_cpu,
            )
            target_aggressors = min(aggressors_per_run, len(same_available))
            same_aggressors = same_available[:target_aggressors]
            same_llc_cpus = [
                cpu
                for cluster in clusters
                if (
                    cluster.llc_id == victim_cluster.llc_id
                    and cluster.cluster_id != victim_cluster.cluster_id
                )
                for cpu in cluster.cpus
            ]
            same_llc_aggressors = _take_aggressors(
                same_llc_cpus,
                count=target_aggressors,
                victim_cpu=victim_cpu,
            )
            other_llc_cpus = [
                cpu
                for cluster in clusters
                if cluster.llc_id != victim_cluster.llc_id
                for cpu in cluster.cpus
            ]
            other_llc_aggressors = _take_aggressors(
                other_llc_cpus,
                count=target_aggressors,
                victim_cpu=victim_cpu,
            )

            modes = [
                ("baseline", []),
                ("same_candidate_cluster", same_aggressors),
                ("other_candidate_cluster_same_llc", same_llc_aggressors),
                ("other_llc_domain", other_llc_aggressors),
            ]
            for run in range(1, runs + 1):
                for mode, aggressors in modes:
                    skip_reason = None
                    if mode != "baseline" and target_aggressors == 0:
                        skip_reason = (
                            f"candidate cluster has no non-victim aggressor CPUs "
                            f"for {mode}"
                        )
                    elif mode != "baseline" and len(aggressors) < target_aggressors:
                        skip_reason = (
                            f"need {target_aggressors} aggressors for {mode}, "
                            f"found {len(aggressors)}"
                        )
                    trials.append(
                        ChaseTrial(
                            candidate_cluster_size=candidate_size,
                            mode=mode,
                            run=run,
                            victim_cpu=victim_cpu,
                            aggressor_cpus=aggressors,
                            victim_cluster_id=victim_cluster.cluster_id,
                            aggressor_cluster_ids=_cluster_ids_for_cpus(
                                clusters,
                                aggressors,
                            ),
                            skip_reason=skip_reason,
                        )
                    )
    return trials


def build_chase_command(
    *,
    cpu: int,
    chase_bin: Path,
    membind: str | None,
    memory_mb: int,
    seconds: int,
) -> list[str]:
    """Build a numactl/taskset command for one chase process."""
    if memory_mb <= 0:
        raise ValueError("memory_mb must be positive")
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    command = [
        "taskset",
        "-c",
        str(cpu),
        _chase_path_text(chase_bin),
        str(memory_mb),
        str(seconds),
    ]
    if membind is None:
        return command
    return ["numactl", f"--membind={membind}", *command]


def run_trial(
    trial: ChaseTrial,
    *,
    chase_bin: Path,
    membind: str | None,
    victim_mb: int,
    victim_sec: int,
    aggressor_mb: int,
    aggressor_sec: int,
    warmup_sec: float,
    dry_run: bool,
) -> ChaseMeasurement:
    """Run one pointer-chase trial and preserve raw command outputs."""
    victim_command = build_chase_command(
        cpu=trial.victim_cpu,
        chase_bin=chase_bin,
        membind=membind,
        memory_mb=victim_mb,
        seconds=victim_sec,
    )
    aggressor_commands = [
        build_chase_command(
            cpu=cpu,
            chase_bin=chase_bin,
            membind=membind,
            memory_mb=aggressor_mb,
            seconds=aggressor_sec,
        )
        for cpu in trial.aggressor_cpus
    ]
    if dry_run or trial.skip_reason is not None:
        return ChaseMeasurement(
            candidate_cluster_size=trial.candidate_cluster_size,
            mode=trial.mode,
            run=trial.run,
            victim_cpu=trial.victim_cpu,
            aggressor_cpus=trial.aggressor_cpus,
            victim_cluster_id=trial.victim_cluster_id,
            aggressor_cluster_ids=trial.aggressor_cluster_ids,
            victim_command=victim_command,
            aggressor_commands=aggressor_commands,
            ns_per_access=None,
            accesses=None,
            stdout="",
            stderr="",
            returncode=None,
            aggressor_returncodes=[],
            skipped=True,
            skip_reason=trial.skip_reason if trial.skip_reason else "dry run",
        )

    processes = [
        subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        for command in aggressor_commands
    ]
    if processes:
        time.sleep(warmup_sec)

    completed = subprocess.run(
        victim_command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    for process in processes:
        process.wait()
    aggressor_returncodes = [process.returncode for process in processes]
    aggressor_stderr = "\n".join(
        stderr
        for process in processes
        for stderr in [process.stderr.read() if process.stderr is not None else ""]
        if stderr
    )

    if completed.returncode != 0:
        ns_per_access, accesses = None, None
        skipped = True
        skip_reason = f"victim process failed with return code {completed.returncode}"
        returncode = completed.returncode
    elif any(code != 0 for code in aggressor_returncodes):
        ns_per_access, accesses = None, None
        skipped = True
        skip_reason = f"aggressor process failed with return codes {aggressor_returncodes}"
        returncode = next(
            int(code) for code in aggressor_returncodes if code is not None and code != 0
        )
    else:
        try:
            ns_per_access, accesses = parse_chase_output(completed.stdout)
        except ValueError as exc:
            ns_per_access, accesses = None, None
            skipped = True
            skip_reason = str(exc)
            returncode = 1
        else:
            skipped = False
            skip_reason = None
            returncode = 0

    return ChaseMeasurement(
        candidate_cluster_size=trial.candidate_cluster_size,
        mode=trial.mode,
        run=trial.run,
        victim_cpu=trial.victim_cpu,
        aggressor_cpus=trial.aggressor_cpus,
        victim_cluster_id=trial.victim_cluster_id,
        aggressor_cluster_ids=trial.aggressor_cluster_ids,
        victim_command=victim_command,
        aggressor_commands=aggressor_commands,
        ns_per_access=ns_per_access,
        accesses=accesses,
        stdout=completed.stdout,
        stderr="\n".join(part for part in [completed.stderr, aggressor_stderr] if part),
        returncode=returncode,
        aggressor_returncodes=aggressor_returncodes,
        skipped=skipped,
        skip_reason=skip_reason,
    )


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _normalized_victim_effects(
    measurements: list[ChaseMeasurement],
) -> list[dict[str, object]]:
    effects: list[dict[str, object]] = []
    keys = sorted(
        {
            (
                item.candidate_cluster_size,
                item.victim_cpu,
                item.victim_cluster_id,
            )
            for item in measurements
            if not item.skipped
        }
    )
    for candidate_size, victim_cpu, victim_cluster_id in keys:
        rows = [
            item
            for item in measurements
            if item.candidate_cluster_size == candidate_size
            and item.victim_cpu == victim_cpu
            and item.victim_cluster_id == victim_cluster_id
            and not item.skipped
            and item.ns_per_access is not None
        ]
        by_mode: dict[str, list[float]] = {}
        for row in rows:
            by_mode.setdefault(row.mode, []).append(row.ns_per_access)

        baseline = _median(by_mode.get("baseline", []))
        same = _median(by_mode.get("same_candidate_cluster", []))
        other_same_llc = _median(by_mode.get("other_candidate_cluster_same_llc", []))
        if baseline is None or baseline <= 0 or same is None or other_same_llc is None:
            continue
        same_delta = same / baseline - 1.0
        other_delta = other_same_llc / baseline - 1.0
        effects.append(
            {
                "candidate_cluster_size": candidate_size,
                "victim_cpu": victim_cpu,
                "victim_cluster_id": victim_cluster_id,
                "baseline_median_ns": baseline,
                "same_candidate_cluster_median_ns": same,
                "other_candidate_cluster_same_llc_median_ns": other_same_llc,
                "same_candidate_cluster_delta_vs_baseline": same_delta,
                "other_candidate_cluster_same_llc_delta_vs_baseline": other_delta,
                "delta_gap": same_delta - other_delta,
            }
        )
    return effects


def summarize_measurements(
    measurements: list[ChaseMeasurement],
    *,
    support_margin_ratio: float,
) -> list[dict[str, object]]:
    """Summarize candidate evidence using per-victim normalized effects."""
    summaries: list[dict[str, object]] = []
    effects = _normalized_victim_effects(measurements)
    candidate_sizes = sorted({item.candidate_cluster_size for item in measurements})
    for candidate_size in candidate_sizes:
        rows = [
            item
            for item in measurements
            if item.candidate_cluster_size == candidate_size and not item.skipped
        ]
        candidate_effects = [
            effect
            for effect in effects
            if effect["candidate_cluster_size"] == candidate_size
        ]
        baseline = _median(
            [float(effect["baseline_median_ns"]) for effect in candidate_effects]
        )
        same = _median(
            [
                float(effect["same_candidate_cluster_median_ns"])
                for effect in candidate_effects
            ]
        )
        other_same_llc = _median(
            [
                float(effect["other_candidate_cluster_same_llc_median_ns"])
                for effect in candidate_effects
            ]
        )
        same_delta = _median(
            [
                float(effect["same_candidate_cluster_delta_vs_baseline"])
                for effect in candidate_effects
            ]
        )
        other_delta = _median(
            [
                float(effect["other_candidate_cluster_same_llc_delta_vs_baseline"])
                for effect in candidate_effects
            ]
        )
        delta_gap = _median([float(effect["delta_gap"]) for effect in candidate_effects])

        if same_delta is None or other_delta is None or delta_gap is None:
            verdict = "insufficient_data"
        elif same_delta >= support_margin_ratio and delta_gap >= support_margin_ratio:
            verdict = "supported_by_interference"
        else:
            verdict = "not_supported_by_interference"

        summaries.append(
            {
                "candidate_cluster_size": candidate_size,
                "baseline_median_ns": baseline,
                "same_candidate_cluster_median_ns": same,
                "other_candidate_cluster_same_llc_median_ns": other_same_llc,
                "same_candidate_cluster_delta_vs_baseline": same_delta,
                "other_candidate_cluster_same_llc_delta_vs_baseline": other_delta,
                "delta_gap": delta_gap,
                "support_margin_ratio": support_margin_ratio,
                "verdict": verdict,
                "per_victim_effects": candidate_effects,
                "completed_measurements": len(rows),
                "skipped_measurements": len(
                    [
                        item
                        for item in measurements
                        if item.candidate_cluster_size == candidate_size and item.skipped
                    ]
                ),
            }
        )
    return summaries


def write_outputs(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    topology: list[CpuTopology],
    measurements: list[ChaseMeasurement],
    summaries: list[dict[str, object]],
) -> None:
    """Write CSV, JSON, and human-readable summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_cluster_size",
        "mode",
        "run",
        "victim_cpu",
        "aggressor_cpus",
        "victim_cluster_id",
        "aggressor_cluster_ids",
        "ns_per_access",
        "accesses",
        "returncode",
        "aggressor_returncodes",
        "skipped",
        "skip_reason",
    ]
    with (output_dir / "measurements.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for item in measurements:
            row = asdict(item)
            writer.writerow(
                {
                    key: json.dumps(row[key]) if isinstance(row[key], list) else row[key]
                    for key in fieldnames
                }
            )

    payload = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "topology": [asdict(item) for item in topology],
        "measurements": [asdict(item) for item in measurements],
        "summary": summaries,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = ["Kunpeng LLC Slice Probe", "=======================", ""]
    for summary in summaries:
        lines.append(
            "candidate_cluster_size={candidate_cluster_size} "
            "verdict={verdict} baseline={baseline_median_ns} "
            "same={same_candidate_cluster_median_ns} "
            "other_same_llc={other_candidate_cluster_same_llc_median_ns}".format(
                **summary
            )
        )
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run pointer-chase interference checks for inferred Kunpeng "
            "sub-LLC cluster sizes."
        )
    )
    parser.add_argument("--sys-cpu-root", type=Path, default=Path("/sys/devices/system/cpu"))
    parser.add_argument("--chase-bin", type=Path, default=Path("./chase"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("traces/experiments/kunpeng_llc_slices"),
    )
    parser.add_argument(
        "--candidate-cluster-sizes",
        default="2,4,8",
        help="Comma-separated candidate physical-core cluster sizes to validate.",
    )
    parser.add_argument(
        "--victim-cpus",
        default="all",
        help="Linux CPU-list for victim CPUs, or 'all' for one victim per candidate cluster.",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--aggressors-per-run",
        type=int,
        default=3,
        help=(
            "Maximum aggressor processes per interference trial. The actual "
            "count is capped by non-victim CPUs available in the candidate "
            "cluster and reused for matched other-cluster controls."
        ),
    )
    parser.add_argument("--victim-mb", type=int, default=3)
    parser.add_argument("--victim-sec", type=int, default=6)
    parser.add_argument("--aggressor-mb", type=int, default=16)
    parser.add_argument("--aggressor-sec", type=int, default=9)
    parser.add_argument("--warmup-sec", type=float, default=1.0)
    parser.add_argument(
        "--membind",
        default="0",
        help="NUMA node passed to numactl --membind, or 'none' to disable numactl.",
    )
    parser.add_argument(
        "--support-margin-ratio",
        type=float,
        default=0.05,
        help=(
            "Descriptive verdict margin: normalized same-cluster slowdown must "
            "exceed both baseline and the other-cluster same-LLC normalized "
            "effect by this ratio to be marked supported."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    parse_positive_int_list(args.candidate_cluster_sizes)
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if args.aggressors_per_run <= 0:
        raise ValueError("--aggressors-per-run must be positive")
    if args.warmup_sec < 0:
        raise ValueError("--warmup-sec must be non-negative")
    if args.aggressor_sec < args.warmup_sec + args.victim_sec:
        raise ValueError(
            "--aggressor-sec must be at least --warmup-sec + --victim-sec "
            "so aggressors cover the victim measurement interval"
        )
    if args.support_margin_ratio < 0:
        raise ValueError("--support-margin-ratio must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    topology = probe_topology(args.sys_cpu_root)
    victim_cpus = None if args.victim_cpus == "all" else parse_cpu_list(args.victim_cpus)
    candidate_sizes = parse_positive_int_list(args.candidate_cluster_sizes)
    trials = build_trials(
        topology,
        candidate_cluster_sizes=candidate_sizes,
        runs=args.runs,
        aggressors_per_run=args.aggressors_per_run,
        victim_cpus=victim_cpus,
    )
    membind = None if args.membind == "none" else args.membind
    measurements = [
        run_trial(
            trial,
            chase_bin=args.chase_bin,
            membind=membind,
            victim_mb=args.victim_mb,
            victim_sec=args.victim_sec,
            aggressor_mb=args.aggressor_mb,
            aggressor_sec=args.aggressor_sec,
            warmup_sec=args.warmup_sec,
            dry_run=args.dry_run,
        )
        for trial in trials
    ]
    summaries = summarize_measurements(
        measurements,
        support_margin_ratio=args.support_margin_ratio,
    )
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root / timestamp
    write_outputs(
        output_dir=output_dir,
        args=args,
        topology=topology,
        measurements=measurements,
        summaries=summaries,
    )
    print(f"Wrote Kunpeng LLC slice probe outputs to {output_dir}")


if __name__ == "__main__":
    main()
