#!/usr/bin/env python3
"""Run one SWE-rebench trace replay case under topology-derived CPU placement.

Taskset compatibility path using ``trace_collect.cli simulate --mode cloud_model`` (no real
LLM inference — replays pre-collected traces with source-trace timing)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.probe_llc_topology import (
    Placement,
    build_placements,
    format_cpu_list,
    probe_topology,
    write_outputs,
)


@dataclass(frozen=True, slots=True)
class RunRecord:
    placement: str
    cpus: list[int] | None
    llc_ids: list[str]
    command: list[str]
    env: dict[str, str]
    run_dir: str
    returncode: int | None
    skipped: bool


def _repo_root() -> Path:
    return REPO_ROOT


def _simulate_base_command(
    *,
    run_dir: Path,
    source_trace: Path,
    task_source: Path,
    container: str,
    num_agents: int,
    ksys_monitoring: str,
    command_timeout: float,
    replay_speed: float,
    network_mode: str,
    extra_simulate_args: list[str],
) -> list[str]:
    """Build the ``trace_collect.cli simulate`` command for cloud_model replay."""
    return [
        sys.executable,
        "-m",
        "trace_collect.cli",
        "simulate",
        "--mode",
        "cloud_model",
        "--source-trace",
        str(source_trace),
        "--task-source",
        str(task_source),
        "--container",
        container,
        "--num-agents",
        str(num_agents),
        "--output-dir",
        str(run_dir),
        "--resource-monitoring",
        "on",
        "--pmu-monitoring",
        "off",
        "--ksys-monitoring",
        ksys_monitoring,
        "--command-timeout",
        str(command_timeout),
        "--replay-speed",
        str(replay_speed),
        "--network-mode",
        network_mode,
        *extra_simulate_args,
    ]


def _with_taskset(command: list[str], placement: Placement) -> list[str]:
    if placement.cpus is None:
        return command
    taskset = shutil.which("taskset")
    if taskset is None:
        raise RuntimeError("taskset is required for explicit CPU placement")
    return [taskset, "-c", format_cpu_list(placement.cpus), *command]


def run_experiment(args: argparse.Namespace) -> list[RunRecord]:
    repo = _repo_root()
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = args.output_root / timestamp
    topology_dir = output_root / "topology"

    topology = probe_topology(args.sys_cpu_root)
    placements = build_placements(
        topology,
        agent_count=args.num_agents,
        sys_cpu_root=args.sys_cpu_root,
        cluster_size=args.cluster_size,
    )
    write_outputs(output_dir=topology_dir, topology=topology, placements=placements)

    selected_names = (
        list(placements)
        if args.placements == "auto"
        else [name.strip() for name in args.placements.split(",") if name.strip()]
    )
    unknown = [name for name in selected_names if name not in placements]
    if unknown:
        raise ValueError(f"unknown placement(s): {', '.join(unknown)}")

    source_trace = args.source_trace.resolve()
    if not source_trace.exists():
        raise FileNotFoundError(f"source trace not found: {source_trace}")

    env = os.environ.copy()
    env["ARM_IMAGE_MODE"] = "qemu"
    env["PYTHONPATH"] = f"{repo / 'src'}{os.pathsep}{repo}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)

    records: list[RunRecord] = []
    for name in selected_names:
        placement = placements[name]
        run_dir = output_root / name
        command = _simulate_base_command(
            run_dir=run_dir,
            source_trace=source_trace,
            task_source=args.task_source,
            container=args.container,
            num_agents=args.num_agents,
            ksys_monitoring=args.ksys_monitoring,
            command_timeout=args.command_timeout,
            replay_speed=args.replay_speed,
            network_mode=args.network_mode,
            extra_simulate_args=args.simulate_args,
        )
        full_command = _with_taskset(command, placement)
        record = RunRecord(
            placement=name,
            cpus=placement.cpus,
            llc_ids=placement.llc_ids,
            command=full_command,
            env={
                "ARM_IMAGE_MODE": env["ARM_IMAGE_MODE"],
                "PYTHONPATH": env["PYTHONPATH"],
            },
            run_dir=str(run_dir),
            returncode=None,
            skipped=args.dry_run,
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_config.json").write_text(
            json.dumps(asdict(record), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[{name}] {' '.join(full_command)}", flush=True)
        if args.dry_run:
            records.append(record)
            continue
        completed = subprocess.run(full_command, cwd=repo, env=env, check=False)
        finished = RunRecord(
            placement=record.placement,
            cpus=record.cpus,
            llc_ids=record.llc_ids,
            command=record.command,
            env=record.env,
            run_dir=record.run_dir,
            returncode=completed.returncode,
            skipped=False,
        )
        (run_dir / "run_config.json").write_text(
            json.dumps(asdict(finished), indent=2) + "\n",
            encoding="utf-8",
        )
        records.append(finished)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)

    (output_root / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "source_trace": str(source_trace),
                "num_agents": args.num_agents,
                "cluster_size": args.cluster_size,
                "topology_dir": str(topology_dir),
                "runs": [asdict(record) for record in records],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a SWE-rebench trace under topology-derived CPU "
            "placement using trace_collect.cli simulate --mode cloud_model "
            "(no real LLM inference). Each placement runs --num-agents "
            "concurrent replay agents from the same --source-trace."
        ),
    )
    parser.add_argument(
        "--source-trace",
        type=Path,
        required=True,
        help="Path to a SWE-rebench trace.jsonl file to replay.",
    )
    parser.add_argument(
        "--task-source",
        type=Path,
        default=Path("data/swe-rebench/tasks.json"),
        help="Path to tasks JSON file (used to resolve per-task metadata).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("traces/experiments/kunpeng_llc"),
    )
    parser.add_argument("--sys-cpu-root", type=Path, default=Path("/sys/devices/system/cpu"))
    parser.add_argument("--container", default="docker", choices=["docker", "podman"])
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument(
        "--placements",
        default="auto",
        help=(
            "Comma-separated placement names, or 'auto' for every valid "
            "placement generated by the topology probe."
        ),
    )
    parser.add_argument(
        "--cluster-size",
        type=int,
        default=4,
        help="Inferred sub-LLC CPU cluster size forwarded to topology probe.",
    )
    parser.add_argument(
        "--ksys-monitoring",
        choices=["auto", "on", "off"],
        default="off",
        help="Forwarded to trace_collect.cli simulate.",
    )
    parser.add_argument(
        "--command-timeout",
        type=float,
        default=600.0,
        help="Fallback timeout in seconds for replayed shell commands.",
    )
    parser.add_argument(
        "--replay-speed",
        type=float,
        default=1.0,
        help="Wall-clock acceleration factor for cloud_model replay.",
    )
    parser.add_argument(
        "--network-mode",
        default="host",
        help="Container network mode (default: host). Use 'none' for isolated replay.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("simulate_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.simulate_args and args.simulate_args[0] == "--":
        args.simulate_args = args.simulate_args[1:]
    if args.num_agents < 1:
        raise ValueError("num_agents must be at least 1")
    run_experiment(args)


if __name__ == "__main__":
    main()
