#!/usr/bin/env python3
"""Run API-free trace replay under Kunpeng LLC/NUMA CPU placements."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.probe_llc_topology import (  # noqa: E402
    Placement,
    build_placements,
    probe_topology,
    write_outputs,
)


@dataclass(frozen=True, slots=True)
class ReplayRunRecord:
    placement: str
    cpus: list[int] | None
    llc_ids: list[str]
    agent_assignments: list[dict[str, object]]
    command: list[str]
    env: dict[str, str]
    run_dir: str
    returncode: int | None
    skipped: bool


def _build_command(
    *,
    source_trace: Path,
    task_source: Path,
    output_dir: Path,
    placement: Placement,
    container: str,
    num_agents: int,
    replay_speed: float,
    network_mode: str,
    command_timeout_s: float,
    workers: int,
    prep_concurrency: int,
    resource_monitoring: str,
    ksys_monitoring: str,
    extra_args: list[str],
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "trace_collect.cli",
        "simulate",
        "--source-trace",
        str(source_trace),
        "--task-source",
        str(task_source),
        "--output-dir",
        str(output_dir),
        "--mode",
        "cloud_model",
        "--container",
        container,
        "--network-mode",
        network_mode,
        "--num-agents",
        str(num_agents),
        "--trace-assignment",
        "manifest",
        "--arrival-mode",
        "closed_loop",
        "--replay-speed",
        str(replay_speed),
        "--cpu-limit",
        "1",
        "--command-timeout",
        str(command_timeout_s),
        "--resource-monitoring",
        resource_monitoring,
        "--pmu-monitoring",
        "off",
        "--ksys-monitoring",
        ksys_monitoring,
        "--workers",
        str(workers),
        "--prep-concurrency",
        str(prep_concurrency),
        *extra_args,
    ]
    if placement.cpus is not None:
        cpusets = [assignment.cpuset_cpus for assignment in placement.agent_assignments]
        if len(cpusets) != num_agents or any(cpuset is None for cpuset in cpusets):
            raise ValueError(
                f"placement {placement.name!r} does not provide one cpuset per agent"
            )
        for cpuset in cpusets:
            command.extend(["--agent-cpuset", str(cpuset)])
    return command


def run_replay(args: argparse.Namespace) -> list[ReplayRunRecord]:
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

    env = os.environ.copy()
    env["ARM_IMAGE_MODE"] = "qemu"
    env["PYTHONPATH"] = (
        f"{REPO_ROOT / 'src'}{os.pathsep}{REPO_ROOT}"
        f"{os.pathsep}{env.get('PYTHONPATH', '')}"
    ).rstrip(os.pathsep)

    records: list[ReplayRunRecord] = []
    for name in selected_names:
        placement = placements[name]
        run_dir = output_root / name
        command = _build_command(
            source_trace=args.source_trace,
            task_source=args.task_source,
            output_dir=run_dir,
            placement=placement,
            container=args.container,
            num_agents=args.num_agents,
            replay_speed=args.replay_speed,
            network_mode=args.network_mode,
            command_timeout_s=args.command_timeout,
            workers=args.workers,
            prep_concurrency=args.prep_concurrency,
            resource_monitoring=args.resource_monitoring,
            ksys_monitoring=args.ksys_monitoring,
            extra_args=args.simulate_args,
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        record = ReplayRunRecord(
            placement=name,
            cpus=placement.cpus,
            llc_ids=placement.llc_ids,
            agent_assignments=[asdict(item) for item in placement.agent_assignments],
            command=command,
            env={
                "ARM_IMAGE_MODE": env["ARM_IMAGE_MODE"],
                "PYTHONPATH": env["PYTHONPATH"],
            },
            run_dir=str(run_dir),
            returncode=None,
            skipped=args.dry_run,
        )
        (run_dir / "run_config.json").write_text(
            json.dumps(asdict(record), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[{name}] {' '.join(command)}", flush=True)
        if args.dry_run:
            records.append(record)
            continue

        completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
        finished = ReplayRunRecord(
            placement=record.placement,
            cpus=record.cpus,
            llc_ids=record.llc_ids,
            agent_assignments=record.agent_assignments,
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
                "source_trace": str(args.source_trace),
                "task_source": str(args.task_source),
                "num_agents": args.num_agents,
                "replay_speed": args.replay_speed,
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
            "Replay one existing trace as N identical cloud_model agents under "
            "topology-derived per-agent Docker cpuset placement. No LLM API is called."
        ),
    )
    parser.add_argument("--source-trace", type=Path, required=True)
    parser.add_argument(
        "--task-source",
        type=Path,
        default=Path("data/swe-rebench/tasks.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("traces/experiments/kunpeng_llc_replay"),
    )
    parser.add_argument(
        "--sys-cpu-root",
        type=Path,
        default=Path("/sys/devices/system/cpu"),
    )
    parser.add_argument("--container", choices=["docker", "podman"], default="docker")
    parser.add_argument(
        "--placements",
        default="auto",
        help=(
            "Comma-separated placement names, or 'auto' for every valid "
            "placement generated by the topology probe."
        ),
    )
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument(
        "--cluster-size",
        type=int,
        default=4,
        help=(
            "Inferred sub-LLC CPU cluster size forwarded to topology probe "
            "(default: 4 for public TaiShan v110 CCL descriptions)."
        ),
    )
    parser.add_argument("--replay-speed", type=float, default=1.0)
    parser.add_argument("--network-mode", default="none")
    parser.add_argument("--command-timeout", type=float, default=600.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--prep-concurrency", type=int, default=8)
    parser.add_argument(
        "--resource-monitoring",
        choices=["auto", "on", "off"],
        default="on",
    )
    parser.add_argument(
        "--ksys-monitoring",
        choices=["auto", "on", "off"],
        default="off",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("simulate_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.simulate_args and args.simulate_args[0] == "--":
        args.simulate_args = args.simulate_args[1:]
    if args.num_agents < 1:
        raise ValueError("num_agents must be at least 1")
    run_replay(args)


if __name__ == "__main__":
    main()
