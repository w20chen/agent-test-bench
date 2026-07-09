#!/usr/bin/env python3
"""Topology-derived Kunpeng LLC scaling replay runner.

This is the non-hardcoded counterpart to ``run_scaling_hardcoded.py``. For
each requested agent count, it probes the host topology, builds placements with
``probe_llc_topology.build_placements()``, and emits one replay command per
valid placement. CPU ids are never baked into this script.
"""

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
    build_placements,
    probe_topology,
    write_outputs,
)
from scripts.experiments.run_kunpeng_llc_replay import _build_command  # noqa: E402


@dataclass(frozen=True, slots=True)
class ScalingRunRecord:
    agent_count: int
    placement: str
    cpus: list[int] | None
    llc_ids: list[str]
    agent_assignments: list[dict[str, object]]
    command: list[str]
    env: dict[str, str]
    run_dir: str
    returncode: int | None
    skipped: bool


def _write_manifest(
    output_root: Path,
    *,
    args: argparse.Namespace,
    agent_counts: list[int],
    records: list[ScalingRunRecord],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "source_trace": str(args.source_trace),
                "task_source": str(args.task_source),
                "agent_counts": agent_counts,
                "placements": args.placements,
                "replay_speed": args.replay_speed,
                "cluster_size": args.cluster_size,
                "runs": [asdict(record) for record in records],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_agent_counts(value: str) -> list[int]:
    """Parse a comma-separated list of positive agent counts."""
    counts: list[int] = []
    for part in value.split(","):
        text = part.strip()
        if not text:
            continue
        count = int(text)
        if count <= 0:
            raise ValueError("agent counts must be positive")
        if count not in counts:
            counts.append(count)
    if not counts:
        raise ValueError("at least one agent count is required")
    return counts


def _select_placements(
    available: dict[str, object],
    requested: str,
    *,
    agent_count: int,
) -> list[str]:
    if requested == "auto":
        return list(available)
    selected = [name.strip() for name in requested.split(",") if name.strip()]
    missing = [name for name in selected if name not in available]
    if missing:
        raise ValueError(
            f"placement(s) unavailable for {agent_count} agents: "
            + ", ".join(missing)
        )
    return selected


def run_scaling(args: argparse.Namespace) -> list[ScalingRunRecord]:
    """Run or dry-run topology-derived scaling placements."""
    agent_counts = parse_agent_counts(args.agent_counts)
    if not args.dry_run:
        if not args.source_trace.exists():
            raise FileNotFoundError(f"--source-trace does not exist: {args.source_trace}")
        if not args.task_source.exists():
            raise FileNotFoundError(f"--task-source does not exist: {args.task_source}")

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = args.output_root / timestamp
    topology = probe_topology(args.sys_cpu_root)

    env = os.environ.copy()
    env["ARM_IMAGE_MODE"] = "qemu"
    env["PYTHONPATH"] = (
        f"{REPO_ROOT / 'src'}{os.pathsep}{REPO_ROOT}"
        f"{os.pathsep}{env.get('PYTHONPATH', '')}"
    ).rstrip(os.pathsep)

    records: list[ScalingRunRecord] = []
    for agent_count in agent_counts:
        placements = build_placements(
            topology,
            agent_count=agent_count,
            sys_cpu_root=args.sys_cpu_root,
            cluster_size=args.cluster_size,
        )
        count_root = output_root / f"n{agent_count}"
        write_outputs(
            output_dir=count_root / "topology",
            topology=topology,
            placements=placements,
        )
        selected_names = _select_placements(
            placements,
            args.placements,
            agent_count=agent_count,
        )
        for placement_name in selected_names:
            placement = placements[placement_name]
            run_dir = count_root / placement_name
            command = _build_command(
                source_trace=args.source_trace,
                task_source=args.task_source,
                output_dir=run_dir,
                placement=placement,
                container=args.container,
                num_agents=agent_count,
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
            record = ScalingRunRecord(
                agent_count=agent_count,
                placement=placement_name,
                cpus=placement.cpus,
                llc_ids=placement.llc_ids,
                agent_assignments=[
                    asdict(item) for item in placement.agent_assignments
                ],
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
            print(
                f"[n={agent_count} {placement_name}] {' '.join(command)}",
                flush=True,
            )
            if args.dry_run:
                records.append(record)
                continue

            completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
            finished = ScalingRunRecord(
                agent_count=record.agent_count,
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
            _write_manifest(
                output_root,
                args=args,
                agent_counts=agent_counts,
                records=records,
            )
            if completed.returncode != 0:
                raise SystemExit(completed.returncode)

    _write_manifest(
        output_root,
        args=args,
        agent_counts=agent_counts,
        records=records,
    )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run topology-derived Kunpeng LLC scaling replay for one or more "
            "agent counts. No LLM API is called."
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
        default=Path("traces/experiments/kunpeng_llc_scaling"),
    )
    parser.add_argument(
        "--sys-cpu-root",
        type=Path,
        default=Path("/sys/devices/system/cpu"),
    )
    parser.add_argument(
        "--agent-counts",
        default="1,2,4,8",
        help="Comma-separated positive agent counts.",
    )
    parser.add_argument(
        "--placements",
        default="auto",
        help=(
            "Comma-separated placement names, or 'auto' for every valid "
            "placement generated for each agent count."
        ),
    )
    parser.add_argument(
        "--cluster-size",
        type=int,
        default=4,
        help="Inferred sub-LLC CPU cluster size forwarded to topology probe.",
    )
    parser.add_argument("--container", choices=["docker", "podman"], default="docker")
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
    parse_agent_counts(args.agent_counts)
    return args


def main() -> None:
    run_scaling(parse_args())


if __name__ == "__main__":
    main()
