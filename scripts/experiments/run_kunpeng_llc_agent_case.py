#!/usr/bin/env python3
"""Run one real 8-agent case under same-LLC and spread-LLC CPU placement."""

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


def _load_default_instance_id(tasks_path: Path) -> str:
    if not tasks_path.exists():
        raise FileNotFoundError(
            f"{tasks_path} does not exist; pass --instance-id explicitly or "
            "prepare SWE-rebench data first"
        )
    payload = json.loads(tasks_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{tasks_path} must contain a non-empty task list")
    first = payload[0]
    if not isinstance(first, dict) or "instance_id" not in first:
        raise ValueError(f"{tasks_path} entries must contain instance_id")
    return str(first["instance_id"])


def _collect_base_command(
    *,
    run_dir: Path,
    instance_id: str,
    container: str,
    mcp_config: str,
    concurrency: int,
    ksys_monitoring: str,
    extra_collect_args: list[str],
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "trace_collect.cli",
        "collect",
        "--benchmark",
        "swe-rebench",
        "--scaffold",
        "openclaw",
        "--container",
        container,
        "--mcp-config",
        mcp_config,
        "--instance-ids",
        instance_id,
        "--concurrency",
        str(concurrency),
        "--resource-monitoring",
        "on",
        "--pmu-monitoring",
        "off",
        "--ksys-monitoring",
        ksys_monitoring,
        "--run-id",
        str(run_dir),
        *extra_collect_args,
    ]


def _with_taskset(command: list[str], placement: Placement) -> list[str]:
    if placement.cpus is None:
        return command
    taskset = shutil.which("taskset")
    if taskset is None:
        raise RuntimeError("taskset is required for same_llc/spread_llc placement")
    return [taskset, "-c", format_cpu_list(placement.cpus), *command]


def run_experiment(args: argparse.Namespace) -> list[RunRecord]:
    repo = _repo_root()
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = args.output_root / timestamp
    topology_dir = output_root / "topology"

    topology = probe_topology(args.sys_cpu_root)
    placements = build_placements(topology, agent_count=args.concurrency)
    write_outputs(output_dir=topology_dir, topology=topology, placements=placements)

    instance_id = args.instance_id or _load_default_instance_id(args.tasks_json)
    selected_names = args.placements.split(",")
    unknown = [name for name in selected_names if name not in placements]
    if unknown:
        raise ValueError(f"unknown placement(s): {', '.join(unknown)}")

    env = os.environ.copy()
    env["ARM_IMAGE_MODE"] = "qemu"
    env["PYTHONPATH"] = f"{repo / 'src'}{os.pathsep}{repo}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)

    records: list[RunRecord] = []
    for name in selected_names:
        placement = placements[name]
        run_dir = output_root / name
        command = _collect_base_command(
            run_dir=run_dir,
            instance_id=instance_id,
            container=args.container,
            mcp_config=args.mcp_config,
            concurrency=args.concurrency,
            ksys_monitoring=args.ksys_monitoring,
            extra_collect_args=args.collect_args,
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
                "instance_id": instance_id,
                "concurrency": args.concurrency,
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
            "Run a fixed SWE-rebench/OpenClaw case with 8-agent same-LLC and "
            "spread-LLC CPU placement. Extra arguments after -- are passed to "
            "trace_collect.cli collect, e.g. -- --provider dashscope --model ..."
        ),
    )
    parser.add_argument("--instance-id", default=None)
    parser.add_argument(
        "--tasks-json",
        type=Path,
        default=Path("data/swe-rebench/tasks.json"),
        help="Used only when --instance-id is omitted.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("traces/experiments/kunpeng_llc"),
    )
    parser.add_argument("--sys-cpu-root", type=Path, default=Path("/sys/devices/system/cpu"))
    parser.add_argument("--container", default="docker", choices=["docker", "podman"])
    parser.add_argument("--mcp-config", default="none")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--placements",
        default="os_default,same_llc,spread_llc",
        help="Comma-separated subset of os_default,same_llc,spread_llc.",
    )
    parser.add_argument(
        "--ksys-monitoring",
        choices=["auto", "on", "off"],
        default="off",
        help="Forwarded to trace_collect.cli collect.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("collect_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.collect_args and args.collect_args[0] == "--":
        args.collect_args = args.collect_args[1:]
    if args.concurrency != 8:
        raise ValueError("this experiment script is intentionally fixed to 8 agents")
    run_experiment(args)


if __name__ == "__main__":
    main()
