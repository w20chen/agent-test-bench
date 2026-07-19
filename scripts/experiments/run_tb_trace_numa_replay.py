#!/usr/bin/env python3
"""Replay one real agent trace under fixed NUMA placement policies."""

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


@dataclass(frozen=True)
class Strategy:
    name: str
    cpuset: str
    numactl_args: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a real trace with trace_collect.cli simulate under NUMA policies."
    )
    parser.add_argument("--source-trace", type=Path, required=True)
    parser.add_argument("--task-source", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("traces/experiments/tb_trace_numa_replay"))
    parser.add_argument("--container", choices=["docker", "podman"], default="docker")
    parser.add_argument("--network-mode", default="none")
    parser.add_argument("--replay-speed", type=float, default=1.0)
    parser.add_argument("--command-timeout", type=float, default=1800.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--prep-concurrency", type=int, default=1)
    parser.add_argument("--same-numa-cpuset", default="0-15")
    parser.add_argument("--cross-numa-cpuset", default="16-63")
    parser.add_argument("--matched-remote-cpuset", default="16-31")
    parser.add_argument("--mem-node", default="0")
    parser.add_argument("--interleave-nodes", default="0-3")
    parser.add_argument(
        "--strategies",
        default="same_numa,cross_numa_remote_mem,cross_numa_interleave",
        help=(
            "Comma-separated: same_numa,cross_numa_remote_mem,"
            "cross_numa_interleave,cross_numa_remote_mem_matched_mask"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("simulate_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.simulate_args and args.simulate_args[0] == "--":
        args.simulate_args = args.simulate_args[1:]
    return args


def strategies(args: argparse.Namespace) -> dict[str, Strategy]:
    return {
        "same_numa": Strategy(
            "same_numa",
            args.same_numa_cpuset,
            [f"--physcpubind={args.same_numa_cpuset}", f"--membind={args.mem_node}"],
        ),
        "cross_numa_remote_mem": Strategy(
            "cross_numa_remote_mem",
            args.cross_numa_cpuset,
            [f"--physcpubind={args.cross_numa_cpuset}", f"--membind={args.mem_node}"],
        ),
        "cross_numa_interleave": Strategy(
            "cross_numa_interleave",
            args.cross_numa_cpuset,
            [f"--physcpubind={args.cross_numa_cpuset}", f"--interleave={args.interleave_nodes}"],
        ),
        "cross_numa_remote_mem_matched_mask": Strategy(
            "cross_numa_remote_mem_matched_mask",
            args.matched_remote_cpuset,
            [f"--physcpubind={args.matched_remote_cpuset}", f"--membind={args.mem_node}"],
        ),
    }


def build_command(args: argparse.Namespace, strategy: Strategy, output_dir: Path) -> list[str]:
    command = [
        "numactl",
        *strategy.numactl_args,
        sys.executable,
        "-m",
        "trace_collect.cli",
        "simulate",
        "--source-trace",
        str(args.source_trace),
        "--output-dir",
        str(output_dir),
        "--mode",
        "cloud_model",
        "--container",
        args.container,
        "--network-mode",
        args.network_mode,
        "--num-agents",
        "1",
        "--trace-assignment",
        "manifest",
        "--arrival-mode",
        "closed_loop",
        "--replay-speed",
        str(args.replay_speed),
        "--cpu-limit",
        str(max(1, len(expand_cpuset(strategy.cpuset)))),
        "--agent-cpuset",
        strategy.cpuset,
        "--command-timeout",
        str(args.command_timeout),
        "--resource-monitoring",
        "on",
        "--pmu-monitoring",
        "off",
        "--ksys-monitoring",
        "off",
        "--workers",
        str(args.workers),
        "--prep-concurrency",
        str(args.prep_concurrency),
        *args.simulate_args,
    ]
    if args.task_source is not None:
        insert_at = command.index("--output-dir")
        command[insert_at:insert_at] = ["--task-source", str(args.task_source)]
    return command


def expand_cpuset(cpuset: str) -> list[int]:
    cpus: list[int] = []
    for part in cpuset.split(","):
        if "-" in part:
            start, end = part.split("-", 1)
            cpus.extend(range(int(start), int(end) + 1))
        elif part:
            cpus.append(int(part))
    return cpus


def main() -> None:
    args = parse_args()
    if not args.source_trace.exists():
        raise SystemExit(f"--source-trace does not exist: {args.source_trace}")
    if args.task_source is not None and not args.task_source.exists():
        raise SystemExit(f"--task-source does not exist: {args.task_source}")
    if shutil.which("numactl") is None and not args.dry_run:
        raise SystemExit("numactl is required for non-dry-run execution")

    available = strategies(args)
    selected = [name.strip() for name in args.strategies.split(",") if name.strip()]
    unknown = [name for name in selected if name not in available]
    if unknown:
        raise SystemExit(f"unknown strategies: {', '.join(unknown)}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = args.output_root / timestamp
    output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}{os.pathsep}{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)

    records = []
    for name in selected:
        strategy = available[name]
        run_dir = output_root / name
        command = build_command(args, strategy, run_dir)
        record = {
            "strategy": asdict(strategy),
            "command": command,
            "run_dir": str(run_dir),
            "returncode": None,
            "skipped": args.dry_run,
        }
        print(" ".join(command), flush=True)
        if not args.dry_run:
            run_dir.mkdir(parents=True, exist_ok=True)
            completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
            record["returncode"] = completed.returncode
            if completed.returncode != 0:
                records.append(record)
                break
        records.append(record)

    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "source_trace": str(args.source_trace),
                "task_source": str(args.task_source) if args.task_source else "auto",
                "records": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if any(record.get("returncode") not in (None, 0) for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
