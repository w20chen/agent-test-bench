#!/usr/bin/env python3
"""Legacy hardcoded scaling experiment for manual reproduction only.

Prefer ``run_kunpeng_llc_replay.py`` for publishable placement experiments:
that runner derives CPU assignments from the host topology instead of assuming
that CPU ids like ``0..7`` mean one physical core per agent.
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

# ---------------------------------------------------------------------------
# Hardcoded experiment configurations.
# Each entry: (experiment_name, list_of_cpuset_strings, num_agents_override)
# num_agents defaults to len(cpusets); set explicitly when they differ
# (e.g. os_default needs 8 agents but passes no --agent-cpuset flags).
# ---------------------------------------------------------------------------
EXPERIMENTS: list[tuple[str, list[str], int | None]] = [
    ("1agent_cpu0",          ["0"],                            None),
    ("2agent_cpu0_2",        ["0", "2"],                       None),
    ("4agent_cpu0_2_4_6",    ["0", "2", "4", "6"],             None),
    ("8agent_cpu0_7",        ["0", "1", "2", "3", "4", "5", "6", "7"], None),
    ("8agent_phys_0_14",     ["0", "2", "4", "6", "8", "10", "12", "14"], None),
    ("8agent_os_default",    [],                               8),
]


@dataclass(frozen=True, slots=True)
class RunRecord:
    name: str
    num_agents: int
    cpusets: list[str]
    command: list[str]
    run_dir: str
    returncode: int | None
    skipped: bool


def build_command(
    *,
    source_trace: Path,
    task_source: Path,
    output_dir: Path,
    cpusets: list[str],
    num_agents: int | None = None,
    container: str = "docker",
    network_mode: str = "none",
    replay_speed: float = 1.0,
    command_timeout_s: float = 600.0,
    workers: int = 1,
    prep_concurrency: int = 8,
    resource_monitoring: str = "on",
    ksys_monitoring: str = "off",
    extra_args: list[str] | None = None,
) -> list[str]:
    n_agents = num_agents if num_agents is not None else len(cpusets)
    cmd = [
        sys.executable,
        "-m", "trace_collect.cli", "simulate",
        "--source-trace", str(source_trace),
        "--task-source", str(task_source),
        "--output-dir", str(output_dir),
        "--mode", "cloud_model",
        "--container", container,
        "--network-mode", network_mode,
        "--num-agents", str(n_agents),
        "--trace-assignment", "manifest",
        "--arrival-mode", "closed_loop",
        "--replay-speed", str(replay_speed),
        "--cpu-limit", "1",
        "--command-timeout", str(command_timeout_s),
        "--resource-monitoring", resource_monitoring,
        "--pmu-monitoring", "off",
        "--ksys-monitoring", ksys_monitoring,
        "--workers", str(workers),
        "--prep-concurrency", str(prep_concurrency),
    ]
    if extra_args:
        cmd.extend(extra_args)
    for cpuset in cpusets:
        cmd.extend(["--agent-cpuset", cpuset])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Legacy hardcoded scaling experiment. Prefer "
            "run_kunpeng_llc_replay.py for topology-derived placements."
        ),
    )
    parser.add_argument(
        "--source-trace", type=Path, required=True,
        help="Path to the source trace.jsonl file.",
    )
    parser.add_argument(
        "--task-source", type=Path, default=Path("data/swe-rebench/tasks.json"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("traces/experiments/scaling_hardcoded"),
    )
    parser.add_argument(
        "--container", choices=["docker", "podman"], default="docker",
    )
    parser.add_argument("--replay-speed", type=float, default=1.0)
    parser.add_argument("--network-mode", default="none")
    parser.add_argument("--command-timeout", type=float, default=600.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--prep-concurrency", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-hardcoded-placement",
        action="store_true",
        help=(
            "Required acknowledgement: CPU ids are hardcoded and may not map "
            "to distinct physical cores or LLC domains on this host."
        ),
    )
    parser.add_argument("simulate_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.simulate_args and args.simulate_args[0] == "--":
        args.simulate_args = args.simulate_args[1:]
    if not args.allow_hardcoded_placement:
        parser.error(
            "refusing to run hardcoded CPU placements without "
            "--allow-hardcoded-placement; use run_kunpeng_llc_replay.py for "
            "topology-derived placements"
        )
    if not args.dry_run:
        if not args.source_trace.exists():
            parser.error(f"--source-trace does not exist: {args.source_trace}")
        if not args.task_source.exists():
            parser.error(f"--task-source does not exist: {args.task_source}")

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = args.output_root / timestamp

    env = os.environ.copy()
    env["ARM_IMAGE_MODE"] = "qemu"
    env["PYTHONPATH"] = (
        f"{REPO_ROOT / 'src'}{os.pathsep}{REPO_ROOT}"
        f"{os.pathsep}{env.get('PYTHONPATH', '')}"
    ).rstrip(os.pathsep)

    records: list[RunRecord] = []

    for name, cpusets, num_override in EXPERIMENTS:
        n_agents = num_override if num_override is not None else len(cpusets)
        run_dir = output_root / name
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = build_command(
            source_trace=args.source_trace,
            task_source=args.task_source,
            output_dir=run_dir,
            cpusets=cpusets,
            num_agents=num_override,
            container=args.container,
            network_mode=args.network_mode,
            replay_speed=args.replay_speed,
            command_timeout_s=args.command_timeout,
            workers=args.workers,
            prep_concurrency=args.prep_concurrency,
            extra_args=list(args.simulate_args) if args.simulate_args else None,
        )

        record = RunRecord(
            name=name,
            num_agents=n_agents,
            cpusets=cpusets,
            command=cmd,
            run_dir=str(run_dir),
            returncode=None,
            skipped=args.dry_run,
        )

        # Write run config before execution
        (run_dir / "run_config.json").write_text(
            json.dumps(asdict(record), indent=2) + "\n", encoding="utf-8"
        )

        print(f"\n{'='*60}")
        print(f"[{name}]  agents={n_agents}  cpusets={cpusets}")
        print(f"[{name}]  {' '.join(cmd)}")
        print(f"{'='*60}\n", flush=True)

        if args.dry_run:
            records.append(record)
            continue

        completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False)

        finished = RunRecord(
            name=record.name,
            num_agents=record.num_agents,
            cpusets=record.cpusets,
            command=record.command,
            run_dir=record.run_dir,
            returncode=completed.returncode,
            skipped=False,
        )
        (run_dir / "run_config.json").write_text(
            json.dumps(asdict(finished), indent=2) + "\n", encoding="utf-8"
        )
        records.append(finished)

        if completed.returncode != 0:
            print(f"[{name}] FAILED with returncode={completed.returncode}", flush=True)
            raise SystemExit(completed.returncode)

    # Write experiment manifest
    (output_root / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "source_trace": str(args.source_trace),
                "task_source": str(args.task_source),
                "replay_speed": args.replay_speed,
                "runs": [asdict(r) for r in records],
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print(f"\nDone. Results in {output_root}")


if __name__ == "__main__":
    main()
