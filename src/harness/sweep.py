from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from harness.trace_logger import build_run_id


@dataclass(slots=True)
class SweepRun:
    """Expanded run cell from the sweep matrix."""

    system: str
    workload: str
    concurrency: int
    tasks_file: str
    output_file: str


@dataclass(slots=True)
class SweepExecutionConfig:
    """Execution parameters that define sweep reproducibility."""

    model: str
    arrival_mode: str
    arrival_rate_per_s: float | None
    arrival_seed: int | None
    task_source_overrides: dict[str, str]
    sweep_config_path: str


OC_ENV_PATTERN = re.compile(r"^\$\{oc\.env:(?P<key>[^,}]+),(?P<default>.+)\}$")


def _resolve_value(value: Any) -> Any:
    if isinstance(value, str):
        match = OC_ENV_PATTERN.match(value)
        if match:
            key = match.group("key").strip()
            default = match.group("default").strip().strip('"').strip("'")
            return os.environ.get(key, default)
        return value
    if isinstance(value, list):
        return [_resolve_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_value(item) for key, item in value.items()}
    return value


def load_config(path: Path) -> dict[str, Any]:
    """Load and resolve a YAML config with simple `${oc.env:...}` support."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _resolve_value(raw)


def resolve_task_source(task_source: str) -> Path:
    """Resolve a workload task source into a concrete JSON/JSONL file."""
    path = Path(task_source)
    if path.is_file():
        return path
    for candidate in ["tasks.json", "tasks.jsonl"]:
        candidate_path = path / candidate
        if candidate_path.is_file():
            return candidate_path
    raise FileNotFoundError(
        f"Could not resolve tasks file from task_source: {task_source}"
    )


def load_tasks(tasks_path: Path) -> list[dict[str, Any]]:
    """Load task payloads from JSON or JSONL."""
    if tasks_path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in tasks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return json.loads(tasks_path.read_text(encoding="utf-8"))


def expand_sweep_matrix(
    *,
    sweep_config_path: Path,
    configs_root: Path,
    output_root: Path,
    systems_filter: set[str] | None = None,
    workloads_filter: set[str] | None = None,
    task_source_overrides: dict[str, str] | None = None,
) -> list[SweepRun]:
    """Expand configs/sweeps/default.yaml into concrete run cells."""
    sweep_config = load_config(sweep_config_path)
    matrix = sweep_config["matrix"]
    task_source_overrides = task_source_overrides or {}
    runs: list[SweepRun] = []
    for system in matrix["systems"]:
        if systems_filter and system not in systems_filter:
            continue
        for workload in matrix["workloads"]:
            if workloads_filter and workload not in workloads_filter:
                continue
            workload_path = configs_root / "workloads" / f"{workload}.yaml"
            workload_config = load_config(workload_path)
            task_source = task_source_overrides.get(
                workload, workload_config["task_source"]
            )
            tasks_file = resolve_task_source(task_source)
            for concurrency in matrix["concurrency"]:
                run_id = (
                    f"{build_run_id(system, workload, concurrency)}_{len(runs):03d}"
                )
                output_file = output_root / f"{run_id}.json"
                runs.append(
                    SweepRun(
                        system=system,
                        workload=workload,
                        concurrency=int(concurrency),
                        tasks_file=str(tasks_file),
                        output_file=str(output_file),
                    )
                )
    return runs


def extract_agent_kwargs(
    workload_name: str, workload_config: dict[str, Any]
) -> dict[str, Any]:
    """Project workload config into the agent constructor kwargs we actually support."""
    if workload_name == "code_agent":
        raise ValueError(
            "code_agent was removed with its legacy scaffold; "
            "use trace_collect.cli benchmark collection instead."
        )
    raise ValueError(f"Unsupported workload config: {workload_name}")


async def execute_sweep(
    *,
    runs: list[SweepRun],
    configs_root: Path,
    model: str,
    arrival_mode: str,
    arrival_rate_per_s: float | None,
    arrival_seed: int | None,
    task_source_overrides: dict[str, str] | None = None,
    sweep_config_path: str | None = None,
) -> list[dict[str, Any]]:
    """Execute expanded run cells.

    The legacy sweep executor depended on the removed code-agent scaffold.
    Keep matrix expansion and dry-run manifests available, but fail execution
    explicitly instead of advertising a broken runtime path.
    """
    if not runs:
        return []
    raise ValueError(
        "Legacy sweep execution was removed with the code_agent scaffold; "
        "use trace_collect.cli benchmark collection instead."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand and execute the benchmark sweep matrix."
    )
    parser.add_argument("--sweep-config", default="configs/sweeps/default.yaml")
    parser.add_argument("--configs-root", default="configs")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--systems")
    parser.add_argument("--workloads")
    parser.add_argument("--task-source-override", action="append", default=[])
    parser.add_argument(
        "--arrival-mode", default="closed_loop", choices=["closed_loop", "poisson"]
    )
    parser.add_argument("--arrival-rate-per-s", type=float)
    parser.add_argument("--arrival-seed", type=int)
    parser.add_argument("--manifest-path")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_overrides(values: list[str]) -> dict[str, str]:
    """Parse repeated KEY=VALUE overrides into a dictionary."""
    overrides: dict[str, str] = {}
    for value in values:
        key, raw = value.split("=", 1)
        overrides[key] = raw
    return overrides


def main() -> None:
    args = parse_args()
    systems_filter = set(args.systems.split(",")) if args.systems else None
    workloads_filter = set(args.workloads.split(",")) if args.workloads else None
    runs = expand_sweep_matrix(
        sweep_config_path=Path(args.sweep_config),
        configs_root=Path(args.configs_root),
        output_root=Path(args.output_root),
        systems_filter=systems_filter,
        workloads_filter=workloads_filter,
        task_source_overrides=parse_overrides(args.task_source_override),
    )
    manifest_path = (
        Path(args.manifest_path)
        if args.manifest_path
        else Path(args.output_root) / "manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        manifest_payload = {
            "execution_config": asdict(
                SweepExecutionConfig(
                    model=args.model,
                    arrival_mode=args.arrival_mode,
                    arrival_rate_per_s=args.arrival_rate_per_s,
                    arrival_seed=args.arrival_seed,
                    task_source_overrides=parse_overrides(args.task_source_override),
                    sweep_config_path=str(Path(args.sweep_config)),
                )
            ),
            "runs": [asdict(run) for run in runs],
        }
        manifest_path.write_text(
            json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8"
        )
        return
    completed_runs = asyncio.run(
        execute_sweep(
            runs=runs,
            configs_root=Path(args.configs_root),
            model=args.model,
            arrival_mode=args.arrival_mode,
            arrival_rate_per_s=args.arrival_rate_per_s,
            arrival_seed=args.arrival_seed,
            task_source_overrides=parse_overrides(args.task_source_override),
            sweep_config_path=str(Path(args.sweep_config)),
        )
    )
    manifest_path.write_text(
        json.dumps(completed_runs, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
