#!/usr/bin/env python3
"""Run stable-kmeans NUMA placement experiments.

The parent process is intentionally small: it builds placement commands,
launches one fixed workload under each placement, and records raw evidence.
The child workload is embedded so the experiment does not depend on a solved
Terminal-Bench container state.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import textwrap
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import resource
except ModuleNotFoundError:  # pragma: no cover - Windows dry-run support.
    resource = None  # type: ignore[assignment]


DEFAULT_STRATEGIES = (
    "same_numa",
    "cross_numa_remote_mem",
    "cross_numa_interleave",
)

EXPECTED_NODE_CPUS = {
    "0": "0-15,64-79",
    "1": "16-31,80-95",
    "2": "32-47,96-111",
    "3": "48-63,112-127",
}


WORKLOAD_SOURCE = r'''
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def _compute_score(
    data: np.ndarray,
    k: int,
    percent_subsampling: float,
    seed: int,
    n_init: int,
    max_iter: int,
) -> float:
    rng = np.random.default_rng(seed)
    n_samples = int(percent_subsampling * len(data))
    if n_samples < 2:
        raise ValueError("subsample size must be at least 2")

    idx1 = rng.choice(len(data), n_samples, replace=True)
    idx2 = rng.choice(len(data), n_samples, replace=True)
    sample1 = data[idx1]
    sample2 = data[idx2]

    km1 = KMeans(
        n_clusters=k,
        random_state=seed,
        n_init=n_init,
        max_iter=max_iter,
        algorithm="lloyd",
    )
    km2 = KMeans(
        n_clusters=k,
        random_state=seed + 1,
        n_init=n_init,
        max_iter=max_iter,
        algorithm="lloyd",
    )
    labels1 = km1.fit_predict(sample1)
    labels2 = km2.fit_predict(sample2)

    common = min(len(labels1), len(labels2), 2048)
    if common < 2:
        return 0.0
    labels1 = labels1[:common]
    labels2 = labels2[:common]
    same1 = labels1[:, None] == labels1[None, :]
    same2 = labels2[:, None] == labels2[None, :]
    np.fill_diagonal(same1, False)
    np.fill_diagonal(same2, False)
    numerator = np.count_nonzero(same1 & same2)
    denom = float(np.sqrt(np.count_nonzero(same1) * np.count_nonzero(same2)))
    return 0.0 if denom == 0.0 else numerator / denom


def _compute_k(
    data: np.ndarray,
    k: int,
    subsamples: int,
    percent_subsampling: float,
    seed: int,
    n_init: int,
    max_iter: int,
) -> tuple[int, list[float]]:
    scores = [
        _compute_score(
            data=data,
            k=k,
            percent_subsampling=percent_subsampling,
            seed=seed + k * 100_000 + i,
            n_init=n_init,
            max_iter=max_iter,
        )
        for i in range(subsamples)
    ]
    return k, scores


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["serial", "parallel"], required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--features", type=int, required=True)
    parser.add_argument("--centers", type=int, required=True)
    parser.add_argument("--k-max", type=int, required=True)
    parser.add_argument("--subsamples", type=int, required=True)
    parser.add_argument("--percent-subsampling", type=float, required=True)
    parser.add_argument("--n-jobs", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-init", type=int, required=True)
    parser.add_argument("--max-iter", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.features <= 0:
        raise ValueError("--features must be positive")
    if args.centers <= 1:
        raise ValueError("--centers must be greater than 1")
    if args.centers > args.samples:
        raise ValueError("--centers must be less than or equal to --samples")
    if args.k_max <= 2:
        raise ValueError("--k-max must be greater than 2")
    if args.subsamples <= 0:
        raise ValueError("--subsamples must be positive")
    if args.n_jobs <= 0:
        raise ValueError("--n-jobs must be positive")
    if not 0.0 < args.percent_subsampling <= 1.0:
        raise ValueError("--percent-subsampling must be in (0, 1]")
    subsample_size = int(args.percent_subsampling * args.samples)
    if args.k_max - 1 > subsample_size:
        raise ValueError("--k-max - 1 must be <= subsample size")

    start = time.perf_counter()
    data, _ = make_blobs(
        n_samples=args.samples,
        n_features=args.features,
        centers=args.centers,
        random_state=args.seed,
        cluster_std=1.5,
    )
    data = np.ascontiguousarray(data, dtype=np.float32)

    k_values = list(range(2, args.k_max))
    if args.mode == "serial":
        results = [
            _compute_k(
                data=data,
                k=k,
                subsamples=args.subsamples,
                percent_subsampling=args.percent_subsampling,
                seed=args.seed,
                n_init=args.n_init,
                max_iter=args.max_iter,
            )
            for k in k_values
        ]
    else:
        results = Parallel(n_jobs=args.n_jobs, backend="loky")(
            delayed(_compute_k)(
                data=data,
                k=k,
                subsamples=args.subsamples,
                percent_subsampling=args.percent_subsampling,
                seed=args.seed,
                n_init=args.n_init,
                max_iter=args.max_iter,
            )
            for k in k_values
        )

    elapsed = time.perf_counter() - start
    summary = {
        str(k): {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        }
        for k, scores in results
    }
    optimal_k = max(summary, key=lambda key: summary[key]["mean"])
    print(
        json.dumps(
            {
                "mode": args.mode,
                "samples": args.samples,
                "features": args.features,
                "centers": args.centers,
                "k_max": args.k_max,
                "subsamples": args.subsamples,
                "percent_subsampling": args.percent_subsampling,
                "n_jobs": args.n_jobs,
                "seed": args.seed,
                "elapsed_s": elapsed,
                "optimal_k": int(optimal_k),
                "summary": summary,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
'''


@dataclass(frozen=True)
class Strategy:
    name: str
    mode: str
    cpu_bind: str | None
    mem_bind: str | None
    interleave: str | None
    n_jobs: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stable-parallel-kmeans under same-NUMA and cross-NUMA placement."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("traces/experiments/tb_stable_kmeans_numa"))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--centers", type=int, default=16)
    parser.add_argument("--k-max", type=int, default=24)
    parser.add_argument("--subsamples", type=int, default=32)
    parser.add_argument("--percent-subsampling", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--same-numa-cpus", default="0-15")
    parser.add_argument("--cross-numa-cpus", default="16-63")
    parser.add_argument("--cross-numa-matched-mask-cpus", default="16-31")
    parser.add_argument("--same-numa-node", default="0")
    parser.add_argument("--interleave-nodes", default="0-3")
    parser.add_argument("--same-numa-jobs", type=int, default=16)
    parser.add_argument("--cross-numa-jobs", type=int, default=16)
    parser.add_argument("--extreme-cross-numa-jobs", type=int, default=48)
    parser.add_argument("--strategies", default=None)
    parser.add_argument("--include-serial-baseline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-topology-mismatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python-bin", default=sys.executable)
    return parser.parse_args()


def run_capture(command: list[str], timeout_s: int = 15) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except Exception as exc:
        return {"command": command, "error": repr(exc)}


def normalize_cpu_list(cpu_list: str) -> str:
    return cpu_list.strip().replace(" ", "")


def read_sysfs_numa_cpus() -> dict[str, str]:
    nodes: dict[str, str] = {}
    node_root = Path("/sys/devices/system/node")
    if not node_root.exists():
        return nodes
    for path in sorted(node_root.glob("node[0-9]*/cpulist")):
        node_id = path.parent.name.removeprefix("node")
        nodes[node_id] = normalize_cpu_list(path.read_text(encoding="utf-8"))
    return nodes


def validate_expected_topology(allow_mismatch: bool) -> dict[str, Any]:
    observed = read_sysfs_numa_cpus()
    if not observed:
        if allow_mismatch:
            return {"status": "not_checked", "reason": "sysfs NUMA topology unavailable"}
        raise SystemExit(
            "could not read /sys/devices/system/node/node*/cpulist; "
            "rerun with --allow-topology-mismatch only for exploratory dry runs"
        )

    mismatches = {
        node: {"expected": expected, "observed": observed.get(node)}
        for node, expected in EXPECTED_NODE_CPUS.items()
        if normalize_cpu_list(observed.get(node, "")) != normalize_cpu_list(expected)
    }
    if mismatches and not allow_mismatch:
        raise SystemExit(
            "host NUMA topology does not match the expected Xeon Gold 6530 mapping: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {
        "status": "matched" if not mismatches else "mismatch_allowed",
        "expected": EXPECTED_NODE_CPUS,
        "observed": observed,
        "mismatches": mismatches,
    }


def build_strategies(args: argparse.Namespace) -> dict[str, Strategy]:
    strategies = {
        "same_numa": Strategy(
            name="same_numa",
            mode="parallel",
            cpu_bind=args.same_numa_cpus,
            mem_bind=args.same_numa_node,
            interleave=None,
            n_jobs=args.same_numa_jobs,
        ),
        "cross_numa_remote_mem": Strategy(
            name="cross_numa_remote_mem",
            mode="parallel",
            cpu_bind=args.cross_numa_cpus,
            mem_bind=args.same_numa_node,
            interleave=None,
            n_jobs=args.cross_numa_jobs,
        ),
        "cross_numa_remote_mem_matched_mask": Strategy(
            name="cross_numa_remote_mem_matched_mask",
            mode="parallel",
            cpu_bind=args.cross_numa_matched_mask_cpus,
            mem_bind=args.same_numa_node,
            interleave=None,
            n_jobs=args.cross_numa_jobs,
        ),
        "cross_numa_interleave": Strategy(
            name="cross_numa_interleave",
            mode="parallel",
            cpu_bind=args.cross_numa_cpus,
            mem_bind=None,
            interleave=args.interleave_nodes,
            n_jobs=args.cross_numa_jobs,
        ),
        "cross_numa_remote_mem_extreme": Strategy(
            name="cross_numa_remote_mem_extreme",
            mode="parallel",
            cpu_bind=args.cross_numa_cpus,
            mem_bind=args.same_numa_node,
            interleave=None,
            n_jobs=args.extreme_cross_numa_jobs,
        ),
    }
    if args.include_serial_baseline:
        strategies["serial_same_numa"] = Strategy(
            name="serial_same_numa",
            mode="serial",
            cpu_bind=args.same_numa_cpus,
            mem_bind=args.same_numa_node,
            interleave=None,
            n_jobs=1,
        )
    return strategies


def build_command(
    *,
    args: argparse.Namespace,
    strategy: Strategy,
    workload_path: Path,
    run_index: int,
) -> list[str]:
    command: list[str] = []
    if strategy.cpu_bind is not None or strategy.mem_bind is not None or strategy.interleave is not None:
        command.append("numactl")
        if strategy.cpu_bind is not None:
            command.append(f"--physcpubind={strategy.cpu_bind}")
        if strategy.mem_bind is not None:
            command.append(f"--membind={strategy.mem_bind}")
        if strategy.interleave is not None:
            command.append(f"--interleave={strategy.interleave}")

    command.extend(
        [
            args.python_bin,
            str(workload_path),
            "--mode",
            strategy.mode,
            "--samples",
            str(args.samples),
            "--features",
            str(args.features),
            "--centers",
            str(args.centers),
            "--k-max",
            str(args.k_max),
            "--subsamples",
            str(args.subsamples),
            "--percent-subsampling",
            str(args.percent_subsampling),
            "--n-jobs",
            str(strategy.n_jobs),
            "--seed",
            str(args.seed + run_index),
            "--n-init",
            str(args.n_init),
            "--max-iter",
            str(args.max_iter),
        ]
    )
    return command


def parse_child_json(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_gnu_time(stderr: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for line in stderr.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metrics[key.strip()] = value.strip()
    return metrics


def write_manifest(
    args: argparse.Namespace,
    output_dir: Path,
    strategies: list[Strategy],
    topology_validation: dict[str, Any],
) -> None:
    manifest = {
        "created_at_unix": time.time(),
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "cpu_count": os.cpu_count(),
        },
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "strategies": [asdict(strategy) for strategy in strategies],
        "topology_validation": topology_validation,
        "tools": {
            "numactl": shutil.which("numactl"),
            "lscpu": shutil.which("lscpu"),
            "gnu_time": shutil.which("time") or ("/usr/bin/time" if Path("/usr/bin/time").exists() else None),
        },
        "lscpu": run_capture(["lscpu"]) if shutil.which("lscpu") else None,
        "numactl_hardware": run_capture(["numactl", "--hardware"]) if shutil.which("numactl") else None,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def summarize(results_path: Path, summary_path: Path) -> None:
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row["returncode"] != 0:
            continue
        child = row.get("child_json") or {}
        elapsed = child.get("elapsed_s", row["wall_elapsed_s"])
        grouped.setdefault(row["strategy"], []).append(float(elapsed))

    lines = ["# Stable K-Means NUMA Summary", ""]
    for strategy, values in sorted(grouped.items()):
        mean = sum(values) / len(values)
        median = statistics.median(values)
        lines.append(
            f"- {strategy}: runs={len(values)}, mean_s={mean:.6f}, median_s={median:.6f}, "
            f"min_s={min(values):.6f}, max_s={max(values):.6f}"
        )
    failed = [row for row in rows if row["returncode"] != 0]
    if failed:
        lines.append("")
        lines.append(f"Failed runs: {len(failed)}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.runs <= 0:
        raise SystemExit("--runs must be positive")
    if shutil.which("numactl") is None and not args.dry_run:
        raise SystemExit("numactl is required on the Linux host for this experiment")
    if args.dry_run and shutil.which("numactl") is None:
        topology_validation = {"status": "not_checked", "reason": "numactl unavailable during dry run"}
    else:
        topology_validation = validate_expected_topology(args.allow_topology_mismatch)

    available = build_strategies(args)
    if args.strategies is None:
        requested_names = list(DEFAULT_STRATEGIES)
        if args.include_serial_baseline:
            requested_names.append("serial_same_numa")
    else:
        requested_names = [name.strip() for name in args.strategies.split(",") if name.strip()]
    unknown = [name for name in requested_names if name not in available]
    if unknown:
        raise SystemExit(f"unknown strategies: {', '.join(unknown)}")
    strategies = [available[name] for name in requested_names]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    if results_path.exists() and not args.resume and not args.dry_run:
        raise SystemExit(
            f"{results_path} already exists; choose a new --output-dir or pass --resume "
            "to intentionally append compatible runs"
        )
    workload_path = args.output_dir / "stable_kmeans_numa_workload.py"
    workload_path.write_text(textwrap.dedent(WORKLOAD_SOURCE).lstrip(), encoding="utf-8")
    write_manifest(args, args.output_dir, strategies, topology_validation)

    planned = [
        {
            "strategy": strategy.name,
            "run_index": run_index,
            "command": build_command(
                args=args,
                strategy=strategy,
                workload_path=workload_path,
                run_index=run_index,
            ),
        }
        for run_index in range(args.runs)
        for strategy in strategies
    ]
    (args.output_dir / "planned_commands.json").write_text(json.dumps(planned, indent=2), encoding="utf-8")
    if args.dry_run:
        for item in planned:
            print(json.dumps(item, sort_keys=True))
        return

    with results_path.open("a", encoding="utf-8") as results_file:
        for item in planned:
            run_dir = args.output_dir / item["strategy"] / f"run_{item['run_index']:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            gnu_time_path = run_dir / "time_verbose.txt"
            gnu_time_bin = "/usr/bin/time" if Path("/usr/bin/time").exists() else None
            execute_command = (
                [gnu_time_bin, "-v", "-o", str(gnu_time_path), *item["command"]]
                if gnu_time_bin is not None
                else item["command"]
            )
            started = time.perf_counter()
            usage_before = resource.getrusage(resource.RUSAGE_CHILDREN) if resource is not None else None
            completed = subprocess.run(
                execute_command,
                check=False,
                text=True,
                capture_output=True,
            )
            usage_after = resource.getrusage(resource.RUSAGE_CHILDREN) if resource is not None else None
            wall_elapsed = time.perf_counter() - started
            user_cpu_s = (
                usage_after.ru_utime - usage_before.ru_utime
                if usage_before is not None and usage_after is not None
                else None
            )
            system_cpu_s = (
                usage_after.ru_stime - usage_before.ru_stime
                if usage_before is not None and usage_after is not None
                else None
            )
            max_rss_kb = usage_after.ru_maxrss if usage_after is not None else None
            gnu_time_text = gnu_time_path.read_text(encoding="utf-8") if gnu_time_path.exists() else ""
            (run_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
            (run_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")

            record = {
                "strategy": item["strategy"],
                "run_index": item["run_index"],
                "command": item["command"],
                "execute_command": execute_command,
                "returncode": completed.returncode,
                "wall_elapsed_s": wall_elapsed,
                "child_user_cpu_s": user_cpu_s,
                "child_system_cpu_s": system_cpu_s,
                "cumulative_child_max_rss_kb": max_rss_kb,
                "gnu_time_metrics": parse_gnu_time(gnu_time_text),
                "run_dir": str(run_dir),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "child_json": parse_child_json(completed.stdout),
            }
            results_file.write(json.dumps(record, sort_keys=True) + "\n")
            results_file.flush()
            print(
                f"{item['strategy']} run={item['run_index']} "
                f"rc={completed.returncode} wall={wall_elapsed:.3f}s",
                flush=True,
            )

    summarize(results_path, args.output_dir / "summary.md")


if __name__ == "__main__":
    main()
