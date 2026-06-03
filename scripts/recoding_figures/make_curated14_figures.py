"""Generate Plot 1-3 for the curated 14 Terminal-Bench recording attempts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DEFAULT_SELECTION = [
    (1, "fix-git", "remote-record-internals-10tasks-100iter-localtasks-20260508T165047Z"),
    (2, "dna-insert", "remote-record-internals-10tasks-100iter-localtasks-20260508T165047Z"),
    (3, "causal-inference-r", "remote-record-internals-10tasks-100iter-localtasks-20260508T165047Z"),
    (
        4,
        "security-celery-redis-rce",
        "remote-record-internals-10tasks-100iter-localtasks-20260508T165047Z",
    ),
    (
        5,
        "schemelike-metacircular-eval",
        "remote-record-internals-10tasks-100iter-localtasks-20260508T165047Z",
    ),
    (
        6,
        "multi-source-data-merger",
        "remote-record-internals-10tasks-100iter-continuation-20260508T191406Z",
    ),
    (
        7,
        "ode-solver-rk4",
        "remote-record-internals-10tasks-100iter-continuation-20260508T191406Z",
    ),
    (
        8,
        "git-leak-recovery",
        "remote-record-internals-2tasks-100iter-gitleak-feal-fix-20260509T041925Z",
    ),
    (
        9,
        "cancel-async-tasks",
        "remote-record-internals-10tasks-100iter-continuation-20260508T191406Z",
    ),
    (
        10,
        "analyze-access-logs",
        "remote-record-internals-10tasks-100iter-continuation-20260508T191406Z",
    ),
    (
        11,
        "jsonl-aggregator",
        "remote-record-internals-10tasks-100iter-continuation-20260508T191406Z",
    ),
    (
        12,
        "assign-seats",
        "remote-record-internals-10tasks-100iter-continuation-20260508T191406Z",
    ),
    (
        13,
        "recover-obfuscated-files",
        "remote-record-internals-10tasks-100iter-continuation-20260508T191406Z",
    ),
    (
        14,
        "countdown-game",
        "remote-record-internals-10tasks-100iter-continuation-20260508T191406Z",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--phase", choices=("all", "prefill", "decode"), default="all")
    parser.add_argument("--layers")
    parser.add_argument("--top-experts", type=int, default=64)
    parser.add_argument("--skip-combined", action="store_true")
    args = parser.parse_args()

    model_root = args.model_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    script = Path(__file__).with_name("make_figures.py").resolve()
    attempts = [_attempt_record(model_root, item) for item in DEFAULT_SELECTION]
    _validate_attempts(attempts)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_root": str(model_root),
        "output_dir": str(output_dir),
        "phase": args.phase,
        "layers": args.layers,
        "top_experts": args.top_experts,
        "tasks": attempts,
    }
    (output_dir / "manifest_inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")

    workers = max(1, int(args.workers))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        if not args.skip_combined:
            future = pool.submit(
                _run_make_figures,
                script,
                [Path(item["attempt_dir"]) for item in attempts],
                output_dir / "combined_all14",
                args.phase,
                args.layers,
                args.top_experts,
            )
            futures[future] = "combined_all14"
        for item in attempts:
            task_output = output_dir / "per_task" / f"{item['order']:02d}_{item['task_id']}"
            future = pool.submit(
                _run_make_figures,
                script,
                [Path(item["attempt_dir"])],
                task_output,
                args.phase,
                args.layers,
                args.top_experts,
            )
            futures[future] = f"{item['order']:02d}_{item['task_id']}"
        for future in as_completed(futures):
            label = futures[future]
            future.result()
            print(f"finished {label}", flush=True)

    print(f"wrote {output_dir}", flush=True)


def _attempt_record(model_root: Path, item: tuple[int, str, str]) -> dict[str, Any]:
    order, task_id, run_id = item
    attempt_dir = model_root / run_id / task_id / "attempt_1"
    recordings_dir = attempt_dir / "recordings"
    complete_iters = [
        path
        for path in sorted(recordings_dir.glob("iter_*"))
        if (path / "attention.npz").is_file()
        and (path / "routing.npz").is_file()
        and (path / "segments.json").is_file()
    ]
    return {
        "order": order,
        "task_id": task_id,
        "run_id": run_id,
        "attempt_dir": str(attempt_dir),
        "complete_iters": len(complete_iters),
    }


def _validate_attempts(attempts: list[dict[str, Any]]) -> None:
    errors: list[str] = []
    for item in attempts:
        attempt_dir = Path(item["attempt_dir"])
        if item["task_id"] == "feal-differential-cryptanalysis":
            errors.append("FEAL is intentionally excluded")
        if not attempt_dir.is_dir():
            errors.append(f"{item['task_id']}: missing attempt dir {attempt_dir}")
        if int(item["complete_iters"]) <= 0:
            errors.append(f"{item['task_id']}: no complete recording iterations")
    if errors:
        raise RuntimeError("\n".join(errors))


def _run_make_figures(
    script: Path,
    inputs: list[Path],
    output_dir: Path,
    phase: str,
    layers: str | None,
    top_experts: int,
) -> None:
    cmd = [
        sys.executable,
        str(script),
        *[str(path) for path in inputs],
        "--output-dir",
        str(output_dir),
        "--phase",
        phase,
        "--top-experts",
        str(top_experts),
    ]
    if layers:
        cmd.extend(["--layers", layers])
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
