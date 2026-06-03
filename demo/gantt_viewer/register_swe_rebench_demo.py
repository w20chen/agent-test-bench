"""Register the swe-rebench Gantt demo traces via the existing REST API."""

from __future__ import annotations

import argparse
import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib import error, request

from demo.gantt_viewer.backend.ingest import ensure_canonical_trace_path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_URL = "http://127.0.0.1:8765"
DEMO_SERVER_PRECONDITION = (
    "demo bootstrap expects the server to run with "
    "demo/gantt_viewer/configs/swe_rebench_demo_bootstrap.yaml and an isolated "
    "GANTT_VIEWER_RUNTIME_STATE; /api/traces must contain only this demo cohort or be empty"
)


@dataclass(frozen=True)
class ExperimentSpec:
    """One experiment root participating in the demo cohort."""

    slot: str
    label_suffix: str
    root: Path


@dataclass(frozen=True)
class RegistrationEntry:
    """One trace registration to create or verify."""

    task_index: int
    task_id: str
    experiment: ExperimentSpec
    label: str
    path: Path


EXPERIMENT_SPECS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        slot="a",
        label_suffix="claude-code-haiku",
        root=REPO_ROOT / "traces" / "swe-rebench" / "claude-code-haiku",
    ),
    ExperimentSpec(
        slot="b",
        label_suffix="openclaw-haiku",
        root=(
            REPO_ROOT
            / "traces"
            / "swe-rebench"
            / "openclaw-top10-haiku"
            / "20260410T063155Z-top10-openclaw-haiku-cc-aligned-100iter-rerun4"
        ),
    ),
    ExperimentSpec(
        slot="c",
        label_suffix="arm",
        root=(
            REPO_ROOT
            / "traces"
            / "swe-rebench"
            / "swe-rebench-arm-10-tasks-openclaw-glm-docker-100"
        ),
    ),
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for demo registration."""
    parser = argparse.ArgumentParser(
        description="Register the swe-rebench demo traces with the Gantt viewer backend.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL for the running Gantt viewer backend.",
    )
    parser.add_argument(
        "--no-verify-payload",
        action="store_true",
        help="Skip the post-registration /api/payload verification request.",
    )
    return parser


def discover_tasks(experiment_root: Path) -> set[str]:
    """Return all task ids with an attempt_1 trace under one experiment root."""
    return {
        trace_path.parent.parent.name
        for trace_path in experiment_root.glob("*/attempt_1/trace.jsonl")
    }


def compute_shared_tasks(tasks_by_experiment: Mapping[str, Iterable[str]]) -> list[str]:
    """Compute the reproducible shared task set across experiment roots."""
    task_sets = [set(tasks) for tasks in tasks_by_experiment.values()]
    if not task_sets:
        raise ValueError("no experiment task sets provided")
    return sorted(set.intersection(*task_sets))


def build_registration_entries(
    shared_tasks: list[str],
    experiment_specs: Iterable[ExperimentSpec],
) -> list[RegistrationEntry]:
    """Build the task-major registration order and sortable labels."""
    entries: list[RegistrationEntry] = []
    for task_index, task_id in enumerate(shared_tasks, start=1):
        for experiment in experiment_specs:
            path = experiment.root / task_id / "attempt_1" / "trace.jsonl"
            if not path.is_file():
                raise FileNotFoundError(f"missing trace path: {path}")
            entries.append(
                RegistrationEntry(
                    task_index=task_index,
                    task_id=task_id,
                    experiment=experiment,
                    label=f"t{task_index:02d}-{experiment.slot}-{experiment.label_suffix}",
                    path=path.resolve(),
                )
            )
    return entries


def _normalize_trace_path(raw_path: str) -> str:
    return str(Path(raw_path).resolve())


@functools.lru_cache(maxsize=None)
def _canonical_demo_path(path: str) -> str:
    return str(ensure_canonical_trace_path(Path(path)).canonical_path.resolve())


def _format_precondition_error(detail: str) -> str:
    return f"{DEMO_SERVER_PRECONDITION}. {detail}"


def classify_existing_traces(
    planned_entries: Iterable[RegistrationEntry],
    existing_traces: Iterable[Mapping[str, Any]],
) -> tuple[list[RegistrationEntry], list[RegistrationEntry]]:
    """Split planned entries into already-matching and still-to-register sets."""
    planned_list = list(planned_entries)
    planned_by_path = {
        _canonical_demo_path(str(entry.path)): entry for entry in planned_list
    }
    existing_paths: list[str] = []

    for trace in existing_traces:
        path = _normalize_trace_path(str(trace["path"]))
        planned = planned_by_path.get(path)
        if planned is None:
            raise ValueError(
                _format_precondition_error(
                    f"found unexpected trace path in /api/traces: {path}"
                )
            )
        if trace.get("label") != planned.label:
            raise ValueError(
                _format_precondition_error(
                    "found a demo trace with a different label in /api/traces for "
                    f"{path}: {trace.get('label')!r} != {planned.label!r}"
                )
            )
        existing_paths.append(path)

    expected_existing_paths = [
        _canonical_demo_path(str(entry.path))
        for entry in planned_list
        if _canonical_demo_path(str(entry.path)) in set(existing_paths)
    ]
    if existing_paths != expected_existing_paths:
        raise ValueError(
            _format_precondition_error(
                "existing /api/traces order does not match the expected task-major demo order"
            )
        )

    existing_path_set = set(existing_paths)
    already_registered = [
        entry
        for entry in planned_list
        if _canonical_demo_path(str(entry.path)) in existing_path_set
    ]
    to_register = [
        entry
        for entry in planned_list
        if _canonical_demo_path(str(entry.path)) not in existing_path_set
    ]
    return already_registered, to_register


def verify_trace_order(
    traces: Iterable[Mapping[str, Any]],
    planned_entries: Iterable[RegistrationEntry],
) -> list[str]:
    """Verify the actual /api/traces order used by the frontend bootstrap."""
    actual_traces = list(traces)
    expected_entries = list(planned_entries)
    if len(actual_traces) != len(expected_entries):
        raise RuntimeError(
            f"expected {len(expected_entries)} traces from /api/traces, found {len(actual_traces)}"
        )

    actual_pairs = [
        (_normalize_trace_path(str(trace["path"])), str(trace.get("label")))
        for trace in actual_traces
    ]
    expected_pairs = [
        (_canonical_demo_path(str(entry.path)), entry.label)
        for entry in expected_entries
    ]
    if actual_pairs != expected_pairs:
        raise RuntimeError(
            "actual /api/traces order does not match the expected task-major demo order"
        )
    return [str(trace["id"]) for trace in actual_traces]


def load_demo_plan() -> list[RegistrationEntry]:
    """Compute the 30-entry demo registration plan from local trace roots."""
    tasks_by_experiment = {
        spec.label_suffix: discover_tasks(spec.root) for spec in EXPERIMENT_SPECS
    }
    shared_tasks = compute_shared_tasks(tasks_by_experiment)
    if len(shared_tasks) != 10:
        raise ValueError(
            f"expected 10 shared tasks, found {len(shared_tasks)}: {shared_tasks}"
        )
    return build_registration_entries(shared_tasks, EXPERIMENT_SPECS)


def fetch_json(
    url: str, *, method: str = "GET", payload: Mapping[str, Any] | None = None
) -> Any:
    """Issue a JSON request with urllib and parse the JSON response."""
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc


def register_demo(base_url: str, *, verify_payload: bool = True) -> dict[str, Any]:
    """Register the demo traces against a running backend and verify ordering."""
    plan = load_demo_plan()
    base = base_url.rstrip("/")

    traces_response = fetch_json(f"{base}/api/traces")
    existing_traces = traces_response.get("traces", [])
    already_registered, to_register = classify_existing_traces(plan, existing_traces)

    if to_register:
        fetch_json(
            f"{base}/api/traces/register",
            method="POST",
            payload={
                "paths": [str(entry.path) for entry in to_register],
                "labels_by_path": {
                    str(entry.path): entry.label for entry in to_register
                },
            },
        )

    refreshed_traces = fetch_json(f"{base}/api/traces").get("traces", [])
    ordered_ids = verify_trace_order(refreshed_traces, plan)

    payload_response: dict[str, Any] | None = None
    if verify_payload:
        payload_response = fetch_json(
            f"{base}/api/payload",
            method="POST",
            payload={"ids": ordered_ids},
        )
        errors = payload_response.get("errors", [])
        if errors:
            raise RuntimeError(f"payload verification reported errors: {errors}")

    return {
        "planned": len(plan),
        "already_registered": len(already_registered),
        "newly_registered": len(to_register),
        "ordered_ids": ordered_ids,
        "payload_traces": 0
        if payload_response is None
        else len(payload_response.get("traces", [])),
        "task_order": [entry.task_id for entry in plan if entry.experiment.slot == "a"],
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = build_parser().parse_args(argv)
    result = register_demo(args.base_url, verify_payload=not args.no_verify_payload)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
