"""Standalone HTML export for the Gantt viewer."""

from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from demo.gantt_viewer.backend.app import FRONTEND_DIST_PATH
from demo.gantt_viewer.backend.discovery import REPO_ROOT
from demo.gantt_viewer.backend.payload import build_gantt_payload_multi
from trace_collect.trace_inspector import CURRENT_TRACE_FORMAT_VERSION, TraceData


PRESET_SWE_REBENCH_GLM_OPENCLAW_100 = "swe-rebench-glm-openclaw-100"

MANIFEST_PATH = REPO_ROOT / "configs" / "simulate" / "openclaw-glm-19-manifest.json"
SIM_SWEEP_ROOT = (
    REPO_ROOT
    / "traces"
    / "simulate"
    / "imported_19tasks_closed_loop_plus5lambda_20260420"
)

EXPECTED_SCAFFOLD = "openclaw"
EXPECTED_MODEL = "z-ai/glm-5.1"
EXPECTED_MAX_ITERATIONS = 100
EXPECTED_TRACE_COUNT = 19
EXPECTED_BENCHMARK = "swe-rebench"
EXPECTED_BENCHMARK_SPLIT = "filtered"
EXPECTED_SOURCE_MODE = "collect"
EXPECTED_SOURCE_TRACE_ROOT = (
    REPO_ROOT / "traces" / "swe-rebench" / "z-ai-glm-5.1" / "20-tasks-combined"
).resolve()
EXPECTED_TASK_SOURCE = (REPO_ROOT / "data" / "swe-rebench" / "tasks.json").resolve()
EXPECTED_SIM_MODE = "simulate"
EXPECTED_SIMULATE_MODE = "cloud_model"
EXPECTED_SIM_REPLAY_TARGET = "cloud_replay"
EXPECTED_SIM_TRACE_MANIFEST = "configs/simulate/openclaw-glm-19-manifest.json"
EXPECTED_SOURCE_TRACE_REF_FRAGMENT = "/traces/swe-rebench/z-ai-glm-5.1/"

GROUP_ALL = "all"
GROUP_RAW = "raw"
GROUP_CLOSED_LOOP = "closed_loop"
POISSON_GROUPS: tuple[str, ...] = (
    "poisson_0.000833_per_s",
    "poisson_0.00167_per_s",
    "poisson_0.00333_per_s",
    "poisson_0.00667_per_s",
    "poisson_0.01333_per_s",
)
EXPORT_GROUPS: tuple[str, ...] = (GROUP_RAW, GROUP_CLOSED_LOOP, *POISSON_GROUPS)


DisplayMode = Literal["sync", "abs"]
ClockMode = Literal["wall", "real"]
ViewMode = Literal["layered", "concise"]
ThemeMode = Literal["dark", "light"]


@dataclass(frozen=True, slots=True)
class ExportTrace:
    task_id: str
    path: Path
    source_ref: str


@dataclass(frozen=True, slots=True)
class ExportGroup:
    name: str
    title: str
    output_name: str
    traces: tuple[ExportTrace, ...]
    time_mode: DisplayMode


@dataclass(frozen=True, slots=True)
class ExportedFile:
    group: str
    path: Path
    n_traces: int
    resource_samples: int
    empty_resource_timelines: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trace_collect.cli gantt-export",
        description="Export standalone Gantt HTML for curated trace cohorts.",
    )
    parser.add_argument(
        "--preset",
        choices=[PRESET_SWE_REBENCH_GLM_OPENCLAW_100],
        default=PRESET_SWE_REBENCH_GLM_OPENCLAW_100,
        help="Curated export preset.",
    )
    parser.add_argument(
        "--group",
        choices=[GROUP_ALL, *EXPORT_GROUPS],
        default=GROUP_ALL,
        help="Export one group or all groups from the preset.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results" / "gantt-viewer"),
        help="Directory for HTML exports and manifest.json.",
    )
    parser.add_argument(
        "--max-resource-samples-per-trace",
        type=int,
        default=None,
        help=(
            "Optionally downsample each trace resource timeline for smaller HTML. "
            "Default preserves all samples."
        ),
    )
    return parser


def export_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.preset != PRESET_SWE_REBENCH_GLM_OPENCLAW_100:
        raise ValueError(f"unsupported Gantt export preset: {args.preset}")
    return export_swe_rebench_glm_openclaw_100(
        output_dir=Path(args.output_dir),
        group=args.group,
        max_resource_samples_per_trace=args.max_resource_samples_per_trace,
    )


def export_swe_rebench_glm_openclaw_100(
    *,
    output_dir: Path,
    group: str = GROUP_ALL,
    max_resource_samples_per_trace: int | None = None,
) -> dict[str, Any]:
    """Export the verified GLM/OpenClaw/100-iteration Gantt cohort."""
    if group not in (GROUP_ALL, *EXPORT_GROUPS):
        raise ValueError(f"unknown export group: {group}")
    if max_resource_samples_per_trace is not None and max_resource_samples_per_trace < 2:
        raise ValueError("--max-resource-samples-per-trace must be at least 2")

    manifest_traces = _load_and_validate_manifest_traces(MANIFEST_PATH)
    groups = _build_export_groups(manifest_traces, group=group)

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frontend = _load_frontend_bundle(FRONTEND_DIST_PATH)

    exported: list[ExportedFile] = []
    for export_group in groups:
        payload = _build_group_payload(
            export_group,
            max_resource_samples_per_trace=max_resource_samples_per_trace,
        )
        resource_samples = sum(
            len(trace.get("resource_timeline") or [])
            for trace in payload["payload"]["traces"]
        )
        empty_resource_timelines = sum(
            1
            for trace in payload["payload"]["traces"]
            if not trace.get("resource_timeline")
        )
        output_path = output_dir / export_group.output_name
        output_path.write_text(
            _render_html(
                title=export_group.title,
                snapshot=payload,
                frontend=frontend,
            ),
            encoding="utf-8",
        )
        exported.append(
            ExportedFile(
                group=export_group.name,
                path=output_path,
                n_traces=len(payload["payload"]["traces"]),
                resource_samples=resource_samples,
                empty_resource_timelines=empty_resource_timelines,
            )
        )

    manifest_payload = {
        "preset": PRESET_SWE_REBENCH_GLM_OPENCLAW_100,
        "source_manifest": str(MANIFEST_PATH),
        "source_trace_count": len(manifest_traces),
        "expected": {
            "benchmark": EXPECTED_BENCHMARK,
            "benchmark_split": EXPECTED_BENCHMARK_SPLIT,
            "scaffold": EXPECTED_SCAFFOLD,
            "model": EXPECTED_MODEL,
            "max_iterations": EXPECTED_MAX_ITERATIONS,
            "trace_count": EXPECTED_TRACE_COUNT,
        },
        "max_resource_samples_per_trace": max_resource_samples_per_trace,
        "exports": [
            {
                "group": item.group,
                "path": str(item.path),
                "n_traces": item.n_traces,
                "resource_samples": item.resource_samples,
                "empty_resource_timelines": item.empty_resource_timelines,
            }
            for item in exported
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_payload["manifest_path"] = str(manifest_path)
    return manifest_payload


def _load_and_validate_manifest_traces(manifest_path: Path) -> tuple[ExportTrace, ...]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{manifest_path} must contain a non-empty list")

    traces: list[ExportTrace] = []
    seen_tasks: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or not isinstance(item.get("source_trace"), str):
            raise ValueError(f"manifest entry #{index} must define source_trace")
        path = (manifest_path.parent / item["source_trace"]).resolve()
        task_id = path.parent.parent.name
        if task_id in seen_tasks:
            raise ValueError(f"duplicate task in manifest: {task_id}")
        seen_tasks.add(task_id)
        source_metadata = _validate_source_trace(path)
        source_ref = _source_ref_from_raw_metadata(source_metadata, trace_path=path)
        _validate_manifest_entry_provenance(
            item,
            trace_path=path,
            task_id=task_id,
            manifest_path=manifest_path,
            index=index,
        )
        traces.append(ExportTrace(task_id=task_id, path=path, source_ref=source_ref))
    if len(traces) != EXPECTED_TRACE_COUNT:
        raise ValueError(
            f"{manifest_path} contains {len(traces)} traces; "
            f"expected exactly {EXPECTED_TRACE_COUNT}"
        )
    return tuple(traces)


def _validate_source_trace(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"source trace not found: {path}")
    metadata = _read_trace_metadata(path)
    expected = {
        "type": "trace_metadata",
        "trace_format_version": CURRENT_TRACE_FORMAT_VERSION,
        "scaffold": EXPECTED_SCAFFOLD,
        "model": EXPECTED_MODEL,
        "max_iterations": EXPECTED_MAX_ITERATIONS,
        "mode": EXPECTED_SOURCE_MODE,
        "benchmark": EXPECTED_BENCHMARK,
        "benchmark_split": EXPECTED_BENCHMARK_SPLIT,
    }
    _validate_expected_metadata(path, metadata, expected)
    task_id = path.parent.parent.name
    if metadata.get("instance_id") != task_id:
        raise ValueError(
            f"{path} does not match export preset: "
            f"instance_id={metadata.get('instance_id')!r} expected {task_id!r}"
        )
    return metadata


def _validate_manifest_entry_provenance(
    item: dict[str, Any],
    *,
    trace_path: Path,
    task_id: str,
    manifest_path: Path,
    index: int,
) -> None:
    if not _is_relative_to(trace_path, EXPECTED_SOURCE_TRACE_ROOT):
        raise ValueError(
            f"manifest entry #{index} source_trace {trace_path} is outside "
            f"{EXPECTED_SOURCE_TRACE_ROOT}"
        )
    _validate_source_trace_reference(item["source_trace"], task_id=task_id)

    task_source = item.get("task_source")
    if not isinstance(task_source, str):
        raise ValueError(f"manifest entry #{index} must define task_source")
    resolved_task_source = (manifest_path.parent / task_source).resolve()
    if resolved_task_source != EXPECTED_TASK_SOURCE:
        raise ValueError(
            f"manifest entry #{index} task_source {resolved_task_source} does not "
            f"match {EXPECTED_TASK_SOURCE}"
        )


def _validate_sim_trace(
    path: Path,
    *,
    source: ExportTrace,
    manifest_source_refs: tuple[str, ...],
) -> None:
    metadata = _read_trace_metadata(path)
    expected = {
        "type": "trace_metadata",
        "trace_format_version": CURRENT_TRACE_FORMAT_VERSION,
        "scaffold": EXPECTED_SCAFFOLD,
        "mode": EXPECTED_SIM_MODE,
        "simulate_mode": EXPECTED_SIMULATE_MODE,
        "source_trace_count": EXPECTED_TRACE_COUNT,
        "trace_manifest": EXPECTED_SIM_TRACE_MANIFEST,
        "replay_target": EXPECTED_SIM_REPLAY_TARGET,
        "instance_id": source.task_id,
    }
    _validate_expected_metadata(path, metadata, expected)
    source_trace = _normalize_source_trace_reference(metadata.get("source_trace"))
    if source_trace != source.source_ref:
        raise ValueError(
            f"{path} does not match export preset: source_trace={source_trace!r} "
            f"expected {source.source_ref!r}"
        )

    source_traces = metadata.get("source_traces")
    if not isinstance(source_traces, list) or len(source_traces) != EXPECTED_TRACE_COUNT:
        raise ValueError(
            f"{path} does not match export preset: source_traces has "
            f"{len(source_traces) if isinstance(source_traces, list) else 'non-list'} "
            f"entries expected {EXPECTED_TRACE_COUNT}"
        )
    normalized_source_traces = tuple(
        _normalize_source_trace_reference(source_trace)
        for source_trace in source_traces
    )
    if normalized_source_traces != manifest_source_refs:
        raise ValueError(f"{path} source_traces do not match curated manifest")


def _read_trace_metadata(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        metadata = json.loads(next(line for line in handle if line.strip()))
    if not isinstance(metadata, dict):
        raise ValueError(f"{path} first JSONL record must be an object")
    return metadata


def _validate_expected_metadata(
    path: Path,
    metadata: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        rendered = ", ".join(
            f"{key}={actual!r} expected {expected!r}"
            for key, (actual, expected) in mismatches.items()
        )
        raise ValueError(f"{path} does not match export preset: {rendered}")


def _validate_source_trace_reference(value: Any, *, task_id: str | None = None) -> None:
    normalized = _normalize_source_trace_reference(value)
    if task_id is not None and f"/{task_id}/attempt_1/trace.jsonl" not in normalized:
        raise ValueError(
            f"source trace reference {value!r} does not point to task {task_id!r}"
        )


def _normalize_source_trace_reference(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"source trace reference must be a string, got {type(value).__name__}")
    normalized = "/" + value.replace("\\", "/").lstrip("/")
    if "/claude-code-import/" in normalized:
        raise ValueError(f"source trace reference points to Claude Code import: {value}")
    if EXPECTED_SOURCE_TRACE_REF_FRAGMENT not in normalized:
        raise ValueError(
            f"source trace reference {value!r} does not contain "
            f"{EXPECTED_SOURCE_TRACE_REF_FRAGMENT}"
        )
    return normalized


def _source_ref_from_raw_metadata(metadata: dict[str, Any], *, trace_path: Path) -> str:
    task_id = trace_path.parent.parent.name
    runtime_proof = metadata.get("runtime_proof")
    sys_path = runtime_proof.get("sys_path") if isinstance(runtime_proof, dict) else None
    if isinstance(sys_path, list):
        for entry in sys_path:
            if not isinstance(entry, str):
                continue
            normalized = "/" + entry.replace("\\", "/").lstrip("/")
            task_attempt = f"/{task_id}/attempt_1/"
            if EXPECTED_SOURCE_TRACE_REF_FRAGMENT not in normalized or task_attempt not in normalized:
                continue
            prefix = normalized.split(task_attempt, 1)[0] + task_attempt
            return _normalize_source_trace_reference(prefix + "trace.jsonl")
    return _normalize_source_trace_reference(str(trace_path))


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _build_export_groups(
    manifest_traces: tuple[ExportTrace, ...],
    *,
    group: str,
) -> list[ExportGroup]:
    requested = EXPORT_GROUPS if group == GROUP_ALL else (group,)
    groups: list[ExportGroup] = []
    for name in requested:
        if name == GROUP_RAW:
            groups.append(
                ExportGroup(
                    name=GROUP_RAW,
                    title="Raw GLM OpenClaw 100-iters SWE-rebench 19",
                    output_name="raw-glm-openclaw-100-19.html",
                    traces=manifest_traces,
                    time_mode="sync",
                )
            )
        elif name == GROUP_CLOSED_LOOP:
            groups.append(
                ExportGroup(
                    name=GROUP_CLOSED_LOOP,
                    title="Closed Loop GLM OpenClaw 100-iters SWE-rebench 19",
                    output_name="sim-closed-loop-glm-openclaw-100-19.html",
                    traces=_sim_group_traces(GROUP_CLOSED_LOOP, manifest_traces),
                    time_mode="sync",
                )
            )
        elif name in POISSON_GROUPS:
            groups.append(
                ExportGroup(
                    name=name,
                    title=(
                        "Poisson "
                        f"{_poisson_rate_label(name)} GLM OpenClaw 100-iters "
                        "SWE-rebench 19"
                    ),
                    output_name=f"sim-{_slugify(name)}-glm-openclaw-100-19.html",
                    traces=_sim_group_traces(name, manifest_traces),
                    time_mode="abs",
                )
            )
        else:
            raise ValueError(f"unknown export group: {name}")
    return groups


def _sim_group_traces(
    group: str,
    manifest_traces: Iterable[ExportTrace],
) -> tuple[ExportTrace, ...]:
    manifest_trace_tuple = tuple(manifest_traces)
    manifest_source_refs = tuple(trace.source_ref for trace in manifest_trace_tuple)
    if len(set(manifest_source_refs)) != len(manifest_source_refs):
        raise ValueError("manifest source trace references must be unique")
    traces: list[ExportTrace] = []
    for source in manifest_trace_tuple:
        path = SIM_SWEEP_ROOT / group / source.task_id / "attempt_1" / "trace.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing simulate trace: {path}")
        _validate_sim_trace(
            path,
            source=source,
            manifest_source_refs=manifest_source_refs,
        )
        traces.append(ExportTrace(task_id=source.task_id, path=path, source_ref=source.source_ref))
    return tuple(traces)


def _build_group_payload(
    group: ExportGroup,
    *,
    max_resource_samples_per_trace: int | None,
) -> dict[str, Any]:
    trace_payload = build_gantt_payload_multi(
        [(trace.task_id, TraceData.load(trace.path)) for trace in group.traces]
    )
    if max_resource_samples_per_trace is not None:
        _downsample_resource_timelines(
            trace_payload,
            max_samples=max_resource_samples_per_trace,
        )
    trace_ids = [trace["id"] for trace in trace_payload["traces"]]
    return {
        "mode": "snapshot",
        "payload": {
            **trace_payload,
            "errors": trace_payload.get("errors", []),
        },
        "trace_ids": trace_ids,
        "visible_trace_ids": trace_ids,
        "display": {
            "clockMode": "real",
            "resourceMetric": "cpu",
            "resourceMetricSecondary": "memory",
            "showResourceChart": True,
            "themeMode": "light",
            "timeMode": group.time_mode,
            "viewMode": "layered",
            "zoom": 1,
        },
    }


def _downsample_resource_timelines(payload: dict[str, Any], *, max_samples: int) -> None:
    for trace in payload["traces"]:
        timeline = trace.get("resource_timeline")
        if not timeline or len(timeline) <= max_samples:
            continue
        trace["resource_timeline"] = _downsample_sequence(timeline, max_samples)


def _downsample_sequence(items: list[Any], max_items: int) -> list[Any]:
    if len(items) <= max_items:
        return items
    last = len(items) - 1
    indexes = sorted({round(i * last / (max_items - 1)) for i in range(max_items)})
    return [items[index] for index in indexes]


def _load_frontend_bundle(dist_path: Path) -> dict[str, str]:
    index_path = dist_path / "index.html"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"missing Gantt frontend build at {index_path}; run make gantt-viewer-build"
        )
    html_text = index_path.read_text(encoding="utf-8")
    script_src = _extract_asset_path(html_text, r'<script[^>]+src="([^"]+)"')
    css_href = _extract_asset_path(html_text, r'<link[^>]+href="([^"]+\.css)"')
    script_path = dist_path / script_src.lstrip("/")
    css_path = dist_path / css_href.lstrip("/")
    if not script_path.is_file():
        raise FileNotFoundError(f"missing frontend script asset: {script_path}")
    if not css_path.is_file():
        raise FileNotFoundError(f"missing frontend CSS asset: {css_path}")
    return {
        "script": script_path.read_text(encoding="utf-8"),
        "style": css_path.read_text(encoding="utf-8"),
    }


def _extract_asset_path(html_text: str, pattern: str) -> str:
    match = re.search(pattern, html_text)
    if not match:
        raise ValueError("could not locate frontend asset in dist/index.html")
    return match.group(1)


def _render_html(
    *,
    title: str,
    snapshot: dict[str, Any],
    frontend: dict[str, str],
) -> str:
    snapshot_json = json.dumps(snapshot, ensure_ascii=False).replace("</", "<\\/")
    style = _escape_raw_text_closing_tag(frontend["style"], "style")
    script = _escape_raw_text_closing_tag(frontend["script"], "script")
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "  <head>",
            '    <meta charset="UTF-8" />',
            '    <meta name="viewport" content="width=device-width, initial-scale=1.0" />',
            f"    <title>{html.escape(title)}</title>",
            f"    <style>{style}</style>",
            "  </head>",
            "  <body>",
            '    <div id="root"></div>',
            (
                '    <script id="gantt-viewer-snapshot-bootstrap" '
                f'type="application/json">{snapshot_json}</script>'
            ),
            f"    <script type=\"module\">{script}</script>",
            "  </body>",
            "</html>",
            "",
        ]
    )


def _escape_raw_text_closing_tag(text: str, tag_name: str) -> str:
    return re.sub(rf"</(?={re.escape(tag_name)}\b)", r"<\\/", text, flags=re.IGNORECASE)


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _poisson_rate_label(group: str) -> str:
    return group.removeprefix("poisson_").removesuffix("_per_s") + "/s"
