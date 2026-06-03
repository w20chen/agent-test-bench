#!/usr/bin/env python3
"""Helpers for figure scripts that use REAL-mode trace timing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (SCRIPT_DIR, REPO_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from demo.gantt_viewer.backend.payload import build_gantt_payload  # noqa: E402
from trace_collect.trace_inspector import TraceData  # noqa: E402


@dataclass(frozen=True)
class CohortSpec:
    label: str
    root: Path
    model_substring: str | None = None


@dataclass(frozen=True)
class RealToolSpan:
    tool_name: str
    midpoint_frac: float
    duration_s: float


@dataclass(frozen=True)
class RealTraceMetrics:
    task: str
    trace_path: Path
    model: str | None
    total_time_s: float
    llm_time_s: float
    tool_time_s: float
    other_time_s: float
    tool_ratio: float
    tool_spans: tuple[RealToolSpan, ...]


def parse_cohort(raw: str) -> CohortSpec:
    if "=" not in raw:
        raise ValueError(f"Invalid cohort spec {raw!r}; expected LABEL=PATH[::MODEL]")
    label, remainder = raw.split("=", 1)
    if "::" in remainder:
        path_str, model_substring = remainder.split("::", 1)
    else:
        path_str, model_substring = remainder, None
    root = Path(path_str).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Cohort path does not exist: {root}")
    return CohortSpec(label=label, root=root, model_substring=model_substring or None)


def load_real_trace_metrics(spec: CohortSpec) -> list[RealTraceMetrics]:
    trace_paths = [
        path
        for path in sorted(spec.root.rglob("trace.jsonl"))
        if _is_attempt_trace(path)
    ]
    metrics: list[RealTraceMetrics] = []
    for path in trace_paths:
        trace_data = TraceData.load(path)
        model = trace_data.metadata.get("model")
        if spec.model_substring and spec.model_substring not in str(model):
            continue
        metric = _build_real_trace_metrics(trace_data, path)
        if metric is not None:
            metrics.append(metric)
    if not metrics:
        raise ValueError(
            f"No matching trace.jsonl files found for cohort {spec.label!r}"
        )
    return metrics


def _is_attempt_trace(path: Path) -> bool:
    if path.name != "trace.jsonl":
        return False
    if any(part == "_task_container_runtime" for part in path.parts):
        return False
    return path.parent.name.startswith("attempt_")


def _build_real_trace_metrics(
    trace_data: TraceData,
    trace_path: Path,
) -> RealTraceMetrics | None:
    model = trace_data.metadata.get("model")
    payload = build_gantt_payload(
        trace_data, label=trace_data.metadata.get("instance_id")
    )
    lanes = payload.get("lanes") or []
    spans = [span for lane in lanes for span in lane.get("spans") or []]
    markers = [marker for lane in lanes for marker in lane.get("markers") or []]
    starts = [_real_span_start(span) for span in spans] + [
        _real_marker_time(marker) for marker in markers
    ]
    ends = [_real_span_end(span) for span in spans] + [
        _real_marker_time(marker) for marker in markers
    ]
    starts = [value for value in starts if value is not None]
    ends = [value for value in ends if value is not None]
    if not starts or not ends:
        return None

    trace_start = min(starts)
    trace_end = max(ends)
    total_time_s = max(0.0, trace_end - trace_start)

    llm_time_s = 0.0
    tool_time_s = 0.0
    tool_spans: list[RealToolSpan] = []
    for span in spans:
        duration_s = max(
            0.0, (_real_span_end(span) or 0.0) - (_real_span_start(span) or 0.0)
        )
        span_type = str(span.get("type") or "")
        if span_type == "llm":
            llm_time_s += duration_s
        elif span_type == "tool":
            tool_time_s += duration_s
            midpoint = (
                (_real_span_start(span) + _real_span_end(span)) / 2.0
                if _real_span_start(span) is not None
                and _real_span_end(span) is not None
                else trace_start
            )
            midpoint_frac = (
                0.0
                if total_time_s <= 0
                else min(max((midpoint - trace_start) / total_time_s, 0.0), 0.999999)
            )
            detail = span.get("detail") or {}
            tool_spans.append(
                RealToolSpan(
                    tool_name=str(detail.get("tool_name") or ""),
                    midpoint_frac=midpoint_frac,
                    duration_s=duration_s,
                )
            )

    active_time_s = llm_time_s + tool_time_s
    other_time_s = max(0.0, total_time_s - active_time_s)
    tool_ratio = tool_time_s / active_time_s if active_time_s > 0 else 0.0
    task = str(trace_data.metadata.get("instance_id") or trace_path.parent.parent.name)
    return RealTraceMetrics(
        task=task,
        trace_path=trace_path,
        model=None if model is None else str(model),
        total_time_s=total_time_s,
        llm_time_s=llm_time_s,
        tool_time_s=tool_time_s,
        other_time_s=other_time_s,
        tool_ratio=tool_ratio,
        tool_spans=tuple(tool_spans),
    )


def _real_span_start(span: dict[str, Any]) -> float | None:
    value = span.get("start_real_abs")
    if isinstance(value, (int, float)):
        return float(value)
    value = span.get("start_abs")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _real_span_end(span: dict[str, Any]) -> float | None:
    value = span.get("end_real_abs")
    if isinstance(value, (int, float)):
        return float(value)
    value = span.get("end_abs")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _real_marker_time(marker: dict[str, Any]) -> float | None:
    value = marker.get("t_real_abs")
    if isinstance(value, (int, float)):
        return float(value)
    value = marker.get("t_abs")
    if isinstance(value, (int, float)):
        return float(value)
    return None


__all__ = [
    "CohortSpec",
    "RealToolSpan",
    "RealTraceMetrics",
    "load_real_trace_metrics",
    "parse_cohort",
]
