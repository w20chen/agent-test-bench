"""Build Gantt payloads from canonical trace data.

The registries live in this module and are shipped inside each payload so
renderers can extend span and marker types without hard-coded frontend logic.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from trace_collect.trace_inspector import TraceData

logger = logging.getLogger(__name__)

_MARKER_CATEGORIES = frozenset({"SCHEDULING", "SESSION", "CONTEXT", "MCP"})

_LLM_CONTENT_MAX = 1000

_TOOL_ARGS_MAX = 200

_TOOL_PRIMARY_FIELDS: tuple[str, ...] = (
    "path",
    "file_path",
    "filepath",
    "command",
    "cmd",
    "pattern",
    "query",
    "url",
)

ACTION_TYPE_MAP: dict[str, str] = {
    "llm_call": "llm",
    "tool_exec": "tool",
    "mcp_call": "mcp",
}

DEFAULT_SPAN_REGISTRY: dict[str, dict[str, Any]] = {
    "llm": {"color": "#00E5FF", "label": "LLM Call", "order": 0},
    "tool": {"color": "#FF6D00", "label": "Tool Exec", "order": 1},
    "scheduling": {"color": "#76FF03", "label": "Scheduling", "order": 2},
    "mcp": {"color": "#AB47BC", "label": "MCP Call", "order": 3},
}

DEFAULT_MARKER_REGISTRY: dict[str, dict[str, str]] = {
    "message_dispatch": {
        "symbol": "diamond",
        "color": "#76FF03",
        "label": "Message Dispatch",
    },
    "session_lock_acquire": {
        "symbol": "diamond",
        "color": "#76FF03",
        "label": "Session Lock Acquire",
    },
    "session_load": {"symbol": "dot", "color": "#76FF03", "label": "Session Load"},
    "message_list_build": {
        "symbol": "dot",
        "color": "#4FC3F7",
        "label": "Message List Build",
    },
    "session_turn_save": {
        "symbol": "dot",
        "color": "#76FF03",
        "label": "Session Turn Save",
    },
    "task_complete": {"symbol": "flag", "color": "#FF6D00", "label": "Task Complete"},
    "llm_error": {"symbol": "cross", "color": "#FF1744", "label": "Llm Error"},
    "max_iterations": {
        "symbol": "cross",
        "color": "#FF1744",
        "label": "Max Iterations",
    },
    "_default": {"symbol": "dot", "color": "#6b7280", "label": "Default"},
}


def _coerce_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _raw_response_message(raw_response: dict[str, Any]) -> dict[str, Any]:
    choices = raw_response.get("choices") or []
    if choices:
        return choices[0].get("message") or {}
    return raw_response.get("message") or {}


def _raw_response_text(raw_response: dict[str, Any]) -> str:
    message = _raw_response_message(raw_response)
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    text_parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(part for part in text_parts if part)


def _raw_response_tool_calls(raw_response: dict[str, Any]) -> list[dict[str, Any]]:
    message = _raw_response_message(raw_response)
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        return [tc for tc in tool_calls if isinstance(tc, dict)]

    content = message.get("content")
    if not isinstance(content, list):
        return []
    calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        calls.append(
            {
                "id": block.get("id"),
                "name": block.get("name"),
                "arguments": block.get("input") or {},
            }
        )
    return calls


def _extract_detail(event: dict[str, Any]) -> dict[str, Any]:
    """Extract lightweight event detail for tooltips."""
    data = dict(event.get("data") or {})
    raw_resp = data.pop("raw_response", None)
    if raw_resp and isinstance(raw_resp, dict):
        content = _raw_response_text(raw_resp)
        if content:
            data["llm_content"] = content[:200] + ("..." if len(content) > 200 else "")

    data.pop("messages_in", None)
    data.pop("tool_result", None)

    for key in ("args_preview", "result_preview", "tool_args"):
        if key in data and isinstance(data[key], str) and len(data[key]) > 100:
            data[key] = data[key][:100] + "..."
    return data


_MEM_RE = re.compile(r"^([\d.]+)\s*(B|[kKMGT]i?B)", re.ASCII)

_MEM_MULTIPLIERS: dict[str, float] = {
    "B": 1 / 1_000_000,
    "kB": 1 / 1_000,
    "KB": 1 / 1_000,
    "KiB": 1 / 1024,
    "MB": 1.0,
    "MiB": 1.0,
    "GB": 1_000.0,
    "GiB": 1024.0,
    "TB": 1_000_000.0,
    "TiB": 1024 * 1024.0,
}


def _parse_mem_usage_mb(raw: str) -> float:
    """Parse '40.42MB / 52.93GB' → 40.42 (left side, in MB)."""
    left = raw.split("/")[0].strip()
    m = _MEM_RE.match(left)
    if not m:
        return 0.0
    return float(m.group(1)) * _MEM_MULTIPLIERS.get(m.group(2), 1.0)


def _parse_percent(raw: str) -> float:
    """Parse '1.20%' → 1.20."""
    try:
        return float(raw.rstrip("% "))
    except (ValueError, TypeError):
        return 0.0


def _load_resource_timeline(
    trace_dir: Path, t0: float,
) -> list[dict[str, Any]] | None:
    """Load resources.json and build timeline aligned to trace t0."""
    resources_path = trace_dir / "resources.json"
    if not resources_path.exists():
        return None
    try:
        with resources_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load %s: %s", resources_path, exc)
        return None

    samples = data.get("samples")
    if not samples:
        return None

    timeline: list[dict[str, Any]] = []
    for s in samples:
        epoch = s.get("epoch")
        if epoch is None:
            continue
        epoch = float(epoch)
        entry: dict[str, Any] = {
            "t": epoch - t0,
            "t_abs": epoch,
            "cpu_percent": _parse_percent(s["cpu_percent"]) if isinstance(s.get("cpu_percent"), str) else float(s.get("cpu_percent") or 0),
            "memory_mb": _parse_mem_usage_mb(s["mem_usage"]) if isinstance(s.get("mem_usage"), str) else float(s.get("memory_mb") or 0),
        }
        for raw_field in (
            "memory_total_mb_s",
            "memory_read_mb_s",
            "memory_write_mb_s",
        ):
            val = s.get(raw_field)
            if val is not None:
                entry[raw_field] = float(val)
        # Per-sample records store raw bytes; convert to MB.
        # Disk uses binary (1 MiB = 1048576), network uses decimal (1 MB = 1e6).
        for raw_field, out_field, divisor in (
            ("disk_read_bytes", "disk_read_mb", 1024 * 1024),
            ("disk_write_bytes", "disk_write_mb", 1024 * 1024),
            ("net_rx_bytes", "net_rx_mb", 1_000_000),
            ("net_tx_bytes", "net_tx_mb", 1_000_000),
        ):
            val = s.get(raw_field)
            if val is not None:
                entry[out_field] = float(val) / divisor
        if "context_switches" in s and s["context_switches"] is not None:
            entry["context_switches"] = int(s["context_switches"])
        timeline.append(entry)

    return timeline if timeline else None


def build_gantt_payload(
    data: TraceData,
    *,
    label: str | None = None,
) -> dict[str, Any]:
    """Build the single-trace payload consumed by the Gantt viewer."""
    meta = data.metadata
    scaffold = meta.get("scaffold", "unknown")
    instance_id = meta.get("instance_id", "")
    trace_label = label or f"{scaffold}/{instance_id}" or str(data.path.stem)

    t0 = _compute_t0(data)
    agents = list(data.agents)
    if not agents:
        agents = ["default"]

    lanes: list[dict[str, Any]] = []
    for agent_id in agents:
        agent_events = [e for e in data.events if e.get("agent_id") == agent_id]
        agent_actions = [a for a in data.actions if a.get("agent_id") == agent_id]

        spans, markers = _build_spans_and_markers(agent_actions, agent_events, t0)

        lanes.append(
            {
                "agent_id": agent_id,
                "spans": spans,
                "markers": markers,
            }
        )

    _apply_real_timeline_to_lanes(lanes, t0)

    distinct_iters = {a.get("iteration", 0) for a in data.actions}
    resource_timeline = _load_resource_timeline(data.path.parent, t0)
    if resource_timeline:
        _apply_real_timeline_to_resources(resource_timeline, lanes)

    return {
        "id": trace_label,
        "label": trace_label,
        "metadata": {
            "scaffold": scaffold,
            "model": meta.get("model"),
            "instance_id": instance_id,
            "mode": meta.get("mode"),
            "max_iterations": meta.get("max_iterations") or meta.get("max_steps"),
            "n_actions": len(data.actions),
            "n_iterations": len(distinct_iters),
            "n_events": len(data.events),
            "elapsed_s": _get_elapsed(data),
        },
        "t0": t0,
        "lanes": lanes,
        "resource_timeline": resource_timeline,
    }


def build_gantt_payload_multi(
    traces: list[tuple[str, TraceData]],
    *,
    span_registry: dict[str, dict[str, Any]] | None = None,
    marker_registry: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build the multi-trace payload, including the active registries."""
    return {
        "registries": {
            "spans": span_registry or DEFAULT_SPAN_REGISTRY,
            "markers": marker_registry or DEFAULT_MARKER_REGISTRY,
        },
        "traces": [build_gantt_payload(td, label=lbl) for lbl, td in traces],
    }


def _compute_t0(data: TraceData) -> float:
    action_t0 = float("inf")
    for step in data.actions:
        ts = step.get("ts_start", 0)
        if ts and ts < action_t0:
            action_t0 = ts
    if action_t0 != float("inf"):
        return action_t0

    event_t0 = float("inf")
    for ev in data.events:
        ts = ev.get("ts", 0)
        if ts and ts < event_t0:
            event_t0 = ts
    return event_t0 if event_t0 != float("inf") else 0.0


def _get_elapsed(data: TraceData) -> float | None:
    for s in data.summaries:
        if "elapsed_s" in s:
            return s["elapsed_s"]
    return None


def _build_spans_and_markers(
    actions: list[dict[str, Any]],
    events: list[dict[str, Any]],
    t0: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build spans from actions and markers from framework events."""
    spans: list[dict[str, Any]] = []
    markers: list[dict[str, Any]] = []

    for act in actions:
        action_type = act.get("action_type")
        if action_type not in ACTION_TYPE_MAP:
            continue
        span_type = ACTION_TYPE_MAP[action_type]
        spans.append(
            {
                "type": span_type,
                "start": act.get("ts_start", 0) - t0,
                "end": act.get("ts_end", 0) - t0,
                "start_abs": act.get("ts_start", 0),
                "end_abs": act.get("ts_end", 0),
                "start_real": act.get("ts_start", 0) - t0,
                "end_real": act.get("ts_end", 0) - t0,
                "start_real_abs": act.get("ts_start", 0),
                "end_real_abs": act.get("ts_end", 0),
                "iteration": act.get("iteration", 0),
                "detail": _extract_detail_from_action(act),
            }
        )

    if spans and events:
        sorted_spans = sorted(spans, key=lambda s: s["start_abs"])
        for i in range(len(sorted_spans) - 1):
            gap_start = sorted_spans[i]["end_abs"]
            gap_end = sorted_spans[i + 1]["start_abs"]
            if gap_end <= gap_start:
                continue
            events_in_gap = [
                e
                for e in events
                if e.get("category") in _MARKER_CATEGORIES
                and gap_start < (e.get("ts") or 0) < gap_end
            ]
            if not events_in_gap:
                continue
            spans.append(
                {
                    "type": "scheduling",
                    "start": gap_start - t0,
                    "end": gap_end - t0,
                    "start_abs": gap_start,
                    "end_abs": gap_end,
                    "start_real": gap_start - t0,
                    "end_real": gap_end - t0,
                    "start_real_abs": gap_start,
                    "end_real_abs": gap_end,
                    "iteration": sorted_spans[i + 1].get("iteration", 0),
                    "detail": {
                        "gap_ms": round((gap_end - gap_start) * 1000, 1),
                        "events": [e.get("event", "?") for e in events_in_gap],
                    },
                }
            )

    for ev in events:
        category = ev.get("category", "")
        if category in _MARKER_CATEGORIES:
            markers.append(
                {
                    "type": category.lower(),
                    "event": ev.get("event", "unknown"),
                    "t": ev.get("ts", 0) - t0,
                    "t_abs": ev.get("ts", 0),
                    "t_real": ev.get("ts", 0) - t0,
                    "t_real_abs": ev.get("ts", 0),
                    "iteration": ev.get("iteration", 0),
                    "detail": _extract_detail_from_event(ev),
                }
            )

    spans.sort(key=lambda s: s["start"])
    markers.sort(key=lambda m: m["t"])
    return spans, markers


def _apply_real_timeline_to_lanes(
    lanes: list[dict[str, Any]],
    trace_t0: float,
) -> None:
    spans = [span for lane in lanes for span in lane.get("spans") or []]
    markers = [marker for lane in lanes for marker in lane.get("markers") or []]
    _apply_real_timeline(spans, markers, trace_t0)


def _apply_real_timeline_to_resources(
    timeline: list[dict[str, Any]],
    lanes: list[dict[str, Any]],
) -> None:
    """Map resource sample timestamps to real (gap-compressed) timeline.

    Uses the same real_t0_abs origin and span sort order as
    _apply_real_timeline so the resource chart aligns with lane content.
    """
    all_spans = sorted(
        (span for lane in lanes for span in lane.get("spans") or []),
        key=lambda s: (
            float(s.get("start_abs", 0)),
            float(s.get("end_abs", 0)),
            str(s.get("type", "")),
        ),
    )
    if not all_spans:
        return

    # Keep resources anchored to actual workload spans. Markers before the
    # first action are allowed to be negative but must not shift the workload.
    real_t0_abs = min(float(s.get("start_real_abs", float("inf"))) for s in all_spans)
    if real_t0_abs == float("inf"):
        return

    # Reuse the same mapping function as markers for correctness with
    # overlapping multi-lane spans.
    for sample in timeline:
        t_real_abs = _map_time_to_real_abs(float(sample["t_abs"]), all_spans)
        sample["t_real_abs"] = t_real_abs
        sample["t_real"] = t_real_abs - real_t0_abs


def _apply_real_timeline(
    spans: list[dict[str, Any]],
    markers: list[dict[str, Any]],
    trace_t0: float,
) -> None:
    if not spans and not markers:
        return

    sorted_spans = sorted(
        spans,
        key=lambda span: (
            float(span.get("start_abs", 0.0)),
            float(span.get("end_abs", 0.0)),
            str(span.get("type", "")),
        ),
    )
    removed_gap_s = 0.0
    prev_real_end_abs: float | None = None

    for span in sorted_spans:
        wall_start_abs = float(span.get("start_abs", 0.0))
        natural_start_abs = wall_start_abs - removed_gap_s
        if prev_real_end_abs is not None and natural_start_abs > prev_real_end_abs:
            removed_gap_s += natural_start_abs - prev_real_end_abs
            natural_start_abs = prev_real_end_abs

        real_duration_s = _real_span_duration_s(span)
        real_end_abs = natural_start_abs + real_duration_s
        span["_real_removed_gap_s"] = removed_gap_s
        span["start_real_abs"] = natural_start_abs
        span["end_real_abs"] = real_end_abs
        prev_real_end_abs = max(prev_real_end_abs or real_end_abs, real_end_abs)

    for marker in markers:
        marker_abs = float(marker.get("t_abs", 0.0))
        marker["t_real_abs"] = _map_time_to_real_abs(marker_abs, sorted_spans)

    if spans:
        real_t0_abs = min(
            float(span.get("start_real_abs", trace_t0)) for span in spans
        )
    else:
        real_t0_abs = min(
            [float(marker.get("t_real_abs", trace_t0)) for marker in markers]
            + [trace_t0]
        )

    for span in spans:
        span["start_real"] = (
            float(span.get("start_real_abs", real_t0_abs)) - real_t0_abs
        )
        span["end_real"] = float(span.get("end_real_abs", real_t0_abs)) - real_t0_abs
        span.pop("_real_removed_gap_s", None)

    for marker in markers:
        marker["t_real"] = float(marker.get("t_real_abs", real_t0_abs)) - real_t0_abs


def _real_span_duration_s(span: dict[str, Any]) -> float:
    wall_duration_s = max(
        0.0,
        float(span.get("end_abs", 0.0)) - float(span.get("start_abs", 0.0)),
    )
    if span.get("type") != "llm":
        return wall_duration_s

    detail = span.get("detail") or {}
    for key in ("llm_call_time_ms", "openrouter_generation_time_ms"):
        value = _coerce_optional_float(detail.get(key))
        if value is not None and value >= 0.0:
            return value / 1000.0

    timing_source = detail.get("llm_timing_source")
    llm_latency_ms = _coerce_optional_float(detail.get("llm_latency_ms"))
    if (
        llm_latency_ms is not None
        and llm_latency_ms >= 0.0
        and isinstance(timing_source, str)
        and timing_source != "wall_clock_ms"
    ):
        return llm_latency_ms / 1000.0
    return wall_duration_s


def _map_time_to_real_abs(timestamp_abs: float, spans: list[dict[str, Any]]) -> float:
    if not spans:
        return timestamp_abs

    first_span = spans[0]
    first_shift_s = float(first_span.get("start_abs", 0.0)) - float(
        first_span.get("start_real_abs", 0.0)
    )
    if timestamp_abs <= float(first_span.get("start_abs", 0.0)):
        return timestamp_abs - first_shift_s

    for index, span in enumerate(spans):
        wall_start_abs = float(span.get("start_abs", 0.0))
        wall_end_abs = float(span.get("end_abs", wall_start_abs))
        real_start_abs = float(span.get("start_real_abs", wall_start_abs))
        real_end_abs = float(span.get("end_real_abs", real_start_abs))

        if wall_start_abs <= timestamp_abs <= wall_end_abs:
            wall_duration_s = wall_end_abs - wall_start_abs
            if wall_duration_s <= 0.0:
                return real_start_abs
            fraction = (timestamp_abs - wall_start_abs) / wall_duration_s
            return real_start_abs + fraction * (real_end_abs - real_start_abs)

        next_start_abs = (
            float(spans[index + 1].get("start_abs", 0.0))
            if index + 1 < len(spans)
            else None
        )
        if next_start_abs is not None and timestamp_abs < next_start_abs:
            return real_end_abs

    last_span = spans[-1]
    last_shift_s = float(last_span.get("end_abs", 0.0)) - float(
        last_span.get("end_real_abs", 0.0)
    )
    return timestamp_abs - last_shift_s


def _summarize_tool_call(tc: dict[str, Any]) -> str | None:
    """Summarize one raw tool call for tooltip display."""
    if not isinstance(tc, dict):
        return None
    fn = tc.get("function") or {}
    name = fn.get("name") or tc.get("name") or "?"

    raw_args = fn.get("arguments") or tc.get("arguments") or ""
    parsed: dict[str, Any] | None = None
    if isinstance(raw_args, dict):
        parsed = raw_args
    elif isinstance(raw_args, str):
        try:
            candidate = json.loads(raw_args)
            if isinstance(candidate, dict):
                parsed = candidate
        except (json.JSONDecodeError, ValueError):
            parsed = None

    if parsed is not None:
        for field in _TOOL_PRIMARY_FIELDS:
            if field in parsed and parsed[field] is not None:
                value = str(parsed[field])
                if len(value) > _TOOL_ARGS_MAX:
                    value = value[:_TOOL_ARGS_MAX] + "..."
                return f'{name}({field}="{value}")'

    if isinstance(raw_args, (dict, list)):
        args_str = json.dumps(raw_args, ensure_ascii=False)
    else:
        args_str = str(raw_args)
    preview = args_str[:_TOOL_ARGS_MAX]
    if len(args_str) > _TOOL_ARGS_MAX:
        preview += "..."
    return f"{name}({preview})"


def _extract_detail_from_action(act: dict[str, Any]) -> dict[str, Any]:
    """Extract action detail while preserving silent tool-call decisions."""
    data = dict(act.get("data") or {})
    raw_resp = data.pop("raw_response", None)
    if raw_resp and isinstance(raw_resp, dict):
        content = _raw_response_text(raw_resp)
        if content:
            if len(content) > _LLM_CONTENT_MAX:
                data["llm_content"] = content[:_LLM_CONTENT_MAX] + "..."
            else:
                data["llm_content"] = content
        tool_calls = _raw_response_tool_calls(raw_resp)
        if tool_calls:
            summaries = [
                s
                for s in (_summarize_tool_call(tc) for tc in tool_calls)
                if s is not None
            ]
            if summaries:
                data["tool_calls_requested"] = summaries

    data.pop("messages_in", None)
    return data


def _extract_detail_from_event(ev: dict[str, Any]) -> dict[str, Any]:
    data = dict(ev.get("data") or {})
    data.pop("messages_in", None)

    raw_resp = data.pop("raw_response", None)
    if raw_resp and isinstance(raw_resp, dict):
        content = _raw_response_text(raw_resp)
        if content:
            data["llm_content"] = content[:200] + ("..." if len(content) > 200 else "")

    for key in ("args_preview", "result_preview", "tool_args", "tool_result"):
        if key in data and isinstance(data[key], str) and len(data[key]) > 100:
            data[key] = data[key][:100] + "..."
    return data
