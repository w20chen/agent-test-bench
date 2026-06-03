"""Regression tests for demo.gantt_viewer.backend.payload.

These tests pin span construction, scheduling-gap rendering, registry export,
and tooltip detail extraction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from demo.gantt_viewer.backend.payload import (
    ACTION_TYPE_MAP,
    DEFAULT_MARKER_REGISTRY,
    DEFAULT_SPAN_REGISTRY,
    build_gantt_payload,
    build_gantt_payload_multi,
)
from trace_collect.trace_inspector import TraceData

# ── Fixture builders ───────────────────────────────────────────────


def _llm_action(
    iteration: int,
    ts_start: float,
    ts_end: float,
    *,
    agent_id: str = "a1",
) -> dict[str, Any]:
    return {
        "type": "action",
        "action_type": "llm_call",
        "action_id": f"llm_{iteration}",
        "agent_id": agent_id,
        "iteration": iteration,
        "ts_start": ts_start,
        "ts_end": ts_end,
        "data": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "llm_latency_ms": (ts_end - ts_start) * 1000,
        },
    }


def _tool_action(
    iteration: int,
    ts_start: float,
    ts_end: float,
    tool_name: str = "bash",
    *,
    agent_id: str = "a1",
) -> dict[str, Any]:
    return {
        "type": "action",
        "action_type": "tool_exec",
        "action_id": f"tool_{iteration}_{tool_name}",
        "agent_id": agent_id,
        "iteration": iteration,
        "ts_start": ts_start,
        "ts_end": ts_end,
        "data": {
            "tool_name": tool_name,
            "duration_ms": (ts_end - ts_start) * 1000,
            "success": True,
        },
    }


def _event(
    category: str,
    event_name: str,
    iteration: int,
    ts: float,
    *,
    agent_id: str = "a1",
) -> dict[str, Any]:
    return {
        "type": "event",
        "agent_id": agent_id,
        "category": category,
        "event": event_name,
        "iteration": iteration,
        "ts": ts,
        "data": {},
    }


def _write_trace(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    trace = tmp_path / "trace.jsonl"
    head = {
        "type": "trace_metadata",
        "scaffold": "synthetic",
        "model": "test-model",
        "trace_format_version": 5,
        "max_iterations": 80,
    }
    with trace.open("w") as fh:
        fh.write(json.dumps(head) + "\n")
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return trace


def _build(tmp_path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    data = TraceData.load(_write_trace(tmp_path, records))
    return build_gantt_payload(data, label="test")


def test_llm_action_becomes_span(tmp_path: Path) -> None:
    payload = _build(tmp_path, [_llm_action(0, 1000.0, 1001.0)])
    assert payload["id"] == "test"
    assert payload["label"] == "test"
    spans = payload["lanes"][0]["spans"]
    llm_spans = [s for s in spans if s["type"] == "llm"]
    assert len(llm_spans) == 1
    assert llm_spans[0]["start"] == pytest.approx(0.0)
    assert llm_spans[0]["end"] == pytest.approx(1.0)
    assert llm_spans[0]["start_real"] == pytest.approx(0.0)
    assert llm_spans[0]["end_real"] == pytest.approx(1.0)
    assert llm_spans[0]["iteration"] == 0
    assert DEFAULT_SPAN_REGISTRY["llm"]["color"] == "#00E5FF"


def test_t0_prefers_first_action_over_pre_action_event(tmp_path: Path) -> None:
    records = [
        _event("SESSION", "session_load", 0, 999.5),
        _llm_action(0, 1000.0, 1001.0),
    ]
    records[1]["data"]["llm_call_time_ms"] = 500.0
    records[1]["data"]["llm_timing_source"] = "openrouter_generation_time_ms"

    payload = _build(tmp_path, records)
    span = next(s for s in payload["lanes"][0]["spans"] if s["type"] == "llm")
    marker = payload["lanes"][0]["markers"][0]

    assert payload["t0"] == pytest.approx(1000.0)
    assert span["start"] == pytest.approx(0.0)
    assert span["start_real"] == pytest.approx(0.0)
    assert marker["t"] == pytest.approx(-0.5)
    assert marker["t_real"] == pytest.approx(-0.5)


def test_research_style_llm_only_trace_renders(tmp_path: Path) -> None:
    trace = tmp_path / "research-trace.jsonl"
    head = {
        "type": "trace_metadata",
        "scaffold": "research-agent",
        "execution_environment": "host",
        "benchmark": "deep-research-bench",
        "trace_format_version": 5,
        "max_iterations": 1,
    }
    llm = _llm_action(0, 1000.0, 1002.0, agent_id="research-1")
    llm["data"]["llm_call_time_ms"] = 1500.0
    with trace.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(head) + "\n")
        fh.write(json.dumps(llm) + "\n")
    (tmp_path / "resources.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "epoch": 1000.5,
                        "cpu_percent": "12.5%",
                        "mem_usage": "64MiB",
                        "disk_read_bytes": 1048576,
                        "context_switches": 7,
                    }
                ],
                "summary": {"sample_count": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = build_gantt_payload(TraceData.load(trace), label="research")
    spans = payload["lanes"][0]["spans"]

    assert payload["metadata"]["scaffold"] == "research-agent"
    assert [span["type"] for span in spans] == ["llm"]
    assert spans[0]["detail"]["prompt_tokens"] == 100
    assert payload["resource_timeline"][0]["cpu_percent"] == pytest.approx(12.5)
    assert payload["resource_timeline"][0]["memory_mb"] == pytest.approx(64.0)
    assert payload["resource_timeline"][0]["disk_read_mb"] == pytest.approx(1.0)
    assert payload["resource_timeline"][0]["context_switches"] == 7


def test_real_timeline_prefers_llm_call_time_when_present(tmp_path: Path) -> None:
    llm = _llm_action(0, 1000.0, 1005.0)
    llm["data"]["llm_call_time_ms"] = 1200.0
    llm["data"]["llm_timing_source"] = "openrouter_generation_time_ms"

    payload = _build(tmp_path, [llm])
    span = next(s for s in payload["lanes"][0]["spans"] if s["type"] == "llm")

    assert span["end"] - span["start"] == pytest.approx(5.0)
    assert span["end_real"] - span["start_real"] == pytest.approx(1.2)


def test_real_timeline_compacts_positive_gaps_between_spans(tmp_path: Path) -> None:
    records = [
        _llm_action(0, 1000.0, 1004.0),
        _tool_action(0, 1006.0, 1007.0, "bash"),
    ]
    records[0]["data"]["llm_call_time_ms"] = 500.0
    records[0]["data"]["llm_timing_source"] = "openrouter_generation_time_ms"

    payload = _build(tmp_path, records)
    spans = [s for s in payload["lanes"][0]["spans"] if s["type"] != "scheduling"]
    spans.sort(key=lambda span: span["start_real_abs"])

    assert spans[0]["end_real_abs"] == pytest.approx(spans[1]["start_real_abs"])
    assert spans[1]["start_abs"] - spans[0]["end_abs"] == pytest.approx(2.0)
    assert spans[1]["start_real_abs"] - spans[0]["end_real_abs"] == pytest.approx(0.0)


def test_real_timeline_shifts_markers_with_compacted_spans(tmp_path: Path) -> None:
    records = [
        _llm_action(0, 1000.0, 1004.0),
        _event("SCHEDULING", "message_dispatch", 0, 1004.5),
        _tool_action(0, 1006.0, 1007.0, "bash"),
    ]
    records[0]["data"]["llm_call_time_ms"] = 1000.0
    records[0]["data"]["llm_timing_source"] = "openrouter_generation_time_ms"

    payload = _build(tmp_path, records)
    marker = payload["lanes"][0]["markers"][0]
    llm_span = next(s for s in payload["lanes"][0]["spans"] if s["type"] == "llm")

    assert marker["t"] == pytest.approx(4.5)
    assert marker["t_real"] == pytest.approx(1.5)
    assert marker["t_real_abs"] > llm_span["end_real_abs"]


def test_real_timeline_compresses_markers_inside_llm_spans(tmp_path: Path) -> None:
    records = [
        _llm_action(0, 1000.0, 1004.0),
        _event("CONTEXT", "session_load", 0, 1002.0),
    ]
    records[0]["data"]["llm_call_time_ms"] = 1000.0
    records[0]["data"]["llm_timing_source"] = "openrouter_generation_time_ms"

    payload = _build(tmp_path, records)
    marker = payload["lanes"][0]["markers"][0]

    assert marker["t"] == pytest.approx(2.0)
    assert marker["t_real"] == pytest.approx(0.5)


def test_real_timeline_shifts_markers_after_final_compacted_span(
    tmp_path: Path,
) -> None:
    records = [
        _llm_action(0, 1000.0, 1004.0),
        _event("MCP", "task_complete", 0, 1004.5),
    ]
    records[0]["data"]["llm_call_time_ms"] = 1000.0
    records[0]["data"]["llm_timing_source"] = "openrouter_generation_time_ms"

    payload = _build(tmp_path, records)
    marker = payload["lanes"][0]["markers"][0]

    assert marker["t"] == pytest.approx(4.5)
    assert marker["t_real"] == pytest.approx(1.5)


def test_real_timeline_uses_trace_level_compaction_across_lanes(tmp_path: Path) -> None:
    records = [
        _llm_action(0, 1000.0, 1004.0, agent_id="agent-a"),
        _tool_action(0, 1005.0, 1006.0, "bash", agent_id="agent-b"),
    ]
    records[0]["data"]["llm_call_time_ms"] = 1000.0
    records[0]["data"]["llm_timing_source"] = "openrouter_generation_time_ms"

    payload = _build(tmp_path, records)
    lane_a = next(lane for lane in payload["lanes"] if lane["agent_id"] == "agent-a")
    lane_b = next(lane for lane in payload["lanes"] if lane["agent_id"] == "agent-b")
    llm_span = lane_a["spans"][0]
    tool_span = lane_b["spans"][0]

    assert llm_span["end_real_abs"] == pytest.approx(1001.0)
    assert tool_span["start_real_abs"] == pytest.approx(1001.0)


def test_tool_action_becomes_span(tmp_path: Path) -> None:
    payload = _build(tmp_path, [_tool_action(0, 1000.0, 1000.5, "bash")])
    spans = payload["lanes"][0]["spans"]
    tool_spans = [s for s in spans if s["type"] == "tool"]
    assert len(tool_spans) == 1
    assert tool_spans[0]["detail"]["tool_name"] == "bash"


def test_scheduling_span_requires_event_in_gap(tmp_path: Path) -> None:
    """A gap between two actions IS rendered as a scheduling span when
    a framework-level event falls inside it — regardless of gap width."""
    records = [
        _llm_action(0, 1000.0, 1001.0),
        _event("SCHEDULING", "message_dispatch", 0, 1001.02),
        _tool_action(0, 1001.05, 1001.10, "bash"),
    ]
    payload = _build(tmp_path, records)
    sched = [s for s in payload["lanes"][0]["spans"] if s["type"] == "scheduling"]
    assert len(sched) == 1
    # Hover detail must surface the underlying event for traceability.
    assert sched[0]["detail"]["events"] == ["message_dispatch"]


def test_scheduling_span_suppressed_without_event(tmp_path: Path) -> None:
    """A gap with NO framework event inside → no scheduling span.

    This is the "trusted evidence" invariant: every green bar in the
    Gantt must be backed by a real SCHEDULING / SESSION / CONTEXT event.
    Pure asyncio/HTTP wake-up noise has no visual representation —
    otherwise the same wall-clock gap would show or hide the span
    across different hardware, making the rendering platform-dependent.
    """
    records = [
        _llm_action(0, 1000.0, 1001.0),
        _tool_action(0, 1001.05, 1001.10, "bash"),  # 50ms gap, no event
    ]
    payload = _build(tmp_path, records)
    sched = [s for s in payload["lanes"][0]["spans"] if s["type"] == "scheduling"]
    assert len(sched) == 0


def test_scheduling_span_has_no_duration_threshold(tmp_path: Path) -> None:
    """A tiny 2ms gap with an event STILL produces a span — the rendering
    is gated purely on event presence, not on gap duration. This pins the
    platform-independence guarantee: same trace → same spans everywhere."""
    records = [
        _llm_action(0, 1000.0, 1001.0),
        _event("SCHEDULING", "session_lock_acquire", 0, 1001.001),
        _tool_action(0, 1001.002, 1001.100, "bash"),  # 2ms gap
    ]
    payload = _build(tmp_path, records)
    sched = [s for s in payload["lanes"][0]["spans"] if s["type"] == "scheduling"]
    assert len(sched) == 1
    assert sched[0]["detail"]["events"] == ["session_lock_acquire"]
    assert sched[0]["detail"]["gap_ms"] == pytest.approx(2.0, abs=0.1)


def test_parallel_tools_share_iteration(tmp_path: Path) -> None:
    """Three tool_exec actions all in iteration=5 → all rendered, no
    scheduling span between them (they may overlap or be close)."""
    records = [
        _tool_action(5, 1000.0, 1000.05, "bash"),
        _tool_action(5, 1000.0, 1000.06, "edit"),
        _tool_action(5, 1000.0, 1000.04, "read"),
    ]
    payload = _build(tmp_path, records)
    tool_spans = [s for s in payload["lanes"][0]["spans"] if s["type"] == "tool"]
    assert len(tool_spans) == 3
    assert {s["iteration"] for s in tool_spans} == {5}

    sched = [s for s in payload["lanes"][0]["spans"] if s["type"] == "scheduling"]
    assert len(sched) == 0


def test_event_becomes_marker(tmp_path: Path) -> None:
    records = [
        _llm_action(0, 1000.0, 1001.0),
        _event("SCHEDULING", "message_dispatch", 0, 1000.5),
    ]
    payload = _build(tmp_path, records)
    markers = payload["lanes"][0]["markers"]
    assert len(markers) == 1
    assert markers[0]["event"] == "message_dispatch"
    assert markers[0]["type"] == "scheduling"


def test_unknown_action_type_skipped(tmp_path: Path) -> None:
    """An unknown action_type is silently skipped, not crashed on."""
    bad = {
        "type": "action",
        "action_type": "future_thing",
        "action_id": "x",
        "agent_id": "a1",
        "iteration": 0,
        "ts_start": 1000.0,
        "ts_end": 1001.0,
        "data": {},
    }
    payload = _build(tmp_path, [bad, _llm_action(1, 1002.0, 1003.0)])
    spans = [s for s in payload["lanes"][0]["spans"] if s["type"] != "scheduling"]
    assert len(spans) == 1
    assert spans[0]["type"] == "llm"


def test_payload_carries_registries(tmp_path: Path) -> None:
    data = TraceData.load(_write_trace(tmp_path, [_llm_action(0, 1.0, 2.0)]))
    payload = build_gantt_payload_multi([("a", data)])
    assert "registries" in payload
    assert "spans" in payload["registries"]
    assert "markers" in payload["registries"]
    assert payload["registries"]["spans"] == DEFAULT_SPAN_REGISTRY
    assert payload["registries"]["markers"] == DEFAULT_MARKER_REGISTRY


def test_payload_registries_can_be_overridden(tmp_path: Path) -> None:
    """Caller-supplied registries replace the defaults."""
    custom_spans = {"llm": {"color": "#FF0000", "label": "Custom", "order": 0}}
    data = TraceData.load(_write_trace(tmp_path, [_llm_action(0, 1.0, 2.0)]))
    payload = build_gantt_payload_multi([("a", data)], span_registry=custom_spans)
    assert payload["registries"]["spans"] == custom_spans
    assert payload["registries"]["markers"] == DEFAULT_MARKER_REGISTRY


def test_metadata_uses_canonical_keys(tmp_path: Path) -> None:
    records = [
        _llm_action(0, 1000.0, 1001.0),
        _tool_action(0, 1001.0, 1001.1),
        _llm_action(1, 1002.0, 1003.0),
    ]
    payload = _build(tmp_path, records)
    meta = payload["metadata"]
    assert "n_actions" in meta
    assert "n_iterations" in meta
    assert "max_iterations" in meta
    assert "max_steps" not in meta, "v3 'max_steps' key must be removed"
    assert meta["n_actions"] == 3
    assert meta["n_iterations"] == 2  # iterations 0 and 1
    assert meta["max_iterations"] == 80  # from trace_metadata


def test_llm_span_detail_surfaces_silent_tool_calls(tmp_path: Path) -> None:
    """An llm_call action whose raw_response has ``content: null`` but
    non-empty ``tool_calls`` must still produce a useful tooltip: the
    requested tool calls become ``tool_calls_requested`` in the detail.
    Otherwise iterations where the LLM calls a tool silently show empty
    hover cards, which is exactly the ``oh why are some tooltips empty``
    regression the reviewer caught.
    """
    act = {
        "type": "action",
        "action_type": "llm_call",
        "action_id": "llm_0",
        "agent_id": "a1",
        "iteration": 0,
        "ts_start": 1000.0,
        "ts_end": 1001.0,
        "data": {
            "prompt_tokens": 100,
            "completion_tokens": 15,
            "raw_response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {
                                        "name": "write_file",
                                        "arguments": '{"path":"src/main.ts"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        },
    }
    payload = _build(tmp_path, [act])
    llm_spans = [s for s in payload["lanes"][0]["spans"] if s["type"] == "llm"]
    assert len(llm_spans) == 1
    detail = llm_spans[0]["detail"]
    assert "llm_content" not in detail, "content was null, should not be set"
    assert detail["tool_calls_requested"] == ['write_file(path="src/main.ts")']


def test_llm_span_detail_surfaces_both_content_and_tool_calls(tmp_path: Path) -> None:
    """When the LLM produces narrative text AND tool calls, both are
    reported so the user sees the reasoning alongside the action."""
    act = {
        "type": "action",
        "action_type": "llm_call",
        "action_id": "llm_0",
        "agent_id": "a1",
        "iteration": 0,
        "ts_start": 1000.0,
        "ts_end": 1001.0,
        "data": {
            "raw_response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Let me write the main entry point.",
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {
                                        "name": "write_file",
                                        "arguments": '{"path":"main.ts"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        },
    }
    payload = _build(tmp_path, [act])
    detail = payload["lanes"][0]["spans"][0]["detail"]
    assert "Let me write" in detail["llm_content"]
    assert detail["tool_calls_requested"] == ['write_file(path="main.ts")']


def test_llm_span_detail_supports_anthropic_raw_response(tmp_path: Path) -> None:
    act = {
        "type": "action",
        "action_type": "llm_call",
        "action_id": "llm_0",
        "agent_id": "a1",
        "iteration": 0,
        "ts_start": 1000.0,
        "ts_end": 1001.0,
        "data": {
            "raw_response": {
                "provider": "anthropic",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I'll read the config first."},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "Read",
                            "input": {"file_path": "config.yaml"},
                        },
                    ],
                },
            },
        },
    }
    payload = _build(tmp_path, [act])
    detail = payload["lanes"][0]["spans"][0]["detail"]
    assert "I'll read the config first." in detail["llm_content"]
    assert detail["tool_calls_requested"] == ['Read(file_path="config.yaml")']


def test_tool_calls_primary_field_path(tmp_path: Path) -> None:
    """A tool call with a JSON ``path`` argument renders as
    ``tool_name(path=\"...\")`` so users immediately see what file the
    model targeted, instead of the first 80 chars of a large ``content``
    field."""
    act = {
        "type": "action",
        "action_type": "llm_call",
        "action_id": "llm_0",
        "agent_id": "a1",
        "iteration": 0,
        "ts_start": 1.0,
        "ts_end": 2.0,
        "data": {
            "raw_response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "write_file",
                                        "arguments": '{"path": "src/main.ts", "content": "console.log(1)"}',
                                    }
                                }
                            ],
                        }
                    }
                ]
            }
        },
    }
    payload = _build(tmp_path, [act])
    detail = payload["lanes"][0]["spans"][0]["detail"]
    assert detail["tool_calls_requested"] == ['write_file(path="src/main.ts")']


def test_tool_calls_primary_field_command(tmp_path: Path) -> None:
    """A shell-style tool call uses the ``command`` field as primary."""
    act = {
        "type": "action",
        "action_type": "llm_call",
        "action_id": "llm_0",
        "agent_id": "a1",
        "iteration": 0,
        "ts_start": 1.0,
        "ts_end": 2.0,
        "data": {
            "raw_response": {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "bash",
                                        "arguments": '{"command": "ls -la src/", "timeout": 30}',
                                    }
                                }
                            ],
                        }
                    }
                ]
            }
        },
    }
    payload = _build(tmp_path, [act])
    detail = payload["lanes"][0]["spans"][0]["detail"]
    assert detail["tool_calls_requested"] == ['bash(command="ls -la src/")']


def test_tool_calls_no_primary_field_falls_back(tmp_path: Path) -> None:
    """When no known primary field is present, fall back to the
    generic 200-char raw argument preview."""
    act = {
        "type": "action",
        "action_type": "llm_call",
        "action_id": "llm_0",
        "agent_id": "a1",
        "iteration": 0,
        "ts_start": 1.0,
        "ts_end": 2.0,
        "data": {
            "raw_response": {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "mystery_tool",
                                        "arguments": '{"foo": "bar", "baz": 42}',
                                    }
                                }
                            ],
                        }
                    }
                ]
            }
        },
    }
    payload = _build(tmp_path, [act])
    detail = payload["lanes"][0]["spans"][0]["detail"]
    assert len(detail["tool_calls_requested"]) == 1
    summary = detail["tool_calls_requested"][0]
    assert summary.startswith("mystery_tool(")
    assert '"foo": "bar"' in summary


def test_llm_content_truncation_at_1000_chars(tmp_path: Path) -> None:
    """Long llm_content is truncated to 1000 chars + ellipsis (was 200)."""
    long_text = "A" * 1500
    act = {
        "type": "action",
        "action_type": "llm_call",
        "action_id": "llm_0",
        "agent_id": "a1",
        "iteration": 0,
        "ts_start": 1.0,
        "ts_end": 2.0,
        "data": {"raw_response": {"choices": [{"message": {"content": long_text}}]}},
    }
    payload = _build(tmp_path, [act])
    detail = payload["lanes"][0]["spans"][0]["detail"]
    assert detail["llm_content"] == "A" * 1000 + "..."


def test_llm_content_short_is_not_truncated(tmp_path: Path) -> None:
    """Content shorter than the cap passes through unchanged (no ...)."""
    act = {
        "type": "action",
        "action_type": "llm_call",
        "action_id": "llm_0",
        "agent_id": "a1",
        "iteration": 0,
        "ts_start": 1.0,
        "ts_end": 2.0,
        "data": {"raw_response": {"choices": [{"message": {"content": "hello"}}]}},
    }
    payload = _build(tmp_path, [act])
    assert payload["lanes"][0]["spans"][0]["detail"]["llm_content"] == "hello"


def test_action_type_map_module_constant() -> None:
    assert ACTION_TYPE_MAP["llm_call"] == "llm"
    assert ACTION_TYPE_MAP["tool_exec"] == "tool"


def test_action_detail_preserves_full_tool_fields(tmp_path: Path) -> None:
    """Action detail should preserve full tool payloads.

    The renderer's pinned tooltip has max-height:60vh + overflow-y:auto,
    so it can display up to the writer-side _TOOL_RESULT_MAX_CHARS=8000
    cap. Truncating to 100 at the data layer makes the pinned tooltip
    useless for inspecting real Bash / pytest output.
    """
    long_args = json.dumps(
        {
            "command": "python3 -m pytest tests/ -x -q --ignore=tests/test_main.py 2>&1 | tail -30",
            "description": "Run focused regression subset with clean output",
        }
    )
    long_result = (
        "........................................ [  8%]\n"
        "........................................ [ 16%]\n"
        "........................................ [ 24%]\n" * 10
    )
    assert len(long_args) > 100
    assert len(long_result) > 100

    tool_act = _tool_action(0, 1000.0, 1000.5, "Bash")
    tool_act["data"]["tool_args"] = long_args
    tool_act["data"]["tool_result"] = long_result

    payload = _build(tmp_path, [_llm_action(0, 999.9, 1000.0), tool_act])
    tool_spans = [s for s in payload["lanes"][0]["spans"] if s["type"] == "tool"]
    assert len(tool_spans) == 1
    detail = tool_spans[0]["detail"]

    assert detail["tool_args"] == long_args, (
        "FIX-A broken: tool_args was truncated at the data layer "
        f"(length={len(detail['tool_args'])}, expected={len(long_args)})"
    )
    assert detail["tool_result"] == long_result, (
        "FIX-A broken: tool_result was truncated at the data layer "
        f"(length={len(detail['tool_result'])}, expected={len(long_result)})"
    )
    assert not detail["tool_args"].endswith("...")
    assert not detail["tool_result"].endswith("...")


def test_event_detail_still_truncates_at_100_chars(tmp_path: Path) -> None:
    """Event detail keeps the shorter hover-only truncation cap."""
    long_value = "x" * 500

    from demo.gantt_viewer.backend.payload import _extract_detail_from_event

    event = {
        "type": "event",
        "category": "SCHEDULING",
        "event": "message_dispatch",
        "iteration": 0,
        "ts": 1000.0,
        "data": {"tool_args": long_value, "tool_result": long_value},
    }
    detail = _extract_detail_from_event(event)
    assert len(detail["tool_args"]) == 103, (
        f"event path should truncate; got length {len(detail['tool_args'])}"
    )
    assert detail["tool_args"].endswith("...")
    assert len(detail["tool_result"]) == 103
    assert detail["tool_result"].endswith("...")
