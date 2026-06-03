"""Regression tests for MCP registry wiring and sim_metrics tooltip fields.

The fixture coverage here checks that additive payload changes do not regress
existing OpenClaw trace rendering.
"""

from __future__ import annotations

from pathlib import Path

from demo.gantt_viewer.backend.payload import (
    ACTION_TYPE_MAP,
    DEFAULT_SPAN_REGISTRY,
    _MARKER_CATEGORIES,
    _extract_detail_from_action,
    build_gantt_payload,
)
from trace_collect.trace_inspector import TraceData

REPO_ROOT = Path(__file__).resolve().parents[3]
OPENCLAW_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "openclaw_minimal_v5.jsonl"

def test_marker_categories_contains_mcp() -> None:
    assert "MCP" in _MARKER_CATEGORIES, (
        "Phase 5 edit 1 missing: _MARKER_CATEGORIES must contain 'MCP' so "
        "MCP-category events become point markers in the Gantt"
    )
    assert "SCHEDULING" in _MARKER_CATEGORIES
    assert "SESSION" in _MARKER_CATEGORIES
    assert "CONTEXT" in _MARKER_CATEGORIES

def test_action_type_map_contains_mcp_call() -> None:
    assert ACTION_TYPE_MAP.get("mcp_call") == "mcp", (
        "Phase 5 edit 2 missing: ACTION_TYPE_MAP must map 'mcp_call' → 'mcp' "
        "so future mcp_call action types render as mcp spans"
    )
    assert ACTION_TYPE_MAP.get("llm_call") == "llm"
    assert ACTION_TYPE_MAP.get("tool_exec") == "tool"

def test_default_span_registry_contains_mcp_entry() -> None:
    assert "mcp" in DEFAULT_SPAN_REGISTRY, (
        "Phase 5 edit 3 missing: DEFAULT_SPAN_REGISTRY must contain 'mcp' "
        "entry so mcp_call spans have a color/label/order"
    )
    mcp_entry = DEFAULT_SPAN_REGISTRY["mcp"]
    assert "color" in mcp_entry
    assert "label" in mcp_entry
    assert "order" in mcp_entry
    assert mcp_entry["label"] == "MCP Call"
    assert mcp_entry["order"] == 3

def test_sim_metrics_flows_through_action_detail_extraction() -> None:
    fake_action = {
        "type": "action",
        "action_type": "llm_call",
        "data": {
            "messages_in": [{"role": "user", "content": "hello"}],
            "raw_response": {"choices": [{"message": {"content": "hi"}}]},
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "sim_metrics": {
                "timing": {
                    "ttft_ms": 123.4,
                    "tpot_ms": 5.6,
                    "total_ms": 200.0,
                },
                "vllm_scheduler_snapshot": {
                    "num_preemptions_total": 42.0,
                    "gpu_cache_usage_perc": 0.85,
                    "cpu_cache_usage_perc": 0.10,
                    "gpu_prefix_cache_hit_rate": 0.72,
                    "cpu_prefix_cache_hit_rate": 0.05,
                },
                "warmup": False,
            },
        },
    }

    detail = _extract_detail_from_action(fake_action)

    assert "sim_metrics" in detail, (
        "_extract_detail_from_action dropped the sim_metrics blob — "
        "Phase 5 preflight classification claim that 'sim_metrics flows "
        "through automatically' is broken"
    )

    sm = detail["sim_metrics"]
    assert sm["timing"]["ttft_ms"] == 123.4
    assert sm["timing"]["tpot_ms"] == 5.6
    assert sm["timing"]["total_ms"] == 200.0

    snap = sm["vllm_scheduler_snapshot"]
    assert snap["num_preemptions_total"] == 42.0
    assert snap["gpu_cache_usage_perc"] == 0.85
    assert snap["gpu_prefix_cache_hit_rate"] == 0.72

def test_sim_metrics_extraction_does_not_drop_other_fields() -> None:
    fake_action = {
        "type": "action",
        "action_type": "llm_call",
        "data": {
            "messages_in": ["this should be dropped"],
            "raw_response": {
                "choices": [{"message": {"content": "this becomes llm_content"}}]
            },
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "sim_metrics": {"warmup": True},
            "ttft_ms": 100.0,
        },
    }

    detail = _extract_detail_from_action(fake_action)

    assert "messages_in" not in detail
    assert "raw_response" not in detail
    assert detail.get("llm_content") == "this becomes llm_content"
    assert detail.get("sim_metrics") == {"warmup": True}
    assert detail.get("ttft_ms") == 100.0
    assert detail.get("prompt_tokens") == 10
    assert detail.get("completion_tokens") == 5

def test_openclaw_minimal_v5_fixture_renders_to_payload() -> None:
    assert OPENCLAW_FIXTURE.exists(), f"missing fixture: {OPENCLAW_FIXTURE}"

    data = TraceData.load(OPENCLAW_FIXTURE)
    payload = build_gantt_payload(data, label="test")

    assert payload["id"] == "test"
    assert payload["metadata"]["scaffold"] == "openclaw"
    assert len(payload["lanes"]) >= 1

def test_openclaw_fixture_spans_unchanged_by_phase5_edits() -> None:
    """The MCP mapping must not affect traces that still use ``tool_exec``."""
    data = TraceData.load(OPENCLAW_FIXTURE)
    payload = build_gantt_payload(data, label="test")

    span_types = set()
    for lane in payload["lanes"]:
        for span in lane["spans"]:
            span_types.add(span["type"])

    assert "llm" in span_types
    assert "tool" in span_types
    assert "mcp" not in span_types, (
        "Phase 5 regression: openclaw_minimal_v5 fixture should NOT produce "
        "mcp spans (its mcp_* tools are tool_exec actions, not mcp_call). "
        "If this fails, the action_type → span_type dispatch is broken."
    )

def test_phase5_payload_still_carries_full_default_span_registry() -> None:
    from demo.gantt_viewer.backend.payload import build_gantt_payload_multi

    data = TraceData.load(OPENCLAW_FIXTURE)
    multi = build_gantt_payload_multi([("test", data)])

    spans_registry = multi["registries"]["spans"]
    assert set(spans_registry.keys()) == {"llm", "tool", "scheduling", "mcp"}

def test_synthetic_mcp_call_action_renders_as_mcp_span() -> None:
    from demo.gantt_viewer.backend.payload import _build_spans_and_markers

    fake_actions = [
        {
            "action_type": "mcp_call",
            "ts_start": 1000.0,
            "ts_end": 1001.0,
            "iteration": 0,
            "data": {"tool_name": "mcp_context7_search"},
        }
    ]
    fake_events: list[dict] = []
    spans, markers = _build_spans_and_markers(fake_actions, fake_events, t0=1000.0)
    assert len(spans) == 1
    assert spans[0]["type"] == "mcp"

def test_synthetic_mcp_event_renders_as_marker() -> None:
    from demo.gantt_viewer.backend.payload import _build_spans_and_markers

    fake_actions = [
        {
            "action_type": "llm_call",
            "ts_start": 1000.0,
            "ts_end": 1001.0,
            "iteration": 0,
            "data": {},
        }
    ]
    fake_events = [
        {
            "category": "MCP",
            "event": "mcp_handshake_start",
            "ts": 1000.5,
            "iteration": 0,
            "data": {"server": "context7"},
        }
    ]
    spans, markers = _build_spans_and_markers(fake_actions, fake_events, t0=1000.0)

    mcp_markers = [m for m in markers if m["type"] == "mcp"]
    assert len(mcp_markers) == 1, (
        "Phase 5 regression: an MCP-category event must produce a marker "
        "after _MARKER_CATEGORIES gains the 'MCP' entry"
    )
    assert mcp_markers[0]["event"] == "mcp_handshake_start"
