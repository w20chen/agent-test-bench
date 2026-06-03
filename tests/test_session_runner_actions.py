"""Regression test for the OpenClaw TraceCollectorHook v4 emission bug.

US-010: Earlier the hook defined ``after_llm_response`` to emit the
``llm_call`` action, but ``AgentLoop``'s CompositeHook never invokes
that method — it only calls ``before_iteration``, ``before_execute_tools``
and ``after_iteration``. The result was traces with ``tool_exec`` actions
but ZERO ``llm_call`` actions, which broke Gantt rendering and the
simulator's iteration grouping.

This test drives ``TraceCollectorHook`` with synthetic ``AgentHookContext``
inputs through the realistic hook order and asserts that the JSONL trace
contains BOTH action types after a single iteration.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

# Skip the entire module if OpenClaw deps are unavailable.
pytest.importorskip("agents.openclaw._session_runner")

from agents.openclaw._session_runner import TraceCollectorHook, _resolve_run_outcome


class _StubResponse:
    def __init__(
        self,
        content: str = "",
        finish_reason: str = "stop",
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.content = content
        self.finish_reason = finish_reason
        self.reasoning_content: str | None = None
        self.extra: dict[str, Any] | None = extra


class _StubToolCall:
    _counter = 0

    def __init__(self, name: str, arguments: dict[str, Any]) -> None:
        _StubToolCall._counter += 1
        self.id = f"call_{_StubToolCall._counter}"
        self.name = name
        self.arguments = arguments


class _StubContext:
    """Mimics the bits of AgentHookContext the trace hook reads."""

    def __init__(
        self,
        iteration: int,
        messages: list[dict[str, Any]],
        tool_calls: list[_StubToolCall] | None = None,
        usage: dict[str, int] | None = None,
        response: _StubResponse | None = None,
    ) -> None:
        self.iteration = iteration
        self.messages = messages
        self.tool_calls = tool_calls or []
        self.usage = usage or {}
        self.response = response
        self.malformed_retry_count = 0


def test_trace_collector_emits_llm_call_action(tmp_path: Path) -> None:
    import asyncio

    asyncio.run(_drive_emits_llm_call_action(tmp_path))


async def _drive_emits_llm_call_action(tmp_path: Path) -> None:
    """One iteration with one tool call must produce ONE llm_call action
    AND ONE tool_exec action — in that chronological order."""
    trace_file = tmp_path / "trace.jsonl"
    hook = TraceCollectorHook(trace_file, instance_id="test-1")

    # ── Iteration 0 ──────────────────────────────────────────────
    msgs_in = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Write hello world."},
    ]
    ctx_before = _StubContext(iteration=0, messages=msgs_in)
    await hook.before_iteration(ctx_before)

    # Simulate LLM response producing a tool call
    msgs_after_llm = msgs_in + [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": '{"path":"a.py"}'},
                }
            ],
        }
    ]
    stub_tc = _StubToolCall("write_file", {"path": "a.py"})
    ctx_before_tools = _StubContext(
        iteration=0,
        messages=msgs_after_llm,
        tool_calls=[stub_tc],
        usage={"prompt_tokens": 100, "completion_tokens": 20},
    )
    await hook.before_execute_tools(ctx_before_tools)

    # Simulate tool result appended to messages
    msgs_after_tool = msgs_after_llm + [
        {"role": "tool", "tool_call_id": stub_tc.id, "name": "write_file", "content": "wrote a.py"}
    ]
    ctx_after = _StubContext(
        iteration=0,
        messages=msgs_after_tool,
        tool_calls=[stub_tc],
        usage={"prompt_tokens": 100, "completion_tokens": 20},
        response=_StubResponse(content="", finish_reason="tool_calls"),
    )
    await hook.after_iteration(ctx_after)
    hook.close()

    # ── Verify ──────────────────────────────────────────────────
    lines = trace_file.read_text().strip().splitlines()
    records = [json.loads(line) for line in lines]
    actions = [r for r in records if r.get("type") == "action"]

    llm_calls = [a for a in actions if a.get("action_type") == "llm_call"]
    tool_execs = [a for a in actions if a.get("action_type") == "tool_exec"]

    assert len(llm_calls) == 1, (
        f"Expected exactly 1 llm_call action, got {len(llm_calls)}. "
        f"All action types: {[a.get('action_type') for a in actions]}"
    )
    assert len(tool_execs) == 1, f"Expected 1 tool_exec, got {len(tool_execs)}"

    # Order: llm_call must come BEFORE tool_exec in the file
    llm_idx = next(
        i
        for i, r in enumerate(records)
        if r.get("type") == "action" and r.get("action_type") == "llm_call"
    )
    tool_idx = next(
        i
        for i, r in enumerate(records)
        if r.get("type") == "action" and r.get("action_type") == "tool_exec"
    )
    assert llm_idx < tool_idx, "llm_call action must precede tool_exec in trace"

    # Verify llm_call action carries the snapshotted messages_in (NOT the
    # post-tool-result messages — that would be a leak from after_iteration)
    llm = llm_calls[0]
    assert llm["data"]["messages_in"] == msgs_in
    assert llm["data"]["prompt_tokens"] == 100
    assert llm["data"]["completion_tokens"] == 20
    assert llm["iteration"] == 0
    assert llm["ts_start"] <= llm["ts_end"]


def test_trace_collector_llm_only_iteration(tmp_path: Path) -> None:
    import asyncio

    asyncio.run(_drive_llm_only_iteration(tmp_path))


async def _drive_llm_only_iteration(tmp_path: Path) -> None:
    """An iteration with NO tool calls (final answer) still emits llm_call."""
    trace_file = tmp_path / "trace.jsonl"
    hook = TraceCollectorHook(trace_file, instance_id="test-2")

    msgs_in = [{"role": "user", "content": "Say hi."}]
    await hook.before_iteration(_StubContext(iteration=0, messages=msgs_in))
    # Note: before_execute_tools is NOT called when there are no tool calls
    msgs_after_llm = msgs_in + [{"role": "assistant", "content": "hi"}]
    await hook.after_iteration(
        _StubContext(
            iteration=0,
            messages=msgs_after_llm,
            tool_calls=[],
            usage={"prompt_tokens": 5, "completion_tokens": 1},
            response=_StubResponse(content="hi", finish_reason="stop"),
        )
    )
    hook.close()

    records = [json.loads(line) for line in trace_file.read_text().strip().splitlines()]
    llm_calls = [
        r
        for r in records
        if r.get("type") == "action" and r.get("action_type") == "llm_call"
    ]
    tool_execs = [
        r
        for r in records
        if r.get("type") == "action" and r.get("action_type") == "tool_exec"
    ]

    assert len(llm_calls) == 1
    assert len(tool_execs) == 0
    # ts_end falls back to "now" when before_execute_tools was never called
    assert llm_calls[0]["ts_end"] >= llm_calls[0]["ts_start"]


def test_trace_collector_records_openrouter_latency_fields(tmp_path: Path) -> None:
    import asyncio

    asyncio.run(_drive_openrouter_latency_fields(tmp_path))


async def _drive_openrouter_latency_fields(tmp_path: Path) -> None:
    trace_file = tmp_path / "trace.jsonl"
    hook = TraceCollectorHook(trace_file, instance_id="test-openrouter")

    await hook.before_iteration(
        _StubContext(iteration=0, messages=[{"role": "user", "content": "Ping"}])
    )
    hook._iter_start_wall = 100.0
    hook._before_exec_wall = 0.0

    openrouter_metadata = {
        "generation_id": "gen-123",
        "request_id": "req-123",
        "provider_name": "Z.AI",
        "latency_ms": 7000.0,
        "generation_time_ms": 6500.0,
        "provider_latency_ms": 6800.0,
        "upstream_id": "up-123",
        "provider_responses": [
            {"provider_name": "Z.AI", "latency_ms": 6800.0, "status": 200}
        ],
    }
    response = _StubResponse(
        content="pong",
        finish_reason="stop",
        extra={
            "llm_wall_ts_end": 115.0,
            "llm_call_time_ms": 6500.0,
            "llm_timing_source": "openrouter_generation_time_ms",
            "openrouter_generation_id": "gen-123",
            "openrouter_request_id": "req-123",
            "openrouter_latency_ms": 7000.0,
            "openrouter_generation_time_ms": 6500.0,
            "openrouter_provider_latency_ms": 6800.0,
            "openrouter_provider_name": "Z.AI",
            "openrouter_upstream_id": "up-123",
            "openrouter_metadata": openrouter_metadata,
        },
    )
    await hook.after_iteration(
        _StubContext(
            iteration=0,
            messages=[
                {"role": "user", "content": "Ping"},
                {"role": "assistant", "content": "pong"},
            ],
            usage={"prompt_tokens": 12, "completion_tokens": 3},
            response=response,
        )
    )
    await hook.write_summary(success=True, elapsed_s=15.0)

    records = [json.loads(line) for line in trace_file.read_text().strip().splitlines()]
    llm_call = next(
        r
        for r in records
        if r.get("type") == "action" and r.get("action_type") == "llm_call"
    )
    llm_event = next(
        r
        for r in records
        if r.get("type") == "event" and r.get("event") == "llm_call_end"
    )
    summary = next(r for r in records if r.get("type") == "summary")

    assert llm_call["data"]["llm_latency_ms"] == 6500.0
    assert llm_call["data"]["llm_call_time_ms"] == 6500.0
    assert llm_call["data"]["llm_wall_latency_ms"] == 15000.0
    assert llm_call["data"]["llm_timing_source"] == "openrouter_generation_time_ms"
    assert llm_call["data"]["openrouter_latency_ms"] == 7000.0
    assert llm_call["data"]["openrouter_generation_time_ms"] == 6500.0
    assert llm_call["data"]["openrouter_provider_latency_ms"] == 6800.0
    assert llm_call["data"]["openrouter_generation_id"] == "gen-123"
    assert (
        llm_call["data"]["raw_response"]["openrouter_metadata"] == openrouter_metadata
    )
    assert llm_event["data"]["llm_latency_ms"] == 6500.0
    assert llm_event["data"]["openrouter_request_id"] == "req-123"
    assert "openrouter_metadata" not in llm_event["data"]
    assert summary["total_llm_ms"] == 6500.0
    assert summary["total_llm_call_time_ms"] == 6500.0
    assert summary["llm_call_time_count"] == 1
    assert summary["llm_timing_source"] == "openrouter_generation_time_ms"
    assert summary["total_llm_wall_ms"] == 15000.0


def test_trace_collector_refetches_late_openrouter_metadata(tmp_path: Path) -> None:
    import asyncio

    asyncio.run(_drive_refetches_late_openrouter_metadata(tmp_path))


async def _drive_refetches_late_openrouter_metadata(tmp_path: Path) -> None:
    trace_file = tmp_path / "trace.jsonl"
    hook = TraceCollectorHook(trace_file, instance_id="test-openrouter-late")

    await hook.before_iteration(
        _StubContext(iteration=0, messages=[{"role": "user", "content": "Ping"}])
    )
    hook._iter_start_wall = 100.0
    hook._before_exec_wall = 115.0

    extra: dict[str, Any] = {
        "llm_wall_ts_end": 115.0,
        "openrouter_generation_id": "gen-late",
        "openrouter_metadata_fetch_status": "pending",
        "openrouter_metadata_fetch_ms": 1.0,
    }

    async def initial_fetch() -> dict[str, Any]:
        result = {
            "openrouter_metadata_fetch_status": "unavailable",
            "openrouter_metadata_fetch_ms": 2.0,
            "openrouter_metadata_fetch_attempt_count": 1,
            "openrouter_metadata_fetch_last_status_code": 404,
            "openrouter_metadata_fetch_last_reason": "not_found",
        }
        extra.update(result)
        return result

    async def refetch() -> dict[str, Any]:
        return {
            "openrouter_metadata_fetch_status": "success",
            "openrouter_metadata_fetch_ms": 3.0,
            "openrouter_metadata_fetch_attempt_count": 1,
            "openrouter_metadata_fetch_last_status_code": 200,
            "openrouter_metadata_fetch_last_reason": "success",
            "openrouter_metadata": {
                "generation_id": "gen-late",
                "generation_time_ms": 4321.0,
                "latency_ms": 5000.0,
            },
            "openrouter_generation_time_ms": 4321.0,
            "openrouter_latency_ms": 5000.0,
            "llm_call_time_ms": 4321.0,
            "llm_timing_source": "openrouter_generation_time_ms",
        }

    extra["_openrouter_metadata_task"] = asyncio.create_task(initial_fetch())
    extra["_openrouter_metadata_refetcher"] = refetch
    response = _StubResponse(content="pong", finish_reason="stop", extra=extra)

    await hook.after_iteration(
        _StubContext(
            iteration=0,
            messages=[
                {"role": "user", "content": "Ping"},
                {"role": "assistant", "content": "pong"},
            ],
            usage={"prompt_tokens": 12, "completion_tokens": 3},
            response=response,
        )
    )
    await hook.write_summary(success=True, elapsed_s=15.0)

    records = [json.loads(line) for line in trace_file.read_text().strip().splitlines()]
    llm_call = next(
        r
        for r in records
        if r.get("type") == "action" and r.get("action_type") == "llm_call"
    )
    llm_event = next(
        r
        for r in records
        if r.get("type") == "event" and r.get("event") == "llm_call_end"
    )
    summary = next(r for r in records if r.get("type") == "summary")

    assert llm_call["data"]["openrouter_metadata_fetch_status"] == "success"
    assert llm_call["data"]["openrouter_metadata_refetch_attempted"] is True
    assert llm_call["data"]["openrouter_metadata_initial_fetch_status"] == "unavailable"
    assert llm_call["data"]["openrouter_generation_time_ms"] == 4321.0
    assert llm_call["data"]["llm_call_time_ms"] == 4321.0
    assert llm_call["data"]["llm_timing_source"] == "openrouter_generation_time_ms"
    assert "_openrouter_metadata_task" not in llm_call["data"]
    assert "_openrouter_metadata_refetcher" not in llm_call["data"]
    assert llm_event["data"]["openrouter_metadata_fetch_status"] == "success"
    assert summary["total_llm_call_time_ms"] == 4321.0
    assert summary["llm_timing_source"] == "openrouter_generation_time_ms"


def test_resolve_run_outcome_uses_trace_llm_error_event(tmp_path: Path) -> None:
    trace_file = tmp_path / "trace.jsonl"
    trace_file.write_text(
        json.dumps(
            {
                "type": "event",
                "event": "llm_error",
                "category": "LLM",
                "data": {"error_message": "credits exhausted"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    stop_reason, error = _resolve_run_outcome(
        outcome={},
        content='Error: {"error":{"message":"This request requires more credits"}}',
        trace_file=trace_file,
    )

    assert stop_reason == "error"
    assert "requires more credits" in (error or "")
