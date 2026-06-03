"""Shared bus-based session runner for CLI and evaluation flows."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.openclaw._hook import AgentHook, AgentHookContext
from agents.openclaw._loop import AgentLoop
from agents.openclaw.config.schema import ExecToolConfig
from agents.openclaw.bus.events import InboundMessage
from agents.openclaw.bus.queue import MessageBus
from agents.openclaw.eval.collector import ResultCollector
from agents.base import TraceAction
from agents.openclaw.eval.types import (
    LLM,
    MCP,
    SUBAGENT,
    TOOL,
    EvalTraceEvent,
    EvalTraceSummary,
)
from agents.openclaw.providers.base import LLMProvider
from agents.openclaw.session.manager import SessionManager
from trace_collect.latency_metrics import summarize_llm_latencies


def _trace_has_llm_error(trace_file: Path | None) -> bool:
    if trace_file is None or not trace_file.exists():
        return False
    try:
        with trace_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                if record.get("type") == "event" and record.get("event") == "llm_error":
                    return True
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    return False


def _resolve_run_outcome(
    *,
    outcome: dict[str, Any],
    content: str | None,
    trace_file: Path | None,
) -> tuple[str, str | None]:
    stop_reason = str(outcome.get("stop_reason") or "completed")
    error = outcome.get("error")
    if (
        stop_reason == "completed"
        and error is None
        and _trace_has_llm_error(trace_file)
    ):
        return "error", content or "LLM returned error."
    if error is None and stop_reason != "completed" and content:
        error = content
    return stop_reason, error


class TraceCollectorHook(AgentHook):
    """Collect per-iteration actions, events, and summaries as JSONL."""

    def __init__(
        self,
        trace_file: Path,
        instance_id: str,
        *,
        agent_id: str | None = None,
        task_id: str | None = None,
    ) -> None:
        self.trace_file = trace_file
        self.instance_id = instance_id
        self.agent_id = agent_id or instance_id
        self.program_id = self.agent_id
        self.task_id = task_id or instance_id
        self.trace_file.parent.mkdir(parents=True, exist_ok=True)
        self._wall_start = time.monotonic()
        self._total_tokens = 0
        self._n_iterations = 0
        self._tool_times: dict[str, float] = {}
        self._tool_timeouts: dict[str, int] = {}
        self._tool_start_ts: dict[str, float] = {}
        self._iter_start_wall: float = 0.0
        self._iter_messages_snapshot: list[dict[str, Any]] | None = None
        self._before_exec_wall: float = 0.0
        self._records: list[dict[str, Any]] = []
        self._actions: list[dict[str, Any]] = []
        self._pending_llm_records: list[dict[str, Any]] = []
        self._flushed = False
        self._fh = open(trace_file, "w", encoding="utf-8")  # noqa: SIM115

    def close(self) -> None:
        if self._flushed:
            return
        if not self._fh.closed:
            self._fh.close()
        tmp_trace_file = self.trace_file.with_suffix(f"{self.trace_file.suffix}.tmp")
        tmp_trace_file.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in self._records
            ),
            encoding="utf-8",
        )
        tmp_trace_file.replace(self.trace_file)
        self._flushed = True

    def add_record(self, record: dict[str, Any]) -> None:
        self._records.append(record)
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def emit_event(
        self,
        category: str,
        event: str,
        data: dict[str, Any],
        *,
        iteration: int = 0,
    ) -> None:
        entry = EvalTraceEvent(
            agent_id=self.agent_id,
            program_id=self.program_id,
            instance_id=self.instance_id,
            event=event,
            category=category,
            data=data,
            ts=time.time(),
            iteration=iteration,
        )
        self.add_record(entry.to_dict())

    async def before_iteration(self, context: AgentHookContext) -> None:
        self._iter_start_wall = time.time()
        self._iter_messages_snapshot = self._clone_messages(context.messages)
        self.emit_event(
            LLM,
            "llm_call_start",
            {"messages_in": self._iter_messages_snapshot},
            iteration=context.iteration,
        )

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        self._before_exec_wall = time.time()
        if context.tool_calls:
            for tc in context.tool_calls:
                self._tool_start_ts[tc.id] = time.monotonic()
                is_mcp = tc.name.startswith("mcp_")
                self.emit_event(
                    MCP if is_mcp else TOOL,
                    "tool_exec_start",
                    {
                        "tool_name": tc.name,
                        "args_preview": json.dumps(tc.arguments, ensure_ascii=False)[
                            :200
                        ],
                    },
                    iteration=context.iteration,
                )

    async def after_iteration(self, context: AgentHookContext) -> None:
        ts_now = time.time()
        self._n_iterations += 1

        usage = context.usage or {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        self._total_tokens += prompt_tokens + completion_tokens

        llm_ts_end = (
            self._resolve_llm_ts_end(context.response)
            or self._before_exec_wall
            or ts_now
        )
        llm_wall_latency_ms = max(0.0, (llm_ts_end - self._iter_start_wall) * 1000)
        llm_call_time_ms = self._resolve_llm_call_time_ms(
            context.response,
            llm_wall_latency_ms,
        )
        llm_timing_source = self._resolve_llm_timing_source(context.response)
        resp_dict = self._build_raw_response(
            context=context,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        llm_event_data = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "llm_latency_ms": round(llm_call_time_ms, 2),
            "llm_call_time_ms": round(llm_call_time_ms, 2),
            "llm_wall_latency_ms": round(llm_wall_latency_ms, 2),
            "llm_timing_source": llm_timing_source,
            "finish_reason": context.response.finish_reason
            if context.response
            else None,
            "is_malformed_retry": context.malformed_retry_count > 0,
        }
        llm_action_data: dict[str, Any] = {
            "messages_in": self._iter_messages_snapshot,
            "raw_response": resp_dict,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "llm_latency_ms": round(llm_call_time_ms, 2),
            "llm_call_time_ms": round(llm_call_time_ms, 2),
            "llm_wall_latency_ms": round(llm_wall_latency_ms, 2),
            "llm_timing_source": llm_timing_source,
            "malformed_retry_count": context.malformed_retry_count,
            "is_malformed_retry": context.malformed_retry_count > 0,
        }
        trace_llm_fields = self._extract_trace_llm_fields(context.response)
        if trace_llm_fields:
            llm_event_data.update(
                {
                    key: value
                    for key, value in trace_llm_fields.items()
                    if key != "openrouter_metadata"
                }
            )
            llm_action_data.update(trace_llm_fields)
        self.emit_event(
            LLM,
            "llm_call_end",
            llm_event_data,
            iteration=context.iteration,
        )
        llm_action = TraceAction(
            action_type="llm_call",
            action_id=f"llm_{context.iteration}",
            agent_id=self.agent_id,
            program_id=self.program_id,
            instance_id=self.instance_id,
            iteration=context.iteration,
            ts_start=self._iter_start_wall,
            ts_end=llm_ts_end,
            data=llm_action_data,
        )
        self._write_action(llm_action)
        if context.response and getattr(context.response, "extra", None):
            if context.response.extra.get("_openrouter_metadata_task") is not None:
                self._pending_llm_records.append(
                    {
                        "response": context.response,
                        "event_data": llm_event_data,
                        "action_data": llm_action_data,
                        "raw_response": resp_dict,
                    }
                )
        self._before_exec_wall = 0.0
        self._iter_messages_snapshot = None

        tool_results_from_messages = self._extract_tool_results(context.messages)
        if tool_results_from_messages:
            tool_args_by_id: dict[str, str] = {}
            tool_name_by_id: dict[str, str] = {}
            if context.tool_calls:
                for tc in context.tool_calls:
                    tool_args_by_id[tc.id] = json.dumps(
                        tc.arguments, ensure_ascii=False
                    )
                    tool_name_by_id[tc.id] = tc.name

            for tc_id, tool_name, tool_content, tool_ok in tool_results_from_messages:
                if tool_name == "_invalid_tool_call" or (
                    tc_id and tc_id.startswith("malformed_retry_")
                ):
                    continue
                tool_start_mono = self._tool_start_ts.pop(tc_id, None)
                duration_ms = (
                    (time.monotonic() - tool_start_mono) * 1000
                    if tool_start_mono
                    else 0.0
                )
                self._tool_times[tool_name] = (
                    self._tool_times.get(tool_name, 0.0) + duration_ms
                )
                if not tool_ok:
                    self._tool_timeouts[tool_name] = (
                        self._tool_timeouts.get(tool_name, 0) + 1
                    )

                is_mcp = tool_name.startswith("mcp_")
                self.emit_event(
                    MCP if is_mcp else TOOL,
                    "tool_exec_end",
                    {
                        "tool_name": tool_name,
                        "success": tool_ok,
                        "duration_ms": round(duration_ms, 1),
                        "result_preview": tool_content[:200],
                    },
                    iteration=context.iteration,
                )
                if tool_name == "spawn":
                    self.emit_event(
                        SUBAGENT,
                        "subagent_complete",
                        {"task_preview": tool_content[:200]},
                        iteration=context.iteration,
                    )

                tool_ts_end = time.time()
                tool_ts_start = (
                    tool_ts_end - duration_ms / 1000 if duration_ms else tool_ts_end
                )
                action_id_suffix = tc_id if tc_id else tool_name
                tool_action = TraceAction(
                    action_type="tool_exec",
                    action_id=f"tool_{context.iteration}_{action_id_suffix}",
                    agent_id=self.agent_id,
                    program_id=self.program_id,
                    instance_id=self.instance_id,
                    iteration=context.iteration,
                    ts_start=tool_ts_start,
                    ts_end=tool_ts_end,
                    data={
                        "tool_name": tool_name,
                        "tool_call_id": tc_id,
                        "tool_args": tool_args_by_id.get(tc_id, ""),
                        "tool_result": tool_content,
                        "duration_ms": round(duration_ms, 1),
                        "success": tool_ok,
                    },
                )
                self._write_action(tool_action)

        if context.response:
            finish_reason = context.response.finish_reason
            if finish_reason == "error":
                error_data: dict[str, Any] = {
                    "error_message": context.response.content[:500]
                    if context.response.content
                    else "",
                    "finish_reason": finish_reason,
                }
                if context.response.extra:
                    error_data.update(
                        {
                            key: value
                            for key, value in context.response.extra.items()
                            if key != "llm_wall_ts_end" and not key.startswith("_")
                        }
                    )
                self.emit_event(
                    LLM,
                    "llm_error",
                    error_data,
                    iteration=context.iteration,
                )
            elif finish_reason == "max_iterations":
                self.emit_event(
                    LLM,
                    "max_iterations",
                    {"total_tokens": self._total_tokens},
                    iteration=context.iteration,
                )

    @staticmethod
    def _extract_tool_results(
        messages: list[dict],
    ) -> list[tuple[str, str, str, bool]]:
        """Extract (tool_call_id, tool_name, content, ok) from trailing tool messages."""
        results: list[tuple[str, str, str, bool]] = []
        i = len(messages) - 1
        while i >= 0 and messages[i].get("role") == "tool":
            m = messages[i]
            tool_call_id = m.get("tool_call_id", "")
            name = m.get("name", "unknown")
            content = str(m.get("content", ""))
            ok = not content.startswith("Error")
            results.append((tool_call_id, name, content, ok))
            i -= 1
        results.reverse()
        return results

    def _write_action(self, action: TraceAction) -> None:
        d = action.to_dict()
        self._actions.append(d)
        self.add_record(d)

    async def _resolve_pending_llm_records(self) -> None:
        for pending in self._pending_llm_records:
            response = pending["response"]
            if response is None or not getattr(response, "extra", None):
                continue
            task = response.extra.get("_openrouter_metadata_task")
            if task is not None:
                try:
                    await task
                except Exception:
                    pass
            await self._refresh_unavailable_openrouter_metadata(response)
            response.extra.pop("_openrouter_metadata_task", None)
            response.extra.pop("_openrouter_metadata_refetcher", None)
            response.extra["openrouter_metadata_task_pending"] = False
            llm_wall_latency_ms = float(pending["action_data"]["llm_wall_latency_ms"])
            llm_call_time_ms = self._resolve_llm_call_time_ms(
                response,
                llm_wall_latency_ms,
            )
            llm_timing_source = self._resolve_llm_timing_source(response)
            pending["event_data"]["llm_latency_ms"] = round(llm_call_time_ms, 2)
            pending["event_data"]["llm_call_time_ms"] = round(llm_call_time_ms, 2)
            pending["event_data"]["llm_timing_source"] = llm_timing_source
            pending["action_data"]["llm_latency_ms"] = round(llm_call_time_ms, 2)
            pending["action_data"]["llm_call_time_ms"] = round(llm_call_time_ms, 2)
            pending["action_data"]["llm_timing_source"] = llm_timing_source
            trace_llm_fields = self._extract_trace_llm_fields(response)
            pending["event_data"].update(
                {
                    key: value
                    for key, value in trace_llm_fields.items()
                    if key != "openrouter_metadata"
                }
            )
            pending["action_data"].update(trace_llm_fields)
            raw_response = pending["raw_response"]
            openrouter_metadata = response.extra.get("openrouter_metadata")
            if openrouter_metadata is not None:
                raw_response["openrouter_metadata"] = openrouter_metadata
            generation_id = response.extra.get("openrouter_generation_id")
            if generation_id is not None:
                raw_response["openrouter_generation_id"] = generation_id
        self._pending_llm_records.clear()

    @staticmethod
    async def _refresh_unavailable_openrouter_metadata(response: Any) -> None:
        if response is None or not getattr(response, "extra", None):
            return
        extra = response.extra
        if extra.get("openrouter_metadata_fetch_status") == "success":
            return
        generation_id = extra.get("openrouter_generation_id")
        refetcher = extra.get("_openrouter_metadata_refetcher")
        if not generation_id or not callable(refetcher):
            return

        initial_status = extra.get("openrouter_metadata_fetch_status")
        initial_fetch_ms = extra.get("openrouter_metadata_fetch_ms")
        try:
            refreshed = await refetcher()
        except Exception as exc:  # pragma: no cover - defensive guard
            extra["openrouter_metadata_refetch_attempted"] = True
            extra["openrouter_metadata_refetch_error"] = str(exc)
            return
        if not isinstance(refreshed, dict):
            return

        extra.update(refreshed)
        extra["openrouter_metadata_refetch_attempted"] = True
        if initial_status is not None:
            extra["openrouter_metadata_initial_fetch_status"] = initial_status
        if initial_fetch_ms is not None:
            extra["openrouter_metadata_initial_fetch_ms"] = initial_fetch_ms

    @staticmethod
    def _clone_messages(messages: list[dict] | None) -> list[dict[str, Any]] | None:
        if not messages:
            return None
        return json.loads(json.dumps(messages, ensure_ascii=False, default=str))

    @staticmethod
    def _resolve_llm_ts_end(response: Any | None) -> float | None:
        if response is None or not getattr(response, "extra", None):
            return None
        value = response.extra.get("llm_wall_ts_end")
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _resolve_llm_call_time_ms(
        response: Any | None,
        llm_wall_latency_ms: float,
    ) -> float:
        if response is None or not getattr(response, "extra", None):
            return llm_wall_latency_ms
        for key in (
            "llm_call_time_ms",
            "openrouter_generation_time_ms",
            "llm_latency_ms",
        ):
            value = response.extra.get(key)
            try:
                if value is not None:
                    return float(value)
            except (TypeError, ValueError):
                continue
        return llm_wall_latency_ms

    @staticmethod
    def _resolve_llm_timing_source(response: Any | None) -> str:
        if response is None or not getattr(response, "extra", None):
            return "wall_clock_ms"
        source = response.extra.get("llm_timing_source")
        if isinstance(source, str) and source:
            return source
        if response.extra.get("openrouter_generation_time_ms") is not None:
            return "openrouter_generation_time_ms"
        return "wall_clock_ms"

    @staticmethod
    def _extract_trace_llm_fields(response: Any | None) -> dict[str, Any]:
        if response is None or not getattr(response, "extra", None):
            return {}
        extra = response.extra
        result = {}
        for key in (
            "llm_call_time_ms",
            "llm_timing_source",
            "openrouter_generation_id",
            "openrouter_request_id",
            "openrouter_latency_ms",
            "openrouter_generation_time_ms",
            "openrouter_moderation_latency_ms",
            "openrouter_provider_latency_ms",
            "openrouter_provider_name",
            "openrouter_upstream_id",
            "openrouter_created_at",
            "openrouter_api_type",
            "openrouter_metadata_capture_enabled",
            "openrouter_metadata_task_pending",
            "openrouter_metadata_retry_delays_s",
            "openrouter_metadata_timeout_s",
            "openrouter_metadata_fetch_ms",
            "openrouter_metadata_fetch_status",
            "openrouter_metadata_fetch_attempt_count",
            "openrouter_metadata_fetch_status_codes",
            "openrouter_metadata_fetch_last_status_code",
            "openrouter_metadata_fetch_last_reason",
            "openrouter_metadata_fetch_last_error_type",
            "openrouter_metadata_refetch_attempted",
            "openrouter_metadata_refetch_error",
            "openrouter_metadata_initial_fetch_status",
            "openrouter_metadata_initial_fetch_ms",
            "openrouter_metadata",
        ):
            if key in extra:
                result[key] = extra[key]
        return result

    def _build_raw_response(
        self,
        *,
        context: AgentHookContext,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> dict[str, Any]:
        resp = context.response
        message: dict[str, Any] = {
            "role": "assistant",
            "content": resp.content if resp else "",
        }
        if resp and resp.reasoning_content:
            message["reasoning_content"] = resp.reasoning_content
        if context.tool_calls:
            tool_calls = []
            for idx, tc in enumerate(context.tool_calls):
                arguments = tc.arguments
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                tool_calls.append(
                    {
                        "id": f"call_{context.iteration}_{idx}",
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": arguments,
                        },
                    }
                )
            message["tool_calls"] = tool_calls
        raw_response = {
            "choices": [
                {
                    "message": message,
                    "finish_reason": resp.finish_reason if resp else None,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }
        if resp and resp.extra:
            openrouter_metadata = resp.extra.get("openrouter_metadata")
            if openrouter_metadata is not None:
                raw_response["openrouter_metadata"] = openrouter_metadata
            generation_id = resp.extra.get("openrouter_generation_id")
            if generation_id is not None:
                raw_response["openrouter_generation_id"] = generation_id
        return raw_response

    async def write_summary(
        self,
        *,
        success: bool | None = None,
        elapsed_s: float = 0.0,
        prepare_ms: float | None = None,
    ) -> None:
        await self._resolve_pending_llm_records()
        llm_summary = summarize_llm_latencies(
            a.get("data") for a in self._actions if a.get("action_type") == "llm_call"
        )
        summary = EvalTraceSummary(
            agent_id=self.agent_id,
            program_id=self.program_id,
            task_id=self.task_id,
            instance_id=self.instance_id,
            n_iterations=self._n_iterations,
            total_llm_ms=float(llm_summary["total_llm_ms"]),
            total_llm_wall_ms=float(llm_summary["total_llm_wall_ms"]),
            total_llm_call_time_ms=float(llm_summary["total_llm_call_time_ms"]),
            llm_call_time_count=int(llm_summary["llm_call_time_count"]),
            llm_timing_source=str(llm_summary["llm_timing_source"]),
            total_tool_ms=sum(self._tool_times.values()),
            total_tokens=self._total_tokens,
            tool_ms_by_name=self._tool_times,
            tool_timeouts=self._tool_timeouts,
            success=success,
            elapsed_s=elapsed_s,
            prepare_ms=prepare_ms,
        )
        self.add_record(summary.to_dict())
        self.close()


def inject_event_callbacks(agent: AgentLoop, hook: TraceCollectorHook) -> None:
    def emit(category: str, event: str, data: dict, iteration: int = 0) -> None:
        hook.emit_event(category, event, data, iteration=iteration)

    if hasattr(agent, "memory_consolidator") and hasattr(
        agent.memory_consolidator, "_event_callback"
    ):
        agent.memory_consolidator._event_callback = lambda cat, evt, d, si=0: emit(
            cat, evt, d, si
        )

    if (
        hasattr(agent, "context")
        and hasattr(agent.context, "skills")
        and hasattr(agent.context.skills, "_event_callback")
    ):
        agent.context.skills._event_callback = lambda cat, evt, d, si=0: emit(
            cat, evt, d, si
        )

    if hasattr(agent, "_mcp_event_callback"):
        agent._mcp_event_callback = lambda cat, evt, d: emit(cat, evt, d)

    if hasattr(agent, "sessions") and hasattr(agent.sessions, "_event_callback"):
        agent.sessions._event_callback = lambda cat, evt, d: emit(cat, evt, d)

    if hasattr(agent, "_event_callback"):
        agent._event_callback = lambda cat, evt, d: emit(cat, evt, d)


@dataclass
class SessionRunResult:
    content: str | None
    elapsed_s: float
    trace_file: Path | None = None
    session_key: str = ""
    session_manager: SessionManager | None = None
    stop_reason: str = "completed"
    error: str | None = None


class SessionRunner:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        model: str | None = None,
        max_iterations: int | None = None,
        context_window_tokens: int | None = None,
        max_tool_result_chars: int | None = None,
        mcp_servers: dict | None = None,
        extra_hooks: list[AgentHook] | None = None,
        exec_config: ExecToolConfig | None = None,
        malformed_retry_budget: int | None = None,
    ) -> None:
        self.provider = provider
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens or 65536
        self.max_tool_result_chars = max_tool_result_chars
        self.mcp_servers = mcp_servers or {}
        self.extra_hooks = extra_hooks or []
        self.exec_config = exec_config or ExecToolConfig()
        self.malformed_retry_budget = malformed_retry_budget

    @staticmethod
    def _scaffold_tools() -> list[str]:
        return [
            "read_file",
            "write_file",
            "edit_file",
            "list_dir",
            "exec",
            "web_search",
            "web_fetch",
            "message",
            "spawn",
        ]

    async def run(
        self,
        prompt: str,
        workspace: Path,
        *,
        tool_workspace: Path | None = None,
        project_workspace: Path | None = None,
        session_key: str,
        trace_file: Path,
        instance_id: str | None = None,
        channel: str = "cli",
        prepare_ms: float | None = None,
    ) -> SessionRunResult:
        workspace.mkdir(parents=True, exist_ok=True)
        iid = instance_id or session_key

        trace_hook = TraceCollectorHook(trace_file, iid, agent_id=iid, task_id=iid)

        metadata = {
            "type": "trace_metadata",
            "scaffold": "openclaw",
            "trace_format_version": 5,
            "mode": "collect",
            "model": self.model,
            "instance_id": iid,
            "session_key": session_key,
            "max_iterations": self.max_iterations,
            "scaffold_capabilities": {
                "tools": self._scaffold_tools(),
                "memory": True,
                "skills": True,
                "file_ops": "structured",
            },
        }
        trace_hook.add_record(metadata)

        bus = MessageBus()
        collector = ResultCollector(bus)
        session_manager = SessionManager(workspace)

        all_hooks: list[AgentHook] = [trace_hook, *self.extra_hooks]
        agent = AgentLoop(
            bus=bus,
            provider=self.provider,
            workspace=workspace,
            tool_workspace=tool_workspace,
            project_workspace=project_workspace,
            model=self.model,
            max_iterations=self.max_iterations,
            context_window_tokens=self.context_window_tokens,
            max_tool_result_chars=self.max_tool_result_chars,
            exec_config=self.exec_config,
            mcp_servers=self.mcp_servers,
            session_manager=session_manager,
            hooks=all_hooks,
            malformed_retry_budget=self.malformed_retry_budget,
        )

        inject_event_callbacks(agent, trace_hook)

        wall_start = time.monotonic()

        chat_id = session_key.split(":", 1)[-1] if ":" in session_key else session_key
        result_key = f"{channel}:{chat_id}"

        async with AsyncExitStack() as stack:
            await collector.start()
            stack.callback(collector.stop)

            agent_task = asyncio.create_task(agent.run())
            stack.callback(agent.stop)

            msg = InboundMessage(
                channel="system",
                sender_id="user",
                chat_id=f"{channel}:{chat_id}",
                content=prompt,
                session_key_override=session_key,
            )
            await bus.publish_inbound(msg)

            content = await collector.wait_for_result(result_key)

        elapsed_s = time.monotonic() - wall_start

        try:
            await asyncio.wait_for(agent_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

        outcomes = getattr(agent, "_last_run_outcomes", {})
        outcome = outcomes.get(session_key, {})
        stop_reason, error = _resolve_run_outcome(
            outcome=outcome,
            content=content,
            trace_file=trace_file,
        )

        await trace_hook.write_summary(
            success=stop_reason == "completed",
            elapsed_s=elapsed_s,
            prepare_ms=prepare_ms,
        )

        return SessionRunResult(
            content=content,
            elapsed_s=elapsed_s,
            trace_file=trace_file,
            session_key=session_key,
            session_manager=session_manager,
            stop_reason=stop_reason,
            error=error,
        )
