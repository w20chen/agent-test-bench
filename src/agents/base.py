from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from harness.trace_logger import TraceLogger

from llm_call import create_async_openai_client
from trace_collect.latency_metrics import (
    summarize_llm_latencies,
)

logger = logging.getLogger(__name__)


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


@dataclass(slots=True)
class ToolCallResult:
    id: str
    name: str
    arguments: str


@dataclass(slots=True)
class TraceAction:
    """A single replayable action in an agent trace (v4 format).

    Each action is one executable operation:
    - ``llm_call``: an LLM inference (input: messages_in; output: raw_response)
    - ``tool_exec``: a tool execution (input: tool_name+args; output: result)

    Multiple actions can share the same ``iteration`` value (e.g., one LLM call
    followed by parallel tool executions).
    """

    action_type: str  # "llm_call" | "tool_exec"
    action_id: str  # unique within trace, e.g. "llm_0", "tool_0_bash"
    agent_id: str = ""
    program_id: str = ""
    instance_id: str = ""
    iteration: int = 0
    ts_start: float = 0.0
    ts_end: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)
    type: str = "action"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "action_type": self.action_type,
            "action_id": self.action_id,
            "agent_id": self.agent_id,
            "program_id": self.program_id,
            "instance_id": self.instance_id,
            "iteration": self.iteration,
            "ts_start": self.ts_start,
            "ts_end": self.ts_end,
            "data": self.data,
        }


@dataclass(slots=True)
class LLMCallResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    llm_latency_ms: float
    raw_response: dict[str, Any]
    tool_calls: list[ToolCallResult] = field(default_factory=list)


class AgentBase(ABC):
    def __init__(
        self,
        agent_id: str,
        api_base: str,
        model: str,
        *,
        api_key: str = "EMPTY",
        request_timeout_s: float = 180.0,
        max_tool_output_chars: int = 8000,
    ) -> None:
        self.agent_id = agent_id
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.max_tool_output_chars = max_tool_output_chars
        self.actions: list[TraceAction] = []
        self.task_id: str = ""
        self.task_success: bool | None = None
        self.task_submission: str = ""
        self.task_exit_status: str | None = None
        self.task_error: str | None = None
        self._trace_logger: TraceLogger | None = None
        self.run_metadata: dict[str, Any] = {}
        self._client = create_async_openai_client(
            api_base=api_base,
            api_key=api_key,
            timeout=request_timeout_s,
        )

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_tool_output_chars:
            return text
        half = self.max_tool_output_chars // 2
        return (
            text[:half]
            + f"\n[... truncated {len(text) - self.max_tool_output_chars} chars ...]\n"
            + text[-half:]
        )

    _RETRY_DELAYS = (2, 4, 8)
    _TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMCallResult:
        """Call the OpenAI-compatible endpoint with retry on transient errors.

        OpenClaw uses nanobot's LLMProvider._run_with_retry() for retry. This
        retry logic covers any future AgentBase subclass that calls
        _call_llm() directly.
        """
        import asyncio as _asyncio
        from openai import APIStatusError, APIConnectionError, APITimeoutError

        extra_body: dict[str, Any] = {
            "program_id": self.agent_id,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            extra_body=extra_body,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        last_exc: BaseException | None = None
        for attempt, delay in enumerate((*self._RETRY_DELAYS, None)):
            try:
                started = time.monotonic()
                response = await self._client.chat.completions.create(**kwargs)
                elapsed_ms = (time.monotonic() - started) * 1000
                if not response.choices:
                    raise RuntimeError(
                        f"LLM returned empty choices: {response.model_dump()}"
                    )
                break
            except (APIConnectionError, APITimeoutError) as exc:
                last_exc = exc
                if delay is None:
                    raise
                logger.warning(
                    "LLM transient error (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1,
                    len(self._RETRY_DELAYS),
                    delay,
                    exc,
                )
                await _asyncio.sleep(delay)
            except APIStatusError as exc:
                last_exc = exc
                if exc.status_code not in self._TRANSIENT_STATUS_CODES or delay is None:
                    raise
                logger.warning(
                    "LLM %d error (attempt %d/%d), retrying in %ds: %s",
                    exc.status_code,
                    attempt + 1,
                    len(self._RETRY_DELAYS),
                    delay,
                    exc,
                )
                await _asyncio.sleep(delay)
        else:
            raise RuntimeError(
                f"LLM call failed after {len(self._RETRY_DELAYS)} retries"
            ) from last_exc

        message = response.choices[0].message
        usage = getattr(response, "usage", None)

        # Extract structured tool calls if present
        parsed_tool_calls: list[ToolCallResult] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                parsed_tool_calls.append(
                    ToolCallResult(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    )
                )

        return LLMCallResult(
            content=_message_content_to_text(message.content),
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            llm_latency_ms=elapsed_ms,
            raw_response=response.model_dump(),
            tool_calls=parsed_tool_calls,
        )

    async def prepare(self, task: dict[str, Any]) -> None:
        """Prepare the agent's environment before the main loop.

        Called during the setup phase so that expensive operations
        (container creation, dependency installation) complete before
        all agents start competing for the GPU simultaneously.

        The default implementation is a no-op.  Subclasses that need
        heavyweight setup (e.g. Podman containers) should override this.
        """

    def _emit_event(
        self,
        category: str,
        event: str,
        data: dict[str, Any],
        *,
        iteration: int = 0,
        ts: float | None = None,
    ) -> None:
        if self._trace_logger is not None:
            self._trace_logger.log_event(
                self.agent_id,
                category,
                event,
                data,
                iteration=iteration,
                ts=ts,
            )

    def _emit_action(self, action: TraceAction) -> None:
        if self.run_metadata:
            action.data.update(self.run_metadata)
        self.actions.append(action)
        if self._trace_logger is not None:
            self._trace_logger.log_trace_action(self.agent_id, action)

    @abstractmethod
    async def run(self, task: dict[str, Any]) -> bool:
        """Run the agent on one task."""

    def get_trace(self) -> list[dict[str, Any]]:
        return [a.to_dict() for a in self.actions]

    def summary(self) -> dict[str, Any]:
        # ``.get(key, 0)`` returns None (not 0) when the key exists with a
        # None value, so use ``(... or 0)`` to coerce missing-or-None fields
        # to a numeric default.
        llm_records = [a.data for a in self.actions if a.action_type == "llm_call"]
        llm_summary = summarize_llm_latencies(llm_records)
        total_tool_ms = sum(
            (a.data.get("duration_ms") or 0)
            for a in self.actions
            if a.action_type == "tool_exec"
        )
        total_tokens = sum(
            (a.data.get("prompt_tokens") or 0) + (a.data.get("completion_tokens") or 0)
            for a in self.actions
            if a.action_type == "llm_call"
        )
        tool_ms_by_name: dict[str, float] = {}
        tool_timeouts: dict[str, int] = {}
        for a in self.actions:
            if a.action_type != "tool_exec":
                continue
            tool_name = a.data.get("tool_name")
            if tool_name is None:
                continue
            tool_ms_by_name[tool_name] = tool_ms_by_name.get(tool_name, 0.0) + (
                a.data.get("duration_ms") or 0.0
            )
            if a.data.get("timeout"):
                tool_timeouts[tool_name] = tool_timeouts.get(tool_name, 0) + 1
        n_iterations = len({a.iteration for a in self.actions})
        return {
            "agent_id": self.agent_id,
            "program_id": self.agent_id,
            "task_id": self.task_id,
            "n_iterations": n_iterations,
            "total_llm_ms": llm_summary["total_llm_ms"],
            "total_llm_wall_ms": llm_summary["total_llm_wall_ms"],
            "total_llm_call_time_ms": llm_summary["total_llm_call_time_ms"],
            "llm_call_time_count": llm_summary["llm_call_time_count"],
            "llm_timing_source": llm_summary["llm_timing_source"],
            "total_tool_ms": total_tool_ms,
            "total_tokens": total_tokens,
            "tool_ms_by_name": tool_ms_by_name,
            "tool_timeouts": tool_timeouts,
            "success": self.task_success,
        }
