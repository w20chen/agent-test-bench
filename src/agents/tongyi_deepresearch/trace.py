"""Adapter-layer trace infrastructure for the Tongyi-DeepResearch scaffold.

This module provides drop-in replacements for vendor's ``OpenAI`` client
symbol and tool classes so the runner can emit canonical v5 TraceAction
records without patching vendor source. The vendor code sees a duck-typed
non-streaming response; internally we use streaming to capture
TTFT/TPOT, 429/503 transport-layer retries, and model-level empty-response
retries.

Key concepts:

- ``logical_turn_id``: one UUID per ``.create()`` call on the shim. All
  transport-layer retries within that call share the UUID. If vendor retries
  the previous turn (because it got an empty content response), the
  *next* turn emits a TraceAction with ``retry_of`` pointing at the prior
  action_id — even though logical_turn_id changes, giving the simulator
  both granular retry linkage and per-turn aggregation.

- ``transport_retry: True``: 429/503/connection retries inside a single
  ``.create()`` call. Distinct from model-level retries which appear as
  separate turns with ``retry_of`` set.

No vendor patches are needed for these hooks. Runner uses a
``contextmanager`` to monkey-patch ``vendor.OpenAI`` and ``vendor.TOOL_CLASS``
for the duration of a single task, then restores module state.
"""

from __future__ import annotations

import inspect
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import openai
import tiktoken

from agents.base import TraceAction

logger = logging.getLogger(__name__)

EmitFn = Callable[[TraceAction], None]
IterationProvider = Callable[[], int]

# openai error types — import lazily so users without openai installed can still
# import this module for static analysis. The shim requires openai at runtime.
_TRANSPORT_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.APIStatusError,
)


class RateLimitExhausted(Exception):
    """Raised by the shim when 429/503 backoff retries are exhausted."""


def _new_turn_id() -> str:
    return uuid.uuid4().hex


def _approx_token_count(text: str) -> int:
    """Fallback token count via tiktoken cl100k_base when stream usage is absent.

    Approximate for non-GPT tokenizers but sufficient for TPOT timing analysis
    (scheduling cares about per-token latency, not exact cost accounting).
    """
    if not text:
        return 0
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _api_status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def _is_retryable_transport_error(exc: Exception) -> bool:
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        status_code = _api_status_code(exc)
        return status_code == 429 or (isinstance(status_code, int) and 500 <= status_code < 600)
    return False


@dataclass
class _DuckMessage:
    content: str


@dataclass
class _DuckChoice:
    message: _DuckMessage
    finish_reason: str | None = None
    index: int = 0


@dataclass
class _DuckChatResponse:
    """Duck-typed response matching the subset of openai ChatCompletion vendor uses."""

    choices: list[_DuckChoice] = field(default_factory=list)
    model: str = ""

    @classmethod
    def from_content(cls, content: str, model: str, finish_reason: str | None = None) -> "_DuckChatResponse":
        return cls(
            choices=[_DuckChoice(message=_DuckMessage(content=content), finish_reason=finish_reason)],
            model=model,
        )


class _ChatCompletions:
    """Implements ``chat.completions.create`` with streaming + trace emit."""

    def __init__(self, parent: "TracedStreamingOpenAI") -> None:
        self._p = parent

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool | None = None,  # ignored; shim always streams
        stream_options: dict[str, Any] | None = None,  # ignored; shim forces include_usage
        **kwargs: Any,
    ) -> _DuckChatResponse:
        return self._p._do_streaming_create(model=model, messages=messages, **kwargs)


class _Chat:
    def __init__(self, parent: "TracedStreamingOpenAI") -> None:
        self.completions = _ChatCompletions(parent)


class TracedStreamingOpenAI:
    """Drop-in replacement for ``openai.OpenAI`` that streams + emits TraceActions.

    Vendor code constructs this via ``OpenAI(api_key=..., base_url=..., timeout=...)``
    and calls ``.chat.completions.create(...)``. Internally we:

    1. Force ``stream=True`` with ``stream_options={'include_usage': True}``.
    2. Capture TTFT at the first non-empty content chunk.
    3. Aggregate content chunks + collect ``usage`` from the terminal chunk.
    4. On 429/503/transport errors, exponential backoff (max 3 retries) with
       each retry emitting a separate TraceAction tagged ``transport_retry: True``.
    5. Build a v5 TraceAction with canonical LLM timing fields, emit it.
    6. Return a duck-typed ``_DuckChatResponse`` vendor can read unchanged.

    Instance-level state tracks the previous action_id for model-level retry
    linkage (``retry_of``): if the last ``.create()`` returned empty content,
    vendor's outer loop will call ``.create()`` again; that next call sets
    ``retry_of`` to point at the prior action_id.
    """

    # Model-retry coupling — if the last call returned empty content, the next
    # call is treated as a model-layer retry of it. Counters and IDs are
    # instance-scoped so concurrent runners don't collide.
    def __init__(
        self,
        api_key: str = "EMPTY",
        base_url: str | None = None,
        timeout: float | None = None,
        *,
        emit_fn: EmitFn,
        agent_id: str,
        instance_id: str,
        iteration_provider: IterationProvider,
        max_transport_retries: int = 3,
        backoff_base_s: float = 1.0,
        backoff_jitter_max_s: float = 1.0,
        call_counter: list[int] | None = None,
        retry_state: dict[str, Any] | None = None,
        llm_iteration_start_fn: Callable[[], int] | None = None,
    ) -> None:
        self._real = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        self._emit = emit_fn
        self._agent_id = agent_id
        self._instance_id = instance_id
        self._iteration_provider = iteration_provider
        self._max_transport_retries = max_transport_retries
        self._backoff_base_s = backoff_base_s
        self._backoff_jitter_max_s = backoff_jitter_max_s
        self._llm_iteration_start_fn = llm_iteration_start_fn

        # Runner-injected shared state so action_ids + retry-linkage survive
        # vendor constructing a fresh OpenAI client per call_server invocation.
        # Each call to vendor's call_server builds a new TracedStreamingOpenAI,
        # but the counter and retry_state refer to the same mutable container
        # owned by _patched_vendor(), keeping action_ids monotonic + retry_of
        # linkage correct across the entire task.
        self._call_counter = call_counter if call_counter is not None else [0]
        self._retry_state = (
            retry_state if retry_state is not None
            else {"last_action_id": None, "last_was_empty": False}
        )

        self.chat = _Chat(self)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _next_action_id(self) -> str:
        self._call_counter[0] += 1
        return f"llm_{self._call_counter[0]}"

    def _do_streaming_create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> _DuckChatResponse:
        logical_turn_id = _new_turn_id()
        action_id = self._next_action_id()
        if self._llm_iteration_start_fn is not None:
            iteration = self._llm_iteration_start_fn()
        else:
            iteration = self._iteration_provider()

        # Model-layer retry detection: vendor retries when prior content was empty
        retry_of: str | None = None
        if self._retry_state["last_was_empty"] and self._retry_state["last_action_id"] is not None:
            retry_of = self._retry_state["last_action_id"]

        last_err: Exception | None = None
        for attempt in range(self._max_transport_retries + 1):
            try:
                result = self._stream_once(
                    model=model,
                    messages=messages,
                    action_id=action_id,
                    iteration=iteration,
                    logical_turn_id=logical_turn_id,
                    retry_of=retry_of,
                    **kwargs,
                )
                # Update retry-linkage state for next call
                self._retry_state["last_action_id"] = action_id
                self._retry_state["last_was_empty"] = not (result.choices[0].message.content or "").strip()
                return result
            except _TRANSPORT_ERRORS as exc:
                if not _is_retryable_transport_error(exc):
                    raise
                last_err = exc
                if attempt < self._max_transport_retries:
                    # Emit a TraceAction for the failed attempt, tagged transport_retry
                    self._emit_transport_retry_action(
                        action_id=f"{action_id}_transport_retry_{attempt}",
                        iteration=iteration,
                        logical_turn_id=logical_turn_id,
                        parent_action_id=action_id,
                        error=exc,
                    )
                    sleep_s = (
                        self._backoff_base_s * (2 ** attempt)
                    ) + random.uniform(0.0, self._backoff_jitter_max_s)
                    logger.warning(
                        "TracedStreamingOpenAI transport retry %d/%d on %s; sleeping %.2fs",
                        attempt + 1, self._max_transport_retries,
                        type(exc).__name__, sleep_s,
                    )
                    time.sleep(sleep_s)
                    continue
                break  # exhausted
        # Exhausted — emit a terminal error TraceAction and raise
        self._emit_transport_retry_action(
            action_id=f"{action_id}_transport_exhausted",
            iteration=iteration,
            logical_turn_id=logical_turn_id,
            parent_action_id=action_id,
            error=last_err,
            messages_in=messages,
            terminal=True,
        )
        raise RateLimitExhausted(f"Transport retries exhausted after {self._max_transport_retries + 1} attempts: {last_err!r}")

    def _stream_once(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        action_id: str,
        iteration: int,
        logical_turn_id: str,
        retry_of: str | None,
        **kwargs: Any,
    ) -> _DuckChatResponse:
        ts_start = time.time()
        mono_start = time.monotonic()
        first_token_mono: float | None = None
        content_parts: list[str] = []
        usage: dict[str, Any] | None = None
        finish_reason: str | None = None

        stream = self._real.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            **kwargs,
        )
        for chunk in stream:
            # Terminal usage chunk has no choices
            if not getattr(chunk, "choices", None):
                maybe_usage = getattr(chunk, "usage", None)
                if maybe_usage is not None:
                    usage = maybe_usage.model_dump() if hasattr(maybe_usage, "model_dump") else dict(maybe_usage)
                continue
            for choice in chunk.choices:
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                text = getattr(delta, "content", None) or ""
                if text:
                    if first_token_mono is None:
                        first_token_mono = time.monotonic()
                    content_parts.append(text)
                fr = getattr(choice, "finish_reason", None)
                if fr:
                    finish_reason = fr

        mono_end = time.monotonic()
        ts_end = time.time()
        elapsed_ms = (mono_end - mono_start) * 1000.0
        content = "".join(content_parts)

        prompt_tokens = int((usage or {}).get("prompt_tokens", 0) or 0)
        completion_tokens = int((usage or {}).get("completion_tokens", 0) or 0)
        if completion_tokens == 0 and content:
            # Tokenizer fallback per Critic v6 AC#4 derivation spec
            completion_tokens = _approx_token_count(content)

        ttft_ms: float | None = None
        tpot_ms: float | None = None
        if first_token_mono is not None:
            ttft_ms = (first_token_mono - mono_start) * 1000.0
            if completion_tokens > 1 and ttft_ms is not None:
                tpot_ms = max(0.0, (elapsed_ms - ttft_ms) / (completion_tokens - 1))

        data: dict[str, Any] = {
            "messages_in": messages,
            "content": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "llm_wall_latency_ms": elapsed_ms,
            "llm_latency_ms": elapsed_ms,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "finish_reason": finish_reason,
            "logical_turn_id": logical_turn_id,
        }
        if retry_of is not None:
            data["retry_of"] = retry_of

        self._emit(TraceAction(
            action_type="llm_call",
            action_id=action_id,
            agent_id=self._agent_id,
            instance_id=self._instance_id,
            iteration=iteration,
            ts_start=ts_start,
            ts_end=ts_end,
            data=data,
        ))

        return _DuckChatResponse.from_content(content=content, model=model, finish_reason=finish_reason)

    def _emit_transport_retry_action(
        self,
        *,
        action_id: str,
        iteration: int,
        logical_turn_id: str,
        parent_action_id: str,
        error: Exception | None,
        messages_in: list[dict[str, Any]] | None = None,
        terminal: bool = False,
    ) -> None:
        ts = time.time()
        data = {
            "transport_retry": True,
            "transport_retry_terminal": terminal,
            "parent_action_id": parent_action_id,
            "logical_turn_id": logical_turn_id,
            "error": f"{type(error).__name__}: {error}" if error else None,
        }
        if terminal and messages_in is not None:
            data["messages_in"] = messages_in
        self._emit(TraceAction(
            action_type="llm_call",
            action_id=action_id,
            agent_id=self._agent_id,
            instance_id=self._instance_id,
            iteration=iteration,
            ts_start=ts,
            ts_end=ts,
            data=data,
        ))


# ----------------------------------------------------------------------
# Tool wrappers
# ----------------------------------------------------------------------


def make_traced_tool_class(
    base_cls: type,
    *,
    emit_fn: EmitFn,
    agent_id: str,
    instance_id: str,
    iteration_provider: IterationProvider,
    action_counter: list[int],
):
    """Return a subclass of ``base_cls`` whose ``.call`` emits a TraceAction.

    Uses a factory rather than inheritance so we can share call-counter state
    and per-runner context across multiple tool subclasses without globals.
    """

    def _safe_tool_args(params: Any) -> Any:
        # Vendor's custom_call_tool aliases tool_args as tool_args["params"]
        # (self-reference), which breaks json.dumps serialization inside
        # the trace logger. Strip the self-ref before recording.
        if isinstance(params, dict):
            return {
                k: v for k, v in params.items()
                if not (k == "params" and v is params)
            }
        return {"raw": str(params)}

    def _next_tool_action_id() -> str:
        action_counter[0] += 1
        return f"tool_{action_counter[0]}"

    def _emit_tool_action(
        *,
        tool_name: str,
        action_id: str,
        params: Any,
        result: Any,
        error: str | None,
        ts_start: float,
        mono_start: float,
    ) -> None:
        mono_end = time.monotonic()
        ts_end = time.time()
        duration_ms = (mono_end - mono_start) * 1000.0
        emit_fn(TraceAction(
            action_type="tool_exec",
            action_id=action_id,
            agent_id=agent_id,
            instance_id=instance_id,
            iteration=iteration_provider(),
            ts_start=ts_start,
            ts_end=ts_end,
            data={
                "tool_name": tool_name,
                "tool_args": _safe_tool_args(params),
                "tool_result": result if isinstance(result, str) else str(result),
                "duration_ms": duration_ms,
                "success": error is None,
                "error": error,
                "logical_turn_id": None,  # Tools sit outside the LLM-turn scope
            },
        ))

    if inspect.iscoroutinefunction(getattr(base_cls, "call", None)):

        class _Traced(base_cls):  # type: ignore[misc, valid-type]
            async def call(self, params: Any, **kwargs: Any) -> Any:  # noqa: ANN401
                action_id = _next_tool_action_id()
                ts_start = time.time()
                mono_start = time.monotonic()
                try:
                    result = await super().call(params, **kwargs)
                    error: str | None = None
                except Exception as exc:  # noqa: BLE001
                    result = f"Error: {exc}"
                    error = str(exc)
                _emit_tool_action(
                    tool_name=getattr(self, "name", base_cls.__name__.lower()),
                    action_id=action_id,
                    params=params,
                    result=result,
                    error=error,
                    ts_start=ts_start,
                    mono_start=mono_start,
                )
                return result

    else:

        class _Traced(base_cls):  # type: ignore[misc, valid-type]
            def call(self, params: Any, **kwargs: Any) -> Any:  # noqa: ANN401
                action_id = _next_tool_action_id()
                ts_start = time.time()
                mono_start = time.monotonic()
                try:
                    result = super().call(params, **kwargs)
                    error: str | None = None
                except Exception as exc:  # noqa: BLE001
                    result = f"Error: {exc}"
                    error = str(exc)
                _emit_tool_action(
                    tool_name=getattr(self, "name", base_cls.__name__.lower()),
                    action_id=action_id,
                    params=params,
                    result=result,
                    error=error,
                    ts_start=ts_start,
                    mono_start=mono_start,
                )
                return result

    _Traced.__name__ = f"Traced{base_cls.__name__}"
    return _Traced
