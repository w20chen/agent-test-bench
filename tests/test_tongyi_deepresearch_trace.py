"""Unit tests for the Tongyi-DeepResearch adapter-layer trace infrastructure.

Covers US-D1 ACs: streaming shim duck-typing, TTFT/TPOT capture, transport
retry with shared logical_turn_id, tool-exec TraceAction emission.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import openai
import pytest

from agents.base import TraceAction
from agents.tongyi_deepresearch.trace import (
    RateLimitExhausted,
    TracedStreamingOpenAI,
    make_traced_tool_class,
)


# ----------------------------------------------------------------------
# Fake OpenAI stream helpers
# ----------------------------------------------------------------------


def _delta_chunk(content: str, finish_reason: str | None = None):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            delta=SimpleNamespace(content=content),
            finish_reason=finish_reason,
            index=0,
        )],
        usage=None,
    )


def _usage_chunk(prompt_tokens: int, completion_tokens: int):
    class _Usage:
        def __init__(self, p, c):
            self.prompt_tokens = p
            self.completion_tokens = c

        def model_dump(self):
            return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens}

    return SimpleNamespace(choices=[], usage=_Usage(prompt_tokens, completion_tokens))


class _FakeOpenAIClient:
    """Minimal stand-in for openai.OpenAI; chat.completions.create yields chunks."""

    def __init__(self, *_a, script_factory=None, **_k) -> None:
        # script_factory is a zero-arg callable returning an iterable of chunks
        self._script_factory = script_factory
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model, messages, stream, stream_options, **kwargs):
        # The shim always passes stream=True; we honor it.
        assert stream is True
        return iter(self._script_factory())


@pytest.fixture
def capture_emits():
    captured: list[TraceAction] = []
    return captured, captured.append


@pytest.fixture
def iter_provider():
    counter = {"i": 0}

    def _get():
        return counter["i"]

    def _advance():
        counter["i"] += 1

    return _get, _advance


def _install_fake_client(script_factory):
    """Context-manager: patch openai.OpenAI constructor in trace module."""
    def _factory(*a, **k):
        return _FakeOpenAIClient(script_factory=script_factory)

    return patch("agents.tongyi_deepresearch.trace.openai.OpenAI", side_effect=_factory)


class _FakeStatusError(openai.APIStatusError):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.response = SimpleNamespace(status_code=status_code)
        self.body = None
        Exception.__init__(self, f"status {status_code}")


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_streaming_shim_returns_duck_typed_response(capture_emits, iter_provider):
    """AC (a): shim yields .choices[0].message.content matching stream content."""
    captured, emit = capture_emits
    get_iter, _ = iter_provider

    def script():
        yield _delta_chunk("Hello ")
        yield _delta_chunk("world")
        yield _delta_chunk("", finish_reason="stop")
        yield _usage_chunk(prompt_tokens=5, completion_tokens=2)

    with _install_fake_client(script):
        shim = TracedStreamingOpenAI(
            api_key="x", base_url=None,
            emit_fn=emit, agent_id="A", instance_id="I",
            iteration_provider=get_iter,
        )
        resp = shim.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])

    assert resp.choices[0].message.content == "Hello world"
    assert resp.model == "m"


def test_streaming_shim_emits_single_llm_call_with_ttft(capture_emits, iter_provider):
    """AC (b): successful call emits exactly 1 llm_call TraceAction with ttft_ms non-None."""
    captured, emit = capture_emits
    get_iter, _ = iter_provider

    def script():
        yield _delta_chunk("foo")
        yield _delta_chunk("bar", finish_reason="stop")
        yield _usage_chunk(prompt_tokens=10, completion_tokens=2)

    with _install_fake_client(script):
        shim = TracedStreamingOpenAI(
            api_key="x", base_url=None,
            emit_fn=emit, agent_id="A", instance_id="I",
            iteration_provider=get_iter,
        )
        shim.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])

    llm_calls = [a for a in captured if a.action_type == "llm_call" and not a.data.get("transport_retry")]
    assert len(llm_calls) == 1
    act = llm_calls[0]
    assert act.data["ttft_ms"] is not None
    assert act.data["ttft_ms"] >= 0.0
    assert act.data["content"] == "foobar"
    assert act.data["prompt_tokens"] == 10
    assert act.data["completion_tokens"] == 2
    assert act.data["logical_turn_id"]
    assert act.data.get("retry_of") is None
    # tpot_ms is defined since completion_tokens > 1
    assert act.data["tpot_ms"] is not None


def test_transport_retry_emits_tagged_action_sharing_turn_id(capture_emits, iter_provider):
    """AC (c): 429 then success -> 2 TraceActions, 1 tagged transport_retry, shared logical_turn_id."""
    captured, emit = capture_emits
    get_iter, _ = iter_provider

    call_count = {"n": 0}

    def _create_that_fails_then_succeeds(*, model, messages, stream, stream_options, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise openai.APIConnectionError(request=None)
        return iter([
            _delta_chunk("ok", finish_reason="stop"),
            _usage_chunk(3, 1),
        ])

    class _FlakyClient:
        def __init__(self, *_a, **_k) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create_that_fails_then_succeeds))

    with patch("agents.tongyi_deepresearch.trace.openai.OpenAI", side_effect=_FlakyClient):
        shim = TracedStreamingOpenAI(
            api_key="x", base_url=None,
            emit_fn=emit, agent_id="A", instance_id="I",
            iteration_provider=get_iter,
            backoff_base_s=0.0,  # no real sleep
        )
        resp = shim.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])

    assert resp.choices[0].message.content == "ok"
    # Expect 2 emits: the transport-retry action + the successful llm_call action
    assert len(captured) == 2
    retry_action = next(a for a in captured if a.data.get("transport_retry"))
    success_action = next(a for a in captured if not a.data.get("transport_retry"))
    assert retry_action.data["logical_turn_id"] == success_action.data["logical_turn_id"]
    assert retry_action.data["parent_action_id"] == success_action.action_id
    assert "APIConnectionError" in retry_action.data["error"]


def test_non_retryable_api_status_error_is_raised_without_transport_retry(
    capture_emits,
    iter_provider,
):
    captured, emit = capture_emits
    get_iter, _ = iter_provider

    def _always_unauthorized(*, model, messages, stream, stream_options, **kwargs):
        raise _FakeStatusError(401)

    class _UnauthorizedClient:
        def __init__(self, *_a, **_k) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=_always_unauthorized))

    with patch("agents.tongyi_deepresearch.trace.openai.OpenAI", side_effect=_UnauthorizedClient):
        shim = TracedStreamingOpenAI(
            api_key="x",
            base_url=None,
            emit_fn=emit,
            agent_id="A",
            instance_id="I",
            iteration_provider=get_iter,
            backoff_base_s=0.0,
            backoff_jitter_max_s=0.0,
        )
        with pytest.raises(_FakeStatusError):
            shim.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])

    assert captured == []


def test_transport_retry_backoff_includes_jitter(capture_emits, iter_provider):
    captured, emit = capture_emits
    get_iter, _ = iter_provider

    call_count = {"n": 0}

    def _fails_then_succeeds(*, model, messages, stream, stream_options, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise _FakeStatusError(503)
        return iter([
            _delta_chunk("ok", finish_reason="stop"),
            _usage_chunk(3, 1),
        ])

    class _RetryableClient:
        def __init__(self, *_a, **_k) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=_fails_then_succeeds))

    with (
        patch("agents.tongyi_deepresearch.trace.openai.OpenAI", side_effect=_RetryableClient),
        patch("agents.tongyi_deepresearch.trace.random.uniform", return_value=0.25),
        patch("agents.tongyi_deepresearch.trace.time.sleep") as mock_sleep,
    ):
        shim = TracedStreamingOpenAI(
            api_key="x",
            base_url=None,
            emit_fn=emit,
            agent_id="A",
            instance_id="I",
            iteration_provider=get_iter,
            backoff_base_s=0.5,
            backoff_jitter_max_s=1.0,
        )
        resp = shim.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])

    assert resp.choices[0].message.content == "ok"
    mock_sleep.assert_called_once_with(0.75)
    retry_action = next(a for a in captured if a.data.get("transport_retry"))
    assert "status 503" in retry_action.data["error"]


def test_transport_exhaustion_raises_rate_limit_exhausted(capture_emits, iter_provider):
    """Transport retries exhausted -> RateLimitExhausted, terminal TraceAction emitted."""
    captured, emit = capture_emits
    get_iter, _ = iter_provider

    def _always_fail(*, model, messages, stream, stream_options, **kwargs):
        raise openai.APIConnectionError(request=None)

    class _BrokenClient:
        def __init__(self, *_a, **_k) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=_always_fail))

    with patch("agents.tongyi_deepresearch.trace.openai.OpenAI", side_effect=_BrokenClient):
        shim = TracedStreamingOpenAI(
            api_key="x", base_url=None,
            emit_fn=emit, agent_id="A", instance_id="I",
            iteration_provider=get_iter,
            max_transport_retries=2,
            backoff_base_s=0.0,
        )
        with pytest.raises(RateLimitExhausted):
            shim.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])

    # max_transport_retries=2 -> 3 total attempts; 2 non-terminal retry actions + 1 terminal
    retry_actions = [a for a in captured if a.data.get("transport_retry")]
    assert len(retry_actions) == 3
    assert sum(1 for a in retry_actions if a.data.get("transport_retry_terminal")) == 1
    terminal = next(a for a in retry_actions if a.data.get("transport_retry_terminal"))
    assert terminal.data["messages_in"] == [{"role": "user", "content": "q"}]


def test_model_layer_retry_detection_sets_retry_of(capture_emits, iter_provider):
    """Empty prior content -> next call's TraceAction has retry_of = prior action_id."""
    captured, emit = capture_emits
    get_iter, _ = iter_provider

    scripts = [
        # First call returns empty content
        lambda: iter([_delta_chunk("", finish_reason="stop"), _usage_chunk(5, 0)]),
        # Second call returns valid content — should be tagged retry_of = llm_1
        lambda: iter([_delta_chunk("recovered", finish_reason="stop"), _usage_chunk(5, 1)]),
    ]
    script_idx = {"i": 0}

    def _sequenced_create(*, model, messages, stream, stream_options, **kwargs):
        s = scripts[script_idx["i"]]
        script_idx["i"] += 1
        return s()

    class _SeqClient:
        def __init__(self, *_a, **_k) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=_sequenced_create))

    with patch("agents.tongyi_deepresearch.trace.openai.OpenAI", side_effect=_SeqClient):
        shim = TracedStreamingOpenAI(
            api_key="x", base_url=None,
            emit_fn=emit, agent_id="A", instance_id="I",
            iteration_provider=get_iter,
        )
        shim.chat.completions.create(model="m", messages=[{"role": "user", "content": "q1"}])
        shim.chat.completions.create(model="m", messages=[{"role": "user", "content": "q2"}])

    llm_actions = [a for a in captured if a.action_type == "llm_call"]
    assert len(llm_actions) == 2
    first, second = llm_actions
    assert first.data.get("retry_of") is None
    assert second.data.get("retry_of") == first.action_id
    # Distinct logical_turn_id (model-level retry = new turn per R3 Principle #2)
    assert first.data["logical_turn_id"] != second.data["logical_turn_id"]


def test_traced_tool_emits_tool_exec_with_canonical_keys(capture_emits, iter_provider):
    """AC (d): tool wrapper emits tool_exec TraceAction with tool_name/tool_args/tool_result/duration_ms."""
    captured, emit = capture_emits
    get_iter, _ = iter_provider

    class _DummyTool:
        name = "search"

        def call(self, params, **kwargs):
            return f"result for query={params.get('query')}"

    counter: list[int] = [0]
    TracedDummy = make_traced_tool_class(
        _DummyTool,
        emit_fn=emit,
        agent_id="A",
        instance_id="I",
        iteration_provider=get_iter,
        action_counter=counter,
    )

    tool = TracedDummy()
    out = tool.call({"query": "Python asyncio"})

    assert out.startswith("result for")
    tool_actions = [a for a in captured if a.action_type == "tool_exec"]
    assert len(tool_actions) == 1
    act = tool_actions[0]
    assert act.data["tool_name"] == "search"
    assert act.data["tool_args"] == {"query": "Python asyncio"}
    assert act.data["tool_result"].startswith("result for")
    assert act.data["duration_ms"] >= 0.0
    assert act.data["success"] is True
    assert act.data["error"] is None


def test_traced_tool_captures_exceptions_without_reraising(capture_emits, iter_provider):
    """Tool .call() raising -> TraceAction with error populated, result contains Error prefix."""
    captured, emit = capture_emits
    get_iter, _ = iter_provider

    class _BrokenTool:
        name = "visit"

        def call(self, params, **kwargs):
            raise ValueError("boom")

    TracedBroken = make_traced_tool_class(
        _BrokenTool,
        emit_fn=emit,
        agent_id="A",
        instance_id="I",
        iteration_provider=get_iter,
        action_counter=[0],
    )

    tool = TracedBroken()
    result = tool.call({"url": "https://example.com"})

    assert result.startswith("Error:")
    act = next(a for a in captured if a.action_type == "tool_exec")
    assert act.data["success"] is False
    assert act.data["error"] == "boom"
    assert act.data["tool_name"] == "visit"


@pytest.mark.asyncio
async def test_traced_async_tool_preserves_async_result_and_emits_success(
    capture_emits,
    iter_provider,
):
    """Async vendored tools should still trace real results and wall time."""
    captured, emit = capture_emits
    get_iter, _ = iter_provider

    class _AsyncTool:
        name = "parse_file"

        async def call(self, params, **kwargs):
            return ["parsed page"]

    TracedAsync = make_traced_tool_class(
        _AsyncTool,
        emit_fn=emit,
        agent_id="A",
        instance_id="I",
        iteration_provider=get_iter,
        action_counter=[0],
    )

    tool = TracedAsync()
    result = await tool.call({"files": ["paper.pdf"]}, file_root_path="/tmp")

    assert result == ["parsed page"]
    act = next(a for a in captured if a.action_type == "tool_exec")
    assert act.data["tool_name"] == "parse_file"
    assert act.data["tool_args"] == {"files": ["paper.pdf"]}
    assert act.data["tool_result"] == "['parsed page']"
    assert act.data["success"] is True
    assert act.data["error"] is None
