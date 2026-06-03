from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from llm_call.openclaw import UnifiedProvider


class _FakeRawResponse:
    def __init__(self, parsed: Any, headers: dict[str, str]) -> None:
        self._parsed = parsed
        self.headers = headers

    def parse(self) -> Any:
        return self._parsed


class _FakeDelta:
    def __init__(self, content: str | None) -> None:
        self.content = content
        self.tool_calls = None


class _FakeChoice:
    def __init__(self, content: str | None, finish_reason: str | None = None) -> None:
        self.delta = _FakeDelta(content)
        self.finish_reason = finish_reason


class _FakeChunk:
    def __init__(
        self,
        content: str | None,
        *,
        finish_reason: str | None = None,
        usage: Any = None,
    ) -> None:
        self.choices = [_FakeChoice(content, finish_reason)]
        self.usage = usage

    def model_dump(self) -> dict[str, Any]:
        usage = None
        if self.usage is not None:
            usage = {
                "prompt_tokens": getattr(self.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(self.usage, "completion_tokens", 0),
                "total_tokens": getattr(self.usage, "total_tokens", 0),
            }
        return {
            "choices": [
                {
                    "delta": {"content": self.choices[0].delta.content},
                    "finish_reason": self.choices[0].finish_reason,
                }
            ],
            "usage": usage,
        }


class _FakeAsyncStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = list(chunks)
        self._index = 0

    def __aiter__(self) -> _FakeAsyncStream:
        return self

    async def __anext__(self) -> Any:
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


class _FakeStreamingRawResponse:
    def __init__(self, stream: _FakeAsyncStream, headers: dict[str, str]) -> None:
        self._stream = stream
        self.headers = headers

    async def parse(self) -> _FakeAsyncStream:
        return self._stream


class _FakeStreamingContext:
    def __init__(self, raw_response: _FakeStreamingRawResponse) -> None:
        self._raw_response = raw_response

    async def __aenter__(self) -> _FakeStreamingRawResponse:
        return self._raw_response

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeCompletions:
    def __init__(
        self,
        *,
        raw_response: _FakeRawResponse | None = None,
        streaming_context: _FakeStreamingContext | None = None,
    ) -> None:
        self._raw_response = raw_response
        self._streaming_context = streaming_context
        self.raw_kwargs: dict[str, Any] | None = None
        self.stream_kwargs: dict[str, Any] | None = None
        self.with_raw_response = SimpleNamespace(create=self.create_raw)
        self.with_streaming_response = SimpleNamespace(create=self.create_streaming)

    async def create_raw(self, **kwargs: Any) -> _FakeRawResponse:
        self.raw_kwargs = kwargs
        assert self._raw_response is not None
        return self._raw_response

    def create_streaming(self, **kwargs: Any) -> _FakeStreamingContext:
        self.stream_kwargs = kwargs
        assert self._streaming_context is not None
        return self._streaming_context


def _make_provider(*, completions: _FakeCompletions) -> UnifiedProvider:
    provider = UnifiedProvider(
        api_key="test-key",
        api_base="https://openrouter.ai/api/v1",
        default_model="z-ai/glm-5.1",
    )
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return provider


def test_unified_provider_reads_llm_timeout_env(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_create_async_openai_client(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setenv("OPENCLAW_LLM_TIMEOUT_S", "1800")
    monkeypatch.setattr(
        "llm_call.openclaw.create_async_openai_client",
        fake_create_async_openai_client,
    )

    UnifiedProvider(
        api_key="test-key",
        api_base="http://localhost:1234/v1",
        default_model="local-model",
    )

    assert captured["timeout"] == 1800.0


def test_normalize_openrouter_generation_metadata_prefers_served_provider() -> None:
    metadata = UnifiedProvider._normalize_openrouter_generation_metadata(
        {
            "id": "gen-123",
            "request_id": "req-123",
            "provider_name": "Z.AI",
            "latency": 7000.0,
            "generation_time": 6500.0,
            "provider_responses": [
                {"provider_name": "OpenAI", "latency": 1200.0, "status": 500},
                {"provider_name": "Z.AI", "latency": 6800.0, "status": 200},
            ],
            "upstream_id": "up-123",
        },
        generation_id="fallback",
    )

    assert metadata["generation_id"] == "gen-123"
    assert metadata["request_id"] == "req-123"
    assert metadata["latency_ms"] == 7000.0
    assert metadata["generation_time_ms"] == 6500.0
    assert metadata["provider_latency_ms"] == 6800.0
    assert metadata["provider_name"] == "Z.AI"
    assert metadata["upstream_id"] == "up-123"


def test_chat_attaches_openrouter_generation_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(_drive_chat_attaches_openrouter_generation_metadata(monkeypatch))


async def _drive_chat_attaches_openrouter_generation_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOBOT_OPENROUTER_CAPTURE_GENERATION_METADATA", "1")
    completions = _FakeCompletions(
        raw_response=_FakeRawResponse(
            {
                "choices": [
                    {
                        "message": {"content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                },
            },
            {"X-Generation-Id": "gen-123"},
        )
    )
    provider = _make_provider(completions=completions)

    async def fake_fetch(
        generation_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        assert generation_id == "gen-123"
        return (
            {
                "generation_id": generation_id,
                "request_id": "req-123",
                "provider_name": "Z.AI",
                "latency_ms": 7000.0,
                "generation_time_ms": 6500.0,
                "moderation_latency_ms": 0.0,
                "provider_latency_ms": 6800.0,
                "upstream_id": "up-123",
                "created_at": "2026-04-12T23:15:00Z",
                "api_type": "completions",
                "model": "z-ai/glm-5.1",
                "streamed": False,
                "provider_responses": [
                    {"provider_name": "Z.AI", "latency_ms": 6800.0, "status": 200}
                ],
            },
            {
                "openrouter_metadata_fetch_attempt_count": 1,
                "openrouter_metadata_fetch_last_reason": "success",
            },
        )

    monkeypatch.setattr(
        provider, "_fetch_openrouter_generation_metadata_with_diagnostics", fake_fetch
    )

    response = await provider.chat(messages=[{"role": "user", "content": "hi"}])
    await _await_openrouter_metadata_task(response)

    assert completions.raw_kwargs is not None
    assert completions.raw_kwargs["model"] == "z-ai/glm-5.1"
    assert response.content == "hello"
    assert response.usage == {
        "prompt_tokens": 5,
        "completion_tokens": 3,
        "total_tokens": 8,
    }
    assert response.extra["openrouter_generation_id"] == "gen-123"
    assert response.extra["openrouter_request_id"] == "req-123"
    assert response.extra["openrouter_latency_ms"] == 7000.0
    assert response.extra["openrouter_generation_time_ms"] == 6500.0
    assert response.extra["openrouter_provider_latency_ms"] == 6800.0
    assert response.extra["openrouter_provider_name"] == "Z.AI"
    assert response.extra["openrouter_upstream_id"] == "up-123"
    assert response.extra["openrouter_metadata"]["generation_id"] == "gen-123"
    assert response.extra["openrouter_metadata_retry_delays_s"] == [
        0.0,
        1.0,
        3.0,
        10.0,
        30.0,
    ]
    assert response.extra["openrouter_metadata_timeout_s"] == 5.0
    assert response.extra["openrouter_metadata_capture_enabled"] is True
    assert response.extra["openrouter_metadata_fetch_status"] == "success"
    assert response.extra["openrouter_metadata_fetch_ms"] >= 0.0
    assert response.extra["llm_call_time_ms"] == 6500.0
    assert response.extra["llm_timing_source"] == "openrouter_generation_time_ms"
    assert isinstance(response.extra["llm_wall_ts_end"], float)


def test_chat_stream_attaches_openrouter_generation_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(_drive_chat_stream_attaches_openrouter_generation_metadata(monkeypatch))


async def _drive_chat_stream_attaches_openrouter_generation_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOBOT_OPENROUTER_CAPTURE_GENERATION_METADATA", "1")
    final_usage = SimpleNamespace(prompt_tokens=7, completion_tokens=4, total_tokens=11)
    stream = _FakeAsyncStream(
        [
            _FakeChunk("hel"),
            _FakeChunk("lo", finish_reason="stop", usage=final_usage),
        ]
    )
    completions = _FakeCompletions(
        streaming_context=_FakeStreamingContext(
            _FakeStreamingRawResponse(stream, {"x-generation-id": "gen-456"})
        )
    )
    provider = _make_provider(completions=completions)
    deltas: list[str] = []

    async def fake_fetch(
        generation_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        assert generation_id == "gen-456"
        return (
            {
                "generation_id": generation_id,
                "request_id": "req-456",
                "provider_name": "Z.AI",
                "latency_ms": 9000.0,
                "generation_time_ms": 7200.0,
                "moderation_latency_ms": 0.0,
                "provider_latency_ms": 8800.0,
                "upstream_id": "up-456",
                "created_at": "2026-04-12T23:15:00Z",
                "api_type": "completions",
                "model": "z-ai/glm-5.1",
                "streamed": True,
                "provider_responses": [
                    {"provider_name": "Z.AI", "latency_ms": 8800.0, "status": 200}
                ],
            },
            {
                "openrouter_metadata_fetch_attempt_count": 1,
                "openrouter_metadata_fetch_last_reason": "success",
            },
        )

    monkeypatch.setattr(
        provider, "_fetch_openrouter_generation_metadata_with_diagnostics", fake_fetch
    )

    response = await provider.chat_stream(
        messages=[{"role": "user", "content": "hi"}],
        on_content_delta=lambda delta: _append_delta(deltas, delta),
    )
    await _await_openrouter_metadata_task(response)

    assert completions.stream_kwargs is not None
    assert completions.stream_kwargs["stream"] is True
    assert response.content == "hello"
    assert response.usage == {
        "prompt_tokens": 7,
        "completion_tokens": 4,
        "total_tokens": 11,
    }
    assert deltas == ["hel", "lo"]
    assert response.extra["openrouter_generation_id"] == "gen-456"
    assert response.extra["openrouter_latency_ms"] == 9000.0
    assert response.extra["openrouter_generation_time_ms"] == 7200.0
    assert response.extra["openrouter_provider_latency_ms"] == 8800.0
    assert response.extra["llm_call_time_ms"] == 7200.0
    assert response.extra["llm_timing_source"] == "openrouter_generation_time_ms"


def test_chat_without_openrouter_header_falls_back_to_wall_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(
        _drive_chat_without_openrouter_header_falls_back_to_wall_clock(monkeypatch)
    )


async def _drive_chat_without_openrouter_header_falls_back_to_wall_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completions = _FakeCompletions(
        raw_response=_FakeRawResponse(
            {
                "choices": [
                    {
                        "message": {"content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                },
            },
            {},
        )
    )
    provider = _make_provider(completions=completions)

    async def fail_fetch(_: str) -> dict[str, Any]:
        raise AssertionError("metadata fetch should not run without generation header")

    monkeypatch.setattr(provider, "_fetch_openrouter_generation_metadata", fail_fetch)

    response = await provider.chat(messages=[{"role": "user", "content": "hi"}])

    assert response.content == "hello"
    assert "openrouter_generation_id" not in response.extra
    assert "llm_call_time_ms" not in response.extra
    assert "llm_timing_source" not in response.extra
    assert isinstance(response.extra["llm_wall_ts_end"], float)


def test_chat_with_missing_openrouter_metadata_keeps_wall_clock_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(
        _drive_chat_with_missing_openrouter_metadata_keeps_wall_clock_fallback(
            monkeypatch
        )
    )


async def _drive_chat_with_missing_openrouter_metadata_keeps_wall_clock_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOBOT_OPENROUTER_CAPTURE_GENERATION_METADATA", "1")
    completions = _FakeCompletions(
        raw_response=_FakeRawResponse(
            {
                "choices": [
                    {
                        "message": {"content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                },
            },
            {"x-generation-id": "gen-missing"},
        )
    )
    provider = _make_provider(completions=completions)

    async def fake_fetch(_: str) -> tuple[dict[str, Any], dict[str, Any]]:
        return (
            {},
            {
                "openrouter_metadata_fetch_attempt_count": 1,
                "openrouter_metadata_fetch_last_reason": "not_found",
            },
        )

    monkeypatch.setattr(
        provider, "_fetch_openrouter_generation_metadata_with_diagnostics", fake_fetch
    )

    response = await provider.chat(messages=[{"role": "user", "content": "hi"}])
    await _await_openrouter_metadata_task(response)

    assert response.content == "hello"
    assert response.extra["openrouter_generation_id"] == "gen-missing"
    assert response.extra["openrouter_metadata_capture_enabled"] is True
    assert response.extra["openrouter_metadata_retry_delays_s"] == [
        0.0,
        1.0,
        3.0,
        10.0,
        30.0,
    ]
    assert response.extra["openrouter_metadata_timeout_s"] == 5.0
    assert response.extra["openrouter_metadata_fetch_status"] == "unavailable"
    assert response.extra["openrouter_metadata_fetch_last_reason"] == "not_found"
    assert response.extra["openrouter_metadata_fetch_ms"] >= 0.0
    assert "openrouter_metadata" not in response.extra
    assert "llm_call_time_ms" not in response.extra
    assert "llm_timing_source" not in response.extra


def test_chat_without_metadata_opt_in_still_fetches_generation_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(
        _drive_chat_without_metadata_opt_in_still_fetches_generation_lookup(monkeypatch)
    )


async def _drive_chat_without_metadata_opt_in_still_fetches_generation_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completions = _FakeCompletions(
        raw_response=_FakeRawResponse(
            {
                "choices": [
                    {
                        "message": {"content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                },
            },
            {"x-generation-id": "gen-disabled"},
        )
    )
    provider = _make_provider(completions=completions)
    seen_generation_ids: list[str] = []

    async def fake_fetch(
        generation_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        seen_generation_ids.append(generation_id)
        return (
            {
                "generation_id": generation_id,
                "latency_ms": 8000.0,
                "generation_time_ms": 7000.0,
            },
            {
                "openrouter_metadata_fetch_attempt_count": 1,
                "openrouter_metadata_fetch_last_reason": "success",
            },
        )

    monkeypatch.setattr(
        provider, "_fetch_openrouter_generation_metadata_with_diagnostics", fake_fetch
    )

    response = await provider.chat(messages=[{"role": "user", "content": "hi"}])
    await _await_openrouter_metadata_task(response)

    assert response.content == "hello"
    assert seen_generation_ids == ["gen-disabled"]
    assert response.extra["openrouter_generation_id"] == "gen-disabled"
    assert response.extra["openrouter_metadata_capture_enabled"] is True
    assert response.extra["openrouter_metadata_fetch_status"] == "success"
    assert response.extra["llm_call_time_ms"] == 7000.0


async def _append_delta(target: list[str], delta: str) -> None:
    target.append(delta)


async def _await_openrouter_metadata_task(response: Any) -> None:
    task = response.extra.get("_openrouter_metadata_task")
    if task is not None:
        await task
        await asyncio.sleep(0)
