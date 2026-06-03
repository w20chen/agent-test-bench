"""OpenClaw provider adapter built on the shared llm_call client layer.

Supports any OpenAI-compatible endpoint (OpenRouter, local servers, etc.).
No ProviderSpec dependency — configuration is passed directly to the constructor.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import os
import secrets
import string
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import httpx
import json_repair
from llm_call.openai_compat import create_async_openai_client, uses_openrouter

from agents.openclaw.providers.base import (
    GenerationSettings,
    LLMProvider,
    LLMResponse,
    ToolCallRequest,
)

_ALLOWED_MSG_KEYS = frozenset(
    {
        "role",
        "content",
        "tool_calls",
        "tool_call_id",
        "name",
    }
)
_ALNUM = string.ascii_letters + string.digits

_STANDARD_TC_KEYS = frozenset({"id", "type", "index", "function"})
_STANDARD_FN_KEYS = frozenset({"name", "arguments"})
_OPENROUTER_GENERATION_ID_HEADER = "x-generation-id"
_DEFAULT_OPENROUTER_METADATA_RETRY_DELAYS_S = (0.0, 1.0, 3.0, 10.0, 30.0)
_DEFAULT_OPENROUTER_METADATA_TIMEOUT_S = 5.0
_OPENCLAW_LLM_TIMEOUT_ENV = "OPENCLAW_LLM_TIMEOUT_S"


def _short_tool_id() -> str:
    """9-char alphanumeric ID compatible with all providers (incl. Mistral)."""
    return "".join(secrets.choice(_ALNUM) for _ in range(9))


def _get(obj: Any, key: str) -> Any:
    """Get a value from dict or object attribute, returning None if absent."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _coerce_dict(value: Any) -> dict[str, Any] | None:
    """Try to coerce *value* to a dict; return None if not possible or empty."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value if value else None
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict) and dumped:
            return dumped
    return None


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return str(value)


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_openrouter_metadata_retry_delays_s() -> tuple[float, ...]:
    raw_value = os.environ.get("NANOBOT_OPENROUTER_METADATA_RETRY_DELAYS_S")
    if not raw_value:
        return _DEFAULT_OPENROUTER_METADATA_RETRY_DELAYS_S
    delays: list[float] = []
    for piece in raw_value.split(","):
        value = _coerce_float(piece.strip())
        if value is not None and value >= 0.0:
            delays.append(value)
    return tuple(delays) or _DEFAULT_OPENROUTER_METADATA_RETRY_DELAYS_S


def _get_openrouter_metadata_timeout_s() -> float:
    raw_value = os.environ.get("NANOBOT_OPENROUTER_METADATA_TIMEOUT_S")
    value = _coerce_float(raw_value)
    if value is None or value <= 0.0:
        return _DEFAULT_OPENROUTER_METADATA_TIMEOUT_S
    return value


def _get_openclaw_llm_timeout_s() -> float | None:
    raw_value = os.environ.get(_OPENCLAW_LLM_TIMEOUT_ENV)
    if raw_value is None or raw_value == "":
        return None
    value = _coerce_float(raw_value)
    if value is None or value <= 0.0:
        raise ValueError(f"{_OPENCLAW_LLM_TIMEOUT_ENV} must be a positive number")
    return value


def _get_openrouter_metadata_policy() -> dict[str, Any]:
    return {
        "retry_delays_s": list(_get_openrouter_metadata_retry_delays_s()),
        "timeout_s": _get_openrouter_metadata_timeout_s(),
    }


def _extract_tc_extras(
    tc: Any,
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """Extract (extra_content, provider_specific_fields, fn_provider_specific_fields)."""
    extra_content = _coerce_dict(_get(tc, "extra_content"))

    tc_dict = _coerce_dict(tc)
    prov = None
    fn_prov = None
    if tc_dict is not None:
        leftover = {
            k: v
            for k, v in tc_dict.items()
            if k not in _STANDARD_TC_KEYS and k != "extra_content" and v is not None
        }
        if leftover:
            prov = leftover
        fn = _coerce_dict(tc_dict.get("function"))
        if fn is not None:
            fn_leftover = {
                k: v
                for k, v in fn.items()
                if k not in _STANDARD_FN_KEYS and v is not None
            }
            if fn_leftover:
                fn_prov = fn_leftover
    else:
        prov = _coerce_dict(_get(tc, "provider_specific_fields"))
        fn_obj = _get(tc, "function")
        if fn_obj is not None:
            fn_prov = _coerce_dict(_get(fn_obj, "provider_specific_fields"))

    return extra_content, prov, fn_prov


class UnifiedProvider(LLMProvider):
    """LLM provider for any OpenAI-compatible endpoint.

    Wraps ``openai.AsyncOpenAI`` directly without a ProviderSpec registry.
    Pass ``api_key`` and ``api_base`` for the target endpoint; when
    ``api_base`` contains "openrouter", attribution headers are added
    automatically.
    """

    def __init__(
        self,
        api_key: str | None,
        api_base: str | None,
        default_model: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        timeout: float | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.generation = GenerationSettings(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        self._client = create_async_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout if timeout is not None else _get_openclaw_llm_timeout_s(),
            include_session_affinity=True,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_tool_call_id(tool_call_id: Any) -> Any:
        """Normalize to a provider-safe 9-char alphanumeric form."""
        if not isinstance(tool_call_id, str):
            return tool_call_id
        if len(tool_call_id) == 9 and tool_call_id.isalnum():
            return tool_call_id
        return hashlib.sha1(tool_call_id.encode()).hexdigest()[:9]

    def _sanitize_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Strip non-standard keys, normalize tool_call IDs."""
        sanitized = LLMProvider._sanitize_request_messages(messages, _ALLOWED_MSG_KEYS)
        id_map: dict[str, str] = {}

        def map_id(value: Any) -> Any:
            if not isinstance(value, str):
                return value
            return id_map.setdefault(value, self._normalize_tool_call_id(value))

        for clean in sanitized:
            if isinstance(clean.get("tool_calls"), list):
                normalized = []
                for tc in clean["tool_calls"]:
                    if not isinstance(tc, dict):
                        normalized.append(tc)
                        continue
                    tc_clean = dict(tc)
                    tc_clean["id"] = map_id(tc_clean.get("id"))
                    normalized.append(tc_clean)
                clean["tool_calls"] = normalized
            if "tool_call_id" in clean and clean["tool_call_id"]:
                clean["tool_call_id"] = map_id(clean["tool_call_id"])
        return sanitized

    def _build_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> dict[str, Any]:
        model_name = model or self.default_model
        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": self._sanitize_messages(self._sanitize_empty_content(messages)),
            "temperature": temperature,
            "max_tokens": max(1, max_tokens),
        }
        if self.generation.top_p is not None:
            kwargs["top_p"] = self.generation.top_p
        extra_body = {
            key: value
            for key, value in {
                "top_k": self.generation.top_k,
                "repetition_penalty": self.generation.repetition_penalty,
            }.items()
            if value is not None
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        return kwargs

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    @staticmethod
    def _normalize_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
        if not headers:
            return {}
        return {str(k).lower(): str(v) for k, v in headers.items()}

    @classmethod
    def _select_openrouter_provider_response(
        cls,
        provider_responses: list[dict[str, Any]],
        provider_name: str | None,
    ) -> dict[str, Any] | None:
        if provider_name:
            served = [
                item
                for item in provider_responses
                if item.get("provider_name") == provider_name
            ]
            ok = [item for item in served if item.get("status") == 200]
            if ok:
                return ok[-1]
            if served:
                return served[-1]
        ok = [item for item in provider_responses if item.get("status") == 200]
        if ok:
            return ok[-1]
        return provider_responses[-1] if provider_responses else None

    @classmethod
    def _normalize_openrouter_generation_metadata(
        cls,
        payload: Mapping[str, Any],
        *,
        generation_id: str,
    ) -> dict[str, Any]:
        provider_name = _coerce_str(payload.get("provider_name"))
        provider_responses_raw = payload.get("provider_responses")
        provider_responses: list[dict[str, Any]] = []
        if isinstance(provider_responses_raw, list):
            for item in provider_responses_raw:
                item_map = cls._maybe_mapping(item) or {}
                provider_responses.append(
                    {
                        "endpoint_id": _coerce_str(item_map.get("endpoint_id")),
                        "id": _coerce_str(item_map.get("id")),
                        "is_byok": item_map.get("is_byok"),
                        "latency_ms": _coerce_float(item_map.get("latency")),
                        "model_permaslug": _coerce_str(item_map.get("model_permaslug")),
                        "provider_name": _coerce_str(item_map.get("provider_name")),
                        "status": item_map.get("status"),
                    }
                )
        selected_provider = cls._select_openrouter_provider_response(
            provider_responses,
            provider_name,
        )
        normalized = {
            "generation_id": _coerce_str(payload.get("id")) or generation_id,
            "request_id": _coerce_str(payload.get("request_id")),
            "provider_name": provider_name,
            "latency_ms": _coerce_float(payload.get("latency")),
            "generation_time_ms": _coerce_float(payload.get("generation_time")),
            "moderation_latency_ms": _coerce_float(payload.get("moderation_latency")),
            "provider_latency_ms": None
            if selected_provider is None
            else selected_provider.get("latency_ms"),
            "upstream_id": _coerce_str(payload.get("upstream_id")),
            "created_at": _coerce_str(payload.get("created_at")),
            "api_type": _coerce_str(payload.get("api_type")),
            "model": _coerce_str(payload.get("model")),
            "streamed": payload.get("streamed"),
            "provider_responses": provider_responses,
        }
        return normalized

    async def _fetch_openrouter_generation_metadata(
        self,
        generation_id: str,
    ) -> dict[str, Any]:
        (
            metadata,
            _diagnostics,
        ) = await self._fetch_openrouter_generation_metadata_with_diagnostics(
            generation_id
        )
        return metadata

    async def _fetch_openrouter_generation_metadata_with_diagnostics(
        self,
        generation_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        attempts: list[dict[str, Any]] = []
        if not generation_id or not self.api_base or not self.api_key:
            return {}, {
                "openrouter_metadata_fetch_attempt_count": 0,
                "openrouter_metadata_fetch_last_reason": (
                    "missing_generation_id_or_credentials"
                ),
            }

        url = f"{self.api_base.rstrip('/')}/generation"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        timeout = httpx.Timeout(_get_openrouter_metadata_timeout_s())
        retry_delays_s = _get_openrouter_metadata_retry_delays_s()

        async with httpx.AsyncClient(timeout=timeout) as client:
            for delay_s in retry_delays_s:
                if delay_s:
                    await asyncio.sleep(delay_s)
                attempt: dict[str, Any] = {
                    "delay_s": delay_s,
                    "attempt": len(attempts) + 1,
                }
                try:
                    response = await client.get(
                        url,
                        params={"id": generation_id},
                        headers=headers,
                    )
                except httpx.HTTPError as exc:
                    attempt["reason"] = "http_exception"
                    attempt["error_type"] = type(exc).__name__
                    attempts.append(attempt)
                    continue

                attempt["status_code"] = response.status_code
                if response.status_code == 404:
                    attempt["reason"] = "not_found"
                    attempts.append(attempt)
                    continue
                if response.is_error:
                    attempt["reason"] = "http_error"
                    attempts.append(attempt)
                    continue

                try:
                    payload = response.json()
                except ValueError:
                    attempt["reason"] = "invalid_json"
                    attempts.append(attempt)
                    continue

                data = payload.get("data") if isinstance(payload, dict) else None
                if isinstance(data, Mapping):
                    attempt["reason"] = "success"
                    attempts.append(attempt)
                    return (
                        self._normalize_openrouter_generation_metadata(
                            data,
                            generation_id=generation_id,
                        ),
                        self._summarize_openrouter_metadata_attempts(attempts),
                    )
                attempt["reason"] = "missing_data"
                attempts.append(attempt)
        return {}, self._summarize_openrouter_metadata_attempts(attempts)

    @staticmethod
    def _summarize_openrouter_metadata_attempts(
        attempts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "openrouter_metadata_fetch_attempt_count": len(attempts)
        }
        status_codes = [
            attempt["status_code"]
            for attempt in attempts
            if attempt.get("status_code") is not None
        ]
        if status_codes:
            summary["openrouter_metadata_fetch_status_codes"] = status_codes
            summary["openrouter_metadata_fetch_last_status_code"] = status_codes[-1]
        if attempts:
            last = attempts[-1]
            if last.get("reason") is not None:
                summary["openrouter_metadata_fetch_last_reason"] = last["reason"]
            if last.get("error_type") is not None:
                summary["openrouter_metadata_fetch_last_error_type"] = last[
                    "error_type"
                ]
        return summary

    async def _fetch_openrouter_extra_fields(
        self, generation_id: str
    ) -> dict[str, Any]:
        metadata_fetch_started_at = time.time()
        (
            metadata,
            diagnostics,
        ) = await self._fetch_openrouter_generation_metadata_with_diagnostics(
            generation_id
        )
        extra_fields: dict[str, Any] = {
            "openrouter_metadata_fetch_ms": (time.time() - metadata_fetch_started_at)
            * 1000.0,
            "openrouter_metadata_fetch_status": (
                "success" if metadata else "unavailable"
            ),
            **diagnostics,
        }
        if not metadata:
            return extra_fields

        llm_call_time_ms = metadata.get("generation_time_ms")
        llm_timing_source = "openrouter_generation_time_ms"
        if llm_call_time_ms is None:
            llm_call_time_ms = metadata.get("latency_ms")
            llm_timing_source = "openrouter_latency_ms"

        extra_fields.update(
            {
                "openrouter_metadata": metadata,
                "openrouter_request_id": metadata.get("request_id"),
                "openrouter_latency_ms": metadata.get("latency_ms"),
                "openrouter_generation_time_ms": metadata.get("generation_time_ms"),
                "openrouter_moderation_latency_ms": metadata.get(
                    "moderation_latency_ms"
                ),
                "openrouter_provider_latency_ms": metadata.get("provider_latency_ms"),
                "openrouter_provider_name": metadata.get("provider_name"),
                "openrouter_upstream_id": metadata.get("upstream_id"),
                "openrouter_created_at": metadata.get("created_at"),
                "openrouter_api_type": metadata.get("api_type"),
            }
        )
        if llm_call_time_ms is not None:
            extra_fields["llm_call_time_ms"] = llm_call_time_ms
            extra_fields["llm_timing_source"] = llm_timing_source
        return extra_fields

    async def _augment_response_extra(
        self,
        response: LLMResponse,
        *,
        response_headers: Mapping[str, str] | None,
    ) -> LLMResponse:
        extra = dict(response.extra)
        extra["llm_wall_ts_end"] = time.time()
        if not uses_openrouter(self.api_base):
            response.extra = extra
            return response

        headers = self._normalize_headers(response_headers)
        generation_id = headers.get(_OPENROUTER_GENERATION_ID_HEADER)
        if generation_id:
            extra["openrouter_generation_id"] = generation_id
        if not generation_id:
            response.extra = extra
            return response

        metadata_policy = _get_openrouter_metadata_policy()
        extra["openrouter_metadata_fetch_status"] = "pending"
        extra["openrouter_metadata_capture_enabled"] = True
        extra["openrouter_metadata_retry_delays_s"] = metadata_policy["retry_delays_s"]
        extra["openrouter_metadata_timeout_s"] = metadata_policy["timeout_s"]
        extra["openrouter_metadata_task_pending"] = True

        async def _refetch_openrouter_metadata() -> dict[str, Any]:
            return await self._fetch_openrouter_extra_fields(generation_id)

        metadata_task = asyncio.create_task(_refetch_openrouter_metadata())
        extra["_openrouter_metadata_task"] = metadata_task
        extra["_openrouter_metadata_refetcher"] = _refetch_openrouter_metadata
        response.extra = extra

        def _apply_openrouter_metadata(task: asyncio.Task[dict[str, Any]]) -> None:
            updated_extra = dict(response.extra)
            updated_extra["openrouter_metadata_task_pending"] = False
            updated_extra.pop("_openrouter_metadata_task", None)
            try:
                updated_extra.update(task.result())
            except Exception as exc:  # pragma: no cover - defensive guard
                updated_extra["openrouter_metadata_fetch_status"] = "task_error"
                updated_extra["openrouter_metadata_fetch_error"] = str(exc)
            response.extra = updated_extra

        metadata_task.add_done_callback(_apply_openrouter_metadata)
        return response

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _maybe_mapping(value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict):
            return value
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump()
            if isinstance(dumped, dict):
                return dumped
        return None

    @classmethod
    def _extract_text_content(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                item_map = cls._maybe_mapping(item)
                if item_map:
                    text = item_map.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                        continue
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
                    continue
                if isinstance(item, str):
                    parts.append(item)
            return "".join(parts) or None
        return str(value)

    @classmethod
    def _extract_usage(cls, response: Any) -> dict[str, int]:
        """Extract token usage from an OpenAI-compatible response."""
        usage_obj = None
        response_map = cls._maybe_mapping(response)
        if response_map is not None:
            usage_obj = response_map.get("usage")
        elif hasattr(response, "usage") and response.usage:
            usage_obj = response.usage

        usage_map = cls._maybe_mapping(usage_obj)
        if usage_map is not None:
            result = {
                "prompt_tokens": int(usage_map.get("prompt_tokens") or 0),
                "completion_tokens": int(usage_map.get("completion_tokens") or 0),
                "total_tokens": int(usage_map.get("total_tokens") or 0),
            }
        elif usage_obj:
            result = {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage_obj, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
            }
        else:
            return {}

        for path in (
            ("prompt_tokens_details", "cached_tokens"),
            ("cached_tokens",),
            ("prompt_cache_hit_tokens",),
        ):
            cached = cls._get_nested_int(usage_map, path)
            if not cached and usage_obj:
                cached = cls._get_nested_int(usage_obj, path)
            if cached:
                result["cached_tokens"] = cached
                break

        return result

    @staticmethod
    def _get_nested_int(obj: Any, path: tuple[str, ...]) -> int:
        current = obj
        for segment in path:
            if current is None:
                return 0
            if isinstance(current, dict):
                current = current.get(segment)
            else:
                current = getattr(current, segment, None)
        return int(current or 0) if current is not None else 0

    def _parse(self, response: Any) -> LLMResponse:
        if isinstance(response, str):
            return LLMResponse(content=response, finish_reason="stop")

        response_map = self._maybe_mapping(response)
        if response_map is not None:
            choices = response_map.get("choices") or []
            if not choices:
                content = self._extract_text_content(
                    response_map.get("content") or response_map.get("output_text")
                )
                if content is not None:
                    return LLMResponse(
                        content=content,
                        finish_reason=str(response_map.get("finish_reason") or "stop"),
                        usage=self._extract_usage(response_map),
                    )
                return LLMResponse(
                    content="Error: API returned empty choices.",
                    finish_reason="error",
                    extra={"error_type": "empty_choices"},
                )

            choice0 = self._maybe_mapping(choices[0]) or {}
            msg0 = self._maybe_mapping(choice0.get("message")) or {}
            content = self._extract_text_content(msg0.get("content"))
            finish_reason = str(choice0.get("finish_reason") or "stop")

            raw_tool_calls: list[Any] = []
            reasoning_content = msg0.get("reasoning_content")
            for ch in choices:
                ch_map = self._maybe_mapping(ch) or {}
                m = self._maybe_mapping(ch_map.get("message")) or {}
                tool_calls = m.get("tool_calls")
                if isinstance(tool_calls, list) and tool_calls:
                    raw_tool_calls.extend(tool_calls)
                    if ch_map.get("finish_reason") in ("tool_calls", "stop"):
                        finish_reason = str(ch_map["finish_reason"])
                if not content:
                    content = self._extract_text_content(m.get("content"))
                if not reasoning_content:
                    reasoning_content = m.get("reasoning_content")

            parsed_tool_calls = []
            for tc in raw_tool_calls:
                tc_map = self._maybe_mapping(tc) or {}
                fn = self._maybe_mapping(tc_map.get("function")) or {}
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    args = json_repair.loads(args)
                ec, prov, fn_prov = _extract_tc_extras(tc)
                parsed_tool_calls.append(
                    ToolCallRequest(
                        id=_short_tool_id(),
                        name=str(fn.get("name") or ""),
                        arguments=args if isinstance(args, dict) else {},
                        extra_content=ec,
                        provider_specific_fields=prov,
                        function_provider_specific_fields=fn_prov,
                    )
                )

            return LLMResponse(
                content=content,
                tool_calls=parsed_tool_calls,
                finish_reason=finish_reason,
                usage=self._extract_usage(response_map),
                reasoning_content=reasoning_content
                if isinstance(reasoning_content, str)
                else None,
            )

        if not response.choices:
            return LLMResponse(
                content="Error: API returned empty choices.",
                finish_reason="error",
                extra={"error_type": "empty_choices"},
            )

        choice = response.choices[0]
        msg = choice.message
        content = msg.content
        finish_reason = choice.finish_reason

        raw_tool_calls_obj: list[Any] = []
        for ch in response.choices:
            m = ch.message
            if hasattr(m, "tool_calls") and m.tool_calls:
                raw_tool_calls_obj.extend(m.tool_calls)
                if ch.finish_reason in ("tool_calls", "stop"):
                    finish_reason = ch.finish_reason
            if not content and m.content:
                content = m.content

        tool_calls = []
        for tc in raw_tool_calls_obj:
            args = tc.function.arguments
            if isinstance(args, str):
                args = json_repair.loads(args)
            ec, prov, fn_prov = _extract_tc_extras(tc)
            tool_calls.append(
                ToolCallRequest(
                    id=_short_tool_id(),
                    name=tc.function.name,
                    arguments=args,
                    extra_content=ec,
                    provider_specific_fields=prov,
                    function_provider_specific_fields=fn_prov,
                )
            )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason or "stop",
            usage=self._extract_usage(response),
            reasoning_content=getattr(msg, "reasoning_content", None) or None,
        )

    @classmethod
    def _parse_chunks(cls, chunks: list[Any]) -> LLMResponse:
        content_parts: list[str] = []
        tc_bufs: dict[int, dict[str, Any]] = {}
        finish_reason = "stop"
        usage: dict[str, int] = {}

        def _accum_tc(tc: Any, idx_hint: int) -> None:
            tc_index: int = (
                _get(tc, "index") if _get(tc, "index") is not None else idx_hint
            )
            buf = tc_bufs.setdefault(
                tc_index,
                {
                    "id": "",
                    "name": "",
                    "arguments": "",
                    "extra_content": None,
                    "prov": None,
                    "fn_prov": None,
                },
            )
            tc_id = _get(tc, "id")
            if tc_id:
                buf["id"] = str(tc_id)
            fn = _get(tc, "function")
            if fn is not None:
                fn_name = _get(fn, "name")
                if fn_name:
                    buf["name"] = str(fn_name)
                fn_args = _get(fn, "arguments")
                if fn_args:
                    buf["arguments"] += str(fn_args)
            ec, prov, fn_prov = _extract_tc_extras(tc)
            if ec:
                buf["extra_content"] = ec
            if prov:
                buf["prov"] = prov
            if fn_prov:
                buf["fn_prov"] = fn_prov

        for chunk in chunks:
            if isinstance(chunk, str):
                content_parts.append(chunk)
                continue

            chunk_map = cls._maybe_mapping(chunk)
            if chunk_map is not None:
                choices = chunk_map.get("choices") or []
                if not choices:
                    usage = cls._extract_usage(chunk_map) or usage
                    text = cls._extract_text_content(
                        chunk_map.get("content") or chunk_map.get("output_text")
                    )
                    if text:
                        content_parts.append(text)
                    continue
                choice = cls._maybe_mapping(choices[0]) or {}
                if choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])
                delta = cls._maybe_mapping(choice.get("delta")) or {}
                text = cls._extract_text_content(delta.get("content"))
                if text:
                    content_parts.append(text)
                for idx, tc in enumerate(delta.get("tool_calls") or []):
                    _accum_tc(tc, idx)
                usage = cls._extract_usage(chunk_map) or usage
                continue

            if not chunk.choices:
                usage = cls._extract_usage(chunk) or usage
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta
            if delta and delta.content:
                content_parts.append(delta.content)
            for tc in (delta.tool_calls or []) if delta else []:
                _accum_tc(tc, getattr(tc, "index", 0))

        return LLMResponse(
            content="".join(content_parts) or None,
            tool_calls=[
                ToolCallRequest(
                    id=b["id"] or _short_tool_id(),
                    name=b["name"],
                    arguments=json_repair.loads(b["arguments"])
                    if b["arguments"]
                    else {},
                    extra_content=b.get("extra_content"),
                    provider_specific_fields=b.get("prov"),
                    function_provider_specific_fields=b.get("fn_prov"),
                )
                for b in tc_bufs.values()
            ],
            finish_reason=finish_reason,
            usage=usage,
        )

    @staticmethod
    def _handle_error(e: Exception) -> LLMResponse:
        body = getattr(e, "doc", None) or getattr(
            getattr(e, "response", None), "text", None
        )
        msg = (
            f"Error: {body.strip()[:500]}"
            if body and body.strip()
            else f"Error calling LLM: {e}"
        )
        extra: dict[str, Any] = {"error_type": type(e).__name__}
        status_code = getattr(e, "status_code", None)
        if status_code is not None:
            extra["http_status"] = status_code
        request_id = getattr(e, "request_id", None)
        if request_id:
            extra["request_id"] = request_id
        raw_body = getattr(e, "body", None)
        if raw_body is not None:
            raw_str = str(raw_body) if not isinstance(raw_body, str) else raw_body
            extra["raw_body"] = raw_str[:1000]
        return LLMResponse(content=msg, finish_reason="error", extra=extra)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        kwargs = self._build_kwargs(
            messages,
            tools,
            model,
            max_tokens,
            temperature,
            reasoning_effort,
            tool_choice,
        )
        try:
            raw_response = await self._client.chat.completions.with_raw_response.create(
                **kwargs
            )
            parsed = await self._maybe_await(raw_response.parse())
            response = self._parse(parsed)
            return await self._augment_response_extra(
                response,
                response_headers=raw_response.headers,
            )
        except Exception as e:
            return self._handle_error(e)

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        kwargs = self._build_kwargs(
            messages,
            tools,
            model,
            max_tokens,
            temperature,
            reasoning_effort,
            tool_choice,
        )
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        idle_timeout_s = int(os.environ.get("NANOBOT_STREAM_IDLE_TIMEOUT_S", "90"))
        try:
            async with self._client.chat.completions.with_streaming_response.create(
                **kwargs
            ) as raw_response:
                stream = await raw_response.parse()
                chunks: list[Any] = []
                stream_iter = stream.__aiter__()
                while True:
                    try:
                        chunk = await asyncio.wait_for(
                            stream_iter.__anext__(),
                            timeout=idle_timeout_s,
                        )
                    except StopAsyncIteration:
                        break
                    chunks.append(chunk)
                    if on_content_delta and chunk.choices:
                        text = getattr(chunk.choices[0].delta, "content", None)
                        if text:
                            await on_content_delta(text)
                response = self._parse_chunks(chunks)
                return await self._augment_response_extra(
                    response,
                    response_headers=raw_response.headers,
                )
        except asyncio.TimeoutError:
            return LLMResponse(
                content=(
                    f"Error calling LLM: stream stalled for more than "
                    f"{idle_timeout_s} seconds"
                ),
                finish_reason="error",
                extra={"error_type": "stream_timeout", "timeout_s": idle_timeout_s},
            )
        except Exception as e:
            return self._handle_error(e)

    def get_default_model(self) -> str:
        return self.default_model
