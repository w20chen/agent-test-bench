"""Helpers for OpenAI-compatible clients shared across the repository."""

from __future__ import annotations

import uuid
from collections.abc import Mapping

from openai import AsyncOpenAI

_DEFAULT_OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/HKUDS/nanobot",
    "X-OpenRouter-Title": "nanobot",
    "X-OpenRouter-Categories": "cli-agent,personal-agent",
}


def uses_openrouter(api_base: str | None) -> bool:
    """Return ``True`` when *api_base* points to OpenRouter."""

    return bool(api_base and "openrouter" in api_base.lower())


def create_async_openai_client(
    *,
    api_key: str | None,
    api_base: str | None,
    timeout: float | None = None,
    extra_headers: Mapping[str, str] | None = None,
    include_session_affinity: bool = False,
) -> AsyncOpenAI:
    """Build a shared OpenAI-compatible async client."""

    headers: dict[str, str] = {}
    if include_session_affinity:
        headers["x-session-affinity"] = uuid.uuid4().hex
    if uses_openrouter(api_base):
        headers.update(_DEFAULT_OPENROUTER_HEADERS)
    if extra_headers:
        headers.update(extra_headers)

    kwargs: dict[str, object] = {
        "api_key": api_key or "no-key",
        "base_url": api_base,
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    if headers:
        kwargs["default_headers"] = headers
    return AsyncOpenAI(**kwargs)
