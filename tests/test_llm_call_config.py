"""Tests for the shared llm_call provider/config helpers."""

from __future__ import annotations

import pytest

from llm_call.config import resolve_llm_config
from llm_call.openai_compat import uses_openrouter


def test_resolve_llm_config_requires_provider() -> None:
    with pytest.raises(ValueError, match="Missing required --provider"):
        resolve_llm_config(
            provider=None,
            api_base=None,
            api_key=None,
            model="z-ai/glm-5.1",
            environ={},
        )


def test_resolve_llm_config_requires_model() -> None:
    with pytest.raises(ValueError, match="Missing required --model"):
        resolve_llm_config(
            provider="openrouter",
            api_base=None,
            api_key=None,
            model=None,
            environ={"OPENROUTER_API_KEY": "test-key"},
        )


def test_resolve_llm_config_uses_provider_metadata_without_defaults() -> None:
    resolved = resolve_llm_config(
        provider="openrouter",
        api_base=None,
        api_key=None,
        model="z-ai/glm-5.1",
        environ={"OPENROUTER_API_KEY": "test-key"},
    )

    assert resolved.name == "openrouter"
    assert resolved.api_base == "https://openrouter.ai/api/v1"
    assert resolved.api_key == "test-key"
    assert resolved.model == "z-ai/glm-5.1"
    assert resolved.env_key == "OPENROUTER_API_KEY"


def test_resolve_llm_config_honors_explicit_overrides() -> None:
    resolved = resolve_llm_config(
        provider="dashscope",
        api_base="https://override.example/v1",
        api_key="override-key",
        model="override-model",
        environ={"DASHSCOPE_API_KEY": "ignored"},
    )

    assert resolved.name == "dashscope"
    assert resolved.api_base == "https://override.example/v1"
    assert resolved.api_key == "override-key"
    assert resolved.model == "override-model"


def test_resolve_llm_config_supports_siliconflow() -> None:
    resolved = resolve_llm_config(
        provider="siliconflow",
        api_base=None,
        api_key=None,
        model="Pro/zai-org/GLM-5.1",
        environ={"SILICONFLOW_API_KEY": "sf-test-key"},
    )

    assert resolved.name == "siliconflow"
    assert resolved.api_base == "https://api.siliconflow.com/v1"
    assert resolved.api_key == "sf-test-key"
    assert resolved.model == "Pro/zai-org/GLM-5.1"
    assert resolved.env_key == "SILICONFLOW_API_KEY"


def test_uses_openrouter_matches_base_url() -> None:
    assert uses_openrouter("https://openrouter.ai/api/v1") is True
    assert uses_openrouter("https://api.openai.com/v1") is False
