"""Registered remote LLM providers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderDefinition:
    """Static configuration for a supported LLM provider."""

    api_base: str
    env_key: str


PROVIDERS: dict[str, ProviderDefinition] = {
    "openrouter": ProviderDefinition(
        api_base="https://openrouter.ai/api/v1",
        env_key="OPENROUTER_API_KEY",
    ),
    "dashscope": ProviderDefinition(
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        env_key="DASHSCOPE_API_KEY",
    ),
    "openai": ProviderDefinition(
        api_base="https://api.openai.com/v1",
        env_key="OPENAI_API_KEY",
    ),
    "siliconflow": ProviderDefinition(
        api_base="https://api.siliconflow.com/v1",
        env_key="SILICONFLOW_API_KEY",
    ),
    "deepseek": ProviderDefinition(
        api_base="https://api.deepseek.com/v1",
        env_key="DEEPSEEK_API_KEY",
    ),
}


def provider_choices() -> list[str]:
    """Return supported provider names in CLI-friendly order."""

    return list(PROVIDERS.keys())
