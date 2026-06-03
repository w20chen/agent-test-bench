"""Shared remote LLM configuration and client helpers.

This package is the only supported entrypoint for provider/model resolution
and OpenAI-compatible client construction across the repository.
"""

from llm_call.config import (
    ResolvedLLMConfig,
    add_llm_config_arguments,
    provider_choices,
    resolve_llm_config,
)
from llm_call.openclaw import UnifiedProvider
from llm_call.openai_compat import create_async_openai_client, uses_openrouter

__all__ = [
    "ResolvedLLMConfig",
    "UnifiedProvider",
    "add_llm_config_arguments",
    "create_async_openai_client",
    "provider_choices",
    "resolve_llm_config",
    "uses_openrouter",
]
