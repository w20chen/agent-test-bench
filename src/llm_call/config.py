"""Shared provider/model configuration resolution."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Mapping

from llm_call.providers import PROVIDERS, provider_choices


@dataclass(frozen=True)
class ResolvedLLMConfig:
    """Resolved provider settings after applying explicit overrides."""

    name: str
    api_base: str
    api_key: str
    model: str
    env_key: str


def add_llm_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the shared provider/model CLI contract on ``parser``."""

    parser.add_argument(
        "--provider",
        choices=provider_choices(),
        default=None,
        help=(
            "LLM provider. Required for execution; resolves the default API base "
            "and provider-specific API key env var."
        ),
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="Override API base URL (default: from --provider).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Override API key (default: from the provider's env var).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model name. Required for execution.",
    )


def nonnegative_float_arg(value: str) -> float:
    parsed = _float_arg(value, name="value")
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"value must be non-negative, got {value!r}")
    return parsed


def positive_float_arg(value: str) -> float:
    parsed = _float_arg(value, name="value")
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"value must be positive, got {value!r}")
    return parsed


def top_p_arg(value: str) -> float:
    parsed = positive_float_arg(value)
    if parsed > 1:
        raise argparse.ArgumentTypeError(f"top_p must be <= 1, got {value!r}")
    return parsed


def positive_int_arg(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"value must be an integer, got {value!r}"
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"value must be positive, got {value!r}")
    return parsed


def _float_arg(value: str, *, name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{name} must be a float, got {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError(f"{name} must be finite, got {value!r}")
    return parsed


def resolve_llm_config(
    *,
    provider: str | None,
    api_base: str | None,
    api_key: str | None,
    model: str | None,
    environ: Mapping[str, str] | None = None,
) -> ResolvedLLMConfig:
    """Resolve provider/model settings with no defaults or presets."""

    if not provider:
        raise ValueError(
            f"Missing required --provider. Choose one of: {', '.join(provider_choices())}."
        )
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unsupported provider {provider!r}. Choose one of: {', '.join(provider_choices())}."
        )
    if not model:
        raise ValueError("Missing required --model. Example: --model z-ai/glm-5.1.")

    env = os.environ if environ is None else environ
    definition = PROVIDERS[provider]
    resolved_api_key = api_key or env.get(definition.env_key, "")
    return ResolvedLLMConfig(
        name=provider,
        api_base=api_base or definition.api_base,
        api_key=resolved_api_key,
        model=model,
        env_key=definition.env_key,
    )
