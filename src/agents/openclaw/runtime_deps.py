"""Minimal Python requirements for running OpenClaw inside task containers."""

from __future__ import annotations

OPENCLAW_CONTAINER_RUNTIME_REQUIREMENTS: tuple[str, ...] = (
    "openai>=2.0,<3.0",
    "httpx>=0.27,<1.0",
    "PyYAML>=6.0,<7.0",
    "json-repair>=0.30,<1.0",
    "loguru>=0.7,<1.0",
    "pydantic>=2.0,<3.0",
    "socksio>=1.0,<2.0",
    "tiktoken>=0.7,<1.0",
)

OPENCLAW_MCP_RUNTIME_REQUIREMENTS: tuple[str, ...] = ("mcp>=1.0",)

