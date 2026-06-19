"""Sessions yield tool — wait for subagent results inline.

The tool blocks inside :meth:`execute` until all expected subagent
results arrive (or a timeout elapses), then returns the collected
findings directly as the tool result.  No runner-level yield / stop /
re-dispatch is needed; the agent continues its existing iteration
loop naturally after the tool returns.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from loguru import logger

from agents.openclaw.tools.base import Tool


class YieldTool(Tool):
    """Block the current turn until subagent results are available.

    The tool polls ``_drain_results`` (the same pending-results queue
    populated by :class:`SubagentManager`) at 1-second intervals and
    returns once results arrive or *timeout_s* is reached.
    """

    _DEFAULT_TIMEOUT_S = 120.0
    _POLL_INTERVAL_S = 1.0

    def __init__(self, drain_results: Callable[[str], list[dict[str, Any]]] | None = None):
        self._drain = drain_results
        self._session_key: str = "cli:direct"

    def set_context(self, channel: str, chat_id: str) -> None:
        """Store the active session key for result polling."""
        self._session_key = f"{channel}:{chat_id}"

    @property
    def name(self) -> str:
        return "sessions_yield"

    @property
    def description(self) -> str:
        return (
            "End the current turn and wait for sub-agent completion events. "
            "Use this after spawning subagents with the spawn tool. "
            "The tool will block until all expected subagent results arrive, "
            "then return the collected findings. "
            "Do NOT poll in a loop; call sessions_yield once and wait."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": (
                        "Optional message describing what you are waiting for."
                    ),
                },
            },
            "required": [],
        }

    async def execute(self, message: str = "", **kwargs: Any) -> str:
        """Block until subagent results arrive, then return them."""
        import json

        wait_msg = message or "Waiting for sub-agent completion events."
        deadline = time.monotonic() + self._DEFAULT_TIMEOUT_S
        collected: list[str] = []

        while time.monotonic() < deadline:
            if self._drain is None:
                break
            pending = self._drain(self._session_key)
            if pending:
                for pmsg in pending:
                    label = pmsg.get("_subagent_label", "subagent")
                    status = pmsg.get("_subagent_status", "ok")
                    content = pmsg.get("content", "")
                    logger.info(
                        "YieldTool collected [{}] ({}) — {} chars",
                        label, status, len(content),
                    )
                    collected.append(f"## {label} ({status})\n\n{content}")
                if collected:
                    break
            await asyncio.sleep(self._POLL_INTERVAL_S)

        if not collected:
            return json.dumps({
                "status": "timeout",
                "message": (
                    f"No subagent results received within "
                    f"{self._DEFAULT_TIMEOUT_S:.0f}s. "
                    "Continue with available information."
                ),
            })

        return json.dumps({
            "status": "ok",
            "message": f"Collected {len(collected)} subagent result(s).",
            "results": collected,
        })
