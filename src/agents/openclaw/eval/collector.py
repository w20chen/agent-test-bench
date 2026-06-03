"""Minimal outbound message collector — replaces ChannelManager._dispatch_outbound.

In gateway mode, ChannelManager routes OutboundMessage to chat platforms
(Telegram, Discord, etc.) with delta coalescing and retry. For SWE-bench
evaluation there are no chat platforms — we just collect the final response
per session key.
"""

from __future__ import annotations

import asyncio

from loguru import logger

from agents.openclaw.bus.queue import MessageBus

class ResultCollector:
    """Collects outbound messages keyed by session_key.

    Each SWE-bench task publishes one InboundMessage with a unique session_key
    (e.g. "eval:{instance_id}"). The collector gathers the corresponding
    OutboundMessage and signals completion via an asyncio.Event.
    """

    def __init__(self, bus: MessageBus):
        self.bus = bus
        self._results: dict[str, str] = {}
        self._done_events: dict[str, asyncio.Event] = {}
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Result collector started")

    def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

    async def _run_loop(self) -> None:
        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_outbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            if msg.metadata.get("_openclaw_tool_message"):
                continue

            session_key = f"{msg.channel}:{msg.chat_id}"
            # Coalesce: accumulate content for the same session_key
            prev = self._results.get(session_key, "")
            self._results[session_key] = prev + msg.content

            # Signal completion when we see a non-streaming message or _stream_end
            if not msg.metadata.get("_stream_delta"):
                event = self._done_events.get(session_key)
                if event:
                    event.set()

    async def wait_for_result(self, session_key: str) -> str | None:
        event = self._done_events.setdefault(session_key, asyncio.Event())
        try:
            # No timeout — the agent loop controls lifecycle
            await event.wait()
        except asyncio.CancelledError:
            return None
        return self._results.get(session_key)

    def get_result(self, session_key: str) -> str | None:
        return self._results.get(session_key)

    def clear(self) -> None:
        self._results.clear()
        self._done_events.clear()
