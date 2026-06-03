"""Targeted tests for OpenClaw tool registration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agents.openclaw._loop import AgentLoop
from agents.openclaw.bus.events import InboundMessage
from agents.openclaw.bus.queue import MessageBus


def _fake_provider() -> SimpleNamespace:
    return SimpleNamespace(
        get_default_model=lambda: "qwen-plus-latest",
        generation=SimpleNamespace(max_tokens=1024),
    )


def test_agent_loop_keeps_spawn_for_local_tools(tmp_path: Path) -> None:
    loop = AgentLoop(
        bus=MessageBus(),
        provider=_fake_provider(),
        workspace=tmp_path / "state",
        tool_workspace=Path("/testbed"),
        model="qwen-plus-latest",
    )

    assert loop.tools.has("spawn") is True


def test_agent_loop_records_error_outcome_when_dispatch_crashes(tmp_path: Path, monkeypatch) -> None:
    bus = MessageBus()
    loop = AgentLoop(
        bus=bus,
        provider=_fake_provider(),
        workspace=tmp_path / "state",
        tool_workspace=Path("/testbed"),
        model="qwen-plus-latest",
    )

    async def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(loop, "_process_message", boom)

    async def run_test() -> None:
        msg = InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="test-chat",
            content="fix bug",
        )
        await loop._dispatch(msg)
        outbound = await bus.consume_outbound()
        assert outbound.content == "Sorry, I encountered an error."
        assert loop._last_run_outcomes[msg.session_key] == {
            "stop_reason": "error",
            "error": "Sorry, I encountered an error.",
        }

    import asyncio

    asyncio.run(run_test())
