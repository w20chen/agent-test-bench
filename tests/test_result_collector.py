from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from agents.openclaw.bus.events import OutboundMessage
from agents.openclaw.bus.queue import MessageBus
from agents.openclaw.eval.collector import ResultCollector
from agents.openclaw.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from agents.openclaw._session_runner import SessionRunner
from agents.openclaw.tools.message import MessageTool
from serving.recording.hooks import LayerCapturer


class _MessageThenFinalProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.second_call_started = asyncio.Event()
        self.release_final = asyncio.Event()

    def get_default_model(self) -> str:
        return "fake-model"

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
        del messages, tools, model, max_tokens, temperature, reasoning_effort, tool_choice
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="message-1",
                        name="message",
                        arguments={"content": "intermediate"},
                    )
                ],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 1, "completion_tokens": 1},
            )
        self.second_call_started.set()
        await self.release_final.wait()
        return LLMResponse(
            content="final",
            finish_reason="stop",
            usage={"prompt_tokens": 1, "completion_tokens": 1},
        )


def test_message_tool_outbound_is_marked_nonfinal() -> None:
    captured: list[OutboundMessage] = []

    async def send(msg: OutboundMessage) -> None:
        captured.append(msg)

    tool = MessageTool(send_callback=send)
    tool.set_context("cli", "session-1", "message-1")

    result = asyncio.run(tool.execute("intermediate"))

    assert result == "Message sent to cli:session-1"
    assert len(captured) == 1
    assert captured[0].metadata["_openclaw_tool_message"] is True
    assert captured[0].metadata["message_id"] == "message-1"


def test_result_collector_ignores_message_tool_outbound() -> None:
    asyncio.run(_drive_result_collector_ignores_message_tool_outbound())


async def _drive_result_collector_ignores_message_tool_outbound() -> None:
    bus = MessageBus()
    collector = ResultCollector(bus)
    await collector.start()
    try:
        wait_task = asyncio.create_task(collector.wait_for_result("cli:session-1"))

        await bus.publish_outbound(
            OutboundMessage(
                channel="cli",
                chat_id="session-1",
                content="intermediate",
                metadata={"_openclaw_tool_message": True},
            )
        )
        await asyncio.sleep(0)

        assert not wait_task.done()
        assert collector.get_result("cli:session-1") is None

        await bus.publish_outbound(
            OutboundMessage(channel="cli", chat_id="session-1", content="final")
        )

        result = await asyncio.wait_for(wait_task, timeout=1.0)
    finally:
        collector.stop()

    assert result == "final"


def test_session_runner_waits_past_message_tool_outbound(tmp_path: Path) -> None:
    asyncio.run(_drive_session_runner_waits_past_message_tool_outbound(tmp_path))


async def _drive_session_runner_waits_past_message_tool_outbound(
    tmp_path: Path,
) -> None:
    provider = _MessageThenFinalProvider()
    runner = SessionRunner(provider=provider, model="fake-model", max_iterations=3)
    trace_file = tmp_path / "trace.jsonl"

    run_task = asyncio.create_task(
        runner.run(
            prompt="send a message, then finish",
            workspace=tmp_path / "workspace",
            session_key="cli:session-1",
            trace_file=trace_file,
        )
    )

    await asyncio.wait_for(provider.second_call_started.wait(), timeout=1.0)
    completed_before_final = run_task.done()
    provider.release_final.set()
    result = await asyncio.wait_for(run_task, timeout=1.0)

    assert not completed_before_final
    assert result.content == "final"
    assert provider.calls == 2

    records = [
        json.loads(line)
        for line in trace_file.read_text(encoding="utf-8").splitlines()
    ]
    llm_calls = [
        record
        for record in records
        if record.get("type") == "action"
        and record.get("action_type") == "llm_call"
    ]
    summary = next(record for record in records if record.get("type") == "summary")

    assert [call["iteration"] for call in llm_calls] == [0, 1]
    assert summary["n_iterations"] == 2

    capturer = object.__new__(LayerCapturer)
    capturer._meta = {"iters": [{"call_idx": 0}, {"call_idx": 1}]}
    capturer._align_meta_to_trace(trace_file)

    assert capturer._meta["alignment"]["aligned_iters"] == 2
    assert capturer._meta["alignment"]["orphan_iters"] == 0
    assert capturer._meta["alignment"]["missing_recording_iters"] == 0
    assert "orphan_iters" not in capturer._meta
    assert "missing_recording_iters" not in capturer._meta
