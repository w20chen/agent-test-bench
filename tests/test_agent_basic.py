from __future__ import annotations

import asyncio
import json
from typing import Any

from agents.base import AgentBase, TraceAction


class DummyAgent(AgentBase):
    async def run(self, task: dict[str, Any]) -> bool:  # pragma: no cover - trivial
        self.task_id = task["task_id"]
        self.task_success = True
        return True


def test_action_export_preserves_fields() -> None:
    agent = DummyAgent(
        agent_id="agent-0001", api_base="http://localhost:8000/v1", model="mock"
    )
    action = TraceAction(
        action_type="llm_call",
        action_id="llm_0",
        agent_id="agent-0001",
        program_id="agent-0001",
        iteration=1,
        ts_start=1.0,
        ts_end=2.0,
        data={
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "llm_latency_ms": 12.5,
            "raw_response": {"id": "resp-1"},
        },
    )
    agent.actions.append(action)
    exported = agent.get_trace()
    assert exported[0]["action_type"] == "llm_call"
    assert exported[0]["data"]["prompt_tokens"] == 10
    assert exported[0]["data"]["raw_response"]["id"] == "resp-1"
    json.dumps(exported)


def test_agent_summary_aggregates_action_fields() -> None:
    agent = DummyAgent(
        agent_id="agent-0002", api_base="http://localhost:8000/v1", model="mock"
    )
    agent.task_id = "task-1"
    agent.task_success = True
    agent.actions = [
        TraceAction(
            action_type="llm_call",
            action_id="llm_0",
            iteration=0,
            data={"prompt_tokens": 12, "completion_tokens": 3, "llm_latency_ms": 10.0},
        ),
        TraceAction(
            action_type="tool_exec",
            action_id="tool_1_bash",
            iteration=1,
            data={"tool_name": "bash", "duration_ms": 25.0},
        ),
    ]
    summary = agent.summary()
    assert summary["program_id"] == "agent-0002"
    assert summary["n_iterations"] == 2  # 2 distinct iterations
    assert summary["total_tokens"] == 15
    assert summary["total_tool_ms"] == 25.0


class FakeUsage:
    prompt_tokens = 11
    completion_tokens = 4


class FakeMessage:
    content = "synthetic reply"
    tool_calls = None


class FakeChoice:
    message = FakeMessage()


class FakeResponse:
    choices = [FakeChoice()]
    usage = FakeUsage()

    def model_dump(self) -> dict[str, Any]:
        return {"id": "resp-2"}


class RecordingCompletions:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> FakeResponse:
        self.kwargs = kwargs
        return FakeResponse()


class RecordingChat:
    def __init__(self) -> None:
        self.completions = RecordingCompletions()


class RecordingClient:
    def __init__(self) -> None:
        self.chat = RecordingChat()


def test_call_llm_attaches_program_id_and_returns_normalized_fields() -> None:
    agent = DummyAgent(
        agent_id="agent-0004", api_base="http://localhost:8000/v1", model="mock"
    )
    recording_client = RecordingClient()
    agent._client = recording_client  # type: ignore[assignment]
    result = asyncio.run(agent._call_llm([{"role": "user", "content": "hi"}]))
    assert recording_client.chat.completions.kwargs is not None
    assert (
        recording_client.chat.completions.kwargs["extra_body"]["program_id"]
        == "agent-0004"
    )
    assert result.content == "synthetic reply"
    assert result.prompt_tokens == 11
    assert result.raw_response["id"] == "resp-2"
