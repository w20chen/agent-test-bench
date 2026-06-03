from __future__ import annotations

import asyncio
from typing import Any

from agents.openclaw._hook import AgentHook, AgentHookContext
from agents.openclaw._runner import AgentRunner, AgentRunSpec
from agents.openclaw.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from agents.openclaw.tools.base import Tool
from agents.openclaw.tools.registry import ToolRegistry


def test_checkpoint_scrub_preserves_nested_payload_keys() -> None:
    payload = {
        "assistant_message": {
            "role": "assistant",
            "_openclaw_message_id": "msg_1",
            "content": {"_openclaw_message_id": "payload_value"},
        },
        "completed_tool_results": [
            {
                "role": "tool",
                "_openclaw_message_id": "msg_2",
                "content": {"_openclaw_message_id": "tool_payload"},
            }
        ],
        "pending_tool_calls": [
            {
                "id": "call_1",
                "function": {
                    "arguments": {"_openclaw_message_id": "arg_payload"}
                },
            }
        ],
    }

    clean = AgentRunner._strip_internal_ids_from_checkpoint_payload(payload)

    assert "_openclaw_message_id" not in clean["assistant_message"]
    assert "_openclaw_message_id" not in clean["completed_tool_results"][0]
    assert clean["assistant_message"]["content"] == {
        "_openclaw_message_id": "payload_value"
    }
    assert clean["completed_tool_results"][0]["content"] == {
        "_openclaw_message_id": "tool_payload"
    }
    assert clean["pending_tool_calls"][0]["function"]["arguments"] == {
        "_openclaw_message_id": "arg_payload"
    }


class _FakeProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__(api_key="test", api_base="http://test")
        self.responses = responses
        self.seen_messages: list[list[dict[str, Any]]] = []

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
        del tools, model, max_tokens, temperature, reasoning_effort, tool_choice
        self.seen_messages.append([dict(message) for message in messages])
        return self.responses.pop(0)

    def get_default_model(self) -> str:
        return "fake-model"


class _ListDirTool(Tool):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "List a directory."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }

    async def execute(self, **kwargs: Any) -> str:
        self.calls.append(dict(kwargs))
        return "ok"


class _MessageSnapshotHook(AgentHook):
    def __init__(self) -> None:
        self.snapshots: list[list[dict[str, Any]]] = []

    async def before_iteration(self, context: AgentHookContext) -> None:
        self.snapshots.append([dict(message) for message in context.messages])

    async def after_iteration(self, context: AgentHookContext) -> None:
        self.snapshots.append([dict(message) for message in context.messages])


def test_runner_reprompts_after_malformed_tool_call_text() -> None:
    asyncio.run(_run_malformed_tool_call_case())


def test_runner_preserves_inline_tool_markup_in_final_answer() -> None:
    asyncio.run(_run_inline_tool_markup_final_case())


def test_runner_hides_internal_message_ids_from_hooks_and_result() -> None:
    asyncio.run(_run_internal_message_id_surface_case())


async def _run_malformed_tool_call_case() -> None:
    provider = _FakeProvider(
        [
            LLMResponse(
                content=(
                    "I'll inspect the workspace.\n"
                    "<function=list_dir>\n"
                    "<parameter=path>\n"
                    "/app\n"
                    "</parameter>\n"
                    "</function>\n"
                    "</tool_call>"
                ),
                usage={"prompt_tokens": 10, "completion_tokens": 8},
            ),
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_list",
                        name="list_dir",
                        arguments={"path": "/app"},
                    )
                ],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 20, "completion_tokens": 4},
            ),
            LLMResponse(
                content="done",
                usage={"prompt_tokens": 30, "completion_tokens": 1},
            ),
        ]
    )
    tool = _ListDirTool()
    registry = ToolRegistry()
    registry.register(tool)

    result = await AgentRunner(provider).run(
        AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Recover the secret."}],
            tools=registry,
            model="fake-model",
            max_iterations=4,
            max_tool_result_chars=1000,
        )
    )

    assert result.final_content == "done"
    assert tool.calls == [{"path": "/app"}]
    assert len(provider.seen_messages) == 3
    retry_messages = provider.seen_messages[1]
    assert retry_messages[-2]["role"] == "assistant"
    assert "<function=list_dir>" in retry_messages[-2]["content"]
    synth_calls = retry_messages[-2].get("tool_calls") or []
    assert len(synth_calls) == 1, "synthetic assistant_call missing"
    synth_id = synth_calls[0]["id"]
    assert synth_id.startswith("malformed_retry_")
    assert synth_calls[0]["function"]["name"] == "_invalid_tool_call"
    assert retry_messages[-1]["role"] == "tool"
    assert retry_messages[-1]["tool_call_id"] == synth_id
    assert retry_messages[-1]["name"] == "_invalid_tool_call"
    assert "not valid and no tool was executed" in retry_messages[-1]["content"]
    assert "<tool_call>" in retry_messages[-1]["content"]
    assert "list_dir" in retry_messages[-1]["content"]


async def _run_inline_tool_markup_final_case() -> None:
    provider = _FakeProvider(
        [
            LLMResponse(
                content=(
                    "The literal text `<function=list_dir>` is malformed tool-call "
                    "markup, not an action I need to execute."
                ),
                usage={"prompt_tokens": 10, "completion_tokens": 8},
            )
        ]
    )
    registry = ToolRegistry()
    registry.register(_ListDirTool())

    result = await AgentRunner(provider).run(
        AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Explain this markup."}],
            tools=registry,
            model="fake-model",
            max_iterations=4,
            max_tool_result_chars=1000,
        )
    )

    assert result.final_content == (
        "The literal text `<function=list_dir>` is malformed tool-call "
        "markup, not an action I need to execute."
    )
    assert len(provider.seen_messages) == 1


async def _run_internal_message_id_surface_case() -> None:
    provider = _FakeProvider(
        [
            LLMResponse(
                content="done",
                usage={"prompt_tokens": 10, "completion_tokens": 1},
            )
        ]
    )
    hook = _MessageSnapshotHook()

    result = await AgentRunner(provider).run(
        AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Finish."}],
            tools=ToolRegistry(),
            hook=hook,
            model="fake-model",
            max_iterations=1,
            max_tool_result_chars=1000,
        )
    )

    assert hook.snapshots
    assert all(
        "_openclaw_message_id" not in message
        for snapshot in hook.snapshots
        for message in snapshot
    )
    assert all(
        "_openclaw_message_id" not in message
        for request in provider.seen_messages
        for message in request
    )
    assert all("_openclaw_message_id" not in message for message in result.messages)


# ---------------------------------------------------------------------------
# Test 1 — budget exhaustion
# ---------------------------------------------------------------------------

def test_malformed_retry_budget_exhaustion() -> None:
    asyncio.run(_run_budget_exhaustion_case())


async def _run_budget_exhaustion_case() -> None:
    # budget=0 means the very first malformed response exhausts the budget
    malformed = LLMResponse(
        content="<tool_call>\n<function=list_dir>",
        usage={"prompt_tokens": 10, "completion_tokens": 5},
    )
    provider = _FakeProvider([malformed])
    registry = ToolRegistry()
    registry.register(_ListDirTool())

    result = await AgentRunner(provider).run(
        AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Do the thing."}],
            tools=registry,
            model="fake-model",
            max_iterations=10,
            max_tool_result_chars=1000,
            malformed_retry_budget=0,
        )
    )

    assert result.stop_reason == "malformed_tool_call_budget_exhausted"
    assert result.error is not None
    assert "after 0 retries" in result.final_content
    assert len(provider.seen_messages) == 1


# ---------------------------------------------------------------------------
# Test 2 — hook context carries malformed_retry_count
# ---------------------------------------------------------------------------

def test_malformed_retry_emits_action_metadata() -> None:
    asyncio.run(_run_trace_metadata_case())


async def _run_trace_metadata_case() -> None:
    provider = _FakeProvider(
        [
            LLMResponse(
                content="<tool_call>\n<function=list_dir>",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
            ),
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_list",
                        name="list_dir",
                        arguments={"path": "/app"},
                    )
                ],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 20, "completion_tokens": 4},
            ),
            LLMResponse(
                content="done",
                usage={"prompt_tokens": 30, "completion_tokens": 1},
            ),
        ]
    )
    registry = ToolRegistry()
    registry.register(_ListDirTool())

    captured: list[dict[str, Any]] = []

    class _CaptureHook(AgentHook):
        async def after_iteration(self, context: AgentHookContext) -> None:
            captured.append(
                {
                    "iteration": context.iteration,
                    "malformed_retry_count": context.malformed_retry_count,
                }
            )

    result = await AgentRunner(provider).run(
        AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Do the thing."}],
            tools=registry,
            model="fake-model",
            max_iterations=5,
            max_tool_result_chars=1000,
            hook=_CaptureHook(),
        )
    )

    assert result.final_content == "done"
    assert len(captured) == 3
    # iter 0: malformed — count is 1
    assert captured[0]["iteration"] == 0
    assert captured[0]["malformed_retry_count"] == 1
    # iter 1: successful tool call — count reset to 0
    assert captured[1]["iteration"] == 1
    assert captured[1]["malformed_retry_count"] == 0
    # iter 2: final answer
    assert captured[2]["iteration"] == 2
    assert captured[2]["malformed_retry_count"] == 0


# ---------------------------------------------------------------------------
# Test 3 — synthetic pair is never orphaned after snip
# ---------------------------------------------------------------------------

def test_malformed_retry_pair_survives_snip() -> None:
    asyncio.run(_run_pair_survives_snip_case())


async def _run_pair_survives_snip_case() -> None:
    provider = _FakeProvider(
        [
            LLMResponse(
                content="<tool_call>\n<function=list_dir>",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
            ),
            LLMResponse(
                content="recovered final answer",
                usage={"prompt_tokens": 15, "completion_tokens": 3},
            ),
        ]
    )
    registry = ToolRegistry()
    registry.register(_ListDirTool())

    await AgentRunner(provider).run(
        AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Do thing. " * 20}],
            tools=registry,
            model="fake-model",
            max_iterations=4,
            max_tool_result_chars=1000,
            context_window_tokens=200,  # tight — forces snip
        )
    )

    assert len(provider.seen_messages) == 2
    iter1_messages = provider.seen_messages[1]

    # Collect all tool messages and all assistant messages that carry tool_calls
    tool_messages = [m for m in iter1_messages if m.get("role") == "tool"]
    asst_with_calls = [
        m
        for m in iter1_messages
        if m.get("role") == "assistant" and m.get("tool_calls")
    ]

    # Every tool message must have a matching declared id in some assistant_call.
    # No orphaned tool message is allowed.
    declared_ids = {
        tc["id"]
        for m in asst_with_calls
        for tc in m["tool_calls"]
    }
    for tm in tool_messages:
        assert tm["tool_call_id"] in declared_ids, (
            f"orphan tool message id={tm['tool_call_id']!r} in snip output"
        )
