from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.openclaw.utils.helpers import build_assistant_message
from agents.terminal_bench.runner import TerminalBenchRunner
from llm_call.openclaw import UnifiedProvider


def _assistant_tool_call_message() -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "tool-call",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }
        ],
        "reasoning_content": "",
    }


def _tool_call_delta() -> dict[str, Any]:
    return {
        "index": 0,
        "id": "tool-call",
        "type": "function",
        "function": {"name": "read_file", "arguments": "{}"},
    }


def _replay_response(provider: UnifiedProvider, response: Any) -> dict[str, Any]:
    message = build_assistant_message(
        response.content,
        tool_calls=[call.to_openai_tool_call() for call in response.tool_calls],
        reasoning_content=response.reasoning_content,
    )
    return provider._build_kwargs(
        messages=[message],
        tools=None,
        model=None,
        max_tokens=16,
        temperature=0.1,
        reasoning_effort=None,
        tool_choice=None,
    )


def test_unified_provider_includes_optional_generation_params() -> None:
    provider = UnifiedProvider(
        api_key="test",
        api_base="http://127.0.0.1:1/v1",
        default_model="test-model",
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.05,
    )

    kwargs = provider._build_kwargs(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        model=None,
        max_tokens=16,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
    )

    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.8
    assert kwargs["extra_body"] == {
        "top_k": 20,
        "repetition_penalty": 1.05,
    }


def test_deepseek_preserves_empty_reasoning_content_for_tool_call_replay() -> None:
    provider = UnifiedProvider(
        api_key="test",
        api_base="https://api.deepseek.com/v1",
        default_model="deepseek-v4-flash",
    )

    kwargs = provider._build_kwargs(
        messages=[_assistant_tool_call_message()],
        tools=None,
        model=None,
        max_tokens=16,
        temperature=0.1,
        reasoning_effort=None,
        tool_choice=None,
    )

    assert kwargs["messages"][0]["reasoning_content"] == ""


def test_deepseek_replays_empty_reasoning_content_from_nonstream_response() -> None:
    provider = UnifiedProvider(
        api_key="test",
        api_base="https://api.deepseek.com/v1",
        default_model="deepseek-v4-flash",
    )
    response = provider._parse(
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [_tool_call_delta()],
                        "reasoning_content": "",
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )

    kwargs = _replay_response(provider, response)

    assert response.reasoning_content == ""
    assert kwargs["messages"][0]["reasoning_content"] == ""


def test_deepseek_replays_empty_reasoning_content_from_stream_response() -> None:
    provider = UnifiedProvider(
        api_key="test",
        api_base="https://api.deepseek.com/v1",
        default_model="deepseek-v4-flash",
    )
    response = provider._parse_chunks(
        [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [_tool_call_delta()],
                            "reasoning_content": "",
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        ]
    )

    kwargs = _replay_response(provider, response)

    assert response.reasoning_content == ""
    assert kwargs["messages"][0]["reasoning_content"] == ""


def test_deepseek_endpoint_detection_accepts_trailing_dns_dot() -> None:
    provider = UnifiedProvider(
        api_key="test",
        api_base="https://api.deepseek.com./v1",
        default_model="deepseek-v4-flash",
    )

    kwargs = provider._build_kwargs(
        messages=[_assistant_tool_call_message()],
        tools=None,
        model=None,
        max_tokens=16,
        temperature=0.1,
        reasoning_effort=None,
        tool_choice=None,
    )

    assert kwargs["messages"][0]["reasoning_content"] == ""


def test_deepseek_endpoint_detection_rejects_lookalike_hostname() -> None:
    provider = UnifiedProvider(
        api_key="test",
        api_base="https://api.deepseek.com.example.org/v1",
        default_model="test-model",
    )

    kwargs = provider._build_kwargs(
        messages=[_assistant_tool_call_message()],
        tools=None,
        model=None,
        max_tokens=16,
        temperature=0.1,
        reasoning_effort=None,
        tool_choice=None,
    )

    assert "reasoning_content" not in kwargs["messages"][0]


def test_other_providers_strip_empty_reasoning_content() -> None:
    provider = UnifiedProvider(
        api_key="test",
        api_base="https://api.openai.com/v1",
        default_model="test-model",
    )

    kwargs = provider._build_kwargs(
        messages=[_assistant_tool_call_message()],
        tools=None,
        model=None,
        max_tokens=16,
        temperature=0.1,
        reasoning_effort=None,
        tool_choice=None,
    )

    assert "reasoning_content" not in kwargs["messages"][0]


def test_terminal_bench_runner_passes_generation_agent_kwargs(tmp_path: Path) -> None:
    runner = TerminalBenchRunner(
        provider_name="openai",
        env_key="OPENAI_API_KEY",
        api_base="http://127.0.0.1:1/v1",
        api_key="test",
        model="test-model",
        workspace_base=tmp_path / "workspace",
        max_iterations=100,
        context_window_tokens=256_000,
        benchmark_slug="terminal-bench",
        benchmark_extras={},
        generation_config={
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.05,
        },
    )

    command = runner._build_tb_command(
        task={
            "dataset_root": str(tmp_path / "dataset"),
            "task_id": "sample-task",
        },
        run_root=tmp_path / "run",
        run_id="sample-task",
        prompt_template="default",
    )

    assert "--agent-kwarg" in command
    assert "temperature=0.7" in command
    assert "top_p=0.8" in command
    assert "top_k=20" in command
    assert "repetition_penalty=1.05" in command
