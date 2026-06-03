from __future__ import annotations

from pathlib import Path

from agents.terminal_bench.runner import TerminalBenchRunner
from llm_call.openclaw import UnifiedProvider


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
