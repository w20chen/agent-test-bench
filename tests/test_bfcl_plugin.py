"""Offline tests for the BFCL host-mode benchmark plugins.

These exercise dataset loading and tool construction against the real BFCL repo
(no LLM, no network). They skip gracefully when the BFCL package is not
importable (e.g. CI without the gorilla checkout).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agents.benchmarks import get_benchmark_class
from agents.benchmarks._bfcl import BFCLOpenClawRunner, build_bfcl_tools
from agents.benchmarks.base import BenchmarkConfig
from agents.openclaw.providers.base import LLMProvider, LLMResponse

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "benchmarks" / "bfcl-multi-turn-base.yaml"


def _plugin():
    cls = get_benchmark_class("bfcl-multi-turn-base")
    try:
        return cls(BenchmarkConfig.from_yaml(CONFIG))
    except FileNotFoundError as exc:  # BFCL repo not present
        pytest.skip(str(exc))


def _load_tasks(plugin):
    try:
        return plugin.load_tasks()
    except ImportError as exc:  # BFCL runtime deps missing
        pytest.skip(f"BFCL deps unavailable: {exc}")


def test_load_tasks_shape():
    plugin = _plugin()
    tasks = _load_tasks(plugin)
    assert tasks, "expected at least one multi_turn_base task"
    task = tasks[0]
    assert task["instance_id"]
    assert task["_bfcl_category"] == "multi_turn_base"
    entry = task["_bfcl_entry"]
    assert entry["function"], "function docs should be populated"
    assert entry["involved_classes"], "involved_classes should be set"
    assert isinstance(entry["question"], list) and entry["question"]


def test_build_tools_and_execute():
    plugin = _plugin()
    tasks = _load_tasks(plugin)
    entry = tasks[0]["_bfcl_entry"]

    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call,
    )

    _, instances = execute_multi_turn_func_call(
        [],
        entry.get("initial_config", {}),
        entry["involved_classes"],
        "test_model",
        entry["id"],
        long_context=False,
    )
    tools = build_bfcl_tools(entry, instances)
    assert tools, "expected non-empty BFCL tool set"

    by_name = {tool.name: tool for tool in tools}
    # multi_turn_base entry 0 involves GorillaFileSystem, which exposes pwd().
    assert "pwd" in by_name
    result = asyncio.run(by_name["pwd"].execute())
    assert isinstance(result, str)
    assert not result.startswith("Error"), result


class _RecordingProvider(LLMProvider):
    """Stub LLM that records the messages it receives and ends each turn.

    Returns no tool calls so every conversation turn completes in exactly one
    LLM call, making per-turn message visibility easy to assert.
    """

    def __init__(self) -> None:
        super().__init__(api_key="stub", api_base="stub")
        self.calls: list[str] = []

    async def chat(self, messages, tools=None, model=None, max_tokens=4096,
                   temperature=0.7, reasoning_effort=None, tool_choice=None):
        import json

        self.calls.append(json.dumps(messages, ensure_ascii=False, default=str))
        return LLMResponse(
            content="turn complete",
            tool_calls=[],
            finish_reason="stop",
            usage={"prompt_tokens": 1, "completion_tokens": 1},
        )

    def get_default_model(self) -> str:
        return "stub-model"


def test_multiturn_only_sees_current_and_past_turns(tmp_path):
    """OpenClaw must see one dataset turn at a time: at turn k the LLM sees
    turns 0..k but never a future turn k+1. This is a pure OpenClaw-side test
    (no BFCL, no network)."""
    provider = _RecordingProvider()
    runner = BFCLOpenClawRunner(
        provider=provider,
        workspace_base=tmp_path,
        max_iterations=5,
        context_window_tokens=8192,
        model="stub-model",
        benchmark_slug="bfcl-test",
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()
    markers = ["TURN_ALPHA", "TURN_BETA", "TURN_GAMMA"]

    asyncio.run(
        runner._drive_conversation(
            test_id="visibility",
            trace_path=tmp_path / "trace.jsonl",
            workspace=workspace,
            turn_texts=list(markers),
            tools=[],
            prompt_template="default",
        )
    )

    # One LLM call per turn (no tool calls), so 3 turns -> 3 calls.
    assert len(provider.calls) == len(markers), provider.calls
    for call_idx, blob in enumerate(provider.calls):
        for marker_idx, marker in enumerate(markers):
            if marker_idx <= call_idx:
                assert marker in blob, (
                    f"call {call_idx} should include current/past turn {marker_idx}"
                )
            else:
                assert marker not in blob, (
                    f"call {call_idx} LEAKED future turn {marker_idx} ({marker})"
                )
