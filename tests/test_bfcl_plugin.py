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
from agents.benchmarks._bfcl import build_bfcl_tools
from agents.benchmarks.base import BenchmarkConfig

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
