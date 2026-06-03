"""Targeted tests for SessionRunner workspace forwarding."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("agents.openclaw._session_runner")

import agents.openclaw._session_runner as session_runner_mod

from agents.openclaw._session_runner import SessionRunner


class _FakeCollector:
    def __init__(self, _bus) -> None:
        self.started = False

    async def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    async def wait_for_result(self, _session_key: str) -> str:
        return "done"


class _FakeAgentLoop:
    last_init: dict[str, object] | None = None

    def __init__(self, **kwargs) -> None:
        type(self).last_init = kwargs
        self.memory_consolidator = SimpleNamespace(_event_callback=None)
        self.context = SimpleNamespace(
            skills=SimpleNamespace(_event_callback=None),
        )
        self.sessions = SimpleNamespace(_event_callback=None)
        self._mcp_event_callback = None
        self._event_callback = None

    async def run(self) -> None:
        return None

    def stop(self) -> None:
        return None


def test_session_runner_forwards_project_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_runner_mod, "ResultCollector", _FakeCollector)
    monkeypatch.setattr(session_runner_mod, "AgentLoop", _FakeAgentLoop)
    monkeypatch.setattr(
        session_runner_mod,
        "inject_event_callbacks",
        lambda *args, **kwargs: None,
    )

    runner = SessionRunner(
        provider=SimpleNamespace(get_default_model=lambda: "qwen-plus-latest"),
        model="qwen-plus-latest",
        max_iterations=1,
    )
    state_workspace = tmp_path / "state"
    tool_workspace = tmp_path / "tool"
    project_workspace = Path("/testbed")

    asyncio.run(
        runner.run(
            prompt="fix bug",
            workspace=state_workspace,
            tool_workspace=tool_workspace,
            project_workspace=project_workspace,
            session_key="eval:test",
            trace_file=tmp_path / "trace.jsonl",
        )
    )

    assert _FakeAgentLoop.last_init is not None
    assert _FakeAgentLoop.last_init["workspace"] == state_workspace
    assert _FakeAgentLoop.last_init["tool_workspace"] == tool_workspace
    assert _FakeAgentLoop.last_init["project_workspace"] == project_workspace
