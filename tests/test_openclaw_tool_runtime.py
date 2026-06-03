from __future__ import annotations

import asyncio
import json

from trace_collect.openclaw_tools import execute_trace_tool


def _nested(tool_name: str, payload: dict) -> str:
    return json.dumps({tool_name: payload}, ensure_ascii=False)


class FakeAgent:
    """Minimal ContainerAgent stub that records requests and returns canned responses."""

    def __init__(self, responses: dict[str, dict] | None = None) -> None:
        self.requests: list[dict] = []
        self.timeouts: list[float] = []
        self._responses = responses or {}
        self._default = {"ok": True, "result": "ok", "returncode": 0}

    async def execute(self, request: dict, *, timeout_s: float = 600.0) -> dict:
        self.requests.append(request)
        self.timeouts.append(timeout_s)
        tool = request.get("tool", "")
        return self._responses.get(tool, self._default)


def test_exec_command_sends_correct_request() -> None:
    agent = FakeAgent({"exec": {"ok": True, "result": "hello\n", "returncode": 0}})
    result, success, _ = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="exec",
            tool_args_json=_nested("exec", {"command": "echo hello"}),
            command_timeout_s=10.0,
        )
    )
    assert success is True
    assert "hello" in result
    assert agent.requests[0]["tool"] == "exec"
    assert agent.requests[0]["args"]["command"] == "echo hello"
    assert agent.requests[0]["args"]["timeout"] == 300.0
    assert agent.timeouts == [300.0]


def test_exec_timeout_from_trace_overrides_simulate_default() -> None:
    agent = FakeAgent({"exec": {"ok": True, "result": "hello\n", "returncode": 0}})
    result, success, _ = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="exec",
            tool_args_json=_nested("exec", {"command": "echo hello", "timeout": 123}),
            command_timeout_s=600.0,
        )
    )
    assert success is True
    assert "hello" in result
    assert agent.requests[0]["args"]["timeout"] == 123.0
    assert agent.timeouts == [123.0]


def test_exec_timeout_from_trace_is_capped_like_openclaw() -> None:
    agent = FakeAgent({"exec": {"ok": True, "result": "hello\n", "returncode": 0}})
    result, success, _ = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="exec",
            tool_args_json=_nested("exec", {"command": "echo hello", "timeout": 999}),
            command_timeout_s=10.0,
        )
    )
    assert success is True
    assert "hello" in result
    assert agent.requests[0]["args"]["timeout"] == 600.0
    assert agent.timeouts == [600.0]


def test_read_file_sends_correct_request() -> None:
    agent = FakeAgent({"read_file": {"ok": True, "result": "file content\n"}})
    result, success, _ = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="read_file",
            tool_args_json=_nested("read_file", {"path": "/testbed/foo.py"}),
            command_timeout_s=10.0,
        )
    )
    assert success is True
    assert "file content" in result
    assert agent.requests[0]["tool"] == "read_file"
    assert agent.requests[0]["args"]["path"] == "/testbed/foo.py"


def test_write_file_sends_correct_request() -> None:
    agent = FakeAgent({"write_file": {"ok": True, "result": "Successfully wrote /testbed/out.txt"}})
    result, success, _ = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="write_file",
            tool_args_json=_nested("write_file", {"path": "/testbed/out.txt", "content": "payload"}),
            command_timeout_s=10.0,
        )
    )
    assert success is True
    assert "Successfully wrote" in result
    assert agent.requests[0]["args"]["content"] == "payload"


def test_edit_file_sends_correct_request() -> None:
    agent = FakeAgent({"edit_file": {"ok": True, "result": "Successfully edited /testbed/x.py"}})
    result, success, _ = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="edit_file",
            tool_args_json=_nested("edit_file", {
                "path": "/testbed/x.py",
                "old_text": "foo",
                "new_text": "bar",
            }),
            command_timeout_s=10.0,
        )
    )
    assert success is True
    assert "Successfully edited" in result
    req = agent.requests[0]
    assert req["tool"] == "edit_file"
    assert req["args"]["old_text"] == "foo"
    assert req["args"]["new_text"] == "bar"
    assert req["args"]["replace_all"] is False


def test_list_dir_sends_correct_request() -> None:
    agent = FakeAgent({"list_dir": {"ok": True, "result": "foo.py\nbar.py"}})
    result, success, _ = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="list_dir",
            tool_args_json=_nested("list_dir", {"path": "/testbed"}),
            command_timeout_s=10.0,
        )
    )
    assert success is True
    assert "foo.py" in result


def test_exec_appends_exit_code() -> None:
    agent = FakeAgent({"exec": {"ok": False, "result": "error msg", "returncode": 1}})
    result, success, _ = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="exec",
            tool_args_json=_nested("exec", {"command": "false"}),
            command_timeout_s=10.0,
        )
    )
    assert success is False
    assert "Exit code: 1" in result


def test_unsupported_tool_returns_error() -> None:
    agent = FakeAgent()
    result, success, _ = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="nope_tool",
            tool_args_json="{}",
            command_timeout_s=5.0,
        )
    )
    assert success is False
    assert "Unsupported replay tool" in result


def test_commands_sends_list() -> None:
    agent = FakeAgent({"commands": {"ok": True, "result": "done", "returncode": 0}})
    result, success, _ = asyncio.run(
        execute_trace_tool(
            agent=agent,
            tool_name="exec",
            tool_args_json=_nested("exec", {"commands": ["echo a", "echo b"]}),
            command_timeout_s=10.0,
        )
    )
    assert success is True
    assert agent.requests[0]["tool"] == "commands"
    assert agent.requests[0]["args"]["commands"] == ["echo a", "echo b"]
    assert agent.requests[0]["args"]["timeout"] == 300.0
    assert agent.timeouts == [300.0]
