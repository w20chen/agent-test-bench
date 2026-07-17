from __future__ import annotations

import asyncio
import json
import shlex
from pathlib import Path

from agents.openclaw._session_runner import TraceCollectorHook
from trace_collect.pytest_script_capture import (
    capture_pytest_scripts_before_tool,
    extract_pytest_targets,
    finalize_pytest_capture,
)


class _StubResponse:
    content = ""
    finish_reason = "tool_calls"
    reasoning_content = None
    extra = None


class _StubToolCall:
    def __init__(self, tool_call_id: str, name: str, arguments: dict[str, object]) -> None:
        self.id = tool_call_id
        self.name = name
        self.arguments = arguments


class _StubContext:
    def __init__(
        self,
        *,
        iteration: int,
        messages: list[dict[str, object]],
        tool_calls: list[_StubToolCall] | None = None,
        tool_events: list[dict[str, object]] | None = None,
    ) -> None:
        self.iteration = iteration
        self.messages = messages
        self.tool_calls = tool_calls or []
        self.tool_events = tool_events or []
        self.usage = {"prompt_tokens": 1, "completion_tokens": 1}
        self.response = _StubResponse()
        self.malformed_retry_count = 0


def test_extract_pytest_targets_from_python_module_command() -> None:
    _cwd, targets, warnings = extract_pytest_targets(
        "cd /testbed && python -m pytest tests/test_alpha.py::TestA::test_one "
        "-xvs 2>&1 | head -100"
    )

    assert targets == ["tests/test_alpha.py"]
    assert warnings == []


def test_extract_pytest_targets_keeps_target_after_no_value_flags() -> None:
    _cwd, targets, warnings = extract_pytest_targets(
        "python -m pytest --cache-clear --self-contained-html "
        "--benchmark-only tests/test_alpha.py"
    )

    assert targets == ["tests/test_alpha.py"]
    assert warnings == []


def test_capture_pytest_scripts_copies_target_file_and_records_timing(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    test_file = project / "tests" / "test_alpha.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_one():\n    assert True\n", encoding="utf-8")

    capture = capture_pytest_scripts_before_tool(
        capture_root=tmp_path / "pytest_scripts",
        project_root=project,
        iteration=6,
        tool_call_id="call/pytest:1",
        tool_name="exec",
        tool_args={
            "command": (
                f"cd {shlex.quote(str(project))} "
                "&& python -m pytest tests/test_alpha.py -q"
            ),
        },
    )

    assert capture is not None
    copied = capture.directory / "files" / "tests" / "test_alpha.py"
    assert copied.read_text(encoding="utf-8") == test_file.read_text(encoding="utf-8")

    finalize_pytest_capture(
        capture,
        action_id="tool_6_call-pytest-1",
        ts_start=10.0,
        ts_end=12.5,
        duration_ms=2500.0,
        success=True,
    )

    manifest = json.loads(capture.manifest_path.read_text(encoding="utf-8"))
    assert manifest["capture_stage"] == "complete"
    assert manifest["duration_ms"] == 2500.0
    assert manifest["action_id"] == "tool_6_call-pytest-1"
    assert manifest["files"][0]["relative_path"] == "tests/test_alpha.py"

    index_lines = (capture.directory.parent / "index.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    index = json.loads(index_lines[0])
    assert index["duration_ms"] == 2500.0
    assert index["file_count"] == 1
    assert index["manifest"].endswith("manifest.json")


def test_capture_pytest_scripts_does_not_discover_files_for_django_test(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    test_file = project / "tests" / "test_alpha.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_one():\n    assert True\n", encoding="utf-8")

    capture = capture_pytest_scripts_before_tool(
        capture_root=tmp_path / "pytest_scripts",
        project_root=project,
        iteration=2,
        tool_call_id="call_django",
        tool_name="exec",
        tool_args={"command": "python -m django test tests"},
    )

    assert capture is not None
    manifest = json.loads(capture.manifest_path.read_text(encoding="utf-8"))
    assert manifest["files"] == []
    assert "no pytest executable" in manifest["warnings"][0]


def test_trace_collector_hook_captures_pytest_and_updates_timing(
    tmp_path: Path,
) -> None:
    asyncio.run(_drive_trace_collector_hook_captures_pytest(tmp_path))


async def _drive_trace_collector_hook_captures_pytest(tmp_path: Path) -> None:
    project = tmp_path / "project"
    test_file = project / "tests" / "test_beta.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_beta():\n    assert 1\n", encoding="utf-8")

    trace_file = tmp_path / "attempt_1" / "trace.jsonl"
    hook = TraceCollectorHook(
        trace_file,
        instance_id="case-1",
        pytest_capture_dir=trace_file.parent / "pytest_scripts",
        pytest_project_root=project,
    )

    messages = [{"role": "user", "content": "run tests"}]
    await hook.before_iteration(_StubContext(iteration=3, messages=messages))

    tool_call = _StubToolCall(
        "call_abc123",
        "exec",
        {
            "command": (
                f"cd {shlex.quote(str(project))} "
                "&& python -m pytest tests/test_beta.py -q"
            )
        },
    )
    await hook.before_execute_tools(
        _StubContext(iteration=3, messages=messages, tool_calls=[tool_call])
    )

    await hook.after_iteration(
        _StubContext(
            iteration=3,
            messages=[
                *messages,
                {
                    "role": "tool",
                    "tool_call_id": "call_abc123",
                    "name": "exec",
                    "content": "passed",
                },
            ],
            tool_calls=[tool_call],
            tool_events=[
                {
                    "tc_id": "call_abc123",
                    "wall_ms": 42.5,
                    "start_mono": 1.0,
                }
            ],
        )
    )
    hook.close()

    capture_dir = trace_file.parent / "pytest_scripts"
    invocation_dirs = [p for p in capture_dir.iterdir() if p.is_dir()]
    assert len(invocation_dirs) == 1
    manifest = json.loads(
        (invocation_dirs[0] / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["iteration"] == 3
    assert manifest["duration_ms"] == 42.5
    assert manifest["success"] is True
    assert manifest["files"][0]["relative_path"] == "tests/test_beta.py"
    assert (invocation_dirs[0] / "files" / "tests" / "test_beta.py").exists()


def test_trace_collector_hook_writes_failure_manifest_on_capture_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    asyncio.run(_drive_trace_collector_hook_writes_failure_manifest(tmp_path, monkeypatch))


async def _drive_trace_collector_hook_writes_failure_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    trace_file = tmp_path / "attempt_1" / "trace.jsonl"
    hook = TraceCollectorHook(
        trace_file,
        instance_id="case-1",
        pytest_capture_dir=trace_file.parent / "pytest_scripts",
        pytest_project_root=project,
    )

    def fail_capture(**_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(
        "agents.openclaw._session_runner.capture_pytest_scripts_before_tool",
        fail_capture,
    )

    messages = [{"role": "user", "content": "run tests"}]
    await hook.before_iteration(_StubContext(iteration=5, messages=messages))
    tool_call = _StubToolCall(
        "call_fail",
        "exec",
        {"command": "python -m pytest tests/test_missing.py -q"},
    )
    await hook.before_execute_tools(
        _StubContext(iteration=5, messages=messages, tool_calls=[tool_call])
    )
    await hook.after_iteration(
        _StubContext(
            iteration=5,
            messages=[
                *messages,
                {
                    "role": "tool",
                    "tool_call_id": "call_fail",
                    "name": "exec",
                    "content": "passed",
                },
            ],
            tool_calls=[tool_call],
            tool_events=[
                {
                    "tc_id": "call_fail",
                    "wall_ms": 99.0,
                    "start_mono": 1.0,
                }
            ],
        )
    )
    hook.close()

    capture_dir = trace_file.parent / "pytest_scripts"
    manifest_path = (
        capture_dir
        / "iter_0005_exec-pytest_call_fail"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["capture_stage"] == "capture_failed"
    assert "disk full" in manifest["capture_error"]
    assert manifest["duration_ms"] == 99.0

    index_lines = (capture_dir / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(index_lines) == 1
    index = json.loads(index_lines[0])
    assert index["capture_stage"] == "capture_failed"
    assert index["file_count"] == 0
    assert index["duration_ms"] == 99.0
