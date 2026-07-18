from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agents.openclaw._session_runner import TraceCollectorHook
from agents.openclaw.tools.shell import ExecTool
from trace_collect.pytest_runtime_prediction import (
    HIDDEN_RUNTIME_DIR_ARG,
    compute_pytest_predictions,
    finalize_pytest_runtime_prediction,
    global_history_median,
    historical_duration,
    is_pytest_tool_call,
    merge_pytest_runtime_environment,
    predict_test_duration,
    prepare_pytest_runtime_environment,
    prepare_pytest_runtime_prediction_before_tool,
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


def test_pytest_command_recognition() -> None:
    assert is_pytest_tool_call("exec", {"command": "pytest tests/"})
    assert is_pytest_tool_call("exec", {"command": "python -m pytest tests/"})
    assert is_pytest_tool_call("exec", {"command": "python3 -m pytest tests/test_x.py -v"})
    assert is_pytest_tool_call(
        "exec",
        {"command": "cd /repo && python -m pytest tests/ -v"},
    )
    assert not is_pytest_tool_call("exec", {"command": "python -m pip install pytest"})
    assert not is_pytest_tool_call("exec", {"command": "python -m django test tests"})
    assert not is_pytest_tool_call("exec", {"command": "PYTHONPATH=src pytest tests"})
    assert not is_pytest_tool_call(
        "exec",
        {"command": "export PYTHONPATH=src && python -m pytest tests"},
    )


def test_history_median_and_fallbacks() -> None:
    history = {
        "tests": {
            "tests/test_a.py::test_1": {"durations": [5.1, 5.3, 5.2]},
            "tests/test_b.py::test_2": {"durations": [58.2, 61.4, 60.1]},
        }
    }

    assert historical_duration(history, "tests/test_a.py::test_1") == pytest.approx(5.2)
    assert global_history_median(history) == pytest.approx(31.75)

    predicted, source = predict_test_duration(history, "tests/test_a.py::test_new")
    assert source == "file"
    assert predicted == pytest.approx(5.2)

    predicted, source = predict_test_duration(history, "tests/test_c.py::test_new")
    assert source == "project"
    assert predicted == pytest.approx(31.75)


def test_prediction_methods_do_not_use_current_run() -> None:
    history = {
        "tests": {
            "tests/test_a.py::test_1": {"durations": [1.0]},
            "tests/test_a.py::test_2": {"durations": [2.0]},
        },
        "commands": {
            "python -m pytest tests/test_a.py": {"durations": [4.0]},
        },
    }

    predictions = compute_pytest_predictions(
        history=history,
        command="python -m pytest tests/test_a.py",
        nodeids=["tests/test_a.py::test_1", "tests/test_a.py::test_2"],
    )

    assert predictions["prediction_last_run_s"] == pytest.approx(4.0)
    assert predictions["prediction_test_count_s"] == pytest.approx(3.0)
    assert predictions["prediction_per_test_s"] == pytest.approx(3.0)


def test_missing_and_corrupt_history_degrade_safely(tmp_path: Path) -> None:
    invocation = tmp_path / "pytest_runtime" / "iter_0001_exec-pytest_call_a"
    invocation.mkdir(parents=True)
    (invocation / "pytest_runtime.json").write_text(
        json.dumps(
            {
                "collected_count": 1,
                "exit_code": 0,
                "tests": [
                    {
                        "nodeid": "tests/test_a.py::test_1",
                        "duration_s": 0.01,
                        "outcome": "passed",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "pytest_runtime" / "history.json").write_text("{bad", encoding="utf-8")
    record = prepare_pytest_runtime_prediction_before_tool(
        prediction_root=tmp_path / "pytest_runtime",
        iteration=1,
        tool_call_id="call_a",
        tool_name="exec",
        tool_args={"command": "python -m pytest tests/test_a.py"},
    )
    assert record is not None
    record.directory = invocation

    payload = finalize_pytest_runtime_prediction(
        record,
        prediction_root=tmp_path / "pytest_runtime",
        action_id="tool_1_call_a",
        ts_start=1.0,
        ts_end=1.2,
        duration_ms=200.0,
        success=True,
        tool_result="passed\nExit code: 0",
    )

    assert payload["prediction_last_run_s"] is None
    assert payload["prediction_test_count_s"] is None
    assert payload["prediction_per_test_s"] is None
    assert payload["warnings"]
    history = json.loads((tmp_path / "pytest_runtime" / "history.json").read_text())
    assert history["tests"]["tests/test_a.py::test_1"]["durations"] == [0.01]


def test_notrun_tests_are_not_added_to_history(tmp_path: Path) -> None:
    invocation = tmp_path / "pytest_runtime" / "iter_0001_exec-pytest_call_a"
    invocation.mkdir(parents=True)
    (invocation / "pytest_runtime.json").write_text(
        json.dumps(
            {
                "collected_count": 2,
                "exit_code": 1,
                "tests": [
                    {
                        "nodeid": "tests/test_a.py::test_ran",
                        "duration_s": 0.02,
                        "outcome": "passed",
                    },
                    {
                        "nodeid": "tests/test_a.py::test_notrun",
                        "duration_s": 0.0,
                        "outcome": "notrun",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    record = prepare_pytest_runtime_prediction_before_tool(
        prediction_root=tmp_path / "pytest_runtime",
        iteration=1,
        tool_call_id="call_a",
        tool_name="exec",
        tool_args={"command": "python -m pytest tests/test_a.py -x"},
    )
    assert record is not None
    record.directory = invocation

    payload = finalize_pytest_runtime_prediction(
        record,
        prediction_root=tmp_path / "pytest_runtime",
        action_id="tool_1_call_a",
        ts_start=1.0,
        ts_end=1.2,
        duration_ms=200.0,
        success=False,
        tool_result="failed\nExit code: 1",
    )

    assert "tests/test_a.py::test_notrun" in payload["collected_tests"]
    history = json.loads((tmp_path / "pytest_runtime" / "history.json").read_text())
    assert "tests/test_a.py::test_ran" in history["tests"]
    assert "tests/test_a.py::test_notrun" not in history["tests"]


def test_runtime_environment_merges_without_wrapping_command(tmp_path: Path) -> None:
    overrides = prepare_pytest_runtime_environment(invocation_dir=tmp_path)
    env = merge_pytest_runtime_environment(
        {"PYTHONPATH": "oldpath", "PYTEST_PLUGINS": "existing"},
        overrides,
    )

    assert env["PYTEST_PLUGINS"].endswith(",openclaw_pytest_runtime_plugin")
    assert env["PYTHONPATH"].endswith("oldpath")
    assert env["OPENCLAW_PYTEST_RUNTIME_JSON"].endswith("pytest_runtime.json")


def test_three_prediction_methods_with_new_test_fallback() -> None:
    history = {
        "tests": {
            "tests/test_a.py::test_1": {"durations": [1.0, 1.2, 1.1]},
            "tests/test_b.py::test_2": {"durations": [10.0]},
        },
        "commands": {
            "pytest tests": {"durations": [13.0]},
        },
    }

    predictions = compute_pytest_predictions(
        history=history,
        command="pytest tests",
        nodeids=[
            "tests/test_a.py::test_1",
            "tests/test_a.py::test_new",
            "tests/test_c.py::test_new",
        ],
    )

    project_median = statistics_median([1.0, 1.2, 1.1, 10.0])
    assert predictions["prediction_last_run_s"] == pytest.approx(13.0)
    assert predictions["prediction_test_count_s"] == pytest.approx(3 * project_median)
    assert predictions["prediction_per_test_s"] == pytest.approx(
        1.1 + 1.1 + project_median
    )


def statistics_median(values: list[float]) -> float:
    values = sorted(values)
    mid = len(values) // 2
    return (values[mid - 1] + values[mid]) / 2


def test_exec_tool_collects_pytest_runtime_json_for_small_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    tests_dir = project / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_sample.py").write_text(
        "\n".join(
            [
                "import time",
                "def test_short():",
                "    assert True",
                "def test_medium():",
                "    time.sleep(0.01)",
                "    assert True",
                "def test_longer():",
                "    time.sleep(0.02)",
                "    assert True",
                "",
            ]
        ),
        encoding="utf-8",
    )
    invocation_dir = tmp_path / "pytest_runtime" / "iter_0000_exec-pytest_call_smoke"
    tool = ExecTool(working_dir=str(project), timeout=30)

    result = asyncio.run(
        tool.execute(
            command="python -m pytest tests -q",
            **{HIDDEN_RUNTIME_DIR_ARG: str(invocation_dir)},
        )
    )

    assert "Exit code: 0" in result
    runtime = json.loads((invocation_dir / "pytest_runtime.json").read_text())
    assert runtime["collected_count"] == 3
    nodeids = {test["nodeid"].replace("\\", "/") for test in runtime["tests"]}
    assert any(nodeid.endswith("tests/test_sample.py::test_short") for nodeid in nodeids)
    assert any(nodeid.endswith("tests/test_sample.py::test_medium") for nodeid in nodeids)
    assert any(nodeid.endswith("tests/test_sample.py::test_longer") for nodeid in nodeids)
    assert all(test["duration_s"] >= 0 for test in runtime["tests"])


def test_small_project_generates_predictions_after_history(tmp_path: Path) -> None:
    project = tmp_path / "project"
    tests_dir = project / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_sample.py").write_text(
        "\n".join(
            [
                "import time",
                "def test_short():",
                "    assert True",
                "def test_medium():",
                "    time.sleep(0.005)",
                "    assert True",
                "def test_longer():",
                "    time.sleep(0.01)",
                "    assert True",
                "",
            ]
        ),
        encoding="utf-8",
    )
    prediction_root = tmp_path / "pytest_runtime"
    tool = ExecTool(working_dir=str(project), timeout=30)

    payloads = []
    for iteration in range(2):
        tool_args = {"command": "python -m pytest tests -q"}
        record = prepare_pytest_runtime_prediction_before_tool(
            prediction_root=prediction_root,
            iteration=iteration,
            tool_call_id=f"call_{iteration}",
            tool_name="exec",
            tool_args=tool_args,
        )
        assert record is not None
        result = asyncio.run(tool.execute(**tool_args))
        assert "Exit code: 0" in result
        payloads.append(
            finalize_pytest_runtime_prediction(
                record,
                prediction_root=prediction_root,
                action_id=f"tool_{iteration}_call_{iteration}",
                ts_start=float(iteration),
                ts_end=float(iteration) + 0.1,
                duration_ms=100.0,
                success=True,
                tool_result=result,
                working_directory=str(project),
            )
        )

    first, second = payloads
    assert first["prediction_per_test_s"] is None
    assert second["prediction_last_run_s"] == pytest.approx(0.1)
    assert second["prediction_test_count_s"] is not None
    assert second["prediction_per_test_s"] is not None
    assert (prediction_root / "history.json").exists()
    assert len((prediction_root / "predictions.jsonl").read_text().splitlines()) == 2


def test_trace_hook_does_not_persist_hidden_runtime_args(tmp_path: Path) -> None:
    asyncio.run(_drive_trace_hook_does_not_persist_hidden_runtime_args(tmp_path))


async def _drive_trace_hook_does_not_persist_hidden_runtime_args(tmp_path: Path) -> None:
    trace_file = tmp_path / "attempt_1" / "trace.jsonl"
    hook = TraceCollectorHook(
        trace_file,
        instance_id="case-1",
        pytest_runtime_dir=trace_file.parent / "pytest_runtime",
        pytest_project_root=tmp_path,
    )
    messages = [{"role": "user", "content": "run tests"}]
    await hook.before_iteration(_StubContext(iteration=0, messages=messages))
    tool_call = _StubToolCall(
        "call_pytest",
        "exec",
        {"command": "python -m pytest tests -q"},
    )
    await hook.before_execute_tools(
        _StubContext(iteration=0, messages=messages, tool_calls=[tool_call])
    )
    assert HIDDEN_RUNTIME_DIR_ARG in tool_call.arguments

    await hook.after_iteration(
        _StubContext(
            iteration=0,
            messages=[
                *messages,
                {
                    "role": "tool",
                    "tool_call_id": "call_pytest",
                    "name": "exec",
                    "content": "Exit code: 0",
                },
            ],
            tool_calls=[tool_call],
            tool_events=[
                {
                    "tc_id": "call_pytest",
                    "wall_ms": 10.0,
                    "start_mono": 1.0,
                }
            ],
        )
    )
    hook.close()

    records = [json.loads(line) for line in trace_file.read_text().splitlines()]
    llm = next(record for record in records if record.get("action_type") == "llm_call")
    tool = next(record for record in records if record.get("action_type") == "tool_exec")
    raw_args = llm["data"]["raw_response"]["choices"][0]["message"]["tool_calls"][0][
        "function"
    ]["arguments"]
    assert HIDDEN_RUNTIME_DIR_ARG not in raw_args
    assert HIDDEN_RUNTIME_DIR_ARG not in tool["data"]["tool_args"]
