from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from trace_collect import pytest_runtime_prediction as pred_mod
from agents.openclaw._session_runner import TraceCollectorHook
from agents.openclaw.tools.shell import ExecTool
from trace_collect.pytest_runtime_prediction import (
    HIDDEN_RUNTIME_DIR_ARG,
    build_pytest_collect_only_invocation,
    compute_pytest_predictions,
    finalize_pytest_runtime_prediction,
    format_pytest_prediction_summary,
    global_history_median,
    global_overhead_median,
    global_unknown_test_median,
    historical_duration,
    is_pytest_tool_call,
    merge_pytest_runtime_environment,
    normalize_pytest_command,
    predict_test_duration,
    prepare_pytest_runtime_environment,
    prepare_pytest_runtime_prediction_before_tool,
    update_pytest_history,
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


def test_normalized_command_merges_interpreter_and_stdout_wrappers() -> None:
    left = normalize_pytest_command(
        "cd /testbed && /usr/local/bin/python -m pytest tests/test_a.py -v 2>&1 | head -80"
    )
    right = normalize_pytest_command(
        "cd /testbed && python3 -m pytest tests/test_a.py --no-header 2>&1 | grep FAILED"
    )

    assert left == "cd /testbed && pytest tests/test_a.py"
    assert right == left


def test_normalized_command_keeps_timeout_as_strict_key() -> None:
    timeout_60 = normalize_pytest_command(
        "cd /testbed && timeout 60 python -m pytest tests/test_a.py -v"
    )
    timeout_1m = normalize_pytest_command(
        "cd /testbed && timeout 1m python -m pytest tests/test_a.py -v"
    )
    timeout_120 = normalize_pytest_command(
        "cd /testbed && timeout 120 python -m pytest tests/test_a.py -v"
    )

    assert timeout_60 == "cd /testbed && timeout=60 pytest tests/test_a.py"
    assert timeout_1m == timeout_60
    assert timeout_120 == "cd /testbed && timeout=120 pytest tests/test_a.py"
    assert timeout_60 != timeout_120


def test_normalized_command_sorts_order_independent_selection_parts() -> None:
    left = normalize_pytest_command(
        "cd /testbed && python -m pytest tests/test_b.py tests/test_a.py "
        "--ignore=tests/slow.py --ignore tests/flaky.py -v"
    )
    right = normalize_pytest_command(
        "cd /testbed && pytest --ignore tests/flaky.py tests/test_a.py "
        "--ignore=tests/slow.py tests/test_b.py"
    )

    assert left == right
    assert left == (
        "cd /testbed && pytest tests/test_a.py tests/test_b.py "
        "--ignore tests/flaky.py --ignore tests/slow.py"
    )


def test_normalized_command_preserves_ordered_selectors_and_unsafe_prelude() -> None:
    with_selector = normalize_pytest_command(
        "cd /testbed && python -m pytest tests/ -k 'a or b' -p no:anyio -v"
    )
    with_prelude = normalize_pytest_command(
        "cd /testbed && git stash && python -m pytest tests/test_a.py -v 2>&1 | tail -20"
    )

    assert with_selector == "cd /testbed && pytest tests/ -k a or b -p no:anyio"
    assert "git stash" in with_prelude
    assert "| tail -20" in with_prelude


def test_normalized_command_keeps_known_value_flags_out_of_targets() -> None:
    normalized = normalize_pytest_command(
        "cd /testbed && python -m pytest tests/test_a.py "
        "--junitxml out.xml --cov src --cov-report term -v"
    )

    assert normalized == (
        "cd /testbed && pytest tests/test_a.py "
        "--junitxml out.xml --cov src --cov-report term"
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
            "pytest tests/test_a.py": {
                "durations": [4.0],
                "collected_counts": [2],
            },
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
    assert predictions["prediction_recommended_s"] == pytest.approx(4.0)
    assert predictions["prediction_recommended_method"] == "last_run"
    assert predictions["prediction_reliability"]["level"] == "high"


def test_command_history_ignores_runs_without_observed_tests() -> None:
    history = update_pytest_history(
        history={},
        command="python -m pytest tests/test_a.py -v",
        total_duration_s=0.02,
        tests=[],
    )

    assert history["commands"] == {}


def test_command_history_records_collected_counts_for_observed_runs() -> None:
    history = update_pytest_history(
        history={},
        command="python -m pytest tests/test_a.py -v",
        total_duration_s=1.5,
        tests=[
            {"nodeid": "tests/test_a.py::test_1", "duration_s": 0.1},
            {"nodeid": "tests/test_a.py::test_2", "duration_s": 0.2},
        ],
    )

    command = history["commands"]["pytest tests/test_a.py"]
    assert command["durations"] == [1.5]
    assert command["collected_counts"] == [2.0]
    assert command["last_observed_test_count"] == 2


def test_recommended_uses_per_test_when_last_run_count_changes() -> None:
    history = {
        "tests": {
            "tests/test_a.py::test_1": {"durations": [1.0]},
            "tests/test_a.py::test_2": {"durations": [2.0]},
        },
        "commands": {
            "pytest tests/test_a.py": {
                "durations": [100.0],
                "collected_counts": [10],
            }
        },
    }

    predictions = compute_pytest_predictions(
        history=history,
        command="python -m pytest tests/test_a.py",
        nodeids=["tests/test_a.py::test_1", "tests/test_a.py::test_2"],
    )

    reliability = predictions["prediction_reliability"]
    assert predictions["prediction_last_run_s"] == pytest.approx(100.0)
    assert predictions["prediction_recommended_method"] == "per_test"
    assert predictions["prediction_recommended_s"] == pytest.approx(3.0)
    assert reliability["level"] == "high"
    assert reliability["collected_count_delta_ratio"] == pytest.approx(0.8)
    assert "same_command_collected_count_changed" in reliability["reasons"]


def test_recommended_last_run_without_collected_count_is_medium() -> None:
    history = {
        "tests": {
            "tests/test_a.py::test_1": {"durations": [1.0]},
        },
        "commands": {
            "pytest tests/test_a.py": {"durations": [2.0]},
        },
    }

    predictions = compute_pytest_predictions(
        history=history,
        command="python -m pytest tests/test_a.py",
        nodeids=["tests/test_a.py::test_1"],
    )

    reliability = predictions["prediction_reliability"]
    assert predictions["prediction_recommended_method"] == "last_run"
    assert predictions["prediction_recommended_s"] == pytest.approx(2.0)
    assert reliability["level"] == "medium"
    assert "previous_collected_count_unavailable" in reliability["reasons"]


def test_recommended_uses_medium_file_history_path() -> None:
    history = {
        "tests": {
            "tests/test_a.py::test_existing": {"durations": [1.0]},
        },
    }

    predictions = compute_pytest_predictions(
        history=history,
        command="python -m pytest tests/test_a.py",
        nodeids=["tests/test_a.py::test_new"],
    )

    reliability = predictions["prediction_reliability"]
    assert predictions["prediction_recommended_method"] == "per_test"
    assert predictions["prediction_recommended_s"] == pytest.approx(1.0)
    assert reliability["level"] == "medium"
    assert reliability["file_fallback_ratio"] == 1.0


def test_recommended_marks_cold_start_unknown_fallback_low_reliability() -> None:
    history = {
        "tests": {
            "tests/test_known.py::test_fast": {"durations": [0.1]},
        },
        "overheads": {"durations": [1.0]},
        "unknown_tests": {"durations": [5.0]},
    }

    predictions = compute_pytest_predictions(
        history=history,
        command="python -m pytest tests/test_new.py",
        nodeids=["tests/test_new.py::test_new"],
    )

    reliability = predictions["prediction_reliability"]
    assert predictions["prediction_recommended_method"] == "unknown_test_fallback"
    assert predictions["prediction_recommended_s"] == pytest.approx(6.0)
    assert reliability["level"] == "low"
    assert reliability["known_node_ratio"] == 0.0
    assert reliability["unknown_fallback_ratio"] == 1.0


def test_recommended_marks_empty_collection_as_error() -> None:
    predictions = compute_pytest_predictions(
        history={},
        command="python -m pytest tests/test_a.py",
        nodeids=[],
    )

    reliability = predictions["prediction_reliability"]
    assert predictions["prediction_recommended_method"] == "none"
    assert predictions["prediction_recommended_s"] is None
    assert reliability["level"] == "error"
    assert "no_collected_tests" in reliability["reasons"]


def test_recommended_marks_pre_execution_history_miss_as_coldstart() -> None:
    predictions = compute_pytest_predictions(
        history={},
        command="python -m pytest tests/test_a.py",
        nodeids=["tests/test_a.py::test_new"],
    )

    reliability = predictions["prediction_reliability"]
    assert predictions["prediction_recommended_method"] == "none"
    assert predictions["prediction_recommended_s"] is None
    assert reliability["level"] == "coldstart"
    assert "no_available_prediction" in reliability["reasons"]


def test_per_test_prediction_adds_historical_overhead() -> None:
    history = {
        "tests": {
            "tests/test_a.py::test_1": {"durations": [1.0]},
            "tests/test_a.py::test_2": {"durations": [2.0]},
        },
        "overheads": {"durations": [9.0, 11.0, 10.0]},
    }

    predictions = compute_pytest_predictions(
        history=history,
        command="python -m pytest tests/test_a.py",
        nodeids=["tests/test_a.py::test_1", "tests/test_a.py::test_2"],
    )

    assert global_overhead_median(history) == pytest.approx(10.0)
    assert predictions["prediction_per_test_without_overhead_s"] == pytest.approx(3.0)
    assert predictions["prediction_per_test_overhead_s"] == pytest.approx(10.0)
    assert predictions["prediction_per_test_s"] == pytest.approx(13.0)


def test_per_test_overhead_field_is_null_when_not_used() -> None:
    predictions = compute_pytest_predictions(
        history={"overheads": {"durations": [1.0]}},
        command="python -m pytest tests/test_a.py",
        nodeids=[],
    )

    assert predictions["prediction_per_test_s"] is None
    assert predictions["prediction_per_test_without_overhead_s"] is None
    assert predictions["prediction_per_test_overhead_s"] is None


def test_unknown_test_fallback_is_fourth_prediction_method() -> None:
    history = {
        "tests": {
            "tests/test_known.py::test_fast": {"durations": [0.1]},
        },
        "overheads": {"durations": [1.0]},
        "unknown_tests": {"durations": [7.0, 9.0, 8.0]},
    }

    predictions = compute_pytest_predictions(
        history=history,
        command="python -m pytest tests/test_new.py",
        nodeids=["tests/test_new.py::test_slow"],
    )

    assert global_unknown_test_median(history) == pytest.approx(8.0)
    assert predictions["prediction_per_test_s"] == pytest.approx(1.1)
    assert predictions["per_test_prediction_details"][0]["source"] == "project"
    assert predictions["prediction_unknown_test_fallback_without_overhead_s"] == pytest.approx(8.0)
    assert predictions["prediction_unknown_test_fallback_overhead_s"] == pytest.approx(1.0)
    assert predictions["prediction_unknown_test_fallback_s"] == pytest.approx(9.0)
    assert predictions["unknown_test_fallback_prediction_details"][0]["source"] == "unknown"


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
    assert payload["schema_version"] == 5
    assert payload["warnings"]
    history = json.loads((tmp_path / "pytest_runtime" / "history.json").read_text())
    assert history["schema_version"] == 5
    assert history["tests"]["tests/test_a.py::test_1"]["durations"] == [0.01]
    assert history["overheads"]["durations"] == pytest.approx([0.19])


def test_prediction_json_includes_pytest_output_but_jsonl_stays_compact(
    tmp_path: Path,
) -> None:
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
        tool_result="test output\nExit code: 0",
    )

    assert payload["pytest_output"]["text"] == "test output\nExit code: 0"
    prediction_json = json.loads((invocation / "prediction.json").read_text())
    prediction_jsonl = json.loads(
        (tmp_path / "pytest_runtime" / "predictions.jsonl").read_text()
    )
    assert prediction_json["pytest_output"]["length_chars"] == len(
        "test output\nExit code: 0"
    )
    assert "pytest_output" not in prediction_jsonl
    assert prediction_jsonl["prediction_recommended_method"] == "none"
    assert "prediction_reliability" in prediction_jsonl
    assert "recommended" in prediction_jsonl["absolute_error"]
    assert "recommended" in prediction_jsonl["relative_error"]


def test_pytest_finalize_nonzero_exit_does_not_update_history(
    tmp_path: Path,
) -> None:
    invocation = tmp_path / "pytest_runtime" / "iter_0001_exec-pytest_call_a"
    invocation.mkdir(parents=True)
    (invocation / "pytest_runtime.json").write_text(
        json.dumps(
            {
                "collected_count": 1,
                "exit_code": 2,
                "tests": [
                    {
                        "nodeid": "tests/test_a.py::test_1",
                        "duration_s": 0.01,
                        "outcome": "failed",
                    }
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
        success=False,
        tool_result="failed\nExit code: 2",
    )

    assert payload["history_updated"] is False
    assert not (tmp_path / "pytest_runtime" / "history.json").exists()


def test_pytest_finalize_uses_pre_execution_prediction_snapshot(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pytest_runtime"
    root.mkdir()
    command = "python -m pytest tests/test_a.py"
    (root / "history.json").write_text(
        json.dumps(
            {
                "tests": {},
                "commands": {
                    "pytest tests/test_a.py": {
                        "durations": [3.0],
                        "collected_counts": [1],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    record = prepare_pytest_runtime_prediction_before_tool(
        prediction_root=root,
        iteration=1,
        tool_call_id="call_snapshot",
        tool_name="exec",
        tool_args={"command": command},
    )
    assert record is not None
    pending = json.loads((record.directory / "pending.json").read_text(encoding="utf-8"))
    assert pending["predictions"]["prediction_recommended_s"] == pytest.approx(3.0)
    assert pending["predictions"]["prediction_recommended_method"] == "last_run"
    assert pending["predictions"]["prediction_reliability"]["level"] == "medium"

    (root / "history.json").write_text(
        json.dumps(
            {
                "tests": {
                    "tests/test_a.py::test_1": {"durations": [99.0]},
                },
                "commands": {
                    "pytest tests/test_a.py": {
                        "durations": [99.0],
                        "collected_counts": [1],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (record.directory / "pytest_runtime.json").write_text(
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

    payload = finalize_pytest_runtime_prediction(
        record,
        prediction_root=root,
        action_id="tool_1_call_snapshot",
        ts_start=1.0,
        ts_end=2.0,
        duration_ms=1000.0,
        success=True,
        tool_result="Exit code: 0",
    )

    assert payload["prediction_recommended_s"] == pytest.approx(3.0)
    assert payload["prediction_last_run_s"] == pytest.approx(3.0)
    assert payload["prediction_per_test_s"] is None


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
    assert payload["history_updated"] is False
    assert not (tmp_path / "pytest_runtime" / "history.json").exists()


def test_unknown_history_records_only_tests_that_were_unknown(tmp_path: Path) -> None:
    invocation = tmp_path / "pytest_runtime" / "iter_0001_exec-pytest_call_a"
    invocation.mkdir(parents=True)
    (tmp_path / "pytest_runtime" / "history.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "tests": {
                    "tests/test_a.py::test_known": {"durations": [0.01]},
                },
                "commands": {},
                "overheads": {},
                "unknown_tests": {},
            }
        ),
        encoding="utf-8",
    )
    (invocation / "pytest_runtime.json").write_text(
        json.dumps(
            {
                "collected_count": 2,
                "exit_code": 0,
                "tests": [
                    {
                        "nodeid": "tests/test_a.py::test_known",
                        "duration_s": 0.02,
                        "outcome": "passed",
                    },
                    {
                        "nodeid": "tests/test_new.py::test_slow",
                        "duration_s": 7.0,
                        "outcome": "passed",
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
        tool_args={"command": "python -m pytest tests"},
    )
    assert record is not None
    record.directory = invocation

    finalize_pytest_runtime_prediction(
        record,
        prediction_root=tmp_path / "pytest_runtime",
        action_id="tool_1_call_a",
        ts_start=1.0,
        ts_end=9.0,
        duration_ms=8000.0,
        success=True,
        tool_result="Exit code: 0",
    )

    history = json.loads((tmp_path / "pytest_runtime" / "history.json").read_text())
    assert history["unknown_tests"]["durations"] == pytest.approx([7.0])


def test_unknown_history_uses_pre_execution_history_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "pytest_runtime"
    root.mkdir()
    (root / "history.json").write_text(
        json.dumps(
            {
                "tests": {
                    "tests/test_known.py::test_known": {"durations": [0.01]},
                },
                "commands": {},
                "overheads": {},
                "unknown_tests": {},
            }
        ),
        encoding="utf-8",
    )
    record = prepare_pytest_runtime_prediction_before_tool(
        prediction_root=root,
        iteration=1,
        tool_call_id="call_snapshot",
        tool_name="exec",
        tool_args={"command": "python -m pytest tests"},
    )
    assert record is not None

    (root / "history.json").write_text(
        json.dumps(
            {
                "tests": {
                    "tests/test_known.py::test_known": {"durations": [0.01]},
                    "tests/test_new.py::test_new": {"durations": [99.0]},
                },
                "commands": {},
                "overheads": {},
                "unknown_tests": {},
            }
        ),
        encoding="utf-8",
    )
    (record.directory / "pytest_runtime.json").write_text(
        json.dumps(
            {
                "collected_count": 1,
                "exit_code": 0,
                "tests": [
                    {
                        "nodeid": "tests/test_new.py::test_new",
                        "duration_s": 2.0,
                        "outcome": "passed",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    finalize_pytest_runtime_prediction(
        record,
        prediction_root=root,
        action_id="tool_1_call_snapshot",
        ts_start=1.0,
        ts_end=3.0,
        duration_ms=2000.0,
        success=True,
        tool_result="Exit code: 0",
    )

    history = json.loads((root / "history.json").read_text(encoding="utf-8"))
    assert history["unknown_tests"]["durations"] == pytest.approx([2.0])


def test_realtime_summary_prints_all_prediction_errors() -> None:
    line = format_pytest_prediction_summary(
        {
            "iteration": 3,
            "collected_count": 2,
            "actual_duration_s": 10.0,
            "collect_only_duration_s": 1.25,
            "prediction_last_run_s": 8.0,
            "prediction_test_count_s": 5.0,
            "prediction_per_test_s": 9.0,
            "prediction_unknown_test_fallback_s": 11.0,
            "prediction_recommended_s": 9.0,
            "prediction_recommended_method": "per_test",
            "prediction_reliability": {"level": "high"},
            "relative_error": {
                "last_run": 0.2,
                "test_count": 0.5,
                "per_test": 0.1,
                "unknown_test_fallback": 0.1,
                "recommended": 0.1,
            },
        }
    )

    assert line.startswith("[pytest-predict] ")
    assert "(collect 1.2s)" in line
    assert "last=8.0s(+20.0%)" in line
    assert "count=5.0s(+50.0%)" in line
    assert "→per=9.0s(+10.0%)" in line
    assert "unk=11.0s(+10.0%)" in line
    assert line.endswith("| high")


def test_realtime_summary_prints_coldstart_and_error_reliability() -> None:
    coldstart_line = format_pytest_prediction_summary(
        {
            "iteration": 1,
            "collected_count": 1,
            "actual_duration_s": 1.0,
            "prediction_recommended_s": None,
            "prediction_recommended_method": "none",
            "prediction_reliability": {"level": "coldstart"},
            "relative_error": {},
        }
    )
    error_line = format_pytest_prediction_summary(
        {
            "iteration": 2,
            "collected_count": 0,
            "actual_duration_s": 1.0,
            "prediction_recommended_s": None,
            "prediction_recommended_method": "none",
            "prediction_reliability": {"level": "error"},
            "relative_error": {},
        }
    )

    assert "last=?(?)" in coldstart_line
    assert coldstart_line.endswith("| coldstart")
    assert "last=?(?)" in error_line
    assert error_line.endswith("| error")


def test_runtime_environment_merges_without_wrapping_command(tmp_path: Path) -> None:
    overrides = prepare_pytest_runtime_environment(invocation_dir=tmp_path)
    env = merge_pytest_runtime_environment(
        {"PYTHONPATH": "oldpath", "PYTEST_PLUGINS": "existing"},
        overrides,
    )

    assert env["PYTEST_PLUGINS"].endswith(",openclaw_pytest_runtime_plugin")
    assert env["PYTHONPATH"].endswith("oldpath")
    assert env["OPENCLAW_PYTEST_RUNTIME_JSON"].endswith("pytest_runtime.json")


def test_collect_only_invocation_strips_side_effect_output_flags(tmp_path: Path) -> None:
    invocation = build_pytest_collect_only_invocation(
        "cd project && python -m pytest tests --junitxml out.xml "
        "--cov src --cov-report html --cache-clear -q",
        working_directory=tmp_path,
        artifact_dir=tmp_path / "artifacts",
    )

    assert invocation is not None
    joined = " ".join(invocation.argv)
    assert "--collect-only" in invocation.argv
    assert "--junitxml" not in invocation.argv
    assert "--cov" not in invocation.argv
    assert "--cov-report" not in invocation.argv
    assert "--cache-clear" not in invocation.argv
    assert "cache_dir=" in joined
    assert Path(invocation.working_directory or "").name == "project"
    assert invocation.warnings


def test_collect_only_invocation_rejects_cache_dependent_selectors(
    tmp_path: Path,
) -> None:
    invocation = build_pytest_collect_only_invocation(
        "python -m pytest --lf tests",
        working_directory=tmp_path,
        artifact_dir=tmp_path / "artifacts",
    )

    assert invocation is None


def test_collect_only_invocation_uses_shell_timeout_prefix(tmp_path: Path) -> None:
    invocation = build_pytest_collect_only_invocation(
        "timeout 5 python -m pytest tests",
        working_directory=tmp_path,
        artifact_dir=tmp_path / "artifacts",
    )

    assert invocation is not None
    assert invocation.timeout_s == pytest.approx(5.0)
    assert "timeout" not in invocation.argv


def test_collect_only_invocation_moves_env_prefix_to_env_overrides(
    tmp_path: Path,
) -> None:
    invocation = build_pytest_collect_only_invocation(
        "FOO=bar python -m pytest tests",
        working_directory=tmp_path,
        artifact_dir=tmp_path / "artifacts",
    )

    assert invocation is not None
    assert invocation.env_overrides == {"FOO": "bar"}
    assert invocation.argv[:3] == ["python", "-m", "pytest"]
    assert "FOO=bar" not in invocation.argv


def test_prepare_collect_only_records_nodeids_and_overhead(tmp_path: Path) -> None:
    project = tmp_path / "project"
    tests_dir = project / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_sample.py").write_text(
        "\n".join(
            [
                "def test_one():",
                "    assert True",
                "def test_two():",
                "    assert True",
                "",
            ]
        ),
        encoding="utf-8",
    )
    root = tmp_path / "pytest_runtime"

    record = prepare_pytest_runtime_prediction_before_tool(
        prediction_root=root,
        iteration=1,
        tool_call_id="call_collect",
        tool_name="exec",
        tool_args={"command": "python -m pytest tests -q", "timeout": 30},
        working_directory=project,
    )

    assert record is not None
    pending = json.loads((record.directory / "pending.json").read_text(encoding="utf-8"))
    assert pending["collect_only"]["status"] == "collected"
    assert pending["collect_only"]["duration_s"] >= 0
    assert pending["prediction_overhead_s"] == pytest.approx(
        pending["collect_only"]["duration_s"]
    )
    nodeids = {nodeid.replace("\\", "/") for nodeid in pending["pre_execution_nodeids"]}
    assert any(nodeid.endswith("tests/test_sample.py::test_one") for nodeid in nodeids)
    assert any(nodeid.endswith("tests/test_sample.py::test_two") for nodeid in nodeids)


def test_prepare_collect_only_distinguishes_empty_collection(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "tests").mkdir(parents=True)
    root = tmp_path / "pytest_runtime"

    record = prepare_pytest_runtime_prediction_before_tool(
        prediction_root=root,
        iteration=1,
        tool_call_id="call_empty",
        tool_name="exec",
        tool_args={"command": "python -m pytest tests -q", "timeout": 30},
        working_directory=project,
    )

    assert record is not None
    pending = json.loads((record.directory / "pending.json").read_text(encoding="utf-8"))
    assert pending["collect_only"]["status"] in {
        "collected_empty",
        "collected_with_pytest_exit",
    }
    assert pending["pre_execution_nodeids"] == []
    assert pending["pre_execution_collected_count"] == 0


def test_collect_only_uses_exec_tool_runtime_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    (project / "tests").mkdir(parents=True)
    userbase = tmp_path / "userbase"
    userbase_bin = userbase / "bin"
    captured: dict[str, object] = {}

    class _Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(*args: object, **kwargs: object) -> _Completed:
        env = kwargs["env"]
        assert isinstance(env, dict)
        runtime_json = Path(str(env["OPENCLAW_PYTEST_RUNTIME_JSON"]))
        runtime_json.parent.mkdir(parents=True, exist_ok=True)
        runtime_json.write_text(
            json.dumps(
                {
                    "collected_count": 1,
                    "exit_code": 0,
                    "tests": [
                        {
                            "nodeid": "tests/test_sample.py::test_one",
                            "duration_s": 0.0,
                            "outcome": "notrun",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        captured["env"] = env
        return _Completed()

    monkeypatch.setenv("OPENCLAW_TASK_USERBASE", str(userbase))
    monkeypatch.setenv("PYTHONPATH", "host-pythonpath-should-not-leak")
    monkeypatch.setattr(pred_mod.subprocess, "run", _fake_run)

    record = prepare_pytest_runtime_prediction_before_tool(
        prediction_root=tmp_path / "pytest_runtime",
        iteration=1,
        tool_call_id="call_env",
        tool_name="exec",
        tool_args={"command": "python -m pytest tests"},
        working_directory=project,
        path_append=str(userbase_bin),
    )

    assert record is not None
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["PYTHONUSERBASE"] == str(userbase)
    assert str(env["PATH"]).startswith(str(userbase_bin))
    assert "host-pythonpath-should-not-leak" not in str(env.get("PYTHONPATH", ""))


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
            working_directory=project,
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
    assert first["collect_only_duration_s"] is not None
    assert first["pre_execution_collected_count"] == 3
    assert second["prediction_last_run_s"] == pytest.approx(0.1)
    assert second["prediction_test_count_s"] is not None
    assert second["prediction_per_test_s"] is not None
    assert second["prediction_recommended_method"] == "last_run"
    assert second["prediction_reliability"]["level"] == "high"
    assert second["collect_only_duration_s"] is not None
    assert second["total_duration_with_prediction_overhead_s"] == pytest.approx(
        second["actual_duration_s"] + second["collect_only_duration_s"]
    )
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
