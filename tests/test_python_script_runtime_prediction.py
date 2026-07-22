from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_collect.python_script_runtime_prediction import (
    compute_python_script_predictions,
    finalize_python_script_runtime_prediction,
    format_python_script_prediction_summary,
    is_python_script_tool_call,
    merge_python_script_predictions_into_shared_history,
    parse_python_script_command,
    prepare_python_script_runtime_prediction_before_tool,
    seed_python_script_history_from_shared,
    update_python_script_history,
)


def test_format_python_script_prediction_summary_includes_runtime_kb() -> None:
    summary = format_python_script_prediction_summary(
        {
            "iteration": 7,
            "script_basename": "train.py",
            "actual_duration_s": 30.0,
            "prediction_last_run_s": None,
            "prediction_family_last_run_s": None,
            "prediction_script_path_median_s": None,
            "prediction_basename_median_s": None,
            "prediction_global_median_s": None,
            "prediction_recommended_s": 25.0,
            "prediction_recommended_method": "common:by_operation/run_script/default",
            "prediction_reliability": {"level": "low"},
            "runtime_knowledge_prediction": {
                "duration_p50_s": 25.0,
                "duration_p90_s": 40.0,
                "prediction_source": "common:by_operation/run_script/default",
            },
            "relative_error": {},
        }
    )

    assert "kb=common:by_operation/run_script/default" in summary
    assert "p50=25.0s p90=40.0s" in summary
    assert summary.endswith("| low")


def test_python_script_command_recognition() -> None:
    assert is_python_script_tool_call("exec", {"command": "python3 eval.py"})
    assert is_python_script_tool_call(
        "exec",
        {"command": "timeout 20 python3 -u /app/explorer.py --seed 1"},
    )
    assert is_python_script_tool_call(
        "exec",
        {"command": "cd /app && . .venv/bin/activate && python test_script.py"},
    )
    assert not is_python_script_tool_call("exec", {"command": "python3 -c 'print(1)'"})
    assert not is_python_script_tool_call("exec", {"command": "python3 -m pytest"})
    assert not is_python_script_tool_call("exec", {"command": "python3 -V eval.py"})
    assert not is_python_script_tool_call(
        "exec",
        {"command": "python3 --version eval.py"},
    )
    assert not is_python_script_tool_call(
        "exec",
        {"command": "python3 << 'PY'\nprint(1)\nPY"},
    )


def test_parse_python_script_normalizes_cd_and_flags() -> None:
    parsed = parse_python_script_command(
        "cd project && . .venv/bin/activate && timeout 30 python3 -u scripts/eval.py --limit 10",
        working_directory="/testbed",
    )

    assert parsed is not None
    assert parsed.script_path == "/testbed/project/scripts/eval.py"
    assert parsed.script_basename == "eval.py"
    assert parsed.python_flags == ["-u"]
    assert parsed.timeout_s == 30.0
    assert parsed.args_signature == "--limit 10"
    assert parsed.normalized_command == (
        "python-script /testbed/project/scripts/eval.py flags -u args --limit 10"
    )


def test_prediction_fallback_order() -> None:
    history = {
        "commands": {
            "python-script /app/eval.py": {"durations": [10.0]},
        },
        "scripts": {
            "/app/other.py": {"durations": [20.0, 30.0]},
            "/app/eval.py": {"durations": [8.0]},
        },
        "basenames": {
            "eval.py": {"durations": [40.0]},
        },
        "tool": {"durations": [50.0]},
    }

    seen = compute_python_script_predictions(
        history=history,
        normalized_command="python-script /app/eval.py",
        script_path="/app/eval.py",
        script_basename="eval.py",
    )
    same_script = compute_python_script_predictions(
        history=history,
        normalized_command="python-script /app/eval.py args --new",
        script_path="/app/eval.py",
        script_basename="eval.py",
    )
    basename = compute_python_script_predictions(
        history=history,
        normalized_command="python-script /tmp/eval.py",
        script_path="/tmp/eval.py",
        script_basename="eval.py",
    )

    assert seen["prediction_recommended_s"] == pytest.approx(10.0)
    assert seen["prediction_recommended_method"] == "last_run"
    assert same_script["prediction_recommended_s"] == pytest.approx(8.0)
    assert same_script["prediction_recommended_method"] == "script_path_median"
    assert basename["prediction_recommended_s"] == pytest.approx(40.0)
    assert basename["prediction_recommended_method"] == "basename_median"


def test_update_history_skips_failed_runs() -> None:
    history = update_python_script_history(
        history={},
        normalized_command="python-script /app/eval.py",
        script_path="/app/eval.py",
        script_basename="eval.py",
        total_duration_s=12.0,
        success=False,
    )

    assert history["commands"] == {}
    assert history["scripts"] == {}
    assert history["basenames"] == {}
    assert history["tool"] == {}


def test_finalize_writes_artifacts_and_cross_attempt_history(tmp_path: Path) -> None:
    shared_root = tmp_path / "python_script_runtime_db"
    first_root = tmp_path / "attempt_1" / "python_script_runtime"
    second_root = tmp_path / "attempt_2" / "python_script_runtime"

    first = prepare_python_script_runtime_prediction_before_tool(
        prediction_root=first_root,
        history_root=shared_root,
        iteration=0,
        tool_call_id="call_1",
        tool_name="exec",
        tool_args={"command": "cd /app && python3 eval.py --limit 10"},
    )
    assert first is not None
    first_payload = finalize_python_script_runtime_prediction(
        first,
        prediction_root=first_root,
        history_root=shared_root,
        action_id="tool_0_call_1",
        ts_start=1.0,
        ts_end=6.0,
        duration_ms=5000.0,
        success=True,
        tool_result="ok\nExit code: 0",
    )
    assert first_payload["history_updated"] is True
    assert str(first_payload["prediction_recommended_method"]).startswith("common:")

    merge_python_script_predictions_into_shared_history(
        shared_history_root=shared_root,
        attempt_prediction_root=first_root,
    )
    seed_python_script_history_from_shared(
        shared_history_root=shared_root,
        attempt_prediction_root=second_root,
    )
    second = prepare_python_script_runtime_prediction_before_tool(
        prediction_root=second_root,
        history_root=shared_root,
        iteration=0,
        tool_call_id="call_2",
        tool_name="exec",
        tool_args={"command": "cd /app && python3 eval.py --limit 10"},
    )
    assert second is not None
    pending = json.loads(
        (second.directory / "pending.json").read_text(encoding="utf-8")
    )

    assert pending["predictions"]["prediction_last_run_s"] == pytest.approx(5.0)
    assert pending["predictions"]["prediction_recommended_method"] == "last_run"


def test_finalize_does_not_update_history_for_or_chain(tmp_path: Path) -> None:
    root = tmp_path / "python_script_runtime"
    record = prepare_python_script_runtime_prediction_before_tool(
        prediction_root=root,
        iteration=0,
        tool_call_id="call_1",
        tool_name="exec",
        tool_args={"command": "python3 missing.py || true"},
    )
    assert record is not None
    payload = finalize_python_script_runtime_prediction(
        record,
        prediction_root=root,
        action_id="tool_0_call_1",
        ts_start=1.0,
        ts_end=3.0,
        duration_ms=2000.0,
        success=True,
        tool_result="ignored\nExit code: 0",
    )

    assert payload["shell_has_or_chain"] is True
    assert payload["history_updated"] is False
    history = json.loads((root / "history.json").read_text(encoding="utf-8"))
    assert history["commands"] == {}


@pytest.mark.parametrize(
    "command",
    [
        "python3 fail.py; true",
        "python3 fail.py | tee log.txt",
        "python3 eval.py && pytest tests/",
        "python3 eval.py\npytest tests/",
        "python3 fail.py\ntrue",
        "python3 eval.py & wait",
    ],
)
def test_finalize_does_not_update_history_for_compound_followups(
    tmp_path: Path,
    command: str,
) -> None:
    root = tmp_path / "python_script_runtime"
    record = prepare_python_script_runtime_prediction_before_tool(
        prediction_root=root,
        iteration=0,
        tool_call_id="call_1",
        tool_name="exec",
        tool_args={"command": command},
    )
    if record is None:
        assert not (root / "history.json").exists()
        return
    payload = finalize_python_script_runtime_prediction(
        record,
        prediction_root=root,
        action_id="tool_0_call_1",
        ts_start=1.0,
        ts_end=3.0,
        duration_ms=2000.0,
        success=True,
        tool_result="done\nExit code: 0",
    )

    assert payload["shell_has_followup_segments"] is True
    assert payload["history_updated"] is False
    history = json.loads((root / "history.json").read_text(encoding="utf-8"))
    assert history["commands"] == {}


@pytest.mark.parametrize(
    "command",
    [
        "make data && python3 eval.py",
        "pytest tests/ && python3 eval.py",
        "curl -fsSL https://example.invalid/data | python3 eval.py",
    ],
)
def test_finalize_does_not_update_history_for_prefix_work(
    tmp_path: Path,
    command: str,
) -> None:
    root = tmp_path / "python_script_runtime"
    record = prepare_python_script_runtime_prediction_before_tool(
        prediction_root=root,
        iteration=0,
        tool_call_id="call_1",
        tool_name="exec",
        tool_args={"command": command},
    )
    if record is None:
        assert not (root / "history.json").exists()
        return
    payload = finalize_python_script_runtime_prediction(
        record,
        prediction_root=root,
        action_id="tool_0_call_1",
        ts_start=1.0,
        ts_end=3.0,
        duration_ms=2000.0,
        success=True,
        tool_result="done\nExit code: 0",
    )

    assert payload["shell_has_prefix_work"] is True
    assert payload["history_updated"] is False
    history = json.loads((root / "history.json").read_text(encoding="utf-8"))
    assert history["commands"] == {}


def test_cd_and_activation_prefixes_can_update_history(tmp_path: Path) -> None:
    root = tmp_path / "python_script_runtime"
    record = prepare_python_script_runtime_prediction_before_tool(
        prediction_root=root,
        iteration=0,
        tool_call_id="call_1",
        tool_name="exec",
        tool_args={
            "command": "cd /app && . .venv/bin/activate && FOO=bar && python3 eval.py"
        },
    )
    assert record is not None
    payload = finalize_python_script_runtime_prediction(
        record,
        prediction_root=root,
        action_id="tool_0_call_1",
        ts_start=1.0,
        ts_end=3.0,
        duration_ms=2000.0,
        success=True,
        tool_result="done\nExit code: 0",
    )

    assert payload["shell_has_prefix_work"] is False
    assert payload["history_updated"] is True
