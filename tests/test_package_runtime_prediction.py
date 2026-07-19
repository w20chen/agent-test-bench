from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_collect.package_runtime_prediction import (
    compute_pip_predictions,
    finalize_pip_runtime_prediction,
    format_pip_prediction_summary,
    is_pip_install_tool_call,
    parse_pip_install_command,
    prepare_pip_runtime_prediction_before_tool,
    update_pip_history,
)


def test_pip_install_command_recognition() -> None:
    assert is_pip_install_tool_call("exec", {"command": "pip install requests"})
    assert is_pip_install_tool_call("exec", {"command": "pip3 install requests"})
    assert is_pip_install_tool_call(
        "exec",
        {"command": "python -m pip install requests"},
    )
    assert not is_pip_install_tool_call("exec", {"command": "pip list"})
    assert not is_pip_install_tool_call("exec", {"command": "python -m pytest"})


def test_normalize_pip_install_sorts_packages_and_ignores_output_flags() -> None:
    parsed = parse_pip_install_command(
        "cd /repo && python -m pip install numpy requests -q --progress-bar off",
        working_directory="/unused",
    )

    assert parsed is not None
    assert parsed.normalized_command == "pip install numpy requests"
    assert parsed.package_count == 2
    assert parsed.packages == ["numpy", "requests"]


def test_normalize_pip_install_hashes_requirement_file(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "\n".join(
            [
                "requests==2.32.0",
                "# comment",
                "--extra-index-url https://example.invalid/simple",
                "numpy>=2",
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_pip_install_command(
        "python -m pip install -r requirements.txt",
        working_directory=tmp_path,
    )

    assert parsed is not None
    assert parsed.package_count == 2
    assert parsed.normalized_command.startswith(
        "pip install -r requirements.txt:sha256="
    )


def test_normalize_pip_install_uses_cd_for_requirement_hash(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "requirements.txt").write_text("requests\n", encoding="utf-8")

    parsed = parse_pip_install_command(
        f"cd {project.as_posix()} && python -m pip install -r requirements.txt",
        working_directory=tmp_path,
    )

    assert parsed is not None
    assert parsed.package_count == 1
    assert parsed.normalized_command.startswith(
        "pip install -r requirements.txt:sha256="
    )


def test_normalize_pip_install_handles_or_chain() -> None:
    parsed = parse_pip_install_command("pip install missing-package || true")

    assert parsed is not None
    assert parsed.normalized_command == "pip install missing-package"
    assert parsed.package_count == 1
    assert parsed.shell_has_or_chain is True


def test_normalize_pip_install_resolves_relative_cd_from_working_dir(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "requirements.txt").write_text("requests\n", encoding="utf-8")

    parsed = parse_pip_install_command(
        "cd project && python -m pip install -r requirements.txt",
        working_directory=tmp_path,
    )

    assert parsed is not None
    assert parsed.package_count == 1
    assert parsed.normalized_command.startswith(
        "pip install -r requirements.txt:sha256="
    )


def test_prediction_prefers_last_run_then_package_count() -> None:
    history = {
        "commands": {
            "pip install numpy": {"durations": [10.0]},
        },
        "tool": {
            "durations": [20.0],
            "per_package_s": [3.0, 5.0],
        },
    }

    seen = compute_pip_predictions(
        history=history,
        normalized_command="pip install numpy",
        package_count=1,
    )
    unseen = compute_pip_predictions(
        history=history,
        normalized_command="pip install pandas scipy",
        package_count=2,
    )

    assert seen["prediction_recommended_s"] == pytest.approx(10.0)
    assert seen["prediction_recommended_method"] == "last_run"
    assert unseen["prediction_package_count_s"] == pytest.approx(8.0)
    assert unseen["prediction_recommended_method"] == "package_count"


def test_format_pip_prediction_summary_includes_all_method_errors() -> None:
    summary = format_pip_prediction_summary(
        {
            "iteration": 4,
            "package_count": 2,
            "actual_duration_s": 10.0,
            "prediction_last_run_s": 9.0,
            "prediction_package_count_s": 11.0,
            "prediction_global_median_s": 8.0,
            "prediction_recommended_s": 11.0,
            "prediction_recommended_method": "package_count",
            "prediction_reliability": {"level": "medium"},
            "relative_error": {
                "last_run": 0.1,
                "package_count": 0.1,
                "global_median": 0.2,
                "recommended": 0.1,
            },
        }
    )

    assert "last=9.00s last_err=10.0%" in summary
    assert "package_count=11.00s package_count_err=10.0%" in summary
    assert "global=8.00s global_err=20.0%" in summary
    assert "recommended=package_count:11.00s rec_err=10.0%" in summary


def test_update_history_skips_failed_runs_for_future_predictions() -> None:
    history = update_pip_history(
        history={},
        normalized_command="pip install missing-package",
        total_duration_s=12.0,
        package_count=1,
        success=False,
    )

    predictions = compute_pip_predictions(
        history=history,
        normalized_command="pip install missing-package",
        package_count=1,
    )

    assert history["commands"] == {}
    assert predictions["prediction_recommended_method"] == "unavailable"


def test_finalize_pip_runtime_prediction_writes_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "pip_runtime"
    tool_args = {"command": "python -m pip install requests"}
    record = prepare_pip_runtime_prediction_before_tool(
        prediction_root=root,
        iteration=3,
        tool_call_id="call-1",
        tool_name="exec",
        tool_args=tool_args,
    )
    assert record is not None

    payload = finalize_pip_runtime_prediction(
        record,
        prediction_root=root,
        action_id="tool_3_call-1",
        ts_start=1.0,
        ts_end=7.0,
        duration_ms=6000.0,
        success=True,
        tool_result="Successfully installed requests\nExit code: 0",
    )

    prediction_path = record.directory / "prediction.json"
    rows = [
        json.loads(line)
        for line in (root / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert prediction_path.exists()
    assert rows == [payload]
    assert payload["normalized_command"] == "pip install requests"
    assert payload["package_count"] == 1
    assert payload["actual_duration_s"] == pytest.approx(6.0)
    assert payload["exit_code"] == 0

    history = json.loads((root / "history.json").read_text(encoding="utf-8"))
    assert history["commands"]["pip install requests"]["durations"] == [6.0]


def test_finalize_uses_pre_execution_prediction_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "pip_runtime"
    record = prepare_pip_runtime_prediction_before_tool(
        prediction_root=root,
        iteration=1,
        tool_call_id="call-snapshot",
        tool_name="exec",
        tool_args={"command": "pip install requests"},
    )
    assert record is not None
    (root / "history.json").write_text(
        json.dumps(
            {
                "commands": {
                    "pip install requests": {"durations": [99.0]},
                },
                "tool": {"durations": [99.0], "per_package_s": [99.0]},
            }
        ),
        encoding="utf-8",
    )

    payload = finalize_pip_runtime_prediction(
        record,
        prediction_root=root,
        action_id="tool_1_call-snapshot",
        ts_start=1.0,
        ts_end=3.0,
        duration_ms=2000.0,
        success=True,
        tool_result="Successfully installed requests\nExit code: 0",
    )

    assert payload["prediction_recommended_method"] == "unavailable"
    assert payload["prediction_recommended_s"] is None


def test_finalize_with_missing_pending_does_not_recompute_predictions(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pip_runtime"
    record = prepare_pip_runtime_prediction_before_tool(
        prediction_root=root,
        iteration=1,
        tool_call_id="call-missing-pending",
        tool_name="exec",
        tool_args={"command": "pip install requests"},
    )
    assert record is not None
    (record.directory / "pending.json").unlink()
    (root / "history.json").write_text(
        json.dumps(
            {
                "commands": {
                    "pip install requests": {"durations": [99.0]},
                },
                "tool": {"durations": [99.0], "per_package_s": [99.0]},
            }
        ),
        encoding="utf-8",
    )

    payload = finalize_pip_runtime_prediction(
        record,
        prediction_root=root,
        action_id="tool_1_call-missing-pending",
        ts_start=1.0,
        ts_end=3.0,
        duration_ms=2000.0,
        success=True,
        tool_result="Successfully installed requests\nExit code: 0",
    )

    assert payload["prediction_recommended_method"] == "unavailable"
    assert payload["prediction_recommended_s"] is None


def test_finalize_nonzero_exit_does_not_update_history(tmp_path: Path) -> None:
    root = tmp_path / "pip_runtime"
    record = prepare_pip_runtime_prediction_before_tool(
        prediction_root=root,
        iteration=1,
        tool_call_id="call-failed",
        tool_name="exec",
        tool_args={"command": "pip install missing-package"},
    )
    assert record is not None

    payload = finalize_pip_runtime_prediction(
        record,
        prediction_root=root,
        action_id="tool_1_call-failed",
        ts_start=1.0,
        ts_end=3.0,
        duration_ms=2000.0,
        success=True,
        tool_result="Could not find a version\nExit code: 1",
    )

    history = json.loads((root / "history.json").read_text(encoding="utf-8"))
    assert payload["history_updated"] is False
    assert history["commands"] == {}


def test_finalize_or_chain_does_not_update_history_on_masked_success(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pip_runtime"
    record = prepare_pip_runtime_prediction_before_tool(
        prediction_root=root,
        iteration=1,
        tool_call_id="call-or",
        tool_name="exec",
        tool_args={"command": "pip install missing-package || true"},
    )
    assert record is not None

    payload = finalize_pip_runtime_prediction(
        record,
        prediction_root=root,
        action_id="tool_1_call-or",
        ts_start=1.0,
        ts_end=3.0,
        duration_ms=2000.0,
        success=True,
        tool_result="Could not find a version\nExit code: 0",
    )

    history = json.loads((root / "history.json").read_text(encoding="utf-8"))
    assert payload["shell_has_or_chain"] is True
    assert payload["history_updated"] is False
    assert history["commands"] == {}
