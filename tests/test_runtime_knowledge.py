from __future__ import annotations

from pathlib import Path

from trace_collect.package_runtime_prediction import compute_pip_predictions
from trace_collect.pytest_runtime_prediction import compute_pytest_predictions
from trace_collect.python_script_runtime_prediction import (
    compute_python_script_predictions,
)
from trace_collect.runtime_knowledge import (
    lookup_common_prediction,
    load_json_object,
    resource_summary_from_profile,
    update_personal_kb,
    write_json_object,
)


def test_common_prior_is_coldstart_fallback_for_pip() -> None:
    predictions = compute_pip_predictions(
        history={},
        common_knowledge={
            "priors": {
                "pip/install/2-10-packages": {
                    "duration": {"p50_s": 40.0, "p90_s": 90.0},
                    "resources": {
                        "load_class": "network_fetch",
                        "expected_cores": 1.2,
                        "peak_memory_mb": 512.0,
                    },
                    "confidence": "low",
                    "sample_count": 120,
                }
            }
        },
        normalized_command="pip install numpy requests",
        package_count=2,
    )

    assert predictions["prediction_recommended_s"] == 40.0
    assert predictions["prediction_recommended_method"] == (
        "common:pip/install/2-10-packages"
    )
    assert predictions["prediction_common_p90_s"] == 90.0
    assert predictions["runtime_knowledge_prediction"]["load_class"] == "network_fetch"


def test_unified_personal_beats_common_after_tool_history_misses() -> None:
    personal = update_personal_kb(
        {},
        tool_name="python",
        tool_family="script_execution",
        operation="run_script",
        normalized_command="python tools/run.py --mode full",
        duration_s=12.0,
        success=True,
    )

    predictions = compute_python_script_predictions(
        history={},
        personal_knowledge=personal,
        common_knowledge={
            "priors": {
                "script_execution/run_script": {
                    "duration": {"p50_s": 60.0, "p90_s": 120.0},
                }
            }
        },
        normalized_command="python tools/run.py --mode full",
        script_path="tools/run.py",
        script_basename="run.py",
    )

    assert predictions["prediction_recommended_s"] == 12.0
    assert predictions["prediction_recommended_method"] == "personal_command"
    assert predictions["prediction_common_p50_s"] is None


def test_personal_tool_bucket_does_not_override_common_coldstart() -> None:
    personal = update_personal_kb(
        {},
        tool_name="python",
        tool_family="script_execution",
        operation="run_script",
        normalized_command="python tools/old.py",
        duration_s=12.0,
        success=True,
    )

    predictions = compute_python_script_predictions(
        history={},
        personal_knowledge=personal,
        common_knowledge={
            "priors": {
                "script_execution/run_script": {
                    "duration": {"p50_s": 60.0, "p90_s": 120.0},
                }
            }
        },
        normalized_command="python tools/new.py",
        script_path="tools/new.py",
        script_basename="new.py",
    )

    assert predictions["prediction_recommended_s"] == 60.0
    assert predictions["prediction_recommended_method"] == (
        "common:script_execution/run_script"
    )


def test_common_prior_preserves_zero_values() -> None:
    prediction = lookup_common_prediction(
        {
            "priors": {
                "generic_process/run": {
                    "duration": {"p50_s": 0.0, "p90_s": 0.0},
                    "resources": {
                        "expected_cores": 0.0,
                        "peak_memory_mb": 0.0,
                    },
                }
            }
        },
        tool_name="unknown",
        tool_family="generic_process",
        operation="run",
    )

    assert prediction is not None
    assert prediction.duration_p50_s == 0.0
    assert prediction.expected_cores == 0.0


def test_pytest_partial_nodeid_prediction_not_overridden_by_common() -> None:
    predictions = compute_pytest_predictions(
        history={},
        common_knowledge={
            "priors": {
                "pytest/run_tests": {
                    "duration": {"p50_s": 30.0, "p90_s": 100.0},
                }
            }
        },
        command="pytest tests/test_a.py",
        nodeids=["tests/test_a.py::test_one"],
        nodeid_coverage="partial",
    )

    assert predictions["prediction_recommended_s"] is None
    assert predictions["prediction_recommended_method"] == "none"
    assert predictions["prediction_common_p50_s"] == 30.0


def test_resource_summary_uses_final_profile_whole_run() -> None:
    summary = resource_summary_from_profile(
        {
            "final_profile": {
                "total_wall_time_s": 8.5,
                "avg_effective_cores": 3.0,
                "p50_effective_cores": 2.5,
                "p90_effective_cores": 4.0,
                "peak_effective_cores": 5.0,
                "rss_peak_bytes": 256 * 1024 * 1024,
                "total_read_bytes": 4 * 1024 * 1024,
                "total_write_bytes": 8 * 1024 * 1024,
                "preliminary_behavior": "cpu_parallel",
            }
        }
    )

    assert summary is not None
    assert summary.wall_time_s == 8.5
    assert summary.peak_memory_mb == 256.0
    assert summary.load_class == "cpu_parallel"


def test_profiler_summary_updates_personal_kb(tmp_path: Path) -> None:
    kb_path = tmp_path / "personal_runtime_knowledge.json"
    resource_summary = resource_summary_from_profile(
        {
            "final_profile": {
                "total_wall_time_s": 5.0,
                "avg_effective_cores": 1.5,
                "p90_effective_cores": 2.0,
                "peak_effective_cores": 2.5,
                "rss_peak_bytes": 64 * 1024 * 1024,
                "preliminary_behavior": "cpu_serial",
            }
        }
    )
    assert resource_summary is not None
    kb = update_personal_kb(
        {},
        tool_name="pytest",
        tool_family="test_runner",
        operation="run_tests",
        normalized_command="pytest tests/test_a.py",
        duration_s=5.0,
        success=True,
        repo_id="repo-A",
        resource_summary=resource_summary,
    )
    write_json_object(kb_path, kb)

    data = load_json_object(kb_path)
    assert data["repo_id"] == "repo-A"
    command_rec = data["commands"]["pytest tests/test_a.py"]
    assert command_rec["durations"] == [5.0]
    assert command_rec["resources"]["peak_memory_mb"] == [64.0]
    assert command_rec["resources"]["load_class"] == "cpu_serial"


def test_failed_observation_does_not_update_personal_kb() -> None:
    kb = update_personal_kb(
        {},
        tool_name="pytest",
        tool_family="test_runner",
        operation="run_tests",
        normalized_command="pytest",
        duration_s=5.0,
        success=False,
    )

    assert kb == {}
