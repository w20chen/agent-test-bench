from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


def test_analyze_pytest_prediction_reports_recommended_and_reliability(
    tmp_path: Path,
) -> None:
    prediction_dir = tmp_path / "case" / "attempt_1" / "pytest_runtime"
    prediction_dir.mkdir(parents=True)
    (prediction_dir / "predictions.jsonl").write_text(
        json.dumps(
            {
                "iteration": 1,
                "command": "pytest tests/test_a.py",
                "actual_duration_s": 10.0,
                "collect_only_duration_s": 1.5,
                "total_duration_with_prediction_overhead_s": 11.5,
                "collected_count": 2,
                "pre_execution_collected_count": 2,
                "prediction_recommended_s": 9.0,
                "prediction_recommended_method": "per_test",
                "prediction_reliability": {
                    "level": "high",
                    "known_node_ratio": 1.0,
                    "file_fallback_ratio": 0.0,
                    "project_fallback_ratio": 0.0,
                    "unknown_fallback_ratio": 0.0,
                    "collected_count_delta_ratio": 0.0,
                },
                "absolute_error": {"recommended": 1.0},
                "relative_error": {"recommended": 0.1},
            }
        )
        + "\n"
        + json.dumps(
            {
                "iteration": 2,
                "command": "pytest tests/test_old.py",
                "actual_duration_s": 5.0,
                "prediction_per_test_s": 4.0,
                "absolute_error": {"per_test": 1.0},
                "relative_error": {"per_test": 0.2},
            }
        )
        + "\n"
        + json.dumps(
            {
                "iteration": 3,
                "command": "pytest tests/test_cold.py",
                "actual_duration_s": 3.0,
                "collect_only_duration_s": 0.5,
                "total_duration_with_prediction_overhead_s": 3.5,
                "prediction_reliability": {"level": "coldstart"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "iteration": 4,
                "command": "pytest tests/test_error.py",
                "actual_duration_s": 1.0,
                "prediction_reliability": {"level": "error"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    csv_path = tmp_path / "summary.csv"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_pytest_prediction.py",
            str(tmp_path),
            "--csv",
            str(csv_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Recommended" in result.stdout
    assert "Average legacy collect-only overhead: 1.0s" in result.stdout
    assert "Reliability buckets:" in result.stdout
    assert "high" in result.stdout
    assert re.search(r"coldstart\s+runs=\s+1\b", result.stdout)
    assert re.search(r"error\s+runs=\s+1\b", result.stdout)
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "prediction_recommended_method" in csv_text
    assert "collect_only_duration_s" in csv_text
    assert "total_duration_with_prediction_overhead_s" in csv_text
    assert "pytest tests/test_cold.py" in csv_text
    assert "pytest tests/test_error.py" in csv_text
    assert "per_test" in csv_text
