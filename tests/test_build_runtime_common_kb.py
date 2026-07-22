from __future__ import annotations

import json
from pathlib import Path

from scripts.build_runtime_common_kb import build_common_kb


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_build_common_kb_from_prediction_artifacts(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "case-a" / "attempt_1" / "pip_runtime" / "predictions.jsonl",
        [
            {
                "success": True,
                "actual_duration_s": 10.0,
                "package_count": 2,
                "command": "pip install private-package",
            },
            {
                "success": True,
                "actual_duration_s": 20.0,
                "package_count": 2,
                "command": "pip install other-private-package",
            },
            {
                "success": False,
                "actual_duration_s": 500.0,
                "package_count": 2,
            },
        ],
    )
    _write_jsonl(
        tmp_path / "case-b" / "attempt_1" / "pytest_runtime" / "predictions.jsonl",
        [
            {
                "success": True,
                "actual_duration_s": 30.0,
                "collected_count": 42,
                "command": "pytest tests/private_test.py",
            }
        ],
    )

    kb = build_common_kb(tmp_path)

    pip_prior = kb["priors"]["pip/install/2-10-packages"]
    assert pip_prior["duration"]["sample_count"] == 2
    assert pip_prior["duration"]["p50_s"] == 15.0
    assert "private-package" not in json.dumps(kb)

    pytest_prior = kb["priors"]["pytest/run_tests/11-100-tests"]
    assert pytest_prior["duration"]["p90_s"] == 30.0


def test_build_common_kb_from_whole_run_profile(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "case-a" / "attempt_1" / "tool_profiles" / "x" / "profile.jsonl",
        [
            {
                "command_string": "python -m pytest tests/test_a.py",
                "exit_code": 0,
                "final_profile": {
                    "total_wall_time_s": 8.0,
                    "avg_effective_cores": 2.0,
                    "p50_effective_cores": 1.5,
                    "p90_effective_cores": 3.0,
                    "peak_effective_cores": 4.0,
                    "rss_peak_bytes": 128 * 1024 * 1024,
                    "total_read_bytes": 2 * 1024 * 1024,
                    "total_write_bytes": 3 * 1024 * 1024,
                    "preliminary_behavior": "cpu_parallel",
                },
            }
        ],
    )

    kb = build_common_kb(tmp_path)
    prior = kb["priors"]["pytest/run_tests"]

    assert prior["duration"]["p50_s"] == 8.0
    assert prior["resources"]["load_class"] == "cpu_parallel"
    assert prior["resources"]["expected_cores"] == 2.0
    assert prior["resources"]["peak_memory_mb"] == 128.0


def test_build_common_kb_from_tool_calls_json(tmp_path: Path) -> None:
    tool_calls = [
        {
            "tool": "read_file",
            "input": {"path": "/testbed/private/module.py"},
            "duration_ms": 100.0,
            "result_preview": "Original size: 70000 chars\nPreview:\n...",
        },
        {
            "tool": "exec-find",
            "input": {"command": "find /testbed -type f -name '*.py'"},
            "duration_ms": 200.0,
            "result_preview": "private/module.py",
        },
        {
            "tool": "exec-pytest",
            "input": {"command": "python -m pytest tests/private_test.py"},
            "duration_ms": 5000.0,
            "result_preview": "1 passed",
        },
    ]
    path = tmp_path / "case" / "attempt_1" / "tool_calls.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tool_calls), encoding="utf-8")

    kb = build_common_kb(tmp_path)

    assert kb["priors"]["read_file/read_file/32k-256k-chars"]["duration"][
        "p50_s"
    ] == 0.1
    assert kb["priors"]["find/search_files"]["tool_family"] == "file_processing"
    assert kb["priors"]["pytest/run_tests"]["duration"]["p50_s"] == 5.0
    assert kb["priors"]["pytest/run_tests/1-10-tests"]["duration"]["p50_s"] == 5.0
    serialized = json.dumps(kb)
    assert "private" not in serialized
    assert "/testbed" not in serialized
    assert "command" not in serialized


def test_builder_emits_family_and_generic_fallback_buckets(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "case" / "attempt_1" / "pip_runtime" / "predictions.jsonl",
        [
            {
                "success": True,
                "actual_duration_s": 10.0,
                "package_count": 2,
                "tool_call_id": "call-1",
            }
        ],
    )

    kb = build_common_kb(tmp_path)

    assert "pip/install/2-10-packages" in kb["priors"]
    assert "package_install/install/2-10-packages" in kb["priors"]
    assert "generic_process/install" in kb["priors"]
    assert "generic_process" in kb["priors"]


def test_builder_deduplicates_prediction_and_tool_call_by_id(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "case" / "attempt_1" / "pytest_runtime" / "predictions.jsonl",
        [
            {
                "success": True,
                "actual_duration_s": 10.0,
                "collected_count": 3,
                "tool_call_id": "same-call",
            }
        ],
    )
    tool_calls = [
        {
            "id": "same-call",
            "tool": "exec-pytest",
            "input": {"command": "pytest tests/test_a.py"},
            "timestamp": "2026-01-01T00:00:01Z",
            "end_timestamp": "2026-01-01T00:00:02Z",
            "duration_ms": 20000.0,
            "result_preview": "3 passed in 20.00s\nExit code: 0",
        }
    ]
    path = tmp_path / "case" / "attempt_1" / "tool_calls.json"
    path.write_text(json.dumps(tool_calls), encoding="utf-8")
    (path.parent / "resources.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "epoch": 1767225601.0,
                        "cpu_percent": "100.0%",
                        "mem_usage": "10.0MiB / 0.0MiB",
                    },
                    {
                        "epoch": 1767225602.0,
                        "cpu_percent": "300.0%",
                        "mem_usage": "20.0MiB / 0.0MiB",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    kb = build_common_kb(tmp_path)

    prior = kb["priors"]["pytest/run_tests/1-10-tests"]
    assert prior["duration"]["sample_count"] == 1
    assert prior["duration"]["p50_s"] == 10.0
    assert prior["resources"]["resource_sample_count"] == 1
    assert prior["resources"]["expected_cores"] == 2.0


def test_builder_suppresses_tool_call_when_prediction_exists_without_shared_id(
    tmp_path: Path,
) -> None:
    _write_jsonl(
        tmp_path / "case" / "attempt_1" / "pytest_runtime" / "predictions.jsonl",
        [
            {
                "success": True,
                "actual_duration_s": 10.0,
                "collected_count": 3,
            }
        ],
    )
    tool_calls = [
        {
            "tool": "exec-pytest",
            "input": {"command": "pytest tests/test_a.py"},
            "duration_ms": 20000.0,
            "result_preview": "3 passed in 20.00s\nExit code: 0",
        }
    ]
    path = tmp_path / "case" / "attempt_1" / "tool_calls.json"
    path.write_text(json.dumps(tool_calls), encoding="utf-8")

    kb = build_common_kb(tmp_path)

    prior = kb["priors"]["pytest/run_tests/1-10-tests"]
    assert prior["duration"]["sample_count"] == 1
    assert prior["duration"]["p50_s"] == 10.0


def test_prediction_duration_suppression_keeps_tool_call_resources(
    tmp_path: Path,
) -> None:
    _write_jsonl(
        tmp_path / "case" / "attempt_1" / "pytest_runtime" / "predictions.jsonl",
        [
            {
                "success": True,
                "actual_duration_s": 10.0,
                "collected_count": 3,
            }
        ],
    )
    attempt = tmp_path / "case" / "attempt_1"
    attempt.mkdir(parents=True, exist_ok=True)
    (attempt / "tool_calls.json").write_text(
        json.dumps(
            [
                {
                    "tool": "exec-pytest",
                    "input": {"command": "pytest tests/test_a.py"},
                    "timestamp": "2026-01-01T00:00:01Z",
                    "end_timestamp": "2026-01-01T00:00:02Z",
                    "duration_ms": 20000.0,
                    "result_preview": "3 passed in 20.00s\nExit code: 0",
                }
            ]
        ),
        encoding="utf-8",
    )
    (attempt / "resources.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "epoch": 1767225601.0,
                        "cpu_percent": "100.0%",
                        "mem_usage": "10.0MiB / 0.0MiB",
                    },
                    {
                        "epoch": 1767225602.0,
                        "cpu_percent": "300.0%",
                        "mem_usage": "20.0MiB / 0.0MiB",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    kb = build_common_kb(tmp_path)

    prior = kb["priors"]["pytest/run_tests/1-10-tests"]
    assert prior["duration"]["sample_count"] == 1
    assert prior["duration"]["p50_s"] == 10.0
    assert prior["resources"]["resource_sample_count"] == 1
    assert prior["resources"]["expected_cores"] == 2.0
    assert prior["resources"]["peak_memory_mb"] == 20.0


def test_tool_call_resources_suppress_duplicate_profile_resources(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "case" / "attempt_1"
    attempt.mkdir(parents=True)
    (attempt / "tool_calls.json").write_text(
        json.dumps(
            [
                {
                    "tool": "exec-pytest",
                    "input": {"command": "pytest tests/test_a.py"},
                    "timestamp": "2026-01-01T00:00:01Z",
                    "end_timestamp": "2026-01-01T00:00:02Z",
                    "duration_ms": 1000.0,
                    "result_preview": "1 passed in 1.00s\nExit code: 0",
                }
            ]
        ),
        encoding="utf-8",
    )
    (attempt / "resources.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "epoch": 1767225601.0,
                        "cpu_percent": "100.0%",
                        "mem_usage": "10.0MiB / 0.0MiB",
                    },
                    {
                        "epoch": 1767225602.0,
                        "cpu_percent": "300.0%",
                        "mem_usage": "20.0MiB / 0.0MiB",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        attempt / "tool_profiles" / "x" / "profile.jsonl",
        [
            {
                "command_string": "pytest tests/test_a.py",
                "exit_code": 0,
                "final_profile": {
                    "total_wall_time_s": 1.0,
                    "avg_effective_cores": 8.0,
                    "rss_peak_bytes": 512 * 1024 * 1024,
                    "preliminary_behavior": "cpu_parallel",
                },
            }
        ],
    )

    kb = build_common_kb(tmp_path)

    resources = kb["priors"]["pytest/run_tests/1-10-tests"]["resources"]
    assert resources["resource_sample_count"] == 1
    assert resources["expected_cores"] == 2.0
    assert resources["peak_memory_mb"] == 20.0


def test_repeated_tool_calls_keep_unmatched_profile_resource_fallback(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "case" / "attempt_1"
    attempt.mkdir(parents=True)
    (attempt / "tool_calls.json").write_text(
        json.dumps(
            [
                {
                    "tool": "exec-pytest",
                    "input": {"command": "pytest tests/test_a.py"},
                    "timestamp": "2026-01-01T00:00:01Z",
                    "end_timestamp": "2026-01-01T00:00:02Z",
                    "duration_ms": 1000.0,
                    "result_preview": "1 passed in 1.00s\nExit code: 0",
                },
                {
                    "tool": "exec-pytest",
                    "input": {"command": "pytest tests/test_b.py"},
                    "timestamp": "2026-01-01T00:00:03Z",
                    "end_timestamp": "2026-01-01T00:00:04Z",
                    "duration_ms": 1000.0,
                    "result_preview": "1 passed in 1.00s\nExit code: 0",
                },
            ]
        ),
        encoding="utf-8",
    )
    (attempt / "resources.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "epoch": 1767225601.0,
                        "cpu_percent": "100.0%",
                        "mem_usage": "10.0MiB / 0.0MiB",
                    },
                    {
                        "epoch": 1767225602.0,
                        "cpu_percent": "300.0%",
                        "mem_usage": "20.0MiB / 0.0MiB",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        attempt / "tool_profiles" / "x" / "profile.jsonl",
        [
            {
                "command_string": "pytest tests/test_b.py",
                "exit_code": 0,
                "final_profile": {
                    "total_wall_time_s": 1.0,
                    "avg_effective_cores": 8.0,
                    "rss_peak_bytes": 512 * 1024 * 1024,
                    "preliminary_behavior": "cpu_parallel",
                },
            }
        ],
    )

    kb = build_common_kb(tmp_path)

    resources = kb["priors"]["pytest/run_tests"]["resources"]
    assert resources["resource_sample_count"] == 2
    assert resources["peak_memory_mb"] > 20.0


def test_builder_skips_failed_tool_call_and_profile(tmp_path: Path) -> None:
    tool_calls = [
        {
            "id": "bad-call",
            "tool": "exec-pytest",
            "input": {"command": "pytest tests/test_a.py"},
            "duration_ms": 5000.0,
            "success": False,
            "result_preview": "failed",
        }
    ]
    tool_call_path = tmp_path / "case" / "attempt_1" / "tool_calls.json"
    tool_call_path.parent.mkdir(parents=True, exist_ok=True)
    tool_call_path.write_text(json.dumps(tool_calls), encoding="utf-8")
    _write_jsonl(
        tmp_path / "case" / "attempt_1" / "tool_profiles" / "x" / "profile.jsonl",
        [
            {
                "command_string": "pytest tests/test_a.py",
                "success": False,
                "final_profile": {
                    "total_wall_time_s": 5.0,
                    "avg_effective_cores": 1.0,
                },
            }
        ],
    )

    kb = build_common_kb(tmp_path)

    assert kb["priors"] == {}


def test_common_output_omits_root_name_and_repo_like_ids(tmp_path: Path) -> None:
    root = tmp_path / "Owner__secret-repo-123"
    _write_jsonl(
        root / "case" / "attempt_1" / "pip_runtime" / "predictions.jsonl",
        [
            {
                "success": True,
                "actual_duration_s": 10.0,
                "package_count": 1,
                "command": "pip install secret-package",
            }
        ],
    )

    kb = build_common_kb(root)
    serialized = json.dumps(kb)

    assert "Owner__secret-repo-123" not in serialized
    assert "secret-package" not in serialized
    assert "command" not in serialized


def test_tool_call_resource_window_updates_common_resources(tmp_path: Path) -> None:
    attempt = tmp_path / "case" / "attempt_1"
    attempt.mkdir(parents=True)
    (attempt / "tool_calls.json").write_text(
        json.dumps(
            [
                {
                    "tool": "exec-pytest",
                    "input": {"command": "pytest tests/test_a.py"},
                    "timestamp": "2026-01-01T00:00:01.000000Z",
                    "end_timestamp": "2026-01-01T00:00:02.000000Z",
                    "duration_ms": 1000.0,
                    "result_preview": "3 passed in 1.00s\nExit code: 0",
                }
            ]
        ),
        encoding="utf-8",
    )
    (attempt / "resources.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "timestamp": "2026-01-01T00:00:00.000000",
                        "epoch": 1767225600.0,
                        "cpu_percent": "900.0%",
                        "mem_usage": "1.0MiB / 0.0MiB",
                        "disk_read_bytes": 0,
                        "disk_write_bytes": 0,
                        "net_rx_bytes": 0,
                        "net_tx_bytes": 0,
                        "context_switches": 0,
                        "memory_read_mb_s": 999999.0,
                    },
                    {
                        "timestamp": "2026-01-01T00:00:01.000000",
                        "epoch": 1767225601.0,
                        "cpu_percent": "100.0%",
                        "mem_usage": "10.0MiB / 0.0MiB",
                        "disk_read_bytes": 0,
                        "disk_write_bytes": 0,
                        "net_rx_bytes": 0,
                        "net_tx_bytes": 0,
                        "context_switches": 10,
                        "l1d_hit_rate": 0.9,
                        "ipc": 1.0,
                    },
                    {
                        "timestamp": "2026-01-01T00:00:02.000000",
                        "epoch": 1767225602.0,
                        "cpu_percent": "300.0%",
                        "mem_usage": "20.0MiB / 0.0MiB",
                        "disk_read_bytes": 2 * 1024 * 1024,
                        "disk_write_bytes": 3 * 1024 * 1024,
                        "net_rx_bytes": 4 * 1024 * 1024,
                        "net_tx_bytes": 5 * 1024 * 1024,
                        "context_switches": 30,
                        "l1d_hit_rate": 0.8,
                        "ipc": 2.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    kb = build_common_kb(tmp_path)
    resources = kb["priors"]["pytest/run_tests/1-10-tests"]["resources"]

    assert resources["expected_cores"] == 2.0
    assert resources["peak_cores_p90"] == 3.0
    assert resources["peak_memory_mb"] == 20.0
    assert resources["disk_read_mb_p90"] == 2.0
    assert resources["disk_write_mb_p90"] == 3.0
    assert resources["net_rx_mb_p90"] == 4.0
    assert resources["net_tx_mb_p90"] == 5.0
    assert resources["context_switches_p90"] == 20.0
    assert resources["ipc_p50"] == 1.5
    assert "memory_read_mb_s" not in json.dumps(kb)
