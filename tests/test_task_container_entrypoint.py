"""Tests for task-container request -> entrypoint mode mapping."""

from __future__ import annotations

import json
from pathlib import Path

from trace_collect.runtime.task_container import (
    exec_task_container_entrypoint,
    write_task_container_request,
)


def test_exec_task_container_entrypoint_uses_run_mode_for_scaffold_requests(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps({"kind": "run_openclaw", "result_path": str(tmp_path / "r.json")}),
        encoding="utf-8",
    )
    seen: list[str] = []

    def fake_run(cmd, **kwargs):
        seen.extend(cmd)

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr("trace_collect.runtime.task_container.subprocess.run", fake_run)

    exec_task_container_entrypoint(
        container_id="cid-1",
        request_path=request_path,
        runtime="/opt/conda/envs/ML/bin/python",
        pythonpath=None,
        container_executable="docker",
        timeout=10,
    )

    assert seen[-1] == "run"


def test_exec_task_container_entrypoint_uses_preflight_mode_for_preflight_requests(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps({"kind": "preflight", "result_path": str(tmp_path / "r.json")}),
        encoding="utf-8",
    )
    seen: list[str] = []

    def fake_run(cmd, **kwargs):
        seen.extend(cmd)

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr("trace_collect.runtime.task_container.subprocess.run", fake_run)

    exec_task_container_entrypoint(
        container_id="cid-1",
        request_path=request_path,
        runtime="/opt/conda/envs/ML/bin/python",
        pythonpath=None,
        container_executable="docker",
        timeout=10,
    )

    assert seen[-1] == "preflight"


def test_write_task_container_request_redacts_api_keys(tmp_path: Path) -> None:
    path = write_task_container_request(
        attempt_dir=tmp_path,
        scaffold="openclaw",
        payload={
            "kind": "run_openclaw",
            "api_key": "secret-value",
            "nested": {"api_key": "nested-secret"},
            "result_path": str(tmp_path / "r.json"),
        },
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["api_key"] == "***REDACTED***"
    assert payload["nested"]["api_key"] == "***REDACTED***"


def test_exec_task_container_entrypoint_uses_original_payload_for_stdin(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = write_task_container_request(
        attempt_dir=tmp_path,
        scaffold="openclaw",
        payload={
            "kind": "run_openclaw",
            "api_key": "secret-value",
            "result_path": str(tmp_path / "r.json"),
        },
    )
    seen: dict[str, str] = {}

    def fake_run(cmd, **kwargs):
        seen["stdin"] = kwargs["input"]

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr("trace_collect.runtime.task_container.subprocess.run", fake_run)

    exec_task_container_entrypoint(
        container_id="cid-1",
        request_path=request_path,
        request_payload={
            "kind": "run_openclaw",
            "api_key": "secret-value",
            "result_path": str(tmp_path / "r.json"),
        },
        runtime="/opt/conda/envs/ML/bin/python",
        pythonpath=None,
        container_executable="docker",
        timeout=10,
    )

    assert "***REDACTED***" not in seen["stdin"]
    assert "secret-value" in seen["stdin"]
