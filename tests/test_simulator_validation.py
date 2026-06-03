from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from trace_collect.simulator import simulate, SimulateError


def test_simulator_rejects_task_without_docker_image(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "trace_metadata",
                        "trace_format_version": 5,
                        "scaffold": "openclaw",
                        "instance_id": "fc_test_001",
                        "model": "dummy",
                        "mode": "collect",
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": "llm_0",
                        "agent_id": "fc_test_001",
                        "iteration": 0,
                        "ts_start": 1.0,
                        "ts_end": 2.0,
                        "data": {"messages_in": [], "completion_tokens": 1},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    task_source = tmp_path / "tasks.json"
    task_source.write_text(
        json.dumps([{"instance_id": "fc_test_001", "problem_statement": "x"}]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SimulateError, match="no resolvable docker_image"):
        asyncio.run(
            simulate(
                source_trace=trace_path,
                task_source=task_source,
                output_dir=tmp_path / "out",
                api_base="http://localhost:8000/v1",
                api_key="EMPTY",
                model="dummy",
            )
        )


def test_simulator_accepts_task_with_image_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Task with image_name passes validation (prepare is mocked)."""
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "trace_metadata",
                        "trace_format_version": 5,
                        "scaffold": "openclaw",
                        "instance_id": "fc_test_002",
                        "model": "dummy",
                        "mode": "collect",
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": "llm_0",
                        "agent_id": "fc_test_002",
                        "iteration": 0,
                        "ts_start": 1.0,
                        "ts_end": 2.0,
                        "data": {"messages_in": [], "completion_tokens": 1},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    task_source = tmp_path / "tasks.json"
    task_source.write_text(
        json.dumps(
            [
                {
                    "instance_id": "fc_test_002",
                    "problem_statement": "x",
                    "image_name": "swebench/test-image",
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    from trace_collect.simulator import PreparedContainer, PreparedTraceSession

    class _FakeAgent:
        async def stop(self): pass

    async def fake_prepare(loaded, *, container_executable, network_mode="host"):
        return PreparedTraceSession(
            loaded=loaded,
            container=PreparedContainer(
                container_id="fake",
                container_executable=container_executable,
                docker_image="fake",
                agent=_FakeAgent(),
            ),
        )

    monkeypatch.setattr("trace_collect.simulator._prepare_container_session", fake_prepare)
    async def _fake_exec(*a, **kw):
        return ("ok", 1.0, True)

    monkeypatch.setattr("trace_collect.simulator._exec_tool", _fake_exec)
    monkeypatch.setattr(
        "trace_collect.simulator.create_async_openai_client",
        lambda **_: (_ for _ in ()).throw(AssertionError("no llm")),
    )

    trace_file = asyncio.run(
        simulate(
            source_trace=trace_path,
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            container_executable="docker",
        )
    )
    assert trace_file.exists()


def test_container_mode_trace_requires_container_executable(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "trace_metadata",
                        "trace_format_version": 5,
                        "scaffold": "openclaw",
                        "instance_id": "fc_test_003",
                        "model": "dummy",
                        "mode": "collect",
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": "llm_0",
                        "agent_id": "fc_test_003",
                        "iteration": 0,
                        "ts_start": 1.0,
                        "ts_end": 2.0,
                        "data": {"messages_in": [], "completion_tokens": 1},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    task_source = tmp_path / "tasks.json"
    task_source.write_text(
        json.dumps(
            [
                {
                    "instance_id": "fc_test_003",
                    "problem_statement": "x",
                    "image_name": "swebench/test-image",
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="container_executable is required"):
        asyncio.run(
            simulate(
                source_trace=trace_path,
                task_source=task_source,
                output_dir=tmp_path / "out",
                mode="cloud_model",
            )
        )


def test_simulator_rejects_duplicate_agent_ids(tmp_path: Path) -> None:
    trace_a = tmp_path / "trace-a.jsonl"
    trace_b = tmp_path / "trace-b.jsonl"
    task_source = tmp_path / "tasks.json"
    manifest = tmp_path / "manifest.json"

    for trace_path in (trace_a, trace_b):
        trace_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "trace_metadata",
                            "scaffold": "openclaw",
                            "instance_id": "same-id",
                            "model": "dummy",
                        }
                    ),
                    json.dumps(
                        {
                            "type": "action",
                            "action_type": "llm_call",
                            "action_id": "llm_0",
                            "agent_id": "same-id",
                            "iteration": 0,
                            "ts_start": 1.0,
                            "ts_end": 2.0,
                            "data": {"messages_in": [], "completion_tokens": 1},
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    task_source.write_text(
        json.dumps([{"instance_id": "same-id", "image_name": "img"}]) + "\n",
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps([
            {"source_trace": trace_a.name},
            {"source_trace": trace_b.name},
        ]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SimulateError, match="Duplicate agent_id"):
        asyncio.run(
            simulate(
                trace_manifest=manifest,
                task_source=task_source,
                output_dir=tmp_path / "out",
                mode="cloud_model",
            )
        )
