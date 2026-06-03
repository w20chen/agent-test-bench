from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from trace_collect.cli import _run_simulate, parse_simulate_args
from trace_collect.simulator import simulate


class _FakeStream:
    def __init__(self) -> None:
        self._emitted = False

    def __aiter__(self) -> "_FakeStream":
        return self

    async def __anext__(self):
        if self._emitted:
            raise StopAsyncIteration
        self._emitted = True
        await asyncio.sleep(0.01)
        delta = type("Delta", (), {"content": "x"})()
        choice = type("Choice", (), {"delta": delta})()
        return type("Chunk", (), {"choices": [choice]})()


class _FakeClient:
    class _Completions:
        async def create(self, **kwargs):
            return _FakeStream()

    class _Chat:
        def __init__(self) -> None:
            self.completions = _FakeClient._Completions()

    def __init__(self) -> None:
        self.chat = self._Chat()


def _write_trace(
    path: Path,
    *,
    agent_id: str,
    scaffold: str = "openclaw",
    llm_start: float = 100.0,
    llm_end: float = 100.2,
    tool_start: float = 100.4,
    tool_end: float = 100.45,
    tool_name: str = "write_file",
    execution_environment: str = "container",
) -> None:
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "trace_metadata",
                        "trace_format_version": 5,
                        "scaffold": scaffold,
                        "instance_id": agent_id,
                        "model": "claude-haiku",
                        "mode": "collect",
                        "execution_environment": execution_environment,
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": f"{agent_id}-llm-0",
                        "agent_id": agent_id,
                        "iteration": 0,
                        "ts_start": llm_start,
                        "ts_end": llm_end,
                        "data": {
                            "messages_in": [{"role": "user", "content": "fix bug"}],
                            "raw_response": {"id": f"resp-{agent_id}"},
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "llm_latency_ms": (llm_end - llm_start) * 1000,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "tool_exec",
                        "action_id": f"{agent_id}-tool-0",
                        "agent_id": agent_id,
                        "iteration": 0,
                        "ts_start": tool_start,
                        "ts_end": tool_end,
                        "data": {
                            "tool_name": tool_name,
                            "tool_args": json.dumps({"path": "/testbed/x.txt"}),
                            "tool_result": "source-result",
                            "duration_ms": (tool_end - tool_start) * 1000,
                            "success": True,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "summary",
                        "agent_id": agent_id,
                        "model": "claude-haiku",
                        "success": True,
                        "n_iterations": 1,
                        "elapsed_s": tool_end - llm_start,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_tasks(path: Path, *agent_ids: str) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "instance_id": agent_id,
                    "problem_statement": f"problem for {agent_id}",
                    "repo": "django/django",
                    "base_commit": "deadbeef",
                    "image_name": f"swebench-test/{agent_id}",
                }
                for agent_id in agent_ids
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_host_tasks(path: Path, *agent_ids: str) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "instance_id": agent_id,
                    "problem_statement": f"problem for {agent_id}",
                    "repo": None,
                    "image_name": None,
                    "docker_image": None,
                }
                for agent_id in agent_ids
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _patch_simulator_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    tool_delay_s: float = 0.0,
    tool_duration_ms: float = 8.0,
    tool_result_prefix: str = "executed",
    llm_client_mode: str = "forbid",
) -> None:
    class _FakeAgent:
        async def stop(self): pass

    async def fake_prepare_container(loaded, *, container_executable, network_mode="host"):
        from trace_collect.simulator import PreparedContainer, PreparedTraceSession
        container = PreparedContainer(
            container_id="fake-cid",
            container_executable=container_executable,
            docker_image="fake-image",
            agent=_FakeAgent(),
        )
        return PreparedTraceSession(loaded=loaded, container=container)

    async def fake_exec_tool(agent, tool_name, tool_args_json, command_timeout_s):
        if tool_delay_s > 0:
            await asyncio.sleep(tool_delay_s)
        return f"{tool_result_prefix}-{tool_name}", tool_duration_ms, True

    monkeypatch.setattr("trace_collect.simulator._prepare_container_session", fake_prepare_container)
    monkeypatch.setattr("trace_collect.simulator._exec_tool", fake_exec_tool)
    if llm_client_mode == "forbid":
        monkeypatch.setattr(
            "trace_collect.simulator.create_async_openai_client",
            lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("cloud_model must not create llm client")
            ),
        )
    elif llm_client_mode == "fake":
        monkeypatch.setattr(
            "trace_collect.simulator.create_async_openai_client",
            lambda **_kwargs: _FakeClient(),
        )
    else:
        raise AssertionError(f"unknown llm_client_mode: {llm_client_mode}")


def test_parse_simulate_args_accepts_cloud_model_manifest_without_llm_args() -> None:
    args = parse_simulate_args(
        [
            "--mode",
            "cloud_model",
            "--trace-manifest",
            "manifest.json",
        ]
    )

    assert args.mode == "cloud_model"
    assert args.trace_manifest == "manifest.json"
    assert args.source_trace is None
    assert args.replay_speed == 1.0


def test_parse_simulate_args_accepts_container_flag() -> None:
    args = parse_simulate_args(
        [
            "--mode",
            "cloud_model",
            "--source-trace",
            "trace.jsonl",
            "--container",
            "podman",
        ]
    )
    assert args.container == "podman"


def test_parse_simulate_args_defaults_container_to_none() -> None:
    args = parse_simulate_args(
        ["--source-trace", "trace.jsonl"]
    )
    assert args.container is None


def test_run_simulate_cloud_model_bypasses_llm_config(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    async def fake_simulate(**kwargs):
        seen.update(kwargs)
        return tmp_path / "out.jsonl"

    monkeypatch.setattr(
        "trace_collect.cli.resolve_llm_config",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not resolve llm config")),
    )
    monkeypatch.setattr("trace_collect.simulator.simulate", fake_simulate)

    args = parse_simulate_args(
        [
            "--mode",
            "cloud_model",
            "--source-trace",
            "trace.jsonl",
        ]
    )

    _run_simulate(args)

    assert seen["mode"] == "cloud_model"
    assert seen["source_trace"] == Path("trace.jsonl")
    assert seen["trace_manifest"] is None
    assert seen["container_executable"] is None


def test_run_simulate_rejects_metrics_url_in_cloud_model(capsys: pytest.CaptureFixture[str]) -> None:
    args = parse_simulate_args(
        [
            "--mode",
            "cloud_model",
            "--source-trace",
            "trace.jsonl",
            "--metrics-url",
            "http://localhost:8000/metrics",
        ]
    )

    with pytest.raises(SystemExit, match="2"):
        _run_simulate(args)

    assert "cloud_model replay does not support --metrics-url" in capsys.readouterr().err


def test_run_simulate_local_model_resolves_llm_config(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    async def fake_simulate(**kwargs):
        seen.update(kwargs)
        return tmp_path / "out.jsonl"

    monkeypatch.setattr(
        "trace_collect.cli.resolve_llm_config",
        lambda **_kwargs: type(
            "Cfg",
            (),
            {
                "api_base": "https://example.com/v1",
                "api_key": "secret",
                "model": "z-ai/glm-5.1",
                "env_key": "OPENROUTER_API_KEY",
            },
        )(),
    )
    monkeypatch.setattr("trace_collect.simulator.simulate", fake_simulate)

    args = parse_simulate_args(
        [
            "--mode",
            "local_model",
            "--source-trace",
            "trace.jsonl",
            "--provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.1",
            "--api-base",
            "https://ignored.example/v1",
        ]
    )

    _run_simulate(args)

    assert seen["mode"] == "local_model"
    assert seen["api_base"] == "https://example.com/v1"
    assert seen["api_key"] == "secret"
    assert seen["model"] == "z-ai/glm-5.1"


def test_run_simulate_local_model_rejects_trace_manifest(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = parse_simulate_args(
        [
            "--mode",
            "local_model",
            "--trace-manifest",
            "manifest.json",
        ]
    )

    with pytest.raises(SystemExit, match="2"):
        _run_simulate(args)

    assert "local_model mode accepts only --source-trace" in capsys.readouterr().err


def test_cloud_model_single_trace_replays_without_llm_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(trace_path, agent_id="task-a")
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(
        monkeypatch,
        tmp_path,
        tool_delay_s=0.01,
        tool_duration_ms=10.0,
        llm_client_mode="forbid",
    )

    started = time.monotonic()
    trace_file = asyncio.run(
        simulate(
            source_trace=trace_path,
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            container_executable="docker",
            replay_speed=10.0,
        )
    )
    elapsed = time.monotonic() - started

    records = _read_jsonl(trace_file)
    assert elapsed >= 0.03
    assert records[0]["simulate_mode"] == "cloud_model"
    assert records[0]["source_model"] == "claude-haiku"
    assert "local_model" not in records[0]
    llm_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "llm_call"
    )
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )
    summary = next(record for record in records if record.get("type") == "summary")

    assert llm_record["data"]["replay_mode"] == "cloud_model"
    assert llm_record["data"]["source_llm_latency_ms"] == pytest.approx(200.0)
    assert llm_record["data"]["sim_metrics"]["warmup"] is False
    assert tool_record["data"]["replay_source"] == "executed_in_container"
    assert tool_record["data"]["source_duration_ms"] == pytest.approx(50.0)
    assert tool_record["data"]["tool_result"] == "executed-write_file"
    assert tool_record["data"]["sim_metrics"]["warmup"] is False
    assert tool_record["data"]["sim_metrics"]["sim_tool_format"] == "container_exec"
    assert summary["success"] is True
    assert summary["source_success"] is True
    assert summary["replay_mode"] == "cloud_model"
    assert summary["replay_speed"] == pytest.approx(10.0)


def test_cloud_model_host_trace_replays_without_container_or_llm_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(
        trace_path,
        agent_id="host-task",
        scaffold="tongyi-deepresearch",
        execution_environment="host",
    )
    _write_host_tasks(task_source, "host-task")

    async def fail_prepare(*args, **kwargs):
        raise AssertionError("host-mode replay must not prepare a container")

    monkeypatch.setattr(
        "trace_collect.simulator._prepare_container_session",
        fail_prepare,
    )
    monkeypatch.setattr(
        "trace_collect.simulator.create_async_openai_client",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("cloud_model must not create llm client")
        ),
    )

    trace_file = asyncio.run(
        simulate(
            source_trace=trace_path,
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            replay_speed=10.0,
        )
    )

    records = _read_jsonl(trace_file)
    metadata = records[0]
    llm_records = [
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "llm_call"
    ]
    tool_records = [
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    ]
    summary = next(record for record in records if record.get("type") == "summary")

    assert metadata["execution_environment"] == "host"
    assert len(llm_records) == 1
    assert llm_records[0]["data"]["sim_metrics"]["warmup"] is False
    assert len(tool_records) == 1
    assert tool_records[0]["data"]["replay_source"] == "skipped_host_mode"
    assert tool_records[0]["data"]["success"] is True
    assert tool_records[0]["data"]["sim_metrics"]["sim_tool_format"] == "skipped_host_mode"
    assert summary["success"] is True

    # Regression: host-mode replay must still emit an empty resources.json so
    # downstream consumers can rely on canonical simulate layout.
    attempt_dir = (tmp_path / "out" / "host-task" / "attempt_1")
    resources_path = attempt_dir / "resources.json"
    assert resources_path.exists(), (
        "host-mode replay must write resources.json even without a sampler"
    )
    payload = json.loads(resources_path.read_text())
    assert payload["samples"] == []
    assert payload["summary"]["sample_count"] == 0


def test_cloud_model_host_trace_skips_mcp_tools(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(
        trace_path,
        agent_id="host-task",
        scaffold="tongyi-deepresearch",
        tool_name="mcp_search",
        execution_environment="host",
    )
    _write_host_tasks(task_source, "host-task")

    async def fail_prepare(*args, **kwargs):
        raise AssertionError("host-mode replay must not prepare a container")

    monkeypatch.setattr(
        "trace_collect.simulator._prepare_container_session",
        fail_prepare,
    )
    monkeypatch.setattr(
        "trace_collect.simulator.create_async_openai_client",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("cloud_model must not create llm client")
        ),
    )

    trace_file = asyncio.run(
        simulate(
            source_trace=trace_path,
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            replay_speed=10.0,
        )
    )

    records = _read_jsonl(trace_file)
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )
    assert tool_record["data"]["replay_source"] == "skipped_host_mode"
    assert tool_record["data"]["sim_metrics"]["source"] == "skipped_host_mode"
    assert tool_record["data"]["success"] is True


def test_cloud_model_replay_marks_warmup_iterations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(trace_path, agent_id="task-a")
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path, llm_client_mode="forbid")

    trace_file = asyncio.run(
        simulate(
            source_trace=trace_path,
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            container_executable="docker",
            replay_speed=10.0,
            warmup_skip_iterations=1,
        )
    )

    records = _read_jsonl(trace_file)
    llm_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "llm_call"
    )
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )

    assert llm_record["data"]["sim_metrics"]["warmup"] is True
    assert tool_record["data"]["sim_metrics"]["warmup"] is True


def test_local_model_single_trace_still_emits_sim_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(trace_path, agent_id="task-a")
    _write_tasks(task_source, "task-a")
    _patch_simulator_runtime(monkeypatch, tmp_path, llm_client_mode="fake")

    trace_file = asyncio.run(
        simulate(
            source_trace=trace_path,
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="local_model",
            container_executable="docker",
            api_base="https://example.com/v1",
            api_key="secret",
            model="local-qwen",
        )
    )

    records = _read_jsonl(trace_file)
    metadata = records[0]
    llm_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "llm_call"
    )
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )

    assert metadata["simulate_mode"] == "local_model"
    assert metadata["local_model"] == "local-qwen"
    assert llm_record["data"]["sim_metrics"]["timing"]["total_ms"] >= 0.0
    assert llm_record["data"]["source_llm_latency_ms"] == pytest.approx(200.0)
    assert tool_record["data"]["sim_metrics"]["source"] == "executed_in_container"
    assert tool_record["data"]["tool_result"] == "executed-write_file"
    summary = next(record for record in records if record.get("type") == "summary")
    assert summary["success"] is True
    assert summary["source_success"] is True


def test_local_model_host_trace_completes_without_container(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(
        trace_path,
        agent_id="host-task",
        scaffold="tongyi-deepresearch",
        execution_environment="host",
    )
    _write_host_tasks(task_source, "host-task")

    async def fail_prepare(*args, **kwargs):
        raise AssertionError("host-mode local simulation must not prepare a container")

    monkeypatch.setattr(
        "trace_collect.simulator._prepare_container_session",
        fail_prepare,
    )
    monkeypatch.setattr(
        "trace_collect.simulator.create_async_openai_client",
        lambda **_kwargs: _FakeClient(),
    )

    trace_file = asyncio.run(
        simulate(
            source_trace=trace_path,
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="local_model",
            api_base="https://example.com/v1",
            api_key="secret",
            model="local-qwen",
        )
    )

    records = _read_jsonl(trace_file)
    metadata = records[0]
    llm_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "llm_call"
    )
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )
    summary = next(record for record in records if record.get("type") == "summary")

    assert metadata["execution_environment"] == "host"
    assert llm_record["data"]["sim_metrics"]["timing"]["total_ms"] >= 0.0
    # Host-mode tools cannot be re-executed in local_model (no container);
    # preserve source-trace timing so total_tool_ms stays faithful.
    assert tool_record["data"]["sim_metrics"]["source"] == "replayed_from_trace"
    assert (
        tool_record["data"]["sim_metrics"]["sim_tool_format"]
        == "replayed_from_trace"
    )
    assert tool_record["data"]["duration_ms"] == pytest.approx(50.0, abs=0.01)
    # ts_end - ts_start should match the replayed duration (0.05s)
    assert tool_record["ts_end"] - tool_record["ts_start"] == pytest.approx(
        0.05, abs=0.01
    )
    assert tool_record["data"]["success"] is True
    assert summary["success"] is True


def test_local_model_host_trace_replays_mcp_tool_timing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(
        trace_path,
        agent_id="host-task",
        scaffold="tongyi-deepresearch",
        tool_name="mcp_search",
        execution_environment="host",
    )
    _write_host_tasks(task_source, "host-task")

    async def fail_prepare(*args, **kwargs):
        raise AssertionError("host-mode local simulation must not prepare a container")

    monkeypatch.setattr(
        "trace_collect.simulator._prepare_container_session",
        fail_prepare,
    )
    monkeypatch.setattr(
        "trace_collect.simulator.create_async_openai_client",
        lambda **_kwargs: _FakeClient(),
    )

    trace_file = asyncio.run(
        simulate(
            source_trace=trace_path,
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="local_model",
            api_base="https://example.com/v1",
            api_key="secret",
            model="local-qwen",
        )
    )

    records = _read_jsonl(trace_file)
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )

    # Both MCP and non-MCP host-mode tools go through the same replay path
    # in local_model — they cannot be re-executed without a container.
    assert tool_record["data"]["sim_metrics"]["source"] == "replayed_from_trace"
    assert (
        tool_record["data"]["sim_metrics"]["sim_tool_format"]
        == "replayed_from_trace"
    )
    assert tool_record["data"]["duration_ms"] == pytest.approx(50.0, abs=0.01)
    assert tool_record["data"]["success"] is True


def test_local_model_host_trace_replay_speed_scales_replayed_tool_timing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(
        trace_path,
        agent_id="host-task",
        scaffold="tongyi-deepresearch",
        execution_environment="host",
    )
    _write_host_tasks(task_source, "host-task")

    async def fail_prepare(*args, **kwargs):
        raise AssertionError("host-mode local simulation must not prepare a container")

    monkeypatch.setattr(
        "trace_collect.simulator._prepare_container_session",
        fail_prepare,
    )
    monkeypatch.setattr(
        "trace_collect.simulator.create_async_openai_client",
        lambda **_kwargs: _FakeClient(),
    )

    trace_file = asyncio.run(
        simulate(
            source_trace=trace_path,
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="local_model",
            api_base="https://example.com/v1",
            api_key="secret",
            model="local-qwen",
            replay_speed=10.0,
        )
    )

    records = _read_jsonl(trace_file)
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )

    assert tool_record["data"]["duration_ms"] == pytest.approx(50.0, abs=0.01)
    assert tool_record["ts_end"] - tool_record["ts_start"] == pytest.approx(
        0.005, abs=0.01
    )


def test_local_model_terminal_transport_retry_marks_failed_iteration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "trace_metadata",
                        "trace_format_version": 5,
                        "scaffold": "tongyi-deepresearch",
                        "instance_id": "task-a",
                        "model": "source-model",
                        "execution_environment": "host",
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": "llm_0_transport_exhausted",
                        "agent_id": "task-a",
                        "iteration": 0,
                        "ts_start": 100.0,
                        "ts_end": 100.0,
                        "data": {
                            "transport_retry": True,
                            "transport_retry_terminal": True,
                            "messages_in": [{"role": "user", "content": "fail please"}],
                            "error": "APIConnectionError: boom",
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "summary",
                        "agent_id": "task-a",
                        "model": "source-model",
                        "success": False,
                        "n_iterations": 1,
                        "elapsed_s": 0.0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_host_tasks(task_source, "task-a")

    monkeypatch.setattr(
        "trace_collect.simulator.create_async_openai_client",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("terminal transport retry should not invoke local model")
        ),
    )
    monkeypatch.setattr(
        "trace_collect.simulator._prepare_container_session",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("host-mode local simulation must not prepare a container")
        ),
    )

    trace_file = asyncio.run(
        simulate(
            source_trace=trace_path,
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="local_model",
            api_base="https://example.com/v1",
            api_key="secret",
            model="local-qwen",
        )
    )

    records = _read_jsonl(trace_file)
    llm_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "llm_call"
    )
    summary = next(record for record in records if record.get("type") == "summary")

    assert llm_record["data"]["transport_retry_terminal"] is True
    assert llm_record["data"]["sim_metrics"]["failed"] is True
    assert llm_record["data"]["messages_in"] == [{"role": "user", "content": "fail please"}]
    assert summary["success"] is False
    assert summary["failed_iterations"] == 1


def test_cloud_model_trace_manifest_replays_multiple_sessions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_a = tmp_path / "trace-a.jsonl"
    trace_b = tmp_path / "trace-b.jsonl"
    task_source = tmp_path / "tasks.json"
    manifest = tmp_path / "manifest.json"
    _write_trace(trace_a, agent_id="task-a", llm_start=100.0, llm_end=100.05, tool_start=100.1, tool_end=100.12)
    _write_trace(trace_b, agent_id="task-b", llm_start=200.0, llm_end=200.05, tool_start=200.1, tool_end=200.12)
    _write_tasks(task_source, "task-a", "task-b")
    manifest.write_text(
        json.dumps(
            [
                {"source_trace": trace_a.name},
                {"source_trace": trace_b.name, "task_source": task_source.name},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _patch_simulator_runtime(
        monkeypatch,
        tmp_path,
        tool_delay_s=0.02,
        tool_duration_ms=20.0,
        tool_result_prefix="ok",
        llm_client_mode="forbid",
    )

    trace_file = asyncio.run(
        simulate(
            trace_manifest=manifest,
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            container_executable="docker",
            replay_speed=10.0,
        )
    )

    records = _read_jsonl(trace_file)
    metadata = records[0]
    summaries = [record for record in records if record.get("type") == "summary"]
    llm_records = [
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "llm_call"
    ]

    assert metadata["trace_manifest"] == str(manifest)
    assert metadata["source_trace_count"] == 2
    assert set(metadata["source_traces"]) == {str(trace_a), str(trace_b)}
    assert {record["agent_id"] for record in summaries} == {"task-a", "task-b"}
    assert {record["agent_id"] for record in llm_records} == {"task-a", "task-b"}
    assert abs(llm_records[0]["ts_start"] - llm_records[1]["ts_start"]) < 0.05


def test_cloud_model_mixed_host_container_manifest_marks_environment_mixed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_a = tmp_path / "trace-container.jsonl"
    trace_b = tmp_path / "trace-host.jsonl"
    task_source = tmp_path / "tasks.json"
    manifest = tmp_path / "manifest.json"
    _write_trace(trace_a, agent_id="task-a")
    _write_trace(
        trace_b,
        agent_id="task-b",
        scaffold="tongyi-deepresearch",
        execution_environment="host",
    )
    _write_tasks(task_source, "task-a", "task-b")
    manifest.write_text(
        json.dumps(
            [
                {"source_trace": trace_a.name},
                {"source_trace": trace_b.name, "task_source": task_source.name},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _patch_simulator_runtime(
        monkeypatch,
        tmp_path,
        tool_result_prefix="ok",
        llm_client_mode="forbid",
    )

    trace_file = asyncio.run(
        simulate(
            trace_manifest=manifest,
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            container_executable="docker",
            replay_speed=10.0,
        )
    )

    records = _read_jsonl(trace_file)
    assert records[0]["execution_environment"] == "mixed"
    container_records = _read_jsonl(tmp_path / "out" / "task-a" / "attempt_1" / "trace.jsonl")
    host_records = _read_jsonl(tmp_path / "out" / "task-b" / "attempt_1" / "trace.jsonl")
    assert container_records[0]["execution_environment"] == "container"
    assert container_records[0]["scaffold"] == "openclaw"
    assert host_records[0]["execution_environment"] == "host"
    assert host_records[0]["scaffold"] == "tongyi-deepresearch"


def test_cloud_model_manifest_with_docker_image_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Manifest-level docker_image overrides task image_name."""
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    manifest = tmp_path / "manifest.json"
    _write_trace(trace_path, agent_id="task-a")
    _write_tasks(task_source, "task-a")
    manifest.write_text(
        json.dumps(
            [{"source_trace": trace_path.name, "docker_image": "custom/override:latest"}]
        )
        + "\n",
        encoding="utf-8",
    )

    prepared_images: list[str] = []

    class _FakeAgent2:
        async def stop(self): pass

    async def capture_prepare(loaded, *, container_executable, network_mode="host"):
        from trace_collect.simulator import PreparedContainer, PreparedTraceSession, _resolve_docker_image
        img = _resolve_docker_image(loaded)
        prepared_images.append(img)
        container = PreparedContainer(
            container_id="fake-cid",
            container_executable=container_executable,
            docker_image=img or "",
            agent=_FakeAgent2(),
        )
        return PreparedTraceSession(loaded=loaded, container=container)

    monkeypatch.setattr("trace_collect.simulator._prepare_container_session", capture_prepare)
    async def _fake_exec(*a, **kw):
        return ("ok", 1.0, True)

    monkeypatch.setattr("trace_collect.simulator._exec_tool", _fake_exec)
    monkeypatch.setattr(
        "trace_collect.simulator.create_async_openai_client",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("no llm")),
    )

    asyncio.run(
        simulate(
            trace_manifest=manifest,
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            container_executable="docker",
        )
    )

    assert prepared_images == ["custom/override:latest"]


def test_cloud_model_rejects_task_without_docker_image(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    _write_trace(trace_path, agent_id="task-a")
    # Task without image_name or docker_image
    task_source.write_text(
        json.dumps([{"instance_id": "task-a", "problem_statement": "x"}]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="no resolvable docker_image"):
        asyncio.run(
            simulate(
                source_trace=trace_path,
                task_source=task_source,
                output_dir=tmp_path / "out",
                mode="cloud_model",
            )
        )


def test_cloud_model_manifest_keeps_default_task_source_cwd_semantics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    manifest_dir = tmp_path / "manifests"
    manifest = manifest_dir / "manifest.json"
    task_source = tmp_path / "tasks.json"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    _write_trace(trace_path, agent_id="task-a")
    _write_tasks(task_source, "task-a")
    manifest.write_text(
        json.dumps([{"source_trace": "../trace.jsonl"}]) + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    _patch_simulator_runtime(
        monkeypatch,
        tmp_path,
        tool_duration_ms=5.0,
        tool_result_prefix="ok",
        llm_client_mode="forbid",
    )

    trace_file = asyncio.run(
        simulate(
            trace_manifest=manifest,
            task_source=Path("tasks.json"),
            output_dir=tmp_path / "out",
            mode="cloud_model",
            container_executable="docker",
            replay_speed=10.0,
        )
    )

    records = _read_jsonl(trace_file)
    assert any(record.get("type") == "summary" for record in records)


def test_cloud_model_host_tool_without_success_field_is_not_mislabeled_as_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Regression: host-mode scaffold tools may not emit 'success'; host replay must
    fall back to `not error` instead of defaulting to False.
    """
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"

    # Hand-crafted trace whose tool_exec action has NO "success" key, matching
    # what vendor host-mode tools emit.
    trace_path.write_text(
        "\n".join(
            [
                json.dumps({
                    "type": "trace_metadata",
                    "trace_format_version": 5,
                    "scaffold": "tongyi-deepresearch",
                    "instance_id": "host-task",
                    "model": "qwen",
                    "mode": "collect",
                    "execution_environment": "host",
                }),
                json.dumps({
                    "type": "action",
                    "action_type": "llm_call",
                    "action_id": "host-task-llm-0",
                    "agent_id": "host-task",
                    "iteration": 0,
                    "ts_start": 100.0,
                    "ts_end": 100.2,
                    "data": {
                        "messages_in": [{"role": "user", "content": "x"}],
                        "raw_response": {"id": "r"},
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "llm_latency_ms": 200.0,
                    },
                }),
                json.dumps({
                    "type": "action",
                    "action_type": "tool_exec",
                    "action_id": "host-task-tool-0",
                    "agent_id": "host-task",
                    "iteration": 0,
                    "ts_start": 100.4,
                    "ts_end": 100.45,
                    "data": {
                        "tool_name": "web_search",
                        "args": {"query": "anything"},
                        "result": "some result",
                        "duration_ms": 50.0,
                        # NOTE: intentionally no "success" key, mirroring
                        # host-mode tool emission
                        "error": None,
                    },
                }),
                json.dumps({
                    "type": "summary",
                    "agent_id": "host-task",
                    "model": "qwen",
                    "success": True,
                    "n_iterations": 1,
                    "elapsed_s": 0.45,
                }),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_host_tasks(task_source, "host-task")

    async def fail_prepare(*args, **kwargs):
        raise AssertionError("host-mode replay must not prepare a container")

    monkeypatch.setattr(
        "trace_collect.simulator._prepare_container_session",
        fail_prepare,
    )
    monkeypatch.setattr(
        "trace_collect.simulator.create_async_openai_client",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("cloud_model must not create llm client")
        ),
    )

    trace_file = asyncio.run(
        simulate(
            source_trace=trace_path,
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            replay_speed=10.0,
        )
    )

    records = _read_jsonl(trace_file)
    tool_record = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    )
    # The bug: without the fallback, a missing "success" would default to
    # False, mislabeling valid host-mode runs and inflating failure rates.
    assert tool_record["data"]["success"] is True


# ----------------------------------------------------------------------
# Ralplan R3 Phase H2: simulator replays tongyi-deepresearch host-mode trace
# ----------------------------------------------------------------------

_TONGYI_FIXTURE = Path(__file__).parent / "fixtures" / "tongyi_deepresearch_minimal_v5.jsonl"


def test_simulator_replays_tongyi_deepresearch_trace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """R3 Principle P3 / Phase H2: host-mode host_controller traces from the
    vendored Tongyi-DeepResearch scaffold are replayed by cloud_model simulator
    without any simulator code changes, and without spinning up a container or
    creating an LLM client (host mode's defining guarantees)."""
    assert _TONGYI_FIXTURE.exists(), f"missing fixture: {_TONGYI_FIXTURE}"
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_bytes(_TONGYI_FIXTURE.read_bytes())

    task_source = tmp_path / "tasks.json"
    _write_host_tasks(task_source, "tongyi-fixture-1")

    async def _fail_prepare(*args, **kwargs):
        raise AssertionError("host-mode replay must not prepare a container")

    monkeypatch.setattr(
        "trace_collect.simulator._prepare_container_session", _fail_prepare,
    )
    monkeypatch.setattr(
        "trace_collect.simulator.create_async_openai_client",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("cloud_model must not create llm client")
        ),
    )

    trace_file = asyncio.run(
        simulate(
            source_trace=trace_path,
            task_source=task_source,
            output_dir=tmp_path / "out",
            mode="cloud_model",
            replay_speed=10.0,
        )
    )

    records = _read_jsonl(trace_file)
    metadata = records[0]
    llm_records = [
        r for r in records
        if r.get("type") == "action" and r.get("action_type") == "llm_call"
    ]
    tool_records = [
        r for r in records
        if r.get("type") == "action" and r.get("action_type") == "tool_exec"
    ]
    summary = next(r for r in records if r.get("type") == "summary")

    # Scaffold-agnostic structural invariants: the simulator respects the
    # source trace's host-mode flag and replays each action span.
    assert metadata["execution_environment"] == "host"
    assert metadata["scaffold"] == "tongyi-deepresearch"
    assert len(llm_records) == 3, "source has 3 llm_calls, simulator must replay all"
    assert len(tool_records) == 2, "source has 2 tool_execs, simulator must replay all"
    # Host-mode tool replay gets the canonical 'skipped_host_mode' tag and
    # success=True fallback, same as any host-mode scaffold.
    for tool_record in tool_records:
        assert tool_record["data"]["replay_source"] == "skipped_host_mode"
        assert tool_record["data"]["success"] is True
    assert summary["success"] is True

    # Host-mode replay must still write an empty resources.json so downstream
    # consumers see a canonical simulate layout.
    attempt_dir = tmp_path / "out" / "tongyi-fixture-1" / "attempt_1"
    resources_path = attempt_dir / "resources.json"
    assert resources_path.exists()
    payload = json.loads(resources_path.read_text())
    assert payload["samples"] == []
    assert payload["summary"]["sample_count"] == 0


def test_execution_environment_infers_host_from_agent_runtime_mode() -> None:
    """Legacy traces that predate execution_environment still replay correctly
    when agent_runtime_mode=host_controller is present. Regression guard for
    Codex P1 feedback on cc3a18a (PR #13)."""
    from types import SimpleNamespace
    from trace_collect.simulator import _execution_environment

    # Legacy host trace: no execution_environment, but agent_runtime_mode is set
    legacy_host = SimpleNamespace(
        metadata={"agent_runtime_mode": "host_controller"},
        source_trace="/tmp/legacy_host.jsonl",
    )
    assert _execution_environment(legacy_host) == "host"

    # Legacy unknown trace: nothing → container default retained
    legacy_unknown = SimpleNamespace(metadata={}, source_trace="/tmp/legacy.jsonl")
    assert _execution_environment(legacy_unknown) == "container"

    # Explicit execution_environment wins over agent_runtime_mode
    explicit_container = SimpleNamespace(
        metadata={
            "execution_environment": "container",
            "agent_runtime_mode": "host_controller",
        },
        source_trace="/tmp/explicit.jsonl",
    )
    assert _execution_environment(explicit_container) == "container"


def test_tongyi_deepresearch_fixture_is_valid_v5() -> None:
    """Sanity: the shipped fixture file parses as valid v5 JSONL with the
    expected record shape. Prevents accidental corruption during edits."""
    records = [json.loads(ln) for ln in _TONGYI_FIXTURE.read_text().splitlines() if ln.strip()]

    # 1 metadata + 3 llm_call + 2 tool_exec + 1 summary = 7 records
    assert len(records) == 7
    metadata = records[0]
    assert metadata["type"] == "trace_metadata"
    assert metadata["trace_format_version"] == 5
    assert metadata["scaffold"] == "tongyi-deepresearch"

    llm_calls = [r for r in records if r.get("action_type") == "llm_call"]
    assert [r["action_id"] for r in llm_calls] == ["llm_1", "llm_2", "llm_3"]
    for call in llm_calls:
        assert call["data"]["ttft_ms"] is not None
        assert call["data"]["tpot_ms"] is not None
        assert "logical_turn_id" in call["data"]

    tool_execs = [r for r in records if r.get("action_type") == "tool_exec"]
    assert [r["action_id"] for r in tool_execs] == ["tool_1", "tool_2"]
    for tool in tool_execs:
        # Canonical keys (R3 Principle P2)
        assert "tool_args" in tool["data"]
        assert "tool_result" in tool["data"]
        assert "duration_ms" in tool["data"]
