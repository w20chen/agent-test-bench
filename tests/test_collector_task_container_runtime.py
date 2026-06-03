"""Tests for collector task-container runtime helpers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from trace_collect.attempt_pipeline import AttemptContext
from trace_collect.collector import (
    _run_openclaw_in_task_container,
)
from trace_collect.runtime.task_container import (
    TaskContainerExecConfig,
    TaskContainerPreflightProof,
    TaskContainerRunResult,
)


def _make_ctx(tmp_path: Path, *, scaffold: str) -> AttemptContext:
    return AttemptContext(
        run_dir=tmp_path / "run",
        instance_id="encode__httpx-2701",
        attempt=1,
        task={
            "instance_id": "encode__httpx-2701",
            "repo": "encode/httpx",
            "base_commit": "deadbeef",
            "problem_statement": "Fix bug",
            "image_name": "swerebench/example",
        },
        model="qwen-plus-latest",
        scaffold=scaffold,
        source_image="swerebench/example",
        prompt_template="cc_aligned",
        agent_runtime_mode="task_container_agent",
    )


def _make_relative_ctx(monkeypatch, tmp_path: Path, *, scaffold: str) -> AttemptContext:
    monkeypatch.chdir(tmp_path)
    return AttemptContext(
        run_dir=Path("run"),
        instance_id="encode__httpx-2701",
        attempt=1,
        task={
            "instance_id": "encode__httpx-2701",
            "repo": "encode/httpx",
            "base_commit": "deadbeef",
            "problem_statement": "Fix bug",
            "image_name": "swerebench/example",
        },
        model="qwen-plus-latest",
        scaffold=scaffold,
        source_image="swerebench/example",
        prompt_template="cc_aligned",
        agent_runtime_mode="task_container_agent",
    )


def test_run_openclaw_in_task_container_normalizes_trace_on_host(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}
    preflight_seen: dict[str, object] = {}
    bootstrap_seen: dict[str, object] = {}
    ctx = _make_relative_ctx(monkeypatch, tmp_path, scaffold="openclaw")
    runtime_dir = ctx.attempt_dir / "_task_container_runtime" / "openclaw"
    stdout_path = runtime_dir / "stdout.txt"
    stderr_path = runtime_dir / "stderr.txt"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text("openclaw stdout", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    raw_trace_path = runtime_dir / "trace.raw.jsonl"
    raw_trace_path.write_text(
        json.dumps(
            {
                "type": "trace_metadata",
                "scaffold": "openclaw",
                "trace_format_version": 5,
                "model": "qwen-plus-latest",
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "action",
                "action_type": "llm_call",
                "action_id": "llm_0",
                "agent_id": "encode__httpx-2701",
                "iteration": 0,
                "ts_start": 1.0,
                "ts_end": 2.0,
                "data": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "trace_collect.collector.start_task_container",
        lambda *args, **kwargs: "cid-openclaw",
    )
    monkeypatch.setattr(
        "trace_collect.collector.stop_task_container",
        lambda *args, **kwargs: "container logs",
    )
    monkeypatch.setattr(
        "trace_collect.collector.resolve_task_container_exec_config",
        lambda **kwargs: TaskContainerExecConfig(
            runtime="/usr/bin/python3",
            pythonpath="/deps:/repo/src:/repo",
            start_extra_args=("--platform", "linux/amd64"),
            bootstrap=True,
            bootstrap_site_dir=ctx.attempt_dir / "_task_container_runtime" / "bootstrap" / "pydeps",
            image_platform="linux/amd64",
        ),
    )
    monkeypatch.setattr(
        "trace_collect.collector.resolve_running_container_exec_config",
        lambda **kwargs: kwargs["exec_config"],
    )
    monkeypatch.setattr(
        "trace_collect.collector.bootstrap_task_container_python",
        lambda **kwargs: bootstrap_seen.update(kwargs),
    )
    monkeypatch.setattr(
        "trace_collect.collector.preflight_task_container_runtime",
        lambda **kwargs: (
            preflight_seen.update(kwargs),
            TaskContainerPreflightProof(
                container_id="cid-openclaw",
                hostname="host-b",
                cwd="/testbed",
                python_executable="/opt/conda/envs/ML/bin/python",
                project_root="/work/project",
                python_prefix="/opt/conda/envs/ML",
                sys_path=["/work/project/src"],
            ),
        )[1],
    )
    monkeypatch.setattr(
        "trace_collect.collector.run_task_container_agent",
        lambda **kwargs: (
            seen.update(kwargs["request"]),
            TaskContainerRunResult(
                success=True,
                trace_path=raw_trace_path,
                model_patch="diff --git a/httpx.py b/httpx.py",
                exit_status="completed",
                error=None,
                n_iterations=4,
                total_llm_ms=12.0,
                total_tool_ms=6.0,
                total_tokens=99,
                runtime_proof={"hostname": "container-b"},
                raw_stdout_path=stdout_path,
                raw_stderr_path=stderr_path,
            ),
        )[1],
    )

    result = asyncio.run(
        _run_openclaw_in_task_container(
            ctx=ctx,
            task=dict(ctx.task),
            benchmark=SimpleNamespace(
                config=SimpleNamespace(slug="swe-rebench", harness_split="filtered")
            ),
            container_executable="docker",
            provider_name="openrouter",
            api_base="https://example.com",
            api_key="test-key",
            model="qwen-plus-latest",
            max_iterations=10,
            generation_config=None,
            max_context_tokens=1024,
            mcp_config=None,
        )
    )

    metadata = json.loads((ctx.attempt_dir / "trace.jsonl").read_text().splitlines()[0])
    assert result.trace_path == ctx.attempt_dir / "trace.jsonl"
    assert metadata["prompt_template"] == "cc_aligned"
    assert metadata["agent_runtime_mode"] == "task_container_agent"
    assert metadata["runtime_proof"]["container_id"] == "cid-openclaw"
    assert seen["kind"] == "run_openclaw"
    assert seen["container_executable"] == "docker"
    assert seen["provider_name"] == "openrouter"
    assert Path(str(seen["result_path"])).is_absolute()
    assert Path(str(seen["workspace_base"])).is_absolute()
    assert Path(str(seen["workspace_dir"])).is_absolute()
    assert Path(str(seen["trace_file"])).is_absolute()
    assert Path(str(seen["raw_stdout_path"])).is_absolute()
    assert Path(str(seen["raw_stderr_path"])).is_absolute()
    assert Path(str(seen["result_path"])) == runtime_dir.resolve() / "run.result.json"
    assert seen["tool_workspace"] == "/testbed"
    assert seen["exec_working_dir"] == "/testbed"
    assert preflight_seen["runtime"] == "/usr/bin/python3"
    assert preflight_seen["pythonpath"] == "/deps:/repo/src:/repo"
    assert preflight_seen["container_executable"] == "docker"
    assert preflight_seen["imports"] == [
        "trace_collect.runtime.entrypoint",
        "agents.openclaw.eval.runner",
        "harness.trace_logger",
    ]
    assert bootstrap_seen["container_executable"] == "docker"
    assert bootstrap_seen["extra_requirements"] == ()
    assert result.total_llm_ms == 12.0
    assert result.total_tool_ms == 6.0
    assert result.total_tokens == 99
    assert "openclaw stdout" in ctx.container_stdout


def test_run_openclaw_in_task_container_adds_mcp_bootstrap_requirements(
    tmp_path: Path,
    monkeypatch,
) -> None:
    preflight_seen: dict[str, object] = {}
    bootstrap_seen: dict[str, object] = {}
    ctx = _make_relative_ctx(monkeypatch, tmp_path, scaffold="openclaw")
    runtime_dir = ctx.attempt_dir / "_task_container_runtime" / "openclaw"
    stdout_path = runtime_dir / "stdout.txt"
    stderr_path = runtime_dir / "stderr.txt"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text("openclaw stdout", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    raw_trace_path = runtime_dir / "trace.raw.jsonl"
    raw_trace_path.write_text(
        '{"type":"trace_metadata","scaffold":"openclaw","trace_format_version":5}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "trace_collect.collector.start_task_container",
        lambda *args, **kwargs: "cid-openclaw",
    )
    monkeypatch.setattr(
        "trace_collect.collector.stop_task_container",
        lambda *args, **kwargs: "container logs",
    )
    monkeypatch.setattr(
        "trace_collect.collector.resolve_task_container_exec_config",
        lambda **kwargs: TaskContainerExecConfig(
            runtime="/usr/bin/python3",
            pythonpath="/deps:/repo/src:/repo",
            start_extra_args=(),
            bootstrap=True,
            bootstrap_site_dir=ctx.attempt_dir / "_task_container_runtime" / "bootstrap" / "pydeps",
            image_platform="linux/amd64",
        ),
    )
    monkeypatch.setattr(
        "trace_collect.collector.resolve_running_container_exec_config",
        lambda **kwargs: kwargs["exec_config"],
    )
    monkeypatch.setattr(
        "trace_collect.collector.bootstrap_task_container_python",
        lambda **kwargs: bootstrap_seen.update(kwargs),
    )
    monkeypatch.setattr(
        "trace_collect.collector.preflight_task_container_runtime",
        lambda **kwargs: (
            preflight_seen.update(kwargs),
            TaskContainerPreflightProof(
                container_id="cid-openclaw",
                hostname="host-b",
                cwd="/testbed",
                python_executable="/usr/bin/python3",
                project_root="/work/project",
                python_prefix="/usr",
                sys_path=["/work/project/src"],
            ),
        )[1],
    )
    monkeypatch.setattr(
        "trace_collect.collector.run_task_container_agent",
        lambda **kwargs: TaskContainerRunResult(
            success=True,
            trace_path=raw_trace_path,
            model_patch="diff --git a/httpx.py b/httpx.py",
            exit_status="completed",
            error=None,
            n_iterations=1,
            total_llm_ms=1.0,
            total_tool_ms=1.0,
            total_tokens=1,
            runtime_proof={"hostname": "container-b"},
            raw_stdout_path=stdout_path,
            raw_stderr_path=stderr_path,
        ),
    )

    asyncio.run(
        _run_openclaw_in_task_container(
            ctx=ctx,
            task=dict(ctx.task),
            benchmark=SimpleNamespace(
                config=SimpleNamespace(slug="swe-rebench", harness_split="filtered")
            ),
            container_executable="docker",
            provider_name="openrouter",
            api_base="https://example.com",
            api_key="test-key",
            model="qwen-plus-latest",
            max_iterations=10,
            generation_config=None,
            max_context_tokens=1024,
            mcp_config="configs/mcp/context7.yaml",
        )
    )

    assert bootstrap_seen["extra_requirements"] == ("mcp>=1.0",)
    assert preflight_seen["imports"] == [
        "trace_collect.runtime.entrypoint",
        "agents.openclaw.eval.runner",
        "harness.trace_logger",
        "agents.openclaw.tools.mcp",
    ]
