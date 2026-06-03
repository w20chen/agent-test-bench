from __future__ import annotations

import asyncio
import json
import signal
import subprocess
import threading
from pathlib import Path

import pytest

from agents.terminal_bench.openclaw_agent import TerminalBenchOpenClawAgent
from agents.terminal_bench.runner import TerminalBenchRunner
from trace_collect.attempt_pipeline import AttemptContext


def _make_runner(**overrides) -> TerminalBenchRunner:
    kwargs = {
        "provider_name": "openrouter",
        "env_key": "OPENROUTER_API_KEY",
        "api_base": "https://openrouter.ai/api/v1",
        "api_key": "test-key",
        "model": "z-ai/glm-5.1",
        "workspace_base": Path("workspace"),
        "max_iterations": 100,
        "context_window_tokens": 256_000,
        "benchmark_slug": "terminal-bench",
        "benchmark_extras": {
            "dataset_name": "terminal-bench-core",
            "dataset_version": "head",
        },
        "mcp_config": None,
    }
    kwargs.update(overrides)
    return TerminalBenchRunner(
        **kwargs,
    )


def _make_ctx(tmp_path: Path) -> AttemptContext:
    return AttemptContext(
        run_dir=tmp_path / "run",
        instance_id="hello-world",
        attempt=1,
        task={"instance_id": "hello-world"},
        model="z-ai/glm-5.1",
        scaffold="openclaw",
        source_image=None,
    )


def test_preflight_requires_tb_and_docker(monkeypatch) -> None:
    runner = _make_runner()
    monkeypatch.setattr(
        "agents.terminal_bench.runner.importlib.util.find_spec", lambda name: object()
    )
    monkeypatch.setattr("agents.terminal_bench.runner.shutil.which", lambda name: None)
    with pytest.raises(RuntimeError, match="tb CLI"):
        runner._preflight()


def test_build_tb_command_uses_agent_import_path() -> None:
    runner = _make_runner()
    cmd = runner._build_tb_command(
        task={"dataset_root": "/tmp/dataset", "task_id": "hello-world"},
        run_root=Path("/tmp/out"),
        run_id="hello-world",
        prompt_template="default",
    )
    joined = " ".join(cmd)
    assert "tb run" in joined
    assert TerminalBenchRunner.AGENT_IMPORT_PATH in joined
    assert "--dataset-path /tmp/dataset" in joined
    assert "--task-id hello-world" in joined
    assert "--n-attempts 1" in joined
    assert "max_iterations=100" in joined
    assert "api_key=test-key" not in joined


def test_build_tb_command_forwards_global_agent_timeout() -> None:
    runner = _make_runner(
        benchmark_extras={
            "dataset_name": "terminal-bench-core",
            "dataset_version": "head",
            "global_agent_timeout_sec": 7200.0,
        },
    )
    cmd = runner._build_tb_command(
        task={"dataset_root": "/tmp/dataset", "task_id": "hello-world"},
        run_root=Path("/tmp/out"),
        run_id="hello-world",
        prompt_template="default",
    )
    joined = " ".join(cmd)
    assert "--global-agent-timeout-sec 7200.0" in joined
    assert "--agent-kwarg agent_timeout_sec=7200.0" in joined


def test_build_tb_command_forwards_llm_timeout_to_agent() -> None:
    runner = _make_runner(
        benchmark_extras={
            "dataset_name": "terminal-bench-core",
            "dataset_version": "head",
            "llm_timeout_sec": 1800.0,
        },
    )
    cmd = runner._build_tb_command(
        task={"dataset_root": "/tmp/dataset", "task_id": "hello-world"},
        run_root=Path("/tmp/out"),
        run_id="hello-world",
        prompt_template="default",
    )
    joined = " ".join(cmd)
    assert "--agent-kwarg llm_timeout_sec=1800.0" in joined


def test_build_tb_command_forwards_task_timeout_without_global() -> None:
    runner = _make_runner(
        benchmark_extras={
            "dataset_name": "terminal-bench-core",
            "dataset_version": "head",
        },
    )
    cmd = runner._build_tb_command(
        task={
            "dataset_root": "/tmp/dataset",
            "task_id": "hello-world",
            "max_agent_timeout_sec": 600.0,
        },
        run_root=Path("/tmp/out"),
        run_id="hello-world",
        prompt_template="default",
    )
    joined = " ".join(cmd)
    assert "--global-agent-timeout-sec" not in joined
    assert "--agent-kwarg agent_timeout_sec=600.0" in joined


def test_terminal_bench_agent_exports_llm_timeout_env() -> None:
    agent = TerminalBenchOpenClawAgent(
        model_name="local-model",
        provider_name="openai",
        api_base="http://127.0.0.1:1234/v1",
        env_key="OPENAI_API_KEY",
        api_key="dummy",
        llm_timeout_sec=1800.0,
    )

    assert agent._env["OPENCLAW_LLM_TIMEOUT_S"] == "1800.0"


def test_global_agent_timeout_must_be_positive() -> None:
    for timeout in (0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="global_agent_timeout_sec"):
            _make_runner(
                benchmark_extras={
                    "dataset_name": "terminal-bench-core",
                    "dataset_version": "head",
                    "global_agent_timeout_sec": timeout,
                },
            )


def test_tb_process_cleanup_grace_must_be_positive() -> None:
    for timeout in (0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="tb_process_cleanup_grace_sec"):
            _make_runner(
                benchmark_extras={
                    "dataset_name": "terminal-bench-core",
                    "dataset_version": "head",
                    "tb_process_cleanup_grace_sec": timeout,
                },
            )


def test_build_tb_command_materializes_prompt_template(tmp_path: Path) -> None:
    runner = _make_runner()
    cmd = runner._build_tb_command(
        task={"dataset_root": "/tmp/dataset", "task_id": "hello-world"},
        run_root=tmp_path,
        run_id="hello-world",
        prompt_template="default",
    )
    kwargs = _agent_kwargs(cmd)
    prompt_template_path = Path(kwargs["prompt_template"])
    assert prompt_template_path.exists()
    content = prompt_template_path.read_text(encoding="utf-8")
    assert "{{ instruction }}" in content
    assert "{{task}}" not in content


def test_build_tb_command_forwards_mcp_config_path(tmp_path: Path) -> None:
    mcp_config = tmp_path / "context7.yaml"
    mcp_config.write_text("mcpServers: {}\n", encoding="utf-8")
    runner = _make_runner(mcp_config=str(mcp_config))
    cmd = runner._build_tb_command(
        task={"dataset_root": "/tmp/dataset", "task_id": "hello-world"},
        run_root=tmp_path,
        run_id="hello-world",
        prompt_template="default",
    )
    kwargs = _agent_kwargs(cmd)
    assert kwargs["mcp_config_path"] == str(mcp_config.resolve())


def test_extract_success_reads_terminal_bench_results(tmp_path: Path) -> None:
    runner = _make_runner()
    run_path = tmp_path / "tb-run"
    run_path.mkdir()
    (run_path / "results.json").write_text(
        json.dumps({"results": [{"is_resolved": True}]}),
        encoding="utf-8",
    )
    assert runner._extract_success(run_path) is True


def test_find_trace_path_prefers_agent_logs(tmp_path: Path) -> None:
    runner = _make_runner()
    trace = tmp_path / "task" / "trial" / "agent-logs" / "openclaw-trace.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text("{}\n", encoding="utf-8")
    assert runner._find_trace_path(tmp_path) == trace


def test_augment_trace_metadata_stamps_terminal_bench_fields(tmp_path: Path) -> None:
    runner = _make_runner(
        mcp_config="configs/mcp/context7.yaml",
        benchmark_extras={
            "dataset_name": "terminal-bench-core",
            "dataset_version": "head",
            "global_agent_timeout_sec": 7200.0,
        },
    )
    src = tmp_path / "src.jsonl"
    src.write_text(
        json.dumps({"type": "trace_metadata", "model": "old", "instance_id": "x"})
        + "\n"
        + json.dumps({"type": "summary", "success": True})
        + "\n",
        encoding="utf-8",
    )
    dst = tmp_path / "dst.jsonl"
    runner._augment_trace_metadata(
        src=src,
        dst=dst,
        task={
            "instance_id": "hello-world",
            "task_source_kind": "terminal_bench_registry",
            "task_source_id": "hello-world",
            "task_source_path": "/tmp/dataset/hello-world",
            "tb_dataset": "terminal-bench-core",
            "tb_registry_source": "registry.json",
        },
        prompt_template="default",
        tb_version="0.2.18",
    )
    metadata = json.loads(dst.read_text(encoding="utf-8").splitlines()[0])
    assert metadata["benchmark"] == "terminal-bench"
    assert metadata["execution_environment"] == "container"
    assert metadata["agent_runtime_mode"] == "host_controller"
    assert metadata["tb_version"] == "0.2.18"
    assert metadata["task_source_kind"] == "terminal_bench_registry"
    assert metadata["run_config"]["mcp_config"] == "context7.yaml"
    assert metadata["run_config"]["global_agent_timeout_sec"] == 7200.0
    assert metadata["run_config"]["tb_process_cleanup_grace_sec"] == 300.0


def test_run_openclaw_task_publishes_terminal_bench_container_name(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _make_runner()
    ctx = _make_ctx(tmp_path)
    task = {
        "instance_id": "hello-world",
        "task_id": "hello-world",
        "dataset_root": "/tmp/dataset",
        "task_source_kind": "terminal_bench_registry",
        "task_source_id": "hello-world",
        "task_source_path": "/tmp/dataset/hello-world",
    }

    monkeypatch.setattr(
        runner,
        "_preflight",
        lambda: {
            "tb_version": "0.2.18",
            "tb_path": "/usr/bin/tb",
            "docker_path": "/usr/bin/docker",
            "agent_runtime_mode": "host_controller",
        },
    )

    def fake_run_tb_process(**kwargs):
        tb_run_path = ctx.attempt_dir / "_terminal_bench_run" / "hello-world"
        tb_run_path.mkdir(parents=True)
        (tb_run_path / "results.json").write_text(
            json.dumps({"results": [{"is_resolved": True}]}),
            encoding="utf-8",
        )
        trace_path = (
            tb_run_path
            / "hello-world"
            / "hello-world.1-of-1.hello-world"
            / "agent-logs"
            / TerminalBenchRunner.TRACE_FILENAME
        )
        trace_path.parent.mkdir(parents=True)
        trace_path.write_text(
            json.dumps({"type": "trace_metadata"}) + "\n",
            encoding="utf-8",
        )
        logs = runner._write_tb_process_logs(
            run_root=kwargs["run_root"],
            stdout="",
            stderr="",
        )
        return subprocess.CompletedProcess(kwargs["command"], 0, "", ""), logs

    monkeypatch.setattr(runner, "_run_tb_process", fake_run_tb_process)

    result = runner._run_openclaw_task_sync(
        task,
        attempt_ctx=ctx,
        prompt_template="default",
    )

    assert result.success is True
    assert ctx.container_id == "hello-world-1-of-1-hello-world"
    metadata = json.loads(result.trace_path.read_text(encoding="utf-8").splitlines()[0])
    assert metadata["execution_environment"] == "container"


def test_tb_process_timeout_kills_process_group_and_container(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _make_runner(
        benchmark_extras={
            "dataset_name": "terminal-bench-core",
            "dataset_version": "head",
            "global_agent_timeout_sec": 10.0,
            "tb_process_cleanup_grace_sec": 2.0,
        },
    )
    task_dir = tmp_path / "tasks" / "hello-world"
    task_dir.mkdir(parents=True)
    (task_dir / "docker-compose.yaml").write_text(
        "\n".join(
            [
                "services:",
                "  client:",
                "    image: ${T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME}",
                "    container_name: ${T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME}",
                "    volumes:",
                "      - ${T_BENCH_TASK_LOGS_PATH}:${T_BENCH_CONTAINER_LOGS_PATH}",
                "      - ${T_BENCH_TASK_AGENT_LOGS_PATH}:${T_BENCH_CONTAINER_AGENT_LOGS_PATH}",
                "      - ${T_BENCH_TEST_DIR}:${T_BENCH_TEST_DIR}:ro",
                "  sidecar:",
                "    image: busybox",
                "    container_name: ${T_BENCH_TASK_DOCKER_NAME_PREFIX}__sidecar",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run_root = tmp_path / "run"
    seen: dict[str, object] = {"docker": [], "killpg": [], "timeouts": []}

    class FakePopen:
        def __init__(self, *args, **kwargs) -> None:
            seen["popen_args"] = args
            seen["popen_kwargs"] = kwargs
            self.pid = 12345
            self.returncode = None
            self._timed_out = False

        def communicate(self, timeout=None):
            seen["timeouts"].append(timeout)
            if not self._timed_out:
                self._timed_out = True
                raise subprocess.TimeoutExpired(
                    cmd=["tb"],
                    timeout=timeout,
                    output="partial stdout",
                    stderr="partial stderr",
                )
            self.returncode = -signal.SIGKILL
            return "post-kill stdout", "post-kill stderr"

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = -signal.SIGTERM
            return self.returncode

    def fake_run(cmd, **kwargs):
        seen["docker"].append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def fake_killpg(pid: int, signum: int) -> None:
        seen["killpg"].append((pid, signum))

    monkeypatch.setattr("agents.terminal_bench.runner.subprocess.Popen", FakePopen)
    monkeypatch.setattr("agents.terminal_bench.runner.subprocess.run", fake_run)
    monkeypatch.setattr("agents.terminal_bench.runner.os.killpg", fake_killpg)

    completed, logs = runner._run_tb_process(
        command=["tb", "run"],
        cwd=tmp_path,
        env={},
        run_root=run_root,
        task={
            "task_id": "hello-world",
            "task_source_path": str(task_dir),
            "max_test_timeout_sec": 3.0,
        },
        run_id="hello-world",
    )

    assert seen["timeouts"][0] == 15.0
    assert (12345, signal.SIGTERM) in seen["killpg"]
    docker_commands = [call[0] for call in seen["docker"]]
    compose_command = [
        "docker",
        "compose",
        "-p",
        "hello-world-1-of-1-hello-world",
        "-f",
        str((task_dir / "docker-compose.yaml").resolve()),
        "down",
        "--remove-orphans",
    ]
    assert compose_command in docker_commands
    assert ["docker", "rm", "-f", "hello-world-1-of-1-hello-world"] in docker_commands
    compose_call = next(call for call in seen["docker"] if call[0] == compose_command)
    compose_env = compose_call[1]["env"]
    trial_path = (
        run_root / "hello-world" / "hello-world" / "hello-world.1-of-1.hello-world"
    )
    assert compose_env["T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME"] == (
        "hello-world-1-of-1-hello-world"
    )
    assert compose_env["T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME"] == (
        "tb__hello-world__client"
    )
    assert compose_env["T_BENCH_TASK_DOCKER_NAME_PREFIX"] == "tb__hello-world"
    assert compose_env["T_BENCH_CONTAINER_LOGS_PATH"] == "/logs"
    assert compose_env["T_BENCH_CONTAINER_AGENT_LOGS_PATH"] == "/agent-logs"
    assert compose_env["T_BENCH_TEST_DIR"] == "/tests"
    assert compose_env["T_BENCH_TASK_LOGS_PATH"] == str(
        (trial_path / "sessions").resolve()
    )
    assert compose_env["T_BENCH_TASK_AGENT_LOGS_PATH"] == str(
        (trial_path / "agent-logs").resolve()
    )
    assert completed.returncode != 0
    assert "timed out after 15.0s" in completed.stderr
    assert "partial stderr" in completed.stderr
    assert "post-kill stderr" in completed.stderr
    assert "partial stdout" in completed.stdout
    assert "post-kill stdout" in completed.stdout
    assert Path(logs["tb_stderr_path"]).read_text(encoding="utf-8") == completed.stderr


def test_active_process_cleanup_reaches_async_to_thread_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _make_runner()
    ctx = _make_ctx(tmp_path)
    task_dir = tmp_path / "tasks" / "hello-world"
    task_dir.mkdir(parents=True)
    (task_dir / "docker-compose.yaml").write_text("services: {}\n", encoding="utf-8")
    task = {
        "instance_id": "hello-world",
        "task_id": "hello-world",
        "dataset_root": "/tmp/dataset",
        "task_source_kind": "terminal_bench_registry",
        "task_source_id": "hello-world",
        "task_source_path": str(task_dir),
        "max_agent_timeout_sec": 600.0,
    }
    communicating = threading.Event()
    cleaned = threading.Event()
    seen: dict[str, object] = {"docker": [], "killpg": [], "timeouts": []}

    monkeypatch.setattr(
        runner,
        "_preflight",
        lambda: {
            "tb_version": "0.2.18",
            "tb_path": "/usr/bin/tb",
            "docker_path": "/usr/bin/docker",
            "agent_runtime_mode": "host_controller",
        },
    )

    class FakePopen:
        def __init__(self, *args, **kwargs) -> None:
            seen["popen_args"] = args
            seen["popen_kwargs"] = kwargs
            self.pid = 54321
            self.returncode = None

        def communicate(self, timeout=None):
            seen["timeouts"].append(timeout)
            communicating.set()
            if not cleaned.wait(timeout=5):
                raise AssertionError("cleanup was not triggered for async process")
            self.returncode = -signal.SIGTERM
            return "stdout after cleanup", "stderr after cleanup"

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = -signal.SIGTERM
            cleaned.set()
            return self.returncode

    def fake_run(cmd, **kwargs):
        seen["docker"].append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def fake_killpg(pid: int, signum: int) -> None:
        seen["killpg"].append((pid, signum))

    monkeypatch.setattr("agents.terminal_bench.runner.subprocess.Popen", FakePopen)
    monkeypatch.setattr("agents.terminal_bench.runner.subprocess.run", fake_run)
    monkeypatch.setattr("agents.terminal_bench.runner.os.killpg", fake_killpg)

    async def run_and_cleanup() -> None:
        task_future = asyncio.create_task(
            runner.run_openclaw_task(
                task,
                attempt_ctx=ctx,
                prompt_template="default",
            )
        )
        assert await asyncio.to_thread(communicating.wait, 5)
        TerminalBenchRunner._cleanup_active_processes()
        with pytest.raises(RuntimeError, match="terminal-bench run failed"):
            await task_future

    asyncio.run(run_and_cleanup())

    assert (54321, signal.SIGTERM) in seen["killpg"]
    docker_commands = [call[0] for call in seen["docker"]]
    assert [
        "docker",
        "compose",
        "-p",
        "hello-world-1-of-1-hello-world",
        "-f",
        str((task_dir / "docker-compose.yaml").resolve()),
        "down",
        "--remove-orphans",
    ] in docker_commands
    assert ["docker", "rm", "-f", "hello-world-1-of-1-hello-world"] in docker_commands


def _agent_kwargs(cmd: list[str]) -> dict[str, str]:
    kwargs: dict[str, str] = {}
    for index, token in enumerate(cmd):
        if token != "--agent-kwarg":
            continue
        key, value = cmd[index + 1].split("=", 1)
        kwargs[key] = value
    return kwargs
