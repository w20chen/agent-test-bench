from __future__ import annotations

import asyncio
import importlib.metadata
import importlib.util
import json
import logging
import math
import os
import signal
import shutil
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trace_collect.attempt_pipeline import AttemptContext, AttemptResult

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ActiveTBProcess:
    process: subprocess.Popen[str]
    cleanup: Callable[[], None]


def _optional_positive_float(value: Any, *, name: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive float, got {value!r}") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be a finite positive float, got {value!r}")
    return parsed


class TerminalBenchRunner:
    AGENT_IMPORT_PATH = (
        "agents.terminal_bench.openclaw_agent:TerminalBenchOpenClawAgent"
    )
    TRACE_FILENAME = "openclaw-trace.jsonl"
    N_ATTEMPTS = 1
    DEFAULT_TB_PROCESS_CLEANUP_GRACE_SEC = 300.0
    PROCESS_TERMINATE_GRACE_SEC = 10.0
    CONTAINER_SESSION_LOGS_PATH = "/logs"
    CONTAINER_AGENT_LOGS_PATH = "/agent-logs"
    CONTAINER_TEST_DIR = "/tests"

    _ACTIVE_PROCESS_LOCK = threading.RLock()
    _ACTIVE_PROCESSES: dict[int, _ActiveTBProcess] = {}
    _SIGNAL_HANDLERS_INSTALLED = False
    _PREVIOUS_SIGNAL_HANDLERS: dict[int, Any] = {}

    def __init__(
        self,
        *,
        provider_name: str | None,
        env_key: str | None,
        api_base: str,
        api_key: str,
        model: str,
        workspace_base: Path,
        max_iterations: int,
        context_window_tokens: int,
        benchmark_slug: str,
        benchmark_extras: dict[str, Any],
        mcp_config: str | None = None,
        generation_config: dict[str, Any] | None = None,
    ) -> None:
        self.provider_name = provider_name or "openai"
        self.env_key = env_key or "OPENAI_API_KEY"
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.workspace_base = Path(workspace_base)
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
        self.benchmark_slug = benchmark_slug
        self.benchmark_extras = dict(benchmark_extras)
        self.generation_config = dict(generation_config or {})
        self.global_agent_timeout_sec = _optional_positive_float(
            self.benchmark_extras.get("global_agent_timeout_sec"),
            name="benchmark_extras.global_agent_timeout_sec",
        )
        self.llm_timeout_sec = _optional_positive_float(
            self.benchmark_extras.get("llm_timeout_sec"),
            name="benchmark_extras.llm_timeout_sec",
        )
        self.tb_process_cleanup_grace_sec = (
            _optional_positive_float(
                self.benchmark_extras.get("tb_process_cleanup_grace_sec"),
                name="benchmark_extras.tb_process_cleanup_grace_sec",
            )
            or self.DEFAULT_TB_PROCESS_CLEANUP_GRACE_SEC
        )
        self.mcp_config = self._resolve_mcp_config(mcp_config)
        self.mcp_config_label = self._mcp_config_label(mcp_config)
        self._install_global_signal_handlers()

    async def run_openclaw_task(
        self,
        task: dict[str, Any],
        *,
        attempt_ctx: AttemptContext,
        prompt_template: str,
    ) -> AttemptResult:
        self._install_global_signal_handlers()
        return await asyncio.to_thread(
            self._run_openclaw_task_sync,
            task,
            attempt_ctx,
            prompt_template,
        )

    async def run_task(
        self,
        task: dict[str, Any],
        *,
        attempt_ctx: AttemptContext,
        prompt_template: str,
    ) -> AttemptResult:
        return await self.run_openclaw_task(
            task,
            attempt_ctx=attempt_ctx,
            prompt_template=prompt_template,
        )

    def _run_openclaw_task_sync(
        self,
        task: dict[str, Any],
        attempt_ctx: AttemptContext,
        prompt_template: str,
    ) -> AttemptResult:
        proof = self._preflight()
        run_root = attempt_ctx.attempt_dir / "_terminal_bench_run"
        run_id = attempt_ctx.instance_id.replace("/", "_")
        run_root.mkdir(parents=True, exist_ok=True)

        attempt_ctx.mark_container_ready(
            self._expected_client_container_name(
                task_id=str(task["task_id"]),
                run_id=run_id,
            )
        )
        command = self._build_tb_command(
            task=task,
            run_root=run_root,
            run_id=run_id,
            prompt_template=prompt_template,
        )
        env = os.environ.copy()
        repo_root = Path(__file__).resolve().parents[3]
        pythonpath = f"{repo_root / 'src'}:{repo_root}:{env.get('PYTHONPATH', '')}"
        env["PYTHONPATH"] = pythonpath.rstrip(":")
        env[self.env_key] = self.api_key
        tb_run_path = run_root / run_id
        completed, tb_process_logs = self._run_tb_process(
            command=command,
            cwd=repo_root,
            env=env,
            run_root=run_root,
            task=task,
            run_id=run_id,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "terminal-bench run failed: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )

        success = self._extract_success(tb_run_path)
        trace_path = self._find_trace_path(tb_run_path)
        summary = self._summary(
            tb_version=proof["tb_version"],
            task=task,
            tb_run_path=tb_run_path,
            tb_process_logs=tb_process_logs,
        )
        if not trace_path.exists():
            raise RuntimeError(
                f"terminal-bench trace file not found under {tb_run_path}"
            )
        normalized_trace = (
            run_root / f"{attempt_ctx.instance_id}-terminal-bench-trace.jsonl"
        )
        self._augment_trace_metadata(
            src=trace_path,
            dst=normalized_trace,
            task=task,
            prompt_template=prompt_template,
            tb_version=proof["tb_version"],
        )
        return AttemptResult(
            success=success,
            exit_status="completed" if success else "failed",
            trace_path=normalized_trace,
            model_patch="",
            error=None if success else "Terminal-Bench task did not resolve",
            summary=summary,
            runtime_proof=proof,
        )

    @staticmethod
    def _write_tb_process_logs(
        *,
        run_root: Path,
        stdout: str,
        stderr: str,
    ) -> dict[str, str]:
        run_root.mkdir(parents=True, exist_ok=True)
        stdout_path = run_root / "tb-run-stdout.txt"
        stderr_path = run_root / "tb-run-stderr.txt"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        return {
            "tb_stdout_path": str(stdout_path),
            "tb_stderr_path": str(stderr_path),
        }

    def _preflight(self) -> dict[str, Any]:
        if importlib.util.find_spec("terminal_bench") is None:
            raise RuntimeError("terminal-bench Python package is not installed")
        tb_bin = shutil.which("tb")
        if not tb_bin:
            raise RuntimeError("tb CLI is not available on PATH")
        docker_bin = shutil.which("docker")
        if not docker_bin:
            raise RuntimeError("docker is not available on PATH")
        return {
            "tb_version": importlib.metadata.version("terminal-bench"),
            "tb_path": tb_bin,
            "docker_path": docker_bin,
            "agent_runtime_mode": "host_controller",
        }

    @classmethod
    def _expected_client_container_name(cls, *, task_id: str, run_id: str) -> str:
        trial_name = cls._trial_name(task_id=task_id, run_id=run_id)
        return trial_name.replace(".", "-")

    @classmethod
    def _trial_name(cls, *, task_id: str, run_id: str) -> str:
        return f"{task_id}.1-of-{cls.N_ATTEMPTS}.{run_id}"

    def _build_tb_command(
        self,
        *,
        task: dict[str, Any],
        run_root: Path,
        run_id: str,
        prompt_template: str,
    ) -> list[str]:
        dataset_root = task["dataset_root"]
        task_id = task["task_id"]
        command = [
            "tb",
            "run",
            "--dataset-path",
            str(dataset_root),
            "--task-id",
            str(task_id),
            "--output-path",
            str(run_root),
            "--run-id",
            run_id,
            "--n-concurrent",
            "1",
            "--n-attempts",
            str(self.N_ATTEMPTS),
            "--agent-import-path",
            self.AGENT_IMPORT_PATH,
            "--model",
            self.model,
            "--agent-kwarg",
            f"provider_name={self.provider_name}",
            "--agent-kwarg",
            f"api_base={self.api_base}",
            "--agent-kwarg",
            f"env_key={self.env_key}",
            "--agent-kwarg",
            f"max_iterations={self.max_iterations}",
        ]
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
        ):
            if key in self.generation_config:
                command.extend(
                    [
                        "--agent-kwarg",
                        f"{key}={self.generation_config[key]}",
                    ]
                )
        if self.llm_timeout_sec is not None:
            command.extend(
                [
                    "--agent-kwarg",
                    f"llm_timeout_sec={self.llm_timeout_sec}",
                ]
            )
        agent_timeout_sec = self._agent_timeout_for_task(task)
        if agent_timeout_sec is not None:
            command.extend(
                [
                    "--agent-kwarg",
                    f"agent_timeout_sec={agent_timeout_sec}",
                ]
            )
        if self.global_agent_timeout_sec is not None:
            command.extend(
                [
                    "--global-agent-timeout-sec",
                    str(self.global_agent_timeout_sec),
                ]
            )
        prompt_template_path = self._materialize_prompt_template(
            prompt_template=prompt_template,
            run_root=run_root,
        )
        command.extend(
            [
                "--agent-kwarg",
                f"prompt_template={prompt_template_path}",
            ]
        )
        if self.mcp_config is not None:
            command.extend(
                [
                    "--agent-kwarg",
                    f"mcp_config_path={self.mcp_config}",
                ]
            )
        return command

    def _agent_timeout_for_task(self, task: dict[str, Any]) -> float | None:
        if self.global_agent_timeout_sec is not None:
            return self.global_agent_timeout_sec
        return _optional_positive_float(
            task.get("max_agent_timeout_sec"),
            name="task.max_agent_timeout_sec",
        )

    def _tb_process_timeout_for_task(self, task: dict[str, Any]) -> float | None:
        agent_timeout_sec = self._agent_timeout_for_task(task)
        if agent_timeout_sec is None:
            return None
        test_timeout_sec = _optional_positive_float(
            task.get("max_test_timeout_sec"),
            name="task.max_test_timeout_sec",
        )
        return (
            agent_timeout_sec
            + (test_timeout_sec or 0.0)
            + self.tb_process_cleanup_grace_sec
        )

    def _run_tb_process(
        self,
        *,
        command: list[str],
        cwd: Path,
        env: dict[str, str],
        run_root: Path,
        task: dict[str, Any],
        run_id: str,
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
        timeout_sec = self._tb_process_timeout_for_task(task)
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )

        def cleanup() -> None:
            self._cleanup_tb_process(
                process=process,
                task=task,
                run_id=run_id,
                run_root=run_root,
            )

        self._register_active_process(process=process, cleanup=cleanup)
        try:
            try:
                stdout, stderr = process.communicate(timeout=timeout_sec)
                completed = subprocess.CompletedProcess(
                    command,
                    process.returncode,
                    stdout,
                    stderr,
                )
            except subprocess.TimeoutExpired as exc:
                partial_stdout = self._timeout_stream_to_text(exc.stdout or exc.output)
                partial_stderr = self._timeout_stream_to_text(exc.stderr)
                self._cleanup_tb_process(
                    process=process,
                    task=task,
                    run_id=run_id,
                    run_root=run_root,
                )
                stdout, stderr = self._communicate_after_termination(process)
                stdout = self._merge_streams(partial_stdout, stdout)
                stderr = self._merge_streams(partial_stderr, stderr)
                message = (
                    f"terminal-bench process timed out after {timeout_sec}s; "
                    "terminated process group and cleaned task container"
                )
                completed = subprocess.CompletedProcess(
                    command,
                    process.returncode
                    if process.returncode is not None
                    else -signal.SIGKILL,
                    stdout,
                    self._append_stderr(stderr, message),
                )
        except BaseException:
            self._cleanup_tb_process(
                process=process,
                task=task,
                run_id=run_id,
                run_root=run_root,
            )
            stdout, stderr = self._communicate_after_termination(process)
            self._write_tb_process_logs(
                run_root=run_root,
                stdout=stdout,
                stderr=stderr,
            )
            raise
        finally:
            self._unregister_active_process(process)

        logs = self._write_tb_process_logs(
            run_root=run_root,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )
        return completed, logs

    @classmethod
    def _install_global_signal_handlers(cls) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        with cls._ACTIVE_PROCESS_LOCK:
            if cls._SIGNAL_HANDLERS_INSTALLED:
                return
            for signum in (signal.SIGINT, signal.SIGTERM):
                cls._PREVIOUS_SIGNAL_HANDLERS[signum] = signal.getsignal(signum)
                signal.signal(signum, cls._handle_process_signal)
            cls._SIGNAL_HANDLERS_INSTALLED = True

    @classmethod
    def _handle_process_signal(cls, signum: int, frame: Any) -> None:
        cls._cleanup_active_processes()
        previous = cls._PREVIOUS_SIGNAL_HANDLERS.get(signum, signal.SIG_DFL)
        if previous == signal.SIG_IGN:
            return
        if callable(previous):
            previous(signum, frame)
            return
        if signum == signal.SIGINT:
            raise KeyboardInterrupt(f"received signal {signum}")
        raise SystemExit(128 + signum)

    @classmethod
    def _register_active_process(
        cls,
        *,
        process: subprocess.Popen[str],
        cleanup: Callable[[], None],
    ) -> None:
        with cls._ACTIVE_PROCESS_LOCK:
            cls._ACTIVE_PROCESSES[process.pid] = _ActiveTBProcess(
                process=process,
                cleanup=cleanup,
            )

    @classmethod
    def _unregister_active_process(
        cls,
        process: subprocess.Popen[str],
    ) -> None:
        with cls._ACTIVE_PROCESS_LOCK:
            cls._ACTIVE_PROCESSES.pop(process.pid, None)

    @classmethod
    def _cleanup_active_processes(cls) -> None:
        with cls._ACTIVE_PROCESS_LOCK:
            active_processes = list(cls._ACTIVE_PROCESSES.values())
        for active in active_processes:
            try:
                active.cleanup()
            except Exception:
                _LOGGER.exception("failed to clean active terminal-bench process")

    def _cleanup_tb_process(
        self,
        *,
        process: subprocess.Popen[str],
        task: dict[str, Any],
        run_id: str,
        run_root: Path,
    ) -> None:
        self._terminate_process_group(process)
        self._cleanup_task_container(task=task, run_id=run_id, run_root=run_root)

    @classmethod
    def _terminate_process_group(cls, process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=cls.PROCESS_TERMINATE_GRACE_SEC)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=cls.PROCESS_TERMINATE_GRACE_SEC)
        except subprocess.TimeoutExpired:
            pass

    @classmethod
    def _communicate_after_termination(
        cls,
        process: subprocess.Popen[str],
    ) -> tuple[str, str]:
        try:
            stdout, stderr = process.communicate(
                timeout=cls.PROCESS_TERMINATE_GRACE_SEC
            )
        except subprocess.TimeoutExpired:
            return "", ""
        return stdout or "", stderr or ""

    @staticmethod
    def _append_stderr(stderr: str, message: str) -> str:
        if stderr:
            return f"{stderr.rstrip()}\n{message}\n"
        return f"{message}\n"

    @staticmethod
    def _timeout_stream_to_text(value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value

    @staticmethod
    def _merge_streams(first: str, second: str) -> str:
        if not first:
            return second
        if not second:
            return first
        if second.startswith(first):
            return second
        return first + second

    def _cleanup_task_container(
        self,
        *,
        task: dict[str, Any],
        run_id: str,
        run_root: Path,
    ) -> None:
        task_id = str(task.get("task_id") or "")
        if not task_id:
            return
        container_name = self._expected_client_container_name(
            task_id=task_id,
            run_id=run_id,
        )
        task_source_path = task.get("task_source_path")
        if task_source_path:
            compose_path = (
                Path(str(task_source_path)).expanduser() / "docker-compose.yaml"
            )
            if compose_path.exists():
                compose_env = os.environ.copy()
                compose_env.update(
                    self._terminal_bench_compose_env(
                        task_id=task_id,
                        run_id=run_id,
                        run_root=run_root,
                    )
                )
                try:
                    result = subprocess.run(
                        [
                            "docker",
                            "compose",
                            "-p",
                            container_name,
                            "-f",
                            str(compose_path.resolve()),
                            "down",
                            "--remove-orphans",
                        ],
                        capture_output=True,
                        text=True,
                        env=compose_env,
                        timeout=120,
                        check=False,
                    )
                    if result.returncode != 0:
                        _LOGGER.warning(
                            "docker compose cleanup failed for %s: %s",
                            container_name,
                            (result.stderr or result.stdout).strip(),
                        )
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    _LOGGER.warning(
                        "docker compose cleanup could not complete for %s",
                        container_name,
                        exc_info=True,
                    )
        try:
            result = subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if result.returncode != 0:
                _LOGGER.warning(
                    "docker rm cleanup failed for %s: %s",
                    container_name,
                    (result.stderr or result.stdout).strip(),
                )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            _LOGGER.warning(
                "docker rm cleanup could not complete for %s",
                container_name,
                exc_info=True,
            )

    @classmethod
    def _task_docker_name_prefix(cls, task_id: str) -> str:
        return f"tb__{task_id}".replace(".", "-")

    @classmethod
    def _terminal_bench_trial_path(
        cls,
        *,
        task_id: str,
        run_id: str,
        run_root: Path,
    ) -> Path:
        return (
            run_root
            / run_id
            / task_id
            / cls._trial_name(
                task_id=task_id,
                run_id=run_id,
            )
        )

    @classmethod
    def _terminal_bench_compose_env(
        cls,
        *,
        task_id: str,
        run_id: str,
        run_root: Path,
    ) -> dict[str, str]:
        container_name = cls._expected_client_container_name(
            task_id=task_id,
            run_id=run_id,
        )
        image_name_prefix = cls._task_docker_name_prefix(task_id)
        trial_path = cls._terminal_bench_trial_path(
            task_id=task_id,
            run_id=run_id,
            run_root=run_root,
        )
        return {
            "T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME": container_name,
            "T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME": f"{image_name_prefix}__client",
            "T_BENCH_TASK_DOCKER_NAME_PREFIX": image_name_prefix,
            "T_BENCH_CONTAINER_LOGS_PATH": cls.CONTAINER_SESSION_LOGS_PATH,
            "T_BENCH_CONTAINER_AGENT_LOGS_PATH": cls.CONTAINER_AGENT_LOGS_PATH,
            "T_BENCH_TEST_DIR": cls.CONTAINER_TEST_DIR,
            "T_BENCH_TASK_LOGS_PATH": str((trial_path / "sessions").resolve()),
            "T_BENCH_TASK_AGENT_LOGS_PATH": str((trial_path / "agent-logs").resolve()),
        }

    def _extract_success(self, tb_run_path: Path) -> bool:
        results_path = tb_run_path / "results.json"
        if not results_path.exists():
            raise RuntimeError(f"terminal-bench results.json missing at {results_path}")
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        results = payload.get("results") or []
        if not results:
            return False
        first = results[0]
        return bool(first.get("is_resolved"))

    def _find_trace_path(self, tb_run_path: Path) -> Path:
        traces = sorted(tb_run_path.glob(f"**/agent-logs/{self._trace_filename()}"))
        if traces:
            return traces[0]
        return tb_run_path / "missing-trace.jsonl"

    def _augment_trace_metadata(
        self,
        *,
        src: Path,
        dst: Path,
        task: dict[str, Any],
        prompt_template: str,
        tb_version: str,
    ) -> None:
        lines = src.read_text(encoding="utf-8").splitlines()
        source_metadata: dict[str, Any] | None = None
        body_start = 0
        for idx, line in enumerate(lines):
            if not line.strip():
                body_start = idx + 1
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                body_start = idx + 1
                continue
            if record.get("type") == "trace_metadata":
                source_metadata = record
                body_start = idx + 1
            else:
                body_start = idx
            break

        merged = self._trace_metadata(
            source_metadata=source_metadata,
            task=task,
            prompt_template=prompt_template,
            tb_version=tb_version,
        )

        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(merged, ensure_ascii=False) + "\n")
            for line in lines[body_start:]:
                if not line.strip():
                    continue
                handle.write(line + "\n")

    def _trace_metadata(
        self,
        *,
        source_metadata: dict[str, Any] | None,
        task: dict[str, Any],
        prompt_template: str,
        tb_version: str,
    ) -> dict[str, Any]:
        merged: dict[str, Any] = dict(source_metadata or {})
        merged.update(
            {
                "type": "trace_metadata",
                "trace_format_version": 5,
                "mode": "collect",
                "scaffold": "openclaw",
                "execution_environment": "container",
                "benchmark": self.benchmark_slug,
                "model": self.model,
                "max_iterations": self.max_iterations,
                "instance_id": task["instance_id"],
                "agent_runtime_mode": "host_controller",
                "prompt_template": prompt_template,
                "task_source_kind": task.get("task_source_kind"),
                "task_source_id": task.get("task_source_id"),
                "task_source_path": task.get("task_source_path"),
                "tb_version": tb_version,
                "tb_dataset": task.get("tb_dataset"),
                "tb_registry_source": task.get("tb_registry_source"),
                "adapter_kind": "terminal_bench_openclaw",
                "agent_import_path": self.AGENT_IMPORT_PATH,
            }
        )
        run_config = dict(merged.get("run_config") or {})
        if self.mcp_config_label is not None:
            run_config["mcp_config"] = self.mcp_config_label
        if self.global_agent_timeout_sec is not None:
            run_config["global_agent_timeout_sec"] = self.global_agent_timeout_sec
        if self.llm_timeout_sec is not None:
            run_config["llm_timeout_sec"] = self.llm_timeout_sec
        run_config["tb_process_cleanup_grace_sec"] = self.tb_process_cleanup_grace_sec
        if self.generation_config:
            run_config["generation"] = dict(self.generation_config)
        if run_config:
            merged["run_config"] = run_config
        return merged

    def _summary(
        self,
        *,
        tb_version: str,
        task: dict[str, Any],
        tb_run_path: Path,
        tb_process_logs: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "tb_version": tb_version,
            "tb_dataset": task.get("tb_dataset"),
            "tb_registry_source": task.get("tb_registry_source"),
            "adapter_kind": "terminal_bench_openclaw",
            "agent_import_path": self.AGENT_IMPORT_PATH,
            "tb_run_path": str(tb_run_path),
        }
        if self.global_agent_timeout_sec is not None:
            summary["global_agent_timeout_sec"] = self.global_agent_timeout_sec
        if self.llm_timeout_sec is not None:
            summary["llm_timeout_sec"] = self.llm_timeout_sec
        summary["tb_process_cleanup_grace_sec"] = self.tb_process_cleanup_grace_sec
        if self.mcp_config_label is not None:
            summary["mcp_config"] = self.mcp_config_label
        if self.generation_config:
            summary["generation"] = dict(self.generation_config)
        if tb_process_logs:
            summary.update(tb_process_logs)
        return summary

    def _trace_filename(self) -> str:
        return self.TRACE_FILENAME

    def _materialize_prompt_template(
        self,
        *,
        prompt_template: str,
        run_root: Path,
    ) -> Path:
        from trace_collect.prompt_loader import load_prompt_template

        run_root.mkdir(parents=True, exist_ok=True)
        template_text = load_prompt_template(
            prompt_template, self.benchmark_slug
        ).replace(
            "{{task}}",
            "{{ instruction }}",
        )
        safe_name = prompt_template.replace("/", "_")
        template_path = run_root / f"prompt-template-{safe_name}.j2"
        template_path.write_text(template_text, encoding="utf-8")
        return template_path.resolve()

    @staticmethod
    def _resolve_mcp_config(mcp_config: str | None) -> str | None:
        if mcp_config in {None, "none"}:
            return None
        return str(Path(mcp_config).expanduser().resolve())

    @staticmethod
    def _mcp_config_label(mcp_config: str | None) -> str | None:
        if mcp_config is None:
            return None
        if mcp_config == "none":
            return "none"
        return Path(mcp_config).name
