"""Orchestrator for one attempt_<N> collection run."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from harness.container_image_prep import (
    ensure_fixed_image,
    ensure_arm_fixed_image,
    is_arm_host,
    use_arm_qemu,
    _inspect_image_platform,
)
from harness.container_runtime import container_run_user_args
from harness.container_stats_sampler import (
    ContainerStatsSampler,
    summarize_samples,
)
from harness.process_stats_sampler import ProcessStatsSampler
from harness.ksys import KsysSession
from harness.disk_preflight import DiskSpaceError, preflight_disk
from trace_collect import attempt_layout
from trace_collect.exec_classifier import rewrite_trace_with_exec_classification

logger = logging.getLogger(__name__)

_NONCOMPLETED_EXIT_STATUSES = frozenset(
    {
        "error",
        "tool_error",
        "empty_final_response",
        "timeout",
        "failed",
    }
)

# max_iterations is handled separately: it produces status "exhausted"
# (not "error") so that resume via --run-id skips it — re-running a task
# that already consumed its full iteration budget is wasteful.

_TASK_CONTAINER_ENV_PASSTHROUGH = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "PIP_INDEX_URL",
    "TASK_CONTAINER_PIP_INDEX_URL",
    "NANOBOT_MAX_CONCURRENT_REQUESTS",
)

# Set TASK_CONTAINER_NO_PROXY=1 to skip forwarding host proxy settings
# into the container.  Useful when the host proxy (e.g. a SOCKS5 proxy
# that breaks HTTPS) should not affect in-container pip operations.
if os.environ.get("TASK_CONTAINER_NO_PROXY") == "1":
    _TASK_CONTAINER_ENV_PASSTHROUGH = tuple(
        v for v in _TASK_CONTAINER_ENV_PASSTHROUGH
        if not v.endswith("PROXY") and not v.endswith("proxy")
    )


def _utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "")


def _stamp_trace_monitoring(
    trace_path: Path,
    monitoring_policy: dict[str, object],
) -> None:
    """Record resolved monitoring policy in canonical trace metadata."""
    if not monitoring_policy:
        return
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    stamped = False
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        if not stamped and record.get("type") == "trace_metadata":
            run_config = dict(record.get("run_config") or {})
            run_config["monitoring"] = monitoring_policy
            record["run_config"] = run_config
            stamped = True
        output.append(json.dumps(record, ensure_ascii=False))
    trace_path.write_text("\n".join(output) + "\n", encoding="utf-8")


@dataclass
class AttemptContext:
    """Shared per-task state between ``run_attempt`` and scaffold code."""

    run_dir: Path
    instance_id: str
    attempt: int
    task: dict[str, Any]
    model: str
    scaffold: str
    source_image: str | None
    prompt_template: str = "default"
    agent_runtime_mode: str = "host_controller"
    execution_environment: str = "container"
    fixed_image: str | None = None
    container_id: str | None = None
    samples: list[dict[str, Any]] | None = None
    disable_resource_monitoring: bool = True
    enable_pmu_monitoring: bool = False
    enable_memory_bandwidth_monitoring: bool = False
    monitoring_policy: dict[str, object] = field(default_factory=dict)
    attempt_dir: Path = field(init=False)
    start_time: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    end_time: datetime | None = None
    container_stdout: str = ""
    permission_fix_time_s: float = 0.0
    # Timing checkpoints for wall-clock breakdown.
    image_ready_time: datetime | None = None
    agent_start_time: datetime | None = None
    agent_end_time: datetime | None = None
    # ARM-native fields (ignored on x86_64 hosts).
    arm_repo: str | None = None
    arm_base_commit: str | None = None
    arm_install_config: list[str] | None = None
    arm_repos_root: Path | None = None

    def __post_init__(self) -> None:
        self.attempt_dir = self.run_dir / self.instance_id / f"attempt_{self.attempt}"

    def mark_container_ready(self, container_id: str) -> None:
        """Publish the task container id so sampling can start."""
        self.container_id = container_id

    @property
    def attempt_label(self) -> str:
        """Stable string like ``attempt_1`` (matches the manifest field)."""
        return f"attempt_{self.attempt}"

    def _delta_s(self, start: datetime | None, end: datetime | None) -> float:
        """Seconds between two optional datetimes; returns 0.0 if either is None."""
        if start is None or end is None:
            return 0.0
        return (end - start).total_seconds()

    def elapsed_seconds(self) -> float:
        """Total wall clock between ``start_time`` and ``end_time`` (or now)."""
        end = self.end_time or datetime.now(tz=timezone.utc)
        return (end - self.start_time).total_seconds()

    def setup_seconds(self) -> float:
        """Wall clock from ``start_time`` to agent start (excl. agent execution)."""
        ref = self.agent_start_time or self.image_ready_time
        return self._delta_s(self.start_time, ref)

    def agent_seconds(self) -> float:
        """Wall clock from ``agent_start_time`` to ``agent_end_time``."""
        return self._delta_s(self.agent_start_time, self.agent_end_time)

    def teardown_seconds(self) -> float:
        """Wall clock from ``agent_end_time`` to ``end_time``."""
        return self._delta_s(self.agent_end_time, self.end_time)

    def start_time_iso(self) -> str:
        return self.start_time.isoformat().replace("+00:00", "")

    def end_time_iso(self) -> str:
        end = self.end_time or datetime.now(tz=timezone.utc)
        return end.isoformat().replace("+00:00", "")


def _resolve_arm_repo_path(
    repo: str,
    *,
    repos_root: Path | None = None,
) -> Path:
    """Resolve the local bare-repo mirror path for *repo*.

    Returns the directory path to use as a Docker volume mount.
    """
    if repos_root is None:
        raise ValueError(
            "ARM host requires repos_root. "
            "Add repos_root to benchmark YAML or run: make setup-swe-rebench-repos"
        )
    bare_name = f"{repo.replace('/', '__')}.git"
    bare_path = repos_root / bare_name
    if not bare_path.exists():
        raise FileNotFoundError(
            f"Repo {repo!r} not found at {bare_path}. "
            f"Run: make setup-swe-rebench-repos"
        )
    return bare_path


def start_task_container(
    fixed_image: str,
    *,
    executable: str,
    extra_args: list[str] | None = None,
    network_mode: str = "host",
) -> str:
    """Launch the task container and return its id."""
    import os
    import subprocess

    home_dir = os.environ.get("HOME", "/root")
    # Always use a container-only PATH to guarantee identical tool resolution
    # regardless of host architecture.  Including host ~/.local/bin leaks
    # host-installed tools (python, pip, etc.) into the container on native
    # arches, producing different behaviour than ARM+QEMU where those host
    # binaries cannot execute.  Task-level --user installs go to an ephemeral
    # in-container userbase, not the shared OpenClaw bootstrap cache.
    task_userbase = "/tmp/openclaw-task-userbase"
    container_path = f"{task_userbase}/bin:/usr/local/bin:/usr/bin:/bin"
    qemu_env = [
        "-e", f"OPENCLAW_TASK_USERBASE={task_userbase}",
        "-e", "PIP_BREAK_SYSTEM_PACKAGES=1",
    ]
    cmd = [
        executable,
        "run",
        "-d",
        "--rm",
        f"--network={network_mode}",
        "-v",
        f"{home_dir}:{home_dir}",
        "-w",
        "/testbed",
        "-e",
        f"HOME={home_dir}",
        "-e",
        f"PATH={container_path}",
    ]
    cmd.extend(qemu_env)
    for env_name in _TASK_CONTAINER_ENV_PASSTHROUGH:
        value = os.environ.get(env_name)
        if value:
            cmd.extend(["-e", f"{env_name}={value}"])
    cmd.extend(container_run_user_args(executable))
    if extra_args:
        cmd.extend(extra_args)
    # Explicitly pin the container platform to the image's native arch.
    # On ARM hosts running x86_64 images this ensures Docker selects the
    # correct QEMU binfmt handler rather than relying on auto-detection.
    image_platform = _inspect_image_platform(fixed_image, executable)
    if image_platform:
        cmd.extend(["--platform", image_platform])
    cmd.extend([fixed_image, "sleep", "infinity"])
    logger.info("starting task container: image=%s platform=%s ...", fixed_image, image_platform or "auto")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to start task container for {fixed_image}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def stop_task_container(container_id: str, *, executable: str) -> str:
    """Capture container logs then stop and remove it. Returns log text."""
    import subprocess

    logs_text = ""
    try:
        logs = subprocess.run(
            [executable, "logs", container_id],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        logs_text = (logs.stdout or "") + (logs.stderr or "")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    try:
        subprocess.run(
            [executable, "stop", container_id],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    try:
        subprocess.run(
            [executable, "rm", "-f", container_id],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return logs_text


@dataclass
class AttemptResult:
    """Result returned by the scaffold ``inner`` coroutine."""

    success: bool
    exit_status: str | None
    trace_path: Path
    model_patch: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    n_iterations: int | None = None
    total_llm_ms: float | None = None
    total_tool_ms: float | None = None
    total_tokens: int | None = None
    runtime_proof: dict[str, Any] = field(default_factory=dict)


async def _watch_for_container_ready(
    ctx: AttemptContext,
    stop_event: threading.Event,
    *,
    container_executable: str,
    enable_pmu: bool,
    enable_memory_bandwidth: bool,
) -> ContainerStatsSampler | None:
    """Wait for ``ctx.container_id`` and start sampling once it appears."""
    while not stop_event.is_set():
        if ctx.container_id:
            if _container_is_inspectable(
                ctx.container_id,
                container_executable=container_executable,
            ):
                sampler = ContainerStatsSampler(
                    container_id=ctx.container_id,
                    interval_s=0.5,
                    executable=container_executable,
                    enable_pmu=enable_pmu,
                    enable_memory_bandwidth=enable_memory_bandwidth,
                )
                sampler.start()
                return sampler
        try:
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            return None
    return None


def _container_is_inspectable(
    container_id: str,
    *,
    container_executable: str,
) -> bool:
    try:
        result = subprocess.run(
            [
                container_executable,
                "inspect",
                "--format",
                "{{.Id}}",
                container_id,
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())




async def _wait_for_recording_provider_idle(
    recording_provider: Any,
    *,
    phase: str,
) -> None:
    wait_until_idle = getattr(recording_provider, "wait_until_idle", None)
    if not callable(wait_until_idle):
        return
    logger.info("waiting for recording provider idle before %s", phase)
    await asyncio.to_thread(wait_until_idle)


async def run_attempt(
    ctx: AttemptContext,
    *,
    inner: Callable[[AttemptContext], Awaitable[AttemptResult]],
    min_free_disk_gb: float = 30.0,
    container_executable: str | None,
    recording_provider: Any | None = None,
    enable_ksys: bool = False,
    disable_resource_monitoring: bool = False,
    enable_pmu_monitoring: bool = True,
    enable_memory_bandwidth_monitoring: bool = True,
    monitoring_policy: dict[str, object] | None = None,
) -> AttemptResult:
    """Execute one scaffold attempt and write its artifacts.

    When *disable_resource_monitoring* is ``True``, built-in resource
    samplers (ContainerStatsSampler, ProcessStatsSampler) are skipped.
    This is used by the concurrent-collection path where ksys provides
    system-level monitoring and per-container sampling would conflict.
    """
    # Propagate to ctx so _run_openclaw_in_task_container (and other
    # inner functions) can also skip their own sampler initialisation.
    ctx.disable_resource_monitoring = disable_resource_monitoring
    ctx.enable_pmu_monitoring = enable_pmu_monitoring
    ctx.enable_memory_bandwidth_monitoring = enable_memory_bandwidth_monitoring
    ctx.monitoring_policy = dict(monitoring_policy or {})

    if recording_provider is not None:
        await _wait_for_recording_provider_idle(
            recording_provider,
            phase="start_attempt",
        )
        ctx.start_time = datetime.now(tz=timezone.utc)

    try:
        free_gb = preflight_disk(ctx.run_dir, min_free_disk_gb)
        logger.info("disk preflight ok: %.2f GB free at %s", free_gb, ctx.run_dir)
        if recording_provider is not None:
            recordings_dir = ctx.attempt_dir / "recordings"
            recording_free_gb = preflight_disk(recordings_dir, min_free_disk_gb)
            logger.info(
                "recording disk preflight ok: %.2f GB free at %s",
                recording_free_gb,
                recordings_dir,
            )
    except DiskSpaceError as exc:
        logger.error("disk preflight failed: %s", exc)
        raise

    attempt_layout.ensure_attempt_dir(ctx.attempt_dir)

    if ctx.source_image:
        if container_executable is None:
            raise ValueError("container_executable is required for container tasks")
        try:
            if (
                is_arm_host()
                and not use_arm_qemu()
                and ctx.arm_repo
                and ctx.arm_base_commit
            ):
                # ARM-native: build fixed image from base + cloned repo.
                if ctx.arm_repos_root is None:
                    raise ValueError(
                        f"ARM host but arm_repos_root not set for {ctx.instance_id}"
                    )
                repo_path = _resolve_arm_repo_path(
                    ctx.arm_repo,
                    repos_root=ctx.arm_repos_root,
                )
                fixed_name, fix_elapsed = ensure_arm_fixed_image(
                    ctx.instance_id,
                    base_image=ctx.source_image,
                    repo_path=repo_path,
                    base_commit=ctx.arm_base_commit,
                    install_config=ctx.arm_install_config,
                    container_executable=container_executable,
                )
            else:
                fixed_name, fix_elapsed = ensure_fixed_image(
                    ctx.source_image,
                    container_executable=container_executable,
                )
            ctx.fixed_image = fixed_name
            ctx.permission_fix_time_s = fix_elapsed
            ctx.image_ready_time = datetime.now(tz=timezone.utc)
            logger.info(
                "image prep: source=%s fixed=%s elapsed=%.2fs",
                ctx.source_image,
                fixed_name,
                fix_elapsed,
            )
        except Exception as exc:
            logger.error("image prep failed: %s", exc)
            ctx.fixed_image = ctx.source_image
    else:
        ctx.fixed_image = None
        ctx.permission_fix_time_s = 0.0
        ctx.image_ready_time = datetime.now(tz=timezone.utc)

    stop_watcher = threading.Event()
    watcher_task: asyncio.Task[ContainerStatsSampler | None] | None = None
    if container_executable is not None and not disable_resource_monitoring:
        watcher_task = asyncio.create_task(
            _watch_for_container_ready(
                ctx,
                stop_watcher,
                container_executable=container_executable,
                enable_pmu=enable_pmu_monitoring,
                enable_memory_bandwidth=enable_memory_bandwidth_monitoring,
            )
        )

    # ksys system metrics collector (Huawei Kunpeng).  Started alongside
    # other samplers so its timeline aligns with the Gantt chart's t0.
    # Gracefully no-ops when ksys is not installed on the host.
    # Controlled by --ksys-monitoring flag (default: off).
    ksys_session: KsysSession | None = None
    ksys_pid: int | None = None
    if enable_ksys:
        ksys_out_dir = ctx.attempt_dir.parent.resolve()
        ksys_session = KsysSession.start(
            output_dir=ksys_out_dir,
            log_dir=ctx.attempt_dir,
        )
        if ksys_session is not None:
            ksys_pid = ksys_session.process.pid

    process_sampler: ProcessStatsSampler | None = None
    if ctx.execution_environment == "host" and not disable_resource_monitoring:
        process_sampler = ProcessStatsSampler(
            pid=os.getpid(),
            interval_s=0.5,
            # Exclude the ksys process tree from host resource
            # accounting so system-level profiling does not
            # pollute the agent's own metrics.  Agent-spawned
            # children (tool calls, etc.) remain included.
            exclude_pids={ksys_pid} if ksys_pid is not None else None,
            enable_pmu=enable_pmu_monitoring,
            enable_memory_bandwidth=enable_memory_bandwidth_monitoring,
        )
        process_sampler.start()
    if recording_provider is not None:
        recording_provider.start_attempt(ctx.attempt_dir / "recordings")

    sampler: ContainerStatsSampler | ProcessStatsSampler | None = None
    samples: list[dict[str, Any]] = []
    result: AttemptResult | None = None
    inner_error: BaseException | None = None

    try:
        ctx.agent_start_time = datetime.now(tz=timezone.utc)
        result = await inner(ctx)
        ctx.agent_end_time = datetime.now(tz=timezone.utc)
    except BaseException as exc:
        ctx.agent_end_time = datetime.now(tz=timezone.utc)
        inner_error = exc
        logger.exception("scaffold inner raised: %s", exc)
    finally:
        if recording_provider is not None:
            await _wait_for_recording_provider_idle(
                recording_provider,
                phase="attempt finalization",
            )
        stop_watcher.set()
        if watcher_task is not None:
            try:
                sampler = await asyncio.wait_for(watcher_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                watcher_task.cancel()
                sampler = None
            except Exception:
                sampler = None
        watcher_samples: list[dict[str, Any]] = []
        if sampler is not None:
            watcher_samples = sampler.stop()
        if ctx.samples is not None:
            # ctx.samples may have been started earlier (e.g. by
            # _run_openclaw_in_task_container before the blocking
            # bootstrap phase) and thus captures a superset of the
            # watcher sampler's timeline.  Prefer the dataset with
            # more samples — the earlier-started sampler covers
            # bootstrap overhead that the watcher would miss.
            if len(ctx.samples) >= len(watcher_samples):
                samples = ctx.samples
            else:
                samples = watcher_samples
        else:
            samples = watcher_samples
        if process_sampler is not None:
            process_samples = process_sampler.stop()
            if not samples:
                samples = process_samples
        if ksys_session is not None:
            ksys_session.stop()
        ctx.end_time = datetime.now(tz=timezone.utc)

    status = "completed"
    if inner_error is not None:
        status = "error"
    elif result is not None and result.exit_status == "max_iterations":
        # Agent exhausted its iteration budget — a legitimate terminal
        # state, not a transient error.  Mark it "exhausted" so that
        # resume via --run-id skips it (re-running would waste compute).
        status = "exhausted"
    elif result is not None and (
        not result.success
        or (
            result.exit_status is not None
            and result.exit_status in _NONCOMPLETED_EXIT_STATUSES
        )
    ):
        status = "error"
    success = bool(result.success) if result is not None else False
    task_payload: dict[str, Any] = {
        "instance_id": ctx.instance_id,
        "repo": ctx.task.get("repo"),
        "docker_image": ctx.source_image,
    }
    for key in ("task_source_kind", "task_source_id", "task_source_path"):
        if key in ctx.task:
            task_payload[key] = ctx.task.get(key)

    manifest = {
        "status": status,
        "task": task_payload,
        "attempt": ctx.attempt_label,
        "model": {"name": ctx.model},
        "runtime": {
            "home": None,
            "wrapper_enabled": False,
            "memory_limit": None,
            "cpu_limit": None,
            "start_time": ctx.start_time_iso(),
            "end_time": ctx.end_time_iso(),
            "min_free_disk_gb": min_free_disk_gb,
            "agent_runtime_mode": ctx.agent_runtime_mode,
            "runtime_proof": result.runtime_proof if result is not None else {},
            "monitoring": ctx.monitoring_policy,
        },
        "replay": {
            "replay_ready": bool(ctx.fixed_image),
            "source_image": ctx.source_image,
            "fixed_image_name": ctx.fixed_image,
        },
        "result_summary": {
            "exit_code": 0 if success else 1,
            "error": str(inner_error)
            if inner_error is not None
            else (result.error if result is not None else None),
            "total_time": ctx.elapsed_seconds(),
            "active_time": (result.total_llm_ms or 0.0) / 1000.0 if result else 0.0,
            "tool_time": (result.total_tool_ms or 0.0) / 1000.0 if result else 0.0,
        },
        "scaffold": ctx.scaffold,
        "prompt_template": ctx.prompt_template,
        "agent_runtime_mode": ctx.agent_runtime_mode,
        "execution_environment": ctx.execution_environment,
        "timing": {
            "wall_total_s": ctx.elapsed_seconds(),
            "setup_s": ctx.setup_seconds(),
            "agent_exec_s": ctx.agent_seconds(),
            "teardown_s": ctx.teardown_seconds(),
            "permission_fix_s": ctx.permission_fix_time_s,
        },
    }

    results_payload: dict[str, Any] = {
        "image": ctx.source_image,
        "start_time": ctx.start_time_iso(),
        "end_time": ctx.end_time_iso(),
        "memory_limit": None,
        "cpu_limit": None,
        "model": ctx.model,
        "output_dir": str(ctx.attempt_dir),
        "permission_fix_time": ctx.permission_fix_time_s,
        "total_time": ctx.elapsed_seconds(),
        "active_time": manifest["result_summary"]["active_time"],
        "tool_time": manifest["result_summary"]["tool_time"],
        "replay_ready": bool(ctx.fixed_image),
        "instance_id": ctx.instance_id,
        "repo": ctx.task.get("repo"),
        "docker_image": ctx.source_image,
        "success": success,
        "scaffold": ctx.scaffold,
        "prompt_template": ctx.prompt_template,
        "agent_runtime_mode": ctx.agent_runtime_mode,
        "timing": {
            "wall_total_s": ctx.elapsed_seconds(),
            "setup_s": ctx.setup_seconds(),
            "agent_exec_s": ctx.agent_seconds(),
            "teardown_s": ctx.teardown_seconds(),
            "permission_fix_s": ctx.permission_fix_time_s,
        },
    }
    for key in (
        "task_source_kind",
        "task_source_id",
        "task_source_path",
        "tb_version",
        "tb_dataset",
        "tb_registry_source",
        "adapter_kind",
        "agent_import_path",
    ):
        if key in ctx.task:
            results_payload[key] = ctx.task.get(key)
        elif result is not None and key in result.summary:
            results_payload[key] = result.summary.get(key)
    if result is not None and result.runtime_proof:
        results_payload["runtime_proof"] = result.runtime_proof
    if result is not None:
        results_payload["n_iterations"] = result.n_iterations
        results_payload["total_tokens"] = result.total_tokens
        if result.summary:
            results_payload["scaffold_summary"] = result.summary

    resources_summary = summarize_samples(samples)

    copied_trace_path: Path | None = None
    try:
        if result is not None and result.trace_path.exists():
            copied_trace_path = attempt_layout.copy_trace_jsonl(
                ctx.attempt_dir,
                result.trace_path,
            )
    finally:
        if recording_provider is not None:
            trace_path = copied_trace_path
            if trace_path is None and result is not None and result.trace_path.exists():
                trace_path = result.trace_path
            recording_provider.finish_attempt(
                trace_path=trace_path if trace_path and trace_path.exists() else None
            )

    # Classify exec tool names (exec → exec-grep, exec-pip, etc.)
    # after the scaffold has finished writing so downstream consumers
    # (visualisers, tool_calls.json, simulators) see classified names.
    if copied_trace_path is not None and copied_trace_path.exists():
        _stamp_trace_monitoring(copied_trace_path, ctx.monitoring_policy)
        n_changed = rewrite_trace_with_exec_classification(copied_trace_path)
        if n_changed:
            logger.debug(
                "Classified %d exec tool calls in %s",
                n_changed, copied_trace_path,
            )

    trace_file = ctx.attempt_dir / attempt_layout.TRACE_FILENAME
    if result is not None and result.tool_calls:
        tool_calls = result.tool_calls
    elif trace_file.exists():
        tool_calls = attempt_layout.build_tool_calls_from_trace(trace_file)
    else:
        tool_calls = []

    attempt_layout.write_run_manifest(ctx.attempt_dir, manifest)
    attempt_layout.write_results_json(ctx.attempt_dir, results_payload)
    if samples:
        resources_summary["monitoring"] = {
            **ctx.monitoring_policy,
            "status": "collected",
        }
        attempt_layout.write_resources_json(
            ctx.attempt_dir, samples, summary=resources_summary
        )
    else:
        # No resource samples collected (e.g. concurrent mode with
        # disable_resource_monitoring=True).  Write a minimal marker
        # so downstream consumers can distinguish "no data" from
        # "zero utilisation".
        monitoring_status = (
            "disabled"
            if disable_resource_monitoring
            else "enabled_no_samples"
        )
        attempt_layout.write_resources_json(
            ctx.attempt_dir,
            [],
            summary={
                **resources_summary,
                "monitoring_disabled": disable_resource_monitoring,
                "monitoring": {
                    **ctx.monitoring_policy,
                    "status": monitoring_status,
                },
            },
        )
    attempt_layout.write_tool_calls_json(ctx.attempt_dir, tool_calls)
    attempt_layout.write_container_stdout(ctx.attempt_dir, ctx.container_stdout)

    if inner_error is not None:
        raise inner_error
    assert result is not None
    return result


__all__ = ["AttemptContext", "AttemptResult", "run_attempt"]
