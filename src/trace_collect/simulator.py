from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import math
import multiprocessing
import os
import random
import re
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import BrokenBarrierError
from typing import Any

from agents.base import TraceAction
from harness.container_image_prep import ensure_fixed_image, normalize_image_reference
from harness.container_stats_sampler import ContainerStatsSampler, summarize_samples
from harness.gpu_resource_sampler import GpuResourceSampler
from harness.ksys import KsysSession
from harness.runner import build_arrival_offsets
from harness.metrics_client import VLLMMetricsClient
from harness.scheduler_hooks import GpuBaseline
from harness.trace_logger import TraceLogger
from llm_call import create_async_openai_client
from trace_collect import attempt_layout
from trace_collect.attempt_pipeline import start_task_container, stop_task_container
from trace_collect.exec_classifier import classify_exec_tool_name
from trace_collect.html_viz import generate_html
from trace_collect.monitoring import MonitoringPolicy, resolve_simulate_monitoring
from trace_collect.openclaw_tools import HostAgent

logger = logging.getLogger(__name__)

_DEFAULT_PREP_CONCURRENCY = 20


class SimulateError(Exception):
    """Raised when simulation encounters a fatal issue."""


def validate_gpu_tracking_args(args: Any) -> None:
    """Validate GPU tracking CLI args. Raises ValueError with a clear message on failure.

    Designed to be called from _run_simulate before any work begins,
    so failures are fast and explicit (CLAUDE.md no-silent-fallback rule).
    """
    gpu_tracking = getattr(args, "gpu_tracking", "off")
    if gpu_tracking != "on":
        return

    mode = getattr(args, "mode", "local_model")
    if mode == "cloud_model":
        raise ValueError("--gpu-tracking on is forbidden in cloud_model mode")

    metrics_url = getattr(args, "metrics_url", None)
    if not metrics_url:
        raise ValueError("--gpu-tracking on requires --metrics-url")

    vllm_pid = getattr(args, "vllm_pid", None)
    if vllm_pid is None:
        raise ValueError("--gpu-tracking on requires --vllm-pid")

    vllm_startup_log = getattr(args, "vllm_startup_log", None)
    if vllm_startup_log is None:
        raise ValueError("--gpu-tracking on requires --vllm-startup-log")


def _ensure_fd_headroom(num_sessions: int, *, concurrent: bool = True) -> None:
    """Raise the process file-descriptor limit if it is too low for *num_sessions*.

    Each concurrent ContainerAgent holds 3 pipes (stdin/stdout/stderr) for
    its persistent ``docker exec`` subprocess.  With N sessions we need at
    least N×5 fds for pipes + Python runtime + Docker API + trace files.
    The default ``ulimit -n 1024`` is exhausted around 200 concurrent
    containers.

    Only acts when *concurrent* is True (serial mode reuses fds).
    """
    if not concurrent or num_sessions <= 1:
        return

    import resource as _resource

    min_needed = num_sessions * 5
    soft, hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
    if soft >= min_needed:
        return  # already sufficient

    target = max(min_needed, 65536)
    if hard >= 0 and target > hard:
        target = hard  # respect hard limit

    try:
        _resource.setrlimit(_resource.RLIMIT_NOFILE, (target, hard))
        logger.warning(
            "Raised fd limit: %d → %d (%d sessions × 5 fds/session)",
            soft, target, num_sessions,
        )
    except (ValueError, OSError) as exc:
        logger.error(
            "Cannot raise fd limit (%d → %d): %s.  "
            "Run 'ulimit -n %d' before starting, or reduce --num-agents.",
            soft, target, exc, target,
        )
        raise SimulateError(
            f"File descriptor limit too low: {soft} < {min_needed} needed "
            f"for {num_sessions} sessions.  "
            f"Run 'ulimit -n {target}' before starting."
        ) from exc


@dataclass(slots=True)
class LoadedTraceSession:
    """Resolved replay inputs for one source trace."""

    source_trace: Path
    task_source: Path
    agent_id: str
    scaffold: str
    metadata: dict[str, Any] | None
    summary: dict[str, Any] | None
    task: dict[str, Any]
    actions: list[dict[str, Any]]
    iterations: dict[int, dict[str, Any]]
    docker_image_override: str | None = None


@dataclass(slots=True)
class PreparedContainer:
    """Container prepared for trace replay."""

    container_id: str
    container_executable: str
    docker_image: str
    agent: Any  # ContainerAgent


@dataclass(slots=True)
class PreparedTraceSession:
    """Container plus the loaded source-trace context."""

    loaded: LoadedTraceSession
    container: PreparedContainer | None = None
    host_agent: HostAgent | None = None
    sampler: ContainerStatsSampler | None = None
    task_output_dir: Path | None = None
    monitoring_policy: MonitoringPolicy | None = None
    _resources_written: bool = False
    # Timing: seconds spent preparing the container/image before replay.
    container_setup_s: float = 0.0
    # Per-tool profiling output directory (vtune/ksys results land here).
    vtune_out_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class WorkerTraceInput:
    '''Picklable worker input that preserves the globally assigned identity.'''

    source_trace: str
    task_source: str
    docker_image_override: str | None
    agent_id: str


def _resolve_prep_concurrency(requested: int, num_sessions: int) -> int:
    '''Resolve the system-wide concurrent container preparation limit.'''
    if requested < 0:
        raise ValueError('prep_concurrency must be >= 0')
    if num_sessions < 1:
        raise ValueError('num_sessions must be >= 1')
    limit = requested or _DEFAULT_PREP_CONCURRENCY
    return min(limit, num_sessions)


def _worker_trace_input(session: LoadedTraceSession) -> WorkerTraceInput:
    '''Serialize a loaded session without discarding its assigned agent ID.'''
    return WorkerTraceInput(
        source_trace=str(session.source_trace),
        task_source=str(session.task_source),
        docker_image_override=session.docker_image_override,
        agent_id=session.agent_id,
    )


def _partition_sessions_and_offsets(
    sessions: list[LoadedTraceSession],
    offsets: list[float],
    workers: int,
) -> list[tuple[list[LoadedTraceSession], list[float]]]:
    """Partition sessions and their global arrival offsets without reordering."""
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if len(sessions) != len(offsets):
        raise ValueError("sessions and offsets must have the same length")
    if not sessions:
        raise ValueError("sessions must not be empty")

    partition_count = min(workers, len(sessions))
    chunk_size, remainder = divmod(len(sessions), partition_count)
    partitions: list[tuple[list[LoadedTraceSession], list[float]]] = []
    start = 0
    for worker_index in range(partition_count):
        size = chunk_size + (1 if worker_index < remainder else 0)
        stop = start + size
        partitions.append((sessions[start:stop], offsets[start:stop]))
        start = stop
    return partitions


def _abort_global_replay_start(barrier: Any, start_event: Any) -> None:
    '''Best-effort release of peers waiting for a failed global start.'''
    try:
        barrier.abort()
    except Exception:
        logger.debug('Failed to abort replay-start barrier', exc_info=True)
    try:
        start_event.set()
    except Exception:
        logger.debug('Failed to set replay-start event', exc_info=True)


async def _wait_for_global_replay_start(
    barrier: Any,
    start_event: Any,
    start_wall_time: Any,
    *,
    coordinator: bool,
) -> float:
    '''Wait until every process is prepared and return one shared time zero.'''
    try:
        await asyncio.to_thread(barrier.wait)
    except BrokenBarrierError as exc:
        raise SimulateError('Global replay start was aborted') from exc

    if coordinator:
        start_wall_time.value = time.time()
        start_event.set()
    else:
        await asyncio.to_thread(start_event.wait)

    shared_wall_zero = float(start_wall_time.value)
    if shared_wall_zero <= 0:
        raise SimulateError('Global replay start has no valid shared time zero')
    return time.monotonic() + (shared_wall_zero - time.time())


async def _call_local_model_streaming(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    n_tokens: int,
) -> tuple[float, float, float]:
    """Send *messages* to the local model and force exactly *n_tokens* of output.

    Returns:
        (ttft_ms, tpot_ms, total_latency_ms)
    """
    t0 = time.monotonic()
    first_token_ts: float | None = None

    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=n_tokens,
        stream=True,
        temperature=0.0,
        extra_body={"min_tokens": n_tokens},
    )
    async for chunk in stream:
        if first_token_ts is None and chunk.choices and chunk.choices[0].delta.content:
            first_token_ts = time.monotonic()

    t_end = time.monotonic()
    total_ms = (t_end - t0) * 1000
    ttft_ms = (first_token_ts - t0) * 1000 if first_token_ts else total_ms
    gen_ms = total_ms - ttft_ms
    tpot_ms = gen_ms / max(1, n_tokens - 1) if n_tokens > 1 else 0.0
    return ttft_ms, tpot_ms, total_ms


async def _exec_tool(
    agent: Any,
    tool_name: str | None,
    tool_args_json: str,
    command_timeout_s: float,
) -> tuple[str, float, bool]:
    """Execute one source-trace tool call via the persistent container agent.

    Returns:
        (tool_result, tool_duration_ms, tool_success)
    """
    from trace_collect.openclaw_tools import execute_trace_tool

    t0 = time.monotonic()
    tool_result, tool_success, inner_duration_ms = await execute_trace_tool(
        agent=agent,
        tool_name=tool_name,
        tool_args_json=tool_args_json,
        command_timeout_s=command_timeout_s,
    )
    wall_duration_ms = (time.monotonic() - t0) * 1000
    # Prefer agent-side timing to exclude pipe transfer overhead
    duration_ms = inner_duration_ms if inner_duration_ms is not None else wall_duration_ms
    return tool_result, duration_ms, tool_success


def _group_actions_by_iteration(
    actions: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Group loaded trace actions into the local-model iteration shape."""

    iterations: dict[int, dict[str, Any]] = {}
    for action in actions:
        it = int(action.get("iteration", 0))
        if it not in iterations:
            iterations[it] = {"llms": [], "tools": []}
        if action.get("action_type") == "llm_call":
            iterations[it]["llms"].append(action)
        elif action.get("action_type") == "tool_exec":
            iterations[it]["tools"].append(action)
    return iterations


def _parse_trace_session_file(
    trace_path: Path,
) -> tuple[str, dict[str, Any] | None, list[dict[str, Any]], dict[str, Any] | None]:
    """Read one canonical trace once and extract the primary replay lane."""

    metadata: dict[str, Any] | None = None
    first_agent_id: str | None = None
    actions: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}

    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            record_type = record.get("type")
            if record_type == "trace_metadata":
                metadata = record
                continue

            agent_id = record.get("agent_id")
            if record_type == "action" and agent_id:
                if first_agent_id is None:
                    first_agent_id = agent_id
                if agent_id == first_agent_id:
                    actions.append(record)
                continue

            if record_type == "summary" and agent_id:
                summaries[agent_id] = record

    if first_agent_id is None or not actions:
        raise SimulateError(f"No action records with agent_id found in {trace_path}")

    # Safe float coercion so non-numeric timestamps (e.g. None, str) don't
    # crash the sort before _validate_loaded_sessions can produce a clear
    # error message.  Malformed actions sort to the front (0.0) and are
    # caught during validation.
    def _safe_ts(value: object) -> float:
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0

    actions.sort(
        key=lambda action: (
            _safe_ts(action.get("ts_start", 0.0)),
            _safe_ts(action.get("ts_end", 0.0)),
            int(action.get("iteration", 0)),
            str(action.get("action_id", "")),
        )
    )
    return first_agent_id, metadata, actions, summaries.get(first_agent_id)


_BENCHMARK_TASK_SOURCES: dict[str, str] = {
    "swe-rebench": "data/swe-rebench/tasks.json",
    "swe-bench-verified": "data/swebench_verified/tasks.json",
    "deep-research-bench": "data/deep-research-bench/tasks.json",
    "browsecomp": "data/browsecomp/tasks.json",
    "terminal-bench": "data/terminal-bench/tasks.json",
}


def _resolve_task_source(
    metadata: dict[str, Any] | None,
    task_source: Path,
    source_trace: Path,
) -> Path:
    """Resolve the canonical task-source path for *source_trace*.

    When *task_source* is the CLI default (or otherwise unreachable) the
    trace metadata's ``benchmark`` field is used to select the
    benchmark-appropriate tasks file.  Falls back to the original
    *task_source* when no metadata is available.
    """
    if task_source.exists():
        return task_source

    benchmark = (metadata or {}).get("benchmark", "")
    if not benchmark:
        return task_source

    mapped = _BENCHMARK_TASK_SOURCES.get(benchmark)
    if mapped:
        candidate = Path(mapped)
        if candidate.exists():
            logger.info(
                "%s: auto-detected task_source %s (benchmark=%s)",
                source_trace.name,
                candidate,
                benchmark,
            )
            return candidate
        # For host-mode benchmarks the tasks file may not exist yet;
        # the caller will synthesise a minimal task record.
        logger.warning(
            "%s: benchmark=%s but %s not found; will synthesise task from metadata",
            source_trace.name,
            benchmark,
            candidate,
        )

    return task_source


# Cache: task_source Path → {instance_id: task_dict}.  Avoids re-reading
# and re-parsing large tasks JSON files for every session (640 sessions
# × 3000+ task entries without this cache = minutes of wasted CPU).
_TASK_CACHE: dict[Path, dict[str, dict[str, Any]]] = {}


def _find_task(task_source: Path, agent_id: str) -> dict[str, Any]:
    if task_source not in _TASK_CACHE:
        tasks = json.loads(task_source.read_text(encoding="utf-8"))
        _TASK_CACHE[task_source] = {
            task["instance_id"]: task for task in tasks
        }
    task = _TASK_CACHE[task_source].get(agent_id)
    if task is None:
        raise SimulateError(f"Task {agent_id!r} not found in {task_source}")
    return task


def _find_or_synthesize_task(
    task_source: Path,
    agent_id: str,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the task record for *agent_id*, synthesising one when needed.

    Host-mode benchmarks (deep-research-bench, browsecomp) may not have a
    static tasks JSON file.  When *task_source* does not exist and the
    metadata declares a host benchmark, we build a minimal task dict from
    the trace metadata.
    """
    if task_source.exists():
        return _find_task(task_source, agent_id)

    benchmark = (metadata or {}).get("benchmark", "")
    execution_env = (metadata or {}).get("execution_environment", "")
    if benchmark and execution_env == "host":
        logger.info(
            "Synthesising task record for %s (benchmark=%s, host mode)",
            agent_id,
            benchmark,
        )
        return {
            "instance_id": agent_id,
            "repo": None,
            "image_name": None,
            "docker_image": None,
        }

    raise SimulateError(
        f"Task source {task_source} not found and cannot synthesise "
        f"task for benchmark={benchmark!r} env={execution_env!r}. "
        f"Pass --task-source explicitly."
    )


def _iteration_count(actions: list[dict[str, Any]]) -> int:
    return len({int(action.get("iteration", 0)) for action in actions})


def _sanitize_run_label(value: str) -> str:
    return value.replace("/", "-").replace(":", "-").replace(" ", "-")


_ATTEMPT_DIR_RE = re.compile(r"^attempt_(\d+)$")


def _next_attempt_number(instance_dir: Path) -> int:
    if not instance_dir.exists():
        return 1
    max_n = 0
    for child in instance_dir.iterdir():
        if not child.is_dir():
            continue
        match = _ATTEMPT_DIR_RE.fullmatch(child.name)
        if match:
            max_n = max(max_n, int(match.group(1)))
    return max_n + 1


def _arrival_tag(arrival_mode: str, arrival_rate_per_s: float | None) -> str:
    if arrival_mode == "poisson" and arrival_rate_per_s:
        return f"poisson_{arrival_rate_per_s:g}_per_s"
    return arrival_mode or "closed_loop"


def _structured_output_subdir(
    sessions: list["LoadedTraceSession"],
    *,
    arrival_mode: str,
    arrival_rate_per_s: float | None,
) -> Path:
    primary = sessions[0].metadata or {}
    benchmark = str(primary.get("benchmark") or "unknown")
    model = str(primary.get("model") or "unknown")
    scaffold = str(primary.get("scaffold") or sessions[0].scaffold or "unknown")
    for session in sessions[1:]:
        other = session.metadata or {}
        if (
            other.get("benchmark") != primary.get("benchmark")
            or other.get("model") != primary.get("model")
            or other.get("scaffold") != primary.get("scaffold")
        ):
            logger.warning(
                "Heterogeneous trace metadata in manifest — primary "
                "benchmark/model/scaffold=%s/%s/%s but %s has %s/%s/%s; "
                "using primary for output path.",
                benchmark, model, scaffold, session.agent_id,
                other.get("benchmark"), other.get("model"), other.get("scaffold"),
            )
            break
    return (
        Path(_sanitize_run_label(benchmark))
        / _sanitize_run_label(model)
        / _sanitize_run_label(scaffold)
        / _arrival_tag(arrival_mode, arrival_rate_per_s)
    )


def _build_run_id(*, mode: str, model: str | None) -> str:
    label = model if model else mode
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"simulate_{_sanitize_run_label(label)}_{ts}"


def _coerce_timestamp(
    value: Any,
    *,
    field: str,
    source_trace: Path,
    action_id: str,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SimulateError(
            f"{source_trace} action {action_id!r} is missing a numeric {field}"
        ) from exc


def _load_trace_session(
    source_trace: Path,
    task_source: Path,
    docker_image_override: str | None = None,
) -> LoadedTraceSession:
    agent_id, metadata, actions, summary = _parse_trace_session_file(source_trace)
    scaffold = metadata.get("scaffold", "unknown") if metadata else "unknown"
    resolved_source = _resolve_task_source(metadata, task_source, source_trace)
    task = _find_or_synthesize_task(resolved_source, agent_id, metadata)
    return LoadedTraceSession(
        source_trace=source_trace,
        task_source=resolved_source,
        agent_id=agent_id,
        scaffold=scaffold,
        metadata=metadata,
        summary=summary,
        task=task,
        actions=actions,
        iterations=_group_actions_by_iteration(actions),
        docker_image_override=docker_image_override,
    )


def _load_trace_manifest(
    trace_manifest: Path,
    *,
    default_task_source: Path,
) -> list[tuple[Path, Path, str | None]]:
    try:
        raw = json.loads(trace_manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SimulateError(f"Invalid trace manifest JSON: {trace_manifest}") from exc
    if not isinstance(raw, list) or not raw:
        raise SimulateError("trace manifest must be a non-empty JSON array")

    base_dir = trace_manifest.parent
    entries: list[tuple[Path, Path, str | None]] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise SimulateError(
                f"trace manifest entry {index} must be an object with source_trace"
            )
        source_value = entry.get("source_trace")
        if not source_value:
            raise SimulateError(f"trace manifest entry {index} is missing source_trace")
        task_value = entry.get("task_source")
        docker_image = entry.get("docker_image")
        source_path = Path(source_value)
        task_path = Path(task_value) if task_value else default_task_source
        if not source_path.is_absolute():
            source_path = (base_dir / source_path).resolve()
        if task_value and not task_path.is_absolute():
            task_path = (base_dir / task_path).resolve()
        entries.append((source_path, task_path, docker_image))
    return entries


def _discover_traces(
    source_dir: Path,
    default_task_source: Path,
) -> list[tuple[Path, Path, str | None]]:
    """Discover all ``trace.jsonl`` files under *source_dir*.

    Each discovered trace is paired with the default task source and no
    docker-image override.  Files are sorted for deterministic ordering.
    """
    found = sorted(source_dir.rglob("trace.jsonl"))
    return [(p, default_task_source, None) for p in found]


def _expand_trace_inputs(
    trace_inputs: list[tuple[Path, Path, str | None]],
    *,
    num_agents: int = 0,
    trace_assignment: str = "manifest",
    trace_assignment_seed: int | None = None,
) -> list[tuple[Path, Path, str | None]]:
    """Expand *trace_inputs* for N:M trace-to-agent mapping.

    When *num_agents* ≤ 0, returns *trace_inputs* unchanged (1:1 mapping).
    When *num_agents* > 0, creates exactly *num_agents* entries using the
    assignment strategy.

    ``manifest`` cycles through the input list; ``random`` picks uniformly
    with replacement using an optional seed for reproducibility.
    """
    if num_agents <= 0:
        return list(trace_inputs)
    if not trace_inputs:
        raise ValueError("num_agents > 0 but no trace inputs available")
    if trace_assignment == "manifest":
        return [trace_inputs[i % len(trace_inputs)] for i in range(num_agents)]
    if trace_assignment == "random":
        rng = random.Random(trace_assignment_seed)
        return [rng.choice(trace_inputs) for _ in range(num_agents)]
    raise ValueError(f"Unknown trace_assignment: {trace_assignment!r}")


def _ensure_unique_agent_ids(sessions: list[LoadedTraceSession]) -> None:
    """Suffix duplicate *agent_id* values with ``--a{N}`` for uniqueness.

    When multiple agents replay the same trace they share the original
    *agent_id* from the source trace.  This function mutates each session's
    ``agent_id`` in-place so that every session has a distinct identity
    for output directories and trace records.
    """
    used: set[str] = set()
    for i, session in enumerate(sessions):
        aid = session.agent_id
        if aid in used:
            suffix = i
            new_id = f"{aid}--a{suffix}"
            while new_id in used:
                suffix += 1
                new_id = f"{aid}--a{suffix}"
            logger.warning(
                "Renamed duplicate agent_id %r -> %r "
                "(session %d of %d sharing this id).",
                aid, new_id, i, len(sessions),
            )
            session.agent_id = new_id
        used.add(session.agent_id)


def _resolve_docker_image(loaded: LoadedTraceSession) -> str | None:
    """Resolve docker image: manifest override > task[image_name] > task[docker_image].

    When the tasks JSON was downloaded directly from HuggingFace (no benchmark
    plugin normalization), neither field is present.  In that case derive the
    canonical SWE-bench image name from the instance_id.
    """
    image = (
        loaded.docker_image_override
        or loaded.task.get("image_name")
        or loaded.task.get("docker_image")
    )
    if image:
        return str(image)
    instance_id = loaded.task.get("instance_id")
    if instance_id:
        docker_compatible_id = str(instance_id).replace("__", "_1776_")
        return (
            f"docker.io/swebench/sweb.eval.x86_64.{docker_compatible_id}:latest"
        ).lower()
    return None


def _execution_environment(loaded: LoadedTraceSession) -> str:
    metadata = loaded.metadata or {}
    value = metadata.get("execution_environment")
    if value is None or value == "":
        # Backward compat for legacy traces that predate the
        # execution_environment field: host_controller agents always ran on
        # the host, so infer "host" from agent_runtime_mode before falling
        # back to the container default.
        if metadata.get("agent_runtime_mode") == "host_controller":
            logger.warning(
                "%s has no execution_environment metadata; inferring host "
                "from agent_runtime_mode=host_controller",
                loaded.source_trace,
            )
            return "host"
        logger.warning(
            "%s has no execution_environment metadata; assuming container",
            loaded.source_trace,
        )
        return "container"
    return str(value)


def _is_host_mode(loaded: LoadedTraceSession) -> bool:
    return _execution_environment(loaded) == "host"


def _validate_loaded_sessions(
    sessions: list[LoadedTraceSession],
    *,
    mode: str,
    replay_speed: float,
) -> None:
    if replay_speed <= 0:
        raise ValueError("replay_speed must be > 0")
    if not sessions:
        raise SimulateError("No trace sessions were loaded")
    if mode == "local_model" and len(sessions) != 1:
        raise SimulateError("local_model mode supports exactly one source trace")
    for session in sessions:
        if _is_host_mode(session):
            continue
        docker_image = _resolve_docker_image(session)
        if not docker_image:
            raise SimulateError(
                f"Task {session.agent_id!r} has no resolvable docker_image "
                "(set docker_image in manifest or ensure task has image_name)"
            )

    seen_agent_ids: set[str] = set()
    for session in sessions:
        if session.agent_id in seen_agent_ids:
            raise SimulateError(
                f"Duplicate agent_id across replay sessions: {session.agent_id!r}"
            )
        seen_agent_ids.add(session.agent_id)

        for action in session.actions:
            action_id = str(action.get("action_id", ""))
            ts_start = _coerce_timestamp(
                action.get("ts_start"),
                field="ts_start",
                source_trace=session.source_trace,
                action_id=action_id,
            )
            ts_end = _coerce_timestamp(
                action.get("ts_end"),
                field="ts_end",
                source_trace=session.source_trace,
                action_id=action_id,
            )
            if ts_end < ts_start:
                raise SimulateError(
                    f"{session.source_trace} action {action_id!r} has ts_end < ts_start"
                )


async def _prepare_container_session(
    loaded: LoadedTraceSession,
    *,
    container_executable: str,
    network_mode: str = "host",
    cpu_limit: float | None = None,
    cpuset_cpus: str | None = None,
    tool_profiling: str = "off",
    tool_profiling_tools: list[str] | None = None,
    output_dir: Path | None = None,
) -> PreparedTraceSession:
    """Prepare a Docker/Podman container and start a persistent replay agent."""
    from trace_collect.openclaw_tools import ContainerAgent
    from trace_collect.runtime.task_container import (
        _SHARED_BOOTSTRAP_CACHE,
        _DEFAULT_RUNTIME_PYTHONPATH,
        _CONTAINER_SYSTEM_PYTHON,
        _inspect_image_platform,
        TaskContainerExecConfig,
        bootstrap_task_container_python,
        resolve_running_container_exec_config,
    )

    t_setup_start = time.monotonic()

    docker_image = _resolve_docker_image(loaded)
    if not docker_image:
        raise SimulateError(
            f"Task {loaded.agent_id!r} has no resolvable docker_image"
        )
    normalized = normalize_image_reference(docker_image)
    fixed_name, _elapsed = await asyncio.to_thread(
        ensure_fixed_image,
        normalized,
        container_executable=container_executable,
    )
    # Inspect the image platform *before* starting the container so that a
    # failed inspection doesn't leak a running container.  This mirrors the
    # ordering in resolve_task_container_exec_config (collect path).
    image_platform = _inspect_image_platform(fixed_name, container_executable=container_executable)
    extra_args: list[str] = []
    if cpu_limit is not None:
        extra_args.append(f"--cpus={cpu_limit}")
    if cpuset_cpus:
        extra_args.append(f"--cpuset-cpus={cpuset_cpus}")
    # When VTune per-tool profiling is active, add capabilities, bind-mounts,
    # and env vars to the container start args.  VTune results are written to
    # <task_dir>/vtune/ (bind-mounted so they survive teardown).
    if tool_profiling == "vtune" and output_dir is not None:
        from trace_collect.vtune_report import _resolve_vtune
        vtune_bin, vtune_root = _resolve_vtune()
        # output_dir is the attempt-level directory (e.g. .../attempt_1/).
        # VTune results go into its vtune/ subdirectory so that each
        # attempt gets an independent profiling directory.
        vtune_out = output_dir.resolve() / "vtune"
        vtune_out.mkdir(parents=True, exist_ok=True)
        resolved_tools = tool_profiling_tools or ["exec-pytest"]
        # Pass SEP driver device node so VTune can use hardware PMU
        # counters instead of falling back to Driverless Perf mode.
        _sep_devices: list[str] = []
        for _candidate in ("/dev/sep5", "/dev/sep"):
            if os.path.exists(_candidate):
                _sep_devices.extend(["--device", _candidate])
                break
        extra_args.extend([
            "--cap-add", "PERFMON",
            "--cap-add", "SYS_ADMIN",
            "--cap-add", "SYS_PTRACE",
            *_sep_devices,
            "-v", f"{vtune_root}:{vtune_root}:ro",
            "-v", f"{output_dir.resolve()}:{output_dir.resolve()}",
            "-e", "VTUNE_PROFILE=1",
            "-e", f"VTUNE_BIN={vtune_bin}",
            "-e", f"VTUNE_OUT={vtune_out.resolve()}",
            "-e", f"VTUNE_TOOLS={','.join(resolved_tools)}",
        ])
    elif tool_profiling == "ksys":
        tools = ",".join(tool_profiling_tools or ["exec-pytest"])
        extra_args.extend([
            "-e", "VTUNE_PROFILE=1",
            "-e", f"VTUNE_TOOLS={tools}",
        ])
    container_id = await asyncio.to_thread(
        start_task_container,
        fixed_name,
        executable=container_executable,
        network_mode=network_mode,
        extra_args=extra_args if extra_args else None,
    )

    # Bootstrap Python runtime dependencies inside the container so that
    # replayed tool commands (e.g. pytest) can find packages like sniffio
    # that were available during the original collect run.  This mirrors
    # what _run_openclaw_in_task_container does in collect mode.
    site_dir = _SHARED_BOOTSTRAP_CACHE / "pydeps"
    pythonpath = f"{site_dir}:{_DEFAULT_RUNTIME_PYTHONPATH}"
    exec_config = TaskContainerExecConfig(
        runtime=_CONTAINER_SYSTEM_PYTHON,
        pythonpath=pythonpath,
        start_extra_args=(),
        bootstrap=True,
        bootstrap_site_dir=site_dir,
        image_platform=image_platform,
    )
    # Probe the running container for Python >=3.11 so that the bootstrap
    # uses the SAME interpreter as ContainerAgent._probe_python() will
    # later select for tool execution.  Without this, the bootstrap would
    # install C extensions (tiktoken, pydantic-core) for /usr/bin/python3
    # (3.10) while ContainerAgent runs /usr/local/bin/python (3.12).
    exec_config = await asyncio.to_thread(
        resolve_running_container_exec_config,
        container_id=container_id,
        exec_config=exec_config,
        container_executable=container_executable,
    )
    try:
        await asyncio.to_thread(
            bootstrap_task_container_python,
            container_id=container_id,
            exec_config=exec_config,
            container_executable=container_executable,
        )
    except Exception:
        await asyncio.to_thread(
            stop_task_container,
            container_id,
            executable=container_executable,
        )
        raise

    agent = ContainerAgent(
        container_id, container_executable, pythonpath=pythonpath,
    )
    try:
        await agent.start()
    except Exception:
        await asyncio.to_thread(
            stop_task_container, container_id, executable=container_executable,
        )
        raise

    container = PreparedContainer(
        container_id=container_id,
        container_executable=container_executable,
        docker_image=normalized,
        agent=agent,
    )
    container_setup_s = time.monotonic() - t_setup_start
    return PreparedTraceSession(
        loaded=loaded, container=container, container_setup_s=container_setup_s,
    )


async def _prepare_host_session(
    loaded: LoadedTraceSession,
) -> PreparedTraceSession:
    """Prepare a host-mode replay session without Docker/Podman.

    Creates a :class:`HostAgent` that executes tools directly on the host
    so that cloud_model replay re-runs tool calls instead of skipping them.
    """
    # Derive workspace from the source trace directory — the parent of
    # the trace file is the attempt dir, its parent is the instance dir.
    workspace = loaded.source_trace.parent.parent
    host_agent = HostAgent(workspace=workspace)
    return PreparedTraceSession(loaded=loaded, host_agent=host_agent)


def _log_trace_metadata(
    *,
    trace_logger: TraceLogger,
    mode: str,
    sessions: list[LoadedTraceSession],
    replay_speed: float,
    source_trace: Path | None,
    source_dir: Path | None,
    trace_manifest: Path | None,
    api_base: str | None,
    model: str | None,
    monitoring_policy: MonitoringPolicy,
    network_mode: str = "host",
    cpu_limit: float | None = None,
    cpuset_cpus: str | None = None,
) -> None:
    scaffolds = {session.scaffold for session in sessions}
    source_models = [
        (session.summary or {}).get("model", "unknown") for session in sessions
    ]
    metadata: dict[str, Any] = {
        "scaffold": sessions[0].scaffold if len(scaffolds) == 1 else "mixed",
        "execution_environment": (
            _execution_environment(sessions[0])
            if len({_execution_environment(session) for session in sessions}) == 1
            else "mixed"
        ),
        "mode": "simulate",
        "simulate_mode": mode,
        "replay_speed": replay_speed,
        "source_trace_count": len(sessions),
        "source_models": source_models,
        "network_mode": network_mode,
        "cpu_limit": cpu_limit,
        "cpuset_cpus": cpuset_cpus,
        "monitoring": monitoring_policy.to_dict(),
    }
    if source_trace is not None:
        metadata["source_trace"] = str(source_trace)
    if source_dir is not None:
        metadata["source_dir"] = str(source_dir)
    if trace_manifest is not None:
        metadata["trace_manifest"] = str(trace_manifest)
        metadata["source_traces"] = [str(session.source_trace) for session in sessions]
    if mode == "local_model":
        metadata["source_model"] = source_models[0]
        metadata["local_model"] = model
        metadata["local_api_base"] = api_base
        metadata["n_source_iterations"] = _iteration_count(sessions[0].actions)
    else:
        metadata["source_model"] = (
            source_models[0] if len(set(source_models)) == 1 else "multiple"
        )
        metadata["model"] = metadata["source_model"]
        metadata["replay_target"] = "cloud_replay"
    trace_logger.log_metadata(**metadata)


def _make_trace_action(
    *,
    loaded: LoadedTraceSession,
    action_type: str,
    action_id: str,
    iteration: int,
    ts_start: float,
    ts_end: float,
    data: dict[str, Any],
) -> TraceAction:
    return TraceAction(
        action_type=action_type,
        action_id=action_id,
        agent_id=loaded.agent_id,
        program_id=loaded.agent_id,
        iteration=iteration,
        ts_start=ts_start,
        ts_end=ts_end,
        data=data,
    )


def _make_trace_summary(
    *,
    loaded: LoadedTraceSession,
    success: bool,
    elapsed_s: float,
    source_model: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    summary = {
        "agent_id": loaded.agent_id,
        "task_id": loaded.agent_id,
        "success": success,
        "source_success": (loaded.summary or {}).get("success"),
        "n_iterations": _iteration_count(loaded.actions),
        "elapsed_s": elapsed_s,
        "source_trace": str(loaded.source_trace),
        "source_model": source_model,
    }
    summary.update(extra)
    return summary


async def _run_local_model_simulation(
    prepared_session: PreparedTraceSession,
    *,
    trace_logger: TraceLogger,
    replay_speed: float,
    api_base: str,
    api_key: str,
    model: str,
    command_timeout_s: float,
    metrics_url: str | None,
    warmup_skip_iterations: int,
    gpu_baseline: GpuBaseline | None = None,
    vllm_pid: int | None = None,
    gpu_sample_hz: float = 10.0,
    gpu_output_path: Path | None = None,
) -> None:
    loaded = prepared_session.loaded
    iterations = loaded.iterations
    source_model = (loaded.summary or {}).get("model", "unknown")
    logger.info(
        "Simulating %s [scaffold=%s]: %d iterations from %s, local model=%s",
        loaded.agent_id,
        loaded.scaffold,
        len(iterations),
        source_model,
        model,
    )

    metrics_client = VLLMMetricsClient(
        metrics_url=metrics_url,
        gpu_baseline=gpu_baseline,
        vllm_pid=vllm_pid,
    )
    logger.info(
        "vLLM metrics client: %s",
        f"enabled (url={metrics_url})" if metrics_client.is_enabled else "disabled",
    )

    gpu_sampler: GpuResourceSampler | None = None
    if gpu_baseline is not None and vllm_pid is not None and metrics_url and gpu_output_path:
        gpu_sampler = GpuResourceSampler(
            metrics_url=metrics_url,
            gpu_baseline=gpu_baseline,
            vllm_pid=vllm_pid,
            output_path=gpu_output_path,
            sample_hz=gpu_sample_hz,
        )
        await gpu_sampler.start()
        logger.info("GPU resource sampler started (%.1f Hz) → %s", gpu_sample_hz, gpu_output_path)

    client = None

    wall_start = time.time()
    total_iters = len(iterations)
    succeeded_iters = 0
    failed_iters = 0
    sorted_iters = sorted(iterations.keys())

    # Accumulate aggregate metrics from *simulated* actions for the
    # summary (and HTML viz header).
    sim_total_tokens = 0
    sim_total_llm_ms = 0.0
    sim_total_tool_ms = 0.0
    sim_llm_call_count = 0
    sim_tool_ms_by_name: dict[str, float] = {}

    try:
        for i, it_num in enumerate(sorted_iters):
            it_group = iterations[it_num]
            llm_actions = it_group.get("llms", [])
            tool_actions = it_group.get("tools", [])

            if not llm_actions:
                logger.warning("Iteration %d: no LLM actions, skipping", it_num)
                continue

            iter_failed = False
            for llm_idx, llm_action in enumerate(llm_actions):
                llm_data = llm_action.get("data", {})
                messages_in = llm_data.get("messages_in")
                n_tokens = llm_data.get("completion_tokens", 1) or 1

                if llm_data.get("transport_retry_terminal"):
                    ts_now = time.time()
                    llm_record = _make_trace_action(
                        loaded=loaded,
                        action_type="llm_call",
                        action_id=f"llm_{it_num}_{llm_idx}",
                        iteration=it_num,
                        ts_start=ts_now,
                        ts_end=ts_now,
                        data={
                            "raw_response": llm_data.get("raw_response", {}),
                            "prompt_tokens": llm_data.get("prompt_tokens", 0),
                            "completion_tokens": 0,
                            "llm_latency_ms": 0.0,
                            "simulate_source": str(loaded.source_trace),
                            "source_llm_latency_ms": llm_data.get("llm_latency_ms"),
                            "transport_retry": True,
                            "transport_retry_terminal": True,
                            "error": llm_data.get("error"),
                            "sim_metrics": {
                                "warmup": i < warmup_skip_iterations,
                                "failed": True,
                            },
                        },
                    )
                    trace_logger.log_trace_action(loaded.agent_id, llm_record)
                    iter_failed = True
                    break

                if not messages_in:
                    logger.warning("Iteration %d llm %d: no messages_in, skipping", it_num, llm_idx)
                    continue

                ts_start = time.time()

                try:
                    if client is None:
                        client = create_async_openai_client(
                            api_base=api_base,
                            api_key=api_key,
                            timeout=180.0,
                        )
                    ttft_ms, tpot_ms, llm_latency_ms = await _call_local_model_streaming(
                        client, model, messages_in, n_tokens
                    )
                except Exception as exc:
                    logger.error("Iteration %d llm %d: LLM call failed: %s", it_num, llm_idx, exc)
                    iter_failed = True
                    break

                ts_after_llm = time.time()

                scheduler_snapshot = metrics_client.get_snapshot()

                llm_record = _make_trace_action(
                    loaded=loaded,
                    action_type="llm_call",
                    action_id=f"llm_{it_num}_{llm_idx}",
                    iteration=it_num,
                    ts_start=ts_start,
                    ts_end=ts_after_llm,
                    data={
                        "raw_response": llm_data.get("raw_response", {}),
                        "prompt_tokens": llm_data.get("prompt_tokens", 0),
                        "completion_tokens": llm_data.get("completion_tokens", 0),
                        "llm_latency_ms": llm_latency_ms,
                        "ttft_ms": ttft_ms,
                        "tpot_ms": tpot_ms,
                        "simulate_source": str(loaded.source_trace),
                        "source_llm_latency_ms": llm_data.get("llm_latency_ms"),
                        "sim_metrics": {
                            "timing": {
                                "ttft_ms": ttft_ms,
                                "tpot_ms": tpot_ms,
                                "total_ms": llm_latency_ms,
                            },
                            "vllm_scheduler_snapshot": dataclasses.asdict(
                                scheduler_snapshot
                            ),
                            "warmup": i < warmup_skip_iterations,
                        },
                    },
                )
                trace_logger.log_trace_action(loaded.agent_id, llm_record)
                sim_llm_call_count += 1
                sim_total_tokens += int(llm_data.get("prompt_tokens") or 0)
                sim_total_tokens += int(llm_data.get("completion_tokens") or 0)
                sim_total_llm_ms += llm_latency_ms

            if iter_failed:
                failed_iters += 1
                continue

            ctr = prepared_session.container
            total_tool_ms = 0.0
            for tool_act in tool_actions:
                td = tool_act.get("data", {})
                tool_name = td.get("tool_name")
                tool_args = td.get("tool_args", "{}")
                if not tool_name:
                    continue

                tool_ts_start = time.time()
                if ctr is None:
                    # Host-mode tool cannot be re-executed without a container;
                    # preserve source-trace timing so total_tool_ms and Gantt
                    # spans remain faithful to the original run.
                    tool_result = td.get("tool_result", td.get("result", ""))
                    tool_duration_ms = float(td.get("duration_ms") or 0.0)
                    tool_success = bool(td.get("success", not td.get("error")))
                    # Advance wall clock so consecutive replayed spans in the
                    # output trace don't overlap (addresses PR #13 Codex P1).
                    await asyncio.sleep(max(0.0, tool_duration_ms / 1000.0 / replay_speed))
                    tool_ts_end = time.time()
                    sim_provenance = "replayed_from_trace"
                elif tool_name is not None and tool_name.startswith("mcp_"):
                    tool_result = td.get("tool_result", "")
                    tool_duration_ms = float(td.get("duration_ms") or 0.0)
                    tool_success = bool(td.get("success", True))
                    await asyncio.sleep(max(0.0, tool_duration_ms / 1000.0 / replay_speed))
                    tool_ts_end = time.time()
                    sim_provenance = "replayed_from_trace"
                else:
                    tool_result, tool_duration_ms, tool_success = await _exec_tool(
                        ctr.agent,
                        tool_name,
                        tool_args,
                        command_timeout_s,
                    )
                    tool_ts_end = time.time()
                    sim_provenance = "executed_in_container"
                total_tool_ms += tool_duration_ms

                _classified_name = classify_exec_tool_name(tool_name, tool_args)
                tool_record = _make_trace_action(
                    loaded=loaded,
                    action_type="tool_exec",
                    action_id=f"tool_{it_num}_{_classified_name}",
                    iteration=it_num,
                    ts_start=tool_ts_start,
                    ts_end=tool_ts_end,
                    data={
                        "tool_name": _classified_name,
                        "tool_args": tool_args,
                        "tool_result": tool_result,
                        "duration_ms": tool_duration_ms,
                        "success": tool_success,
                        "sim_metrics": {
                            "source": sim_provenance,
                            "sim_tool_format": (
                                "replayed_from_trace"
                                if sim_provenance == "replayed_from_trace"
                                else "container_exec"
                            ),
                            "warmup": i < warmup_skip_iterations,
                        },
                    },
                )
                trace_logger.log_trace_action(loaded.agent_id, tool_record)
                sim_total_tool_ms += tool_duration_ms
                sim_tool_ms_by_name[_classified_name] = (
                    sim_tool_ms_by_name.get(_classified_name, 0.0) + tool_duration_ms
                )

            succeeded_iters += 1

            logger.info(
                "[%d/%d] iter %d: %d llm calls, tool=%.0fms",
                i + 1,
                total_iters,
                it_num,
                len(llm_actions),
                total_tool_ms,
            )
    finally:
        if gpu_sampler is not None:
            await gpu_sampler.stop()
            logger.info("GPU resource sampler stopped → %s", gpu_output_path)

        wall_end = time.time()

        simulate_summary = _make_trace_summary(
            loaded=loaded,
            success=failed_iters == 0 and succeeded_iters == total_iters,
            elapsed_s=wall_end - wall_start,
            source_model=source_model,
            extra={
                "local_model": model,
                "local_api_base": api_base,
                "succeeded_iterations": succeeded_iters,
                "failed_iterations": failed_iters,
                "total_tokens": sim_total_tokens,
                "total_llm_ms": sim_total_llm_ms,
                "total_tool_ms": sim_total_tool_ms,
                "tool_ms_by_name": sim_tool_ms_by_name,
                "llm_call_time_count": sim_llm_call_count,
                "timing": {
                    "agent_exec_s": wall_end - wall_start,
                    "container_setup_s": prepared_session.container_setup_s,
                },
            },
        )
        trace_logger.log_summary(loaded.agent_id, simulate_summary)


async def _sleep_until_offset(
    *,
    replay_zero_monotonic: float,
    target_offset_s: float,
) -> None:
    delay_s = target_offset_s - (time.monotonic() - replay_zero_monotonic)
    if delay_s > 0:
        await asyncio.sleep(delay_s)


async def _delayed_replay(
    delay_s: float,
    prepared_session: PreparedTraceSession,
    **kwargs: Any,
) -> None:
    """Wait *delay_s* then run a single cloud-model session replay."""
    if delay_s > 0:
        logger.info(
            "Poisson delay %.1fs for %s",
            delay_s, prepared_session.loaded.agent_id,
        )
    await _sleep_until_offset(
        replay_zero_monotonic=float(kwargs['replay_zero_monotonic']),
        target_offset_s=delay_s,
    )
    await _replay_cloud_model_session(prepared_session, **kwargs)


async def _run_cloud_model_replay(
    prepared_sessions: list[PreparedTraceSession],
    *,
    trace_logger: TraceLogger | None,
    replay_speed: float,
    command_timeout_s: float,
    warmup_skip_iterations: int,
    serial: bool = False,
    arrival_offsets: list[float] | None = None,
    replay_zero_monotonic: float | None = None,
) -> None:
    if replay_zero_monotonic is None:
        replay_zero_monotonic = time.monotonic()
    offsets = arrival_offsets or [0.0] * len(prepared_sessions)

    if serial:
        for i in range(len(prepared_sessions)):
            await _delayed_replay(
                offsets[i],
                prepared_sessions[i],
                trace_logger=trace_logger,
                replay_zero_monotonic=replay_zero_monotonic,
                replay_speed=replay_speed,
                command_timeout_s=command_timeout_s,
                warmup_skip_iterations=warmup_skip_iterations,
            )
    else:
        # return_exceptions=True prevents the cascade failure where one
        # task raising causes trace_logger.close() in the outer finally
        # block while other tasks are still writing ("I/O operation on
        # closed file").  When per-agent logging is active (trace_logger
        # is None), each session writes to its own independent file so
        # there is no shared-handle risk.
        results = await asyncio.gather(
            *[
                _delayed_replay(
                    offsets[i],
                    prepared_sessions[i],
                    trace_logger=trace_logger,
                    replay_zero_monotonic=replay_zero_monotonic,
                    replay_speed=replay_speed,
                    command_timeout_s=command_timeout_s,
                    warmup_skip_iterations=warmup_skip_iterations,
                )
                for i in range(len(prepared_sessions))
            ],
            return_exceptions=True,
        )

        # Surface the first exception after all tasks have completed (and
        # the shared trace_logger is still open so other sessions can log
        # their partial results cleanly).
        exceptions = [
            (i, r) for i, r in enumerate(results)
            if isinstance(r, BaseException)
        ]
        if exceptions:
            for i, exc in exceptions:
                logger.error(
                    "Replay failed for session %d (%s): %s",
                    i,
                    prepared_sessions[i].loaded.agent_id,
                    exc,
                )
            first_i, first_exc = exceptions[0]
            raise SimulateError(
                f"{len(exceptions)}/{len(results)} replay sessions failed; "
                f"first: session {first_i} "
                f"({prepared_sessions[first_i].loaded.agent_id}): {first_exc}"
            ) from first_exc


async def _replay_cloud_model_session(
    prepared_session: PreparedTraceSession,
    *,
    trace_logger: TraceLogger | None = None,
    replay_zero_monotonic: float,
    replay_speed: float,
    command_timeout_s: float,
    warmup_skip_iterations: int,
) -> None:
    loaded = prepared_session.loaded
    ctr = prepared_session.container
    host_agent = prepared_session.host_agent
    source_model = (loaded.summary or {}).get("model", "unknown")

    logger.info(
        "Replaying %s [scaffold=%s]: %d actions from %s at %.2fx (event-driven)",
        loaded.agent_id,
        loaded.scaffold,
        len(loaded.actions),
        source_model,
        replay_speed,
    )

    wall_start = time.time()
    succeeded_actions = 0
    failed_actions = 0

    # ── Per-agent trace logger (avoids shared-file contention) ─────
    # When *trace_logger* is None and a per-agent output directory is
    # available, each session writes to its own trace.jsonl.  This
    # eliminates the serialisation bottleneck of a single shared file
    # with 640 concurrent flush() calls.
    _own_logger: TraceLogger | None = None
    _log: TraceLogger
    if trace_logger is not None:
        _log = trace_logger
    elif prepared_session.task_output_dir is not None:
        _own_logger = TraceLogger(
            prepared_session.task_output_dir, "trace",
        )
        _log = _own_logger
        # Write per-agent metadata (equivalent to _log_trace_metadata
        # but scoped to this single session).
        _log.log_metadata(
            scaffold=loaded.scaffold,
            execution_environment=_execution_environment(loaded),
            mode="simulate",
            simulate_mode="cloud_model",
            replay_speed=replay_speed,
            source_trace_count=1,
            source_models=[source_model],
            source_model=source_model,
            model=source_model,
            replay_target="cloud_replay",
            instance_id=loaded.agent_id,
            source_trace=str(loaded.source_trace),
            network_mode="host",
            cpu_limit=None,
            monitoring={},
        )
    else:
        raise RuntimeError(
            "trace_logger is None and task_output_dir is None — "
            "cannot create per-agent trace logger."
        )

    # Accumulate aggregate metrics from *replayed* actions so the
    # summary (and HTML viz header) reflects actual replay timing,
    # not the source trace durations.
    total_tokens = 0
    total_llm_ms = 0.0
    total_tool_ms = 0.0
    tool_ms_by_name: dict[str, float] = {}
    llm_call_time_count = 0

    total_actions = len(loaded.actions)
    report_interval = max(1, total_actions // 10)
    _action_seq = 0  # sequential counter for progress reporting

    for action in loaded.actions:
        _action_seq += 1
        action_id = str(action.get("action_id", ""))
        action_type = str(action.get("action_type", ""))
        iteration = int(action.get("iteration", 0))
        data = action.get("data", {})
        action_ts_start = _coerce_timestamp(
            action.get("ts_start"),
            field="ts_start",
            source_trace=loaded.source_trace,
            action_id=action_id,
        )
        action_ts_end = _coerce_timestamp(
            action.get("ts_end"),
            field="ts_end",
            source_trace=loaded.source_trace,
            action_id=action_id,
        )
        source_duration_s = max(0.0, action_ts_end - action_ts_start)

        # Periodic progress: first action + every report_interval actions
        if _action_seq == 1:
            print(
                f"  [{loaded.agent_id}] started ({total_actions} actions)",
                flush=True,
            )
        elif _action_seq % report_interval == 0:
            print(
                f"  [{loaded.agent_id}] {_action_seq}/{total_actions} actions",
                flush=True,
            )

        # Event-driven: each action starts immediately after the previous
        # one completes.  Tool execution may be faster or slower than the
        # original trace — we don't wait for the original ts_start.
        try:
            if action_type == "llm_call":
                record_ts_start = time.time()
                if source_duration_s > 0:
                    await asyncio.sleep(source_duration_s / replay_speed)
                record_ts_end = time.time()
                prompt_tokens = int(data.get("prompt_tokens") or 0)
                completion_tokens = int(data.get("completion_tokens") or 0)
                record = _make_trace_action(
                    loaded=loaded,
                    action_type="llm_call",
                    action_id=action_id or f"llm_{iteration}",
                    iteration=iteration,
                    ts_start=record_ts_start,
                    ts_end=record_ts_end,
                    data={
                        "raw_response": data.get("raw_response", {}),
                        "prompt_tokens": data.get("prompt_tokens", 0),
                        "completion_tokens": data.get("completion_tokens", 0),
                        "llm_latency_ms": (record_ts_end - record_ts_start) * 1000,
                        "simulate_source": str(loaded.source_trace),
                        "source_llm_latency_ms": data.get("llm_latency_ms"),
                        "replay_mode": "cloud_model",
                        "replay_speed": replay_speed,
                        "sim_metrics": {
                            "warmup": iteration < warmup_skip_iterations,
                        },
                    },
                )
                _log.log_trace_action(loaded.agent_id, record)
                succeeded_actions += 1
                llm_call_time_count += 1
                total_tokens += prompt_tokens
                total_tokens += completion_tokens
                total_llm_ms += (record_ts_end - record_ts_start) * 1000
                continue

            if action_type != "tool_exec":
                logger.warning(
                    "Skipping unsupported action_type=%s in %s",
                    action_type,
                    loaded.source_trace,
                )
                continue

            tool_name = data.get("tool_name")
            tool_args = data.get("tool_args", "{}")
            if not tool_name:
                logger.warning(
                    "Skipping tool action without tool_name in %s",
                    loaded.source_trace,
                )
                continue

            record_ts_start = time.time()
            source_duration_ms = float(data.get("duration_ms") or 0.0)

            # ── Resolve which agent executes the tool ──────────────
            # Prefer host-agent (host-mode benchmarks) over
            # container-agent (container-mode benchmarks).  Fall back
            # to trace replay for MCP tools (not supported by either).
            agent_for_tool: Any = host_agent if host_agent is not None else (
                ctr.agent if ctr is not None else None
            )

            if agent_for_tool is not None and (
                tool_name is None or not tool_name.startswith("mcp_")
            ):
                tool_result, duration_ms, tool_success = await _exec_tool(
                    agent_for_tool,
                    tool_name,
                    tool_args,
                    command_timeout_s,
                )
                replay_source = (
                    "executed_on_host"
                    if host_agent is not None
                    else "executed_in_container"
                )
            elif tool_name is not None and tool_name.startswith("mcp_"):
                if source_duration_ms > 0:
                    await asyncio.sleep(source_duration_ms / 1000 / replay_speed)
                tool_result = data.get("tool_result", "")
                tool_success = bool(data.get("success", True))
                duration_ms = (time.time() - record_ts_start) * 1000
                replay_source = "replayed_from_trace"
            else:
                logger.warning(
                    "No agent available for %s action=%s tool=%s; replaying from trace",
                    loaded.agent_id,
                    action_id,
                    tool_name,
                )
                replay_source = "replayed_from_trace"
                tool_result = data.get("tool_result", data.get("result", ""))
                tool_success = bool(data.get("success", not data.get("error")))
                if source_duration_ms > 0:
                    await asyncio.sleep(source_duration_ms / 1000 / replay_speed)
                duration_ms = (time.time() - record_ts_start) * 1000
            record_ts_end = time.time()
            _classified_name = classify_exec_tool_name(tool_name, tool_args)
            tool_record = _make_trace_action(
                loaded=loaded,
                action_type="tool_exec",
                action_id=action_id or f"tool_{iteration}_{_classified_name}",
                iteration=iteration,
                ts_start=record_ts_start,
                ts_end=record_ts_end,
                data={
                    "tool_name": _classified_name,
                    "tool_args": tool_args,
                    "tool_result": tool_result,
                    "duration_ms": duration_ms,
                    "success": tool_success,
                    "simulate_source": str(loaded.source_trace),
                    "source_duration_ms": source_duration_ms,
                    "replay_mode": "cloud_model",
                    "replay_speed": replay_speed,
                    "replay_source": replay_source,
                    "sim_metrics": {
                        "warmup": iteration < warmup_skip_iterations,
                        "source": replay_source,
                        "sim_tool_format": (
                            "host_exec" if replay_source == "executed_on_host"
                            else "container_exec" if replay_source == "executed_in_container"
                            else "trace_replay"
                        ),
                    },
                },
            )
            _log.log_trace_action(loaded.agent_id, tool_record)
            succeeded_actions += 1
            total_tool_ms += duration_ms
            tool_ms_by_name[_classified_name] = (
                tool_ms_by_name.get(_classified_name, 0.0) + duration_ms
            )
        except Exception as exc:
            logger.error(
                "Replay action failed for %s action=%s: %s",
                loaded.agent_id,
                action_id,
                exc,
            )
            failed_actions += 1

    wall_end = time.time()

    _elapsed = wall_end - wall_start
    print(
        f"  [{loaded.agent_id}] done: {succeeded_actions}/{total_actions} "
        f"actions in {_elapsed:.1f}s "
        f"({failed_actions} failed)",
        flush=True,
    )

    # Post-loop cleanup and logging.  Wrapped in try/except so an
    # exception here (e.g. sampler I/O error) doesn't cascade into
    # other concurrent sessions via the shared trace_logger close.
    try:
        # Stop the resource sampler immediately so resources.json doesn't
        # include the idle gap between replay end and container teardown.
        if prepared_session.sampler is not None:
            samples = prepared_session.sampler.stop()
            prepared_session.sampler = None
            if (
                prepared_session.task_output_dir is not None
                and prepared_session.monitoring_policy is not None
            ):
                summary = summarize_samples(samples)
                summary["monitoring"] = {
                    **prepared_session.monitoring_policy.to_dict(),
                    "status": "collected" if samples else "enabled_no_samples",
                }
                attempt_layout.write_resources_json(
                    prepared_session.task_output_dir, samples, summary,
                )
                logger.info(
                    "Wrote %d resource samples → %s",
                    len(samples),
                    prepared_session.task_output_dir / "resources.json",
                )
            prepared_session._resources_written = True

        _log.log_summary(
            loaded.agent_id,
            _make_trace_summary(
                loaded=loaded,
                success=failed_actions == 0,
                elapsed_s=_elapsed,
                source_model=source_model,
                extra={
                    "replay_mode": "cloud_model",
                    "replay_speed": replay_speed,
                    "succeeded_actions": succeeded_actions,
                    "failed_actions": failed_actions,
                    "total_tokens": total_tokens,
                    "total_llm_ms": total_llm_ms,
                    "total_tool_ms": total_tool_ms,
                    "tool_ms_by_name": tool_ms_by_name,
                    "llm_call_time_count": llm_call_time_count,
                    "timing": {
                        "agent_exec_s": _elapsed,
                        "container_setup_s": prepared_session.container_setup_s,
                    },
                },
            ),
        )
    except Exception as exc:
        logger.error(
            "Post-replay cleanup failed for %s: %s",
            loaded.agent_id,
            exc,
        )
        raise
    finally:
        if _own_logger is not None:
            _own_logger.close()


def _split_trace_by_agent(
    combined_path: Path,
    sessions: list[PreparedTraceSession],
) -> int:
    """Write per-task trace.jsonl from the combined JSONL, filtered by agent_id.

    Returns the number of per-task trace files written.
    """
    agent_dirs = {
        s.loaded.agent_id: s.task_output_dir
        for s in sessions
        if s.task_output_dir is not None
    }
    sessions_by_agent = {s.loaded.agent_id: s for s in sessions}
    if not agent_dirs:
        return 0

    per_agent: dict[str, list[str]] = {aid: [] for aid in agent_dirs}
    metadata_line: str | None = None

    try:
        with combined_path.open(encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                rtype = record.get("type")
                if rtype == "trace_metadata":
                    metadata_line = stripped
                    continue
                agent_id = record.get("agent_id")
                if agent_id in per_agent:
                    per_agent[agent_id].append(stripped)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to split trace %s: %s", combined_path, exc)
        return 0

    for agent_id, lines in per_agent.items():
        out_dir = agent_dirs[agent_id]
        out_path = out_dir / "trace.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            if metadata_line:
                metadata = json.loads(metadata_line)
                session = sessions_by_agent[agent_id].loaded
                metadata["scaffold"] = session.scaffold
                metadata["execution_environment"] = _execution_environment(session)
                metadata["instance_id"] = session.agent_id
                metadata["source_trace"] = str(session.source_trace)
                fh.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            for ln in lines:
                fh.write(ln + "\n")
        logger.info("Wrote per-task trace (%d records) → %s", len(lines), out_path)
    return len(per_agent)


def _flush_session_trace(
    combined_path: Path,
    prepared: PreparedTraceSession,
) -> None:
    """Write per-task ``trace.jsonl`` and HTML viz for a single finished session.

    Called after each serial replay so the output is visible immediately,
    not just at the end of a batch run.
    """
    agent_id = prepared.loaded.agent_id
    out_dir = prepared.task_output_dir
    if out_dir is None:
        return

    metadata_line: str | None = None
    lines: list[str] = []

    try:
        with combined_path.open(encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                rtype = record.get("type")
                if rtype == "trace_metadata":
                    metadata_line = stripped
                    continue
                if record.get("agent_id") == agent_id:
                    lines.append(stripped)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Failed to flush trace for %s: %s", agent_id, exc
        )
        return

    # Write per-task trace.jsonl
    out_path = out_dir / "trace.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        if metadata_line:
            metadata = json.loads(metadata_line)
            metadata["scaffold"] = prepared.loaded.scaffold
            metadata["instance_id"] = agent_id
            metadata["source_trace"] = str(prepared.loaded.source_trace)
            fh.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for ln in lines:
            fh.write(ln + "\n")
    logger.info(
        "Wrote per-task trace (%d records) → %s", len(lines), out_path
    )

    # Auto-generate HTML viz
    try:
        html = generate_html(out_dir)
        viz_path = out_dir / "trace_viz.html"
        viz_path.write_text(html, encoding="utf-8")
        logger.info("HTML viz written -> %s", viz_path)
    except Exception:
        logger.warning(
            "Failed to generate HTML viz for %s", out_dir, exc_info=True,
        )


def _worker_run_cloud_model(
    worker_id: int,
    trace_inputs: list[WorkerTraceInput],
    *,
    arrival_offsets: list[float],
    output_dir: str,
    container_executable: str,
    network_mode: str,
    replay_speed: float,
    command_timeout_s: float,
    warmup_skip_iterations: int,
    cpu_limit: float | None,
    cpuset_cpus: str | None,
    prep_semaphore: Any,
    replay_start_barrier: Any,
    replay_start_event: Any,
    replay_start_wall_time: Any,
    resource_monitoring: str = "auto",
    pmu_monitoring: str = "auto",
    ksys_monitoring: str = "auto",
    tool_profiling: str = "off",
    tool_profiling_tools: list[str] | None = None,
) -> dict[str, Any]:
    """Run a subset of cloud_model sessions in a subprocess.

    Each worker process owns its own asyncio event loop, so N agents
    spread across *workers* processes see N/workers agents per event
    loop — dramatically reducing sleep-wake contention.
    """
    # ── Setup logging for this worker ──────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format=(
            f"%(asctime)s [Worker-{worker_id}] "
            f"%(levelname)s %(name)s: %(message)s"
        ),
        force=True,
    )
    _worker_logger = logging.getLogger(f"worker.{worker_id}")

    async def _run() -> dict[str, Any]:
        # 1. Load sessions from paths
        loaded_sessions: list[LoadedTraceSession] = []
        for worker_input in trace_inputs:
            loaded = _load_trace_session(
                Path(worker_input.source_trace),
                Path(worker_input.task_source),
                worker_input.docker_image_override,
            )
            loaded.agent_id = worker_input.agent_id
            loaded_sessions.append(loaded)
        _validate_loaded_sessions(
            loaded_sessions,
            mode="cloud_model",
            replay_speed=replay_speed,
        )

        # 2. Compute monitoring policy — respect the user's CLI flags
        #    (passed through from the main process); compute host/container
        #    flags from this worker's own session subset.
        has_host = any(_is_host_mode(s) for s in loaded_sessions)
        has_container = any(
            not _is_host_mode(s) for s in loaded_sessions
        )
        monitoring_policy = resolve_simulate_monitoring(
            resource=resource_monitoring,
            pmu=pmu_monitoring,
            ksys=ksys_monitoring,
            concurrent=True,
            has_host_session=has_host,
            has_container_session=has_container,
        )
        output_path = Path(output_dir)

        # 3. Ensure file-descriptor headroom for this worker's sessions.
        _ensure_fd_headroom(len(loaded_sessions), concurrent=True)

        # 4. Prepare all sessions concurrently
        prepared_sessions: list[PreparedTraceSession] = []

        async def _cleanup_prepared() -> None:
            for prepared in prepared_sessions:
                try:
                    await _teardown_one_worker(prepared, monitoring_policy)
                except Exception:
                    _worker_logger.warning(
                        "Teardown error for %s",
                        prepared.loaded.agent_id,
                        exc_info=True,
                    )

        async def _prepare_one_worker(
            loaded: LoadedTraceSession,
        ) -> PreparedTraceSession:
            uses_container = not _is_host_mode(loaded)
            # Compute attempt directory BEFORE container preparation so
            # that VTune output goes into the attempt-level directory
            # (prevents subsequent attempts from overwriting VTune data).
            instance_dir = output_path / loaded.agent_id
            attempt_n = _next_attempt_number(instance_dir)
            task_dir = instance_dir / f"attempt_{attempt_n}"
            task_dir.mkdir(parents=True, exist_ok=True)
            if uses_container and prep_semaphore is not None:
                await asyncio.to_thread(prep_semaphore.acquire)
            try:
                if _is_host_mode(loaded):
                    prepared = await _prepare_host_session(loaded)
                else:
                    prepare_kwargs: dict[str, Any] = {
                        "container_executable": container_executable,
                        "network_mode": network_mode,
                        "tool_profiling": tool_profiling,
                        "tool_profiling_tools": tool_profiling_tools,
                        "output_dir": task_dir,
                    }
                    if cpu_limit is not None:
                        prepare_kwargs["cpu_limit"] = cpu_limit
                    if cpuset_cpus:
                        prepare_kwargs["cpuset_cpus"] = cpuset_cpus
                    prepared = await _prepare_container_session(
                        loaded, **prepare_kwargs,
                    )
            finally:
                if uses_container and prep_semaphore is not None:
                    prep_semaphore.release()
            prepared.monitoring_policy = monitoring_policy
            prepared.task_output_dir = task_dir
            if tool_profiling != "off":
                prepared.vtune_out_dir = task_dir / "vtune"
            try:
                if (
                    prepared.container is not None
                    and monitoring_policy.resource_enabled
                ):
                    sampler = ContainerStatsSampler(
                        container_id=prepared.container.container_id,
                        interval_s=0.5,
                        executable=prepared.container.container_executable,
                        enable_pmu=(
                            monitoring_policy.pmu_enabled
                            and tool_profiling != "vtune"
                        ),
                        enable_memory_bandwidth=(
                            monitoring_policy.memory_bandwidth_enabled
                        ),
                    )
                    sampler.start()
                    prepared.sampler = sampler
            except Exception:
                await _teardown_one_worker(prepared, monitoring_policy)
                raise
            return prepared

        try:
            results = await asyncio.gather(
                *[_prepare_one_worker(s) for s in loaded_sessions],
                return_exceptions=True,
            )
            # With return_exceptions=True, ALL tasks complete. Collect ALL
            # successful results first, then check for failures — otherwise
            # a failure at index i would leave sessions at i+1..N prepared
            # but never torn down (orphaned containers).
            first_exc: BaseException | None = None
            for loaded, r in zip(loaded_sessions, results):
                if isinstance(r, BaseException):
                    if first_exc is None:
                        first_exc = r
                    logger.error(
                        "Preparation failed for %s: %s",
                        loaded.agent_id, r,
                    )
                else:
                    prepared_sessions.append(r)

            if first_exc is not None:
                _abort_global_replay_start(
                    replay_start_barrier,
                    replay_start_event,
                )
                _worker_logger.error(
                    "Preparation failed: %d/%d sessions had errors",
                    sum(1 for r in results if isinstance(r, BaseException)),
                    len(results),
                )
                await _cleanup_prepared()
                return {
                    "worker_id": worker_id,
                    "success": False,
                    "error": str(first_exc),
                }

            _worker_logger.info(
                "Prepared %d/%d sessions",
                len(prepared_sessions), len(loaded_sessions),
            )
        except Exception as exc:
            _abort_global_replay_start(
                replay_start_barrier,
                replay_start_event,
            )
            _worker_logger.error(
                "Preparation failed: %s", exc, exc_info=True,
            )
            await _cleanup_prepared()
            return {"worker_id": worker_id, "success": False, "error": str(exc)}

        # 5. Run replay
        replay_ok = True
        replay_error: str | None = None
        if len(arrival_offsets) != len(prepared_sessions):
            _abort_global_replay_start(
                replay_start_barrier,
                replay_start_event,
            )
            replay_ok = False
            replay_error = 'Worker arrival-offset count does not match sessions'
        try:
            replay_zero = await _wait_for_global_replay_start(
                replay_start_barrier,
                replay_start_event,
                replay_start_wall_time,
                coordinator=False,
            )
        except Exception as exc:
            replay_ok = False
            replay_error = str(exc)
            replay_zero = time.monotonic()
        try:
            if replay_ok:
                await _run_cloud_model_replay(
                    prepared_sessions,
                    trace_logger=None,  # per-agent files
                    replay_speed=replay_speed,
                    command_timeout_s=command_timeout_s,
                    warmup_skip_iterations=warmup_skip_iterations,
                    arrival_offsets=arrival_offsets,
                    replay_zero_monotonic=replay_zero,
                )
        except Exception as exc:
            _worker_logger.error(
                "Replay failed: %s", exc, exc_info=True,
            )
            replay_ok = False
            replay_error = str(exc)

        await _cleanup_prepared()

        return {
            "worker_id": worker_id,
            "success": replay_ok,
            "n_sessions": len(prepared_sessions),
            "error": replay_error,
        }

    try:
        return asyncio.run(_run())
    except Exception as exc:
        _abort_global_replay_start(
            replay_start_barrier,
            replay_start_event,
        )
        _worker_logger.error("Fatal worker error: %s", exc, exc_info=True)
        return {"worker_id": worker_id, "success": False, "error": str(exc)}


async def _teardown_one_worker(
    prepared: PreparedTraceSession,
    monitoring_policy: MonitoringPolicy,
) -> None:
    """Teardown one prepared session: stop sampler, write resources, stop container.

    Used by both the main process (via the _teardown_one closure) and worker
    subprocesses.  Must be a module-level function so it is picklable by
    ProcessPoolExecutor.
    """
    first_error: Exception | None = None
    session_resource_enabled = (
        monitoring_policy.resource_enabled and prepared.container is not None
    )
    _container_samples: list[dict[str, Any]] = []
    try:
        if prepared.sampler is not None:
            sampler = prepared.sampler
            prepared.sampler = None
            _container_samples = sampler.stop()
            if prepared.task_output_dir is not None:
                summary = summarize_samples(_container_samples)
                summary["monitoring"] = {
                    **monitoring_policy.to_dict(),
                    "status": (
                        "collected" if _container_samples else "enabled_no_samples"
                    ),
                }
                attempt_layout.write_resources_json(
                    prepared.task_output_dir, _container_samples, summary,
                )
                logger.info(
                    "Wrote %d resource samples → %s",
                    len(_container_samples),
                    prepared.task_output_dir / "resources.json",
                )
        elif (
            prepared.task_output_dir is not None
            and not prepared._resources_written
        ):
            summary = summarize_samples([])
            summary["monitoring_disabled"] = not session_resource_enabled
            summary["monitoring"] = {
                **monitoring_policy.to_dict(),
                "status": (
                    "disabled" if not session_resource_enabled
                    else "enabled_no_samples"
                ),
            }
            attempt_layout.write_resources_json(
                prepared.task_output_dir,
                samples=[],
                summary=summary,
            )
    except Exception as exc:
        first_error = exc
        logger.error(
            "Resource teardown failed for %s: %s",
            prepared.loaded.agent_id,
            exc,
        )

    # Finalize per-tool profiling (VTune/ksys) data if available.
    # Must run before the container is stopped so the bind-mounted
    # VTune result directory is still accessible.
    if prepared.vtune_out_dir is not None and prepared.vtune_out_dir.is_dir():
        try:
            from trace_collect.vtune_report import finalize_vtune
            finalize_vtune(prepared.vtune_out_dir, _container_samples)
        except Exception as exc:
            if first_error is None:
                first_error = exc
            logger.error(
                "VTune finalization failed for %s: %s",
                prepared.loaded.agent_id,
                exc,
            )

    ctr = prepared.container
    if ctr is None:
        if prepared.host_agent is not None:
            try:
                await prepared.host_agent.stop()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
        return

    try:
        if ctr.agent is not None:
            await ctr.agent.stop()
    except Exception as exc:
        if first_error is None:
            first_error = exc
        logger.error(
            "Agent stop failed for %s: %s",
            prepared.loaded.agent_id,
            exc,
        )
    try:
        await asyncio.to_thread(
            stop_task_container,
            ctr.container_id,
            executable=ctr.container_executable,
        )
        prepared.container = None
    except Exception as exc:
        if first_error is None:
            first_error = exc
        logger.error(
            "Container stop failed for %s: %s",
            prepared.loaded.agent_id,
            exc,
        )
    if first_error is not None:
        raise first_error


async def simulate(
    *,
    source_trace: Path | None = None,
    source_dir: Path | None = None,
    trace_manifest: Path | None = None,
    task_source: Path,
    output_dir: Path,
    mode: str = "local_model",
    serial: bool = False,
    container_executable: str | None = None,
    network_mode: str = "host",
    api_base: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    command_timeout_s: float = 120.0,
    metrics_url: str | None = None,
    warmup_skip_iterations: int = 0,
    replay_speed: float = 1.0,
    arrival_mode: str = "closed_loop",
    arrival_rate_per_s: float | None = None,
    arrival_seed: int | None = None,
    structured_output: bool = False,
    gpu_baseline: GpuBaseline | None = None,
    vllm_pid: int | None = None,
    gpu_sample_hz: float = 10.0,
    resource_monitoring: str = "auto",
    pmu_monitoring: str = "auto",
    ksys_monitoring: str = "auto",
    num_agents: int = 0,
    trace_assignment: str = "manifest",
    trace_assignment_seed: int | None = None,
    cpu_limit: float | None = None,
    cpuset_cpus: str | None = None,
    workers: int = 1,
    prep_concurrency: int = 0,
    tool_profiling: str = "off",
    tool_profiling_tools: list[str] | None = None,
) -> Path:
    if workers < 1:
        raise ValueError('workers must be >= 1')
    if prep_concurrency < 0:
        raise ValueError('prep_concurrency must be >= 0')
    if source_trace is not None and trace_manifest is not None:
        raise ValueError("source_trace and trace_manifest are mutually exclusive")
    if source_trace is not None and source_dir is not None:
        raise ValueError("source_trace and source_dir are mutually exclusive")
    if source_dir is not None and trace_manifest is not None:
        raise ValueError("source_dir and trace_manifest are mutually exclusive")
    if source_trace is None and trace_manifest is None and source_dir is None:
        raise ValueError("simulate requires source_trace, source_dir, or trace_manifest")
    if mode not in {"local_model", "cloud_model"}:
        raise ValueError(f"Unsupported simulate mode: {mode}")
    if tool_profiling not in {"off", "vtune", "ksys"}:
        raise ValueError(
            f"--tool-profiling must be 'off', 'vtune', or 'ksys', got {tool_profiling!r}"
        )
    vtune = tool_profiling == "vtune"
    if vtune and container_executable is None:
        raise ValueError(
            "--tool-profiling vtune wraps in-container commands and only applies "
            "to container-mode traces (--container docker|podman)."
        )
    if tool_profiling_tools is None:
        tool_profiling_tools = ["exec-pytest"]

    if trace_manifest is not None:
        trace_inputs = _load_trace_manifest(
            trace_manifest,
            default_task_source=task_source.resolve(),
        )
    elif source_dir is not None:
        trace_inputs = _discover_traces(source_dir, task_source.resolve())
        if not trace_inputs:
            raise ValueError(
                f"No trace.jsonl files found under {source_dir}"
            )
    else:
        assert source_trace is not None
        trace_inputs = [(source_trace, task_source, None)]

    # Expand trace_inputs for N:M mapping (num_agents > 0).
    trace_inputs = _expand_trace_inputs(
        trace_inputs,
        num_agents=num_agents,
        trace_assignment=trace_assignment,
        trace_assignment_seed=trace_assignment_seed,
    )

    print(
        f"  [load] loading {len(trace_inputs)} trace sessions...",
        flush=True,
    )
    loaded_sessions = [
        _load_trace_session(source_path, task_path, docker_image_override=img)
        for source_path, task_path, img in trace_inputs
    ]
    print(
        f"  [load] {len(loaded_sessions)} sessions loaded, validating...",
        flush=True,
    )
    # Ensure unique agent_ids after possible N:M expansion.
    _ensure_unique_agent_ids(loaded_sessions)
    _validate_loaded_sessions(
        loaded_sessions,
        mode=mode,
        replay_speed=replay_speed,
    )
    concurrent = (
        mode == "cloud_model" and not serial and len(loaded_sessions) > 1
    )
    # Each concurrent ContainerAgent holds 3 pipes (stdin/stdout/stderr);
    # N sessions need at least N*5 fds including Python, Docker API, and
    # trace files.  Default ulimit -n 1024 is insufficient beyond ~200
    # concurrent containers.
    _ensure_fd_headroom(len(loaded_sessions), concurrent=concurrent)
    has_host_session = any(_is_host_mode(session) for session in loaded_sessions)
    has_container_session = any(
        not _is_host_mode(session) for session in loaded_sessions
    )
    monitoring_policy = resolve_simulate_monitoring(
        resource=resource_monitoring,
        pmu=pmu_monitoring,
        ksys=ksys_monitoring,
        concurrent=concurrent,
        has_host_session=has_host_session,
        has_container_session=has_container_session,
    )

    # Apply host-side CPU affinity when --cpu-limit is set and at least
    # one session runs in host mode (container sessions get --cpus below).
    if cpu_limit is not None and has_host_session:
        try:
            import psutil
            p = psutil.Process()
            # CPU affinity does not support fractional cores — round up
            # so that --cpu-limit 0.5 pins to at least 1 logical core
            # rather than setting an empty (unrestricted) mask.
            effective_cores = max(1, math.ceil(cpu_limit))
            total_cores = psutil.cpu_count()
            if total_cores is not None:
                effective_cores = min(effective_cores, total_cores)
            p.cpu_affinity(list(range(effective_cores)))
            logger.info(
                "Set CPU affinity to cores 0-%d "
                "(cpu_limit=%.1f, effective=%d)",
                effective_cores - 1, cpu_limit, effective_cores,
            )
        except Exception as exc:
            logger.warning(
                "Failed to set CPU affinity (cpu_limit=%.1f): %s",
                cpu_limit, exc,
            )

    output_path = Path(output_dir)
    if structured_output:
        output_path = output_path / _structured_output_subdir(
            loaded_sessions,
            arrival_mode=arrival_mode,
            arrival_rate_per_s=arrival_rate_per_s,
        )

    prepared_sessions: list[PreparedTraceSession] = []
    trace_logger: TraceLogger | None = None
    output_path.mkdir(parents=True, exist_ok=True)
    shared_prep_semaphore: Any = None

    # ── Helpers for session lifecycle ──────────────────────────────
    async def _prepare_one(loaded: LoadedTraceSession) -> PreparedTraceSession:
        # Compute attempt directory BEFORE container preparation so that
        # VTune output goes into the attempt-level directory (prevents
        # subsequent attempts from overwriting VTune data).
        instance_dir = output_path / loaded.agent_id
        attempt_n = _next_attempt_number(instance_dir)
        task_dir = instance_dir / f"attempt_{attempt_n}"
        task_dir.mkdir(parents=True, exist_ok=True)
        if _is_host_mode(loaded):
            prepared = await _prepare_host_session(loaded)
        else:
            if container_executable is None:
                raise ValueError(
                    "container_executable is required for container-mode traces"
                )
            prepare_kwargs: dict[str, Any] = {
                "container_executable": container_executable,
                "network_mode": network_mode,
                "tool_profiling": tool_profiling,
                "tool_profiling_tools": tool_profiling_tools,
                "output_dir": task_dir,
            }
            if cpu_limit is not None:
                prepare_kwargs["cpu_limit"] = cpu_limit
            if cpuset_cpus:
                prepare_kwargs["cpuset_cpus"] = cpuset_cpus
            if shared_prep_semaphore is not None:
                await asyncio.to_thread(shared_prep_semaphore.acquire)
            try:
                prepared = await _prepare_container_session(
                    loaded,
                    **prepare_kwargs,
                )
            finally:
                if shared_prep_semaphore is not None:
                    shared_prep_semaphore.release()
        prepared.monitoring_policy = monitoring_policy
        prepared.task_output_dir = task_dir
        if tool_profiling != "off":
            prepared.vtune_out_dir = task_dir / "vtune"
        return prepared

    async def _setup_one(prepared: PreparedTraceSession) -> None:
        if prepared.container is None or not monitoring_policy.resource_enabled:
            return
        sampler = ContainerStatsSampler(
            container_id=prepared.container.container_id,
            interval_s=0.5,
            executable=prepared.container.container_executable,
            # When VTune is active, the host-side perf-stat PMU sampling
            # competes with VTune's own PMU collection for the same set of
            # hardware counters, causing kernel multiplexing and degraded
            # accuracy on both sides.  Disable the host PMU to give VTune
            # exclusive counter access during profiled pytest windows.
            enable_pmu=monitoring_policy.pmu_enabled and not vtune,
            enable_memory_bandwidth=(
                monitoring_policy.memory_bandwidth_enabled
            ),
        )
        sampler.start()
        prepared.sampler = sampler

    async def _teardown_one(prepared: PreparedTraceSession) -> None:
        """Thin closure that delegates to the standalone worker teardown."""
        await _teardown_one_worker(prepared, monitoring_policy)

    run_id = _build_run_id(mode=mode, model=model)
    # Concurrent cloud-model uses per-agent trace files (no shared
    # file contention); skip the combined trace_logger.
    if concurrent:
        trace_logger = None
    else:
        trace_logger = TraceLogger(output_path, run_id)
        _log_trace_metadata(
            trace_logger=trace_logger,
            mode=mode,
            sessions=loaded_sessions,
            replay_speed=replay_speed,
            source_trace=source_trace,
            source_dir=source_dir,
            trace_manifest=trace_manifest,
            api_base=api_base,
            model=model,
            network_mode=network_mode,
            monitoring_policy=monitoring_policy,
            cpu_limit=cpu_limit,
            cpuset_cpus=cpuset_cpus,
        )
    ksys_session = (
        KsysSession.start(output_dir=output_path, log_dir=output_path)
        if monitoring_policy.ksys_enabled
        else None
    )
    sync_manager: Any = None
    worker_executor: ProcessPoolExecutor | None = None
    replay_start_barrier: Any = None
    replay_start_event: Any = None

    try:
        if mode == "local_model":
            # ── Single trace: batch prepare then local-model replay ──
            prepared = await _prepare_one(loaded_sessions[0])
            await _setup_one(prepared)
            prepared_sessions.append(prepared)

            assert api_base is not None
            assert api_key is not None
            assert model is not None
            gpu_output_path: Path | None = None
            if gpu_baseline is not None and vllm_pid is not None and metrics_url:
                task_dir = prepared.task_output_dir
                if task_dir is not None:
                    gpu_output_path = task_dir / "gpu_resources.json"
            await _run_local_model_simulation(
                prepared,
                trace_logger=trace_logger,
                replay_speed=replay_speed,
                api_base=api_base,
                api_key=api_key,
                model=model,
                command_timeout_s=command_timeout_s,
                metrics_url=metrics_url,
                warmup_skip_iterations=warmup_skip_iterations,
                gpu_baseline=gpu_baseline,
                vllm_pid=vllm_pid,
                gpu_sample_hz=gpu_sample_hz,
                gpu_output_path=gpu_output_path,
            )
        elif serial:
            # ── Serial cloud-model: prepare → replay → cleanup one at a time ──
            offsets = build_arrival_offsets(
                len(loaded_sessions),
                arrival_mode=arrival_mode,
                arrival_rate_per_s=arrival_rate_per_s,
                arrival_seed=arrival_seed,
            )
            for i, loaded in enumerate(loaded_sessions):
                logger.info(
                    "Serial replay [%d/%d]: preparing %s",
                    i + 1, len(loaded_sessions), loaded.agent_id,
                )
                prepared = await _prepare_one(loaded)
                await _setup_one(prepared)
                prepared_sessions.append(prepared)
                try:
                    await _replay_cloud_model_session(
                        prepared,
                        trace_logger=trace_logger,
                        replay_zero_monotonic=time.monotonic(),
                        replay_speed=replay_speed,
                        command_timeout_s=command_timeout_s,
                        warmup_skip_iterations=warmup_skip_iterations,
                    )
                finally:
                    await _teardown_one(prepared)
                    # Immediately write per-task trace for this session
                    _flush_session_trace(trace_logger.path, prepared)
        else:
            # ── Concurrent cloud-model ──
            # Local helper: concurrent prepare + replay for ONE process's batch
            async def _concurrent_run(
                sessions: list[LoadedTraceSession],
                offsets: list[float],
                *,
                replay_start_barrier: Any = None,
                replay_start_event: Any = None,
                replay_start_wall_time: Any = None,
                coordinator: bool = False,
            ) -> list[PreparedTraceSession]:
                """Prepare, setup, and replay *sessions* in this process."""
                local_prep_limit = (
                    len(sessions)
                    if shared_prep_semaphore is not None
                    else _resolve_prep_concurrency(
                        prep_concurrency,
                        len(sessions),
                    )
                )
                prep_sem = asyncio.Semaphore(local_prep_limit)
                total = len(sessions)
                _done: list[int] = [0]
                _report_every = max(1, min(20, total // 10))

                async def _prepare_with_limit(
                    loaded: LoadedTraceSession,
                ) -> PreparedTraceSession:
                    async with prep_sem:
                        p = await _prepare_one(loaded)
                        try:
                            await _setup_one(p)
                        except Exception:
                            await _teardown_one(p)
                            raise
                        _done[0] += 1
                        if _done[0] % _report_every == 0 or _done[0] == total:
                            print(
                                f"  [prepare] {_done[0]}/{total} containers ready",
                                flush=True,
                            )
                        return p

                print(
                    f"  [prepare] starting {total} containers "
                    f"(system-wide concurrency="
                    f"{_resolve_prep_concurrency(prep_concurrency, len(loaded_sessions))}"
                    ")...",
                    flush=True,
                )
                results = await asyncio.gather(
                    *[_prepare_with_limit(s) for s in sessions],
                    return_exceptions=True,
                )
                n_errors = sum(
                    1 for r in results if isinstance(r, BaseException)
                )
                if n_errors:
                    print(
                        f"  [prepare] {_done[0]}/{total} containers ready "
                        f"({n_errors} failed)",
                        flush=True,
                    )
                else:
                    print(
                        f"  [prepare] all {_done[0]} containers ready",
                        flush=True,
                    )

                out: list[PreparedTraceSession] = []
                # With return_exceptions=True, ALL tasks complete. Collect
                # ALL successful results before checking failures — otherwise
                # a failure at index i leaks sessions at i+1..N.
                first_exc: BaseException | None = None
                for i, r in enumerate(results):
                    if isinstance(r, BaseException):
                        if first_exc is None:
                            first_exc = r
                        logger.error(
                            "Preparation failed for session %d (%s): %s",
                            i, sessions[i].agent_id, r,
                        )
                    else:
                        out.append(r)

                if first_exc is not None:
                    if replay_start_barrier is not None:
                        _abort_global_replay_start(
                            replay_start_barrier,
                            replay_start_event,
                        )
                    n_failed = len(results) - len(out)
                    for p in out:
                        try:
                            await _teardown_one(p)
                        except Exception:
                            pass
                    raise SimulateError(
                        f"Preparation failed: {n_failed}/{len(results)} "
                        f"sessions had errors"
                    ) from first_exc

                if len(offsets) != len(out):
                    if replay_start_barrier is not None:
                        _abort_global_replay_start(
                            replay_start_barrier,
                            replay_start_event,
                        )
                    raise SimulateError(
                        "Arrival-offset count does not match prepared sessions"
                    )
                try:
                    replay_zero = time.monotonic()
                    if replay_start_barrier is not None:
                        replay_zero = await _wait_for_global_replay_start(
                            replay_start_barrier,
                            replay_start_event,
                            replay_start_wall_time,
                            coordinator=coordinator,
                        )
                    await _run_cloud_model_replay(
                        out,
                        trace_logger=trace_logger,
                        replay_speed=replay_speed,
                        command_timeout_s=command_timeout_s,
                        warmup_skip_iterations=warmup_skip_iterations,
                        arrival_offsets=offsets,
                        replay_zero_monotonic=replay_zero,
                    )
                except BaseException:
                    await asyncio.gather(
                        *[_teardown_one(p) for p in out],
                        return_exceptions=True,
                    )
                    raise
                return out

            global_offsets = build_arrival_offsets(
                len(loaded_sessions),
                arrival_mode=arrival_mode,
                arrival_rate_per_s=arrival_rate_per_s,
                arrival_seed=arrival_seed,
            )
            if workers > 1:
                # ── Multi-process concurrent ────────────────────────────────
                # Split N agents across W worker processes so each event loop
                # only handles N/W agents — eliminating single-loop congestion.
                n_total = len(loaded_sessions)
                n_workers = min(workers, n_total)
                partitions = _partition_sessions_and_offsets(
                    loaded_sessions,
                    global_offsets,
                    n_workers,
                )
                main_chunk, main_offsets = partitions[0]
                worker_chunks = [
                    sessions for sessions, _offsets in partitions[1:]
                ]
                worker_offset_chunks = [
                    offsets for _sessions, offsets in partitions[1:]
                ]

                print(
                    f"  [workers] {n_total} sessions × {n_workers} workers "
                    f"({len(main_chunk)} main / "
                    f"{[len(c) for c in worker_chunks]} workers)",
                    flush=True,
                )

                # Spawn worker processes (they independently prepare, replay,
                # teardown and write per-agent trace files into output_path).
                # NOTE: Do NOT use `with ProcessPoolExecutor(...)` here — its
                # __exit__ calls shutdown(wait=True) which BLOCKS the event
                # loop thread until ALL workers finish, preventing the main
                # process from processing its chunk in parallel.
                worker_futures: list[Any] = []
                sync_manager = multiprocessing.Manager()
                shared_prep_semaphore = sync_manager.BoundedSemaphore(
                    _resolve_prep_concurrency(prep_concurrency, n_total)
                )
                replay_start_barrier = sync_manager.Barrier(n_workers)
                replay_start_event = sync_manager.Event()
                replay_start_wall_time = sync_manager.Value("d", 0.0)
                if worker_chunks:
                    worker_executor = ProcessPoolExecutor(
                        max_workers=len(worker_chunks)
                    )
                    for wid, (chunk, chunk_offsets) in enumerate(
                        zip(worker_chunks, worker_offset_chunks),
                        start=1,
                    ):
                        inputs = [_worker_trace_input(s) for s in chunk]
                        fut = worker_executor.submit(
                            _worker_run_cloud_model,
                            wid,
                            inputs,
                            arrival_offsets=chunk_offsets,
                            output_dir=str(output_path),
                            container_executable=(
                                container_executable or "docker"
                            ),
                            network_mode=network_mode,
                            replay_speed=replay_speed,
                            command_timeout_s=command_timeout_s,
                            warmup_skip_iterations=warmup_skip_iterations,
                            cpu_limit=cpu_limit,
                            cpuset_cpus=cpuset_cpus,
                            prep_semaphore=shared_prep_semaphore,
                            replay_start_barrier=replay_start_barrier,
                            replay_start_event=replay_start_event,
                            replay_start_wall_time=replay_start_wall_time,
                            resource_monitoring=resource_monitoring,
                            pmu_monitoring=pmu_monitoring,
                            ksys_monitoring=ksys_monitoring,
                            tool_profiling=tool_profiling,
                            tool_profiling_tools=tool_profiling_tools,
                        )

                        def _abort_if_worker_exits_early(
                            _future: Any,
                            *,
                            barrier: Any = replay_start_barrier,
                            start_event: Any = replay_start_event,
                        ) -> None:
                            if not start_event.is_set():
                                _abort_global_replay_start(
                                    barrier,
                                    start_event,
                                )

                        fut.add_done_callback(_abort_if_worker_exits_early)
                        worker_futures.append(fut)

                # Main process handles its chunk IN PARALLEL with workers
                main_error: BaseException | None = None
                print(
                    f"  [workers] main process handling "
                    f"{len(main_chunk)} sessions...",
                    flush=True,
                )
                try:
                    prepared_sessions = await _concurrent_run(
                        main_chunk,
                        main_offsets,
                        replay_start_barrier=replay_start_barrier,
                        replay_start_event=replay_start_event,
                        replay_start_wall_time=replay_start_wall_time,
                        coordinator=True,
                    )
                except BaseException as exc:
                    main_error = exc
                    _abort_global_replay_start(
                        replay_start_barrier,
                        replay_start_event,
                    )

                # Wait for all workers and check results
                worker_errors: list[str] = []
                try:
                    worker_results = await asyncio.gather(
                        *[
                            asyncio.wrap_future(fut)
                            for fut in worker_futures
                        ],
                        return_exceptions=True,
                    )
                    for worker_index, result in enumerate(
                        worker_results,
                        start=1,
                    ):
                        if isinstance(result, BaseException):
                            worker_errors.append(
                                f"Worker {worker_index}: {result}"
                            )
                        elif not result.get("success"):
                            wid = result.get("worker_id")
                            err = result.get("error") or "unknown error"
                            worker_errors.append(f"Worker {wid}: {err}")
                finally:
                    _abort_global_replay_start(
                        replay_start_barrier,
                        replay_start_event,
                    )
                    if worker_executor is not None:
                        await asyncio.to_thread(
                            worker_executor.shutdown,
                            wait=True,
                            cancel_futures=False,
                        )
                    await asyncio.to_thread(sync_manager.shutdown)
                    shared_prep_semaphore = None
                    worker_executor = None
                    sync_manager = None
                    replay_start_barrier = None
                    replay_start_event = None

                if main_error is not None:
                    raise main_error
                if worker_errors:
                    raise SimulateError(
                        f"{len(worker_errors)}/{len(worker_futures)} "
                        f"workers failed: {'; '.join(worker_errors)}"
                    )

                print("  [workers] all workers completed", flush=True)

            else:
                # ── Single-process concurrent (original behaviour) ─────
                prepared_sessions = await _concurrent_run(
                    loaded_sessions,
                    global_offsets,
                )
    finally:
        if replay_start_barrier is not None:
            _abort_global_replay_start(
                replay_start_barrier,
                replay_start_event,
            )
        if worker_executor is not None:
            await asyncio.to_thread(
                worker_executor.shutdown,
                wait=True,
                cancel_futures=False,
            )
        if sync_manager is not None:
            await asyncio.to_thread(sync_manager.shutdown)
        if ksys_session is not None:
            ksys_session.stop()
        if trace_logger is not None:
            trace_logger.close()
            logger.info("Combined trace -> %s", trace_logger.path)
            n = _split_trace_by_agent(trace_logger.path, prepared_sessions)
            logger.info("Split: %d per-task trace files written", n)
        # Teardown any sessions that weren't cleaned up inline
        # (in serial mode these have already been torn down; in concurrent
        # mode they haven't).
        pending_teardown = [
            prepared
            for prepared in prepared_sessions
            if prepared.sampler is not None or prepared.container is not None
        ]
        if pending_teardown:
            logger.info(
                "Tearing down %d sessions (containers + samplers)...",
                len(pending_teardown),
            )
            await asyncio.gather(
                *[_teardown_one(p) for p in pending_teardown],
                return_exceptions=True,
            )
            logger.info("Teardown complete.")

    trace_file: Path
    if trace_logger is not None:
        trace_file = trace_logger.path
    else:
        trace_file = output_path  # per-agent files already written
    logger.info("Simulate complete [%s] -> %s", mode, trace_file)

    # ── Auto-generate HTML viz (concurrent, offloaded to threads) ─
    viz_tasks: list[tuple[Path, Path]] = []
    # Collect task dirs from main process sessions
    for prepared in prepared_sessions:
        task_dir = prepared.task_output_dir
        if task_dir is None or not task_dir.is_dir():
            continue
        trace_jsonl = task_dir / "trace.jsonl"
        if trace_jsonl.exists():
            viz_tasks.append((task_dir, task_dir / "trace_viz.html"))
    # In multi-process mode, also scan output_path for worker-created dirs
    # that aren't already in prepared_sessions.
    if workers > 1:
        for agent_dir in sorted(output_path.iterdir()):
            if not agent_dir.is_dir():
                continue
            # Already handled by main process?
            already = any(
                p.task_output_dir is not None
                and p.task_output_dir.parent == agent_dir
                for p in prepared_sessions
            )
            if already:
                continue
            for attempt_dir in sorted(agent_dir.iterdir()):
                if not attempt_dir.is_dir():
                    continue
                trace_jsonl = attempt_dir / "trace.jsonl"
                if trace_jsonl.exists():
                    viz_tasks.append(
                        (attempt_dir, attempt_dir / "trace_viz.html")
                    )

    if viz_tasks:
        logger.info("Generating HTML viz for %d sessions...", len(viz_tasks))

        async def _gen_viz(task_dir: Path, viz_path: Path) -> None:
            try:
                html = await asyncio.to_thread(generate_html, task_dir)
                viz_path.write_text(html, encoding="utf-8")
                logger.info("HTML viz written -> %s", viz_path)
            except Exception:
                logger.warning(
                    "Failed to generate HTML viz for %s",
                    task_dir,
                    exc_info=True,
                )

        await asyncio.gather(
            *[_gen_viz(d, p) for d, p in viz_tasks],
            return_exceptions=True,
        )
        logger.info("HTML viz complete.")

    return trace_file
