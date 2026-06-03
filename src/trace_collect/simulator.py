from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.base import TraceAction
from harness.container_image_prep import ensure_fixed_image, normalize_image_reference
from harness.container_stats_sampler import ContainerStatsSampler, summarize_samples
from harness.gpu_resource_sampler import GpuResourceSampler
from harness.runner import build_arrival_offsets
from harness.metrics_client import VLLMMetricsClient
from harness.scheduler_hooks import GpuBaseline
from harness.trace_logger import TraceLogger
from llm_call import create_async_openai_client
from trace_collect import attempt_layout
from trace_collect.attempt_pipeline import start_task_container, stop_task_container

logger = logging.getLogger(__name__)


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
    sampler: ContainerStatsSampler | None = None
    task_output_dir: Path | None = None


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

    actions.sort(
        key=lambda action: (
            float(action.get("ts_start", 0.0)),
            float(action.get("ts_end", 0.0)),
            int(action.get("iteration", 0)),
            str(action.get("action_id", "")),
        )
    )
    return first_agent_id, metadata, actions, summaries.get(first_agent_id)


def _find_task(task_source: Path, agent_id: str) -> dict[str, Any]:
    tasks = json.loads(task_source.read_text(encoding="utf-8"))
    for task in tasks:
        if task["instance_id"] == agent_id:
            return task
    raise SimulateError(f"Task {agent_id!r} not found in {task_source}")


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
    task = _find_task(task_source, agent_id)
    return LoadedTraceSession(
        source_trace=source_trace,
        task_source=task_source,
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


def _resolve_docker_image(loaded: LoadedTraceSession) -> str | None:
    """Resolve docker image: manifest override > task[image_name] > task[docker_image]."""
    return (
        loaded.docker_image_override
        or loaded.task.get("image_name")
        or loaded.task.get("docker_image")
    )


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
) -> PreparedTraceSession:
    """Prepare a Docker/Podman container and start a persistent replay agent."""
    from trace_collect.openclaw_tools import ContainerAgent

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
    container_id = await asyncio.to_thread(
        start_task_container,
        fixed_name,
        executable=container_executable,
        network_mode=network_mode,
    )

    agent = ContainerAgent(container_id, container_executable)
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
    return PreparedTraceSession(loaded=loaded, container=container)


async def _prepare_host_session(
    loaded: LoadedTraceSession,
) -> PreparedTraceSession:
    """Prepare a host-mode replay session without Docker/Podman."""
    return PreparedTraceSession(loaded=loaded, container=None)


def _log_trace_metadata(
    *,
    trace_logger: TraceLogger,
    mode: str,
    sessions: list[LoadedTraceSession],
    replay_speed: float,
    source_trace: Path | None,
    trace_manifest: Path | None,
    api_base: str | None,
    model: str | None,
    network_mode: str = "host",
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
    }
    if source_trace is not None:
        metadata["source_trace"] = str(source_trace)
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
                            "messages_in": messages_in,
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
                        "messages_in": messages_in,
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

                tool_record = _make_trace_action(
                    loaded=loaded,
                    action_type="tool_exec",
                    action_id=f"tool_{it_num}_{tool_name}",
                    iteration=it_num,
                    ts_start=tool_ts_start,
                    ts_end=tool_ts_end,
                    data={
                        "tool_name": tool_name,
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
        await asyncio.sleep(delay_s)
    await _replay_cloud_model_session(prepared_session, **kwargs)


async def _run_cloud_model_replay(
    prepared_sessions: list[PreparedTraceSession],
    *,
    trace_logger: TraceLogger,
    replay_speed: float,
    command_timeout_s: float,
    warmup_skip_iterations: int,
    arrival_offsets: list[float] | None = None,
) -> None:
    replay_zero_monotonic = time.monotonic()
    offsets = arrival_offsets or [0.0] * len(prepared_sessions)
    await asyncio.gather(
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
        ]
    )


async def _replay_cloud_model_session(
    prepared_session: PreparedTraceSession,
    *,
    trace_logger: TraceLogger,
    replay_zero_monotonic: float,
    replay_speed: float,
    command_timeout_s: float,
    warmup_skip_iterations: int,
) -> None:
    loaded = prepared_session.loaded
    ctr = prepared_session.container
    source_model = (loaded.summary or {}).get("model", "unknown")
    source_zero = _coerce_timestamp(
        loaded.actions[0].get("ts_start"),
        field="ts_start",
        source_trace=loaded.source_trace,
        action_id=str(loaded.actions[0].get("action_id", "")),
    )

    logger.info(
        "Replaying %s [scaffold=%s]: %d actions from %s at %.2fx",
        loaded.agent_id,
        loaded.scaffold,
        len(loaded.actions),
        source_model,
        replay_speed,
    )

    wall_start = time.time()
    succeeded_actions = 0
    failed_actions = 0

    for action in loaded.actions:
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

        await _sleep_until_offset(
            replay_zero_monotonic=replay_zero_monotonic,
            target_offset_s=(action_ts_start - source_zero) / replay_speed,
        )

        try:
            if action_type == "llm_call":
                record_ts_start = time.time()
                if source_duration_s > 0:
                    await asyncio.sleep(source_duration_s / replay_speed)
                record_ts_end = time.time()
                record = _make_trace_action(
                    loaded=loaded,
                    action_type="llm_call",
                    action_id=action_id or f"llm_{iteration}",
                    iteration=iteration,
                    ts_start=record_ts_start,
                    ts_end=record_ts_end,
                    data={
                        "messages_in": data.get("messages_in"),
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
                trace_logger.log_trace_action(loaded.agent_id, record)
                succeeded_actions += 1
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
            if ctr is None:
                logger.info(
                    "Skipping host-mode tool action for %s action=%s tool=%s",
                    loaded.agent_id,
                    action_id,
                    tool_name,
                )
                replay_source = "skipped_host_mode"
                tool_result = data.get("tool_result", data.get("result", ""))
                # Research-agent tools don't emit "success"; fall back to
                # absence-of-error so successful runs aren't mislabeled.
                tool_success = bool(data.get("success", not data.get("error")))
                if source_duration_ms > 0:
                    await asyncio.sleep(source_duration_ms / 1000 / replay_speed)
                duration_ms = (time.time() - record_ts_start) * 1000
            elif tool_name is not None and tool_name.startswith("mcp_"):
                if source_duration_ms > 0:
                    await asyncio.sleep(source_duration_ms / 1000 / replay_speed)
                tool_result = data.get("tool_result", "")
                tool_success = bool(data.get("success", True))
                duration_ms = (time.time() - record_ts_start) * 1000
                replay_source = "replayed_from_trace"
            elif ctr is not None:
                tool_result, duration_ms, tool_success = await _exec_tool(
                    ctr.agent,
                    tool_name,
                    tool_args,
                    command_timeout_s,
                )
                replay_source = "executed_in_container"
            record_ts_end = time.time()
            tool_record = _make_trace_action(
                loaded=loaded,
                action_type="tool_exec",
                action_id=action_id or f"tool_{iteration}_{tool_name}",
                iteration=iteration,
                ts_start=record_ts_start,
                ts_end=record_ts_end,
                data={
                    "tool_name": tool_name,
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
                        "sim_tool_format": replay_source
                        if replay_source == "skipped_host_mode"
                        else "container_exec",
                    },
                },
            )
            trace_logger.log_trace_action(loaded.agent_id, tool_record)
            succeeded_actions += 1
        except Exception as exc:
            logger.error(
                "Replay action failed for %s action=%s: %s",
                loaded.agent_id,
                action_id,
                exc,
            )
            failed_actions += 1

    wall_end = time.time()
    trace_logger.log_summary(
        loaded.agent_id,
        _make_trace_summary(
            loaded=loaded,
            success=failed_actions == 0,
            elapsed_s=wall_end - wall_start,
            source_model=source_model,
            extra={
                "replay_mode": "cloud_model",
                "replay_speed": replay_speed,
                "succeeded_actions": succeeded_actions,
                "failed_actions": failed_actions,
            },
        ),
    )


def _split_trace_by_agent(
    combined_path: Path,
    sessions: list[PreparedTraceSession],
) -> None:
    """Write per-task trace.jsonl from the combined JSONL, filtered by agent_id."""
    agent_dirs = {
        s.loaded.agent_id: s.task_output_dir
        for s in sessions
        if s.task_output_dir is not None
    }
    sessions_by_agent = {s.loaded.agent_id: s for s in sessions}
    if not agent_dirs:
        return

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
        return

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


async def simulate(
    *,
    source_trace: Path | None = None,
    trace_manifest: Path | None = None,
    task_source: Path,
    output_dir: Path,
    mode: str = "local_model",
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
) -> Path:
    if source_trace is not None and trace_manifest is not None:
        raise ValueError("source_trace and trace_manifest are mutually exclusive")
    if source_trace is None and trace_manifest is None:
        raise ValueError("simulate requires source_trace or trace_manifest")
    if mode not in {"local_model", "cloud_model"}:
        raise ValueError(f"Unsupported simulate mode: {mode}")

    if trace_manifest is not None:
        trace_inputs = _load_trace_manifest(
            trace_manifest,
            default_task_source=task_source.resolve(),
        )
    else:
        assert source_trace is not None
        trace_inputs = [(source_trace, task_source, None)]

    loaded_sessions = [
        _load_trace_session(source_path, task_path, docker_image_override=img)
        for source_path, task_path, img in trace_inputs
    ]
    _validate_loaded_sessions(
        loaded_sessions,
        mode=mode,
        replay_speed=replay_speed,
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

    try:
        for loaded in loaded_sessions:
            if _is_host_mode(loaded):
                prepared_sessions.append(await _prepare_host_session(loaded))
                continue
            if container_executable is None:
                raise ValueError(
                    "container_executable is required for container-mode traces"
                )
            prepared_sessions.append(
                await _prepare_container_session(
                    loaded,
                    container_executable=container_executable,
                    network_mode=network_mode,
                )
            )

        for prepared in prepared_sessions:
            instance_dir = output_path / prepared.loaded.agent_id
            attempt_n = _next_attempt_number(instance_dir)
            task_dir = instance_dir / f"attempt_{attempt_n}"
            task_dir.mkdir(parents=True, exist_ok=True)
            prepared.task_output_dir = task_dir
            if prepared.container is None:
                continue
            sampler = ContainerStatsSampler(
                container_id=prepared.container.container_id,
                interval_s=1.0,
                executable=prepared.container.container_executable,
            )
            sampler.start()
            prepared.sampler = sampler

        run_id = _build_run_id(mode=mode, model=model)
        trace_logger = TraceLogger(output_path, run_id)
        _log_trace_metadata(
            trace_logger=trace_logger,
            mode=mode,
            sessions=loaded_sessions,
            replay_speed=replay_speed,
            source_trace=source_trace,
            trace_manifest=trace_manifest,
            api_base=api_base,
            model=model,
            network_mode=network_mode,
        )

        if mode == "local_model":
            assert trace_logger is not None
            assert api_base is not None
            assert api_key is not None
            assert model is not None
            # Compute gpu_output_path from the single session's attempt dir
            gpu_output_path: Path | None = None
            if gpu_baseline is not None and vllm_pid is not None and metrics_url:
                task_dir = prepared_sessions[0].task_output_dir
                if task_dir is not None:
                    gpu_output_path = task_dir / "gpu_resources.json"
            await _run_local_model_simulation(
                prepared_sessions[0],
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
        else:
            assert trace_logger is not None
            offsets = build_arrival_offsets(
                len(prepared_sessions),
                arrival_mode=arrival_mode,
                arrival_rate_per_s=arrival_rate_per_s,
                arrival_seed=arrival_seed,
            )
            await _run_cloud_model_replay(
                prepared_sessions,
                trace_logger=trace_logger,
                replay_speed=replay_speed,
                command_timeout_s=command_timeout_s,
                warmup_skip_iterations=warmup_skip_iterations,
                arrival_offsets=offsets,
            )
    finally:
        if trace_logger is not None:
            trace_logger.close()
            _split_trace_by_agent(trace_logger.path, prepared_sessions)
        for prepared in prepared_sessions:
            if prepared.sampler is not None:
                samples = prepared.sampler.stop()
                if prepared.task_output_dir is not None:
                    summary = summarize_samples(samples)
                    attempt_layout.write_resources_json(
                        prepared.task_output_dir, samples, summary,
                    )
                    logger.info(
                        "Wrote %d resource samples → %s",
                        len(samples),
                        prepared.task_output_dir / "resources.json",
                    )
            elif prepared.task_output_dir is not None:
                # Host-mode replay has no container sampler; emit a canonical
                # empty resources.json so downstream consumers can rely on
                # the file always existing.
                attempt_layout.write_resources_json(
                    prepared.task_output_dir,
                    samples=[],
                    summary=summarize_samples([]),
                )
            ctr = prepared.container
            if ctr is None:
                continue
            if ctr.agent is not None:
                await ctr.agent.stop()
            await asyncio.to_thread(
                stop_task_container,
                ctr.container_id,
                executable=ctr.container_executable,
            )

    trace_file = output_path / f"{run_id}.jsonl"
    logger.info("Simulate complete [%s] -> %s", mode, trace_file)
    return trace_file
