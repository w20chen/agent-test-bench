from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from trace_collect.cli import _run_simulate, parse_simulate_args
from trace_collect.simulator import (
    LoadedTraceSession,
    PreparedTraceSession,
    SimulateError,
    WorkerTraceInput,
    _ensure_unique_agent_ids,
    _partition_sessions_and_offsets,
    _prepare_container_session,
    _resolve_prep_concurrency,
    _teardown_one_worker,
    _wait_for_global_replay_start,
    _worker_run_cloud_model,
    _worker_trace_input,
    simulate,
)


def test_prep_concurrency_is_system_wide_and_preserves_default() -> None:
    assert _resolve_prep_concurrency(0, 640) == 20
    assert _resolve_prep_concurrency(64, 640) == 64
    assert _resolve_prep_concurrency(64, 2) == 2
    with pytest.raises(ValueError, match="must be >= 0"):
        _resolve_prep_concurrency(-1, 640)


def test_partition_preserves_global_ids_and_offsets_for_640_agents() -> None:
    sessions = [
        SimpleNamespace(
            source_trace=Path("trace.jsonl"),
            task_source=Path("tasks.json"),
            docker_image_override=None,
            agent_id=f"task--a{index}",
        )
        for index in range(640)
    ]
    offsets = [float(index) for index in range(640)]

    partitions = _partition_sessions_and_offsets(sessions, offsets, 320)
    specs = [
        _worker_trace_input(session)
        for chunk, _chunk_offsets in partitions
        for session in chunk
    ]
    reconstructed_offsets = [
        offset
        for _chunk, chunk_offsets in partitions
        for offset in chunk_offsets
    ]

    assert len(partitions) == 320
    assert {len(chunk) for chunk, _offsets in partitions} == {2}
    assert len({spec.agent_id for spec in specs}) == 640
    assert reconstructed_offsets == offsets


def test_unique_agent_ids_do_not_collide_with_existing_suffixes() -> None:
    sessions = [
        SimpleNamespace(agent_id="task"),
        SimpleNamespace(agent_id="task"),
        SimpleNamespace(agent_id="task--a1"),
        SimpleNamespace(agent_id="task"),
    ]

    _ensure_unique_agent_ids(sessions)

    ids = [session.agent_id for session in sessions]
    assert len(ids) == len(set(ids))


def test_global_start_waits_for_every_participant() -> None:
    async def run() -> tuple[list[float], threading.Event]:
        barrier = threading.Barrier(2)
        start_event = threading.Event()
        start_wall_time = SimpleNamespace(value=0.0)
        worker_entered = threading.Event()

        async def worker() -> float:
            worker_entered.set()
            return await _wait_for_global_replay_start(
                barrier,
                start_event,
                start_wall_time,
                coordinator=False,
            )

        worker_task = asyncio.create_task(worker())
        await asyncio.to_thread(worker_entered.wait)
        await asyncio.sleep(0)
        assert not worker_task.done()

        coordinator_zero = await _wait_for_global_replay_start(
            barrier,
            start_event,
            start_wall_time,
            coordinator=True,
        )
        worker_zero = await worker_task
        return [coordinator_zero, worker_zero], start_event

    zeros, start_event = asyncio.run(run())
    assert start_event.is_set()
    assert abs(zeros[0] - zeros[1]) < 0.05


def test_global_start_abort_is_explicit() -> None:
    barrier = threading.Barrier(2)
    barrier.abort()
    with pytest.raises(SimulateError, match="aborted"):
        asyncio.run(
            _wait_for_global_replay_start(
                barrier,
                threading.Event(),
                SimpleNamespace(value=0.0),
                coordinator=False,
            )
        )


def test_cli_rejects_negative_prep_concurrency(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = parse_simulate_args(
        [
            "--mode",
            "cloud_model",
            "--source-trace",
            "trace.jsonl",
            "--prep-concurrency",
            "-1",
        ]
    )
    with pytest.raises(SystemExit, match="2"):
        _run_simulate(args)
    assert "--prep-concurrency must be >= 0" in capsys.readouterr().err


def test_simulate_api_rejects_invalid_worker_controls(tmp_path: Path) -> None:
    common = {
        "source_trace": tmp_path / "trace.jsonl",
        "task_source": tmp_path / "tasks.json",
        "output_dir": tmp_path / "out",
        "mode": "cloud_model",
    }
    with pytest.raises(ValueError, match="workers must be >= 1"):
        asyncio.run(simulate(**common, workers=0))
    with pytest.raises(ValueError, match="prep_concurrency must be >= 0"):
        asyncio.run(simulate(**common, prep_concurrency=-1))


@pytest.mark.parametrize(
    ("tool_profiling", "tool_profiling_tools"),
    [
        ("off", []),
        ("ksys", ["exec-pytest"]),
        ("vtune", ["exec-pytest"]),
    ],
)
def test_worker_cloud_model_passes_tool_profiling_to_prepare(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tool_profiling: str,
    tool_profiling_tools: list[str],
) -> None:
    loaded = LoadedTraceSession(
        source_trace=tmp_path / "trace.jsonl",
        task_source=tmp_path / "tasks.json",
        agent_id="task",
        scaffold="openclaw",
        metadata=None,
        summary=None,
        task={"image_name": "example/image:latest"},
        actions=[],
        iterations={},
    )
    seen_prepare_kwargs: list[dict[str, object]] = []

    monkeypatch.setattr(
        "trace_collect.simulator._load_trace_session",
        lambda *_args, **_kwargs: loaded,
    )
    monkeypatch.setattr(
        "trace_collect.simulator._validate_loaded_sessions",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "trace_collect.simulator._is_host_mode",
        lambda _session: False,
    )

    async def fake_prepare_container_session(
        loaded_session: LoadedTraceSession,
        **kwargs: object,
    ) -> PreparedTraceSession:
        seen_prepare_kwargs.append(kwargs)
        return PreparedTraceSession(loaded=loaded_session)

    async def fake_run_cloud_model_replay(*_args: object, **_kwargs: object) -> None:
        return None

    async def fake_teardown_one_worker(
        *_args: object,
        **_kwargs: object,
    ) -> None:
        return None

    monkeypatch.setattr(
        "trace_collect.simulator._prepare_container_session",
        fake_prepare_container_session,
    )
    monkeypatch.setattr(
        "trace_collect.simulator._run_cloud_model_replay",
        fake_run_cloud_model_replay,
    )
    monkeypatch.setattr(
        "trace_collect.simulator._teardown_one_worker",
        fake_teardown_one_worker,
    )

    start_event = threading.Event()
    start_event.set()
    result = _worker_run_cloud_model(
        1,
        [
            WorkerTraceInput(
                source_trace=str(loaded.source_trace),
                task_source=str(loaded.task_source),
                docker_image_override=None,
                agent_id=loaded.agent_id,
            )
        ],
        arrival_offsets=[0.0],
        output_dir=str(tmp_path / "out"),
        container_executable="docker",
        network_mode="host",
        replay_speed=1.0,
        command_timeout_s=120.0,
        warmup_skip_iterations=0,
        cpu_limit=None,
        prep_semaphore=None,
        replay_start_barrier=threading.Barrier(1),
        replay_start_event=start_event,
        replay_start_wall_time=SimpleNamespace(value=time.time()),
        tool_profiling=tool_profiling,
        tool_profiling_tools=tool_profiling_tools,
    )

    assert result["success"] is True
    assert seen_prepare_kwargs == [
        {
            "container_executable": "docker",
            "network_mode": "host",
            "tool_profiling": tool_profiling,
            "tool_profiling_tools": tool_profiling_tools,
            "output_dir": tmp_path / "out" / "task" / "attempt_1",
        }
    ]


def test_bootstrap_failure_stops_started_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped: list[tuple[str, str]] = []
    loaded = SimpleNamespace(
        agent_id="task",
        task={"image_name": "example/image:latest"},
        docker_image_override=None,
    )

    monkeypatch.setattr(
        "trace_collect.simulator.ensure_fixed_image",
        lambda image, *, container_executable: ("fixed-image", 0.0),
    )
    monkeypatch.setattr(
        "trace_collect.simulator.start_task_container",
        lambda *args, **kwargs: "container-id",
    )
    monkeypatch.setattr(
        "trace_collect.simulator.stop_task_container",
        lambda container_id, *, executable: stopped.append(
            (container_id, executable)
        ),
    )

    def fail_bootstrap(**_kwargs: object) -> None:
        raise RuntimeError("bootstrap failed")

    monkeypatch.setattr(
        "trace_collect.runtime.task_container.bootstrap_task_container_python",
        fail_bootstrap,
    )

    with pytest.raises(RuntimeError, match="bootstrap failed"):
        asyncio.run(
            _prepare_container_session(
                loaded,
                container_executable="docker",
            )
        )
    assert stopped == [("container-id", "docker")]


def test_teardown_stops_container_after_sampler_and_agent_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped: list[str] = []

    class FailingSampler:
        def stop(self) -> list[object]:
            raise ValueError("sampler failed")

    class FailingAgent:
        async def stop(self) -> None:
            raise RuntimeError("agent failed")

    prepared = SimpleNamespace(
        loaded=SimpleNamespace(agent_id="task"),
        sampler=FailingSampler(),
        container=SimpleNamespace(
            agent=FailingAgent(),
            container_id="container-id",
            container_executable="docker",
        ),
        host_agent=None,
        task_output_dir=None,
        _resources_written=False,
    )
    policy = SimpleNamespace(
        resource_enabled=True,
        to_dict=lambda: {},
    )
    monkeypatch.setattr(
        "trace_collect.simulator.stop_task_container",
        lambda container_id, *, executable: stopped.append(container_id),
    )

    with pytest.raises(ValueError, match="sampler failed"):
        asyncio.run(_teardown_one_worker(prepared, policy))

    assert stopped == ["container-id"]
    assert prepared.container is None
