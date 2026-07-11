"""Tests for benchmark-owned runtime selection in collector orchestration."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace

import pytest

from trace_collect.attempt_pipeline import AttemptResult
from trace_collect.collector import (
    _cleanup_task_images,
    _container_ids_for_attempt,
    _ensure_task_source_ready,
    _low_disk_locations,
    _remove_containers_for_attempt,
    _run_scaffold_tasks,
    _select_tasks,
)


@pytest.fixture(autouse=True)
def _mock_fixed_image(monkeypatch) -> None:
    monkeypatch.setattr(
        "trace_collect.attempt_pipeline.ensure_fixed_image",
        lambda source_image, *, container_executable: ((source_image or ""), 0.0),
    )


def _write_trace(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"type":"trace_metadata","scaffold":"openclaw","trace_format_version":5}\n',
        encoding="utf-8",
    )


def test_run_scaffold_tasks_uses_benchmark_prompt_default_and_runtime_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trace_path = tmp_path / "trace-source" / "trace.jsonl"
    _write_trace(trace_path)
    seen: dict[str, str] = {}

    monkeypatch.setattr(
        "trace_collect.collector.ensure_source_image",
        lambda source_image, *, container_executable: None,
    )
    monkeypatch.setattr(
        "trace_collect.collector.remove_image",
        lambda image, *, container_executable: False,
    )
    monkeypatch.setattr(
        "trace_collect.collector.drop_cached_fixed_image",
        lambda source_image: None,
    )
    monkeypatch.setattr(
        "trace_collect.collector.prune_dangling_images",
        lambda *, container_executable: None,
    )

    benchmark = SimpleNamespace(
        config=SimpleNamespace(
            slug="swe-rebench",
            harness_split="filtered",
            trace_root=tmp_path / "traces",
            default_prompt_template="cc_aligned",
        ),
        runtime_mode_for=lambda scaffold: "task_container_agent",
        image_name_for=lambda task: task.get("image_name"),
    )

    def make_inner(task: dict):
        async def inner(ctx) -> AttemptResult:
            seen["prompt_template"] = ctx.prompt_template
            seen["agent_runtime_mode"] = ctx.agent_runtime_mode
            return AttemptResult(
                success=True,
                exit_status="ok",
                trace_path=trace_path,
            )

        return inner

    asyncio.run(
        _run_scaffold_tasks(
            benchmark=benchmark,
            tasks=[{"instance_id": "encode__httpx-2701", "image_name": "img"}],
            run_dir=tmp_path / "run",
            model="qwen-plus-latest",
            scaffold="openclaw",
            container_executable="docker",
            prompt_template=None,
            min_free_disk_gb=0.001,
            inner_factory=make_inner,
        )
    )

    assert seen == {
        "prompt_template": "cc_aligned",
        "agent_runtime_mode": "task_container_agent",
    }


@pytest.mark.parametrize(
    ("existing_attempt_dirs", "expected_attempt_dir"),
    [([], "attempt_1"), (["attempt_1"], "attempt_2")],
)
def test_run_scaffold_tasks_allocates_next_attempt_dir(
    tmp_path: Path,
    monkeypatch,
    existing_attempt_dirs: list[str],
    expected_attempt_dir: str,
) -> None:
    trace_path = tmp_path / "trace-source" / "trace.jsonl"
    _write_trace(trace_path)
    instance_dir = tmp_path / "run" / "encode__httpx-2701"
    for attempt_dir in existing_attempt_dirs:
        (instance_dir / attempt_dir).mkdir(parents=True, exist_ok=True)
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        "trace_collect.collector.ensure_source_image",
        lambda source_image, *, container_executable: None,
    )
    monkeypatch.setattr(
        "trace_collect.collector.remove_image",
        lambda image, *, container_executable: False,
    )
    monkeypatch.setattr(
        "trace_collect.collector.drop_cached_fixed_image",
        lambda source_image: None,
    )
    monkeypatch.setattr(
        "trace_collect.collector.prune_dangling_images",
        lambda *, container_executable: None,
    )

    async def fake_run_attempt(
        ctx,
        *,
        inner,
        min_free_disk_gb,
        container_executable,
        **monitoring_kwargs,
    ) -> AttemptResult:
        seen["attempt"] = ctx.attempt
        seen["attempt_dir_name"] = ctx.attempt_dir.name
        return AttemptResult(success=True, exit_status="ok", trace_path=trace_path)

    monkeypatch.setattr("trace_collect.collector.run_attempt", fake_run_attempt)

    benchmark = SimpleNamespace(
        config=SimpleNamespace(
            slug="swe-rebench",
            harness_split="filtered",
            trace_root=tmp_path / "traces",
            default_prompt_template="cc_aligned",
        ),
        runtime_mode_for=lambda scaffold: "task_container_agent",
        image_name_for=lambda task: task.get("image_name"),
    )

    asyncio.run(
        _run_scaffold_tasks(
            benchmark=benchmark,
            tasks=[{"instance_id": "encode__httpx-2701", "image_name": "img"}],
            run_dir=tmp_path / "run",
            model="qwen-plus-latest",
            scaffold="openclaw",
            container_executable="docker",
            prompt_template=None,
            min_free_disk_gb=0.001,
            inner_factory=lambda task: lambda ctx: None,
        )
    )

    assert seen == {
        "attempt": int(expected_attempt_dir.removeprefix("attempt_")),
        "attempt_dir_name": expected_attempt_dir,
    }


def test_run_scaffold_tasks_uses_max_sparse_attempt_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trace_path = tmp_path / "trace-source" / "trace.jsonl"
    _write_trace(trace_path)
    instance_dir = tmp_path / "run" / "encode__httpx-2701"
    (instance_dir / "attempt_1").mkdir(parents=True)
    (instance_dir / "attempt_3").mkdir()
    (instance_dir / "attempt_bad").mkdir()
    (instance_dir / "not_an_attempt").mkdir()
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        "trace_collect.collector.ensure_source_image",
        lambda source_image, *, container_executable: None,
    )
    monkeypatch.setattr(
        "trace_collect.collector.remove_image",
        lambda image, *, container_executable: False,
    )
    monkeypatch.setattr(
        "trace_collect.collector.drop_cached_fixed_image",
        lambda source_image: None,
    )
    monkeypatch.setattr(
        "trace_collect.collector.prune_dangling_images",
        lambda *, container_executable: None,
    )

    async def fake_run_attempt(
        ctx,
        *,
        inner,
        min_free_disk_gb,
        container_executable,
        **monitoring_kwargs,
    ) -> AttemptResult:
        seen["attempt"] = ctx.attempt
        seen["attempt_dir_name"] = ctx.attempt_dir.name
        return AttemptResult(success=True, exit_status="ok", trace_path=trace_path)

    monkeypatch.setattr("trace_collect.collector.run_attempt", fake_run_attempt)

    benchmark = SimpleNamespace(
        config=SimpleNamespace(
            slug="swe-rebench",
            harness_split="filtered",
            trace_root=tmp_path / "traces",
            default_prompt_template="cc_aligned",
        ),
        runtime_mode_for=lambda scaffold: "task_container_agent",
        image_name_for=lambda task: task.get("image_name"),
    )

    asyncio.run(
        _run_scaffold_tasks(
            benchmark=benchmark,
            tasks=[{"instance_id": "encode__httpx-2701", "image_name": "img"}],
            run_dir=tmp_path / "run",
            model="qwen-plus-latest",
            scaffold="openclaw",
            container_executable="docker",
            prompt_template=None,
            min_free_disk_gb=0.001,
            inner_factory=lambda task: lambda ctx: None,
        )
    )

    assert seen == {"attempt": 4, "attempt_dir_name": "attempt_4"}


def test_run_scaffold_tasks_prompt_override_stays_independent_of_runtime_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trace_path = tmp_path / "trace-source" / "trace.jsonl"
    _write_trace(trace_path)
    seen: dict[str, str] = {}

    monkeypatch.setattr(
        "trace_collect.collector.ensure_source_image",
        lambda source_image, *, container_executable: None,
    )
    monkeypatch.setattr(
        "trace_collect.collector.remove_image",
        lambda image, *, container_executable: False,
    )
    monkeypatch.setattr(
        "trace_collect.collector.drop_cached_fixed_image",
        lambda source_image: None,
    )
    monkeypatch.setattr(
        "trace_collect.collector.prune_dangling_images",
        lambda *, container_executable: None,
    )

    benchmark = SimpleNamespace(
        config=SimpleNamespace(
            slug="swe-rebench",
            harness_split="filtered",
            trace_root=tmp_path / "traces",
            default_prompt_template="cc_aligned",
        ),
        runtime_mode_for=lambda scaffold: "task_container_agent",
        image_name_for=lambda task: task.get("image_name"),
    )

    def make_inner(task: dict):
        async def inner(ctx) -> AttemptResult:
            seen["prompt_template"] = ctx.prompt_template
            seen["agent_runtime_mode"] = ctx.agent_runtime_mode
            return AttemptResult(
                success=True,
                exit_status="ok",
                trace_path=trace_path,
            )

        return inner

    asyncio.run(
        _run_scaffold_tasks(
            benchmark=benchmark,
            tasks=[{"instance_id": "encode__httpx-2701", "image_name": "img"}],
            run_dir=tmp_path / "run",
            model="qwen-plus-latest",
            scaffold="openclaw",
            container_executable="docker",
            prompt_template="default",
            min_free_disk_gb=0.001,
            inner_factory=make_inner,
        )
    )

    assert seen == {
        "prompt_template": "default",
        "agent_runtime_mode": "task_container_agent",
    }


def test_run_scaffold_tasks_uses_benchmark_image_name_for_source_image(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trace_path = tmp_path / "trace-source" / "trace.jsonl"
    _write_trace(trace_path)
    seen: dict[str, str] = {}

    monkeypatch.setattr(
        "trace_collect.collector.ensure_source_image",
        lambda source_image, *, container_executable: seen.setdefault(
            "ensure_source_image", source_image
        ),
    )
    monkeypatch.setattr(
        "trace_collect.collector.remove_image",
        lambda image, *, container_executable: False,
    )
    monkeypatch.setattr(
        "trace_collect.collector.drop_cached_fixed_image",
        lambda source_image: None,
    )
    monkeypatch.setattr(
        "trace_collect.collector.prune_dangling_images",
        lambda *, container_executable: None,
    )

    benchmark = SimpleNamespace(
        config=SimpleNamespace(
            slug="swe-bench-verified",
            harness_split="test",
            trace_root=tmp_path / "traces",
            default_prompt_template="default",
        ),
        runtime_mode_for=lambda scaffold: "host_controller",
        image_name_for=lambda task: (
            "docker.io/swebench/sweb.eval.x86_64.kinto_1776_kinto-http.py-384:latest"
        ),
    )

    def make_inner(task: dict):
        async def inner(ctx) -> AttemptResult:
            seen["ctx_source_image"] = ctx.source_image
            return AttemptResult(
                success=True,
                exit_status="ok",
                trace_path=trace_path,
            )

        return inner

    asyncio.run(
        _run_scaffold_tasks(
            benchmark=benchmark,
            tasks=[{"instance_id": "Kinto__kinto-http.py-384"}],
            run_dir=tmp_path / "run",
            model="qwen-plus-latest",
            scaffold="openclaw",
            container_executable="docker",
            prompt_template=None,
            min_free_disk_gb=0.001,
            inner_factory=make_inner,
        )
    )

    assert seen == {
        "ensure_source_image": (
            "docker.io/swebench/sweb.eval.x86_64.kinto_1776_kinto-http.py-384:latest"
        ),
        "ctx_source_image": (
            "docker.io/swebench/sweb.eval.x86_64.kinto_1776_kinto-http.py-384:latest"
        ),
    }


def test_run_scaffold_tasks_allows_non_image_tasks_and_uses_attempt_success(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trace_path = tmp_path / "trace-source" / "trace.jsonl"
    _write_trace(trace_path)
    seen: list[str | None] = []

    monkeypatch.setattr(
        "trace_collect.collector.ensure_source_image",
        lambda source_image, executable="podman": seen.append(source_image),
    )
    monkeypatch.setattr(
        "trace_collect.collector.remove_image",
        lambda image, executable="podman": False,
    )
    monkeypatch.setattr(
        "trace_collect.collector.drop_cached_fixed_image",
        lambda source_image: None,
    )
    monkeypatch.setattr(
        "trace_collect.collector.prune_dangling_images",
        lambda executable="podman": None,
    )

    benchmark = SimpleNamespace(
        config=SimpleNamespace(
            slug="terminal-bench",
            harness_split=None,
            trace_root=tmp_path / "traces",
            default_prompt_template="default",
        ),
        runtime_mode_for=lambda scaffold: "host_controller",
        image_name_for=lambda task: None,
    )

    def make_inner(task: dict):
        async def inner(ctx) -> AttemptResult:
            return AttemptResult(
                success=True,
                exit_status="completed",
                trace_path=trace_path,
                model_patch="",
            )

        return inner

    run_dir = asyncio.run(
        _run_scaffold_tasks(
            benchmark=benchmark,
            tasks=[{"instance_id": "tb-1"}],
            run_dir=tmp_path / "run",
            model="z-ai/glm-5.1",
            scaffold="openclaw",
            container_executable="docker",
            prompt_template=None,
            min_free_disk_gb=0.001,
            inner_factory=make_inner,
        )
    )

    assert seen == []
    results_jsonl = (run_dir / "results.jsonl").read_text(encoding="utf-8")
    assert '"success": true' in results_jsonl


def test_select_tasks_preserves_explicit_instance_order() -> None:
    tasks = [
        {"instance_id": "mozilla__bleach-259"},
        {"instance_id": "encode__httpx-2701"},
        {"instance_id": "Kinto__kinto-http.py-384"},
    ]

    selected = _select_tasks(
        tasks,
        instance_ids=[
            "encode__httpx-2701",
            "Kinto__kinto-http.py-384",
        ],
        sample=None,
    )

    assert [task["instance_id"] for task in selected] == [
        "encode__httpx-2701",
        "Kinto__kinto-http.py-384",
    ]


def test_cleanup_task_images_keeps_next_source_image(monkeypatch) -> None:
    removed: list[str] = []
    cached: list[str] = []
    pruned: list[str] = []

    monkeypatch.setattr(
        "trace_collect.collector.remove_image",
        lambda image, *, container_executable: removed.append(image) or True,
    )
    monkeypatch.setattr(
        "trace_collect.collector.drop_cached_fixed_image",
        lambda source_image: cached.append(source_image),
    )
    monkeypatch.setattr(
        "trace_collect.collector.prune_dangling_images",
        lambda *, container_executable: pruned.append("pruned"),
    )

    _cleanup_task_images(
        instance_id="encode__httpx-2701",
        source_image="shared-image",
        fixed_image="fixed-shared-image",
        keep_source_image="shared-image",
        container_executable="docker",
    )

    assert removed == ["fixed-shared-image"]
    assert cached == ["shared-image"]
    assert pruned == ["pruned"]


def test_cleanup_task_images_force_cleanup_ignores_keep_images_threshold(
    monkeypatch,
    tmp_path: Path,
) -> None:
    removed: list[str] = []

    monkeypatch.setenv("KEEP_IMAGES_ABOVE_GB", "1")
    monkeypatch.setattr("trace_collect.collector._free_disk_gb", lambda path: 10.0)
    monkeypatch.setattr(
        "trace_collect.collector.remove_image",
        lambda image, *, container_executable: removed.append(image) or True,
    )
    monkeypatch.setattr(
        "trace_collect.collector.drop_cached_fixed_image",
        lambda source_image: None,
    )
    monkeypatch.setattr(
        "trace_collect.collector.prune_dangling_images",
        lambda *, container_executable: None,
    )

    _cleanup_task_images(
        instance_id="task-a",
        source_image="source-image",
        fixed_image="fixed-image",
        keep_source_image=None,
        container_executable="docker",
        run_dir=tmp_path,
        force_cleanup=True,
    )

    assert removed == ["fixed-image", "source-image"]


@pytest.mark.parametrize("container_executable", ["docker", "podman"])
def test_container_ids_for_attempt_filters_by_image_and_labels(
    monkeypatch,
    container_executable: str,
) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="abc123\n\n def456 \n",
            stderr="",
        )

    monkeypatch.setattr("trace_collect.collector.subprocess.run", fake_run)

    ids = _container_ids_for_attempt(
        image="fixed-image",
        container_executable=container_executable,
        labels={
            "trace_collect.attempt": "1",
            "trace_collect.instance_id": "task-a",
        },
    )

    assert calls == [
        [
            container_executable,
            "ps",
            "-a",
            "--filter",
            "ancestor=fixed-image",
            "--filter",
            "label=trace_collect.attempt=1",
            "--filter",
            "label=trace_collect.instance_id=task-a",
            "--format",
            "{{.ID}}",
        ]
    ]
    assert ids == ["abc123", "def456"]


def test_remove_containers_for_attempt_skips_without_labels(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="abc\n", stderr="")

    monkeypatch.setattr("trace_collect.collector.subprocess.run", fake_run)

    removed = _remove_containers_for_attempt(
        instance_id="task-a",
        image="fixed-image",
        container_executable="docker",
        labels=None,
    )

    assert removed is False
    assert calls == []


def test_remove_containers_for_attempt_skips_empty_container_list(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("trace_collect.collector.subprocess.run", fake_run)

    removed = _remove_containers_for_attempt(
        instance_id="task-a",
        image="fixed-image",
        container_executable="docker",
        labels={"trace_collect.attempt_dir": "/runs/task-a/attempt_1"},
    )

    assert removed is False
    assert len(calls) == 1
    assert calls[0][:3] == ["docker", "ps", "-a"]
    assert "label=trace_collect.attempt_dir=/runs/task-a/attempt_1" in calls[0]


def test_remove_containers_for_attempt_removes_labeled_ids(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:3] == ["docker", "ps", "-a"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="abc\ndef\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("trace_collect.collector.subprocess.run", fake_run)

    removed = _remove_containers_for_attempt(
        instance_id="task-a",
        image="fixed-image",
        container_executable="docker",
        labels={"trace_collect.attempt_dir": "/runs/task-a/attempt_1"},
    )

    assert removed is True
    assert calls[-1] == ["docker", "rm", "-f", "abc", "def"]


def test_low_disk_locations_includes_container_storage_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    docker_root = tmp_path / "docker-root"
    docker_root.mkdir()

    monkeypatch.setattr(
        "trace_collect.collector._container_storage_root",
        lambda container_executable: docker_root,
    )
    monkeypatch.setattr(
        "trace_collect.collector._free_disk_gb",
        lambda path: 100.0 if path == tmp_path else 2.0,
    )

    low = _low_disk_locations(
        run_dir=tmp_path,
        container_executable="docker",
        min_free_disk_gb=30.0,
    )

    assert low == [("container_storage", docker_root, 2.0)]


def test_ensure_task_source_ready_falls_back_after_prefetch_failure(
    monkeypatch,
) -> None:
    seen: list[str] = []
    failed_prefetch: Future[None] = Future()
    failed_prefetch.set_exception(RuntimeError("prefetch boom"))

    monkeypatch.setattr(
        "trace_collect.collector.ensure_source_image",
        lambda source_image, *, container_executable: seen.append(source_image),
    )

    _ensure_task_source_ready(
        instance_id="encode__httpx-2701",
        source_image="docker.io/library/img-a",
        prefetched_source_image="docker.io/library/img-a",
        prefetch_future=failed_prefetch,
        container_executable="docker",
    )

    assert seen == ["docker.io/library/img-a"]


def test_run_scaffold_tasks_prefetches_next_image_and_cleans_after_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trace_path = tmp_path / "trace-source" / "trace.jsonl"
    _write_trace(trace_path)
    benchmark = SimpleNamespace(
        config=SimpleNamespace(
            slug="swe-rebench",
            harness_split="filtered",
            trace_root=tmp_path / "traces",
            default_prompt_template="cc_aligned",
        ),
        runtime_mode_for=lambda scaffold: "task_container_agent",
        image_name_for=lambda task: task.get("image_name"),
    )
    events: list[tuple[str, str]] = []
    prefetch_started = threading.Event()
    allow_prefetch_finish = threading.Event()

    def fake_ensure_source_image(image: str, *, container_executable: str) -> None:
        events.append(("ensure_source", image))
        if image == "docker.io/library/img-b":
            prefetch_started.set()
            assert allow_prefetch_finish.wait(timeout=1.0)

    async def fake_run_attempt(
        ctx,
        *,
        inner,
        min_free_disk_gb,
        container_executable,
        **monitoring_kwargs,
    ):
        events.append(("run_start", ctx.instance_id))
        if ctx.instance_id == "task-a":
            assert prefetch_started.wait(timeout=1.0)
            allow_prefetch_finish.set()
        ctx.fixed_image = f"fixed-{ctx.source_image}"
        events.append(("run_end", ctx.instance_id))
        return AttemptResult(
            success=True,
            exit_status="ok",
            trace_path=trace_path,
            model_patch=f"diff --git a/{ctx.instance_id} b/{ctx.instance_id}",
        )

    monkeypatch.setattr(
        "trace_collect.collector.ensure_source_image",
        fake_ensure_source_image,
    )
    monkeypatch.setattr(
        "trace_collect.collector.run_attempt",
        fake_run_attempt,
    )
    monkeypatch.setattr(
        "trace_collect.collector.remove_image",
        lambda image, *, container_executable: (
            events.append(("remove_image", image)) or True
        ),
    )
    monkeypatch.setattr(
        "trace_collect.collector.drop_cached_fixed_image",
        lambda source_image: events.append(("drop_cache", source_image)),
    )
    monkeypatch.setattr(
        "trace_collect.collector.prune_dangling_images",
        lambda *, container_executable: events.append(("prune", "done")),
    )

    asyncio.run(
        _run_scaffold_tasks(
            benchmark=benchmark,
            tasks=[
                {"instance_id": "task-a", "image_name": "img-a"},
                {"instance_id": "task-b", "image_name": "img-b"},
                {"instance_id": "task-c", "image_name": "img-c"},
            ],
            run_dir=tmp_path / "run",
            model="qwen-plus-latest",
            scaffold="openclaw",
            container_executable="docker",
            prompt_template=None,
            min_free_disk_gb=0.001,
            inner_factory=lambda task: lambda ctx: None,
        )
    )

    assert [event for event in events if event[0] == "ensure_source"] == [
        ("ensure_source", "docker.io/library/img-a"),
        ("ensure_source", "docker.io/library/img-b"),
        ("ensure_source", "docker.io/library/img-c"),
    ]
    assert events.index(("ensure_source", "docker.io/library/img-b")) < events.index(
        ("run_end", "task-a")
    )
    assert events.index(("run_end", "task-a")) < events.index(
        ("remove_image", "fixed-docker.io/library/img-a")
    )
    assert events.index(("run_end", "task-b")) < events.index(
        ("remove_image", "fixed-docker.io/library/img-b")
    )
    assert events.index(("run_end", "task-c")) < events.index(
        ("remove_image", "fixed-docker.io/library/img-c")
    )


def test_run_scaffold_tasks_reuses_source_image_for_consecutive_tasks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trace_path = tmp_path / "trace-source" / "trace.jsonl"
    _write_trace(trace_path)
    benchmark = SimpleNamespace(
        config=SimpleNamespace(
            slug="swe-rebench",
            harness_split="filtered",
            trace_root=tmp_path / "traces",
            default_prompt_template="cc_aligned",
        ),
        runtime_mode_for=lambda scaffold: "task_container_agent",
        image_name_for=lambda task: task.get("image_name"),
    )
    events: list[tuple[str, str]] = []

    async def fake_run_attempt(
        ctx,
        *,
        inner,
        min_free_disk_gb,
        container_executable,
        **monitoring_kwargs,
    ):
        ctx.fixed_image = f"fixed-{ctx.instance_id}"
        events.append(("run_end", ctx.instance_id))
        return AttemptResult(
            success=True,
            exit_status="ok",
            trace_path=trace_path,
            model_patch=f"diff --git a/{ctx.instance_id} b/{ctx.instance_id}",
        )

    monkeypatch.setattr(
        "trace_collect.collector.ensure_source_image",
        lambda source_image, *, container_executable: events.append(
            ("ensure_source", source_image)
        ),
    )
    monkeypatch.setattr(
        "trace_collect.collector.run_attempt",
        fake_run_attempt,
    )
    monkeypatch.setattr(
        "trace_collect.collector.remove_image",
        lambda image, *, container_executable: (
            events.append(("remove_image", image)) or True
        ),
    )
    monkeypatch.setattr(
        "trace_collect.collector.drop_cached_fixed_image",
        lambda source_image: events.append(("drop_cache", source_image)),
    )
    monkeypatch.setattr(
        "trace_collect.collector.prune_dangling_images",
        lambda *, container_executable: events.append(("prune", "done")),
    )

    asyncio.run(
        _run_scaffold_tasks(
            benchmark=benchmark,
            tasks=[
                {"instance_id": "task-a", "image_name": "shared-image"},
                {"instance_id": "task-b", "image_name": "shared-image"},
            ],
            run_dir=tmp_path / "run",
            model="qwen-plus-latest",
            scaffold="openclaw",
            container_executable="docker",
            prompt_template=None,
            min_free_disk_gb=0.001,
            inner_factory=lambda task: lambda ctx: None,
        )
    )

    assert [
        event
        for event in events
        if event == ("remove_image", "docker.io/library/shared-image")
    ] == [("remove_image", "docker.io/library/shared-image")]
    assert events.index(("run_end", "task-a")) < events.index(
        ("remove_image", "fixed-task-a")
    )
    assert events.index(("remove_image", "fixed-task-a")) < events.index(
        ("run_end", "task-b")
    )


def test_run_scaffold_tasks_low_disk_does_not_keep_next_source_image(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trace_path = tmp_path / "trace-source" / "trace.jsonl"
    _write_trace(trace_path)
    benchmark = SimpleNamespace(
        config=SimpleNamespace(
            slug="swe-rebench",
            harness_split="filtered",
            trace_root=tmp_path / "traces",
            default_prompt_template="cc_aligned",
        ),
        runtime_mode_for=lambda scaffold: "task_container_agent",
        image_name_for=lambda task: task.get("image_name"),
    )
    events: list[tuple[str, str]] = []

    async def fake_run_attempt(
        ctx,
        *,
        inner,
        min_free_disk_gb,
        container_executable,
        **monitoring_kwargs,
    ):
        ctx.fixed_image = f"fixed-{ctx.instance_id}"
        events.append(("run_end", ctx.instance_id))
        return AttemptResult(
            success=True,
            exit_status="ok",
            trace_path=trace_path,
            model_patch=f"diff --git a/{ctx.instance_id} b/{ctx.instance_id}",
        )

    monkeypatch.setattr(
        "trace_collect.collector.ensure_source_image",
        lambda source_image, *, container_executable: events.append(
            ("ensure_source", source_image)
        ),
    )
    monkeypatch.setattr(
        "trace_collect.collector.run_attempt",
        fake_run_attempt,
    )
    monkeypatch.setattr(
        "trace_collect.collector.remove_image",
        lambda image, *, container_executable: (
            events.append(("remove_image", image)) or True
        ),
    )
    monkeypatch.setattr(
        "trace_collect.collector._remove_containers_for_attempt",
        lambda *, instance_id, image, container_executable, labels: (
            events.append(("remove_containers", image))
            or events.append(("container_label", labels["trace_collect.instance_id"]))
            or True
        ),
    )
    monkeypatch.setattr(
        "trace_collect.collector.drop_cached_fixed_image",
        lambda source_image: events.append(("drop_cache", source_image)),
    )
    monkeypatch.setattr(
        "trace_collect.collector.prune_dangling_images",
        lambda *, container_executable: events.append(("prune", "done")),
    )
    monkeypatch.setattr("trace_collect.collector._free_disk_gb", lambda path: 0.0)

    asyncio.run(
        _run_scaffold_tasks(
            benchmark=benchmark,
            tasks=[
                {"instance_id": "task-a", "image_name": "shared-image"},
                {"instance_id": "task-b", "image_name": "shared-image"},
            ],
            run_dir=tmp_path / "run",
            model="qwen-plus-latest",
            scaffold="openclaw",
            container_executable="docker",
            prompt_template=None,
            min_free_disk_gb=30.0,
            inner_factory=lambda task: lambda ctx: None,
        )
    )

    assert [
        event
        for event in events
        if event == ("remove_image", "docker.io/library/shared-image")
    ] == [
        ("remove_image", "docker.io/library/shared-image"),
        ("remove_image", "docker.io/library/shared-image"),
    ]
    assert ("remove_containers", "fixed-task-a") in events
    assert ("container_label", "task-a") in events
    assert ("remove_containers", "docker.io/library/shared-image") not in events
    assert events.index(("remove_image", "docker.io/library/shared-image")) < (
        events.index(("run_end", "task-b"))
    )


@pytest.mark.parametrize("container_executable", ["docker", "podman"])
def test_run_scaffold_tasks_propagates_container_executable(
    tmp_path: Path,
    monkeypatch,
    container_executable: str,
) -> None:
    trace_path = tmp_path / "trace-source" / "trace.jsonl"
    _write_trace(trace_path)
    seen: list[tuple[str, str]] = []

    benchmark = SimpleNamespace(
        config=SimpleNamespace(
            slug="swe-rebench",
            harness_split="filtered",
            trace_root=tmp_path / "traces",
            default_prompt_template="cc_aligned",
        ),
        runtime_mode_for=lambda scaffold: "task_container_agent",
        image_name_for=lambda task: task.get("image_name"),
    )

    monkeypatch.setattr(
        "trace_collect.collector.ensure_source_image",
        lambda source_image, *, container_executable: seen.append(
            ("ensure_source_image", container_executable)
        ),
    )
    monkeypatch.setattr(
        "trace_collect.collector.remove_image",
        lambda image, *, container_executable: (
            seen.append(("remove_image", container_executable)) or True
        ),
    )
    monkeypatch.setattr(
        "trace_collect.collector.drop_cached_fixed_image",
        lambda source_image: None,
    )
    monkeypatch.setattr(
        "trace_collect.collector.prune_dangling_images",
        lambda *, container_executable: seen.append(
            ("prune_dangling_images", container_executable)
        ),
    )

    async def fake_run_attempt(
        ctx,
        *,
        inner,
        min_free_disk_gb,
        container_executable,
        **monitoring_kwargs,
    ):
        seen.append(("run_attempt", container_executable))
        ctx.fixed_image = f"fixed-{ctx.source_image}"
        return AttemptResult(
            success=True,
            exit_status="ok",
            trace_path=trace_path,
            model_patch="diff --git a/x b/x",
        )

    monkeypatch.setattr("trace_collect.collector.run_attempt", fake_run_attempt)

    asyncio.run(
        _run_scaffold_tasks(
            benchmark=benchmark,
            tasks=[{"instance_id": "task-a", "image_name": "img-a"}],
            run_dir=tmp_path / "run",
            model="qwen-plus-latest",
            scaffold="openclaw",
            container_executable=container_executable,
            prompt_template=None,
            min_free_disk_gb=0.001,
            inner_factory=lambda task: lambda ctx: None,
        )
    )

    assert seen == [
        ("ensure_source_image", container_executable),
        ("run_attempt", container_executable),
        ("remove_image", container_executable),
        ("remove_image", container_executable),
        ("prune_dangling_images", container_executable),
    ]
