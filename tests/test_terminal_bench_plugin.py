from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from agents.benchmarks import REGISTRY, get_benchmark_class
from agents.benchmarks.base import BenchmarkConfig


@pytest.fixture(autouse=True)
def _mock_terminal_bench_package(monkeypatch) -> None:
    monkeypatch.setattr(
        "agents.benchmarks.terminal_bench.importlib.util.find_spec",
        lambda name: object(),
    )


@pytest.fixture
def tb_stub(monkeypatch, tmp_path: Path):
    task_root = tmp_path / "dataset"
    task_dir = task_root / "hello-world"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text(
        "instruction: fix it\ndifficulty: easy\n", encoding="utf-8"
    )

    terminal_bench_pkg = types.ModuleType("terminal_bench")
    dataset_pkg = types.ModuleType("terminal_bench.dataset")
    dataset_mod = types.ModuleType("terminal_bench.dataset.dataset")
    handlers_pkg = types.ModuleType("terminal_bench.handlers")
    handler_mod = types.ModuleType("terminal_bench.handlers.trial_handler")

    class Dataset:
        def __init__(
            self,
            *,
            path=None,
            name=None,
            version=None,
            registry_url=None,
            local_registry_path=None,
        ):
            self.config = types.SimpleNamespace(path=path)
            self._path = Path(path) if path else task_root
            self._tasks = [task_dir]

        def __iter__(self):
            return iter(self._tasks)

    class Task:
        @classmethod
        def from_yaml(cls, path: Path):
            return types.SimpleNamespace(
                instruction="fix it",
                max_agent_timeout_sec=360.0,
                max_test_timeout_sec=60.0,
            )

    class TaskPaths:
        def __init__(self, input_path: Path):
            self.task_config_path = input_path / "task.yaml"

    dataset_mod.Dataset = Dataset
    handler_mod.Task = Task
    handler_mod.TaskPaths = TaskPaths

    monkeypatch.setitem(sys.modules, "terminal_bench", terminal_bench_pkg)
    monkeypatch.setitem(sys.modules, "terminal_bench.dataset", dataset_pkg)
    monkeypatch.setitem(sys.modules, "terminal_bench.dataset.dataset", dataset_mod)
    monkeypatch.setitem(sys.modules, "terminal_bench.handlers", handlers_pkg)
    monkeypatch.setitem(
        sys.modules, "terminal_bench.handlers.trial_handler", handler_mod
    )
    return task_root, task_dir


def _make_config(**extra_overrides) -> BenchmarkConfig:
    extras = {
        "task_source_kind": "terminal_bench_registry",
        "dataset_name": "terminal-bench-core",
        "dataset_version": "head",
        **extra_overrides,
    }
    return BenchmarkConfig(
        slug="terminal-bench",
        display_name="Terminal-Bench",
        trace_root=Path("traces/terminal-bench"),
        default_max_iterations=100,
        selection_n=32,
        selection_seed=42,
        extras=extras,
    )


def test_terminal_bench_registered() -> None:
    assert "terminal-bench" in REGISTRY
    plugin_cls = get_benchmark_class("terminal-bench")
    assert plugin_cls.__name__ == "TerminalBenchBenchmark"


def test_terminal_bench_config_requires_dataset_locator() -> None:
    with pytest.raises(
        ValueError,
        match="dataset_path or both extras.dataset_name and extras.dataset_version",
    ):
        get_benchmark_class("terminal-bench")(
            BenchmarkConfig(
                slug="terminal-bench",
                display_name="Terminal-Bench",
                trace_root=Path("traces/terminal-bench"),
                default_max_iterations=100,
                selection_n=32,
                selection_seed=42,
                extras={"task_source_kind": "terminal_bench_registry"},
            )
        )


def test_terminal_bench_runtime_mode_and_scaffold_gating() -> None:
    plugin = get_benchmark_class("terminal-bench")(_make_config())
    assert plugin.runtime_mode_for("openclaw") == "host_controller"
    with pytest.raises(NotImplementedError, match="openclaw"):
        plugin.validate_scaffold_support("unsupported")
    assert plugin.execution_environment == "container"


def test_terminal_bench_normalize_task_preserves_non_swe_shape(tb_stub) -> None:
    task_root, task_dir = tb_stub
    plugin = get_benchmark_class("terminal-bench")(
        _make_config(
            dataset_path=str(task_root), task_source_kind="terminal_bench_local"
        )
    )

    normalized = plugin.normalize_task(
        {"task_path": str(task_dir), "dataset_root": str(task_root)}
    )

    assert normalized["instance_id"] == "hello-world"
    assert normalized["problem_statement"] == "fix it"
    assert normalized["max_agent_timeout_sec"] == 360.0
    assert normalized["max_test_timeout_sec"] == 60.0
    assert normalized["dataset_root"] == str(task_root)
    assert normalized["task_source_path"] == str(task_dir)
    assert normalized["image_name"] is None
    assert normalized["docker_image"] is None
    assert normalized["repo"] is None


def test_terminal_bench_build_runner_preserves_mcp_config() -> None:
    plugin = get_benchmark_class("terminal-bench")(_make_config())
    runner = plugin.build_runner(
        scaffold="openclaw",
        provider=types.SimpleNamespace(
            api_base="https://openrouter.ai/api/v1",
            api_key="test-key",
        ),
        workspace_base=Path("workspace"),
        max_iterations=100,
        context_window_tokens=256_000,
        model="z-ai/glm-5.1",
        provider_name="openrouter",
        env_key="OPENROUTER_API_KEY",
        api_base="https://openrouter.ai/api/v1",
        api_key="test-key",
        mcp_config="configs/mcp/context7.yaml",
    )
    assert runner.mcp_config == str(Path("configs/mcp/context7.yaml").resolve())
