"""Terminal-Bench benchmark plugin."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, ClassVar

from agents.benchmarks.base import Benchmark
from agents.terminal_bench.runner import TerminalBenchRunner


class TerminalBenchBenchmark(Benchmark):
    slug: ClassVar[str] = "terminal-bench"
    SUPPORTED_SCAFFOLDS: ClassVar[set[str]] = {"openclaw"}

    def validate_config(self) -> None:
        if sys.version_info < (3, 12):
            raise ValueError("terminal-bench benchmark requires Python 3.12+")
        if importlib.util.find_spec("terminal_bench") is None:
            raise ValueError(
                "terminal-bench benchmark requires the `terminal-bench` Python package. "
                "Install project dependencies (including terminal-bench) before running this benchmark."
            )
        extras = self.config.extras
        has_dataset_path = bool(extras.get("dataset_path"))
        has_named_dataset = bool(extras.get("dataset_name")) and bool(
            extras.get("dataset_version")
        )
        if not has_dataset_path and not has_named_dataset:
            raise ValueError(
                "terminal-bench benchmark requires either extras.dataset_path or "
                "both extras.dataset_name and extras.dataset_version"
            )
        if "task_source_kind" not in extras:
            raise ValueError(
                "terminal-bench benchmark requires extras.task_source_kind"
            )

    def validate_scaffold_support(self, scaffold: str) -> None:
        if scaffold not in self.SUPPORTED_SCAFFOLDS:
            raise NotImplementedError(
                "terminal-bench phase 1 supports scaffold='openclaw' only"
            )

    def runtime_mode_for(self, scaffold: str) -> str:
        self.validate_scaffold_support(scaffold)
        return "host_controller"

    @property
    def execution_environment(self) -> str:
        return "container"

    def load_tasks(self) -> list[dict[str, Any]]:
        dataset_root, task_paths = self._load_dataset_paths()
        return [
            self.normalize_task(
                {
                    "task_path": str(task_path),
                    "dataset_root": str(dataset_root),
                }
            )
            for task_path in task_paths
        ]

    def normalize_task(self, raw: dict[str, Any]) -> dict[str, Any]:
        from terminal_bench.handlers.trial_handler import Task, TaskPaths

        task_path = Path(raw["task_path"]).resolve()
        task_paths = TaskPaths(task_path)
        task = Task.from_yaml(task_paths.task_config_path)
        dataset_root = Path(raw["dataset_root"]).resolve()
        extras = self.config.extras
        return {
            "instance_id": task_path.name,
            "task_id": task_path.name,
            "dataset_root": str(dataset_root),
            "problem_statement": task.instruction,
            "max_agent_timeout_sec": task.max_agent_timeout_sec,
            "max_test_timeout_sec": task.max_test_timeout_sec,
            "task_source_kind": extras["task_source_kind"],
            "task_source_id": task_path.name,
            "task_source_path": str(task_path),
            "tb_dataset": extras.get("dataset_name"),
            "tb_version": extras.get("dataset_version"),
            "tb_registry_source": extras.get("registry_url")
            or extras.get("local_registry_path"),
            "repo": None,
            "image_name": None,
            "docker_image": None,
        }

    def build_runner(
        self,
        *,
        scaffold: str,
        provider: Any,
        workspace_base: Path,
        max_iterations: int,
        context_window_tokens: int,
        model: str,
        **kwargs: Any,
    ) -> Any:
        self.validate_scaffold_support(scaffold)
        return TerminalBenchRunner(
            provider_name=kwargs.get("provider_name"),
            env_key=kwargs.get("env_key"),
            api_base=kwargs.get("api_base")
            or getattr(provider, "api_base", None)
            or "",
            api_key=kwargs.get("api_key") or getattr(provider, "api_key", None) or "",
            model=model,
            workspace_base=workspace_base,
            max_iterations=max_iterations,
            context_window_tokens=context_window_tokens,
            benchmark_slug=self.config.slug,
            benchmark_extras=self.config.extras,
            mcp_config=kwargs.get("mcp_config"),
            generation_config=kwargs.get("generation_config") or {},
        )

    def _load_dataset_paths(self) -> tuple[Path, list[Path]]:
        from terminal_bench.dataset.dataset import Dataset

        extras = self.config.extras
        if extras.get("dataset_path"):
            dataset_root = Path(str(extras["dataset_path"])).expanduser().resolve()
            dataset = Dataset(path=dataset_root)
            return dataset_root, [Path(p).resolve() for p in dataset]

        dataset = Dataset(
            name=str(extras["dataset_name"]),
            version=str(extras["dataset_version"]),
            registry_url=extras.get("registry_url"),
            local_registry_path=(
                Path(str(extras["local_registry_path"])).expanduser().resolve()
                if extras.get("local_registry_path")
                else None
            ),
        )
        dataset_root = Path(dataset.config.path or dataset._path).resolve()
        return dataset_root, [Path(p).resolve() for p in dataset]
