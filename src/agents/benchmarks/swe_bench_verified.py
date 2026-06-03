"""SWE-bench Verified benchmark plugin.

Implements the :class:`~agents.benchmarks.base.Benchmark` protocol for the
``princeton-nlp/SWE-bench_Verified`` dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from agents.benchmarks._swebench_selection import (
    select_tool_intensive as _select_tool_intensive,
)
from agents.benchmarks.base import Benchmark

# Repo-level selection constants.

#: Repos known to have heavy test suites (large codebase, slow pytest).
CLASS_LEVEL_HEAVY_REPOS: frozenset[str] = frozenset(
    {
        "django/django",
        "sympy/sympy",
        "scikit-learn/scikit-learn",
        "matplotlib/matplotlib",
        "sphinx-doc/sphinx",
        "pytest-dev/pytest",
    }
)

#: Target allocation per repo for a 32-task selection.
CLASS_LEVEL_REPO_QUOTAS: dict[str, int] = {
    "django/django": 10,
    "sympy/sympy": 6,
    "scikit-learn/scikit-learn": 5,
    "matplotlib/matplotlib": 4,
    # Remaining 7 slots filled from other repos
}

class SWEBenchVerified(Benchmark):
    """Benchmark plugin for SWE-bench Verified.

    Dataset: ``princeton-nlp/SWE-bench_Verified`` (HuggingFace).
    """

    slug: ClassVar[str] = "swe-bench-verified"
    SUPPORTED_SCAFFOLDS: ClassVar[set[str]] = {"openclaw"}

    # Abstract method implementations

    def load_tasks(self) -> list[dict[str, Any]]:
        """Load all tasks from HuggingFace and normalize each row.

        Requires the ``datasets`` package (``pip install datasets``).
        """
        from datasets import load_dataset  # type: ignore[import]

        ds = load_dataset(self.config.harness_dataset, split=self.config.harness_split)
        tasks: list[dict[str, Any]] = []
        for row in ds:
            tasks.append(self.normalize_task(dict(row)))
        return tasks

    def normalize_task(self, raw: dict[str, Any]) -> dict[str, Any]:
        task = dict(raw)
        task["test_cmd"] = self.derive_test_cmd(task)
        task["image_name"] = self.image_name_for(task)
        return task

    @staticmethod
    def _default_image_name(instance_id: str) -> str:
        docker_compatible_id = instance_id.replace("__", "_1776_")
        return (
            f"docker.io/swebench/sweb.eval.x86_64.{docker_compatible_id}:latest"
        ).lower()

    def image_name_for(self, task: dict[str, Any]) -> str | None:
        image_name = task.get("image_name") or task.get("docker_image")
        if image_name:
            return str(image_name)
        instance_id = task.get("instance_id")
        if not instance_id:
            return None
        return self._default_image_name(str(instance_id))

    # Override: task selection uses SWE-bench-specific repo quotas

    def select_subset(
        self,
        tasks: list[dict[str, Any]],
        n: int | None = None,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Select *n* tool-intensive tasks using repo-quota stratification.

        Selection strategy:
        1. Prioritise large repos whose test suites are naturally slow.
        2. Within each repo, rank by FAIL_TO_PASS test count (more = longer).
        3. Exclude trivial tasks (empty FAIL_TO_PASS or very short statement).
        4. Stratified sampling ensures repo diversity.
        """
        effective_n = n if n is not None else self.config.selection_n
        effective_seed = seed if seed is not None else self.config.selection_seed
        return _select_tool_intensive(
            tasks,
            repo_quotas=CLASS_LEVEL_REPO_QUOTAS,
            n=effective_n,
            seed=effective_seed,
        )

    # Override: build a SWE-bench runner for the given scaffold

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
        """Return a :class:`~agents.openclaw.eval.runner.SWEBenchRunner`.

        Reuses the shared SWE runner for repo-backed patch tasks.
        """
        if scaffold != "openclaw":
            raise NotImplementedError(
                f"SWE-bench Verified does not support scaffold={scaffold!r}; "
                f"use scaffold='openclaw'."
            )
        from agents.openclaw.eval.runner import SWEBenchRunner

        return SWEBenchRunner(
            provider=provider,
            workspace_base=workspace_base,
            benchmark_slug=self.config.slug,
            max_iterations=max_iterations,
            context_window_tokens=context_window_tokens,
            model=model,
            **kwargs,
        )

    def runtime_mode_for(self, scaffold: str) -> str:
        if scaffold == "openclaw":
            return "task_container_agent"
        raise NotImplementedError(
            f"SWE-bench Verified does not support scaffold={scaffold!r}"
        )
