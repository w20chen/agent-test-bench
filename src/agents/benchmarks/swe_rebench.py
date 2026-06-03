"""SWE-rebench benchmark plugin.

This plugin absorbs the benchmark-specific schema quirks that differ from
SWE-bench Verified: native test-id lists, explicit docker image URIs, and the
optional opt-in ``exclude_lite`` filter. Keeping that logic here prevents the
rest of the harness from depending on dataset-specific branches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from agents.benchmarks.base import Benchmark

class SWERebenchBenchmark(Benchmark):
    """Benchmark plugin for ``nebius/SWE-rebench`` (filtered or test split).

    Dataset schema is a superset of SWE-Bench Verified; see the module
    docstring for the three schema quirks this plugin absorbs.
    """

    slug: ClassVar[str] = "swe-rebench"
    SUPPORTED_SCAFFOLDS: ClassVar[set[str]] = {"openclaw"}

    # Abstract method implementations

    def load_tasks(self) -> list[dict[str, Any]]:
        """Load all rows from ``nebius/SWE-rebench`` and normalize each.

        Requires the ``datasets`` package (``pip install datasets``).
        """
        from datasets import load_dataset  # type: ignore[import]

        ds = load_dataset(self.config.harness_dataset, split=self.config.harness_split)
        tasks: list[dict[str, Any]] = []
        for row in ds:
            tasks.append(self.normalize_task(dict(row)))
        return tasks

    def normalize_task(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Normalize a SWE-rebench row and preserve its dataset-specific fields."""
        task = dict(raw)

        # Quirk 2: pin explicit docker image so the harness uses the pre-built
        # swerebench/sweb.eval.* image instead of deriving one.
        docker_image = raw.get("docker_image")
        if docker_image:
            task["image_name"] = docker_image

        # Derive test_cmd. The base helper handles native lists directly;
        # no conversion needed to avoid a lossy round-trip.
        task["test_cmd"] = self.derive_test_cmd(task)
        return task

    # Override: opt-in ``meta.is_lite`` filter via YAML knob

    def select_subset(
        self,
        tasks: list[dict[str, Any]],
        n: int | None = None,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return a subset of tasks, honoring the ``exclude_lite`` config knob.

        When ``self.config.exclude_lite`` is ``True``, tasks whose
        ``meta.is_lite`` is truthy are dropped from the candidate pool
        **before** stratified selection runs. The default is ``False`` —
        we do not silently exclude "lite" tasks from research runs.
        """
        if self.config.exclude_lite:
            pool = [
                t
                for t in tasks
                if not (t.get("meta") or {}).get("is_lite", False)
            ]
        else:
            pool = list(tasks)
        return super().select_subset(pool, n=n, seed=seed)

    # Override: reuse the SWEBenchRunner for swe_patch tasks

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
                f"SWE-rebench does not support scaffold={scaffold!r}; "
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
        if scaffold != "openclaw":
            raise NotImplementedError(
                f"SWE-rebench does not support scaffold={scaffold!r}"
            )
        return "task_container_agent"
