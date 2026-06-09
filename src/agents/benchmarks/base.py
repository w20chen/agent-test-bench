"""Base classes and configuration for the benchmark plugin architecture.

New code should instantiate benchmarks via::

    from agents.benchmarks import get_benchmark_class
    cls = get_benchmark_class("swe-bench-verified")
    plugin = cls(config)
"""

from __future__ import annotations

import json
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol

# ---------------------------------------------------------------------------
# datasets>=3.0 backward-compatibility shims.
#
# (1) datasets>=3.0 renamed `List` to `LargeList`, but some HF datasets were
#     created with older versions whose cached dataset_info.json still
#     references the `List` type name.  Register the alias.
# (2) Arrow files encode list-typed columns as ``[Value(...)]`` (which
#     datasets interprets as ``Sequence``), while cached dataset_info.json
#     may encode them as ``LargeList(...)``.  Patch
#     ``Features.reorder_fields_as`` to normalise these representations
#     before the inner ``recursive_reorder`` type comparison runs.
# ---------------------------------------------------------------------------
try:
    import datasets.features.features as _ff  # type: ignore[import]
    from datasets.features.features import LargeList, _FEATURE_TYPES  # type: ignore[import]

    if "List" not in _FEATURE_TYPES:
        _FEATURE_TYPES["List"] = LargeList

    # Features is a dict subclass; we walk it and replace every LargeList
    # leaf with the equivalent [feature] representation.
    def _normalise_largelist(obj: object) -> object:
        if isinstance(obj, dict):
            return {k: _normalise_largelist(v) for k, v in obj.items()}
        if isinstance(obj, LargeList):
            return [obj.feature]
        return obj

    _orig_reorder_fields_as = _ff.Features.reorder_fields_as

    def _patched_reorder_fields_as(
        self: _ff.Features, other: _ff.Features
    ) -> _ff.Features:
        # self came from dataset_info.json (may contain LargeList leaves);
        # normalise to [feature] so it matches the arrow-inferred schema.
        return _orig_reorder_fields_as(_normalise_largelist(self), other)

    _ff.Features.reorder_fields_as = _patched_reorder_fields_as  # type: ignore[assignment]
except ImportError:
    pass  # datasets not installed — no-op


class Runner(Protocol):
    """Static host-mode runner interface used by collector dispatch.

    Container-mode runners such as SWEBenchRunner have their own in-container
    adapter path and are not required to satisfy this Protocol directly.
    """

    async def run_task(
        self,
        task: dict[str, Any],
        *,
        attempt_ctx: Any,
        prompt_template: str,
    ) -> Any:
        """Run one normalized benchmark task."""
        ...

@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark plugin.

    All path fields are stored as :class:`pathlib.Path` objects.
    """

    slug: str
    display_name: str
    trace_root: Path
    default_max_iterations: int
    selection_n: int
    selection_seed: int
    harness_dataset: str | None = None
    harness_split: str | None = None
    data_root: Path | None = None
    repos_root: Path | None = None
    default_prompt_template: str = "default"
    exclude_lite: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "BenchmarkConfig":
        """Load a :class:`BenchmarkConfig` from a YAML file.

        Path fields (``data_root``, ``repos_root``, ``trace_root``) are
        wrapped in :class:`pathlib.Path`.  ``repos_root`` is ``None`` when
        absent or explicitly set to ``null`` in the YAML.
        """
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))

        repos_root_raw = raw.get("repos_root")
        repos_root: Path | None = (
            Path(repos_root_raw) if repos_root_raw is not None else None
        )
        data_root_raw = raw.get("data_root")
        data_root: Path | None = Path(data_root_raw) if data_root_raw is not None else None

        return cls(
            slug=raw["slug"],
            display_name=raw["display_name"],
            harness_dataset=raw.get("harness_dataset"),
            harness_split=raw.get("harness_split"),
            data_root=data_root,
            repos_root=repos_root,
            trace_root=Path(raw["trace_root"]),
            default_max_iterations=int(raw["default_max_iterations"]),
            selection_n=int(raw["selection_n"]),
            selection_seed=int(raw["selection_seed"]),
            default_prompt_template=str(raw.get("default_prompt_template", "default")),
            exclude_lite=bool(raw.get("exclude_lite", False)),
            extras=dict(raw.get("extras", {})),
        )

class Benchmark(ABC):
    """Abstract base class for all benchmark plugins.

    Subclasses must set :attr:`slug` and implement :meth:`load_tasks`
    and :meth:`normalize_task`.
    """

    slug: ClassVar[str]

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.validate_config()

    # Abstract interface

    @abstractmethod
    def load_tasks(self) -> list[dict[str, Any]]:
        """Load and return all tasks for this benchmark.

        Each task is a plain dict with at minimum an ``instance_id`` key.
        """
        ...

    @abstractmethod
    def normalize_task(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Normalize a raw task row into the canonical task dict format.

        Args:
            raw: A single raw row as returned by the upstream dataset.

        Returns:
            A normalized task dict suitable for use by scaffolds.
        """
        ...

    # Concrete defaults

    def validate_config(self) -> None:
        """Validate benchmark-specific configuration.

        Subclasses can raise :class:`ValueError` for missing or invalid config.
        """
        return None

    @property
    def execution_environment(self) -> str:
        """Return the required task execution environment."""
        return "container"

    def validate_scaffold_support(self, scaffold: str) -> None:
        """Raise when *scaffold* is unsupported for this benchmark."""
        self.runtime_mode_for(scaffold)

    def derive_test_cmd(self, task: dict[str, Any]) -> str:
        """Derive a pytest command from ``task["FAIL_TO_PASS"]``.

        Handles both native list (SWE-rebench) and JSON-encoded string
        (SWE-Bench Verified) forms.
        """
        raw = task.get("FAIL_TO_PASS", "[]")
        if isinstance(raw, str):
            try:
                test_ids = json.loads(raw)
            except json.JSONDecodeError:
                test_ids = [raw] if raw else []
        else:
            test_ids = list(raw)
        if not test_ids:
            return "python -m pytest --no-header -q"
        return f"python -m pytest {' '.join(test_ids)} -x --no-header -q"

    def select_subset(
        self,
        tasks: list[dict[str, Any]],
        n: int | None = None,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return the first ``n`` tasks, sorted by instance_id for determinism.

        This is the simplest possible benchmark-agnostic default. Subclasses
        with specific selection needs (repo-stratified like SWE-Bench Verified,
        lite-filtering like SWE-rebench) MUST override this.
        """
        effective_n = n if n is not None else self.config.selection_n
        sorted_tasks = sorted(tasks, key=lambda t: t.get("instance_id", ""))
        return sorted_tasks[:effective_n]

    def image_name_for(self, task: dict[str, Any]) -> str | None:
        return task.get("image_name")

    def viz_filename(self, instance_id: str) -> str:
        """Filename for the per-task HTML report written by the collector.

        Defaults to ``trace_viz.html`` (unchanged for all existing benchmarks).
        Subclasses may override to embed identifying info in the name.
        """
        return "trace_viz.html"

    def runtime_mode_for(self, scaffold: str) -> str:
        """Return the runtime strategy label for the given scaffold."""
        return "host_controller"

    def build_runner(self, *, scaffold: str, **kwargs: Any) -> Any:
        """Build and return a scaffold runner for this benchmark.

        The base implementation always raises :exc:`NotImplementedError`.
        Subclasses that support concrete scaffold integrations **must** override
        this method.

        Raises:
            NotImplementedError: Always — subclasses must override.
        """
        raise NotImplementedError(
            f"Benchmark {self.slug!r} does not implement build_runner; "
            f"subclasses must override this method for scaffold={scaffold!r}"
        )
