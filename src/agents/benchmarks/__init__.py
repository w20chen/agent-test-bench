"""Benchmark plugin registry.

Usage::

    from agents.benchmarks import get_benchmark_class
    cls = get_benchmark_class("swe-bench-verified")
    plugin = cls(config)

New benchmarks register here by adding an entry to :data:`REGISTRY`.
"""

from __future__ import annotations

from agents.benchmarks.base import Benchmark, BenchmarkConfig, Runner
from agents.benchmarks.browsecomp import BrowseCompBenchmark
from agents.benchmarks.deep_research_bench import DeepResearchBenchBenchmark
from agents.benchmarks.swe_bench_verified import SWEBenchVerified
from agents.benchmarks.swe_rebench import SWERebenchBenchmark
from agents.benchmarks.terminal_bench import TerminalBenchBenchmark

__all__ = [
    "REGISTRY",
    "get_benchmark_class",
    "Benchmark",
    "BenchmarkConfig",
    "BrowseCompBenchmark",
    "DeepResearchBenchBenchmark",
    "Runner",
    "SWEBenchVerified",
    "SWERebenchBenchmark",
    "TerminalBenchBenchmark",
]

#: Maps benchmark slug → concrete :class:`~agents.benchmarks.base.Benchmark` subclass.
REGISTRY: dict[str, type[Benchmark]] = {
    "browsecomp": BrowseCompBenchmark,
    "deep-research-bench": DeepResearchBenchBenchmark,
    "swe-bench-verified": SWEBenchVerified,
    "swe-rebench": SWERebenchBenchmark,
    "terminal-bench": TerminalBenchBenchmark,
}


def get_benchmark_class(slug: str) -> type[Benchmark]:
    """Return the :class:`~agents.benchmarks.base.Benchmark` subclass for *slug*.

    Args:
        slug: Benchmark identifier, e.g. ``"swe-bench-verified"``.

    Returns:
        The registered benchmark class.

    Raises:
        KeyError: If *slug* is not registered.  The error message lists known
            slugs so callers can diagnose typos quickly.
    """
    if slug not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(
            f"Benchmark slug {slug!r} is not registered. "
            f"Known slugs: {known}. "
            "To add a new benchmark, create src/agents/benchmarks/<slug>.py "
            "and register its class in agents.benchmarks.REGISTRY."
        )
    return REGISTRY[slug]
