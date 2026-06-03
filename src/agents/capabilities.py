"""Scaffold/benchmark capability matrix derived from benchmark plugins."""

from __future__ import annotations


def _supported_scaffolds_for(cls: type[object]) -> set[str]:
    raw = getattr(cls, "SUPPORTED_SCAFFOLDS", None)
    if raw is None:
        return set()
    return {str(scaffold) for scaffold in raw}


def all_scaffolds() -> tuple[str, ...]:
    """Return every scaffold declared by registered benchmark plugins."""
    from agents.benchmarks import REGISTRY

    scaffolds: set[str] = set()
    for cls in REGISTRY.values():
        scaffolds.update(_supported_scaffolds_for(cls))
    return tuple(sorted(scaffolds))


def scaffold_benchmark_matrix() -> dict[str, set[str]]:
    """Return scaffold -> supported benchmark slugs."""
    from agents.benchmarks import REGISTRY

    known_scaffolds = all_scaffolds()
    matrix: dict[str, set[str]] = {scaffold: set() for scaffold in known_scaffolds}
    for slug, cls in REGISTRY.items():
        supported = _supported_scaffolds_for(cls)
        for scaffold in known_scaffolds:
            if scaffold in supported:
                matrix[scaffold].add(slug)
    return matrix


def validate_scaffold_benchmark(scaffold: str, benchmark_slug: str) -> None:
    """Raise ValueError when a scaffold/benchmark pair is unsupported."""
    from agents.benchmarks import REGISTRY

    known_scaffolds = all_scaffolds()
    if scaffold not in known_scaffolds:
        raise ValueError(
            f"Unknown scaffold {scaffold!r}; known scaffolds: {', '.join(known_scaffolds)}"
        )
    if benchmark_slug not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise ValueError(
            f"Unknown benchmark {benchmark_slug!r}; known benchmarks: {known}"
        )
    if benchmark_slug not in scaffold_benchmark_matrix()[scaffold]:
        raise ValueError(
            f"scaffold={scaffold!r} does not support benchmark={benchmark_slug!r}"
        )
