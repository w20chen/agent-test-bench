"""Load externalized agent prompt templates for scaffolds.

Templates live under ``configs/prompts/<benchmark_slug>/<name>.md`` where
``benchmark_slug`` is the benchmark slug with hyphens replaced by underscores
(e.g. ``terminal-bench`` → ``terminal_bench``). Templates must contain the
literal ``{{task}}`` placeholder that the caller substitutes with the task's
problem statement at run time.
"""

from __future__ import annotations

from pathlib import Path

_PLACEHOLDER = "{{task}}"
_PROMPTS_ROOT = Path(__file__).resolve().parents[2] / "configs" / "prompts"


def load_prompt_template(name: str, benchmark: str) -> str:
    """Return the raw template text for the named prompt.

    Args:
        name: Template name (stem of the ``.md`` file, e.g. ``"default"``).
        benchmark: Benchmark slug (e.g. ``"swe-rebench"``). Hyphens are
            converted to underscores to form the subdirectory name.

    Raises:
        FileNotFoundError: the template file does not exist.
        ValueError: the template does not contain the required ``{{task}}``
            placeholder.
    """
    slug_dir = benchmark.replace("-", "_")
    bench_dir = _PROMPTS_ROOT / slug_dir
    path = bench_dir / f"{name}.md"
    if not path.exists():
        available = sorted(p.stem for p in bench_dir.glob("*.md")) if bench_dir.exists() else []
        raise FileNotFoundError(
            f"Prompt template {name!r} not found at {path}. Available templates: {available}"
        )
    text = path.read_text(encoding="utf-8")
    if _PLACEHOLDER not in text:
        raise ValueError(
            f"Prompt template {name!r} at {path} is missing the required "
            f"{_PLACEHOLDER!r} placeholder."
        )
    return text


def render_prompt(template: str, problem_statement: str) -> str:
    """Substitute the ``{{task}}`` placeholder with the task's problem statement."""
    return template.replace(_PLACEHOLDER, problem_statement)
