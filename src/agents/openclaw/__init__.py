"""Expose the OpenClaw package and register its scaffold adapter."""

def __getattr__(name: str):  # noqa: ANN001
    """Lazy imports to avoid cascading dependency failures."""
    _imports = {
        "SWEBenchRunner": ("agents.openclaw.eval.runner", "SWEBenchRunner"),
        "EvalTask": ("agents.openclaw.eval.types", "EvalTask"),
        "EvalResult": ("agents.openclaw.eval.types", "EvalResult"),
    }
    if name in _imports:
        module_path, attr = _imports[name]
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "SWEBenchRunner",
    "EvalTask",
    "EvalResult",
]
