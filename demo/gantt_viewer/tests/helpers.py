"""Shared test helpers for the Gantt viewer backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_trace(
    trace_path: Path,
    records: list[dict[str, Any]],
    *,
    scaffold: str = "synthetic",
    model: str = "test-model",
    max_iterations: int = 10,
    metadata_overrides: dict[str, Any] | None = None,
) -> Path:
    """Write a minimal canonical trace JSONL file."""
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "type": "trace_metadata",
        "scaffold": scaffold,
        "model": model,
        "trace_format_version": 5,
        "max_iterations": max_iterations,
    }
    if metadata_overrides:
        header.update(metadata_overrides)
    with trace_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header) + "\n")
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return trace_path


def write_config(config_path: Path, trace_paths: list[str]) -> Path:
    """Write a minimal discovery config."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["groups:", '  - name: "AC1 — test group"', "    paths:"]
    for trace_path in trace_paths:
        lines.append(f"      - {trace_path}")
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path
