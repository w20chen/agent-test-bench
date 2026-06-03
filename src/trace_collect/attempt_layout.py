"""Write the canonical ``attempt_<N>`` artifact layout.

This module is intentionally write-only: callers provide already-computed
payloads and it persists them under the stable filenames used downstream.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

# Filenames — single source of truth so callers don't hardcode strings.

RUN_MANIFEST_FILENAME = "run_manifest.json"
RESULTS_FILENAME = "results.json"
RESOURCES_FILENAME = "resources.json"
TOOL_CALLS_FILENAME = "tool_calls.json"
CONTAINER_STDOUT_FILENAME = "container_stdout.txt"
TRACE_FILENAME = "trace.jsonl"

SCHEMA_VERSION = 1

def ensure_attempt_dir(attempt_dir: Path) -> Path:
    attempt_dir.mkdir(parents=True, exist_ok=True)
    return attempt_dir

def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

# Writers — one per artifact. Payloads are plain dicts/lists so the caller can
# choose dataclass -> dict conversion whichever way suits them.

def write_run_manifest(attempt_dir: Path, manifest: dict[str, Any]) -> Path:
    """Write run_manifest.json after stamping schema_version and defaults.

    The *manifest* dict must contain at least:
      - ``task`` (dict with instance_id, repo, docker_image)
      - ``attempt`` (e.g. "attempt_1")
      - ``model`` (dict with at least ``requested``)
      - ``runtime`` (dict with start_time / end_time ISO strings)
      - ``result_summary`` (dict with exit_code, total_time, etc.)

    Missing top-level keys are filled with stable defaults so the
    downstream viewer / analysis layer always sees the same shape.
    """
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": manifest.get("status", "completed"),
        "task": manifest.get("task", {}),
        "attempt": manifest.get("attempt", "attempt_1"),
        "model": manifest.get("model", {}),
        "runtime": manifest.get("runtime", {}),
        "artifacts": manifest.get(
            "artifacts",
            {
                "results_json": RESULTS_FILENAME,
                "resources_json": RESOURCES_FILENAME,
                "trace_jsonl": TRACE_FILENAME,
                "tool_calls_json": TOOL_CALLS_FILENAME,
                "container_stdout_txt": CONTAINER_STDOUT_FILENAME,
            },
        ),
        "replay": manifest.get("replay", {"replay_ready": False}),
        "result_summary": manifest.get("result_summary", {}),
    }
    # Any extra top-level keys the caller provided (e.g. prompt_template,
    # host_platform) are preserved verbatim.
    for key, value in manifest.items():
        if key not in payload:
            payload[key] = value
    path = attempt_dir / RUN_MANIFEST_FILENAME
    _write_json(path, payload)
    return path

def write_results_json(attempt_dir: Path, result: dict[str, Any]) -> Path:
    path = attempt_dir / RESULTS_FILENAME
    _write_json(path, result)
    return path

def write_resources_json(
    attempt_dir: Path,
    samples: list[dict[str, Any]],
    summary: dict[str, Any] | None = None,
) -> Path:
    """Write resources.json in the ``{samples: [...], summary: {...}}`` shape.

    An empty sample list + empty summary still produces a valid file so the
    downstream consumers can rely on the file always existing.
    """
    payload = {
        "samples": samples,
        "summary": summary if summary is not None else {},
    }
    path = attempt_dir / RESOURCES_FILENAME
    _write_json(path, payload)
    return path

def write_tool_calls_json(
    attempt_dir: Path, tool_calls: list[dict[str, Any]]
) -> Path:
    path = attempt_dir / TOOL_CALLS_FILENAME
    _write_json(path, tool_calls)
    return path

def write_container_stdout(attempt_dir: Path, stdout_text: str) -> Path:
    path = attempt_dir / CONTAINER_STDOUT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stdout_text or "", encoding="utf-8")
    return path

def build_tool_calls_from_trace(trace_path: Path) -> list[dict[str, Any]]:
    """Convert a canonical trace's ``tool_exec`` actions into the tool_calls.json shape.

    The harness stores ``tool_calls.json`` as a flat list of
    ``{timestamp, tool, id, input, end_timestamp, duration_ms, result_preview}``.
    OpenClaw emits ``tool_exec`` action records under canonical trace format,
    which we translate here so downstream analysis can use a single schema.
    """
    if not trace_path.exists():
        return []
    tool_calls: list[dict[str, Any]] = []
    try:
        text = trace_path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("type") != "action":
            continue
        if entry.get("action_type") != "tool_exec":
            continue
        data = entry.get("data") or {}
        ts_start = entry.get("ts_start")
        ts_end = entry.get("ts_end")
        duration_ms: float | None = None
        if isinstance(ts_start, (int, float)) and isinstance(ts_end, (int, float)):
            duration_ms = (ts_end - ts_start) * 1000.0

        raw_args = data.get("tool_args")
        if isinstance(raw_args, str):
            try:
                parsed_input: Any = json.loads(raw_args)
            except (json.JSONDecodeError, ValueError):
                parsed_input = {"raw": raw_args}
        else:
            parsed_input = raw_args or {}

        result_text = str(data.get("tool_result") or "")
        if len(result_text) > 500:
            result_text = result_text[:500] + "..."

        tool_calls.append(
            {
                "timestamp": _format_epoch(ts_start),
                "tool": data.get("tool_name", ""),
                "id": entry.get("action_id", ""),
                "input": parsed_input,
                "end_timestamp": _format_epoch(ts_end),
                "duration_ms": round(duration_ms, 1) if duration_ms is not None else None,
                "result_preview": result_text,
            }
        )
    return tool_calls

def _format_epoch(ts: Any) -> str | None:
    if not isinstance(ts, (int, float)):
        return None
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")

def copy_trace_jsonl(attempt_dir: Path, source_path: Path) -> Path:
    """Copy a produced trace.jsonl into the attempt dir under the canonical name.

    ``source_path`` and ``attempt_dir / TRACE_FILENAME`` may be the same file,
    in which case this is a no-op.
    """
    target = attempt_dir / TRACE_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() == target.resolve():
        return target
    shutil.copy2(source_path, target)
    return target
