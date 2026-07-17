"""Capture pytest test scripts referenced by trace-collected tool calls."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
import shlex
import shutil
from pathlib import Path
from typing import Any

from trace_collect.exec_classifier import classify_exec_tool_name

_PYTHON_INTERPRETERS = frozenset(
    {"python", "python3", "python3.9", "python3.10", "python3.11", "python3.12"}
)
_SHELL_STOPS = frozenset({"|", "||", "&&", ";", "&"})
_PYTEST_VALUE_OPTIONS = frozenset(
    {
        "-k",
        "-m",
        "-c",
        "-o",
        "--rootdir",
        "--confcutdir",
        "--basetemp",
        "--import-mode",
        "--ignore",
        "--ignore-glob",
        "--deselect",
        "--junitxml",
        "--junit-prefix",
        "--html",
        "--cov",
        "--cov-report",
        "--cov-config",
        "--maxfail",
        "--tb",
        "--color",
        "--capture",
        "--log-cli-level",
        "--log-file",
        "--log-file-level",
        "--durations",
        "--durations-min",
        "--reruns",
        "--reruns-delay",
        "--timeout",
    }
)
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._+-]+")


@dataclass(slots=True)
class PytestCaptureRecord:
    """Reference to one captured pytest invocation."""

    tool_call_id: str
    iteration: int
    directory: Path
    manifest_path: Path


def _safe_component(value: str, *, fallback: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", value.strip())[:80].strip("._-")
    return cleaned or fallback


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _append_index(
    capture_root: Path,
    record: dict[str, Any],
) -> None:
    capture_root.mkdir(parents=True, exist_ok=True)
    index_path = capture_root / "index.jsonl"
    with index_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _split_command(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return []


def _find_pytest_args(tokens: list[str]) -> tuple[int | None, list[str]]:
    """Return the pytest token index and raw pytest argument tokens."""
    for idx, token in enumerate(tokens):
        base = token.rsplit("/", 1)[-1]
        if base == "pytest":
            return idx, tokens[idx + 1 :]
        if (
            base in _PYTHON_INTERPRETERS
            and idx + 2 < len(tokens)
            and tokens[idx + 1] == "-m"
            and tokens[idx + 2] == "pytest"
        ):
            return idx, tokens[idx + 3 :]
    return None, []


def _resolve_command_cwd(tokens: list[str], pytest_idx: int | None, root: Path) -> Path:
    """Best-effort resolution of simple `cd DIR && pytest ...` prefixes."""
    cwd = root
    if pytest_idx is None:
        return cwd
    idx = 0
    while idx < pytest_idx:
        if tokens[idx] == "cd" and idx + 1 < pytest_idx:
            raw = tokens[idx + 1]
            path = Path(raw)
            cwd = path if path.is_absolute() else (cwd / path)
            idx += 2
            continue
        idx += 1
    return cwd


def _strip_node_id(value: str) -> str:
    return value.split("::", 1)[0]


def _option_name(token: str) -> str:
    return token.split("=", 1)[0]


def extract_pytest_targets(command: str) -> tuple[Path, list[str], list[str]]:
    """Parse pytest positional path targets from a shell command.

    Returns `(relative_cwd_hint, targets, warnings)`.  The cwd hint is only
    useful for tests of the parser; capture resolves it against the project
    root with `_resolve_command_cwd`.
    """
    tokens = _split_command(command)
    pytest_idx, raw_args = _find_pytest_args(tokens)
    warnings: list[str] = []
    if pytest_idx is None:
        return Path("."), [], ["could not locate pytest executable in command"]

    targets: list[str] = []
    idx = 0
    while idx < len(raw_args):
        token = raw_args[idx]
        if token in _SHELL_STOPS or token.startswith(">") or token.startswith("<"):
            break
        if re.fullmatch(r"\d?>&\d", token) is not None:
            break
        if token == "--":
            idx += 1
            continue
        if token.startswith("-"):
            name = _option_name(token)
            if name in _PYTEST_VALUE_OPTIONS and "=" not in token:
                idx += 2
            else:
                idx += 1
            continue
        targets.append(_strip_node_id(token))
        idx += 1

    cwd_hint = Path(".")
    if tokens:
        cwd_hint = _resolve_command_cwd(tokens, pytest_idx, Path("."))
    return cwd_hint, targets, warnings


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _discover_pytest_files(directory: Path) -> list[Path]:
    if not directory.exists() or not directory.is_dir():
        return []
    ignored = {".git", ".tox", ".venv", "venv", "__pycache__", "node_modules"}
    found: list[Path] = []
    for path in directory.rglob("*.py"):
        if any(part in ignored for part in path.parts):
            continue
        name = path.name
        if name == "conftest.py" or name.startswith("test") or name.endswith("_test.py"):
            found.append(path)
    return sorted(found)


def _resolve_target_files(
    *,
    command: str,
    project_root: Path,
) -> tuple[Path, list[dict[str, Any]], list[Path], list[str]]:
    tokens = _split_command(command)
    pytest_idx, _raw_args = _find_pytest_args(tokens)
    cwd = _resolve_command_cwd(tokens, pytest_idx, project_root)
    _cwd_hint, targets, warnings = extract_pytest_targets(command)
    if not targets:
        warnings.append("pytest command has no explicit path targets; captured tests discovered from cwd")
        targets = ["."]

    target_records: list[dict[str, Any]] = []
    files: list[Path] = []
    seen: set[Path] = set()
    for raw_target in targets:
        raw_target = raw_target.strip()
        if not raw_target:
            continue
        target_path = Path(raw_target)
        resolved = target_path if target_path.is_absolute() else cwd / target_path
        target_record: dict[str, Any] = {
            "argument": raw_target,
            "resolved_path": str(resolved),
            "exists": resolved.exists(),
        }
        if not _is_under(resolved, project_root):
            target_record["skipped"] = "outside_project_root"
            target_records.append(target_record)
            warnings.append(f"skipped target outside project root: {raw_target}")
            continue
        if resolved.is_file():
            target_record["kind"] = "file"
            candidates = [resolved] if resolved.suffix == ".py" else []
        elif resolved.is_dir():
            target_record["kind"] = "directory"
            candidates = _discover_pytest_files(resolved)
        else:
            target_record["kind"] = "missing"
            candidates = []
            warnings.append(f"pytest target does not exist: {raw_target}")
        target_record["file_count"] = len(candidates)
        target_records.append(target_record)
        for path in candidates:
            canonical = path.resolve()
            if canonical in seen:
                continue
            seen.add(canonical)
            files.append(path)
    return cwd, target_records, sorted(files), warnings


def capture_pytest_scripts_before_tool(
    *,
    capture_root: Path,
    project_root: Path,
    iteration: int,
    tool_call_id: str,
    tool_name: str,
    tool_args: dict[str, Any],
) -> PytestCaptureRecord | None:
    """Capture pytest scripts for an about-to-run tool call, if applicable."""
    classified = classify_exec_tool_name(tool_name, tool_args)
    if classified != "exec-pytest":
        return None
    command = str(tool_args.get("command") or "")
    if not command:
        return None

    safe_id = _safe_component(tool_call_id, fallback="tool")
    invocation_dir = (
        capture_root
        / f"iter_{iteration:04d}_exec-pytest_{safe_id}"
    )
    files_dir = invocation_dir / "files"
    invocation_dir.mkdir(parents=True, exist_ok=True)
    files_dir.mkdir(parents=True, exist_ok=True)

    command_path = invocation_dir / "command.sh"
    command_path.write_text(f"#!/usr/bin/env bash\n{command}\n", encoding="utf-8")

    tokens = _split_command(command)
    pytest_idx, _raw_args = _find_pytest_args(tokens)
    if pytest_idx is None:
        manifest_path = invocation_dir / "manifest.json"
        manifest = {
            "schema_version": 1,
            "capture_kind": "pytest_scripts",
            "capture_stage": "before_tool_execution",
            "created_at": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "iteration": iteration,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "classified_tool_name": classified,
            "command": command,
            "command_artifact": "command.sh",
            "project_root": str(project_root),
            "targets": [],
            "files": [],
            "warnings": [
                "classified as exec-pytest, but no pytest executable was found; "
                "no pytest scripts captured"
            ],
        }
        _json_dump(manifest_path, manifest)
        return PytestCaptureRecord(
            tool_call_id=tool_call_id,
            iteration=iteration,
            directory=invocation_dir,
            manifest_path=manifest_path,
        )

    cwd, target_records, source_files, warnings = _resolve_target_files(
        command=command,
        project_root=project_root,
    )

    copied_files: list[dict[str, Any]] = []
    for source in source_files:
        try:
            rel = source.resolve().relative_to(project_root.resolve())
        except ValueError:
            warnings.append(f"skipped source outside project root: {source}")
            continue
        dest = files_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        copied_files.append(
            {
                "source_path": str(source),
                "relative_path": rel.as_posix(),
                "artifact_path": str(dest.relative_to(invocation_dir)),
                "size_bytes": dest.stat().st_size,
                "sha256": _sha256_file(dest),
            }
        )

    manifest = {
        "schema_version": 1,
        "capture_kind": "pytest_scripts",
        "capture_stage": "before_tool_execution",
        "created_at": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "iteration": iteration,
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "classified_tool_name": classified,
        "command": command,
        "command_artifact": "command.sh",
        "project_root": str(project_root),
        "command_cwd": str(cwd),
        "targets": target_records,
        "files": copied_files,
        "warnings": warnings,
    }
    manifest_path = invocation_dir / "manifest.json"
    _json_dump(manifest_path, manifest)
    return PytestCaptureRecord(
        tool_call_id=tool_call_id,
        iteration=iteration,
        directory=invocation_dir,
        manifest_path=manifest_path,
    )


def record_pytest_capture_failure(
    *,
    capture_root: Path,
    iteration: int,
    tool_call_id: str,
    tool_name: str,
    tool_args: dict[str, Any],
    error: str,
) -> PytestCaptureRecord | None:
    """Persist a failure marker for a pytest capture attempt."""
    classified = classify_exec_tool_name(tool_name, tool_args)
    if classified != "exec-pytest":
        return None
    command = str(tool_args.get("command") or "")
    safe_id = _safe_component(tool_call_id, fallback="tool")
    invocation_dir = (
        capture_root
        / f"iter_{iteration:04d}_exec-pytest_{safe_id}"
    )
    invocation_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = invocation_dir / "manifest.json"
    manifest = {
        "schema_version": 1,
        "capture_kind": "pytest_scripts",
        "capture_stage": "capture_failed",
        "created_at": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "iteration": iteration,
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "classified_tool_name": classified,
        "command": command,
        "files": [],
        "warnings": [],
        "capture_error": error,
    }
    _json_dump(manifest_path, manifest)
    return PytestCaptureRecord(
        tool_call_id=tool_call_id,
        iteration=iteration,
        directory=invocation_dir,
        manifest_path=manifest_path,
    )


def record_pytest_finalize_failure(
    record: PytestCaptureRecord,
    *,
    action_id: str,
    ts_start: float,
    ts_end: float,
    duration_ms: float,
    success: bool,
    error: str,
) -> None:
    """Persist timing plus finalize error when manifest update fails."""
    fallback_manifest = {
        "schema_version": 1,
        "capture_kind": "pytest_scripts",
        "capture_stage": "finalize_failed",
        "iteration": record.iteration,
        "tool_call_id": record.tool_call_id,
        "action_id": action_id,
        "ts_start": ts_start,
        "ts_end": ts_end,
        "duration_ms": duration_ms,
        "success": success,
        "finalize_error": error,
    }
    try:
        existing = _load_json(record.manifest_path)
        existing.update(fallback_manifest)
        _json_dump(record.manifest_path, existing)
    except OSError:
        record.directory.mkdir(parents=True, exist_ok=True)
        _json_dump(record.directory / "manifest.finalize_failed.json", fallback_manifest)
    _append_index(
        record.directory.parent,
        {
            "iteration": record.iteration,
            "tool_call_id": record.tool_call_id,
            "action_id": action_id,
            "duration_ms": duration_ms,
            "ts_start": ts_start,
            "ts_end": ts_end,
            "success": success,
            "capture_stage": "finalize_failed",
            "manifest": str(record.manifest_path.relative_to(record.directory.parent)),
            "file_count": 0,
            "finalize_error": error,
        },
    )


def finalize_pytest_capture(
    record: PytestCaptureRecord,
    *,
    action_id: str,
    ts_start: float,
    ts_end: float,
    duration_ms: float,
    success: bool,
) -> None:
    """Add final trace timing metadata to a captured pytest invocation."""
    manifest = _load_json(record.manifest_path)
    capture_stage = str(manifest.get("capture_stage") or "complete")
    if capture_stage not in {"capture_failed", "finalize_failed"}:
        capture_stage = "complete"
    manifest.update(
        {
            "capture_stage": capture_stage,
            "action_id": action_id,
            "ts_start": ts_start,
            "ts_end": ts_end,
            "duration_ms": duration_ms,
            "success": success,
        }
    )
    _json_dump(record.manifest_path, manifest)

    index_record = {
        "iteration": record.iteration,
        "tool_call_id": record.tool_call_id,
        "action_id": action_id,
        "duration_ms": duration_ms,
        "ts_start": ts_start,
        "ts_end": ts_end,
        "success": success,
        "capture_stage": capture_stage,
        "manifest": str(record.manifest_path.relative_to(record.directory.parent)),
        "file_count": len(manifest.get("files") or []),
        "command": manifest.get("command", ""),
    }
    if manifest.get("capture_error"):
        index_record["capture_error"] = manifest.get("capture_error")
    _append_index(record.directory.parent, index_record)
