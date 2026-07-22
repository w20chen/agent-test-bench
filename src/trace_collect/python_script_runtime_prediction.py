"""Runtime prediction artifacts for Python script tool commands."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import statistics
import time
from typing import Any, Iterator

from trace_collect.exec_classifier import classify_exec_tool_name
from trace_collect.runtime_knowledge import (
    default_common_kb_path,
    default_personal_kb_path,
    format_runtime_knowledge_summary,
    load_json_object,
    select_unified_prediction,
    update_personal_kb,
    write_json_object,
)

HISTORY_FILENAME = "history.json"
PREDICTIONS_FILENAME = "predictions.jsonl"
PREDICTION_FILENAME = "prediction.json"
HISTORY_LIMIT = 5
PREDICTION_SCHEMA_VERSION = 1
HISTORY_SCHEMA_VERSION = 1
LOCK_STALE_AFTER_S = 600.0

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._+-]+")
_EXIT_CODE_RE = re.compile(r"Exit code:\s*(-?\d+)")
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$", re.DOTALL)
_PYTHON_INTERPRETERS = {
    "python",
    "python3",
    "python3.9",
    "python3.10",
    "python3.11",
    "python3.12",
}
_PYTHON_FLAG_WITH_VALUE = {"-W", "-X"}
_PYTHON_BOOL_FLAGS = {
    "-b",
    "-bb",
    "-B",
    "-d",
    "-E",
    "-I",
    "-i",
    "-O",
    "-OO",
    "-P",
    "-q",
    "-R",
    "-s",
    "-S",
    "-u",
    "-v",
}
_PYTHON_NON_EXECUTING_FLAGS = {"-V", "--version", "-h", "--help"}


@dataclass(slots=True)
class PythonScriptRuntimeRecord:
    """Reference to one Python script runtime prediction invocation."""

    tool_call_id: str
    iteration: int
    directory: Path
    command: str
    working_directory: str | None = None
    history_root: Path | None = None


@dataclass(slots=True)
class PythonScriptCommand:
    """Normalized Python script command metadata."""

    normalized_command: str
    script_path: str
    script_basename: str
    args_signature: str
    python_flags: list[str]
    timeout_s: float | None = None
    shell_has_or_chain: bool = False
    shell_has_prefix_work: bool = False
    shell_has_followup_segments: bool = False


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_component(value: str, *, fallback: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", value.strip())[:80].strip("._-")
    return cleaned or fallback


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.monotonic_ns()}")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_json(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return {}, f"failed to read {path.name}: {exc}"
    if not isinstance(loaded, dict):
        return {}, f"{path.name} is not a JSON object"
    return loaded, None


def _history_path(history_root: Path | None, prediction_root: Path) -> Path:
    return (history_root or prediction_root) / HISTORY_FILENAME


@contextmanager
def _history_lock(history_path: Path, timeout_s: float = 30.0) -> Iterator[None]:
    lock_path = history_path.with_suffix(f"{history_path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age_s = time.time() - lock_path.stat().st_mtime
            except OSError:
                age_s = 0.0
            if age_s > LOCK_STALE_AFTER_S:
                try:
                    lock_path.unlink()
                    continue
                except OSError:
                    pass
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for {lock_path}")
            time.sleep(0.05)
    try:
        os.write(fd, _utc_now().encode("utf-8"))
        yield
    finally:
        os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _median(values: Any) -> float | None:
    clean = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and value >= 0
    ]
    if not clean:
        return None
    return float(statistics.median(clean))


def _bounded_append(values: Any, value: float) -> list[float]:
    clean = [float(v) for v in values if isinstance(v, (int, float)) and v >= 0]
    clean.append(float(value))
    return clean[-HISTORY_LIMIT:]


def _basename(token: str) -> str:
    return token.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]


def _split_shell_segments(command: str) -> list[tuple[str, str | None]]:
    segments: list[tuple[str, str | None]] = []
    current = ""
    in_single = False
    in_double = False
    idx = 0
    previous_operator: str | None = None
    while idx < len(command):
        ch = command[idx]
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single and (idx == 0 or command[idx - 1] != "\\"):
            in_double = not in_double
        if not in_single and not in_double:
            if ch == "\n" and (idx == 0 or command[idx - 1] != "\\"):
                segments.append((current, previous_operator))
                current = ""
                previous_operator = "\n"
                idx += 1
                continue
            if ch == "|" and (idx == 0 or command[idx - 1] != "\\"):
                if idx + 1 < len(command) and command[idx + 1] == "|":
                    segments.append((current, previous_operator))
                    current = ""
                    previous_operator = "||"
                    idx += 2
                else:
                    segments.append((current, previous_operator))
                    current = ""
                    previous_operator = "|"
                    idx += 1
                continue
            if ch == "&" and idx + 1 < len(command) and command[idx + 1] == "&":
                segments.append((current, previous_operator))
                current = ""
                previous_operator = "&&"
                idx += 2
                continue
            if ch == "&" and (idx == 0 or command[idx - 1] != "\\"):
                segments.append((current, previous_operator))
                current = ""
                previous_operator = "&"
                idx += 1
                continue
            if ch == ";":
                segments.append((current, previous_operator))
                current = ""
                previous_operator = ";"
                idx += 1
                continue
        current += ch
        idx += 1
    segments.append((current, previous_operator))
    return segments


def _strip_leading_comments(segment: str) -> str:
    lines = []
    for line in segment.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines)


def _posix_join(base: str | None, path_text: str) -> str:
    path = PurePosixPath(path_text)
    if path.is_absolute():
        return str(path)
    if base:
        return str(PurePosixPath(base) / path)
    return path_text


def _normalize_path_text(path_text: str) -> str:
    path = PurePosixPath(path_text)
    parts: list[str] = []
    for part in path.parts:
        if part in {"", ".", "/"}:
            continue
        if part == "..":
            if parts and parts[-1] != "..":
                parts.pop()
            else:
                parts.append(part)
            continue
        parts.append(part)
    prefix = "/" if path.is_absolute() else ""
    return prefix + "/".join(parts)


def _resolve_cd_path(cd_arg: str, current_dir: str | None) -> str:
    return _normalize_path_text(_posix_join(current_dir, cd_arg))


def _looks_like_activation(tokens: list[str]) -> bool:
    if len(tokens) < 2 or tokens[0] not in {".", "source"}:
        return False
    return "activate" in _basename(tokens[1])


def _strip_wrapper(tokens: list[str]) -> tuple[list[str], float | None]:
    timeout_s: float | None = None
    idx = 0
    while idx < len(tokens) and _ENV_ASSIGN_RE.fullmatch(tokens[idx]):
        idx += 1
    tokens = tokens[idx:]
    while tokens and _basename(tokens[0]) in {"sudo", "time", "timeout", "nohup"}:
        wrapper = _basename(tokens[0])
        tokens = tokens[1:]
        while tokens and tokens[0].startswith("-"):
            option = tokens[0]
            tokens = tokens[1:]
            if option in {"-u", "-g", "-k", "-s"} and tokens:
                tokens = tokens[1:]
        if wrapper == "timeout":
            if not tokens:
                return [], timeout_s
            try:
                timeout_s = float(tokens[0].rstrip("s"))
            except ValueError:
                timeout_s = None
            tokens = tokens[1:]
    return tokens, timeout_s


def _parse_python_script_tokens(
    tokens: list[str],
    *,
    current_dir: str | None,
    shell_has_or_chain: bool,
    shell_has_followup_segments: bool,
) -> PythonScriptCommand | None:
    tokens, timeout_s = _strip_wrapper(tokens)
    if not tokens:
        return None
    executable = _basename(tokens[0])
    if executable not in _PYTHON_INTERPRETERS:
        return None
    idx = 1
    python_flags: list[str] = []
    while idx < len(tokens):
        token = tokens[idx]
        if token in {"-c", "-m"} or token == "-":
            return None
        if token in _PYTHON_NON_EXECUTING_FLAGS:
            return None
        if token in _PYTHON_FLAG_WITH_VALUE:
            if idx + 1 >= len(tokens):
                return None
            python_flags.extend([token, tokens[idx + 1]])
            idx += 2
            continue
        if token.startswith("-W") and token != "-W":
            python_flags.append("-W")
            idx += 1
            continue
        if token.startswith("-X") and token != "-X":
            python_flags.append("-X")
            idx += 1
            continue
        if token in _PYTHON_BOOL_FLAGS:
            python_flags.append(token)
            idx += 1
            continue
        if token.startswith("-"):
            return None
        break
    if idx >= len(tokens):
        return None
    script_token = tokens[idx]
    if not script_token.endswith(".py"):
        return None
    script_path = _normalize_path_text(_posix_join(current_dir, script_token))
    script_basename = _basename(script_path)
    args = tokens[idx + 1 :]
    args_signature = " ".join(args)
    normalized_parts = ["python-script", script_path]
    if python_flags:
        normalized_parts.extend(["flags", *python_flags])
    if args:
        normalized_parts.extend(["args", *args])
    return PythonScriptCommand(
        normalized_command=" ".join(normalized_parts),
        script_path=script_path,
        script_basename=script_basename,
        args_signature=args_signature,
        python_flags=python_flags,
        timeout_s=timeout_s,
        shell_has_or_chain=shell_has_or_chain,
        shell_has_followup_segments=shell_has_followup_segments,
    )


def parse_python_script_command(
    command: str,
    *,
    working_directory: str | Path | None = None,
) -> PythonScriptCommand | None:
    """Return normalized metadata for a supported Python script command."""

    current_dir = (
        Path(working_directory).as_posix() if working_directory else None
    )
    segments = _split_shell_segments(command)
    shell_has_or_chain = any(operator == "||" for _, operator in segments)
    shell_has_prefix_work = False
    for segment_index, (segment, _operator) in enumerate(segments):
        segment = _strip_leading_comments(segment)
        if not segment.strip():
            continue
        try:
            tokens = shlex.split(segment, posix=True)
        except ValueError:
            continue
        if not tokens:
            continue
        if tokens[0] == "cd":
            if len(tokens) >= 2:
                current_dir = _resolve_cd_path(tokens[1], current_dir)
            continue
        if _looks_like_activation(tokens):
            continue
        if all(_ENV_ASSIGN_RE.fullmatch(token) for token in tokens):
            continue
        parsed = _parse_python_script_tokens(
            tokens,
            current_dir=current_dir,
            shell_has_or_chain=shell_has_or_chain,
            shell_has_followup_segments=any(
                _strip_leading_comments(later_segment).strip()
                for later_segment, _later_operator in segments[segment_index + 1 :]
            ),
        )
        if parsed is not None:
            parsed.shell_has_prefix_work = shell_has_prefix_work
            return parsed
        shell_has_prefix_work = True
    return None


def is_python_script_tool_call(tool_name: str, tool_args: dict[str, Any]) -> bool:
    """Return True when the tool call is a supported Python script command."""

    command = str(tool_args.get("command") or "")
    if not command:
        return False
    if classify_exec_tool_name(tool_name, tool_args) != "exec-python":
        return False
    return (
        parse_python_script_command(
            command,
            working_directory=tool_args.get("working_dir")
            or tool_args.get("working_directory"),
        )
        is not None
    )


def _history_commands(history: dict[str, Any]) -> dict[str, dict[str, Any]]:
    commands = history.get("commands")
    return commands if isinstance(commands, dict) else {}


def _history_scripts(history: dict[str, Any]) -> dict[str, dict[str, Any]]:
    scripts = history.get("scripts")
    return scripts if isinstance(scripts, dict) else {}


def _history_basenames(history: dict[str, Any]) -> dict[str, dict[str, Any]]:
    basenames = history.get("basenames")
    return basenames if isinstance(basenames, dict) else {}


def _history_tool(history: dict[str, Any]) -> dict[str, Any]:
    tool = history.get("tool")
    return tool if isinstance(tool, dict) else {}


def compute_python_script_predictions(
    *,
    history: dict[str, Any],
    family_history: dict[str, Any] | None = None,
    personal_knowledge: dict[str, Any] | None = None,
    common_knowledge: dict[str, Any] | None = None,
    normalized_command: str,
    script_path: str,
    script_basename: str,
) -> dict[str, Any]:
    """Compute instance-scoped Python script runtime predictions.

    Baselines (cascading priority):
    1. ``last_run`` (instance history, ``high``)
    2. ``family_last_run`` (same-repo siblings, ``medium``) — NEW
    3. ``script_path_median`` (``medium``)
    4. ``basename_median`` (``low``)
    5. ``global_median`` (``low``)
    """

    command_rec = _history_commands(history).get(normalized_command)
    command_durations = (
        command_rec.get("durations")
        if isinstance(command_rec, dict)
        and isinstance(command_rec.get("durations"), list)
        else []
    )
    last_run = (
        float(command_durations[-1])
        if command_durations and isinstance(command_durations[-1], (int, float))
        else None
    )

    family_last_run: float | None = None
    if family_history is not None:
        family_command_rec = _history_commands(family_history).get(normalized_command)
        family_durations = (
            family_command_rec.get("durations")
            if isinstance(family_command_rec, dict)
            and isinstance(family_command_rec.get("durations"), list)
            else []
        )
        family_last_run = (
            float(family_durations[-1])
            if family_durations and isinstance(family_durations[-1], (int, float))
            else None
        )

    script_rec = _history_scripts(history).get(script_path)
    script_median = _median(
        script_rec.get("durations")
        if isinstance(script_rec, dict)
        and isinstance(script_rec.get("durations"), list)
        else []
    )
    basename_rec = _history_basenames(history).get(script_basename)
    basename_median = _median(
        basename_rec.get("durations")
        if isinstance(basename_rec, dict)
        and isinstance(basename_rec.get("durations"), list)
        else []
    )
    global_median = _median(
        _history_tool(history).get("durations")
        if isinstance(_history_tool(history).get("durations"), list)
        else []
    )
    unified = select_unified_prediction(
        personal_kb=personal_knowledge or {},
        common_kb=common_knowledge or {},
        tool_name="python",
        tool_family="script_execution",
        operation="run_script",
        normalized_command=normalized_command,
        workload_bucket=None,
    )
    knowledge_p50 = unified.duration_p50_s if unified is not None else None
    knowledge_p90 = unified.duration_p90_s if unified is not None else None
    common_p50 = (
        knowledge_p50
        if unified is not None and unified.prediction_source.startswith("common:")
        else None
    )
    common_p90 = (
        knowledge_p90
        if unified is not None and unified.prediction_source.startswith("common:")
        else None
    )
    if last_run is not None:
        recommended = last_run
        method = "last_run"
        reliability = "high"
    elif family_last_run is not None:
        recommended = family_last_run
        method = "family_last_run"
        reliability = "medium"
    elif script_median is not None:
        recommended = script_median
        method = "script_path_median"
        reliability = "medium"
    elif basename_median is not None:
        recommended = basename_median
        method = "basename_median"
        reliability = "low"
    elif knowledge_p50 is not None:
        recommended = knowledge_p50
        method = unified.prediction_source if unified is not None else "knowledge_prior"
        reliability = unified.confidence if unified is not None else "low"
    else:
        recommended = global_median
        method = "global_median" if global_median is not None else "unavailable"
        reliability = "low" if global_median is not None else "unavailable"
    return {
        "prediction_last_run_s": last_run,
        "prediction_family_last_run_s": family_last_run,
        "prediction_script_path_median_s": script_median,
        "prediction_basename_median_s": basename_median,
        "prediction_global_median_s": global_median,
        "prediction_knowledge_p50_s": knowledge_p50,
        "prediction_knowledge_p90_s": knowledge_p90,
        "prediction_common_p50_s": common_p50,
        "prediction_common_p90_s": common_p90,
        "prediction_recommended_s": recommended,
        "prediction_recommended_method": method,
        "prediction_reliability": {"level": reliability},
        "runtime_knowledge_prediction": unified.to_dict() if unified else None,
    }


def update_python_script_history(
    *,
    history: dict[str, Any],
    normalized_command: str,
    script_path: str,
    script_basename: str,
    total_duration_s: float,
    success: bool,
) -> dict[str, Any]:
    """Return updated bounded history after a successful Python script run."""

    updated = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "updated_at": _utc_now(),
        "history_limit": HISTORY_LIMIT,
        "commands": dict(_history_commands(history)),
        "scripts": dict(_history_scripts(history)),
        "basenames": dict(_history_basenames(history)),
        "tool": dict(_history_tool(history)),
    }
    if not success:
        return updated
    for container_name, key in (
        ("commands", normalized_command),
        ("scripts", script_path),
        ("basenames", script_basename),
    ):
        rec = dict(updated[container_name].get(key) or {})
        rec["durations"] = _bounded_append(rec.get("durations") or [], total_duration_s)
        rec["last_seen_at"] = updated["updated_at"]
        updated[container_name][key] = rec
    tool_rec = dict(updated["tool"])
    tool_rec["durations"] = _bounded_append(
        tool_rec.get("durations") or [],
        total_duration_s,
    )
    tool_rec["last_seen_at"] = updated["updated_at"]
    updated["tool"] = tool_rec
    return updated


def prepare_python_script_runtime_prediction_before_tool(
    *,
    prediction_root: Path,
    history_root: Path | None = None,
    family_history_root: Path | None = None,
    iteration: int,
    tool_call_id: str,
    tool_name: str,
    tool_args: dict[str, Any],
    working_directory: str | Path | None = None,
) -> PythonScriptRuntimeRecord | None:
    """Prepare an artifact directory for one Python script invocation."""

    if not is_python_script_tool_call(tool_name, tool_args):
        return None
    command = str(tool_args.get("command") or "")
    effective_working_directory = (
        tool_args.get("working_dir")
        or tool_args.get("working_directory")
        or working_directory
        or ""
    )
    working_directory_text = str(effective_working_directory)
    parsed = parse_python_script_command(
        command,
        working_directory=working_directory_text or None,
    )
    if parsed is None:
        return None
    safe_id = _safe_component(tool_call_id, fallback="tool")
    invocation_dir = prediction_root / f"iter_{iteration:04d}_python-script_{safe_id}"
    invocation_dir.mkdir(parents=True, exist_ok=True)
    history_path = _history_path(history_root, prediction_root)
    history, history_warning = _load_json(history_path)
    family_history: dict[str, Any] | None = None
    if family_history_root is not None:
        family_history_path = _history_path(family_history_root, prediction_root)
        family_history, _family_warning = _load_json(family_history_path)
    else:
        family_history_path = prediction_root / "family_history.json"
        if family_history_path.exists():
            family_history, _family_warning = _load_json(family_history_path)
    common_knowledge = load_json_object(default_common_kb_path())
    personal_knowledge = load_json_object(
        default_personal_kb_path(history_root, prediction_root)
    )
    predictions = compute_python_script_predictions(
        history=history,
        family_history=family_history,
        personal_knowledge=personal_knowledge,
        common_knowledge=common_knowledge,
        normalized_command=parsed.normalized_command,
        script_path=parsed.script_path,
        script_basename=parsed.script_basename,
    )
    pending = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "iteration": iteration,
        "tool_call_id": tool_call_id,
        "command": command,
        "normalized_command": parsed.normalized_command,
        "script_path": parsed.script_path,
        "script_basename": parsed.script_basename,
        "args_signature": parsed.args_signature,
        "python_flags": parsed.python_flags,
        "timeout_s": parsed.timeout_s,
        "shell_has_or_chain": parsed.shell_has_or_chain,
        "shell_has_prefix_work": parsed.shell_has_prefix_work,
        "shell_has_followup_segments": parsed.shell_has_followup_segments,
        "predictions": predictions,
        "warnings": [history_warning] if history_warning else [],
        "working_directory": working_directory_text,
        "history_path": str(history_path),
        "history_scope": "shared" if history_root is not None else "attempt",
        "status": "pending_execution",
    }
    _write_json(invocation_dir / "pending.json", pending)
    return PythonScriptRuntimeRecord(
        tool_call_id=tool_call_id,
        iteration=iteration,
        directory=invocation_dir,
        command=command,
        working_directory=working_directory_text or None,
        history_root=history_root,
    )


def seed_python_script_history_from_shared(
    *,
    shared_history_root: Path,
    attempt_prediction_root: Path,
) -> None:
    """Copy shared Python script history into an attempt-local directory."""

    shared_history_path = _history_path(shared_history_root, attempt_prediction_root)
    attempt_history_path = _history_path(None, attempt_prediction_root)
    with _history_lock(shared_history_path):
        history, warning = _load_json(shared_history_path)
    if warning:
        return
    if history:
        _write_json(attempt_history_path, history)


def seed_python_script_family_history_from_shared(
    *,
    shared_history_root: Path,
    attempt_prediction_root: Path,
) -> None:
    """Copy shared family Python script history into ``family_history.json``."""

    shared_history_path = _history_path(shared_history_root, attempt_prediction_root)
    attempt_family_path = attempt_prediction_root / "family_history.json"
    with _history_lock(shared_history_path):
        history, warning = _load_json(shared_history_path)
    if warning:
        return
    if history:
        attempt_family_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = attempt_family_path.with_name(
            f".{attempt_family_path.name}.tmp-{os.getpid()}-{time.monotonic_ns()}"
        )
        tmp_path.write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(attempt_family_path)


def merge_python_script_predictions_into_shared_history(
    *,
    shared_history_root: Path,
    attempt_prediction_root: Path,
) -> None:
    """Merge successful attempt-local prediction rows into shared history."""

    predictions_path = attempt_prediction_root / PREDICTIONS_FILENAME
    if not predictions_path.exists():
        return
    rows: list[dict[str, Any]] = []
    try:
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                loaded = json.loads(line)
                if isinstance(loaded, dict):
                    rows.append(loaded)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return
    if not rows:
        return

    shared_history_path = _history_path(shared_history_root, attempt_prediction_root)
    with _history_lock(shared_history_path):
        history, warning = _load_json(shared_history_path)
        if warning:
            history = {}
        for row in rows:
            if row.get("history_updated") is not True:
                continue
            if row.get("history_scope") == "shared":
                continue
            normalized_command = row.get("normalized_command")
            script_path = row.get("script_path")
            script_basename = row.get("script_basename")
            total_duration_s = row.get("actual_duration_s")
            if not (
                isinstance(normalized_command, str)
                and isinstance(script_path, str)
                and isinstance(script_basename, str)
                and isinstance(total_duration_s, (int, float))
            ):
                continue
            history = update_python_script_history(
                history=history,
                normalized_command=normalized_command,
                script_path=script_path,
                script_basename=script_basename,
                total_duration_s=float(total_duration_s),
                success=True,
            )
        _write_json(shared_history_path, history)


def _absolute_error(prediction: float | None, actual: float) -> float | None:
    return None if prediction is None else abs(float(prediction) - actual)


def _relative_error(prediction: float | None, actual: float) -> float | None:
    if prediction is None or actual <= 0:
        return None
    return abs(float(prediction) - actual) / actual


def parse_exit_code(tool_result: str) -> int | None:
    """Parse the structured trailer emitted by OpenClaw's shell tool."""

    match = _EXIT_CODE_RE.search(tool_result)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def finalize_python_script_runtime_prediction(
    record: PythonScriptRuntimeRecord,
    *,
    prediction_root: Path,
    history_root: Path | None = None,
    action_id: str,
    ts_start: float,
    ts_end: float,
    duration_ms: float,
    success: bool,
    tool_result: str,
    working_directory: str | None = None,
) -> dict[str, Any]:
    """Persist prediction results for one Python script run."""

    actual_duration_s = max(0.0, float(duration_ms) / 1000.0)
    effective_working_dir = working_directory or record.working_directory
    pending, pending_warning = _load_json(record.directory / "pending.json")
    parsed = parse_python_script_command(
        record.command,
        working_directory=effective_working_dir,
    )
    if parsed is None:
        raise ValueError("record command is not a supported Python script command")
    normalized_command = (
        str(pending.get("normalized_command"))
        if isinstance(pending.get("normalized_command"), str)
        else parsed.normalized_command
    )
    script_path = (
        str(pending.get("script_path"))
        if isinstance(pending.get("script_path"), str)
        else parsed.script_path
    )
    script_basename = (
        str(pending.get("script_basename"))
        if isinstance(pending.get("script_basename"), str)
        else parsed.script_basename
    )
    shell_has_or_chain = (
        bool(pending.get("shell_has_or_chain"))
        if isinstance(pending.get("shell_has_or_chain"), bool)
        else parsed.shell_has_or_chain
    )
    shell_has_followup_segments = (
        bool(pending.get("shell_has_followup_segments"))
        if isinstance(pending.get("shell_has_followup_segments"), bool)
        else parsed.shell_has_followup_segments
    )
    shell_has_prefix_work = (
        bool(pending.get("shell_has_prefix_work"))
        if isinstance(pending.get("shell_has_prefix_work"), bool)
        else parsed.shell_has_prefix_work
    )
    effective_history_root = history_root or record.history_root
    history_path = _history_path(effective_history_root, prediction_root)
    _, history_warning = _load_json(history_path)
    predictions = (
        pending.get("predictions")
        if isinstance(pending.get("predictions"), dict)
        else {
            "prediction_last_run_s": None,
            "prediction_family_last_run_s": None,
            "prediction_script_path_median_s": None,
            "prediction_basename_median_s": None,
            "prediction_global_median_s": None,
            "prediction_knowledge_p50_s": None,
            "prediction_knowledge_p90_s": None,
            "prediction_common_p50_s": None,
            "prediction_common_p90_s": None,
            "prediction_recommended_s": None,
            "prediction_recommended_method": "unavailable",
            "prediction_reliability": {"level": "unavailable"},
            "runtime_knowledge_prediction": None,
        }
    )
    prediction_values = {
        "last_run": predictions["prediction_last_run_s"],
        "family_last_run": predictions["prediction_family_last_run_s"],
        "script_path_median": predictions["prediction_script_path_median_s"],
        "basename_median": predictions["prediction_basename_median_s"],
        "global_median": predictions["prediction_global_median_s"],
        "knowledge_p50": predictions.get("prediction_knowledge_p50_s"),
        "knowledge_p90": predictions.get("prediction_knowledge_p90_s"),
        "common_p50": predictions.get("prediction_common_p50_s"),
        "common_p90": predictions.get("prediction_common_p90_s"),
        "recommended": predictions["prediction_recommended_s"],
    }
    exit_code = parse_exit_code(tool_result)
    successful_run = (
        bool(success)
        and (exit_code is None or exit_code == 0)
        and not shell_has_or_chain
        and not shell_has_prefix_work
        and not shell_has_followup_segments
    )
    warnings = [
        warning for warning in (pending_warning, history_warning) if warning
    ]
    warnings.extend(
        str(warning)
        for warning in pending.get("warnings", [])
        if isinstance(warning, str)
    )
    payload: dict[str, Any] = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "run_id": action_id,
        "iteration": record.iteration,
        "tool_call_id": record.tool_call_id,
        "action_id": action_id,
        "command": record.command,
        "normalized_command": normalized_command,
        "script_path": script_path,
        "script_basename": script_basename,
        "args_signature": pending.get("args_signature", parsed.args_signature),
        "python_flags": pending.get("python_flags", parsed.python_flags),
        "timeout_s": pending.get("timeout_s", parsed.timeout_s),
        "shell_has_or_chain": shell_has_or_chain,
        "shell_has_prefix_work": shell_has_prefix_work,
        "shell_has_followup_segments": shell_has_followup_segments,
        "working_directory": effective_working_dir,
        "ts_start": ts_start,
        "ts_end": ts_end,
        "total_duration_s": actual_duration_s,
        "actual_duration_s": actual_duration_s,
        "exit_code": exit_code,
        "success": success,
        "history_updated": successful_run,
        "history_path": str(history_path),
        "history_scope": "shared" if effective_history_root is not None else "attempt",
        **predictions,
        "absolute_error": {
            key: _absolute_error(value, actual_duration_s)
            for key, value in prediction_values.items()
        },
        "relative_error": {
            key: _relative_error(value, actual_duration_s)
            for key, value in prediction_values.items()
        },
        "warnings": warnings,
    }
    try:
        with _history_lock(history_path):
            locked_history, locked_warning = _load_json(history_path)
            if locked_warning and locked_warning not in payload["warnings"]:
                payload["warnings"].append(locked_warning)
            updated = update_python_script_history(
                history=locked_history,
                normalized_command=normalized_command,
                script_path=script_path,
                script_basename=script_basename,
                total_duration_s=actual_duration_s,
                success=successful_run,
            )
            _write_json(history_path, updated)
    except Exception as exc:
        payload["history_updated"] = False
        payload["warnings"].append(f"failed to update history: {exc!r}")
    personal_kb_path = default_personal_kb_path(effective_history_root, prediction_root)
    try:
        personal_kb = load_json_object(personal_kb_path)
        updated_personal_kb = update_personal_kb(
            personal_kb,
            tool_name="python",
            tool_family="script_execution",
            operation="run_script",
            normalized_command=normalized_command,
            duration_s=actual_duration_s,
            success=successful_run,
            repo_id=effective_working_dir,
            features={
                "script_path": script_path,
                "script_basename": script_basename,
            },
        )
        write_json_object(personal_kb_path, updated_personal_kb)
        payload["personal_kb_path"] = str(personal_kb_path)
        payload["personal_kb_updated"] = successful_run
    except Exception as exc:
        payload["personal_kb_updated"] = False
        payload["warnings"].append(f"failed to update personal kb: {exc!r}")
    _write_json(record.directory / PREDICTION_FILENAME, payload)
    _append_jsonl(prediction_root / PREDICTIONS_FILENAME, payload)
    summary = format_python_script_prediction_summary(payload)
    if summary:
        print(summary, flush=True)
    return payload


def format_python_script_prediction_summary(payload: dict[str, Any]) -> str:
    """Return a compact human-readable prediction line showing all strategies."""

    actual = payload.get("actual_duration_s")
    if not isinstance(actual, (int, float)):
        return ""
    rel = payload.get("relative_error") or {}
    recommended = payload.get("prediction_recommended_method", "?")
    reliability = (payload.get("prediction_reliability") or {}).get("level", "?")

    def _s(value: Any) -> str:
        return "?" if value is None else f"{float(value):.1f}s"

    def _pct(method: str) -> str:
        v = rel.get(method)
        return "?" if v is None else f"{float(v) * 100:+.1f}%"

    def _strat(label: str, pred_key: str, err_key: str) -> str:
        val = payload.get(pred_key)
        arrow = "\u2192" if (err_key == recommended and val is not None) else " "
        return f"{arrow}{label}={_s(val)}({_pct(err_key)})"

    kb_part = format_runtime_knowledge_summary(payload)

    parts = [
        f"[py-predict] #{payload.get('iteration')}",
        f"{payload.get('script_basename')}",
        f"| {_s(actual)} actual",
        _strat("last", "prediction_last_run_s", "last_run"),
        _strat("fam", "prediction_family_last_run_s", "family_last_run"),
        _strat("path", "prediction_script_path_median_s", "script_path_median"),
        _strat("name", "prediction_basename_median_s", "basename_median"),
        _strat("glob", "prediction_global_median_s", "global_median"),
        kb_part,
        f"| {reliability}",
    ]
    return " ".join(part for part in parts if part)
