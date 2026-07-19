"""Minimal pip install runtime prediction artifacts for trace collection."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import statistics
import time
from typing import Any, Iterator

from trace_collect.exec_classifier import classify_exec_tool_name

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
_PIP_INTERPRETERS = {
    "python",
    "python3",
    "python3.9",
    "python3.10",
    "python3.11",
    "python3.12",
}
_OUTPUT_FLAGS = {"-q", "--quiet", "-v", "--verbose"}
_BOOLEAN_FLAGS = {
    "--no-cache-dir",
    "--upgrade",
    "-U",
    "--force-reinstall",
    "--ignore-installed",
    "-I",
    "--editable",
    "-e",
}
_VALUE_FLAGS = {
    "-r",
    "--requirement",
    "-c",
    "--constraint",
    "-i",
    "--index-url",
    "--extra-index-url",
    "-f",
    "--find-links",
    "--trusted-host",
    "--progress-bar",
    "--python-version",
    "--platform",
    "--implementation",
    "--abi",
    "--root",
    "--prefix",
    "--src",
    "--target",
}
_STOP_TOKENS = {"|", "||", ">", ">>", "<", "2>", "2>>", "&>"}


@dataclass(slots=True)
class PipRuntimeRecord:
    """Reference to one pip runtime prediction invocation."""

    tool_call_id: str
    iteration: int
    directory: Path
    command: str
    working_directory: str | None = None
    history_root: Path | None = None


@dataclass(slots=True)
class PipInstallCommand:
    """Normalized pip install command metadata."""

    normalized_command: str
    package_count: int | None
    packages: list[str]
    requirement_files: list[str]
    shell_has_or_chain: bool = False


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


def _median(values: list[float]) -> float | None:
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


def _resolve_requirement_path(path_text: str, base_dir: Path | None) -> Path | None:
    if base_dir is None:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = base_dir / path
    try:
        resolved = path.resolve()
    except OSError:
        return None
    return resolved if resolved.exists() and resolved.is_file() else None


def _requirement_file_key(
    path_text: str,
    base_dir: Path | None,
) -> tuple[str, int | None]:
    resolved = _resolve_requirement_path(path_text, base_dir)
    if resolved is None:
        return f"{path_text}:missing", None
    try:
        raw = resolved.read_bytes()
    except OSError:
        return f"{path_text}:unreadable", None
    digest = hashlib.sha256(raw).hexdigest()[:16]
    count = _count_requirement_lines(raw.decode("utf-8", errors="replace").splitlines())
    return f"{path_text}:sha256={digest}", count


def _count_requirement_lines(lines: list[str]) -> int:
    count = 0
    continued = ""
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            continued += line[:-1].strip() + " "
            continue
        line = (continued + line).strip()
        continued = ""
        if not line or line.startswith(("-", "--")):
            continue
        count += 1
    if continued.strip() and not continued.strip().startswith(("-", "--")):
        count += 1
    return count


def _pip_install_tokens(
    command: str,
    *,
    working_directory: str | Path | None = None,
) -> tuple[list[str], Path | None, bool] | None:
    current_dir: Path | None = None
    working_dir = Path(working_directory) if working_directory else None
    segments = _split_shell_segments(command)
    has_or_chain = any(operator == "||" for _, operator in segments)
    for segment, _operator in segments:
        try:
            tokens = shlex.split(segment, posix=True)
        except ValueError:
            continue
        if not tokens:
            continue
        if tokens[0] == "cd":
            if len(tokens) >= 2:
                current_dir = Path(tokens[1])
                if not current_dir.is_absolute() and working_dir is not None:
                    current_dir = working_dir / current_dir
            continue
        while tokens and _ENV_ASSIGN_RE.fullmatch(tokens[0]):
            tokens.pop(0)
        if tokens and _basename(tokens[0]) == "sudo":
            tokens = tokens[1:]
        if not tokens:
            continue
        executable = _basename(tokens[0])
        if executable in {"pip", "pip3"}:
            args = tokens[1:]
        elif (
            executable in _PIP_INTERPRETERS
            and len(tokens) >= 4
            and tokens[1] == "-m"
            and tokens[2] == "pip"
        ):
            args = tokens[3:]
        else:
            continue
        if args and args[0] == "install":
            return args[1:], current_dir, has_or_chain
    return None


def parse_pip_install_command(
    command: str,
    *,
    working_directory: str | Path | None = None,
) -> PipInstallCommand | None:
    """Return normalized metadata for supported pip install commands."""

    parsed = _pip_install_tokens(command, working_directory=working_directory)
    if parsed is None:
        return None
    args, command_dir, has_or_chain = parsed
    base_dir = command_dir or (Path(working_directory) if working_directory else None)

    packages: list[str] = []
    requirement_files: list[str] = []
    normalized_flags: list[str] = []
    package_count = 0
    idx = 0
    while idx < len(args):
        token = args[idx]
        if token in _STOP_TOKENS:
            break
        split_name, split_value = (
            token.split("=", 1)
            if token.startswith("--") and "=" in token
            else (token, None)
        )
        if token in _OUTPUT_FLAGS or split_name in _OUTPUT_FLAGS:
            idx += 1
            continue
        if token in _BOOLEAN_FLAGS:
            normalized_flags.append(token)
            idx += 1
            continue
        if split_name in _BOOLEAN_FLAGS and split_value is not None:
            normalized_flags.append(split_name)
            idx += 1
            continue
        if token in {"-r", "--requirement"} or split_name in {"-r", "--requirement"}:
            value = (
                split_value
                if split_value is not None
                else args[idx + 1]
                if idx + 1 < len(args)
                else ""
            )
            if value:
                key, count = _requirement_file_key(value, base_dir)
                requirement_files.append(key)
                package_count += count if count is not None else 1
            idx += 1 if split_value is not None else 2
            continue
        if token in _VALUE_FLAGS or split_name in _VALUE_FLAGS:
            idx += 1 if split_value is not None else 2
            continue
        if token.startswith("-"):
            idx += 1
            continue
        packages.append(token)
        package_count += 1
        idx += 1

    normalized_parts = ["pip", "install"]
    normalized_parts.extend(sorted(packages))
    for req in sorted(requirement_files):
        normalized_parts.extend(["-r", req])
    normalized_parts.extend(sorted(normalized_flags))
    if len(normalized_parts) == 2:
        return None
    return PipInstallCommand(
        normalized_command=" ".join(normalized_parts),
        package_count=package_count if package_count > 0 else None,
        packages=sorted(packages),
        requirement_files=sorted(requirement_files),
        shell_has_or_chain=has_or_chain,
    )


def is_pip_install_tool_call(tool_name: str, tool_args: dict[str, Any]) -> bool:
    """Return True when the tool call is a supported pip install command."""

    command = str(tool_args.get("command") or "")
    if not command:
        return False
    return (
        classify_exec_tool_name(tool_name, tool_args) == "exec-pip"
        and parse_pip_install_command(
            command,
            working_directory=tool_args.get("working_dir")
            or tool_args.get("working_directory"),
        )
        is not None
    )


def prepare_pip_runtime_prediction_before_tool(
    *,
    prediction_root: Path,
    history_root: Path | None = None,
    iteration: int,
    tool_call_id: str,
    tool_name: str,
    tool_args: dict[str, Any],
) -> PipRuntimeRecord | None:
    """Prepare an artifact directory for one pip install invocation."""

    if not is_pip_install_tool_call(tool_name, tool_args):
        return None
    command = str(tool_args.get("command") or "")
    safe_id = _safe_component(tool_call_id, fallback="tool")
    invocation_dir = prediction_root / f"iter_{iteration:04d}_exec-pip_{safe_id}"
    invocation_dir.mkdir(parents=True, exist_ok=True)
    working_directory = str(
        tool_args.get("working_dir") or tool_args.get("working_directory") or ""
    )
    parsed = parse_pip_install_command(
        command,
        working_directory=working_directory or None,
    )
    if parsed is None:
        return None
    history_path = _history_path(history_root, prediction_root)
    history, history_warning = _load_json(history_path)
    predictions = compute_pip_predictions(
        history=history,
        normalized_command=parsed.normalized_command,
        package_count=parsed.package_count,
    )
    record = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "iteration": iteration,
        "tool_call_id": tool_call_id,
        "command": command,
        "normalized_command": parsed.normalized_command,
        "package_count": parsed.package_count,
        "packages": parsed.packages,
        "requirement_files": parsed.requirement_files,
        "shell_has_or_chain": parsed.shell_has_or_chain,
        "predictions": predictions,
        "warnings": [history_warning] if history_warning else [],
        "working_directory": working_directory,
        "history_path": str(history_path),
        "history_scope": "shared" if history_root is not None else "attempt",
        "status": "pending_execution",
    }
    _write_json(invocation_dir / "pending.json", record)
    return PipRuntimeRecord(
        tool_call_id=tool_call_id,
        iteration=iteration,
        directory=invocation_dir,
        command=command,
        working_directory=working_directory or None,
        history_root=history_root,
    )


def _history_commands(history: dict[str, Any]) -> dict[str, dict[str, Any]]:
    commands = history.get("commands")
    return commands if isinstance(commands, dict) else {}


def _history_tool(history: dict[str, Any]) -> dict[str, Any]:
    tool = history.get("tool")
    return tool if isinstance(tool, dict) else {}


def compute_pip_predictions(
    *,
    history: dict[str, Any],
    normalized_command: str,
    package_count: int | None,
) -> dict[str, Any]:
    """Compute simple pip runtime predictions from bounded prior history."""

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
    tool_history = _history_tool(history)
    per_package_median = _median(
        tool_history.get("per_package_s")
        if isinstance(tool_history.get("per_package_s"), list)
        else []
    )
    package_count_prediction = (
        package_count * per_package_median
        if package_count is not None and per_package_median is not None
        else None
    )
    global_median = _median(
        tool_history.get("durations")
        if isinstance(tool_history.get("durations"), list)
        else []
    )
    if last_run is not None:
        recommended = last_run
        method = "last_run"
        reliability = "high"
    elif package_count_prediction is not None:
        recommended = package_count_prediction
        method = "package_count"
        reliability = "medium"
    else:
        recommended = global_median
        method = "global_median" if global_median is not None else "unavailable"
        reliability = "low" if global_median is not None else "unavailable"
    return {
        "prediction_last_run_s": last_run,
        "prediction_package_count_s": package_count_prediction,
        "prediction_global_median_s": global_median,
        "prediction_recommended_s": recommended,
        "prediction_recommended_method": method,
        "prediction_reliability": {"level": reliability},
    }


def update_pip_history(
    *,
    history: dict[str, Any],
    normalized_command: str,
    total_duration_s: float,
    package_count: int | None,
    success: bool,
) -> dict[str, Any]:
    """Return updated bounded history after a successful pip install run."""

    updated = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "updated_at": _utc_now(),
        "history_limit": HISTORY_LIMIT,
        "commands": dict(_history_commands(history)),
        "tool": dict(_history_tool(history)),
    }
    if not success:
        return updated
    command_rec = dict(updated["commands"].get(normalized_command) or {})
    command_rec["durations"] = _bounded_append(
        command_rec.get("durations") or [],
        total_duration_s,
    )
    if package_count is not None:
        command_rec["package_counts"] = _bounded_append(
            command_rec.get("package_counts") or [],
            float(package_count),
        )
    command_rec["last_seen_at"] = updated["updated_at"]
    updated["commands"][normalized_command] = command_rec

    tool_rec = dict(updated["tool"])
    tool_rec["durations"] = _bounded_append(
        tool_rec.get("durations") or [],
        total_duration_s,
    )
    if package_count is not None and package_count > 0:
        tool_rec["per_package_s"] = _bounded_append(
            tool_rec.get("per_package_s") or [],
            total_duration_s / package_count,
        )
    tool_rec["last_seen_at"] = updated["updated_at"]
    updated["tool"] = tool_rec
    return updated


def seed_pip_history_from_shared(
    *,
    shared_history_root: Path,
    attempt_prediction_root: Path,
) -> None:
    """Copy shared pip history into an attempt-local prediction directory."""

    shared_history_path = _history_path(shared_history_root, attempt_prediction_root)
    attempt_history_path = _history_path(None, attempt_prediction_root)
    with _history_lock(shared_history_path):
        history, warning = _load_json(shared_history_path)
    if warning:
        return
    if history:
        _write_json(attempt_history_path, history)


def merge_pip_predictions_into_shared_history(
    *,
    shared_history_root: Path,
    attempt_prediction_root: Path,
) -> None:
    """Merge successful attempt-local pip prediction rows into shared history."""

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
            normalized_command = row.get("normalized_command")
            total_duration_s = row.get("actual_duration_s")
            if not isinstance(normalized_command, str) or not isinstance(
                total_duration_s, (int, float)
            ):
                continue
            package_count = row.get("package_count")
            history = update_pip_history(
                history=history,
                normalized_command=normalized_command,
                total_duration_s=float(total_duration_s),
                package_count=package_count if isinstance(package_count, int) else None,
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


def finalize_pip_runtime_prediction(
    record: PipRuntimeRecord,
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
    """Compute and persist prediction results for one pip install run."""

    actual_duration_s = max(0.0, float(duration_ms) / 1000.0)
    effective_working_dir = working_directory or record.working_directory
    pending, pending_warning = _load_json(record.directory / "pending.json")
    parsed = parse_pip_install_command(
        record.command,
        working_directory=effective_working_dir,
    )
    if parsed is None:
        raise ValueError("record command is not a supported pip install command")
    normalized_command = (
        str(pending.get("normalized_command"))
        if isinstance(pending.get("normalized_command"), str)
        else parsed.normalized_command
    )
    package_count = (
        int(pending["package_count"])
        if isinstance(pending.get("package_count"), int)
        else parsed.package_count
    )
    packages = (
        pending.get("packages")
        if isinstance(pending.get("packages"), list)
        else parsed.packages
    )
    requirement_files = (
        pending.get("requirement_files")
        if isinstance(pending.get("requirement_files"), list)
        else parsed.requirement_files
    )
    shell_has_or_chain = (
        bool(pending.get("shell_has_or_chain"))
        if isinstance(pending.get("shell_has_or_chain"), bool)
        else parsed.shell_has_or_chain
    )
    effective_history_root = history_root or record.history_root
    history_path = _history_path(effective_history_root, prediction_root)
    _, history_warning = _load_json(history_path)
    predictions = (
        pending.get("predictions")
        if isinstance(pending.get("predictions"), dict)
        else {
            "prediction_last_run_s": None,
            "prediction_package_count_s": None,
            "prediction_global_median_s": None,
            "prediction_recommended_s": None,
            "prediction_recommended_method": "unavailable",
            "prediction_reliability": {"level": "unavailable"},
        }
    )
    prediction_values = {
        "last_run": predictions["prediction_last_run_s"],
        "package_count": predictions["prediction_package_count_s"],
        "global_median": predictions["prediction_global_median_s"],
        "recommended": predictions["prediction_recommended_s"],
    }
    exit_code = parse_exit_code(tool_result)
    successful_install = (
        bool(success)
        and (exit_code is None or exit_code == 0)
        and not shell_has_or_chain
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
        "working_directory": effective_working_dir,
        "ts_start": ts_start,
        "ts_end": ts_end,
        "total_duration_s": actual_duration_s,
        "actual_duration_s": actual_duration_s,
        "package_count": package_count,
        "packages": packages,
        "requirement_files": requirement_files,
        "shell_has_or_chain": shell_has_or_chain,
        "exit_code": exit_code,
        "success": success,
        "history_updated": successful_install,
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
            updated = update_pip_history(
                history=locked_history,
                normalized_command=normalized_command,
                total_duration_s=actual_duration_s,
                package_count=package_count,
                success=successful_install,
            )
            _write_json(history_path, updated)
    except Exception as exc:
        payload["history_updated"] = False
        payload["warnings"].append(f"failed to update history: {exc!r}")
    _write_json(record.directory / PREDICTION_FILENAME, payload)
    _append_jsonl(prediction_root / PREDICTIONS_FILENAME, payload)
    summary = format_pip_prediction_summary(payload)
    if summary:
        print(summary, flush=True)
    return payload


def format_pip_prediction_summary(payload: dict[str, Any]) -> str:
    """Return a compact human-readable prediction line."""

    actual = payload.get("actual_duration_s")
    if not isinstance(actual, (int, float)):
        return ""
    rel = payload.get("relative_error") or {}

    def _fmt(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.2f}s"

    def _err(method: str) -> str:
        value = rel.get(method)
        return "n/a" if value is None else f"{float(value) * 100:.1f}%"

    return (
        "[pip-predict] "
        f"iter={payload.get('iteration')} "
        f"packages={payload.get('package_count')} "
        f"actual={_fmt(actual)} "
        f"last={_fmt(payload.get('prediction_last_run_s'))} "
        f"last_err={_err('last_run')} "
        f"package_count={_fmt(payload.get('prediction_package_count_s'))} "
        f"package_count_err={_err('package_count')} "
        f"global={_fmt(payload.get('prediction_global_median_s'))} "
        f"global_err={_err('global_median')} "
        f"recommended={payload.get('prediction_recommended_method')}:"
        f"{_fmt(payload.get('prediction_recommended_s'))} "
        f"rec_err={_err('recommended')} "
        f"reliability={(payload.get('prediction_reliability') or {}).get('level', 'unavailable')}"
    )
