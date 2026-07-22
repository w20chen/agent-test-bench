"""Minimal pytest runtime prediction artifacts for trace collection.

The prototype is intentionally small: it records per-node pytest durations from
the real command, computes predictions from bounded history, and writes
attempt-local JSON artifacts.  It does not run pre-execution pytest probes,
change benchmark labels, or use oracle data.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import statistics
import time
from typing import Any, Iterator

from trace_collect.exec_classifier import classify_exec_tool_name

HIDDEN_RUNTIME_DIR_ARG = "__openclaw_pytest_runtime_dir"
HIDDEN_RUNTIME_ENABLED_ARG = "__openclaw_pytest_runtime_enabled"

HISTORY_FILENAME = "history.json"
PREDICTIONS_FILENAME = "predictions.jsonl"
RUNTIME_JSON_FILENAME = "pytest_runtime.json"
INSTRUMENTATION_FILENAME = "instrumentation.json"
PLUGIN_MODULE = "openclaw_pytest_runtime_plugin"
HISTORY_LIMIT = 5
PREDICTION_SCHEMA_VERSION = 5
HISTORY_SCHEMA_VERSION = 5
LOCK_STALE_AFTER_S = 600.0
KNOWN_NODE_HIGH_RATIO = 0.80
KNOWN_OR_FILE_MEDIUM_RATIO = 0.80
COMMAND_COUNT_STABLE_REL_DELTA = 0.10

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._+-]+")
_EXIT_CODE_RE = re.compile(r"Exit code:\s*(-?\d+)")
_PYTHONPATH_ASSIGN_RE = re.compile(
    r"(?i)(?:^|[\s;&|])(?:export\s+|env\s+|set\s+)?PYTHONPATH\s*="
)
_SHELL_STOP_TOKENS = {"|", ">", ">>", "<", "2>", "2>>", "&>"}
_PYTEST_OUTPUT_FLAGS = {
    "-v",
    "-vv",
    "-vvv",
    "-q",
    "-qq",
    "--no-header",
    "--disable-warnings",
}
_PYTEST_OUTPUT_VALUE_FLAGS = {"--tb", "--color"}
_PYTEST_SORTABLE_VALUE_FLAGS = {"--ignore", "--ignore-glob", "--deselect"}
_PYTEST_ORDERED_VALUE_FLAGS = {
    "-k",
    "-m",
    "-p",
    "-o",
    "--confcutdir",
    "--rootdir",
    "--timeout",
    "--maxfail",
    "--junitxml",
    "--html",
    "--cov",
    "--cov-report",
    "--basetemp",
    "--cache-dir",
    "--capture",
    "--durations",
    "--durations-min",
    "--import-mode",
    "--junit-prefix",
    "--log-cli-level",
    "--log-file",
    "--log-file-level",
    "--reruns",
    "--reruns-delay",
}
_PYTEST_SELECTOR_FLAGS = {"-k", "-m", "--deselect"}


@dataclass(slots=True)
class ExplicitPytestNodeids:
    """Nodeids explicitly named in the pytest command text."""

    nodeids: list[str]
    coverage: str
    unmatched_positional_args: list[str]
    selector_flags: list[str]
    warnings: list[str]


@dataclass(slots=True)
class PytestRuntimeRecord:
    """Reference to one pytest runtime prediction invocation."""

    tool_call_id: str
    iteration: int
    directory: Path
    command: str
    working_directory: str | None = None
    history_root: Path | None = None


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_component(value: str, *, fallback: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", value.strip())[:80].strip("._-")
    return cleaned or fallback


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


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
    clean = [float(v) for v in values if isinstance(v, (int, float)) and v >= 0]
    if not clean:
        return None
    return float(statistics.median(clean))


def _node_file(nodeid: str) -> str:
    return nodeid.split("::", 1)[0]


def _squash_shell(command: str) -> str:
    try:
        return " ".join(shlex.split(command, posix=True))
    except ValueError:
        return " ".join(command.split())


def _shell_basename(token: str) -> str:
    return token.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]


def _is_redirection_token(token: str) -> bool:
    return (
        token in _SHELL_STOP_TOKENS
        or bool(re.fullmatch(r"\d?>&?\d", token))
        or bool(re.fullmatch(r"\d?>>?.+", token))
    )


def _split_option_value(token: str) -> tuple[str, str] | None:
    if not token.startswith("--") or "=" not in token:
        return None
    name, value = token.split("=", 1)
    return name, value


def _normalize_timeout_value(value: str) -> str:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([smhdSMHD]?)", value)
    if match is None:
        return value
    amount = float(match.group(1))
    unit = match.group(2).lower() or "s"
    multiplier = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}[unit]
    seconds = amount * multiplier
    if seconds.is_integer():
        return str(int(seconds))
    return f"{seconds:.3f}".rstrip("0").rstrip(".")


def _strip_supported_pytest_command_prefix(
    tokens: list[str],
) -> tuple[list[str] | None, list[str]]:
    """Return pytest args after simple shell prefixes, or None if unsupported."""
    warnings: list[str] = []
    if len(tokens) >= 3 and tokens[0] == "cd" and tokens[2] == "&&":
        tokens = tokens[3:]

    if "&&" in tokens or ";" in tokens:
        warnings.append("unsupported compound command before pytest")
        return None, warnings

    if tokens and _shell_basename(tokens[0]) == "timeout":
        if len(tokens) < 3:
            warnings.append("unsupported timeout prefix")
            return None, warnings
        tokens = tokens[2:]

    env_prefix: list[str] = []
    while tokens and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[0]):
        env_prefix.append(tokens.pop(0))

    pytest_idx: int | None = None
    pytest_args_start: int | None = None
    for idx, token in enumerate(tokens):
        base = _shell_basename(token)
        if base == "pytest":
            pytest_idx = idx
            pytest_args_start = idx + 1
            break
        if (
            base.startswith("python")
            and idx + 2 < len(tokens)
            and tokens[idx + 1] == "-m"
            and tokens[idx + 2] == "pytest"
        ):
            pytest_idx = idx
            pytest_args_start = idx + 3
            break
    if pytest_idx is None or pytest_args_start is None:
        warnings.append("pytest executable not found")
        return None, warnings
    if any(tok not in env_prefix for tok in tokens[:pytest_idx]):
        warnings.append("unsupported command prefix before pytest")
        return None, warnings
    return tokens[pytest_args_start:], warnings


def normalize_pytest_command(command: str) -> str:
    """Return a stable pytest selection key for the Last Run baseline.

    The key intentionally keeps timeout and test-selection semantics while
    dropping interpreter paths, pytest verbosity/reporting flags, and stdout
    post-processing such as ``| head`` or ``| grep``.
    """
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return _squash_shell(command)
    if not tokens:
        return ""

    cwd: str | None = None
    if len(tokens) >= 3 and tokens[0] == "cd" and tokens[2] == "&&":
        cwd = tokens[1]
        tokens = tokens[3:]

    if "&&" in tokens or ";" in tokens:
        return _squash_shell(command)

    timeout_s: str | None = None
    if tokens and _shell_basename(tokens[0]) == "timeout":
        if len(tokens) < 3:
            return _squash_shell(command)
        timeout_s = _normalize_timeout_value(tokens[1])
        tokens = tokens[2:]

    env_prefix: list[str] = []
    while tokens and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[0]):
        env_prefix.append(tokens.pop(0))

    pytest_idx: int | None = None
    pytest_args_start: int | None = None
    for idx, token in enumerate(tokens):
        base = _shell_basename(token)
        if base == "pytest":
            pytest_idx = idx
            pytest_args_start = idx + 1
            break
        if (
            base.startswith("python")
            and idx + 2 < len(tokens)
            and tokens[idx + 1] == "-m"
            and tokens[idx + 2] == "pytest"
        ):
            pytest_idx = idx
            pytest_args_start = idx + 3
            break
    if pytest_idx is None or pytest_args_start is None:
        return _squash_shell(command)
    if any(tok not in env_prefix for tok in tokens[:pytest_idx]):
        return _squash_shell(command)

    pytest_args: list[str] = []
    for token in tokens[pytest_args_start:]:
        if token in _SHELL_STOP_TOKENS or _is_redirection_token(token):
            break
        pytest_args.append(token)

    targets: list[str] = []
    sortable_flags: dict[str, list[str]] = {name: [] for name in _PYTEST_SORTABLE_VALUE_FLAGS}
    ordered_parts: list[str] = []
    idx = 0
    while idx < len(pytest_args):
        token = pytest_args[idx]
        split = _split_option_value(token)
        option_name = split[0] if split is not None else token

        if token in _PYTEST_OUTPUT_FLAGS or option_name in _PYTEST_OUTPUT_VALUE_FLAGS:
            idx += 1 if split is not None else 2 if token in _PYTEST_OUTPUT_VALUE_FLAGS else 1
            continue
        if split is not None and option_name in _PYTEST_SORTABLE_VALUE_FLAGS:
            sortable_flags[option_name].append(split[1])
            idx += 1
            continue
        if token in _PYTEST_SORTABLE_VALUE_FLAGS and idx + 1 < len(pytest_args):
            sortable_flags[token].append(pytest_args[idx + 1])
            idx += 2
            continue
        if split is not None and option_name in _PYTEST_ORDERED_VALUE_FLAGS:
            ordered_parts.extend([option_name, split[1]])
            idx += 1
            continue
        if token in _PYTEST_ORDERED_VALUE_FLAGS and idx + 1 < len(pytest_args):
            ordered_parts.extend([token, pytest_args[idx + 1]])
            idx += 2
            continue
        if token.startswith("-"):
            ordered_parts.append(token)
            idx += 1
            continue
        targets.append(token.removeprefix("/testbed/"))
        idx += 1

    parts: list[str] = []
    if cwd is not None:
        parts.extend(["cd", cwd, "&&"])
    if timeout_s is not None:
        parts.append(f"timeout={timeout_s}")
    parts.extend(env_prefix)
    parts.append("pytest")
    parts.extend(sorted(targets))
    for flag in sorted(sortable_flags):
        for value in sorted(sortable_flags[flag]):
            parts.extend([flag, value.removeprefix("/testbed/")])
    parts.extend(ordered_parts)
    return " ".join(parts)


def extract_explicit_pytest_nodeids(command: str) -> ExplicitPytestNodeids:
    """Extract pytest nodeids that are explicitly present in command text.

    This is intentionally a pure parser: it does not expand files, inspect the
    workspace, import tests, or run pytest. A mixed command such as
    ``pytest tests/test_a.py::test_one tests/test_b.py`` yields one explicit
    nodeid with ``coverage='partial'``.
    """
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError as exc:
        return ExplicitPytestNodeids([], "unknown", [], [], [f"parse_error: {exc}"])
    if not tokens:
        return ExplicitPytestNodeids([], "unknown", [], [], ["empty command"])

    pytest_args, warnings = _strip_supported_pytest_command_prefix(tokens)
    if pytest_args is None:
        return ExplicitPytestNodeids([], "unknown", [], [], warnings)

    nodeids: list[str] = []
    unmatched_positional_args: list[str] = []
    selector_flags: list[str] = []
    seen: set[str] = set()
    idx = 0
    while idx < len(pytest_args):
        token = pytest_args[idx]
        if token in _SHELL_STOP_TOKENS or _is_redirection_token(token):
            break
        split = _split_option_value(token)
        option_name = split[0] if split is not None else token

        if option_name in _PYTEST_SELECTOR_FLAGS:
            selector_flags.append(option_name)

        if token in _PYTEST_OUTPUT_FLAGS:
            idx += 1
            continue
        if option_name in _PYTEST_OUTPUT_VALUE_FLAGS:
            idx += 1 if split is not None else 2
            continue
        if option_name in _PYTEST_SORTABLE_VALUE_FLAGS:
            idx += 1 if split is not None else 2
            continue
        if option_name in _PYTEST_ORDERED_VALUE_FLAGS:
            idx += 1 if split is not None else 2
            continue
        if token.startswith("-"):
            idx += 1
            continue

        normalized = token.removeprefix("/testbed/")
        if "::" in normalized:
            if normalized not in seen:
                seen.add(normalized)
                nodeids.append(normalized)
        else:
            unmatched_positional_args.append(normalized)
        idx += 1

    if not nodeids:
        coverage = "unknown"
    elif unmatched_positional_args or selector_flags:
        coverage = "partial"
    else:
        coverage = "explicit_only"
    return ExplicitPytestNodeids(
        nodeids=nodeids,
        coverage=coverage,
        unmatched_positional_args=unmatched_positional_args,
        selector_flags=selector_flags,
        warnings=warnings,
    )


def command_contains_literal_pytest(command: str) -> bool:
    """Detect common literal pytest CLI forms without full shell parsing."""
    try:
        tokens = shlex.split(command, posix=(os.name != "nt"))
    except ValueError:
        return False
    for idx, token in enumerate(tokens):
        base = token.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if base == "pytest":
            return True
        if (
            base.startswith("python")
            and idx + 2 < len(tokens)
            and tokens[idx + 1] == "-m"
            and tokens[idx + 2] == "pytest"
        ):
            return True
    return False


def command_overrides_pythonpath(command: str) -> bool:
    """Return True when shell text may hide the injected plugin module path."""
    return bool(_PYTHONPATH_ASSIGN_RE.search(command))


def is_pytest_tool_call(tool_name: str, tool_args: dict[str, Any]) -> bool:
    """Return True when the existing exec classifier identifies pytest."""
    command = str(tool_args.get("command") or "")
    return (
        classify_exec_tool_name(tool_name, tool_args) == "exec-pytest"
        and command_contains_literal_pytest(command)
        and not command_overrides_pythonpath(command)
    )


def parse_exit_code(tool_result: str) -> int | None:
    """Parse the structured trailer emitted by OpenClaw's shell tool."""
    match = _EXIT_CODE_RE.search(tool_result)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _plugin_source() -> str:
    return r'''
from __future__ import annotations

import json
import os
from pathlib import Path

_COLLECTED = []
_TESTS = {}


def pytest_collection_finish(session):
    global _COLLECTED
    try:
        _COLLECTED = [item.nodeid for item in getattr(session, "items", [])]
    except Exception:
        _COLLECTED = []


def pytest_runtest_logreport(report):
    try:
        if report.when not in {"setup", "call", "teardown"}:
            return
        rec = _TESTS.setdefault(
            report.nodeid,
            {"nodeid": report.nodeid, "duration_s": 0.0, "outcome": "unknown"},
        )
        rec["duration_s"] += float(getattr(report, "duration", 0.0) or 0.0)
        if report.when == "call" or report.failed or report.skipped:
            rec["outcome"] = str(report.outcome)
    except Exception:
        return


def pytest_sessionfinish(session, exitstatus):
    try:
        out = os.environ.get("OPENCLAW_PYTEST_RUNTIME_JSON")
        if not out:
            return
        tests = []
        seen = set()
        for nodeid in _COLLECTED:
            rec = dict(_TESTS.get(nodeid) or {
                "nodeid": nodeid,
                "duration_s": 0.0,
                "outcome": "notrun",
            })
            rec["duration_s"] = round(float(rec.get("duration_s") or 0.0), 9)
            tests.append(rec)
            seen.add(nodeid)
        for nodeid, rec in sorted(_TESTS.items()):
            if nodeid in seen:
                continue
            item = dict(rec)
            item["duration_s"] = round(float(item.get("duration_s") or 0.0), 9)
            tests.append(item)
        payload = {
            "schema_version": 1,
            "collected_count": len(_COLLECTED),
            "exit_code": int(exitstatus),
            "tests": tests,
        }
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)
    except Exception:
        return
'''.lstrip()


def prepare_pytest_runtime_environment(
    *,
    invocation_dir: Path,
) -> dict[str, str]:
    """Write the temporary pytest plugin and return environment overrides."""
    invocation_dir.mkdir(parents=True, exist_ok=True)
    plugin_dir = invocation_dir / "_plugin"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    plugin_path = plugin_dir / f"{PLUGIN_MODULE}.py"
    source = _plugin_source()
    compile(source, str(plugin_path), "exec")
    plugin_path.write_text(source, encoding="utf-8")
    runtime_json = invocation_dir / RUNTIME_JSON_FILENAME

    existing, _warning = _load_json(invocation_dir / INSTRUMENTATION_FILENAME)
    payload = {
        **existing,
        "schema_version": 1,
        "prepared_at": _utc_now(),
        "plugin_module": PLUGIN_MODULE,
        "plugin_path": str(plugin_path),
        "runtime_json": str(runtime_json),
        "status": "prepared",
    }
    _write_json(invocation_dir / INSTRUMENTATION_FILENAME, payload)
    return {
        "OPENCLAW_PYTEST_RUNTIME_JSON": str(runtime_json),
        "OPENCLAW_PYTEST_RUNTIME_PLUGIN_DIR": str(plugin_dir),
        "OPENCLAW_PYTEST_RUNTIME_PLUGIN": PLUGIN_MODULE,
    }


def merge_pytest_runtime_environment(
    env: dict[str, str],
    overrides: dict[str, str],
) -> dict[str, str]:
    """Merge pytest runtime plugin settings into a subprocess environment."""
    merged = dict(env)
    plugin_dir = overrides["OPENCLAW_PYTEST_RUNTIME_PLUGIN_DIR"]
    plugin_name = overrides["OPENCLAW_PYTEST_RUNTIME_PLUGIN"]
    existing_plugins = merged.get("PYTEST_PLUGINS", "").strip()
    if existing_plugins:
        merged["PYTEST_PLUGINS"] = f"{existing_plugins},{plugin_name}"
    else:
        merged["PYTEST_PLUGINS"] = plugin_name
    existing_pythonpath = merged.get("PYTHONPATH", "")
    if existing_pythonpath:
        merged["PYTHONPATH"] = plugin_dir + os.pathsep + existing_pythonpath
    else:
        merged["PYTHONPATH"] = plugin_dir
    merged["OPENCLAW_PYTEST_RUNTIME_JSON"] = overrides[
        "OPENCLAW_PYTEST_RUNTIME_JSON"
    ]
    return merged


def prepare_pytest_runtime_prediction_before_tool(
    *,
    prediction_root: Path,
    history_root: Path | None = None,
    family_history_root: Path | None = None,
    iteration: int,
    tool_call_id: str,
    tool_name: str,
    tool_args: dict[str, Any],
    working_directory: str | Path | None = None,
    path_append: str = "",
) -> PytestRuntimeRecord | None:
    """Prepare historical prediction artifacts without running pytest."""
    _ = path_append  # Kept for caller compatibility; pre-run probing is forbidden.
    if not is_pytest_tool_call(tool_name, tool_args):
        return None
    command = str(tool_args.get("command") or "")
    if not command:
        return None

    safe_id = _safe_component(tool_call_id, fallback="tool")
    invocation_dir = prediction_root / f"iter_{iteration:04d}_exec-pytest_{safe_id}"
    invocation_dir.mkdir(parents=True, exist_ok=True)
    tool_args[HIDDEN_RUNTIME_DIR_ARG] = str(invocation_dir)
    tool_args[HIDDEN_RUNTIME_ENABLED_ARG] = True
    effective_working_directory = str(
        tool_args.get("working_dir")
        or tool_args.get("working_directory")
        or working_directory
        or ""
    )
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
    explicit_nodeids = extract_explicit_pytest_nodeids(command)
    predictions = compute_pytest_predictions(
        history=history,
        family_history=family_history,
        command=command,
        nodeids=explicit_nodeids.nodeids,
        nodeid_coverage=explicit_nodeids.coverage,
    )
    per_test_details = predictions.get("per_test_prediction_details")
    exact_explicit_nodeids = (
        bool(explicit_nodeids.nodeids)
        and isinstance(per_test_details, list)
        and all(
            isinstance(detail, dict) and detail.get("source") == "nodeid"
            for detail in per_test_details
        )
    )
    pre_execution_test_set_known = (
        explicit_nodeids.coverage == "explicit_only" and exact_explicit_nodeids
    )
    pre_execution_collected_count = (
        len(explicit_nodeids.nodeids) if pre_execution_test_set_known else None
    )

    record = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "iteration": iteration,
        "tool_call_id": tool_call_id,
        "command": command,
        "working_directory": effective_working_directory,
        "history_path": str(history_path),
        "history_scope": "shared" if history_root is not None else "attempt",
        "history_snapshot": history,
        "collect_only": {
            "schema_version": 1,
            "status": "disabled",
            "reason": "pre-execution pytest probing is disabled",
            "command": None,
            "argv": [],
            "nodeids": [],
            "collected_count": None,
            "duration_s": None,
        },
        "pre_execution_nodeid_source": "command_explicit"
        if explicit_nodeids.nodeids
        else "none",
        "pre_execution_nodeid_coverage": explicit_nodeids.coverage,
        "pre_execution_test_set_known": pre_execution_test_set_known,
        "pre_execution_nodeids": explicit_nodeids.nodeids,
        "pre_execution_explicit_nodeid_count": len(explicit_nodeids.nodeids),
        "pre_execution_unmatched_positional_args": (
            explicit_nodeids.unmatched_positional_args
        ),
        "pre_execution_selector_flags": explicit_nodeids.selector_flags,
        "pre_execution_collected_count": pre_execution_collected_count,
        "prediction_overhead_s": None,
        "history_before": {
            "test_count": len(_history_tests(history)),
            "command_count": len(_history_commands(history)),
            "project_test_median_s": predictions["project_test_median_s"],
            "project_overhead_median_s": predictions["project_overhead_median_s"],
            "project_unknown_test_median_s": predictions[
                "project_unknown_test_median_s"
            ],
            "recommended_method": predictions["prediction_recommended_method"],
            "reliability_level": predictions["prediction_reliability"]["level"],
        },
        "predictions": predictions,
        "warnings": [
            warning
            for warning in [history_warning, *explicit_nodeids.warnings]
            if warning
        ],
        "status": "pending_execution",
    }
    _write_json(invocation_dir / "pending.json", record)
    _write_json(invocation_dir / INSTRUMENTATION_FILENAME, record)
    return PytestRuntimeRecord(
        tool_call_id=tool_call_id,
        iteration=iteration,
        directory=invocation_dir,
        command=command,
        working_directory=effective_working_directory or None,
        history_root=history_root,
    )


def sanitize_tool_args_for_trace(tool_args: dict[str, Any]) -> dict[str, Any]:
    """Remove internal pytest runtime keys before persisting model-visible args."""
    return {
        key: value
        for key, value in tool_args.items()
        if key not in {HIDDEN_RUNTIME_DIR_ARG, HIDDEN_RUNTIME_ENABLED_ARG}
    }


def _history_tests(history: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tests = history.get("tests")
    return tests if isinstance(tests, dict) else {}


def _history_commands(history: dict[str, Any]) -> dict[str, dict[str, Any]]:
    commands = history.get("commands")
    return commands if isinstance(commands, dict) else {}


def _history_overheads(history: dict[str, Any]) -> dict[str, Any]:
    overheads = history.get("overheads")
    return overheads if isinstance(overheads, dict) else {}


def _history_unknown_tests(history: dict[str, Any]) -> dict[str, Any]:
    unknown_tests = history.get("unknown_tests")
    return unknown_tests if isinstance(unknown_tests, dict) else {}


def historical_duration(history: dict[str, Any], nodeid: str) -> float | None:
    rec = _history_tests(history).get(nodeid)
    if not isinstance(rec, dict):
        return None
    durations = rec.get("durations")
    return _median(durations if isinstance(durations, list) else [])


def global_history_median(history: dict[str, Any]) -> float | None:
    values: list[float] = []
    for rec in _history_tests(history).values():
        if not isinstance(rec, dict):
            continue
        durations = rec.get("durations")
        if isinstance(durations, list):
            values.extend(
                float(v) for v in durations if isinstance(v, (int, float)) and v >= 0
            )
    return _median(values)


def global_overhead_median(history: dict[str, Any]) -> float | None:
    overheads = _history_overheads(history)
    durations = overheads.get("durations")
    return _median(durations if isinstance(durations, list) else [])


def global_unknown_test_median(history: dict[str, Any]) -> float | None:
    unknown_tests = _history_unknown_tests(history)
    durations = unknown_tests.get("durations")
    return _median(durations if isinstance(durations, list) else [])


def _last_command_collected_count(command_rec: Any) -> int | None:
    if not isinstance(command_rec, dict):
        return None
    counts = command_rec.get("collected_counts")
    if isinstance(counts, list):
        for value in reversed(counts):
            if isinstance(value, int) and value > 0:
                return value
            if isinstance(value, float) and value > 0 and value.is_integer():
                return int(value)
    value = command_rec.get("last_observed_test_count")
    if isinstance(value, int) and value > 0:
        return value
    return None


def _collected_count_delta_ratio(current: int, previous: int | None) -> float | None:
    if previous is None or previous <= 0:
        return None
    return abs(float(current) - float(previous)) / float(previous)


def file_history_median(history: dict[str, Any], file_path: str) -> float | None:
    values: list[float] = []
    for nodeid, rec in _history_tests(history).items():
        if _node_file(str(nodeid)) != file_path or not isinstance(rec, dict):
            continue
        durations = rec.get("durations")
        if isinstance(durations, list):
            values.extend(
                float(v) for v in durations if isinstance(v, (int, float)) and v >= 0
            )
    return _median(values)


def predict_test_duration(
    history: dict[str, Any],
    nodeid: str,
    *,
    project_median: float | None = None,
) -> tuple[float | None, str]:
    """Predict one test duration using node, file, then project fallback."""
    exact = historical_duration(history, nodeid)
    if exact is not None:
        return exact, "nodeid"
    by_file = file_history_median(history, _node_file(nodeid))
    if by_file is not None:
        return by_file, "file"
    if project_median is None:
        project_median = global_history_median(history)
    if project_median is not None:
        return project_median, "project"
    return None, "unavailable"


def predict_test_duration_with_unknown_fallback(
    history: dict[str, Any],
    nodeid: str,
    *,
    project_median: float | None = None,
    unknown_median: float | None = None,
) -> tuple[float | None, str]:
    """Predict one test duration, preferring cold-start unknown-test history."""
    exact = historical_duration(history, nodeid)
    if exact is not None:
        return exact, "nodeid"
    by_file = file_history_median(history, _node_file(nodeid))
    if by_file is not None:
        return by_file, "file"
    if unknown_median is None:
        unknown_median = global_unknown_test_median(history)
    if unknown_median is not None:
        return unknown_median, "unknown"
    if project_median is None:
        project_median = global_history_median(history)
    if project_median is not None:
        return project_median, "project"
    return None, "unavailable"


def _source_ratio(source_counts: dict[str, int], source: str, total: int) -> float:
    return 0.0 if total <= 0 else source_counts.get(source, 0) / total


def compute_prediction_reliability(
    *,
    nodeids: list[str],
    command_rec: dict[str, Any] | None,
    predictions: dict[str, Any],
    family_last_run: float | None = None,
    nodeid_coverage: str = "complete",
) -> dict[str, Any]:
    """Choose an interpretable recommended prediction from pre-run history."""
    total = len(nodeids)
    per_test_details = predictions.get("per_test_prediction_details")
    unknown_details = predictions.get("unknown_test_fallback_prediction_details")
    per_sources = [
        str(detail.get("source") or "unavailable")
        for detail in (per_test_details if isinstance(per_test_details, list) else [])
        if isinstance(detail, dict)
    ]
    unknown_sources = [
        str(detail.get("source") or "unavailable")
        for detail in (unknown_details if isinstance(unknown_details, list) else [])
        if isinstance(detail, dict)
    ]
    per_counts = {source: per_sources.count(source) for source in set(per_sources)}
    unknown_counts = {
        source: unknown_sources.count(source) for source in set(unknown_sources)
    }
    known_node_ratio = _source_ratio(per_counts, "nodeid", total)
    file_fallback_ratio = _source_ratio(per_counts, "file", total)
    project_fallback_ratio = _source_ratio(per_counts, "project", total)
    unavailable_ratio = _source_ratio(per_counts, "unavailable", total)
    unknown_fallback_ratio = _source_ratio(unknown_counts, "unknown", total)
    known_or_file_ratio = known_node_ratio + file_fallback_ratio

    previous_count = _last_command_collected_count(command_rec)
    count_delta = _collected_count_delta_ratio(total, previous_count)
    repeated_command = isinstance(predictions.get("prediction_last_run_s"), (int, float))

    reasons: list[str] = []
    method = "none"
    value: float | None = None
    level = "error"
    last_run = predictions.get("prediction_last_run_s")
    per_test = predictions.get("prediction_per_test_s")
    unknown_fallback = predictions.get("prediction_unknown_test_fallback_s")
    explicit_partial = bool(nodeids) and nodeid_coverage == "partial"
    explicit_only = bool(nodeids) and nodeid_coverage == "explicit_only"

    if total <= 0:
        if repeated_command and isinstance(last_run, (int, float)):
            method = "last_run"
            value = float(last_run)
            level = "medium"
            reasons.append("same_normalized_command_seen_before")
            reasons.append("pre_execution_test_set_unknown")
        elif isinstance(family_last_run, (int, float)):
            method = "family_last_run"
            value = float(family_last_run)
            level = "medium"
            reasons.append("family_last_run_fallback")
            reasons.append("pre_execution_test_set_unknown")
        else:
            level = "coldstart"
            reasons.append("pre_execution_test_set_unknown")
            reasons.append("no_available_prediction")
    elif (
        repeated_command
        and isinstance(last_run, (int, float))
        and count_delta is not None
        and count_delta <= COMMAND_COUNT_STABLE_REL_DELTA
    ):
        method = "last_run"
        value = float(last_run)
        level = "high"
        reasons.append("same_normalized_command_seen_before")
        reasons.append("collected_count_stable")
    elif repeated_command and isinstance(last_run, (int, float)) and count_delta is None:
        method = "last_run"
        value = float(last_run)
        level = "medium"
        reasons.append("same_normalized_command_seen_before")
        reasons.append("previous_collected_count_unavailable")
    elif isinstance(family_last_run, (int, float)):
        method = "family_last_run"
        value = float(family_last_run)
        level = "medium"
        reasons.append("family_last_run_fallback")
    elif (
        isinstance(per_test, (int, float))
        and known_node_ratio >= KNOWN_NODE_HIGH_RATIO
        and not explicit_partial
    ):
        method = "per_test"
        value = float(per_test)
        level = "high"
        reasons.append("known_node_ratio_high")
    elif (
        isinstance(per_test, (int, float))
        and known_or_file_ratio >= KNOWN_OR_FILE_MEDIUM_RATIO
        and not explicit_only
        and not explicit_partial
    ):
        method = "per_test"
        value = float(per_test)
        level = "medium"
        reasons.append("known_or_file_ratio_high")
    elif isinstance(unknown_fallback, (int, float)):
        if explicit_partial or explicit_only:
            level = "coldstart"
            if explicit_partial:
                reasons.append("partial_prediction_lower_bound")
            else:
                reasons.append("explicit_nodeids_not_exact")
            reasons.append("no_full_command_prediction")
        else:
            method = "unknown_test_fallback"
            value = float(unknown_fallback)
            level = "low"
            reasons.append("cold_start_or_project_fallback")
    else:
        if total > 0:
            level = "coldstart"
        reasons.append("no_available_prediction")

    if repeated_command and count_delta is not None and count_delta > COMMAND_COUNT_STABLE_REL_DELTA:
        reasons.append("same_command_collected_count_changed")
    if project_fallback_ratio > 0:
        reasons.append("project_fallback_used")
    if unknown_fallback_ratio > 0:
        reasons.append("unknown_test_fallback_used")
    if unavailable_ratio > 0:
        reasons.append("some_tests_unavailable")
    if explicit_partial and "explicit_nodeids_partial" not in reasons:
        reasons.append("explicit_nodeids_partial")
    if explicit_only and known_node_ratio < KNOWN_NODE_HIGH_RATIO:
        reasons.append("explicit_nodeids_not_exact")

    return {
        "level": level,
        "reasons": reasons,
        "known_node_ratio": known_node_ratio,
        "file_fallback_ratio": file_fallback_ratio,
        "project_fallback_ratio": project_fallback_ratio,
        "unknown_fallback_ratio": unknown_fallback_ratio,
        "unavailable_ratio": unavailable_ratio,
        "repeated_command": repeated_command,
        "previous_collected_count": previous_count,
        "collected_count_delta_ratio": count_delta,
        "nodeid_coverage": nodeid_coverage,
        "recommended_method": method,
        "recommended_prediction_s": value,
    }


def compute_pytest_predictions(
    *,
    history: dict[str, Any],
    family_history: dict[str, Any] | None = None,
    command: str,
    nodeids: list[str],
    nodeid_coverage: str | None = None,
) -> dict[str, Any]:
    """Compute Last Run, Test Count, and overhead-adjusted Per-Test Sum."""
    effective_nodeid_coverage = nodeid_coverage or ("complete" if nodeids else "unknown")
    project_median = global_history_median(history)
    overhead_median = global_overhead_median(history)
    unknown_median = global_unknown_test_median(history)
    command_key = normalize_pytest_command(command)
    command_rec = _history_commands(history).get(command_key)
    command_durations = (
        command_rec.get("durations")
        if isinstance(command_rec, dict) and isinstance(command_rec.get("durations"), list)
        else []
    )
    last_run = (
        float(command_durations[-1])
        if command_durations and isinstance(command_durations[-1], (int, float))
        else None
    )

    family_last_run: float | None = None
    if family_history is not None:
        family_command_rec = _history_commands(family_history).get(command_key)
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
    test_count = (
        len(nodeids) * project_median
        if project_median is not None and nodeids
        else None
    )

    per_test_values: list[dict[str, Any]] = []
    unknown_fallback_values: list[dict[str, Any]] = []
    per_test_total = 0.0
    unknown_fallback_total = 0.0
    per_test_available = True
    unknown_fallback_available = True
    for nodeid in nodeids:
        predicted, source = predict_test_duration(
            history,
            nodeid,
            project_median=project_median,
        )
        per_test_values.append(
            {
                "nodeid": nodeid,
                "predicted_duration_s": predicted,
                "source": source,
            }
        )
        if predicted is None:
            per_test_available = False
        else:
            per_test_total += predicted

        unknown_predicted, unknown_source = predict_test_duration_with_unknown_fallback(
            history,
            nodeid,
            project_median=project_median,
            unknown_median=unknown_median,
        )
        unknown_fallback_values.append(
            {
                "nodeid": nodeid,
                "predicted_duration_s": unknown_predicted,
                "source": unknown_source,
            }
        )
        if unknown_predicted is None:
            unknown_fallback_available = False
        else:
            unknown_fallback_total += unknown_predicted

    per_test_without_overhead = (
        per_test_total if per_test_available and nodeids else None
    )
    overhead_used = (
        overhead_median
        if per_test_without_overhead is not None and overhead_median is not None
        else None
    )
    per_test_with_overhead = (
        per_test_without_overhead + overhead_used
        if per_test_without_overhead is not None and overhead_used is not None
        else per_test_without_overhead
    )
    unknown_without_overhead = (
        unknown_fallback_total if unknown_fallback_available and nodeids else None
    )
    unknown_overhead_used = (
        overhead_median
        if unknown_without_overhead is not None and overhead_median is not None
        else None
    )
    unknown_with_overhead = (
        unknown_without_overhead + unknown_overhead_used
        if unknown_without_overhead is not None and unknown_overhead_used is not None
        else unknown_without_overhead
    )

    result = {
        "command_key": command_key,
        "project_test_median_s": project_median,
        "project_overhead_median_s": overhead_median,
        "project_unknown_test_median_s": unknown_median,
        "prediction_last_run_s": last_run,
        "prediction_family_last_run_s": family_last_run,
        "prediction_nodeid_coverage": effective_nodeid_coverage,
        "prediction_test_count_s": test_count,
        "prediction_per_test_without_overhead_s": per_test_without_overhead,
        "prediction_per_test_overhead_s": overhead_used,
        "prediction_per_test_s": per_test_with_overhead,
        "prediction_explicit_nodeid_lower_bound_s": (
            per_test_with_overhead
            if effective_nodeid_coverage in {"partial", "explicit_only"}
            else None
        ),
        "per_test_prediction_details": per_test_values,
        "prediction_unknown_test_fallback_without_overhead_s": unknown_without_overhead,
        "prediction_unknown_test_fallback_overhead_s": unknown_overhead_used,
        "prediction_unknown_test_fallback_s": unknown_with_overhead,
        "unknown_test_fallback_prediction_details": unknown_fallback_values,
    }
    reliability = compute_prediction_reliability(
        nodeids=nodeids,
        command_rec=command_rec if isinstance(command_rec, dict) else None,
        predictions=result,
        family_last_run=family_last_run,
        nodeid_coverage=effective_nodeid_coverage,
    )
    result["prediction_reliability"] = reliability
    result["prediction_recommended_s"] = reliability["recommended_prediction_s"]
    result["prediction_recommended_method"] = reliability["recommended_method"]
    return result


def _empty_pytest_predictions(command: str) -> dict[str, Any]:
    return {
        "command_key": normalize_pytest_command(command),
        "project_test_median_s": None,
        "project_overhead_median_s": None,
        "project_unknown_test_median_s": None,
        "prediction_last_run_s": None,
        "prediction_family_last_run_s": None,
        "prediction_nodeid_coverage": "unknown",
        "prediction_test_count_s": None,
        "prediction_per_test_without_overhead_s": None,
        "prediction_per_test_overhead_s": None,
        "prediction_per_test_s": None,
        "prediction_explicit_nodeid_lower_bound_s": None,
        "per_test_prediction_details": [],
        "prediction_unknown_test_fallback_without_overhead_s": None,
        "prediction_unknown_test_fallback_overhead_s": None,
        "prediction_unknown_test_fallback_s": None,
        "unknown_test_fallback_prediction_details": [],
        "prediction_reliability": {
            "level": "error",
            "reasons": ["pre_execution_prediction_missing"],
            "known_node_ratio": 0.0,
            "file_fallback_ratio": 0.0,
            "project_fallback_ratio": 0.0,
            "unknown_fallback_ratio": 0.0,
            "unavailable_ratio": 0.0,
            "repeated_command": False,
            "previous_collected_count": None,
            "collected_count_delta_ratio": None,
            "nodeid_coverage": "unknown",
            "recommended_method": "none",
            "recommended_prediction_s": None,
        },
        "prediction_recommended_s": None,
        "prediction_recommended_method": "none",
    }


def _coerce_pytest_predictions(value: Any, command: str) -> dict[str, Any]:
    defaults = _empty_pytest_predictions(command)
    if not isinstance(value, dict):
        return defaults
    merged = {**defaults, **value}
    reliability = merged.get("prediction_reliability")
    if isinstance(reliability, dict):
        merged["prediction_reliability"] = {
            **defaults["prediction_reliability"],
            **reliability,
        }
    else:
        merged["prediction_reliability"] = defaults["prediction_reliability"]
    return merged


def _bounded_append(values: list[Any], value: float) -> list[float]:
    clean = [float(v) for v in values if isinstance(v, (int, float)) and v >= 0]
    clean.append(float(value))
    return clean[-HISTORY_LIMIT:]


def update_pytest_history(
    *,
    history: dict[str, Any],
    command: str,
    total_duration_s: float,
    tests: list[dict[str, Any]],
    per_test_prediction_details: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return updated bounded history after this run."""
    updated = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "updated_at": _utc_now(),
        "history_limit": HISTORY_LIMIT,
        "tests": dict(_history_tests(history)),
        "commands": dict(_history_commands(history)),
        "overheads": dict(_history_overheads(history)),
        "unknown_tests": dict(_history_unknown_tests(history)),
    }
    test_history = updated["tests"]
    prediction_source_by_nodeid = {
        str(detail.get("nodeid")): str(detail.get("source") or "")
        for detail in (per_test_prediction_details or [])
        if isinstance(detail, dict) and isinstance(detail.get("nodeid"), str)
    }
    unknown_observed_durations: list[float] = []
    observed_test_duration_s = 0.0
    observed_test_count = 0
    for test in tests:
        nodeid = test.get("nodeid")
        duration = test.get("duration_s")
        if not isinstance(nodeid, str) or not isinstance(duration, (int, float)):
            continue
        if str(test.get("outcome") or "").lower() == "notrun":
            continue
        observed_test_duration_s += float(duration)
        observed_test_count += 1
        if prediction_source_by_nodeid.get(nodeid) in {"project", "unavailable"}:
            unknown_observed_durations.append(float(duration))
        rec = dict(test_history.get(nodeid) or {})
        rec["durations"] = _bounded_append(rec.get("durations") or [], float(duration))
        rec["last_outcome"] = test.get("outcome")
        rec["last_seen_at"] = updated["updated_at"]
        test_history[nodeid] = rec

    if observed_test_count > 0:
        command_key = normalize_pytest_command(command)
        command_history = updated["commands"]
        command_rec = dict(command_history.get(command_key) or {})
        command_rec["durations"] = _bounded_append(
            command_rec.get("durations") or [],
            total_duration_s,
        )
        command_rec["collected_counts"] = _bounded_append(
            command_rec.get("collected_counts") or [],
            float(observed_test_count),
        )
        command_rec["last_seen_at"] = updated["updated_at"]
        command_rec["last_observed_test_count"] = observed_test_count
        command_history[command_key] = command_rec

        overhead_s = max(0.0, float(total_duration_s) - observed_test_duration_s)
        overheads = dict(updated["overheads"])
        overheads["durations"] = _bounded_append(overheads.get("durations") or [], overhead_s)
        overheads["last_seen_at"] = updated["updated_at"]
        overheads["last_observed_test_duration_s"] = observed_test_duration_s
        overheads["last_observed_test_count"] = observed_test_count
        updated["overheads"] = overheads
    if unknown_observed_durations:
        unknown_tests = dict(updated["unknown_tests"])
        durations = unknown_tests.get("durations") or []
        for duration in unknown_observed_durations:
            durations = _bounded_append(durations, duration)
        unknown_tests["durations"] = durations
        unknown_tests["last_seen_at"] = updated["updated_at"]
        unknown_tests["last_observed_count"] = len(unknown_observed_durations)
        updated["unknown_tests"] = unknown_tests
    return updated


def seed_pytest_history_from_shared(
    *,
    shared_history_root: Path,
    attempt_prediction_root: Path,
) -> None:
    """Copy shared pytest history into an attempt-local prediction directory."""

    shared_history_path = _history_path(shared_history_root, attempt_prediction_root)
    attempt_history_path = _history_path(None, attempt_prediction_root)
    with _history_lock(shared_history_path):
        history, warning = _load_json(shared_history_path)
    if warning:
        return
    if history:
        _write_json(attempt_history_path, history)


def seed_pytest_family_history_from_shared(
    *,
    shared_history_root: Path,
    attempt_prediction_root: Path,
) -> None:
    """Copy shared family pytest history into ``family_history.json``."""

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


def merge_pytest_predictions_into_shared_history(
    *,
    shared_history_root: Path,
    attempt_prediction_root: Path,
) -> None:
    """Merge successful attempt-local pytest prediction rows into shared history."""

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
            command = row.get("command")
            total_duration_s = row.get("actual_duration_s")
            tests = row.get("tests")
            details = row.get("history_update_per_test_prediction_details")
            if not isinstance(details, list):
                details = row.get("per_test_prediction_details")
            if not (
                isinstance(command, str)
                and isinstance(total_duration_s, (int, float))
                and isinstance(tests, list)
            ):
                continue
            history = update_pytest_history(
                history=history,
                command=command,
                total_duration_s=float(total_duration_s),
                tests=[test for test in tests if isinstance(test, dict)],
                per_test_prediction_details=[
                    detail for detail in details if isinstance(detail, dict)
                ]
                if isinstance(details, list)
                else None,
            )
        _write_json(shared_history_path, history)


def _relative_error(prediction: float | None, actual: float) -> float | None:
    if prediction is None or actual <= 0:
        return None
    return abs(prediction - actual) / actual


def _absolute_error(prediction: float | None, actual: float) -> float | None:
    return None if prediction is None else abs(prediction - actual)


def _load_runtime_tests(runtime_payload: dict[str, Any]) -> list[dict[str, Any]]:
    tests = runtime_payload.get("tests")
    if not isinstance(tests, list):
        return []
    result: list[dict[str, Any]] = []
    for item in tests:
        if not isinstance(item, dict):
            continue
        nodeid = item.get("nodeid")
        duration = item.get("duration_s")
        if not isinstance(nodeid, str):
            continue
        try:
            duration_s = float(duration or 0.0)
        except (TypeError, ValueError):
            duration_s = 0.0
        result.append(
            {
                "nodeid": nodeid,
                "duration_s": duration_s,
                "outcome": str(item.get("outcome") or "unknown"),
            }
        )
    return result


def finalize_pytest_runtime_prediction(
    record: PytestRuntimeRecord,
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
    """Persist results for one pytest run using the pre-execution prediction."""
    effective_history_root = history_root or record.history_root
    history_path = _history_path(effective_history_root, prediction_root)
    history, history_warning = _load_json(history_path)
    pending, pending_warning = _load_json(record.directory / "pending.json")
    runtime_payload, runtime_warning = _load_json(record.directory / RUNTIME_JSON_FILENAME)
    instrumentation, instrumentation_warning = _load_json(
        record.directory / INSTRUMENTATION_FILENAME
    )

    actual_duration_s = max(0.0, float(duration_ms) / 1000.0)
    tests = _load_runtime_tests(runtime_payload)
    nodeids = [str(test["nodeid"]) for test in tests]
    predictions = _coerce_pytest_predictions(pending.get("predictions"), record.command)
    collect_only = pending.get("collect_only") if isinstance(pending.get("collect_only"), dict) else {}
    collect_only_duration_s = collect_only.get("duration_s")
    if not isinstance(collect_only_duration_s, (int, float)):
        collect_only_duration_s = None
    pre_execution_nodeids = [
        str(nodeid)
        for nodeid in pending.get("pre_execution_nodeids", [])
        if isinstance(nodeid, str)
    ]
    pre_execution_collected_count = pending.get("pre_execution_collected_count")
    if not isinstance(pre_execution_collected_count, int):
        pre_execution_collected_count = None
    pre_execution_test_set_known = (
        pending.get("pre_execution_test_set_known")
        if isinstance(pending.get("pre_execution_test_set_known"), bool)
        else bool(pre_execution_nodeids)
    )
    pre_execution_nodeid_source = pending.get("pre_execution_nodeid_source")
    if not isinstance(pre_execution_nodeid_source, str):
        pre_execution_nodeid_source = "unknown"
    pre_execution_nodeid_coverage = pending.get("pre_execution_nodeid_coverage")
    if not isinstance(pre_execution_nodeid_coverage, str):
        pre_execution_nodeid_coverage = (
            "complete" if pre_execution_test_set_known else "unknown"
        )
    pre_execution_explicit_nodeid_count = pending.get(
        "pre_execution_explicit_nodeid_count"
    )
    if not isinstance(pre_execution_explicit_nodeid_count, int):
        pre_execution_explicit_nodeid_count = len(pre_execution_nodeids)
    pre_execution_history = (
        pending.get("history_snapshot")
        if isinstance(pending.get("history_snapshot"), dict)
        else {}
    )
    history_update_predictions = compute_pytest_predictions(
        history=pre_execution_history,
        command=record.command,
        nodeids=nodeids,
    )
    exit_code = (
        runtime_payload.get("exit_code")
        if isinstance(runtime_payload.get("exit_code"), int)
        else parse_exit_code(tool_result)
    )
    successful_pytest_run = (
        bool(success)
        and bool(tests)
        and (exit_code is None or exit_code == 0)
    )

    warnings = [
        warning
        for warning in (
            pending_warning,
            history_warning,
            runtime_warning,
            instrumentation_warning,
        )
        if warning
    ]
    warnings.extend(
        str(warning)
        for warning in pending.get("warnings", [])
        if isinstance(warning, str)
    )
    if not tests:
        warnings.append("pytest runtime JSON missing or contains no tests")

    prediction_values = {
        "last_run": predictions["prediction_last_run_s"],
        "family_last_run": predictions["prediction_family_last_run_s"],
        "test_count": predictions["prediction_test_count_s"],
        "per_test": predictions["prediction_per_test_s"],
        "unknown_test_fallback": predictions["prediction_unknown_test_fallback_s"],
        "recommended": predictions["prediction_recommended_s"],
    }
    absolute_error = {
        key: _absolute_error(value, actual_duration_s)
        for key, value in prediction_values.items()
    }
    relative_error = {
        key: _relative_error(value, actual_duration_s)
        for key, value in prediction_values.items()
    }

    payload: dict[str, Any] = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "run_id": action_id,
        "iteration": record.iteration,
        "tool_call_id": record.tool_call_id,
        "action_id": action_id,
        "command": record.command,
        "normalized_command": predictions["command_key"],
        "working_directory": (
            working_directory
            or record.working_directory
            or instrumentation.get("working_directory")
        ),
        "ts_start": ts_start,
        "ts_end": ts_end,
        "total_duration_s": actual_duration_s,
        "actual_duration_s": actual_duration_s,
        "exit_code": exit_code,
        "success": success,
        "history_updated": successful_pytest_run,
        "history_path": str(history_path),
        "history_scope": "shared" if effective_history_root is not None else "attempt",
        "collected_count": (
            runtime_payload.get("collected_count")
            if isinstance(runtime_payload.get("collected_count"), int)
            else len(nodeids)
        ),
        "collected_tests": nodeids,
        "tests": tests,
        "pre_execution_test_set_known": pre_execution_test_set_known,
        "pre_execution_nodeid_source": pre_execution_nodeid_source,
        "pre_execution_nodeid_coverage": pre_execution_nodeid_coverage,
        "pre_execution_collected_count": pre_execution_collected_count,
        "pre_execution_explicit_nodeid_count": pre_execution_explicit_nodeid_count,
        "pre_execution_unmatched_positional_args": pending.get(
            "pre_execution_unmatched_positional_args"
        )
        if isinstance(pending.get("pre_execution_unmatched_positional_args"), list)
        else [],
        "pre_execution_selector_flags": pending.get("pre_execution_selector_flags")
        if isinstance(pending.get("pre_execution_selector_flags"), list)
        else [],
        "pre_execution_nodeids": pre_execution_nodeids,
        "collect_only": collect_only,
        "collect_only_duration_s": collect_only_duration_s,
        "prediction_overhead_s": collect_only_duration_s,
        "total_duration_with_prediction_overhead_s": (
            actual_duration_s + float(collect_only_duration_s)
            if collect_only_duration_s is not None
            else None
        ),
        "prediction_last_run_s": predictions["prediction_last_run_s"],
        "prediction_family_last_run_s": predictions["prediction_family_last_run_s"],
        "prediction_test_count_s": predictions["prediction_test_count_s"],
        "prediction_per_test_s": predictions["prediction_per_test_s"],
        "prediction_per_test_without_overhead_s": predictions[
            "prediction_per_test_without_overhead_s"
        ],
        "prediction_per_test_overhead_s": predictions["prediction_per_test_overhead_s"],
        "prediction_explicit_nodeid_lower_bound_s": predictions[
            "prediction_explicit_nodeid_lower_bound_s"
        ],
        "per_test_prediction_details": predictions["per_test_prediction_details"],
        "history_update_per_test_prediction_details": history_update_predictions[
            "per_test_prediction_details"
        ],
        "prediction_unknown_test_fallback_s": predictions[
            "prediction_unknown_test_fallback_s"
        ],
        "prediction_unknown_test_fallback_without_overhead_s": predictions[
            "prediction_unknown_test_fallback_without_overhead_s"
        ],
        "prediction_unknown_test_fallback_overhead_s": predictions[
            "prediction_unknown_test_fallback_overhead_s"
        ],
        "unknown_test_fallback_prediction_details": predictions[
            "unknown_test_fallback_prediction_details"
        ],
        "prediction_recommended_s": predictions["prediction_recommended_s"],
        "prediction_recommended_method": predictions[
            "prediction_recommended_method"
        ],
        "prediction_reliability": predictions["prediction_reliability"],
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "history_before": pending.get("history_before")
        if isinstance(pending.get("history_before"), dict)
        else {
            "test_count": None,
            "command_count": None,
            "project_test_median_s": predictions["project_test_median_s"],
            "project_overhead_median_s": predictions["project_overhead_median_s"],
            "project_unknown_test_median_s": predictions[
                "project_unknown_test_median_s"
            ],
            "recommended_method": predictions["prediction_recommended_method"],
            "reliability_level": predictions["prediction_reliability"]["level"],
        },
        "warnings": warnings,
    }
    if successful_pytest_run:
        try:
            with _history_lock(history_path):
                locked_history, locked_warning = _load_json(history_path)
                if locked_warning and locked_warning not in payload["warnings"]:
                    payload["warnings"].append(locked_warning)
                updated_history = update_pytest_history(
                    history=locked_history,
                    command=record.command,
                    total_duration_s=actual_duration_s,
                    tests=tests,
                    per_test_prediction_details=history_update_predictions[
                        "per_test_prediction_details"
                    ],
                )
                _write_json(history_path, updated_history)
        except Exception as exc:
            payload["history_updated"] = False
            payload["warnings"].append(f"failed to update history: {exc!r}")

    detailed_payload: dict[str, Any] = {
        **payload,
        "pytest_output": {
            "text": tool_result,
            "length_chars": len(tool_result),
            "truncated_by_tool": "... chars truncated" in tool_result,
        },
    }
    _write_json(record.directory / "prediction.json", detailed_payload)
    _append_jsonl(prediction_root / PREDICTIONS_FILENAME, payload)

    print(format_pytest_prediction_summary(payload), flush=True)
    return detailed_payload


def format_pytest_prediction_summary(payload: dict[str, Any]) -> str:
    """Return a concise realtime collect-mode summary line showing all strategies."""
    actual = payload.get("actual_duration_s")
    count = payload.get("collected_count")
    rel = payload.get("relative_error") or {}
    recommended = payload.get("prediction_recommended_method", "?")
    reliability = (payload.get("prediction_reliability") or {}).get("level", "?")
    overhead = payload.get("collect_only_duration_s")

    def _s(value: Any) -> str:
        return "?" if value is None else f"{float(value):.1f}s"

    def _pct(method: str) -> str:
        v = rel.get(method)
        return "?" if v is None else f"{float(v) * 100:+.1f}%"

    def _strat(label: str, pred_key: str, err_key: str) -> str:
        val = payload.get(pred_key)
        arrow = "\u2192" if (err_key == recommended and val is not None) else " "
        return f"{arrow}{label}={_s(val)}({_pct(err_key)})"

    parts = [
        f"[pytest-predict] #{payload.get('iteration')}",
        f"tests={count}",
        f"| {_s(actual)} actual",
        _strat("last", "prediction_last_run_s", "last_run"),
        _strat("fam", "prediction_family_last_run_s", "family_last_run"),
        _strat("count", "prediction_test_count_s", "test_count"),
        _strat("per", "prediction_per_test_s", "per_test"),
        _strat("unk", "prediction_unknown_test_fallback_s", "unknown_test_fallback"),
        f"| {reliability}",
    ]
    if overhead is not None:
        parts.insert(3, f"(probe {_s(overhead)})")
    return " ".join(parts)
