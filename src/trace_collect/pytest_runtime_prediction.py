"""Minimal pytest runtime prediction artifacts for trace collection.

The prototype is intentionally small: it records per-node pytest durations,
computes three simple predictions from bounded history, and writes attempt-local
JSON artifacts.  It does not change benchmark labels or use oracle data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import statistics
from typing import Any

from trace_collect.exec_classifier import classify_exec_tool_name

HIDDEN_RUNTIME_DIR_ARG = "__openclaw_pytest_runtime_dir"
HIDDEN_RUNTIME_ENABLED_ARG = "__openclaw_pytest_runtime_enabled"

HISTORY_FILENAME = "history.json"
PREDICTIONS_FILENAME = "predictions.jsonl"
RUNTIME_JSON_FILENAME = "pytest_runtime.json"
INSTRUMENTATION_FILENAME = "instrumentation.json"
PLUGIN_MODULE = "openclaw_pytest_runtime_plugin"
HISTORY_LIMIT = 5

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._+-]+")
_EXIT_CODE_RE = re.compile(r"Exit code:\s*(-?\d+)")
_PYTHONPATH_ASSIGN_RE = re.compile(
    r"(?i)(?:^|[\s;&|])(?:export\s+|env\s+|set\s+)?PYTHONPATH\s*="
)


@dataclass(slots=True)
class PytestRuntimeRecord:
    """Reference to one pytest runtime prediction invocation."""

    tool_call_id: str
    iteration: int
    directory: Path
    command: str
    working_directory: str | None = None


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


def _median(values: list[float]) -> float | None:
    clean = [float(v) for v in values if isinstance(v, (int, float)) and v >= 0]
    if not clean:
        return None
    return float(statistics.median(clean))


def _node_file(nodeid: str) -> str:
    return nodeid.split("::", 1)[0]


def normalize_pytest_command(command: str) -> str:
    """Return a conservative normalized command key for last-run baseline."""
    try:
        return " ".join(shlex.split(command, posix=(os.name != "nt")))
    except ValueError:
        return " ".join(command.split())


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
    iteration: int,
    tool_call_id: str,
    tool_name: str,
    tool_args: dict[str, Any],
) -> PytestRuntimeRecord | None:
    """Prepare artifact directory and annotate pytest tool args for ExecTool."""
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

    record = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "iteration": iteration,
        "tool_call_id": tool_call_id,
        "command": command,
        "working_directory": str(tool_args.get("working_dir") or tool_args.get("working_directory") or ""),
        "status": "pending_execution",
    }
    _write_json(invocation_dir / "pending.json", record)
    _write_json(invocation_dir / INSTRUMENTATION_FILENAME, record)
    return PytestRuntimeRecord(
        tool_call_id=tool_call_id,
        iteration=iteration,
        directory=invocation_dir,
        command=command,
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


def compute_pytest_predictions(
    *,
    history: dict[str, Any],
    command: str,
    nodeids: list[str],
) -> dict[str, Any]:
    """Compute Last Run, Test Count, and Per-Test Historical Sum."""
    project_median = global_history_median(history)
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
    test_count = (
        len(nodeids) * project_median
        if project_median is not None and nodeids
        else None
    )

    per_test_values: list[dict[str, Any]] = []
    per_test_total = 0.0
    per_test_available = True
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

    return {
        "command_key": command_key,
        "project_test_median_s": project_median,
        "prediction_last_run_s": last_run,
        "prediction_test_count_s": test_count,
        "prediction_per_test_s": per_test_total if per_test_available and nodeids else None,
        "per_test_prediction_details": per_test_values,
    }


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
) -> dict[str, Any]:
    """Return updated bounded history after this run."""
    updated = {
        "schema_version": 1,
        "updated_at": _utc_now(),
        "history_limit": HISTORY_LIMIT,
        "tests": dict(_history_tests(history)),
        "commands": dict(_history_commands(history)),
    }
    test_history = updated["tests"]
    for test in tests:
        nodeid = test.get("nodeid")
        duration = test.get("duration_s")
        if not isinstance(nodeid, str) or not isinstance(duration, (int, float)):
            continue
        if str(test.get("outcome") or "").lower() == "notrun":
            continue
        rec = dict(test_history.get(nodeid) or {})
        rec["durations"] = _bounded_append(rec.get("durations") or [], float(duration))
        rec["last_outcome"] = test.get("outcome")
        rec["last_seen_at"] = updated["updated_at"]
        test_history[nodeid] = rec

    command_key = normalize_pytest_command(command)
    command_history = updated["commands"]
    command_rec = dict(command_history.get(command_key) or {})
    command_rec["durations"] = _bounded_append(
        command_rec.get("durations") or [],
        total_duration_s,
    )
    command_rec["last_seen_at"] = updated["updated_at"]
    command_history[command_key] = command_rec
    return updated


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
    action_id: str,
    ts_start: float,
    ts_end: float,
    duration_ms: float,
    success: bool,
    tool_result: str,
    working_directory: str | None = None,
) -> dict[str, Any]:
    """Compute, persist, and print prediction results for one pytest run."""
    history_path = prediction_root / HISTORY_FILENAME
    history, history_warning = _load_json(history_path)
    runtime_payload, runtime_warning = _load_json(record.directory / RUNTIME_JSON_FILENAME)
    instrumentation, instrumentation_warning = _load_json(
        record.directory / INSTRUMENTATION_FILENAME
    )

    actual_duration_s = max(0.0, float(duration_ms) / 1000.0)
    tests = _load_runtime_tests(runtime_payload)
    nodeids = [str(test["nodeid"]) for test in tests]
    predictions = compute_pytest_predictions(
        history=history,
        command=record.command,
        nodeids=nodeids,
    )
    exit_code = (
        runtime_payload.get("exit_code")
        if isinstance(runtime_payload.get("exit_code"), int)
        else parse_exit_code(tool_result)
    )

    warnings = [
        warning
        for warning in (history_warning, runtime_warning, instrumentation_warning)
        if warning
    ]
    if not tests:
        warnings.append("pytest runtime JSON missing or contains no tests")

    prediction_values = {
        "last_run": predictions["prediction_last_run_s"],
        "test_count": predictions["prediction_test_count_s"],
        "per_test": predictions["prediction_per_test_s"],
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
        "schema_version": 1,
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
        "collected_count": (
            runtime_payload.get("collected_count")
            if isinstance(runtime_payload.get("collected_count"), int)
            else len(nodeids)
        ),
        "collected_tests": nodeids,
        "tests": tests,
        "prediction_last_run_s": predictions["prediction_last_run_s"],
        "prediction_test_count_s": predictions["prediction_test_count_s"],
        "prediction_per_test_s": predictions["prediction_per_test_s"],
        "per_test_prediction_details": predictions["per_test_prediction_details"],
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "history_before": {
            "test_count": len(_history_tests(history)),
            "command_count": len(_history_commands(history)),
            "project_test_median_s": predictions["project_test_median_s"],
        },
        "warnings": warnings,
    }
    _write_json(record.directory / "prediction.json", payload)
    _append_jsonl(prediction_root / PREDICTIONS_FILENAME, payload)

    if tests:
        updated_history = update_pytest_history(
            history=history,
            command=record.command,
            total_duration_s=actual_duration_s,
            tests=tests,
        )
        _write_json(history_path, updated_history)

    print(format_pytest_prediction_summary(payload), flush=True)
    return payload


def format_pytest_prediction_summary(payload: dict[str, Any]) -> str:
    """Return a concise realtime collect-mode summary line."""
    actual = payload.get("actual_duration_s")
    count = payload.get("collected_count")

    def _fmt(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.2f}s"

    rel = payload.get("relative_error") or {}
    per_test_rel = rel.get("per_test")
    per_test_err = "n/a" if per_test_rel is None else f"{float(per_test_rel) * 100:.1f}%"
    return (
        "[pytest-predict] "
        f"iter={payload.get('iteration')} tests={count} actual={_fmt(actual)} "
        f"last={_fmt(payload.get('prediction_last_run_s'))} "
        f"count={_fmt(payload.get('prediction_test_count_s'))} "
        f"per_test={_fmt(payload.get('prediction_per_test_s'))} "
        f"per_test_err={per_test_err}"
    )
