from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shlex
import statistics
from typing import Any, Iterable


SCHEMA_VERSION = 1
_PYTHON_NAMES = {
    "python",
    "python3",
    "python3.9",
    "python3.10",
    "python3.11",
    "python3.12",
}
_ORIGINAL_SIZE_RE = re.compile(r"Original size:\s*(\d+)\s*chars")
_PYTEST_COUNT_RE = re.compile(
    r"(?P<count>\d+)\s+"
    r"(?P<kind>passed|failed|error|errors|skipped|xfailed|xpassed|deselected)",
    re.IGNORECASE,
)
_EXIT_CODE_RE = re.compile(r"Exit code:\s*(-?\d+)")


@dataclass(slots=True)
class PriorBucket:
    tool_name: str
    tool_family: str
    operation: str
    workload_bucket: str | None
    durations: list[float] = field(default_factory=list)
    avg_cores: list[float] = field(default_factory=list)
    p50_cores: list[float] = field(default_factory=list)
    p90_cores: list[float] = field(default_factory=list)
    peak_cores: list[float] = field(default_factory=list)
    peak_memory_mb: list[float] = field(default_factory=list)
    disk_read_mb: list[float] = field(default_factory=list)
    disk_write_mb: list[float] = field(default_factory=list)
    net_rx_mb: list[float] = field(default_factory=list)
    net_tx_mb: list[float] = field(default_factory=list)
    context_switches: list[float] = field(default_factory=list)
    l1d_hit_rate: list[float] = field(default_factory=list)
    l1i_hit_rate: list[float] = field(default_factory=list)
    ipc: list[float] = field(default_factory=list)
    instructions_per_s: list[float] = field(default_factory=list)
    load_classes: Counter[str] = field(default_factory=Counter)
    resource_sample_count: int = 0

    @property
    def key(self) -> str:
        base = f"{self.tool_name}/{self.operation}"
        return f"{base}/{self.workload_bucket}" if self.workload_bucket else base


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return
    for line in lines:
        if not line.strip():
            continue
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            yield loaded


def _finite_nonnegative(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)) and value >= 0:
        return float(value)
    return None


def _percent_value(value: Any) -> float | None:
    finite = _finite_nonnegative(value)
    if finite is not None:
        return finite
    if not isinstance(value, str):
        return None
    text = value.strip().removesuffix("%")
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _resource_summary_from_profile(record: dict[str, Any]) -> dict[str, Any] | None:
    final = record.get("final_profile")
    if not isinstance(final, dict):
        return None

    runtime_s = _finite_nonnegative(final.get("total_wall_time_s"))
    if runtime_s is None:
        runtime_s = _finite_nonnegative(record.get("runtime_s"))
    if runtime_s is None:
        return None

    rss_peak_bytes = _finite_nonnegative(final.get("rss_peak_bytes"))
    read_bytes = _finite_nonnegative(final.get("total_read_bytes"))
    write_bytes = _finite_nonnegative(final.get("total_write_bytes"))
    sample_count = final.get("n_samples")
    if not isinstance(sample_count, int):
        sample_count = final.get("num_samples")
    return {
        "wall_time_s": runtime_s,
        "avg_cores": _finite_nonnegative(final.get("avg_effective_cores")),
        "p50_cores": _finite_nonnegative(
            final.get("p50_effective_cores")
            if "p50_effective_cores" in final
            else final.get("median_effective_cores")
        ),
        "p90_cores": _finite_nonnegative(final.get("p90_effective_cores")),
        "peak_cores": _finite_nonnegative(final.get("peak_effective_cores")),
        "peak_memory_mb": (rss_peak_bytes / (1024.0 * 1024.0))
        if rss_peak_bytes is not None
        else None,
        "read_mb": (read_bytes / (1024.0 * 1024.0))
        if read_bytes is not None
        else None,
        "write_mb": (write_bytes / (1024.0 * 1024.0))
        if write_bytes is not None
        else None,
        "load_class": final.get("preliminary_behavior")
        if isinstance(final.get("preliminary_behavior"), str)
        else None,
        "sample_count": sample_count if isinstance(sample_count, int) else None,
    }


def _parse_epoch(value: Any) -> float | None:
    finite = _finite_nonnegative(value)
    if finite is not None:
        return finite
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _mem_usage_mb(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    first = value.split("/", 1)[0].strip()
    match = re.match(r"([0-9.]+)\s*([KMGTP]?i?B)", first, re.IGNORECASE)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2).lower()
    scale = {
        "b": 1 / (1024 * 1024),
        "kb": 1 / 1024,
        "kib": 1 / 1024,
        "mb": 1,
        "mib": 1,
        "gb": 1024,
        "gib": 1024,
        "tb": 1024 * 1024,
        "tib": 1024 * 1024,
    }.get(unit)
    return amount * scale if scale is not None else None


def _resource_samples(resources_path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(resources_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list):
        return []
    result: list[dict[str, Any]] = []
    for row in samples:
        if not isinstance(row, dict):
            continue
        epoch = _parse_epoch(row.get("epoch"))
        if epoch is None:
            epoch = _parse_epoch(row.get("timestamp"))
        if epoch is None:
            continue
        copied = dict(row)
        copied["_epoch"] = epoch
        result.append(copied)
    result.sort(key=lambda item: float(item["_epoch"]))
    return result


def _resource_window(
    samples: list[dict[str, Any]],
    *,
    start_epoch: float,
    end_epoch: float,
) -> list[dict[str, Any]]:
    if end_epoch < start_epoch:
        start_epoch, end_epoch = end_epoch, start_epoch
    return [
        sample
        for sample in samples
        if start_epoch <= float(sample["_epoch"]) <= end_epoch
    ]


def _delta_from_cumulative(samples: list[dict[str, Any]], key: str) -> float | None:
    values = [
        value
        for value in (_finite_nonnegative(sample.get(key)) for sample in samples)
        if value is not None
    ]
    if len(values) < 2:
        return None
    return max(values[-1] - values[0], 0.0)


def _mean_resource_metric(samples: list[dict[str, Any]], key: str) -> float | None:
    values = [
        value
        for value in (_finite_nonnegative(sample.get(key)) for sample in samples)
        if value is not None
    ]
    return statistics.mean(values) if values else None


def _classify_load(
    avg_cores: float | None,
    disk_read: float | None,
    disk_write: float | None,
    net_rx: float | None,
    net_tx: float | None,
) -> str:
    io_bytes = sum(
        value
        for value in (disk_read, disk_write, net_rx, net_tx)
        if value is not None
    )
    if avg_cores is not None and avg_cores >= 2.0:
        return "cpu_parallel"
    if avg_cores is not None and avg_cores >= 0.6:
        return "cpu_serial"
    if io_bytes >= 1024 * 1024:
        return "io_active"
    return "idle_or_waiting"


def _summary_from_resource_window(
    samples: list[dict[str, Any]],
    *,
    duration_s: float,
) -> dict[str, Any] | None:
    if not samples:
        return None
    cpu_values = [
        value
        for value in (_percent_value(sample.get("cpu_percent")) for sample in samples)
        if value is not None
    ]
    rss_values = [
        value
        for value in (_mem_usage_mb(sample.get("mem_usage")) for sample in samples)
        if value is not None
    ]
    disk_read = _delta_from_cumulative(samples, "disk_read_bytes")
    disk_write = _delta_from_cumulative(samples, "disk_write_bytes")
    net_rx = _delta_from_cumulative(samples, "net_rx_bytes")
    net_tx = _delta_from_cumulative(samples, "net_tx_bytes")
    context_switches = _delta_from_cumulative(samples, "context_switches")
    cores = [value / 100.0 for value in cpu_values]
    avg_cores = statistics.mean(cores) if cores else None
    return {
        "wall_time_s": duration_s,
        "avg_cores": avg_cores,
        "p50_cores": _percentile(cores, 50) if cores else None,
        "p90_cores": _percentile(cores, 90) if cores else None,
        "peak_cores": max(cores) if cores else None,
        "peak_memory_mb": max(rss_values) if rss_values else None,
        "read_mb": (disk_read / (1024.0 * 1024.0)) if disk_read is not None else None,
        "write_mb": (disk_write / (1024.0 * 1024.0))
        if disk_write is not None
        else None,
        "net_rx_mb": (net_rx / (1024.0 * 1024.0)) if net_rx is not None else None,
        "net_tx_mb": (net_tx / (1024.0 * 1024.0)) if net_tx is not None else None,
        "context_switches": context_switches,
        "l1d_hit_rate": _mean_resource_metric(samples, "l1d_hit_rate"),
        "l1i_hit_rate": _mean_resource_metric(samples, "l1i_hit_rate"),
        "ipc": _mean_resource_metric(samples, "ipc"),
        "instructions_per_s": _mean_resource_metric(samples, "instructions_per_s"),
        "load_class": _classify_load(avg_cores, disk_read, disk_write, net_rx, net_tx),
        "sample_count": len(samples),
    }


def _percentile(values: list[float], p: float) -> float | None:
    clean = sorted(values)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    k = (p / 100.0) * (len(clean) - 1)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return clean[int(k)]
    return clean[lo] * (hi - k) + clean[hi] * (k - lo)


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _std(values: list[float]) -> float | None:
    return statistics.pstdev(values) if len(values) > 1 else None


def _confidence(sample_count: int, *, min_samples: int) -> str:
    if sample_count >= max(min_samples * 4, 100):
        return "high"
    if sample_count >= min_samples:
        return "medium"
    if sample_count > 0:
        return "low"
    return "unavailable"


def _duration_block(values: list[float]) -> dict[str, Any]:
    return {
        "p50_s": _percentile(values, 50),
        "p75_s": _percentile(values, 75),
        "p90_s": _percentile(values, 90),
        "p95_s": _percentile(values, 95),
        "mean_s": _mean(values),
        "std_s": _std(values),
        "sample_count": len(values),
    }


def _resource_block(bucket: PriorBucket) -> dict[str, Any]:
    load_class = (
        bucket.load_classes.most_common(1)[0][0] if bucket.load_classes else "unknown"
    )
    return {
        "load_class": load_class,
        "expected_cores": _percentile(bucket.avg_cores, 50),
        "avg_cores_p50": _percentile(bucket.avg_cores, 50),
        "p50_cores_p50": _percentile(bucket.p50_cores, 50),
        "p90_cores_p50": _percentile(bucket.p90_cores, 50),
        "peak_cores_p90": _percentile(bucket.peak_cores, 90),
        "peak_memory_mb": _percentile(bucket.peak_memory_mb, 90),
        "disk_read_mb_p90": _percentile(bucket.disk_read_mb, 90),
        "disk_write_mb_p90": _percentile(bucket.disk_write_mb, 90),
        "net_rx_mb_p90": _percentile(bucket.net_rx_mb, 90),
        "net_tx_mb_p90": _percentile(bucket.net_tx_mb, 90),
        "context_switches_p90": _percentile(bucket.context_switches, 90),
        "l1d_hit_rate_p50": _percentile(bucket.l1d_hit_rate, 50),
        "l1i_hit_rate_p50": _percentile(bucket.l1i_hit_rate, 50),
        "ipc_p50": _percentile(bucket.ipc, 50),
        "instructions_per_s_p50": _percentile(bucket.instructions_per_s, 50),
        "resource_sample_count": bucket.resource_sample_count,
        "load_class_counts": dict(bucket.load_classes),
    }


def _strip_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_none(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_strip_none(item) for item in value if item is not None]
    return value


def _pip_workload_bucket(package_count: Any) -> str | None:
    count = _finite_nonnegative(package_count)
    if count is None:
        return None
    if count <= 1:
        return "1-package"
    if count <= 10:
        return "2-10-packages"
    return "10-plus-packages"


def _count_pip_packages(command: str) -> int | None:
    tokens = _command_tokens(command)
    if len(tokens) >= 2 and _basename(tokens[0]) in {"pip", "pip3"}:
        tokens = tokens[2:] if tokens[1] == "install" else []
    elif len(tokens) >= 4 and _basename(tokens[0]) in _PYTHON_NAMES:
        tokens = tokens[4:] if tokens[1:4] == ["-m", "pip", "install"] else []
    else:
        return None
    count = 0
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token in {"-r", "--requirement", "-c", "--constraint"}:
            count += 1
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        count += 1
    return count if count > 0 else None


def _python_workload_bucket(row: dict[str, Any]) -> str | None:
    timeout_s = _finite_nonnegative(row.get("timeout_s"))
    if timeout_s is not None:
        if timeout_s <= 30:
            return "timeout-0-30s"
        if timeout_s <= 120:
            return "timeout-31-120s"
        return "timeout-120-plus-s"
    return None


def _pytest_workload_bucket(count_value: Any) -> str | None:
    count = _finite_nonnegative(count_value)
    if count is None:
        return None
    if count <= 10:
        return "1-10-tests"
    if count <= 100:
        return "11-100-tests"
    if count <= 500:
        return "101-500-tests"
    return "500-plus-tests"


def _pytest_count_from_preview(row: dict[str, Any]) -> int | None:
    preview = row.get("result_preview")
    if not isinstance(preview, str):
        return None
    total = 0
    for match in _PYTEST_COUNT_RE.finditer(preview):
        total += int(match.group("count"))
    return total if total > 0 else None


def _size_bucket(chars: int | None) -> str | None:
    if chars is None:
        return None
    if chars <= 4_096:
        return "0-4k-chars"
    if chars <= 32_768:
        return "4k-32k-chars"
    if chars <= 262_144:
        return "32k-256k-chars"
    return "256k-plus-chars"


def _preview_size(row: dict[str, Any]) -> int | None:
    preview = row.get("result_preview")
    if not isinstance(preview, str):
        return None
    match = _ORIGINAL_SIZE_RE.search(preview)
    if match:
        return int(match.group(1))
    return len(preview)


def _tool_call_descriptor(row: dict[str, Any]) -> tuple[str, str, str, str | None] | None:
    tool = row.get("tool")
    if not isinstance(tool, str):
        return None
    input_payload = row.get("input")
    if not isinstance(input_payload, dict):
        input_payload = {}
    command = str(input_payload.get("command") or "")
    if _looks_like_pip_install(command):
        return (
            "pip",
            "package_install",
            "install",
            _pip_workload_bucket(_count_pip_packages(command)),
        )
    if _looks_like_pytest(command):
        return (
            "pytest",
            "test_runner",
            "run_tests",
            _pytest_workload_bucket(_pytest_count_from_preview(row)),
        )
    if _looks_like_python_script(command):
        return ("python", "script_execution", "run_script", None)

    if tool in {"read_file", "write_file", "edit_file"}:
        operation = tool
        return (
            tool,
            "file_processing",
            operation,
            _size_bucket(_preview_size(row)),
        )
    if tool == "list_dir":
        recursive = bool(input_payload.get("recursive"))
        return (
            "list_dir",
            "file_processing",
            "list_dir",
            "recursive" if recursive else "single-dir",
        )
    if tool.startswith("exec-"):
        name = tool.removeprefix("exec-") or "exec"
        family = "generic_process"
        operation = "run"
        if name in {"find", "grep", "rg"}:
            family = "file_processing"
            operation = "search_files"
        elif name in {"ls", "cat", "sed", "head", "tail"}:
            family = "file_processing"
            operation = name
        return (name, family, operation, None)
    return (tool, "generic_process", "run", None)


def _bucket_for_prediction_file(path: Path, row: dict[str, Any]) -> tuple[str, str, str, str | None] | None:
    parts = set(path.parts)
    if "pip_runtime" in parts:
        return (
            "pip",
            "package_install",
            "install",
            _pip_workload_bucket(row.get("package_count")),
        )
    if "python_script_runtime" in parts:
        return (
            "python",
            "script_execution",
            "run_script",
            _python_workload_bucket(row),
        )
    if "pytest_runtime" in parts:
        count_value = row.get("collected_count")
        if count_value is None:
            count_value = row.get("pre_execution_collected_count")
        return (
            "pytest",
            "test_runner",
            "run_tests",
            _pytest_workload_bucket(count_value),
        )
    return None


def _ensure_bucket(
    buckets: dict[str, PriorBucket],
    *,
    tool_name: str,
    tool_family: str,
    operation: str,
    workload_bucket: str | None,
) -> PriorBucket:
    if not operation:
        key = f"{tool_name}/{workload_bucket}" if workload_bucket else tool_name
    else:
        key = (
            f"{tool_name}/{operation}/{workload_bucket}"
            if workload_bucket
            else f"{tool_name}/{operation}"
        )
    bucket = buckets.get(key)
    if bucket is None:
        bucket = PriorBucket(
            tool_name=tool_name,
            tool_family=tool_family,
            operation=operation,
            workload_bucket=workload_bucket,
        )
        buckets[key] = bucket
    return bucket


def _bucket_specs(
    *,
    tool_name: str,
    tool_family: str,
    operation: str,
    workload_bucket: str | None,
) -> list[tuple[str, str, str, str | None]]:
    specs = [
        (tool_name, tool_family, operation, workload_bucket),
        (tool_name, tool_family, operation, None),
    ]
    if tool_family != tool_name:
        specs.append((tool_family, tool_family, operation, workload_bucket))
        specs.append((tool_family, tool_family, operation, None))
    if tool_family != "generic_process":
        specs.append(("generic_process", "generic_process", operation, None))
    specs.append(("generic_process", "generic_process", "", None))

    deduped: list[tuple[str, str, str, str | None]] = []
    seen: set[tuple[str, str, str, str | None]] = set()
    for spec in specs:
        if spec not in seen:
            seen.add(spec)
            deduped.append(spec)
    return deduped


def _add_duration_observation(
    buckets: dict[str, PriorBucket],
    *,
    tool_name: str,
    tool_family: str,
    operation: str,
    workload_bucket: str | None,
    duration_s: float,
) -> None:
    for spec_tool, spec_family, spec_operation, spec_workload in _bucket_specs(
        tool_name=tool_name,
        tool_family=tool_family,
        operation=operation,
        workload_bucket=workload_bucket,
    ):
        bucket = _ensure_bucket(
            buckets,
            tool_name=spec_tool,
            tool_family=spec_family,
            operation=spec_operation,
            workload_bucket=spec_workload,
        )
        bucket.durations.append(duration_s)


def _add_resource_observation_to_buckets(
    buckets: dict[str, PriorBucket],
    *,
    tool_name: str,
    tool_family: str,
    operation: str,
    workload_bucket: str | None,
    summary: dict[str, Any],
) -> None:
    for spec_tool, spec_family, spec_operation, spec_workload in _bucket_specs(
        tool_name=tool_name,
        tool_family=tool_family,
        operation=operation,
        workload_bucket=workload_bucket,
    ):
        bucket = _ensure_bucket(
            buckets,
            tool_name=spec_tool,
            tool_family=spec_family,
            operation=spec_operation,
            workload_bucket=spec_workload,
        )
        _add_resource_observation(bucket, summary)


def _attempt_key(path: Path) -> str:
    for parent in path.parents:
        if parent.name.startswith("attempt_"):
            return str(parent)
    return str(path.parent)


def _duration_key(
    path: Path,
    *,
    tool_name: str,
    operation: str,
) -> tuple[str, str, str]:
    return (_attempt_key(path), tool_name, operation)


def _observation_id(path: Path, row: dict[str, Any]) -> str:
    for key in ("tool_call_id", "id", "action_id", "run_id", "invocation_id"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return f"{_attempt_key(path)}::{value}"
    timestamp = row.get("timestamp") or row.get("ts_start")
    return f"{_attempt_key(path)}::{timestamp!r}::{len(json.dumps(row, sort_keys=True))}"


def _iter_prediction_files(root: Path) -> Iterable[Path]:
    patterns = (
        "pip_runtime/predictions.jsonl",
        "python_script_runtime/predictions.jsonl",
        "pytest_runtime/predictions.jsonl",
    )
    if root.is_file() and root.name == "predictions.jsonl":
        yield root
        return
    for pattern in patterns:
        yield from sorted(root.rglob(pattern))


def collect_prediction_observations(
    root: Path,
    *,
    seen_observations: set[str],
    duration_keys: set[tuple[str, str, str]],
) -> dict[str, PriorBucket]:
    buckets: dict[str, PriorBucket] = {}
    for path in _iter_prediction_files(root):
        for row in _jsonl_rows(path):
            if row.get("success") is not True:
                continue
            duration_s = _finite_nonnegative(row.get("actual_duration_s"))
            if duration_s is None:
                continue
            descriptor = _bucket_for_prediction_file(path, row)
            if descriptor is None:
                continue
            observation_id = _observation_id(path, row)
            if observation_id in seen_observations:
                continue
            seen_observations.add(observation_id)
            tool_name, tool_family, operation, workload_bucket = descriptor
            duration_keys.add(
                _duration_key(path, tool_name=tool_name, operation=operation)
            )
            _add_duration_observation(
                buckets,
                tool_name=tool_name,
                tool_family=tool_family,
                operation=operation,
                workload_bucket=workload_bucket,
                duration_s=duration_s,
            )
    return buckets


def _iter_tool_call_files(root: Path) -> Iterable[Path]:
    if root.is_file() and root.name == "tool_calls.json":
        yield root
        return
    yield from sorted(root.rglob("tool_calls.json"))


def _load_tool_call_rows(path: Path) -> list[dict[str, Any]]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    if isinstance(loaded, list):
        return [row for row in loaded if isinstance(row, dict)]
    if isinstance(loaded, dict):
        rows = loaded.get("tool_calls")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def collect_tool_call_observations(
    root: Path,
    buckets: dict[str, PriorBucket],
    *,
    seen_observations: set[str],
    suppress_duration_keys: set[tuple[str, str, str]],
    duration_keys: set[tuple[str, str, str]],
    tool_call_counts: Counter[tuple[str, str, str]],
    tool_call_resource_counts: Counter[tuple[str, str, str]],
    tool_call_resource_ids: set[str],
) -> None:
    for path in _iter_tool_call_files(root):
        resources = _resource_samples(path.with_name("resources.json"))
        for row in _load_tool_call_rows(path):
            duration_ms = _finite_nonnegative(row.get("duration_ms"))
            if duration_ms is None:
                continue
            if not _tool_call_success(row):
                continue
            descriptor = _tool_call_descriptor(row)
            if descriptor is None:
                continue
            observation_id = _observation_id(path, row)
            tool_name, tool_family, operation, workload_bucket = descriptor
            duration_key = _duration_key(path, tool_name=tool_name, operation=operation)
            already_seen = observation_id in seen_observations
            if already_seen and duration_key not in suppress_duration_keys:
                continue
            if not already_seen:
                seen_observations.add(observation_id)
            tool_call_counts[duration_key] += 1
            duration_s = duration_ms / 1000.0
            if not already_seen and duration_key not in suppress_duration_keys:
                duration_keys.add(duration_key)
                _add_duration_observation(
                    buckets,
                    tool_name=tool_name,
                    tool_family=tool_family,
                    operation=operation,
                    workload_bucket=workload_bucket,
                    duration_s=duration_s,
                )
            resource_summary = _tool_call_resource_summary(
                row,
                resources=resources,
                duration_s=duration_s,
            )
            if resource_summary is not None:
                tool_call_resource_counts[duration_key] += 1
                tool_call_resource_ids.add(observation_id)
                _add_resource_observation_to_buckets(
                    buckets,
                    tool_name=tool_name,
                    tool_family=tool_family,
                    operation=operation,
                    workload_bucket=workload_bucket,
                    summary=resource_summary,
                )


def _tool_call_resource_summary(
    row: dict[str, Any],
    *,
    resources: list[dict[str, Any]],
    duration_s: float,
) -> dict[str, Any] | None:
    if not resources:
        return None
    start_epoch = _parse_epoch(row.get("timestamp"))
    end_epoch = _parse_epoch(row.get("end_timestamp"))
    if start_epoch is None or end_epoch is None:
        return None
    return _summary_from_resource_window(
        _resource_window(resources, start_epoch=start_epoch, end_epoch=end_epoch),
        duration_s=duration_s,
    )


def _tool_call_success(row: dict[str, Any]) -> bool:
    success = row.get("success")
    if success is False:
        return False
    exit_code = row.get("exit_code")
    if isinstance(exit_code, int) and exit_code != 0:
        return False
    status = row.get("status")
    if isinstance(status, str) and status.lower() in {"failed", "error", "timeout"}:
        return False
    preview = row.get("result_preview")
    if not isinstance(preview, str):
        return True
    match = _EXIT_CODE_RE.search(preview)
    return match is None or int(match.group(1)) == 0


def _command_text(row: dict[str, Any]) -> str:
    command = row.get("command_string")
    if not isinstance(command, str) or not command.strip():
        raw = row.get("command")
        if isinstance(raw, list):
            command = " ".join(str(part) for part in raw)
        else:
            command = str(raw or "")
    return " ".join(command.split())


def _basename(token: str) -> str:
    return token.replace("\\", "/").rsplit("/", 1)[-1].lower()


def _command_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return command.split()


def _looks_like_pip_install(command: str) -> bool:
    tokens = _command_tokens(command)
    if len(tokens) >= 2 and _basename(tokens[0]) in {"pip", "pip3"}:
        return tokens[1] == "install"
    if len(tokens) >= 4 and _basename(tokens[0]) in _PYTHON_NAMES:
        return tokens[1:4] == ["-m", "pip", "install"]
    return False


def _looks_like_python_script(command: str) -> bool:
    tokens = _command_tokens(command)
    if len(tokens) >= 3 and _basename(tokens[0]) == "timeout":
        tokens = tokens[2:]
    if len(tokens) < 2 or _basename(tokens[0]) not in _PYTHON_NAMES:
        return False
    if tokens[1] in {"-m", "-c"}:
        return False
    return any(token.endswith(".py") for token in tokens[1:3])


def _looks_like_pytest(command: str) -> bool:
    tokens = _command_tokens(command)
    if not tokens:
        return False
    if _basename(tokens[0]) == "pytest":
        return True
    return (
        len(tokens) >= 3
        and _basename(tokens[0]) in _PYTHON_NAMES
        and tokens[1:3] == ["-m", "pytest"]
    )


def _descriptor_from_profile_row(row: dict[str, Any]) -> tuple[str, str, str]:
    command = _command_text(row)
    if _looks_like_pip_install(command):
        return ("pip", "package_install", "install")
    if _looks_like_python_script(command):
        return ("python", "script_execution", "run_script")
    if _looks_like_pytest(command):
        return ("pytest", "test_runner", "run_tests")
    return ("generic_process", "generic_process", "run")


def collect_profile_observations(
    root: Path,
    buckets: dict[str, PriorBucket],
    *,
    seen_observations: set[str],
    duration_keys: set[tuple[str, str, str]],
    tool_call_counts: Counter[tuple[str, str, str]],
    tool_call_resource_counts: Counter[tuple[str, str, str]],
    tool_call_resource_ids: set[str],
) -> None:
    profile_paths = sorted(root.rglob("tool_profiles/*/profile.jsonl"))
    profile_paths.extend(sorted(root.rglob("tool_scheduler/*/profile.jsonl")))
    for path in profile_paths:
        for row in _jsonl_rows(path):
            if not _profile_success(row):
                continue
            summary = _resource_summary_from_profile(row)
            if summary is None:
                continue
            duration_s = _finite_nonnegative(summary.get("wall_time_s"))
            if duration_s is None:
                continue
            tool_name, tool_family, operation = _descriptor_from_profile_row(row)
            final = row.get("final_profile") if isinstance(row.get("final_profile"), dict) else {}
            profile_tool = final.get("short_tool")
            workload_bucket = "short" if profile_tool is True else None
            observation_id = _observation_id(path, row)
            duration_key = _duration_key(path, tool_name=tool_name, operation=operation)
            if observation_id not in seen_observations and duration_key not in duration_keys:
                seen_observations.add(observation_id)
                duration_keys.add(duration_key)
                _add_duration_observation(
                    buckets,
                    tool_name=tool_name,
                    tool_family=tool_family,
                    operation=operation,
                    workload_bucket=workload_bucket,
                    duration_s=duration_s,
                )
            tool_call_interval_covers_profile = (
                observation_id in tool_call_resource_ids
                or (
                    tool_call_counts[duration_key] == 1
                    and tool_call_resource_counts[duration_key] == 1
                )
            )
            if not tool_call_interval_covers_profile:
                _add_resource_observation_to_buckets(
                    buckets,
                    tool_name=tool_name,
                    tool_family=tool_family,
                    operation=operation,
                    workload_bucket=workload_bucket,
                    summary=summary,
                )


def _profile_success(row: dict[str, Any]) -> bool:
    success = row.get("success")
    if success is False:
        return False
    exit_code = row.get("exit_code")
    if isinstance(exit_code, int) and exit_code != 0:
        return False
    final = row.get("final_profile")
    if isinstance(final, dict):
        final_exit_code = final.get("exit_code")
        if isinstance(final_exit_code, int) and final_exit_code != 0:
            return False
    status = row.get("status")
    return not (
        isinstance(status, str) and status.lower() in {"failed", "error", "timeout"}
    )


def _add_resource_observation(bucket: PriorBucket, summary: dict[str, Any]) -> None:
    for attr, target in (
        ("avg_cores", bucket.avg_cores),
        ("p50_cores", bucket.p50_cores),
        ("p90_cores", bucket.p90_cores),
        ("peak_cores", bucket.peak_cores),
        ("peak_memory_mb", bucket.peak_memory_mb),
        ("read_mb", bucket.disk_read_mb),
        ("write_mb", bucket.disk_write_mb),
        ("net_rx_mb", bucket.net_rx_mb),
        ("net_tx_mb", bucket.net_tx_mb),
        ("context_switches", bucket.context_switches),
        ("l1d_hit_rate", bucket.l1d_hit_rate),
        ("l1i_hit_rate", bucket.l1i_hit_rate),
        ("ipc", bucket.ipc),
        ("instructions_per_s", bucket.instructions_per_s),
    ):
        value = _finite_nonnegative(summary.get(attr))
        if value is not None:
            target.append(value)
    load_class = summary.get("load_class")
    if isinstance(load_class, str) and load_class:
        bucket.load_classes[load_class] += 1
    bucket.resource_sample_count += 1


def build_common_kb(root: Path, *, min_samples: int = 1) -> dict[str, Any]:
    seen_observations: set[str] = set()
    duration_keys: set[tuple[str, str, str]] = set()
    tool_call_counts: Counter[tuple[str, str, str]] = Counter()
    tool_call_resource_counts: Counter[tuple[str, str, str]] = Counter()
    tool_call_resource_ids: set[str] = set()
    buckets = collect_prediction_observations(
        root,
        seen_observations=seen_observations,
        duration_keys=duration_keys,
    )
    prediction_duration_keys = set(duration_keys)
    collect_tool_call_observations(
        root,
        buckets,
        seen_observations=seen_observations,
        suppress_duration_keys=prediction_duration_keys,
        duration_keys=duration_keys,
        tool_call_counts=tool_call_counts,
        tool_call_resource_counts=tool_call_resource_counts,
        tool_call_resource_ids=tool_call_resource_ids,
    )
    collect_profile_observations(
        root,
        buckets,
        seen_observations=seen_observations,
        duration_keys=duration_keys,
        tool_call_counts=tool_call_counts,
        tool_call_resource_counts=tool_call_resource_counts,
        tool_call_resource_ids=tool_call_resource_ids,
    )
    priors: dict[str, Any] = {}
    for key, bucket in sorted(buckets.items()):
        sample_count = len(bucket.durations)
        if sample_count < min_samples:
            continue
        prior = {
            "tool_name": bucket.tool_name,
            "tool_family": bucket.tool_family,
            "operation": bucket.operation,
            "workload_bucket": bucket.workload_bucket,
            "duration": _duration_block(bucket.durations),
            "resources": _resource_block(bucket),
            "sample_count": sample_count,
            "confidence": _confidence(sample_count, min_samples=min_samples),
            "quality": {
                "min_samples": min_samples,
                "outlier_policy": "none",
                "source_version": "runtime-common-builder-v1",
                "privacy_level": "aggregate",
            },
        }
        priors[key] = _strip_none(prior)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "source": {
            "kind": "historical_runtime_artifacts",
        },
        "priors": priors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a de-identified Common runtime KB from trace artifacts.",
    )
    parser.add_argument("root", type=Path, help="Root containing attempt/runtime artifacts")
    parser.add_argument(
        "--output",
        required=True,
        help="Output Common KB JSON path, or '-' for stdout",
    )
    parser.add_argument("--min-samples", type=int, default=1)
    args = parser.parse_args()

    if args.min_samples <= 0:
        raise SystemExit("--min-samples must be positive")
    kb = build_common_kb(args.root, min_samples=args.min_samples)
    text = json.dumps(kb, indent=2, ensure_ascii=False)
    if args.output == "-":
        print(text)
    else:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(
            f"wrote {len(kb['priors'])} priors to {output}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
