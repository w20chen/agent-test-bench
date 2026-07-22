"""Unified runtime knowledge bases for tool duration and resource priors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any


PERSONAL_KB_FILENAME = "personal_runtime_knowledge.json"
COMMON_KB_FILENAME = "runtime_common_kb_swe_rebench_p1.json"
COMMON_KB_ENV = "TOOL_RUNTIME_COMMON_KB"
HISTORY_LIMIT = 10
SCHEMA_VERSION = 1


@dataclass(slots=True)
class RuntimePrediction:
    """Common prediction output shared by tool-specific predictors."""

    duration_p50_s: float | None
    duration_p90_s: float | None
    load_class: str | None
    expected_cores: float | None
    peak_memory_mb: float | None
    confidence: str
    prediction_source: str
    source_detail: str | None = None
    sample_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ResourceSummary:
    """Whole-process resource summary for one completed invocation."""

    wall_time_s: float | None = None
    avg_cores: float | None = None
    p50_cores: float | None = None
    p90_cores: float | None = None
    peak_cores: float | None = None
    peak_memory_mb: float | None = None
    read_mb: float | None = None
    write_mb: float | None = None
    load_class: str | None = None
    sample_count: int | None = None
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_personal_kb_path(history_root: Path | None, prediction_root: Path) -> Path:
    """Resolve the Personal KB path.

    When *history_root* points to a repo-level directory (e.g.
    ``runtime_kb/repo/repo_<name>/pip/``), the Personal KB lives alongside
    it at ``runtime_kb/repo/repo_<name>/personal_runtime_knowledge.json``
    so that all tools within the same repo share a single knowledge base.

    Falls back to *prediction_root* when no shared history directory is
    configured.
    """
    if history_root is not None and history_root.parent.parent != history_root.parent:
        return history_root.parent / PERSONAL_KB_FILENAME
    return prediction_root / PERSONAL_KB_FILENAME


def repo_personal_kb_path(run_dir: Path, repo_key: str) -> Path:
    """Repo-level Personal KB path, independent of any tool subdirectory.

    ``repo_key`` is a safe directory name for the repository (e.g. the
    output of ``_safe_scope_dir_name``).
    """
    return run_dir / "runtime_kb" / "repo" / repo_key / PERSONAL_KB_FILENAME


def default_common_kb_path() -> Path | None:
    value = os.environ.get(COMMON_KB_ENV)
    if value:
        return Path(value)
    repo_default = Path(__file__).resolve().parents[2] / COMMON_KB_FILENAME
    return repo_default if repo_default.exists() else None


def load_json_object(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_json_object(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def bounded_append(values: Any, value: float, *, limit: int = HISTORY_LIMIT) -> list[float]:
    existing = [
        float(item)
        for item in (values if isinstance(values, list) else [])
        if isinstance(item, (int, float)) and math.isfinite(float(item)) and item >= 0
    ]
    existing.append(float(value))
    return existing[-limit:]


def percentile(values: list[float], p: float) -> float | None:
    clean = sorted(
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and value >= 0
    )
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


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def resource_summary_from_profile(record: dict[str, Any]) -> ResourceSummary | None:
    """Parse a whole-run profiler/scheduler JSONL row into a resource summary."""

    final = record.get("final_profile")
    if not isinstance(final, dict):
        return None
    runtime_s = _float_or_none(final.get("total_wall_time_s"))
    if runtime_s is None:
        runtime_s = _float_or_none(record.get("runtime_s"))

    avg_cores = _float_or_none(final.get("avg_effective_cores"))
    p50_cores = _float_or_none(final.get("p50_effective_cores"))
    if p50_cores is None:
        p50_cores = _float_or_none(final.get("median_effective_cores"))
    p90_cores = _float_or_none(final.get("p90_effective_cores"))
    peak_cores = _float_or_none(final.get("peak_effective_cores"))

    rss_peak_bytes = _float_or_none(final.get("rss_peak_bytes"))
    read_bytes = _float_or_none(final.get("total_read_bytes"))
    write_bytes = _float_or_none(final.get("total_write_bytes"))
    sample_count = final.get("n_samples")
    if not isinstance(sample_count, int):
        sample_count = final.get("num_samples")
    if not isinstance(sample_count, int):
        sample_count = None

    return ResourceSummary(
        wall_time_s=runtime_s,
        avg_cores=avg_cores,
        p50_cores=p50_cores,
        p90_cores=p90_cores,
        peak_cores=peak_cores,
        peak_memory_mb=(rss_peak_bytes / (1024.0 * 1024.0))
        if rss_peak_bytes is not None
        else None,
        read_mb=(read_bytes / (1024.0 * 1024.0)) if read_bytes is not None else None,
        write_mb=(write_bytes / (1024.0 * 1024.0)) if write_bytes is not None else None,
        load_class=final.get("preliminary_behavior")
        if isinstance(final.get("preliminary_behavior"), str)
        else None,
        sample_count=sample_count,
        source="tool_profile_final",
    )


def _records(container: dict[str, Any], key: str) -> dict[str, Any]:
    value = container.get(key)
    return value if isinstance(value, dict) else {}


def _duration_prediction(rec: dict[str, Any], *, source: str) -> RuntimePrediction | None:
    durations = rec.get("durations")
    clean = (
        [
            float(value)
            for value in durations
            if isinstance(value, (int, float)) and math.isfinite(float(value)) and value >= 0
        ]
        if isinstance(durations, list)
        else []
    )
    if not clean:
        return None
    resources = rec.get("resources")
    if not isinstance(resources, dict):
        resources = {}
    return RuntimePrediction(
        duration_p50_s=percentile(clean, 50),
        duration_p90_s=percentile(clean, 90),
        load_class=resources.get("load_class")
        if isinstance(resources.get("load_class"), str)
        else None,
        expected_cores=percentile(resources.get("avg_cores", []), 50)
        if isinstance(resources.get("avg_cores"), list)
        else None,
        peak_memory_mb=percentile(resources.get("peak_memory_mb", []), 90)
        if isinstance(resources.get("peak_memory_mb"), list)
        else None,
        confidence="high" if len(clean) >= 3 else "medium",
        prediction_source=source,
        source_detail=rec.get("last_seen_at")
        if isinstance(rec.get("last_seen_at"), str)
        else None,
        sample_count=len(clean),
    )


def lookup_personal_prediction(
    kb: dict[str, Any],
    *,
    tool_name: str,
    tool_family: str,
    operation: str,
    normalized_command: str | None = None,
) -> RuntimePrediction | None:
    """Lookup exact-command personal history for duration prediction.

    Tool and family aggregates are retained in Personal KB for resource
    analysis, but cold-start duration fallback must come from Common KB.
    """

    _ = (tool_name, tool_family, operation)
    commands = _records(kb, "commands")
    if normalized_command and isinstance(commands.get(normalized_command), dict):
        pred = _duration_prediction(commands[normalized_command], source="personal_command")
        if pred is not None:
            return pred
    return None


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _confidence_from_common_prior(prior: dict[str, Any]) -> str:
    confidence = prior.get("confidence")
    if isinstance(confidence, str):
        return confidence
    if isinstance(confidence, dict):
        duration_confidence = confidence.get("duration")
        if isinstance(duration_confidence, str):
            return duration_confidence
    return "low"


def _common_prior_to_prediction(prior: dict[str, Any], *, source: str) -> RuntimePrediction:
    duration = prior.get("duration")
    resources = prior.get("resources")
    if not isinstance(duration, dict):
        duration = prior
    if not isinstance(resources, dict):
        resources = prior
    counts = prior.get("counts")
    sample_count = prior.get("sample_count")
    if not isinstance(sample_count, int) and isinstance(counts, dict):
        sample_count = counts.get("duration")
    if not isinstance(sample_count, int):
        sample_count = duration.get("sample_count")
    return RuntimePrediction(
        duration_p50_s=_float_or_none(_first_present(duration, "p50_s", "p50")),
        duration_p90_s=_float_or_none(_first_present(duration, "p90_s", "p90")),
        load_class=resources.get("load_class")
        if isinstance(resources.get("load_class"), str)
        else None,
        expected_cores=_float_or_none(
            _first_present(resources, "expected_cores", "avg_cores")
        ),
        peak_memory_mb=_float_or_none(
            _first_present(resources, "peak_memory_mb", "rss_p90_mb")
        ),
        confidence=_confidence_from_common_prior(prior),
        prediction_source=source,
        source_detail=prior.get("source_detail")
        if isinstance(prior.get("source_detail"), str)
        else None,
        sample_count=sample_count,
    )


def _lookup_layered_operation(
    node: dict[str, Any],
    *,
    workload_bucket: str | None,
) -> tuple[str, dict[str, Any]] | None:
    if workload_bucket:
        buckets = node.get("buckets")
        if isinstance(buckets, dict):
            prior = buckets.get(workload_bucket)
            if isinstance(prior, dict):
                return f"buckets/{workload_bucket}", prior
    prior = node.get("default")
    if isinstance(prior, dict) and prior:
        return "default", prior
    return None


def _lookup_v2_common_prediction(
    kb: dict[str, Any],
    *,
    tool_name: str,
    tool_family: str,
    operation: str,
    workload_bucket: str | None,
) -> RuntimePrediction | None:
    by_tool = kb.get("by_tool")
    if isinstance(by_tool, dict):
        tool_node = by_tool.get(tool_name)
        if isinstance(tool_node, dict):
            operation_node = tool_node.get(operation)
            if isinstance(operation_node, dict):
                found = _lookup_layered_operation(
                    operation_node,
                    workload_bucket=workload_bucket,
                )
                if found is not None:
                    detail, prior = found
                    return _common_prior_to_prediction(
                        prior,
                        source=f"common:by_tool/{tool_name}/{operation}/{detail}",
                    )

    by_family = kb.get("by_family")
    if isinstance(by_family, dict):
        family_node = by_family.get(tool_family)
        if isinstance(family_node, dict):
            operation_node = family_node.get(operation)
            if isinstance(operation_node, dict):
                found = _lookup_layered_operation(
                    operation_node,
                    workload_bucket=workload_bucket,
                )
                if found is not None:
                    detail, prior = found
                    return _common_prior_to_prediction(
                        prior,
                        source=f"common:by_family/{tool_family}/{operation}/{detail}",
                    )

    by_operation = kb.get("by_operation")
    if isinstance(by_operation, dict):
        operation_node = by_operation.get(operation)
        if isinstance(operation_node, dict):
            prior = operation_node.get("default")
            if isinstance(prior, dict) and prior:
                return _common_prior_to_prediction(
                    prior,
                    source=f"common:by_operation/{operation}/default",
                )

    global_prior = kb.get("global")
    if isinstance(global_prior, dict) and global_prior:
        return _common_prior_to_prediction(global_prior, source="common:global")

    return None


def lookup_common_prediction(
    kb: dict[str, Any],
    *,
    tool_name: str,
    tool_family: str,
    operation: str,
    workload_bucket: str | None = None,
) -> RuntimePrediction | None:
    """Read-only Common fallback: tool -> family -> generic."""

    v2_prediction = _lookup_v2_common_prediction(
        kb,
        tool_name=tool_name,
        tool_family=tool_family,
        operation=operation,
        workload_bucket=workload_bucket,
    )
    if v2_prediction is not None:
        return v2_prediction

    candidates: list[str] = []
    if workload_bucket:
        candidates.append(f"{tool_name}/{operation}/{workload_bucket}")
    candidates.append(f"{tool_name}/{operation}")
    if workload_bucket:
        candidates.append(f"{tool_family}/{operation}/{workload_bucket}")
    candidates.append(f"{tool_family}/{operation}")
    candidates.append(f"generic_process/{operation}")
    candidates.append("generic_process")

    priors = kb.get("priors")
    if isinstance(priors, dict):
        for key in candidates:
            prior = priors.get(key)
            if isinstance(prior, dict):
                return _common_prior_to_prediction(prior, source=f"common:{key}")

    for container_name in ("tools", "families", "generic"):
        container = kb.get(container_name)
        if not isinstance(container, dict):
            continue
        for key in candidates:
            prior = container.get(key)
            if isinstance(prior, dict):
                return _common_prior_to_prediction(prior, source=f"common:{key}")
    return None


def select_unified_prediction(
    *,
    personal_kb: dict[str, Any],
    common_kb: dict[str, Any],
    tool_name: str,
    tool_family: str,
    operation: str,
    normalized_command: str | None,
    workload_bucket: str | None = None,
) -> RuntimePrediction | None:
    personal = lookup_personal_prediction(
        personal_kb,
        tool_name=tool_name,
        tool_family=tool_family,
        operation=operation,
        normalized_command=normalized_command,
    )
    if personal is not None:
        return personal
    return lookup_common_prediction(
        common_kb,
        tool_name=tool_name,
        tool_family=tool_family,
        operation=operation,
        workload_bucket=workload_bucket,
    )


def _update_resource_history(rec: dict[str, Any], resource: ResourceSummary | None) -> None:
    if resource is None:
        return
    resources = rec.get("resources")
    if not isinstance(resources, dict):
        resources = {}
    for attr in (
        "avg_cores",
        "p50_cores",
        "p90_cores",
        "peak_cores",
        "peak_memory_mb",
        "read_mb",
        "write_mb",
    ):
        value = getattr(resource, attr)
        if value is not None:
            resources[attr] = bounded_append(resources.get(attr), float(value))
    if resource.load_class:
        counts = resources.get("load_class_counts")
        if not isinstance(counts, dict):
            counts = {}
        counts[resource.load_class] = int(counts.get(resource.load_class, 0)) + 1
        resources["load_class_counts"] = counts
        resources["load_class"] = max(counts, key=lambda key: int(counts[key]))
    if resource.sample_count is not None:
        resources["sample_counts"] = bounded_append(
            resources.get("sample_counts"),
            float(resource.sample_count),
        )
    rec["resources"] = resources


def update_personal_kb(
    kb: dict[str, Any],
    *,
    tool_name: str,
    tool_family: str,
    operation: str,
    normalized_command: str | None,
    duration_s: float,
    success: bool,
    repo_id: str | None = None,
    features: dict[str, Any] | None = None,
    resource_summary: ResourceSummary | None = None,
) -> dict[str, Any]:
    """Return updated Personal KB. Common KB is intentionally not touched."""

    if not success:
        return dict(kb)

    updated = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": utc_now(),
        "history_limit": int(kb.get("history_limit") or HISTORY_LIMIT),
        "repo_id": repo_id or kb.get("repo_id"),
        "commands": dict(_records(kb, "commands")),
        "tools": dict(_records(kb, "tools")),
        "families": dict(_records(kb, "families")),
    }
    limit = int(updated["history_limit"])

    def touch(container: str, key: str) -> None:
        rec = dict(updated[container].get(key) or {})
        rec["durations"] = bounded_append(rec.get("durations"), duration_s, limit=limit)
        rec["last_seen_at"] = updated["updated_at"]
        rec["sample_count"] = len(rec["durations"])
        if features:
            rec["last_features"] = features
        _update_resource_history(rec, resource_summary)
        updated[container][key] = rec

    if normalized_command:
        touch("commands", normalized_command)
    touch("tools", f"{tool_name}/{operation}")
    touch("families", f"{tool_family}/{operation}")
    return updated
