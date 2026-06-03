"""Helpers for selecting and reporting LLM latency metrics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _coerce_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_llm_wall_latency_ms(data: Mapping[str, Any] | None) -> float:
    """Return the locally observed wall-clock latency for one LLM call."""

    if not data:
        return 0.0
    wall = _coerce_optional_float(data.get("llm_wall_latency_ms"))
    if wall is None:
        wall = _coerce_optional_float(data.get("llm_latency_ms"))
    return 0.0 if wall is None else wall


def get_llm_call_time_ms(data: Mapping[str, Any] | None) -> float | None:
    """Return canonical model call time, preferring explicit call-time fields."""

    if not data:
        return None
    for key in ("llm_call_time_ms", "openrouter_generation_time_ms"):
        value = _coerce_optional_float(data.get(key))
        if value is not None:
            return value
    return None


def get_openrouter_latency_ms(data: Mapping[str, Any] | None) -> float | None:
    """Return OpenRouter total latency when the trace includes it."""

    if not data:
        return None
    return _coerce_optional_float(data.get("openrouter_latency_ms"))


def get_preferred_llm_latency_ms(data: Mapping[str, Any] | None) -> float:
    """Return canonical LLM call time, falling back to local wall time."""

    llm_call_time = get_llm_call_time_ms(data)
    if llm_call_time is not None:
        return llm_call_time
    if data:
        llm_latency = _coerce_optional_float(data.get("llm_latency_ms"))
        if llm_latency is not None:
            return llm_latency
    return get_llm_wall_latency_ms(data)


def summarize_llm_latencies(
    records: Iterable[Mapping[str, Any] | None],
) -> dict[str, float | int | str]:
    """Summarize LLM latencies without mixing call time and wall time totals."""

    record_list = [record for record in records if record is not None]
    total_llm_wall_ms = sum(get_llm_wall_latency_ms(record) for record in record_list)

    llm_call_times: list[float] = []
    llm_call_sources: set[str] = set()
    for record in record_list:
        llm_call_time = get_llm_call_time_ms(record)
        if llm_call_time is None:
            continue
        llm_call_times.append(llm_call_time)
        source = record.get("llm_timing_source")
        if isinstance(source, str) and source:
            llm_call_sources.add(source)

    llm_call_time_count = len(llm_call_times)
    total_llm_call_time_ms = sum(llm_call_times)
    if record_list and llm_call_time_count == len(record_list):
        total_llm_ms = total_llm_call_time_ms
        if len(llm_call_sources) == 1:
            llm_timing_source = next(iter(llm_call_sources))
        else:
            llm_timing_source = "llm_call_time_ms"
    elif llm_call_time_count == 0:
        total_llm_ms = total_llm_wall_ms
        llm_timing_source = "wall_clock_ms"
    else:
        total_llm_ms = total_llm_wall_ms
        llm_timing_source = "mixed_fallback_to_wall_clock_ms"

    return {
        "total_llm_ms": total_llm_ms,
        "total_llm_wall_ms": total_llm_wall_ms,
        "total_llm_call_time_ms": total_llm_call_time_ms,
        "llm_call_time_count": llm_call_time_count,
        "llm_timing_source": llm_timing_source,
    }
