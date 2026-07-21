"""Metric computation: deltas, effective cores, percentiles, stability."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .sampler import SamplePoint
from .classifier import classify, ClassificationResult


@dataclass
class AggregatedMetrics:
    """Aggregated statistics over a set of sample points."""

    num_samples: int
    elapsed_start_s: float
    elapsed_end_s: float

    avg_effective_cores: float
    p50_effective_cores: float
    p90_effective_cores: float
    peak_effective_cores: float

    avg_process_count: float
    peak_process_count: int

    avg_thread_count: float
    peak_thread_count: int

    total_cpu_user_time_s: float
    total_cpu_system_time_s: float
    total_cpu_time_s: float

    rss_start_bytes: int
    rss_end_bytes: int
    rss_peak_bytes: int

    total_read_bytes: int
    total_write_bytes: int

    total_voluntary_context_switches: Optional[int]
    total_involuntary_context_switches: Optional[int]

    total_minor_page_faults: Optional[int]
    total_major_page_faults: Optional[int]

    parallelism_cv: float
    profile_stability: str
    preliminary_behavior: str


@dataclass
class WindowMetrics:
    """Metrics for a single delta window between two samples."""

    timestamp_s: float
    elapsed_s: float
    root_pid: int

    process_count: int
    thread_count: int

    cpu_user_time_s: float
    cpu_system_time_s: float
    cpu_total_time_s: float

    cpu_time_delta_s: float
    wall_time_delta_s: float
    effective_cpu_cores: float

    rss_bytes: int
    vms_bytes: int

    read_bytes: Optional[int]
    write_bytes: Optional[int]
    read_count: Optional[int]
    write_count: Optional[int]

    voluntary_context_switches: Optional[int]
    involuntary_context_switches: Optional[int]

    minor_page_faults: Optional[int]
    major_page_faults: Optional[int]


def _safe_delta(new_val: Optional[float], old_val: Optional[float]) -> Optional[float]:
    """Compute delta between two optional values. Returns None if either is None."""
    if new_val is None or old_val is None:
        return None
    return new_val - old_val


def _safe_int_delta(new_val: Optional[int], old_val: Optional[int]) -> Optional[int]:
    """Compute delta between two optional int values."""
    if new_val is None or old_val is None:
        return None
    return new_val - old_val


def compute_window_metrics(
    current: SamplePoint,
    previous: SamplePoint,
) -> WindowMetrics:
    """Compute per-window metrics from two consecutive samples.

    Args:
        current: The current sample point.
        previous: The immediately preceding sample point.

    Returns:
        WindowMetrics with delta-based calculations.
    """
    wall_delta = current.elapsed_s - previous.elapsed_s
    cpu_delta = max(current.cpu_total_time_s - previous.cpu_total_time_s, 0.0)

    # effective_cpu_cores = CPU time delta / wall time delta
    effective_cores = cpu_delta / wall_delta if wall_delta > 0 else 0.0

    return WindowMetrics(
        timestamp_s=current.timestamp_s,
        elapsed_s=current.elapsed_s,
        root_pid=current.root_pid,
        process_count=current.process_count,
        thread_count=current.thread_count,
        cpu_user_time_s=current.cpu_user_time_s,
        cpu_system_time_s=current.cpu_system_time_s,
        cpu_total_time_s=current.cpu_total_time_s,
        cpu_time_delta_s=cpu_delta,
        wall_time_delta_s=wall_delta,
        effective_cpu_cores=effective_cores,
        rss_bytes=current.rss_bytes,
        vms_bytes=current.vms_bytes,
        read_bytes=current.read_bytes,
        write_bytes=current.write_bytes,
        read_count=current.read_count,
        write_count=current.write_count,
        voluntary_context_switches=current.voluntary_context_switches,
        involuntary_context_switches=current.involuntary_context_switches,
        minor_page_faults=current.minor_page_faults,
        major_page_faults=current.major_page_faults,
    )


def compute_windows(samples: list[SamplePoint]) -> list[WindowMetrics]:
    """Convert a time series of SamplePoints into WindowMetrics.

    Returns empty list if fewer than 2 samples.
    """
    if len(samples) < 2:
        return []
    windows: list[WindowMetrics] = []
    for i in range(1, len(samples)):
        wm = compute_window_metrics(samples[i], samples[i - 1])
        windows.append(wm)
    return windows


def percentile(sorted_values: list[float], p: float) -> float:
    """Compute the p-th percentile (0..100) of sorted values using linear interpolation.

    Args:
        sorted_values: Already sorted list of floats.
        p: Percentile value, e.g. 50 for median, 90 for p90.

    Returns:
        The interpolated percentile value.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]

    k = (p / 100.0) * (len(sorted_values) - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    d0 = sorted_values[int(f)] * (c - k)
    d1 = sorted_values[int(c)] * (k - f)
    return d0 + d1


def compute_cv(values: list[float]) -> float:
    """Coefficient of variation: std / |mean|. Returns 0 if mean near zero."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if abs(mean) < 1e-12:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance) / abs(mean)


def aggregate_samples(
    samples: list[SamplePoint],
    windows: list[WindowMetrics],
) -> AggregatedMetrics:
    """Aggregate a set of samples into summary statistics.

    Args:
        samples: All raw sample points.
        windows: Pre-computed window metrics (deltas).

    Returns:
        AggregatedMetrics with summary statistics.
    """
    if not samples:
        return AggregatedMetrics(
            num_samples=0,
            elapsed_start_s=0.0,
            elapsed_end_s=0.0,
            avg_effective_cores=0.0,
            p50_effective_cores=0.0,
            p90_effective_cores=0.0,
            peak_effective_cores=0.0,
            avg_process_count=0.0,
            peak_process_count=0,
            avg_thread_count=0.0,
            peak_thread_count=0,
            total_cpu_user_time_s=0.0,
            total_cpu_system_time_s=0.0,
            total_cpu_time_s=0.0,
            rss_start_bytes=0,
            rss_end_bytes=0,
            rss_peak_bytes=0,
            total_read_bytes=0,
            total_write_bytes=0,
            total_voluntary_context_switches=None,
            total_involuntary_context_switches=None,
            total_minor_page_faults=None,
            total_major_page_faults=None,
            parallelism_cv=0.0,
            profile_stability="unknown",
            preliminary_behavior="unknown",
        )

    # Window-level effective cores (for percentiles)
    effective_cores_list = [w.effective_cpu_cores for w in windows]
    sorted_cores = sorted(effective_cores_list)

    # Aggregates
    avg_cores = sum(effective_cores_list) / len(effective_cores_list) if effective_cores_list else 0.0
    peak_cores = max(effective_cores_list) if effective_cores_list else 0.0

    process_counts = [s.process_count for s in samples]
    thread_counts = [s.thread_count for s in samples]
    rss_values = [s.rss_bytes for s in samples]

    # CPU times: use last sample minus first sample for cumulative deltas
    cpu_user_total = max(samples[-1].cpu_user_time_s - samples[0].cpu_user_time_s, 0.0)
    cpu_sys_total = max(samples[-1].cpu_system_time_s - samples[0].cpu_system_time_s, 0.0)
    cpu_total = cpu_user_total + cpu_sys_total

    # I/O: use last sample values (cumulative from OS)
    def _final_io_val(
        attr: str,
    ) -> int:
        for s in reversed(samples):
            val = getattr(s, attr)
            if val is not None:
                return val
        return 0

    total_read = _final_io_val("read_bytes")
    total_write = _final_io_val("write_bytes")

    # Context switches
    vol_ctx: Optional[int] = None
    invol_ctx: Optional[int] = None
    for s in reversed(samples):
        if s.voluntary_context_switches is not None:
            vol_ctx = (_opt_final(s.voluntary_context_switches, samples[0].voluntary_context_switches)
                       if samples[0].voluntary_context_switches is not None
                       else s.voluntary_context_switches)
            break
    for s in reversed(samples):
        if s.involuntary_context_switches is not None:
            invol_ctx = (_opt_final(s.involuntary_context_switches, samples[0].involuntary_context_switches)
                        if samples[0].involuntary_context_switches is not None
                        else s.involuntary_context_switches)
            break

    # Page faults
    minflt: Optional[int] = None
    majflt: Optional[int] = None
    for s in reversed(samples):
        if s.minor_page_faults is not None:
            minflt = (_opt_final(s.minor_page_faults, samples[0].minor_page_faults)
                     if samples[0].minor_page_faults is not None
                     else s.minor_page_faults)
            break
    for s in reversed(samples):
        if s.major_page_faults is not None:
            majflt = (_opt_final(s.major_page_faults, samples[0].major_page_faults)
                     if samples[0].major_page_faults is not None
                     else s.major_page_faults)
            break

    parallelism_cv = compute_cv(effective_cores_list)

    wall_total = samples[-1].elapsed_s - samples[0].elapsed_s
    cpu_time_ratio = cpu_total / wall_total if wall_total > 0 else 0.0

    classification = classify(
        avg_effective_cores=avg_cores,
        cpu_time_ratio=cpu_time_ratio,
        total_read_bytes=total_read,
        total_write_bytes=total_write,
        parallelism_cv=parallelism_cv,
        num_samples=len(samples),
    )

    return AggregatedMetrics(
        num_samples=len(samples),
        elapsed_start_s=samples[0].elapsed_s,
        elapsed_end_s=samples[-1].elapsed_s,
        avg_effective_cores=avg_cores,
        p50_effective_cores=percentile(sorted_cores, 50),
        p90_effective_cores=percentile(sorted_cores, 90),
        peak_effective_cores=peak_cores,
        avg_process_count=sum(process_counts) / len(process_counts),
        peak_process_count=max(process_counts),
        avg_thread_count=sum(thread_counts) / len(thread_counts),
        peak_thread_count=max(thread_counts),
        total_cpu_user_time_s=cpu_user_total,
        total_cpu_system_time_s=cpu_sys_total,
        total_cpu_time_s=cpu_total,
        rss_start_bytes=samples[0].rss_bytes,
        rss_end_bytes=samples[-1].rss_bytes,
        rss_peak_bytes=max(rss_values),
        total_read_bytes=total_read,
        total_write_bytes=total_write,
        total_voluntary_context_switches=vol_ctx,
        total_involuntary_context_switches=invol_ctx,
        total_minor_page_faults=minflt,
        total_major_page_faults=majflt,
        parallelism_cv=parallelism_cv,
        profile_stability=classification.profile_stability,
        preliminary_behavior=classification.label,
    )


def _opt_final(final_val: int, first_val: int) -> int:
    """Compute final cumulative delta."""
    return max(final_val - first_val, 0)
