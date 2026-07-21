"""Weak behavior classification based on observed resource metrics.

IMPORTANT: `preliminary_behavior` is a quick profile result, NOT a rigorous
performance bottleneck classification. In particular, without DRAM bandwidth
or interference experiments, we cannot reliably claim a tool is
memory-bandwidth-bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BehaviorLabel = Literal[
    "cpu_parallel",
    "cpu_serial",
    "io_active",
    "mixed",
    "idle_or_waiting",
    "unknown",
]

# ---- Configurable thresholds (centralized, easy to tune) ----

# cpu_parallel: avg effective cores >= this AND cpu_time_ratio >= this
CPU_PARALLEL_MIN_CORES = 2.0
CPU_PARALLEL_MIN_CPU_RATIO = 1.5

# cpu_serial: avg effective cores in [MIN, MAX) AND I/O not significant
CPU_SERIAL_MIN_CORES = 0.6
CPU_SERIAL_MAX_CORES = 2.0

# io_active: cpu_time_ratio < this AND I/O bytes > threshold
IO_ACTIVE_MAX_CPU_RATIO = 0.5
IO_ACTIVE_MIN_BYTES = 1024 * 1024  # 1 MiB

# idle_or_waiting: cpu_time_ratio < this AND I/O also low
IDLE_MAX_CPU_RATIO = 0.2
IDLE_MAX_IO_BYTES = 1024 * 1024  # 1 MiB

# Stability: parallelism_cv <= this => "stable"
STABILITY_CV_THRESHOLD = 0.30


@dataclass
class ClassificationResult:
    """Result of weak behavior classification."""

    label: BehaviorLabel
    cpu_time_ratio: float
    avg_effective_cores: float
    total_read_bytes: int
    total_write_bytes: int
    parallelism_cv: float
    profile_stability: str  # "stable" | "unstable"


def classify(
    *,
    avg_effective_cores: float,
    cpu_time_ratio: float,
    total_read_bytes: int,
    total_write_bytes: int,
    parallelism_cv: float,
    num_samples: int = 0,
) -> ClassificationResult:
    """Classify tool behavior from aggregated metrics.

    Args:
        avg_effective_cores: Mean effective CPU cores across samples.
        cpu_time_ratio: Total CPU time delta / total wall time delta.
        total_read_bytes: Cumulative bytes read.
        total_write_bytes: Cumulative bytes written.
        parallelism_cv: Coefficient of variation of effective_cores.
        num_samples: Number of samples (used to detect insufficient data).

    Returns:
        ClassificationResult with label and stability.
    """
    stability = (
        "stable" if parallelism_cv <= STABILITY_CV_THRESHOLD else "unstable"
    )
    io_bytes = total_read_bytes + total_write_bytes

    # Insufficient data
    if num_samples < 2:
        return ClassificationResult(
            label="unknown",
            cpu_time_ratio=cpu_time_ratio,
            avg_effective_cores=avg_effective_cores,
            total_read_bytes=total_read_bytes,
            total_write_bytes=total_write_bytes,
            parallelism_cv=parallelism_cv,
            profile_stability=stability,
        )

    # CPU parallel
    if (
        avg_effective_cores >= CPU_PARALLEL_MIN_CORES
        and cpu_time_ratio >= CPU_PARALLEL_MIN_CPU_RATIO
    ):
        return ClassificationResult(
            label="cpu_parallel",
            cpu_time_ratio=cpu_time_ratio,
            avg_effective_cores=avg_effective_cores,
            total_read_bytes=total_read_bytes,
            total_write_bytes=total_write_bytes,
            parallelism_cv=parallelism_cv,
            profile_stability=stability,
        )

    # Idle or waiting
    if cpu_time_ratio < IDLE_MAX_CPU_RATIO and io_bytes < IDLE_MAX_IO_BYTES:
        return ClassificationResult(
            label="idle_or_waiting",
            cpu_time_ratio=cpu_time_ratio,
            avg_effective_cores=avg_effective_cores,
            total_read_bytes=total_read_bytes,
            total_write_bytes=total_write_bytes,
            parallelism_cv=parallelism_cv,
            profile_stability=stability,
        )

    # I/O active
    if cpu_time_ratio < IO_ACTIVE_MAX_CPU_RATIO and io_bytes >= IO_ACTIVE_MIN_BYTES:
        return ClassificationResult(
            label="io_active",
            cpu_time_ratio=cpu_time_ratio,
            avg_effective_cores=avg_effective_cores,
            total_read_bytes=total_read_bytes,
            total_write_bytes=total_write_bytes,
            parallelism_cv=parallelism_cv,
            profile_stability=stability,
        )

    # CPU serial
    if CPU_SERIAL_MIN_CORES <= avg_effective_cores < CPU_SERIAL_MAX_CORES and io_bytes < IO_ACTIVE_MIN_BYTES:
        return ClassificationResult(
            label="cpu_serial",
            cpu_time_ratio=cpu_time_ratio,
            avg_effective_cores=avg_effective_cores,
            total_read_bytes=total_read_bytes,
            total_write_bytes=total_write_bytes,
            parallelism_cv=parallelism_cv,
            profile_stability=stability,
        )

    # Mixed: CPU and I/O both present
    if (
        avg_effective_cores >= CPU_SERIAL_MIN_CORES
        and io_bytes >= IO_ACTIVE_MIN_BYTES
    ):
        return ClassificationResult(
            label="mixed",
            cpu_time_ratio=cpu_time_ratio,
            avg_effective_cores=avg_effective_cores,
            total_read_bytes=total_read_bytes,
            total_write_bytes=total_write_bytes,
            parallelism_cv=parallelism_cv,
            profile_stability=stability,
        )

    return ClassificationResult(
        label="unknown",
        cpu_time_ratio=cpu_time_ratio,
        avg_effective_cores=avg_effective_cores,
        total_read_bytes=total_read_bytes,
        total_write_bytes=total_write_bytes,
        parallelism_cv=parallelism_cv,
        profile_stability=stability,
    )
