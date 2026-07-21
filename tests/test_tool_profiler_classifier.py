"""Tests for the classifier module."""

from __future__ import annotations

import pytest

from prototype.tool_profiler.classifier import classify, STABILITY_CV_THRESHOLD


class TestClassify:
    """Unit tests for weak behavior classification."""

    def test_cpu_parallel(self) -> None:
        result = classify(
            avg_effective_cores=8.0,
            cpu_time_ratio=1.8,
            total_read_bytes=0,
            total_write_bytes=0,
            parallelism_cv=0.1,
            num_samples=10,
        )
        assert result.label == "cpu_parallel"
        assert result.profile_stability == "stable"

    def test_cpu_serial(self) -> None:
        result = classify(
            avg_effective_cores=0.9,
            cpu_time_ratio=0.8,
            total_read_bytes=0,
            total_write_bytes=0,
            parallelism_cv=0.05,
            num_samples=10,
        )
        assert result.label == "cpu_serial"

    def test_io_active(self) -> None:
        result = classify(
            avg_effective_cores=0.3,
            cpu_time_ratio=0.3,
            total_read_bytes=10_000_000,
            total_write_bytes=5_000_000,
            parallelism_cv=0.2,
            num_samples=10,
        )
        assert result.label == "io_active"

    def test_idle_or_waiting(self) -> None:
        result = classify(
            avg_effective_cores=0.05,
            cpu_time_ratio=0.1,
            total_read_bytes=100,
            total_write_bytes=50,
            parallelism_cv=0.3,
            num_samples=10,
        )
        assert result.label == "idle_or_waiting"

    def test_mixed(self) -> None:
        result = classify(
            avg_effective_cores=1.5,
            cpu_time_ratio=0.6,
            total_read_bytes=10_000_000,
            total_write_bytes=5_000_000,
            parallelism_cv=0.25,
            num_samples=10,
        )
        assert result.label == "mixed"

    def test_unknown_insufficient_samples(self) -> None:
        result = classify(
            avg_effective_cores=1.0,
            cpu_time_ratio=0.5,
            total_read_bytes=0,
            total_write_bytes=0,
            parallelism_cv=0.0,
            num_samples=1,
        )
        assert result.label == "unknown"

    def test_stability_stable(self) -> None:
        result = classify(
            avg_effective_cores=1.0,
            cpu_time_ratio=0.9,
            total_read_bytes=0,
            total_write_bytes=0,
            parallelism_cv=0.20,
            num_samples=10,
        )
        assert result.profile_stability == "stable"

    def test_stability_unstable(self) -> None:
        result = classify(
            avg_effective_cores=2.0,
            cpu_time_ratio=0.9,
            total_read_bytes=0,
            total_write_bytes=0,
            parallelism_cv=0.50,
            num_samples=10,
        )
        assert result.profile_stability == "unstable"

    def test_stability_boundary(self) -> None:
        """Test exactly at the stability threshold."""
        result = classify(
            avg_effective_cores=1.0,
            cpu_time_ratio=0.9,
            total_read_bytes=0,
            total_write_bytes=0,
            parallelism_cv=STABILITY_CV_THRESHOLD,
            num_samples=10,
        )
        # At exactly the threshold, it's stable (<=)
        assert result.profile_stability == "stable"
