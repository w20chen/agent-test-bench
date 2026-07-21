"""Tests for the metrics module."""

from __future__ import annotations

import pytest

from prototype.tool_profiler.metrics import (
    compute_windows,
    compute_cv,
    percentile,
    aggregate_samples,
)
from prototype.tool_profiler.sampler import SamplePoint


def _make_sample(
    elapsed_s: float,
    cpu_total: float,
    cpu_user: float,
    cpu_system: float,
    rss: int = 100_000_000,
    vms: int = 200_000_000,
    procs: int = 1,
    threads: int = 1,
) -> SamplePoint:
    """Helper: create a SamplePoint with minimal fields."""
    return SamplePoint(
        timestamp_s=1000.0 + elapsed_s,
        elapsed_s=elapsed_s,
        root_pid=1,
        process_count=procs,
        thread_count=threads,
        cpu_user_time_s=cpu_user,
        cpu_system_time_s=cpu_system,
        cpu_total_time_s=cpu_total,
        rss_bytes=rss,
        vms_bytes=vms,
        read_bytes=None,
        write_bytes=None,
        read_count=None,
        write_count=None,
        voluntary_context_switches=None,
        involuntary_context_switches=None,
        minor_page_faults=None,
        major_page_faults=None,
    )


class TestPercentile:
    def test_median_odd(self) -> None:
        assert percentile([1.0, 2.0, 3.0], 50) == 2.0

    def test_median_even(self) -> None:
        assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5

    def test_p90(self) -> None:
        vals = sorted([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        # p90 of 1..10 with 10 values => k = 0.9 * 9 = 8.1 => between index 8 and 9
        # vals[8] = 9.0, vals[9] = 10.0 => 9.0*(9-8.1) + 10.0*(8.1-8) = 9.0*0.9 + 10.0*0.1 = 8.1 + 1.0 = 9.1
        result = percentile(vals, 90)
        assert abs(result - 9.1) < 1e-9

    def test_single_value(self) -> None:
        assert percentile([5.0], 50) == 5.0

    def test_empty(self) -> None:
        assert percentile([], 50) == 0.0


class TestCV:
    def test_constant(self) -> None:
        assert compute_cv([1.0, 1.0, 1.0, 1.0]) == 0.0

    def test_variable(self) -> None:
        cv = compute_cv([1.0, 2.0, 3.0, 4.0, 5.0])
        assert cv > 0.0

    def test_single_value(self) -> None:
        assert compute_cv([1.0]) == 0.0

    def test_mean_zero(self) -> None:
        assert compute_cv([-1.0, 1.0]) == 0.0


class TestComputeWindows:
    def test_single_sample(self) -> None:
        s = _make_sample(0.0, 0.0, 0.0, 0.0)
        assert compute_windows([s]) == []

    def test_two_samples_effective_cores(self) -> None:
        s0 = _make_sample(0.0, 0.0, 0.0, 0.0)
        s1 = _make_sample(0.2, 0.4, 0.3, 0.1)
        windows = compute_windows([s0, s1])
        assert len(windows) == 1
        assert windows[0].wall_time_delta_s == 0.2
        assert windows[0].cpu_time_delta_s == 0.4
        assert windows[0].effective_cpu_cores == 2.0

    def test_cpu_time_no_regression(self) -> None:
        """CPU time should not decrease between samples."""
        s0 = _make_sample(0.0, 10.0, 8.0, 2.0)
        s1 = _make_sample(0.2, 9.5, 7.5, 2.0)  # regression
        windows = compute_windows([s0, s1])
        assert windows[0].cpu_time_delta_s == 0.0  # clamped to 0


class TestAggregateSamples:
    def test_empty(self) -> None:
        agg = aggregate_samples([], [])
        assert agg.num_samples == 0

    def test_basic_aggregation(self) -> None:
        samples = [
            _make_sample(0.0, 0.0, 0.0, 0.0, rss=100, procs=1, threads=1),
            _make_sample(0.2, 0.4, 0.3, 0.1, rss=120, procs=1, threads=1),
            _make_sample(0.4, 0.8, 0.6, 0.2, rss=150, procs=1, threads=1),
        ]
        windows = compute_windows(samples)
        agg = aggregate_samples(samples, windows)
        assert agg.num_samples == 3
        assert agg.avg_effective_cores == 2.0  # each window: 0.4/0.2 = 2.0
        assert agg.rss_start_bytes == 100
        assert agg.rss_end_bytes == 150
        assert agg.rss_peak_bytes == 150
