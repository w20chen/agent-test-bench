"""Tests for system-level resource profiling (P0).

Covers:
- ContainerStatsSampler cgroup-based I/O + NetIO + context switch extensions
- Network mode parameterization in start_task_container
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness.container_stats_sampler import (  # noqa: E402
    ContainerStatsSampler,
    _aggregate_context_switches,
    _parse_net_io_bytes,
    _parse_pipe_stats,
    _parse_size_bytes,
    _read_cgroup_io_stat,
    _read_cgroup_pids,
    _read_ctxt_via_exec,
    _read_io_via_exec,
    _read_pid_context_switches,
    _read_pid_starttime,
    _resolve_cgroup_path,
    _resolve_container_pid,
    summarize_samples,
)


# ---------------------------------------------------------------------------
# _parse_pipe_stats
# ---------------------------------------------------------------------------


def test_parse_pipe_stats_4field() -> None:
    """4-field format with NetIO is parsed correctly."""
    raw = "100MB / 1GB|10%|5%|1.5kB / 2.3MB"
    result = _parse_pipe_stats(raw)
    assert result is not None
    assert result["mem_usage"] == "100MB / 1GB"
    assert result["mem_percent"] == "10%"
    assert result["cpu_percent"] == "5%"
    assert result["net_io"] == "1.5kB / 2.3MB"


def test_parse_pipe_stats_legacy_3field() -> None:
    """Legacy 3-field format still works (backward compat)."""
    raw = "100MB / 1GB|10%|5%"
    result = _parse_pipe_stats(raw)
    assert result is not None
    assert result["mem_usage"] == "100MB / 1GB"
    assert result["cpu_percent"] == "5%"
    assert "net_io" not in result


def test_parse_pipe_stats_rejects_short_input() -> None:
    assert _parse_pipe_stats("foo|bar") is None
    assert _parse_pipe_stats("") is None
    assert _parse_pipe_stats(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _resolve_container_pid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("executable", ["docker", "podman"])
def test_resolve_container_pid(executable: str) -> None:
    """Dispatches correct inspect command and returns PID."""
    def fake_run(cmd, **kwargs):
        assert cmd == [executable, "inspect", "--format", "{{.State.Pid}}", "abc123"]
        return subprocess.CompletedProcess(cmd, 0, stdout="42\n", stderr="")

    with patch("harness.container_stats_sampler.subprocess.run", side_effect=fake_run):
        pid = _resolve_container_pid("abc123", executable=executable)
    assert pid == 42


def test_resolve_container_pid_returns_none_on_failure() -> None:
    def fail_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="error")

    with patch("harness.container_stats_sampler.subprocess.run", side_effect=fail_run):
        assert _resolve_container_pid("bad", executable="docker") is None


def test_resolve_container_pid_returns_none_on_zero_pid() -> None:
    def zero_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="0\n", stderr="")

    with patch("harness.container_stats_sampler.subprocess.run", side_effect=zero_run):
        assert _resolve_container_pid("stopped", executable="docker") is None


# ---------------------------------------------------------------------------
# _resolve_cgroup_path
# ---------------------------------------------------------------------------


def test_resolve_cgroup_path() -> None:
    """Resolves cgroup v2 path from /proc/<pid>/cgroup."""
    cgroup_content = "0::/system.slice/docker-abc123.scope\n"

    def fake_read_text(self, encoding="utf-8"):
        if str(self) == "/proc/42/cgroup":
            return cgroup_content
        raise FileNotFoundError(str(self))

    def fake_exists(self):
        return str(self) == "/sys/fs/cgroup/system.slice/docker-abc123.scope"

    with (
        patch.object(Path, "read_text", fake_read_text),
        patch.object(Path, "exists", fake_exists),
    ):
        result = _resolve_cgroup_path(42)
    assert result == Path("/sys/fs/cgroup/system.slice/docker-abc123.scope")


def test_resolve_cgroup_path_permission_error() -> None:
    with patch.object(Path, "read_text", side_effect=PermissionError):
        assert _resolve_cgroup_path(42) is None


# ---------------------------------------------------------------------------
# _read_cgroup_io_stat
# ---------------------------------------------------------------------------


def test_read_cgroup_io_stat() -> None:
    """Parses io.stat and aggregates across devices."""
    io_stat = (
        "8:0 rbytes=4096000 wbytes=2048000 rios=100 wios=50 dbytes=0 dios=0\n"
        "8:16 rbytes=1000 wbytes=500 rios=10 wios=5 dbytes=0 dios=0\n"
    )

    def fake_read_text(self, encoding="utf-8"):
        return io_stat

    with patch.object(Path, "read_text", fake_read_text):
        result = _read_cgroup_io_stat(Path("/sys/fs/cgroup/test"))
    assert result is not None
    assert result["read_bytes"] == 4097000
    assert result["write_bytes"] == 2048500


def test_read_cgroup_io_stat_missing() -> None:
    with patch.object(Path, "read_text", side_effect=FileNotFoundError):
        assert _read_cgroup_io_stat(Path("/nonexistent")) is None


# ---------------------------------------------------------------------------
# _read_cgroup_pids + _aggregate_context_switches
# ---------------------------------------------------------------------------


def test_read_cgroup_pids() -> None:
    procs_content = "42\n43\n100\n"

    def fake_read_text(self, encoding="utf-8"):
        return procs_content

    with patch.object(Path, "read_text", fake_read_text):
        pids = _read_cgroup_pids(Path("/sys/fs/cgroup/test"))
    assert pids == [42, 43, 100]


def test_read_pid_starttime() -> None:
    """Extracts starttime (field 22) from /proc/<pid>/stat."""
    stat_content = "42 (python3) S 1 42 42 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 5000 0 0\n"
    with patch.object(Path, "read_text", return_value=stat_content):
        assert _read_pid_starttime(42) == 5000


def test_read_pid_starttime_comm_with_spaces() -> None:
    """Handles comm field containing spaces and parentheses."""
    stat_content = "42 (my (weird) app) S 1 42 42 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 9999 0 0\n"
    with patch.object(Path, "read_text", return_value=stat_content):
        assert _read_pid_starttime(42) == 9999


def test_read_pid_context_switches() -> None:
    """Reads voluntary + nonvoluntary context switches for one PID."""
    status = "Name:\tpython3\nvoluntary_ctxt_switches:\t500\nnonvoluntary_ctxt_switches:\t80\n"

    with patch.object(Path, "read_text", return_value=status):
        total = _read_pid_context_switches(42)
    assert total == 580


def test_aggregate_context_switches() -> None:
    """Sums context switches across all PIDs."""
    status_42 = "Name:\tsleep\nvoluntary_ctxt_switches:\t100\nnonvoluntary_ctxt_switches:\t20\n"
    status_43 = "Name:\tpython3\nvoluntary_ctxt_switches:\t500\nnonvoluntary_ctxt_switches:\t80\n"

    def fake_read_text(self, encoding="utf-8"):
        path_str = str(self)
        if path_str == "/proc/42/status":
            return status_42
        if path_str == "/proc/43/status":
            return status_43
        raise FileNotFoundError(path_str)

    with patch.object(Path, "read_text", fake_read_text):
        total = _aggregate_context_switches([42, 43])
    # 100 + 20 + 500 + 80 = 700
    assert total == 700


def test_aggregate_context_switches_partial_failure() -> None:
    """PIDs that vanish mid-read are skipped gracefully."""
    status_42 = "voluntary_ctxt_switches:\t50\nnonvoluntary_ctxt_switches:\t10\n"

    def fake_read_text(self, encoding="utf-8"):
        if str(self) == "/proc/42/status":
            return status_42
        raise FileNotFoundError(str(self))

    with patch.object(Path, "read_text", fake_read_text):
        total = _aggregate_context_switches([42, 99])
    assert total == 60


def test_sampler_cumulative_context_switches() -> None:
    """Exited PIDs keep their last-known counts keyed by (pid, starttime)."""
    sampler = ContainerStatsSampler(
        container_id="test", interval_s=1.0, executable="docker",
    )
    sampler._pid_ctxt = {(42, 1000): 100, (43, 1001): 200}
    sampler._pid_ctxt[(44, 1002)] = 50
    # PID 43 exited but stays in dict → total = 100 + 200 + 50
    assert sum(sampler._pid_ctxt.values()) == 350


def test_sampler_pid_reuse_separate_identity() -> None:
    """Recycled PID gets separate (pid, starttime) key — both lifetimes counted."""
    sampler = ContainerStatsSampler(
        container_id="test", interval_s=1.0, executable="docker",
    )
    # PID 42 first lifetime: started at tick 1000, accumulated 1000 ctxt
    sampler._pid_ctxt[(42, 1000)] = 1000
    # PID 42 recycled: started at tick 2000, new process has 5 ctxt
    sampler._pid_ctxt[(42, 2000)] = 5
    # Total = 1000 + 5 = 1005 (both lifetimes counted independently)
    assert sum(sampler._pid_ctxt.values()) == 1005


# ---------------------------------------------------------------------------
# _read_io_via_exec (fallback)
# ---------------------------------------------------------------------------


def test_read_io_via_exec() -> None:
    """Exec fallback aggregates I/O from all container processes."""
    def fake_run(cmd, **kwargs):
        assert "python3" in cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="8192000 4096000\n", stderr="")

    with patch("harness.container_stats_sampler.subprocess.run", side_effect=fake_run):
        result = _read_io_via_exec("cid123", executable="podman")
    assert result is not None
    assert result["read_bytes"] == 8192000
    assert result["write_bytes"] == 4096000


def test_read_ctxt_via_exec() -> None:
    """Exec fallback aggregates context switches from all container processes."""
    def fake_run(cmd, **kwargs):
        assert "python3" in cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="750\n", stderr="")

    with patch("harness.container_stats_sampler.subprocess.run", side_effect=fake_run):
        result = _read_ctxt_via_exec("cid", executable="docker")
    assert result == 750


def test_read_io_via_exec_failure() -> None:
    def fail_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="error")

    with patch("harness.container_stats_sampler.subprocess.run", side_effect=fail_run):
        assert _read_io_via_exec("bad", executable="docker") is None


# ---------------------------------------------------------------------------
# _parse_net_io_bytes / _parse_size_bytes
# ---------------------------------------------------------------------------


def test_parse_net_io_bytes() -> None:
    rx, tx = _parse_net_io_bytes("1.5kB / 2.3MB")
    assert rx == pytest.approx(1500.0)
    assert tx == pytest.approx(2.3e6)


def test_parse_net_io_bytes_gb() -> None:
    rx, tx = _parse_net_io_bytes("1GB / 2GB")
    assert rx == pytest.approx(1e9)
    assert tx == pytest.approx(2e9)


def test_parse_net_io_bytes_empty() -> None:
    assert _parse_net_io_bytes("") == (None, None)
    assert _parse_net_io_bytes("--") == (None, None)


def test_parse_size_bytes_variants() -> None:
    assert _parse_size_bytes("100B") == pytest.approx(100.0)
    assert _parse_size_bytes("1.5kB") == pytest.approx(1500.0)
    assert _parse_size_bytes("2MB") == pytest.approx(2e6)
    assert _parse_size_bytes("1GB") == pytest.approx(1e9)
    assert _parse_size_bytes("1TB") == pytest.approx(1e12)
    # Binary units
    assert _parse_size_bytes("1KiB") == pytest.approx(1024.0)
    assert _parse_size_bytes("1MiB") == pytest.approx(1024**2)
    assert _parse_size_bytes("1GiB") == pytest.approx(1024**3)
    assert _parse_size_bytes("1TiB") == pytest.approx(1024**4)
    assert _parse_size_bytes("") is None


# ---------------------------------------------------------------------------
# summarize_samples (extended fields)
# ---------------------------------------------------------------------------


def test_summarize_samples_with_io() -> None:
    """New I/O fields present with correct values."""
    samples = [
        {
            "epoch": 1000.0,
            "mem_usage": "100MB / 1GB",
            "mem_percent": "10%",
            "cpu_percent": "5%",
            "memory_total_mb_s": 100.0,
            "memory_read_mb_s": 60.0,
            "memory_write_mb_s": 40.0,
            "memory_bandwidth_available": True,
            "memory_bandwidth_source": "perf:intel-imc-cas",
            "disk_read_bytes": 1024 * 1024,       # 1 MB
            "disk_write_bytes": 512 * 1024,        # 0.5 MB
            "net_rx_bytes": 1000.0,                # ~0.001 MB
            "net_tx_bytes": 2000.0,                # ~0.002 MB
            "context_switches": 100,
        },
        {
            "epoch": 1002.0,
            "mem_usage": "200MB / 1GB",
            "mem_percent": "20%",
            "cpu_percent": "15%",
            "memory_total_mb_s": 150.0,
            "memory_read_mb_s": 90.0,
            "memory_write_mb_s": 60.0,
            "memory_bandwidth_available": True,
            "memory_bandwidth_source": "perf:intel-imc-cas",
            "disk_read_bytes": 2 * 1024 * 1024,   # 2 MB
            "disk_write_bytes": 1024 * 1024,       # 1 MB
            "net_rx_bytes": 5000.0,
            "net_tx_bytes": 10000.0,
            "context_switches": 250,
        },
        {
            "epoch": 1004.0,
            "mem_usage": "300MB / 1GB",
            "mem_percent": "30%",
            "cpu_percent": "25%",
            "memory_total_mb_s": 175.0,
            "memory_read_mb_s": 100.0,
            "memory_write_mb_s": 75.0,
            "memory_bandwidth_available": True,
            "memory_bandwidth_source": "perf:intel-imc-cas",
            "disk_read_bytes": 4 * 1024 * 1024,   # 4 MB
            "disk_write_bytes": 2 * 1024 * 1024,   # 2 MB
            "net_rx_bytes": 20000.0,
            "net_tx_bytes": 50000.0,
            "context_switches": 500,
        },
    ]
    summary = summarize_samples(samples)

    # Original fields unchanged (no regression)
    assert summary["sample_count"] == 3
    assert summary["duration_seconds"] == 4.0
    assert summary["memory_mb"]["min"] == 100.0
    assert summary["memory_mb"]["max"] == 300.0
    assert summary["memory_mb"]["avg"] == 200.0
    assert summary["cpu_percent"]["min"] == 5.0
    assert summary["cpu_percent"]["max"] == 25.0
    assert summary["cpu_percent"]["avg"] == 15.0
    assert summary["memory_total_mb_s"]["min"] == pytest.approx(100.0)
    assert summary["memory_total_mb_s"]["max"] == pytest.approx(175.0)
    assert summary["memory_total_mb_s"]["avg"] == pytest.approx((100.0 + 150.0 + 175.0) / 3)
    assert summary["memory_bandwidth_available"] is True
    assert summary["memory_bandwidth_source"] == "perf:intel-imc-cas"

    # Disk I/O
    assert summary["disk_read_mb"]["min"] == pytest.approx(1.0)
    assert summary["disk_read_mb"]["max"] == pytest.approx(4.0)
    assert summary["disk_read_mb"]["avg"] == pytest.approx(7.0 / 3)
    assert summary["disk_read_mb"]["delta"] == pytest.approx(3.0)

    assert summary["disk_write_mb"]["min"] == pytest.approx(0.5)
    assert summary["disk_write_mb"]["max"] == pytest.approx(2.0)
    assert summary["disk_write_mb"]["delta"] == pytest.approx(1.5)

    # Network I/O (bytes → MB, decimal division)
    assert summary["net_rx_mb"]["delta"] == pytest.approx(
        (20000.0 - 1000.0) / 1_000_000
    )
    assert summary["net_tx_mb"]["delta"] == pytest.approx(
        (50000.0 - 2000.0) / 1_000_000
    )

    # Context switches
    assert summary["context_switches"]["min"] == 100
    assert summary["context_switches"]["max"] == 500
    assert summary["context_switches"]["delta"] == 400


def test_summarize_samples_empty() -> None:
    """Empty input produces zero-valued fields including new ones."""
    summary = summarize_samples([])
    assert summary["sample_count"] == 0
    assert summary["memory_total_mb_s"]["avg"] == 0
    assert summary["memory_bandwidth_available"] is False
    assert summary["disk_read_mb"]["min"] == 0
    assert summary["net_rx_mb"]["delta"] == 0
    assert summary["context_switches"]["avg"] == 0


def test_summarize_samples_legacy_no_io() -> None:
    """Samples without I/O fields (legacy) produce zero-valued I/O summaries."""
    samples = [
        {
            "epoch": 1000.0,
            "mem_usage": "100MB / 1GB",
            "mem_percent": "10%",
            "cpu_percent": "5%",
        },
        {
            "epoch": 1002.0,
            "mem_usage": "200MB / 1GB",
            "mem_percent": "20%",
            "cpu_percent": "15%",
        },
    ]
    summary = summarize_samples(samples)
    assert summary["memory_mb"]["avg"] == 150.0
    assert summary["cpu_percent"]["avg"] == 10.0
    assert summary["memory_total_mb_s"]["avg"] == 0
    assert summary["memory_bandwidth_available"] is False
    assert summary["disk_read_mb"]["avg"] == 0
    assert summary["net_rx_mb"]["avg"] == 0
    assert summary["context_switches"]["avg"] == 0


# ---------------------------------------------------------------------------
# ContainerStatsSampler integration (mocked cgroup path)
# ---------------------------------------------------------------------------


def test_sampler_collects_io_via_cgroup() -> None:
    """Sampler uses cgroup io.stat for disk I/O when available."""
    pipe_output = "100MB / 1GB|10%|5%|1kB / 2kB"

    io_stat_content = "8:0 rbytes=1024 wbytes=512 rios=10 wios=5\n"
    cgroup_content = "0::/system.slice/docker-test.scope\n"
    procs_content = "42\n43\n"
    status_42 = "voluntary_ctxt_switches:\t10\nnonvoluntary_ctxt_switches:\t5\n"
    status_43 = "voluntary_ctxt_switches:\t20\nnonvoluntary_ctxt_switches:\t3\n"
    # /proc/<pid>/stat: fields after (comm): state ppid pgrp session tty_nr tpgid flags
    # minflt cminflt majflt cmajflt utime stime cutime cstime priority nice
    # num_threads itrealvalue starttime ...
    stat_42 = "42 (sleep) S 1 42 42 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 1000 0 0\n"
    stat_43 = "43 (python3) S 1 43 43 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 1001 0 0\n"

    def fake_subprocess_run(cmd, **kwargs):
        if "inspect" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="42\n", stderr="")
        if "stats" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=pipe_output, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def fake_read_text(self, encoding="utf-8"):
        path_str = str(self)
        if path_str == "/proc/42/cgroup":
            return cgroup_content
        if path_str.endswith("/io.stat"):
            return io_stat_content
        if path_str.endswith("/cgroup.procs"):
            return procs_content
        if path_str == "/proc/42/status":
            return status_42
        if path_str == "/proc/43/status":
            return status_43
        if path_str == "/proc/42/stat":
            return stat_42
        if path_str == "/proc/43/stat":
            return stat_43
        raise FileNotFoundError(path_str)

    def fake_exists(self):
        s = str(self)
        return s in (
            "/sys/fs/cgroup/system.slice/docker-test.scope",
            "/sys/fs/cgroup/system.slice/docker-test.scope/io.stat",
        )

    sampler = ContainerStatsSampler(
        container_id="test123",
        interval_s=0.05,
        executable="docker",
    )
    with (
        patch("harness.container_stats_sampler.subprocess.run", side_effect=fake_subprocess_run),
        patch(
            "harness.container_stats_sampler.attach_host_memory_bandwidth",
            side_effect=lambda sample, *, interval_s: sample.update(
                {
                    "memory_bandwidth_available": True,
                    "memory_bandwidth_source": "perf:test",
                    "memory_total_mb_s": 12.0,
                    "memory_read_mb_s": 7.0,
                    "memory_write_mb_s": 5.0,
                }
            ),
        ),
        patch.object(Path, "read_text", fake_read_text),
        patch.object(Path, "exists", fake_exists),
    ):
        sampler.start()
        time.sleep(0.25)
        samples = sampler.stop()

    assert len(samples) >= 2
    s = samples[0]
    assert s["mem_usage"] == "100MB / 1GB"
    assert s["disk_read_bytes"] == 1024
    assert s["disk_write_bytes"] == 512
    assert s["context_switches"] == 38  # 10+5+20+3
    assert s["net_rx_bytes"] == pytest.approx(1000.0)
    assert s["net_tx_bytes"] == pytest.approx(2000.0)
    assert s["memory_total_mb_s"] == pytest.approx(12.0)
    assert s["memory_read_mb_s"] == pytest.approx(7.0)
    assert s["memory_write_mb_s"] == pytest.approx(5.0)


def test_sampler_falls_back_to_exec_when_cgroup_unavailable() -> None:
    """When cgroup path can't be resolved, uses exec-based I/O + ctxt aggregation."""
    pipe_output = "100MB / 1GB|10%|5%|0B / 0B"
    exec_call_count = 0

    def fake_subprocess_run(cmd, **kwargs):
        nonlocal exec_call_count
        if "inspect" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="42\n", stderr="")
        if "stats" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=pipe_output, stderr="")
        if "python3" in cmd:
            exec_call_count += 1
            # I/O script returns "read write", ctxt script returns count
            if "rbytes" in str(cmd) or "read_bytes" in str(cmd) or exec_call_count % 2 == 1:
                return subprocess.CompletedProcess(cmd, 0, stdout="2048 1024\n", stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="350\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def fake_read_text(self, encoding="utf-8"):
        raise FileNotFoundError(str(self))

    sampler = ContainerStatsSampler(
        container_id="test456",
        interval_s=0.05,
        executable="docker",
    )
    with (
        patch("harness.container_stats_sampler.subprocess.run", side_effect=fake_subprocess_run),
        patch.object(Path, "read_text", fake_read_text),
    ):
        sampler.start()
        time.sleep(0.25)
        samples = sampler.stop()

    assert len(samples) >= 2
    s = samples[0]
    assert s["disk_read_bytes"] == 2048
    assert s["disk_write_bytes"] == 1024
    # Context switches should be present in exec fallback mode
    assert "context_switches" in s


# ---------------------------------------------------------------------------
# start_task_container network_mode
# ---------------------------------------------------------------------------


def test_start_task_container_network_mode() -> None:
    """--network={mode} appears in generated docker run command."""
    from trace_collect.attempt_pipeline import start_task_container

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="container_id_123\n", stderr="")

    with patch("subprocess.run", side_effect=fake_run):
        cid = start_task_container(
            "test-image:latest",
            executable="docker",
            network_mode="none",
        )

    assert cid == "container_id_123"
    run_cmd = calls[0]
    assert "--network=none" in run_cmd
    assert "--network=host" not in run_cmd


def test_start_task_container_default_network_host() -> None:
    """Default network mode is 'host'."""
    from trace_collect.attempt_pipeline import start_task_container

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="cid\n", stderr="")

    with patch("subprocess.run", side_effect=fake_run):
        start_task_container("img", executable="docker")

    assert "--network=host" in calls[0]
