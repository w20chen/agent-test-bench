from __future__ import annotations

import os
import sys
import time
import types
from types import SimpleNamespace

from harness.container_stats_sampler import summarize_samples
from harness.process_stats_sampler import ProcessStatsSampler, _sample_with_psutil


def test_process_stats_sampler_collects_current_process_sample() -> None:
    sampler = ProcessStatsSampler(pid=os.getpid(), interval_s=0.01)
    sampler.start()
    time.sleep(0.03)
    samples = sampler.stop()

    assert samples
    assert "epoch" in samples[0]
    assert "cpu_percent" in samples[0]
    assert "mem_usage" in samples[0]
    summary = summarize_samples(samples)
    assert summary["sample_count"] == len(samples)


def test_process_stats_sampler_stop_samples_fast_process_attempt() -> None:
    sampler = ProcessStatsSampler(pid=os.getpid(), interval_s=60.0)
    samples = sampler.stop()

    assert len(samples) == 1
    assert "mem_usage" in samples[0]


def test_psutil_sampler_aggregates_recursive_children(monkeypatch) -> None:
    class FakeProcess:
        def __init__(self, pid: int, rss: int, cpu: float) -> None:
            self.pid = pid
            self._rss = rss
            self._cpu = cpu

        def children(self, *, recursive: bool):
            assert recursive is True
            return [
                FakeProcess(2, rss=20 * 1024 * 1024, cpu=2.5),
                FakeProcess(3, rss=30 * 1024 * 1024, cpu=3.5),
            ]

        def memory_info(self):
            return SimpleNamespace(rss=self._rss)

        def cpu_percent(self, *, interval=None):
            assert interval is None
            return self._cpu

        def io_counters(self):
            return SimpleNamespace(read_bytes=self.pid * 100, write_bytes=self.pid * 10)

        def num_ctx_switches(self):
            return SimpleNamespace(voluntary=self.pid, involuntary=self.pid + 1)

    fake_psutil = types.ModuleType("psutil")
    fake_psutil.Process = lambda pid: FakeProcess(pid, rss=10 * 1024 * 1024, cpu=1.5)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    sample = _sample_with_psutil(1)

    assert sample is not None
    assert sample["mem_usage"] == "60.000MiB"
    assert sample["cpu_percent"] == "7.500%"
    assert sample["disk_read_bytes"] == 600
    assert sample["disk_write_bytes"] == 60
    assert sample["context_switches"] == 15
    assert sample["process_count"] == 3


def test_psutil_sampler_keeps_root_process_when_child_enumeration_fails(
    monkeypatch,
) -> None:
    class FakeProcess:
        pid = 1

        def children(self, *, recursive: bool):
            assert recursive is True
            raise PermissionError("process list unavailable")

        def memory_info(self):
            return SimpleNamespace(rss=44 * 1024 * 1024)

        def cpu_percent(self, *, interval=None):
            assert interval is None
            return 12.25

        def io_counters(self):
            return SimpleNamespace(read_bytes=4096, write_bytes=8192)

        def num_ctx_switches(self):
            return SimpleNamespace(voluntary=3, involuntary=4)

    fake_psutil = types.ModuleType("psutil")
    fake_psutil.Process = lambda pid: FakeProcess()
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    sample = _sample_with_psutil(1)

    assert sample is not None
    assert sample["mem_usage"] == "44.000MiB"
    assert sample["cpu_percent"] == "12.250%"
    assert sample["disk_read_bytes"] == 4096
    assert sample["disk_write_bytes"] == 8192
    assert sample["context_switches"] == 7
    assert "process_count" not in sample


def test_process_sampler_keeps_psutil_child_io_over_proc_parent(monkeypatch) -> None:
    sampler = ProcessStatsSampler(pid=1, interval_s=60.0)
    monkeypatch.setattr(
        "harness.process_stats_sampler._sample_with_psutil",
        lambda pid, *, process_cache: {
            "epoch": 1.0,
            "timestamp": "ts",
            "mem_usage": "60MiB",
            "mem_percent": "0%",
            "cpu_percent": "7.5%",
            "disk_read_bytes": 600,
            "disk_write_bytes": 60,
            "context_switches": 15,
        },
    )
    monkeypatch.setattr(
        "harness.process_stats_sampler._read_proc_io",
        lambda pid: {"read_bytes": 1, "write_bytes": 1},
    )
    monkeypatch.setattr(
        "harness.process_stats_sampler._read_proc_context_switches",
        lambda pid: 1,
    )

    sample = sampler._collect_sample()

    assert sample is not None
    assert sample["disk_read_bytes"] == 600
    assert sample["disk_write_bytes"] == 60
    assert sample["context_switches"] == 15


def test_process_sampler_attaches_host_memory_bandwidth(monkeypatch) -> None:
    sampler = ProcessStatsSampler(pid=1, interval_s=60.0)
    monkeypatch.setattr(
        "harness.process_stats_sampler._sample_with_psutil",
        lambda pid, *, process_cache: {
            "epoch": 1.0,
            "timestamp": "ts",
            "mem_usage": "60MiB",
            "mem_percent": "0%",
            "cpu_percent": "7.5%",
        },
    )
    monkeypatch.setattr(
        "harness.process_stats_sampler._read_proc_io",
        lambda pid: None,
    )
    monkeypatch.setattr(
        "harness.process_stats_sampler._read_proc_context_switches",
        lambda pid: None,
    )
    monkeypatch.setattr(
        "harness.process_stats_sampler.attach_host_memory_bandwidth",
        lambda sample, *, interval_s: sample.update(
            {
                "memory_bandwidth_available": True,
                "memory_bandwidth_source": "perf:test",
                "memory_total_mb_s": 12.0,
                "memory_read_mb_s": 7.0,
                "memory_write_mb_s": 5.0,
            }
        ),
    )

    sample = sampler._collect_sample()

    assert sample is not None
    assert sample["memory_total_mb_s"] == 12.0
    assert sample["memory_read_mb_s"] == 7.0
    assert sample["memory_write_mb_s"] == 5.0


def test_psutil_sampler_reuses_process_handles_for_cpu_deltas(monkeypatch) -> None:
    created: list[int] = []

    class FakeProcess:
        def __init__(self, pid: int) -> None:
            created.append(pid)
            self.pid = pid
            self.calls = 0

        def children(self, *, recursive: bool):
            assert recursive is True
            return []

        def memory_info(self):
            return SimpleNamespace(rss=10 * 1024 * 1024)

        def cpu_percent(self, *, interval=None):
            assert interval is None
            self.calls += 1
            return 0.0 if self.calls == 1 else 12.5

        def io_counters(self):
            return SimpleNamespace(read_bytes=100, write_bytes=10)

        def num_ctx_switches(self):
            return SimpleNamespace(voluntary=1, involuntary=2)

    fake_psutil = types.ModuleType("psutil")
    fake_psutil.Process = FakeProcess
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    cache: dict[int, object] = {}

    first = _sample_with_psutil(1, process_cache=cache)
    second = _sample_with_psutil(1, process_cache=cache)

    assert created == [1]
    assert first is not None
    assert second is not None
    assert first["cpu_percent"] == "12.500%"
    assert second["cpu_percent"] == "12.500%"


def test_psutil_sampler_primes_new_child_handles_before_sampling(monkeypatch) -> None:
    class FakeChild:
        def __init__(self, pid: int) -> None:
            self.pid = pid
            self.calls = 0

        def cpu_percent(self, *, interval=None):
            assert interval is None
            self.calls += 1
            return 0.0 if self.calls == 1 else 4.0

        def memory_info(self):
            return SimpleNamespace(rss=5 * 1024 * 1024)

        def io_counters(self):
            return SimpleNamespace(read_bytes=10, write_bytes=20)

        def num_ctx_switches(self):
            return SimpleNamespace(voluntary=1, involuntary=1)

    class FakeRoot:
        pid = 1

        def __init__(self) -> None:
            self.calls = 0
            self.child = FakeChild(2)

        def children(self, *, recursive: bool):
            assert recursive is True
            return [self.child]

        def cpu_percent(self, *, interval=None):
            assert interval is None
            self.calls += 1
            return 0.0 if self.calls == 1 else 6.0

        def memory_info(self):
            return SimpleNamespace(rss=10 * 1024 * 1024)

        def io_counters(self):
            return SimpleNamespace(read_bytes=100, write_bytes=50)

        def num_ctx_switches(self):
            return SimpleNamespace(voluntary=2, involuntary=3)

    root = FakeRoot()
    fake_psutil = types.ModuleType("psutil")
    fake_psutil.Process = lambda pid: root
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    cache: dict[int, object] = {}

    sample = _sample_with_psutil(1, process_cache=cache)

    assert sample is not None
    assert sample["cpu_percent"] == "10.000%"
