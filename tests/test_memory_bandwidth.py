from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from harness.memory_bandwidth import (
    INTEL_CAS_BYTES,
    MemoryBandwidthReading,
    attach_host_memory_bandwidth,
    detect_perf_backend,
    reset_host_memory_bandwidth_collector_for_tests,
    sample_memory_bandwidth_once,
)


@pytest.fixture(autouse=True)
def _reset_collector() -> None:
    reset_host_memory_bandwidth_collector_for_tests()
    yield
    reset_host_memory_bandwidth_collector_for_tests()


def test_detect_perf_backend_prefers_intel_imc(tmp_path: Path) -> None:
    events = tmp_path / "uncore_imc_0" / "events"
    events.mkdir(parents=True)
    (events / "cas_count_read").write_text("event=0x01\n", encoding="utf-8")
    (events / "cas_count_write").write_text("event=0x02\n", encoding="utf-8")

    backend = detect_perf_backend(tmp_path)

    assert backend is not None
    assert backend.kind == "intel_imc_cas"
    assert backend.read_specs == ("uncore_imc_0/cas_count_read/",)
    assert backend.write_specs == ("uncore_imc_0/cas_count_write/",)


def test_detect_perf_backend_falls_back_to_explicit_byte_events(tmp_path: Path) -> None:
    events = tmp_path / "ddrc0" / "events"
    events.mkdir(parents=True)
    (events / "read_bytes").write_text("event=0x11\n", encoding="utf-8")
    (events / "write_bytes").write_text("event=0x12\n", encoding="utf-8")

    backend = detect_perf_backend(tmp_path)

    assert backend is not None
    assert backend.kind == "explicit_byte_events"
    assert backend.read_specs == ("ddrc0/read_bytes/",)
    assert backend.write_specs == ("ddrc0/write_bytes/",)


def test_sample_memory_bandwidth_once_intel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    events = tmp_path / "uncore_imc_0" / "events"
    events.mkdir(parents=True)
    (events / "cas_count_read").write_text("", encoding="utf-8")
    (events / "cas_count_write").write_text("", encoding="utf-8")
    backend = detect_perf_backend(tmp_path)
    assert backend is not None

    def fake_run(cmd, **kwargs):
        assert "perf" in cmd[0]
        stderr = "\n".join(
            [
                "1000,,uncore_imc_0/cas_count_read/,1.00,100.00",
                "2000,,uncore_imc_0/cas_count_write/,1.00,100.00",
            ]
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=stderr)

    monkeypatch.setattr("harness.memory_bandwidth.subprocess.run", fake_run)
    reading = sample_memory_bandwidth_once(backend, interval_s=2.0)

    assert reading.available is True
    expected_read = 1000 * INTEL_CAS_BYTES / (2.0 * 1024 * 1024)
    expected_write = 2000 * INTEL_CAS_BYTES / (2.0 * 1024 * 1024)
    assert reading.read_mb_s == pytest.approx(expected_read)
    assert reading.write_mb_s == pytest.approx(expected_write)
    assert reading.total_mb_s == pytest.approx(expected_read + expected_write)


def test_sample_memory_bandwidth_once_permission_denied(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    events = tmp_path / "uncore_imc_0" / "events"
    events.mkdir(parents=True)
    (events / "cas_count_read").write_text("", encoding="utf-8")
    (events / "cas_count_write").write_text("", encoding="utf-8")
    backend = detect_perf_backend(tmp_path)
    assert backend is not None

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            255,
            stdout="",
            stderr="Error: No permission to enable uncore_imc_0/cas_count_read/ event.\n",
        )

    monkeypatch.setattr("harness.memory_bandwidth.subprocess.run", fake_run)
    reading = sample_memory_bandwidth_once(backend, interval_s=1.0)

    assert reading.available is False
    assert reading.reason == "permission_denied"


def test_attach_host_memory_bandwidth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "harness.memory_bandwidth.get_host_memory_bandwidth_sample",
        lambda **kwargs: MemoryBandwidthReading(
            available=True,
            source="perf:intel-imc-cas",
            total_mb_s=12.5,
            read_mb_s=7.5,
            write_mb_s=5.0,
        ),
    )
    sample: dict[str, object] = {}

    attach_host_memory_bandwidth(sample, interval_s=1.0)

    assert sample["memory_bandwidth_available"] is True
    assert sample["memory_bandwidth_source"] == "perf:intel-imc-cas"
    assert sample["memory_total_mb_s"] == 12.5
    assert sample["memory_read_mb_s"] == 7.5
    assert sample["memory_write_mb_s"] == 5.0
