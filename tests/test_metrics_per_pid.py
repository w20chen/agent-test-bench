from __future__ import annotations

import subprocess

import pytest

from harness.metrics import (
    sample_nvidia_smi_compute_apps,
    sample_nvidia_smi_per_pid,
    parse_nvidia_smi_csv,
)

# Two-process fake output for reuse across tests
_TWO_PROCESS_OUTPUT = "12345, GPU-1234abcd, 4096\n67890, GPU-1234abcd, 2048\n"
_GPU_QUERY_OUTPUT = "0, GPU-1234abcd\n"


def test_compute_apps_parses_two_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: _TWO_PROCESS_OUTPUT)
    rows = sample_nvidia_smi_compute_apps()
    assert len(rows) == 2
    assert rows[0] == {"pid": 12345, "gpu_serial": "GPU-1234abcd", "memory_used_mib": 4096.0}
    assert rows[1] == {"pid": 67890, "gpu_serial": "GPU-1234abcd", "memory_used_mib": 2048.0}
    assert isinstance(rows[0]["pid"], int)
    assert isinstance(rows[0]["memory_used_mib"], float)


def test_per_pid_returns_row_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_check_output(cmd: list[str], **kw: object) -> str:
        calls.append(cmd)
        if "--query-compute-apps" in cmd[1]:
            return _TWO_PROCESS_OUTPUT
        return _GPU_QUERY_OUTPUT

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)
    result = sample_nvidia_smi_per_pid(12345)
    assert result == {"pid": 12345, "gpu_index": 0, "memory_used_mib": 4096.0}
    assert len(calls) == 2  # one for apps, one for gpu index


def test_per_pid_returns_none_when_pid_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: _TWO_PROCESS_OUTPUT)
    assert sample_nvidia_smi_per_pid(99999) is None


def test_per_pid_returns_none_when_nvidia_smi_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_fnf(cmd: list[str], **kw: object) -> str:
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(subprocess, "check_output", raise_fnf)
    assert sample_nvidia_smi_per_pid(12345) is None


def test_compute_apps_handles_malformed_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    # Mix: one good row, one missing field, one non-numeric pid, one empty line
    fake = "12345, GPU-abc, 1024\nbadpid, GPU-abc, 512\nonly_one_field\n\n"
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: fake)
    rows = sample_nvidia_smi_compute_apps()
    assert len(rows) == 1
    assert rows[0]["pid"] == 12345


def test_compute_apps_returns_empty_when_no_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: "")
    assert sample_nvidia_smi_compute_apps() == []


def test_existing_sample_nvidia_smi_unchanged() -> None:
    # Regression: original parse_nvidia_smi_csv still parses device-aggregate format
    samples = parse_nvidia_smi_csv("10, 2000\n20, 2100\n")
    assert len(samples) == 2
    assert samples[0] == {"utilization_gpu": 10.0, "memory_used_mib": 2000.0}
    assert samples[1] == {"utilization_gpu": 20.0, "memory_used_mib": 2100.0}
