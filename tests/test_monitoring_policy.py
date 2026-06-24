from __future__ import annotations

from pathlib import Path

import pytest

from harness.container_stats_sampler import ContainerStatsSampler
from harness.process_stats_sampler import ProcessStatsSampler
from trace_collect.cli import parse_collect_args, parse_simulate_args
from trace_collect.monitoring import (
    resolve_collect_monitoring,
    resolve_ksys_request,
    resolve_simulate_monitoring,
)


def test_collect_auto_preserves_serial_defaults() -> None:
    policy = resolve_collect_monitoring(
        resource="auto",
        pmu="auto",
        ksys="auto",
        concurrency=1,
        execution_environment="container",
    )

    assert policy.resource_enabled is True
    assert policy.pmu_enabled is True
    assert policy.memory_bandwidth_enabled is True
    assert policy.ksys_enabled is False


def test_collect_auto_disables_all_builtin_monitoring_concurrently() -> None:
    policy = resolve_collect_monitoring(
        resource="auto",
        pmu="auto",
        ksys="auto",
        concurrency=2,
        execution_environment="container",
    )

    assert policy.resource_enabled is False
    assert policy.pmu_enabled is False
    assert policy.memory_bandwidth_enabled is False


def test_collect_concurrent_resource_on_never_enables_pmu_or_bandwidth() -> None:
    policy = resolve_collect_monitoring(
        resource="on",
        pmu="auto",
        ksys="off",
        concurrency=2,
        execution_environment="container",
    )

    assert policy.resource_enabled is True
    assert policy.pmu_enabled is False
    assert policy.memory_bandwidth_enabled is False


def test_collect_concurrent_pmu_on_is_forbidden() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        resolve_collect_monitoring(
            resource="on",
            pmu="on",
            ksys="off",
            concurrency=2,
            execution_environment="container",
        )


def test_collect_concurrent_host_resource_on_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be isolated"):
        resolve_collect_monitoring(
            resource="on",
            pmu="off",
            ksys="off",
            concurrency=2,
            execution_environment="host",
        )


def test_simulate_concurrent_container_keeps_base_sampler_without_perf() -> None:
    policy = resolve_simulate_monitoring(
        resource="auto",
        pmu="auto",
        ksys="auto",
        concurrent=True,
        has_host_session=False,
        has_container_session=True,
    )

    assert policy.resource_enabled is True
    assert policy.pmu_enabled is False
    assert policy.memory_bandwidth_enabled is False


def test_simulate_host_auto_disables_builtin_monitoring() -> None:
    policy = resolve_simulate_monitoring(
        resource="auto",
        pmu="auto",
        ksys="on",
        concurrent=False,
        has_host_session=True,
        has_container_session=False,
    )

    assert policy.resource_enabled is False
    assert policy.pmu_enabled is False
    assert policy.memory_bandwidth_enabled is False
    assert policy.ksys_enabled is True


def test_simulate_host_resource_on_is_rejected() -> None:
    with pytest.raises(ValueError, match="no isolated process PID"):
        resolve_simulate_monitoring(
            resource="on",
            pmu="off",
            ksys="off",
            concurrent=False,
            has_host_session=True,
            has_container_session=False,
        )


def test_pmu_on_requires_base_resource_monitoring() -> None:
    with pytest.raises(ValueError, match="requires built-in resource monitoring"):
        resolve_collect_monitoring(
            resource="off",
            pmu="on",
            ksys="off",
            concurrency=1,
            execution_environment="container",
        )


def test_legacy_ksys_alias_and_conflict() -> None:
    assert resolve_ksys_request("auto", legacy_ksys=True) == "on"
    with pytest.raises(ValueError, match="conflicts"):
        resolve_ksys_request("off", legacy_ksys=True)


def test_monitoring_cli_defaults_are_auto() -> None:
    collect = parse_collect_args(
        ["--provider", "openrouter", "--model", "test-model"]
    )
    simulate = parse_simulate_args(
        ["--source-trace", "trace.jsonl", "--provider", "openai", "--model", "m"]
    )

    for args in (collect, simulate):
        assert args.resource_monitoring == "auto"
        assert args.pmu_monitoring == "auto"
        assert args.ksys_monitoring == "auto"
        assert args.ksys is False


def test_monitoring_cli_accepts_explicit_values() -> None:
    args = parse_simulate_args(
        [
            "--source-trace",
            "trace.jsonl",
            "--provider",
            "openai",
            "--model",
            "m",
            "--resource-monitoring",
            "off",
            "--pmu-monitoring",
            "off",
            "--ksys-monitoring",
            "on",
        ]
    )

    assert args.resource_monitoring == "off"
    assert args.pmu_monitoring == "off"
    assert args.ksys_monitoring == "on"


def test_process_sampler_can_disable_perf_collectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"pmu": 0, "bandwidth": 0}
    monkeypatch.setattr(
        "harness.process_stats_sampler._sample_with_psutil",
        lambda *args, **kwargs: {
            "mem_usage": "1MiB",
            "mem_percent": "0%",
            "cpu_percent": "0%",
        },
    )
    monkeypatch.setattr(
        "harness.process_stats_sampler._read_proc_context_switches",
        lambda pid: None,
    )
    monkeypatch.setattr(
        "harness.process_stats_sampler._read_proc_net_dev",
        lambda pid: None,
    )
    monkeypatch.setattr(
        "harness.process_stats_sampler.attach_micro_arch",
        lambda *args, **kwargs: calls.__setitem__("pmu", calls["pmu"] + 1),
    )
    monkeypatch.setattr(
        "harness.process_stats_sampler.attach_host_memory_bandwidth",
        lambda *args, **kwargs: calls.__setitem__(
            "bandwidth", calls["bandwidth"] + 1
        ),
    )
    sampler = ProcessStatsSampler(
        pid=1,
        enable_pmu=False,
        enable_memory_bandwidth=False,
    )

    assert sampler._collect_sample() is not None
    assert calls == {"pmu": 0, "bandwidth": 0}


def test_container_sampler_can_disable_perf_collectors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = {"pmu": 0, "bandwidth": 0}
    sampler = ContainerStatsSampler(
        "container-id",
        enable_pmu=False,
        enable_memory_bandwidth=False,
    )
    sampler._io_mode = "cgroup"
    sampler._cgroup_path = tmp_path
    monkeypatch.setattr(
        "harness.container_stats_sampler._read_cgroup_io_stat",
        lambda path: {"read_bytes": 1, "write_bytes": 2},
    )
    monkeypatch.setattr(
        "harness.container_stats_sampler._read_cgroup_pids",
        lambda path: [],
    )
    monkeypatch.setattr(
        "harness.container_stats_sampler.attach_micro_arch",
        lambda *args, **kwargs: calls.__setitem__("pmu", calls["pmu"] + 1),
    )
    monkeypatch.setattr(
        "harness.container_stats_sampler.attach_host_memory_bandwidth",
        lambda *args, **kwargs: calls.__setitem__(
            "bandwidth", calls["bandwidth"] + 1
        ),
    )

    sample: dict[str, object] = {}
    sampler._sample_io(sample)
    assert calls == {"pmu": 0, "bandwidth": 0}
