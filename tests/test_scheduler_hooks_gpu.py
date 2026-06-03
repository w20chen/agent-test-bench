"""Tests for GPU memory tracking dataclasses added in US-1."""

import json
from dataclasses import asdict

import pytest

from harness.scheduler_hooks import (
    GpuBaseline,
    GpuComponentBreakdown,
    GpuMemoryBreakdown,
    PreemptionSnapshot,
    empty_snapshot,
    parse_prometheus_metrics,
)


def _make_gpu_breakdown() -> GpuMemoryBreakdown:
    return GpuMemoryBreakdown(
        gpu_index=0,
        pid=12345,
        total_pid_mib=16384.0,
        weights_mib=8192.0,
        kv_cache_used_mib=1024.0,
        kv_cache_total_mib=4096.0,
        activations_mib=256.0,
        ts=1_700_000_000.0,
    )


def test_gpu_memory_breakdown_construct_and_asdict_roundtrip() -> None:
    bd = _make_gpu_breakdown()
    d = asdict(bd)
    # Must be JSON-serializable
    s = json.dumps(d)
    assert json.loads(s)["gpu_index"] == 0
    assert json.loads(s)["total_pid_mib"] == 16384.0


def test_gpu_baseline_construct_and_asdict_roundtrip() -> None:
    baseline = GpuBaseline(
        weights_mib=8192.0,
        kv_cache_total_mib=4096.0,
        model="meta-llama/Llama-3-8b",
        dtype="bfloat16",
        tensor_parallel_size=1,
    )
    d = asdict(baseline)
    s = json.dumps(d)
    assert json.loads(s)["model"] == "meta-llama/Llama-3-8b"
    assert json.loads(s)["tensor_parallel_size"] == 1


def test_gpu_component_breakdown_per_module_serializable() -> None:
    per_module = [{"path": "layers.0.attn", "class": "Attention", "value_mib": 12.5}]
    cbd = GpuComponentBreakdown(
        step_index=3,
        attn_mib=12.5,
        mlp_mib=8.0,
        other_activations_mib=2.0,
        per_module=per_module,
        measurement_kind="peak",
    )
    d = asdict(cbd)
    s = json.dumps(d)
    loaded = json.loads(s)
    assert loaded["per_module"][0]["path"] == "layers.0.attn"
    assert loaded["per_module"][0]["value_mib"] == 12.5


def test_preemption_snapshot_gpu_field_default_none() -> None:
    snap = empty_snapshot()
    assert snap.gpu_memory_breakdown is None


def test_preemption_snapshot_with_gpu_breakdown_serializes_nested() -> None:
    snap = PreemptionSnapshot(
        num_preemptions_total=5.0,
        gpu_cache_usage_perc=0.72,
        cpu_cache_usage_perc=0.1,
        gpu_prefix_cache_hit_rate=0.85,
        cpu_prefix_cache_hit_rate=0.0,
        gpu_memory_breakdown=_make_gpu_breakdown(),
    )
    d = asdict(snap)
    s = json.dumps(d)
    loaded = json.loads(s)
    assert isinstance(loaded["gpu_memory_breakdown"], dict)
    assert loaded["gpu_memory_breakdown"]["pid"] == 12345
    assert loaded["gpu_memory_breakdown"]["weights_mib"] == 8192.0


def test_existing_parse_prometheus_metrics_unchanged() -> None:
    # Minimal valid Prometheus text with the five fields vLLM emits
    payload = (
        "# HELP vllm:num_preemptions_total\n"
        "# TYPE vllm:num_preemptions_total counter\n"
        "vllm:num_preemptions_total 3.0\n"
        "# HELP vllm:gpu_cache_usage_perc\n"
        "# TYPE vllm:gpu_cache_usage_perc gauge\n"
        "vllm:gpu_cache_usage_perc 0.5\n"
        "# HELP vllm:cpu_cache_usage_perc\n"
        "# TYPE vllm:cpu_cache_usage_perc gauge\n"
        "vllm:cpu_cache_usage_perc 0.0\n"
        "# HELP vllm:gpu_prefix_cache_hit_rate\n"
        "# TYPE vllm:gpu_prefix_cache_hit_rate gauge\n"
        "vllm:gpu_prefix_cache_hit_rate 0.9\n"
        "# HELP vllm:cpu_prefix_cache_hit_rate\n"
        "# TYPE vllm:cpu_prefix_cache_hit_rate gauge\n"
        "vllm:cpu_prefix_cache_hit_rate 0.0\n"
    )
    snap = parse_prometheus_metrics(payload)
    assert snap.gpu_memory_breakdown is None
    assert snap.num_preemptions_total == 3.0
    assert snap.gpu_cache_usage_perc == 0.5
