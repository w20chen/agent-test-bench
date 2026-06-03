"""Tests for VLLMMetricsClient GPU tracking (US-4)."""

from __future__ import annotations

import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from harness.metrics_client import GpuPidNotFoundError, VLLMMetricsClient
from harness.scheduler_hooks import GpuBaseline


# Minimal Prometheus payload with gpu_cache_usage_perc = 50.0 (percent convention)
_PROMETHEUS_PAYLOAD = "\n".join([
    "vllm:num_preemptions_total 0",
    "vllm:gpu_cache_usage_perc 50.0",
    "vllm:cpu_cache_usage_perc 0.0",
    "vllm:gpu_prefix_cache_hit_rate 0.0",
    "vllm:cpu_prefix_cache_hit_rate 0.0",
])

# Minimal Prometheus payload with gpu_cache_usage_perc = 0.5 (fraction convention)
_PROMETHEUS_PAYLOAD_FRACTION = "\n".join([
    "vllm:num_preemptions_total 0",
    "vllm:gpu_cache_usage_perc 0.5",
    "vllm:cpu_cache_usage_perc 0.0",
    "vllm:gpu_prefix_cache_hit_rate 0.0",
    "vllm:cpu_prefix_cache_hit_rate 0.0",
])


def _make_http_server(body: str) -> tuple[ThreadingHTTPServer, str]:
    """Start a local HTTP server returning body; return (server, url)."""

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            payload = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/metrics"
    return server, url


def _baseline(weights_mib: float = 3400.0, kv_total_mib: float = 1000.0) -> GpuBaseline:
    return GpuBaseline(
        weights_mib=weights_mib,
        kv_cache_total_mib=kv_total_mib,
        model="test-model",
        dtype="float16",
        tensor_parallel_size=1,
    )


# ---------------------------------------------------------------------------
# is_gpu_tracking_enabled property tests
# ---------------------------------------------------------------------------

def test_gpu_tracking_disabled_when_no_baseline() -> None:
    client = VLLMMetricsClient("http://x", vllm_pid=123)
    assert client.is_gpu_tracking_enabled is False


def test_gpu_tracking_disabled_when_no_pid() -> None:
    client = VLLMMetricsClient("http://x", gpu_baseline=_baseline())
    assert client.is_gpu_tracking_enabled is False


def test_gpu_tracking_enabled_when_all_provided() -> None:
    client = VLLMMetricsClient("http://x", gpu_baseline=_baseline(), vllm_pid=42)
    assert client.is_gpu_tracking_enabled is True


# ---------------------------------------------------------------------------
# get_snapshot integration tests
# ---------------------------------------------------------------------------

def test_get_snapshot_includes_breakdown_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """baseline weights=3400, kv_total=1000; Prometheus gpu_cache_usage_perc=50.0 (percent);
    nvidia-smi total=4500 → kv_used=500.0, activations=600.0"""
    server, url = _make_http_server(_PROMETHEUS_PAYLOAD)
    try:
        monkeypatch.setattr(
            "harness.metrics_client.sample_nvidia_smi_per_pid",
            lambda pid: {"pid": pid, "gpu_index": 0, "memory_used_mib": 4500.0},
        )
        client = VLLMMetricsClient(url, gpu_baseline=_baseline(), vllm_pid=999)
        snap = client.get_snapshot()
        assert snap.gpu_memory_breakdown is not None
        bd = snap.gpu_memory_breakdown
        assert bd.kv_cache_used_mib == pytest.approx(500.0)
        assert bd.activations_mib == pytest.approx(600.0)
        assert bd.total_pid_mib == pytest.approx(4500.0)
        assert bd.weights_mib == pytest.approx(3400.0)
        assert bd.kv_cache_total_mib == pytest.approx(1000.0)
    finally:
        server.shutdown()


def test_get_snapshot_returns_none_breakdown_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """No baseline/pid → gpu_memory_breakdown is None; _sample_gpu_memory never called."""
    server, url = _make_http_server(_PROMETHEUS_PAYLOAD)
    called = []

    def _should_not_be_called(pid: int):  # type: ignore[return]
        called.append(pid)

    try:
        monkeypatch.setattr("harness.metrics_client.sample_nvidia_smi_per_pid", _should_not_be_called)
        client = VLLMMetricsClient(url)
        snap = client.get_snapshot()
        assert snap.gpu_memory_breakdown is None
        assert called == []
    finally:
        server.shutdown()


def test_get_snapshot_raises_when_pid_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    server, url = _make_http_server(_PROMETHEUS_PAYLOAD)
    try:
        monkeypatch.setattr(
            "harness.metrics_client.sample_nvidia_smi_per_pid",
            lambda pid: None,
        )
        client = VLLMMetricsClient(url, gpu_baseline=_baseline(), vllm_pid=404)
        with pytest.raises(GpuPidNotFoundError, match="404"):
            client.get_snapshot()
    finally:
        server.shutdown()


def test_negative_residual_clamps_to_zero(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """total=3500, weights=3400, kv_used=500 → residual=-400 → clamped to 0 with warning."""
    server, url = _make_http_server(_PROMETHEUS_PAYLOAD)
    try:
        monkeypatch.setattr(
            "harness.metrics_client.sample_nvidia_smi_per_pid",
            lambda pid: {"pid": pid, "gpu_index": 0, "memory_used_mib": 3500.0},
        )
        client = VLLMMetricsClient(url, gpu_baseline=_baseline(), vllm_pid=1)
        with caplog.at_level(logging.WARNING):
            snap = client.get_snapshot()
        assert snap.gpu_memory_breakdown is not None
        assert snap.gpu_memory_breakdown.activations_mib == pytest.approx(0.0)
        assert "clamping to 0" in caplog.text
    finally:
        server.shutdown()


def test_kv_usage_percent_or_fraction_handled(monkeypatch: pytest.MonkeyPatch) -> None:
    """gpu_cache_usage_perc=0.5 (fraction) → kv_used=500; 50.0 (percent) → also 500."""
    monkeypatch.setattr(
        "harness.metrics_client.sample_nvidia_smi_per_pid",
        lambda pid: {"pid": pid, "gpu_index": 0, "memory_used_mib": 5000.0},
    )

    # fraction convention (0.5)
    server_frac, url_frac = _make_http_server(_PROMETHEUS_PAYLOAD_FRACTION)
    try:
        client_frac = VLLMMetricsClient(url_frac, gpu_baseline=_baseline(), vllm_pid=1)
        snap_frac = client_frac.get_snapshot()
        assert snap_frac.gpu_memory_breakdown is not None
        assert snap_frac.gpu_memory_breakdown.kv_cache_used_mib == pytest.approx(500.0)
    finally:
        server_frac.shutdown()

    # percent convention (50.0)
    server_perc, url_perc = _make_http_server(_PROMETHEUS_PAYLOAD)
    try:
        client_perc = VLLMMetricsClient(url_perc, gpu_baseline=_baseline(), vllm_pid=1)
        snap_perc = client_perc.get_snapshot()
        assert snap_perc.gpu_memory_breakdown is not None
        assert snap_perc.gpu_memory_breakdown.kv_cache_used_mib == pytest.approx(500.0)
    finally:
        server_perc.shutdown()
