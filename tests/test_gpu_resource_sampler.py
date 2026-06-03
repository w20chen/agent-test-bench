"""Tests for GpuResourceSampler (US-5)."""

from __future__ import annotations

import asyncio
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from harness.gpu_resource_sampler import GpuResourceSampler
from harness.scheduler_hooks import GpuBaseline


# Minimal Prometheus payload with gpu_cache_usage_perc = 50.0
_PROMETHEUS_PAYLOAD = "\n".join([
    "vllm:num_preemptions_total 0",
    "vllm:gpu_cache_usage_perc 50.0",
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
# test 1: sampler writes baseline and samples
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sampler_writes_baseline_and_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """50 Hz × 0.2 s ≈ 10 samples; output has gpu_baseline and gpu_samples."""
    # nvidia-smi fake: total=4500 MiB
    monkeypatch.setattr(
        "harness.metrics_client.sample_nvidia_smi_per_pid",
        lambda pid: {"pid": pid, "gpu_index": 0, "memory_used_mib": 4500.0},
    )
    server, url = _make_http_server(_PROMETHEUS_PAYLOAD)
    try:
        output = tmp_path / "gpu_resources.json"
        sampler = GpuResourceSampler(
            metrics_url=url,
            gpu_baseline=_baseline(),
            vllm_pid=123,
            output_path=output,
            sample_hz=50.0,
        )
        await sampler.start()
        await asyncio.sleep(0.2)
        await sampler.stop()

        assert output.exists()
        import json
        data = json.loads(output.read_text())

        assert "gpu_baseline" in data
        assert "gpu_samples" in data
        assert "summary" in data

        assert data["summary"]["n_samples"] > 0
        assert len(data["gpu_samples"]) >= 5  # 50 Hz × 0.2s ≈ 10
        assert data["summary"]["peak_total_pid_mib"] == pytest.approx(4500.0)

        # Baseline fields round-trip correctly
        assert data["gpu_baseline"]["weights_mib"] == pytest.approx(3400.0)
        assert data["gpu_baseline"]["kv_cache_total_mib"] == pytest.approx(1000.0)
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# test 2: fails fast when PID missing at start
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sampler_fails_fast_when_pid_missing_at_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When nvidia-smi returns None, start() raises GpuPidNotFoundError; no file written."""
    from harness.metrics_client import GpuPidNotFoundError

    monkeypatch.setattr(
        "harness.metrics_client.sample_nvidia_smi_per_pid",
        lambda pid: None,
    )
    server, url = _make_http_server(_PROMETHEUS_PAYLOAD)
    try:
        output = tmp_path / "gpu_resources.json"
        sampler = GpuResourceSampler(
            metrics_url=url,
            gpu_baseline=_baseline(),
            vllm_pid=404,
            output_path=output,
            sample_hz=50.0,
        )
        with pytest.raises(GpuPidNotFoundError):
            await sampler.start()

        assert not output.exists()
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# test 3: stop is idempotent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sampler_stop_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Calling stop() twice must not raise."""
    monkeypatch.setattr(
        "harness.metrics_client.sample_nvidia_smi_per_pid",
        lambda pid: {"pid": pid, "gpu_index": 0, "memory_used_mib": 4000.0},
    )
    server, url = _make_http_server(_PROMETHEUS_PAYLOAD)
    try:
        output = tmp_path / "gpu_resources.json"
        sampler = GpuResourceSampler(
            metrics_url=url,
            gpu_baseline=_baseline(),
            vllm_pid=1,
            output_path=output,
            sample_hz=50.0,
        )
        await sampler.start()
        await sampler.stop()
        await sampler.stop()  # must not raise
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# test 4: PID disappears mid-run
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sampler_handles_pid_disappearing_mid_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """After 2 successful samples, PID vanishes → sampler logs error, loop halts,
    flush still writes what was collected (n_samples >= 1)."""
    call_count = 0

    def _fake_nvidia_smi(pid: int) -> dict | None:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return {"pid": pid, "gpu_index": 0, "memory_used_mib": 4000.0}
        return None

    monkeypatch.setattr(
        "harness.metrics_client.sample_nvidia_smi_per_pid",
        _fake_nvidia_smi,
    )
    server, url = _make_http_server(_PROMETHEUS_PAYLOAD)
    try:
        output = tmp_path / "gpu_resources.json"
        sampler = GpuResourceSampler(
            metrics_url=url,
            gpu_baseline=_baseline(),
            vllm_pid=1,
            output_path=output,
            sample_hz=50.0,
        )
        with caplog.at_level(logging.ERROR, logger="harness.gpu_resource_sampler"):
            await sampler.start()
            await asyncio.sleep(0.5)
            await sampler.stop()

        assert "PID disappeared" in caplog.text

        import json
        assert output.exists()
        data = json.loads(output.read_text())
        assert data["summary"]["n_samples"] >= 1
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# test 5: invalid sample_hz raises ValueError
# ---------------------------------------------------------------------------

def test_sampler_invalid_hz_raises() -> None:
    """sample_hz <= 0 must raise ValueError at construction."""
    bl = _baseline()
    with pytest.raises(ValueError, match="sample_hz"):
        GpuResourceSampler(
            metrics_url="http://x",
            gpu_baseline=bl,
            vllm_pid=1,
            output_path=Path("/tmp/x.json"),
            sample_hz=0,
        )
    with pytest.raises(ValueError, match="sample_hz"):
        GpuResourceSampler(
            metrics_url="http://x",
            gpu_baseline=bl,
            vllm_pid=1,
            output_path=Path("/tmp/x.json"),
            sample_hz=-5.0,
        )


# ---------------------------------------------------------------------------
# test 6: output parent directory created on flush
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_output_path_parent_dir_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Passing a path under a non-existent dir → directory is created on flush."""
    monkeypatch.setattr(
        "harness.metrics_client.sample_nvidia_smi_per_pid",
        lambda pid: {"pid": pid, "gpu_index": 0, "memory_used_mib": 4000.0},
    )
    server, url = _make_http_server(_PROMETHEUS_PAYLOAD)
    try:
        output = tmp_path / "nested" / "deep" / "gpu_resources.json"
        assert not output.parent.exists()

        sampler = GpuResourceSampler(
            metrics_url=url,
            gpu_baseline=_baseline(),
            vllm_pid=1,
            output_path=output,
            sample_hz=50.0,
        )
        await sampler.start()
        await sampler.stop()

        assert output.exists()
    finally:
        server.shutdown()
