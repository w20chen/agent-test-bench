from __future__ import annotations

import asyncio
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from harness.metrics import (
    VLLMMetricsCollector,
    dump_nvidia_samples,
    parse_nvidia_smi_csv,
)


class MetricsHandler(BaseHTTPRequestHandler):
    body = "\n".join(
        [
            "# HELP vllm:num_requests_running running requests",
            "vllm:num_requests_running 2",
            "vllm:num_requests_waiting 1",
            "vllm:gpu_cache_usage_perc 75.5",
            "vllm:cpu_cache_usage_perc 0.0",
            "vllm:num_preemptions_total 4",
            "vllm:avg_prompt_throughput_toks_per_s 12.5",
            "vllm:avg_generation_throughput_toks_per_s 40.0",
        ]
    )

    def do_GET(self) -> None:  # noqa: N802
        payload = self.body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


def test_vllm_metrics_collector_polls_and_dumps(tmp_path: Path) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        collector = VLLMMetricsCollector(
            metrics_url=f"http://127.0.0.1:{server.server_address[1]}/metrics",
            gpu_sample_provider=lambda: [
                {"utilization_gpu": 10.0, "memory_used_mib": 2000.0}
            ],
        )
        snapshots = asyncio.run(collector.poll(interval_s=0.01, max_samples=2))
        assert len(snapshots) == 2
        assert snapshots[0]["vllm:num_requests_running"] == 2.0
        assert snapshots[0]["gpu_samples"][0]["memory_used_mib"] == 2000.0
        output = tmp_path / "metrics.json"
        collector.dump_json(output)
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert len(payload) == 2
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_vllm_metrics_collector_warns_on_incomplete_payload(
    caplog: logging.Handler,
) -> None:
    collector = VLLMMetricsCollector(metrics_url="http://localhost:8000/metrics")
    # Should NOT raise — now logs a warning instead
    with caplog.at_level(logging.WARNING):
        collector._validate_snapshot({"vllm:num_requests_running": 1.0})
    assert "Incomplete metrics snapshot" in caplog.text


def test_parse_nvidia_smi_csv_and_dump(tmp_path: Path) -> None:
    # noheader,nounits format: raw numbers without header row
    samples = parse_nvidia_smi_csv("10, 2000\n20, 2100\n")
    assert samples[0]["utilization_gpu"] == 10.0
    output = tmp_path / "gpu.json"
    dump_nvidia_samples(samples, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload[1]["memory_used_mib"] == 2100.0
