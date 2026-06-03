"""End-to-end integration test for GPU tracking (US-8).

All tests run on Mac without Docker, without vLLM, without nvidia-smi.
Strategy: use host-mode source traces so _prepare_host_session is called
(no container required), mock the OpenAI LLM client, mock nvidia-smi,
and spin up a local Prometheus HTTP server.
"""
from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from trace_collect.simulator import simulate

# ---------------------------------------------------------------------------
# Prometheus mock server
# ---------------------------------------------------------------------------

_PROMETHEUS_PAYLOAD = "\n".join([
    "vllm:num_preemptions_total 0",
    "vllm:gpu_cache_usage_perc 50.0",
    "vllm:cpu_cache_usage_perc 0.0",
    "vllm:gpu_prefix_cache_hit_rate 0.0",
    "vllm:cpu_prefix_cache_hit_rate 0.0",
])


def _make_http_server(body: str) -> tuple[ThreadingHTTPServer, str]:
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


# ---------------------------------------------------------------------------
# Fake OpenAI async streaming client (same pattern as test_simulate_cloud_model)
# ---------------------------------------------------------------------------

class _FakeStream:
    def __init__(self) -> None:
        self._emitted = False

    def __aiter__(self) -> "_FakeStream":
        return self

    async def __anext__(self):
        if self._emitted:
            raise StopAsyncIteration
        self._emitted = True
        delta = type("Delta", (), {"content": "x"})()
        choice = type("Choice", (), {"delta": delta})()
        return type("Chunk", (), {"choices": [choice]})()


class _FakeClient:
    class _Completions:
        async def create(self, **kwargs):
            return _FakeStream()

    class _Chat:
        def __init__(self) -> None:
            self.completions = _FakeClient._Completions()

    def __init__(self) -> None:
        self.chat = self._Chat()


# ---------------------------------------------------------------------------
# Minimal host-mode source trace fixture (3 LLM calls, 1 tool call)
# ---------------------------------------------------------------------------

_STARTUP_LOG = Path(__file__).parent / "fixtures" / "vllm_startup_0_5.log"

# Expected values from vllm_startup_0_5.log:
#   weights: 3.32 GiB → 3399.68 MiB
#   kv_cache_total: 0.5 GiB → 512.0 MiB
#   kv_cache_used: 50% of 512 → 256.0 MiB
_EXPECTED_WEIGHTS_MIB = 3.32 * 1024  # 3399.68
_EXPECTED_KV_TOTAL_MIB = 0.5 * 1024  # 512.0
_EXPECTED_KV_USED_MIB = 0.50 * _EXPECTED_KV_TOTAL_MIB  # 256.0
_FAKE_PID = 12345
_FAKE_TOTAL_MIB = 4500.0


def _write_host_trace(path: Path, agent_id: str) -> None:
    """Write a minimal host-mode trace with 3 LLM calls and 1 tool call."""
    records = [
        {
            "type": "trace_metadata",
            "trace_format_version": 5,
            "scaffold": "tongyi-deepresearch",
            "instance_id": agent_id,
            "model": "qwen-test",
            "mode": "collect",
            "execution_environment": "host",
        },
        {
            "type": "action",
            "action_type": "llm_call",
            "action_id": "llm_0",
            "agent_id": agent_id,
            "iteration": 0,
            "ts_start": 100.0,
            "ts_end": 100.2,
            "data": {
                "messages_in": [{"role": "user", "content": "solve the task"}],
                "raw_response": {"id": "r0"},
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "llm_latency_ms": 200.0,
            },
        },
        {
            "type": "action",
            "action_type": "tool_exec",
            "action_id": "tool_0",
            "agent_id": agent_id,
            "iteration": 0,
            "ts_start": 100.4,
            "ts_end": 100.45,
            "data": {
                "tool_name": "web_search",
                "tool_args": json.dumps({"query": "test"}),
                "tool_result": "some result",
                "duration_ms": 50.0,
                "success": True,
            },
        },
        {
            "type": "action",
            "action_type": "llm_call",
            "action_id": "llm_1",
            "agent_id": agent_id,
            "iteration": 1,
            "ts_start": 100.5,
            "ts_end": 100.7,
            "data": {
                "messages_in": [{"role": "user", "content": "continue"}],
                "raw_response": {"id": "r1"},
                "prompt_tokens": 20,
                "completion_tokens": 8,
                "llm_latency_ms": 200.0,
            },
        },
        {
            "type": "action",
            "action_type": "llm_call",
            "action_id": "llm_2",
            "agent_id": agent_id,
            "iteration": 2,
            "ts_start": 100.8,
            "ts_end": 101.0,
            "data": {
                "messages_in": [{"role": "user", "content": "finalize"}],
                "raw_response": {"id": "r2"},
                "prompt_tokens": 30,
                "completion_tokens": 10,
                "llm_latency_ms": 200.0,
            },
        },
        {
            "type": "summary",
            "agent_id": agent_id,
            "model": "qwen-test",
            "success": True,
            "n_iterations": 3,
            "elapsed_s": 1.0,
        },
    ]
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )


def _write_host_tasks(path: Path, agent_id: str) -> None:
    path.write_text(
        json.dumps([{
            "instance_id": agent_id,
            "problem_statement": f"problem for {agent_id}",
            "repo": None,
            "image_name": None,
            "docker_image": None,
        }]) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _patch_gpu_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock nvidia-smi to return a fixed PID/memory reading."""
    monkeypatch.setattr(
        "harness.metrics_client.sample_nvidia_smi_per_pid",
        lambda pid: {"pid": pid, "gpu_index": 0, "memory_used_mib": _FAKE_TOTAL_MIB},
    )


def _patch_fake_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "trace_collect.simulator.create_async_openai_client",
        lambda **_kw: _FakeClient(),
    )


def _patch_no_container(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fail_prepare(*args, **kwargs):
        raise AssertionError("host-mode simulation must not prepare a container")

    monkeypatch.setattr(
        "trace_collect.simulator._prepare_container_session",
        _fail_prepare,
    )


# ---------------------------------------------------------------------------
# Test 1: happy path — GPU tracking on writes breakdown into every llm_call
# ---------------------------------------------------------------------------

def test_simulate_local_model_with_gpu_tracking_on_writes_breakdown_to_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Full happy path: GPU breakdown present in every llm_call and gpu_resources.json written."""
    assert _STARTUP_LOG.exists(), f"missing fixture: {_STARTUP_LOG}"

    from harness.vllm_startup_parser import parse_startup_log_file
    gpu_baseline = parse_startup_log_file(_STARTUP_LOG)
    assert gpu_baseline is not None, "vllm_startup_0_5.log must parse successfully"

    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    agent_id = "gpu-tracking-task"
    _write_host_trace(trace_path, agent_id)
    _write_host_tasks(task_source, agent_id)

    _patch_no_container(monkeypatch)
    _patch_fake_llm(monkeypatch)
    _patch_gpu_mocks(monkeypatch)

    server, metrics_url = _make_http_server(_PROMETHEUS_PAYLOAD)
    try:
        trace_file = asyncio.run(
            simulate(
                source_trace=trace_path,
                task_source=task_source,
                output_dir=tmp_path / "out",
                mode="local_model",
                api_base="http://localhost:8000/v1",
                api_key="dummy",
                model="local-qwen",
                metrics_url=metrics_url,
                gpu_baseline=gpu_baseline,
                vllm_pid=_FAKE_PID,
                gpu_sample_hz=50.0,
            )
        )
    finally:
        server.shutdown()

    records = _read_jsonl(trace_file)
    llm_records = [
        r for r in records
        if r.get("type") == "action" and r.get("action_type") == "llm_call"
    ]
    assert len(llm_records) >= 1, "trace must have at least one llm_call"

    for r in llm_records:
        bd = r["data"]["sim_metrics"]["vllm_scheduler_snapshot"]["gpu_memory_breakdown"]
        assert bd is not None, f"gpu_memory_breakdown must be non-None for action {r.get('action_id')}"

        required_keys = {
            "gpu_index", "pid", "total_pid_mib",
            "weights_mib", "kv_cache_used_mib", "kv_cache_total_mib",
            "activations_mib", "ts",
        }
        assert required_keys <= bd.keys(), f"missing keys: {required_keys - bd.keys()}"

        assert bd["total_pid_mib"] == pytest.approx(_FAKE_TOTAL_MIB)
        assert bd["weights_mib"] == pytest.approx(_EXPECTED_WEIGHTS_MIB)
        assert bd["kv_cache_used_mib"] == pytest.approx(_EXPECTED_KV_USED_MIB)

    # gpu_resources.json must exist in the attempt dir
    attempt_dir = tmp_path / "out" / agent_id / "attempt_1"
    gpu_path = attempt_dir / "gpu_resources.json"
    assert gpu_path.exists(), "gpu_resources.json must be written to attempt dir"

    gpu_data = json.loads(gpu_path.read_text())
    assert gpu_data.get("gpu_baseline") is not None
    assert isinstance(gpu_data.get("gpu_samples"), list)
    assert len(gpu_data["gpu_samples"]) >= 1, "must have at least one GPU sample"


# ---------------------------------------------------------------------------
# Test 2: control — GPU tracking off → breakdown is None, no gpu_resources.json
# ---------------------------------------------------------------------------

def test_simulate_local_model_with_gpu_tracking_off_skips_breakdown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """With GPU tracking off (no gpu_baseline/vllm_pid), breakdown must be None."""
    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    agent_id = "no-gpu-task"
    _write_host_trace(trace_path, agent_id)
    _write_host_tasks(task_source, agent_id)

    _patch_no_container(monkeypatch)
    _patch_fake_llm(monkeypatch)
    # nvidia-smi must NOT be called when tracking is off
    monkeypatch.setattr(
        "harness.metrics_client.sample_nvidia_smi_per_pid",
        lambda pid: (_ for _ in ()).throw(AssertionError("nvidia-smi must not be called when gpu tracking off")),
    )

    server, metrics_url = _make_http_server(_PROMETHEUS_PAYLOAD)
    try:
        trace_file = asyncio.run(
            simulate(
                source_trace=trace_path,
                task_source=task_source,
                output_dir=tmp_path / "out",
                mode="local_model",
                api_base="http://localhost:8000/v1",
                api_key="dummy",
                model="local-qwen",
                metrics_url=metrics_url,
                # gpu_baseline omitted → None (default)
                # vllm_pid omitted → None (default)
            )
        )
    finally:
        server.shutdown()

    records = _read_jsonl(trace_file)
    llm_records = [
        r for r in records
        if r.get("type") == "action" and r.get("action_type") == "llm_call"
    ]
    assert len(llm_records) >= 1

    for r in llm_records:
        bd = r["data"]["sim_metrics"]["vllm_scheduler_snapshot"]["gpu_memory_breakdown"]
        assert bd is None, (
            f"gpu_memory_breakdown must be None when tracking is off, got {bd} "
            f"for action {r.get('action_id')}"
        )

    # gpu_resources.json must NOT exist
    attempt_dir = tmp_path / "out" / agent_id / "attempt_1"
    gpu_path = attempt_dir / "gpu_resources.json"
    assert not gpu_path.exists(), "gpu_resources.json must not be written when gpu tracking is off"


# ---------------------------------------------------------------------------
# Test 3: cross-source consistency — per-call weights_mib == gpu_resources.json weights_mib
# ---------------------------------------------------------------------------

def test_baseline_consistency_per_call_vs_resources_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """weights_mib in every trace llm_call must equal gpu_resources.json.gpu_baseline.weights_mib."""
    assert _STARTUP_LOG.exists(), f"missing fixture: {_STARTUP_LOG}"

    from harness.vllm_startup_parser import parse_startup_log_file
    gpu_baseline = parse_startup_log_file(_STARTUP_LOG)
    assert gpu_baseline is not None

    trace_path = tmp_path / "trace.jsonl"
    task_source = tmp_path / "tasks.json"
    agent_id = "consistency-task"
    _write_host_trace(trace_path, agent_id)
    _write_host_tasks(task_source, agent_id)

    _patch_no_container(monkeypatch)
    _patch_fake_llm(monkeypatch)
    _patch_gpu_mocks(monkeypatch)

    server, metrics_url = _make_http_server(_PROMETHEUS_PAYLOAD)
    try:
        trace_file = asyncio.run(
            simulate(
                source_trace=trace_path,
                task_source=task_source,
                output_dir=tmp_path / "out",
                mode="local_model",
                api_base="http://localhost:8000/v1",
                api_key="dummy",
                model="local-qwen",
                metrics_url=metrics_url,
                gpu_baseline=gpu_baseline,
                vllm_pid=_FAKE_PID,
                gpu_sample_hz=50.0,
            )
        )
    finally:
        server.shutdown()

    # Read gpu_resources.json baseline
    attempt_dir = tmp_path / "out" / agent_id / "attempt_1"
    gpu_path = attempt_dir / "gpu_resources.json"
    assert gpu_path.exists()
    gpu_data = json.loads(gpu_path.read_text())
    resources_weights_mib = gpu_data["gpu_baseline"]["weights_mib"]

    # Check every llm_call's breakdown matches
    records = _read_jsonl(trace_file)
    llm_records = [
        r for r in records
        if r.get("type") == "action" and r.get("action_type") == "llm_call"
    ]
    assert len(llm_records) >= 1

    for r in llm_records:
        bd = r["data"]["sim_metrics"]["vllm_scheduler_snapshot"]["gpu_memory_breakdown"]
        assert bd is not None
        assert bd["weights_mib"] == pytest.approx(resources_weights_mib), (
            f"per-call weights_mib={bd['weights_mib']} != "
            f"gpu_resources.json weights_mib={resources_weights_mib} "
            f"for action {r.get('action_id')}"
        )
