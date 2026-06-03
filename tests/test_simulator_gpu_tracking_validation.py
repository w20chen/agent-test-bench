"""Tests for GPU tracking CLI validation and wiring (US-6).

All tests operate on pure functions or lightweight monkeypatched paths —
no real vLLM, no real nvidia-smi, no real containers.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from trace_collect.simulator import validate_gpu_tracking_args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _args(
    *,
    gpu_tracking: str = "off",
    mode: str = "local_model",
    metrics_url: str | None = None,
    vllm_pid: int | None = None,
    vllm_startup_log: Path | None = None,
    gpu_sample_hz: float = 10.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        gpu_tracking=gpu_tracking,
        mode=mode,
        metrics_url=metrics_url,
        vllm_pid=vllm_pid,
        vllm_startup_log=vllm_startup_log,
        gpu_sample_hz=gpu_sample_hz,
    )


# ---------------------------------------------------------------------------
# validate_gpu_tracking_args: required-arg checks
# ---------------------------------------------------------------------------

def test_gpu_tracking_off_is_always_valid() -> None:
    # All other fields absent — no error when off
    validate_gpu_tracking_args(_args(gpu_tracking="off"))


def test_gpu_tracking_on_requires_metrics_url() -> None:
    with pytest.raises(ValueError, match="--metrics-url"):
        validate_gpu_tracking_args(
            _args(
                gpu_tracking="on",
                metrics_url=None,
                vllm_pid=12345,
                vllm_startup_log=Path("/tmp/startup.log"),
            )
        )


def test_gpu_tracking_on_requires_vllm_pid() -> None:
    with pytest.raises(ValueError, match="--vllm-pid"):
        validate_gpu_tracking_args(
            _args(
                gpu_tracking="on",
                metrics_url="http://localhost:8000/metrics",
                vllm_pid=None,
                vllm_startup_log=Path("/tmp/startup.log"),
            )
        )


def test_gpu_tracking_on_requires_startup_log() -> None:
    with pytest.raises(ValueError, match="--vllm-startup-log"):
        validate_gpu_tracking_args(
            _args(
                gpu_tracking="on",
                metrics_url="http://localhost:8000/metrics",
                vllm_pid=12345,
                vllm_startup_log=None,
            )
        )


def test_gpu_tracking_on_forbidden_in_cloud_model_mode() -> None:
    with pytest.raises(ValueError, match="cloud_model"):
        validate_gpu_tracking_args(
            _args(
                gpu_tracking="on",
                mode="cloud_model",
                metrics_url="http://localhost:8000/metrics",
                vllm_pid=12345,
                vllm_startup_log=Path("/tmp/startup.log"),
            )
        )


def test_gpu_tracking_on_all_args_provided_passes() -> None:
    # No exception when all required args are present
    validate_gpu_tracking_args(
        _args(
            gpu_tracking="on",
            mode="local_model",
            metrics_url="http://localhost:8000/metrics",
            vllm_pid=12345,
            vllm_startup_log=Path("/tmp/startup.log"),
        )
    )


# ---------------------------------------------------------------------------
# CLI _run_simulate: startup-log parse failure raises SystemExit(2)
# ---------------------------------------------------------------------------

def test_startup_log_parse_failure_raises(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """_run_simulate must exit(2) when parse_startup_log_file returns None."""
    from trace_collect.cli import _run_simulate, parse_simulate_args

    garbage_log = tmp_path / "startup.log"
    garbage_log.write_text("this is not a vllm startup log\n", encoding="utf-8")

    args = parse_simulate_args(
        [
            "--mode", "local_model",
            "--source-trace", "trace.jsonl",
            "--provider", "openai",
            "--api-base", "http://localhost:8000/v1",
            "--api-key", "dummy",
            "--model", "local-model",
            "--gpu-tracking", "on",
            "--vllm-pid", "9999",
            "--vllm-startup-log", str(garbage_log),
            "--metrics-url", "http://localhost:8000/metrics",
        ]
    )

    with pytest.raises(SystemExit, match="2"):
        _run_simulate(args)

    stderr = capsys.readouterr().err
    assert "Failed to parse vLLM startup log" in stderr
    assert str(garbage_log) in stderr


# ---------------------------------------------------------------------------
# CLI _run_simulate: gpu_tracking off does not construct sampler/baseline client
# ---------------------------------------------------------------------------

def test_gpu_tracking_off_does_not_alter_behavior(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """With --gpu-tracking off, simulate() is called without gpu_baseline/vllm_pid."""
    from trace_collect.cli import _run_simulate, parse_simulate_args

    seen_kwargs: dict = {}

    async def fake_simulate(**kwargs):  # type: ignore[misc]
        seen_kwargs.update(kwargs)
        return tmp_path / "out.jsonl"

    monkeypatch.setattr(
        "trace_collect.cli.resolve_llm_config",
        lambda **_kw: SimpleNamespace(
            api_base="http://localhost:8000/v1",
            api_key="dummy",
            model="local-model",
            env_key="OPENAI_API_KEY",
        ),
    )
    monkeypatch.setattr("trace_collect.simulator.simulate", fake_simulate)

    args = parse_simulate_args(
        [
            "--mode", "local_model",
            "--source-trace", "trace.jsonl",
            "--provider", "openai",
            "--api-base", "http://localhost:8000/v1",
            "--api-key", "dummy",
            "--model", "local-model",
            # gpu-tracking is off (default)
        ]
    )

    _run_simulate(args)

    # gpu_baseline and vllm_pid must not appear when tracking is off
    assert "gpu_baseline" not in seen_kwargs
    assert "vllm_pid" not in seen_kwargs
    assert "gpu_sample_hz" not in seen_kwargs


# ---------------------------------------------------------------------------
# CLI _run_simulate: cloud_model + gpu_tracking on → SystemExit(2) via validate
# ---------------------------------------------------------------------------

def test_gpu_tracking_on_cloud_model_exits_via_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """validate_gpu_tracking_args is called before any work in _run_simulate."""
    from trace_collect.cli import _run_simulate, parse_simulate_args

    args = parse_simulate_args(
        [
            "--mode", "cloud_model",
            "--source-trace", "trace.jsonl",
            "--gpu-tracking", "on",
            "--vllm-pid", "9999",
            "--vllm-startup-log", "/tmp/startup.log",
            "--metrics-url", "http://localhost:8000/metrics",
        ]
    )

    with pytest.raises(SystemExit, match="2"):
        _run_simulate(args)

    stderr = capsys.readouterr().err
    assert "cloud_model" in stderr


# ---------------------------------------------------------------------------
# CLI _run_simulate: gpu_tracking on + all valid args → calls simulate with baseline
# ---------------------------------------------------------------------------

def test_gpu_tracking_on_valid_args_passes_baseline_to_simulate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When gpu_tracking on and startup log parses OK, simulate() receives gpu_baseline."""
    from harness.scheduler_hooks import GpuBaseline
    from trace_collect.cli import _run_simulate, parse_simulate_args

    startup_log = tmp_path / "startup.log"
    startup_log.write_text("(irrelevant content)", encoding="utf-8")

    fake_baseline = GpuBaseline(
        weights_mib=10240.0,
        kv_cache_total_mib=4096.0,
        model="Qwen/Qwen3-32B",
        dtype="bfloat16",
        tensor_parallel_size=1,
    )

    # _run_simulate does a local import: `from harness.vllm_startup_parser import parse_startup_log_file`
    # Patching at the module level makes it visible to the already-imported reference.
    monkeypatch.setattr(
        "harness.vllm_startup_parser.parse_startup_log_file",
        lambda _path: fake_baseline,
    )

    seen_kwargs: dict = {}

    async def fake_simulate(**kwargs):  # type: ignore[misc]
        seen_kwargs.update(kwargs)
        return tmp_path / "out.jsonl"

    monkeypatch.setattr(
        "trace_collect.cli.resolve_llm_config",
        lambda **_kw: SimpleNamespace(
            api_base="http://localhost:8000/v1",
            api_key="dummy",
            model="local-model",
            env_key="OPENAI_API_KEY",
        ),
    )
    monkeypatch.setattr("trace_collect.simulator.simulate", fake_simulate)

    args = parse_simulate_args(
        [
            "--mode", "local_model",
            "--source-trace", "trace.jsonl",
            "--provider", "openai",
            "--api-base", "http://localhost:8000/v1",
            "--api-key", "dummy",
            "--model", "local-model",
            "--gpu-tracking", "on",
            "--vllm-pid", "9999",
            "--vllm-startup-log", str(startup_log),
            "--metrics-url", "http://localhost:8000/metrics",
            "--gpu-sample-hz", "5.0",
        ]
    )

    _run_simulate(args)

    assert seen_kwargs.get("gpu_baseline") == fake_baseline
    assert seen_kwargs.get("vllm_pid") == 9999
    assert seen_kwargs.get("gpu_sample_hz") == pytest.approx(5.0)
