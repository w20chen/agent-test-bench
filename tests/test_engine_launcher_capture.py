from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from serving.engine_launcher import (
    VLLMServerConfig,
    _exec_with_stderr_tee,
    build_vllm_command,
)


def test_default_command_unchanged_when_capture_disabled() -> None:
    """build_vllm_command output is not affected by capture_startup_log_path=None."""
    config_with = VLLMServerConfig(
        model_path="/model",
        capture_startup_log_path="/tmp/x.log",
    )
    config_without = VLLMServerConfig(model_path="/model")
    assert build_vllm_command(config_with) == build_vllm_command(config_without)


def test_config_dataclass_has_capture_field() -> None:
    cfg = VLLMServerConfig(model_path="x", capture_startup_log_path="/tmp/x.log")
    assert cfg.capture_startup_log_path == "/tmp/x.log"


def test_print_only_includes_capture_field(tmp_path: Path) -> None:
    """--print-only JSON output reflects capture_startup_log_path in the config block."""
    log_path = str(tmp_path / "out.log")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "serving.engine_launcher",
            "--model-path",
            "/fake",
            "--capture-startup-log",
            log_path,
            "--print-only",
        ],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).parent.parent / "src")},
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    # The config dict must carry the field
    assert data["config"]["capture_startup_log_path"] == log_path


def test_exec_with_stderr_tee_writes_file_and_streams(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    """stderr is teed: written to file AND forwarded to parent stderr."""
    log = tmp_path / "out.log"
    rc = _exec_with_stderr_tee(
        [sys.executable, "-c", "import sys; sys.stderr.write('hello vllm\\n')"],
        str(log),
    )
    assert rc == 0
    assert log.read_text(encoding="utf-8") == "hello vllm\n"
    captured = capfd.readouterr()
    assert "hello vllm" in captured.err


def test_exec_with_stderr_tee_creates_parent_dir(tmp_path: Path) -> None:
    """_exec_with_stderr_tee creates missing parent directories."""
    log = tmp_path / "deep" / "dir" / "x.log"
    rc = _exec_with_stderr_tee(
        [sys.executable, "-c", "import sys; sys.stderr.write('ok\\n')"],
        str(log),
    )
    assert rc == 0
    assert log.exists()
    assert "ok" in log.read_text(encoding="utf-8")
