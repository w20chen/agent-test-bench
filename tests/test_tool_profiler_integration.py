"""Integration tests for the tool profiler end-to-end.

These tests run actual commands through the profiler and verify
the output JSONL is well-formed and contains expected fields.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

from prototype.tool_profiler.runner import run_tool


def _path_to_python() -> str:
    """Return path to the current Python interpreter."""
    return sys.executable


def _path_to_workload(name: str) -> str:
    """Return absolute path to a workload script."""
    return str(Path(__file__).parent.parent / "workloads" / name)


class TestRunTool:
    """End-to-end tests using the runner directly."""

    def test_short_tool(self, tmp_path: Path) -> None:
        """Short tool (< warmup) should produce final profile with short_tool=true."""
        output = str(tmp_path / "profiles.jsonl")
        exit_code = run_tool(
            command=[_path_to_python(), "-c", "import time; time.sleep(0.1)"],
            warmup_seconds=2.0,
            sample_interval=0.2,
            output_path=output,
            verbose=False,
            save_samples=False,
        )
        assert exit_code == 0

        with open(output) as f:
            lines = f.readlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["final_profile"]["short_tool"] is True
        assert rec["early_profile"]["available"] is False

    def test_long_tool_early_profile(self, tmp_path: Path) -> None:
        """Tool running > warmup should emit early profile."""
        output = str(tmp_path / "profiles.jsonl")

        # Use a workload that runs for at least 3 seconds
        workload = _path_to_workload("cpu_serial.py")
        exit_code = run_tool(
            command=[_path_to_python(), workload, "--seconds", "3"],
            warmup_seconds=2.0,
            sample_interval=0.2,
            output_path=output,
            verbose=False,
            save_samples=False,
        )
        assert exit_code == 0

        with open(output) as f:
            lines = f.readlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["final_profile"]["short_tool"] is False
        assert rec["early_profile"]["available"] is True
        assert "elapsed_s" in rec["early_profile"]
        assert "avg_effective_cores" in rec["early_profile"]

    def test_failed_command(self, tmp_path: Path) -> None:
        """Non-existent command should return non-zero exit code."""
        output = str(tmp_path / "profiles.jsonl")
        exit_code = run_tool(
            command=["nonexistent_command_xyz"],
            warmup_seconds=2.0,
            sample_interval=0.2,
            output_path=output,
            verbose=False,
            save_samples=False,
        )
        assert exit_code != 0

    def test_save_samples(self, tmp_path: Path) -> None:
        """With --save-samples, raw samples should appear in output."""
        output = str(tmp_path / "profiles.jsonl")
        exit_code = run_tool(
            command=[_path_to_python(), "-c", "import time; time.sleep(0.5)"],
            warmup_seconds=2.0,
            sample_interval=0.1,
            output_path=output,
            verbose=False,
            save_samples=True,
        )
        assert exit_code == 0

        with open(output) as f:
            rec = json.loads(f.readline())
        assert "samples" in rec
        assert isinstance(rec["samples"], list)
        assert len(rec["samples"]) > 0
        # Each sample should have the required fields
        s0 = rec["samples"][0]
        assert "timestamp_s" in s0
        assert "elapsed_s" in s0
        assert "cpu_total_time_s" in s0

    def test_output_directory_created(self, tmp_path: Path) -> None:
        """Output path in non-existent directory should be created."""
        output = str(tmp_path / "nonexistent" / "profiles.jsonl")
        exit_code = run_tool(
            command=[_path_to_python(), "-c", "pass"],
            warmup_seconds=2.0,
            sample_interval=0.2,
            output_path=output,
            verbose=False,
            save_samples=False,
        )
        assert exit_code == 0
        assert os.path.exists(output)

    def test_cpu_serial_behavior(self, tmp_path: Path) -> None:
        """CPU serial workload should be classified as cpu_serial."""
        output = str(tmp_path / "profiles.jsonl")
        workload = _path_to_workload("cpu_serial.py")
        exit_code = run_tool(
            command=[_path_to_python(), workload, "--seconds", "4"],
            warmup_seconds=2.0,
            sample_interval=0.2,
            output_path=output,
            verbose=False,
            save_samples=False,
        )
        assert exit_code == 0

        with open(output) as f:
            rec = json.loads(f.readline())
        # Single-threaded CPU should have ~1 effective core
        cores = rec["final_profile"]["avg_effective_cores"]
        # Allow some tolerance for system noise
        assert 0.3 < cores < 2.5, f"Expected ~1 core for serial CPU, got {cores:.2f}"

    def test_cpu_parallel_multi_process(self, tmp_path: Path) -> None:
        """Multi-process CPU workload should show > 1 effective cores."""
        output = str(tmp_path / "profiles.jsonl")
        workload = _path_to_workload("cpu_parallel.py")
        exit_code = run_tool(
            command=[_path_to_python(), workload, "--workers", "4", "--seconds", "4"],
            warmup_seconds=2.0,
            sample_interval=0.2,
            output_path=output,
            verbose=False,
            save_samples=False,
        )
        assert exit_code == 0

        with open(output) as f:
            rec = json.loads(f.readline())
        cores = rec["final_profile"]["avg_effective_cores"]
        assert cores > 1.5, f"Expected > 1.5 cores for parallel CPU, got {cores:.2f}"
        # Peak process count should include workers
        assert rec["final_profile"]["peak_process_count"] >= 1

    def test_process_tree_workload(self, tmp_path: Path) -> None:
        """Process tree workload should detect multiple processes."""
        output = str(tmp_path / "profiles.jsonl")
        workload = _path_to_workload("process_tree.py")
        exit_code = run_tool(
            command=[_path_to_python(), workload, "--children", "3", "--seconds", "3"],
            warmup_seconds=2.0,
            sample_interval=0.2,
            output_path=output,
            verbose=False,
            save_samples=False,
        )
        assert exit_code == 0

        with open(output) as f:
            rec = json.loads(f.readline())
        assert rec["final_profile"]["peak_process_count"] > 1

    def test_early_final_comparison_fields(self, tmp_path: Path) -> None:
        """Records with early profile should have comparison fields."""
        output = str(tmp_path / "profiles.jsonl")
        workload = _path_to_workload("cpu_parallel.py")
        exit_code = run_tool(
            command=[_path_to_python(), workload, "--workers", "2", "--seconds", "4"],
            warmup_seconds=2.0,
            sample_interval=0.2,
            output_path=output,
            verbose=False,
            save_samples=False,
        )
        assert exit_code == 0

        with open(output) as f:
            rec = json.loads(f.readline())

        assert "early_final_comparison" in rec
        comp = rec["early_final_comparison"]
        assert "effective_cores_relative_error" in comp
        assert "behavior_changed" in comp
        assert "stability_changed" in comp
        assert isinstance(comp["effective_cores_relative_error"], (int, float))

    def test_jsonl_schema(self, tmp_path: Path) -> None:
        """Verify JSONL output has all required top-level fields."""
        output = str(tmp_path / "profiles.jsonl")
        exit_code = run_tool(
            command=[_path_to_python(), "-c", "import time; time.sleep(0.3)"],
            warmup_seconds=2.0,
            sample_interval=0.1,
            output_path=output,
            verbose=False,
            save_samples=False,
        )
        assert exit_code == 0

        with open(output) as f:
            rec = json.loads(f.readline())

        required_fields = [
            "schema_version",
            "invocation_id",
            "command",
            "command_string",
            "cwd",
            "root_pid",
            "start_time",
            "warmup_seconds",
            "sample_interval",
            "exit_code",
            "early_profile",
            "final_profile",
        ]
        for field in required_fields:
            assert field in rec, f"Missing required field: {field}"

        final_required = [
            "total_wall_time_s",
            "short_tool",
            "avg_effective_cores",
            "p50_effective_cores",
            "p90_effective_cores",
            "peak_effective_cores",
            "peak_process_count",
            "peak_thread_count",
            "rss_peak_bytes",
            "total_read_bytes",
            "total_write_bytes",
            "parallelism_cv",
            "profile_stability",
            "preliminary_behavior",
        ]
        for field in final_required:
            assert field in rec["final_profile"], f"Missing final_profile.{field}"
