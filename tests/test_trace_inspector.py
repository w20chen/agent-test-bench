"""Unit tests for canonical trace inspection helpers."""

from __future__ import annotations

import json
from pathlib import Path

from trace_collect.trace_inspector import (
    TraceData,
    cmd_overview,
    cmd_step,
    cmd_timeline,
)


def test_cmd_overview_counts_distinct_iterations(tmp_path: Path, capsys) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "trace_metadata",
                        "scaffold": "openclaw",
                        "trace_format_version": 5,
                        "model": "test-model",
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": "llm_0",
                        "agent_id": "agent-1",
                        "iteration": 0,
                        "ts_start": 1.0,
                        "ts_end": 2.0,
                        "data": {"prompt_tokens": 10, "completion_tokens": 5},
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "tool_exec",
                        "action_id": "tool_0",
                        "agent_id": "agent-1",
                        "iteration": 0,
                        "ts_start": 2.0,
                        "ts_end": 2.1,
                        "data": {
                            "tool_name": "bash",
                            "duration_ms": 100.0,
                            "success": True,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": "llm_1",
                        "agent_id": "agent-1",
                        "iteration": 1,
                        "ts_start": 3.0,
                        "ts_end": 4.0,
                        "data": {"prompt_tokens": 10, "completion_tokens": 5},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cmd_overview(TraceData.load(trace_path), as_json=True)
    output = json.loads(capsys.readouterr().out)
    assert output["n_iterations"] == 2


def test_cmd_timeline_renders_cloud_model_simulation_truthfully(
    tmp_path: Path,
    capsys,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "trace_metadata",
                        "scaffold": "openclaw",
                        "trace_format_version": 5,
                        "mode": "simulate",
                        "simulate_mode": "cloud_model",
                        "source_model": "claude-haiku",
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": "llm_0",
                        "agent_id": "agent-1",
                        "iteration": 0,
                        "ts_start": 1.0,
                        "ts_end": 2.0,
                        "data": {"prompt_tokens": 10, "completion_tokens": 5},
                    }
                ),
                json.dumps(
                    {
                        "type": "summary",
                        "agent_id": "agent-1",
                        "success": True,
                        "n_iterations": 1,
                        "elapsed_s": 1.0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cmd_timeline(TraceData.load(trace_path))
    output = capsys.readouterr().out
    assert "Simulate: claude-haiku → cloud replay" in output
    assert "Model: cloud_model" not in output
    assert "✓ success" in output


def test_cmd_overview_prefers_openrouter_latency(tmp_path: Path, capsys) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "trace_metadata",
                        "scaffold": "openclaw",
                        "trace_format_version": 5,
                        "model": "test-model",
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": "llm_0",
                        "agent_id": "agent-1",
                        "iteration": 0,
                        "ts_start": 1.0,
                        "ts_end": 901.0,
                        "data": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "llm_latency_ms": 6500.0,
                            "llm_call_time_ms": 6500.0,
                            "llm_wall_latency_ms": 900000.0,
                            "llm_timing_source": "openrouter_generation_time_ms",
                            "openrouter_latency_ms": 7000.0,
                            "openrouter_generation_time_ms": 6500.0,
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cmd_overview(TraceData.load(trace_path), as_json=True)
    output = json.loads(capsys.readouterr().out)
    assert output["total_llm_ms"] == 6500.0
    assert output["total_llm_call_time_ms"] == 6500.0
    assert output["llm_timing_source"] == "openrouter_generation_time_ms"
    assert output["total_llm_wall_ms"] == 900000.0


def test_cmd_step_prints_openrouter_latency_fields(tmp_path: Path, capsys) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "trace_metadata",
                        "scaffold": "openclaw",
                        "trace_format_version": 5,
                        "model": "test-model",
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": "llm_0",
                        "agent_id": "agent-1",
                        "iteration": 0,
                        "ts_start": 1.0,
                        "ts_end": 901.0,
                        "data": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "llm_latency_ms": 6500.0,
                            "llm_call_time_ms": 6500.0,
                            "llm_wall_latency_ms": 900000.0,
                            "openrouter_generation_id": "gen-123",
                            "openrouter_request_id": "req-123",
                            "openrouter_provider_name": "Z.AI",
                            "openrouter_upstream_id": "up-123",
                            "openrouter_latency_ms": 7000.0,
                            "openrouter_generation_time_ms": 6500.0,
                            "openrouter_provider_latency_ms": 6800.0,
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cmd_step(TraceData.load(trace_path), 0)
    output = capsys.readouterr().out
    assert "llm_latency_ms  : 6500.0" in output
    assert "llm_wall_latency_ms: 900000.0" in output
    assert "openrouter_generation_id: gen-123" in output
    assert "openrouter_request_id: req-123" in output
