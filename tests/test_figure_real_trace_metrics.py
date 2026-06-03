from __future__ import annotations

import json
from pathlib import Path
import sys

from trace_collect.trace_inspector import CURRENT_TRACE_FORMAT_VERSION


REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_SCRIPTS = REPO_ROOT / "scripts" / "figures"
if str(FIGURE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(FIGURE_SCRIPTS))

from _real_trace_metrics import load_real_trace_metrics, parse_cohort  # noqa: E402


def test_real_metrics_drop_gap_created_by_shorter_llm_latency(tmp_path: Path) -> None:
    trace_path = _write_trace(
        tmp_path,
        [
            {
                "type": "action",
                "action_type": "llm_call",
                "action_id": "llm_0",
                "agent_id": "agent-a",
                "program_id": "task-a",
                "instance_id": "task-a",
                "iteration": 0,
                "ts_start": 1000.0,
                "ts_end": 1004.0,
                "data": {
                    "llm_call_time_ms": 1000.0,
                    "llm_timing_source": "openrouter_generation_time_ms",
                    "llm_latency_ms": 1000.0,
                },
            },
            {
                "type": "action",
                "action_type": "tool_exec",
                "action_id": "tool_0_exec",
                "agent_id": "agent-a",
                "program_id": "task-a",
                "instance_id": "task-a",
                "iteration": 0,
                "ts_start": 1005.0,
                "ts_end": 1006.0,
                "data": {"tool_name": "exec"},
            },
            {
                "type": "event",
                "agent_id": "agent-a",
                "program_id": "task-a",
                "instance_id": "task-a",
                "event": "task_complete",
                "category": "MCP",
                "ts": 1006.5,
                "iteration": 0,
                "data": {},
            },
        ],
    )

    metrics = load_real_trace_metrics(parse_cohort(f"demo={trace_path.parents[2]}"))

    assert len(metrics) == 1
    metric = metrics[0]
    assert metric.total_time_s == 2.5
    assert metric.llm_time_s == 1.0
    assert metric.tool_time_s == 1.0
    assert metric.tool_spans[0].midpoint_frac == 0.6


def _write_trace(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    trace_path = tmp_path / "demo" / "task-a" / "attempt_1" / "trace.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "type": "trace_metadata",
        "trace_format_version": CURRENT_TRACE_FORMAT_VERSION,
        "scaffold": "openclaw",
        "model": "z-ai/glm-5.1",
        "instance_id": "task-a",
    }
    summary = {
        "type": "summary",
        "agent_id": "agent-a",
        "program_id": "task-a",
        "task_id": "task-a",
        "instance_id": "task-a",
        "n_iterations": 1,
        "total_llm_ms": 1000.0,
        "total_tool_ms": 1000.0,
        "total_tokens": 0,
        "tool_ms_by_name": {"exec": 1000.0},
        "tool_timeouts": {},
        "success": True,
        "elapsed_s": 6.5,
    }
    with trace_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata) + "\n")
        for record in records:
            handle.write(json.dumps(record) + "\n")
        handle.write(json.dumps(summary) + "\n")
    return trace_path
