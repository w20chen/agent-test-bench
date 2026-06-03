from __future__ import annotations

import json
from pathlib import Path
from agents.base import AgentBase, TraceAction
from harness.trace_logger import TraceLogger


def test_trace_logger_writes_action_entries(tmp_path: Path) -> None:
    run_id = "demo_run"
    logger = TraceLogger(tmp_path, run_id)
    action = TraceAction(
        action_type="llm_call",
        action_id="llm_0",
        agent_id="agent-0001",
        program_id="agent-0001",
        iteration=0,
        ts_start=1.0,
        ts_end=2.0,
        data={"prompt_tokens": 10, "completion_tokens": 3, "llm_latency_ms": 11.0},
    )
    logger.log_trace_action("agent-0001", action)
    logger.log_summary("agent-0001", {"task_id": "t-1", "success": True})
    logger.close()

    lines = (tmp_path / f"{run_id}.jsonl").read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["type"] == "action"
    assert parsed[0]["action_type"] == "llm_call"
    assert parsed[0]["agent_id"] == "agent-0001"
    assert parsed[1]["type"] == "summary"
    assert parsed[1]["success"] is True


def test_trace_logger_appends_on_resume(tmp_path: Path) -> None:
    run_id = "resume"
    logger1 = TraceLogger(tmp_path, run_id)
    logger1.log_event("agent-1", "LLM", "llm_call_start", {}, iteration=0, ts=1.0)
    logger1.close()

    logger2 = TraceLogger(tmp_path, run_id)
    logger2.log_event("agent-1", "LLM", "llm_call_end", {}, iteration=0, ts=2.0)
    logger2.close()

    lines = (tmp_path / f"{run_id}.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    e0 = json.loads(lines[0])
    e1 = json.loads(lines[1])
    assert e0["type"] == "event"
    assert e0["event"] == "llm_call_start"
    assert e1["event"] == "llm_call_end"


def test_trace_logger_log_event_writes_correct_record(tmp_path: Path) -> None:
    logger = TraceLogger(tmp_path, "evt_run")
    logger.log_event("agent-42", "LLM", "llm_call_start", {"prompt_tokens": 10},
                     iteration=0, ts=1.0)
    logger.close()

    lines = (tmp_path / "evt_run.jsonl").read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[0])
    assert entry["type"] == "event"
    assert entry["event"] == "llm_call_start"
    assert entry["category"] == "LLM"
    assert entry["agent_id"] == "agent-42"
    assert entry["iteration"] == 0
    assert entry["ts"] == 1.0
    assert entry["data"]["prompt_tokens"] == 10


def _make_action(iteration: int = 0, action_type: str = "tool_exec",
                 tool_name: str = "bash") -> TraceAction:
    return TraceAction(
        action_type=action_type,
        action_id=f"{action_type}_{iteration}_{tool_name}",
        agent_id="agent-1",
        program_id="agent-1",
        iteration=iteration,
        ts_start=1.0,
        ts_end=2.0,
        data={
            "tool_name": tool_name,
            "duration_ms": 300.0,
            "timeout": False,
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "llm_latency_ms": 100.0,
        },
    )


def _make_concrete_agent(api_base: str = "http://localhost") -> AgentBase:
    class _ConcreteAgent(AgentBase):
        async def run(self, task):  # type: ignore[override]
            return False
    return _ConcreteAgent(agent_id="agent-1", api_base=api_base, model="test-model")


def test_emit_action_appends_and_merges_metadata(tmp_path: Path) -> None:
    logger = TraceLogger(tmp_path, "action_run")
    agent = _make_concrete_agent()
    agent._trace_logger = logger
    agent.run_metadata = {"model": "qwen-plus", "prepare_ms": 500.0}

    action = _make_action(iteration=0)
    agent._emit_action(action)
    logger.close()

    assert len(agent.actions) == 1
    assert agent.actions[0].iteration == 0
    assert agent.actions[0].data["model"] == "qwen-plus"

    lines = (tmp_path / "action_run.jsonl").read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[0])
    assert entry["type"] == "action"
    assert entry["data"]["model"] == "qwen-plus"


def test_summary_tool_stats() -> None:
    agent = _make_concrete_agent()

    a1 = TraceAction(action_type="tool_exec", action_id="t0_bash",
                     iteration=0, data={"tool_name": "bash", "duration_ms": 200.0})
    a2 = TraceAction(action_type="tool_exec", action_id="t1_bash",
                     iteration=1, data={"tool_name": "bash", "duration_ms": 100.0, "timeout": True})
    a3 = TraceAction(action_type="tool_exec", action_id="t2_submit",
                     iteration=2, data={"tool_name": "submit", "duration_ms": 50.0})

    for a in (a1, a2, a3):
        agent.actions.append(a)

    s = agent.summary()
    assert s["tool_ms_by_name"]["bash"] == 300.0
    assert s["tool_ms_by_name"]["submit"] == 50.0
    assert s["tool_timeouts"]["bash"] == 1
    assert "submit" not in s["tool_timeouts"]
