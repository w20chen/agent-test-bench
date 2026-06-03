"""Unit tests for the Claude Code → canonical trace converter.

Phase US-002 of the claude-code-gantt-import plan. Verifies that:
- The metadata header is well-formed (canonical trace + scaffold + run_config backfill)
- Assistant records become llm_call actions with all backfill fields
- Thinking blocks are preserved under data.thinking
- tool_result pairs with the matching tool_use via tool_use_id
- toolUseResult sidecar fields flow into data.subagent_meta + subagent_tokens
- Unpaired tool_results still emit a tool_exec action with a descriptive note
- Discarded record types produce zero emitted trace records
- Sidechain files fold into separate agent_id lanes
- The converted output loads via TraceData and renders through the Gantt payload
- Timestamp conversion handles ISO 8601 with Z suffix
- Per-lane iteration counters are independent
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from trace_collect.claude_code_import import (
    SCAFFOLD_LABEL,
    V5_FORMAT_VERSION,
    import_claude_code_session,
    looks_like_claude_code_session,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "claude_code_minimal.jsonl"
SUBAGENT_DIR = REPO_ROOT / "tests" / "fixtures" / "claude_code_minimal" / "subagents"


# ─── Shared fixture: run the converter once per test session ────────────


@pytest.fixture(scope="module")
def converted_trace(tmp_path_factory) -> Path:
    """Convert the minimal CC fixture once and share the output path."""
    out_dir = tmp_path_factory.mktemp("cc-import-default")
    return import_claude_code_session(
        session_path=FIXTURE,
        output_dir=out_dir,
        include_sidechains=True,
    )


@pytest.fixture(scope="module")
def converted_trace_no_sidechains(tmp_path_factory) -> Path:
    """Same fixture but with include_sidechains=False."""
    out_dir = tmp_path_factory.mktemp("cc-import-main-only")
    return import_claude_code_session(
        session_path=FIXTURE,
        output_dir=out_dir,
        include_sidechains=False,
    )


def _load_records(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def test_looks_like_claude_code_session_accepts_fixture() -> None:
    assert looks_like_claude_code_session(FIXTURE) is True


def test_looks_like_claude_code_session_accepts_queue_operation_first_record(
    tmp_path: Path,
) -> None:
    session_path = tmp_path / "queue-first.jsonl"
    session_path.write_text(
        json.dumps(
            {
                "type": "queue-operation",
                "operation": "enqueue",
                "timestamp": "2026-04-08T10:00:00.000Z",
                "sessionId": "demo-session",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert looks_like_claude_code_session(session_path) is True


def test_looks_like_claude_code_session_rejects_canonical_trace(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "type": "trace_metadata",
                "trace_format_version": V5_FORMAT_VERSION,
                "scaffold": "openclaw",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert looks_like_claude_code_session(trace_path) is False


# ─── Timestamp conversion helpers ───────────────────────────────────────

# ─── Metadata header ────────────────────────────────────────────────────


def test_metadata_has_required_fields(converted_trace: Path) -> None:
    records = _load_records(converted_trace)
    assert records[0]["type"] == "trace_metadata"
    meta = records[0]
    assert meta["trace_format_version"] == V5_FORMAT_VERSION
    assert meta["scaffold"] == SCAFFOLD_LABEL
    assert meta["mode"] == "import"
    assert meta["instance_id"] == "claude_code_minimal"
    assert meta["model"] == "claude-sonnet-4-6"
    assert "source_trace" in meta
# ─── Assistant → llm_call conversion ───────────────────────────────────


def test_assistant_becomes_llm_call_action(converted_trace: Path) -> None:
    records = _load_records(converted_trace)
    llm_main = [
        r
        for r in records
        if r.get("type") == "action"
        and r.get("action_type") == "llm_call"
        and r.get("agent_id") == "claude_code_minimal"
    ]
    assert len(llm_main) == 2  # two main-lane assistant records in fixture

    # Check monotonic iteration numbering per lane
    iterations = [r["iteration"] for r in llm_main]
    assert iterations == [0, 1]

    # Check backfill fields on the first llm_call
    first = llm_main[0]
    data = first["data"]
    assert data["prompt_tokens"] == 42
    assert data["completion_tokens"] == 28
    assert data["cache_creation_tokens"] == 1500
    assert data["cache_read_tokens"] == 0
    assert data["cache_ephemeral_5m_tokens"] == 1500
    assert data["cache_ephemeral_1h_tokens"] == 0
    assert data["message_id"] == "msg_test_a0001"
    assert data["service_tier"] == "standard"
    assert data["llm_latency_ms"] > 0
# ─── user + tool_result → tool_exec conversion ──────────────────────────


def test_tool_result_pairs_with_tool_use(converted_trace: Path) -> None:
    records = _load_records(converted_trace)
    tool_actions = [
        r
        for r in records
        if r.get("type") == "action"
        and r.get("action_type") == "tool_exec"
        and r.get("agent_id") == "claude_code_minimal"
    ]
    assert len(tool_actions) == 1, (
        f"expected exactly 1 tool_exec in main lane; got {len(tool_actions)}"
    )
    ta = tool_actions[0]
    data = ta["data"]
    assert data["tool_name"] == "Read"
    # tool_args is a JSON-string-encoded input dict
    args = json.loads(data["tool_args"])
    assert args["file_path"] == "/etc/hosts"
    assert "127.0.0.1 localhost" in data["tool_result"]
    # Duration came from toolUseResult.totalDurationMs (450.0)
    assert data["duration_ms"] == 450.0
    assert data["success"] is True
    # Iteration inherited from the paired tool_use (not the user record position)
    assert ta["iteration"] == 0


def test_tooluseresult_backfill_fields(converted_trace: Path) -> None:
    """The fixture's user record has toolUseResult with totalTokens + usage."""
    records = _load_records(converted_trace)
    tool_actions = [
        r
        for r in records
        if r.get("type") == "action"
        and r.get("action_type") == "tool_exec"
        and r.get("agent_id") == "claude_code_minimal"
    ]
    data = tool_actions[0]["data"]
    # subagent_tokens backfill from toolUseResult.totalTokens + usage
    assert "subagent_tokens" in data
    assert data["subagent_tokens"]["total"] == 12
    assert data["subagent_tokens"]["usage"]["input_tokens"] == 6
    # subagent_tool_use_count from toolUseResult.totalToolUseCount
    assert data.get("subagent_tool_use_count") == 1


def test_tool_exec_success_inferred_from_is_error(converted_trace: Path) -> None:
    records = _load_records(converted_trace)
    tool_actions = [
        r
        for r in records
        if r.get("type") == "action"
        and r.get("action_type") == "tool_exec"
        and r.get("agent_id") == "claude_code_minimal"
    ]
    # Fixture has is_error: false → success: true
    assert tool_actions[0]["data"]["success"] is True
# ─── Sidechain folding ──────────────────────────────────────────────────


def test_sidechain_becomes_separate_lane(converted_trace: Path) -> None:
    records = _load_records(converted_trace)
    agents = sorted(
        {r.get("agent_id") for r in records if r.get("type") == "action"}
    )
    # Main lane + one subagent lane from the fixture
    assert len(agents) == 2, f"expected 2 agent lanes, got {agents}"
    assert "claude_code_minimal" in agents
    assert any(a.startswith("agent-test") for a in agents), (
        f"expected a sidechain lane starting with 'agent-test'; got {agents}"
    )


def test_no_sidechains_when_opted_out(
    converted_trace_no_sidechains: Path,
) -> None:
    records = _load_records(converted_trace_no_sidechains)
    agents = sorted(
        {r.get("agent_id") for r in records if r.get("type") == "action"}
    )
    assert len(agents) == 1, (
        f"expected only main lane with include_sidechains=False; got {agents}"
    )
    assert agents[0] == "claude_code_minimal"


def test_iteration_counter_per_agent_lane(converted_trace: Path) -> None:
    """Main lane and subagent lane each have their own iteration counter."""
    records = _load_records(converted_trace)
    main_iters = sorted(
        r["iteration"]
        for r in records
        if r.get("type") == "action"
        and r.get("action_type") == "llm_call"
        and r.get("agent_id") == "claude_code_minimal"
    )
    sub_iters = sorted(
        r["iteration"]
        for r in records
        if r.get("type") == "action"
        and r.get("action_type") == "llm_call"
        and r.get("agent_id", "").startswith("agent-test")
    )
    # Both start from 0 — counter is per-lane, not global
    assert main_iters == [0, 1]
    assert sub_iters == [0, 1]


def _make_inline_sidechain_session(tmp_path: Path) -> Path:
    """Build a synthetic main-lane session with an embedded inline sidechain."""
    session = tmp_path / "cc-inline-sidechain.jsonl"
    _write_synthetic_session(
        session,
        [
            {
                "type": "user",
                "sessionId": "cc-inline-sidechain",
                "uuid": "u0",
                "timestamp": "2026-04-08T10:00:00.000Z",
                "cwd": "/tmp",
                "gitBranch": "main",
                "version": "2.1.96",
                "isSidechain": False,
                "message": {"role": "user", "content": "Use Task to inspect the repo."},
            },
            {
                "type": "assistant",
                "sessionId": "cc-inline-sidechain",
                "uuid": "a0",
                "parentUuid": "u0",
                "timestamp": "2026-04-08T10:00:01.000Z",
                "cwd": "/tmp",
                "gitBranch": "main",
                "version": "2.1.96",
                "isSidechain": False,
                "message": {
                    "id": "msg_main_task",
                    "model": "claude-sonnet-4-6",
                    "role": "assistant",
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 5,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation": {},
                    },
                    "content": [
                        {"type": "text", "text": "I'll delegate this search."},
                        {
                            "type": "tool_use",
                            "id": "toolu_task_0001",
                            "name": "Task",
                            "input": {
                                "description": "Search the repo",
                                "prompt": "Find parsing code paths.",
                            },
                        },
                    ],
                },
            },
            {
                "type": "user",
                "sessionId": "cc-inline-sidechain",
                "uuid": "su0",
                "timestamp": "2026-04-08T10:00:01.100Z",
                "cwd": "/tmp",
                "gitBranch": "main",
                "version": "2.1.96",
                "isSidechain": True,
                "message": {"role": "user", "content": "Find parsing code paths."},
            },
            {
                "type": "assistant",
                "sessionId": "cc-inline-sidechain",
                "uuid": "sa0",
                "parentUuid": "su0",
                "timestamp": "2026-04-08T10:00:01.500Z",
                "cwd": "/tmp",
                "gitBranch": "main",
                "version": "2.1.96",
                "isSidechain": True,
                "message": {
                    "id": "msg_side_0",
                    "model": "claude-3-5-haiku-20241022",
                    "role": "assistant",
                    "usage": {
                        "input_tokens": 8,
                        "output_tokens": 4,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation": {},
                    },
                    "content": [{"type": "text", "text": "I'll inspect the repo."}],
                },
            },
            {
                "type": "assistant",
                "sessionId": "cc-inline-sidechain",
                "uuid": "sa1",
                "parentUuid": "sa0",
                "timestamp": "2026-04-08T10:00:02.000Z",
                "cwd": "/tmp",
                "gitBranch": "main",
                "version": "2.1.96",
                "isSidechain": True,
                "message": {
                    "id": "msg_side_1",
                    "model": "claude-3-5-haiku-20241022",
                    "role": "assistant",
                    "usage": {
                        "input_tokens": 9,
                        "output_tokens": 6,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation": {},
                    },
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_side_bash_0001",
                            "name": "Bash",
                            "input": {"command": "rg -n parser src"},
                        }
                    ],
                },
            },
            {
                "type": "user",
                "sessionId": "cc-inline-sidechain",
                "uuid": "su1",
                "parentUuid": "sa1",
                "timestamp": "2026-04-08T10:00:02.200Z",
                "cwd": "/tmp",
                "gitBranch": "main",
                "version": "2.1.96",
                "isSidechain": True,
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_side_bash_0001",
                            "content": [{"type": "text", "text": "src/parser.py:1:def parse()"}],
                            "is_error": False,
                        }
                    ],
                },
                "toolUseResult": {
                    "stdout": "src/parser.py:1:def parse()",
                    "stderr": "",
                    "interrupted": False,
                },
            },
            {
                "type": "assistant",
                "sessionId": "cc-inline-sidechain",
                "uuid": "sa2",
                "parentUuid": "su1",
                "timestamp": "2026-04-08T10:00:02.600Z",
                "cwd": "/tmp",
                "gitBranch": "main",
                "version": "2.1.96",
                "isSidechain": True,
                "message": {
                    "id": "msg_side_2",
                    "model": "claude-3-5-haiku-20241022",
                    "role": "assistant",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 7,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation": {},
                    },
                    "content": [{"type": "text", "text": "Parser code is in src/parser.py."}],
                },
            },
            {
                "type": "user",
                "sessionId": "cc-inline-sidechain",
                "uuid": "u1",
                "parentUuid": "sa2",
                "timestamp": "2026-04-08T10:00:03.500Z",
                "cwd": "/tmp",
                "gitBranch": "main",
                "version": "2.1.96",
                "isSidechain": False,
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_task_0001",
                            "content": [{"type": "text", "text": "Parser code is in src/parser.py."}],
                            "is_error": False,
                        }
                    ],
                },
                "toolUseResult": {
                    "content": [
                        {"type": "text", "text": "Parser code is in src/parser.py."}
                    ],
                    "totalDurationMs": 2400.0,
                    "totalTokens": 17,
                    "usage": {"input_tokens": 9, "output_tokens": 8},
                    "totalToolUseCount": 1,
                },
            },
        ],
    )
    return session


def test_inline_sidechain_records_become_separate_lane(tmp_path: Path) -> None:
    session = _make_inline_sidechain_session(tmp_path)
    result = import_claude_code_session(
        session_path=session,
        output_dir=tmp_path / "out",
        include_sidechains=False,
    )
    records = _load_records(result)

    action_records = [r for r in records if r.get("type") == "action"]
    agent_ids = sorted({r["agent_id"] for r in action_records})
    assert "cc-inline-sidechain" in agent_ids
    inline_agents = [
        agent for agent in agent_ids if ":inline-sidechain-" in agent
    ]
    assert len(inline_agents) == 1, (
        "expected one synthetic inline sidechain lane; "
        f"got {agent_ids}"
    )

    main_llm = [
        r for r in action_records
        if r.get("action_type") == "llm_call"
        and r["agent_id"] == "cc-inline-sidechain"
    ]
    main_tools = [
        r for r in action_records
        if r.get("action_type") == "tool_exec"
        and r["agent_id"] == "cc-inline-sidechain"
    ]
    assert [r["iteration"] for r in main_llm] == [0]
    assert len(main_tools) == 1
    assert main_tools[0]["data"]["tool_name"] == "Task"
    assert main_tools[0]["data"]["duration_ms"] == 2400.0

    inline_llm = [
        r for r in action_records
        if r.get("action_type") == "llm_call"
        and r["agent_id"] == inline_agents[0]
    ]
    inline_tools = [
        r for r in action_records
        if r.get("action_type") == "tool_exec"
        and r["agent_id"] == inline_agents[0]
    ]
    assert [r["iteration"] for r in inline_llm] == [0, 1, 2]
    assert len(inline_tools) == 1
    assert inline_tools[0]["data"]["tool_name"] == "Bash"


def test_inline_sidechain_payload_renders_multiple_lanes(tmp_path: Path) -> None:
    session = _make_inline_sidechain_session(tmp_path)
    result = import_claude_code_session(
        session_path=session,
        output_dir=tmp_path / "out",
        include_sidechains=False,
    )

    from demo.gantt_viewer.backend.payload import build_gantt_payload_multi
    from trace_collect.trace_inspector import TraceData

    data = TraceData.load(result)
    payload = build_gantt_payload_multi([("inline-sidechain", data)])
    lanes = payload["traces"][0]["lanes"]
    assert len(lanes) == 2, f"expected 2 lanes after import, got {len(lanes)}"
    lane_ids = {lane["agent_id"] for lane in lanes}
    assert "cc-inline-sidechain" in lane_ids
    assert any(":inline-sidechain-" in lane_id for lane_id in lane_ids)
# ─── Summary record ─────────────────────────────────────────────────────


def test_summary_record_per_lane(converted_trace: Path) -> None:
    records = _load_records(converted_trace)
    summaries = [r for r in records if r.get("type") == "summary"]
    # One summary per agent lane (main + sidechain = 2)
    assert len(summaries) == 2

    main_summary = next(
        s for s in summaries if s["agent_id"] == "claude_code_minimal"
    )
    # 2 assistant records in main lane fixture
    assert main_summary["n_iterations"] == 2
    assert main_summary["n_tool_actions"] == 1
    assert main_summary["elapsed_s"] > 0
    assert main_summary["total_tokens"] > 0


# ─── Follow-ups from the Gate-C reviewer (M1-M5) ────────────────────────


def _write_synthetic_session(path: Path, records: list[dict]) -> None:
    """Helper: dump a list of records as JSONL to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def test_tooluseresult_string_variant_does_not_crash(tmp_path: Path) -> None:
    """M1 (reviewer): some CC records carry toolUseResult as a plain string.

    The converter must type-check defensively and fall back to an empty
    sidecar rather than crashing with ``AttributeError: 'str' object has
    no attribute 'get'``. Caught on a real 1.6MB session during the
    US-001 smoke run.
    """
    session = tmp_path / "cc-str-sidecar.jsonl"
    _write_synthetic_session(
        session,
        [
            {
                "type": "assistant",
                "sessionId": "cc-str-sidecar",
                "uuid": "a1",
                "timestamp": "2026-04-08T10:00:00.000Z",
                "cwd": "/tmp",
                "gitBranch": "main",
                "version": "2.1.92",
                "isSidechain": False,
                "message": {
                    "id": "msg_str_test",
                    "model": "claude-sonnet-4-6",
                    "role": "assistant",
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 3,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation": {},
                    },
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_str_0001",
                            "name": "Read",
                            "input": {"file_path": "/tmp/x"},
                        }
                    ],
                },
            },
            {
                "type": "user",
                "sessionId": "cc-str-sidecar",
                "uuid": "u1",
                "timestamp": "2026-04-08T10:00:00.500Z",
                "cwd": "/tmp",
                "gitBranch": "main",
                "version": "2.1.92",
                "isSidechain": False,
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_str_0001",
                            "content": [{"type": "text", "text": "file contents"}],
                            "is_error": False,
                        }
                    ],
                },
                # The quirk: toolUseResult is a plain string instead of a dict.
                "toolUseResult": "file contents",
            },
        ],
    )

    # Should not crash
    result = import_claude_code_session(
        session_path=session, output_dir=tmp_path / "out", include_sidechains=False
    )
    records = _load_records(result)
    tool_actions = [
        r for r in records if r.get("action_type") == "tool_exec"
    ]
    assert len(tool_actions) == 1
    # duration_ms was derived from timestamps, not from the string sidecar
    assert tool_actions[0]["data"]["duration_ms"] > 0
    # subagent_tokens should NOT be present because sidecar wasn't a dict
    assert "subagent_tokens" not in tool_actions[0]["data"]


def test_tool_result_with_is_error_true(tmp_path: Path) -> None:
    """M2 (reviewer): tool_result with is_error: true → data.success is False."""
    session = tmp_path / "cc-is-error.jsonl"
    _write_synthetic_session(
        session,
        [
            {
                "type": "assistant",
                "sessionId": "cc-is-error",
                "uuid": "a1",
                "timestamp": "2026-04-08T10:00:00.000Z",
                "cwd": "/tmp",
                "gitBranch": "main",
                "version": "2.1.92",
                "isSidechain": False,
                "message": {
                    "id": "msg_err_test",
                    "model": "claude-sonnet-4-6",
                    "role": "assistant",
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 3,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation": {},
                    },
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_err_0001",
                            "name": "Bash",
                            "input": {"command": "false"},
                        }
                    ],
                },
            },
            {
                "type": "user",
                "sessionId": "cc-is-error",
                "uuid": "u1",
                "timestamp": "2026-04-08T10:00:00.500Z",
                "cwd": "/tmp",
                "gitBranch": "main",
                "version": "2.1.92",
                "isSidechain": False,
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_err_0001",
                            "content": [{"type": "text", "text": "exit 1"}],
                            "is_error": True,
                        }
                    ],
                },
                "toolUseResult": {
                    "status": "error",
                    "totalDurationMs": 50.0,
                    "totalTokens": 2,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "totalToolUseCount": 1,
                },
            },
        ],
    )

    result = import_claude_code_session(
        session_path=session, output_dir=tmp_path / "out", include_sidechains=False
    )
    records = _load_records(result)
    tool_actions = [r for r in records if r.get("action_type") == "tool_exec"]
    assert len(tool_actions) == 1
    assert tool_actions[0]["data"]["success"] is False


def test_orphan_tool_use_is_drained_with_note(tmp_path: Path) -> None:
    """M3 (reviewer): orphan tool_use (no matching tool_result) must emit a stub.

    CLAUDE.md §5 "preserve all intermediate outputs" — silently dropping
    an orphaned tool_use loses the fact that the invocation ever
    happened, which is not acceptable.
    """
    session = tmp_path / "cc-orphan.jsonl"
    _write_synthetic_session(
        session,
        [
            {
                "type": "assistant",
                "sessionId": "cc-orphan",
                "uuid": "a1",
                "timestamp": "2026-04-08T10:00:00.000Z",
                "cwd": "/tmp",
                "gitBranch": "main",
                "version": "2.1.92",
                "isSidechain": False,
                "message": {
                    "id": "msg_orphan",
                    "model": "claude-sonnet-4-6",
                    "role": "assistant",
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 3,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation": {},
                    },
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_orphan_0001",
                            "name": "Read",
                            "input": {"file_path": "/never/arrived"},
                        }
                    ],
                },
            },
            # No user record with a matching tool_result — session cut off.
        ],
    )

    result = import_claude_code_session(
        session_path=session, output_dir=tmp_path / "out", include_sidechains=False
    )
    records = _load_records(result)
    tool_actions = [r for r in records if r.get("action_type") == "tool_exec"]
    assert len(tool_actions) == 1, (
        f"orphan tool_use should still produce a stub tool_exec; got {len(tool_actions)}"
    )
    orphan = tool_actions[0]
    assert orphan["data"]["tool_name"] == "Read"
    assert orphan["data"]["success"] is False
    assert orphan["data"]["duration_ms"] == 0.0
    assert "orphan tool_use" in orphan["data"].get("note", "")


# ─── FIX-4: per-tool rich backfill (Bash/Edit/Read) ─────────────────────


def _make_synthetic_tool_session(
    tmp_path: Path,
    tool_name: str,
    tool_input: dict[str, Any],
    tool_use_result: Any,
    is_error: bool | None = False,
    result_text: str = "result content",
) -> Path:
    """Helper: build a 2-record session (assistant→user) for a single tool."""
    session = tmp_path / f"cc-{tool_name.lower()}-test.jsonl"
    tu_id = f"toolu_{tool_name.lower()}_0001"
    user_msg: dict[str, Any] = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tu_id,
                "content": [{"type": "text", "text": result_text}],
            }
        ],
    }
    if is_error is not None:
        user_msg["content"][0]["is_error"] = is_error
    user_record: dict[str, Any] = {
        "type": "user",
        "sessionId": f"cc-{tool_name.lower()}-test",
        "uuid": "u1",
        "timestamp": "2026-04-08T10:00:00.500Z",
        "cwd": "/tmp",
        "gitBranch": "main",
        "version": "2.1.96",
        "isSidechain": False,
        "message": user_msg,
        "toolUseResult": tool_use_result,
    }
    _write_synthetic_session(
        session,
        [
            {
                "type": "assistant",
                "sessionId": f"cc-{tool_name.lower()}-test",
                "uuid": "a1",
                "timestamp": "2026-04-08T10:00:00.000Z",
                "cwd": "/tmp",
                "gitBranch": "main",
                "version": "2.1.96",
                "isSidechain": False,
                "message": {
                    "id": f"msg_{tool_name.lower()}_test",
                    "model": "claude-sonnet-4-6",
                    "role": "assistant",
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 3,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation": {},
                    },
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tu_id,
                            "name": tool_name,
                            "input": tool_input,
                        }
                    ],
                },
            },
            user_record,
        ],
    )
    return session


def _convert_and_get_tool(session: Path, tmp_path: Path) -> dict[str, Any]:
    """Helper: convert a single-tool session and return its tool_exec action."""
    out_dir = tmp_path / f"out-{session.stem}"
    result = import_claude_code_session(
        session_path=session, output_dir=out_dir, include_sidechains=False
    )
    records = _load_records(result)
    tool_actions = [r for r in records if r.get("action_type") == "tool_exec"]
    assert len(tool_actions) == 1, f"expected 1 tool_exec, got {len(tool_actions)}"
    return tool_actions[0]


def test_bash_interrupted_marks_success_false(tmp_path: Path) -> None:
    """FIX-4: Bash.interrupted=True is the only reliable failure signal
    for interrupted bash commands. tool_result.is_error is typically
    None/False even when the command was killed."""
    session = _make_synthetic_tool_session(
        tmp_path,
        tool_name="Bash",
        tool_input={"command": "sleep 10"},
        tool_use_result={
            "interrupted": True,
            "isImage": False,
            "noOutputExpected": False,
            "stderr": "",
            "stdout": "",
        },
        is_error=None,  # the real shape: is_error absent on interrupted bash
    )
    tool_action = _convert_and_get_tool(session, tmp_path)
    assert tool_action["data"]["success"] is False, (
        "FIX-4 broken: interrupted Bash should be marked success=False"
    )
def test_edit_structured_patch_preserved(tmp_path: Path) -> None:
    """FIX-4: Edit.structuredPatch (a parsed diff) is preserved as
    data.structured_patch — far more useful than re-derivable from newString."""
    patch = [
        {
            "oldStart": 10,
            "newStart": 10,
            "lines": ["-old line", "+new line"],
        }
    ]
    session = _make_synthetic_tool_session(
        tmp_path,
        tool_name="Edit",
        tool_input={"file_path": "/etc/hosts", "old_string": "old", "new_string": "new"},
        tool_use_result={
            "filePath": "/etc/hosts",
            "oldString": "old",
            "newString": "new",
            "originalFile": "old line",
            "replaceAll": False,
            "structuredPatch": patch,
            "userModified": False,
        },
    )
    tool_action = _convert_and_get_tool(session, tmp_path)
    assert tool_action["data"].get("structured_patch") == patch
# ─── FIX-5: requestId provenance backfill ───────────────────────────────


def _make_assistant_only_session(
    tmp_path: Path,
    extra_top_level: dict[str, Any] | None = None,
) -> Path:
    """Build a 1-record session with one assistant message."""
    session = tmp_path / "cc-request-id-test.jsonl"
    record: dict[str, Any] = {
        "type": "assistant",
        "sessionId": "cc-request-id-test",
        "uuid": "a1",
        "timestamp": "2026-04-08T10:00:00.000Z",
        "cwd": "/tmp",
        "gitBranch": "main",
        "version": "2.1.96",
        "isSidechain": False,
        "message": {
            "id": "msg_req_test",
            "model": "claude-sonnet-4-6",
            "role": "assistant",
            "usage": {
                "input_tokens": 5,
                "output_tokens": 3,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation": {},
            },
            "content": [{"type": "text", "text": "ack"}],
        },
    }
    if extra_top_level:
        record.update(extra_top_level)
    _write_synthetic_session(session, [record])
    return session


def test_request_id_preserved(tmp_path: Path) -> None:
    """FIX-5: top-level requestId is backfilled into data.request_id."""
    session = _make_assistant_only_session(
        tmp_path,
        extra_top_level={"requestId": "req_011CZrDvajP6jsqd1oCpD8RG"},
    )
    result = import_claude_code_session(
        session_path=session, output_dir=tmp_path / "out", include_sidechains=False
    )
    records = _load_records(result)
    llm = next(r for r in records if r.get("action_type") == "llm_call")
    assert llm["data"]["request_id"] == "req_011CZrDvajP6jsqd1oCpD8RG"


def test_request_id_absent_is_empty_string(tmp_path: Path) -> None:
    """FIX-5: missing requestId yields empty string (stable schema, not missing key)."""
    session = _make_assistant_only_session(tmp_path)  # no requestId
    result = import_claude_code_session(
        session_path=session, output_dir=tmp_path / "out", include_sidechains=False
    )
    records = _load_records(result)
    llm = next(r for r in records if r.get("action_type") == "llm_call")
    assert "request_id" in llm["data"], (
        "FIX-5 broken: request_id key should be present even when absent at source"
    )
    assert llm["data"]["request_id"] == ""
def test_is_error_takes_precedence_over_interrupted(tmp_path: Path) -> None:
    """FIX-4 precedence: tool_result.is_error wins over per-tool overrides.

    is_error=False + interrupted=True → success=True (is_error said
    'not an error', so we trust the most authoritative direct signal).
    """
    session = _make_synthetic_tool_session(
        tmp_path,
        tool_name="Bash",
        tool_input={"command": "echo ok"},
        tool_use_result={
            "interrupted": True,
            "isImage": False,
            "noOutputExpected": False,
            "stderr": "",
            "stdout": "ok",
        },
        is_error=False,  # explicit "not an error" — must win
    )
    tool_action = _convert_and_get_tool(session, tmp_path)
    assert tool_action["data"]["success"] is True, (
        "FIX-4 precedence broken: is_error=False should override interrupted=True"
    )


# ─── End-to-end integration: real swe-rebench Claude Code trace ────────


REAL_CC_TRACE = (
    REPO_ROOT
    / "traces"
    / "swe-rebench"
    / "claude-code-haiku"
    / "encode__httpx-2701"
    / "attempt_1"
    / "trace.jsonl"
)
REAL_CC_SESSION_UUID = "2a49ce6f-616e-4072-9a35-6934dcce7383"


def test_real_swe_rebench_claude_code_trace_converts_cleanly(
    tmp_path: Path,
) -> None:
    """End-to-end smoke against the real swe-rebench Claude Code trace.

    This test exercises every fix in this Ralph pass against a real
    collector-produced CC session: FIX-1 (UUID fallback because the
    file is named ``trace.jsonl``), FIX-2 (the trailing ``last-prompt``
    record is discarded), FIX-4 (Edit tool actions get
    ``structured_patch`` backfill), FIX-5 (every assistant has
    ``request_id``).

    Skipped gracefully when the artifact is not present so the suite
    remains runnable on machines without the trace.
    """
    if not REAL_CC_TRACE.exists():
        pytest.skip(f"real CC trace not present at {REAL_CC_TRACE}")

    # FIX-2 source-side check: confirm the real trace actually contains a
    # last-prompt record. Without this guard, a future schema drift that
    # removes last-prompt from CC would render the discard assertion
    # below vacuous (no record to discard → trivially passes).
    with open(REAL_CC_TRACE, encoding="utf-8") as f:
        has_last_prompt = any(
            json.loads(line).get("type") == "last-prompt" for line in f
        )
    assert has_last_prompt, (
        "FIX-2 verification gap: real trace no longer contains a "
        "last-prompt record; this test cannot validate the discard"
    )

    out_dir = tmp_path / "real-cc-out"
    result = import_claude_code_session(
        session_path=REAL_CC_TRACE,
        output_dir=out_dir,
        include_sidechains=False,
    )

    # FIX-1: output landed under the real session UUID, not "trace"
    expected = (
        out_dir
        / "claude-code-import"
        / REAL_CC_SESSION_UUID
        / f"{REAL_CC_SESSION_UUID}.jsonl"
    )
    assert result == expected, (
        f"FIX-1 broken: expected {expected}, got {result}"
    )
    assert not (out_dir / "claude-code-import" / "trace").exists()

    records = _load_records(result)

    # Metadata header reflects the real session
    meta = records[0]
    assert meta["type"] == "trace_metadata"
    assert meta["trace_format_version"] == V5_FORMAT_VERSION
    assert meta["instance_id"] == REAL_CC_SESSION_UUID
    assert meta["model"] == "claude-sonnet-4-6"
    rc = meta.get("run_config", {})
    assert rc.get("cwd") == "/testbed"
    assert rc.get("git_branch") == "master"
    assert rc.get("cli_version") == "2.1.96"

    # Action counts match the verified record-type inventory
    llm_calls = [
        r for r in records
        if r.get("type") == "action" and r.get("action_type") == "llm_call"
    ]
    tool_execs = [
        r for r in records
        if r.get("type") == "action" and r.get("action_type") == "tool_exec"
    ]
    assert len(llm_calls) == 32, f"expected 32 llm_calls, got {len(llm_calls)}"
    assert len(tool_execs) == 22, f"expected 22 tool_execs, got {len(tool_execs)}"

    # FIX-1: every action's agent_id is the real UUID
    for action in llm_calls + tool_execs:
        assert action["agent_id"] == REAL_CC_SESSION_UUID

    # No orphans, no unpaired
    for tool in tool_execs:
        note = tool["data"].get("note", "")
        assert not note.startswith("orphan"), f"unexpected orphan: {note}"
        assert not note.startswith("unpaired"), f"unexpected unpaired: {note}"

    # Real trace has 5 thinking-only assistant records
    thinking_count = sum(
        1 for r in llm_calls if r["data"].get("thinking", "").strip()
    )
    assert thinking_count >= 5, (
        f"expected >=5 llm_calls with thinking; got {thinking_count}"
    )

    # FIX-5: every llm_call has request_id (real CC 2.1.96 always emits it)
    missing_req = [
        r for r in llm_calls if not r["data"].get("request_id", "").startswith("req_")
    ]
    assert not missing_req, (
        f"FIX-5 broken: {len(missing_req)} llm_calls missing request_id"
    )

    # FIX-4: the Edit tool calls (3 of them) all have structured_patch
    edits = [t for t in tool_execs if t["data"]["tool_name"] == "Edit"]
    assert len(edits) == 3
    for edit in edits:
        assert edit["data"].get("structured_patch") is not None, (
            f"FIX-4 broken: Edit action {edit['action_id']} missing structured_patch"
        )

    # Trace loads through the strict canonical trace reader and renders via the Gantt
    from demo.gantt_viewer.backend.payload import build_gantt_payload_multi
    from trace_collect.trace_inspector import TraceData

    data = TraceData.load(result)
    assert data.metadata["trace_format_version"] == 5
    assert data.metadata["scaffold"] == SCAFFOLD_LABEL

    payload = build_gantt_payload_multi([("real-cc", data)])
    assert payload["traces"][0]["lanes"], "expected at least one lane"
    span_types: set[str] = set()
    for lane in payload["traces"][0]["lanes"]:
        for span in lane.get("spans", []):
            span_types.add(span["type"])
    assert "llm" in span_types
    assert "tool" in span_types
