
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

SCAFFOLD_LABEL = "claude-code"
V5_FORMAT_VERSION = 5
_TOOL_RESULT_MAX_CHARS = 8000  # storage cap; viewer truncates display separately

# Distinguishes native Claude Code filenames from copied ``trace.jsonl`` files.
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)

# Claude Code record types that do not produce trace actions.
_DISCARDABLE_TYPES = frozenset(
    {
        "file-history-snapshot",
        "permission-mode",
        "system",
        "custom-title",
        "agent-name",
        "queue-operation",
        "attachment",  # could become a CONTEXT event in a follow-up
        "last-prompt",  # CC >= 2.1.96 session bookmark; no trace-relevant info
    }
)
_CLAUDE_CODE_RECORD_TYPES = _DISCARDABLE_TYPES | {"assistant", "user"}

def _iso_to_unix(iso_str: str | None) -> float | None:
    if not iso_str:
        return None
    try:
        normalized = iso_str.replace("Z", "+00:00") if iso_str.endswith("Z") else iso_str
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return None

def _harvest_session_metadata(session_path: Path) -> dict[str, Any]:
    harvested: dict[str, Any] = {
        "model": None,
        "cwd": None,
        "git_branch": None,
        "cli_version": None,
        "session_id": None,
    }
    with open(session_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if harvested["cwd"] is None and record.get("cwd"):
                harvested["cwd"] = record["cwd"]
            if harvested["git_branch"] is None and record.get("gitBranch"):
                harvested["git_branch"] = record["gitBranch"]
            if harvested["cli_version"] is None and record.get("version"):
                harvested["cli_version"] = record["version"]
            if harvested["session_id"] is None and record.get("sessionId"):
                harvested["session_id"] = record["sessionId"]
            if harvested["model"] is None and record.get("type") == "assistant":
                model = (record.get("message") or {}).get("model")
                if model:
                    harvested["model"] = model

            if all(v is not None for v in harvested.values()):
                break

    return harvested


def looks_like_claude_code_session(
    session_path: Path,
    *,
    max_records: int = 20,
) -> bool:
    """Best-effort sniff for raw Claude Code session JSONL."""

    session_path = Path(session_path).expanduser().resolve()
    seen_records = 0
    with open(session_path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                return False
            if not isinstance(record, dict):
                return False

            rtype = record.get("type")
            if rtype == "trace_metadata":
                return False
            if rtype in _CLAUDE_CODE_RECORD_TYPES:
                return True

            seen_records += 1
            if seen_records >= max_records:
                return False

    return False
def _split_assistant_content(
    content: list[dict[str, Any]] | str | None,
) -> tuple[str, str, list[dict[str, Any]]]:
    if content is None or isinstance(content, str):
        text = content or ""
        return text, "", []

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_uses: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "thinking":
            thinking_parts.append(block.get("thinking", ""))
        elif btype == "tool_use":
            tool_uses.append(block)

    return "\n".join(text_parts), "\n".join(thinking_parts), tool_uses

def _build_claude_raw_response(
    *,
    model: str | None,
    message_id: str | None,
    content: list[dict[str, Any]] | str | None,
) -> dict[str, Any]:
    return {
        "provider": "anthropic",
        "id": message_id or "",
        "model": model or "",
        "message": {
            "role": "assistant",
            "content": content if content is not None else [],
        },
    }

def _extract_tool_result_text(
    tool_result_content: list[dict[str, Any]] | str | None,
) -> str:
    if tool_result_content is None:
        return ""
    if isinstance(tool_result_content, str):
        return tool_result_content
    parts: list[str] = []
    for block in tool_result_content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            else:
                parts.append(json.dumps(block, ensure_ascii=False))
        else:
            parts.append(str(block))
    return "\n".join(parts)
# Keep stderr previews short; full tool_result text is stored separately.
_BASH_STDERR_PREVIEW_MAX = 2000

def _backfill_per_tool(
    tool_name: str,
    result_dict: dict[str, Any],
) -> tuple[dict[str, Any], bool | None]:
    backfill: dict[str, Any] = {}
    success_override: bool | None = None

    if tool_name == "Bash":
        if result_dict.get("interrupted"):
            success_override = False
        stderr = result_dict.get("stderr")
        if isinstance(stderr, str) and stderr:
            backfill["stderr_preview"] = stderr[:_BASH_STDERR_PREVIEW_MAX]
    elif tool_name == "Edit":
        patch = result_dict.get("structuredPatch")
        if patch is not None:
            backfill["structured_patch"] = patch
        if result_dict.get("userModified"):
            backfill["user_modified"] = True
    elif tool_name == "Read":
        file_meta = result_dict.get("file")
        if isinstance(file_meta, dict):
            keep = {
                k: file_meta[k]
                for k in ("filePath", "numLines", "totalLines", "startLine")
                if k in file_meta
            }
            if keep:
                backfill["file_meta"] = keep

    return backfill, success_override


@dataclass(slots=True)
class _LaneState:
    pending_tool_uses: dict[str, dict[str, Any]] = field(default_factory=dict)
    iteration: int = 0
    last_lane_ts: float | None = None
    n_tool_actions: int = 0
    n_llm_actions: int = 0
    total_llm_ms: float = 0.0
    total_tool_ms: float = 0.0
    total_tokens: int = 0
    first_ts: float | None = None
    last_ts: float | None = None


def _get_lane_state(
    lane_states: dict[str, _LaneState],
    lane_id: str,
    *,
    default_start_ts: float | None,
) -> _LaneState:
    state = lane_states.get(lane_id)
    if state is None:
        state = _LaneState(last_lane_ts=default_start_ts)
        lane_states[lane_id] = state
    return state

def _convert_session_records(
    *,
    session_path: Path,
    agent_id: str,
    session_start_ts: float | None = None,
) -> Iterator[dict[str, Any]]:
    lane_states: dict[str, _LaneState] = {
        agent_id: _LaneState(last_lane_ts=session_start_ts)
    }
    inline_sidechain_lane_id: str | None = None
    inline_sidechain_count = 0
    saw_non_sidechain_record = False

    with open(session_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("skipping malformed JSONL line in %s", session_path)
                continue

            rtype = record.get("type")
            if rtype in _DISCARDABLE_TYPES:
                continue

            is_sidechain = bool(record.get("isSidechain"))
            if is_sidechain and saw_non_sidechain_record:
                if inline_sidechain_lane_id is None:
                    inline_sidechain_count += 1
                    inline_sidechain_lane_id = (
                        f"{agent_id}:inline-sidechain-{inline_sidechain_count}"
                    )
                lane_id = inline_sidechain_lane_id
            else:
                lane_id = agent_id
                if not is_sidechain:
                    saw_non_sidechain_record = True
                inline_sidechain_lane_id = None

            state = _get_lane_state(
                lane_states,
                lane_id,
                default_start_ts=session_start_ts if lane_id == agent_id else None,
            )

            if rtype == "assistant":
                ts_end = _iso_to_unix(record.get("timestamp"))
                if ts_end is None:
                    logger.debug(
                        "assistant record missing timestamp in %s; skipping",
                        session_path,
                    )
                    continue
                ts_start = state.last_lane_ts if state.last_lane_ts is not None else ts_end
                state.last_lane_ts = ts_end
                if state.first_ts is None:
                    state.first_ts = ts_start
                state.last_ts = ts_end

                message = record.get("message") or {}
                content = message.get("content")
                text, thinking, tool_uses = _split_assistant_content(content)

                raw_response = _build_claude_raw_response(
                    model=message.get("model"),
                    message_id=message.get("id"),
                    content=content,
                )

                usage = message.get("usage") or {}
                prompt_tokens = usage.get("input_tokens", 0) or 0
                completion_tokens = usage.get("output_tokens", 0) or 0
                cache_creation_obj = usage.get("cache_creation") or {}

                llm_latency_ms = max(0.0, (ts_end - ts_start) * 1000)

                llm_content_preview = text
                if len(llm_content_preview) > 1000:
                    llm_content_preview = (
                        llm_content_preview[:1000] + "\n[...truncated at 1000 chars]"
                    )

                data: dict[str, Any] = {
                    "raw_response": raw_response,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "llm_latency_ms": round(llm_latency_ms, 2),
                    "llm_content": llm_content_preview,
                    # Backfill: Claude-specific fields preserved under data.*
                    "message_id": message.get("id", ""),
                    # records since CC CLI 2.1.96). Empty string when
                    # absent so the schema is stable across versions.
                    "request_id": record.get("requestId", "") or "",
                    "cache_read_tokens": usage.get("cache_read_input_tokens", 0) or 0,
                    "cache_creation_tokens": usage.get(
                        "cache_creation_input_tokens", 0
                    )
                    or 0,
                    "cache_ephemeral_5m_tokens": cache_creation_obj.get(
                        "ephemeral_5m_input_tokens", 0
                    )
                    or 0,
                    "cache_ephemeral_1h_tokens": cache_creation_obj.get(
                        "ephemeral_1h_input_tokens", 0
                    )
                    or 0,
                }
                if thinking:
                    data["thinking"] = thinking
                if usage.get("service_tier"):
                    data["service_tier"] = usage["service_tier"]

                yield {
                    "type": "action",
                    "action_type": "llm_call",
                    "action_id": f"llm_{state.iteration}",
                    "agent_id": lane_id,
                    "iteration": state.iteration,
                    "ts_start": ts_start,
                    "ts_end": ts_end,
                    "data": data,
                }
                state.n_llm_actions += 1
                state.total_llm_ms += llm_latency_ms
                state.total_tokens += prompt_tokens + completion_tokens

                for tu in tool_uses:
                    tu_id = tu.get("id") or ""
                    if not tu_id:
                        continue
                    state.pending_tool_uses[tu_id] = {
                        "ts_start": ts_end,
                        "tool_name": tu.get("name", "unknown"),
                        "tool_args": json.dumps(
                            tu.get("input") or {}, ensure_ascii=False
                        ),
                        "iteration": state.iteration,
                    }

                state.iteration += 1

            elif rtype == "user":
                message = record.get("message") or {}
                content = message.get("content")
                user_ts = _iso_to_unix(record.get("timestamp"))

                if user_ts is not None:
                    state.last_lane_ts = max(state.last_lane_ts or 0.0, user_ts)

                if not isinstance(content, list):
                    continue
                tool_use_result_raw = record.get("toolUseResult")
                tool_use_result: dict[str, Any] = (
                    tool_use_result_raw
                    if isinstance(tool_use_result_raw, dict)
                    else {}
                )

                for block in content:
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        continue

                    tu_id = block.get("tool_use_id") or ""
                    pending = state.pending_tool_uses.pop(tu_id, None)
                    unpaired = pending is None

                    if pending is not None:
                        tool_ts_start = pending["ts_start"]
                        tool_iteration = pending["iteration"]
                        tool_name = pending["tool_name"]
                        tool_args = pending["tool_args"]
                    else:
                        tool_ts_start = user_ts if user_ts is not None else (
                            state.last_lane_ts or 0.0
                        )
                        tool_iteration = max(0, state.iteration - 1)
                        tool_name = "unknown_tool"
                        tool_args = "{}"

                    sidecar_duration = tool_use_result.get("totalDurationMs")
                    if sidecar_duration is not None:
                        duration_ms = float(sidecar_duration)
                    elif user_ts is not None:
                        duration_ms = max(0.0, (user_ts - tool_ts_start) * 1000)
                    else:
                        duration_ms = 0.0
                    tool_ts_end = tool_ts_start + (duration_ms / 1000.0)
                    state.total_tool_ms += duration_ms

                    per_tool_backfill, success_override = _backfill_per_tool(
                        tool_name, tool_use_result
                    )

                    is_error = block.get("is_error")
                    if is_error is not None:
                        success = not bool(is_error)
                    elif success_override is not None:
                        success = success_override
                    else:
                        success = True

                    result_text = _extract_tool_result_text(block.get("content"))
                    if len(result_text) > _TOOL_RESULT_MAX_CHARS:
                        result_text = (
                            result_text[:_TOOL_RESULT_MAX_CHARS]
                            + f"\n[...truncated at {_TOOL_RESULT_MAX_CHARS} chars]"
                        )

                    tool_data: dict[str, Any] = {
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "tool_result": result_text,
                        "duration_ms": round(duration_ms, 2),
                        "success": success,
                    }
                    if unpaired:
                        tool_data["note"] = (
                            f"unpaired tool_result: no matching tool_use for id "
                            f"{tu_id[:12]}..."
                        )
                    if tool_use_result.get("agentId") or tool_use_result.get("agentType"):
                        tool_data["subagent_meta"] = {
                            "agent_id": tool_use_result.get("agentId"),
                            "agent_type": tool_use_result.get("agentType"),
                        }
                    if (
                        tool_use_result.get("totalTokens") is not None
                        or tool_use_result.get("usage") is not None
                    ):
                        tool_data["subagent_tokens"] = {
                            "total": tool_use_result.get("totalTokens"),
                            "usage": tool_use_result.get("usage"),
                        }
                    if tool_use_result.get("totalToolUseCount") is not None:
                        tool_data["subagent_tool_use_count"] = tool_use_result[
                            "totalToolUseCount"
                        ]
                    tool_data.update(per_tool_backfill)

                    if state.first_ts is None:
                        state.first_ts = tool_ts_start
                    state.last_ts = tool_ts_end

                    tu_id_suffix = (tu_id or "noid")[-8:]
                    yield {
                        "type": "action",
                        "action_type": "tool_exec",
                        "action_id": (
                            f"tool_{tool_iteration}_{tool_name}_{tu_id_suffix}"
                        ),
                        "agent_id": lane_id,
                        "iteration": tool_iteration,
                        "ts_start": tool_ts_start,
                        "ts_end": tool_ts_end,
                        "data": tool_data,
                    }
                    state.n_tool_actions += 1

                    state.last_lane_ts = max(state.last_lane_ts or 0.0, tool_ts_end)

    for lane_id, state in lane_states.items():
        for orphan_id, pending in state.pending_tool_uses.items():
            orphan_ts = pending["ts_start"]
            orphan_id_suffix = orphan_id[-8:] if orphan_id else "noid"
            if state.first_ts is None:
                state.first_ts = orphan_ts
            state.last_ts = max(state.last_ts or 0.0, orphan_ts)
            yield {
                "type": "action",
                "action_type": "tool_exec",
                "action_id": (
                    f"tool_{pending['iteration']}_{pending['tool_name']}"
                    f"_{orphan_id_suffix}_orphan"
                ),
                "agent_id": lane_id,
                "iteration": pending["iteration"],
                "ts_start": orphan_ts,
                "ts_end": orphan_ts,
                "data": {
                    "tool_name": pending["tool_name"],
                    "tool_args": pending["tool_args"],
                    "tool_result": "",
                    "duration_ms": 0.0,
                    "success": False,
                    "note": (
                        f"orphan tool_use: no tool_result received for id "
                        f"{orphan_id[:12]}... (partial write or interrupted session)"
                    ),
                },
            }
            state.n_tool_actions += 1

        yield {
            "__summary__": True,
            "agent_id": lane_id,
            "n_iterations": state.n_llm_actions,
            "n_tool_actions": state.n_tool_actions,
            "total_llm_ms": round(state.total_llm_ms, 2),
            "total_tool_ms": round(state.total_tool_ms, 2),
            "total_tokens": state.total_tokens,
            "elapsed_s": round((state.last_ts or 0.0) - (state.first_ts or 0.0), 3),
        }

def import_claude_code_session(
    *,
    session_path: Path,
    output_dir: Path,
    include_sidechains: bool = True,
    run_id: str | None = None,
) -> Path:
    session_path = Path(session_path).expanduser().resolve()
    if not session_path.exists():
        raise FileNotFoundError(f"Claude Code session not found: {session_path}")

    # Copied ``trace.jsonl`` files need the in-record ``sessionId`` fallback.
    harvested = _harvest_session_metadata(session_path)
    if _UUID_RE.match(session_path.stem):
        session_uuid = session_path.stem
    elif harvested["session_id"]:
        session_uuid = harvested["session_id"]
    else:
        session_uuid = session_path.stem

    run_dir_name = run_id or session_uuid
    run_dir = (
        Path(output_dir).expanduser().resolve()
        / "claude-code-import"
        / run_dir_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    target_path = run_dir / f"{session_uuid}.jsonl"

    run_config: dict[str, Any] = {}
    if harvested["cwd"]:
        run_config["cwd"] = harvested["cwd"]
    if harvested["git_branch"]:
        run_config["git_branch"] = harvested["git_branch"]
    if harvested["cli_version"]:
        run_config["cli_version"] = harvested["cli_version"]

    metadata: dict[str, Any] = {
        "type": "trace_metadata",
        "trace_format_version": V5_FORMAT_VERSION,
        "scaffold": SCAFFOLD_LABEL,
        "mode": "import",
        "instance_id": session_uuid,
        "source_trace": str(session_path),
    }
    if harvested["model"]:
        metadata["model"] = harvested["model"]
    if run_config:
        metadata["run_config"] = run_config

    all_actions: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    main_agent_id = session_uuid
    for item in _convert_session_records(
        session_path=session_path, agent_id=main_agent_id
    ):
        if item.get("__summary__"):
            summaries.append(item)
        else:
            all_actions.append(item)

    if include_sidechains:
        sidechain_dir = session_path.parent / session_uuid / "subagents"
        if sidechain_dir.exists() and sidechain_dir.is_dir():
            for agent_file in sorted(sidechain_dir.glob("agent-*.jsonl")):
                sub_agent_id = agent_file.stem  # e.g. "agent-a3db9c581cbafb09d"
                for item in _convert_session_records(
                    session_path=agent_file, agent_id=sub_agent_id
                ):
                    if item.get("__summary__"):
                        summaries.append(item)
                    else:
                        all_actions.append(item)

    main_summary = next(
        (s for s in summaries if s["agent_id"] == main_agent_id), None
    )
    if main_summary:
        metadata["max_iterations"] = main_summary["n_iterations"]

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as dst:
        dst.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for action in all_actions:
            dst.write(json.dumps(action, ensure_ascii=False) + "\n")
        for summary_data in summaries:
            record = {
                "type": "summary",
                "agent_id": summary_data["agent_id"],
                "n_iterations": summary_data["n_iterations"],
                "n_tool_actions": summary_data["n_tool_actions"],
                "total_llm_ms": summary_data["total_llm_ms"],
                "total_tool_ms": summary_data["total_tool_ms"],
                "total_tokens": summary_data["total_tokens"],
                "elapsed_s": summary_data["elapsed_s"],
                "model": harvested["model"] or "",
            }
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        "Imported Claude Code session %s → %s (%d actions, %d lanes)",
        session_uuid,
        target_path,
        len(all_actions),
        len(summaries),
    )
    return target_path
