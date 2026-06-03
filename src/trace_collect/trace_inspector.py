from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trace_collect.latency_metrics import (
    get_llm_wall_latency_ms,
    get_openrouter_latency_ms,
    get_preferred_llm_latency_ms,
    summarize_llm_latencies,
)

CURRENT_TRACE_FORMAT_VERSION = 5


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"... ({len(text) - limit} chars truncated)"


def _to_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _action_get(action: dict[str, Any] | None, key: str, default: Any = None) -> Any:
    if not action:
        return default
    data = action.get("data") or {}
    if key in data and data[key] is not None:
        return data[key]
    return action.get(key, default)


def _raw_response_text(raw_response: dict[str, Any]) -> str:
    choices = raw_response.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content

    message = raw_response.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    text_parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(part for part in text_parts if part)


@dataclass
class TraceData:
    path: Path
    metadata: dict[str, Any]  # trace_metadata record
    actions: list[dict[str, Any]]  # sorted by iteration, then ts_start
    events: list[dict[str, Any]]  # sorted by ts
    summaries: list[dict[str, Any]]
    agents: list[str]  # unique agent_ids in order seen

    @classmethod
    def load(cls, path: Path, agent_filter: str | None = None) -> "TraceData":
        metadata: dict[str, Any] = {}
        actions: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        seen_agents: dict[str, None] = {}

        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                rec_type = record.get("type", "")
                agent_id = record.get("agent_id")

                if agent_filter is not None and agent_id is not None:
                    if agent_filter not in agent_id:
                        continue

                if agent_id is not None:
                    seen_agents[agent_id] = None

                if rec_type == "trace_metadata":
                    metadata.update(record)
                elif rec_type == "action":
                    actions.append(record)
                elif rec_type == "event":
                    events.append(record)
                elif rec_type == "summary":
                    summaries.append(record)
                else:
                    raise ValueError(
                        f"Unsupported record type {rec_type!r} in {path}:{lineno}; "
                        "expected a canonical trace JSONL."
                    )

        actions.sort(key=lambda r: (r.get("iteration", 0), r.get("ts_start", 0)))
        events.sort(key=lambda r: r.get("ts", 0.0))

        if not metadata:
            raise ValueError(
                f"Missing trace_metadata record in {path}; expected a canonical trace JSONL."
            )
        if metadata.get("trace_format_version") != CURRENT_TRACE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported trace_format_version {metadata.get('trace_format_version')!r} "
                f"in {path}; expected canonical trace format "
                f"{CURRENT_TRACE_FORMAT_VERSION}."
            )

        for act in actions:
            data = act.get("data") or {}
            if "llm_output" not in act:
                llm_content = data.get("llm_content")
                if llm_content:
                    act["llm_output"] = llm_content
                else:
                    raw = data.get("raw_response") or {}
                    act["llm_output"] = _raw_response_text(raw)

        return cls(
            path=path,
            metadata=metadata,
            actions=actions,
            events=events,
            summaries=summaries,
            agents=list(seen_agents.keys()),
        )


def cmd_overview(data: TraceData, as_json: bool = False) -> None:
    total_prompt = sum(_action_get(s, "prompt_tokens", 0) for s in data.actions)
    total_completion = sum(_action_get(s, "completion_tokens", 0) for s in data.actions)
    total_tokens = total_prompt + total_completion
    llm_summary = summarize_llm_latencies(
        s.get("data") for s in data.actions if s.get("action_type") == "llm_call"
    )
    total_llm_ms = float(llm_summary["total_llm_ms"])
    total_llm_wall_ms = float(llm_summary["total_llm_wall_ms"])
    total_tool_ms = sum(
        _action_get(s, "tool_duration_ms", 0) or _action_get(s, "duration_ms", 0)
        for s in data.actions
    )

    elapsed_s: float | None = None
    success: bool | None = None
    for summary in data.summaries:
        if "elapsed_s" in summary:
            elapsed_s = summary["elapsed_s"]
        if "success" in summary:
            success = summary["success"]

    tool_counts: dict[str, int] = {}
    for act in data.actions:
        tool = _action_get(act, "tool_name", None)
        if tool:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
    distinct_iterations = {
        act.get("iteration") for act in data.actions if act.get("iteration") is not None
    }

    info: dict[str, Any] = {
        "path": str(data.path),
        "agents": data.agents,
        "scaffold": data.metadata.get("scaffold"),
        "mode": data.metadata.get("mode"),
        "model": data.metadata.get("model"),
        "n_iterations": len(distinct_iterations),
        "n_events": len(data.events),
        "tool_counts": tool_counts,
        "total_tokens": total_tokens,
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_llm_ms": total_llm_ms,
        "total_llm_wall_ms": total_llm_wall_ms,
        "total_llm_call_time_ms": llm_summary["total_llm_call_time_ms"],
        "llm_call_time_count": llm_summary["llm_call_time_count"],
        "llm_timing_source": llm_summary["llm_timing_source"],
        "total_tool_ms": total_tool_ms,
        "elapsed_s": elapsed_s,
        "success": success,
    }

    if as_json:
        print(json.dumps(info, indent=2))
        return

    print(f"Trace: {data.path}")
    print(f"  Agents    : {', '.join(data.agents) if data.agents else '(none)'}")
    print(f"  Scaffold  : {info['scaffold']}")
    print(f"  Mode      : {info['mode']}")
    print(f"  Model     : {info['model']}")
    print(f"  Steps     : {info['n_iterations']}")
    print(f"  Events    : {info['n_events']}")
    if tool_counts:
        counts_str = ", ".join(
            f"{k}={v}" for k, v in sorted(tool_counts.items(), key=lambda x: -x[1])
        )
        print(f"  Tools     : {counts_str}")
    print(
        f"  Tokens    : {total_tokens} (prompt={total_prompt}, completion={total_completion})"
    )
    print(f"  LLM time  : {total_llm_ms:.0f} ms")
    if abs(total_llm_wall_ms - total_llm_ms) > 0.5:
        print(f"  LLM wall  : {total_llm_wall_ms:.0f} ms")
    print(f"  LLM source: {llm_summary['llm_timing_source']}")
    print(f"  Tool time : {total_tool_ms:.0f} ms")
    if elapsed_s is not None:
        print(f"  Elapsed   : {elapsed_s:.1f} s")
    if success is not None:
        print(f"  Success   : {success}")


def cmd_step(
    data: TraceData,
    step_idx: int,
    *,
    truncate: int = 2000,
    as_json: bool = False,
) -> None:
    matching = [s for s in data.actions if s.get("iteration") == step_idx]
    if not matching:
        avail = sorted(
            {s.get("iteration") for s in data.actions if s.get("iteration") is not None}
        )
        msg = f"step {step_idx} not found (available: {avail})"
        if as_json:
            print(json.dumps({"error": msg}))
        else:
            print(f"ERROR: {msg}")
        return

    llm_step = next((s for s in matching if s.get("action_type") == "llm_call"), None)
    tool_step = next((s for s in matching if s.get("action_type") == "tool_exec"), None)
    step = llm_step or tool_step or matching[0]

    if as_json:
        out = dict(step)
        if tool_step is not None and tool_step is not step:
            out["tool_exec"] = tool_step
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    print(f"--- Step {step_idx} ---")
    print(f"  agent_id        : {step.get('agent_id')}")
    print(f"  phase           : {step.get('phase')}")
    print(f"  prompt_tokens   : {_action_get(llm_step or step, 'prompt_tokens')}")
    print(f"  completion_tokens: {_action_get(llm_step or step, 'completion_tokens')}")
    llm_data = (llm_step or step).get("data") or {}
    print(f"  llm_latency_ms  : {get_preferred_llm_latency_ms(llm_data)}")
    print(f"  llm_wall_latency_ms: {get_llm_wall_latency_ms(llm_data)}")
    print(f"  openrouter_latency_ms: {get_openrouter_latency_ms(llm_data)}")
    print(
        "  openrouter_generation_time_ms: "
        f"{_action_get(llm_step or step, 'openrouter_generation_time_ms')}"
    )
    print(
        "  openrouter_provider_latency_ms: "
        f"{_action_get(llm_step or step, 'openrouter_provider_latency_ms')}"
    )
    print(
        f"  openrouter_generation_id: "
        f"{_action_get(llm_step or step, 'openrouter_generation_id')}"
    )
    print(
        f"  openrouter_request_id: "
        f"{_action_get(llm_step or step, 'openrouter_request_id')}"
    )
    print(
        f"  openrouter_provider_name: "
        f"{_action_get(llm_step or step, 'openrouter_provider_name')}"
    )
    print(
        f"  openrouter_upstream_id: "
        f"{_action_get(llm_step or step, 'openrouter_upstream_id')}"
    )
    print(f"  ttft_ms         : {_action_get(llm_step or step, 'ttft_ms')}")
    print(f"  tpot_ms         : {_action_get(llm_step or step, 'tpot_ms')}")
    print(f"  ts_start        : {step.get('ts_start')}")
    print(f"  ts_end          : {step.get('ts_end')}")
    print(f"  tool_name       : {_action_get(tool_step, 'tool_name')}")
    print(
        f"  tool_duration_ms: "
        f"{_action_get(tool_step, 'tool_duration_ms', _action_get(tool_step, 'duration_ms'))}"
    )
    print(f"  success         : {_action_get(tool_step, 'success')}")
    print(f"  tool_ts_start   : {tool_step.get('ts_start') if tool_step else None}")
    print(f"  tool_ts_end     : {tool_step.get('ts_end') if tool_step else None}")
    if tool_step is not None and _action_get(tool_step, "tool_args") is not None:
        print(
            f"  tool_args       : "
            f"{_truncate(_to_str(_action_get(tool_step, 'tool_args')), truncate)}"
        )
    if tool_step is not None and _action_get(tool_step, "tool_result") is not None:
        print(
            f"  tool_result     : "
            f"{_truncate(_to_str(_action_get(tool_step, 'tool_result')), truncate)}"
        )
    if "llm_output" in step:
        print(f"  llm_output      : {_truncate(_to_str(step['llm_output']), truncate)}")


def cmd_messages(
    data: TraceData,
    step_idx: int,
    *,
    role_filter: str | None = None,
    truncate: int = 2000,
    as_json: bool = False,
) -> None:
    step = next(
        (
            s
            for s in data.actions
            if s.get("iteration") == step_idx and s.get("action_type") == "llm_call"
        ),
        None,
    )
    if step is None:
        if as_json:
            print(json.dumps({"error": f"step {step_idx} not found"}))
        else:
            print(f"ERROR: step {step_idx} not found")
        return

    messages: list[dict[str, Any]] = (step.get("data") or {}).get(
        "messages_in", []
    ) or []
    if role_filter:
        messages = [m for m in messages if m.get("role") == role_filter]

    if as_json:
        out = [
            {
                "role": m.get("role"),
                "content": _truncate(_to_str(m.get("content", "")), truncate),
            }
            for m in messages
        ]
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    print(
        f"--- Messages for step {step_idx}"
        + (f" (role={role_filter})" if role_filter else "")
        + " ---"
    )
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        content = _truncate(_to_str(msg.get("content", "")), truncate)
        print(f"  [{i}] {role}: {content}")


def cmd_response(
    data: TraceData,
    step_idx: int,
    *,
    truncate: int = 2000,
    as_json: bool = False,
) -> None:
    step = next(
        (
            s
            for s in data.actions
            if s.get("iteration") == step_idx and s.get("action_type") == "llm_call"
        ),
        None,
    )
    if step is None:
        if as_json:
            print(json.dumps({"error": f"step {step_idx} not found"}))
        else:
            print(f"ERROR: step {step_idx} not found")
        return

    raw = (step.get("data") or {}).get("raw_response")
    if raw is None:
        if as_json:
            print(json.dumps({"error": f"Step {step_idx} has no raw_response field."}))
        else:
            print(f"Step {step_idx} has no raw_response field.")
        return

    text = json.dumps(raw, indent=2, ensure_ascii=False)
    text = _truncate(text, truncate)

    if as_json:
        print(
            json.dumps(
                {"iteration": step_idx, "raw_response": raw},
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    print(f"--- raw_response for step {step_idx} ---")
    print(text)


def cmd_events(
    data: TraceData,
    *,
    category: str | None = None,
    iteration: int | None = None,
    as_json: bool = False,
) -> None:
    events = data.events

    if category is not None:
        cat_upper = category.upper()
        events = [e for e in events if e.get("category", "").upper() == cat_upper]
    if iteration is not None:
        events = [e for e in events if e.get("iteration") == iteration]

    if as_json:
        print(json.dumps(events, indent=2, ensure_ascii=False))
        return

    if not events:
        print("No events found.")
        return

    print(f"--- Events ({len(events)} total) ---")
    for ev in events:
        ts = ev.get("ts") or ev.get("ts_start") or "?"
        name = ev.get("event", "?")
        cat = ev.get("category", "?")
        itr = ev.get("iteration", "?")
        data_fields = ev.get("data", {})

        data_str = ""
        if isinstance(data_fields, dict) and data_fields:
            items = []
            for k, v in list(data_fields.items())[:5]:
                if k == "tool_args":
                    v = str(v)[:80]
                items.append(f"{k}={v}")
            data_str = " | " + ", ".join(items)
        print(f"  ts={ts:<12} event={name:<20} cat={cat:<8} step={itr}{data_str}")


def cmd_tools(
    data: TraceData,
    *,
    step_idx: int | None = None,
    as_json: bool = False,
) -> None:
    steps = data.actions
    if step_idx is not None:
        steps = [s for s in steps if s.get("iteration") == step_idx]

    # name -> {count, total_ms, successes}
    agg: dict[str, dict[str, Any]] = {}
    for step in steps:
        if step.get("action_type") != "tool_exec":
            continue
        d = step.get("data") or {}
        tool = d.get("tool_name")
        if not tool:
            continue
        if tool not in agg:
            agg[tool] = {"count": 0, "total_duration_ms": 0.0, "successes": 0}
        agg[tool]["count"] += 1
        agg[tool]["total_duration_ms"] += d.get("duration_ms", 0.0) or 0.0
        if d.get("success"):
            agg[tool]["successes"] += 1

    rows = []
    for name, stats in agg.items():
        count = stats["count"]
        total_ms = stats["total_duration_ms"]
        successes = stats["successes"]
        success_rate = successes / count if count > 0 else 0.0
        rows.append(
            {
                "tool_name": name,
                "count": count,
                "total_duration_ms": total_ms,
                "success_rate": success_rate,
            }
        )
    rows.sort(key=lambda r: -r["count"])

    if as_json:
        print(json.dumps(rows, indent=2))
        return

    if not rows:
        print("No tool calls found.")
        return

    header = f"{'Tool':<20} {'Count':>6} {'Total ms':>10} {'Success%':>10}"
    print(
        f"--- Tool Usage{' (step=' + str(step_idx) + ')' if step_idx is not None else ''} ---"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"  {row['tool_name']:<18} {row['count']:>6} "
            f"{row['total_duration_ms']:>10.0f} {row['success_rate'] * 100:>9.1f}%"
        )


def cmd_search(
    data: TraceData,
    pattern: str,
    *,
    truncate: int = 200,
    as_json: bool = False,
) -> None:
    if not pattern:
        if as_json:
            print(json.dumps({"error": "search pattern is required."}))
        else:
            print("ERROR: search pattern is required.")
        return

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        if as_json:
            print(json.dumps({"error": f"invalid regex pattern: {exc}"}))
        else:
            print(f"ERROR: invalid regex pattern: {exc}")
        return

    results = []
    for step in data.actions:
        llm_output = step.get("llm_output", "") or ""
        if not isinstance(llm_output, str):
            llm_output = json.dumps(llm_output)
        match = regex.search(llm_output)
        if match:
            start = max(0, match.start() - 60)
            end = min(len(llm_output), match.end() + 60)
            context = llm_output[start:end]
            if truncate > 0:
                context = _truncate(context, truncate)
            results.append(
                {
                    "iteration": step.get("iteration"),
                    "match_start": match.start(),
                    "context": context,
                }
            )

    if as_json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    if not results:
        print(f"No matches for pattern: {pattern!r}")
        return

    print(f"--- Search results for {pattern!r} ({len(results)} match(es)) ---")
    for r in results:
        print(f"  iter {r['iteration']}: ...{r['context']}...")


_TIMELINE_ICONS: dict[str, str] = {
    "skill_load": "📦",
    "skill_load_failed": "❌📦",
    "skills_summary_build": "📋",
    "memory_context_load": "🧠",
    "system_prompt_build": "📐",
    "message_list_build": "📬",
    "consolidation_trigger": "🧹",
    "consolidation_llm_call": "🧠⚡",
    "memory_write": "💾",
    "history_append": "📝",
    "consolidation_failure": "⚠️🧠",
    "raw_archive": "📦⚠️",
    "consolidation_complete": "✅🧠",
    "background_consolidation_scheduled": "🔄🧠",
    "mcp_connect_start": "🔌",
    "mcp_server_connect": "🔗",
    "mcp_server_connected": "✅🔌",
    "mcp_server_failed": "❌🔌",
    "mcp_tool_register": "📝🔧",
    "mcp_tool_call": "⚡🔧",
    "mcp_tool_timeout": "⏰🔧",
    "mcp_disconnect": "🔌❌",
    "session_create": "🆕",
    "session_load": "📂",
    "session_save": "💾",
    "session_turn_save": "💾↩️",
    "llm_request": "▶️🤖",
    "llm_response": "◀️🤖",
    "llm_retry": "🔄🤖",
    "llm_error": "❌🤖",
    "finalization_retry": "🔄📝",
    "max_iterations": "⏹️",
    "llm_call_start": "▶️🤖",
    "llm_call_end": "◀️🤖",
    "llm_action": "▶️🤖",
    "tool_exec_start": "⚙️",
    "tool_exec_end": "✅",
    "tool_prepare": "🔧",
    "tool_prepare_error": "❌🔧",
    "tool_execute": "⚙️",
    "tool_complete": "✅",
    "tool_error": "❌",
    "tool_timeout": "⏰",
    "tool_cancelled": "🚫",
    "external_lookup_blocked": "🚫🔍",
    "read_file": "📖",
    "write_file": "📝",
    "edit_file": "✏️",
    "list_dir": "📁",
    "exec": "⚙️💻",
    "exec_safety_block": "🛡️",
    "web_search": "🔍",
    "web_fetch": "🌐",
    "message": "📨",
    "subagent_spawn": "🌱",
    "subagent_start": "🏃🌱",
    "subagent_tool_execute": "⚙️🌱",
    "subagent_complete": "✅🌱",
    "subagent_error": "❌🌱",
    "subagent_cancel": "🚫🌱",
    "subagent_announcement": "📢🌱",
    "message_dispatch": "📤",
    "session_lock_acquire": "🔒",
    "session_lock_release": "🔓",
    "concurrency_gate_acquire": "🚦",
    "task_complete": "🏁",
    "priority_command_bypass": "⚡",
}

_CATEGORY_SHORT: dict[str, str] = {
    "SCHEDULING": "sched",
    "SESSION": "session",
    "CONTEXT": "context",
    "LLM": "llm",
    "TOOL": "tool",
    "MCP": "mcp",
    "MEMORY": "memory",
    "SUBAGENT": "subagent",
}


def _fmt_tl_event(rec: dict[str, Any], t0: float = 0.0) -> str:
    event_name = rec.get("event", "unknown")
    category = rec.get("category", "")
    data = rec.get("data", {})
    itr = rec.get("iteration", "?")
    ts = rec.get("ts", 0.0)
    rel = ts - t0 if t0 > 0 and ts > 0 else 0.0

    icon = _TIMELINE_ICONS.get(event_name, "  ")
    cat = _CATEGORY_SHORT.get(category, category.lower()[:6])

    parts: list[str] = []
    for key in (
        "skill_name",
        "source",
        "server_name",
        "tool_name",
        "transport",
        "tools_registered",
        "session_key",
        "task_id",
        "label",
        "command_preview",
        "path",
        "query",
        "error_message",
        "error_type",
        "request_id",
        "openrouter_request_id",
        "openrouter_generation_id",
        "openrouter_provider_name",
    ):
        if key in data:
            parts.append(f"{key}={data[key]}")
    for key in (
        "http_status",
        "wait_ms",
        "dispatch_duration_ms",
        "history_messages",
        "total_messages",
        "memory_size_chars",
        "messages_count",
        "duration_ms",
        "consecutive_failures",
        "result_count",
    ):
        if key in data:
            parts.append(f"{key}={data[key]}")
    if "success" in data:
        parts.append("ok" if data["success"] else "FAIL")

    detail = "  ".join(parts)
    return f"  +{rel:7.1f}s {icon} [{cat:>7}] {event_name:<30} iter={itr:<3} {detail}"


def _fmt_tl_action(rec: dict[str, Any], t0: float = 0.0) -> str:
    action_type = rec.get("action_type", "?")
    iteration = rec.get("iteration", "?")
    data = rec.get("data") or {}
    ts_start = rec.get("ts_start", 0)
    rel = ts_start - t0 if t0 > 0 and ts_start > 0 else 0.0

    if action_type == "llm_call":
        pt = data.get("prompt_tokens", 0)
        ct = data.get("completion_tokens", 0)
        lat = get_preferred_llm_latency_ms(data) / 1000
        wall = get_llm_wall_latency_ms(data) / 1000
        wall_suffix = f"  wall={wall:.1f}s" if abs(wall - lat) > 0.05 else ""
        return (
            f"  +{rel:7.1f}s iter={iteration:<3}  ◀ LLM  {pt}+{ct}tok  "
            f"lat={lat:.1f}s{wall_suffix}"
        )
    elif action_type == "tool_exec":
        tool_name = data.get("tool_name", "?")
        dur = (data.get("duration_ms") or 0) / 1000
        ok = "✓" if data.get("success") else "✗"
        result_preview = (data.get("tool_result") or "")[:80].replace("\n", "↵")
        if result_preview:
            result_preview = f"  {result_preview}"
        return (
            f"  +{rel:7.1f}s iter={iteration:<3}  {ok}  {tool_name}  "
            f"dur={dur:.2f}s{result_preview}"
        )
    return f"  +{rel:7.1f}s iter={iteration:<3}  {action_type}"


def _print_tl_summary(summary: dict[str, Any]) -> None:
    llm_s = summary.get("total_llm_ms", 0) / 1000
    llm_wall_s = summary.get("total_llm_wall_ms", summary.get("total_llm_ms", 0)) / 1000
    tool_s = summary.get("total_tool_ms", 0) / 1000
    elapsed = summary.get("elapsed_s", 0)
    n = summary.get("n_iterations", 0)
    tokens = summary.get("total_tokens", 0)
    ok = "✓ success" if summary.get("success") else "✗ failed"
    prepare_ms = summary.get("prepare_ms")
    prepare_str = f"  prepare={prepare_ms:.0f}ms" if prepare_ms else ""
    wall_str = f"  wall={llm_wall_s:.1f}s" if abs(llm_wall_s - llm_s) > 0.05 else ""

    print("─" * 80)
    print(
        f"  {ok}  {n} steps  "
        f"elapsed={elapsed:.0f}s  LLM={llm_s:.1f}s{wall_str}  tool={tool_s:.1f}s  "
        f"tokens={tokens}{prepare_str}"
    )
    tool_ms = summary.get("tool_ms_by_name", {})
    if tool_ms:
        print("  Tool time breakdown:")
        for name, ms in sorted(tool_ms.items(), key=lambda x: -x[1]):
            if ms > 0:
                print(f"    {name:20s}: {ms / 1000:.1f}s")
    timeouts = summary.get("tool_timeouts", {})
    if timeouts:
        print("  Tool timeouts:")
        for name, count in sorted(timeouts.items()):
            print(f"    {name:20s}: {count}")


def cmd_timeline(data: TraceData) -> None:

    scaffold = data.metadata.get("scaffold", "")
    mode = data.metadata.get("mode", "collect")
    simulate_mode = data.metadata.get("simulate_mode", "")
    model = data.metadata.get("model", "")
    if not model and not (mode == "simulate" and simulate_mode == "cloud_model"):
        model = data.metadata.get("local_model", "")

    print(f"Trace: {data.path.name}")
    print(f"  Scaffold: {scaffold}  Mode: {mode}")
    if model:
        print(f"  Model: {model}")
    if mode == "simulate":
        src = data.metadata.get("source_model", "?")
        if simulate_mode == "cloud_model":
            local = "cloud replay"
        else:
            local = data.metadata.get("local_model", "?")
        print(f"  Simulate: {src} → {local}")

    for agent_id in data.agents:
        agent_actions = [s for s in data.actions if s.get("agent_id") == agent_id]
        agent_events = [e for e in data.events if e.get("agent_id") == agent_id]
        agent_summaries = [s for s in data.summaries if s.get("agent_id") == agent_id]

        print(f"\nTimeline: {agent_id}")
        print("─" * 80)

        entries: list[tuple[float, str, dict[str, Any]]] = []
        for a in agent_actions:
            entries.append((a.get("ts_start", 0), "action", a))
        for e in agent_events:
            entries.append((e.get("ts", 0), "event", e))
        entries.sort(key=lambda x: x[0])

        t0 = min((ts for ts, _, _ in entries if ts > 0), default=0.0)

        for _, entry_type, rec in entries:
            if entry_type == "action":
                print(_fmt_tl_action(rec, t0))
            else:
                print(_fmt_tl_event(rec, t0))

        summary = agent_summaries[0] if agent_summaries else None
        if summary:
            _print_tl_summary(summary)
