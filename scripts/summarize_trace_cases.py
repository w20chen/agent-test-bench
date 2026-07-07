#!/usr/bin/env python3
"""Summarize selected canonical trace attempts for a compact case-study deck."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from demo.gantt_viewer.backend.payload import build_gantt_payload
from trace_collect.latency_metrics import summarize_llm_latencies
from trace_collect.trace_inspector import TraceData, _action_get


DEFAULT_ATTEMPTS: tuple[tuple[str, str, str], ...] = (
    (
        "SWE-bench Verified",
        "django__django-10880",
        r"C:\Users\29068\Desktop\agent-tool-predictor\swe-bench-verified\django__django-10880\attempt_1",
    ),
    (
        "SWE-bench Verified",
        "astropy__astropy-7336",
        r"C:\Users\29068\Desktop\agent-tool-predictor\swe-bench-verified\astropy__astropy-7336\attempt_1",
    ),
    (
        "SWE-rebench",
        "12rambau__sepal_ui-411",
        r"C:\Users\29068\Desktop\agent-tool-predictor\swe-rebench\12rambau__sepal_ui-411\attempt_1",
    ),
    (
        "SWE-rebench",
        "AzureAD__msal-python-77",
        r"C:\Users\29068\Desktop\agent-tool-predictor\swe-rebench\AzureAD__microsoft-authentication-library-for-python-77\attempt_1",
    ),
    (
        "Terminal-Bench",
        "causal-inference-r",
        r"C:\Users\29068\Desktop\agent-tool-predictor\terminal-bench\causal-inference-r\attempt_1",
    ),
    (
        "Terminal-Bench",
        "query-optimize",
        r"C:\Users\29068\Desktop\agent-tool-predictor\terminal-bench\query-optimize\attempt_1",
    ),
    (
        "DeepResearch-Bench",
        "51",
        r"C:\Users\29068\Desktop\agent-tool-predictor\deep-research-bench\51\attempt_1",
    ),
    (
        "DeepResearch-Bench",
        "66",
        r"C:\Users\29068\Desktop\agent-tool-predictor\deep-research-bench\66\attempt_1",
    ),
)


TOOL_LABELS: dict[str, str] = {
    "read_file": "Read",
    "write_file": "Write",
    "edit_file": "Edit",
    "list_dir": "List",
    "web_search": "Search",
    "web_fetch": "Fetch",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/trace_case_ppt/scratch/case_summary.json"),
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = []
    for benchmark, label, attempt_raw in DEFAULT_ATTEMPTS:
        cases.append(_summarize_attempt(benchmark, label, Path(attempt_raw)))

    payload = {
        "title": "Representative benchmark trace cases",
        "timing_note": (
            "LLM/tool times are recorded from canonical trace spans. Timelines use "
            "gap-compressed real spans so active work remains legible."
        ),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


def _summarize_attempt(benchmark: str, label: str, attempt_dir: Path) -> dict[str, Any]:
    trace_path = attempt_dir / "trace.jsonl"
    if not trace_path.exists():
        raise FileNotFoundError(f"Missing canonical trace: {trace_path}")

    data = TraceData.load(trace_path)
    payload = build_gantt_payload(data, label=label)
    spans = [
        span
        for lane in payload.get("lanes", [])
        for span in lane.get("spans", [])
        if span.get("type") in {"llm", "tool"}
    ]
    spans.sort(key=lambda s: float(s.get("start_real", s.get("start", 0.0))))

    llm_summary = summarize_llm_latencies(
        action.get("data")
        for action in data.actions
        if action.get("action_type") == "llm_call"
    )
    elapsed_s = _first_summary_value(data, "elapsed_s")
    success = _first_summary_value(data, "success")
    tool_counts: Counter[str] = Counter()
    tool_ms_by_name: defaultdict[str, float] = defaultdict(float)
    highlights: list[dict[str, Any]] = []

    for action in data.actions:
        if action.get("action_type") != "tool_exec":
            continue
        tool_name = str(_action_get(action, "tool_name", "unknown"))
        duration_ms = float(
            _action_get(action, "tool_duration_ms", 0)
            or _action_get(action, "duration_ms", 0)
            or 0.0
        )
        tool_counts[tool_name] += 1
        tool_ms_by_name[tool_name] += duration_ms

    compact_spans = [_compact_span(span) for span in spans]
    highlights = _select_highlights(compact_spans)
    total_tool_ms = sum(tool_ms_by_name.values())
    total_llm_ms = float(llm_summary["total_llm_ms"])

    return {
        "benchmark": benchmark,
        "label": label,
        "instance_id": data.metadata.get("instance_id") or label,
        "attempt_dir": str(attempt_dir),
        "model": data.metadata.get("model"),
        "success": success,
        "actions": len(data.actions),
        "elapsed_s": elapsed_s,
        "llm_s": total_llm_ms / 1000.0,
        "tool_s": total_tool_ms / 1000.0,
        "llm_timing_source": llm_summary["llm_timing_source"],
        "top_tools": [
            {
                "name": name,
                "count": count,
                "seconds": round(tool_ms_by_name[name] / 1000.0, 3),
            }
            for name, count in tool_counts.most_common(8)
        ],
        "n_llm_calls": sum(1 for span in compact_spans if span["kind"] == "llm"),
        "n_tool_calls": sum(1 for span in compact_spans if span["kind"] == "tool"),
        "timeline_end_s": _timeline_end(compact_spans),
        "timeline": compact_spans,
        "highlights": highlights,
        "narrative": _narrative(benchmark, label, tool_counts),
    }


def _first_summary_value(data: TraceData, key: str) -> Any:
    for summary in data.summaries:
        if key in summary:
            return summary[key]
    return None


def _compact_span(span: dict[str, Any]) -> dict[str, Any]:
    detail = span.get("detail") or {}
    start = float(span.get("start_real", span.get("start", 0.0)) or 0.0)
    end = float(span.get("end_real", span.get("end", start)) or start)
    kind = "llm" if span.get("type") == "llm" else "tool"
    tool_name = str(detail.get("tool_name") or "")
    return {
        "kind": kind,
        "start_s": round(start, 3),
        "end_s": round(max(end, start), 3),
        "duration_s": round(max(end - start, 0.0), 3),
        "iteration": span.get("iteration"),
        "tool": tool_name if kind == "tool" else "",
        "tool_label": _tool_label(tool_name) if kind == "tool" else "LLM",
        "detail": _important_detail(tool_name, detail),
    }


def _important_detail(tool_name: str, detail: dict[str, Any]) -> str:
    if not tool_name:
        return ""
    for key in ("command", "path", "file_path", "pattern", "query", "url"):
        value = detail.get(key)
        if isinstance(value, str) and value:
            return value[:120]
    preview = detail.get("args_preview") or detail.get("tool_args")
    if isinstance(preview, str):
        return preview[:120]
    return ""


def _tool_label(tool_name: str) -> str:
    if tool_name in TOOL_LABELS:
        return TOOL_LABELS[tool_name]
    if tool_name.startswith("exec-"):
        return tool_name.removeprefix("exec-")
    return tool_name or "Tool"


def _timeline_end(spans: list[dict[str, Any]]) -> float:
    return max((float(span["end_s"]) for span in spans), default=0.0)


def _select_highlights(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tool_spans = [span for span in spans if span["kind"] == "tool"]
    priority_prefixes = (
        "exec-pytest",
        "exec-python",
        "edit_file",
        "write_file",
        "web_search",
        "web_fetch",
        "exec-xxd",
        "exec-grep",
    )
    selected: list[dict[str, Any]] = []
    for prefix in priority_prefixes:
        matches = [
            span
            for span in tool_spans
            if str(span.get("tool", "")).startswith(prefix)
        ]
        if matches:
            selected.append(max(matches, key=lambda s: float(s["duration_s"])))
    if len(selected) < 4:
        selected.extend(
            sorted(tool_spans, key=lambda s: float(s["duration_s"]), reverse=True)
        )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    for span in selected:
        key = (str(span.get("tool", "")), float(span["start_s"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(
            {
                "tool": span.get("tool"),
                "label": span.get("tool_label"),
                "start_s": span.get("start_s"),
                "duration_s": span.get("duration_s"),
                "detail": span.get("detail"),
            }
        )
        if len(deduped) >= 5:
            break
    return deduped


def _narrative(benchmark: str, label: str, counts: Counter[str]) -> str:
    if benchmark.startswith("SWE") and counts.get("edit_file", 0):
        return "Reads the code path, probes behavior with shell/Python, patches files, then validates with targeted tests."
    if benchmark == "Terminal-Bench" and label == "causal-inference-r":
        return "Runs R scripts and package/system checks repeatedly, using the terminal as the main computation surface for statistical analysis."
    if benchmark == "Terminal-Bench" and label == "query-optimize":
        return "Exercises SQLite queries and file checks, comparing plans/results until the optimized query behavior is verified."
    if benchmark == "DeepResearch-Bench":
        return "Alternates web search and page fetches to gather evidence, then consolidates findings into the final answer."
    return "Alternates LLM planning with tool execution to inspect, change, and verify the task state."


if __name__ == "__main__":
    main()
