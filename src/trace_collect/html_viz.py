"""Generate a self-contained HTML visualization from a trace attempt directory.

Usage::

    PYTHONPATH=src python -m trace_collect.html_viz <attempt_dir>
    # or:
    PYTHONPATH=src python -m trace_collect.html_viz traces/swe-rebench/deepseek-chat/20260603T051712/12rambau__sepal_ui-411/attempt_1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Any


from trace_collect.exec_classifier import classify_exec_tool_name


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_iso(ts_str: str) -> float:
    """Parse ISO datetime string to UTC epoch seconds."""
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return 0.0


def _ts_to_str(ts: float) -> str:
    """Convert epoch seconds to readable time."""
    if ts <= 0:
        return "N/A"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S")


def _esc(text: Any) -> str:
    """HTML-escape a value."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _safe_float(val: Any) -> float:
    """Convert a possibly-string float to float, stripping % and whitespace."""
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().rstrip("%").strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def _parse_mem_mb(mem_usage: str) -> float:
    """Parse Docker mem_usage like '1.2GiB / 4GiB' → MB."""
    if not mem_usage:
        return 0.0
    left = mem_usage.split("/")[0].strip()
    try:
        if left.endswith("GiB"):
            return float(left[:-3]) * 1024
        if left.endswith("MiB"):
            return float(left[:-3])
        if left.endswith("KiB"):
            return float(left[:-3]) / 1024
        if left.endswith("GB"):
            return float(left[:-2]) * 1000
        if left.endswith("MB"):
            return float(left[:-2])
        if left.endswith("KB"):
            return float(left[:-2]) / 1000
        if left.endswith("B"):
            return float(left[:-1]) / (1024 * 1024)
    except ValueError:
        pass
    return 0.0


def generate_html(attempt_dir: Path) -> str:
    """Generate a self-contained HTML report string."""
    trace_path = attempt_dir / "trace.jsonl"
    resources_path = attempt_dir / "resources.json"
    manifest_path = attempt_dir / "run_manifest.json"

    records = _load_jsonl(trace_path)
    resources = _load_json(resources_path)
    manifest = _load_json(manifest_path)

    # ── Classify exec tool names in-memory ─────────────────────────
    # Post-process actions and rebuild summary so Gantt bars, pie
    # chart, and stats all reflect exec-grep / exec-find / etc.
    # Works on both fresh traces (already processed by
    # attempt_pipeline) and older unprocessed traces.
    tool_ms_by_name: dict[str, float] = {}
    tool_timeouts: dict[str, int] = {}
    for rec in records:
        if rec.get("type") == "action" and rec.get("action_type") == "tool_exec":
            data = rec.get("data")
            if isinstance(data, dict):
                tn = data.get("tool_name", "")
                ta = data.get("tool_args", "")
                classified = classify_exec_tool_name(tn, ta)
                if classified != tn:
                    data["tool_name"] = classified
                dur = data.get("duration_ms", 0.0) or 0.0
                ok = data.get("success", True)
                tool_ms_by_name[classified] = tool_ms_by_name.get(classified, 0.0) + dur
                if not ok:
                    tool_timeouts[classified] = tool_timeouts.get(classified, 0) + 1
    for rec in records:
        if rec.get("type") == "summary":
            if tool_ms_by_name:
                rec["tool_ms_by_name"] = tool_ms_by_name
            if tool_timeouts:
                rec["tool_timeouts"] = tool_timeouts
            break

    # ── Extract metadata ──────────────────────────────────────────
    meta: dict[str, Any] = {}
    for rec in records:
        if rec.get("type") == "trace_metadata":
            meta = rec
            break

    # ── Extract actions (llm_call + tool_exec) ────────────────────
    actions: list[dict[str, Any]] = []
    for rec in records:
        if rec.get("type") == "action":
            actions.append(rec)

    # ── Resource samples ──────────────────────────────────────────
    resource_samples = resources.get("samples", [])
    resource_summary = resources.get("summary", {})

    # Determine time origin: earliest of (first action ts_start, first resource epoch)
    t0 = 0.0
    for a in actions:
        ts = a.get("ts_start", 0)
        if ts > 0:
            t0 = ts
            break
    for s in resource_samples:
        epoch = s.get("epoch", 0)
        if isinstance(epoch, (int, float)) and epoch > 0:
            if t0 == 0 or epoch < t0:
                t0 = epoch
            break
    if t0 == 0:
        for a in actions:
            ts = a.get("ts", 0)
            if ts > 0:
                t0 = ts
                break

    # ── Extract summary ───────────────────────────────────────────
    summary: dict[str, Any] = {}
    for rec in records:
        if rec.get("type") == "summary":
            summary = rec
            break

    # ── Build Gantt data ──────────────────────────────────────────
    gantt_items: list[dict[str, Any]] = []
    for a in actions:
        ts_s = a.get("ts_start", 0)
        ts_e = a.get("ts_end", 0)
        if ts_s <= 0 or ts_e <= 0:
            continue
        atype = a.get("action_type", "?")
        iteration = a.get("iteration", 0)
        label = ""
        color = ""
        tooltip_extra = ""
        if atype == "llm_call":
            label = f"LLM #{iteration}"
            color = "#4a90d9"
        elif atype == "tool_exec":
            tdata = a.get("data") or {}
            tool_name = tdata.get("tool_name", "?")
            label = f"{tool_name} #{iteration}"
            # Extract a short preview of tool arguments for the tooltip.
            tooltip_extra = ""
            tool_args_raw = tdata.get("tool_args", "")
            if tool_args_raw:
                try:
                    args_dict = json.loads(tool_args_raw) if isinstance(tool_args_raw, str) else tool_args_raw
                    if isinstance(args_dict, dict):
                        # Show the first meaningful value (command, path, query, etc.)
                        preview_keys = ("command", "path", "query", "url", "text", "content", "pattern", "name")
                        for k in preview_keys:
                            if k in args_dict and args_dict[k]:
                                val = str(args_dict[k])
                                tooltip_extra = f" | {val[:200]}"
                                break
                        if not tooltip_extra:
                            # Fallback: first key-value pair
                            first_kv = next(iter(args_dict.items()), None)
                            if first_kv:
                                tooltip_extra = f" | {first_kv[0]}={str(first_kv[1])[:60]}"
                except (json.JSONDecodeError, TypeError, StopIteration):
                    pass
            # Distinct colors per tool type
            tool_colors = {
                "exec": "#e67e22",
                "read_file": "#27ae60",
                "write_file": "#8e44ad",
                "edit_file": "#c0392b",
                "list_dir": "#16a085",
                "web_search": "#2980b9",
                "web_fetch": "#1abc9c",
                "message": "#7f8c8d",
                "spawn": "#d35400",
            }
            color = tool_colors.get(tool_name)
            if color is None and tool_name.startswith("exec-"):
                color = tool_colors["exec"]
            if color is None:
                color = "#95a5a6"
        else:
            continue
        gantt_items.append({
            "label": label,
            "color": color,
            "atype": atype,
            "iteration": iteration,
            "x_start": ts_s - t0,
            "x_end": ts_e - t0,
            "duration": ts_e - ts_s,
            "tooltip_extra": tooltip_extra,
        })

    total_span = max((it["x_end"] for it in gantt_items), default=1)

    # ── Build resource chart data ─────────────────────────────────
    res_data: dict[str, list[float]] = {
        "timestamps": [],
        "cpu": [],
        "mem": [],
        "net_rx": [],
        "net_tx": [],
        "disk_r": [],
        "disk_w": [],
        "disk_r_rate": [],
        "disk_w_rate": [],
        "ctx_switches": [],
        "mem_bw_total": [],
        "mem_bw_read": [],
        "mem_bw_write": [],
    }
    prev_dr: float = 0.0
    prev_dw: float = 0.0
    prev_ts: float = 0.0
    first: bool = True
    for s in resource_samples:
        ts_val = s.get("epoch", s.get("timestamp", 0))
        if isinstance(ts_val, str):
            ts_val = _parse_iso(ts_val)
        elif not isinstance(ts_val, (int, float)):
            ts_val = 0.0
        rel_ts = float(ts_val) - t0
        res_data["timestamps"].append(rel_ts)
        res_data["cpu"].append(_safe_float(s.get("cpu_percent", 0)))
        res_data["mem"].append(_parse_mem_mb(str(s.get("mem_usage", ""))))
        # Network: net_rx_bytes / net_tx_bytes → MB
        rx = s.get("net_rx_bytes", 0)
        tx = s.get("net_tx_bytes", 0)
        res_data["net_rx"].append(float(rx) / 1e6 if rx else 0.0)
        res_data["net_tx"].append(float(tx) / 1e6 if tx else 0.0)
        # Disk: absolute MB
        dr = s.get("disk_read_bytes", 0) or 0
        dw = s.get("disk_write_bytes", 0) or 0
        dr_mb = float(dr) / 1e6
        dw_mb = float(dw) / 1e6
        res_data["disk_r"].append(dr_mb)
        res_data["disk_w"].append(dw_mb)
        # Disk rate (MB/s) from deltas
        if not first:
            dt = rel_ts - prev_ts
            dr_rate = (dr_mb - prev_dr) / dt if dt > 0 else 0.0
            dw_rate = (dw_mb - prev_dw) / dt if dt > 0 else 0.0
        else:
            dr_rate = 0.0
            dw_rate = 0.0
            first = False
        res_data["disk_r_rate"].append(dr_rate)
        res_data["disk_w_rate"].append(dw_rate)
        prev_dr = dr_mb
        prev_dw = dw_mb
        prev_ts = rel_ts
        # Context switches
        ctx = s.get("context_switches")
        res_data["ctx_switches"].append(float(ctx) if ctx is not None else 0.0)
        # Host memory bandwidth (MB/s)
        res_data["mem_bw_total"].append(_safe_float(s.get("memory_total_mb_s", 0)))
        res_data["mem_bw_read"].append(_safe_float(s.get("memory_read_mb_s", 0)))
        res_data["mem_bw_write"].append(_safe_float(s.get("memory_write_mb_s", 0)))

    # ── Unified time span (covers both Gantt and resource data) ────
    res_max_ts = max(res_data["timestamps"]) if res_data["timestamps"] else 0.0
    unified_total = max(total_span, res_max_ts, 1.0)

    # ── Memory bandwidth availability ──────────────────────────────
    mem_bw_reason = ""
    mem_bw_all_zero = all(v == 0.0 for v in res_data["mem_bw_total"])
    if mem_bw_all_zero:
        for s in resource_samples:
            reason = s.get("memory_bandwidth_reason", "")
            if reason:
                mem_bw_reason = reason
                break

    # ── Serialize data as JSON for JavaScript ─────────────────────
    gantt_json = json.dumps(gantt_items, ensure_ascii=False)
    res_json = json.dumps(res_data, ensure_ascii=False)

    instance_id = meta.get("instance_id", manifest.get("task", {}).get("instance_id", "?"))
    model = meta.get("model", manifest.get("model", {}).get("name", "?"))
    benchmark = meta.get("benchmark", "?")
    scaffold = meta.get("scaffold", "?")
    total_time = unified_total
    n_iterations = summary.get("n_iterations", len(set(a.get("iteration", 0) for a in actions if a.get("action_type") == "llm_call")))

    return _HTML_TEMPLATE.format(
        instance_id=_esc(instance_id),
        model=_esc(model),
        benchmark=_esc(benchmark),
        scaffold=_esc(scaffold),
        total_time=f"{total_time:.1f}",
        n_iterations=n_iterations,
        n_llm_calls=summary.get("llm_call_time_count", sum(1 for a in actions if a.get("action_type") == "llm_call")),
        n_tool_calls=len([a for a in actions if a.get("action_type") == "tool_exec"]),
        total_tokens=summary.get("total_tokens", 0),
        total_llm_ms=f"{summary.get('total_llm_ms', 0):.0f}",
        total_tool_ms=f"{summary.get('total_tool_ms', 0):.0f}",
        tool_breakdown=json.dumps(summary.get("tool_ms_by_name", {}), ensure_ascii=False),
        resource_summary=json.dumps(resource_summary, ensure_ascii=False),
        resource_count=len(resource_samples),
        gantt_items=gantt_json,
        gantt_total=unified_total,
        res_data=res_json,
        mem_bw_reason=_esc(mem_bw_reason),
        ts_start=_ts_to_str(t0) if t0 else "N/A",
        ts_end=_ts_to_str(t0 + unified_total) if t0 else "N/A",
    )


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trace: {instance_id}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
* {{margin:0;padding:0;box-sizing:border-box}}
body {{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f0f2f5;color:#333}}
.header {{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);color:#fff;padding:18px 24px}}
.header h1 {{font-size:18px;margin-bottom:4px}}
.header .meta {{font-size:12px;opacity:0.75}}
.stats {{display:flex;flex-wrap:wrap;gap:12px;padding:16px 24px;background:#fff;border-bottom:1px solid #e0e0e0}}
.stat-box {{flex:1;min-width:120px;text-align:center;padding:10px;background:#f8f9fa;border-radius:8px}}
.stat-box .val {{font-size:22px;font-weight:700;color:#1a1a2e}}
.stat-box .lbl {{font-size:11px;color:#888;margin-top:2px}}
.section {{margin:16px 24px}}
.section h2 {{font-size:15px;margin-bottom:10px;color:#1a1a2e;border-bottom:2px solid #4a90d9;padding-bottom:4px;display:inline-block}}
.chart-wrap {{background:#fff;border-radius:8px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.08);margin-bottom:16px}}
.chart-wrap canvas {{max-height:300px}}
/* Gantt */
.gantt {{position:relative;background:#fff;border-radius:8px;padding:8px 16px 16px;box-shadow:0 1px 3px rgba(0,0,0,0.08);margin-bottom:16px;overflow-x:auto}}
.gantt-lanes {{position:relative;min-height:200px}}
.gantt-row {{display:flex;align-items:center;height:22px;margin:1px 0;position:relative}}
.gantt-label {{width:130px;min-width:130px;font-size:10px;text-align:right;padding-right:8px;color:#555;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.gantt-track {{flex:1;position:relative;height:100%;background:#f5f5f5;border-radius:3px;overflow:hidden}}
.gantt-bar {{position:absolute;top:2px;height:18px;border-radius:3px;opacity:0.9;min-width:2px}}
.gantt-tick {{position:absolute;top:0;width:1px;height:100%;background:#e0e0e0}}
.gantt-tick-label {{position:absolute;top:100%;font-size:9px;color:#aaa;transform:translateX(-50%);white-space:nowrap;margin-top:2px}}
.gantt-legend {{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px;font-size:10px}}
.gantt-legend span {{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:3px;vertical-align:middle}}
/* Tooltips */
.tooltip {{position:fixed;background:#222;color:#fff;padding:6px 10px;border-radius:4px;font-size:11px;pointer-events:none;z-index:999;display:none;max-width:500px;word-break:break-all}}
.footer {{text-align:center;padding:16px;font-size:11px;color:#aaa}}
</style>
</head>
<body>

<div class="header">
  <h1>🔬 {instance_id}</h1>
  <div class="meta">Model: {model} &nbsp;|&nbsp; Benchmark: {benchmark} &nbsp;|&nbsp; Scaffold: {scaffold} &nbsp;|&nbsp; {ts_start} → {ts_end}</div>
</div>

<div class="stats">
  <div class="stat-box"><div class="val">{total_time}s</div><div class="lbl">Total Time</div></div>
  <div class="stat-box"><div class="val">{n_iterations}</div><div class="lbl">Iterations</div></div>
  <div class="stat-box"><div class="val">{n_llm_calls}</div><div class="lbl">LLM Calls</div></div>
  <div class="stat-box"><div class="val">{n_tool_calls}</div><div class="lbl">Tool Execs</div></div>
  <div class="stat-box"><div class="val">{total_tokens}</div><div class="lbl">Total Tokens</div></div>
  <div class="stat-box"><div class="val">{total_llm_ms}ms</div><div class="lbl">LLM Time</div></div>
  <div class="stat-box"><div class="val">{total_tool_ms}ms</div><div class="lbl">Tool Time</div></div>
</div>

<div class="section" id="gantt-section">
  <h2>⏱ Execution Timeline</h2>
  <div class="gantt" id="gantt"></div>
</div>

<div class="section" id="resources-section">
  <h2>📊 Resource Monitoring</h2>
  <div id="res-charts"></div>
</div>

<div class="tooltip" id="tooltip"></div>
<div class="footer">Generated by trace_collect.html_viz &mdash; agent-test-bench</div>

<script>
// ── Data ──────────────────────────────────────────────────────────
var GANTT = {gantt_items};
var GANTT_TOTAL = {gantt_total};
var RES_DATA = {res_data};
var TOOL_BREAKDOWN = {tool_breakdown};
var RES_SUMMARY = {resource_summary};
var RES_COUNT = {resource_count};
var MEM_BW_REASON = '{mem_bw_reason}';

// ── Gantt Chart ───────────────────────────────────────────────────
(function() {{
    var container = document.getElementById('gantt');
    if (!GANTT.length) {{ container.innerHTML = '<p style="color:#999;padding:16px">No action records found.</p>'; return; }}

    var labels = GANTT.map(function(it) {{ return it.label; }});
    var total = GANTT_TOTAL || 1;

    // Time ticks
    var tickCount = 10;
    var tickHtml = '';
    for (var i = 0; i <= tickCount; i++) {{
        var pct = (i / tickCount) * 100;
        var t = (i / tickCount) * total;
        tickHtml += '<div class="gantt-tick" style="left:' + pct + '%"></div>';
        tickHtml += '<div class="gantt-tick-label" style="left:' + pct + '%">' + t.toFixed(1) + 's</div>';
    }}

    // Legend — group by label (tool name), not color, so that
    // exec-grep, exec-find, exec-cd each get their own entry
    // even though they share the same orange hue.
    var seenLabels = {{}};
    var legendItems = [];
    GANTT.forEach(function(it) {{
        var lbl = it.atype === 'llm_call' ? 'LLM call' : it.label.split(' #')[0];
        if (!seenLabels[lbl]) {{
            seenLabels[lbl] = true;
            legendItems.push({{label: lbl, color: it.color}});
        }}
    }});
    var legendHtml = legendItems.map(function(li) {{
        return '<span style="background:' + li.color + '"></span> ' + li.label;
    }}).join(' ');

    var rowsHtml = GANTT.map(function(it, idx) {{
        var left = (it.x_start / total) * 100;
        var width = Math.max((it.duration / total) * 100, 0.3);
        var info = it.label + (it.tooltip_extra || '') + ' | ' + it.duration.toFixed(2) + 's | t=' +
            it.x_start.toFixed(2) + 's \u2192 ' + it.x_end.toFixed(2) + 's';
        return '<div class="gantt-row">' +
            '<div class="gantt-label" title="' + it.label + '">' + it.label + '</div>' +
            '<div class="gantt-track">' +
                '<div class="gantt-bar" style="left:' + left + '%;width:' + width + '%;background:' + it.color +
                '" data-info="' + info.replace(/"/g, '&quot;') + '" ' +
                'onmouseenter="showTT(event,this)" onmouseleave="hideTT()"></div>' +
            '</div>' +
        '</div>';
    }}).join('');

    container.innerHTML = '<div class="gantt-lanes">' + rowsHtml + '</div>' +
        '<div style="position:relative;height:20px;margin-left:130px">' + tickHtml + '</div>' +
        '<div class="gantt-legend">' + legendHtml + '</div>';
}})();

// ── Tooltip ───────────────────────────────────────────────────────
function showTT(e, el) {{
    var tt = document.getElementById('tooltip');
    tt.textContent = el.getAttribute('data-info');
    tt.style.display = 'block';
    positionTT(e);
}}
function hideTT() {{ document.getElementById('tooltip').style.display = 'none'; }}
function positionTT(e) {{
    var tt = document.getElementById('tooltip');
    var x = e.clientX + 14, y = e.clientY - 10;
    if (x + tt.offsetWidth > window.innerWidth) x = e.clientX - tt.offsetWidth - 8;
    if (y + tt.offsetHeight > window.innerHeight) y = e.clientY - tt.offsetHeight - 8;
    if (x < 0) x = 4;
    if (y < 0) y = 4;
    tt.style.left = x + 'px';
    tt.style.top = y + 'px';
}}
document.addEventListener('mousemove', function(e) {{
    var tt = document.getElementById('tooltip');
    if (tt.style.display === 'block') positionTT(e);
}});

// ── Action Span Overlay Plugin ────────────────────────────────────
// beforeDraw: tool strip (backdrop + bars) drawn first, then LLM bands,
//   then Chart.js renders data on top – strip sits at the bottom layer.
// afterDraw:  restores chart area, no drawing.
Chart.register({{
    id: 'traceActionSpans',
    beforeDraw: function(chart) {{
        if (!chart.scales.x || !GANTT.length) return;
        var ctx = chart.ctx;
        var xAxis = chart.scales.x;
        var ca = chart.chartArea;

        // ── Collect tool-exec actions & assign lanes ──────────────
        var tools = [];
        GANTT.forEach(function(s) {{
            if (s.atype === 'tool_exec') tools.push({{
                x_start: s.x_start, x_end: s.x_end,
                color: s.color, label: s.label
            }});
        }});
        tools.sort(function(a, b) {{ return a.x_start - b.x_start; }});
        var lanes = [];
        tools.forEach(function(t) {{
            var lane = 0;
            while (lane < lanes.length && lanes[lane] > t.x_start) lane++;
            t._lane = lane;
            if (lane >= lanes.length) lanes.push(t.x_end);
            else lanes[lane] = t.x_end;
        }});

        var numLanes = Math.max(lanes.length, tools.length ? 1 : 0);
        var barH = Math.min(10, Math.max(5, 34 / Math.max(numLanes, 1)));
        var gap = 2, pad = 3;
        var stripH = numLanes * (barH + gap) + pad * 2;

        // Store original bottom for afterDraw restore
        var origBottom = ca.bottom;
        chart._ts_origBottom = origBottom;

        // ── Shrink chart area so data curves stay above the strip ─
        ca.bottom -= stripH;

        // ── 1) Tool strip (no backdrop – just separator + bars) ────
        var stripTop = origBottom - stripH;

        // Separator line
        ctx.strokeStyle = '#cccccc';
        ctx.lineWidth = 1;
        ctx.globalAlpha = 0.6;
        ctx.beginPath();
        ctx.moveTo(ca.left, stripTop);
        ctx.lineTo(ca.right, stripTop);
        ctx.stroke();

        // Tool bars (min 2 px so short calls never vanish)
        ctx.save();
        tools.forEach(function(t) {{
            var x1 = xAxis.getPixelForValue(t.x_start);
            var x2 = xAxis.getPixelForValue(t.x_end);
            if (x2 < ca.left || x1 > ca.right) return;
            x1 = Math.max(x1, ca.left + 1);
            x2 = Math.min(x2, ca.right - 1);
            var w = Math.max(x2 - x1, 2);
            var y = stripTop + pad + t._lane * (barH + gap);
            ctx.fillStyle = t.color;
            ctx.globalAlpha = 0.88;
            ctx.fillRect(x1, y, w, barH);
        }});
        ctx.restore();

        // ── 2) LLM iteration bands (on top of strip, under data) ──
        GANTT.forEach(function(span) {{
            if (span.atype !== 'llm_call') return;
            var x1 = xAxis.getPixelForValue(span.x_start);
            var x2 = xAxis.getPixelForValue(span.x_end);
            if (x2 < ca.left || x1 > ca.right) return;
            x1 = Math.max(x1, ca.left);
            x2 = Math.min(x2, ca.right);
            var w = x2 - x1;
            if (w < 1) return;
            ctx.save();
            ctx.fillStyle = span.color;
            ctx.globalAlpha = 0.07;
            ctx.fillRect(x1, ca.top, w, ca.bottom - ca.top);
            ctx.restore();
        }});
    }},
    afterDraw: function(chart) {{
        // Restore original chart area
        if (chart._ts_origBottom !== undefined) {{
            chart.chartArea.bottom = chart._ts_origBottom;
        }}
        delete chart._ts_origBottom;
    }}
}});

// ── Resource Charts ───────────────────────────────────────────────
(function() {{
    var parent = document.getElementById('res-charts');
    if (!RES_COUNT) {{
        parent.innerHTML = '<div class="chart-wrap"><p style="color:#999">No resource samples collected (container stats sampling may not have been active).</p></div>' +
            '<div class="chart-wrap"><h3 style="font-size:13px;margin-bottom:8px">🔧 Tool Time Breakdown (ms)</h3><div id="tool-pie-container"><canvas id="chart-tool-pie"></canvas></div></div>';
    }} else {{
        parent.innerHTML =
            '<div class="chart-wrap"><h3 style="font-size:13px;margin-bottom:8px">🖥 CPU & Memory</h3><div style="height:280px"><canvas id="chart-cpu-mem"></canvas></div></div>' +
            '<div class="chart-wrap"><h3 style="font-size:13px;margin-bottom:8px">🧠 Memory Bandwidth (host)</h3><div style="height:240px"><canvas id="chart-mem-bw"></canvas></div></div>' +
            '<div class="chart-wrap"><h3 style="font-size:13px;margin-bottom:8px">🌐 Network I/O (cumulative)</h3><div style="height:240px"><canvas id="chart-net"></canvas></div></div>' +
            '<div class="chart-wrap"><h3 style="font-size:13px;margin-bottom:8px">💾 Disk I/O (rate)</h3><div style="height:240px"><canvas id="chart-disk"></canvas></div></div>' +
            '<div class="chart-wrap"><h3 style="font-size:13px;margin-bottom:8px">⚡ Context Switches (cumulative)</h3><div style="height:200px"><canvas id="chart-ctx"></canvas></div></div>' +
            '<div class="chart-wrap"><h3 style="font-size:13px;margin-bottom:8px">🔧 Tool Time Breakdown (ms)</h3><div id="tool-pie-container"><canvas id="chart-tool-pie"></canvas></div></div>';
    }}

    // CPU + Memory
    if (RES_COUNT) {{
        new Chart(document.getElementById('chart-cpu-mem'), {{
            type: 'line',
            data: {{
                labels: RES_DATA.timestamps.map(function(t) {{ return t.toFixed(1); }}),
                datasets: [
                    {{ label:'CPU % (all cores)', data:RES_DATA.cpu, borderColor:'#e74c3c', backgroundColor:'rgba(231,76,60,0.1)', fill:true, tension:0, yAxisID:'y', borderWidth:1.5, pointBackgroundColor:'#fff', pointRadius:1.5, pointHoverRadius:6, pointHitRadius:10, pointStyle:'circle' }},
                    {{ label:'Memory (MB)', data:RES_DATA.mem, borderColor:'#3498db', backgroundColor:'rgba(52,152,219,0.1)', fill:true, tension:0, yAxisID:'y1', borderWidth:1.5, pointBackgroundColor:'#fff', pointRadius:1.5, pointHoverRadius:6, pointHitRadius:10, pointStyle:'circle' }}
                ]
            }},
            options: {{
                responsive:true, maintainAspectRatio:false,
                interaction:{{mode:'index',intersect:false}},
                scales: {{
                    y:{{ type:'linear', position:'left', title:{{display:true,text:'CPU % (all cores)'}}, min:0 }},
                    y1:{{ type:'linear', position:'right', title:{{display:true,text:'Memory MB'}}, min:0, grid:{{drawOnChartArea:false}} }},
                    x:{{ type:'linear', title:{{display:true,text:'Time (s)'}} , min:0, max:GANTT_TOTAL }}
                }}
            }}
        }});

        // Memory Bandwidth (host)
        var allZero = RES_DATA.mem_bw_total.every(function(v) {{ return v === 0; }});
        if (MEM_BW_REASON && allZero) {{
            document.getElementById('chart-mem-bw').parentNode.innerHTML =
                '<p style="color:#c0392b;font-size:12px;padding:20px 0;text-align:center">'
                + '⚠ Memory bandwidth unavailable: <code>' + MEM_BW_REASON + '</code>'
                + '<br><span style="color:#888">(requires Intel IMC PMU: check <code>ls /sys/bus/event_source/devices/ | grep uncore</code>)</span></p>';
        }} else {{
        new Chart(document.getElementById('chart-mem-bw'), {{
            type: 'line',
            data: {{
                labels: RES_DATA.timestamps.map(function(t) {{ return t.toFixed(1); }}),
                datasets: [
                    {{ label:'Total (MB/s)', data:RES_DATA.mem_bw_total, borderColor:'#8e44ad', tension:0, borderWidth:1.5, pointBackgroundColor:'#fff', pointRadius:1.5, pointHoverRadius:6, pointHitRadius:10, pointStyle:'circle' }},
                    {{ label:'Read (MB/s)', data:RES_DATA.mem_bw_read, borderColor:'#2980b9', tension:0, borderWidth:1.5, pointBackgroundColor:'#fff', pointRadius:1.5, pointHoverRadius:6, pointHitRadius:10, pointStyle:'circle' }},
                    {{ label:'Write (MB/s)', data:RES_DATA.mem_bw_write, borderColor:'#c0392b', tension:0, borderWidth:1.5, pointBackgroundColor:'#fff', pointRadius:1.5, pointHoverRadius:6, pointHitRadius:10, pointStyle:'circle' }}
                ]
            }},
            options: {{
                responsive:true, maintainAspectRatio:false,
                interaction:{{mode:'index',intersect:false}},
                scales: {{
                    y:{{ title:{{display:true,text:'MB/s'}}, min:0 }},
                    x:{{ type:'linear', title:{{display:true,text:'Time (s)'}} , min:0, max:GANTT_TOTAL }}
                }}
            }}
        }});
        }}  // end else (mem_bw available)

        // Network
        new Chart(document.getElementById('chart-net'), {{
            type: 'line',
            data: {{
                labels: RES_DATA.timestamps.map(function(t) {{ return t.toFixed(1); }}),
                datasets: [
                    {{ label:'RX (MB)', data:RES_DATA.net_rx, borderColor:'#2ecc71', tension:0, borderWidth:1.5, pointBackgroundColor:'#fff', pointRadius:1.5, pointHoverRadius:6, pointHitRadius:10, pointStyle:'circle' }},
                    {{ label:'TX (MB)', data:RES_DATA.net_tx, borderColor:'#9b59b6', tension:0, borderWidth:1.5, pointBackgroundColor:'#fff', pointRadius:1.5, pointHoverRadius:6, pointHitRadius:10, pointStyle:'circle' }}
                ]
            }},
            options: {{
                responsive:true, maintainAspectRatio:false,
                interaction:{{mode:'index',intersect:false}},
                scales: {{
                    y:{{ title:{{display:true,text:'MB'}}, min:0 }},
                    x:{{ type:'linear', title:{{display:true,text:'Time (s)'}} , min:0, max:GANTT_TOTAL }}
                }}
            }}
        }});

        // Disk (rate MB/s)
        new Chart(document.getElementById('chart-disk'), {{
            type: 'line',
            data: {{
                labels: RES_DATA.timestamps.map(function(t) {{ return t.toFixed(1); }}),
                datasets: [
                    {{ label:'Read (MB/s)', data:RES_DATA.disk_r_rate, borderColor:'#2ecc71', tension:0, borderWidth:1.5, pointBackgroundColor:'#fff', pointRadius:1.5, pointHoverRadius:6, pointHitRadius:10, pointStyle:'circle' }},
                    {{ label:'Write (MB/s)', data:RES_DATA.disk_w_rate, borderColor:'#e67e22', tension:0, borderWidth:1.5, pointBackgroundColor:'#fff', pointRadius:1.5, pointHoverRadius:6, pointHitRadius:10, pointStyle:'circle' }}
                ]
            }},
            options: {{
                responsive:true, maintainAspectRatio:false,
                interaction:{{mode:'index',intersect:false}},
                scales: {{
                    y:{{ title:{{display:true,text:'MB/s'}}, min:0 }},
                    x:{{ type:'linear', title:{{display:true,text:'Time (s)'}} , min:0, max:GANTT_TOTAL }}
                }}
            }}
        }});

        // Context Switches
        new Chart(document.getElementById('chart-ctx'), {{
            type: 'line',
            data: {{
                labels: RES_DATA.timestamps.map(function(t) {{ return t.toFixed(1); }}),
                datasets: [
                    {{ label:'Context Switches', data:RES_DATA.ctx_switches, borderColor:'#1abc9c', backgroundColor:'rgba(26,188,156,0.1)', fill:true, tension:0, borderWidth:1.5, pointBackgroundColor:'#fff', pointRadius:1.5, pointHoverRadius:6, pointHitRadius:10, pointStyle:'circle' }}
                ]
            }},
            options: {{
                responsive:true, maintainAspectRatio:false,
                interaction:{{mode:'index',intersect:false}},
                scales: {{
                    y:{{ title:{{display:true,text:'Count'}}, min:0 }},
                    x:{{ type:'linear', title:{{display:true,text:'Time (s)'}} , min:0, max:GANTT_TOTAL }}
                }}
            }}
        }});
    }}

    // Tool Time Pie
    var toolNames = Object.keys(TOOL_BREAKDOWN);
    if (toolNames.length) {{
        var toolColors = {{exec:'#e67e22',read_file:'#27ae60',write_file:'#8e44ad',edit_file:'#c0392b',list_dir:'#16a085',web_search:'#2980b9',web_fetch:'#1abc9c',message:'#7f8c8d',spawn:'#d35400'}};
        var bgColors = toolNames.map(function(n) {{
            if (toolColors[n]) return toolColors[n];
            if (n.startsWith('exec-')) return toolColors['exec'];
            if (n.startsWith('mcp_')) return '#8e44ad';
            return '#95a5a6';
        }});
        new Chart(document.getElementById('chart-tool-pie'), {{
            type: 'doughnut',
            data: {{
                labels: toolNames,
                datasets: [{{ data:toolNames.map(function(n){{return TOOL_BREAKDOWN[n]}}), backgroundColor:bgColors }}]
            }},
            options: {{
                responsive: true, maintainAspectRatio: true,
                plugins: {{ legend: {{ position:'right', labels:{{font:{{size:11}}}} }} }}
            }}
        }});
    }} else {{
        document.getElementById('tool-pie-container').innerHTML = '<p style="color:#999">No tool timing data.</p>';
    }}
}})();
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HTML visualization from a trace attempt directory.")
    parser.add_argument("attempt_dir", type=str, help="Path to the attempt_<N> directory containing trace.jsonl, resources.json, run_manifest.json")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output HTML file path (default: <attempt_dir>/trace_viz.html)")
    args = parser.parse_args()

    attempt_dir = Path(args.attempt_dir).resolve()
    if not attempt_dir.is_dir():
        print(f"ERROR: {attempt_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    trace_file = attempt_dir / "trace.jsonl"
    if not trace_file.exists():
        print(f"ERROR: {trace_file} not found", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else (attempt_dir / "trace_viz.html")
    print(f"Generating: {output_path}")
    html = generate_html(attempt_dir)
    output_path.write_text(html, encoding="utf-8")
    print(f"Done → {output_path}  ({len(html)} bytes)")


if __name__ == "__main__":
    main()
