#!/usr/bin/env python3
"""Generate a self-contained HTML visualization from system_resources.jsonl.

Reads the JSONL file produced by ``scripts/system_resource_monitor.py`` and
writes an interactive HTML page with Chart.js time-series charts for CPU,
memory, load, network I/O, disk I/O, and container count.  Metric
descriptions are included inline so every chart is self-documenting.

Optionally accepts ``--timeline agent_timeline.jsonl`` to add an agent
throughput & latency summary section.

Usage::

    python scripts/plot_system_resources.py \\
        --input traces/simulate/swe-rebench/sweep_320a_1cpu/system_resources.jsonl \\
        --timeline traces/simulate/swe-rebench/sweep_320a_1cpu/agent_timeline.jsonl \\
        --output traces/simulate/swe-rebench/sweep_320a_1cpu/system_viz.html

Dependencies: none beyond the Python standard library.  Chart.js is loaded
from CDN in the generated HTML.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_samples(input_path: Path) -> list[dict]:
    """Read the JSONL file and return a list of sample dicts."""
    samples: list[dict] = []
    with input_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return samples


def _load_timeline(timeline_path: Path) -> list[dict] | None:
    """Read agent_timeline.jsonl, or return None if unavailable."""
    if not timeline_path.is_file():
        return None
    records: list[dict] = []
    with timeline_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records if records else None


def _percentile(data: list[float], p: float) -> float:
    """Compute the *p*-th percentile of *data* (linear interpolation)."""
    if not data:
        return 0.0
    s = sorted(data)
    k = (p / 100.0) * (len(s) - 1)
    f = int(k)
    c = k - f
    if f + 1 < len(s):
        return s[f] + c * (s[f + 1] - s[f])
    return s[f]


def _build_dataset(
    samples: list[dict],
    key: str,
    *,
    label: str,
    color: str,
    y_axis_id: str = "y",
    hidden: bool = False,
) -> str:
    """Build a Chart.js dataset JSON fragment for a single metric."""
    values = []
    for s in samples:
        v = s.get(key)
        values.append(v if v is not None else "null")

    return json.dumps({
        "label": label,
        "data": values,
        "borderColor": color,
        "backgroundColor": color + "20",
        "borderWidth": 1.5,
        "pointRadius": 0,
        "fill": False,
        "tension": 0.1,
        "yAxisID": y_axis_id,
        "hidden": hidden,
    })


def generate_html(samples: list[dict], timeline: list[dict] | None = None) -> str:
    """Build a self-contained HTML page with resource charts and optional agent summary."""
    if not samples:
        return "<html><body><h2>No samples found</h2></body></html>"

    t0 = samples[0]["ts"]
    timestamps = [s["ts"] - t0 for s in samples]  # relative seconds
    ts_labels = json.dumps([f"{t:.0f}s" for t in timestamps])

    n = len(samples)

    # Summary stats from system monitor
    cpu_vals = [s.get("cpu_percent", 0) or 0 for s in samples]
    mem_vals = [s.get("mem_used_gb", 0) or 0 for s in samples]
    container_vals = [s.get("container_count") for s in samples]
    max_containers = max((v for v in container_vals if v is not None), default=0)
    load_vals = [s.get("load_1m", 0) or 0 for s in samples]
    load_max = max(load_vals) if load_vals else 0
    cpu_count = samples[0].get("cpu_count", 0) or 1

    cpu_avg = sum(cpu_vals) / n if n else 0
    cpu_max = max(cpu_vals) if cpu_vals else 0
    mem_avg = sum(mem_vals) / n if n else 0
    mem_max = max(mem_vals) if mem_vals else 0

    # ── Agent timeline summary (if available) ─────────────────────
    agent_summary_html = ""
    if timeline:
        valid = [r for r in timeline if r.get("start_ts") is not None]
        if valid:
            elapsed_list = [float(r["elapsed_s"]) for r in valid]
            start_all = min(float(r["start_ts"]) for r in valid)
            end_all = max(float(r["end_ts"]) for r in valid)
            wall_s = end_all - start_all
            wall_min = wall_s / 60.0
            total_agents = len(valid)
            aps = total_agents / wall_s if wall_s > 0 else 0
            apm = aps * 60
            total_actions = sum(int(r.get("n_actions", 0)) for r in valid)
            total_llm = sum(int(r.get("n_llm_calls", 0)) for r in valid)
            total_tool = sum(int(r.get("n_tool_execs", 0)) for r in valid)

            agent_summary_html = (
                f'<div class="section-title">Throughput &amp; Agent Summary</div>'
                f'<div class="stats">'
                f'<div class="stat-box">'
                f'<div class="value" style="color:#E91E63">{aps:.3f}</div>'
                f'<div class="label">Throughput (agents/s)</div></div>'
                f'<div class="stat-box">'
                f'<div class="value" style="color:#E91E63">{apm:.1f}</div>'
                f'<div class="label">Throughput (agents/min)</div></div>'
                f'<div class="stat-box">'
                f'<div class="value" style="color:#FF9800">{wall_s:.0f}s</div>'
                f'<div class="label">Wall Time ({wall_min:.1f} min)</div></div>'
                f'<div class="stat-box">'
                f'<div class="value" style="color:#9C27B0">{total_agents}</div>'
                f'<div class="label">Total Agents</div></div>'
                f'</div>'
                f'<div class="stats">'
                f'<div class="stat-box">'
                f'<div class="value" style="color:#4CAF50">{sum(elapsed_list)/len(elapsed_list):.1f}s</div>'
                f'<div class="label">Agent Elapsed (mean)</div></div>'
                f'<div class="stat-box">'
                f'<div class="value" style="color:#4CAF50">{_percentile(elapsed_list, 50):.1f}s</div>'
                f'<div class="label">Agent Elapsed (p50)</div></div>'
                f'<div class="stat-box">'
                f'<div class="value" style="color:#FF5722">{_percentile(elapsed_list, 95):.1f}s</div>'
                f'<div class="label">Agent Elapsed (p95)</div></div>'
                f'<div class="stat-box">'
                f'<div class="value" style="color:#FF5722">{_percentile(elapsed_list, 99):.1f}s</div>'
                f'<div class="label">Agent Elapsed (p99)</div></div>'
                f'</div>'
                f'<div class="stats">'
                f'<div class="stat-box">'
                f'<div class="value">{total_actions}</div>'
                f'<div class="label">Total Actions (llm={total_llm}, tool={total_tool})</div></div>'
                f'<div class="stat-box">'
                f'<div class="value">{max(elapsed_list):.1f}s</div>'
                f'<div class="label">Agent Elapsed (max)</div></div>'
                f'<div class="stat-box">'
                f'<div class="value" style="color:#795548">{load_max:.1f}</div>'
                f'<div class="label">Load 1m Peak (cpu_count={cpu_count})</div></div>'
                f'<div class="stat-box"></div>'
                f'</div>'
            )

    datasets = []

    # ── CPU % ──────────────────────────────────────────────────────────
    datasets.append(_build_dataset(samples, "cpu_percent", label="CPU %", color="#2196F3"))

    # ── Load averages ──────────────────────────────────────────────────
    for key, label, color in [
        ("load_1m", "Load 1m", "#FF9800"),
        ("load_5m", "Load 5m", "#FF5722"),
        ("load_15m", "Load 15m", "#795548"),
    ]:
        datasets.append(_build_dataset(samples, key, label=label, color=color,
                                        y_axis_id="y_load", hidden=(key != "load_1m")))

    # ── Memory ─────────────────────────────────────────────────────────
    datasets.append(_build_dataset(samples, "mem_used_gb", label="Mem Used (GB)", color="#4CAF50"))
    datasets.append(_build_dataset(samples, "mem_total_gb", label="Mem Total (GB)", color="#9E9E9E",
                                    y_axis_id="y_mem", hidden=False))

    # ── Container count ────────────────────────────────────────────────
    datasets.append(_build_dataset(samples, "container_count", label="Containers", color="#9C27B0"))

    # ── Network I/O rate (computed as delta) ───────────────────────────
    net_rx_rate, net_tx_rate = _compute_rate(samples, "net_rx_mb", "net_tx_mb")
    datasets.append(json.dumps({
        "label": "Net RX (MB/s)",
        "data": net_rx_rate,
        "borderColor": "#00BCD4",
        "backgroundColor": "#00BCD420",
        "borderWidth": 1.5,
        "pointRadius": 0,
        "fill": False,
        "tension": 0.1,
        "yAxisID": "y_net",
    }))
    datasets.append(json.dumps({
        "label": "Net TX (MB/s)",
        "data": net_tx_rate,
        "borderColor": "#0097A7",
        "backgroundColor": "#0097A720",
        "borderWidth": 1.5,
        "pointRadius": 0,
        "fill": False,
        "tension": 0.1,
        "yAxisID": "y_net",
    }))

    # ── Disk I/O rate ──────────────────────────────────────────────────
    disk_r_rate, disk_w_rate = _compute_rate(samples, "disk_read_mb", "disk_write_mb")
    datasets.append(json.dumps({
        "label": "Disk Read (MB/s)",
        "data": disk_r_rate,
        "borderColor": "#E91E63",
        "backgroundColor": "#E91E6320",
        "borderWidth": 1.5,
        "pointRadius": 0,
        "fill": False,
        "tension": 0.1,
        "yAxisID": "y_disk",
    }))
    datasets.append(json.dumps({
        "label": "Disk Write (MB/s)",
        "data": disk_w_rate,
        "borderColor": "#C2185B",
        "backgroundColor": "#C2185B20",
        "borderWidth": 1.5,
        "pointRadius": 0,
        "fill": False,
        "tension": 0.1,
        "yAxisID": "y_disk",
    }))

    datasets_js = ",\n        ".join(datasets)

    duration_s = f"{timestamps[-1]:.0f}" if timestamps else "0"

    return _HTML_TEMPLATE.format(
        title="System Resource Timeline",
        ts_labels=ts_labels,
        datasets=datasets_js,
        cpu_avg=f"{cpu_avg:.1f}",
        cpu_max=f"{cpu_max:.1f}",
        mem_avg=f"{mem_avg:.1f}",
        mem_max=f"{mem_max:.1f}",
        max_containers=str(max_containers),
        total_samples=str(n),
        duration_s=duration_s,
        cpu_count=str(cpu_count),
        load_max=f"{load_max:.1f}",
        agent_summary_section=agent_summary_html,
    )


def _compute_rate(
    samples: list[dict], key_rx: str, key_tx: str
) -> tuple[list, list]:
    """Compute per-second rate from cumulative counters."""
    rate_rx: list = []
    rate_tx: list = []
    prev_rx = None
    prev_tx = None
    prev_ts = None
    for s in samples:
        rx = s.get(key_rx)
        tx = s.get(key_tx)
        ts = s.get("ts")
        if (prev_rx is not None and prev_tx is not None and prev_ts is not None
                and rx is not None and tx is not None):
            dt = ts - prev_ts
            if dt > 0:
                rate_rx.append(round(max(0, rx - prev_rx) / dt, 3))
                rate_tx.append(round(max(0, tx - prev_tx) / dt, 3))
            else:
                rate_rx.append(0.0)
                rate_tx.append(0.0)
        else:
            rate_rx.append("null")
            rate_tx.append("null")
        prev_rx, prev_tx, prev_ts = rx, tx, ts
    return rate_rx, rate_tx


_HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js">
</script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #f5f5f5; color: #333; padding: 20px; }}
  .header {{ background: linear-gradient(135deg, #1a237e, #283593); color: #fff;
             padding: 20px 28px; border-radius: 10px; margin-bottom: 24px; }}
  .header h1 {{ font-size: 22px; font-weight: 600; }}
  .header .meta {{ font-size: 13px; opacity: 0.8; margin-top: 6px; }}
  .section-title {{ font-size: 16px; font-weight: 600; color: #555; margin-bottom: 12px;
                     padding-bottom: 6px; border-bottom: 2px solid #e0e0e0; }}
  .stats {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }}
  .stat-box {{ background: #fff; border-radius: 8px; padding: 14px 20px; flex: 1;
               min-width: 130px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .stat-box .value {{ font-size: 28px; font-weight: 700; }}
  .stat-box .label {{ font-size: 12px; color: #888; margin-top: 2px; }}
  .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
  .chart-row.full {{ grid-template-columns: 1fr; }}
  .chart-card {{ background: #fff; border-radius: 8px; padding: 16px;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .chart-card h3 {{ font-size: 14px; font-weight: 600; margin-bottom: 6px; color: #555; }}
  .chart-card canvas {{ width: 100%; max-height: 280px; }}
  .info-note {{ font-size: 11px; color: #999; line-height: 1.5; margin-top: 8px;
                padding: 6px 10px; background: #fafafa; border-left: 3px solid #e0e0e0;
                border-radius: 3px; }}
  .info-note code {{ background: #eee; padding: 1px 4px; border-radius: 2px; font-size: 10px; }}
  @media (max-width: 900px) {{ .chart-row {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<div class="header">
  <h1>{title}</h1>
  <div class="meta">
    Samples: {total_samples} &middot; Monitor Duration: {duration_s}s
    &middot; Max Containers: {max_containers}
  </div>
</div>

{agent_summary_section}

<div class="section-title">System Resource Metrics</div>
<div class="stats">
  <div class="stat-box">
    <div class="value" style="color:#2196F3">{cpu_avg}%</div>
    <div class="label">CPU Avg (max {cpu_max}%)</div>
  </div>
  <div class="stat-box">
    <div class="value" style="color:#4CAF50">{mem_avg} GB</div>
    <div class="label">Mem Avg (max {mem_max} GB)</div>
  </div>
  <div class="stat-box">
    <div class="value" style="color:#9C27B0">{max_containers}</div>
    <div class="label">Max Containers</div>
  </div>
  <div class="stat-box">
    <div class="value" style="color:#FF9800">{duration_s}s</div>
    <div class="label">Monitor Duration</div>
  </div>
</div>

<div class="chart-row">
  <div class="chart-card">
    <h3>CPU Utilization &amp; System Load</h3>
    <canvas id="chart_cpu"></canvas>
    <div class="info-note">
      <strong>CPU %</strong> (blue, left axis): Whole-host utilization &mdash; 100% = all cores saturated.<br>
      <strong>Load 1m/5m/15m</strong> (right axis): Linux load average &mdash;
      exponentially-weighted count of runnable + I/O-waiting processes.
      <strong>load &asymp; cpu_count ({cpu_count})</strong> = saturated, no queueing;
      <strong>load &gt;&gt; cpu_count</strong> = severe contention. Peak load 1m: <strong>{load_max}</strong>.
      Toggle 5m / 15m in the legend.
    </div>
  </div>
  <div class="chart-card">
    <h3>Memory Usage</h3>
    <canvas id="chart_mem"></canvas>
    <div class="info-note">
      <strong>Mem Used</strong> (green): <code>psutil.virtual_memory().used</code> &mdash;
      includes OS page cache, not just process RSS. High t=0 values reflect
      pre-existing host memory usage. Docker CoW image sharing means N containers
      add relatively little incremental memory.<br>
      <strong>Mem Total</strong> (grey): physical RAM installed.
    </div>
  </div>
</div>

<div class="chart-row">
  <div class="chart-card">
    <h3>Container Count</h3>
    <canvas id="chart_containers"></canvas>
    <div class="info-note">
      Running Docker containers (<code>docker ps -q</code>).  Ramp-up is
      rate-limited by <code>asyncio.Semaphore(20)</code>.  Steady state = all
      agents replaying concurrently.  Tear-down aligns with replay completion.
    </div>
  </div>
  <div class="chart-card">
    <h3>Network I/O Rate</h3>
    <canvas id="chart_net"></canvas>
    <div class="info-note">
      Per-second delta from <code>/proc/net/dev</code> cumulative counters,
      all interfaces.  In <code>cloud_model</code> (no real LLM API calls),
      traffic comes from Docker daemon communication (<code>docker exec</code>
      stdout/stderr streaming), image pulls, SSH, and NFS if outputs are on
      network mounts.  ~20 MB/s RX is expected with 320 concurrent containers.
    </div>
  </div>
</div>

<div class="chart-row full">
  <div class="chart-card">
    <h3>Disk I/O Rate</h3>
    <canvas id="chart_disk"></canvas>
    <div class="info-note">
      Per-second delta from <code>/proc/diskstats</code> cumulative counters.
      Peaks typically align with container startup (image layer reads) and
      teardown (log flushing).
    </div>
  </div>
</div>

<div class="info-note" style="margin-top: 20px; font-size: 12px;">
  <strong>Measurement:</strong> All system metrics via <code>psutil</code> at 1 Hz
  (same as <code>top</code>/<code>htop</code>). <strong>Monitor Duration</strong>
  = last sample &minus; first sample &mdash; includes ~1s of monitor warm-up
  before simulate starts and the trace-split/teardown tail after it exits.
  For throughput calculations use the <strong>Wall Time</strong> reported in the
  agent summary above (derived from per-agent <code>trace.jsonl</code> timestamps).
</div>

<script>
const labels = {ts_labels};
const allDatasets = [{datasets}];

function makeChart(canvasId, datasetIndices, scales) {{
  const ctx = document.getElementById(canvasId).getContext('2d');
  return new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: labels,
      datasets: datasetIndices.map(i => allDatasets[i]),
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{ legend: {{ position: 'top', labels: {{ boxWidth: 12, padding: 12, font: {{ size: 11 }} }} }} }},
      scales: scales,
    }},
  }});
}}

// CPU + Load (indices 0-3)
makeChart('chart_cpu', [0, 1, 2, 3], {{
  x: {{ display: true, title: {{ text: 'Elapsed', display: true }} }},
  y: {{ type: 'linear', position: 'left', title: {{ text: 'CPU %', display: true }}, min: 0, max: 100 }},
  y_load: {{ type: 'linear', position: 'right', title: {{ text: 'Load Avg', display: true }}, min: 0, grid: {{ drawOnChartArea: false }} }},
}});

// Memory (indices 4-5)
makeChart('chart_mem', [4, 5], {{
  x: {{ display: true, title: {{ text: 'Elapsed', display: true }} }},
  y: {{ type: 'linear', position: 'left', title: {{ text: 'GB', display: true }}, min: 0 }},
  y_mem: {{ type: 'linear', position: 'left', title: {{ text: 'GB', display: true }}, min: 0 }},
}});

// Container count (index 6)
makeChart('chart_containers', [6], {{
  x: {{ display: true, title: {{ text: 'Elapsed', display: true }} }},
  y: {{ type: 'linear', position: 'left', title: {{ text: 'Containers', display: true }}, min: 0, ticks: {{ stepSize: 1 }} }},
}});

// Network I/O rate (indices 7-8)
makeChart('chart_net', [7, 8], {{
  x: {{ display: true, title: {{ text: 'Elapsed', display: true }} }},
  y_net: {{ type: 'linear', position: 'left', title: {{ text: 'MB/s', display: true }}, min: 0 }},
}});

// Disk I/O rate (indices 9-10)
makeChart('chart_disk', [9, 10], {{
  x: {{ display: true, title: {{ text: 'Elapsed', display: true }} }},
  y_disk: {{ type: 'linear', position: 'left', title: {{ text: 'MB/s', display: true }}, min: 0 }},
}});
</script>

</body>
</html>'''


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate HTML visualization from system_resources.jsonl.",
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to system_resources.jsonl file.",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Path to write the output HTML file.",
    )
    parser.add_argument(
        "--timeline", type=Path, default=None,
        help="Optional path to agent_timeline.jsonl for throughput summary.",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"ERROR: --input not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    samples = _load_samples(args.input)
    if not samples:
        print(f"WARNING: {args.input} is empty, generating placeholder.", file=sys.stderr)

    timeline = _load_timeline(args.timeline) if args.timeline else None

    html = generate_html(samples, timeline=timeline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Written: {args.output} ({len(samples)} samples, {len(html)} bytes)")


if __name__ == "__main__":
    main()
