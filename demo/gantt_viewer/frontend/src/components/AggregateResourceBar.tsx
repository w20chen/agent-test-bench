import { createEffect, createMemo, createSignal, Show } from "solid-js";

import type { ResourceSample, TracePayload } from "../api/client";
import { RESOURCE_METRIC_COLORS, findNearestSample } from "../canvas/hit";
import { resourceMetricUnit, resourceMetricValues } from "../canvas/resourceMetrics";
import { displayColor } from "../theme/displayColor";
import { canvasTimeRange } from "../state/signals";
import type { ClockMode, ResourceMetric, ThemeMode, TimeMode } from "../state/signals";

interface AggregateResourceBarProps {
  traces: TracePayload[];
  visibility: Record<string, boolean>;
  resourceMetric: ResourceMetric;
  resourceMetricSecondary: ResourceMetric;
  showResourceChart: boolean;
  timeMode: TimeMode;
  clockMode: ClockMode;
  themeMode: ThemeMode;
}

const BAR_HEIGHT = 80;
const CHART_PAD = 4;

function selectTime(sample: ResourceSample, timeMode: TimeMode, clockMode: ClockMode): number {
  if (clockMode === "real") {
    const realT = timeMode === "sync" ? sample.t_real : sample.t_real_abs;
    if (typeof realT === "number" && Number.isFinite(realT)) return realT;
  }
  return timeMode === "sync" ? sample.t : sample.t_abs;
}

function selectSpanStart(
  span: TracePayload["lanes"][number]["spans"][number],
  timeMode: TimeMode,
  clockMode: ClockMode,
): number {
  if (clockMode === "real") {
    const realStart = timeMode === "sync" ? span.start_real : span.start_real_abs;
    if (typeof realStart === "number" && Number.isFinite(realStart)) return realStart;
  }
  return timeMode === "sync" ? span.start : span.start_abs;
}

function selectSpanEnd(
  span: TracePayload["lanes"][number]["spans"][number],
  timeMode: TimeMode,
  clockMode: ClockMode,
): number {
  if (clockMode === "real") {
    const realEnd = timeMode === "sync" ? span.end_real : span.end_real_abs;
    if (typeof realEnd === "number" && Number.isFinite(realEnd)) return realEnd;
  }
  return timeMode === "sync" ? span.end : span.end_abs;
}

function traceSpanBounds(
  trace: TracePayload,
  timeMode: TimeMode,
  clockMode: ClockMode,
): { min: number; max: number } | null {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (const lane of trace.lanes) {
    for (const span of lane.spans) {
      min = Math.min(min, selectSpanStart(span, timeMode, clockMode));
      max = Math.max(max, selectSpanEnd(span, timeMode, clockMode));
    }
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) return null;
  return { min, max };
}

/** Binary-search the nearest sample, then linearly interpolate. */
function interpolateSample(
  timeline: ResourceSample[],
  times: number[],
  targetT: number,
): ResourceSample {
  if (targetT <= times[0]) return timeline[0];
  if (targetT >= times[times.length - 1]) return timeline[timeline.length - 1];

  let lo = 0;
  let hi = times.length - 1;
  while (lo < hi - 1) {
    const mid = (lo + hi) >> 1;
    if (times[mid] <= targetT) lo = mid;
    else hi = mid;
  }

  const t0 = times[lo];
  const t1 = times[hi];
  const frac = t1 === t0 ? 0 : (targetT - t0) / (t1 - t0);
  const s0 = timeline[lo];
  const s1 = timeline[hi];

  const lerp = (a: number, b: number) => a + (b - a) * frac;
  const lerpOptional = (
    a: number | null | undefined,
    b: number | null | undefined,
  ): number | null => {
    if (a == null && b == null) return null;
    if (a == null) return b ?? null;
    if (b == null) return a;
    return lerp(a, b);
  };
  return {
    t: targetT,
    t_abs: targetT,
    t_real: targetT,
    t_real_abs: targetT,
    cpu_percent: lerp(s0.cpu_percent, s1.cpu_percent),
    memory_mb: lerp(s0.memory_mb, s1.memory_mb),
    memory_total_mb_s: lerpOptional(s0.memory_total_mb_s, s1.memory_total_mb_s),
    memory_read_mb_s: lerpOptional(s0.memory_read_mb_s, s1.memory_read_mb_s),
    memory_write_mb_s: lerpOptional(s0.memory_write_mb_s, s1.memory_write_mb_s),
    disk_read_mb: lerp(s0.disk_read_mb ?? 0, s1.disk_read_mb ?? 0),
    disk_write_mb: lerp(s0.disk_write_mb ?? 0, s1.disk_write_mb ?? 0),
  };
}

function pushDistinctSample(
  samples: ResourceSample[],
  sample: ResourceSample,
  time: number,
  timeMode: TimeMode,
  clockMode: ClockMode,
): void {
  if (samples.length > 0) {
    const previous = samples[samples.length - 1];
    const previousTime = selectTime(previous, timeMode, clockMode);
    if (Math.abs(previousTime - time) < 1e-9) {
      samples[samples.length - 1] = sample;
      return;
    }
  }
  samples.push(sample);
}

function clipTimelineToBounds(
  timeline: ResourceSample[],
  min: number,
  max: number,
  timeMode: TimeMode,
  clockMode: ClockMode,
): ResourceSample[] {
  if (max < min) return [];
  const entries = timeline
    .map((sample) => ({ sample, time: selectTime(sample, timeMode, clockMode) }))
    .filter((entry) => Number.isFinite(entry.time))
    .sort((left, right) => left.time - right.time);
  if (entries.length === 0 || max < entries[0].time || min > entries[entries.length - 1].time) {
    return [];
  }

  const sortedTimeline = entries.map((entry) => entry.sample);
  const times = entries.map((entry) => entry.time);
  const clipped: ResourceSample[] = [];
  pushDistinctSample(
    clipped,
    interpolateSample(sortedTimeline, times, min),
    min,
    timeMode,
    clockMode,
  );
  for (const { sample, time } of entries) {
    if (time > min && time < max) {
      pushDistinctSample(clipped, sample, time, timeMode, clockMode);
    }
  }
  pushDistinctSample(
    clipped,
    interpolateSample(sortedTimeline, times, max),
    max,
    timeMode,
    clockMode,
  );
  return clipped;
}

function aggregateTimelines(
  traces: TracePayload[],
  visibility: Record<string, boolean>,
  timeMode: TimeMode,
  clockMode: ClockMode,
): ResourceSample[] {
  const valid = traces.filter(
    (t) => visibility[t.id] !== false && t.resource_timeline?.length,
  );
  if (valid.length === 0) return [];

  const clippedTimelines = valid.map((t) => {
    const bounds = traceSpanBounds(t, timeMode, clockMode);
    if (!bounds) return t.resource_timeline!;
    return clipTimelineToBounds(
      t.resource_timeline!,
      bounds.min,
      bounds.max,
      timeMode,
      clockMode,
    );
  }).filter((tl) => tl.length > 0);

  if (clippedTimelines.length === 0) return [];
  if (clippedTimelines.length === 1) return clippedTimelines[0];

  // Pre-compute per-trace time arrays
  const traceTimes = clippedTimelines.map((tl) =>
    tl.map((s) => selectTime(s, timeMode, clockMode)),
  );

  // Collect all unique timestamps, down-sample to ~500 points max for perf
  const allTimes = new Set<number>();
  for (const times of traceTimes) {
    for (const t of times) allTimes.add(t);
  }
  let sorted = [...allTimes].sort((a, b) => a - b);
  if (sorted.length > 500) {
    const step = Math.ceil(sorted.length / 500);
    sorted = sorted.filter((_, i) => i % step === 0 || i === sorted.length - 1);
  }

  return sorted.map((t) => {
    let cpu = 0;
    let mem = 0;
    let memTotal: number | null = null;
    let memRead: number | null = null;
    let memWrite: number | null = null;
    let dr = 0;
    let dw = 0;
    for (let i = 0; i < clippedTimelines.length; i++) {
      const s = interpolateSample(clippedTimelines[i], traceTimes[i], t);
      cpu += s.cpu_percent;
      mem += s.memory_mb;
      if (s.memory_total_mb_s != null) {
        memTotal = memTotal == null ? s.memory_total_mb_s : Math.max(memTotal, s.memory_total_mb_s);
      }
      if (s.memory_read_mb_s != null) {
        memRead = memRead == null ? s.memory_read_mb_s : Math.max(memRead, s.memory_read_mb_s);
      }
      if (s.memory_write_mb_s != null) {
        memWrite = memWrite == null ? s.memory_write_mb_s : Math.max(memWrite, s.memory_write_mb_s);
      }
      dr += s.disk_read_mb ?? 0;
      dw += s.disk_write_mb ?? 0;
    }
    return {
      t,
      t_abs: t,
      t_real: t,
      t_real_abs: t,
      cpu_percent: cpu,
      memory_mb: mem,
      memory_total_mb_s: memTotal,
      memory_read_mb_s: memRead,
      memory_write_mb_s: memWrite,
      disk_read_mb: dr,
      disk_write_mb: dw,
    };
  });
}

function drawMetricOverlay(
  ctx: CanvasRenderingContext2D,
  timeline: ResourceSample[],
  metric: ResourceMetric,
  timeMode: TimeMode,
  clockMode: ClockMode,
  y: number,
  h: number,
  pad: number,
  canvasW: number,
  side: "left" | "right",
  externalTimeMin?: number,
  externalTimeRange?: number,
): void {
  const values = resourceMetricValues(timeline, metric);
  const finiteValues = values.filter((v): v is number => v != null && Number.isFinite(v));
  if (finiteValues.length === 0) {
    return;
  }
  let vMin = Number.POSITIVE_INFINITY;
  let vMax = Number.NEGATIVE_INFINITY;
  for (const v of finiteValues) {
    if (v < vMin) vMin = v;
    if (v > vMax) vMax = v;
  }
  if (vMin === vMax) vMax = vMin + 1;
  const vRange = vMax - vMin;
  const innerH = h - pad * 2;
  const baselineY = y + pad + innerH;

  // Time → X mapping: use external (shared) range if provided
  const times = timeline.map((s) => selectTime(s, timeMode, clockMode));
  let tMin: number;
  let tRange: number;
  if (externalTimeMin != null && externalTimeRange != null) {
    tMin = externalTimeMin;
    tRange = externalTimeRange;
  } else {
    tMin = times[0];
    const tMax = times[times.length - 1];
    tRange = tMax - tMin || 1;
  }
  const timeToX = (t: number) => ((t - tMin) / tRange) * canvasW;

  const points = times.flatMap((t, i) => {
    const value = values[i];
    if (value == null || !Number.isFinite(value)) {
      return [];
    }
    return [{
      x: timeToX(t),
      y: y + pad + innerH - ((value - vMin) / vRange) * innerH,
    }];
  });

  if (points.length === 0) return;

  const color = displayColor(RESOURCE_METRIC_COLORS[metric] ?? "#94A3B8");

  // Area fill
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
  ctx.lineTo(points[points.length - 1].x, baselineY);
  ctx.lineTo(points[0].x, baselineY);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.globalAlpha = side === "left" ? 0.25 : 0.18;
  ctx.fill();
  ctx.globalAlpha = 1;

  // Stroke
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.globalAlpha = 0.8;
  ctx.stroke();
  ctx.globalAlpha = 1;
  ctx.lineWidth = 1;

  // Y-axis labels
  ctx.fillStyle = color;
  ctx.font = '9px "JetBrains Mono", monospace';
  const unit = resourceMetricUnit(metric);
  if (side === "left") {
    ctx.textAlign = "left";
    ctx.fillText(`${vMax.toFixed(1)}${unit}`, 4, y + pad + 8);
    ctx.fillText(`${vMin.toFixed(1)}${unit}`, 4, y + h - pad - 2);
  } else {
    ctx.textAlign = "right";
    ctx.fillText(`${vMax.toFixed(1)}${unit}`, canvasW - 4, y + pad + 8);
    ctx.fillText(`${vMin.toFixed(1)}${unit}`, canvasW - 4, y + h - pad - 2);
  }
}

export default function AggregateResourceBar(props: AggregateResourceBarProps) {
  let canvasEl!: HTMLCanvasElement;
  let wrapEl!: HTMLDivElement;

  const [expanded, setExpanded] = createSignal(false);
  const [hoverInfo, setHoverInfo] = createSignal<{ x: number; y: number; sample: ResourceSample } | null>(null);

  const aggregated = createMemo(() =>
    aggregateTimelines(props.traces, props.visibility, props.timeMode, props.clockMode),
  );

  const hasAnyMetric = () => props.resourceMetric !== "none" || props.resourceMetricSecondary !== "none";
  const hasData = () => aggregated().length > 0 && props.showResourceChart && hasAnyMetric();

  function render() {
    if (!canvasEl || !wrapEl) return;
    const data = aggregated();
    if (data.length === 0) return;

    const dpr = window.devicePixelRatio || 1;
    const w = wrapEl.clientWidth;
    const h = BAR_HEIGHT;
    if (w <= 0) return;
    canvasEl.width = w * dpr;
    canvasEl.height = h * dpr;
    canvasEl.style.width = `${w}px`;
    canvasEl.style.height = `${h}px`;

    const ctx = canvasEl.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    // Background
    const isDark = props.themeMode === "dark";
    ctx.fillStyle = isDark ? "#1a1f2e" : "#f0f2f5";
    ctx.fillRect(0, 0, w, h);

    // Use shared time range from main canvas for alignment
    const tr = canvasTimeRange();
    const tMin = tr?.minTime;
    const tRange = tr?.totalRange;

    // Primary metric
    if (props.resourceMetric !== "none") {
      drawMetricOverlay(ctx, data, props.resourceMetric, props.timeMode, props.clockMode, 0, h, CHART_PAD, w, "left", tMin, tRange);
    }

    // Secondary metric (skip if same or none)
    if (props.resourceMetricSecondary !== "none" && props.resourceMetricSecondary !== props.resourceMetric) {
      drawMetricOverlay(ctx, data, props.resourceMetricSecondary, props.timeMode, props.clockMode, 0, h, CHART_PAD, w, "right", tMin, tRange);
    }
  }

  function scheduleRender() {
    requestAnimationFrame(() => render());
  }

  createEffect(() => {
    if (!expanded() || !hasData()) return;
    // Touch reactive deps
    aggregated();
    canvasTimeRange();
    props.resourceMetric;
    props.resourceMetricSecondary;
    props.themeMode;
    scheduleRender();
  });

  return (
    <>
      <Show when={hasData()}>
        <button
          class="aggregate-resource-toggle"
          onClick={() => setExpanded((v) => !v)}
          title={expanded() ? "Collapse aggregate" : "Expand aggregate"}
        >
          {expanded() ? "\u25BC" : "\u25B2"}
        </button>
      </Show>
      <Show when={hasData() && expanded()}>
        <div
          class="aggregate-resource-bar"
          ref={(el) => { wrapEl = el; scheduleRender(); }}
          onMouseMove={(e) => {
            const data = aggregated();
            if (!data.length || !wrapEl) return;
            const rect = wrapEl.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const w = rect.width;
            // Use shared time range for alignment
            const tr = canvasTimeRange();
            let tMin: number;
            let tRange: number;
            if (tr) {
              tMin = tr.minTime;
              tRange = tr.totalRange;
            } else {
              const times = data.map((s) => selectTime(s, props.timeMode, props.clockMode));
              tMin = times[0];
              tRange = (times[times.length - 1] - tMin) || 1;
            }
            const time = tMin + (mx / w) * tRange;
            const sample = findNearestSample(data, time, props.timeMode, props.clockMode);
            setHoverInfo({ x: e.clientX, y: e.clientY, sample });
          }}
          onMouseLeave={() => setHoverInfo(null)}
        >
          <canvas ref={canvasEl} />
        </div>
        <Show when={hoverInfo()}>
          {(info) => (
            <div
              class="aggregate-hover-tooltip"
              style={{
                left: `${info().x + 12}px`,
                top: `${info().y - 80}px`,
              }}
            >
              <strong>Aggregate</strong>
              <div>CPU: {info().sample.cpu_percent.toFixed(1)}%</div>
              <div>Mem: {info().sample.memory_mb.toFixed(1)} MB</div>
              {(() => {
                const timeline = aggregated();
                const index = timeline.indexOf(info().sample);
                if (index < 0) return null;
                const memTotal = resourceMetricValues(timeline, "mem_total")[index];
                const memRead = resourceMetricValues(timeline, "mem_read")[index];
                const memWrite = resourceMetricValues(timeline, "mem_write")[index];
                const total = resourceMetricValues(timeline, "disk_total")[index] ?? 0;
                const read = resourceMetricValues(timeline, "disk_read")[index] ?? 0;
                const write = resourceMetricValues(timeline, "disk_write")[index] ?? 0;
                return (
                  <>
                    <div>Mem Total: {memTotal == null ? "N/A" : `${memTotal.toFixed(1)} MB/s`}</div>
                    <div>Mem Read: {memRead == null ? "N/A" : `${memRead.toFixed(1)} MB/s`}</div>
                    <div>Mem Write: {memWrite == null ? "N/A" : `${memWrite.toFixed(1)} MB/s`}</div>
                    <div>Disk Total: {total.toFixed(1)} MB/s</div>
                    <div>Disk Read: {read.toFixed(1)} MB/s</div>
                    <div>Disk Write: {write.toFixed(1)} MB/s</div>
                  </>
                );
              })()}
            </div>
          )}
        </Show>
      </Show>
    </>
  );
}
