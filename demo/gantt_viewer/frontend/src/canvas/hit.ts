import type { Registries, ResourceSample, TracePayload } from "../api/client";
import { resourceMetricValueAt } from "./resourceMetrics";
import { displayColor } from "../theme/displayColor";
import { formatAbsTime } from "./time";

export interface SpanHit {
  item: TracePayload["lanes"][number]["spans"][number];
  kind: "span";
  laneAgentId: string;
  traceId: string;
  traceLabel: string;
}

export interface MarkerHit {
  item: TracePayload["lanes"][number]["markers"][number];
  kind: "marker";
  laneAgentId: string;
  traceId: string;
  traceLabel: string;
}

export interface LaneHit {
  kind: "lane";
  laneAgentId: string;
  trace: TracePayload;
  traceId: string;
  traceLabel: string;
}

export interface ResourceHit {
  kind: "resource";
  traceId: string;
  traceLabel: string;
  timeline: ResourceSample[];
  metric: string;
  metricSecondary?: string;
  chartY: number;
  chartH: number;
  chartPad: number;
  vMin: number;
  vMax: number;
  vMinSecondary?: number;
  vMaxSecondary?: number;
  /** Time range for X→time reverse mapping */
  timeMin: number;
  timeRange: number;
  canvasWidth: number;
  /** Populated during hover (not click) */
  hoveredSample?: ResourceSample;
  hoveredTime?: number;
}

export type Hit = LaneHit | MarkerHit | ResourceHit | SpanHit;

export interface HitCard {
  hit: Hit;
  x: number;
  y: number;
}

export function sameHit(a: Hit | null, b: Hit | null): boolean {
  if (!a || !b || a.kind !== b.kind || a.traceId !== b.traceId) {
    return false;
  }
  if (a.kind === "lane" && b.kind === "lane") {
    return a.laneAgentId === b.laneAgentId;
  }
  if (a.kind === "resource" && b.kind === "resource") {
    return a.traceId === b.traceId && a.metric === b.metric && a.metricSecondary === b.metricSecondary;
  }
  if (a.kind === "span" && b.kind === "span") {
    return (
      a.laneAgentId === b.laneAgentId &&
      a.item.type === b.item.type &&
      a.item.iteration === b.item.iteration &&
      a.item.start_abs === b.item.start_abs &&
      a.item.end_abs === b.item.end_abs
    );
  }
  if (a.kind === "marker" && b.kind === "marker") {
    return (
      a.laneAgentId === b.laneAgentId &&
      a.item.type === b.item.type &&
      a.item.event === b.item.event &&
      a.item.t_abs === b.item.t_abs
    );
  }
  return false;
}

export const RESOURCE_METRIC_COLORS: Record<string, string> = {
  cpu: "#00E5FF",
  memory: "#76FF03",
  mem_total: "#26C6DA",
  mem_read: "#4DD0E1",
  mem_write: "#0097A7",
  disk_total: "#FF6D00",
  disk_read: "#FFA726",
  disk_write: "#FF7043",
};

const RESOURCE_METRIC_LABELS: Record<string, string> = {
  cpu: "CPU %",
  memory: "Memory MB",
  mem_total: "Mem Total MB/s",
  mem_read: "Mem Read MB/s",
  mem_write: "Mem Write MB/s",
  disk_total: "Disk Total MB/s",
  disk_read: "Disk Read MB/s",
  disk_write: "Disk Write MB/s",
};

export function hitAccent(hit: Hit, registries: Registries | null): string {
  if (hit.kind === "lane") {
    return displayColor("#00E5FF");
  }
  if (hit.kind === "resource") {
    return displayColor(RESOURCE_METRIC_COLORS[hit.metric] ?? "#94A3B8");
  }
  if (hit.kind === "span") {
    return displayColor(registries?.spans[hit.item.type]?.color ?? "#94A3B8");
  }
  const marker = registries?.markers[hit.item.event] ?? registries?.markers[hit.item.type];
  return displayColor(marker?.color ?? "#94A3B8");
}

export function hitTitle(hit: Hit, registries: Registries | null): string {
  if (hit.kind === "lane") {
    return `Trace ${hit.traceLabel}`;
  }
  if (hit.kind === "resource") {
    const primary = RESOURCE_METRIC_LABELS[hit.metric] ?? hit.metric;
    const secondary = hit.metricSecondary ? (RESOURCE_METRIC_LABELS[hit.metricSecondary] ?? hit.metricSecondary) : null;
    return secondary && secondary !== primary ? `Resource · ${primary} + ${secondary}` : `Resource · ${primary}`;
  }
  if (hit.kind === "span") {
    const label = registries?.spans[hit.item.type]?.label ?? hit.item.type;
    return `${label} · Iter ${hit.item.iteration}`;
  }
  return `${hit.item.event} · Iter ${hit.item.iteration}`;
}

export function hitRows(hit: Hit): Array<[string, string]> {
  if (hit.kind === "resource") {
    const s = hit.hoveredSample;
    if (s) {
      const rows: Array<[string, string]> = [
        ["trace", hit.traceLabel],
        ["CPU", `${s.cpu_percent.toFixed(1)}%`],
        ["Memory", `${s.memory_mb.toFixed(1)} MB`],
      ];
      const sampleIndex = hit.timeline.indexOf(s);
      if (sampleIndex >= 0) {
        const memTotal = resourceMetricValueAt(hit.timeline, sampleIndex, "mem_total");
        const memRead = resourceMetricValueAt(hit.timeline, sampleIndex, "mem_read");
        const memWrite = resourceMetricValueAt(hit.timeline, sampleIndex, "mem_write");
        rows.push(["Mem Total", memTotal == null ? "N/A" : `${memTotal.toFixed(1)} MB/s`]);
        rows.push(["Mem Read", memRead == null ? "N/A" : `${memRead.toFixed(1)} MB/s`]);
        rows.push(["Mem Write", memWrite == null ? "N/A" : `${memWrite.toFixed(1)} MB/s`]);
        const diskTotal = resourceMetricValueAt(hit.timeline, sampleIndex, "disk_total") ?? 0;
        const diskRead = resourceMetricValueAt(hit.timeline, sampleIndex, "disk_read") ?? 0;
        const diskWrite = resourceMetricValueAt(hit.timeline, sampleIndex, "disk_write") ?? 0;
        rows.push(["Disk Total", `${diskTotal.toFixed(1)} MB/s`]);
        rows.push(["Disk Read", `${diskRead.toFixed(1)} MB/s`]);
        rows.push(["Disk Write", `${diskWrite.toFixed(1)} MB/s`]);
      }
      if (s.context_switches != null) {
        rows.push(["Ctx Switches", String(s.context_switches)]);
      }
      return rows;
    }
    // Fallback: no hover sample (e.g. pinned card)
    const rows: Array<[string, string]> = [
      ["trace", hit.traceLabel],
      [RESOURCE_METRIC_LABELS[hit.metric] ?? hit.metric, `${hit.vMin.toFixed(1)} – ${hit.vMax.toFixed(1)}`],
    ];
    if (hit.metricSecondary && hit.metricSecondary !== hit.metric) {
      rows.push(
        [RESOURCE_METRIC_LABELS[hit.metricSecondary] ?? hit.metricSecondary, `${(hit.vMinSecondary ?? 0).toFixed(1)} – ${(hit.vMaxSecondary ?? 0).toFixed(1)}`],
      );
    }
    rows.push(["samples", String(hit.timeline.length)]);
    return rows;
  }
  if (hit.kind === "lane") {
    const metadata = hit.trace.metadata;
    const rows: Array<[string, string]> = [
      ["trace", hit.traceLabel],
      ["agent", hit.laneAgentId],
      ["scaffold", metadata.scaffold],
      ["model", metadata.model ?? "?"],
      ["instance", metadata.instance_id || "?"],
      ["actions", String(metadata.n_actions)],
      ["iterations", String(metadata.n_iterations)],
      ["events", String(metadata.n_events)],
      ["elapsed", `${(metadata.elapsed_s ?? 0).toFixed(2)} s`],
      ["lanes", String(hit.trace.lanes.length)],
    ];
    if (metadata.mode) {
      rows.splice(4, 0, ["mode", metadata.mode]);
    }
    return rows;
  }

  if (hit.kind === "span") {
    const rows: Array<[string, string]> = [
      ["trace", hit.traceLabel],
      ["agent", hit.laneAgentId],
      ["duration", `${((hit.item.end_abs - hit.item.start_abs) * 1000).toFixed(1)} ms`],
      ["start", formatAbsTime(hit.item.start_abs)],
      ["relative", `+${hit.item.start.toFixed(3)}s`],
    ];
    rows.push(...spanDetailRows(hit.item.type, hit.item.detail ?? {}));
    return rows;
  }

  const rows: Array<[string, string]> = [
    ["trace", hit.traceLabel],
    ["agent", hit.laneAgentId],
    ["event", hit.item.event],
    ["time", formatAbsTime(hit.item.t_abs)],
    ["relative", `+${hit.item.t.toFixed(3)}s`],
  ];
  for (const [key, value] of Object.entries(hit.item.detail ?? {})) {
    rows.push([key, formatDetailValue(value)]);
  }
  return rows;
}

function spanDetailRows(spanType: string, detail: Record<string, unknown>): Array<[string, string]> {
  if (spanType !== "llm") {
    return Object.entries(detail).map(([key, value]) => [key, formatDetailValue(value)]);
  }

  const rows: Array<[string, string]> = [];
  pushDetailRow(rows, detail, "finish_reason", "finish");
  pushDetailRow(rows, detail, "prompt_tokens", "prompt_tokens");
  pushDetailRow(rows, detail, "completion_tokens", "completion_tokens");
  pushDetailRow(rows, detail, "llm_content", "response");
  pushDetailRow(rows, detail, "tool_calls_requested", "tool_calls");
  pushDetailRow(rows, detail, "llm_call_time_ms", "llm_call_time_ms");
  pushDetailRow(rows, detail, "llm_wall_latency_ms", "llm_wall_latency_ms");
  pushDetailRow(rows, detail, "llm_timing_source", "timing_source");

  const openrouterSummary = formatOpenRouterSummary(detail);
  if (openrouterSummary) {
    rows.push(["openrouter", openrouterSummary]);
  }

  const shown = new Set([
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "llm_content",
    "tool_calls_requested",
    "llm_call_time_ms",
    "llm_latency_ms",
    "llm_wall_latency_ms",
    "llm_timing_source",
    ...OPENROUTER_DETAIL_FIELDS,
  ]);
  for (const [key, value] of Object.entries(detail)) {
    if (shown.has(key) || key.startsWith("openrouter_")) {
      continue;
    }
    rows.push([key, formatDetailValue(value)]);
  }
  return rows;
}

const OPENROUTER_DETAIL_FIELDS = [
  "openrouter_generation_id",
  "openrouter_metadata_fetch_status",
  "openrouter_generation_time_ms",
  "openrouter_latency_ms",
  "openrouter_provider_latency_ms",
  "openrouter_provider_name",
  "openrouter_model",
  "openrouter_metadata_fetch_last_reason",
  "openrouter_metadata_refetch_attempted",
  "openrouter_metadata_initial_fetch_status",
  "openrouter_metadata_fetch_attempt_count",
  "openrouter_metadata_fetch_status_codes",
  "openrouter_metadata_fetch_last_status_code",
  "openrouter_metadata_capture_enabled",
  "openrouter_metadata_task_pending",
  "openrouter_metadata_retry_delays_s",
  "openrouter_metadata_timeout_s",
  "openrouter_metadata_fetch_ms",
  "openrouter_request_id",
  "openrouter_upstream_id",
  "openrouter_created_at",
  "openrouter_api_type",
  "openrouter_metadata",
];

function pushDetailRow(
  rows: Array<[string, string]>,
  detail: Record<string, unknown>,
  key: string,
  label: string,
): void {
  if (detail[key] == null) {
    return;
  }
  rows.push([label, formatDetailValue(detail[key])]);
}

function formatOpenRouterSummary(detail: Record<string, unknown>): string | null {
  const status = detail.openrouter_metadata_fetch_status;
  const generationMs = detail.openrouter_generation_time_ms;
  const latencyMs = detail.openrouter_latency_ms;
  const provider = detail.openrouter_provider_name;
  const parts: string[] = [];
  if (typeof status === "string" && status) {
    parts.push(status);
  }
  if (typeof generationMs === "number" && Number.isFinite(generationMs)) {
    parts.push(`generation ${generationMs.toFixed(0)} ms`);
  }
  if (typeof latencyMs === "number" && Number.isFinite(latencyMs)) {
    parts.push(`latency ${latencyMs.toFixed(0)} ms`);
  }
  if (typeof provider === "string" && provider) {
    parts.push(provider);
  }
  return parts.length ? parts.join(" · ") : null;
}

type TimeMode = "sync" | "abs";
type ClockMode = "wall" | "real";

function selectResourceTime(sample: ResourceSample, timeMode: TimeMode, clockMode: ClockMode): number {
  if (clockMode === "real") {
    const realT = timeMode === "sync" ? sample.t_real : sample.t_real_abs;
    if (typeof realT === "number" && Number.isFinite(realT)) return realT;
  }
  return timeMode === "sync" ? sample.t : sample.t_abs;
}

export function findNearestSample(
  timeline: ResourceSample[],
  targetTime: number,
  timeMode: TimeMode,
  clockMode: ClockMode,
): ResourceSample {
  if (timeline.length === 0) return timeline[0];
  let best = 0;
  let bestDist = Math.abs(selectResourceTime(timeline[0], timeMode, clockMode) - targetTime);
  // Binary search for efficiency on large timelines
  let lo = 0;
  let hi = timeline.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const t = selectResourceTime(timeline[mid], timeMode, clockMode);
    const dist = Math.abs(t - targetTime);
    if (dist < bestDist) {
      bestDist = dist;
      best = mid;
    }
    if (t < targetTime) lo = mid + 1;
    else if (t > targetTime) hi = mid - 1;
    else break;
  }
  return timeline[best];
}

export function formatDetailValue(value: unknown): string {
  if (Array.isArray(value)) {
    return value.map((item) => String(item)).join("\n");
  }
  if (value && typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}
