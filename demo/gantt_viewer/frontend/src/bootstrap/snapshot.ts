import type { GanttPayload, TraceDescriptor, TracePayload } from "../api/client";
import { ZOOM_PRESETS } from "../canvas/CanvasRenderer";
import {
  DEFAULT_CLOCK_MODE,
  DEFAULT_THEME_MODE,
  DEFAULT_TIME_MODE,
  DEFAULT_VIEW_MODE,
  DEFAULT_ZOOM,
  type ClockMode,
  type ResourceMetric,
  type ThemeMode,
  type TimeMode,
  type ViewMode,
} from "../state/signals";

const SNAPSHOT_BOOTSTRAP_ELEMENT_ID = "gantt-viewer-snapshot-bootstrap";

export interface SnapshotBootstrapData {
  mode: "snapshot";
  payload: GanttPayload;
  trace_ids: string[];
  visible_trace_ids: string[];
  display?: SnapshotDisplayOptions;
}

export interface SnapshotDisplayOptions {
  clockMode?: ClockMode;
  resourceMetric?: ResourceMetric;
  resourceMetricSecondary?: ResourceMetric;
  showResourceChart?: boolean;
  themeMode?: ThemeMode;
  timeMode?: TimeMode;
  viewMode?: ViewMode;
  zoom?: number;
}

export const SNAPSHOT_DEFAULTS: {
  clockMode: ClockMode;
  resourceMetric: ResourceMetric;
  resourceMetricSecondary: ResourceMetric;
  showResourceChart: boolean;
  themeMode: ThemeMode;
  timeMode: TimeMode;
  viewMode: ViewMode;
  zoom: number;
} = {
  clockMode: DEFAULT_CLOCK_MODE,
  resourceMetric: "cpu",
  resourceMetricSecondary: "memory",
  showResourceChart: true,
  themeMode: DEFAULT_THEME_MODE,
  timeMode: DEFAULT_TIME_MODE,
  viewMode: DEFAULT_VIEW_MODE,
  zoom: DEFAULT_ZOOM,
};

const CLOCK_MODES = ["wall", "real"] as const;
const RESOURCE_METRICS = [
  "cpu",
  "memory",
  "mem_total",
  "mem_read",
  "mem_write",
  "disk_total",
  "disk_read",
  "disk_write",
  "none",
] as const;
const THEME_MODES = ["dark", "light"] as const;
const TIME_MODES = ["sync", "abs"] as const;
const VIEW_MODES = ["layered", "concise"] as const;
const MIN_ZOOM = ZOOM_PRESETS[0];
const MAX_ZOOM = ZOOM_PRESETS[ZOOM_PRESETS.length - 1];

declare global {
  interface Window {
    __GANTT_VIEWER_BOOTSTRAP__?: SnapshotBootstrapData;
  }
}

function readBootstrapScriptPayload(): unknown {
  if (typeof document === "undefined") {
    return null;
  }
  const element = document.getElementById(SNAPSHOT_BOOTSTRAP_ELEMENT_ID);
  if (!(element instanceof HTMLScriptElement) || element.textContent === null) {
    return null;
  }
  try {
    return JSON.parse(element.textContent);
  } catch {
    return null;
  }
}

function isTracePayloadArray(value: unknown): value is TracePayload[] {
  return Array.isArray(value) && value.every((trace) => typeof trace?.id === "string");
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((entry) => typeof entry === "string");
}

function isOptionalObject(value: unknown): boolean {
  return value === undefined || (typeof value === "object" && value !== null);
}

function enumOrDefault<TValue extends string>(
  value: unknown,
  allowed: readonly TValue[],
  fallback: TValue,
): TValue {
  return typeof value === "string" && allowed.includes(value as TValue)
    ? (value as TValue)
    : fallback;
}

function booleanOrDefault(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function zoomOrDefault(value: unknown, fallback: number): number {
  return typeof value === "number" &&
    Number.isFinite(value) &&
    value >= MIN_ZOOM &&
    value <= MAX_ZOOM
    ? value
    : fallback;
}

export function normalizeSnapshotDisplay(
  display: SnapshotDisplayOptions | undefined,
): typeof SNAPSHOT_DEFAULTS {
  return {
    clockMode: enumOrDefault(display?.clockMode, CLOCK_MODES, SNAPSHOT_DEFAULTS.clockMode),
    resourceMetric: enumOrDefault(
      display?.resourceMetric,
      RESOURCE_METRICS,
      SNAPSHOT_DEFAULTS.resourceMetric,
    ),
    resourceMetricSecondary: enumOrDefault(
      display?.resourceMetricSecondary,
      RESOURCE_METRICS,
      SNAPSHOT_DEFAULTS.resourceMetricSecondary,
    ),
    showResourceChart: booleanOrDefault(
      display?.showResourceChart,
      SNAPSHOT_DEFAULTS.showResourceChart,
    ),
    themeMode: enumOrDefault(display?.themeMode, THEME_MODES, SNAPSHOT_DEFAULTS.themeMode),
    timeMode: enumOrDefault(display?.timeMode, TIME_MODES, SNAPSHOT_DEFAULTS.timeMode),
    viewMode: enumOrDefault(display?.viewMode, VIEW_MODES, SNAPSHOT_DEFAULTS.viewMode),
    zoom: zoomOrDefault(display?.zoom, SNAPSHOT_DEFAULTS.zoom),
  };
}

function isSnapshotBootstrapData(value: unknown): value is SnapshotBootstrapData {
  if (typeof value !== "object" || value === null) {
    return false;
  }

  const candidate = value as Partial<SnapshotBootstrapData>;
  const payload = candidate.payload;
  return (
    candidate.mode === "snapshot" &&
    typeof payload === "object" &&
    payload !== null &&
    typeof payload.registries === "object" &&
    payload.registries !== null &&
    isTracePayloadArray(payload.traces) &&
    isStringArray(candidate.trace_ids) &&
    isStringArray(candidate.visible_trace_ids) &&
    isOptionalObject(candidate.display)
  );
}

export function readSnapshotBootstrap(): SnapshotBootstrapData | null {
  const globalBootstrap = typeof window === "undefined" ? null : window.__GANTT_VIEWER_BOOTSTRAP__;
  if (isSnapshotBootstrapData(globalBootstrap)) {
    return globalBootstrap;
  }

  const scriptBootstrap = readBootstrapScriptPayload();
  if (!isSnapshotBootstrapData(scriptBootstrap)) {
    return null;
  }

  if (typeof window !== "undefined") {
    window.__GANTT_VIEWER_BOOTSTRAP__ = scriptBootstrap;
  }
  return scriptBootstrap;
}

export function snapshotDescriptorsFromTraces(traces: TracePayload[]): TraceDescriptor[] {
  return traces.map((trace, index) => ({
    id: trace.id,
    label: trace.label,
    mtime: index,
    path: `snapshot:${trace.id}`,
    size_bytes: 0,
    source_format: "trace",
  }));
}

export function visibilityFromTraceIds(
  traceIds: string[],
  visibleTraceIds: string[] = traceIds,
): Record<string, boolean> {
  const visibleSet = new Set(visibleTraceIds);
  return Object.fromEntries(traceIds.map((traceId) => [traceId, visibleSet.has(traceId)]));
}
