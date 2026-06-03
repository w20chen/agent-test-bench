import type { TracePayload } from "../api/client";
import type { ViewMode } from "../state/signals";

export const LANE_H = 60;
export const LANE_H_CONCISE = 26;
export const MARKER_H = 6;
export const SPAN_H = 18;
export const SPAN_PAD = 4;
export const TIME_AXIS_H = 28;

// Stratified lane layout (layered mode): top strip for control flow, bottom strip for tools.
export const CONTROL_FLOW_SPAN_TYPES: ReadonlySet<string> = new Set([
  "llm",
  "scheduling",
  "mcp",
]);
export const MIN_TOOL_TRACK_H = 4;
export const MIN_SPAN_H = 3;
export const MARKER_RESERVED_H = MARKER_H + 2;
// Tool strip dynamically fills the whole strip height with N sub-tracks.
// N=1 → one fat track filling the strip; N>1 compresses proportionally.
const MIN_TOOL_SLOTS = 1;

export interface TrackLayout {
  topStripY: number;   // y offset (relative to laneY) where top strip begins (= SPAN_PAD)
  topTrackH: number;   // per-track height for top strip
  toolStripY: number;  // y offset (relative to laneY) where tool strip begins
  toolTrackH: number;  // per-track height for tool strip (may be < SPAN_H when compressed)
  toolSpanH: number;   // drawable span height within a tool track (toolTrackH - gap)
}

export function computeTrackLayout(
  topTrackCount: number,
  toolTrackCount: number,
  laneH: number,
): TrackLayout {
  const topTrackH = SPAN_H;
  const topSlots = Math.max(0, topTrackCount);
  const topStripH = topSlots > 0 ? topSlots * (SPAN_H + 2) : 0;
  const topStripY = SPAN_PAD;
  const toolStripY = topStripY + topStripH;
  const toolStripH = Math.max(
    0,
    laneH - toolStripY - SPAN_PAD - MARKER_RESERVED_H,
  );
  const slots = Math.max(MIN_TOOL_SLOTS, toolTrackCount);
  const rawTrackH = slots > 0 ? toolStripH / slots : SPAN_H + 2;
  // Cap at top-row height (SPAN_H + 2) so sequential tool spans match the
  // LLM row height visually; compress below only when concurrency requires.
  const toolTrackH = Math.min(
    SPAN_H + 2,
    Math.max(MIN_TOOL_TRACK_H, rawTrackH),
  );
  const toolSpanH = Math.max(MIN_SPAN_H, toolTrackH - 2);
  return { topStripY, topTrackH, toolStripY, toolTrackH, toolSpanH };
}

export interface LaneRow {
  laneAgentId: string;
  traceId: string;
  traceLabel: string;
  trace: TracePayload;
}

export const RESOURCE_CHART_H_CONCISE = 60;
export const RESOURCE_CHART_H_LAYERED = 40;

export function effectiveLaneH(viewMode: ViewMode): number {
  return viewMode === "concise" ? LANE_H_CONCISE : LANE_H;
}

export function resourceChartH(viewMode: ViewMode): number {
  return viewMode === "concise" ? RESOURCE_CHART_H_CONCISE : RESOURCE_CHART_H_LAYERED;
}

export function flattenVisibleLanes(
  traces: TracePayload[],
  visible: Record<string, boolean>,
): LaneRow[] {
  return traces.flatMap((trace) => {
    if (visible[trace.id] === false) {
      return [];
    }
    return trace.lanes.map((lane) => ({
      laneAgentId: lane.agent_id,
      traceId: trace.id,
      traceLabel: trace.label,
      trace,
    }));
  });
}

export function computeTotalContentHeight(
  traces: TracePayload[],
  visible: Record<string, boolean>,
  viewMode: ViewMode,
  minHeight: number,
  showResource = false,
): number {
  const laneCount = flattenVisibleLanes(traces, visible).length;
  let resourceH = 0;
  if (showResource) {
    const visibleWithResources = traces.filter(
      (t) => visible[t.id] !== false && t.resource_timeline?.length,
    );
    resourceH = visibleWithResources.length * resourceChartH(viewMode);
  }
  return Math.max(
    TIME_AXIS_H + laneCount * effectiveLaneH(viewMode) + resourceH,
    minHeight,
  );
}
