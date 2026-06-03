import type { GanttPayload, ResourceSample } from "../api/client";
import { displayColor } from "../theme/displayColor";
import { RESOURCE_METRIC_COLORS, findNearestSample, sameHit, type Hit, type HitCard } from "./hit";
import { resourceMetricUnit, resourceMetricValues } from "./resourceMetrics";
import {
  CONTROL_FLOW_SPAN_TYPES,
  computeTotalContentHeight,
  computeTrackLayout,
  effectiveLaneH,
  LANE_H,
  MARKER_H,
  resourceChartH,
  SPAN_H,
  SPAN_PAD,
  TIME_AXIS_H,
} from "./layout";
import { setCanvasTimeRange } from "../state/signals";
import { formatTimeLabel, niceStep } from "./time";
import { assignTracks } from "./tracks";

type TimeMode = "sync" | "abs";
type ViewMode = "layered" | "concise";
type ClockMode = "wall" | "real";
type ResourceMetric =
  | "cpu"
  | "memory"
  | "mem_total"
  | "mem_read"
  | "mem_write"
  | "disk_total"
  | "disk_read"
  | "disk_write"
  | "none";

interface RenderHitBox {
  h: number;
  hit: Hit;
  w: number;
  x: number;
  y: number;
}

const GRID_COLOR = "#141a24";
const LANE_BG = "#0a0e14";
const LANE_BORDER = "#1e2633";
const UNKNOWN_SPAN_COLOR = "#6b7280";
const TEXT_COLOR = "#c5c8d4";
const AXIS_TEXT_COLOR = "#6b7280";
const SPAN_LABEL_COLOR = "#0a0e14";
const ZOOM_MAX = 32;
const ZOOM_MIN = 0.25;
const ZOOM_STEP = 1.25;
export const ZOOM_PRESETS = [0.25, 0.5, 1, 2, 4, 8, 16, 32] as const;

export class CanvasRenderer extends EventTarget {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private hitBoxes: RenderHitBox[] = [];
  private keydownHandler: (event: KeyboardEvent) => void;
  private payload: GanttPayload | null = null;
  private rafId: number | null = null;
  private resizeRafId: number | null = null;
  private resizeObserver: ResizeObserver;
  private clockMode: ClockMode = "wall";
  private resourceMetric: ResourceMetric = "cpu";
  private resourceMetricSecondary: ResourceMetric = "memory";
  private showResourceChart = true;
  private timeMode: TimeMode = "sync";
  private viewMode: ViewMode = "layered";
  private visibility: Record<string, boolean> = {};
  private wrap: HTMLDivElement;
  private zoomFactor = 1;

  constructor(canvas: HTMLCanvasElement, wrap: HTMLDivElement) {
    super();
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      throw new Error("2d canvas context unavailable");
    }

    this.canvas = canvas;
    this.ctx = ctx;
    this.wrap = wrap;

    this.resizeObserver = new ResizeObserver(() => this.scheduleResize());
    this.resizeObserver.observe(this.wrap);

    this.canvas.addEventListener("mousemove", (event) => this.onMouseMove(event));
    this.canvas.addEventListener("mouseleave", () => this.emitHover(null));
    this.canvas.addEventListener("click", (event) => this.onClick(event));
    this.canvas.addEventListener(
      "wheel",
      (event) => this.onWheel(event),
      { passive: false },
    );
    this.wrap.addEventListener("scroll", () => {
      this.dispatchEvent(
        new CustomEvent("scroll", {
          detail: {
            scrollLeft: this.wrap.scrollLeft,
            scrollTop: this.wrap.scrollTop,
          },
        }),
      );
    });
    this.keydownHandler = (event) => this.onKeyDown(event);
    document.addEventListener("keydown", this.keydownHandler);

    this.resizeCanvasImmediate();
  }

  destroy(): void {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    if (this.resizeRafId !== null) {
      cancelAnimationFrame(this.resizeRafId);
      this.resizeRafId = null;
    }
    this.resizeObserver.disconnect();
    document.removeEventListener("keydown", this.keydownHandler);
  }

  setPayload(payload: GanttPayload | null): void {
    this.payload = payload;
    this.scheduleResize();
  }

  setTimeMode(mode: TimeMode): void {
    if (this.timeMode === mode) {
      return;
    }
    this.timeMode = mode;
    this.queueRender();
  }

  setClockMode(mode: ClockMode): void {
    if (this.clockMode === mode) {
      return;
    }
    this.clockMode = mode;
    this.queueRender();
  }

  setViewMode(mode: ViewMode): void {
    if (this.viewMode === mode) {
      return;
    }
    this.viewMode = mode;
    this.scheduleResize();
  }

  setZoom(factor: number, cursorFrac?: number): void {
    const clamped = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, factor));
    const currentWidth = this.canvas.clientWidth || this.wrap.clientWidth || 1;
    const anchorFrac =
      cursorFrac !== undefined
        ? (this.wrap.scrollLeft + cursorFrac * this.wrap.clientWidth) / currentWidth
        : 0.5;

    if (clamped === this.zoomFactor && cursorFrac === undefined) {
      return;
    }

    this.zoomFactor = clamped;
    // Synchronous resize for cursor-anchored zoom: scrollLeft needs the
    // new logical width, which scheduleResize would only produce next frame.
    this.resizeCanvasImmediate();

    const nextWidth = this.canvas.clientWidth || this.wrap.clientWidth || 1;
    const offset =
      cursorFrac !== undefined
        ? cursorFrac * this.wrap.clientWidth
        : this.wrap.clientWidth / 2;
    this.wrap.scrollLeft = anchorFrac * nextWidth - offset;
    this.dispatchEvent(
      new CustomEvent("zoom", {
        detail: {
          factor: this.zoomFactor,
        },
      }),
    );
  }

  setVisibility(map: Record<string, boolean>): void {
    this.visibility = map;
    this.scheduleResize();
  }

  setResourceMetric(metric: ResourceMetric): void {
    if (this.resourceMetric === metric) return;
    const wasNone = this.resourceMetric === "none";
    this.resourceMetric = metric;
    if (wasNone || metric === "none") this.scheduleResize();
    else this.queueRender();
  }

  setResourceMetricSecondary(metric: ResourceMetric): void {
    if (this.resourceMetricSecondary === metric) return;
    const wasNone = this.resourceMetricSecondary === "none";
    this.resourceMetricSecondary = metric;
    if (wasNone || metric === "none") this.scheduleResize();
    else this.queueRender();
  }

  setShowResourceChart(show: boolean): void {
    if (this.showResourceChart === show) return;
    this.showResourceChart = show;
    this.scheduleResize();
  }

  rerender(): void {
    this.queueRender();
  }

  hitTest(mx: number, my: number): Hit | null {
    for (let index = this.hitBoxes.length - 1; index >= 0; index -= 1) {
      const box = this.hitBoxes[index];
      if (mx >= box.x && mx <= box.x + box.w && my >= box.y && my <= box.y + box.h) {
        return box.hit;
      }
    }
    return null;
  }

  locateHit(hit: Hit): HitCard | null {
    const box = this.hitBoxes.find((candidate) => sameHit(candidate.hit, hit));
    if (!box) {
      return null;
    }
    const rect = this.canvas.getBoundingClientRect();
    return {
      hit: box.hit,
      x: rect.left + box.x + box.w / 2,
      y: rect.top + box.y + box.h / 2,
    };
  }

  private emitHover(card: HitCard | null): void {
    this.dispatchEvent(new CustomEvent("hover", { detail: card }));
  }

  private onClick(event: MouseEvent): void {
    const hit = this.hitFromEvent(event);
    // Resource charts are hover-only, not clickable
    const effective = hit?.kind === "resource" ? null : hit;
    this.dispatchEvent(
      new CustomEvent("click", {
        detail: effective
          ? {
              hit: effective,
              x: event.clientX,
              y: event.clientY,
            }
          : null,
      }),
    );
  }

  private onKeyDown(event: KeyboardEvent): void {
    const target = event.target as HTMLElement | null;
    if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) {
      return;
    }
    if (event.key === "+" || event.key === "=") {
      event.preventDefault();
      this.setZoom(this.zoomFactor * ZOOM_STEP);
    } else if (event.key === "-" || event.key === "_") {
      event.preventDefault();
      this.setZoom(this.zoomFactor / ZOOM_STEP);
    } else if (event.key === "0") {
      event.preventDefault();
      this.setZoom(1);
    }
  }

  private onMouseMove(event: MouseEvent): void {
    const hit = this.hitFromEvent(event);
    if (hit?.kind === "resource") {
      const rect = this.canvas.getBoundingClientRect();
      const mx = event.clientX - rect.left;
      const time = hit.timeMin + (mx / hit.canvasWidth) * hit.timeRange;
      hit.hoveredTime = time;
      hit.hoveredSample = findNearestSample(hit.timeline, time, this.timeMode, this.clockMode);
    }
    this.emitHover(
      hit
        ? {
            hit,
            x: event.clientX,
            y: event.clientY,
          }
        : null,
    );
  }

  private onWheel(event: WheelEvent): void {
    if (!event.ctrlKey && !event.metaKey) {
      return;
    }
    event.preventDefault();
    const rect = this.canvas.getBoundingClientRect();
    const cursorX = event.clientX - rect.left;
    const cursorFrac = this.canvas.clientWidth > 0 ? cursorX / this.canvas.clientWidth : 0.5;
    this.setZoom(
      this.zoomFactor * (event.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP),
      cursorFrac,
    );
  }

  private hitFromEvent(event: MouseEvent): Hit | null {
    const rect = this.canvas.getBoundingClientRect();
    return this.hitTest(event.clientX - rect.left, event.clientY - rect.top);
  }

  private queueRender(): void {
    if (this.rafId !== null) {
      return;
    }
    this.rafId = requestAnimationFrame(() => {
      this.rafId = null;
      this.render();
    });
  }

  private themeColor(variable: string, fallback: string): string {
    const value = getComputedStyle(document.documentElement)
      .getPropertyValue(variable)
      .trim();
    return value || fallback;
  }

  private scheduleResize(): void {
    if (this.resizeRafId !== null) {
      return;
    }
    this.resizeRafId = requestAnimationFrame(() => {
      this.resizeRafId = null;
      this.resizeCanvasImmediate();
    });
  }

  private resizeCanvasImmediate(): void {
    const dpr = window.devicePixelRatio || 1;
    const logicalWidth = Math.max(this.wrap.clientWidth * this.zoomFactor, this.wrap.clientWidth || 1);
    const effectiveShowResource = this.showResourceChart &&
      (this.resourceMetric !== "none" || this.resourceMetricSecondary !== "none");
    const logicalHeight = computeTotalContentHeight(
      this.payload?.traces ?? [],
      this.visibility,
      this.viewMode,
      this.wrap.clientHeight || 320,
      effectiveShowResource,
    );

    // canvas.width/height reset the backing buffer (and clear it) even on a
    // no-op write. Gate the writes so rapid re-renders from the same viewport
    // size don't thrash the GPU texture at high zoom.
    const nextBufW = Math.round(logicalWidth * dpr);
    const nextBufH = Math.round(logicalHeight * dpr);
    if (this.canvas.width !== nextBufW) {
      this.canvas.width = nextBufW;
    }
    if (this.canvas.height !== nextBufH) {
      this.canvas.height = nextBufH;
    }
    this.canvas.style.width = `${logicalWidth}px`;
    this.canvas.style.height = `${logicalHeight}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.queueRender();
  }

  private render(): void {
    const width = this.canvas.clientWidth || this.wrap.clientWidth || 1;
    const height = this.canvas.clientHeight || this.wrap.clientHeight || 1;

    this.ctx.clearRect(0, 0, width, height);
    this.hitBoxes = [];

    const gridColor = this.themeColor("--grid", GRID_COLOR);
    const laneBackground = this.themeColor("--bg", LANE_BG);
    const laneBorder = this.themeColor("--border", LANE_BORDER);
    const textColor = this.themeColor("--text", TEXT_COLOR);
    const axisTextColor = this.themeColor("--text-dim", AXIS_TEXT_COLOR);
    const spanLabelColor = this.themeColor("--bg", SPAN_LABEL_COLOR);

    const traces = (this.payload?.traces ?? []).filter(
      (trace) => this.visibility[trace.id] !== false,
    );
    if (traces.length === 0) {
      this.ctx.fillStyle = axisTextColor;
      this.ctx.font = '12px "JetBrains Mono", monospace';
      this.ctx.fillText("Load a trace to render the timeline.", 20, 48);
      return;
    }

    let minTime = Number.POSITIVE_INFINITY;
    let maxTime = Number.NEGATIVE_INFINITY;

    for (const trace of traces) {
        for (const lane of trace.lanes) {
          for (const span of lane.spans) {
          const start = this.selectSpanStart(span);
          const end = this.selectSpanEnd(span);
          minTime = Math.min(minTime, start);
          maxTime = Math.max(maxTime, end);
        }
        for (const marker of lane.markers) {
          const point = this.selectMarkerTime(marker);
          minTime = Math.min(minTime, point);
          maxTime = Math.max(maxTime, point);
        }
      }
      if (this.showResourceChart && trace.resource_timeline?.length) {
        const timeline = this.displayResourceTimeline(trace);
        if (timeline.length) {
          const first = this.selectResourceTime(timeline[0]);
          const last = this.selectResourceTime(timeline[timeline.length - 1]);
          minTime = Math.min(minTime, first);
          maxTime = Math.max(maxTime, last);
        }
      }
    }

    if (!Number.isFinite(minTime)) {
      minTime = 0;
      maxTime = 1;
    }

    const range = maxTime - minTime || 1;
    const margin = range * 0.02;
    // Sync mode anchors t=0 on the trace start — a negative left margin
    // would display "-Xs" ticks that have no referent.
    if (this.timeMode === "sync") {
      minTime = 0;
    } else {
      minTime -= margin;
    }
    maxTime += margin;
    const totalRange = maxTime - minTime || 1;
    setCanvasTimeRange({ minTime, totalRange });
    const laneHeight = effectiveLaneH(this.viewMode);
    const gridStep = niceStep(totalRange, width / 80);
    const timeToX = (value: number) => ((value - minTime) / totalRange) * width;

    this.ctx.strokeStyle = gridColor;
    this.ctx.lineWidth = 1;
    for (
      let tick = Math.ceil(minTime / gridStep) * gridStep;
      tick <= maxTime;
      tick += gridStep
    ) {
      const x = timeToX(tick);
      this.ctx.beginPath();
      this.ctx.moveTo(x, 0);
      this.ctx.lineTo(x, height);
      this.ctx.stroke();
    }

    this.ctx.fillStyle = axisTextColor;
    this.ctx.font = '10px "JetBrains Mono", "Menlo", monospace';
    this.ctx.textAlign = "center";
    for (
      let tick = Math.ceil(minTime / gridStep) * gridStep;
      tick <= maxTime;
      tick += gridStep
    ) {
      this.ctx.fillText(formatTimeLabel(this.timeMode, tick), timeToX(tick), TIME_AXIS_H - 6);
    }

    let laneY = TIME_AXIS_H;
    for (const trace of traces) {
      for (const lane of trace.lanes) {
        this.ctx.fillStyle = laneBackground;
        this.ctx.fillRect(0, laneY, width, laneHeight);

        this.ctx.strokeStyle = laneBorder;
        this.ctx.beginPath();
        this.ctx.moveTo(0, laneY + laneHeight);
        this.ctx.lineTo(width, laneY + laneHeight);
        this.ctx.stroke();

        const spansByIteration = new Map<number, typeof lane.spans>();
        for (const span of lane.spans) {
          const bucket = spansByIteration.get(span.iteration) ?? [];
          bucket.push(span);
          spansByIteration.set(span.iteration, bucket);
        }

        for (const spans of spansByIteration.values()) {
          spans.sort((left, right) => {
            const leftOrder = this.payload?.registries.spans[left.type]?.order ?? 99;
            const rightOrder = this.payload?.registries.spans[right.type]?.order ?? 99;
            return leftOrder - rightOrder;
          });

          // Stratified layout (layered mode): top strip for control-flow
          // span types (llm/scheduling/mcp) and bottom strip for tool spans.
          // Concise mode keeps all spans collapsed onto track 0.
          const isConcise = this.viewMode === "concise";
          const topSpans = isConcise
            ? spans
            : spans.filter((s) => CONTROL_FLOW_SPAN_TYPES.has(s.type));
          const toolSpans = isConcise
            ? []
            : spans.filter((s) => !CONTROL_FLOW_SPAN_TYPES.has(s.type));

          const topTracks = assignTracks(
            topSpans,
            (s) => this.selectSpanStart(s),
            (s) => this.selectSpanEnd(s),
          );
          const toolTracks = assignTracks(
            toolSpans,
            (s) => this.selectSpanStart(s),
            (s) => this.selectSpanEnd(s),
          );
          const topTrackCount = topTracks.length
            ? Math.max(...topTracks) + 1
            : 0;
          const toolTrackCount = toolTracks.length
            ? Math.max(...toolTracks) + 1
            : 0;
          const topTrackBySpan = new Map(
            topSpans.map((span, idx) => [span, topTracks[idx] ?? 0] as const),
          );
          const toolTrackBySpan = new Map(
            toolSpans.map((span, idx) => [span, toolTracks[idx] ?? 0] as const),
          );
          const layout = computeTrackLayout(
            topTrackCount,
            toolTrackCount,
            LANE_H,
          );

          const spanYH = (span: (typeof spans)[number]) => {
            if (isConcise) {
              return { y: laneY + SPAN_PAD, h: SPAN_H };
            }
            if (CONTROL_FLOW_SPAN_TYPES.has(span.type)) {
              const track = topTrackBySpan.get(span) ?? 0;
              return {
                y: laneY + layout.topStripY + track * (layout.topTrackH + 2),
                h: layout.topTrackH,
              };
            }
            const track = toolTrackBySpan.get(span) ?? 0;
            return {
              y: laneY + layout.toolStripY + track * layout.toolTrackH,
              h: layout.toolSpanH,
            };
          };

          spans.forEach((span) => {
            const start = this.selectSpanStart(span);
            const end = this.selectSpanEnd(span);
            const x0 = timeToX(start);
            const x1 = timeToX(end);
            const widthPx = Math.max(x1 - x0, 3);
            const { y: yOffset, h: spanH } = spanYH(span);
            const color =
              displayColor(
                this.payload?.registries.spans[span.type]?.color ?? UNKNOWN_SPAN_COLOR,
              );

            this.ctx.fillStyle = color;
            this.ctx.globalAlpha = 0.88;
            this.ctx.fillRect(x0, yOffset, widthPx, spanH);
            this.ctx.globalAlpha = 1;

            if (widthPx > 22 && spanH >= 10) {
              this.ctx.fillStyle = spanLabelColor;
              this.ctx.textAlign = "left";
              this.ctx.fillText(String(span.iteration), x0 + 4, yOffset + Math.min(12, spanH - 2));
            }

            this.hitBoxes.push({
              h: spanH,
              hit: {
                item: span,
                kind: "span",
                laneAgentId: lane.agent_id,
                traceId: trace.id,
                traceLabel: trace.label,
              },
              w: widthPx,
              x: x0,
              y: yOffset,
            });
          });
        }

        for (const marker of lane.markers) {
          const point = this.selectMarkerTime(marker);
          const x = timeToX(point);
          const markerReg =
            this.payload?.registries.markers[marker.event] ??
            this.payload?.registries.markers[marker.type];
          const color = displayColor(markerReg?.color ?? UNKNOWN_SPAN_COLOR);
          const symbol = markerReg?.symbol ?? "dot";
          const centerY = laneY + laneHeight - MARKER_H - 2;

          this.ctx.fillStyle = color;
          this.ctx.globalAlpha = 0.8;
          if (symbol === "diamond") {
            this.ctx.save();
            this.ctx.translate(x, centerY);
            this.ctx.rotate(Math.PI / 4);
            this.ctx.fillRect(-3, -3, 6, 6);
            this.ctx.restore();
          } else if (symbol === "flag") {
            this.ctx.fillRect(x - 1, centerY - 6, 2, 10);
            this.ctx.fillRect(x + 1, centerY - 6, 6, 4);
          } else if (symbol === "cross") {
            this.ctx.fillRect(x - 4, centerY - 1, 8, 2);
            this.ctx.fillRect(x - 1, centerY - 4, 2, 8);
          } else {
            this.ctx.beginPath();
            this.ctx.arc(x, centerY, 3, 0, Math.PI * 2);
            this.ctx.fill();
          }
          this.ctx.globalAlpha = 1;

          this.hitBoxes.push({
            h: 12,
            hit: {
              item: marker,
              kind: "marker",
              laneAgentId: lane.agent_id,
              traceId: trace.id,
              traceLabel: trace.label,
            },
            w: 12,
            x: x - 6,
            y: centerY - 6,
          });
        }

        laneY += laneHeight;
      }

      // Resource utilization chart for this trace (dual-metric overlay)
      const hasAnyMetric = this.resourceMetric !== "none" || this.resourceMetricSecondary !== "none";
      if (this.showResourceChart && hasAnyMetric && trace.resource_timeline?.length) {
        const timeline = this.displayResourceTimeline(trace);
        if (!timeline.length) { laneY += resourceChartH(this.viewMode); continue; }
        const chartH = resourceChartH(this.viewMode);
        const chartPad = 3;

        // Background
        this.ctx.fillStyle = laneBackground;
        this.ctx.fillRect(0, laneY, width, chartH);
        this.ctx.strokeStyle = laneBorder;
        this.ctx.beginPath();
        this.ctx.moveTo(0, laneY + chartH);
        this.ctx.lineTo(width, laneY + chartH);
        this.ctx.stroke();

        // Primary metric (left Y-axis labels) — skip if "none"
        let primaryRange: { vMin: number; vMax: number } = { vMin: 0, vMax: 1 };
        if (this.resourceMetric !== "none") {
          primaryRange = this.renderMetricOverlay(
            timeline, this.resourceMetric, timeToX, laneY, chartH, chartPad, "left",
          );
        }

        // Secondary metric (right Y-axis labels) — skip if "none" or same as primary
        let secondaryRange: { vMin: number; vMax: number } | undefined;
        if (this.resourceMetricSecondary !== "none" && this.resourceMetricSecondary !== this.resourceMetric) {
          secondaryRange = this.renderMetricOverlay(
            timeline, this.resourceMetricSecondary, timeToX, laneY, chartH, chartPad, "right",
          );
        }

        // Hit box for the entire resource chart area
        this.hitBoxes.push({
          h: chartH,
          hit: {
            kind: "resource",
            traceId: trace.id,
            traceLabel: trace.label,
            timeline,
            metric: this.resourceMetric,
            metricSecondary: this.resourceMetricSecondary !== this.resourceMetric ? this.resourceMetricSecondary : undefined,
            chartY: laneY,
            chartH,
            chartPad,
            vMin: primaryRange.vMin,
            vMax: primaryRange.vMax,
            vMinSecondary: secondaryRange?.vMin,
            vMaxSecondary: secondaryRange?.vMax,
            timeMin: minTime,
            timeRange: totalRange,
            canvasWidth: width,
          } as Hit,
          w: width,
          x: 0,
          y: laneY,
        });

        laneY += chartH;
      }
    }

    this.dispatchEvent(new CustomEvent("render"));
  }

  private selectSpanStart(span: GanttPayload["traces"][number]["lanes"][number]["spans"][number]): number {
    if (this.clockMode === "real") {
      const realStart = this.timeMode === "sync" ? span.start_real : span.start_real_abs;
      if (typeof realStart === "number" && Number.isFinite(realStart)) {
        return realStart;
      }
    }
    return this.timeMode === "sync" ? span.start : span.start_abs;
  }

  private selectSpanEnd(span: GanttPayload["traces"][number]["lanes"][number]["spans"][number]): number {
    if (this.clockMode === "real") {
      const realEnd = this.timeMode === "sync" ? span.end_real : span.end_real_abs;
      if (typeof realEnd === "number" && Number.isFinite(realEnd)) {
        return realEnd;
      }
    }
    return this.timeMode === "sync" ? span.end : span.end_abs;
  }

  private selectMarkerTime(marker: GanttPayload["traces"][number]["lanes"][number]["markers"][number]): number {
    if (this.clockMode === "real") {
      const realPoint = this.timeMode === "sync" ? marker.t_real : marker.t_real_abs;
      if (typeof realPoint === "number" && Number.isFinite(realPoint)) {
        return realPoint;
      }
    }
    return this.timeMode === "sync" ? marker.t : marker.t_abs;
  }

  private selectResourceTime(sample: ResourceSample): number {
    if (this.clockMode === "real") {
      const realT = this.timeMode === "sync" ? sample.t_real : sample.t_real_abs;
      if (typeof realT === "number" && Number.isFinite(realT)) {
        return realT;
      }
    }
    return this.timeMode === "sync" ? sample.t : sample.t_abs;
  }

  private displayResourceTimeline(trace: GanttPayload["traces"][number]): ResourceSample[] {
    const timeline = trace.resource_timeline ?? [];
    if (timeline.length === 0) {
      return [];
    }
    const bounds = this.traceSpanTimeBounds(trace);
    if (!bounds) {
      return timeline;
    }
    return this.clipResourceTimelineToBounds(timeline, bounds.min, bounds.max);
  }

  private traceSpanTimeBounds(trace: GanttPayload["traces"][number]): { min: number; max: number } | null {
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    for (const lane of trace.lanes) {
      for (const span of lane.spans) {
        min = Math.min(min, this.selectSpanStart(span));
        max = Math.max(max, this.selectSpanEnd(span));
      }
    }
    if (!Number.isFinite(min) || !Number.isFinite(max)) {
      return null;
    }
    return { min, max };
  }

  private clipResourceTimelineToBounds(
    timeline: ResourceSample[],
    min: number,
    max: number,
  ): ResourceSample[] {
    if (max < min) {
      return [];
    }

    const entries = timeline
      .map((sample) => ({ sample, time: this.selectResourceTime(sample) }))
      .filter((entry) => Number.isFinite(entry.time))
      .sort((left, right) => left.time - right.time);
    if (entries.length === 0) {
      return [];
    }

    const clipped: ResourceSample[] = [];
    const startSample = this.resourceSampleAt(entries, min);
    if (startSample) {
      this.appendDistinctResourceSample(clipped, startSample);
    }

    for (const { sample, time } of entries) {
      if (time > min && time < max) {
        this.appendDistinctResourceSample(clipped, sample);
      }
    }

    const endSample = this.resourceSampleAt(entries, max);
    if (endSample) {
      this.appendDistinctResourceSample(clipped, endSample);
    }
    return clipped;
  }

  private resourceSampleAt(
    entries: Array<{ sample: ResourceSample; time: number }>,
    targetTime: number,
  ): ResourceSample | null {
    if (targetTime < entries[0].time || targetTime > entries[entries.length - 1].time) {
      return null;
    }

    let previous = entries[0];
    for (const entry of entries) {
      if (Math.abs(entry.time - targetTime) < 1e-9) {
        return this.withSelectedResourceTime({ ...entry.sample }, targetTime);
      }
      if (entry.time > targetTime) {
        return this.interpolateResourceSample(previous, entry, targetTime);
      }
      previous = entry;
    }
    return this.withSelectedResourceTime({ ...entries[entries.length - 1].sample }, targetTime);
  }

  private interpolateResourceSample(
    left: { sample: ResourceSample; time: number },
    right: { sample: ResourceSample; time: number },
    targetTime: number,
  ): ResourceSample {
    const range = right.time - left.time;
    const frac = range === 0 ? 0 : (targetTime - left.time) / range;
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

    const contextSwitches = lerpOptional(
      left.sample.context_switches,
      right.sample.context_switches,
    );
    const sample: ResourceSample = {
      t: lerp(left.sample.t, right.sample.t),
      t_abs: lerp(left.sample.t_abs, right.sample.t_abs),
      t_real: lerpOptional(left.sample.t_real, right.sample.t_real),
      t_real_abs: lerpOptional(left.sample.t_real_abs, right.sample.t_real_abs),
      cpu_percent: lerp(left.sample.cpu_percent, right.sample.cpu_percent),
      memory_mb: lerp(left.sample.memory_mb, right.sample.memory_mb),
      memory_total_mb_s: lerpOptional(
        left.sample.memory_total_mb_s,
        right.sample.memory_total_mb_s,
      ),
      memory_read_mb_s: lerpOptional(
        left.sample.memory_read_mb_s,
        right.sample.memory_read_mb_s,
      ),
      memory_write_mb_s: lerpOptional(
        left.sample.memory_write_mb_s,
        right.sample.memory_write_mb_s,
      ),
      disk_read_mb: lerpOptional(left.sample.disk_read_mb, right.sample.disk_read_mb),
      disk_write_mb: lerpOptional(left.sample.disk_write_mb, right.sample.disk_write_mb),
      net_rx_mb: lerpOptional(left.sample.net_rx_mb, right.sample.net_rx_mb),
      net_tx_mb: lerpOptional(left.sample.net_tx_mb, right.sample.net_tx_mb),
      context_switches: contextSwitches == null ? null : Math.round(contextSwitches),
    };
    return this.withSelectedResourceTime(sample, targetTime);
  }

  private withSelectedResourceTime(sample: ResourceSample, targetTime: number): ResourceSample {
    if (this.clockMode === "real") {
      if (this.timeMode === "sync") {
        sample.t_real = targetTime;
      } else {
        sample.t_real_abs = targetTime;
      }
    } else if (this.timeMode === "sync") {
      sample.t = targetTime;
    } else {
      sample.t_abs = targetTime;
    }
    return sample;
  }

  private appendDistinctResourceSample(samples: ResourceSample[], sample: ResourceSample): void {
    const previous = samples.at(-1);
    if (previous) {
      const previousTime = this.selectResourceTime(previous);
      const nextTime = this.selectResourceTime(sample);
      if (Math.abs(previousTime - nextTime) < 1e-9) {
        samples[samples.length - 1] = sample;
        return;
      }
    }
    samples.push(sample);
  }

  private resourceMetricUnit(): string {
    return resourceMetricUnit(this.resourceMetric);
  }

  private renderMetricOverlay(
    timeline: ResourceSample[],
    metric: ResourceMetric,
    timeToX: (t: number) => number,
    laneY: number,
    chartH: number,
    chartPad: number,
    labelSide: "left" | "right",
  ): { vMin: number; vMax: number } {
    const values = resourceMetricValues(timeline, metric);
    const finiteValues = values.filter((v): v is number => v != null && Number.isFinite(v));
    if (finiteValues.length === 0) {
      return { vMin: 0, vMax: 1 };
    }
    let vMin = Number.POSITIVE_INFINITY;
    let vMax = Number.NEGATIVE_INFINITY;
    for (const v of finiteValues) {
      if (v < vMin) vMin = v;
      if (v > vMax) vMax = v;
    }
    if (vMin === vMax) { vMax = vMin + 1; }
    const vRange = vMax - vMin;
    const innerH = chartH - chartPad * 2;
    const baselineY = laneY + chartPad + innerH;

    const points = timeline.flatMap((sample, i) => {
      const value = values[i];
      if (value == null || !Number.isFinite(value)) {
        return [];
      }
      return [{
        x: timeToX(this.selectResourceTime(sample)),
        y: laneY + chartPad + innerH - ((value - vMin) / vRange) * innerH,
      }];
    });

    if (points.length > 0) {
      const metricColor = displayColor(
        RESOURCE_METRIC_COLORS[metric] ?? "#94A3B8",
      );

      // Area fill
      this.ctx.beginPath();
      this.ctx.moveTo(points[0].x, points[0].y);
      for (let i = 1; i < points.length; i++) {
        this.ctx.lineTo(points[i].x, points[i].y);
      }
      this.ctx.lineTo(points[points.length - 1].x, baselineY);
      this.ctx.lineTo(points[0].x, baselineY);
      this.ctx.closePath();
      this.ctx.fillStyle = metricColor;
      this.ctx.globalAlpha = labelSide === "left" ? 0.25 : 0.18;
      this.ctx.fill();
      this.ctx.globalAlpha = 1;

      // Top stroke line
      this.ctx.beginPath();
      this.ctx.moveTo(points[0].x, points[0].y);
      for (let i = 1; i < points.length; i++) {
        this.ctx.lineTo(points[i].x, points[i].y);
      }
      this.ctx.strokeStyle = metricColor;
      this.ctx.lineWidth = 1.5;
      this.ctx.globalAlpha = 0.8;
      this.ctx.stroke();
      this.ctx.globalAlpha = 1;
      this.ctx.lineWidth = 1;
    }

    // Y-axis labels colored by metric
    const canvasW = this.canvas.width / (window.devicePixelRatio || 1);
    const metricLabelColor = displayColor(RESOURCE_METRIC_COLORS[metric] ?? "#94A3B8");
    this.ctx.fillStyle = metricLabelColor;
    this.ctx.font = '9px "JetBrains Mono", monospace';
    const unit = resourceMetricUnit(metric);
    if (labelSide === "left") {
      this.ctx.textAlign = "left";
      this.ctx.fillText(`${vMax.toFixed(1)}${unit}`, 4, laneY + chartPad + 8);
      this.ctx.fillText(`${vMin.toFixed(1)}${unit}`, 4, laneY + chartH - chartPad - 2);
    } else {
      this.ctx.textAlign = "right";
      this.ctx.fillText(`${vMax.toFixed(1)}${unit}`, canvasW - 4, laneY + chartPad + 8);
      this.ctx.fillText(`${vMin.toFixed(1)}${unit}`, canvasW - 4, laneY + chartH - chartPad - 2);
    }

    return { vMin, vMax };
  }
}
