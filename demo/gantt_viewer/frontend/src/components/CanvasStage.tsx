import { createEffect, onCleanup, onMount } from "solid-js";

import type { GanttPayload } from "../api/client";
import { CanvasRenderer } from "../canvas/CanvasRenderer";
import type { Hit, HitCard } from "../canvas/hit";
import type { ClockMode, ResourceMetric, ThemeMode, TimeMode, ViewMode } from "../state/signals";

interface CanvasStageProps {
  clockMode: ClockMode;
  onClick: (card: HitCard | null) => void;
  onHover: (card: HitCard | null) => void;
  onPinnedReanchor: (card: HitCard | null) => void;
  onScroll: (scrollTop: number) => void;
  onZoom: (factor: number) => void;
  payload: GanttPayload | null;
  pinnedHit: Hit | null;
  resourceMetric: ResourceMetric;
  resourceMetricSecondary: ResourceMetric;
  showResourceChart: boolean;
  themeMode: ThemeMode;
  timeMode: TimeMode;
  viewMode: ViewMode;
  visibility: Record<string, boolean>;
  zoom: number;
}

export default function CanvasStage(props: CanvasStageProps) {
  let canvasEl!: HTMLCanvasElement;
  let wrapEl!: HTMLDivElement;
  let renderer: CanvasRenderer | null = null;

  const reanchorPinned = () => {
    if (!renderer || !props.pinnedHit || props.pinnedHit.kind === "lane") {
      return;
    }
    props.onPinnedReanchor(renderer.locateHit(props.pinnedHit));
  };

  onMount(() => {
    renderer = new CanvasRenderer(canvasEl, wrapEl);
    renderer.addEventListener("hover", (event) =>
      props.onHover((event as CustomEvent<HitCard | null>).detail),
    );
    renderer.addEventListener("click", (event) =>
      props.onClick((event as CustomEvent<HitCard | null>).detail),
    );
    renderer.addEventListener("scroll", (event) =>
      props.onScroll((event as CustomEvent<{ scrollTop: number }>).detail.scrollTop),
    );
    renderer.addEventListener("scroll", reanchorPinned);
    renderer.addEventListener("zoom", (event) =>
      props.onZoom((event as CustomEvent<{ factor: number }>).detail.factor),
    );
    renderer.addEventListener("render", reanchorPinned);
  });

  createEffect(() => {
    renderer?.setPayload(props.payload);
  });

  createEffect(() => {
    renderer?.setClockMode(props.clockMode);
  });

  createEffect(() => {
    renderer?.setTimeMode(props.timeMode);
  });

  createEffect(() => {
    renderer?.setViewMode(props.viewMode);
  });

  createEffect(() => {
    renderer?.setVisibility(props.visibility);
  });

  createEffect(() => {
    renderer?.setZoom(props.zoom);
  });

  createEffect(() => {
    renderer?.setResourceMetric(props.resourceMetric);
  });

  createEffect(() => {
    renderer?.setResourceMetricSecondary(props.resourceMetricSecondary);
  });

  createEffect(() => {
    renderer?.setShowResourceChart(props.showResourceChart);
  });

  createEffect(() => {
    props.themeMode;
    renderer?.rerender();
  });

  createEffect(() => {
    if (!renderer || !props.pinnedHit || props.pinnedHit.kind === "lane") {
      return;
    }
    props.onPinnedReanchor(renderer.locateHit(props.pinnedHit));
  });

  onCleanup(() => {
    renderer?.destroy();
    renderer = null;
  });

  return (
    <div class="canvas-panel">
      <div class="canvas-wrap" ref={wrapEl}>
        <canvas ref={canvasEl} />
      </div>
    </div>
  );
}
