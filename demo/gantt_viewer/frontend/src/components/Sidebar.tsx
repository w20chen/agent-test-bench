import { For, Show, createEffect, createMemo } from "solid-js";

import type { TracePayload } from "../api/client";
import type { HitCard } from "../canvas/hit";
import { flattenVisibleLanes, resourceChartH, type LaneRow } from "../canvas/layout";
import type { ResourceMetric, ViewMode } from "../state/signals";

type SidebarEntry =
  | { kind: "lane"; row: LaneRow }
  | { kind: "spacer"; height: number; traceId: string };

interface SidebarProps {
  onPinLane: (card: HitCard) => void;
  onScroll?: (scrollTop: number) => void;
  scrollTop: number;
  traces: TracePayload[];
  visibility: Record<string, boolean>;
  showResourceChart: boolean;
  resourceMetric: ResourceMetric;
  resourceMetricSecondary: ResourceMetric;
  viewMode: ViewMode;
}

export default function Sidebar(props: SidebarProps) {
  const rows = createMemo(() => flattenVisibleLanes(props.traces, props.visibility));

  const entries = createMemo<SidebarEntry[]>(() => {
    const laneRows = rows();
    const result: SidebarEntry[] = [];
    for (let i = 0; i < laneRows.length; i++) {
      result.push({ kind: "lane", row: laneRows[i] });
      const isLastOfTrace =
        i === laneRows.length - 1 || laneRows[i + 1].traceId !== laneRows[i].traceId;
      const hasAnyMetric = props.resourceMetric !== "none" || props.resourceMetricSecondary !== "none";
      if (
        isLastOfTrace &&
        props.showResourceChart &&
        hasAnyMetric &&
        laneRows[i].trace.resource_timeline?.length
      ) {
        result.push({
          kind: "spacer",
          height: resourceChartH(props.viewMode),
          traceId: laneRows[i].traceId,
        });
      }
    }
    return result;
  });

  let shellEl!: HTMLElement;
  let suppressNextScroll = false;

  createEffect(() => {
    const target = props.scrollTop;
    if (!shellEl || shellEl.scrollTop === target) {
      return;
    }
    suppressNextScroll = true;
    shellEl.scrollTop = target;
  });

  const handleScroll = () => {
    if (suppressNextScroll) {
      suppressNextScroll = false;
      return;
    }
    props.onScroll?.(shellEl.scrollTop);
  };

  return (
    <aside class="sidebar-shell" onScroll={handleScroll} ref={shellEl}>
      <div class="sidebar-track">
        <div class="sidebar-time">TIME</div>
        <Show
          fallback={<div class="sidebar-empty">Load traces from the bar above.</div>}
          when={entries().length > 0}
        >
          <For each={entries()}>
            {(entry) =>
              entry.kind === "lane" ? (
                <button
                  class="lane-label"
                  onClick={(event) =>
                    props.onPinLane({
                      hit: {
                        kind: "lane",
                        laneAgentId: entry.row.laneAgentId,
                        trace: entry.row.trace,
                        traceId: entry.row.traceId,
                        traceLabel: entry.row.traceLabel,
                      },
                      x: event.clientX,
                      y: event.clientY,
                    })
                  }
                  type="button"
                >
                  <strong>{entry.row.traceLabel}</strong>
                  <span>{entry.row.trace.metadata.scaffold}</span>
                  <span>{entry.row.laneAgentId}</span>
                </button>
              ) : (
                <div
                  class="sidebar-resource-spacer"
                  style={{ height: `${entry.height}px` }}
                />
              )
            }
          </For>
        </Show>
      </div>
    </aside>
  );
}
