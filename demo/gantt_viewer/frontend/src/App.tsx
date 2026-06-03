import { Match, Show, Switch, createMemo, onCleanup, onMount } from "solid-js";

import {
  getPayload,
  getTraces,
  unregisterTraces,
  uploadTrace,
  type GanttPayload,
  type TraceDescriptor,
  type TracePayload,
} from "./api/client";
import CanvasStage from "./components/CanvasStage";
import DropZone from "./components/DropZone";
import Header from "./components/Header";
import Legend from "./components/Legend";
import Sidebar from "./components/Sidebar";
import Tooltip from "./components/Tooltip";
import AggregateResourceBar from "./components/AggregateResourceBar";
import TraceChipBar from "./components/TraceChipBar";
import { selectInitialTraceIds } from "./bootstrap/autoload";
import {
  normalizeSnapshotDisplay,
  readSnapshotBootstrap,
  snapshotDescriptorsFromTraces,
  visibilityFromTraceIds,
} from "./bootstrap/snapshot";
import { sameHit } from "./canvas/hit";
import { enableDisplaySync, enablePersistence } from "./state/persist";
import {
  appError,
  clockMode,
  descriptors,
  hoverCard,
  loadedTraces,
  loadingIds,
  pinnedCard,
  registries,
  scrollTop,
  setAppError,
  setClockMode,
  setDescriptors,
  setHoverCard,
  setLoadedTraces,
  setLoadingIds,
  setPinnedCard,
  setRegistries,
  setScrollTop,
  setThemeMode,
  setTimeMode,
  setViewMode,
  setVisibility,
  setZoom,
  themeMode,
  timeMode,
  viewMode,
  visibility,
  zoom,
  resourceMetric,
  setResourceMetric,
  resourceMetricSecondary,
  setResourceMetricSecondary,
  showResourceChart,
  setShowResourceChart,
} from "./state/signals";

function mergeById<T extends { id: string }>(existing: T[], incoming: T[]): T[] {
  const byId = new Map(existing.map((item) => [item.id, item]));
  incoming.forEach((item) => byId.set(item.id, item));
  return [...byId.values()];
}

function mergeTraces(existing: TracePayload[], incoming: TracePayload[]): TracePayload[] {
  return mergeById(existing, incoming);
}

function mergeDescriptors(existing: TraceDescriptor[], incoming: TraceDescriptor[]): TraceDescriptor[] {
  return mergeById(existing, incoming).sort((left, right) => left.id.localeCompare(right.id));
}

function mapIds(ids: string[], value: boolean): Record<string, boolean> {
  return Object.fromEntries(ids.map((id) => [id, value]));
}

function removeIdFlag(flags: Record<string, boolean>, id: string): Record<string, boolean> {
  const next = { ...flags };
  delete next[id];
  return next;
}

function applySnapshotBootstrap(snapshotBootstrap: NonNullable<ReturnType<typeof readSnapshotBootstrap>>): void {
  const display = normalizeSnapshotDisplay(snapshotBootstrap.display);
  setClockMode(display.clockMode);
  setThemeMode(display.themeMode);
  setTimeMode(display.timeMode);
  setViewMode(display.viewMode);
  setZoom(display.zoom);
  setResourceMetric(display.resourceMetric);
  setResourceMetricSecondary(display.resourceMetricSecondary);
  setShowResourceChart(display.showResourceChart);
  setDescriptors(snapshotDescriptorsFromTraces(snapshotBootstrap.payload.traces));
  setRegistries(snapshotBootstrap.payload.registries);
  setLoadedTraces(snapshotBootstrap.payload.traces);
  setVisibility(
    visibilityFromTraceIds(snapshotBootstrap.trace_ids, snapshotBootstrap.visible_trace_ids),
  );
  setLoadingIds({});
  setAppError(null);
}

export default function App() {
  const snapshotBootstrap = readSnapshotBootstrap();
  const snapshotMode = snapshotBootstrap !== null;
  if (snapshotBootstrap) {
    enableDisplaySync();
    applySnapshotBootstrap(snapshotBootstrap);
  } else {
    enablePersistence();
  }

  const payload = createMemo<GanttPayload | null>(() => {
    const currentRegistries = registries();
    if (!currentRegistries) {
      return null;
    }
    return {
      registries: currentRegistries,
      traces: loadedTraces(),
    };
  });

  const loadedIds = createMemo(() => loadedTraces().map((trace) => trace.id));

  async function initialize(): Promise<void> {
    if (snapshotBootstrap) {
      return;
    }

    try {
      const traceList = await getTraces();
      setDescriptors(traceList.traces);
      setRegistries(traceList.registries);
      const initialIds = selectInitialTraceIds(traceList.traces, window.location.search);
      if (initialIds.length > 0) {
        await loadTraceIds(initialIds);
      }
    } catch (error) {
      setAppError(String(error));
    }
  }

  async function loadTraceIds(ids: string[]): Promise<void> {
    const currentlyLoaded = new Set(loadedIds());
    const inFlight = loadingIds();
    const nextIds = ids.filter((id) => !currentlyLoaded.has(id) && !inFlight[id]);
    if (nextIds.length === 0) {
      return;
    }

    setLoadingIds((current) => ({
      ...current,
      ...mapIds(nextIds, true),
    }));

    try {
      const nextPayload = await getPayload(nextIds);
      setRegistries(nextPayload.registries);
      setLoadedTraces((current) => mergeTraces(current, nextPayload.traces));
      setVisibility((current) => ({
        ...current,
        ...mapIds(
          nextPayload.traces.map((trace) => trace.id),
          true,
        ),
      }));
      setAppError(null);
    } catch (error) {
      setAppError(String(error));
    } finally {
      setLoadingIds((current) => nextIds.reduce(removeIdFlag, current));
    }
  }

  async function removeTrace(id: string): Promise<void> {
    setLoadedTraces((current) => current.filter((trace) => trace.id !== id));
    setVisibility((current) => removeIdFlag(current, id));
    setDescriptors((current) => current.filter((descriptor) => descriptor.id !== id));
    if (pinnedCard()?.hit.traceId === id) {
      setPinnedCard(null);
    }
    if (hoverCard()?.hit.traceId === id) {
      setHoverCard(null);
    }
    try {
      await unregisterTraces([id]);
      setAppError(null);
    } catch (error) {
      console.error("Failed to unregister gantt trace", id, error);
      setAppError(String(error));
    }
  }

  function toggleVisibility(id: string): void {
    setVisibility((current) => ({
      ...current,
      [id]: current[id] === false,
    }));
  }

  async function handleUpload(files: File[]): Promise<void> {
    for (const file of files) {
      try {
        const { descriptor, payload_fragment } = await uploadTrace(file);
        setDescriptors((current) => mergeDescriptors(current, [descriptor]));
        setLoadedTraces((current) => mergeTraces(current, [payload_fragment]));
        setVisibility((current) => ({
          ...current,
          [descriptor.id]: true,
        }));
        setAppError(null);
      } catch (error) {
        setAppError(String(error));
      }
    }
  }

  onMount(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setPinnedCard(null);
      }
    };

    void initialize();
    document.addEventListener("keydown", onKeyDown);
    onCleanup(() => document.removeEventListener("keydown", onKeyDown));
  });

  return (
    <main class="app-shell">
      <Header
        clockMode={clockMode}
        onClockModeChange={setClockMode}
        onThemeModeChange={setThemeMode}
        onTimeModeChange={setTimeMode}
        onViewModeChange={setViewMode}
        themeMode={themeMode}
        onZoomChange={setZoom}
        timeMode={timeMode}
        viewMode={viewMode}
        zoom={zoom}
        resourceMetric={resourceMetric}
        onResourceMetricChange={setResourceMetric}
        resourceMetricSecondary={resourceMetricSecondary}
        onResourceMetricSecondaryChange={setResourceMetricSecondary}
        showResourceChart={showResourceChart}
        onShowResourceChartChange={setShowResourceChart}
      />

      <Legend registries={registries()} />

      <TraceChipBar
        descriptors={descriptors()}
        loadedIds={loadedIds()}
        loadingIds={loadingIds()}
        onLoad={loadTraceIds}
        onUpload={handleUpload}
        onRemove={removeTrace}
        onToggleVisibility={toggleVisibility}
        snapshotMode={snapshotMode}
        visibility={visibility()}
      />

      <Show when={appError()}>
        <section class="error-banner">{appError()}</section>
      </Show>

      <section class="workspace-card">
        <Sidebar
          onPinLane={(card) => setPinnedCard(card)}
          onScroll={setScrollTop}
          scrollTop={scrollTop()}
          traces={loadedTraces()}
          visibility={visibility()}
          showResourceChart={showResourceChart()}
          resourceMetric={resourceMetric()}
          resourceMetricSecondary={resourceMetricSecondary()}
          viewMode={viewMode()}
        />
        <CanvasStage
          onClick={(card) =>
            setPinnedCard((current) => {
              if (!card) {
                return null;
              }
              return current && sameHit(current.hit, card.hit) ? null : card;
            })
          }
          onHover={(card) => {
            if (!pinnedCard()) {
              setHoverCard(card);
            }
          }}
          onPinnedReanchor={(card) =>
            setPinnedCard((current) => {
              if (!current || current.hit.kind === "lane") {
                return current;
              }
              return card;
            })
          }
          onScroll={setScrollTop}
          onZoom={setZoom}
          clockMode={clockMode()}
          payload={payload()}
          pinnedHit={pinnedCard()?.hit ?? null}
          resourceMetric={resourceMetric()}
          resourceMetricSecondary={resourceMetricSecondary()}
          showResourceChart={showResourceChart()}
          themeMode={themeMode()}
          timeMode={timeMode()}
          viewMode={viewMode()}
          visibility={visibility()}
          zoom={zoom()}
        />
      </section>

      <AggregateResourceBar
        traces={loadedTraces()}
        visibility={visibility()}
        resourceMetric={resourceMetric()}
        resourceMetricSecondary={resourceMetricSecondary()}
        showResourceChart={showResourceChart()}
        timeMode={timeMode()}
        clockMode={clockMode()}
        themeMode={themeMode()}
      />

      <DropZone enabled={!snapshotMode} onUpload={handleUpload} />

      <Switch>
        <Match when={pinnedCard()}>
          <Tooltip
            card={pinnedCard()}
            onClose={() => setPinnedCard(null)}
            pinned={true}
            registries={registries()}
          />
        </Match>
        <Match when={hoverCard()}>
          <Tooltip
            card={hoverCard()}
            onClose={() => setHoverCard(null)}
            pinned={false}
            registries={registries()}
          />
        </Match>
      </Switch>
    </main>
  );
}
