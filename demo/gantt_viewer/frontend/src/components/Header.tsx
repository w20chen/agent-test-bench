import { For, type Accessor } from "solid-js";

import { ZOOM_PRESETS } from "../canvas/CanvasRenderer";
import type { ClockMode, ResourceMetric, ThemeMode, TimeMode, ViewMode } from "../state/signals";

interface HeaderProps {
  clockMode: Accessor<ClockMode>;
  onClockModeChange: (mode: ClockMode) => void;
  onThemeModeChange: (mode: ThemeMode) => void;
  onTimeModeChange: (mode: TimeMode) => void;
  onViewModeChange: (mode: ViewMode) => void;
  themeMode: Accessor<ThemeMode>;
  onZoomChange: (factor: number) => void;
  timeMode: Accessor<TimeMode>;
  viewMode: Accessor<ViewMode>;
  zoom: Accessor<number>;
  resourceMetric: Accessor<ResourceMetric>;
  onResourceMetricChange: (metric: ResourceMetric) => void;
  resourceMetricSecondary: Accessor<ResourceMetric>;
  onResourceMetricSecondaryChange: (metric: ResourceMetric) => void;
  showResourceChart: Accessor<boolean>;
  onShowResourceChartChange: (show: boolean) => void;
}

function getPresetValue(zoom: number): string {
  const matchingPreset = ZOOM_PRESETS.find((preset) => Math.abs(preset - zoom) < 0.001);
  return matchingPreset === undefined ? "" : String(matchingPreset);
}

export default function Header(props: HeaderProps) {
  const currentPresetValue = () => getPresetValue(props.zoom());
  const currentZoomLabel = () => `${Math.round(props.zoom() * 100)}%`;

  function handleZoomChange(event: Event): void {
    const value = (event.currentTarget as HTMLSelectElement).value;
    if (value !== "") {
      props.onZoomChange(Number(value));
    }
  }

  return (
    <header class="toolbar-card">
      <span class="toolbar-title">TRACE GANTT</span>
      <div class="toolbar-spacer" />
      <label class="zoom-select-wrap" title="Zoom presets (Ctrl+wheel for free zoom)">
        <span class="zoom-select-label">zoom</span>
        <select
          class="zoom-select"
          value={currentPresetValue()}
          onChange={handleZoomChange}
        >
          {currentPresetValue() === "" && (
            <option value="" disabled>
              {currentZoomLabel()}
            </option>
          )}
          <For each={ZOOM_PRESETS}>
            {(preset) => <option value={String(preset)}>{`${preset}x`}</option>}
          </For>
        </select>
      </label>
      <div class="toggle-group">
        <button
          classList={{ active: props.themeMode() === "dark" }}
          onClick={() => props.onThemeModeChange("dark")}
          type="button"
        >
          DARK
        </button>
        <button
          classList={{ active: props.themeMode() === "light" }}
          onClick={() => props.onThemeModeChange("light")}
          type="button"
        >
          LIGHT
        </button>
      </div>
      <div class="toggle-group">
        <button
          classList={{ active: props.timeMode() === "sync" }}
          onClick={() => props.onTimeModeChange("sync")}
          type="button"
        >
          SYNC
        </button>
        <button
          classList={{ active: props.timeMode() === "abs" }}
          onClick={() => props.onTimeModeChange("abs")}
          type="button"
        >
          ABS
        </button>
      </div>
      <div class="toggle-group">
        <button
          classList={{ active: props.clockMode() === "wall" }}
          onClick={() => props.onClockModeChange("wall")}
          type="button"
        >
          WALL
        </button>
        <button
          classList={{ active: props.clockMode() === "real" }}
          onClick={() => props.onClockModeChange("real")}
          type="button"
        >
          REAL
        </button>
      </div>
      <div class="toggle-group">
        <button
          classList={{ active: props.viewMode() === "layered" }}
          onClick={() => props.onViewModeChange("layered")}
          type="button"
        >
          LAYER
        </button>
        <button
          classList={{ active: props.viewMode() === "concise" }}
          onClick={() => props.onViewModeChange("concise")}
          type="button"
        >
          CONCISE
        </button>
      </div>
      <label class="zoom-select-wrap" title="Primary resource metric">
        <span class="zoom-select-label" style={{ color: "#00E5FF" }}>res1</span>
        <select
          class="zoom-select"
          value={props.resourceMetric()}
          onChange={(e) =>
            props.onResourceMetricChange(
              (e.currentTarget as HTMLSelectElement).value as ResourceMetric,
            )
          }
        >
          <option value="none">None</option>
          <option value="cpu">CPU %</option>
          <option value="memory">Memory</option>
          <option value="mem_total">Mem Total</option>
          <option value="mem_read">Mem Read</option>
          <option value="mem_write">Mem Write</option>
          <option value="disk_total">Disk Total</option>
          <option value="disk_read">Disk Read</option>
          <option value="disk_write">Disk Write</option>
        </select>
      </label>
      <label class="zoom-select-wrap" title="Secondary resource metric">
        <span class="zoom-select-label" style={{ color: "#76FF03" }}>res2</span>
        <select
          class="zoom-select"
          value={props.resourceMetricSecondary()}
          onChange={(e) =>
            props.onResourceMetricSecondaryChange(
              (e.currentTarget as HTMLSelectElement).value as ResourceMetric,
            )
          }
        >
          <option value="none">None</option>
          <option value="cpu">CPU %</option>
          <option value="memory">Memory</option>
          <option value="mem_total">Mem Total</option>
          <option value="mem_read">Mem Read</option>
          <option value="mem_write">Mem Write</option>
          <option value="disk_total">Disk Total</option>
          <option value="disk_read">Disk Read</option>
          <option value="disk_write">Disk Write</option>
        </select>
      </label>
      <div class="toggle-group">
        <button
          classList={{ active: props.showResourceChart() }}
          onClick={() => props.onShowResourceChartChange(!props.showResourceChart())}
          type="button"
        >
          RES
        </button>
      </div>
    </header>
  );
}
