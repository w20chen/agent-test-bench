import { render } from "solid-js/web";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Registries, TraceDescriptor, TracePayload } from "../api/client";
import type { SnapshotBootstrapData } from "../bootstrap/snapshot";
import {
  __resetSignalsForTests,
  clockMode,
  descriptors as descriptorState,
  loadedTraces,
  resourceMetric,
  resourceMetricSecondary,
  setClockMode,
  setLoadedTraces,
  setRegistries,
  setThemeMode,
  setTimeMode,
  setViewMode,
  setVisibility,
  setZoom,
  showResourceChart,
  themeMode,
  timeMode,
  viewMode,
  visibility,
  zoom,
} from "../state/signals";

const apiClient = vi.hoisted(() => ({
  getPayload: vi.fn(),
  getTraces: vi.fn(),
  unregisterTraces: vi.fn(),
  uploadTrace: vi.fn(),
}));

const persist = vi.hoisted(() => ({
  enableDisplaySync: vi.fn(),
  enablePersistence: vi.fn(),
}));

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return {
    ...actual,
    getPayload: apiClient.getPayload,
    getTraces: apiClient.getTraces,
    unregisterTraces: apiClient.unregisterTraces,
    uploadTrace: apiClient.uploadTrace,
  };
});

vi.mock("../components/CanvasStage", () => ({
  default: () => <div data-testid="canvas-stage" />,
}));
vi.mock("../components/Legend", () => ({
  default: () => <div data-testid="legend" />,
}));
vi.mock("../components/Sidebar", () => ({
  default: () => <div data-testid="sidebar" />,
}));
vi.mock("../components/Tooltip", () => ({
  default: () => <div data-testid="tooltip" />,
}));
vi.mock("../state/persist", () => ({
  enableDisplaySync: persist.enableDisplaySync,
  enablePersistence: persist.enablePersistence,
}));

import App from "../App";

function flush(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function setSnapshotBootstrap(value: unknown): void {
  Object.defineProperty(window, "__GANTT_VIEWER_BOOTSTRAP__", {
    configurable: true,
    value,
    writable: true,
  });
}

function setSnapshotBootstrapScript(value: unknown): void {
  const existing = document.getElementById("gantt-viewer-snapshot-bootstrap");
  existing?.remove();
  const element = document.createElement("script");
  element.id = "gantt-viewer-snapshot-bootstrap";
  element.type = "application/json";
  element.textContent = JSON.stringify(value);
  document.body.append(element);
}

const baseRegistries: Registries = {
  markers: {
    done: { color: "#0f0", label: "Done", symbol: "circle" },
  },
  spans: {
    work: { color: "#00f", label: "Work", order: 1 },
  },
};

const baseTrace: TracePayload = {
  id: "trace-a",
  label: "Trace A",
  lanes: [],
  metadata: {
    instance_id: "instance-a",
    n_actions: 2,
    n_events: 2,
    n_iterations: 1,
    scaffold: "openclaw",
  },
  t0: 0,
};

const descriptors: TraceDescriptor[] = [
  {
    id: baseTrace.id,
    label: baseTrace.label,
    mtime: 0,
    path: "/tmp/trace-a.jsonl",
    size_bytes: 10,
    source_format: "trace",
  },
];

function createSnapshotTrace(id: string, label: string, instanceId: string): TracePayload {
  return {
    ...baseTrace,
    id,
    label,
    metadata: {
      ...baseTrace.metadata,
      instance_id: instanceId,
    },
  };
}

function createSnapshotBootstrap(traces: TracePayload[]): SnapshotBootstrapData {
  const traceIds = traces.map((trace) => trace.id);
  return {
    mode: "snapshot" as const,
    payload: {
      errors: [],
      registries: baseRegistries,
      traces,
    },
    trace_ids: traceIds,
    visible_trace_ids: traceIds,
  };
}

function traceChipLabels(host: HTMLElement): string[] {
  return Array.from(host.querySelectorAll(".trace-chip-main"), (button) => button.textContent ?? "");
}

function mountApp() {
  const host = document.createElement("div");
  document.body.append(host);
  const dispose = render(() => <App />, host);
  return { dispose, host };
}

beforeEach(() => {
  setSnapshotBootstrap(undefined);
  __resetSignalsForTests();
  apiClient.getTraces.mockReset();
  apiClient.getPayload.mockReset();
  apiClient.uploadTrace.mockReset();
  apiClient.unregisterTraces.mockReset();
  persist.enableDisplaySync.mockReset();
  persist.enablePersistence.mockReset();
  apiClient.getTraces.mockResolvedValue({ traces: descriptors, registries: baseRegistries });
  apiClient.getPayload.mockResolvedValue({ traces: [], registries: baseRegistries });
  apiClient.uploadTrace.mockResolvedValue(undefined);
  apiClient.unregisterTraces.mockResolvedValue({ missing_ids: [], removed_ids: [baseTrace.id] });
});

afterEach(() => {
  document.body.innerHTML = "";
});

describe("App", () => {
  it("boots live mode with upload controls and persistence", async () => {
    const { dispose, host } = mountApp();
    try {
      await flush();

      expect(apiClient.getTraces).toHaveBeenCalledOnce();
      expect(host.textContent).toContain("+ JSONL");
      expect(host.textContent).toContain("Load all");
      expect(traceChipLabels(host)).toEqual([baseTrace.label]);
      expect(persist.enableDisplaySync).not.toHaveBeenCalled();
      expect(persist.enablePersistence).toHaveBeenCalledOnce();

      window.dispatchEvent(new Event("dragenter", { bubbles: true }));
      await flush();
      expect(host.textContent).toContain("Drop JSONL");
    } finally {
      dispose();
    }
  });

  it("boots snapshot mode without live-only controls and preserves trace order", async () => {
    const snapshotTraceB = createSnapshotTrace("trace-b", "Trace B", "instance-b");
    setSnapshotBootstrap(createSnapshotBootstrap([snapshotTraceB, baseTrace]));
    setThemeMode("light");
    setClockMode("real");
    setTimeMode("abs");
    setViewMode("concise");
    setZoom(2);

    const { dispose, host } = mountApp();
    try {
      const activeButtons = Array.from(host.querySelectorAll(".toggle-group button.active"));
      expect(activeButtons.map((button) => button.textContent)).toEqual(["LIGHT", "SYNC", "WALL", "LAYER", "RES"]);

      await flush();

      expect(traceChipLabels(host)).toEqual(["Trace B", "Trace A"]);
      expect(host.textContent).not.toContain("+ JSONL");
      expect(host.textContent).not.toContain("Load all");

      window.dispatchEvent(new Event("dragenter", { bubbles: true }));
      await flush();
      expect(host.textContent).not.toContain("Drop JSONL");

      expect(apiClient.getTraces).not.toHaveBeenCalled();
      expect(apiClient.getPayload).not.toHaveBeenCalled();
      expect(descriptorState().map((descriptor) => descriptor.id)).toEqual(["trace-b", "trace-a"]);
      expect(loadedTraces().map((trace) => trace.id)).toEqual(["trace-b", "trace-a"]);
      expect(visibility()).toEqual({ "trace-b": true, "trace-a": true });
      expect(themeMode()).toBe("light");
      expect(timeMode()).toBe("sync");
      expect(viewMode()).toBe("layered");
      expect(zoom()).toBe(1);
      expect(persist.enableDisplaySync).toHaveBeenCalledOnce();
      expect(persist.enablePersistence).not.toHaveBeenCalled();
    } finally {
      dispose();
    }
  });

  it("applies embedded snapshot display defaults", async () => {
    const snapshot = createSnapshotBootstrap([baseTrace]);
    snapshot.display = {
      clockMode: "real",
      resourceMetric: "disk_total",
      resourceMetricSecondary: "none",
      showResourceChart: false,
      themeMode: "light",
      timeMode: "abs",
      viewMode: "concise",
      zoom: 4,
    };
    setSnapshotBootstrap(snapshot);

    const { dispose } = mountApp();
    try {
      await flush();

      expect(clockMode()).toBe("real");
      expect(resourceMetric()).toBe("disk_total");
      expect(resourceMetricSecondary()).toBe("none");
      expect(showResourceChart()).toBe(false);
      expect(themeMode()).toBe("light");
      expect(timeMode()).toBe("abs");
      expect(viewMode()).toBe("concise");
      expect(zoom()).toBe(4);
      expect(apiClient.getTraces).not.toHaveBeenCalled();
      expect(apiClient.getPayload).not.toHaveBeenCalled();
    } finally {
      dispose();
    }
  });

  it("falls back from invalid embedded snapshot display values", async () => {
    const snapshot = createSnapshotBootstrap([baseTrace]);
    snapshot.display = {
      clockMode: "invalid",
      resourceMetric: "invalid",
      resourceMetricSecondary: "invalid",
      showResourceChart: "yes",
      themeMode: "invalid",
      timeMode: "invalid",
      viewMode: "invalid",
      zoom: -1,
    } as unknown as SnapshotBootstrapData["display"];
    setSnapshotBootstrap(snapshot);

    const { dispose } = mountApp();
    try {
      await flush();

      expect(clockMode()).toBe("wall");
      expect(resourceMetric()).toBe("cpu");
      expect(resourceMetricSecondary()).toBe("memory");
      expect(showResourceChart()).toBe(true);
      expect(themeMode()).toBe("light");
      expect(timeMode()).toBe("sync");
      expect(viewMode()).toBe("layered");
      expect(zoom()).toBe(1);
    } finally {
      dispose();
    }
  });

  it("boots from the embedded snapshot script payload without calling live APIs", async () => {
    const snapshotTraceB = createSnapshotTrace("trace-b", "Trace B", "instance-b");
    setSnapshotBootstrapScript(createSnapshotBootstrap([snapshotTraceB, baseTrace]));

    const { dispose, host } = mountApp();
    try {
      await flush();

      expect(traceChipLabels(host)).toEqual(["Trace B", "Trace A"]);
      expect(apiClient.getTraces).not.toHaveBeenCalled();
      expect(apiClient.getPayload).not.toHaveBeenCalled();
      expect(persist.enableDisplaySync).toHaveBeenCalledOnce();
      expect(persist.enablePersistence).not.toHaveBeenCalled();
      expect(window.__GANTT_VIEWER_BOOTSTRAP__).toMatchObject({
        mode: "snapshot",
        trace_ids: ["trace-b", "trace-a"],
      });
    } finally {
      dispose();
    }
  });

  it("unregisters traces when remove is clicked and clears the chip", async () => {
    setRegistries(baseRegistries);
    setLoadedTraces([baseTrace]);
    setVisibility({ [baseTrace.id]: true });

    const { dispose, host } = mountApp();
    try {
      await flush();

      const removeButton = host.querySelector("button.trace-remove") as HTMLButtonElement;
      expect(removeButton).not.toBeNull();

      removeButton.click();
      await flush();

      expect(apiClient.unregisterTraces).toHaveBeenCalledWith([baseTrace.id]);
      expect(traceChipLabels(host)).toEqual([]);
      expect(host.textContent).not.toContain(baseTrace.label);
    } finally {
      dispose();
    }
  });
});
