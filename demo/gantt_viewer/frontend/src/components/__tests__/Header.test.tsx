import { render } from "solid-js/web";
import { afterEach, describe, expect, it } from "vitest";

import Header from "../Header";

function mountHeader() {
  const host = document.createElement("div");
  document.body.append(host);
  const dispose = render(
    () => (
      <Header
        clockMode={() => "wall"}
        onClockModeChange={() => undefined}
        onThemeModeChange={() => undefined}
        onTimeModeChange={() => undefined}
        onViewModeChange={() => undefined}
        themeMode={() => "dark"}
        onZoomChange={() => undefined}
        timeMode={() => "sync"}
        viewMode={() => "layered"}
        zoom={() => 1}
        resourceMetric={() => "cpu"}
        onResourceMetricChange={() => undefined}
        resourceMetricSecondary={() => "memory" as const}
        onResourceMetricSecondaryChange={() => undefined}
        showResourceChart={() => true}
        onShowResourceChartChange={() => undefined}
      />
    ),
    host,
  );
  return { dispose, host };
}

function headerText(host: HTMLElement): string {
  return host.textContent ?? "";
}

afterEach(() => {
  document.body.innerHTML = "";
});

describe("Header", () => {
  it("renders zoom and all toggle groups without legacy summary text", () => {
    const { dispose, host } = mountHeader();
    try {
      const header = host.querySelector("header");

      expect(header?.textContent).toContain("TRACE GANTT");
      expect(header?.textContent).not.toContain("actions total");
      expect(header?.textContent).not.toContain("loaded ");
      expect(headerText(host)).toContain("DARK");
      expect(headerText(host)).toContain("LIGHT");
      expect(headerText(host)).toContain("SYNC");
      expect(headerText(host)).toContain("ABS");
      expect(headerText(host)).toContain("WALL");
      expect(headerText(host)).toContain("REAL");
      expect(headerText(host)).toContain("LAYER");
      expect(headerText(host)).toContain("CONCISE");
      expect(headerText(host)).toContain("Disk Total");
      expect(headerText(host)).toContain("Disk Read");
      expect(headerText(host)).toContain("Disk Write");
      expect(headerText(host)).toContain("Mem Total");
      expect(headerText(host)).toContain("Mem Read");
      expect(headerText(host)).toContain("Mem Write");
      expect(headerText(host)).not.toContain("Net I/O");
    } finally {
      dispose();
    }
  });
});
