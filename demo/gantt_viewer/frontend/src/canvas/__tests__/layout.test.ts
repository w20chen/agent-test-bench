import { describe, expect, it } from "vitest";

import {
  CONTROL_FLOW_SPAN_TYPES,
  LANE_H,
  MARKER_RESERVED_H,
  MIN_SPAN_H,
  MIN_TOOL_TRACK_H,
  SPAN_H,
  SPAN_PAD,
  computeTrackLayout,
} from "../layout";

describe("CONTROL_FLOW_SPAN_TYPES", () => {
  it("contains llm, scheduling, mcp", () => {
    expect(CONTROL_FLOW_SPAN_TYPES.has("llm")).toBe(true);
    expect(CONTROL_FLOW_SPAN_TYPES.has("scheduling")).toBe(true);
    expect(CONTROL_FLOW_SPAN_TYPES.has("mcp")).toBe(true);
    expect(CONTROL_FLOW_SPAN_TYPES.has("tool")).toBe(false);
  });
});

describe("layout constants", () => {
  it("MIN_TOOL_TRACK_H floor is high enough to show a span + gap", () => {
    expect(MIN_TOOL_TRACK_H).toBeGreaterThanOrEqual(4);
    expect(MIN_SPAN_H).toBeGreaterThanOrEqual(3);
  });

  it("MARKER_RESERVED_H leaves room for marker + padding", () => {
    expect(MARKER_RESERVED_H).toBeGreaterThanOrEqual(6);
  });
});

describe("computeTrackLayout", () => {
  it("places top strip at SPAN_PAD and tool strip below it", () => {
    const layout = computeTrackLayout(1, 2, LANE_H);
    expect(layout.topStripY).toBe(SPAN_PAD);
    expect(layout.toolStripY).toBe(SPAN_PAD + SPAN_H + 2);
    expect(layout.topTrackH).toBe(SPAN_H);
  });

  it("reserves top-strip height for each control-flow track", () => {
    const layout = computeTrackLayout(3, 1, LANE_H);
    expect(layout.toolStripY).toBe(SPAN_PAD + 3 * (SPAN_H + 2));
  });

  it("matches the LLM row height for a single tool track (N=1)", () => {
    // Sequential tools (N=1): keep SPAN_H height so tool row matches LLM row.
    const layout = computeTrackLayout(1, 1, LANE_H);
    expect(layout.toolTrackH).toBeCloseTo(SPAN_H + 2, 1);
    expect(layout.toolSpanH).toBeCloseTo(SPAN_H, 1);
  });

  it("splits the tool strip in half for 2 concurrent tracks", () => {
    const layout = computeTrackLayout(1, 2, LANE_H);
    // 24 / 2 = 12 per track
    expect(layout.toolTrackH).toBeCloseTo(12, 1);
    expect(layout.toolSpanH).toBeCloseTo(10, 1);
  });

  it("compresses tool tracks when toolTrackCount = 3 (research-agent fetch)", () => {
    const layout = computeTrackLayout(0, 3, LANE_H);
    // With no control-flow strip reserved, tools get the full drawable height.
    expect(layout.toolTrackH).toBeCloseTo(14.666, 1);
    expect(layout.toolSpanH).toBeCloseTo(12.666, 1);
    expect(layout.toolSpanH).toBeGreaterThanOrEqual(MIN_SPAN_H);
  });

  it("keeps tool tracks >= MIN_TOOL_TRACK_H when concurrency is very high", () => {
    const layout = computeTrackLayout(1, 20, LANE_H);
    expect(layout.toolTrackH).toBeGreaterThanOrEqual(MIN_TOOL_TRACK_H);
    expect(layout.toolSpanH).toBeGreaterThanOrEqual(MIN_SPAN_H);
  });

  it("returns only finite, non-negative values", () => {
    for (const [top, tool] of [[0, 0], [1, 1], [1, 3], [0, 5], [2, 8], [1, 50]]) {
      const layout = computeTrackLayout(top, tool, LANE_H);
      for (const value of Object.values(layout)) {
        expect(Number.isFinite(value)).toBe(true);
        expect(value).toBeGreaterThanOrEqual(0);
      }
    }
  });

  it("fits 3 tool tracks within the lane without overflowing the marker zone", () => {
    const layout = computeTrackLayout(1, 3, LANE_H);
    const bottom = layout.toolStripY + 3 * layout.toolTrackH;
    // Must stay above the marker-reserved zone
    expect(bottom).toBeLessThanOrEqual(LANE_H - MARKER_RESERVED_H + 0.5);
  });

  it("fits 5 tool tracks via compression (marginal but still bounded)", () => {
    const layout = computeTrackLayout(1, 5, LANE_H);
    const bottom = layout.toolStripY + 5 * layout.toolTrackH;
    // When compression keeps tracks at natural size, still bounded
    expect(bottom).toBeLessThanOrEqual(LANE_H - MARKER_RESERVED_H + 0.5);
  });

  it("keeps the real-trace DRB iter=2 (3 fetches) bottom above the marker zone", () => {
    // DRB iter=2 has 5 fetches with max_concurrent=3 → 3 tracks in tool strip.
    const layout = computeTrackLayout(0, 3, LANE_H);
    const bottom = layout.toolStripY + 3 * layout.toolTrackH;
    // Must not overlap marker zone (laneY + LANE_H - MARKER_RESERVED_H)
    expect(bottom).toBeLessThanOrEqual(LANE_H - MARKER_RESERVED_H + 0.5);
    // And the last track's span should remain visible (>= MIN_SPAN_H)
    expect(layout.toolSpanH).toBeGreaterThanOrEqual(MIN_SPAN_H);
  });

  it("fits within lane at typical concurrency (N <= 5) and keeps spans visible at extreme N", () => {
    // Typical case: research-agent / most scaffolds stay within N <= 5.
    for (let n = 1; n <= 5; n++) {
      const layout = computeTrackLayout(1, n, LANE_H);
      const bottom = layout.toolStripY + n * layout.toolTrackH;
      expect(bottom).toBeLessThanOrEqual(LANE_H - MARKER_RESERVED_H + 0.5);
    }
    // Extreme case: MIN_TOOL_TRACK_H floor engages, lane may bleed into
    // marker zone but every span stays >= MIN_SPAN_H. This is the documented
    // graceful-degradation tradeoff.
    const extreme = computeTrackLayout(1, 20, LANE_H);
    expect(extreme.toolSpanH).toBeGreaterThanOrEqual(MIN_SPAN_H);
    expect(extreme.toolTrackH).toBeGreaterThanOrEqual(MIN_TOOL_TRACK_H);
  });
});
