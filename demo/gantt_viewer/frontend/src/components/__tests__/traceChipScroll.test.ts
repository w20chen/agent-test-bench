import { describe, expect, it } from "vitest";

import { scrollTraceChipGrid } from "../traceChipScroll";

describe("scrollTraceChipGrid", () => {
  it("scrolls horizontally using vertical wheel delta when overflow exists", () => {
    const element = {
      scrollWidth: 600,
      clientWidth: 300,
      scrollLeft: 0,
    };

    const consumed = scrollTraceChipGrid(element, 0, 120);

    expect(consumed).toBe(true);
    expect(element.scrollLeft).toBe(120);
  });

  it("prefers horizontal wheel delta when it dominates", () => {
    const element = {
      scrollWidth: 600,
      clientWidth: 300,
      scrollLeft: 10,
    };

    const consumed = scrollTraceChipGrid(element, 80, 20);

    expect(consumed).toBe(true);
    expect(element.scrollLeft).toBe(90);
  });

  it("does nothing when there is no horizontal overflow", () => {
    const element = {
      scrollWidth: 300,
      clientWidth: 300,
      scrollLeft: 15,
    };

    const consumed = scrollTraceChipGrid(element, 0, 120);

    expect(consumed).toBe(false);
    expect(element.scrollLeft).toBe(15);
  });
});
