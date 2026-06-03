import { describe, expect, it } from "vitest";

import { displayColor } from "../displayColor";

describe("displayColor", () => {
  it("tones down llm cyan in light theme", () => {
    document.documentElement.classList.add("theme-light");
    expect(displayColor("#00E5FF")).toBe("#0b6e95");
    document.documentElement.classList.remove("theme-light");
  });

  it("preserves original colors in dark theme", () => {
    document.documentElement.classList.remove("theme-light");
    expect(displayColor("#00E5FF")).toBe("#00E5FF");
  });
});
