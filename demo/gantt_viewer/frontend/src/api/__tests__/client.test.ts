import { afterEach, describe, expect, it, vi } from "vitest";

import { unregisterTraces } from "../client";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("unregisterTraces", () => {
  it("posts ids to the unregister endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({ missing_ids: [], removed_ids: ["trace-a"] }),
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(unregisterTraces(["trace-a"])).resolves.toEqual({
      missing_ids: [],
      removed_ids: ["trace-a"],
    });
    expect(fetchMock).toHaveBeenCalledWith("/api/traces/unregister", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ ids: ["trace-a"] }),
    });
  });
});
