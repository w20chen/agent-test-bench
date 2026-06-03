import { describe, expect, it } from "vitest";

import { assignTracks } from "../tracks";

interface Action {
  type: string;
  action_type?: string;
  iteration?: number;
  ts_start?: number;
  ts_end?: number;
}

function action(startTs: number, endTs: number): Action {
  return { type: "action", action_type: "tool_exec", ts_start: startTs, ts_end: endTs };
}

const sequentialSearches = [
  action(0, 1),
  action(1, 2),
  action(2, 3),
  action(3, 4),
  action(4, 5),
];

const concurrentFetches = [
  action(0, 5),
  action(1, 4),
  action(2, 6),
  action(3, 7),
  action(4, 8),
];

const start = (a: Action) => a.ts_start ?? 0;
const end = (a: Action) => a.ts_end ?? 0;

describe("assignTracks against representative research traces", () => {
  it("places all 5 sequential searches on track 0", () => {
    const tracks = assignTracks(sequentialSearches, start, end);
    expect(tracks).toEqual([0, 0, 0, 0, 0]);
  });

  it("distributes representative concurrent fetches across >= 3 distinct tracks", () => {
    const tracks = assignTracks(concurrentFetches, start, end);
    expect(new Set(tracks).size).toBeGreaterThanOrEqual(3);
  });
});
