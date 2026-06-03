import { describe, expect, it } from "vitest";

import { assignTracks } from "../tracks";

interface Span {
  start: number;
  end: number;
}

const start = (s: Span) => s.start;
const end = (s: Span) => s.end;

describe("assignTracks", () => {
  it("returns empty array for empty input", () => {
    expect(assignTracks<Span>([], start, end)).toEqual([]);
  });

  it("places sequential non-overlapping spans all on track 0", () => {
    const spans: Span[] = [
      { start: 0, end: 5 },
      { start: 5, end: 10 },
      { start: 10, end: 15 },
      { start: 15, end: 20 },
      { start: 20, end: 25 },
    ];
    expect(assignTracks(spans, start, end)).toEqual([0, 0, 0, 0, 0]);
  });

  it("places fully concurrent spans on separate tracks", () => {
    const spans: Span[] = [
      { start: 0, end: 10 },
      { start: 0, end: 10 },
      { start: 0, end: 10 },
    ];
    expect(assignTracks(spans, start, end)).toEqual([0, 1, 2]);
  });

  it("reuses released tracks for partially overlapping spans (3+2 fetch pattern)", () => {
    const spans: Span[] = [
      { start: 0, end: 5 },
      { start: 0, end: 8 },
      { start: 0, end: 5 },
      { start: 5, end: 9 },
      { start: 5, end: 15 },
    ];
    const result = assignTracks(spans, start, end);

    // 3 concurrent spans at start → need at least 3 distinct tracks
    expect(new Set(result).size).toBeGreaterThanOrEqual(3);
    expect(result.length).toBe(5);

    // Verify no two spans on the same track overlap
    const byTrack = new Map<number, Span[]>();
    spans.forEach((s, i) => {
      const arr = byTrack.get(result[i]) ?? [];
      arr.push(s);
      byTrack.set(result[i], arr);
    });
    for (const trackSpans of byTrack.values()) {
      trackSpans.sort((a, b) => a.start - b.start);
      for (let i = 1; i < trackSpans.length; i++) {
        expect(trackSpans[i].start).toBeGreaterThanOrEqual(trackSpans[i - 1].end);
      }
    }

    // Track count should be exactly 3 (minimum for this conflict graph)
    expect(new Set(result).size).toBe(3);
  });

  it("treats adjacent zero-gap spans (end == next start) as non-overlapping", () => {
    const spans: Span[] = [
      { start: 0, end: 5 },
      { start: 5, end: 10 },
    ];
    expect(assignTracks(spans, start, end)).toEqual([0, 0]);
  });

  it("is deterministic for identical inputs with same start/end", () => {
    const spans: Span[] = [
      { start: 0, end: 10 },
      { start: 0, end: 10 },
    ];
    const a = assignTracks(spans, start, end);
    const b = assignTracks(spans, start, end);
    expect(a).toEqual(b);
    // Same start+end but need distinct tracks (they truly overlap)
    expect(a).toEqual([0, 1]);
  });

  it("preserves input-order alignment (tracks[i] corresponds to spans[i])", () => {
    // Feed in reverse order to prove the function handles unsorted input
    const spans: Span[] = [
      { start: 20, end: 25 },
      { start: 10, end: 15 },
      { start: 0, end: 5 },
    ];
    const result = assignTracks(spans, start, end);
    // All three are sequential, so all share track 0 regardless of input order
    expect(result).toEqual([0, 0, 0]);
  });
});
