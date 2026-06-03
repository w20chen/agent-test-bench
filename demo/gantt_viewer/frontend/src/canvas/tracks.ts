/**
 * Greedy interval-graph coloring for span track allocation.
 *
 * Given a list of spans with start/end times, assigns each span to the lowest
 * track index such that no two spans on the same track overlap in time.
 * Non-overlapping (sequential) spans share track 0; concurrent spans get
 * distinct tracks.
 *
 * The returned array is index-aligned with the input: tracks[i] is the
 * track for spans[i], regardless of the internal processing order.
 */

export function assignTracks<T>(
  spans: readonly T[],
  selectStart: (span: T) => number,
  selectEnd: (span: T) => number,
): number[] {
  const n = spans.length;
  if (n === 0) {
    return [];
  }

  // Process spans in (start, end) order for a stable, deterministic allocation.
  const order = spans
    .map((_, i) => i)
    .sort((a, b) => {
      const startDiff = selectStart(spans[a]) - selectStart(spans[b]);
      if (startDiff !== 0) return startDiff;
      return selectEnd(spans[a]) - selectEnd(spans[b]);
    });

  const trackEnds: number[] = [];
  const assignment = new Array<number>(n).fill(0);

  for (const idx of order) {
    const start = selectStart(spans[idx]);
    const end = selectEnd(spans[idx]);
    let assigned = -1;
    for (let t = 0; t < trackEnds.length; t++) {
      if (trackEnds[t] <= start) {
        trackEnds[t] = end;
        assigned = t;
        break;
      }
    }
    if (assigned === -1) {
      trackEnds.push(end);
      assigned = trackEnds.length - 1;
    }
    assignment[idx] = assigned;
  }

  return assignment;
}
