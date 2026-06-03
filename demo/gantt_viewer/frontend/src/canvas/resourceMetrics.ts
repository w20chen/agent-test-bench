import type { ResourceSample } from "../api/client";
import type { ResourceMetric } from "../state/signals";

function sampleTime(sample: ResourceSample): number {
  if (typeof sample.t_abs === "number" && Number.isFinite(sample.t_abs)) {
    return sample.t_abs;
  }
  return sample.t;
}

function throughputMbPerS(
  current: number | null | undefined,
  previous: number | null | undefined,
  dtSeconds: number,
): number {
  if (!Number.isFinite(dtSeconds) || dtSeconds <= 0) {
    return 0;
  }
  const delta = (current ?? 0) - (previous ?? 0);
  return Math.max(0, delta) / dtSeconds;
}

export function resourceMetricUnit(metric: ResourceMetric): string {
  switch (metric) {
    case "cpu":
      return "%";
    case "memory":
      return "MB";
    case "mem_total":
    case "mem_read":
    case "mem_write":
    case "disk_total":
    case "disk_read":
    case "disk_write":
      return "MB/s";
    case "none":
      return "";
  }
}

export function resourceMetricValueAt(
  timeline: ResourceSample[],
  index: number,
  metric: ResourceMetric,
): number | null {
  const sample = timeline[index];
  if (!sample) {
    return null;
  }
  switch (metric) {
    case "cpu":
      return sample.cpu_percent;
    case "memory":
      return sample.memory_mb;
    case "mem_total":
      return sample.memory_total_mb_s ?? null;
    case "mem_read":
      return sample.memory_read_mb_s ?? null;
    case "mem_write":
      return sample.memory_write_mb_s ?? null;
    case "disk_total":
    case "disk_read":
    case "disk_write": {
      if (index === 0) {
        return 0;
      }
      const previous = timeline[index - 1];
      const dtSeconds = sampleTime(sample) - sampleTime(previous);
      if (metric === "disk_total") {
        const read = throughputMbPerS(sample.disk_read_mb, previous.disk_read_mb, dtSeconds);
        const write = throughputMbPerS(sample.disk_write_mb, previous.disk_write_mb, dtSeconds);
        return read + write;
      }
      if (metric === "disk_read") {
        return throughputMbPerS(sample.disk_read_mb, previous.disk_read_mb, dtSeconds);
      }
      return throughputMbPerS(sample.disk_write_mb, previous.disk_write_mb, dtSeconds);
    }
    case "none":
      return 0;
  }
}

export function resourceMetricValues(
  timeline: ResourceSample[],
  metric: ResourceMetric,
): Array<number | null> {
  return timeline.map((_sample, index) => resourceMetricValueAt(timeline, index, metric));
}
