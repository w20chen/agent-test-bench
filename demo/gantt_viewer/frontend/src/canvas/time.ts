import type { TimeMode } from "../state/signals";

export function formatSyncTime(seconds: number): string {
  if (seconds < 0) {
    return `${seconds.toFixed(1)}s`;
  }
  if (seconds < 60) {
    return `+${seconds.toFixed(1)}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  return `+${minutes}m${String(remainder).padStart(2, "0")}s`;
}

export function formatAbsTime(timestampSeconds: number): string {
  const date = new Date(timestampSeconds * 1000);
  return date.toTimeString().slice(0, 8);
}

export function formatTimeLabel(mode: TimeMode, value: number): string {
  return mode === "sync" ? formatSyncTime(value) : formatAbsTime(value);
}

export function niceStep(range: number, minTickCount: number): number {
  const raw = range / Math.max(1, Math.floor(minTickCount));
  const magnitude = Math.pow(10, Math.floor(Math.log10(raw)));
  const normalized = raw / magnitude;

  if (normalized <= 1) {
    return magnitude;
  }
  if (normalized <= 2) {
    return 2 * magnitude;
  }
  if (normalized <= 5) {
    return 5 * magnitude;
  }
  return 10 * magnitude;
}
