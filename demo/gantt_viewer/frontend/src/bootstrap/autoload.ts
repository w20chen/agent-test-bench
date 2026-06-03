import type { TraceDescriptor } from "../api/client";

export function shouldAutoloadAll(search: string): boolean {
  return new URLSearchParams(search).get("autoload") === "all";
}

export function selectInitialTraceIds(
  traceDescriptors: TraceDescriptor[],
  search: string,
): string[] {
  if (traceDescriptors.length === 0) {
    return [];
  }
  if (shouldAutoloadAll(search)) {
    return traceDescriptors.map((trace) => trace.id);
  }
  return [traceDescriptors[0].id];
}
