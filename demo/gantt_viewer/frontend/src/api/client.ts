import type { components } from "./schema.gen";

export type GanttPayload = components["schemas"]["GanttPayload"];
export type HealthResponse = components["schemas"]["HealthResponse"];
export type Registries = components["schemas"]["Registries"];
export type TraceDescriptor = components["schemas"]["TraceDescriptor"];
export type TraceListResponse = components["schemas"]["TraceListResponse"];
export type TracePayload = components["schemas"]["TracePayload-Output"];
export type ResourceSample = components["schemas"]["ResourceSample"];
export type PayloadError = components["schemas"]["PayloadError"];
export type UnregisterTracesResponse = components["schemas"]["UnregisterTracesResponse"];
export type UploadTraceResponse = components["schemas"]["UploadTraceResponse"];

const JSON_HEADERS = {
  "Content-Type": "application/json",
} as const;

async function parseJson<T>(response: Response, message: string): Promise<T> {
  if (!response.ok) {
    throw new Error(`${message}: ${response.status}`);
  }
  return (await response.json()) as T;
}

async function postJson<T>(path: string, body: unknown, message: string): Promise<T> {
  return parseJson<T>(
    await fetch(path, {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify(body),
    }),
    message,
  );
}

export async function getHealth(): Promise<HealthResponse> {
  return parseJson<HealthResponse>(await fetch("/api/health"), "GET /api/health failed");
}

export async function getTraces(): Promise<TraceListResponse> {
  return parseJson<TraceListResponse>(await fetch("/api/traces"), "GET /api/traces failed");
}

export async function getPayload(ids: string[]): Promise<GanttPayload> {
  return postJson<GanttPayload>("/api/payload", { ids }, "POST /api/payload failed");
}

export async function uploadTrace(file: File): Promise<UploadTraceResponse> {
  const form = new FormData();
  form.append("file", file);
  return parseJson<UploadTraceResponse>(
    await fetch("/api/traces/upload", {
      method: "POST",
      body: form,
    }),
    "POST /api/traces/upload failed",
  );
}

export async function unregisterTraces(ids: string[]): Promise<UnregisterTracesResponse> {
  return postJson<UnregisterTracesResponse>(
    "/api/traces/unregister",
    { ids },
    "POST /api/traces/unregister failed",
  );
}
