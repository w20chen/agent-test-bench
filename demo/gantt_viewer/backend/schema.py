"""Typed API schema for the dynamic Gantt viewer."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SpanDef(BaseModel):
    color: str
    label: str
    order: int


class MarkerDef(BaseModel):
    symbol: str
    color: str
    label: str


class Registries(BaseModel):
    spans: dict[str, SpanDef]
    markers: dict[str, MarkerDef]


class Span(BaseModel):
    type: str
    start: float
    end: float
    start_abs: float
    end_abs: float
    start_real: float | None = None
    end_real: float | None = None
    start_real_abs: float | None = None
    end_real_abs: float | None = None
    iteration: int
    detail: dict[str, Any]


class Marker(BaseModel):
    type: str
    event: str
    t: float
    t_abs: float
    t_real: float | None = None
    t_real_abs: float | None = None
    iteration: int
    detail: dict[str, Any]


class Lane(BaseModel):
    agent_id: str
    spans: list[Span]
    markers: list[Marker]


class TraceMetadata(BaseModel):
    scaffold: str
    model: str | None = None
    instance_id: str = ""
    mode: str | None = None
    max_iterations: int | None = None
    n_actions: int
    n_iterations: int
    n_events: int
    elapsed_s: float | None = None


class ResourceSample(BaseModel):
    t: float
    t_abs: float
    t_real: float | None = None
    t_real_abs: float | None = None
    cpu_percent: float
    memory_mb: float
    memory_total_mb_s: float | None = None
    memory_read_mb_s: float | None = None
    memory_write_mb_s: float | None = None
    disk_read_mb: float | None = None
    disk_write_mb: float | None = None
    net_rx_mb: float | None = None
    net_tx_mb: float | None = None
    context_switches: int | None = None


class TracePayload(BaseModel):
    id: str
    label: str
    metadata: TraceMetadata
    t0: float
    lanes: list[Lane]
    resource_timeline: list[ResourceSample] | None = None


class PayloadError(BaseModel):
    trace_id: str
    stage: str
    error: str


class GanttPayload(BaseModel):
    registries: Registries
    traces: list[TracePayload]
    errors: list[PayloadError] = Field(default_factory=list)


class TraceDescriptor(BaseModel):
    id: str
    label: str
    source_format: Literal["trace"]
    path: str
    size_bytes: int
    mtime: float


class TraceListResponse(BaseModel):
    traces: list[TraceDescriptor]
    registries: Registries


class HealthResponse(BaseModel):
    status: str
    n_discovered: int


class PayloadRequest(BaseModel):
    ids: list[str] = Field(min_length=1)


class RegisterTracesRequest(BaseModel):
    paths: list[str] = Field(min_length=1)
    labels_by_path: dict[str, str] = Field(default_factory=dict)


class RegisterTracesResponse(BaseModel):
    registered: list[TraceDescriptor]


class UnregisterTracesRequest(BaseModel):
    ids: list[str] = Field(min_length=1)


class UnregisterTracesResponse(BaseModel):
    removed_ids: list[str]
    missing_ids: list[str]


class UploadTraceResponse(BaseModel):
    descriptor: TraceDescriptor
    payload_fragment: TracePayload
