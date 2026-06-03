"""FastAPI routes for the dynamic Gantt viewer backend."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile

from demo.gantt_viewer.backend.payload import (
    DEFAULT_MARKER_REGISTRY,
    DEFAULT_SPAN_REGISTRY,
    build_gantt_payload_multi,
)
from demo.gantt_viewer.backend.ingest import (
    CanonicalizedTrace,
    ensure_canonical_trace_path,
)
from demo.gantt_viewer.backend.runtime_registry import (
    RuntimeRegistryConflictError,
    RuntimeTraceRegistry,
)
from demo.gantt_viewer.backend.schema import (
    GanttPayload,
    HealthResponse,
    PayloadError,
    PayloadRequest,
    RegisterTracesRequest,
    RegisterTracesResponse,
    Registries,
    TraceDescriptor,
    TraceListResponse,
    TracePayload,
    UnregisterTracesRequest,
    UnregisterTracesResponse,
    UploadTraceResponse,
)
from demo.gantt_viewer.backend.uploads import build_upload_id, persist_upload
from trace_collect.trace_inspector import TraceData


class _TracePayloadError(Exception):
    """Internal exception carrying structured partial-failure context."""

    def __init__(self, trace_id: str, stage: str, error: str) -> None:
        super().__init__(error)
        self.trace_id = trace_id
        self.stage = stage
        self.error = error


router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse)
def health_endpoint(request: Request) -> HealthResponse:
    trace_registry = _get_trace_registry(request)
    return HealthResponse(status="ok", n_discovered=len(trace_registry.descriptors))


@router.get("/traces", response_model=TraceListResponse)
def list_traces_endpoint(request: Request) -> TraceListResponse:
    return _build_trace_list_response(_get_trace_registry(request))


@router.post("/traces/reload", response_model=TraceListResponse)
def reload_traces_endpoint(request: Request) -> TraceListResponse:
    trace_registry = _get_trace_registry(request)
    trace_registry.reload()
    return _build_trace_list_response(trace_registry)


@router.post("/traces/register", response_model=RegisterTracesResponse)
def register_traces_endpoint(
    register_request: RegisterTracesRequest,
    request: Request,
) -> RegisterTracesResponse:
    trace_registry = _get_trace_registry(request)
    try:
        registered = trace_registry.register_paths(
            register_request.paths,
            labels_by_path=register_request.labels_by_path,
        )
    except RuntimeRegistryConflictError as exc:
        raise HTTPException(status_code=409, detail={"message": str(exc)}) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail={"message": str(exc)}) from exc
    return RegisterTracesResponse(registered=registered)


@router.post("/traces/unregister", response_model=UnregisterTracesResponse)
def unregister_traces_endpoint(
    unregister_request: UnregisterTracesRequest,
    request: Request,
) -> UnregisterTracesResponse:
    trace_registry = _get_trace_registry(request)
    removed_ids, missing_ids = trace_registry.unregister_ids(unregister_request.ids)
    return UnregisterTracesResponse(
        removed_ids=removed_ids,
        missing_ids=missing_ids,
    )


@router.post("/traces/upload", response_model=UploadTraceResponse)
async def upload_trace_endpoint(
    file: UploadFile,
    request: Request,
) -> UploadTraceResponse:
    filename = file.filename or "upload.jsonl"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=422, detail={"message": "empty upload"})

    upload_path = persist_upload(
        filename,
        content,
        upload_root=request.app.state.upload_root,
    )
    canonicalized = _canonicalize_or_422(upload_path)
    descriptor = TraceDescriptor(
        id=build_upload_id(filename, content),
        label=Path(filename).stem or "upload",
        source_format=canonicalized.source_format,
        path=str(canonicalized.canonical_path),
        size_bytes=canonicalized.canonical_path.stat().st_size,
        mtime=canonicalized.canonical_path.stat().st_mtime,
    )

    try:
        trace_payload = await _build_trace_payload(descriptor)
    except _TracePayloadError as exc:
        raise _trace_payload_http_exception(exc) from exc
    _get_trace_registry(request).register_uploaded_descriptor(descriptor)
    return UploadTraceResponse(descriptor=descriptor, payload_fragment=trace_payload)


@router.post("/payload", response_model=GanttPayload)
async def payload_endpoint(
    payload_request: PayloadRequest,
    request: Request,
) -> GanttPayload:
    trace_registry = _get_trace_registry(request)
    descriptors = _resolve_descriptors(trace_registry, payload_request.ids)

    traces: list[TracePayload] = []
    errors: list[PayloadError] = []
    for descriptor in descriptors:
        try:
            traces.append(await _build_trace_payload(descriptor))
        except _TracePayloadError as exc:
            errors.append(
                PayloadError(trace_id=exc.trace_id, stage=exc.stage, error=exc.error)
            )

    if not traces and errors:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "all requested traces failed",
                "errors": [e.model_dump() for e in errors],
            },
        )

    return GanttPayload(
        registries=_build_registries(),
        traces=traces,
        errors=errors,
    )


def _get_trace_registry(request: Request) -> RuntimeTraceRegistry:
    return request.app.state.trace_registry


def _build_trace_list_response(
    trace_registry: RuntimeTraceRegistry,
) -> TraceListResponse:
    return TraceListResponse(
        traces=trace_registry.descriptors, registries=_build_registries()
    )


def _trace_payload_http_exception(exc: _TracePayloadError) -> HTTPException:
    return HTTPException(
        status_code=422,
        detail={"trace_id": exc.trace_id, "stage": exc.stage, "error": exc.error},
    )


async def _build_trace_payload(descriptor: TraceDescriptor) -> TracePayload:
    try:
        trace_data = TraceData.load(Path(descriptor.path))
    except Exception as exc:
        raise _TracePayloadError(
            trace_id=descriptor.id,
            stage="trace_load",
            error=str(exc),
        ) from exc

    try:
        raw_payload = build_gantt_payload_multi(
            [(descriptor.id, trace_data)],
            span_registry=deepcopy(DEFAULT_SPAN_REGISTRY),
            marker_registry=deepcopy(DEFAULT_MARKER_REGISTRY),
        )
        raw_payload["traces"][0]["label"] = descriptor.label
        return TracePayload.model_validate(raw_payload["traces"][0])
    except Exception as exc:
        raise _TracePayloadError(
            trace_id=descriptor.id,
            stage="payload_build",
            error=str(exc),
        ) from exc


def _canonicalize_or_422(path: Path) -> CanonicalizedTrace:
    try:
        return ensure_canonical_trace_path(path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail={"message": str(exc)}) from exc


def _resolve_descriptors(
    trace_registry: RuntimeTraceRegistry,
    trace_ids: list[str],
) -> list[TraceDescriptor]:
    descriptors = [trace_registry.get_descriptor(trace_id) for trace_id in trace_ids]
    missing = [
        trace_id
        for trace_id, descriptor in zip(trace_ids, descriptors, strict=False)
        if descriptor is None
    ]
    if missing:
        raise HTTPException(
            status_code=404,
            detail={"message": "unknown trace ids", "trace_ids": missing},
        )
    return [descriptor for descriptor in descriptors if descriptor is not None]


def _build_registries() -> Registries:
    return Registries(
        spans=deepcopy(DEFAULT_SPAN_REGISTRY),
        markers=deepcopy(DEFAULT_MARKER_REGISTRY),
    )
