"""Tests for the FastAPI backend scaffold."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from demo.gantt_viewer.backend.app import create_app
from demo.gantt_viewer.backend.payload import (
    DEFAULT_MARKER_REGISTRY,
    build_gantt_payload_multi,
)
from demo.gantt_viewer.backend.schema import MarkerDef
from demo.gantt_viewer.tests.helpers import write_config, write_trace
from trace_collect.trace_inspector import TraceData


REPO_ROOT = Path(__file__).resolve().parents[3]
CC_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "claude_code_minimal.jsonl"
OPENAPI_SNAPSHOT = (
    REPO_ROOT / "demo" / "gantt_viewer" / "tests" / "fixtures" / "openapi.snapshot.json"
)


def _normalize_openapi_snapshot(text: str) -> str:
    """Normalize known FastAPI/Pydantic schema drift across envs."""

    def _rewrite(value):
        if isinstance(value, dict):
            rewritten = {}
            for key, item in value.items():
                normalized_key = (
                    "TracePayload" if key in {"TracePayload-Input", "TracePayload-Output"} else key
                )
                if normalized_key == "TracePayload" and normalized_key in rewritten:
                    continue
                rewritten[normalized_key] = _rewrite(item)
            return rewritten
        if isinstance(value, list):
            return [_rewrite(item) for item in value]
        if isinstance(value, str):
            return (
                value.replace("TracePayload-Input", "TracePayload")
                .replace("TracePayload-Output", "TracePayload")
            )
        return value

    payload = _rewrite(json.loads(text))
    file_schema = (
        payload.get("components", {})
        .get("schemas", {})
        .get("Body_upload_trace_endpoint_api_traces_upload_post", {})
        .get("properties", {})
        .get("file")
    )
    if isinstance(file_schema, dict) and file_schema.get("contentMediaType") == "application/octet-stream":
        file_schema.pop("contentMediaType", None)
        file_schema["format"] = "binary"
    return json.dumps(payload, indent=2, sort_keys=True)


def _write_legacy_trace(trace_path: Path) -> Path:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        json.dumps(
            {
                "type": "trace_metadata",
                "scaffold": "openclaw",
                "trace_format_version": 4,
            }
        )
        + "\n"
        + json.dumps({"type": "step", "iteration": 0})
        + "\n",
        encoding="utf-8",
    )
    return trace_path


def _llm_action() -> dict:
    return {
        "type": "action",
        "action_type": "llm_call",
        "action_id": "llm_0",
        "agent_id": "agent-1",
        "iteration": 0,
        "ts_start": 1000.0,
        "ts_end": 1001.0,
        "data": {
            "raw_response": {
                "choices": [{"message": {"content": "hello from llm"}}]
            }
        },
    }


def _make_client(tmp_path: Path) -> tuple[TestClient, Path]:
    runtime_state_path = tmp_path / "runtime-state.json"
    trace_path = write_trace(
        tmp_path / "runs" / "repo__issue-1" / "trace.jsonl",
        [_llm_action()],
        scaffold="openclaw",
    )
    config_path = write_config(
        tmp_path / "config.yaml",
        [str(trace_path)],
    )
    client = TestClient(
        create_app(
            config_path=config_path,
            runtime_state_path=runtime_state_path,
        )
    )
    return client, trace_path


def test_health_endpoint_reports_discovered_traces(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "n_discovered": 1}


def test_list_traces_returns_descriptors_and_registries(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    response = client.get("/api/traces")
    assert response.status_code == 200

    body = response.json()
    assert len(body["traces"]) == 1
    assert body["traces"][0]["source_format"] == "trace"
    assert body["registries"]["spans"]["llm"]["label"] == "LLM Call"
    assert body["registries"]["markers"]["message_dispatch"]["label"] == "Message Dispatch"


def test_payload_endpoint_returns_trace_payload(tmp_path: Path) -> None:
    client, trace_path = _make_client(tmp_path)
    response = client.post("/api/payload", json={"ids": ["ac1-repo__issue-1"]})
    assert response.status_code == 200

    body = response.json()
    assert len(body["traces"]) == 1
    assert body["traces"][0]["id"] == "ac1-repo__issue-1"
    assert body["traces"][0]["label"] == "repo__issue-1"
    assert body["traces"][0]["metadata"]["scaffold"] == "openclaw"

    direct_payload = build_gantt_payload_multi(
        [("ac1-repo__issue-1", TraceData.load(trace_path))]
    )
    direct_payload["traces"][0]["label"] = "repo__issue-1"
    assert body["traces"] == direct_payload["traces"]


def test_payload_endpoint_rejects_unknown_id(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    response = client.post("/api/payload", json={"ids": ["missing-id"]})
    assert response.status_code == 404
    assert response.json()["detail"]["trace_ids"] == ["missing-id"]


def test_reload_endpoint_rewalks_discovery(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    response = client.post("/api/traces/reload")
    assert response.status_code == 200
    assert len(response.json()["traces"]) == 1


def test_register_trace_path_adds_descriptor(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    extra_path = write_trace(
        tmp_path / "runtime" / "repo__issue-2" / "trace.jsonl",
        [_llm_action()],
        scaffold="openclaw",
    )

    response = client.post(
        "/api/traces/register",
        json={"paths": [str(extra_path)]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["registered"][0]["source_format"] == "trace"

    traces = client.get("/api/traces").json()["traces"]
    assert any(trace["path"] == str(extra_path.resolve()) for trace in traces)


def test_register_raw_claude_code_trace_auto_imports(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    client, _ = _make_client(tmp_path)
    response = client.post(
        "/api/traces/register",
        json={"paths": [str(CC_FIXTURE)]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["registered"][0]["source_format"] == "trace"
    assert "gantt-cc-import" in body["registered"][0]["path"]

    payload_response = client.post("/api/payload", json={"ids": [body["registered"][0]["id"]]})
    assert payload_response.status_code == 200
    assert payload_response.json()["traces"][0]["metadata"]["scaffold"] == "claude-code"


def test_register_legacy_trace_returns_422(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    legacy_path = _write_legacy_trace(tmp_path / "legacy" / "trace.jsonl")
    response = client.post(
        "/api/traces/register",
        json={"paths": [str(legacy_path)]},
    )
    assert response.status_code == 422
    assert "trace_format_version=4" in response.json()["detail"]["message"]


def test_unregister_runtime_trace_removes_descriptor(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    extra_path = write_trace(
        tmp_path / "runtime" / "repo__issue-2" / "trace.jsonl",
        [_llm_action()],
        scaffold="openclaw",
    )
    registered = client.post(
        "/api/traces/register",
        json={"paths": [str(extra_path)]},
    ).json()["registered"][0]

    response = client.post(
        "/api/traces/unregister",
        json={"ids": [registered["id"]]},
    )
    assert response.status_code == 200
    assert response.json()["removed_ids"] == [registered["id"]]
    traces = client.get("/api/traces").json()["traces"]
    assert all(trace["id"] != registered["id"] for trace in traces)


def test_unregister_config_trace_persists_across_reload(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    config_id = "ac1-repo__issue-1"

    response = client.post("/api/traces/unregister", json={"ids": [config_id]})
    assert response.status_code == 200
    assert response.json()["removed_ids"] == [config_id]

    traces = client.get("/api/traces").json()["traces"]
    assert all(trace["id"] != config_id for trace in traces)

    reloaded = client.post("/api/traces/reload").json()["traces"]
    assert all(trace["id"] != config_id for trace in reloaded)


def test_uploaded_trace_persists_across_app_restart(tmp_path: Path) -> None:
    runtime_state_path = tmp_path / "runtime-state.json"
    trace_path = write_trace(
        tmp_path / "runs" / "repo__issue-1" / "trace.jsonl",
        [_llm_action()],
        scaffold="openclaw",
    )
    config_path = write_config(tmp_path / "config.yaml", [str(trace_path)])

    client = TestClient(create_app(config_path=config_path, runtime_state_path=runtime_state_path))
    upload_path = write_trace(
        tmp_path / "upload" / "persisted_upload.jsonl",
        [_llm_action()],
        scaffold="openclaw",
    )

    with upload_path.open("rb") as handle:
        response = client.post(
            "/api/traces/upload",
            files={"file": (upload_path.name, handle, "application/json")},
        )

    assert response.status_code == 200
    uploaded_id = response.json()["descriptor"]["id"]

    restarted = TestClient(
        create_app(config_path=config_path, runtime_state_path=runtime_state_path)
    )
    traces = restarted.get("/api/traces").json()["traces"]
    assert any(trace["id"] == uploaded_id for trace in traces)


def test_upload_trace_registers_descriptor(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    upload_path = write_trace(
        tmp_path / "upload" / "demo_trace.jsonl",
        [_llm_action()],
        scaffold="openclaw",
    )

    with upload_path.open("rb") as handle:
        response = client.post(
            "/api/traces/upload",
            files={"file": (upload_path.name, handle, "application/json")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["descriptor"]["source_format"] == "trace"
    assert body["payload_fragment"]["metadata"]["scaffold"] == "openclaw"

    payload_response = client.post("/api/payload", json={"ids": [body["descriptor"]["id"]]})
    assert payload_response.status_code == 200


def test_upload_raw_claude_code_trace_auto_imports(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    client, _ = _make_client(tmp_path)
    with CC_FIXTURE.open("rb") as handle:
        response = client.post(
            "/api/traces/upload",
            files={"file": ("claude_code_minimal.jsonl", handle, "application/json")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["descriptor"]["source_format"] == "trace"
    assert body["payload_fragment"]["metadata"]["scaffold"] == "claude-code"


def test_upload_legacy_trace_returns_422(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    legacy_path = _write_legacy_trace(tmp_path / "upload" / "legacy.jsonl")
    with legacy_path.open("rb") as handle:
        response = client.post(
            "/api/traces/upload",
            files={"file": (legacy_path.name, handle, "application/json")},
        )

    assert response.status_code == 422
    assert "trace_format_version=4" in response.json()["detail"]["message"]


def test_upload_malformed_trace_returns_422(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    response = client.post(
        "/api/traces/upload",
        files={"file": ("broken.jsonl", b"not jsonl", "application/octet-stream")},
    )
    assert response.status_code == 422


def test_payload_missing_trace_file_returns_422(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    runtime_path = write_trace(
        tmp_path / "runtime" / "repo__issue-2" / "trace.jsonl",
        [_llm_action()],
        scaffold="openclaw",
    )
    registered = client.post(
        "/api/traces/register",
        json={"paths": [str(runtime_path)]},
    ).json()["registered"][0]
    runtime_path.unlink()

    response = client.post("/api/payload", json={"ids": [registered["id"]]})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["message"] == "all requested traces failed"
    assert detail["errors"][0]["stage"] == "trace_load"


def test_default_marker_registry_matches_schema() -> None:
    """Every DEFAULT_MARKER_REGISTRY entry must pass MarkerDef validation directly."""
    assert DEFAULT_MARKER_REGISTRY, "registry must not be empty"
    for key, entry in DEFAULT_MARKER_REGISTRY.items():
        assert set(entry.keys()) >= {"symbol", "color", "label"}, (
            f"{key} missing required fields: {entry}"
        )
        validated = MarkerDef.model_validate(entry)
        assert validated.label, f"{key} has empty label"


def test_openapi_frozen(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("GANTT_VIEWER_CONFIG", raising=False)
    monkeypatch.delenv("GANTT_VIEWER_DEV", raising=False)
    empty_config = write_config(
        tmp_path / "config.yaml",
        [str(tmp_path / "nothing" / "*.jsonl")],
    )
    client = TestClient(
        create_app(
            config_path=empty_config,
            runtime_state_path=tmp_path / "runtime-state.json",
        )
    )
    response = client.get("/openapi.json")
    assert response.status_code == 200

    actual = json.loads(
        _normalize_openapi_snapshot(json.dumps(response.json(), indent=2, sort_keys=True))
    )
    expected = json.loads(
        _normalize_openapi_snapshot(OPENAPI_SNAPSHOT.read_text(encoding="utf-8").strip())
    )

    assert set(actual["paths"]) == set(expected["paths"]), (
        "OpenAPI path surface drifted. Regenerate "
        "demo/gantt_viewer/tests/fixtures/openapi.snapshot.json and "
        "demo/gantt_viewer/frontend/src/api/schema.gen.ts together."
    )
    for path, expected_methods in expected["paths"].items():
        actual_methods = actual["paths"][path]
        assert set(actual_methods) == set(expected_methods), (
            f"OpenAPI methods drifted for {path}. Regenerate "
            "demo/gantt_viewer/tests/fixtures/openapi.snapshot.json and "
            "demo/gantt_viewer/frontend/src/api/schema.gen.ts together."
        )
        for method, expected_operation in expected_methods.items():
            assert actual_methods[method]["operationId"] == expected_operation["operationId"]

    assert {
        "GanttPayload",
        "HealthResponse",
        "TraceDescriptor",
        "TraceListResponse",
        "UploadTraceResponse",
    }.issubset(actual["components"]["schemas"])
