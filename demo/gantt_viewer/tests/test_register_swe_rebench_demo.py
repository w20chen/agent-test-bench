"""Tests for the swe-rebench demo registration helper."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from demo.gantt_viewer.backend.app import create_app
from demo.gantt_viewer.register_swe_rebench_demo import (
    DEMO_SERVER_PRECONDITION,
    ExperimentSpec,
    build_registration_entries,
    classify_existing_traces,
    compute_shared_tasks,
    register_demo,
    verify_trace_order,
)
from demo.gantt_viewer.tests.helpers import write_config, write_trace


def _llm_action() -> dict:
    return {
        "type": "action",
        "action_type": "llm_call",
        "action_id": "llm_0",
        "agent_id": "agent-1",
        "iteration": 0,
        "ts_start": 1000.0,
        "ts_end": 1001.0,
        "data": {"raw_response": {"choices": [{"message": {"content": "hello"}}]}},
    }


def _make_trace(path: Path) -> Path:
    return write_trace(path, [_llm_action()], scaffold="openclaw")


def test_compute_shared_tasks_returns_sorted_intersection() -> None:
    shared = compute_shared_tasks(
        {
            "a": ["task-c", "task-a", "task-b"],
            "b": ["task-b", "task-a", "task-d"],
            "c": ["task-a", "task-b"],
        }
    )

    assert shared == ["task-a", "task-b"]


def test_build_registration_entries_orders_task_major(tmp_path: Path) -> None:
    spec_a = ExperimentSpec(
        slot="a", label_suffix="claude-code-haiku", root=tmp_path / "exp-a"
    )
    spec_b = ExperimentSpec(
        slot="b", label_suffix="openclaw-haiku", root=tmp_path / "exp-b"
    )
    for spec in (spec_a, spec_b):
        for task in ("task-a", "task-b"):
            _make_trace(spec.root / task / "attempt_1" / "trace.jsonl")

    entries = build_registration_entries(["task-a", "task-b"], [spec_a, spec_b])

    assert [entry.label for entry in entries] == [
        "t01-a-claude-code-haiku",
        "t01-b-openclaw-haiku",
        "t02-a-claude-code-haiku",
        "t02-b-openclaw-haiku",
    ]


def test_classify_existing_traces_accepts_clean_rerun_subset(tmp_path: Path) -> None:
    spec = ExperimentSpec(
        slot="a", label_suffix="claude-code-haiku", root=tmp_path / "exp-a"
    )
    _make_trace(spec.root / "task-a" / "attempt_1" / "trace.jsonl")
    _make_trace(spec.root / "task-b" / "attempt_1" / "trace.jsonl")
    entries = build_registration_entries(["task-a", "task-b"], [spec])

    already_registered, to_register = classify_existing_traces(
        entries,
        [{"path": str(entries[0].path), "label": entries[0].label}],
    )

    assert already_registered == [entries[0]]
    assert to_register == [entries[1]]


def test_classify_existing_traces_accepts_canonicalized_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = ExperimentSpec(
        slot="a", label_suffix="claude-code-haiku", root=tmp_path / "exp-a"
    )
    _make_trace(spec.root / "task-a" / "attempt_1" / "trace.jsonl")
    [entry] = build_registration_entries(["task-a"], [spec])
    canonical_path = tmp_path / "imports" / "task-a.jsonl"

    monkeypatch.setattr(
        "demo.gantt_viewer.register_swe_rebench_demo._canonical_demo_path",
        lambda path: str(canonical_path) if path == str(entry.path) else path,
    )

    already_registered, to_register = classify_existing_traces(
        [entry],
        [{"path": str(canonical_path), "label": entry.label}],
    )

    assert already_registered == [entry]
    assert to_register == []


def test_classify_existing_traces_rejects_mixed_registry_state(tmp_path: Path) -> None:
    spec = ExperimentSpec(
        slot="a", label_suffix="claude-code-haiku", root=tmp_path / "exp-a"
    )
    _make_trace(spec.root / "task-a" / "attempt_1" / "trace.jsonl")
    [entry] = build_registration_entries(["task-a"], [spec])

    with pytest.raises(
        ValueError, match="/api/traces must contain only this demo cohort"
    ):
        classify_existing_traces(
            [entry],
            [{"path": str(tmp_path / "other" / "trace.jsonl"), "label": "other"}],
        )
    assert "isolated GANTT_VIEWER_RUNTIME_STATE" in DEMO_SERVER_PRECONDITION


def test_classify_existing_traces_rejects_wrong_existing_order(tmp_path: Path) -> None:
    spec = ExperimentSpec(
        slot="a", label_suffix="claude-code-haiku", root=tmp_path / "exp-a"
    )
    _make_trace(spec.root / "task-a" / "attempt_1" / "trace.jsonl")
    _make_trace(spec.root / "task-b" / "attempt_1" / "trace.jsonl")
    entries = build_registration_entries(["task-a", "task-b"], [spec])

    with pytest.raises(ValueError, match="existing /api/traces order does not match"):
        classify_existing_traces(
            entries,
            [
                {"path": str(entries[1].path), "label": entries[1].label},
                {"path": str(entries[0].path), "label": entries[0].label},
            ],
        )


def test_verify_trace_order_rejects_api_order_mismatch(tmp_path: Path) -> None:
    spec = ExperimentSpec(
        slot="a", label_suffix="claude-code-haiku", root=tmp_path / "exp-a"
    )
    _make_trace(spec.root / "task-a" / "attempt_1" / "trace.jsonl")
    _make_trace(spec.root / "task-b" / "attempt_1" / "trace.jsonl")
    entries = build_registration_entries(["task-a", "task-b"], [spec])

    with pytest.raises(RuntimeError, match="actual /api/traces order does not match"):
        verify_trace_order(
            [
                {
                    "id": "second",
                    "path": str(entries[1].path),
                    "label": entries[1].label,
                },
                {
                    "id": "first",
                    "path": str(entries[0].path),
                    "label": entries[0].label,
                },
            ],
            entries,
        )


def test_verify_trace_order_accepts_canonicalized_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = ExperimentSpec(
        slot="a", label_suffix="claude-code-haiku", root=tmp_path / "exp-a"
    )
    _make_trace(spec.root / "task-a" / "attempt_1" / "trace.jsonl")
    [entry] = build_registration_entries(["task-a"], [spec])
    canonical_path = tmp_path / "imports" / "task-a.jsonl"

    monkeypatch.setattr(
        "demo.gantt_viewer.register_swe_rebench_demo._canonical_demo_path",
        lambda path: str(canonical_path) if path == str(entry.path) else path,
    )

    ordered_ids = verify_trace_order(
        [{"id": "trace-1", "path": str(canonical_path), "label": entry.label}],
        [entry],
    )

    assert ordered_ids == ["trace-1"]


def test_register_demo_fails_when_refreshed_api_order_is_wrong(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = ExperimentSpec(
        slot="a", label_suffix="claude-code-haiku", root=tmp_path / "exp-a"
    )
    _make_trace(spec.root / "task-a" / "attempt_1" / "trace.jsonl")
    _make_trace(spec.root / "task-b" / "attempt_1" / "trace.jsonl")
    entries = build_registration_entries(["task-a", "task-b"], [spec])
    responses = iter(
        [
            {"traces": []},
            {"registered": [{"id": "ignored"}]},
            {
                "traces": [
                    {
                        "id": "second",
                        "path": str(entries[1].path),
                        "label": entries[1].label,
                    },
                    {
                        "id": "first",
                        "path": str(entries[0].path),
                        "label": entries[0].label,
                    },
                ]
            },
        ]
    )

    monkeypatch.setattr(
        "demo.gantt_viewer.register_swe_rebench_demo.load_demo_plan",
        lambda: entries,
    )
    monkeypatch.setattr(
        "demo.gantt_viewer.register_swe_rebench_demo.fetch_json",
        lambda *args, **kwargs: next(responses),
    )

    with pytest.raises(RuntimeError, match="actual /api/traces order does not match"):
        register_demo("http://127.0.0.1:8765", verify_payload=False)


def test_api_traces_order_matches_task_major_labels(tmp_path: Path) -> None:
    runtime_state_path = tmp_path / "runtime-state.json"
    config_path = write_config(
        tmp_path / "config.yaml",
        [str(tmp_path / "nothing" / "*.jsonl")],
    )
    client = TestClient(
        create_app(config_path=config_path, runtime_state_path=runtime_state_path)
    )
    trace_a = _make_trace(tmp_path / "demo" / "task-a" / "attempt_1" / "trace.jsonl")
    trace_b = _make_trace(tmp_path / "demo" / "task-b" / "attempt_1" / "trace.jsonl")
    trace_c = _make_trace(tmp_path / "demo" / "task-c" / "attempt_1" / "trace.jsonl")

    response = client.post(
        "/api/traces/register",
        json={
            "paths": [str(trace_c), str(trace_a), str(trace_b)],
            "labels_by_path": {
                str(trace_a): "t01-a-claude-code-haiku",
                str(trace_b): "t01-b-openclaw-haiku",
                str(trace_c): "t01-c-arm",
            },
        },
    )
    assert response.status_code == 200

    traces = client.get("/api/traces").json()["traces"]
    assert [trace["label"] for trace in traces] == [
        "t01-a-claude-code-haiku",
        "t01-b-openclaw-haiku",
        "t01-c-arm",
    ]
