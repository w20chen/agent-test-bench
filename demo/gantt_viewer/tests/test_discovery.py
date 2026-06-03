"""Tests for canonical trace discovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from demo.gantt_viewer.backend.discovery import (
    discover_traces,
    load_discovery_config,
    sniff_format,
)
from demo.gantt_viewer.tests.helpers import write_config, write_trace

REPO_ROOT = Path(__file__).resolve().parents[3]
CC_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "claude_code_minimal.jsonl"


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


def test_sniff_format_trace(tmp_path: Path) -> None:
    trace_path = write_trace(tmp_path / "runs" / "task-1" / "trace.jsonl", [])
    assert sniff_format(trace_path) == "trace"


def test_sniff_format_rejects_raw_claude_code_session() -> None:
    with pytest.raises(ValueError, match="not a canonical trace JSONL"):
        sniff_format(CC_FIXTURE)


def test_sniff_format_rejects_legacy_trace_version(tmp_path: Path) -> None:
    trace_path = _write_legacy_trace(tmp_path / "legacy.jsonl")
    with pytest.raises(ValueError, match="trace_format_version=4"):
        sniff_format(trace_path)


def test_sniff_format_empty_file_raises(tmp_path: Path) -> None:
    trace_path = tmp_path / "empty.jsonl"
    trace_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty JSONL file"):
        sniff_format(trace_path)


def test_discover_traces_builds_expected_ids(tmp_path: Path) -> None:
    trace_path = write_trace(
        tmp_path / "runs" / "repo__issue-1" / "trace.jsonl",
        [],
    )
    config_path = write_config(
        tmp_path / "config.yaml",
        [str(trace_path)],
    )

    config = load_discovery_config(config_path)
    descriptors = discover_traces(config)

    assert [descriptor.id for descriptor in descriptors] == ["ac1-repo__issue-1"]
    assert descriptors[0].label == "repo__issue-1"
    assert descriptors[0].source_format == "trace"


def test_sniff_format_unknown_shape_raises(tmp_path: Path) -> None:
    trace_path = tmp_path / "unknown.jsonl"
    trace_path.write_text(json.dumps({"foo": "bar"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a canonical trace JSONL"):
        sniff_format(trace_path)


def test_discover_traces_expands_glob_patterns(tmp_path: Path) -> None:
    """A glob pattern in the config should expand to every matching trace."""
    runs_root = tmp_path / "runs"
    for instance in ("repo__issue-1", "repo__issue-2", "repo__issue-3"):
        write_trace(runs_root / instance / "trace.jsonl", [])

    config_path = write_config(
        tmp_path / "config.yaml",
        [str(runs_root / "*" / "trace.jsonl")],
    )

    config = load_discovery_config(config_path)
    descriptors = discover_traces(config)

    ids = [descriptor.id for descriptor in descriptors]
    assert ids == [
        "ac1-repo__issue-1",
        "ac1-repo__issue-2",
        "ac1-repo__issue-3",
    ]
    assert all(descriptor.source_format == "trace" for descriptor in descriptors)
