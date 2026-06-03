"""Tests for load_completed_ids attempt-layout dedup."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trace_collect.collector import load_completed_ids, next_attempt_number  # noqa: E402


def _write_manifest(path: Path, *, status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": 1, "status": status}), encoding="utf-8"
    )


def test_load_completed_ids_reads_attempt_manifests(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path / "mozilla__bleach-259" / "attempt_1" / "run_manifest.json",
        status="completed",
    )
    _write_manifest(
        tmp_path / "encode__httpx-2701" / "attempt_1" / "run_manifest.json",
        status="error",
    )
    _write_manifest(
        tmp_path / "tobymao__sqlglot-3425" / "attempt_2" / "run_manifest.json",
        status="completed",
    )
    completed = load_completed_ids(tmp_path)
    assert completed == {"mozilla__bleach-259", "tobymao__sqlglot-3425"}


def test_load_completed_ids_skips_instances_without_manifest(tmp_path: Path) -> None:
    (tmp_path / "half__written" / "attempt_1").mkdir(parents=True)
    (tmp_path / "no__attempts_dir").mkdir()
    assert load_completed_ids(tmp_path) == set()


def test_load_completed_ids_missing_run_dir_returns_empty(tmp_path: Path) -> None:
    assert load_completed_ids(tmp_path / "does_not_exist") == set()


def test_next_attempt_number_returns_one_for_fresh_instance(tmp_path: Path) -> None:
    assert next_attempt_number(tmp_path, "mozilla__bleach-259") == 1


def test_next_attempt_number_increments_after_existing_attempt(tmp_path: Path) -> None:
    (tmp_path / "mozilla__bleach-259" / "attempt_1").mkdir(parents=True)

    assert next_attempt_number(tmp_path, "mozilla__bleach-259") == 2


def test_next_attempt_number_uses_sparse_max_and_ignores_malformed_dirs(
    tmp_path: Path,
) -> None:
    instance_dir = tmp_path / "mozilla__bleach-259"
    (instance_dir / "attempt_1").mkdir(parents=True)
    (instance_dir / "attempt_3").mkdir()
    (instance_dir / "attempt_latest").mkdir()
    (instance_dir / "attempt_abc").mkdir()
    (instance_dir / "attempt_2.txt").mkdir()
    (instance_dir / "other").mkdir()

    assert next_attempt_number(tmp_path, "mozilla__bleach-259") == 4
