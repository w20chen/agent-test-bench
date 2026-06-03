from __future__ import annotations

import json
from pathlib import Path

from scripts.recoding_figures.recording_loader import load_session_history


def _write_meta(attempt_dir: Path, payload: dict) -> None:
    recordings = attempt_dir / "recordings"
    recordings.mkdir(parents=True)
    (recordings / "meta.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_load_session_history_joins_attempt_and_iter(tmp_path: Path) -> None:
    attempt_dir = tmp_path / "task-a" / "attempt_1"
    _write_meta(
        attempt_dir,
        {
            "iters": [{"call_idx": 0, "dir": "iter_0000"}],
            "orphan_iters": [{"call_idx": 1, "dir": "iter_0001"}],
            "session_history": [
                {
                    "call_idx": 0,
                    "used_session_cache": True,
                    "lcp": 0,
                    "cached_len_before": 0,
                    "new_len": 10,
                    "delta_len": 10,
                    "diverged": False,
                },
                {
                    "call_idx": 1,
                    "used_session_cache": True,
                    "lcp": 3,
                    "cached_len_before": 8,
                    "new_len": 12,
                    "delta_len": 12,
                    "diverged": True,
                },
            ],
        },
    )

    rows = load_session_history([attempt_dir])

    assert [row["call_idx"] for row in rows] == [0, 1]
    assert rows[0]["task"] == "task-a"
    assert rows[0]["attempt_dir"] == attempt_dir
    assert rows[0]["iter_dir"] == attempt_dir / "recordings" / "iter_0000"
    assert rows[1]["iter_dir"] == attempt_dir / "recordings" / "iter_0001"
    assert rows[1]["diverged"] is True


def test_load_session_history_legacy_meta_returns_empty(tmp_path: Path) -> None:
    attempt_dir = tmp_path / "task-a" / "attempt_1"
    _write_meta(attempt_dir, {"iters": [{"call_idx": 0, "dir": "iter_0000"}]})

    assert load_session_history([attempt_dir]) == []
