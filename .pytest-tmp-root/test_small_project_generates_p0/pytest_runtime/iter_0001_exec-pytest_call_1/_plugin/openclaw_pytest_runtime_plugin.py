from __future__ import annotations

import json
import os
from pathlib import Path

_COLLECTED = []
_TESTS = {}


def pytest_collection_finish(session):
    global _COLLECTED
    try:
        _COLLECTED = [item.nodeid for item in getattr(session, "items", [])]
    except Exception:
        _COLLECTED = []


def pytest_runtest_logreport(report):
    try:
        if report.when not in {"setup", "call", "teardown"}:
            return
        rec = _TESTS.setdefault(
            report.nodeid,
            {"nodeid": report.nodeid, "duration_s": 0.0, "outcome": "unknown"},
        )
        rec["duration_s"] += float(getattr(report, "duration", 0.0) or 0.0)
        if report.when == "call" or report.failed or report.skipped:
            rec["outcome"] = str(report.outcome)
    except Exception:
        return


def pytest_sessionfinish(session, exitstatus):
    try:
        out = os.environ.get("OPENCLAW_PYTEST_RUNTIME_JSON")
        if not out:
            return
        tests = []
        seen = set()
        for nodeid in _COLLECTED:
            rec = dict(_TESTS.get(nodeid) or {
                "nodeid": nodeid,
                "duration_s": 0.0,
                "outcome": "notrun",
            })
            rec["duration_s"] = round(float(rec.get("duration_s") or 0.0), 9)
            tests.append(rec)
            seen.add(nodeid)
        for nodeid, rec in sorted(_TESTS.items()):
            if nodeid in seen:
                continue
            item = dict(rec)
            item["duration_s"] = round(float(item.get("duration_s") or 0.0), 9)
            tests.append(item)
        payload = {
            "schema_version": 1,
            "collected_count": len(_COLLECTED),
            "exit_code": int(exitstatus),
            "tests": tests,
        }
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)
    except Exception:
        return
