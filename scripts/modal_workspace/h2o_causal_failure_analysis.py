"""Modal analysis for the H2O causal-inference-r capstone failure.

The input is the private Hugging Face dataset archive uploaded during the
emergency backup:

    HCAHOI/agent-sched-bench-kv-evict-capstone-20260514
    archives/20260514T155149Z/causal-inference-r__h2o-b4096-T140506.tar.zst

This script keeps the analysis post-hoc: it downloads and extracts the archive,
then inspects `trace.jsonl`, `segments.json`, and `kv_eviction.npz` artifacts.
It does not rerun the benchmark or modify source traces.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import modal


APP_NAME = "asb-h2o-causal-failure-analysis"
VOLUME_NAME = "asb-terminal-recordings"
HF_SECRET_NAME = "asb-hf-token"
HF_REPO_ID = "HCAHOI/agent-sched-bench-kv-evict-capstone-20260514"
HF_REPO_TYPE = "dataset"
ARCHIVE_PATH_IN_REPO = (
    "archives/20260514T155149Z/"
    "causal-inference-r__h2o-b4096-T140506.tar.zst"
)
ARCHIVE_NAME = Path(ARCHIVE_PATH_IN_REPO).name
RUN_LABEL = "causal-inference-r__h2o-b4096-T140506"

VOLUME_ROOT = Path("/data")
ARCHIVE_DIR = VOLUME_ROOT / "hf-archives" / "kv-evict-capstone-20260514"
EXTRACT_DIR = VOLUME_ROOT / "extracted" / RUN_LABEL
OUTPUT_DIR = VOLUME_ROOT / "outputs" / "h2o-causal-failure" / "20260515-h2o-t140506"

LOCAL_FILE = Path(__file__).resolve()
LOCAL_RECODING_FIGURES = (
    LOCAL_FILE.parents[2] / "scripts" / "recoding_figures"
    if len(LOCAL_FILE.parents) > 2
    else Path("/opt/recoding_figures")
)
RECODING_FIGURES = (
    LOCAL_RECODING_FIGURES
    if LOCAL_RECODING_FIGURES.exists()
    else Path("/opt/recoding_figures")
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ca-certificates", "tar", "zstd")
    .pip_install("huggingface_hub", "numpy")
    .add_local_dir(RECODING_FIGURES, remote_path="/opt/recoding_figures", copy=True)
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
secret = modal.Secret.from_name(HF_SECRET_NAME, required_keys=["HF_TOKEN"])
app = modal.App(APP_NAME)


@app.function(
    image=image,
    volumes={VOLUME_ROOT: volume},
    secrets=[secret],
    cpu=4,
    memory=32768,
    timeout=60 * 60 * 2,
)
def prepare_archive(force: bool = False) -> dict[str, Any]:
    """Download the HF archive and extract it into the Modal volume."""
    from huggingface_hub import hf_hub_download

    token = os.environ["HF_TOKEN"]
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = ARCHIVE_DIR / ARCHIVE_NAME
    extracted_marker = EXTRACT_DIR / ".complete"

    if not archive_path.exists() or force:
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            filename=ARCHIVE_PATH_IN_REPO,
            token=token,
            local_dir=ARCHIVE_DIR,
        )
        downloaded_path = Path(downloaded)
        if downloaded_path != archive_path:
            archive_path.write_bytes(downloaded_path.read_bytes())

    if force and extracted_marker.exists():
        extracted_marker.unlink()
    if not extracted_marker.exists():
        subprocess.run(
            ["tar", "--zstd", "-xf", str(archive_path), "-C", str(EXTRACT_DIR)],
            check=True,
        )
        extracted_marker.write_text("ok\n", encoding="utf-8")

    volume.commit()
    return {
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "extract_dir": str(EXTRACT_DIR),
        "top_level": [p.name for p in sorted(EXTRACT_DIR.iterdir())],
    }


@app.function(
    image=image,
    volumes={VOLUME_ROOT: volume},
    cpu=16,
    memory=65536,
    timeout=60 * 60 * 4,
)
def analyze_failure() -> dict[str, Any]:
    """Analyze loop onset and H2O KV eviction behavior."""
    sys.path.insert(0, "/opt/recoding_figures")
    import numpy as np
    from kv_eviction_metrics import compute_phase_distribution, load_segments_by_iter_dir
    from recording_loader import find_attempt_dirs, load_iteration_records, load_kv_eviction

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    attempt_dirs = find_attempt_dirs([EXTRACT_DIR])
    if not attempt_dirs:
        raise FileNotFoundError(f"no attempt dirs under {EXTRACT_DIR}")
    records = load_iteration_records(attempt_dirs)
    frame = load_kv_eviction(records)
    segments_by_iter_dir = load_segments_by_iter_dir(records)

    trace_files = sorted(EXTRACT_DIR.rglob("attempt_*/trace.jsonl"))
    tool_files = sorted(EXTRACT_DIR.rglob("attempt_*/tool_calls.json"))
    result_files = sorted(EXTRACT_DIR.rglob("results.jsonl"))
    run_manifest_files = sorted(EXTRACT_DIR.rglob("attempt_*/run_manifest.json"))
    meta_files = sorted(EXTRACT_DIR.rglob("recordings/meta.json"))

    actions = []
    for trace in trace_files:
        actions.extend(_extract_llm_actions(trace))
    actions.sort(key=lambda row: row["call_idx"])
    _write_csv(
        OUTPUT_DIR / "llm_actions.csv",
        actions,
        [
            "call_idx",
            "finish_reason",
            "tool_count",
            "tool_summary",
            "tool_signature",
            "content_chars",
            "content_excerpt",
        ],
    )

    loop = _detect_loop(actions)
    (OUTPUT_DIR / "loop_detection.json").write_text(
        json.dumps(loop, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    eviction_rows = _eviction_rows(frame)
    _write_csv(
        OUTPUT_DIR / "eviction_events.csv",
        eviction_rows,
        [
            "call_idx",
            "layer",
            "step",
            "phase",
            "pre_len",
            "post_len",
            "budget",
            "n_evicted",
            "reason",
        ],
    )

    role_rows, segment_rows = _role_and_segment_eviction_rows(
        frame, segments_by_iter_dir
    )
    _write_csv(
        OUTPUT_DIR / "role_eviction_by_row.csv",
        role_rows,
        [
            "call_idx",
            "layer",
            "step",
            "phase",
            "reason",
            "pre_len",
            "post_len",
            "role",
            "total_tokens",
            "kept_tokens",
            "evicted_tokens",
            "evicted_rate",
        ],
    )
    _write_csv(
        OUTPUT_DIR / "segment_eviction_top.csv",
        segment_rows[:500],
        [
            "call_idx",
            "layer",
            "step",
            "phase",
            "reason",
            "pre_len",
            "post_len",
            "role",
            "message_index",
            "token_start",
            "token_end",
            "segment_tokens",
            "kept_tokens",
            "evicted_tokens",
            "evicted_rate",
        ],
    )

    phase_distribution = compute_phase_distribution(frame, run_label=RUN_LABEL)
    layer_summary = _layer_summary(frame)
    call_summary = _call_summary(frame)
    critical = _critical_eviction_context(
        frame=frame,
        loop_start_call=loop.get("start_call_idx"),
        segments_by_iter_dir=segments_by_iter_dir,
    )

    summary = {
        "run_label": RUN_LABEL,
        "extract_dir": str(EXTRACT_DIR),
        "attempt_dirs": [str(p) for p in attempt_dirs],
        "trace_files": [str(p) for p in trace_files],
        "tool_files": [str(p) for p in tool_files],
        "results": [_load_jsonl(path) for path in result_files],
        "run_manifest": [_load_json(path) for path in run_manifest_files[:1]],
        "recording_meta": [_load_json(path) for path in meta_files[:1]],
        "n_iteration_records": len(records),
        "iteration_records": [
            {
                "call_idx": r.call_idx,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "total_tokens": r.total_tokens,
                "iter_dir": str(r.iter_dir),
            }
            for r in records
        ],
        "n_llm_actions": len(actions),
        "loop_detection": loop,
        "kv_eviction": {
            "n_rows": frame.n_rows,
            "policy_names": sorted({str(x) for x in frame.policy_name.tolist()}),
            "phase_distribution": phase_distribution,
            "reason_counts": dict(Counter(str(x) for x in frame.evict_reason.tolist())),
            "layer_summary": layer_summary,
            "call_summary": call_summary,
            "critical_context": critical,
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_report(OUTPUT_DIR / "REPORT.md", summary)
    volume.commit()
    return {
        "output_dir": str(OUTPUT_DIR),
        "summary": {
            "n_llm_actions": len(actions),
            "loop_start_call_idx": loop.get("start_call_idx"),
            "loop_reason": loop.get("reason"),
            "n_eviction_rows": frame.n_rows,
            "reason_counts": summary["kv_eviction"]["reason_counts"],
            "critical_rows": len(critical),
        },
    }


@app.local_entrypoint()
def main(action: str = "prepare", force: bool = False) -> None:
    """actions: prepare | analyze | all"""
    if action == "prepare":
        print(json.dumps(prepare_archive.remote(force), indent=2))
        return
    if action == "analyze":
        print(json.dumps(analyze_failure.remote(), indent=2))
        return
    if action == "all":
        print(json.dumps(prepare_archive.remote(force), indent=2))
        print(json.dumps(analyze_failure.remote(), indent=2))
        return
    raise ValueError("action must be one of: prepare, analyze, all")


def _extract_llm_actions(trace_path: Path) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("action_type") != "llm_call":
                continue
            data = rec.get("data") or {}
            rr = data.get("raw_response") or {}
            msg = _first_choice_message(rr)
            tool_calls = _tool_calls_from_message(msg)
            content = str((msg or {}).get("content") or "")
            finish_reason = _first_choice_finish_reason(rr)
            summary = _summarize_tool_calls(tool_calls)
            actions.append(
                {
                    "call_idx": int(rec.get("iteration") or len(actions)),
                    "finish_reason": finish_reason,
                    "tool_count": len(tool_calls),
                    "tool_summary": summary,
                    "tool_signature": _normalize_tool_signature(tool_calls),
                    "content_chars": len(content),
                    "content_excerpt": content[:240].replace("\n", "\\n"),
                }
            )
    return actions


def _first_choice_message(raw_response: Any) -> dict[str, Any] | None:
    if not isinstance(raw_response, dict):
        return None
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    msg = first.get("message")
    return msg if isinstance(msg, dict) else None


def _first_choice_finish_reason(raw_response: Any) -> str:
    if not isinstance(raw_response, dict):
        return ""
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    return str(first.get("finish_reason") or "")


def _tool_calls_from_message(msg: dict[str, Any] | None) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not msg:
        return out
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") if isinstance(tc, dict) else None
        if not isinstance(fn, dict):
            continue
        out.append(
            {
                "name": str(fn.get("name") or ""),
                "arguments": str(fn.get("arguments") or ""),
            }
        )
    return out


def _summarize_tool_calls(tool_calls: list[dict[str, str]]) -> str:
    if not tool_calls:
        return "<empty>"
    parts: list[str] = []
    for tc in tool_calls:
        args = tc["arguments"]
        if len(args) > 100:
            args = args[:97] + "..."
        parts.append(f"{tc['name']}({args})")
    return " | ".join(parts)


def _normalize_tool_signature(tool_calls: list[dict[str, str]]) -> str:
    if not tool_calls:
        return "<empty>"
    parts: list[str] = []
    for tc in tool_calls:
        name = tc["name"]
        args = tc["arguments"]
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                # Keep the operation shape but drop incidental whitespace and
                # long scalar tails; exact command repetition still survives.
                args = json.dumps(parsed, sort_keys=True, ensure_ascii=False)
        except Exception:
            args = re.sub(r"\s+", " ", args).strip()
        if len(args) > 300:
            args = args[:300]
        parts.append(f"{name}:{args}")
    return " || ".join(parts)


def _detect_loop(actions: list[dict[str, Any]]) -> dict[str, Any]:
    if not actions:
        return {"status": "no_actions", "start_call_idx": None}
    signatures = [str(a["tool_signature"]) for a in actions]
    runs: list[dict[str, Any]] = []
    i = 0
    while i < len(signatures):
        j = i + 1
        while j < len(signatures) and signatures[j] == signatures[i]:
            j += 1
        if j - i >= 3:
            runs.append(
                {
                    "start_call_idx": int(actions[i]["call_idx"]),
                    "end_call_idx": int(actions[j - 1]["call_idx"]),
                    "length": j - i,
                    "signature": signatures[i],
                    "tool_summary": actions[i]["tool_summary"],
                }
            )
        i = j
    if runs:
        first = runs[0]
        return {
            "status": "loop_detected",
            "reason": ">=3 consecutive identical tool-call signatures",
            **first,
            "all_runs": runs,
        }

    # Fallback: repeated short cycle near the tail.
    for period in (2, 3, 4):
        if len(signatures) < period * 3:
            continue
        tail = signatures[-period * 3 :]
        if tail[:period] == tail[period : 2 * period] == tail[2 * period :]:
            start = len(signatures) - period * 3
            return {
                "status": "cycle_detected",
                "reason": f"tail repeats with period {period}",
                "start_call_idx": int(actions[start]["call_idx"]),
                "end_call_idx": int(actions[-1]["call_idx"]),
                "length": period * 3,
                "signature": " <cycle> ".join(tail[:period]),
                "tool_summary": " <cycle> ".join(
                    str(a["tool_summary"]) for a in actions[start : start + period]
                ),
            }
    repeated = Counter(signatures).most_common(5)
    return {
        "status": "no_strict_loop_detected",
        "reason": "no >=3 consecutive identical signatures or repeated tail cycle",
        "start_call_idx": None,
        "most_common_signatures": [
            {"count": count, "signature": sig} for sig, count in repeated
        ],
    }


def _eviction_rows(frame: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(frame.n_rows):
        rows.append(
            {
                "call_idx": int(frame.call_idx[i]),
                "layer": int(frame.record_layer[i]),
                "step": int(frame.record_step[i]),
                "phase": str(frame.record_phase[i]),
                "pre_len": int(frame.pre_len[i]),
                "post_len": int(frame.post_len[i]),
                "budget": int(frame.budget[i]),
                "n_evicted": int(frame.pre_len[i]) - int(frame.post_len[i]),
                "reason": str(frame.evict_reason[i]),
            }
        )
    return rows


def _role_and_segment_eviction_rows(
    frame: Any, segments_by_iter_dir: dict[str, dict]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    role_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    for i in range(frame.n_rows):
        payload = segments_by_iter_dir.get(str(frame.iter_dir[i]))
        if not payload:
            continue
        segments = payload.get("segments", [])
        pre_len = int(frame.pre_len[i])
        kept = _sorted_row_indices(frame.kept_per_row[i])
        evicted = _sorted_row_indices(frame.evicted_per_row[i])
        role_totals: dict[str, Counter[str]] = defaultdict(Counter)
        segment_impacts: list[dict[str, Any]] = []
        for seg in segments:
            start = int(seg.get("token_start", seg.get("start", 0)) or 0)
            end = int(seg.get("token_end", seg.get("end", start)) or start)
            if start >= pre_len or end <= 0:
                continue
            lo, hi = max(0, start), min(pre_len, end)
            if hi <= lo:
                continue
            role = _normalize_role(seg)
            kept_n = _count_sorted_in_range(kept, lo, hi)
            evicted_n = _count_sorted_in_range(evicted, lo, hi)
            total = hi - lo
            role_totals[role]["total"] += total
            role_totals[role]["kept"] += kept_n
            role_totals[role]["evicted"] += evicted_n
            if evicted_n:
                segment_impacts.append(
                    {
                        **_row_id(frame, i),
                        "role": role,
                        "message_index": seg.get("message_index"),
                        "token_start": lo,
                        "token_end": hi,
                        "segment_tokens": total,
                        "kept_tokens": kept_n,
                        "evicted_tokens": evicted_n,
                        "evicted_rate": evicted_n / total if total else 0.0,
                    }
                )
        for role, counts in sorted(role_totals.items()):
            total = int(counts["total"])
            evicted_n = int(counts["evicted"])
            kept_n = int(counts["kept"])
            role_rows.append(
                {
                    **_row_id(frame, i),
                    "role": role,
                    "total_tokens": total,
                    "kept_tokens": kept_n,
                    "evicted_tokens": evicted_n,
                    "evicted_rate": evicted_n / total if total else 0.0,
                }
            )
        segment_impacts.sort(
            key=lambda row: (row["evicted_tokens"], row["evicted_rate"]),
            reverse=True,
        )
        segment_rows.extend(segment_impacts[:10])
    segment_rows.sort(
        key=lambda row: (
            row["call_idx"],
            row["layer"],
            -int(row["evicted_tokens"]),
        )
    )
    return role_rows, segment_rows


def _critical_eviction_context(
    *,
    frame: Any,
    loop_start_call: int | None,
    segments_by_iter_dir: dict[str, dict],
) -> list[dict[str, Any]]:
    if frame.n_rows == 0:
        return []
    candidates: set[int] = set()
    # First eviction row and all first score_missing rows.
    candidates.add(0)
    for i in range(frame.n_rows):
        if str(frame.evict_reason[i]) == "score_missing":
            candidates.add(i)
            if len(candidates) > 20:
                break
    if loop_start_call is not None:
        # Rows immediately before / during loop onset, plus largest eviction
        # row in that local window.
        window = [
            i
            for i in range(frame.n_rows)
            if int(frame.call_idx[i]) in range(max(0, loop_start_call - 2), loop_start_call + 2)
        ]
        candidates.update(window[:20])
        if window:
            candidates.add(
                max(
                    window,
                    key=lambda idx: int(frame.pre_len[idx]) - int(frame.post_len[idx]),
                )
            )
    # Largest eviction rows overall.
    largest = sorted(
        range(frame.n_rows),
        key=lambda idx: int(frame.pre_len[idx]) - int(frame.post_len[idx]),
        reverse=True,
    )[:10]
    candidates.update(largest)

    out: list[dict[str, Any]] = []
    for i in sorted(candidates):
        payload = segments_by_iter_dir.get(str(frame.iter_dir[i]), {})
        segments = payload.get("segments", [])
        role_summary, top_segments = _row_role_segment_summary(frame, i, segments)
        out.append(
            {
                **_row_id(frame, i),
                "n_evicted": int(frame.pre_len[i]) - int(frame.post_len[i]),
                "top_score_indices": _top_score_pairs(frame, i)[:20],
                "role_summary": role_summary,
                "top_evicted_segments": top_segments[:10],
            }
        )
    return out


def _row_role_segment_summary(
    frame: Any, i: int, segments: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pre_len = int(frame.pre_len[i])
    kept = _sorted_row_indices(frame.kept_per_row[i])
    evicted = _sorted_row_indices(frame.evicted_per_row[i])
    roles: dict[str, Counter[str]] = defaultdict(Counter)
    top_segments: list[dict[str, Any]] = []
    for seg in segments:
        start = int(seg.get("token_start", seg.get("start", 0)) or 0)
        end = int(seg.get("token_end", seg.get("end", start)) or start)
        lo, hi = max(0, start), min(pre_len, end)
        if hi <= lo:
            continue
        role = _normalize_role(seg)
        kept_n = _count_sorted_in_range(kept, lo, hi)
        evicted_n = _count_sorted_in_range(evicted, lo, hi)
        total = hi - lo
        roles[role]["total"] += total
        roles[role]["kept"] += kept_n
        roles[role]["evicted"] += evicted_n
        if evicted_n:
            top_segments.append(
                {
                    "role": role,
                    "message_index": seg.get("message_index"),
                    "token_start": lo,
                    "token_end": hi,
                    "segment_tokens": total,
                    "kept_tokens": kept_n,
                    "evicted_tokens": evicted_n,
                    "evicted_rate": evicted_n / total if total else 0.0,
                    "has_content": bool(seg.get("has_content")),
                    "has_tool_calls": bool(seg.get("has_tool_calls")),
                }
            )
    role_summary = []
    for role, counts in sorted(roles.items()):
        total = int(counts["total"])
        evicted_n = int(counts["evicted"])
        role_summary.append(
            {
                "role": role,
                "total_tokens": total,
                "kept_tokens": int(counts["kept"]),
                "evicted_tokens": evicted_n,
                "evicted_rate": evicted_n / total if total else 0.0,
            }
        )
    top_segments.sort(
        key=lambda row: (row["evicted_tokens"], row["evicted_rate"]),
        reverse=True,
    )
    return role_summary, top_segments


def _row_id(frame: Any, i: int) -> dict[str, Any]:
    return {
        "call_idx": int(frame.call_idx[i]),
        "layer": int(frame.record_layer[i]),
        "step": int(frame.record_step[i]),
        "phase": str(frame.record_phase[i]),
        "reason": str(frame.evict_reason[i]),
        "pre_len": int(frame.pre_len[i]),
        "post_len": int(frame.post_len[i]),
    }


def _top_score_pairs(frame: Any, i: int) -> list[dict[str, Any]]:
    if frame.score_topk_index.size == 0:
        return []
    pairs = []
    for idx, val in zip(frame.score_topk_index[i], frame.score_topk_value[i], strict=False):
        idx_i = int(idx)
        if idx_i < 0:
            continue
        value = None if not _is_finite(float(val)) else float(val)
        pairs.append({"index": idx_i, "score": value})
    return pairs


def _sorted_row_indices(indices: Any) -> Any:
    import numpy as np

    arr = np.asarray(indices, dtype=np.int64)
    if arr.size <= 1:
        return arr
    if bool(np.all(arr[:-1] <= arr[1:])):
        return arr
    return np.sort(arr)


def _count_sorted_in_range(indices: Any, lo: int, hi: int) -> int:
    import numpy as np

    if hi <= lo or indices.size == 0:
        return 0
    left = int(np.searchsorted(indices, lo, side="left"))
    right = int(np.searchsorted(indices, hi, side="left"))
    return right - left


def _layer_summary(frame: Any) -> list[dict[str, Any]]:
    by_layer: dict[int, dict[str, Any]] = {}
    for i in range(frame.n_rows):
        layer = int(frame.record_layer[i])
        row = by_layer.setdefault(
            layer,
            {
                "layer": layer,
                "rows": 0,
                "evicted_total": 0,
                "max_pre_len": 0,
                "reason_counts": Counter(),
            },
        )
        row["rows"] += 1
        row["evicted_total"] += int(frame.pre_len[i]) - int(frame.post_len[i])
        row["max_pre_len"] = max(row["max_pre_len"], int(frame.pre_len[i]))
        row["reason_counts"][str(frame.evict_reason[i])] += 1
    out = []
    for row in sorted(by_layer.values(), key=lambda r: r["layer"]):
        row = dict(row)
        row["reason_counts"] = dict(row["reason_counts"])
        out.append(row)
    return out


def _call_summary(frame: Any) -> list[dict[str, Any]]:
    by_call: dict[int, dict[str, Any]] = {}
    for i in range(frame.n_rows):
        call = int(frame.call_idx[i])
        row = by_call.setdefault(
            call,
            {
                "call_idx": call,
                "rows": 0,
                "evicted_total": 0,
                "max_pre_len": 0,
                "min_post_len": None,
                "reason_counts": Counter(),
            },
        )
        row["rows"] += 1
        row["evicted_total"] += int(frame.pre_len[i]) - int(frame.post_len[i])
        row["max_pre_len"] = max(row["max_pre_len"], int(frame.pre_len[i]))
        post = int(frame.post_len[i])
        row["min_post_len"] = post if row["min_post_len"] is None else min(row["min_post_len"], post)
        row["reason_counts"][str(frame.evict_reason[i])] += 1
    out = []
    for row in sorted(by_call.values(), key=lambda r: r["call_idx"]):
        row = dict(row)
        row["reason_counts"] = dict(row["reason_counts"])
        out.append(row)
    return out


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    loop = summary["loop_detection"]
    kv = summary["kv_eviction"]
    lines = [
        "# H2O Causal Failure Analysis",
        "",
        f"- Run: `{summary['run_label']}`",
        f"- LLM calls: {summary['n_llm_actions']}",
        f"- Iteration records: {summary['n_iteration_records']}",
        f"- KV eviction rows: {kv['n_rows']}",
        f"- Loop status: {loop.get('status')} ({loop.get('reason')})",
        f"- Loop start call: {loop.get('start_call_idx')}",
        "",
        "## Eviction Reason Counts",
        "",
    ]
    for reason, count in kv["reason_counts"].items():
        lines.append(f"- `{reason}`: {count}")
    lines.extend(["", "## Critical Eviction Rows", ""])
    for row in kv["critical_context"][:12]:
        lines.append(
            f"- call={row['call_idx']} layer={row['layer']} step={row['step']} "
            f"phase={row['phase']} reason={row['reason']} "
            f"pre={row['pre_len']} post={row['post_len']} evicted={row['n_evicted']}"
        )
        for role in row["role_summary"]:
            if role["evicted_tokens"]:
                lines.append(
                    f"  - {role['role']}: evicted {role['evicted_tokens']}/"
                    f"{role['total_tokens']} ({role['evicted_rate']:.3f})"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _normalize_role(segment: dict[str, Any]) -> str:
    role = str(segment.get("role") or "other")
    if role == "assistant" and bool(segment.get("has_tool_calls")):
        return "assistant_call"
    if role == "assistant":
        return "assistant_message"
    if role in {"tool", "tool_result"}:
        return "tool_result"
    return role


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[Any]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def _is_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value
