"""Plot segment attention after filtering by recorded sparse keep sets.

This is a post-hoc diagnostic for observe-only sparse-attention runs. The model
generation remains dense, while `sparse_attention.npz` records the keep set that
the sparse method would have used. This script filters dense `attention.npz`
top-k rows by that runtime keep set and then rebuilds the segment grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.recoding_figures.recording_loader import (  # noqa: E402
    IterationRecord,
    decode_attention_topk,
    find_attempt_dirs,
    load_iteration_records,
)
from scripts.recoding_figures.sparse_keep_sets import (  # noqa: E402
    SparseParams,
    reconstruct_keep_set,
)

SUPPORTED_METHODS = {"sliding", "streaming", "heavy_hitter", "block_topk", "quest"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="attempt, task, or run directories")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-orphans", action="store_true")
    parser.add_argument("--max-iters", type=int)
    parser.add_argument(
        "--split-by-task",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write one grid per task. Default: true.",
    )
    args = parser.parse_args()

    summary = build_sparse_segment_grids(
        inputs=args.inputs,
        output_dir=args.output_dir,
        include_orphans=args.include_orphans,
        max_iters=args.max_iters,
        split_by_task=args.split_by_task,
    )
    print(json.dumps(_json_ready(summary), indent=2, sort_keys=True))


def build_sparse_segment_grids(
    *,
    inputs: Sequence[Path],
    output_dir: Path,
    include_orphans: bool = False,
    max_iters: int | None = None,
    split_by_task: bool = True,
) -> dict[str, Any]:
    """Build sparse-filtered segment grids for one or more attempt paths."""
    records = load_iteration_records(
        inputs,
        include_orphans=include_orphans,
        max_iters=max_iters,
    )
    attempt_dirs = find_attempt_dirs(inputs)
    sparse_meta = _load_and_validate_sparse_meta(attempt_dirs)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups: list[tuple[str, list[IterationRecord], Path]]
    if split_by_task:
        by_task: dict[str, list[IterationRecord]] = {}
        for record in records:
            by_task.setdefault(record.task, []).append(record)
        groups = [
            (task, task_records, output_dir / _safe_name(task))
            for task, task_records in sorted(by_task.items())
        ]
    else:
        groups = [("all_tasks", records, output_dir)]

    group_summaries = []
    for label, group_records, group_dir in groups:
        trajectory_rows, layer_rows = sparse_filtered_segment_rows(
            group_records,
            method_name=str(sparse_meta["method"]),
            method_params=SparseParams(
                sink_size=int(sparse_meta["sink_size"]),
                recent_window=int(sparse_meta["recent_window"]),
            ),
        )
        if not trajectory_rows:
            raise ValueError(f"{label}: no segment attention observations were found")
        summary_rows = _segment_summary_rows(trajectory_rows)
        group_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(group_dir / "segment_attention_sparse_filtered_trajectory.csv", trajectory_rows)
        _write_csv(group_dir / "segment_attention_sparse_filtered_by_layer.csv", layer_rows)
        _write_csv(group_dir / "segment_attention_sparse_filtered_summary.csv", summary_rows)
        plot_summary = _plot_segment_attention_grid(
            trajectory_rows,
            summary_rows,
            group_dir / "segment_attention_sparse_filtered_grid",
        )
        group_summary = {
            "label": label,
            "output_dir": str(group_dir),
            "n_records": len(group_records),
            "n_segments": len(summary_rows),
            "n_trajectory_rows": len(trajectory_rows),
            "n_layer_rows": len(layer_rows),
            "role_counts": _role_counts(summary_rows),
            "plot": plot_summary,
        }
        (group_dir / "summary.json").write_text(
            json.dumps(_json_ready(group_summary), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (group_dir / "summary.md").write_text(
            _summary_markdown(group_summary, sparse_meta),
            encoding="utf-8",
        )
        group_summaries.append(group_summary)

    run_summary = {
        "inputs": [str(path) for path in inputs],
        "attempt_dirs": [str(path) for path in attempt_dirs],
        "sparse_attention": sparse_meta,
        "split_by_task": split_by_task,
        "groups": group_summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_ready(run_summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return run_summary


def sparse_filtered_segment_rows(
    records: Sequence[IterationRecord],
    *,
    method_name: str,
    method_params: SparseParams,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return trajectory and by-layer rows after runtime sparse filtering."""
    trajectory: dict[tuple[str, int, str], dict[str, Any]] = {}
    by_layer: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    metadata: dict[str, dict[str, Any]] = {}

    for record in sorted(records, key=lambda item: (item.task, item.call_idx)):
        segments_payload = _load_json_required(record.iter_dir / "segments.json")
        segments = list(segments_payload.get("segments", []))
        segment_items = _segments_for_record(
            segments,
            record=record,
            metadata=metadata,
        )
        if not segment_items:
            continue

        with np.load(record.iter_dir / "attention.npz", allow_pickle=True) as attention:
            record_layers = attention["record_layer"].astype(np.int64)
            record_phases = attention["record_phase"].astype(str)
            record_decode_steps = attention["record_decode_step"].astype(np.int64)
            offsets = attention["query_row_offsets"].astype(np.int64)
            query_positions = attention["query_positions"].astype(np.int64)
            topk_indices, topk_weights = decode_attention_topk(attention)

            if offsets.shape[0] != record_layers.shape[0] + 1:
                raise ValueError(
                    f"{record.iter_dir}: query_row_offsets length does not match records"
                )
            if topk_indices.shape != topk_weights.shape:
                raise ValueError(
                    f"{record.iter_dir}: decoded top-k index/weight shape mismatch"
                )
            if topk_indices.shape[0] != query_positions.shape[0]:
                raise ValueError(
                    f"{record.iter_dir}: top-k rows {topk_indices.shape[0]} "
                    f"!= query_positions {query_positions.shape[0]}"
                )

            sparse_lookup = _sparse_lookup(record.iter_dir / "sparse_attention.npz")
            attention_keys: set[tuple[int, str, int]] = set()
            for record_idx, layer in enumerate(record_layers):
                start = int(offsets[record_idx])
                end = int(offsets[record_idx + 1])
                if end <= start:
                    continue
                if end > topk_indices.shape[0] or end > query_positions.shape[0]:
                    raise ValueError(f"{record.iter_dir}: query row offset exceeds stored rows")
                phase = str(record_phases[record_idx])
                if phase not in {"prefill", "decode"}:
                    continue
                layer_int = int(layer)
                decode_step = int(record_decode_steps[record_idx])
                sparse_key = (layer_int, phase, decode_step)
                attention_keys.add(sparse_key)
                sparse_row = sparse_lookup.get(sparse_key)
                if sparse_row is None:
                    raise ValueError(
                        f"{record.iter_dir.name}: attention.npz has record "
                        f"(layer={layer_int}, phase={phase}, dstep={decode_step}) "
                        "but no matching row in sparse_attention.npz"
                    )
                keep_arr = reconstruct_keep_set(
                    method_name=method_name,
                    method_params=method_params,
                    key_len=int(sparse_row["key_len"]),
                    extras=dict(sparse_row["extras"]),
                )
                if int(keep_arr.shape[0]) != int(sparse_row["kept_count"]):
                    raise ValueError(
                        f"{record.iter_dir / 'sparse_attention.npz'} row "
                        f"{sparse_row['row_idx']}: reconstructed keep_set size "
                        f"{int(keep_arr.shape[0])} != recorded kept_count "
                        f"{int(sparse_row['kept_count'])}"
                    )

                segment_mass_rows = _segment_mass_from_topk(
                    topk_indices[start:end],
                    topk_weights[start:end],
                    segments=segments,
                    keep_indices=keep_arr,
                )
                query_slice = query_positions[start:end]
                for segment in segment_items:
                    stats = _segment_attention_stats(
                        segment_mass_rows,
                        query_slice,
                        segment_idx=segment["segment_idx"],
                        token_start=segment["token_start"],
                        token_end=segment["token_end"],
                    )
                    _accumulate_segment_observation(
                        trajectory,
                        key=(segment["segment_id"], int(record.call_idx), phase),
                        metadata=segment,
                        record_call_idx=int(record.call_idx),
                        phase=phase,
                        layer=layer_int,
                        stats=stats,
                    )
                    _accumulate_segment_observation(
                        by_layer,
                        key=(segment["segment_id"], int(record.call_idx), phase, layer_int),
                        metadata=segment,
                        record_call_idx=int(record.call_idx),
                        phase=phase,
                        layer=layer_int,
                        stats=stats,
                    )

            sparse_keys = set(sparse_lookup.keys())
            missing = sorted(attention_keys - sparse_keys)
            if missing:
                raise ValueError(
                    f"{record.iter_dir}: sparse_attention.npz keys do not exactly "
                    f"cover attention.npz records; missing={missing}"
                )

    return _finalize_segment_rows(trajectory), _finalize_segment_rows(
        by_layer,
        include_layer=True,
    )


def _load_and_validate_sparse_meta(attempt_dirs: Sequence[Path]) -> dict[str, Any]:
    sparse_blocks: list[dict[str, Any]] = []
    for attempt_dir in attempt_dirs:
        meta_path = attempt_dir / "recordings" / "meta.json"
        meta = _load_json_required(meta_path)
        sparse_block = meta.get("sparse_attention")
        if not isinstance(sparse_block, dict):
            raise ValueError(f"{meta_path} has no sparse_attention block")
        method = str(sparse_block.get("method", ""))
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"{meta_path}: unsupported sparse method {method!r}")
        if sparse_block.get("observe_only") is not True:
            raise ValueError(
                f"{meta_path}: expected sparse_attention.observe_only == true; "
                "enforced sparse traces do not contain dense counterfactual top-k"
            )
        for item in meta.get("iters", []):
            if not isinstance(item, dict):
                continue
            integrity = item.get("recording_integrity")
            if not isinstance(integrity, dict):
                continue
            if integrity.get("sparse_attention_recording_enabled") is False:
                raise ValueError(f"{meta_path}: sparse attention recording was disabled")
            if integrity.get("sparse_attention_observe_only") is not True:
                raise ValueError(f"{meta_path}: iter integrity is not observe-only")
            if integrity.get("sparse_attention_records_match_expected") is not True:
                raise ValueError(f"{meta_path}: sparse attention record count mismatch")
            if integrity.get("sparse_attention_hooks_balanced") is not True:
                raise ValueError(f"{meta_path}: sparse attention hooks are unbalanced")
        for key in ("sink_size", "recent_window"):
            if key not in sparse_block:
                raise ValueError(f"{meta_path}: sparse_attention missing {key!r}")
        sparse_blocks.append(
            {
                "method": method,
                "sink_size": int(sparse_block["sink_size"]),
                "recent_window": int(sparse_block["recent_window"]),
                "budget": _optional_int(sparse_block.get("budget")),
                "block_size": _optional_int(sparse_block.get("block_size")),
                "score_reduction": sparse_block.get("score_reduction"),
                "phase_scope": sparse_block.get("phase_scope"),
                "observe_only": bool(sparse_block.get("observe_only", False)),
            }
        )
    if not sparse_blocks:
        raise ValueError("no attempt sparse_attention metadata found")
    first = sparse_blocks[0]
    for block in sparse_blocks[1:]:
        if block != first:
            raise ValueError(
                "all attempts must use identical sparse_attention config; "
                f"got {first} and {block}"
            )
    return first


def _sparse_lookup(npz_path: Path) -> dict[tuple[int, str, int], dict[str, Any]]:
    if not npz_path.is_file():
        raise FileNotFoundError(f"missing sparse attention artifact: {npz_path}")
    lookup: dict[tuple[int, str, int], dict[str, Any]] = {}
    with np.load(npz_path, allow_pickle=True) as data:
        layers = data["record_layer"].astype(np.int32)
        phases = data["record_phase"].astype(str)
        dsteps = data["record_decode_step"].astype(np.int32)
        key_lens = data["key_len"].astype(np.int32)
        kept_counts = data["kept_count"].astype(np.int32)
        extras_json = data["extras_json"]
        for row_idx in range(layers.shape[0]):
            key = (int(layers[row_idx]), str(phases[row_idx]), int(dsteps[row_idx]))
            if key in lookup:
                raise ValueError(f"{npz_path}: duplicate sparse row key {key}")
            lookup[key] = {
                "row_idx": row_idx,
                "key_len": int(key_lens[row_idx]),
                "kept_count": int(kept_counts[row_idx]),
                "extras": json.loads(str(extras_json[row_idx])),
            }
    return lookup


def _segment_mass_from_topk(
    topk_indices: np.ndarray,
    topk_weights: np.ndarray,
    *,
    segments: Sequence[dict[str, Any]],
    keep_indices: np.ndarray,
) -> np.ndarray:
    if topk_indices.shape != topk_weights.shape:
        raise ValueError("top-k indices and weights must have the same shape")
    keep = np.zeros(max(_max_key_index(topk_indices), int(keep_indices.max(initial=-1))) + 1, dtype=bool)
    if keep_indices.size:
        keep[keep_indices.astype(np.int64)] = True
    valid = topk_indices >= 0
    in_keep = np.zeros_like(valid, dtype=bool)
    safe_indices = topk_indices.astype(np.int64, copy=False)
    indexable = valid & (safe_indices < keep.shape[0])
    in_keep[indexable] = keep[safe_indices[indexable]]

    masses = np.zeros((topk_indices.shape[0], len(segments)), dtype=np.float64)
    weights = topk_weights.astype(np.float64, copy=False)
    for segment_idx, segment in enumerate(segments):
        token_start = _int_field(segment, "token_start", default=0)
        token_end = _int_field(segment, "token_end", default=token_start)
        if token_end <= token_start:
            continue
        in_segment = (
            in_keep
            & (safe_indices >= int(token_start))
            & (safe_indices < int(token_end))
        )
        masses[:, segment_idx] = np.where(in_segment, weights, 0.0).sum(axis=1)
    return masses


def _max_key_index(topk_indices: np.ndarray) -> int:
    if topk_indices.size == 0:
        return -1
    valid = topk_indices[topk_indices >= 0]
    if valid.size == 0:
        return -1
    return int(valid.max())


def _segments_for_record(
    segments: Sequence[dict[str, Any]],
    *,
    record: IterationRecord,
    metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for segment_idx, segment in enumerate(segments):
        role = str(segment.get("role") or "other")
        first_seen_call = _int_field(segment, "first_seen_call", default=record.call_idx)
        if int(record.call_idx) < first_seen_call:
            continue
        token_start = _int_field(segment, "token_start", default=0)
        token_end = _int_field(segment, "token_end", default=token_start)
        token_count = max(0, token_end - token_start)
        if token_count <= 0:
            continue
        message_index = _int_field(segment, "message_index", default=segment_idx)
        segment_id, identity_source = _segment_identity(
            segment,
            task=record.task,
            role=role,
            first_seen_call=first_seen_call,
            message_index=message_index,
            segment_idx=segment_idx,
        )
        existing = metadata.get(segment_id)
        if existing is None:
            existing = {
                "segment_ordinal": len(metadata) + 1,
                "segment_id": segment_id,
                "identity_source": identity_source,
                "task": record.task,
                "role": role,
                "tool_call_id": segment.get("tool_call_id"),
                "tool_name": segment.get("name"),
                "first_seen_call": first_seen_call,
                "first_seen_call_inferred": bool(segment.get("first_seen_call_inferred", False)),
                "message_index": message_index,
                "has_content": bool(segment.get("has_content", False)),
                "has_tool_calls": bool(segment.get("has_tool_calls", False)),
                "initial_token_count": token_count,
            }
            metadata[segment_id] = existing
        items.append(
            {
                **existing,
                "segment_idx": segment_idx,
                "token_start": token_start,
                "token_end": token_end,
                "token_count": token_count,
            }
        )
    return items


def _segment_identity(
    segment: dict[str, Any],
    *,
    task: str,
    role: str,
    first_seen_call: int,
    message_index: int,
    segment_idx: int,
) -> tuple[str, str]:
    if role == "tool_result" and segment.get("tool_call_id"):
        return f"{task}:tool_result:{segment['tool_call_id']}", "task_tool_call_id"
    return (
        f"{task}:{role}:call{first_seen_call}:message{message_index}:segment{segment_idx}",
        "task_segment",
    )


def _segment_attention_stats(
    segment_mass_rows: np.ndarray,
    query_positions: np.ndarray,
    *,
    segment_idx: int,
    token_start: int,
    token_end: int,
) -> dict[str, Any]:
    values = np.asarray(segment_mass_rows[:, segment_idx], dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values < -1e-12):
        raise ValueError("segment attention mass contains negative or non-finite values")
    key_lengths = np.asarray(query_positions, dtype=np.float64) + 1.0
    visible_tokens = np.minimum(float(token_end), key_lengths) - float(token_start)
    visible_tokens = np.clip(visible_tokens, 0.0, float(max(token_end - token_start, 0)))
    baseline = np.divide(
        visible_tokens,
        key_lengths,
        out=np.zeros_like(visible_tokens, dtype=np.float64),
        where=key_lengths > 0,
    )
    visible_mask = visible_tokens > 0
    return {
        "row_count": int(values.shape[0]),
        "visible_row_count": int(visible_mask.sum()),
        "mass_sum": float(values.sum()),
        "visible_mass_sum": float(values[visible_mask].sum()) if bool(visible_mask.any()) else 0.0,
        "baseline_sum": float(baseline.sum()),
        "visible_baseline_sum": (
            float(baseline[visible_mask].sum()) if bool(visible_mask.any()) else 0.0
        ),
    }


def _accumulate_segment_observation(
    target: dict[Any, dict[str, Any]],
    *,
    key: Any,
    metadata: dict[str, Any],
    record_call_idx: int,
    phase: str,
    layer: int,
    stats: dict[str, Any],
) -> None:
    row = target.setdefault(
        key,
        {
            "segment_ordinal": metadata["segment_ordinal"],
            "segment_id": metadata["segment_id"],
            "identity_source": metadata["identity_source"],
            "task": metadata["task"],
            "role": metadata["role"],
            "tool_call_id": metadata.get("tool_call_id"),
            "tool_name": metadata.get("tool_name"),
            "first_seen_call": metadata["first_seen_call"],
            "first_seen_call_inferred": metadata["first_seen_call_inferred"],
            "message_index": metadata["message_index"],
            "has_content": metadata["has_content"],
            "has_tool_calls": metadata["has_tool_calls"],
            "observed_call_idx": record_call_idx,
            "age": record_call_idx - int(metadata["first_seen_call"]),
            "phase": phase,
            "token_count": metadata["token_count"],
            "initial_token_count": metadata["initial_token_count"],
            "layers": set(),
            "attention_records": 0,
            "row_count": 0,
            "visible_row_count": 0,
            "mass_sum": 0.0,
            "visible_mass_sum": 0.0,
            "baseline_sum": 0.0,
            "visible_baseline_sum": 0.0,
        },
    )
    row["layers"].add(layer)
    row["attention_records"] += 1
    row["row_count"] += int(stats["row_count"])
    row["visible_row_count"] += int(stats["visible_row_count"])
    row["mass_sum"] += float(stats["mass_sum"])
    row["visible_mass_sum"] += float(stats["visible_mass_sum"])
    row["baseline_sum"] += float(stats["baseline_sum"])
    row["visible_baseline_sum"] += float(stats["visible_baseline_sum"])


def _finalize_segment_rows(
    aggregates: dict[Any, dict[str, Any]],
    *,
    include_layer: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, aggregate in aggregates.items():
        row_count = int(aggregate["row_count"])
        visible_row_count = int(aggregate["visible_row_count"])
        attention_share = _safe_div(aggregate["mass_sum"], row_count)
        visible_attention_share = _safe_div(aggregate["visible_mass_sum"], visible_row_count)
        baseline_share = _safe_div(aggregate["baseline_sum"], row_count)
        visible_baseline_share = _safe_div(
            aggregate["visible_baseline_sum"],
            visible_row_count,
        )
        row = {
            "segment_ordinal": aggregate["segment_ordinal"],
            "segment_id": aggregate["segment_id"],
            "identity_source": aggregate["identity_source"],
            "task": aggregate["task"],
            "role": aggregate["role"],
            "tool_call_id": aggregate.get("tool_call_id"),
            "tool_name": aggregate.get("tool_name"),
            "first_seen_call": aggregate["first_seen_call"],
            "first_seen_call_inferred": aggregate["first_seen_call_inferred"],
            "message_index": aggregate["message_index"],
            "has_content": aggregate["has_content"],
            "has_tool_calls": aggregate["has_tool_calls"],
            "observed_call_idx": aggregate["observed_call_idx"],
            "age": aggregate["age"],
            "phase": aggregate["phase"],
            "token_count": aggregate["token_count"],
            "initial_token_count": aggregate["initial_token_count"],
            "n_layers": len(aggregate["layers"]),
            "attention_records": aggregate["attention_records"],
            "row_count": row_count,
            "visible_row_count": visible_row_count,
            "attention_share_mean": attention_share,
            "visible_attention_share_mean": visible_attention_share,
            "baseline_share_mean": baseline_share,
            "visible_baseline_share_mean": visible_baseline_share,
            "attention_excess": (
                attention_share - baseline_share
                if attention_share is not None and baseline_share is not None
                else None
            ),
            "visible_attention_excess": (
                visible_attention_share - visible_baseline_share
                if visible_attention_share is not None and visible_baseline_share is not None
                else None
            ),
            "attention_over_baseline": _safe_div(attention_share, baseline_share),
            "visible_attention_over_baseline": _safe_div(
                visible_attention_share,
                visible_baseline_share,
            ),
        }
        if include_layer:
            row["layer"] = int(key[3])
        rows.append(row)
    phase_order = {"prefill": 0, "decode": 1}
    return sorted(
        rows,
        key=lambda row: (
            str(row["task"]),
            int(row["segment_ordinal"]),
            int(row["observed_call_idx"]),
            phase_order.get(str(row["phase"]), 99),
            int(row.get("layer", -1)),
        ),
    )


def _segment_summary_rows(trajectory_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in trajectory_rows:
        grouped.setdefault(str(row["segment_id"]), []).append(dict(row))

    summary_rows: list[dict[str, Any]] = []
    for rows in grouped.values():
        first = min(
            rows,
            key=lambda row: (
                str(row["task"]),
                int(row["first_seen_call"]),
                int(row["message_index"]),
            ),
        )
        out = {
            "segment_ordinal": first["segment_ordinal"],
            "segment_id": first["segment_id"],
            "identity_source": first["identity_source"],
            "task": first["task"],
            "role": first["role"],
            "tool_call_id": first.get("tool_call_id"),
            "tool_name": first.get("tool_name"),
            "first_seen_call": first["first_seen_call"],
            "first_seen_call_inferred": first["first_seen_call_inferred"],
            "message_index": first["message_index"],
            "has_content": first["has_content"],
            "has_tool_calls": first["has_tool_calls"],
            "initial_token_count": first["initial_token_count"],
            "max_observed_age": max(int(row["age"]) for row in rows),
        }
        for phase in ("prefill", "decode"):
            phase_rows = sorted(
                [row for row in rows if row["phase"] == phase],
                key=lambda row: int(row["age"]),
            )
            out.update(_segment_phase_summary(phase, phase_rows))
        summary_rows.append(out)

    return sorted(
        summary_rows,
        key=lambda row: (
            str(row["task"]),
            int(row["first_seen_call"]),
            int(row["message_index"]),
            str(row["role"]),
        ),
    )


def _segment_phase_summary(phase: str, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            f"{phase}_observed_calls": 0,
            f"mean_{phase}_visible_attention_share": None,
            f"peak_{phase}_visible_attention_share": None,
            f"peak_{phase}_age": None,
            f"mean_{phase}_visible_attention_over_baseline": None,
            f"peak_{phase}_visible_attention_over_baseline": None,
            f"peak_{phase}_ratio_age": None,
        }
    shares = np.asarray(
        [
            np.nan
            if row["visible_attention_share_mean"] is None
            else float(row["visible_attention_share_mean"])
            for row in rows
        ],
        dtype=np.float64,
    )
    ratios = np.asarray(
        [
            np.nan
            if row["visible_attention_over_baseline"] is None
            else float(row["visible_attention_over_baseline"])
            for row in rows
        ],
        dtype=np.float64,
    )
    ages = np.asarray([int(row["age"]) for row in rows], dtype=np.float64)
    peak_idx = _nanargmax_or_none(shares)
    ratio_peak_idx = _nanargmax_or_none(ratios)
    return {
        f"{phase}_observed_calls": len(rows),
        f"mean_{phase}_visible_attention_share": _nanmean_or_none(shares),
        f"peak_{phase}_visible_attention_share": (
            float(shares[peak_idx]) if peak_idx is not None else None
        ),
        f"peak_{phase}_age": int(ages[peak_idx]) if peak_idx is not None else None,
        f"mean_{phase}_visible_attention_over_baseline": _nanmean_or_none(ratios),
        f"peak_{phase}_visible_attention_over_baseline": (
            float(ratios[ratio_peak_idx]) if ratio_peak_idx is not None else None
        ),
        f"peak_{phase}_ratio_age": int(ages[ratio_peak_idx]) if ratio_peak_idx is not None else None,
    }


def _plot_segment_attention_grid(
    trajectory_rows: Sequence[dict[str, Any]],
    summary_rows: Sequence[dict[str, Any]],
    output_stem: Path,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    phases = ("prefill", "decode")
    segment_order = [str(row["segment_id"]) for row in summary_rows]
    segment_to_row = {segment_id: idx for idx, segment_id in enumerate(segment_order)}
    max_age = max(int(row["age"]) for row in trajectory_rows)
    share_matrices = {
        phase: np.full((len(segment_order), max_age + 1), np.nan, dtype=np.float64)
        for phase in phases
    }
    ratio_matrices = {
        phase: np.full((len(segment_order), max_age + 1), np.nan, dtype=np.float64)
        for phase in phases
    }
    for row in trajectory_rows:
        phase = str(row["phase"])
        if phase not in share_matrices:
            continue
        row_idx = segment_to_row.get(str(row["segment_id"]))
        if row_idx is None:
            continue
        age = int(row["age"])
        share = _optional_float(row["visible_attention_share_mean"])
        ratio = _optional_float(row["visible_attention_over_baseline"])
        if share is not None:
            share_matrices[phase][row_idx, age] = share * 100.0
        if ratio is not None and ratio > 0:
            ratio_matrices[phase][row_idx, age] = float(np.log2(ratio))

    labels = [_segment_plot_label(row) for row in summary_rows]
    share_cmap = LinearSegmentedColormap.from_list(
        "asb_sparse_segment_share",
        ["#f7f4ed", "#d8d1c7", "#9faf9b", "#4f7771"],
    )
    ratio_cmap = LinearSegmentedColormap.from_list(
        "asb_sparse_segment_ratio",
        ["#b88c8c", "#f3eee8", "#7f8f84"],
    )
    share_cmap.set_bad("#ece7df")
    ratio_cmap.set_bad("#ece7df")
    finite_share = np.concatenate(
        [matrix[np.isfinite(matrix)] for matrix in share_matrices.values()]
    )
    share_vmax = float(np.percentile(finite_share, 95)) if finite_share.size else 1.0
    share_vmax = max(share_vmax, 0.01)
    finite_ratio = np.concatenate(
        [matrix[np.isfinite(matrix)] for matrix in ratio_matrices.values()]
    )
    ratio_abs = float(np.percentile(np.abs(finite_ratio), 95)) if finite_ratio.size else 1.0
    ratio_abs = max(ratio_abs, 0.25)

    width = max(11.0, min(18.0, 6.8 + 0.42 * (max_age + 1)))
    height = max(8.5, min(22.0, 2.8 + 0.30 * len(segment_order)))
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(width, height),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    images = []
    for col, phase in enumerate(phases):
        images.append(
            axes[0, col].imshow(
                share_matrices[phase],
                aspect="auto",
                cmap=share_cmap,
                vmin=0.0,
                vmax=share_vmax,
            )
        )
        images.append(
            axes[1, col].imshow(
                ratio_matrices[phase],
                aspect="auto",
                cmap=ratio_cmap,
                norm=TwoSlopeNorm(vmin=-ratio_abs, vcenter=0.0, vmax=ratio_abs),
            )
        )
        axes[0, col].set_title(f"{phase}: retained recorded attention share")
        axes[1, col].set_title(f"{phase}: retained attention / visible-token baseline")
        axes[1, col].set_xlabel("age in LLM calls since first visible")
    for row_idx in range(2):
        axes[row_idx, 0].set_ylabel("segment")
        axes[row_idx, 0].set_yticks(range(len(labels)))
        axes[row_idx, 0].set_yticklabels(labels, fontsize=5.5)
        axes[row_idx, 1].tick_params(axis="y", length=0)
    for ax in axes.ravel():
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", length=0)
        ax.set_xticks(range(max_age + 1))
        ax.set_xticklabels([str(age) for age in range(max_age + 1)])
        ax.set_xlim(-0.5, max_age + 0.5)
    cbar_share = fig.colorbar(images[0], ax=axes[0, :], fraction=0.025, pad=0.012)
    cbar_share.set_label("retained recorded attention share (%)")
    cbar_ratio = fig.colorbar(images[1], ax=axes[1, :], fraction=0.025, pad=0.012)
    cbar_ratio.set_label("log2(retained attention / token baseline)")
    fig.text(
        0.01,
        0.002,
        "Each row is one concrete segment. Missing cells mean the segment is not yet visible "
        "or that phase has no recorded rows. Values are dense recorded top-k mass retained "
        "by the runtime sparse keep set, not renormalized sparse-softmax mass.",
        fontsize=7,
        color="#555555",
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {
        "grid_png": str(output_stem.with_suffix(".png")),
        "grid_pdf": str(output_stem.with_suffix(".pdf")),
        "n_segments": len(segment_order),
        "max_age": max_age,
        "share_vmax_percentile_95": share_vmax,
        "ratio_log2_abs_percentile_95": ratio_abs,
    }


def _segment_plot_label(row: dict[str, Any]) -> str:
    role = str(row["role"])
    tool_name = row.get("tool_name")
    if role == "tool_result" and tool_name:
        role = f"tool:{tool_name}"
    return (
        f"{int(row['segment_ordinal']):02d} {role} "
        f"c{int(row['first_seen_call'])} m{int(row['message_index'])} "
        f"({int(row['initial_token_count'])}t)"
    )


def _role_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter(str(row["role"]) for row in rows)
    return dict(sorted(counts.items()))


def _summary_markdown(summary: dict[str, Any], sparse_meta: dict[str, Any]) -> str:
    lines = [
        "# Sparse-Filtered Segment Attention",
        "",
        f"- Label: `{summary['label']}`",
        f"- Output: `{summary['output_dir']}`",
        f"- Records: `{summary['n_records']}`",
        f"- Segments analyzed: `{summary['n_segments']}`",
        f"- Sparse method: `{sparse_meta['method']}`",
        f"- Budget: `{sparse_meta.get('budget')}`; sink: `{sparse_meta['sink_size']}`; recent: `{sparse_meta['recent_window']}`; block size: `{sparse_meta.get('block_size')}`",
        "",
        "## Figure",
        "",
        "- Top row: dense recorded top-k attention mass retained by the runtime sparse keep set.",
        "- Bottom row: log2(retained attention divided by dense visible-token baseline).",
        "- Left column: prefill. Right column: decode.",
        "",
        "## Caveat",
        "",
        "This is a counterfactual retained-mass diagnostic over observe-only artifacts. "
        "It does not rerun inference with sparse attention and does not renormalize "
        "post-sparse softmax mass.",
        "",
    ]
    return "\n".join(lines)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, set):
        return json.dumps(sorted(value))
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_ready(value), sort_keys=True)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_ready(item) for item in value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    if isinstance(value, Path):
        return str(value)
    return value


def _load_json_required(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _int_field(mapping: dict[str, Any], key: str, *, default: int) -> int:
    value = mapping.get(key, default)
    if value is None:
        return default
    return int(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    return result


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _nanmean_or_none(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(finite.mean())


def _nanargmax_or_none(values: np.ndarray) -> int | None:
    finite_mask = np.isfinite(values)
    if not bool(finite_mask.any()):
        return None
    masked = np.where(finite_mask, values, -np.inf)
    return int(masked.argmax())


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


if __name__ == "__main__":
    main()
