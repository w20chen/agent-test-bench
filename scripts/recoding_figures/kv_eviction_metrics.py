"""Aggregate metrics for KV eviction trace analysis.

Pure data transformations + small loaders. No matplotlib here so the
module stays test-friendly. Callers wire the outputs into figures.

Function tiers:

  Pure (frame in, dict-rows out):
    compute_phase_distribution
    compute_role_survival_rows
    js_divergence / jaccard

  Loader (records in, per-layer aggregates out):
    aggregate_attention_role_per_layer
    aggregate_heavy_hitters_per_layer
    aggregate_sink_recent_share_per_layer
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np

from recording_loader import (
    IterationRecord,
    KVEvictionFrame,
    average_layer_matrix,
    collect_role_labels,
    decode_attention_topk,
    load_attention_distributions,
)


# ---------------------------------------------------------------------------
# Pure scalar primitives
# ---------------------------------------------------------------------------


def js_divergence(p: np.ndarray, q: np.ndarray, *, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence in nats. Returns 0 for matching dists."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {q.shape}")
    p_sum = float(p.sum())
    q_sum = float(q.sum())
    if p_sum <= 0 or q_sum <= 0:
        return 0.0
    p = p / p_sum
    q = q / q_sum
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m, eps) + 0.5 * _kl(q, m, eps)


def _kl(p: np.ndarray, q: np.ndarray, eps: float) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * (np.log(p[mask] + eps) - np.log(q[mask] + eps))))


def jaccard(a: set[int], b: set[int]) -> float:
    """Jaccard coefficient. 0 for two empties (convention)."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Eviction profile
# ---------------------------------------------------------------------------


def compute_phase_distribution(
    frame: KVEvictionFrame, run_label: str
) -> list[dict]:
    """Group eviction decisions by phase, return per-phase counters.

    One row per (run_label, phase) with totals + reason histogram. Empty
    frames return an empty list.
    """
    if frame.is_empty:
        return []
    rows: dict[str, dict] = {}
    for i in range(frame.n_rows):
        phase = str(frame.record_phase[i])
        budget = int(frame.budget[i])
        n_evicted = int(frame.pre_len[i]) - int(frame.post_len[i])
        reason = str(frame.evict_reason[i])
        row = rows.setdefault(
            phase,
            {
                "run": run_label,
                "phase": phase,
                "budget": budget,
                "n_decisions": 0,
                "n_decisions_with_evict": 0,
                "n_evicted_total": 0,
                "reasons": defaultdict(int),
            },
        )
        if row["budget"] != budget:
            row["budget"] = -1  # mixed budgets within one run shouldn't happen
        row["n_decisions"] += 1
        if n_evicted > 0:
            row["n_decisions_with_evict"] += 1
        row["n_evicted_total"] += n_evicted
        row["reasons"][reason] += 1

    out: list[dict] = []
    for phase, row in rows.items():
        row["reasons"] = dict(row["reasons"])
        out.append(row)
    out.sort(key=lambda r: r["phase"])
    return out


def compute_eviction_profile_rows(
    frame: KVEvictionFrame, run_label: str
) -> list[dict]:
    """One row per eviction decision in the frame; ready for CSV."""
    if frame.is_empty:
        return []
    out: list[dict] = []
    for i in range(frame.n_rows):
        n_evicted = int(frame.pre_len[i]) - int(frame.post_len[i])
        out.append(
            {
                "run": run_label,
                "task": str(frame.task[i]),
                "call_idx": int(frame.call_idx[i]),
                "layer": int(frame.record_layer[i]),
                "step": int(frame.record_step[i]),
                "phase": str(frame.record_phase[i]),
                "pre_len": int(frame.pre_len[i]),
                "post_len": int(frame.post_len[i]),
                "budget": int(frame.budget[i]),
                "n_evicted": n_evicted,
                "reason": str(frame.evict_reason[i]),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Role survival
# ---------------------------------------------------------------------------


def compute_role_survival_rows(
    frame: KVEvictionFrame,
    segments_by_iter_dir: dict[str, dict],
    role_labels: Sequence[str],
    run_label: str,
) -> list[dict]:
    """For each eviction decision × role, compute kept/total token counts.

    `segments_by_iter_dir` maps the str(record.iter_dir) → parsed
    `segments.json` payload (dict with "segments" key).  `role_labels`
    is the canonical role order (typically `ROLE_ORDER`); roles outside
    the list are folded into "other".
    """
    if frame.is_empty:
        return []
    role_index = {role: idx for idx, role in enumerate(role_labels)}
    fallback = role_index.get("other")

    out: list[dict] = []
    for i in range(frame.n_rows):
        iter_dir_key = str(frame.iter_dir[i])
        payload = segments_by_iter_dir.get(iter_dir_key)
        if payload is None:
            # Skip rows whose segments.json we did not preload; surface count
            # at the caller via a `missing` counter rather than failing here.
            continue
        segments = payload.get("segments", [])
        key_len = int(frame.pre_len[i])
        total_by_role = _role_token_totals_under_key_len(
            segments, role_index, fallback, key_len
        )
        kept_by_role = _kept_role_totals(
            segments, role_index, fallback, frame.kept_per_row[i]
        )
        for role, role_col in role_index.items():
            total = float(total_by_role[role_col])
            kept = float(kept_by_role[role_col])
            if total <= 0 and kept <= 0:
                continue
            out.append(
                {
                    "run": run_label,
                    "task": str(frame.task[i]),
                    "call_idx": int(frame.call_idx[i]),
                    "layer": int(frame.record_layer[i]),
                    "step": int(frame.record_step[i]),
                    "phase": str(frame.record_phase[i]),
                    "role": role,
                    "total_tokens": total,
                    "kept_tokens": kept,
                    "survival_rate": float(kept / total) if total > 0 else 0.0,
                }
            )
    return out


def _role_token_totals_under_key_len(
    segments: Sequence[dict],
    role_index: dict[str, int],
    fallback: int | None,
    key_len: int,
) -> np.ndarray:
    """Per-role count of tokens whose absolute position lies in [0, key_len)."""
    counts = np.zeros(len(role_index), dtype=np.float64)
    if key_len <= 0:
        return counts
    for segment in segments:
        start = int(segment.get("token_start", segment.get("start", 0)) or 0)
        end = int(segment.get("token_end", segment.get("end", start)) or start)
        if end <= 0 or start >= key_len:
            continue
        length = max(0, min(end, key_len) - max(start, 0))
        if length <= 0:
            continue
        col = role_index.get(_normalize_role(segment), fallback)
        if col is None:
            continue
        counts[col] += float(length)
    return counts


def _kept_role_totals(
    segments: Sequence[dict],
    role_index: dict[str, int],
    fallback: int | None,
    kept_indices: np.ndarray,
) -> np.ndarray:
    """Bucket `kept_indices` (absolute positions) by segment role."""
    counts = np.zeros(len(role_index), dtype=np.float64)
    if kept_indices.size == 0:
        return counts
    # Sort segments by start so we can binary-search; segments.json is usually
    # already sorted but assume nothing.
    sorted_segments = sorted(
        segments,
        key=lambda s: int(s.get("token_start", s.get("start", 0)) or 0),
    )
    starts = np.asarray(
        [int(s.get("token_start", s.get("start", 0)) or 0) for s in sorted_segments],
        dtype=np.int64,
    )
    ends = np.asarray(
        [int(s.get("token_end", s.get("end", 0)) or 0) for s in sorted_segments],
        dtype=np.int64,
    )
    roles = [_normalize_role(s) for s in sorted_segments]

    for idx in kept_indices:
        # find segment whose [start, end) contains idx; bisect_right on starts
        # then back up one position
        pos = int(np.searchsorted(starts, int(idx), side="right")) - 1
        if pos < 0 or pos >= len(sorted_segments):
            continue
        if int(idx) >= int(ends[pos]):
            continue
        col = role_index.get(roles[pos], fallback)
        if col is None:
            continue
        counts[col] += 1.0
    return counts


def _normalize_role(segment: dict) -> str:
    role = str(segment.get("role") or "other")
    has_tool_calls = bool(segment.get("has_tool_calls"))
    if role == "assistant" and has_tool_calls:
        return "assistant_call"
    if role == "assistant":
        return "assistant_message"
    if role in {"tool", "tool_result"}:
        return "tool_result"
    return role


# ---------------------------------------------------------------------------
# Attention aggregation
# ---------------------------------------------------------------------------


def aggregate_attention_role_per_layer(
    records: Sequence[IterationRecord],
    role_labels: Sequence[str] | None = None,
) -> tuple[list[str], dict[int, np.ndarray]]:
    """Per-layer 1D role distribution averaged across all records of a run."""
    labels = list(role_labels or collect_role_labels(records))
    dataset = load_attention_distributions(records, role_labels=labels, phase="all")
    layers, matrix, _counts = average_layer_matrix(dataset, equal_iter_weight=True)
    return labels, {int(layer): matrix[i] for i, layer in enumerate(layers)}


def aggregate_heavy_hitters_per_layer(
    records: Sequence[IterationRecord],
) -> dict[int, set[int]]:
    """Per-layer union of `heavy_indices` across all records.

    Reads attention.npz directly because `load_attention_distributions`
    aggregates over segments and discards raw key positions.
    """
    per_layer: dict[int, set[int]] = defaultdict(set)
    for record in records:
        path = record.iter_dir / "attention.npz"
        if not path.is_file():
            continue
        with np.load(path) as attn:
            if "heavy_indices" not in attn.files or "record_layer" not in attn.files:
                continue
            layers = attn["record_layer"].astype(np.int64)
            heavy = attn["heavy_indices"].astype(np.int64)
            if heavy.ndim == 1:
                # one heavy slot per record → broadcast to (R, 1)
                heavy = heavy[:, None]
            if int(heavy.shape[0]) != int(layers.shape[0]):
                raise ValueError(
                    f"{path}: heavy_indices rows {heavy.shape[0]} "
                    f"!= record_layer rows {layers.shape[0]}"
                )
            for row_idx in range(heavy.shape[0]):
                layer = int(layers[row_idx])
                for value in heavy[row_idx]:
                    v = int(value)
                    if v < 0:
                        continue
                    per_layer[layer].add(v)
    return dict(per_layer)


def aggregate_sink_recent_share_per_layer(
    records: Sequence[IterationRecord],
    sink: int,
    recent: int,
) -> dict[int, dict[str, float]]:
    """Per-layer fraction of topk-weight mass in sink/recent/middle key bands.

    Uses CSR top-k fields from attention.npz. Each record-row contributes
    weight in proportion to its top-k mass; sums are normalized at the end
    so the three shares add to 1.
    """
    per_layer_sums: dict[int, dict[str, float]] = defaultdict(
        lambda: {"sink": 0.0, "recent": 0.0, "middle": 0.0}
    )
    for record in records:
        path = record.iter_dir / "attention.npz"
        if not path.is_file():
            continue
        with np.load(path) as attn:
            needed = {"record_layer", "query_row_offsets", "query_positions"}
            if not needed.issubset(set(attn.files)):
                continue
            csr_fields = {
                "topk_csr_offsets",
                "topk_csr_indices",
                "topk_csr_weights",
            }
            has_csr_topk = csr_fields.issubset(set(attn.files))
            if not has_csr_topk:
                if any(name.startswith("topk_csr_") for name in attn.files):
                    raise ValueError(f"{path}: incomplete attention top-k schema")
                continue
            layers = attn["record_layer"].astype(np.int64)
            try:
                topk_idx, topk_w = decode_attention_topk(attn)
            except (KeyError, ValueError) as exc:
                raise ValueError(f"{path}: invalid attention top-k schema: {exc}") from exc
            topk_idx = topk_idx.astype(np.int64, copy=False)
            topk_w = topk_w.astype(np.float64, copy=False)
            offsets = attn["query_row_offsets"].astype(np.int64)
            query_positions = attn["query_positions"].astype(np.int64)

            for rec_idx in range(int(layers.shape[0])):
                layer = int(layers[rec_idx])
                start = int(offsets[rec_idx])
                end = int(offsets[rec_idx + 1])
                if end <= start:
                    continue
                # Each query row's effective key_len is its query_position + 1
                # under causal attention. Use the max across rows in this record
                # as a stable per-record key_len bound (since these query rows
                # share a record_phase/decode_step).
                row_positions = query_positions[start:end]
                if row_positions.size == 0:
                    continue
                key_len = int(row_positions.max()) + 1
                if key_len <= 0:
                    continue
                rec_idx_topk = topk_idx[start:end]
                rec_w_topk = topk_w[start:end]
                sink_mask = rec_idx_topk < sink
                recent_mask = rec_idx_topk >= max(0, key_len - recent)
                middle_mask = (~sink_mask) & (~recent_mask)
                # If sink and recent overlap (very small key_len), prefer sink.
                recent_mask = recent_mask & (~sink_mask)
                middle_mask = (~sink_mask) & (~recent_mask)
                per_layer_sums[layer]["sink"] += float(rec_w_topk[sink_mask].sum())
                per_layer_sums[layer]["recent"] += float(rec_w_topk[recent_mask].sum())
                per_layer_sums[layer]["middle"] += float(rec_w_topk[middle_mask].sum())

    out: dict[int, dict[str, float]] = {}
    for layer, sums in per_layer_sums.items():
        total = sums["sink"] + sums["recent"] + sums["middle"]
        if total <= 0:
            out[layer] = {"sink_share": 0.0, "recent_share": 0.0, "middle_share": 0.0}
            continue
        out[layer] = {
            "sink_share": sums["sink"] / total,
            "recent_share": sums["recent"] / total,
            "middle_share": sums["middle"] / total,
        }
    return out


# ---------------------------------------------------------------------------
# Pairwise comparisons
# ---------------------------------------------------------------------------


def compute_attention_js_per_layer(
    per_run_layer_dist: dict[str, dict[int, np.ndarray]],
    baseline_label: str,
) -> list[dict]:
    """Per-layer JS divergence between `baseline_label` and each other run."""
    if baseline_label not in per_run_layer_dist:
        raise KeyError(f"baseline label {baseline_label!r} not present in inputs")
    baseline = per_run_layer_dist[baseline_label]
    out: list[dict] = []
    for run, layer_dist in per_run_layer_dist.items():
        if run == baseline_label:
            continue
        common_layers = sorted(set(baseline) & set(layer_dist))
        for layer in common_layers:
            out.append(
                {
                    "layer": layer,
                    "run_pair": f"{run}_vs_{baseline_label}",
                    "js": js_divergence(baseline[layer], layer_dist[layer]),
                }
            )
    return out


def compute_heavy_jaccard_per_layer(
    per_run_heavy: dict[str, dict[int, set[int]]],
    baseline_label: str,
) -> list[dict]:
    """Per-layer Jaccard between `baseline_label` heavy set and each other."""
    if baseline_label not in per_run_heavy:
        raise KeyError(f"baseline label {baseline_label!r} not present in inputs")
    baseline = per_run_heavy[baseline_label]
    out: list[dict] = []
    for run, layer_set in per_run_heavy.items():
        if run == baseline_label:
            continue
        common_layers = sorted(set(baseline) & set(layer_set))
        for layer in common_layers:
            out.append(
                {
                    "layer": layer,
                    "run_pair": f"{run}_vs_{baseline_label}",
                    "jaccard": jaccard(baseline[layer], layer_set[layer]),
                    "baseline_size": len(baseline[layer]),
                    "variant_size": len(layer_set[layer]),
                }
            )
    return out


# ---------------------------------------------------------------------------
# Convenience: pre-load segments.json by iter_dir
# ---------------------------------------------------------------------------


def load_segments_by_iter_dir(records: Iterable[IterationRecord]) -> dict[str, dict]:
    """Pre-parse segments.json for every record. Cheap (~20 KB each)."""
    out: dict[str, dict] = {}
    for record in records:
        path = record.iter_dir / "segments.json"
        if not path.is_file():
            continue
        out[str(record.iter_dir)] = json.loads(path.read_text(encoding="utf-8"))
    return out
