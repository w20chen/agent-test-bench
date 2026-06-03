"""Follow-up metrics for task-change, residual, and expert-cache analyses."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np

try:
    from metrics import js_divergence, normalized_distribution, pairwise_js
    from recording_loader import (
        LayerDistributionSet,
        decode_attention_topk,
        derive_moe_record_phases,
        segment_role_indices_for_record,
    )
except ModuleNotFoundError:  # pragma: no cover - used by local pytest package imports.
    from scripts.recoding_figures.metrics import (
        js_divergence,
        normalized_distribution,
        pairwise_js,
    )
    from scripts.recoding_figures.recording_loader import (
        LayerDistributionSet,
        decode_attention_topk,
        derive_moe_record_phases,
        segment_role_indices_for_record,
    )


DEFAULT_ROLE_GROUPS: dict[str, tuple[str, ...]] = {
    "system": ("system",),
    "tool": ("assistant_call", "tool_result", "tool"),
    "recent_gen": ("generation", "recent_gen"),
}

ROUTED_TOKEN_ROLE_GROUPS: dict[str, tuple[str, ...]] = {
    "system": ("system",),
    "tool": ("assistant_call", "tool_result", "tool"),
    "generation": ("generation",),
}

DEFAULT_DISTANCE_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("self", 0, 0),
    ("d1_4", 1, 4),
    ("d5_16", 5, 16),
    ("d17_64", 17, 64),
    ("d65_256", 65, 256),
    ("d257_1024", 257, 1024),
    ("d1025_plus", 1025, None),
)

DEFAULT_DECODE_STEP_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("step0", 0, 0),
    ("step1_4", 1, 4),
    ("step5_16", 5, 16),
    ("step17_64", 17, 64),
    ("step65_plus", 65, None),
)

DEFAULT_RANK_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("rank1", 0, 0),
    ("rank2_4", 1, 3),
    ("rank5_16", 4, 15),
    ("rank17_plus", 16, None),
)

SEGMENT_RECENCY_LABELS: tuple[str, ...] = (
    "system",
    "latest_user",
    "earlier_user",
    "latest_tool_result",
    "earlier_tool_result",
    "latest_assistant_call",
    "earlier_assistant_call",
    "latest_assistant_message",
    "earlier_assistant_message",
    "gen_prompt",
    "generation",
    "other",
)


def load_attention_distance_bucket_distributions(
    records: Sequence[Any],
    *,
    phase: str = "decode",
    distance_buckets: Sequence[tuple[str, int, int | None]] = DEFAULT_DISTANCE_BUCKETS,
) -> LayerDistributionSet:
    """Load top-k attention mass distributions over query-key distance buckets."""
    labels = [name for name, _low, _high in distance_buckets]
    per_layer: dict[int, list[np.ndarray | None]] = {}
    per_layer_counts: dict[int, list[float]] = {}

    for record_index, record in enumerate(records):
        with np.load(record.iter_dir / "attention.npz") as attention:
            record_layers = attention["record_layer"].astype(np.int64)
            record_phases = attention["record_phase"].astype(str)
            offsets = attention["query_row_offsets"].astype(np.int64)
            query_positions = attention["query_positions"].astype(np.int64)
            topk_indices, topk_weights = decode_attention_topk(attention)
            topk_indices = topk_indices.astype(np.int64, copy=False)
            topk_weights = topk_weights.astype(np.float64, copy=False)

            layer_sums: dict[int, np.ndarray] = {}
            layer_counts: dict[int, int] = {}
            for row_idx, layer in enumerate(record_layers):
                if phase != "all" and str(record_phases[row_idx]) != phase:
                    continue
                start = int(offsets[row_idx])
                end = int(offsets[row_idx + 1])
                if end <= start:
                    continue
                positions = query_positions[start:end]
                key_indices = topk_indices[start:end]
                weights = topk_weights[start:end]
                if positions.shape[0] != key_indices.shape[0]:
                    raise ValueError(f"{record.iter_dir}: query/topk row mismatch")
                distances = positions[:, None] - key_indices
                bucket_values = np.zeros(len(labels), dtype=np.float64)
                finite_weights = np.isfinite(weights) & (weights > 0) & (distances >= 0)
                for bucket_idx, (_name, low, high) in enumerate(distance_buckets):
                    if high is None:
                        mask = (distances >= low) & finite_weights
                    else:
                        mask = (distances >= low) & (distances <= high) & finite_weights
                    bucket_values[bucket_idx] = float(weights[mask].sum())
                layer_int = int(layer)
                layer_sums.setdefault(layer_int, np.zeros(len(labels), dtype=np.float64))
                layer_sums[layer_int] += bucket_values
                layer_counts[layer_int] = layer_counts.get(layer_int, 0) + int(end - start)

        for layer, values in layer_sums.items():
            _ensure_record_slots(per_layer, layer, record_index)
            _ensure_count_slots(per_layer_counts, layer, record_index)
            distribution, effective_count = _distribution_or_zero(
                values,
                float(layer_counts[layer]),
            )
            per_layer[layer][record_index] = distribution
            per_layer_counts[layer][record_index] = effective_count

    return LayerDistributionSet(
        modality=f"attention_distance_{phase}",
        records=list(records),
        layers=sorted(per_layer),
        axis_labels=labels,
        distributions=_finalize_distribution_slots(per_layer, len(records), len(labels)),
        observation_counts=_finalize_count_slots(per_layer_counts, len(records)),
    )


def load_attention_decode_step_distributions(
    records: Sequence[Any],
    *,
    role_labels: Sequence[str],
    phase: str = "decode",
    step_buckets: Sequence[tuple[str, int, int | None]] = DEFAULT_DECODE_STEP_BUCKETS,
) -> LayerDistributionSet:
    """Load sampled query-row distributions over generated-token step buckets."""
    labels = [name for name, _low, _high in step_buckets]
    per_layer: dict[int, list[np.ndarray | None]] = {}
    per_layer_counts: dict[int, list[float]] = {}

    for record_index, record in enumerate(records):
        payload = json.loads((record.iter_dir / "segments.json").read_text(encoding="utf-8"))
        token_segment_id = [int(item) for item in payload.get("token_segment_id", [])]
        if not token_segment_id:
            raise ValueError(f"{record.iter_dir}: segments.json lacks token_segment_id")
        segment_role_cols = segment_role_indices_for_record(record, role_labels)
        decode_steps = _token_decode_steps(
            payload.get("segments", []),
            token_segment_id,
            segment_role_cols,
            role_labels,
        )

        with np.load(record.iter_dir / "attention.npz") as attention:
            record_layers = attention["record_layer"].astype(np.int64)
            record_phases = attention["record_phase"].astype(str)
            offsets = attention["query_row_offsets"].astype(np.int64)
            query_positions = attention["query_positions"].astype(np.int64)

            layer_sums: dict[int, np.ndarray] = {}
            layer_counts: dict[int, int] = {}
            for row_idx, layer in enumerate(record_layers):
                if phase != "all" and str(record_phases[row_idx]) != phase:
                    continue
                start = int(offsets[row_idx])
                end = int(offsets[row_idx + 1])
                if end <= start:
                    continue
                positions = query_positions[start:end]
                valid = (positions >= 0) & (positions < decode_steps.shape[0])
                steps = decode_steps[positions[valid]]
                steps = steps[steps >= 0]
                bucket_values = np.zeros(len(labels), dtype=np.float64)
                for bucket_idx, (_name, low, high) in enumerate(step_buckets):
                    if high is None:
                        mask = steps >= low
                    else:
                        mask = (steps >= low) & (steps <= high)
                    bucket_values[bucket_idx] = float(mask.sum())
                layer_int = int(layer)
                layer_sums.setdefault(layer_int, np.zeros(len(labels), dtype=np.float64))
                layer_sums[layer_int] += bucket_values
                layer_counts[layer_int] = layer_counts.get(layer_int, 0) + int(end - start)

        for layer, values in layer_sums.items():
            _ensure_record_slots(per_layer, layer, record_index)
            _ensure_count_slots(per_layer_counts, layer, record_index)
            distribution, effective_count = _distribution_or_zero(
                values,
                float(layer_counts[layer]),
            )
            per_layer[layer][record_index] = distribution
            per_layer_counts[layer][record_index] = effective_count

    return LayerDistributionSet(
        modality=f"attention_decode_step_{phase}",
        records=list(records),
        layers=sorted(per_layer),
        axis_labels=labels,
        distributions=_finalize_distribution_slots(per_layer, len(records), len(labels)),
        observation_counts=_finalize_count_slots(per_layer_counts, len(records)),
    )


def load_attention_segment_recency_distributions(
    records: Sequence[Any],
    *,
    role_labels: Sequence[str],
    phase: str = "decode",
) -> LayerDistributionSet:
    """Load attention mass over latest/earlier message-segment groups."""
    labels = list(SEGMENT_RECENCY_LABELS)
    per_layer: dict[int, list[np.ndarray | None]] = {}
    per_layer_counts: dict[int, list[float]] = {}

    for record_index, record in enumerate(records):
        payload = json.loads((record.iter_dir / "segments.json").read_text(encoding="utf-8"))
        segments = list(payload.get("segments", []))
        segment_role_cols = segment_role_indices_for_record(record, role_labels)
        if len(segments) != len(segment_role_cols):
            raise ValueError(
                f"{record.iter_dir}: segment count mismatch "
                f"{len(segments)} vs {len(segment_role_cols)}"
            )
        segment_groups = _segment_recency_group_ids(segment_role_cols, role_labels)

        with np.load(record.iter_dir / "attention.npz") as attention:
            record_layers = attention["record_layer"].astype(np.int64)
            record_phases = attention["record_phase"].astype(str)
            offsets = attention["query_row_offsets"].astype(np.int64)
            segment_mass = attention["segment_mass"].astype(np.float64)

            layer_sums: dict[int, np.ndarray] = {}
            layer_counts: dict[int, int] = {}
            for row_idx, layer in enumerate(record_layers):
                if phase != "all" and str(record_phases[row_idx]) != phase:
                    continue
                start = int(offsets[row_idx])
                end = int(offsets[row_idx + 1])
                if end <= start:
                    continue
                rows = segment_mass[start:end]
                if rows.shape[1] != len(segment_groups):
                    raise ValueError(
                        f"{record.iter_dir}: segment_mass width {rows.shape[1]} "
                        f"!= segment groups {len(segment_groups)}"
                    )
                if np.any(~np.isfinite(rows)):
                    raise ValueError(f"{record.iter_dir}: segment_mass contains non-finite values")
                values = np.bincount(
                    segment_groups,
                    weights=rows.sum(axis=0),
                    minlength=len(labels),
                ).astype(np.float64)
                layer_int = int(layer)
                layer_sums.setdefault(layer_int, np.zeros(len(labels), dtype=np.float64))
                layer_sums[layer_int] += values
                layer_counts[layer_int] = layer_counts.get(layer_int, 0) + int(end - start)

        for layer, values in layer_sums.items():
            _ensure_record_slots(per_layer, layer, record_index)
            _ensure_count_slots(per_layer_counts, layer, record_index)
            distribution, effective_count = _distribution_or_zero(
                values,
                float(layer_counts[layer]),
            )
            per_layer[layer][record_index] = distribution
            per_layer_counts[layer][record_index] = effective_count

    return LayerDistributionSet(
        modality=f"attention_segment_recency_{phase}",
        records=list(records),
        layers=sorted(per_layer),
        axis_labels=labels,
        distributions=_finalize_distribution_slots(per_layer, len(records), len(labels)),
        observation_counts=_finalize_count_slots(per_layer_counts, len(records)),
    )


def load_attention_rank_bucket_distributions(
    records: Sequence[Any],
    *,
    phase: str = "decode",
    rank_buckets: Sequence[tuple[str, int, int | None]] = DEFAULT_RANK_BUCKETS,
) -> LayerDistributionSet:
    """Load top-k attention mass distributions over saved key-rank buckets."""
    labels = [name for name, _low, _high in rank_buckets]
    per_layer: dict[int, list[np.ndarray | None]] = {}
    per_layer_counts: dict[int, list[float]] = {}

    for record_index, record in enumerate(records):
        with np.load(record.iter_dir / "attention.npz") as attention:
            record_layers = attention["record_layer"].astype(np.int64)
            record_phases = attention["record_phase"].astype(str)
            offsets = attention["query_row_offsets"].astype(np.int64)
            _, topk_weights = decode_attention_topk(attention)
            topk_weights = topk_weights.astype(np.float64, copy=False)

            layer_sums: dict[int, np.ndarray] = {}
            layer_counts: dict[int, int] = {}
            for row_idx, layer in enumerate(record_layers):
                if phase != "all" and str(record_phases[row_idx]) != phase:
                    continue
                start = int(offsets[row_idx])
                end = int(offsets[row_idx + 1])
                if end <= start:
                    continue
                weights = topk_weights[start:end]
                if np.any(~np.isfinite(weights)):
                    raise ValueError(f"{record.iter_dir}: topk_weights contains non-finite values")
                weights = np.where(weights > 0, weights, 0.0)
                bucket_values = np.zeros(len(labels), dtype=np.float64)
                width = int(weights.shape[1])
                for bucket_idx, (_name, low, high) in enumerate(rank_buckets):
                    if low >= width:
                        continue
                    end_col = width if high is None else min(width, high + 1)
                    if end_col <= low:
                        continue
                    bucket_values[bucket_idx] = float(weights[:, low:end_col].sum())
                layer_int = int(layer)
                layer_sums.setdefault(layer_int, np.zeros(len(labels), dtype=np.float64))
                layer_sums[layer_int] += bucket_values
                layer_counts[layer_int] = layer_counts.get(layer_int, 0) + int(end - start)

        for layer, values in layer_sums.items():
            _ensure_record_slots(per_layer, layer, record_index)
            _ensure_count_slots(per_layer_counts, layer, record_index)
            distribution, effective_count = _distribution_or_zero(
                values,
                float(layer_counts[layer]),
            )
            per_layer[layer][record_index] = distribution
            per_layer_counts[layer][record_index] = effective_count

    return LayerDistributionSet(
        modality=f"attention_rank_bucket_{phase}",
        records=list(records),
        layers=sorted(per_layer),
        axis_labels=labels,
        distributions=_finalize_distribution_slots(per_layer, len(records), len(labels)),
        observation_counts=_finalize_count_slots(per_layer_counts, len(records)),
    )


def load_attention_head_role_distributions(
    records: Sequence[Any],
    *,
    role_labels: Sequence[str],
    phase: str = "decode",
    head_count: int | None = None,
) -> LayerDistributionSet:
    """Load flattened head-by-role attention profiles for sampled query rows."""
    inferred_head_count = int(head_count or _infer_attention_head_count(records))
    if inferred_head_count <= 0:
        return LayerDistributionSet(
            modality=f"attention_head_role_{phase}",
            records=list(records),
            layers=[],
            axis_labels=[],
            distributions={},
            observation_counts={},
        )
    labels = [
        f"h{head}:{role}"
        for head in range(inferred_head_count)
        for role in role_labels
    ]
    role_width = len(role_labels)
    width = inferred_head_count * role_width
    per_layer: dict[int, list[np.ndarray | None]] = {}
    per_layer_counts: dict[int, list[float]] = {}

    for record_index, record in enumerate(records):
        segment_role_cols = segment_role_indices_for_record(record, role_labels)
        with np.load(record.iter_dir / "attention.npz") as attention:
            if "query_heads" not in attention.files:
                continue
            record_layers = attention["record_layer"].astype(np.int64)
            record_phases = attention["record_phase"].astype(str)
            offsets = attention["query_row_offsets"].astype(np.int64)
            segment_mass = attention["segment_mass"].astype(np.float64)
            query_heads = attention["query_heads"].astype(np.int64)
            if int(query_heads.shape[0]) != int(segment_mass.shape[0]):
                raise ValueError(
                    f"{record.iter_dir}: query_heads length {query_heads.shape[0]} "
                    f"does not match segment_mass rows {segment_mass.shape[0]}"
                )

            layer_sums: dict[int, np.ndarray] = {}
            layer_counts: dict[int, int] = {}
            for row_idx, layer in enumerate(record_layers):
                if phase != "all" and str(record_phases[row_idx]) != phase:
                    continue
                start = int(offsets[row_idx])
                end = int(offsets[row_idx + 1])
                if end <= start:
                    continue
                rows = segment_mass[start:end]
                heads = query_heads[start:end]
                if rows.shape[1] != len(segment_role_cols):
                    raise ValueError(
                        f"{record.iter_dir}: segment_mass width {rows.shape[1]} "
                        f"!= segment roles {len(segment_role_cols)}"
                    )
                values = np.zeros((inferred_head_count, role_width), dtype=np.float64)
                valid_heads = (heads >= 0) & (heads < inferred_head_count)
                for head in np.unique(heads[valid_heads]):
                    mask = heads == int(head)
                    segment_totals = rows[mask].sum(axis=0)
                    for segment_idx, role_col in enumerate(segment_role_cols):
                        values[int(head), int(role_col)] += float(segment_totals[segment_idx])
                layer_int = int(layer)
                layer_sums.setdefault(layer_int, np.zeros(width, dtype=np.float64))
                layer_sums[layer_int] += values.reshape(width)
                layer_counts[layer_int] = layer_counts.get(layer_int, 0) + int(end - start)

        for layer, values in layer_sums.items():
            _ensure_record_slots(per_layer, layer, record_index)
            _ensure_count_slots(per_layer_counts, layer, record_index)
            distribution, effective_count = _distribution_or_zero(
                values,
                float(layer_counts[layer]),
            )
            per_layer[layer][record_index] = distribution
            per_layer_counts[layer][record_index] = effective_count

    return LayerDistributionSet(
        modality=f"attention_head_role_{phase}",
        records=list(records),
        layers=sorted(per_layer),
        axis_labels=labels,
        distributions=_finalize_distribution_slots(per_layer, len(records), width),
        observation_counts=_finalize_count_slots(per_layer_counts, len(records)),
    )


def load_attention_query_role_distributions(
    records: Sequence[Any],
    *,
    role_labels: Sequence[str],
    phase: str = "decode",
) -> LayerDistributionSet:
    """Load query-token role distributions from sampled attention query rows."""
    labels = list(role_labels)
    per_layer: dict[int, list[np.ndarray | None]] = {}
    per_layer_counts: dict[int, list[float]] = {}

    for record_index, record in enumerate(records):
        segment_role_cols = segment_role_indices_for_record(record, labels)
        payload = json.loads((record.iter_dir / "segments.json").read_text(encoding="utf-8"))
        token_segment_id = [int(item) for item in payload.get("token_segment_id", [])]
        if not token_segment_id:
            raise ValueError(f"{record.iter_dir}: segments.json lacks token_segment_id")
        token_to_role_col = _token_to_role_column(token_segment_id, segment_role_cols)

        with np.load(record.iter_dir / "attention.npz") as attention:
            record_layers = attention["record_layer"].astype(np.int64)
            record_phases = attention["record_phase"].astype(str)
            offsets = attention["query_row_offsets"].astype(np.int64)
            query_positions = attention["query_positions"].astype(np.int64)

            layer_sums: dict[int, np.ndarray] = {}
            layer_counts: dict[int, int] = {}
            for row_idx, layer in enumerate(record_layers):
                if phase != "all" and str(record_phases[row_idx]) != phase:
                    continue
                start = int(offsets[row_idx])
                end = int(offsets[row_idx + 1])
                if end <= start:
                    continue
                positions = query_positions[start:end]
                valid = (positions >= 0) & (positions < len(token_to_role_col))
                role_cols = token_to_role_col[positions[valid]]
                values = np.bincount(
                    role_cols[role_cols >= 0],
                    minlength=len(labels),
                ).astype(np.float64)
                layer_int = int(layer)
                layer_sums.setdefault(layer_int, np.zeros(len(labels), dtype=np.float64))
                layer_sums[layer_int] += values
                layer_counts[layer_int] = layer_counts.get(layer_int, 0) + int(end - start)

        for layer, values in layer_sums.items():
            _ensure_record_slots(per_layer, layer, record_index)
            _ensure_count_slots(per_layer_counts, layer, record_index)
            distribution, effective_count = _distribution_or_zero(
                values,
                float(layer_counts[layer]),
            )
            per_layer[layer][record_index] = distribution
            per_layer_counts[layer][record_index] = effective_count

    return LayerDistributionSet(
        modality=f"query_role_{phase}",
        records=list(records),
        layers=sorted(per_layer),
        axis_labels=labels,
        distributions=_finalize_distribution_slots(per_layer, len(records), len(labels)),
        observation_counts=_finalize_count_slots(per_layer_counts, len(records)),
    )


def load_attention_context_group_distributions(
    records: Sequence[Any],
    *,
    role_labels: Sequence[str],
    phase: str = "decode",
    recent_token_window: int = 256,
) -> LayerDistributionSet:
    """Load top-k attention mass over system/tool/recent_gen/other context groups."""
    if recent_token_window <= 0:
        raise ValueError("recent_token_window must be positive")
    labels = ["system", "tool", "recent_gen", "other"]
    per_layer: dict[int, list[np.ndarray | None]] = {}
    per_layer_counts: dict[int, list[float]] = {}

    for record_index, record in enumerate(records):
        segment_role_cols = segment_role_indices_for_record(record, role_labels)
        payload = json.loads((record.iter_dir / "segments.json").read_text(encoding="utf-8"))
        token_segment_id = [int(item) for item in payload.get("token_segment_id", [])]
        if not token_segment_id:
            raise ValueError(f"{record.iter_dir}: segments.json lacks token_segment_id")
        token_to_context = _token_to_context_base(token_segment_id, segment_role_cols, role_labels)

        with np.load(record.iter_dir / "attention.npz") as attention:
            record_layers = attention["record_layer"].astype(np.int64)
            record_phases = attention["record_phase"].astype(str)
            offsets = attention["query_row_offsets"].astype(np.int64)
            query_positions = attention["query_positions"].astype(np.int64)
            topk_indices, topk_weights = decode_attention_topk(attention)
            topk_indices = topk_indices.astype(np.int64, copy=False)
            topk_weights = topk_weights.astype(np.float64, copy=False)

            layer_sums: dict[int, np.ndarray] = {}
            layer_counts: dict[int, int] = {}
            for row_idx, layer in enumerate(record_layers):
                if phase != "all" and str(record_phases[row_idx]) != phase:
                    continue
                start = int(offsets[row_idx])
                end = int(offsets[row_idx + 1])
                if end <= start:
                    continue
                values = np.zeros(len(labels), dtype=np.float64)
                positions = query_positions[start:end]
                key_indices = topk_indices[start:end]
                weights = topk_weights[start:end]
                distances = positions[:, None] - key_indices
                valid = (
                    np.isfinite(weights)
                    & (weights > 0)
                    & (distances >= 0)
                    & (key_indices >= 0)
                    & (key_indices < len(token_to_context))
                )
                if bool(valid.any()):
                    group_ids = np.full(key_indices.shape, -1, dtype=np.int64)
                    group_ids[valid] = token_to_context[key_indices[valid]]
                    recent_mask = (
                        valid
                        & (group_ids == 2)
                        & (distances <= recent_token_window)
                    )
                    far_generation_mask = valid & (group_ids == 2) & ~recent_mask
                    group_ids[far_generation_mask] = 3
                    flat_groups = group_ids[valid]
                    flat_weights = weights[valid]
                    values = np.bincount(
                        flat_groups[flat_groups >= 0],
                        weights=flat_weights[flat_groups >= 0],
                        minlength=len(labels),
                    ).astype(np.float64)
                layer_int = int(layer)
                layer_sums.setdefault(layer_int, np.zeros(len(labels), dtype=np.float64))
                layer_sums[layer_int] += values
                layer_counts[layer_int] = layer_counts.get(layer_int, 0) + int(end - start)

        for layer, values in layer_sums.items():
            _ensure_record_slots(per_layer, layer, record_index)
            _ensure_count_slots(per_layer_counts, layer, record_index)
            distribution, effective_count = _distribution_or_zero(
                values,
                float(layer_counts[layer]),
            )
            per_layer[layer][record_index] = distribution
            per_layer_counts[layer][record_index] = effective_count

    return LayerDistributionSet(
        modality=f"attention_context_group_{phase}",
        records=list(records),
        layers=sorted(per_layer),
        axis_labels=labels,
        distributions=_finalize_distribution_slots(per_layer, len(records), len(labels)),
        observation_counts=_finalize_count_slots(per_layer_counts, len(records)),
    )


def sliding_window_detection_summary(
    dataset: Any,
    *,
    windows: Sequence[int] = (2, 4, 8),
    tolerance: int = 2,
    mad_multiplier: float = 3.0,
) -> dict[str, Any]:
    """Evaluate an online previous-window JS detector on task-sorted records."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if mad_multiplier <= 0:
        raise ValueError("mad_multiplier must be positive")

    boundaries = _task_boundary_indices(dataset.records)
    rows: list[dict[str, Any]] = []
    for window_value in windows:
        window = int(window_value)
        if window <= 0:
            raise ValueError("window sizes must be positive")
        scores = sliding_window_js_scores(dataset, window=window)
        budget = len(boundaries)
        rank_alerts = _select_top_alerts(scores, budget=budget, nms_radius=tolerance)
        mad_alerts = _rolling_mad_alerts(
            scores,
            multiplier=mad_multiplier,
            min_history=max(8, window * 2),
            nms_radius=tolerance,
        )
        rank_eval = _evaluate_alerts(
            rank_alerts,
            boundaries=boundaries,
            tolerance=tolerance,
        )
        mad_eval = _evaluate_alerts(
            mad_alerts,
            boundaries=boundaries,
            tolerance=tolerance,
        )
        rows.append(
            {
                "window": float(window),
                "n_records": float(len(dataset.records)),
                "n_task_blocks": float(len(boundaries) + 1),
                "n_synthetic_task_transitions": float(len(boundaries)),
                "tolerance": float(tolerance),
                "rank_budget": float(budget),
                "rank_alerts": rank_alerts,
                "rank_hits": float(rank_eval["hits"]),
                "rank_detection_rate": rank_eval["detection_rate"],
                "rank_precision": rank_eval["precision"],
                "rolling_mad_multiplier": float(mad_multiplier),
                "rolling_mad_alerts": mad_alerts,
                "rolling_mad_hits": float(mad_eval["hits"]),
                "rolling_mad_detection_rate": mad_eval["detection_rate"],
                "rolling_mad_precision": mad_eval["precision"],
                "mean_boundary_score": _mean_boundary_score(
                    scores,
                    boundaries=boundaries,
                    tolerance=tolerance,
                ),
                "mean_nonboundary_score": _mean_nonboundary_score(
                    scores,
                    boundaries=boundaries,
                    tolerance=tolerance,
                ),
                "top_scores": _top_score_rows(scores, dataset.records, limit=12),
            }
        )

    return {
        "metric": "online_previous_window_js",
        "definition": (
            "At index i, score the JS divergence between the current "
            "per-layer distribution and the observation-weighted mean of the "
            "previous W records. Scores use no future records. Task labels are "
            "used only after scoring to evaluate synthetic task transitions in "
            "the task-sorted curated-14 order."
        ),
        "boundary_note": (
            "The curated-14 loader sorts records by task and call index, so "
            "task transitions here are synthetic task-order splices rather "
            "than chronological runtime task switches."
        ),
        "boundary_indices": [
            {
                "index": float(index),
                "prev_task": str(dataset.records[index - 1].task),
                "next_task": str(dataset.records[index].task),
            }
            for index in boundaries
        ],
        "rows": rows,
    }


def sliding_window_js_scores(dataset: Any, *, window: int) -> np.ndarray:
    """Return online previous-window JS scores for each record index."""
    if window <= 0:
        raise ValueError("window must be positive")
    n_records = len(dataset.records)
    scores = np.full(n_records, np.nan, dtype=np.float64)
    for idx in range(window, n_records):
        layer_scores: list[float] = []
        layer_weights: list[float] = []
        for layer in dataset.layers:
            matrix = dataset.distributions[layer]
            obs = dataset.observation_counts[layer].astype(np.float64)
            if idx >= matrix.shape[0] or obs[idx] <= 0:
                continue
            start = max(0, idx - window)
            prev_obs = obs[start:idx]
            valid = prev_obs > 0
            if not bool(valid.any()):
                continue
            prev_matrix = matrix[start:idx][valid]
            prev_weights = prev_obs[valid]
            prev_dist = _weighted_rows(prev_matrix, prev_weights)
            current_dist = matrix[idx]
            layer_scores.append(js_divergence(prev_dist, current_dist))
            layer_weights.append(float(obs[idx]))
        scores[idx] = _weighted_mean(layer_scores, layer_weights)
    return scores


def decode_residual_closure_summary(
    attention_dataset: Any,
    key_role_dataset: Any,
    distance_dataset: Any,
    query_role_dataset: Any,
    *,
    max_lag: int = 8,
) -> dict[str, Any]:
    """Stratify decode attention residuals by distance, call position, and query role."""
    if max_lag <= 0:
        raise ValueError("max_lag must be positive")
    records = attention_dataset.records
    layers = sorted(
        set(attention_dataset.layers)
        & set(key_role_dataset.layers)
        & set(distance_dataset.layers)
        & set(query_role_dataset.layers)
    )
    if not layers:
        raise ValueError("no common layers across residual closure datasets")

    position_fraction = _task_position_fraction(records)
    distance_layer_rows: list[dict[str, float]] = []
    query_layer_rows: list[dict[str, float]] = []
    position_buckets = {
        "early": _WeightedAccumulator(),
        "middle": _WeightedAccumulator(),
        "late": _WeightedAccumulator(),
    }
    lag_buckets = {lag: _WeightedAccumulator() for lag in range(1, max_lag + 1)}

    for layer in layers:
        attention_js = pairwise_js(attention_dataset.distributions[layer])
        key_role_js = pairwise_js(key_role_dataset.distributions[layer])
        distance_js = pairwise_js(distance_dataset.distributions[layer])
        query_role_js = pairwise_js(query_role_dataset.distributions[layer])
        residual = _linear_residual_matrix(attention_js, key_role_js)

        upper = np.triu_indices_from(residual, k=1)
        finite = (
            np.isfinite(residual[upper])
            & np.isfinite(distance_js[upper])
            & np.isfinite(query_role_js[upper])
        )
        abs_residual = np.abs(residual[upper][finite])
        distance_values = distance_js[upper][finite]
        query_values = query_role_js[upper][finite]
        distance_layer_rows.append(
            {
                "layer": float(layer),
                "corr_abs_residual_vs_distance_js": _pearson(abs_residual, distance_values),
                "mean_distance_js": _mean(distance_values),
                "mean_abs_residual": _mean(abs_residual),
            }
        )
        query_layer_rows.append(
            {
                "layer": float(layer),
                "corr_abs_residual_vs_query_role_js": _pearson(abs_residual, query_values),
                "mean_query_role_js": _mean(query_values),
                "mean_abs_residual": _mean(abs_residual),
            }
        )

        for left in range(len(records)):
            for right in range(left + 1, len(records)):
                value = float(abs(residual[left, right]))
                if not np.isfinite(value):
                    continue
                if str(records[left].task) != str(records[right].task):
                    continue
                midpoint = 0.5 * (position_fraction[left] + position_fraction[right])
                position_buckets[_position_bucket(midpoint)].add(value, 1.0)
                lag = right - left
                if lag in lag_buckets:
                    lag_buckets[lag].add(value, 1.0)

    query_audit = _query_role_audit(query_role_dataset)
    query_role_variation_available = int(query_audit["n_nonzero_roles"]) > 1
    return {
        "definition": (
            "Decode residual is the pairwise attention-role JS residual after "
            "a per-layer linear control for visible-key role JS. Distance decay "
            "uses top-k attention mass over query-key distance buckets. "
            "Sub-call position uses task-local normalized call position."
        ),
        "distance_decay": {
            "mean_corr_abs_residual_vs_distance_js": _mean(
                row["corr_abs_residual_vs_distance_js"] for row in distance_layer_rows
            ),
            "mean_distance_js": _mean(row["mean_distance_js"] for row in distance_layer_rows),
            "layer_rows": distance_layer_rows,
        },
        "sub_call_position": {
            "bucket_rows": [
                {
                    "bucket": bucket,
                    "mean_abs_residual": accumulator.mean(),
                    "n_layer_pairs": float(accumulator.count),
                }
                for bucket, accumulator in position_buckets.items()
            ],
            "lag_rows": [
                {
                    "lag": float(lag),
                    "mean_abs_residual": accumulator.mean(),
                    "n_layer_pairs": float(accumulator.count),
                }
                for lag, accumulator in lag_buckets.items()
            ],
        },
        "query_token_semantic_type": {
            "available_semantics": "segment_role_only",
            "lexical_token_semantics_available": False,
            "schema_limitation": (
                "Current attention artifacts store query positions and segment "
                "roles, but not token ids or token text. Query semantic type is "
                "therefore limited to segment-role categories."
            ),
            "role_share": query_audit["role_share"],
            "dominant_role": query_audit["dominant_role"],
            "dominant_role_share": query_audit["dominant_role_share"],
            "n_nonzero_roles": query_audit["n_nonzero_roles"],
            "query_role_variation_available": query_role_variation_available,
            "query_role_variation_can_explain_residual": query_role_variation_available,
            "mean_query_role_js": _mean(row["mean_query_role_js"] for row in query_layer_rows),
            "mean_corr_abs_residual_vs_query_role_js": _mean(
                row["corr_abs_residual_vs_query_role_js"] for row in query_layer_rows
            ),
            "layer_rows": query_layer_rows,
        },
    }


def record_scalar_feature_arrays(
    records: Sequence[Any],
    *,
    role_labels: Sequence[str],
) -> dict[str, np.ndarray]:
    """Return per-record scalar features that may explain residual magnitude."""
    features = {
        "task_position_fraction": _task_position_fraction(records),
        "call_idx": np.full(len(records), np.nan, dtype=np.float64),
        "input_tokens": np.full(len(records), np.nan, dtype=np.float64),
        "output_tokens": np.full(len(records), np.nan, dtype=np.float64),
        "total_tokens": np.full(len(records), np.nan, dtype=np.float64),
        "n_segments": np.full(len(records), np.nan, dtype=np.float64),
        "n_user_segments": np.full(len(records), np.nan, dtype=np.float64),
        "n_tool_result_segments": np.full(len(records), np.nan, dtype=np.float64),
        "n_assistant_call_segments": np.full(len(records), np.nan, dtype=np.float64),
        "n_assistant_message_segments": np.full(len(records), np.nan, dtype=np.float64),
        "n_generation_tokens": np.full(len(records), np.nan, dtype=np.float64),
        "n_prompt_tokens_from_segments": np.full(len(records), np.nan, dtype=np.float64),
    }

    for idx, record in enumerate(records):
        payload = json.loads((record.iter_dir / "segments.json").read_text(encoding="utf-8"))
        segments = list(payload.get("segments", []))
        segment_role_cols = segment_role_indices_for_record(record, role_labels)
        roles = [role_labels[int(col)] for col in segment_role_cols]
        lengths = np.asarray([_segment_token_length(segment) for segment in segments])
        if lengths.shape[0] != len(roles):
            raise ValueError(f"{record.iter_dir}: segment role/length mismatch")

        features["call_idx"][idx] = float(getattr(record, "call_idx", np.nan))
        features["input_tokens"][idx] = _record_numeric_value(
            record,
            payload,
            "input_tokens",
            fallback=float(lengths[[role != "generation" for role in roles]].sum()),
        )
        features["output_tokens"][idx] = _record_numeric_value(
            record,
            payload,
            "output_tokens",
            fallback=float(lengths[[role == "generation" for role in roles]].sum()),
        )
        features["total_tokens"][idx] = _record_numeric_value(
            record,
            payload,
            "total_tokens",
            fallback=float(lengths.sum()),
        )
        features["n_segments"][idx] = float(len(segments))
        features["n_user_segments"][idx] = float(sum(role == "user" for role in roles))
        features["n_tool_result_segments"][idx] = float(
            sum(role in {"tool", "tool_result"} for role in roles)
        )
        features["n_assistant_call_segments"][idx] = float(
            sum(role == "assistant_call" for role in roles)
        )
        features["n_assistant_message_segments"][idx] = float(
            sum(role == "assistant_message" for role in roles)
        )
        features["n_generation_tokens"][idx] = float(
            lengths[[role == "generation" for role in roles]].sum()
        )
        features["n_prompt_tokens_from_segments"][idx] = float(
            lengths[[role != "generation" for role in roles]].sum()
        )

    return features


def record_pair_feature_matrices(records: Sequence[Any]) -> dict[str, np.ndarray]:
    """Return analysis-only pairwise labels such as task mismatch."""
    tasks = np.asarray([str(record.task) for record in records], dtype=object)
    call_idx = np.asarray([float(getattr(record, "call_idx", np.nan)) for record in records])
    return {
        "oracle_task_mismatch": (tasks[:, None] != tasks[None, :]).astype(np.float64),
        "call_order_distance": np.abs(call_idx[:, None] - call_idx[None, :]),
    }


def residual_explanation_leaderboard(
    attention_dataset: Any,
    key_role_dataset: Any,
    *,
    distribution_features: dict[str, Any] | None = None,
    scalar_features: dict[str, np.ndarray] | None = None,
    pair_features: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Rank candidate explanations for the decode attention residual."""
    _validate_record_alignment(attention_dataset.records, key_role_dataset.records)
    records = attention_dataset.records
    base_layers = sorted(set(attention_dataset.layers) & set(key_role_dataset.layers))
    if not base_layers:
        raise ValueError("attention and key-role datasets have no common layers")

    base = {
        layer: _residual_base_for_layer(attention_dataset, key_role_dataset, layer)
        for layer in base_layers
    }
    rows: list[dict[str, Any]] = []

    for name, dataset in (distribution_features or {}).items():
        _validate_record_alignment(records, dataset.records)
        layer_rows: list[dict[str, Any]] = []
        for layer in sorted(set(base_layers) & set(dataset.layers)):
            feature_js = pairwise_js(dataset.distributions[layer])
            feature_obs = dataset.observation_counts[layer].astype(np.float64)
            layer_rows.append(
                _score_pair_matrix_for_layer(
                    base[layer],
                    feature_js,
                    feature_obs=feature_obs,
                    layer=layer,
                )
            )
        rows.append(
            _summarize_feature_layer_rows(
                feature=f"distribution:{name}",
                feature_type="distribution_js",
                layer_rows=layer_rows,
            )
        )

    for name, values in (scalar_features or {}).items():
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape != (len(records),):
            raise ValueError(f"scalar feature {name} shape {arr.shape} != {(len(records),)}")
        scalar_specs = {
            "abs_diff": np.abs(arr[:, None] - arr[None, :]),
            "pair_mean": 0.5 * (arr[:, None] + arr[None, :]),
        }
        feature_obs = np.isfinite(arr).astype(np.float64)
        for mode, matrix in scalar_specs.items():
            layer_rows = [
                _score_pair_matrix_for_layer(
                    base[layer],
                    matrix,
                    feature_obs=feature_obs,
                    layer=layer,
                )
                for layer in base_layers
            ]
            rows.append(
                _summarize_feature_layer_rows(
                    feature=f"scalar:{name}:{mode}",
                    feature_type="scalar_pair_matrix",
                    layer_rows=layer_rows,
                )
            )

    for name, matrix in (pair_features or {}).items():
        arr = np.asarray(matrix, dtype=np.float64)
        expected = (len(records), len(records))
        if arr.shape != expected:
            raise ValueError(f"pair feature {name} shape {arr.shape} != {expected}")
        layer_rows = [
            _score_pair_matrix_for_layer(base[layer], arr, layer=layer)
            for layer in base_layers
        ]
        rows.append(
            _summarize_feature_layer_rows(
                feature=f"pair:{name}",
                feature_type="provided_pair_matrix",
                layer_rows=layer_rows,
            )
        )

    rows = sorted(
        rows,
        key=lambda row: (
            -float(row["mean_abs_corr"])
            if np.isfinite(float(row["mean_abs_corr"]))
            else float("inf")
        ),
    )
    return {
        "definition": (
            "For each decode layer, compute pairwise attention-role JS and "
            "linearly control for visible-key role JS. Candidate features are "
            "scored by Pearson correlation between their pairwise distance or "
            "pairwise scalar matrix and the absolute residual."
        ),
        "n_records": float(len(records)),
        "n_layers": float(len(base_layers)),
        "residual_baseline": _residual_baseline_summary(base),
        "rows": rows,
    }


def distribution_component_leaderboard(
    attention_dataset: Any,
    key_role_dataset: Any,
    feature_dataset: Any,
    *,
    modes: Sequence[str] = ("abs_diff", "pair_mean"),
) -> dict[str, Any]:
    """Rank individual feature-distribution components against residual."""
    _validate_record_alignment(attention_dataset.records, key_role_dataset.records)
    _validate_record_alignment(attention_dataset.records, feature_dataset.records)
    base_layers = sorted(
        set(attention_dataset.layers)
        & set(key_role_dataset.layers)
        & set(feature_dataset.layers)
    )
    if not base_layers:
        raise ValueError("datasets have no common layers")
    base = {
        layer: _residual_base_for_layer(attention_dataset, key_role_dataset, layer)
        for layer in base_layers
    }

    rows: list[dict[str, Any]] = []
    for component_idx, label in enumerate(feature_dataset.axis_labels):
        for mode in modes:
            if mode not in {"abs_diff", "pair_mean"}:
                raise ValueError(f"unsupported component mode {mode!r}")
            layer_rows: list[dict[str, Any]] = []
            for layer in base_layers:
                values = feature_dataset.distributions[layer][:, component_idx].astype(
                    np.float64
                )
                if mode == "abs_diff":
                    matrix = np.abs(values[:, None] - values[None, :])
                else:
                    matrix = 0.5 * (values[:, None] + values[None, :])
                layer_rows.append(
                    _score_pair_matrix_for_layer(
                        base[layer],
                        matrix,
                        feature_obs=feature_dataset.observation_counts[layer],
                        layer=layer,
                    )
                )
            rows.append(
                _summarize_feature_layer_rows(
                    feature=f"component:{feature_dataset.modality}:{label}:{mode}",
                    feature_type="distribution_component_pair_matrix",
                    layer_rows=layer_rows,
                    include_layer_rows=False,
                )
            )

    rows = sorted(
        rows,
        key=lambda row: (
            -float(row["mean_abs_corr"])
            if np.isfinite(float(row["mean_abs_corr"]))
            else float("inf")
        ),
    )
    return {
        "definition": (
            "Component probe scores each single feature-distribution column by "
            "pairwise absolute difference and pairwise mean against the same "
            "absolute decode residual."
        ),
        "source_modality": feature_dataset.modality,
        "n_components": float(len(feature_dataset.axis_labels)),
        "n_layers": float(len(base_layers)),
        "rows": rows,
    }


def context_role_cache_summary(
    moe_dataset: Any,
    attention_dataset: Any,
    *,
    ks: Sequence[int] = (8, 16, 32, 64),
    role_groups: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """Compare layer-static and attention-context role-aware expert hotsets."""
    groups = _complete_role_groups(attention_dataset.axis_labels, role_groups)
    group_names = list(groups)
    group_weights = _attention_group_weights(attention_dataset, groups)
    static_by_layer = {
        layer: _weighted_rows(
            moe_dataset.distributions[layer],
            moe_dataset.observation_counts[layer],
        )
        for layer in moe_dataset.layers
    }
    role_by_layer = _role_conditioned_expert_distributions(
        moe_dataset,
        group_weights,
        group_names,
        fallback_by_layer=static_by_layer,
    )

    rows: list[dict[str, Any]] = []
    for k_value in ks:
        k = min(int(k_value), len(moe_dataset.axis_labels))
        if k <= 0:
            raise ValueError("k must be positive")
        rows.append(
            _evaluate_context_role_hotsets(
                moe_dataset,
                group_weights,
                group_names,
                static_by_layer,
                role_by_layer,
                k,
            )
        )

    return {
        "role_groups": {
            name: list(labels)
            for name, labels in groups.items()
        },
        "definition": (
            "Layer-static uses one top-k expert hotset per layer. "
            "Context-role uses one top-k hotset per layer and attention role "
            "group, where role-group expert distributions are fitted from "
            "decode MoE distributions weighted by same-layer decode attention "
            "mass to that role group."
        ),
        "capacity_note": (
            "Dominant-context coverage uses k experts for the selected role at "
            "each record/layer. If all role hotsets are kept resident at once, "
            "the union can exceed k; average union size is reported separately."
        ),
        "rows": rows,
    }


def routed_token_role_load_audit(
    records: Sequence[Any],
    role_labels: Sequence[str],
    *,
    phase: str = "decode",
    role_groups: dict[str, tuple[str, ...]] | None = ROUTED_TOKEN_ROLE_GROUPS,
) -> dict[str, Any]:
    """Audit which segment roles carry MoE routed-token load."""
    groups = _complete_role_groups(role_labels, role_groups)
    role_to_group = _role_to_group(groups)
    group_load = {name: 0.0 for name in groups}
    n_routing_records = 0
    n_iteration_records = 0

    for record in records:
        with np.load(record.iter_dir / "routing.npz") as routing:
            expert_load = routing["expert_load"].astype(np.float64)
            phases = derive_moe_record_phases(record, routing, expert_load=expert_load)
            segment_role_indices = segment_role_indices_for_record(record, role_labels)
            n_iteration_records += 1
            for idx, record_phase in enumerate(phases.astype(str)):
                if record_phase != phase:
                    continue
                n_routing_records += 1
                segment_mass = expert_load[idx].sum(axis=1)
                for segment_idx, value in enumerate(segment_mass):
                    if float(value) <= 0:
                        continue
                    role = role_labels[segment_role_indices[segment_idx]]
                    group = role_to_group.get(role, "other")
                    group_load[group] += float(value)

    total = sum(group_load.values())
    system_tool_load = group_load.get("system", 0.0) + group_load.get("tool", 0.0)
    system_tool_share = system_tool_load / total if total > 0 else float("nan")
    dominant_group = max(group_load, key=lambda key: group_load[key]) if group_load else None
    dominant_group_share = (
        group_load[dominant_group] / total
        if dominant_group is not None and total > 0
        else float("nan")
    )
    return {
        "phase": phase,
        "n_iteration_records": float(n_iteration_records),
        "n_routing_records": float(n_routing_records),
        "group_load": group_load,
        "group_share": {
            group: (value / total if total > 0 else float("nan"))
            for group, value in group_load.items()
        },
        "dominant_group": dominant_group,
        "dominant_group_share": dominant_group_share,
        "routed_token_system_tool_share": system_tool_share,
        "routed_token_system_tool_hotsets_supported": bool(system_tool_load > 0.0),
        "schema_note": (
            "This audits roles of tokens routed through MoE. It is separate "
            "from attention-context role conditioning over visible KV roles. "
            "Routing records do not contain query-key recency, so routed-token "
            "roles use generation rather than recent_gen."
        ),
    }


def alpha_blend_summary(
    dataset: Any,
    *,
    alphas: Sequence[float] = (0.0, 0.3, 0.6, 1.0),
    ks: Sequence[int] = (8, 16, 32, 64),
) -> dict[str, Any]:
    """Evaluate layer-static / previous-iteration hotset blend coverage."""
    static_by_layer = {
        layer: _weighted_rows(
            dataset.distributions[layer],
            dataset.observation_counts[layer],
        )
        for layer in dataset.layers
    }
    rows: list[dict[str, Any]] = []
    for k_value in ks:
        k = min(int(k_value), len(dataset.axis_labels))
        if k <= 0:
            raise ValueError("k must be positive")
        for alpha_value in alphas:
            alpha = float(alpha_value)
            if alpha < 0.0 or alpha > 1.0:
                raise ValueError("alpha values must be in [0, 1]")
            rows.append(_evaluate_alpha(dataset, static_by_layer, k=k, alpha=alpha))

    return {
        "definition": (
            "For each layer and adjacent iteration pair, build a blended "
            "expert score distribution: (1-alpha) * layer_static + alpha * "
            "previous_iteration_distribution. The top-k of that blend is "
            "evaluated against current expert-load mass."
        ),
        "rows": rows,
    }


def _evaluate_context_role_hotsets(
    moe_dataset: Any,
    group_weights: dict[int, np.ndarray],
    group_names: Sequence[str],
    static_by_layer: dict[int, np.ndarray],
    role_by_layer: dict[int, dict[str, np.ndarray]],
    k: int,
) -> dict[str, Any]:
    layer_static = _WeightedAccumulator()
    dominant_context = _WeightedAccumulator()
    mixture_context = _WeightedAccumulator()
    per_group = {name: _WeightedAccumulator() for name in group_names}
    union_sizes: list[float] = []

    for layer in moe_dataset.layers:
        matrix = moe_dataset.distributions[layer]
        obs = moe_dataset.observation_counts[layer].astype(np.float64)
        weights = group_weights[layer]
        static_hotset = _top_indices(static_by_layer[layer], k)
        role_hotsets = {
            group: _top_indices(role_by_layer[layer][group], k)
            for group in group_names
        }
        union_sizes.append(float(len(set().union(*(set(v) for v in role_hotsets.values())))))

        for idx in range(matrix.shape[0]):
            if obs[idx] <= 0:
                continue
            if float(weights[idx].sum()) <= 0.0:
                continue
            current = matrix[idx]
            weight = float(obs[idx])
            group_row = _normalize_or_zero(weights[idx])
            layer_static.add(float(current[static_hotset].sum()), weight)

            dominant_idx = int(np.argmax(group_row))
            dominant_group = group_names[dominant_idx]
            dominant_coverage = float(current[role_hotsets[dominant_group]].sum())
            dominant_context.add(dominant_coverage, weight)
            per_group[dominant_group].add(dominant_coverage, weight)

            mixture_coverage = 0.0
            for group_idx, group in enumerate(group_names):
                coverage = float(current[role_hotsets[group]].sum())
                mixture_coverage += float(group_row[group_idx]) * coverage
            mixture_context.add(mixture_coverage, weight)

    return {
        "k": float(k),
        "layer_static_coverage": layer_static.mean(),
        "dominant_context_role_coverage": dominant_context.mean(),
        "attention_mixture_role_coverage": mixture_context.mean(),
        "dominant_minus_layer_static": dominant_context.mean() - layer_static.mean(),
        "mixture_minus_layer_static": mixture_context.mean() - layer_static.mean(),
        "mean_role_hotset_union_size": _mean(union_sizes),
        "per_dominant_group": [
            {
                "group": group,
                "coverage": accumulator.mean(),
                "weight": accumulator.weight,
                "n_layer_records": float(accumulator.count),
            }
            for group, accumulator in per_group.items()
        ],
    }


def _evaluate_alpha(
    dataset: Any,
    static_by_layer: dict[int, np.ndarray],
    *,
    k: int,
    alpha: float,
) -> dict[str, Any]:
    overall = _WeightedAccumulator()
    same_task = _WeightedAccumulator()
    cross_task_splice = _WeightedAccumulator()

    records = dataset.records
    for layer in dataset.layers:
        matrix = dataset.distributions[layer]
        obs = dataset.observation_counts[layer].astype(np.float64)
        static_dist = static_by_layer[layer]
        for idx in range(1, len(records)):
            if obs[idx - 1] <= 0 or obs[idx] <= 0:
                continue
            blend = (1.0 - alpha) * static_dist + alpha * matrix[idx - 1]
            hotset = _top_indices(blend, k)
            coverage = float(matrix[idx, hotset].sum())
            weight = float(obs[idx])
            overall.add(coverage, weight)
            if records[idx - 1].task == records[idx].task:
                same_task.add(coverage, weight)
            else:
                cross_task_splice.add(coverage, weight)

    return {
        "k": float(k),
        "alpha": float(alpha),
        "overall_coverage": overall.mean(),
        "same_task_coverage": same_task.mean(),
        "synthetic_cross_task_splice_coverage": cross_task_splice.mean(),
        "n_adjacent_layer_transitions": float(overall.count),
        "n_same_task_layer_transitions": float(same_task.count),
        "n_cross_task_splice_layer_transitions": float(cross_task_splice.count),
    }


def _role_conditioned_expert_distributions(
    moe_dataset: Any,
    group_weights: dict[int, np.ndarray],
    group_names: Sequence[str],
    *,
    fallback_by_layer: dict[int, np.ndarray],
) -> dict[int, dict[str, np.ndarray]]:
    out: dict[int, dict[str, np.ndarray]] = {}
    for layer in moe_dataset.layers:
        matrix = moe_dataset.distributions[layer]
        obs = moe_dataset.observation_counts[layer].astype(np.float64)
        weights = group_weights[layer]
        out[layer] = {}
        for group_idx, group in enumerate(group_names):
            row_weights = obs * weights[:, group_idx]
            if float(row_weights.sum()) <= 0:
                out[layer][group] = fallback_by_layer[layer]
            else:
                out[layer][group] = _weighted_rows(matrix, row_weights)
    return out


def _attention_group_weights(
    attention_dataset: Any,
    groups: dict[str, tuple[str, ...]],
) -> dict[int, np.ndarray]:
    label_to_index = {label: idx for idx, label in enumerate(attention_dataset.axis_labels)}
    out: dict[int, np.ndarray] = {}
    for layer in attention_dataset.layers:
        matrix = attention_dataset.distributions[layer]
        values = np.zeros((matrix.shape[0], len(groups)), dtype=np.float64)
        for group_idx, labels in enumerate(groups.values()):
            cols = [label_to_index[label] for label in labels if label in label_to_index]
            if cols:
                values[:, group_idx] = matrix[:, cols].sum(axis=1)
        out[layer] = np.vstack([_normalize_or_zero(row) for row in values])
    return out


def _complete_role_groups(
    role_labels: Sequence[str],
    role_groups: dict[str, tuple[str, ...]] | None,
) -> dict[str, tuple[str, ...]]:
    groups = dict(role_groups or DEFAULT_ROLE_GROUPS)
    assigned = {role for labels in groups.values() for role in labels}
    other = tuple(role for role in role_labels if role not in assigned)
    groups["other"] = other
    return groups


def _role_to_group(groups: dict[str, tuple[str, ...]]) -> dict[str, str]:
    return {role: group for group, roles in groups.items() for role in roles}


def _token_to_role_column(
    token_segment_id: Sequence[int],
    segment_role_cols: Sequence[int],
) -> np.ndarray:
    values = np.full(len(token_segment_id), -1, dtype=np.int64)
    for token_idx, segment_idx in enumerate(token_segment_id):
        if 0 <= int(segment_idx) < len(segment_role_cols):
            values[token_idx] = int(segment_role_cols[int(segment_idx)])
    return values


def _token_to_context_base(
    token_segment_id: Sequence[int],
    segment_role_cols: Sequence[int],
    role_labels: Sequence[str],
) -> np.ndarray:
    """Map key token to base context group id.

    Group ids are aligned with `["system", "tool", "recent_gen", "other"]`.
    Generation tokens are initially assigned to `recent_gen`; caller moves them
    to `other` when query-key distance exceeds the fixed recency window.
    """
    values = np.full(len(token_segment_id), -1, dtype=np.int64)
    for token_idx, segment_idx in enumerate(token_segment_id):
        if not 0 <= int(segment_idx) < len(segment_role_cols):
            continue
        role = role_labels[int(segment_role_cols[int(segment_idx)])]
        if role == "system":
            values[token_idx] = 0
        elif role in {"assistant_call", "tool_result", "tool"}:
            values[token_idx] = 1
        elif role == "generation":
            values[token_idx] = 2
        else:
            values[token_idx] = 3
    return values


def _weighted_rows(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    weight_arr = np.asarray(weights, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected matrix, got {arr.shape}")
    if weight_arr.shape[0] != arr.shape[0]:
        raise ValueError(f"weight length {weight_arr.shape[0]} != rows {arr.shape[0]}")
    valid = np.isfinite(weight_arr) & (weight_arr > 0)
    if not bool(valid.any()):
        return np.zeros(arr.shape[1], dtype=np.float64)
    weighted = np.sum(arr[valid] * weight_arr[valid, None], axis=0)
    return normalized_distribution(weighted)


def _distribution_or_zero(values: np.ndarray, count: float) -> tuple[np.ndarray, float]:
    arr = np.asarray(values, dtype=np.float64)
    if np.any(arr < 0) or np.any(~np.isfinite(arr)):
        raise ValueError("distribution contains negative or non-finite values")
    total = float(arr.sum())
    if total <= 0.0:
        return np.zeros(arr.shape, dtype=np.float64), 0.0
    return arr / total, float(count)


def _normalize_or_zero(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if np.any(arr < 0) or np.any(~np.isfinite(arr)):
        raise ValueError("distribution contains negative or non-finite values")
    total = float(arr.sum())
    if total <= 0.0:
        return np.zeros(arr.shape, dtype=np.float64)
    return arr / total


def _linear_residual_matrix(target: np.ndarray, control: np.ndarray) -> np.ndarray:
    if target.shape != control.shape:
        raise ValueError(f"shape mismatch: {target.shape} vs {control.shape}")
    upper = np.triu_indices_from(target, k=1)
    x = control[upper]
    y = target[upper]
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 2 or float(np.std(x[finite])) == 0.0:
        prediction = np.full_like(target, float(np.nanmean(y[finite])) if finite.any() else 0.0)
    else:
        slope, intercept = np.polyfit(x[finite], y[finite], deg=1)
        prediction = slope * control + intercept
    return target - prediction


def _task_position_fraction(records: Sequence[Any]) -> np.ndarray:
    by_task: dict[str, list[int]] = {}
    for idx, record in enumerate(records):
        by_task.setdefault(str(record.task), []).append(idx)
    fractions = np.zeros(len(records), dtype=np.float64)
    for indices in by_task.values():
        denom = max(len(indices) - 1, 1)
        for position, record_idx in enumerate(indices):
            fractions[record_idx] = float(position / denom)
    return fractions


def _position_bucket(value: float) -> str:
    if value < 1.0 / 3.0:
        return "early"
    if value < 2.0 / 3.0:
        return "middle"
    return "late"


def _query_role_audit(query_role_dataset: Any) -> dict[str, Any]:
    totals = np.zeros(len(query_role_dataset.axis_labels), dtype=np.float64)
    total_weight = 0.0
    for layer in query_role_dataset.layers:
        matrix = query_role_dataset.distributions[layer]
        weights = query_role_dataset.observation_counts[layer].astype(np.float64)
        valid = weights > 0
        if not bool(valid.any()):
            continue
        totals += np.sum(matrix[valid] * weights[valid, None], axis=0)
        total_weight += float(weights[valid].sum())
    share = normalized_distribution(totals) if total_weight > 0 else totals
    dominant_idx = int(np.argmax(share)) if share.size else -1
    return {
        "role_share": {
            label: float(share[idx])
            for idx, label in enumerate(query_role_dataset.axis_labels)
        },
        "n_nonzero_roles": float(np.count_nonzero(share > 0.0)),
        "dominant_role": (
            str(query_role_dataset.axis_labels[dominant_idx])
            if dominant_idx >= 0
            else None
        ),
        "dominant_role_share": float(share[dominant_idx]) if dominant_idx >= 0 else float("nan"),
    }


def _token_decode_steps(
    segments: Sequence[dict],
    token_segment_id: Sequence[int],
    segment_role_cols: Sequence[int],
    role_labels: Sequence[str],
) -> np.ndarray:
    steps = np.full(len(token_segment_id), -1, dtype=np.int64)
    if len(segments) != len(segment_role_cols):
        raise ValueError(
            f"segment count mismatch {len(segments)} vs role cols {len(segment_role_cols)}"
        )
    token_segments = np.asarray(token_segment_id, dtype=np.int64)
    for segment_idx, segment in enumerate(segments):
        role = role_labels[int(segment_role_cols[segment_idx])]
        if role != "generation":
            continue
        start = int(segment.get("token_start", segment.get("start", 0)) or 0)
        end = int(segment.get("token_end", segment.get("end", start)) or start)
        low = max(0, start)
        high = min(len(steps), end)
        if high <= low:
            continue
        positions = np.arange(low, high, dtype=np.int64)
        valid = token_segments[positions] == int(segment_idx)
        steps[positions[valid]] = positions[valid] - int(start)
    return steps


def _segment_recency_group_ids(
    segment_role_cols: Sequence[int],
    role_labels: Sequence[str],
) -> np.ndarray:
    label_to_group = {label: idx for idx, label in enumerate(SEGMENT_RECENCY_LABELS)}
    roles = [role_labels[int(col)] for col in segment_role_cols]
    latest: dict[str, int] = {}
    for idx, role in enumerate(roles):
        if role in {
            "user",
            "tool",
            "tool_result",
            "assistant_call",
            "assistant_message",
        }:
            latest[_recency_role_key(role)] = idx

    groups = np.zeros(len(roles), dtype=np.int64)
    for idx, role in enumerate(roles):
        if role == "system":
            groups[idx] = label_to_group["system"]
        elif role == "user":
            groups[idx] = label_to_group[
                "latest_user" if latest.get("user") == idx else "earlier_user"
            ]
        elif role in {"tool", "tool_result"}:
            groups[idx] = label_to_group[
                "latest_tool_result"
                if latest.get("tool_result") == idx
                else "earlier_tool_result"
            ]
        elif role == "assistant_call":
            groups[idx] = label_to_group[
                "latest_assistant_call"
                if latest.get("assistant_call") == idx
                else "earlier_assistant_call"
            ]
        elif role == "assistant_message":
            groups[idx] = label_to_group[
                "latest_assistant_message"
                if latest.get("assistant_message") == idx
                else "earlier_assistant_message"
            ]
        elif role == "gen_prompt":
            groups[idx] = label_to_group["gen_prompt"]
        elif role == "generation":
            groups[idx] = label_to_group["generation"]
        else:
            groups[idx] = label_to_group["other"]
    return groups


def _recency_role_key(role: str) -> str:
    return "tool_result" if role in {"tool", "tool_result"} else role


def _infer_attention_head_count(records: Sequence[Any]) -> int:
    max_head = -1
    for record in records:
        with np.load(record.iter_dir / "attention.npz") as attention:
            if "query_heads" not in attention.files:
                continue
            heads = attention["query_heads"].astype(np.int64)
            finite_heads = heads[heads >= 0]
            if finite_heads.size:
                max_head = max(max_head, int(finite_heads.max()))
    return max_head + 1


def _segment_token_length(segment: dict) -> float:
    start = int(segment.get("token_start", segment.get("start", 0)) or 0)
    end = int(segment.get("token_end", segment.get("end", start)) or start)
    return float(max(0, end - start))


def _record_numeric_value(
    record: Any,
    payload: dict[str, Any],
    field: str,
    *,
    fallback: float,
) -> float:
    value = getattr(record, field, None)
    if value is None:
        value = payload.get(field)
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(fallback)
    return number if np.isfinite(number) else float(fallback)


def _validate_record_alignment(left: Sequence[Any], right: Sequence[Any]) -> None:
    if len(left) != len(right):
        raise ValueError(f"record count mismatch {len(left)} vs {len(right)}")
    for idx, (left_record, right_record) in enumerate(zip(left, right, strict=True)):
        left_key = (
            getattr(left_record, "task", None),
            getattr(left_record, "call_idx", None),
            str(getattr(left_record, "iter_dir", "")),
        )
        right_key = (
            getattr(right_record, "task", None),
            getattr(right_record, "call_idx", None),
            str(getattr(right_record, "iter_dir", "")),
        )
        if left_key != right_key:
            raise ValueError(f"record alignment mismatch at {idx}: {left_key} vs {right_key}")


def _residual_base_for_layer(
    attention_dataset: Any,
    key_role_dataset: Any,
    layer: int,
) -> dict[str, Any]:
    attention_js = pairwise_js(attention_dataset.distributions[layer])
    key_role_js = pairwise_js(key_role_dataset.distributions[layer])
    residual = _linear_residual_matrix(attention_js, key_role_js)
    left, right = np.triu_indices_from(residual, k=1)
    attention_obs = attention_dataset.observation_counts[layer].astype(np.float64)
    key_role_obs = key_role_dataset.observation_counts[layer].astype(np.float64)
    valid_obs = (
        (attention_obs[left] > 0)
        & (attention_obs[right] > 0)
        & (key_role_obs[left] > 0)
        & (key_role_obs[right] > 0)
    )
    values = np.abs(residual[left, right])
    valid = valid_obs & np.isfinite(values)
    return {
        "layer": int(layer),
        "left": left,
        "right": right,
        "abs_residual": values,
        "valid": valid,
        "mean_abs_residual": _mean(values[valid]),
        "n_valid_pairs": float(valid.sum()),
    }


def _score_pair_matrix_for_layer(
    base: dict[str, Any],
    feature_matrix: np.ndarray,
    *,
    layer: int,
    feature_obs: np.ndarray | None = None,
) -> dict[str, Any]:
    matrix = np.asarray(feature_matrix, dtype=np.float64)
    left = base["left"]
    right = base["right"]
    if matrix.shape[0] <= int(max(left.max(initial=0), right.max(initial=0))):
        raise ValueError(f"feature matrix too small for layer {layer}: {matrix.shape}")
    values = matrix[left, right]
    valid = np.asarray(base["valid"], dtype=bool) & np.isfinite(values)
    if feature_obs is not None:
        obs = np.asarray(feature_obs, dtype=np.float64)
        if obs.shape[0] <= int(max(left.max(initial=0), right.max(initial=0))):
            raise ValueError(f"feature obs too short for layer {layer}: {obs.shape}")
        valid &= (obs[left] > 0) & (obs[right] > 0)
    abs_residual = np.asarray(base["abs_residual"], dtype=np.float64)
    corr = _pearson(abs_residual[valid], values[valid])
    return {
        "layer": float(layer),
        "corr": corr,
        "abs_corr": abs(corr) if np.isfinite(corr) else float("nan"),
        "mean_feature_pair_value": _mean(values[valid]),
        "mean_abs_residual": _mean(abs_residual[valid]),
        "n_pairs": float(valid.sum()),
    }


def _summarize_feature_layer_rows(
    *,
    feature: str,
    feature_type: str,
    layer_rows: Sequence[dict[str, Any]],
    include_layer_rows: bool = True,
) -> dict[str, Any]:
    finite_rows = [row for row in layer_rows if np.isfinite(float(row["corr"]))]
    corrs = [float(row["corr"]) for row in finite_rows]
    abs_corrs = [abs(value) for value in corrs]
    top_layers = sorted(
        finite_rows,
        key=lambda row: abs(float(row["corr"])),
        reverse=True,
    )[:5]
    positive = sum(1 for value in corrs if value > 0)
    negative = sum(1 for value in corrs if value < 0)
    denom = len(corrs)
    summary = {
        "feature": feature,
        "feature_type": feature_type,
        "mean_corr": _mean(corrs),
        "median_corr": _median(corrs),
        "mean_abs_corr": _mean(abs_corrs),
        "median_abs_corr": _median(abs_corrs),
        "positive_layer_fraction": (
            float(positive / denom) if denom > 0 else float("nan")
        ),
        "negative_layer_fraction": (
            float(negative / denom) if denom > 0 else float("nan")
        ),
        "n_layers_scored": float(denom),
        "n_pairs_total": float(sum(float(row["n_pairs"]) for row in layer_rows)),
        "top_layers_by_abs_corr": [
            {
                "layer": row["layer"],
                "corr": row["corr"],
                "abs_corr": row["abs_corr"],
                "mean_feature_pair_value": row["mean_feature_pair_value"],
                "mean_abs_residual": row["mean_abs_residual"],
                "n_pairs": row["n_pairs"],
            }
            for row in top_layers
        ],
    }
    if include_layer_rows:
        summary["layer_rows"] = list(layer_rows)
    return summary


def _residual_baseline_summary(base_by_layer: dict[int, dict[str, Any]]) -> dict[str, Any]:
    layer_rows = [
        {
            "layer": float(layer),
            "mean_abs_residual": float(item["mean_abs_residual"]),
            "n_valid_pairs": float(item["n_valid_pairs"]),
        }
        for layer, item in sorted(base_by_layer.items())
    ]
    return {
        "mean_abs_residual": _mean(row["mean_abs_residual"] for row in layer_rows),
        "layer_rows": layer_rows,
    }


def _median(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    value_arr = np.asarray(values, dtype=np.float64)
    weight_arr = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(value_arr) & np.isfinite(weight_arr) & (weight_arr > 0)
    if not bool(finite.any()):
        return float("nan")
    return float(np.average(value_arr[finite], weights=weight_arr[finite]))


def _top_indices(values: np.ndarray, k: int) -> np.ndarray:
    arr = normalized_distribution(values)
    if k >= arr.size:
        return np.arange(arr.size)
    return np.argpartition(-arr, kth=k - 1)[:k]


def _task_boundary_indices(records: Sequence[Any]) -> list[int]:
    return [
        idx
        for idx in range(1, len(records))
        if str(records[idx - 1].task) != str(records[idx].task)
    ]


def _select_top_alerts(
    scores: np.ndarray,
    *,
    budget: int,
    nms_radius: int,
) -> list[int]:
    finite_indices = [idx for idx, value in enumerate(scores) if np.isfinite(value)]
    ranked = sorted(finite_indices, key=lambda idx: float(scores[idx]), reverse=True)
    selected: list[int] = []
    for idx in ranked:
        if any(abs(idx - chosen) <= nms_radius for chosen in selected):
            continue
        selected.append(int(idx))
        if len(selected) >= budget:
            break
    return sorted(selected)


def _rolling_mad_alerts(
    scores: np.ndarray,
    *,
    multiplier: float,
    min_history: int,
    nms_radius: int,
) -> list[int]:
    alerts: list[int] = []
    finite_history: list[float] = []
    for idx, value in enumerate(scores):
        if not np.isfinite(value):
            continue
        if len(finite_history) >= min_history:
            history = np.asarray(finite_history, dtype=np.float64)
            median = float(np.median(history))
            mad = float(np.median(np.abs(history - median)))
            scale = 1.4826 * mad
            if scale <= 1e-12:
                scale = float(np.std(history))
            threshold = median + multiplier * scale
            if value > threshold and not any(
                abs(idx - alert) <= nms_radius for alert in alerts
            ):
                alerts.append(int(idx))
        finite_history.append(float(value))
    return alerts


def _evaluate_alerts(
    alerts: Sequence[int],
    *,
    boundaries: Sequence[int],
    tolerance: int,
) -> dict[str, float]:
    hit_boundaries: set[int] = set()
    hit_alerts = 0
    for alert in alerts:
        matched = False
        for boundary in boundaries:
            if boundary <= alert <= boundary + tolerance:
                hit_boundaries.add(int(boundary))
                matched = True
        if matched:
            hit_alerts += 1
    return {
        "hits": float(len(hit_boundaries)),
        "detection_rate": (
            float(len(hit_boundaries) / len(boundaries)) if boundaries else float("nan")
        ),
        "precision": float(hit_alerts / len(alerts)) if alerts else float("nan"),
    }


def _mean_boundary_score(
    scores: np.ndarray,
    *,
    boundaries: Sequence[int],
    tolerance: int,
) -> float:
    values: list[float] = []
    for boundary in boundaries:
        for idx in range(boundary, min(scores.shape[0], boundary + tolerance + 1)):
            if np.isfinite(scores[idx]):
                values.append(float(scores[idx]))
                break
    return _mean(values)


def _mean_nonboundary_score(
    scores: np.ndarray,
    *,
    boundaries: Sequence[int],
    tolerance: int,
) -> float:
    boundary_window = {
        idx
        for boundary in boundaries
        for idx in range(boundary, min(scores.shape[0], boundary + tolerance + 1))
    }
    values = [
        float(value)
        for idx, value in enumerate(scores)
        if np.isfinite(value) and idx not in boundary_window
    ]
    return _mean(values)


def _top_score_rows(scores: np.ndarray, records: Sequence[Any], *, limit: int) -> list[dict[str, Any]]:
    indices = [
        idx
        for idx, value in sorted(
            enumerate(scores),
            key=lambda item: float(item[1]) if np.isfinite(item[1]) else -np.inf,
            reverse=True,
        )
        if np.isfinite(value)
    ][:limit]
    return [
        {
            "index": float(idx),
            "score": float(scores[idx]),
            "prev_task": str(records[idx - 1].task) if idx > 0 else None,
            "task": str(records[idx].task),
            "call_idx": float(records[idx].call_idx),
        }
        for idx in indices
    ]


def _mean(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    left_arr = np.asarray(list(left), dtype=np.float64)
    right_arr = np.asarray(list(right), dtype=np.float64)
    if left_arr.shape != right_arr.shape:
        raise ValueError(f"shape mismatch: {left_arr.shape} vs {right_arr.shape}")
    finite = np.isfinite(left_arr) & np.isfinite(right_arr)
    if int(finite.sum()) < 2:
        return float("nan")
    x = left_arr[finite]
    y = right_arr[finite]
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _ensure_record_slots(
    mapping: dict[int, list[np.ndarray | None]],
    layer: int,
    record_index: int,
) -> None:
    slots = mapping.setdefault(layer, [])
    while len(slots) <= record_index:
        slots.append(None)


def _ensure_count_slots(
    mapping: dict[int, list[float]],
    layer: int,
    record_index: int,
) -> None:
    slots = mapping.setdefault(layer, [])
    while len(slots) <= record_index:
        slots.append(0.0)


def _finalize_distribution_slots(
    mapping: dict[int, list[np.ndarray | None]],
    n_records: int,
    width: int,
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for layer, slots in mapping.items():
        rows: list[np.ndarray] = []
        for idx in range(n_records):
            value = slots[idx] if idx < len(slots) else None
            rows.append(
                np.zeros(width, dtype=np.float64)
                if value is None
                else np.asarray(value, dtype=np.float64)
            )
        out[layer] = np.vstack(rows)
    return out


def _finalize_count_slots(
    mapping: dict[int, list[float]],
    n_records: int,
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for layer, slots in mapping.items():
        out[layer] = np.asarray(
            [float(slots[idx]) if idx < len(slots) else 0.0 for idx in range(n_records)],
            dtype=np.float64,
        )
    return out


class _WeightedAccumulator:
    def __init__(self) -> None:
        self.total = 0.0
        self.weight = 0.0
        self.count = 0

    def add(self, value: float, weight: float) -> None:
        if not np.isfinite(value) or not np.isfinite(weight) or weight <= 0:
            return
        self.total += float(value) * float(weight)
        self.weight += float(weight)
        self.count += 1

    def mean(self) -> float:
        if self.weight <= 0:
            return float("nan")
        return self.total / self.weight


def json_default(value: Any) -> Any:
    """JSON fallback for NumPy scalar values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def dumps_json(payload: Any) -> str:
    """Serialize payload with stable indentation."""
    return json.dumps(payload, indent=2, default=json_default) + "\n"
