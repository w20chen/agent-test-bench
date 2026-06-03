"""MoE phase denominator audits for routing recordings."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from recording_loader import IterationRecord, derive_moe_record_phases


PHASES = ("prefill", "decode", "mixed", "unknown")


def compute_moe_phase_denominator_audit(
    records: Sequence[IterationRecord],
) -> dict[str, Any]:
    """Compute routing-record, token-row, assignment, and load denominators."""
    totals = _empty_phase_totals()
    input_tokens = 0
    output_tokens = 0
    top_k_values: set[int] = set()

    for record in records:
        input_tokens += int(record.input_tokens or 0)
        output_tokens += int(record.output_tokens or 0)
        with np.load(record.iter_dir / "routing.npz") as routing:
            expert_load = routing["expert_load"].astype(np.float64)
            record_phases = derive_moe_record_phases(
                record,
                routing,
                expert_load=expert_load,
            )
            offsets = routing["token_row_offsets"].astype(np.int64)
            top_k = _scalar_int(routing["top_k_experts"])
            top_k_values.add(top_k)
            _accumulate_phase_denominators(
                totals,
                record_phases=record_phases,
                token_row_offsets=offsets,
                expert_load=expert_load,
                top_k=top_k,
            )

    return {
        "n_records": len(records),
        "input_tokens_total": input_tokens,
        "output_tokens_total": output_tokens,
        "top_k_experts_values": sorted(top_k_values),
        "phases": totals,
        "ratios_decode_over_prefill": _decode_prefill_ratios(totals),
        "derivation": {
            "routing_records": "number of saved gate-forward records after layer/step expansion",
            "token_rows": "sum(diff(token_row_offsets)) for routing records in that phase",
            "topk_assignments": "token_rows * top_k_experts",
            "expert_load_sum": "sum(expert_load), i.e. top-k softmax mass accumulated by segment/expert",
        },
    }


def _accumulate_phase_denominators(
    totals: dict[str, dict[str, float]],
    *,
    record_phases: np.ndarray,
    token_row_offsets: np.ndarray,
    expert_load: np.ndarray,
    top_k: int,
) -> None:
    if expert_load.ndim != 3:
        raise ValueError(f"expected expert_load rank 3, got {expert_load.shape}")
    if int(token_row_offsets.shape[0]) != int(expert_load.shape[0]) + 1:
        raise ValueError(
            "token_row_offsets length must equal expert_load records + 1: "
            f"{token_row_offsets.shape[0]} vs {expert_load.shape[0]}"
        )
    if int(record_phases.shape[0]) != int(expert_load.shape[0]):
        raise ValueError(
            "record_phases length must equal expert_load records: "
            f"{record_phases.shape[0]} vs {expert_load.shape[0]}"
        )

    token_rows = np.diff(token_row_offsets).astype(np.float64)
    for phase in PHASES:
        mask = record_phases.astype(str) == phase
        if not bool(mask.any()):
            continue
        phase_token_rows = float(token_rows[mask].sum())
        totals[phase]["routing_records"] += float(mask.sum())
        totals[phase]["token_rows"] += phase_token_rows
        totals[phase]["topk_assignments"] += phase_token_rows * float(top_k)
        totals[phase]["expert_load_sum"] += float(expert_load[mask].sum())


def _empty_phase_totals() -> dict[str, dict[str, float]]:
    return {
        phase: {
            "routing_records": 0.0,
            "token_rows": 0.0,
            "topk_assignments": 0.0,
            "expert_load_sum": 0.0,
        }
        for phase in PHASES
    }


def _decode_prefill_ratios(
    totals: dict[str, dict[str, float]],
) -> dict[str, float | None]:
    return {
        key: (
            totals["decode"][key] / totals["prefill"][key]
            if totals["prefill"][key] > 0
            else None
        )
        for key in totals["prefill"]
    }


def _scalar_int(value: Any) -> int:
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"expected scalar integer value, got shape {arr.shape}")
    return int(arr.reshape(-1)[0])
