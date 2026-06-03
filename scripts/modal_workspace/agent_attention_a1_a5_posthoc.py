"""Modal post-hoc A1-A5 statistics for agent attention/MoE recordings.

This script is intentionally offline-only: it reads existing curated-14
recording artifacts from the Modal Volume and never reruns inference or touches
benchmark/evaluation code.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import modal


APP_NAME = "asb-agent-attention-a1-a5-revised"
VOLUME_NAME = "asb-terminal-recordings"
VOLUME_ROOT = Path("/data")
EXTRACT_DIR = VOLUME_ROOT / "extracted" / "curated14"
OUTPUT_DIR = VOLUME_ROOT / "outputs" / "agent_attention_a1_a5_revised_20260510"
PHASES = ("all", "prefill", "decode")

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
    .apt_install("zstd")
    .pip_install("matplotlib", "numpy")
    .add_local_dir(RECODING_FIGURES, remote_path="/opt/recoding_figures", copy=True)
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
app = modal.App(APP_NAME)


@app.function(
    image=image,
    volumes={VOLUME_ROOT: volume},
    cpu=4,
    memory=16384,
    timeout=60 * 30,
)
def run_schema(max_records: int = 8) -> dict[str, Any]:
    """Audit recording schemas without computing pairwise statistics."""
    sys.path.insert(0, "/opt/recoding_figures")
    from recording_loader import load_iteration_records

    attempts = _attempt_dirs()
    if not attempts:
        raise FileNotFoundError(f"no extracted attempts under {EXTRACT_DIR}")
    records = load_iteration_records(attempts)
    audit = _schema_audit(records, max_records=max_records)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = _json_ready(
        {
            "n_records": len(records),
            "n_tasks": len({record.task for record in records}),
            "schema_audit": audit,
        }
    )
    (OUTPUT_DIR / "schema_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    volume.commit()
    return payload


@app.function(
    image=image,
    volumes={VOLUME_ROOT: volume},
    cpu=16,
    memory=65536,
    timeout=60 * 60 * 4,
)
def run_posthoc(
    max_lag: int = 8,
    candidate_rows: int = 15,
    workers: int = 16,
) -> dict[str, Any]:
    """Run A1-A5 post-hoc statistics over all curated-14 recordings."""
    if max_lag <= 0:
        raise ValueError("max_lag must be positive")
    if candidate_rows <= 0:
        raise ValueError("candidate_rows must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")

    sys.path.insert(0, "/opt/recoding_figures")
    import matplotlib

    matplotlib.use("Agg")
    from expert_cache_metrics import expert_cache_coverage_summary
    from metrics import pairwise_js
    from moe_phase_audit import compute_moe_phase_denominator_audit
    from plot_iter_distance import compute_iter_distance_matrices
    from recording_loader import (
        collect_role_labels,
        load_iteration_records,
        load_token_role_distributions,
    )

    attempts = _attempt_dirs()
    if not attempts:
        raise FileNotFoundError(f"no extracted attempts under {EXTRACT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    records = load_iteration_records(attempts)
    role_labels = collect_role_labels(records)
    loaded_role_labels, token_matrix = load_token_role_distributions(
        records,
        role_labels=role_labels,
    )
    if loaded_role_labels != role_labels:
        raise ValueError("role label collection is not stable")

    schema = _schema_audit(records, max_records=8)
    denominator_audit = compute_moe_phase_denominator_audit(records)
    token_js = pairwise_js(token_matrix)
    token_distance = _distance_summary_extended(records, {0: token_js}, max_lag=max_lag)
    n_experts = _infer_n_experts_for_records(records)
    parallel = _parallel_distribution_sets(
        records=records,
        role_labels=role_labels,
        n_experts=n_experts,
        requested_workers=workers,
        phase_moe_supported=bool(schema.get("phase_separated_moe_supported")),
    )

    attention_distances: dict[str, Any] = {}
    key_role_distances: dict[str, Any] = {}
    residuals: dict[str, Any] = {}
    residual_matrices_by_phase: dict[str, dict[int, Any]] = {}
    attention_matrices_by_phase: dict[str, dict[int, Any]] = {}
    key_role_matrices_by_phase: dict[str, dict[int, Any]] = {}

    for phase in ("all", "prefill", "decode"):
        attention = parallel["attention"][phase]
        key_roles = parallel["key_role"][phase]
        attention_matrices, _ = compute_iter_distance_matrices(attention)
        key_role_matrices, _ = compute_iter_distance_matrices(key_roles)
        attention_matrices_by_phase[phase] = attention_matrices
        key_role_matrices_by_phase[phase] = key_role_matrices
        attention_distances[phase] = _distance_summary_extended(
            records,
            attention_matrices,
            max_lag=max_lag,
        )
        key_role_distances[phase] = _distance_summary_extended(
            records,
            key_role_matrices,
            max_lag=max_lag,
        )
        residual_summary, residual_matrices = _residual_summary_extended(
            records,
            phase,
            attention_matrices,
            key_role_matrices,
            max_lag=max_lag,
        )
        residuals[phase] = residual_summary
        residual_matrices_by_phase[phase] = residual_matrices

    decode_head_level = _head_level_summary_from_aggregates(
        parallel["head_level"],
        role_labels,
        schema,
    )
    moe = parallel["moe"]["all"]
    moe_matrices, _ = compute_iter_distance_matrices(moe)
    moe_distance = _distance_summary_extended(records, moe_matrices, max_lag=max_lag)
    moe_matrices_by_phase: dict[str, dict[int, Any]] = {}
    if bool(schema.get("phase_separated_moe_supported")):
        for phase in ("prefill", "decode"):
            dataset = parallel["moe"][phase]
            if dataset.layers:
                matrices, _ = compute_iter_distance_matrices(dataset)
                moe_matrices_by_phase[phase] = matrices

    moe_phase_summary = _moe_phase_summary(
        records,
        schema,
        denominator_audit,
        moe_matrices_by_phase,
        max_lag=max_lag,
    )
    a4_moe_matrices = moe_matrices_by_phase.get("decode", moe_matrices)
    a4_moe_scope = (
        _moe_scope_label("decode", schema)
        if "decode" in moe_matrices_by_phase
        else "all-token"
    )
    a4_rows = _activity_annotation_rows(
        records=records,
        role_labels=role_labels,
        token_matrix=token_matrix,
        attention_decode_matrices=attention_matrices_by_phase["decode"],
        key_role_decode_matrices=key_role_matrices_by_phase["decode"],
        residual_decode_matrices=residual_matrices_by_phase["decode"],
        moe_matrices=a4_moe_matrices,
        moe_scope=a4_moe_scope,
    )
    blind_rows = _blind_activity_annotation_rows(a4_rows)
    ranked_a4_rows = _rank_candidate_rows(a4_rows)
    _write_activity_template(
        OUTPUT_DIR / "activity_annotation_blind_template.csv",
        blind_rows,
    )
    _write_activity_template(
        OUTPUT_DIR / "activity_transition_candidates_ranked.csv",
        ranked_a4_rows,
    )

    a5 = _attention_moe_independence(
        attention_matrices_by_phase=attention_matrices_by_phase,
        key_role_matrices_by_phase=key_role_matrices_by_phase,
        residual_matrices_by_phase=residual_matrices_by_phase,
        moe_matrices=moe_matrices,
        moe_matrices_by_phase=moe_matrices_by_phase,
        schema=schema,
    )
    prev_iter_dynamic = _prev_iter_dynamic_task_split(
        all_token_moe=moe,
        phase_moe=parallel["moe"],
        moe_matrices_by_phase=moe_matrices_by_phase,
        expert_cache_coverage_summary=expert_cache_coverage_summary,
    )

    summary = {
        "n_records": len(records),
        "n_tasks": len({record.task for record in records}),
        "task_counts": _task_counts(records),
        "role_labels": role_labels,
        "config": {
            "max_lag": max_lag,
            "candidate_rows": candidate_rows,
            "workers": parallel["workers"],
            "input_root": str(EXTRACT_DIR),
            "output_root": str(OUTPUT_DIR),
        },
        "schema_audit": schema,
        "a1_decode_residual": {
            "summary": residuals["decode"],
            "head_level": decode_head_level,
        },
        "a2_pairwise_iter_distance": {
            "token_role": token_distance,
            "phase_aligned_key_role": key_role_distances,
            "attention": attention_distances,
            "moe": moe_distance,
        },
        "a3_moe_phase_label": moe_phase_summary,
        "a4_activity_transition": {
            "manual_annotation_required": True,
            "blind_template_path": str(
                OUTPUT_DIR / "activity_annotation_blind_template.csv"
            ),
            "diagnostic_ranked_path": str(
                OUTPUT_DIR / "activity_transition_candidates_ranked.csv"
            ),
            "moe_scope": a4_moe_scope,
            "candidate_score_definition": (
                "Mean of finite diagnostic components listed in each row's "
                "candidate_score_components field."
            ),
            "n_candidate_transitions": len(a4_rows),
            "top_candidate_rows": ranked_a4_rows[:candidate_rows],
            "verdict": (
                "Current artifacts can rank candidate activity boundaries, but "
                "activity-stratified claims require manual labels."
            ),
        },
        "a5_cross_modality_independence": a5,
        "prev_iter_dynamic_task_split": prev_iter_dynamic,
    }

    clean_summary = _json_ready(summary)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(clean_summary, indent=2) + "\n")
    (OUTPUT_DIR / "summary.md").write_text(
        _summary_markdown(summary),
        encoding="utf-8",
    )
    _plot_a1(summary, figures_dir)
    _plot_a2(summary, figures_dir)
    _plot_a5(summary, figures_dir)

    tar_path = OUTPUT_DIR.with_suffix(".tar.zst")
    if tar_path.exists():
        tar_path.unlink()
    _run(
        [
            "tar",
            "-I",
            "zstd -T0 -3",
            "-cf",
            str(tar_path),
            "-C",
            str(OUTPUT_DIR.parent),
            OUTPUT_DIR.name,
        ]
    )
    volume.commit()
    return {
        "output_dir": str(OUTPUT_DIR),
        "output_tar": str(tar_path),
        "output_tar_bytes": tar_path.stat().st_size,
        "summary": clean_summary,
    }


@app.local_entrypoint()
def main(
    action: str = "posthoc",
    background: bool = False,
    max_lag: int = 8,
    candidate_rows: int = 15,
    schema_records: int = 8,
    workers: int = 16,
) -> None:
    """Run staged A1-A5 post-hoc actions."""
    if action not in {"schema", "posthoc", "all"}:
        raise ValueError("action must be one of: schema, posthoc, all")
    if background:
        if action == "all":
            raise ValueError("background supports action=schema or action=posthoc")
        if action == "schema":
            call = run_schema.spawn(schema_records)
        else:
            call = run_posthoc.spawn(max_lag, candidate_rows, workers)
        print(f"spawned {action}: {call.object_id}")
        print(call.get_dashboard_url())
        return

    if action in {"schema", "all"}:
        print(json.dumps(run_schema.remote(schema_records), indent=2))
    if action in {"posthoc", "all"}:
        result = run_posthoc.remote(max_lag, candidate_rows, workers)
        print(json.dumps(result["summary"], indent=2))


def _attempt_dirs() -> list[Path]:
    return sorted(EXTRACT_DIR.glob("*/attempt_1"))


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _task_counts(records: Sequence[Any]) -> dict[str, int]:
    return {task: sum(record.task == task for record in records) for task in sorted({record.task for record in records})}


def _infer_n_experts_for_records(records: Sequence[Any]) -> int:
    import numpy as np

    n_experts = 0
    for record in records:
        with np.load(record.iter_dir / "routing.npz") as routing:
            n_experts = max(n_experts, int(routing["n_experts"]))
            if "expert_load" in routing.files:
                n_experts = max(n_experts, int(routing["expert_load"].shape[2]))
    if n_experts <= 0:
        raise ValueError("could not infer a positive expert count")
    return n_experts


def _parallel_distribution_sets(
    *,
    records: Sequence[Any],
    role_labels: Sequence[str],
    n_experts: int,
    requested_workers: int,
    phase_moe_supported: bool,
) -> dict[str, Any]:
    from recording_loader import LayerDistributionSet

    n_workers = _resolve_worker_count(requested_workers, len(records))
    descriptors = [_record_descriptor(record) for record in records]
    worker_args = [
        (idx, descriptor, list(role_labels), n_experts, phase_moe_supported)
        for idx, descriptor in enumerate(descriptors)
    ]
    print(
        f"aggregating {len(records)} records with {n_workers} worker processes",
        flush=True,
    )
    if n_workers == 1:
        results = [_record_distribution_worker(args) for args in worker_args]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_record_distribution_worker, args) for args in worker_args]
            for completed, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                if completed == len(futures) or completed % 25 == 0:
                    print(
                        f"aggregated {completed}/{len(futures)} records",
                        flush=True,
                    )
    results = sorted(results, key=lambda item: int(item["index"]))

    return {
        "workers": n_workers,
        "attention": {
            phase: _dataset_from_results(
                modality=f"attention_{phase}",
                records=records,
                axis_labels=role_labels,
                width=len(role_labels),
                results=results,
                aggregate_key="attention",
                phase=phase,
                dataset_cls=LayerDistributionSet,
            )
            for phase in PHASES
        },
        "key_role": {
            phase: _dataset_from_results(
                modality=f"attention_key_role_{phase}",
                records=records,
                axis_labels=role_labels,
                width=len(role_labels),
                results=results,
                aggregate_key="key_role",
                phase=phase,
                dataset_cls=LayerDistributionSet,
            )
            for phase in PHASES
        },
        "moe": {
            phase: _dataset_from_results(
                modality="moe" if phase == "all" else f"moe_{phase}",
                records=records,
                axis_labels=[str(idx) for idx in range(n_experts)],
                width=n_experts,
                results=results,
                aggregate_key="moe",
                phase=phase,
                dataset_cls=LayerDistributionSet,
            )
            for phase in PHASES
        },
        "head_level": {
            "profiles": _merge_head_aggregates(results),
            "n_records": len(results),
            "n_records_with_query_heads": sum(
                1 for result in results if bool(result.get("has_query_heads"))
            ),
        },
    }


def _resolve_worker_count(requested_workers: int, n_records: int) -> int:
    if requested_workers <= 0:
        raise ValueError("workers must be positive")
    available = os.cpu_count() or requested_workers
    return max(1, min(int(requested_workers), int(available), int(n_records)))


def _record_descriptor(record: Any) -> dict[str, Any]:
    return {
        "task": record.task,
        "attempt_dir": str(record.attempt_dir),
        "recordings_dir": str(record.recordings_dir),
        "iter_dir": str(record.iter_dir),
        "call_idx": record.call_idx,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "total_tokens": record.total_tokens,
        "trace_iteration": record.trace_iteration,
        "is_orphan": record.is_orphan,
    }


def _record_from_descriptor(descriptor: dict[str, Any]) -> Any:
    from recording_loader import IterationRecord

    return IterationRecord(
        task=str(descriptor["task"]),
        attempt_dir=Path(str(descriptor["attempt_dir"])),
        recordings_dir=Path(str(descriptor["recordings_dir"])),
        iter_dir=Path(str(descriptor["iter_dir"])),
        call_idx=int(descriptor["call_idx"]),
        input_tokens=descriptor.get("input_tokens"),
        output_tokens=descriptor.get("output_tokens"),
        total_tokens=descriptor.get("total_tokens"),
        trace_iteration=descriptor.get("trace_iteration"),
        is_orphan=bool(descriptor.get("is_orphan", False)),
    )


def _record_distribution_worker(
    args: tuple[int, dict[str, Any], list[str], int, bool],
) -> dict[str, Any]:
    sys.path.insert(0, "/opt/recoding_figures")
    import numpy as np
    from recording_loader import (
        derive_moe_record_phases,
        role_token_counts_for_key_len,
        segment_role_indices_for_record,
    )

    index, descriptor, role_labels, n_experts, phase_moe_supported = args
    record = _record_from_descriptor(descriptor)
    role_width = len(role_labels)
    attention = _new_phase_layer_store()
    key_role = _new_phase_layer_store()
    moe = _new_phase_layer_store()
    head_level: dict[tuple[int, int], dict[str, Any]] = {}
    has_query_heads = False

    segment_payload = json.loads((record.iter_dir / "segments.json").read_text(encoding="utf-8"))
    segments = list(segment_payload.get("segments", []))
    segment_role_cols = segment_role_indices_for_record(record, role_labels)
    visible_cache: dict[int, Any] = {}

    with np.load(record.iter_dir / "attention.npz") as attention_npz:
        record_layers = attention_npz["record_layer"].astype(np.int64)
        record_phases = attention_npz["record_phase"].astype(str)
        offsets = attention_npz["query_row_offsets"].astype(np.int64)
        query_positions = attention_npz["query_positions"].astype(np.int64)
        query_heads = (
            attention_npz["query_heads"].astype(np.int64)
            if "query_heads" in attention_npz.files
            else None
        )
        has_query_heads = query_heads is not None
        segment_mass = attention_npz["segment_mass"].astype(np.float64)
        if np.any(~np.isfinite(segment_mass)):
            raise ValueError(f"{record.iter_dir}: segment_mass contains non-finite values")
        if int(segment_mass.shape[1]) != len(segment_role_cols):
            raise ValueError(
                f"{record.iter_dir}: segment count mismatch "
                f"{segment_mass.shape[1]} vs {len(segment_role_cols)}"
            )
        if int(query_positions.shape[0]) != int(segment_mass.shape[0]):
            raise ValueError(
                f"{record.iter_dir}: query_positions length {query_positions.shape[0]} "
                f"does not match segment_mass rows {segment_mass.shape[0]}"
            )
        if query_heads is not None and int(query_heads.shape[0]) != int(segment_mass.shape[0]):
            raise ValueError(
                f"{record.iter_dir}: query_heads length {query_heads.shape[0]} "
                f"does not match segment_mass rows {segment_mass.shape[0]}"
            )

        for rec_idx, layer in enumerate(record_layers):
            phase = str(record_phases[rec_idx])
            start = int(offsets[rec_idx])
            end = int(offsets[rec_idx + 1])
            if end <= start:
                continue
            rows = segment_mass[start:end]
            row_count = int(end - start)
            role_values = np.zeros(role_width, dtype=np.float64)
            segment_totals = rows.sum(axis=0)
            for segment_idx, role_col in enumerate(segment_role_cols):
                role_values[role_col] += float(segment_totals[segment_idx])
            _add_phase_layer(attention, "all", int(layer), role_values, row_count)
            if phase in {"prefill", "decode"}:
                _add_phase_layer(attention, phase, int(layer), role_values, row_count)

            positions = query_positions[start:end]
            key_values = np.zeros(role_width, dtype=np.float64)
            unique_positions, position_counts = np.unique(positions, return_counts=True)
            for position, count in zip(unique_positions, position_counts, strict=True):
                key_len = int(position) + 1
                if key_len <= 0:
                    continue
                cached = visible_cache.get(key_len)
                if cached is None:
                    cached = role_token_counts_for_key_len(segments, role_labels, key_len)
                    visible_cache[key_len] = cached
                key_values += cached * float(count)
            _add_phase_layer(key_role, "all", int(layer), key_values, row_count)
            if phase in {"prefill", "decode"}:
                _add_phase_layer(key_role, phase, int(layer), key_values, row_count)

            if phase == "decode" and query_heads is not None:
                block_heads = query_heads[start:end]
                for head in np.unique(block_heads):
                    mask = block_heads == int(head)
                    if not bool(mask.any()):
                        continue
                    head_values = np.zeros(role_width, dtype=np.float64)
                    head_segment_totals = rows[mask].sum(axis=0)
                    for segment_idx, role_col in enumerate(segment_role_cols):
                        head_values[role_col] += float(head_segment_totals[segment_idx])
                    _add_head_profile(
                        head_level,
                        int(layer),
                        int(head),
                        head_values,
                        int(mask.sum()),
                    )

    with np.load(record.iter_dir / "routing.npz") as routing:
        record_layers = routing["record_layer"].astype(np.int64)
        expert_load = routing["expert_load"].astype(np.float64)
        if expert_load.ndim != 3:
            raise ValueError(
                f"{record.iter_dir}: expected expert_load rank 3, got {expert_load.shape}"
            )
        expert_values = expert_load.sum(axis=1)
        _add_moe_phase_values(moe, "all", record_layers, expert_values, n_experts)
        if phase_moe_supported:
            record_phases = derive_moe_record_phases(record, routing, expert_load=expert_load)
            for phase in ("prefill", "decode"):
                phase_mask = record_phases.astype(str) == phase
                if bool(phase_mask.any()):
                    _add_moe_phase_values(
                        moe,
                        phase,
                        record_layers[phase_mask],
                        expert_values[phase_mask],
                        n_experts,
                    )

    return {
        "index": index,
        "attention": attention,
        "key_role": key_role,
        "moe": moe,
        "head_level": head_level,
        "has_query_heads": has_query_heads,
    }


def _new_phase_layer_store() -> dict[str, dict[int, dict[str, Any]]]:
    return {phase: {} for phase in PHASES}


def _add_phase_layer(
    store: dict[str, dict[int, dict[str, Any]]],
    phase: str,
    layer: int,
    values: Any,
    count: float,
) -> None:
    import numpy as np

    if count <= 0:
        return
    layer_store = store[phase]
    item = layer_store.get(layer)
    if item is None:
        layer_store[layer] = {
            "values": np.asarray(values, dtype=np.float64).copy(),
            "count": float(count),
        }
        return
    item["values"] += np.asarray(values, dtype=np.float64)
    item["count"] = float(item["count"]) + float(count)


def _add_moe_phase_values(
    store: dict[str, dict[int, dict[str, Any]]],
    phase: str,
    record_layers: Any,
    expert_values: Any,
    n_experts: int,
) -> None:
    import numpy as np

    if int(record_layers.shape[0]) == 0:
        return
    for layer in np.unique(record_layers):
        layer_values = expert_values[record_layers == int(layer)].sum(axis=0)
        values = np.zeros(n_experts, dtype=np.float64)
        values[: int(layer_values.shape[0])] = layer_values
        _add_phase_layer(store, phase, int(layer), values, float(values.sum()))


def _add_head_profile(
    store: dict[tuple[int, int], dict[str, Any]],
    layer: int,
    head: int,
    values: Any,
    count: int,
) -> None:
    import numpy as np

    if count <= 0:
        return
    key = (layer, head)
    item = store.get(key)
    if item is None:
        store[key] = {
            "values": np.asarray(values, dtype=np.float64).copy(),
            "count": int(count),
        }
        return
    item["values"] += np.asarray(values, dtype=np.float64)
    item["count"] = int(item["count"]) + int(count)


def _dataset_from_results(
    *,
    modality: str,
    records: Sequence[Any],
    axis_labels: Sequence[str],
    width: int,
    results: Sequence[dict[str, Any]],
    aggregate_key: str,
    phase: str,
    dataset_cls: Any,
) -> Any:
    import numpy as np

    layers = sorted(
        {
            int(layer)
            for result in results
            for layer in result[aggregate_key][phase].keys()
        }
    )
    distributions: dict[int, Any] = {}
    observation_counts: dict[int, Any] = {}
    for layer in layers:
        matrix = np.zeros((len(records), width), dtype=np.float64)
        counts = np.zeros(len(records), dtype=np.float64)
        for result in results:
            item = result[aggregate_key][phase].get(layer)
            if item is None:
                continue
            index = int(result["index"])
            count = float(item["count"])
            counts[index] = count
            if count > 0:
                matrix[index] = _normalize_distribution(item["values"], width)
        distributions[layer] = matrix
        observation_counts[layer] = counts
    return dataset_cls(
        modality=modality,
        records=list(records),
        layers=layers,
        axis_labels=list(axis_labels),
        distributions=distributions,
        observation_counts=observation_counts,
    )


def _merge_head_aggregates(results: Sequence[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    import numpy as np

    merged: dict[tuple[int, int], dict[str, Any]] = {}
    for result in results:
        for key, item in result["head_level"].items():
            existing = merged.get(key)
            if existing is None:
                merged[key] = {
                    "values": np.asarray(item["values"], dtype=np.float64).copy(),
                    "count": int(item["count"]),
                }
                continue
            existing["values"] += np.asarray(item["values"], dtype=np.float64)
            existing["count"] = int(existing["count"]) + int(item["count"])
    return merged


def _normalize_distribution(values: Any, width: int) -> Any:
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (width,):
        raise ValueError(f"expected distribution width {width}, got {arr.shape}")
    if np.any(arr < 0) or np.any(~np.isfinite(arr)):
        raise ValueError("distribution contains negative or non-finite values")
    total = float(arr.sum())
    if total <= 0:
        return np.zeros(width, dtype=np.float64)
    return arr / total


def _schema_audit(records: Sequence[Any], *, max_records: int) -> dict[str, Any]:
    import numpy as np
    from recording_loader import count_moe_record_phases

    if not records:
        raise ValueError("records must not be empty")
    sample_indices = _sample_indices(len(records), max_records)
    attention_file_counts: dict[str, int] = {}
    routing_file_counts: dict[str, int] = {}
    attention_examples: dict[str, Any] = {}
    routing_examples: dict[str, Any] = {}

    sample_set = set(sample_indices)
    for index, record in enumerate(records):
        with np.load(record.iter_dir / "attention.npz") as attention:
            _count_files(attention.files, attention_file_counts)
            if index in sample_set:
                _capture_examples(attention, attention_examples)
        with np.load(record.iter_dir / "routing.npz") as routing:
            _count_files(routing.files, routing_file_counts)
            if index in sample_set:
                _capture_examples(routing, routing_examples)

    attention_files = sorted(attention_file_counts)
    routing_files = sorted(routing_file_counts)
    routing_phase_fields = [name for name in routing_files if "phase" in name.lower()]
    routing_has_record_phase = routing_file_counts.get("record_phase", 0) == len(records)
    routing_has_token_offsets = routing_file_counts.get("token_row_offsets", 0) == len(records)
    attention_head_fields = [name for name in attention_files if "head" in name.lower()]
    n_records_with_query_heads = attention_file_counts.get("query_heads", 0)
    head_record_coverage = float(n_records_with_query_heads / len(records))
    moe_phase_audit = count_moe_record_phases(records)
    moe_phase_counts = dict(moe_phase_audit["counts"])
    n_prefill = int(moe_phase_counts.get("prefill", 0))
    n_decode = int(moe_phase_counts.get("decode", 0))
    n_mixed = int(moe_phase_counts.get("mixed", 0))
    n_unknown = int(moe_phase_counts.get("unknown", 0))
    phase_derivation_supported = (
        routing_has_token_offsets
        and int(moe_phase_audit["n_iteration_records_failed"]) == 0
        and n_prefill > 0
        and n_decode > 0
    )
    phase_separated_supported = bool(routing_has_record_phase or phase_derivation_supported)
    phase_source = (
        "record_phase"
        if routing_has_record_phase
        else "token_row_offsets+segments+expert_load"
        if phase_derivation_supported
        else "unavailable"
    )
    phase_complete = phase_separated_supported and n_mixed == 0 and n_unknown == 0

    return {
        "sampled_records": [
            {
                "task": records[index].task,
                "call_idx": records[index].call_idx,
                "iter_dir": str(records[index].iter_dir),
            }
            for index in sample_indices
        ],
        "attention_files": attention_files,
        "routing_files": routing_files,
        "attention_file_presence": attention_file_counts,
        "routing_file_presence": routing_file_counts,
        "attention_examples": attention_examples,
        "routing_examples": routing_examples,
        "attention_head_fields": attention_head_fields,
        "head_level_supported": bool(
            attention_head_fields and n_records_with_query_heads == len(records)
        ),
        "attention_head_record_coverage": head_record_coverage,
        "n_records_with_query_heads": n_records_with_query_heads,
        "routing_phase_fields": routing_phase_fields,
        "routing_phase_supported": bool(routing_has_record_phase),
        "routing_token_row_offsets_supported": bool(routing_has_token_offsets),
        "moe_phase_audit": moe_phase_audit,
        "moe_phase_source": phase_source,
        "phase_separated_moe_supported": phase_separated_supported,
        "phase_separated_moe_complete": phase_complete,
        "moe_phase_limitation": _moe_phase_verdict(
            source=phase_source,
            phase_complete=phase_complete,
            counts=moe_phase_counts,
            n_failed=int(moe_phase_audit["n_iteration_records_failed"]),
        ),
    }


def _sample_indices(size: int, max_records: int) -> list[int]:
    if size <= 0:
        return []
    count = min(max_records, size)
    if count == 1:
        return [0]
    return sorted({round(idx * (size - 1) / (count - 1)) for idx in range(count)})


def _moe_phase_verdict(
    *,
    source: str,
    phase_complete: bool,
    counts: dict[str, int],
    n_failed: int,
) -> str:
    if source == "record_phase":
        return (
            "routing.npz contains record_phase, so phase-separated MoE is "
            "computed directly from saved routing labels."
        )
    if source == "token_row_offsets+segments+expert_load":
        unresolved = int(counts.get("mixed", 0)) + int(counts.get("unknown", 0))
        if phase_complete:
            return (
                "routing.npz has no record_phase, but token_row_offsets, "
                "segments.json, and per-segment expert_load fully determine "
                "prefill/decode routing records."
            )
        return (
            "routing.npz has no record_phase, but token_row_offsets, "
            "segments.json, and per-segment expert_load derive usable "
            f"prefill/decode labels; {unresolved} mixed/unknown routing records "
            "are reported separately and excluded from phase-specific MoE."
        )
    return (
        "routing.npz does not expose record_phase and phase derivation did not "
        f"produce both prefill and decode labels; MoE remains all-token. "
        f"Failed records: {n_failed}."
    )


def _count_files(files: Sequence[str], counts: dict[str, int]) -> None:
    for name in files:
        counts[name] = counts.get(name, 0) + 1


def _capture_examples(npz: Any, out: dict[str, Any]) -> None:
    for name in npz.files:
        if name in out:
            continue
        arr = npz[name]
        out[name] = {
            "shape": [int(item) for item in arr.shape],
            "dtype": str(arr.dtype),
        }


def _head_level_status(schema: dict[str, Any]) -> dict[str, Any]:
    fields = list(schema.get("attention_head_fields", []))
    record_coverage = schema.get("attention_head_record_coverage")
    fully_supported = bool(fields) and record_coverage == 1.0
    return {
        "supported": fully_supported,
        "fields": fields,
        "schema_record_coverage": record_coverage,
        "verdict": (
            "Head-level analysis is available for every attention record."
            if fully_supported
            else "Head-level fields are present only for a subset of attention "
            "records; head-level summaries must report record coverage."
            if fields
            else "Current attention.npz samples expose no head-level fields; A1 "
            "head-level claims require updated instrumentation."
        ),
    }


def _head_level_verdict(
    status: dict[str, Any],
    n_records_with_query_heads: int,
    n_records: int,
) -> str:
    if n_records_with_query_heads == n_records:
        return (
            "Head-level decode role profiles were computed from query_heads and "
            "segment_mass during the parallel aggregation pass; interpretation "
            "should use the reported concentration and role-share rows rather "
            "than a binary head-specialization claim."
        )
    return (
        f"Head-level decode role profiles were computed for "
        f"{n_records_with_query_heads}/{n_records} records with query_heads. "
        "Coverage is partial, so head-level claims must be treated as a "
        "schema-limited diagnostic rather than a full-run statistic."
    )


def _head_level_summary_from_aggregates(
    head_level: dict[str, Any],
    role_labels: Sequence[str],
    schema: dict[str, Any],
) -> dict[str, Any]:
    status = _head_level_status(schema)
    n_records = int(head_level.get("n_records", 0))
    n_records_with_query_heads = int(head_level.get("n_records_with_query_heads", 0))
    coverage = (
        float(n_records_with_query_heads / n_records)
        if n_records > 0
        else float("nan")
    )
    coverage_payload = {
        "n_records": float(n_records),
        "n_records_with_query_heads": float(n_records_with_query_heads),
        "record_coverage": coverage,
    }
    if n_records_with_query_heads == 0:
        return {**status, **coverage_payload, "computed": False}

    import numpy as np
    from metrics import pairwise_js, specialization_score

    rows: list[dict[str, Any]] = []
    profiles: list[Any] = []
    labels = list(role_labels)
    profile_items = dict(head_level.get("profiles", {}))
    for (layer, head), item in sorted(profile_items.items()):
        profile = _normalize_distribution(item["values"], len(labels))
        if float(profile.sum()) <= 0:
            continue
        profiles.append(profile)
        top_idx = int(np.argmax(profile))
        rows.append(
            {
                "layer": float(layer),
                "head": float(head),
                "n_decode_query_rows": float(item["count"]),
                "top_role": labels[top_idx],
                "top_role_share": float(profile[top_idx]),
                "specialization": float(specialization_score(profile)),
                "generation_share": _role_share(profile, labels, "generation"),
                "tool_result_share": _role_share(profile, labels, "tool_result"),
                "assistant_call_share": _role_share(profile, labels, "assistant_call"),
            }
        )

    if len(profiles) >= 2:
        matrix = np.vstack(profiles)
        js_matrix = pairwise_js(matrix)
        upper = js_matrix[np.triu_indices(js_matrix.shape[0], k=1)]
        mean_profile_js = _mean(upper)
    else:
        mean_profile_js = float("nan")

    return {
        **status,
        **coverage_payload,
        "computed": True,
        "n_layer_heads": float(len(rows)),
        "n_decode_query_rows": float(sum(int(item["count"]) for item in profile_items.values())),
        "mean_pairwise_head_profile_js": mean_profile_js,
        "median_head_specialization": _median(
            float(row["specialization"]) for row in rows
        ),
        "top_concentrated_heads": sorted(
            rows,
            key=lambda row: float(row["specialization"]),
            reverse=True,
        )[:12],
        "top_generation_heads": sorted(
            rows,
            key=lambda row: float(row["generation_share"]),
            reverse=True,
        )[:12],
        "verdict": _head_level_verdict(status, n_records_with_query_heads, n_records),
    }


def _role_share(profile: Any, labels: Sequence[str], role: str) -> float:
    try:
        idx = labels.index(role)
    except ValueError:
        return float("nan")
    return float(profile[idx])


def _distance_summary_extended(
    records: Sequence[Any],
    matrices: dict[int, Any],
    *,
    max_lag: int,
) -> dict[str, Any]:
    import numpy as np

    all_values: list[float] = []
    adjacent_values: list[float] = []
    same_task_values: list[float] = []
    cross_task_values: list[float] = []
    same_adjacent_values: list[float] = []
    boundary_adjacent_values: list[float] = []
    same_non_adjacent_values: list[float] = []
    lag_values: dict[int, list[float]] = {lag: [] for lag in range(1, max_lag + 1)}

    for matrix in matrices.values():
        n_records = matrix.shape[0]
        for left in range(n_records - 1):
            value = matrix[left, left + 1]
            if np.isfinite(value):
                adjacent_values.append(float(value))
                if records[left].task == records[left + 1].task:
                    same_adjacent_values.append(float(value))
                else:
                    boundary_adjacent_values.append(float(value))
        for left in range(n_records):
            for right in range(left + 1, n_records):
                value = matrix[left, right]
                if not np.isfinite(value):
                    continue
                value_f = float(value)
                all_values.append(value_f)
                if records[left].task == records[right].task:
                    same_task_values.append(value_f)
                    lag = right - left
                    if lag <= max_lag:
                        lag_values[lag].append(value_f)
                    if lag > 1:
                        same_non_adjacent_values.append(value_f)
                else:
                    cross_task_values.append(value_f)

    same_mean = _mean(same_task_values)
    cross_mean = _mean(cross_task_values)
    return {
        "n_layers": float(len(matrices)),
        "n_pairs": float(len(all_values)),
        "mean_pairwise_js": _mean(all_values),
        "mean_adjacent_js": _mean(adjacent_values),
        "mean_same_task_js": same_mean,
        "mean_cross_task_js": cross_mean,
        "cross_over_same_ratio": float(cross_mean / same_mean) if same_mean > 0 else float("nan"),
        "mean_same_task_adjacent_js": _mean(same_adjacent_values),
        "mean_task_boundary_adjacent_js": _mean(boundary_adjacent_values),
        "mean_same_task_non_adjacent_js": _mean(same_non_adjacent_values),
        "lag_profile": [
            {
                "lag": float(lag),
                "mean_js": _mean(values),
                "n_values": float(len(values)),
            }
            for lag, values in lag_values.items()
        ],
    }


def _residual_summary_extended(
    records: Sequence[Any],
    phase: str,
    attention_matrices: dict[int, Any],
    key_role_matrices: dict[int, Any],
    *,
    max_lag: int,
) -> tuple[dict[str, Any], dict[int, Any]]:
    import numpy as np

    rows: list[dict[str, float | str]] = []
    residual_matrices: dict[int, Any] = {}
    for layer, attention_matrix in sorted(attention_matrices.items()):
        if layer not in key_role_matrices:
            continue
        key_matrix = key_role_matrices[layer]
        upper = np.triu_indices(attention_matrix.shape[0], k=1)
        finite = np.isfinite(attention_matrix[upper]) & np.isfinite(key_matrix[upper])
        if int(finite.sum()) < 2:
            continue
        x = key_matrix[upper][finite].astype(np.float64)
        y = attention_matrix[upper][finite].astype(np.float64)
        fit = _linear_fit(x, y)
        residual_upper = y - (fit["intercept"] + fit["slope"] * x)

        residual_matrix = np.full(attention_matrix.shape, np.nan, dtype=np.float64)
        full_finite = np.isfinite(attention_matrix) & np.isfinite(key_matrix)
        residual_matrix[full_finite] = attention_matrix[full_finite] - (
            fit["intercept"] + fit["slope"] * key_matrix[full_finite]
        )
        np.fill_diagonal(residual_matrix, np.nan)
        residual_matrices[int(layer)] = residual_matrix

        same_mask = np.asarray(
            [records[i].task == records[j].task for i, j in zip(upper[0][finite], upper[1][finite])],
            dtype=bool,
        )
        adjacent_residuals = [
            float(residual_matrix[idx, idx + 1])
            for idx in range(residual_matrix.shape[0] - 1)
            if np.isfinite(residual_matrix[idx, idx + 1])
        ]
        rows.append(
            {
                "layer": float(layer),
                "phase": phase,
                "corr_attention_vs_visible_key_role_js": fit["corr"],
                "r2_attention_explained_by_visible_key_role_js": fit["r2"],
                "slope": fit["slope"],
                "intercept": fit["intercept"],
                "mean_abs_residual": float(np.mean(np.abs(residual_upper))),
                "same_task_mean_residual": _mean(residual_upper[same_mask]),
                "cross_task_mean_residual": _mean(residual_upper[~same_mask]),
                "cross_minus_same_residual": _mean(residual_upper[~same_mask]) - _mean(residual_upper[same_mask]),
                "adjacent_mean_residual": _mean(adjacent_residuals),
                "adjacent_mean_abs_residual": _mean(abs(value) for value in adjacent_residuals),
            }
        )

    summary = {
        "layer_rows": rows,
        "mean_corr_attention_vs_visible_key_role_js": _mean(
            float(row["corr_attention_vs_visible_key_role_js"]) for row in rows
        ),
        "median_r2_attention_explained_by_visible_key_role_js": _median(
            float(row["r2_attention_explained_by_visible_key_role_js"]) for row in rows
        ),
        "mean_abs_residual": _mean(float(row["mean_abs_residual"]) for row in rows),
        "mean_cross_minus_same_residual": _mean(
            float(row["cross_minus_same_residual"]) for row in rows
        ),
        "mean_adjacent_residual": _mean(float(row["adjacent_mean_residual"]) for row in rows),
        "mean_adjacent_abs_residual": _mean(
            float(row["adjacent_mean_abs_residual"]) for row in rows
        ),
        "highest_residual_layers": sorted(
            rows,
            key=lambda item: float(item["mean_abs_residual"]),
            reverse=True,
        )[:8],
        "lowest_explained_layers": sorted(
            rows,
            key=lambda item: float(item["r2_attention_explained_by_visible_key_role_js"]),
        )[:8],
        "lag_profile": _residual_lag_profile(records, residual_matrices, max_lag=max_lag),
        "position_bucket_profile": _adjacent_residual_position_buckets(
            records,
            residual_matrices,
        ),
        "adjacent_residual_autocorr": _adjacent_residual_autocorr(records, residual_matrices),
    }
    return summary, residual_matrices


def _residual_lag_profile(
    records: Sequence[Any],
    residual_matrices: dict[int, Any],
    *,
    max_lag: int,
) -> list[dict[str, float]]:
    import numpy as np

    rows: list[dict[str, float]] = []
    for lag in range(1, max_lag + 1):
        values: list[float] = []
        for matrix in residual_matrices.values():
            for left in range(0, matrix.shape[0] - lag):
                right = left + lag
                if records[left].task != records[right].task:
                    continue
                value = matrix[left, right]
                if np.isfinite(value):
                    values.append(float(value))
        rows.append(
            {
                "lag": float(lag),
                "mean_signed_residual": _mean(values),
                "mean_abs_residual": _mean(abs(value) for value in values),
                "n_values": float(len(values)),
            }
        )
    return rows


def _adjacent_residual_position_buckets(
    records: Sequence[Any],
    residual_matrices: dict[int, Any],
) -> list[dict[str, Any]]:
    import numpy as np

    buckets = {
        "early": [],
        "middle": [],
        "late": [],
    }
    positions = _record_positions(records)
    for matrix in residual_matrices.values():
        for idx in range(matrix.shape[0] - 1):
            if records[idx].task != records[idx + 1].task:
                continue
            value = matrix[idx, idx + 1]
            if not np.isfinite(value):
                continue
            bucket = _position_bucket(positions[idx + 1])
            buckets[bucket].append(float(value))
    return [
        {
            "bucket": bucket,
            "mean_signed_residual": _mean(values),
            "mean_abs_residual": _mean(abs(value) for value in values),
            "n_values": float(len(values)),
        }
        for bucket, values in buckets.items()
    ]


def _adjacent_residual_autocorr(
    records: Sequence[Any],
    residual_matrices: dict[int, Any],
) -> dict[str, float]:
    import numpy as np

    correlations: list[float] = []
    for matrix in residual_matrices.values():
        for start, end in _task_spans(records):
            seq = [
                float(matrix[idx, idx + 1])
                for idx in range(start, end - 1)
                if np.isfinite(matrix[idx, idx + 1])
            ]
            if len(seq) >= 3:
                corr = _pearson(seq[:-1], seq[1:])
                if math.isfinite(corr):
                    correlations.append(corr)
    return {
        "mean_autocorr": _mean(correlations),
        "median_autocorr": _median(correlations),
        "n_layer_task_sequences": float(len(correlations)),
    }


def _record_positions(records: Sequence[Any]) -> dict[int, float]:
    positions: dict[int, float] = {}
    for start, end in _task_spans(records):
        denom = max(1, end - start - 1)
        for idx in range(start, end):
            positions[idx] = float((idx - start) / denom)
    return positions


def _position_bucket(position: float) -> str:
    if position < 1.0 / 3.0:
        return "early"
    if position < 2.0 / 3.0:
        return "middle"
    return "late"


def _task_spans(records: Sequence[Any]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(records):
        end = start + 1
        while end < len(records) and records[end].task == records[start].task:
            end += 1
        spans.append((start, end))
        start = end
    return spans


def _moe_phase_summary(
    records: Sequence[Any],
    schema: dict[str, Any],
    denominator_audit: dict[str, Any],
    matrices_by_phase: dict[str, dict[int, Any]],
    *,
    max_lag: int,
) -> dict[str, Any]:
    base = {
        "schema_verdict": schema["moe_phase_limitation"],
        "phase_source": schema.get("moe_phase_source", "unavailable"),
        "phase_complete": bool(schema.get("phase_separated_moe_complete")),
        "phase_count_unit": "routing_records",
        "phase_counts": schema.get("moe_phase_audit", {}).get("counts", {}),
        "denominator_audit": denominator_audit,
        "n_phase_derivation_failures": schema.get("moe_phase_audit", {}).get(
            "n_iteration_records_failed",
            0,
        ),
    }
    if not matrices_by_phase:
        return {
            **base,
            "phase_separated_available": False,
            "phase_summaries": {},
        }
    return {
        **base,
        "phase_separated_available": True,
        "phase_summaries": {
            phase: _distance_summary_extended(records, matrices, max_lag=max_lag)
            for phase, matrices in matrices_by_phase.items()
        },
    }


def _prev_iter_dynamic_task_split(
    *,
    all_token_moe: Any,
    phase_moe: dict[str, Any],
    moe_matrices_by_phase: dict[str, dict[int, Any]],
    expert_cache_coverage_summary: Any,
) -> dict[str, Any]:
    """Compute the task-stratified version of adjacent dynamic expert coverage."""
    ks = (8, 16, 32, 64)
    phase_summaries: dict[str, Any] = {
        "all": expert_cache_coverage_summary(all_token_moe, ks=ks)
    }
    for phase in ("prefill", "decode"):
        dataset = phase_moe.get(phase)
        if dataset is not None and phase in moe_matrices_by_phase and dataset.layers:
            phase_summaries[phase] = expert_cache_coverage_summary(dataset, ks=ks)

    top32 = _coverage_row_for_k(phase_summaries["all"]["coverage_rows"], 32)
    return {
        "metric_scope": (
            "Adjacent previous-iteration dynamic expert coverage, task-stratified. "
            "The all-token top-32 overall field is directly comparable to the "
            "old 0.6417 finding. Cross-task adjacent pairs are synthetic splices "
            "from the task-sorted curated-14 record order, not chronological task "
            "switches in one live run."
        ),
        "ks": [float(item) for item in ks],
        "phase_summaries": phase_summaries,
        "headline_top32_all_token": top32,
        "method_design_conclusion": _prev_iter_method_conclusion(top32),
    }


def _coverage_row_for_k(rows: Sequence[dict[str, Any]], k_value: int) -> dict[str, Any]:
    for row in rows:
        if int(row["k"]) == int(k_value):
            return row
    return {}


def _prev_iter_method_conclusion(row: dict[str, Any]) -> str:
    if not row:
        return "Top-32 adjacent dynamic coverage was not computed."
    same = _float_or_nan(row.get("adjacent_same_task_coverage"))
    splice = _float_or_nan(row.get("adjacent_cross_task_splice_coverage"))
    layer_static = _float_or_nan(row.get("static_layer_coverage"))
    overall = _float_or_nan(row.get("adjacent_prev_iter_coverage"))
    if not all(math.isfinite(value) for value in (same, splice, layer_static, overall)):
        return "Top-32 coverage has missing values; inspect per-task/per-splice rows."
    gap = same - splice
    if splice >= layer_static and gap < 0.05:
        return (
            "Previous-iteration expert coverage is not explained only by "
            "same-task continuity: the synthetic cross-task splice remains close "
            "to same-task coverage and above the layer-static baseline. This is "
            "not evidence of chronological task-switch persistence."
        )
    if splice >= layer_static:
        return (
            "Previous-iteration expert coverage is stronger within task, but the "
            "synthetic cross-task splice still beats the layer-static baseline. "
            "Use this as a cross-task generalization diagnostic, not a real "
            "runtime boundary result."
        )
    if gap >= 0.10:
        return (
            "Previous-iteration expert coverage is mainly a within-task signal; "
            "synthetic cross-task splice coverage falls well below same-task "
            "coverage and below the layer-static baseline."
        )
    return (
        "Previous-iteration expert coverage weakens on synthetic cross-task "
        "splices, but the same-task/splice gap is modest; inspect per-splice rows "
        "before treating it as a task-local prefetch signal."
    )


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _moe_scope_label(phase: str, schema: dict[str, Any]) -> str:
    source = str(schema.get("moe_phase_source", "unavailable"))
    if source == "record_phase":
        return phase
    if source == "token_row_offsets+segments+expert_load":
        return f"{phase}-derived"
    return "all-token"


def _activity_annotation_rows(
    *,
    records: Sequence[Any],
    role_labels: Sequence[str],
    token_matrix: Any,
    attention_decode_matrices: dict[int, Any],
    key_role_decode_matrices: dict[int, Any],
    residual_decode_matrices: dict[int, Any],
    moe_matrices: dict[int, Any],
    moe_scope: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    positions = _record_positions(records)
    for idx in range(len(records) - 1):
        if records[idx].task != records[idx + 1].task:
            continue
        attn_js = _mean_pair_value(attention_decode_matrices, idx, idx + 1)
        key_js = _mean_pair_value(key_role_decode_matrices, idx, idx + 1)
        residual = _mean_pair_value(residual_decode_matrices, idx, idx + 1)
        moe_js = _mean_pair_value(moe_matrices, idx, idx + 1)
        score_components = {
            "decode_attention_adjacent_js": attn_js,
            "abs_decode_residual_adjacent": abs(residual),
            "moe_adjacent_js": moe_js,
        }
        finite_components = [
            name
            for name, value in score_components.items()
            if math.isfinite(float(value))
        ]
        score = _finite_mean(score_components.values())
        row: dict[str, Any] = {
            "task": records[idx + 1].task,
            "prev_call_idx": records[idx].call_idx,
            "call_idx": records[idx + 1].call_idx,
            "prev_trace_iteration": records[idx].trace_iteration,
            "trace_iteration": records[idx + 1].trace_iteration,
            "prev_input_tokens": records[idx].input_tokens,
            "input_tokens": records[idx + 1].input_tokens,
            "prev_output_tokens": records[idx].output_tokens,
            "output_tokens": records[idx + 1].output_tokens,
            "position_bucket": _position_bucket(positions[idx + 1]),
            "decode_attention_adjacent_js": attn_js,
            "decode_key_role_adjacent_js": key_js,
            "decode_residual_adjacent": residual,
            "moe_adjacent_js": moe_js,
            "moe_scope": moe_scope,
            "moe_available": math.isfinite(float(moe_js)),
            "candidate_score": score,
            "candidate_score_n_components": len(finite_components),
            "candidate_score_components": "|".join(finite_components),
            "activity_label": "",
            "notes": "",
        }
        for role_idx, role in enumerate(role_labels):
            row[f"current_role_share_{role}"] = float(token_matrix[idx + 1, role_idx])
        rows.append(row)
    return rows


def _blind_activity_annotation_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    blind_fields = [
        "task",
        "prev_call_idx",
        "call_idx",
        "prev_trace_iteration",
        "trace_iteration",
        "prev_input_tokens",
        "input_tokens",
        "prev_output_tokens",
        "output_tokens",
        "position_bucket",
        "activity_label",
        "notes",
    ]
    return [{field: row.get(field, "") for field in blind_fields} for row in rows]


def _rank_candidate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: _finite_sort_key(row["candidate_score"]),
        reverse=True,
    )


def _write_activity_template(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_ready(row))


def _mean_pair_value(matrices: dict[int, Any], left: int, right: int) -> float:
    values = [
        float(matrix[left, right])
        for matrix in matrices.values()
        if math.isfinite(float(matrix[left, right]))
    ]
    return _mean(values)


def _finite_mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _finite_sort_key(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return -1.0
    return number if math.isfinite(number) else -1.0


def _attention_moe_independence(
    *,
    attention_matrices_by_phase: dict[str, dict[int, Any]],
    key_role_matrices_by_phase: dict[str, dict[int, Any]],
    residual_matrices_by_phase: dict[str, dict[int, Any]],
    moe_matrices: dict[int, Any],
    moe_matrices_by_phase: dict[str, dict[int, Any]],
    schema: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np

    phase_rows: dict[str, list[dict[str, float | str]]] = {}
    phase_summary: dict[str, Any] = {}

    for phase, attention_matrices in attention_matrices_by_phase.items():
        phase_moe_matrices = moe_matrices_by_phase.get(phase, moe_matrices)
        moe_scope = (
            _moe_scope_label(phase, schema)
            if phase in moe_matrices_by_phase
            else "all-token"
        )
        moe_layer_mean = _layer_mean_pairwise(phase_moe_matrices)
        rows: list[dict[str, float | str]] = []
        attention_layer_mean = _layer_mean_pairwise(attention_matrices)
        residual_layer_mean = {
            layer: _matrix_upper_mean_abs(matrix)
            for layer, matrix in residual_matrices_by_phase[phase].items()
        }
        for layer in sorted(set(attention_matrices).intersection(phase_moe_matrices)):
            if (
                layer not in key_role_matrices_by_phase[phase]
                or layer not in residual_matrices_by_phase[phase]
            ):
                continue
            attention_matrix = attention_matrices[layer]
            key_matrix = key_role_matrices_by_phase[phase][layer]
            residual_matrix = residual_matrices_by_phase[phase][layer]
            moe_matrix = phase_moe_matrices[layer]
            upper = np.triu_indices(attention_matrix.shape[0], k=1)
            finite_raw = np.isfinite(attention_matrix[upper]) & np.isfinite(moe_matrix[upper])
            finite_residual = np.isfinite(residual_matrix[upper]) & np.isfinite(moe_matrix[upper])
            finite_key = np.isfinite(key_matrix[upper]) & np.isfinite(moe_matrix[upper])
            raw_corr = _pearson(
                attention_matrix[upper][finite_raw],
                moe_matrix[upper][finite_raw],
            )
            residual_corr = _pearson(
                residual_matrix[upper][finite_residual],
                moe_matrix[upper][finite_residual],
            )
            key_corr = _pearson(
                key_matrix[upper][finite_key],
                moe_matrix[upper][finite_key],
            )
            rows.append(
                {
                    "phase": phase,
                    "moe_scope": moe_scope,
                    "layer": float(layer),
                    "mean_attention_js": attention_layer_mean.get(layer, float("nan")),
                    "mean_abs_attention_residual": residual_layer_mean.get(layer, float("nan")),
                    "mean_moe_js": moe_layer_mean.get(layer, float("nan")),
                    "corr_attention_js_vs_moe_js": raw_corr,
                    "corr_attention_residual_vs_moe_js": residual_corr,
                    "corr_key_role_js_vs_moe_js": key_corr,
                }
            )
        phase_rows[phase] = rows
        phase_summary[phase] = {
            "n_layers": float(len(rows)),
            "moe_scope": moe_scope,
            "mean_corr_attention_js_vs_moe_js": _mean(float(row["corr_attention_js_vs_moe_js"]) for row in rows),
            "mean_corr_attention_residual_vs_moe_js": _mean(
                float(row["corr_attention_residual_vs_moe_js"]) for row in rows
            ),
            "max_abs_corr_attention_residual_vs_moe_js": _max(
                abs(float(row["corr_attention_residual_vs_moe_js"])) for row in rows
            ),
            "spearman_layer_mean_attention_vs_moe": _rank_corr(
                [float(row["mean_attention_js"]) for row in rows],
                [float(row["mean_moe_js"]) for row in rows],
            ),
            "spearman_layer_residual_vs_moe": _rank_corr(
                [float(row["mean_abs_attention_residual"]) for row in rows],
                [float(row["mean_moe_js"]) for row in rows],
            ),
            "top_abs_raw_correlation_layers": sorted(
                rows,
                key=lambda row: abs(float(row["corr_attention_js_vs_moe_js"]))
                if math.isfinite(float(row["corr_attention_js_vs_moe_js"]))
                else -1.0,
                reverse=True,
            )[:8],
            "top_abs_residual_correlation_layers": sorted(
                rows,
                key=lambda row: abs(float(row["corr_attention_residual_vs_moe_js"]))
                if math.isfinite(float(row["corr_attention_residual_vs_moe_js"]))
                else -1.0,
                reverse=True,
            )[:8],
        }

    return {
        "comparison_limitation": (
            "MoE scope is reported per phase. Derived MoE phases exclude "
            "mixed/unknown routing records; when a phase-specific MoE matrix is "
            "unavailable the comparison falls back to all-token MoE and remains "
            "asymmetric. Prefill attention rows are sampled by the attention "
            "recorder, while prefill MoE routing aggregates full-token expert "
            "load; prefill attention-vs-MoE is therefore a proxy check rather "
            "than a token-level matched comparison."
        ),
        "phase_summary": phase_summary,
        "layer_rows": [row for rows in phase_rows.values() for row in rows],
    }


def _layer_mean_pairwise(matrices: dict[int, Any]) -> dict[int, float]:
    return {layer: _matrix_upper_mean(matrix) for layer, matrix in matrices.items()}


def _matrix_upper_mean(matrix: Any) -> float:
    import numpy as np

    upper = matrix[np.triu_indices(matrix.shape[0], k=1)]
    return _mean(upper[np.isfinite(upper)])


def _matrix_upper_mean_abs(matrix: Any) -> float:
    import numpy as np

    upper = matrix[np.triu_indices(matrix.shape[0], k=1)]
    finite = upper[np.isfinite(upper)]
    return _mean(abs(float(value)) for value in finite)


def _linear_fit(x_values: Any, y_values: Any) -> dict[str, float]:
    import numpy as np

    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    if x.size < 2:
        return {"intercept": float("nan"), "slope": float("nan"), "corr": float("nan"), "r2": float("nan")}
    design = np.column_stack([np.ones_like(x), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    corr = _pearson(x, y)
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "corr": corr,
        "r2": float(corr * corr) if math.isfinite(corr) else float("nan"),
    }


def _pearson(x_values: Any, y_values: Any) -> float:
    import numpy as np

    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rank_corr(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    import numpy as np

    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2:
        return float("nan")
    return _pearson(_ranks(x), _ranks(y))


def _ranks(values: Any) -> Any:
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.shape, dtype=np.float64)
    ranks[order] = np.arange(arr.size, dtype=np.float64)
    return ranks


def _mean(values: Any) -> float:
    import numpy as np

    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _median(values: Any) -> float:
    import numpy as np

    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _max(values: Any) -> float:
    import numpy as np

    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if arr.size else float("nan")


def _fmt(value: Any) -> str:
    if value is None:
        return "nan"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "nan"
    return f"{number:.4f}"


def _fmt_count(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "nan"
    return f"{int(round(number)):,}"


def _summary_markdown(summary: dict[str, Any]) -> str:
    a1 = summary["a1_decode_residual"]["summary"]
    a2_attention = summary["a2_pairwise_iter_distance"]["attention"]
    a2_moe = summary["a2_pairwise_iter_distance"]["moe"]
    a3 = summary["a3_moe_phase_label"]
    a4 = summary["a4_activity_transition"]
    a5 = summary["a5_cross_modality_independence"]["phase_summary"]
    prev_iter = summary["prev_iter_dynamic_task_split"]
    head = summary["a1_decode_residual"]["head_level"]
    denominator = a3.get("denominator_audit", {})
    denominator_phases = denominator.get("phases", {})
    denominator_ratios = denominator.get("ratios_decode_over_prefill", {})
    prefill_denominator = denominator_phases.get("prefill", {})
    decode_denominator = denominator_phases.get("decode", {})
    top32 = prev_iter.get("headline_top32_all_token", {})

    lines = [
        "# A1-A5 Agent Attention/MoE Post-Hoc Statistics",
        "",
        f"- Records: `{summary['n_records']}` across `{summary['n_tasks']}` tasks.",
        f"- Input: `{summary['config']['input_root']}`.",
        "- Offline analysis only: no inference rerun and no benchmark code changes.",
        "",
        "## A1 - Decode Residual Structure",
        "",
        f"- Decode median R2 after visible-key role control: `{_fmt(a1['median_r2_attention_explained_by_visible_key_role_js'])}`.",
        f"- Decode mean abs residual JS: `{_fmt(a1['mean_abs_residual'])}`.",
        f"- Decode adjacent mean abs residual: `{_fmt(a1['mean_adjacent_abs_residual'])}`.",
        f"- Adjacent residual autocorr: mean `{_fmt(a1['adjacent_residual_autocorr']['mean_autocorr'])}`, median `{_fmt(a1['adjacent_residual_autocorr']['median_autocorr'])}`.",
        f"- Head-level instrumentation: {head['verdict']}",
    ]
    if head.get("computed"):
        lines.append(
            f"- Decode head profiles: `{_fmt(head['n_layer_heads'])}` layer-heads, "
            f"mean profile JS `{_fmt(head['mean_pairwise_head_profile_js'])}`, "
            f"median specialization `{_fmt(head['median_head_specialization'])}`."
        )
    lines.extend(["", "## A2 - Pairwise Iter Distance Matrix", ""])
    for phase in ("all", "prefill", "decode"):
        item = a2_attention[phase]
        lines.append(
            f"- Attention `{phase}`: pairwise `{_fmt(item['mean_pairwise_js'])}`, "
            f"adjacent `{_fmt(item['mean_adjacent_js'])}`, same-task `{_fmt(item['mean_same_task_js'])}`, "
            f"cross-task `{_fmt(item['mean_cross_task_js'])}`, cross/same `{_fmt(item['cross_over_same_ratio'])}`."
        )
    lines.append(
        f"- MoE all-token: pairwise `{_fmt(a2_moe['mean_pairwise_js'])}`, adjacent `{_fmt(a2_moe['mean_adjacent_js'])}`, same-task `{_fmt(a2_moe['mean_same_task_js'])}`, cross-task `{_fmt(a2_moe['mean_cross_task_js'])}`, cross/same `{_fmt(a2_moe['cross_over_same_ratio'])}`."
    )
    lines.extend(
        [
            "",
            "## A3 - MoE Phase Label Audit",
            "",
            f"- Phase-separated MoE available: `{a3['phase_separated_available']}`.",
            f"- Phase source: `{a3['phase_source']}`; complete: `{a3['phase_complete']}`.",
            f"- Phase count unit: `{a3['phase_count_unit']}`.",
            f"- Routing-record phase counts: `{a3['phase_counts']}`.",
            f"- Routing records: prefill `{_fmt_count(prefill_denominator.get('routing_records'))}`, "
            f"decode `{_fmt_count(decode_denominator.get('routing_records'))}`, "
            f"decode/prefill `{_fmt(denominator_ratios.get('routing_records'))}`.",
            f"- Token rows: prefill `{_fmt_count(prefill_denominator.get('token_rows'))}`, "
            f"decode `{_fmt_count(decode_denominator.get('token_rows'))}`, "
            f"decode/prefill `{_fmt(denominator_ratios.get('token_rows'))}`.",
            f"- Top-k assignments: prefill `{_fmt_count(prefill_denominator.get('topk_assignments'))}`, "
            f"decode `{_fmt_count(decode_denominator.get('topk_assignments'))}`, "
            f"decode/prefill `{_fmt(denominator_ratios.get('topk_assignments'))}`.",
            f"- Expert-load sum: prefill `{_fmt(prefill_denominator.get('expert_load_sum'))}`, "
            f"decode `{_fmt(decode_denominator.get('expert_load_sum'))}`, "
            f"decode/prefill `{_fmt(denominator_ratios.get('expert_load_sum'))}`.",
            f"- Schema verdict: {a3['schema_verdict']}",
            "",
            "## A4 - Activity Transition Stratification",
            "",
            f"- Candidate transition rows: `{a4['n_candidate_transitions']}`.",
            f"- Blind annotation template: `{a4['blind_template_path']}`.",
            f"- Diagnostic ranked candidates: `{a4['diagnostic_ranked_path']}`.",
            f"- MoE scope for diagnostic ranking: `{a4['moe_scope']}`.",
            f"- Candidate score: {a4['candidate_score_definition']}",
            f"- Verdict: {a4['verdict']}",
            "",
            "## A5 - Cross-Modality Independence",
            "",
        ]
    )
    for phase in ("all", "prefill", "decode"):
        item = a5[phase]
        lines.append(
            f"- `{phase}` vs MoE `{item['moe_scope']}`: raw attention-vs-MoE mean r `{_fmt(item['mean_corr_attention_js_vs_moe_js'])}`, "
            f"residual-vs-MoE mean r `{_fmt(item['mean_corr_attention_residual_vs_moe_js'])}`, "
            f"layer-rank residual-vs-MoE rho `{_fmt(item['spearman_layer_residual_vs_moe'])}`."
        )
    lines.extend(
        [
            "",
            "## Prev-Iter Dynamic Expert Coverage",
            "",
            f"- All-token top-32 overall adjacent coverage: `{_fmt(top32.get('adjacent_prev_iter_coverage'))}`.",
            f"- Same-task adjacent coverage: `{_fmt(top32.get('adjacent_same_task_coverage'))}`.",
            f"- Synthetic cross-task splice coverage: `{_fmt(top32.get('adjacent_cross_task_splice_coverage'))}`.",
            f"- Layer-static top-32 baseline: `{_fmt(top32.get('static_layer_coverage'))}`.",
            f"- Equal-task same-task coverage: `{_fmt(top32.get('adjacent_same_task_equal_task_coverage'))}`.",
            f"- Equal-splice cross-task coverage: `{_fmt(top32.get('adjacent_cross_task_splice_equal_splice_coverage'))}`.",
            f"- Method-design conclusion: {prev_iter['method_design_conclusion']}",
            "",
            "## Interpretation Guardrails",
            "",
            "- A1 should be interpreted from the reported residual magnitudes, lag profile, and head-level profile summary; this report does not convert them into a binary claim.",
            "- A2 reports block, lag, and task-structure statistics rather than a thresholded phase/no-phase verdict.",
            "- A3 `phase_counts` are routing-record/schema coverage, not token-volume coverage.",
            "- A4 blind labels must be collected from the blind template before making activity-boundary claims; ranked diagnostics are separated from the labeling file.",
            "- A5 prefill attention-vs-MoE is a sampled-attention/full-token-MoE proxy check, not a strict token-level matched comparison.",
            "- Prev-iter dynamic expert claims must use the task-stratified split above; do not cite the old 64% overall number alone.",
            "- Cross-task splice rows are adjacent only in the task-sorted analysis order, not real chronological task-switch boundaries.",
        ]
    )
    return "\n".join(lines) + "\n"


def _plot_a1(summary: dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rows = summary["a1_decode_residual"]["summary"]["lag_profile"]
    x_values = [int(row["lag"]) for row in rows]
    y_values = [float(row["mean_abs_residual"]) for row in rows]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.plot(x_values, y_values, marker="o", color="#b2796f")
    ax.set_xlabel("within-task lag")
    ax.set_ylabel("mean abs residual JS")
    ax.set_title("A1 decode residual lag profile")
    ax.grid(alpha=0.25)
    _save_figure(fig, output_dir, "a1_decode_residual_lag")


def _plot_a2(summary: dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    phases = ["prefill", "decode", "moe"]
    same = [
        summary["a2_pairwise_iter_distance"]["attention"]["prefill"]["mean_same_task_js"],
        summary["a2_pairwise_iter_distance"]["attention"]["decode"]["mean_same_task_js"],
        summary["a2_pairwise_iter_distance"]["moe"]["mean_same_task_js"],
    ]
    cross = [
        summary["a2_pairwise_iter_distance"]["attention"]["prefill"]["mean_cross_task_js"],
        summary["a2_pairwise_iter_distance"]["attention"]["decode"]["mean_cross_task_js"],
        summary["a2_pairwise_iter_distance"]["moe"]["mean_cross_task_js"],
    ]
    x_positions = np.arange(len(phases))
    width = 0.36
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    ax.bar(x_positions - width / 2, same, width=width, label="same task", color="#4c78a8")
    ax.bar(x_positions + width / 2, cross, width=width, label="cross task", color="#8c8c8c")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(phases)
    ax.set_ylabel("mean JS")
    ax.set_title("A2 task block structure")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _save_figure(fig, output_dir, "a2_same_cross_distance")


def _plot_a5(summary: dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    phases = ["all", "prefill", "decode"]
    phase_summary = summary["a5_cross_modality_independence"]["phase_summary"]
    raw = [phase_summary[phase]["mean_corr_attention_js_vs_moe_js"] for phase in phases]
    residual = [
        phase_summary[phase]["mean_corr_attention_residual_vs_moe_js"]
        for phase in phases
    ]
    x_positions = np.arange(len(phases))
    width = 0.36
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    ax.bar(x_positions - width / 2, raw, width=width, color="#4c78a8", label="raw attention")
    ax.bar(x_positions + width / 2, residual, width=width, color="#b2796f", label="residual")
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(phases)
    ax.set_ylabel("mean Pearson r with MoE JS")
    ax.set_title("A5 attention/MoE proxy check")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _save_figure(fig, output_dir, "a5_attention_moe_proxy")


def _save_figure(fig: Any, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    fig.tight_layout()
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def _json_ready(value: Any) -> Any:
    import numpy as np

    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


if __name__ == "__main__":
    main()
