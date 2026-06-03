"""Modal follow-up experiments for agent attention/MoE recordings."""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import modal


APP_NAME = "asb-agent-attention-modal-followup"
VOLUME_NAME = "asb-terminal-recordings"
VOLUME_ROOT = Path("/data")
EXTRACT_DIR = VOLUME_ROOT / "extracted" / "curated14"
OUTPUT_PARENT = VOLUME_ROOT / "outputs"
OUTPUT_PREFIX = "agent_attention_modal_followup"
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
    .pip_install("matplotlib", "numpy", "scikit-learn", "umap-learn")
    .add_local_dir(RECODING_FIGURES, remote_path="/opt/recoding_figures", copy=True)
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
app = modal.App(APP_NAME)


@app.function(
    image=image,
    volumes={VOLUME_ROOT: volume},
    cpu=16,
    memory=65536,
    timeout=60 * 60 * 4,
)
def run_followup(
    workers: int = 16,
    random_state: int = 0,
    n_bootstrap: int = 5000,
    run_id: str = "",
) -> dict[str, Any]:
    """Run the full follow-up analysis over curated-14 recordings."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")

    sys.path.insert(0, "/opt/recoding_figures")
    import matplotlib
    from modal_followup_metrics import (
        fit_log_linear_half_life,
        layer_hotset_jaccard_summary,
        summarize_correlation_heterogeneity,
        tool_result_segment_ages,
    )
    from plot_iter_distance import compute_iter_distance_matrices
    from recording_loader import (
        LayerDistributionSet,
        collect_role_labels,
        load_iteration_records,
    )

    matplotlib.use("Agg")

    attempts = _attempt_dirs()
    if not attempts:
        raise FileNotFoundError(f"no extracted attempts under {EXTRACT_DIR}")

    output_dir = _new_output_dir(run_id)
    output_dir.mkdir(parents=True, exist_ok=False)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    records = load_iteration_records(attempts)
    role_labels = collect_role_labels(records)
    tool_ages, tool_age_diagnostics = tool_result_segment_ages(records)
    n_experts = _infer_n_experts_for_records(records)
    aggregate = _parallel_record_aggregates(
        records=records,
        role_labels=role_labels,
        n_experts=n_experts,
        tool_ages=tool_ages,
        requested_workers=workers,
    )

    head = _head_specialization_analysis(
        aggregate["head_profiles"],
        random_state=random_state,
        figures_dir=figures_dir,
        output_dir=output_dir,
    )
    tool_decay = _tool_decay_analysis(
        aggregate["tool_decay"],
        fit_log_linear_half_life=fit_log_linear_half_life,
        figures_dir=figures_dir,
        output_dir=output_dir,
    )
    datasets = {
        "attention": {
            phase: _dataset_from_results(
                modality=f"attention_{phase}",
                records=records,
                axis_labels=role_labels,
                width=len(role_labels),
                results=aggregate["results"],
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
                results=aggregate["results"],
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
                results=aggregate["results"],
                aggregate_key="moe",
                phase=phase,
                dataset_cls=LayerDistributionSet,
            )
            for phase in PHASES
        },
    }

    a5 = _attention_moe_layer_correlation(
        attention=datasets["attention"],
        key_role=datasets["key_role"],
        moe=datasets["moe"],
        compute_iter_distance_matrices=compute_iter_distance_matrices,
    )
    a5["heterogeneity"] = summarize_correlation_heterogeneity(
        a5["layer_rows"],
        phases=PHASES,
        random_state=random_state,
        n_bootstrap=n_bootstrap,
    )
    _plot_a5_layer_correlations(a5["layer_rows"], figures_dir)

    hotset_jaccard = layer_hotset_jaccard_summary(datasets["moe"]["all"])
    preferred_k = _preferred_k(hotset_jaccard["rows"], 32)
    _write_jaccard_csv(
        hotset_jaccard,
        output_dir / f"layer_hotset_jaccard_k{preferred_k}.csv",
        preferred_k=preferred_k,
    )
    _plot_jaccard_heatmap(hotset_jaccard, figures_dir)

    summary = {
        "n_records": len(records),
        "n_tasks": len({record.task for record in records}),
        "role_labels": role_labels,
        "config": {
            "workers": aggregate["workers"],
            "random_state": random_state,
            "n_bootstrap": n_bootstrap,
            "run_id": output_dir.name.removeprefix(f"{OUTPUT_PREFIX}_"),
            "input_root": str(EXTRACT_DIR),
            "output_root": str(output_dir),
        },
        "head_specialization": head,
        "tool_result_decay": {
            **tool_decay,
            "age_diagnostics": tool_age_diagnostics,
        },
        "residual_moe_layer_correlation": a5,
        "layer_hotset_jaccard": hotset_jaccard,
    }
    clean_summary = _json_ready(summary)
    (output_dir / "summary.json").write_text(json.dumps(clean_summary, indent=2) + "\n")
    (output_dir / "summary.md").write_text(
        _summary_markdown(clean_summary),
        encoding="utf-8",
    )

    tar_path = _tar_path_for_output_dir(output_dir)
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
            str(output_dir.parent),
            output_dir.name,
        ]
    )
    volume.commit()
    return {
        "output_dir": str(output_dir),
        "output_tar": str(tar_path),
        "output_tar_bytes": tar_path.stat().st_size,
        "summary": clean_summary,
    }


@app.local_entrypoint()
def main(
    background: bool = False,
    workers: int = 16,
    random_state: int = 0,
    n_bootstrap: int = 5000,
    run_id: str = "",
) -> None:
    """Run follow-up experiments."""
    if background:
        call = run_followup.spawn(workers, random_state, n_bootstrap, run_id)
        print(f"spawned followup: {call.object_id}")
        print(call.get_dashboard_url())
        return
    result = run_followup.remote(workers, random_state, n_bootstrap, run_id)
    print(json.dumps(result["summary"], indent=2))


def _attempt_dirs() -> list[Path]:
    return sorted(EXTRACT_DIR.glob("*/attempt_1"))


def _new_output_dir(run_id: str) -> Path:
    clean_id = str(run_id or "").strip()
    if not clean_id:
        clean_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if any(char in clean_id for char in {"/", "\\", ".."}):
        raise ValueError(f"invalid run_id: {clean_id!r}")
    output_dir = OUTPUT_PARENT / f"{OUTPUT_PREFIX}_{clean_id}"
    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists; use a fresh run_id to avoid stale artifacts"
        )
    return output_dir


def _tar_path_for_output_dir(output_dir: Path) -> Path:
    """Return an archive path that preserves the full output directory name."""
    return output_dir.parent / f"{output_dir.name}.tar.zst"


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


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


def _parallel_record_aggregates(
    *,
    records: Sequence[Any],
    role_labels: Sequence[str],
    n_experts: int,
    tool_ages: dict[int, list[int | None]],
    requested_workers: int,
) -> dict[str, Any]:
    descriptors = [_record_descriptor(record) for record in records]
    worker_args = [
        (
            idx,
            descriptor,
            list(role_labels),
            n_experts,
            list(tool_ages.get(idx, [])),
        )
        for idx, descriptor in enumerate(descriptors)
    ]
    n_workers = _resolve_worker_count(requested_workers, len(records))
    print(
        f"aggregating {len(records)} records with {n_workers} worker processes",
        flush=True,
    )
    if n_workers == 1:
        results = [_record_worker(args) for args in worker_args]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_record_worker, args) for args in worker_args]
            for completed, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                if completed == len(futures) or completed % 25 == 0:
                    print(f"aggregated {completed}/{len(futures)} records", flush=True)
    results = sorted(results, key=lambda item: int(item["index"]))
    return {
        "workers": n_workers,
        "results": results,
        "head_profiles": _merge_head_profiles(results),
        "tool_decay": _merge_tool_decay(results),
    }


def _resolve_worker_count(requested_workers: int, n_records: int) -> int:
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


def _record_worker(
    args: tuple[int, dict[str, Any], list[str], int, list[int | None]],
) -> dict[str, Any]:
    sys.path.insert(0, "/opt/recoding_figures")
    import numpy as np
    from recording_loader import (
        derive_moe_record_phases,
        role_token_counts_for_key_len,
        segment_role_indices_for_record,
    )

    index, descriptor, role_labels, n_experts, tool_ages = args
    record = _record_from_descriptor(descriptor)
    role_width = len(role_labels)
    attention = _new_phase_layer_store()
    key_role = _new_phase_layer_store()
    moe = _new_phase_layer_store()
    head_profiles: dict[tuple[int, int], dict[str, Any]] = {}
    tool_decay: dict[tuple[int, int], dict[str, float]] = {}

    segment_payload = json.loads((record.iter_dir / "segments.json").read_text(encoding="utf-8"))
    segments = list(segment_payload.get("segments", []))
    segment_role_cols = segment_role_indices_for_record(record, role_labels)
    if len(tool_ages) != len(segments):
        tool_ages = [None for _segment in segments]
    four_way_indices = _four_way_segment_indices(segments, role_labels, segment_role_cols)
    visible_cache: dict[int, Any] = {}

    with np.load(record.iter_dir / "attention.npz") as attention_npz:
        record_layers = attention_npz["record_layer"].astype(np.int64)
        record_phases = attention_npz["record_phase"].astype(str)
        offsets = attention_npz["query_row_offsets"].astype(np.int64)
        query_positions = attention_npz["query_positions"].astype(np.int64)
        query_heads = attention_npz["query_heads"].astype(np.int64)
        segment_mass = attention_npz["segment_mass"].astype(np.float64)
        if np.any(~np.isfinite(segment_mass)):
            raise ValueError(f"{record.iter_dir}: segment_mass contains non-finite values")
        if int(segment_mass.shape[1]) != len(segment_role_cols):
            raise ValueError(
                f"{record.iter_dir}: segment count mismatch "
                f"{segment_mass.shape[1]} vs {len(segment_role_cols)}"
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
                role_values[int(role_col)] += float(segment_totals[segment_idx])
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

            if phase != "decode":
                continue
            block_heads = query_heads[start:end]
            for head in np.unique(block_heads):
                head_mask = block_heads == int(head)
                if not bool(head_mask.any()):
                    continue
                head_rows = rows[head_mask]
                _add_head_profile(
                    head_profiles,
                    int(layer),
                    int(head),
                    _four_way_values(head_rows, four_way_indices),
                    int(head_mask.sum()),
                )
            for age in sorted({value for value in tool_ages if value is not None}):
                segment_ids = [
                    segment_idx
                    for segment_idx, segment_age in enumerate(tool_ages)
                    if segment_age == age
                ]
                if not segment_ids:
                    continue
                _add_tool_decay(
                    tool_decay,
                    int(layer),
                    int(age),
                    float(rows[:, segment_ids].sum()),
                    float(row_count),
                )

    with np.load(record.iter_dir / "routing.npz") as routing:
        record_layers = routing["record_layer"].astype(np.int64)
        expert_load = routing["expert_load"].astype(np.float64)
        expert_values = expert_load.sum(axis=1)
        _add_moe_phase_values(moe, "all", record_layers, expert_values, n_experts)
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
        "head_profiles": head_profiles,
        "tool_decay": tool_decay,
    }


def _four_way_segment_indices(
    segments: Sequence[dict[str, Any]],
    role_labels: Sequence[str],
    segment_role_cols: Sequence[int],
) -> dict[str, list[int]]:
    roles = [role_labels[int(col)] for col in segment_role_cols]
    latest_user = _latest_index(roles, {"user"})
    latest_tool = _latest_index(roles, {"tool", "tool_result"})
    return {
        "system_mass": [idx for idx, role in enumerate(roles) if role == "system"],
        "latest_user_mass": [] if latest_user is None else [latest_user],
        "latest_tool_mass": [] if latest_tool is None else [latest_tool],
        "generation_mass": [
            idx
            for idx, role in enumerate(roles)
            if role == "generation" and idx < len(segments)
        ],
    }


def _latest_index(roles: Sequence[str], targets: set[str]) -> int | None:
    for idx in range(len(roles) - 1, -1, -1):
        if roles[idx] in targets:
            return idx
    return None


def _four_way_values(rows: Any, indices: dict[str, list[int]]) -> Any:
    import numpy as np

    return np.asarray(
        [
            float(rows[:, indices[name]].sum()) if indices[name] else 0.0
            for name in (
                "system_mass",
                "latest_user_mass",
                "latest_tool_mass",
                "generation_mass",
            )
        ],
        dtype=np.float64,
    )


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
    item = store[phase].get(layer)
    if item is None:
        store[phase][layer] = {
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
    key = (int(layer), int(head))
    item = store.get(key)
    if item is None:
        store[key] = {
            "values": np.asarray(values, dtype=np.float64).copy(),
            "count": float(count),
        }
        return
    item["values"] += np.asarray(values, dtype=np.float64)
    item["count"] = float(item["count"]) + float(count)


def _add_tool_decay(
    store: dict[tuple[int, int], dict[str, float]],
    layer: int,
    age: int,
    mass: float,
    query_rows: float,
) -> None:
    if query_rows <= 0:
        return
    key = (int(layer), int(age))
    item = store.setdefault(key, {"mass": 0.0, "query_rows": 0.0})
    item["mass"] += float(mass)
    item["query_rows"] += float(query_rows)


def _merge_head_profiles(results: Sequence[dict[str, Any]]) -> dict[tuple[int, int], Any]:
    merged: dict[tuple[int, int], dict[str, Any]] = {}
    for result in results:
        for key, item in result["head_profiles"].items():
            _add_head_profile(
                merged,
                int(key[0]),
                int(key[1]),
                item["values"],
                int(item["count"]),
            )
    return merged


def _merge_tool_decay(results: Sequence[dict[str, Any]]) -> dict[tuple[int, int], Any]:
    merged: dict[tuple[int, int], dict[str, float]] = {}
    for result in results:
        for key, item in result["tool_decay"].items():
            _add_tool_decay(
                merged,
                int(key[0]),
                int(key[1]),
                float(item["mass"]),
                float(item["query_rows"]),
            )
    return merged


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
            for layer in result[aggregate_key].get(phase, {})
        }
    )
    distributions: dict[int, Any] = {}
    counts: dict[int, Any] = {}
    for layer in layers:
        matrix = np.zeros((len(records), width), dtype=np.float64)
        obs = np.zeros(len(records), dtype=np.float64)
        for result in results:
            row_idx = int(result["index"])
            item = result[aggregate_key].get(phase, {}).get(layer)
            if item is None:
                continue
            count = float(item["count"])
            values = np.asarray(item["values"], dtype=np.float64)
            if count <= 0 or float(values.sum()) <= 0:
                continue
            matrix[row_idx] = values / float(values.sum())
            obs[row_idx] = count
        distributions[layer] = matrix
        counts[layer] = obs
    return dataset_cls(
        modality=modality,
        records=list(records),
        layers=layers,
        axis_labels=list(axis_labels),
        distributions=distributions,
        observation_counts=counts,
    )


def _head_specialization_analysis(
    profiles: dict[tuple[int, int], Any],
    *,
    random_state: int,
    figures_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    import numpy as np
    from modal_followup_metrics import name_head_cluster
    from sklearn.manifold import TSNE

    try:
        import umap
    except ImportError as exc:  # pragma: no cover - Modal image installs it.
        raise RuntimeError("umap-learn is required for this analysis") from exc

    keys = sorted(profiles)
    rows: list[dict[str, Any]] = []
    matrix_values: list[Any] = []
    for layer, head in keys:
        item = profiles[(layer, head)]
        count = float(item["count"])
        if count <= 0:
            continue
        vector = np.asarray(item["values"], dtype=np.float64) / count
        rows.append(
            {
                "layer": float(layer),
                "head": float(head),
                "n_decode_query_rows": count,
                "system_mass": float(vector[0]),
                "latest_user_mass": float(vector[1]),
                "latest_tool_mass": float(vector[2]),
                "generation_mass": float(vector[3]),
            }
        )
        matrix_values.append(vector)
    matrix = np.vstack(matrix_values)
    cluster_payload = _cluster_heads(matrix, random_state=random_state)
    clusters = cluster_payload["labels"]
    centers = cluster_payload["centers"]
    cluster_names = {
        cluster: name_head_cluster(centers[cluster])
        for cluster in sorted(set(int(item) for item in clusters))
    }

    tsne = TSNE(
        n_components=2,
        perplexity=min(30.0, max(5.0, (matrix.shape[0] - 1) / 3.0)),
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    ).fit_transform(matrix)
    umap_embedding = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.05,
        random_state=random_state,
    ).fit_transform(matrix)

    for idx, row in enumerate(rows):
        cluster = int(clusters[idx])
        row["cluster"] = float(cluster)
        row["cluster_name"] = cluster_names[cluster]
        row["tsne_x"] = float(tsne[idx, 0])
        row["tsne_y"] = float(tsne[idx, 1])
        row["umap_x"] = float(umap_embedding[idx, 0])
        row["umap_y"] = float(umap_embedding[idx, 1])

    layer_cluster = _layer_cluster_counts(rows)
    _write_head_csv(rows, output_dir / "head_specialization_clusters.csv")
    _plot_head_embeddings(rows, figures_dir)
    _plot_layer_cluster_heatmap(layer_cluster, figures_dir)
    return {
        "n_layer_heads": float(len(rows)),
        "vector_labels": [
            "system_mass",
            "latest_user_mass",
            "latest_tool_mass",
            "generation_mass",
        ],
        "selected_k": float(cluster_payload["selected_k"]),
        "candidate_silhouette": cluster_payload["candidate_silhouette"],
        "cluster_centers": [
            {
                "cluster": float(idx),
                "cluster_name": cluster_names[idx],
                "system_mass": float(center[0]),
                "latest_user_mass": float(center[1]),
                "latest_tool_mass": float(center[2]),
                "generation_mass": float(center[3]),
                "n_heads": float(sum(int(row["cluster"]) == idx for row in rows)),
            }
            for idx, center in enumerate(centers)
        ],
        "layer_cluster_counts": layer_cluster,
        "top_rows": sorted(
            rows,
            key=lambda row: max(
                float(row["system_mass"]),
                float(row["latest_user_mass"]),
                float(row["latest_tool_mass"]),
                float(row["generation_mass"]),
            ),
            reverse=True,
        )[:16],
    }


def _cluster_heads(matrix: Any, *, random_state: int) -> dict[str, Any]:
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    scores: dict[int, float] = {}
    models: dict[int, Any] = {}
    for k in (4, 5, 6):
        if matrix.shape[0] <= k:
            continue
        model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = model.fit_predict(matrix)
        scores[k] = float(silhouette_score(matrix, labels))
        models[k] = model
    if not scores:
        raise ValueError("not enough heads for k=4..6 clustering")
    selected_k = max(scores, key=lambda key: scores[key])
    model = models[selected_k]
    return {
        "selected_k": selected_k,
        "labels": np.asarray(model.labels_, dtype=np.int64),
        "centers": np.asarray(model.cluster_centers_, dtype=np.float64),
        "candidate_silhouette": {str(key): value for key, value in sorted(scores.items())},
    }


def _layer_cluster_counts(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    layers = sorted({int(row["layer"]) for row in rows})
    clusters = sorted({int(row["cluster"]) for row in rows})
    out: list[dict[str, Any]] = []
    for layer in layers:
        row: dict[str, Any] = {"layer": float(layer)}
        for cluster in clusters:
            row[f"cluster_{cluster}"] = float(
                sum(int(item["layer"]) == layer and int(item["cluster"]) == cluster for item in rows)
            )
        out.append(row)
    return out


def _tool_decay_analysis(
    tool_decay: dict[tuple[int, int], Any],
    *,
    fit_log_linear_half_life: Any,
    figures_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    layer_age_rows: list[dict[str, Any]] = []
    global_by_age: dict[int, dict[str, float]] = {}
    for (layer, age), item in sorted(tool_decay.items()):
        mass = float(item["mass"])
        query_rows = float(item["query_rows"])
        mean_mass = mass / query_rows if query_rows > 0 else float("nan")
        layer_age_rows.append(
            {
                "layer": float(layer),
                "age_in_iters": float(age),
                "attention_mass": mass,
                "query_rows": query_rows,
                "mean_attention_mass": mean_mass,
            }
        )
        global_item = global_by_age.setdefault(age, {"mass": 0.0, "query_rows": 0.0})
        global_item["mass"] += mass
        global_item["query_rows"] += query_rows

    global_rows = [
        {
            "age_in_iters": float(age),
            "attention_mass": item["mass"],
            "query_rows": item["query_rows"],
            "mean_attention_mass": (
                item["mass"] / item["query_rows"]
                if item["query_rows"] > 0
                else float("nan")
            ),
        }
        for age, item in sorted(global_by_age.items())
    ]
    per_layer_half_life = []
    for layer in sorted({int(row["layer"]) for row in layer_age_rows}):
        rows = [row for row in layer_age_rows if int(row["layer"]) == layer]
        per_layer_half_life.append({"layer": float(layer), **fit_log_linear_half_life(rows)})

    _write_rows_csv(layer_age_rows, output_dir / "tool_result_decay_by_layer.csv")
    _plot_tool_decay(global_rows, layer_age_rows, figures_dir)
    global_fit = fit_log_linear_half_life(global_rows)
    finite_halves = [
        float(row["half_life_iters"])
        for row in per_layer_half_life
        if row.get("half_life_iters") is not None
    ]
    return {
        "phase": "decode",
        "definition": (
            "For each tool_result segment, age is current call_idx minus the "
            "first same-task call_idx where that message segment is visible. "
            "Mean attention mass is conditional on the age bucket being visible "
            "in that record/layer."
        ),
        "global_age_rows": global_rows,
        "global_log_linear_fit": global_fit,
        "per_layer_half_life": per_layer_half_life,
        "median_layer_half_life_iters": _median(finite_halves),
        "n_layers_with_half_life": float(len(finite_halves)),
    }


def _attention_moe_layer_correlation(
    *,
    attention: dict[str, Any],
    key_role: dict[str, Any],
    moe: dict[str, Any],
    compute_iter_distance_matrices: Any,
) -> dict[str, Any]:
    layer_rows: list[dict[str, Any]] = []
    phase_summary: dict[str, Any] = {}
    for phase in PHASES:
        attention_matrices, _ = compute_iter_distance_matrices(attention[phase])
        key_matrices, _ = compute_iter_distance_matrices(key_role[phase])
        moe_matrices, _ = compute_iter_distance_matrices(moe[phase])
        residual_matrices = _residual_matrices(attention_matrices, key_matrices)
        phase_rows: list[dict[str, Any]] = []
        for layer in sorted(set(attention_matrices).intersection(moe_matrices)):
            if layer not in key_matrices or layer not in residual_matrices:
                continue
            upper = _upper_indices(attention_matrices[layer])
            raw_corr = _pearson(
                attention_matrices[layer][upper],
                moe_matrices[layer][upper],
            )
            residual_corr = _pearson(
                residual_matrices[layer][upper],
                moe_matrices[layer][upper],
            )
            key_corr = _pearson(key_matrices[layer][upper], moe_matrices[layer][upper])
            row = {
                "phase": phase,
                "layer": float(layer),
                "mean_attention_js": _matrix_upper_mean(attention_matrices[layer]),
                "mean_key_role_js": _matrix_upper_mean(key_matrices[layer]),
                "mean_abs_attention_residual": _matrix_upper_mean_abs(
                    residual_matrices[layer]
                ),
                "mean_moe_js": _matrix_upper_mean(moe_matrices[layer]),
                "corr_attention_js_vs_moe_js": raw_corr,
                "corr_attention_residual_vs_moe_js": residual_corr,
                "corr_key_role_js_vs_moe_js": key_corr,
            }
            phase_rows.append(row)
            layer_rows.append(row)
        phase_summary[phase] = {
            "n_layers": float(len(phase_rows)),
            "mean_corr_attention_js_vs_moe_js": _mean(
                float(row["corr_attention_js_vs_moe_js"]) for row in phase_rows
            ),
            "mean_corr_attention_residual_vs_moe_js": _mean(
                float(row["corr_attention_residual_vs_moe_js"]) for row in phase_rows
            ),
            "mean_corr_key_role_js_vs_moe_js": _mean(
                float(row["corr_key_role_js_vs_moe_js"]) for row in phase_rows
            ),
        }
    return {
        "definition": (
            "Per-layer Pearson correlations over pairwise iteration-distance "
            "upper triangles. Residual attention is attention-role JS after a "
            "per-layer linear control for visible-key role JS."
        ),
        "phase_summary": phase_summary,
        "layer_rows": layer_rows,
    }


def _residual_matrices(
    attention_matrices: dict[int, Any],
    key_matrices: dict[int, Any],
) -> dict[int, Any]:
    import numpy as np

    residuals: dict[int, Any] = {}
    for layer in sorted(set(attention_matrices).intersection(key_matrices)):
        attention = np.asarray(attention_matrices[layer], dtype=np.float64)
        key = np.asarray(key_matrices[layer], dtype=np.float64)
        upper = _upper_indices(attention)
        x = key[upper]
        y = attention[upper]
        finite = np.isfinite(x) & np.isfinite(y)
        residual = np.full_like(attention, np.nan, dtype=np.float64)
        if np.sum(finite) >= 2 and float(np.std(x[finite])) > 0:
            slope, intercept = np.polyfit(x[finite], y[finite], deg=1)
            predicted = slope * x + intercept
        else:
            predicted = np.full_like(y, np.nanmean(y[finite]))
        values = y - predicted
        residual[upper] = values
        residual[(upper[1], upper[0])] = values
        np.fill_diagonal(residual, 0.0)
        residuals[layer] = residual
    return residuals


def _upper_indices(matrix: Any) -> Any:
    import numpy as np

    return np.triu_indices(np.asarray(matrix).shape[0], k=1)


def _pearson(left: Any, right: Any) -> float:
    import numpy as np

    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left_arr) & np.isfinite(right_arr)
    if np.sum(finite) < 2:
        return float("nan")
    x = left_arr[finite]
    y = right_arr[finite]
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _matrix_upper_mean(matrix: Any) -> float:
    import numpy as np

    values = np.asarray(matrix, dtype=np.float64)[_upper_indices(matrix)]
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def _matrix_upper_mean_abs(matrix: Any) -> float:
    import numpy as np

    values = np.asarray(matrix, dtype=np.float64)[_upper_indices(matrix)]
    values = values[np.isfinite(values)]
    return float(np.mean(np.abs(values))) if values.size else float("nan")


def _write_head_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    _write_rows_csv(rows, path)


def _write_rows_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jaccard_csv(
    summary: dict[str, Any],
    path: Path,
    *,
    preferred_k: int,
) -> None:
    matrix_payload = summary["matrices"][str(preferred_k)]
    layers = [int(layer) for layer in matrix_payload["layers"]]
    matrix = matrix_payload["matrix"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer"] + layers)
        for layer, row in zip(layers, matrix, strict=True):
            writer.writerow([layer] + row)


def _plot_head_embeddings(rows: Sequence[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    clusters = sorted({int(row["cluster"]) for row in rows})
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=False, sharey=False)
    for ax, prefix, title in [
        (axes[0], "tsne", "t-SNE"),
        (axes[1], "umap", "UMAP"),
    ]:
        for cluster in clusters:
            selected = [row for row in rows if int(row["cluster"]) == cluster]
            ax.scatter(
                [float(row[f"{prefix}_x"]) for row in selected],
                [float(row[f"{prefix}_y"]) for row in selected],
                s=12,
                alpha=0.75,
                color=colors(cluster % 10),
                label=f"c{cluster}",
            )
        ax.set_title(title)
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        ax.grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    _save_figure(fig, output_dir, "head_specialization_tsne_umap")


def _plot_layer_cluster_heatmap(
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    layers = [int(row["layer"]) for row in rows]
    cluster_cols = [key for key in rows[0] if key.startswith("cluster_")]
    matrix = np.asarray([[float(row[col]) for col in cluster_cols] for row in rows])
    fig, ax = plt.subplots(figsize=(7.0, 8.2))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title("Head cluster count by layer")
    ax.set_xlabel("cluster")
    ax.set_ylabel("layer")
    ax.set_xticks(range(len(cluster_cols)))
    ax.set_xticklabels([col.replace("cluster_", "c") for col in cluster_cols])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    _save_figure(fig, output_dir, "head_cluster_by_layer")


def _plot_tool_decay(
    global_rows: Sequence[dict[str, Any]],
    layer_age_rows: Sequence[dict[str, Any]],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.plot(
        [float(row["age_in_iters"]) for row in global_rows],
        [float(row["mean_attention_mass"]) for row in global_rows],
        marker="o",
        linewidth=1.8,
        color="#4c78a8",
    )
    ax.set_xlabel("tool-result age in iterations")
    ax.set_ylabel("decode attention mass")
    ax.set_title("Tool-result attention decay")
    ax.grid(alpha=0.25)
    _save_figure(fig, output_dir, "tool_result_decay_global")

    layers = sorted({int(row["layer"]) for row in layer_age_rows})
    ages = sorted({int(row["age_in_iters"]) for row in layer_age_rows})
    matrix = np.full((len(layers), len(ages)), np.nan, dtype=np.float64)
    for row in layer_age_rows:
        matrix[layers.index(int(row["layer"])), ages.index(int(row["age_in_iters"]))] = float(
            row["mean_attention_mass"]
        )
    fig, ax = plt.subplots(figsize=(9.2, 8.2))
    vmax = float(np.nanpercentile(matrix, 98)) if np.isfinite(matrix).any() else None
    image = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0.0, vmax=vmax)
    ax.set_title("Tool-result attention mass by layer and age")
    ax.set_xlabel("age in iterations")
    ax.set_ylabel("layer")
    ax.set_xticks(range(len(ages)))
    ax.set_xticklabels(ages, rotation=90)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    _save_figure(fig, output_dir, "tool_result_decay_by_layer")


def _plot_a5_layer_correlations(rows: Sequence[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    phases = list(PHASES)
    fig, axes = plt.subplots(len(phases), 1, figsize=(8.2, 7.2), sharex=True)
    for ax, phase in zip(axes, phases, strict=True):
        phase_rows = [row for row in rows if row["phase"] == phase]
        ax.plot(
            [float(row["layer"]) for row in phase_rows],
            [float(row["corr_attention_js_vs_moe_js"]) for row in phase_rows],
            marker="o",
            markersize=3,
            linewidth=1.0,
            color="#4c78a8",
            label="raw attention",
        )
        ax.plot(
            [float(row["layer"]) for row in phase_rows],
            [float(row["corr_attention_residual_vs_moe_js"]) for row in phase_rows],
            marker="o",
            markersize=3,
            linewidth=1.0,
            color="#b2796f",
            label="residual",
        )
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set_ylabel(f"{phase}\nr")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("layer")
    fig.suptitle("Per-layer attention/MoE correlation")
    _save_figure(fig, output_dir, "residual_moe_per_layer_correlation")


def _plot_jaccard_heatmap(summary: dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    preferred_k = _preferred_k(summary["rows"], 32)
    payload = summary["matrices"][str(preferred_k)]
    layers = [int(layer) for layer in payload["layers"]]
    matrix = np.asarray(payload["matrix"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.0, 7.2))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(f"Layer-static expert hotset Jaccard (k={preferred_k})")
    ax.set_xlabel("layer")
    ax.set_ylabel("layer")
    tick_positions = list(range(0, len(layers), 4))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([layers[idx] for idx in tick_positions])
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([layers[idx] for idx in tick_positions])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    _save_figure(fig, output_dir, f"layer_hotset_jaccard_heatmap_k{preferred_k}")


def _save_figure(fig: Any, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def _summary_markdown(summary: dict[str, Any]) -> str:
    head = summary["head_specialization"]
    decay = summary["tool_result_decay"]
    residual = summary["residual_moe_layer_correlation"]
    jaccard = summary["layer_hotset_jaccard"]
    preferred_k = _preferred_k(jaccard["rows"], 32)
    k32 = _row_for_k(jaccard["rows"], preferred_k)
    lines = [
        "# Agent Attention Modal Follow-up - 2026-05-11",
        "",
        f"- Records: `{summary['n_records']}` across `{summary['n_tasks']}` tasks.",
        "- Offline analysis only: no inference rerun and no benchmark code changes.",
        "",
        "## Head Specialization",
        "",
        f"- Layer-head vectors: `{_fmt(head['n_layer_heads'])}`.",
        f"- Selected KMeans k: `{_fmt(head['selected_k'])}`.",
        "- Cluster centers:",
        "",
        "| Cluster | Name | system | latest user | latest tool | generation | heads |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in head["cluster_centers"]:
        lines.append(
            "| "
            f"{_fmt(row['cluster'])} | {row['cluster_name']} | "
            f"{_fmt(row['system_mass'])} | {_fmt(row['latest_user_mass'])} | "
            f"{_fmt(row['latest_tool_mass'])} | {_fmt(row['generation_mass'])} | "
            f"{_fmt(row['n_heads'])} |"
        )
    lines.extend(
        [
            "",
            "## Tool-Result Decay",
            "",
            f"- Global half-life: `{_fmt(decay['global_log_linear_fit']['half_life_iters'])}` iterations.",
            f"- Layers with finite half-life: `{_fmt(decay['n_layers_with_half_life'])}`.",
            f"- Median layer half-life: `{_fmt(decay['median_layer_half_life_iters'])}` iterations.",
            "",
            "## Residual Attention vs MoE",
            "",
            "| Phase | Mean raw r | Mean residual r | CI low | CI high | positive / negative layers |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for phase in PHASES:
        phase_summary = residual["phase_summary"][phase]
        hetero = residual["heterogeneity"][phase]
        lines.append(
            "| "
            f"{phase} | {_fmt(phase_summary['mean_corr_attention_js_vs_moe_js'])} | "
            f"{_fmt(phase_summary['mean_corr_attention_residual_vs_moe_js'])} | "
            f"{_fmt(hetero['bootstrap_mean_residual_corr_ci95_low'])} | "
            f"{_fmt(hetero['bootstrap_mean_residual_corr_ci95_high'])} | "
            f"{_fmt(hetero['n_positive_residual_layers'])} / "
            f"{_fmt(hetero['n_negative_residual_layers'])} |"
        )
    lines.extend(
        [
            "",
            "## Layer-Hotset Jaccard",
            "",
            "| k | pairwise | adjacent | non-adjacent | adjacent - non-adjacent |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in jaccard["rows"]:
        lines.append(
            "| "
            f"{_fmt(row['k'])} | {_fmt(row['mean_pairwise_jaccard'])} | "
            f"{_fmt(row['adjacent_layer_jaccard'])} | "
            f"{_fmt(row['non_adjacent_layer_jaccard'])} | "
            f"{_fmt(row['adjacent_minus_non_adjacent'])} |"
        )
    lines.extend(
        [
            "",
            f"Primary k={preferred_k} result: pairwise layer-hotset Jaccard "
            f"`{_fmt(k32['mean_pairwise_jaccard'])}`, adjacent "
            f"`{_fmt(k32['adjacent_layer_jaccard'])}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def _preferred_k(rows: Sequence[dict[str, Any]], requested_k: int) -> int:
    available = sorted({int(float(row["k"])) for row in rows})
    if not available:
        raise ValueError("no hotset Jaccard rows available")
    eligible = [value for value in available if value <= requested_k]
    return max(eligible) if eligible else min(available)


def _row_for_k(rows: Sequence[dict[str, Any]], k: int) -> dict[str, Any]:
    for row in rows:
        if int(float(row["k"])) == int(k):
            return row
    raise KeyError(k)


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(number):
        return "nan"
    if abs(number - round(number)) < 1e-9 and abs(number) >= 1:
        return str(int(round(number)))
    return f"{number:.4f}"


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


def _json_ready(value: Any) -> Any:
    import numpy as np

    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


if __name__ == "__main__":
    main()
