"""Modal offline research analysis for agent attention, KV value, and MoE locality."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import modal


APP_NAME = "asb-agent-attention-moe-research"
VOLUME_NAME = "asb-terminal-recordings"
VOLUME_ROOT = Path("/data")
EXTRACT_DIR = VOLUME_ROOT / "extracted" / "curated14"
OUTPUT_DIR = VOLUME_ROOT / "outputs" / "agent_attention_moe_research_20260509"

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
    cpu=16,
    memory=65536,
    timeout=60 * 60 * 4,
)
def run_research() -> dict[str, Any]:
    """Run the full offline research analysis over the curated-14 recordings."""
    sys.path.insert(0, "/opt/recoding_figures")

    from expert_cache_metrics import expert_cache_coverage_summary
    from metrics import pairwise_js
    from plot_iter_distance import compute_iter_distance_matrices
    from recording_loader import (
        load_attention_distributions,
        load_attention_key_role_distributions,
        load_iteration_records,
        load_moe_distributions,
        load_token_role_distributions,
    )

    attempts = _attempt_dirs()
    if not attempts:
        raise FileNotFoundError(f"no extracted attempts under {EXTRACT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_iteration_records(attempts)
    role_labels, token_matrix = load_token_role_distributions(records)
    token_js = pairwise_js(token_matrix)

    attention_phase_summaries: dict[str, Any] = {}
    residual_summaries: dict[str, Any] = {}
    role_value_summaries: dict[str, Any] = {}
    attention_matrices_by_phase: dict[str, dict[int, Any]] = {}
    key_role_matrices_by_phase: dict[str, dict[int, Any]] = {}
    key_role_summaries: dict[str, Any] = {}

    for phase in ("all", "prefill", "decode"):
        attention = load_attention_distributions(
            records,
            role_labels=role_labels,
            phase=phase,
        )
        key_roles = load_attention_key_role_distributions(
            records,
            role_labels=role_labels,
            phase=phase,
        )
        matrices, _finite_values = compute_iter_distance_matrices(attention)
        key_role_matrices, _finite_values = compute_iter_distance_matrices(key_roles)
        attention_matrices_by_phase[phase] = matrices
        key_role_matrices_by_phase[phase] = key_role_matrices
        attention_phase_summaries[phase] = _distance_summary(records, matrices)
        key_role_summaries[phase] = _distance_summary(records, key_role_matrices)
        residual_summaries[phase] = _residual_summary(
            records,
            phase,
            matrices,
            key_role_matrices,
        )
        role_value_summaries[phase] = _role_value_summary(
            attention,
            key_roles,
            role_labels,
        )

    moe = load_moe_distributions(records)
    moe_matrices, _finite_values = compute_iter_distance_matrices(moe)
    moe_summary = _distance_summary(records, moe_matrices)
    expert_cache = _expert_cache_summary(
        moe,
        ks=(8, 16, 32, 64),
        expert_cache_coverage_summary=expert_cache_coverage_summary,
    )
    coupling = _attention_moe_coupling(
        records,
        attention_matrices_by_phase,
        key_role_matrices_by_phase,
        moe_matrices,
    )

    summary = {
        "n_records": len(records),
        "n_tasks": len({record.task for record in records}),
        "task_counts": {
            task: sum(record.task == task for record in records)
            for task in sorted({record.task for record in records})
        },
        "role_labels": role_labels,
        "token_role_mean": _role_dict(role_labels, token_matrix.mean(axis=0)),
        "token_role_distance": _distance_summary(records, {0: token_js}),
        "phase_aligned_key_role_distance": key_role_summaries,
        "attention_phase_distance": attention_phase_summaries,
        "phase1_measurement_residuals": residual_summaries,
        "phase2_role_kv_value": role_value_summaries,
        "phase3_moe_distance": moe_summary,
        "phase3_moe_cacheability": expert_cache,
        "phase4_attention_moe_coupling": coupling,
    }

    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUTPUT_DIR / "research_findings.md").write_text(_summary_markdown(summary))

    tar_path = OUTPUT_DIR.with_suffix(".tar.zst")
    if tar_path.exists():
        tar_path.unlink()
    _run(["tar", "-I", "zstd -T0 -3", "-cf", str(tar_path), "-C", str(OUTPUT_DIR.parent), OUTPUT_DIR.name])
    volume.commit()
    return {
        "output_dir": str(OUTPUT_DIR),
        "output_tar": str(tar_path),
        "output_tar_bytes": tar_path.stat().st_size,
        "summary": summary,
    }


@app.local_entrypoint()
def main(background: bool = False) -> None:
    """Run the research analysis."""
    if background:
        call = run_research.spawn()
        print(f"spawned research: {call.object_id}")
        print(call.get_dashboard_url())
        return
    result = run_research.remote()
    print(json.dumps(result["summary"], indent=2))


def _attempt_dirs() -> list[Path]:
    return sorted(EXTRACT_DIR.glob("*/attempt_1"))


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _role_dict(role_labels: Sequence[str], values: Any) -> dict[str, float]:
    return {role: float(values[idx]) for idx, role in enumerate(role_labels)}


def _distance_summary(records: Sequence[Any], matrices: dict[int, Any]) -> dict[str, float]:
    import numpy as np

    all_values: list[float] = []
    adjacent_values: list[float] = []
    same_task_values: list[float] = []
    cross_task_values: list[float] = []
    for matrix in matrices.values():
        for idx in range(matrix.shape[0] - 1):
            value = matrix[idx, idx + 1]
            if np.isfinite(value):
                adjacent_values.append(float(value))
        for left in range(matrix.shape[0]):
            for right in range(left + 1, matrix.shape[0]):
                value = matrix[left, right]
                if not np.isfinite(value):
                    continue
                value_f = float(value)
                all_values.append(value_f)
                if records[left].task == records[right].task:
                    same_task_values.append(value_f)
                else:
                    cross_task_values.append(value_f)
    same_mean = _mean(same_task_values)
    cross_mean = _mean(cross_task_values)
    return {
        "mean_pairwise_js": _mean(all_values),
        "mean_adjacent_js": _mean(adjacent_values),
        "mean_same_task_js": same_mean,
        "mean_cross_task_js": cross_mean,
        "cross_over_same_ratio": float(cross_mean / same_mean) if same_mean > 0 else float("nan"),
        "n_pairs": float(len(all_values)),
    }


def _residual_summary(
    records: Sequence[Any],
    phase: str,
    attention_matrices: dict[int, Any],
    key_role_matrices: dict[int, Any],
) -> dict[str, Any]:
    import numpy as np

    rows: list[dict[str, float]] = []

    for layer, matrix in sorted(attention_matrices.items()):
        if layer not in key_role_matrices:
            continue
        key_role_js = key_role_matrices[layer]
        upper = np.triu_indices(key_role_js.shape[0], k=1)
        key_role_upper_all = key_role_js[upper]
        same_mask_all = np.asarray(
            [records[i].task == records[j].task for i, j in zip(upper[0], upper[1])],
            dtype=bool,
        )
        attn_upper_all = matrix[upper]
        finite = np.isfinite(attn_upper_all) & np.isfinite(key_role_upper_all)
        if int(finite.sum()) < 2:
            continue
        x = key_role_upper_all[finite].astype(np.float64)
        y = attn_upper_all[finite].astype(np.float64)
        fit = _linear_fit(x, y)
        residual = y - (fit["intercept"] + fit["slope"] * x)
        same_mask = same_mask_all[finite]
        adjacent_residuals: list[float] = []
        for idx in range(matrix.shape[0] - 1):
            if not np.isfinite(matrix[idx, idx + 1]) or not np.isfinite(
                key_role_js[idx, idx + 1]
            ):
                continue
            pred = fit["intercept"] + fit["slope"] * float(key_role_js[idx, idx + 1])
            adjacent_residuals.append(float(matrix[idx, idx + 1] - pred))
        rows.append(
            {
                "layer": float(layer),
                "phase": phase,
                "corr_attention_vs_visible_key_role_js": fit["corr"],
                "r2_attention_explained_by_visible_key_role_js": fit["r2"],
                "slope": fit["slope"],
                "intercept": fit["intercept"],
                "mean_abs_residual": float(np.mean(np.abs(residual))),
                "same_task_mean_residual": _nanmean_array(residual[same_mask]),
                "cross_task_mean_residual": _nanmean_array(residual[~same_mask]),
                "cross_minus_same_residual": _nanmean_array(residual[~same_mask])
                - _nanmean_array(residual[same_mask]),
                "adjacent_mean_residual": _mean(adjacent_residuals),
            }
        )

    return {
        "layer_rows": rows,
        "mean_corr_attention_vs_visible_key_role_js": _mean(
            row["corr_attention_vs_visible_key_role_js"] for row in rows
        ),
        "median_r2_attention_explained_by_visible_key_role_js": _median(
            row["r2_attention_explained_by_visible_key_role_js"] for row in rows
        ),
        "mean_abs_residual": _mean(row["mean_abs_residual"] for row in rows),
        "mean_cross_minus_same_residual": _mean(
            row["cross_minus_same_residual"] for row in rows
        ),
        "mean_adjacent_residual": _mean(row["adjacent_mean_residual"] for row in rows),
        "highest_residual_layers": sorted(
            rows,
            key=lambda item: item["mean_abs_residual"],
            reverse=True,
        )[:5],
        "lowest_explained_layers": sorted(
            rows,
            key=lambda item: item["r2_attention_explained_by_visible_key_role_js"],
        )[:5],
    }


def _role_value_summary(
    attention_dataset: Any,
    key_role_dataset: Any,
    role_labels: Sequence[str],
) -> dict[str, Any]:
    import numpy as np
    from recording_loader import average_layer_matrix

    layers, layer_attention, _counts = average_layer_matrix(
        attention_dataset,
        equal_iter_weight=True,
    )
    key_layers, layer_key_roles, _counts = average_layer_matrix(
        key_role_dataset,
        layers=layers,
        equal_iter_weight=True,
    )
    if key_layers != layers:
        raise ValueError("attention and key-role layers are not aligned")

    equal_layer_attention = layer_attention.mean(axis=0)
    equal_layer_key_roles = layer_key_roles.mean(axis=0)
    equal_layer_enrichment = _safe_divide(equal_layer_attention, equal_layer_key_roles)
    layer_enrichment = _safe_divide(layer_attention, layer_key_roles)

    row_weighted_attention = _weighted_profile(attention_dataset, layers)
    row_weighted_key_roles = _weighted_profile(key_role_dataset, layers)
    row_weighted_enrichment = _safe_divide(row_weighted_attention, row_weighted_key_roles)

    role_rows: list[dict[str, float | str]] = []
    for idx, role in enumerate(role_labels):
        finite_layer_values = layer_enrichment[:, idx]
        finite_layer_values = finite_layer_values[np.isfinite(finite_layer_values)]
        max_layer_idx = int(np.nanargmax(layer_enrichment[:, idx])) if finite_layer_values.size else -1
        role_rows.append(
            {
                "role": role,
                "equal_layer_key_role_share": float(equal_layer_key_roles[idx]),
                "equal_layer_attention_mass": float(equal_layer_attention[idx]),
                "equal_layer_attention_per_key_token": float(equal_layer_enrichment[idx]),
                "row_weighted_key_role_share": float(row_weighted_key_roles[idx]),
                "row_weighted_attention_mass": float(row_weighted_attention[idx]),
                "row_weighted_attention_per_key_token": float(row_weighted_enrichment[idx]),
                "median_layer_enrichment": _nanmedian_array(layer_enrichment[:, idx]),
                "max_layer_enrichment": _nanmax_array(layer_enrichment[:, idx]),
                "max_enrichment_layer": float(layers[max_layer_idx]) if max_layer_idx >= 0 else float("nan"),
            }
        )

    layer_rows = []
    for layer_idx, layer in enumerate(layers):
        layer_rows.append(
            {
                "layer": float(layer),
                "equal_iter_attention_mass": _role_dict(role_labels, layer_attention[layer_idx]),
                "equal_iter_key_role_share": _role_dict(role_labels, layer_key_roles[layer_idx]),
                "equal_iter_attention_per_key_token": _role_dict(
                    role_labels,
                    layer_enrichment[layer_idx],
                ),
            }
        )

    return {
        "role_rows": role_rows,
        "layer_rows": layer_rows,
        "top_enriched_roles": sorted(
            role_rows,
            key=lambda item: float(item["row_weighted_attention_per_key_token"])
            if math.isfinite(float(item["row_weighted_attention_per_key_token"]))
            else -1.0,
            reverse=True,
        ),
    }


def _expert_cache_summary(
    dataset: Any,
    *,
    ks: Sequence[int],
    expert_cache_coverage_summary: Any,
) -> dict[str, Any]:
    summary = expert_cache_coverage_summary(dataset, ks=ks)
    summary["moe_recording_phase_limit"] = (
        "routing.npz does not record prefill/decode labels in this historical "
        "script; revised A1-A5 analysis derives phase labels from "
        "token_row_offsets, segments.json, and expert_load."
    )
    return summary


def _attention_moe_coupling(
    records: Sequence[Any],
    attention_matrices_by_phase: dict[str, dict[int, Any]],
    key_role_matrices_by_phase: dict[str, dict[int, Any]],
    moe_matrices: dict[int, Any],
) -> dict[str, Any]:
    import numpy as np

    rows: list[dict[str, float | str]] = []
    for phase, attention_matrices in attention_matrices_by_phase.items():
        key_role_matrices = key_role_matrices_by_phase[phase]
        for layer in sorted(set(attention_matrices).intersection(moe_matrices)):
            if layer not in key_role_matrices:
                continue
            attn = attention_matrices[layer]
            key_role = key_role_matrices[layer]
            moe = moe_matrices[layer]
            upper = np.triu_indices(key_role.shape[0], k=1)
            finite = (
                np.isfinite(attn[upper])
                & np.isfinite(moe[upper])
                & np.isfinite(key_role[upper])
            )
            if int(finite.sum()) < 2:
                continue
            attn_values = attn[upper][finite].astype(np.float64)
            moe_values = moe[upper][finite].astype(np.float64)
            key_role_values = key_role[upper][finite].astype(np.float64)
            fit = _linear_fit(key_role_values, attn_values)
            residual = attn_values - (fit["intercept"] + fit["slope"] * key_role_values)
            rows.append(
                {
                    "phase": phase,
                    "layer": float(layer),
                    "corr_attention_js_vs_moe_js": _pearson(attn_values, moe_values),
                    "corr_attention_residual_vs_moe_js": _pearson(residual, moe_values),
                }
            )

    by_phase: dict[str, dict[str, float]] = {}
    for phase in sorted({str(row["phase"]) for row in rows}):
        phase_rows = [row for row in rows if row["phase"] == phase]
        by_phase[phase] = {
            "mean_corr_attention_js_vs_moe_js": _mean(
                float(row["corr_attention_js_vs_moe_js"]) for row in phase_rows
            ),
            "mean_corr_attention_residual_vs_moe_js": _mean(
                float(row["corr_attention_residual_vs_moe_js"]) for row in phase_rows
            ),
        }

    return {
        "layer_rows": rows,
        "phase_summary": by_phase,
        "highest_residual_coupling_layers": sorted(
            rows,
            key=lambda item: float(item["corr_attention_residual_vs_moe_js"])
            if math.isfinite(float(item["corr_attention_residual_vs_moe_js"]))
            else -2.0,
            reverse=True,
        )[:8],
    }


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


def _safe_divide(numerator: Any, denominator: Any) -> Any:
    import numpy as np

    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)


def _weighted_profile(dataset: Any, layers: Sequence[int]) -> Any:
    import numpy as np

    total: Any | None = None
    total_weight = 0.0
    for layer in layers:
        matrix = dataset.distributions[layer]
        weights = dataset.observation_counts[layer].astype(np.float64)
        valid = weights > 0
        if not bool(valid.any()):
            continue
        weighted = np.sum(matrix[valid] * weights[valid, None], axis=0)
        total = weighted if total is None else total + weighted
        total_weight += float(weights[valid].sum())
    if total is None or total_weight <= 0:
        return np.zeros(len(dataset.axis_labels), dtype=np.float64)
    return total / float(total.sum())


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


def _nanmean_array(values: Any) -> float:
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _nanmedian_array(values: Any) -> float:
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _nanmax_array(values: Any) -> float:
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if arr.size else float("nan")


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "nan"
    return f"{number:.4f}"


def _row_for_k(rows: Sequence[dict[str, Any]], k_value: int) -> dict[str, Any] | None:
    for row in rows:
        if int(row["k"]) == k_value:
            return row
    return None


def _role_row(rows: Sequence[dict[str, Any]], role: str) -> dict[str, Any] | None:
    for row in rows:
        if row["role"] == role:
            return row
    return None


def _phase1_evidence_lines(phase1: dict[str, Any]) -> list[str]:
    lines = [
        "- Residuals are computed after controlling for phase-aligned visible-key role composition, not final-call token composition."
    ]
    prefill_r2 = float(
        phase1["prefill"]["median_r2_attention_explained_by_visible_key_role_js"]
    )
    decode_r2 = float(
        phase1["decode"]["median_r2_attention_explained_by_visible_key_role_js"]
    )
    prefill_residual = float(phase1["prefill"]["mean_abs_residual"])
    decode_residual = float(phase1["decode"]["mean_abs_residual"])
    if math.isfinite(prefill_r2) and math.isfinite(decode_r2):
        if prefill_r2 > decode_r2:
            lines.append(
                f"- Visible-key role composition explains prefill attention distances more strongly than decode: median R2 `{_fmt(prefill_r2)}` vs `{_fmt(decode_r2)}`."
            )
        else:
            lines.append(
                f"- Decode is at least as role-composition-explained as prefill in median R2: `{_fmt(decode_r2)}` vs `{_fmt(prefill_r2)}`."
            )
    if math.isfinite(prefill_residual) and math.isfinite(decode_residual):
        if decode_residual > prefill_residual:
            lines.append(
                f"- Decode has larger post-composition residual distance than prefill: `{_fmt(decode_residual)}` vs `{_fmt(prefill_residual)}`."
            )
        else:
            lines.append(
                f"- Prefill residual distance is not lower than decode: `{_fmt(prefill_residual)}` vs `{_fmt(decode_residual)}`."
            )
    lines.append(
        "- Any paper claim about phase/block structure should report both raw attention distance and this residual view."
    )
    return lines


def _phase2_evidence_lines(
    system_prefill: dict[str, Any] | None,
    system_decode: dict[str, Any] | None,
    generation_decode: dict[str, Any] | None,
) -> list[str]:
    lines: list[str] = []
    if system_prefill is not None:
        enrichment = float(system_prefill["row_weighted_attention_per_key_token"])
        if math.isfinite(enrichment) and enrichment > 1.0:
            lines.append(
                f"- `system` is row-weighted attention-enriched in prefill with enrichment `{_fmt(enrichment)}`."
            )
        else:
            lines.append(
                f"- `system` is not row-weighted attention-enriched in prefill by this metric: enrichment `{_fmt(enrichment)}`."
            )
    if system_decode is not None:
        lines.append(
            f"- Decode `system` attention mass is `{_fmt(system_decode['row_weighted_attention_mass'])}` against visible-key share `{_fmt(system_decode['row_weighted_key_role_share'])}`."
        )
    if generation_decode is not None:
        lines.append(
            f"- Decode `generation` attention mass is `{_fmt(generation_decode['row_weighted_attention_mass'])}` against visible-key share `{_fmt(generation_decode['row_weighted_key_role_share'])}`."
        )
    lines.append(
        "- Roles with very small visible-key share should be treated as diagnostics unless the effect is replicated per task."
    )
    return lines


def _phase3_evidence_lines(top_row: dict[str, Any]) -> list[str]:
    layer_gain = float(top_row["static_layer_minus_global"])
    adjacent = float(top_row["adjacent_prev_iter_coverage"])
    layer_static = float(top_row["static_layer_coverage"])
    global_static = float(top_row["static_global_coverage"])
    lines = [
        f"- At top-{int(top_row['k'])}, layer-static coverage is `{_fmt(layer_static)}` vs global-static `{_fmt(global_static)}`."
    ]
    if math.isfinite(layer_gain) and layer_gain > 0:
        lines.append(f"- Layer-specific hotsets improve over a global hotset by `{_fmt(layer_gain)}` coverage.")
    else:
        lines.append(f"- Layer-specific hotsets do not improve over global hotsets at this k: delta `{_fmt(layer_gain)}`.")
    if math.isfinite(adjacent) and math.isfinite(layer_static):
        if adjacent > layer_static:
            lines.append(
                f"- Adjacent previous-iteration dynamic coverage beats layer-static coverage at this k: `{_fmt(adjacent)}` vs `{_fmt(layer_static)}`."
            )
        else:
            lines.append(
                f"- Adjacent previous-iteration dynamic coverage does not beat layer-static coverage at this k: `{_fmt(adjacent)}` vs `{_fmt(layer_static)}`."
            )
    lines.append(
        f"- Layer hotset pairwise Jaccard is `{_fmt(top_row['layer_hotset_pairwise_jaccard'])}`, measuring how much expert identity is shared across layers."
    )
    return lines


def _phase4_evidence_lines(phase_summary: dict[str, Any]) -> list[str]:
    residual_values = [
        abs(float(item["mean_corr_attention_residual_vs_moe_js"]))
        for item in phase_summary.values()
        if math.isfinite(float(item["mean_corr_attention_residual_vs_moe_js"]))
    ]
    lines = [
        "- These are historical asymmetric correlations: phase-specific attention "
        "vs all-token MoE. The revised A1-A5 workflow derives phase-specific MoE "
        "from token_row_offsets, segments.json, and expert_load."
    ]
    if residual_values and max(residual_values) < 0.3:
        lines.append(
            f"- Residual attention distance is a weak proxy for MoE distance here; max absolute mean residual correlation is `{_fmt(max(residual_values))}`."
        )
    elif residual_values:
        lines.append(
            f"- Some residual attention/MoE coupling is visible; max absolute mean residual correlation is `{_fmt(max(residual_values))}`."
        )
    else:
        lines.append("- Residual attention/MoE coupling is undefined for this run.")
    return lines


def _phase5_lines(
    phase1: dict[str, Any],
    system_prefill: dict[str, Any] | None,
    system_decode: dict[str, Any] | None,
    top_row: dict[str, Any],
) -> list[str]:
    supported: list[str] = [
        "- Role-composition controls are required before using role-aggregated attention distance plots as evidence of latent phase structure."
    ]
    if system_prefill is not None and float(system_prefill["row_weighted_attention_per_key_token"]) > 1.0:
        supported.append("- `system` KV preservation/compression deserves explicit policy treatment in prefill.")
    if system_decode is not None and float(system_decode["row_weighted_attention_mass"]) > 0.1:
        supported.append("- Decode still assigns non-trivial mass to `system`, so decode-only eviction policies should not drop system KV blindly.")
    if float(top_row["static_layer_minus_global"]) > 0:
        supported.append("- MoE expert cache baselines should be layer-specific before considering a global hotset.")
    if float(phase1["decode"]["mean_abs_residual"]) > float(phase1["prefill"]["mean_abs_residual"]):
        supported.append("- Decode deserves separate serving analysis because its post-composition attention residual is larger than prefill.")

    lines = ["Supported by current evidence:", "", *supported]
    lines.extend(
        [
            "",
            "Not yet supported:",
            "",
            "- A claim that role-aggregated block/stripe plots alone prove intrinsic phase transitions.",
            "- Phase-separated MoE serving conclusions from this historical all-token report; use the revised A1-A5 phase-derived MoE analysis for that question.",
            "- A dynamic adjacent-iteration expert policy unless its coverage beats the layer-static baseline for the chosen cache budget.",
        ]
    )
    return lines


def _summary_markdown(summary: dict[str, Any]) -> str:
    phase1 = summary["phase1_measurement_residuals"]
    phase2 = summary["phase2_role_kv_value"]
    cache_rows = summary["phase3_moe_cacheability"]["coverage_rows"]
    top32 = _row_for_k(cache_rows, 32) or cache_rows[-1]
    system_prefill = _role_row(phase2["prefill"]["role_rows"], "system")
    system_decode = _role_row(phase2["decode"]["role_rows"], "system")
    generation_decode = _role_row(phase2["decode"]["role_rows"], "generation")

    lines = [
        "# Agent Attention/MoE Research Findings - 2026-05-09",
        "",
        f"- Records: `{summary['n_records']}` trace-aligned LLM calls across `{summary['n_tasks']}` tasks.",
        "- Source: curated-14 Terminal-Bench Qwen3-Coder internal recordings on Modal Volume.",
        "- This is offline analysis only: no inference rerun and no benchmark code changes.",
        "- Token-role controls in this report are phase-aligned: each attention record is compared to only the key tokens visible at that record's key length.",
        "",
        "## Phase 1 - Measurement Artifact Separation",
        "",
    ]

    for phase in ("all", "prefill", "decode"):
        item = phase1[phase]
        lines.append(
            f"- `{phase}`: mean corr(attention JS, visible-key-role JS) "
            f"`{_fmt(item['mean_corr_attention_vs_visible_key_role_js'])}`, median R2 "
            f"`{_fmt(item['median_r2_attention_explained_by_visible_key_role_js'])}`, "
            f"mean abs residual `{_fmt(item['mean_abs_residual'])}`, "
            f"cross-minus-same residual `{_fmt(item['mean_cross_minus_same_residual'])}`."
        )
    lines.extend(["", "Evidence status:", ""])
    lines.extend(_phase1_evidence_lines(phase1))
    lines.extend(["", "## Phase 2 - KV Value by Role and Phase", ""])

    for phase in ("all", "prefill", "decode"):
        lines.append(f"### {phase}")
        for row in phase2[phase]["role_rows"]:
            lines.append(
                f"- `{row['role']}`: visible-key share "
                f"`{_fmt(row['row_weighted_key_role_share'])}`, attention mass "
                f"`{_fmt(row['row_weighted_attention_mass'])}`, "
                f"attention/key-token enrichment "
                f"`{_fmt(row['row_weighted_attention_per_key_token'])}`, "
                f"equal-layer enrichment "
                f"`{_fmt(row['equal_layer_attention_per_key_token'])}`, "
                f"max layer `{_fmt(row['max_enrichment_layer'])}`."
            )
        lines.append("")

    lines.extend(["Evidence status:", ""])
    lines.extend(_phase2_evidence_lines(system_prefill, system_decode, generation_decode))
    lines.extend(
        [
            "",
            "## Phase 3 - MoE Expert Locality and Cacheability",
            "",
            f"- MoE pairwise JS: mean `{_fmt(summary['phase3_moe_distance']['mean_pairwise_js'])}`, "
            f"adjacent `{_fmt(summary['phase3_moe_distance']['mean_adjacent_js'])}`, "
            f"same-task `{_fmt(summary['phase3_moe_distance']['mean_same_task_js'])}`, "
            f"cross-task `{_fmt(summary['phase3_moe_distance']['mean_cross_task_js'])}`.",
            f"- Recording limitation: {summary['phase3_moe_cacheability']['moe_recording_phase_limit']}.",
            "",
        ]
    )
    for row in cache_rows:
        lines.append(
            f"- top-{int(row['k'])}: global static coverage `{_fmt(row['static_global_coverage'])}`, "
            f"layer static `{_fmt(row['static_layer_coverage'])}`, "
            f"adjacent previous-iter `{_fmt(row['adjacent_prev_iter_coverage'])}`, "
            f"same-task adjacent `{_fmt(row['adjacent_same_task_coverage'])}`, "
            f"layer hotset Jaccard `{_fmt(row['layer_hotset_pairwise_jaccard'])}`."
        )
    lines.extend(["", "Evidence status:", ""])
    lines.extend(_phase3_evidence_lines(top32))
    lines.extend(["", "## Phase 4 - Attention/MoE Coupling", ""])
    lines.append(
        "- Coupling rows in this historical report compare phase-specific "
        "attention distances against all-token MoE routing distances. Revised "
        "A1-A5 outputs use phase-derived MoE where available."
    )
    for phase, item in summary["phase4_attention_moe_coupling"]["phase_summary"].items():
        lines.append(
            f"- `{phase}` attention vs all-token MoE: mean corr(attention JS, MoE JS) "
            f"`{_fmt(item['mean_corr_attention_js_vs_moe_js'])}`, "
            f"mean corr(attention residual, MoE JS) "
            f"`{_fmt(item['mean_corr_attention_residual_vs_moe_js'])}`."
        )
    lines.extend(["", "Evidence status:", ""])
    lines.extend(_phase4_evidence_lines(summary["phase4_attention_moe_coupling"]["phase_summary"]))
    lines.extend(["", "## Phase 5 - Serving Research Implications", ""])
    lines.extend(_phase5_lines(phase1, system_prefill, system_decode, top32))
    lines.extend(
        [
            "",
            "Next measurements to add:",
            "",
            "- Prefer explicit MoE routing phase labels in future recordings even though the current artifact can derive phases for this analysis.",
            "- Add position/recency buckets within role so KV value can distinguish old system, recent tool output, and generated context.",
            "- Validate per-task phase structure before using combined-all14 plots for paper claims.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
