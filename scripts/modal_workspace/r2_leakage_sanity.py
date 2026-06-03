"""Sanity checks for high attention-vs-visible-key-role R2 values."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import modal


APP_NAME = "asb-agent-attention-r2-sanity"
VOLUME_NAME = "asb-terminal-recordings"
VOLUME_ROOT = Path("/data")
EXTRACT_DIR = VOLUME_ROOT / "extracted" / "curated14"
OUTPUT_DIR = VOLUME_ROOT / "outputs" / "agent_attention_r2_sanity_20260509"

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
    .pip_install("numpy")
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
def run_sanity(n_permutations: int = 20, seed: int = 20260509) -> dict[str, Any]:
    """Run full-data R2 leakage sanity checks over curated-14 recordings."""
    sys.path.insert(0, "/opt/recoding_figures")

    from metrics import pairwise_js
    from recording_loader import (
        collect_role_labels,
        load_attention_distributions,
        load_attention_key_role_distributions,
        load_iteration_records,
    )

    attempts = sorted(EXTRACT_DIR.glob("*/attempt_1"))
    if not attempts:
        raise FileNotFoundError(f"no extracted attempts under {EXTRACT_DIR}")
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive")

    records = load_iteration_records(attempts)
    role_labels = collect_role_labels(records)
    tasks = [record.task for record in records]
    rng = __import__("numpy").random.default_rng(seed)

    phase_summaries: dict[str, Any] = {}
    for phase in ("all", "prefill", "decode"):
        print(f"loading phase={phase}", flush=True)
        attention = load_attention_distributions(records, role_labels=role_labels, phase=phase)
        key_roles = load_attention_key_role_distributions(
            records,
            role_labels=role_labels,
            phase=phase,
        )
        attention_matrices = _compute_distance_matrices(attention, pairwise_js)
        key_role_matrices = _compute_distance_matrices(key_roles, pairwise_js)
        phase_summaries[phase] = _phase_sanity(
            phase=phase,
            tasks=tasks,
            role_labels=role_labels,
            attention=attention,
            key_roles=key_roles,
            attention_matrices=attention_matrices,
            key_role_matrices=key_role_matrices,
            n_permutations=n_permutations,
            rng=rng,
            pairwise_js=pairwise_js,
        )

    summary = {
        "n_records": len(records),
        "n_tasks": len(set(tasks)),
        "role_labels": role_labels,
        "n_permutations": n_permutations,
        "seed": seed,
        "feature_target_definition": {
            "pairwise_feature": (
                "JS distance between phase-aligned visible-key role distributions "
                "for two LLM calls at a given layer"
            ),
            "pairwise_target": (
                "JS distance between attention mass distributions over the same "
                "normalized role axis for the same two calls/layer"
            ),
            "system_mass_check": (
                "per-call linear fit of system visible-key share to system attention mass"
            ),
            "not_cross_validated_original_r2": (
                "the original figure used in-sample Pearson r^2 over all finite call pairs"
            ),
        },
        "phase_summaries": phase_summaries,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUTPUT_DIR / "summary.md").write_text(_summary_markdown(summary))
    tar_path = OUTPUT_DIR.with_suffix(".tar.zst")
    if tar_path.exists():
        tar_path.unlink()
    _run(["tar", "-I", "zstd -T0 -3", "-cf", str(tar_path), "-C", str(OUTPUT_DIR.parent), OUTPUT_DIR.name])
    volume.commit()
    return {"output_dir": str(OUTPUT_DIR), "output_tar": str(tar_path), "summary": summary}


@app.local_entrypoint()
def main(background: bool = False, n_permutations: int = 20, seed: int = 20260509) -> None:
    """Run or spawn the sanity checks."""
    if background:
        call = run_sanity.spawn(n_permutations, seed)
        print(f"spawned sanity: {call.object_id}")
        print(call.get_dashboard_url())
        return
    result = run_sanity.remote(n_permutations, seed)
    print(json.dumps(result["summary"], indent=2))


def _phase_sanity(
    *,
    phase: str,
    tasks: Sequence[str],
    role_labels: Sequence[str],
    attention: Any,
    key_roles: Any,
    attention_matrices: dict[int, Any],
    key_role_matrices: dict[int, Any],
    n_permutations: int,
    rng: Any,
    pairwise_js: Any,
) -> dict[str, Any]:
    import numpy as np

    layer_rows: list[dict[str, float]] = []
    system_rows: list[dict[str, float]] = []
    system_idx = role_labels.index("system") if "system" in role_labels else None
    task_arr = np.asarray(tasks)

    for layer in sorted(set(attention_matrices).intersection(key_role_matrices)):
        attn_matrix = attention_matrices[layer]
        key_matrix = key_role_matrices[layer]
        upper = np.triu_indices(attn_matrix.shape[0], k=1)
        finite = np.isfinite(attn_matrix[upper]) & np.isfinite(key_matrix[upper])
        x_values = key_matrix[upper][finite].astype(np.float64)
        y_values = attn_matrix[upper][finite].astype(np.float64)
        left = upper[0][finite]
        right = upper[1][finite]

        if x_values.size < 4:
            continue

        valid_records = (
            (attention.observation_counts[layer] > 0)
            & (key_roles.observation_counts[layer] > 0)
        )
        key_dist = key_roles.distributions[layer]
        attn_dist = attention.distributions[layer]
        record_shuffle_r2: list[float] = []
        role_shuffle_r2: list[float] = []
        for _ in range(n_permutations):
            permuted_records = rng.permutation(key_dist)
            record_shuffle_matrix = pairwise_js(permuted_records)
            record_shuffle_r2.append(_r2_corr(record_shuffle_matrix[upper][finite], y_values))

            shuffled_roles = key_dist.copy()
            for row_idx in range(shuffled_roles.shape[0]):
                shuffled_roles[row_idx] = shuffled_roles[row_idx, rng.permutation(shuffled_roles.shape[1])]
            role_shuffle_matrix = pairwise_js(shuffled_roles)
            role_shuffle_r2.append(_r2_corr(role_shuffle_matrix[upper][finite], y_values))

        same_mask = task_arr[left] == task_arr[right]
        row = {
            "layer": float(layer),
            "in_sample_r2": _r2_corr(x_values, y_values),
            "random_pair_cv_r2": _random_pair_cv_r2(x_values, y_values, rng),
            "leave_task_out_pair_cv_r2": _leave_task_out_pair_cv_r2(
                x_values,
                y_values,
                task_arr[left],
                task_arr[right],
            ),
            "train_same_test_cross_r2": _train_test_r2(
                x_values,
                y_values,
                train_mask=same_mask,
                test_mask=~same_mask,
            ),
            "train_cross_test_same_r2": _train_test_r2(
                x_values,
                y_values,
                train_mask=~same_mask,
                test_mask=same_mask,
            ),
            "record_shuffle_r2_median": _median(record_shuffle_r2),
            "record_shuffle_r2_max": _max(record_shuffle_r2),
            "within_record_role_shuffle_r2_median": _median(role_shuffle_r2),
            "within_record_role_shuffle_r2_max": _max(role_shuffle_r2),
        }
        layer_rows.append(row)

        if system_idx is not None and bool(valid_records.any()):
            system_x = key_dist[valid_records, system_idx]
            system_y = attn_dist[valid_records, system_idx]
            system_rows.append(
                {
                    "layer": float(layer),
                    "system_share_to_attention_mass_r2": _r2_corr(system_x, system_y),
                    "system_share_to_attention_mass_random_pair_cv_r2": _random_pair_cv_r2(
                        system_x,
                        system_y,
                        rng,
                    ),
                    "system_share_mean": float(np.mean(system_x)),
                    "system_attention_mass_mean": float(np.mean(system_y)),
                }
            )

    return {
        "phase": phase,
        "layer_rows": layer_rows,
        "system_mass_layer_rows": system_rows,
        "median_in_sample_r2": _median(row["in_sample_r2"] for row in layer_rows),
        "median_random_pair_cv_r2": _median(row["random_pair_cv_r2"] for row in layer_rows),
        "median_leave_task_out_pair_cv_r2": _median(
            row["leave_task_out_pair_cv_r2"] for row in layer_rows
        ),
        "median_train_same_test_cross_r2": _median(
            row["train_same_test_cross_r2"] for row in layer_rows
        ),
        "median_train_cross_test_same_r2": _median(
            row["train_cross_test_same_r2"] for row in layer_rows
        ),
        "median_record_shuffle_r2": _median(
            row["record_shuffle_r2_median"] for row in layer_rows
        ),
        "max_record_shuffle_r2": _max(row["record_shuffle_r2_max"] for row in layer_rows),
        "median_within_record_role_shuffle_r2": _median(
            row["within_record_role_shuffle_r2_median"] for row in layer_rows
        ),
        "max_within_record_role_shuffle_r2": _max(
            row["within_record_role_shuffle_r2_max"] for row in layer_rows
        ),
        "median_system_share_to_attention_mass_r2": _median(
            row["system_share_to_attention_mass_r2"] for row in system_rows
        ),
        "median_system_share_to_attention_mass_random_pair_cv_r2": _median(
            row["system_share_to_attention_mass_random_pair_cv_r2"]
            for row in system_rows
        ),
    }


def _compute_distance_matrices(dataset: Any, pairwise_js: Any) -> dict[int, Any]:
    import numpy as np

    matrices: dict[int, Any] = {}
    n_records = len(dataset.records)
    for layer in dataset.layers:
        matrix = dataset.distributions[layer]
        obs = dataset.observation_counts[layer]
        valid = obs > 0
        distances = np.full((n_records, n_records), np.nan)
        if int(valid.sum()) >= 1:
            valid_distances = pairwise_js(matrix[valid])
            valid_indices = np.flatnonzero(valid)
            distances[np.ix_(valid_indices, valid_indices)] = valid_distances
        matrices[layer] = distances
    return matrices


def _linear_fit(x_values: Any, y_values: Any) -> tuple[float, float]:
    import numpy as np

    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    design = np.column_stack([np.ones_like(x), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(intercept), float(slope)


def _predict_r2(train_x: Any, train_y: Any, test_x: Any, test_y: Any) -> float:
    import numpy as np

    if len(train_x) < 2 or len(test_x) < 2:
        return float("nan")
    intercept, slope = _linear_fit(train_x, train_y)
    pred = intercept + slope * np.asarray(test_x, dtype=np.float64)
    y = np.asarray(test_y, dtype=np.float64)
    denom = float(np.sum((y - float(np.mean(y))) ** 2))
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((y - pred) ** 2) / denom)


def _r2_corr(x_values: Any, y_values: Any) -> float:
    import numpy as np

    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    corr = float(np.corrcoef(x, y)[0, 1])
    return corr * corr


def _random_pair_cv_r2(x_values: Any, y_values: Any, rng: Any, n_folds: int = 5) -> float:
    import numpy as np

    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < n_folds * 2:
        return float("nan")
    order = rng.permutation(x.size)
    fold_scores: list[float] = []
    for test_indices in np.array_split(order, n_folds):
        train_mask = np.ones(x.size, dtype=bool)
        train_mask[test_indices] = False
        score = _predict_r2(x[train_mask], y[train_mask], x[test_indices], y[test_indices])
        if math.isfinite(score):
            fold_scores.append(score)
    return _mean(fold_scores)


def _leave_task_out_pair_cv_r2(
    x_values: Any,
    y_values: Any,
    left_tasks: Sequence[str],
    right_tasks: Sequence[str],
) -> float:
    import numpy as np

    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    left = np.asarray(left_tasks)
    right = np.asarray(right_tasks)
    scores: list[float] = []
    for task in sorted(set(left.tolist()) | set(right.tolist())):
        test_mask = (left == task) | (right == task)
        train_mask = ~test_mask
        score = _train_test_r2(x, y, train_mask=train_mask, test_mask=test_mask)
        if math.isfinite(score):
            scores.append(score)
    return _mean(scores)


def _train_test_r2(
    x_values: Any,
    y_values: Any,
    *,
    train_mask: Any,
    test_mask: Any,
) -> float:
    import numpy as np

    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    train = np.asarray(train_mask, dtype=bool) & np.isfinite(x) & np.isfinite(y)
    test = np.asarray(test_mask, dtype=bool) & np.isfinite(x) & np.isfinite(y)
    return _predict_r2(x[train], y[train], x[test], y[test])


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
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "nan"
    return f"{number:.4f}"


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Attention R2 Leakage Sanity Check",
        "",
        f"- Records: `{summary['n_records']}` across `{summary['n_tasks']}` tasks.",
        f"- Permutations per layer/phase: `{summary['n_permutations']}`.",
        "",
        "## Feature/Target Definition",
        "",
        f"- Pairwise feature: {summary['feature_target_definition']['pairwise_feature']}.",
        f"- Pairwise target: {summary['feature_target_definition']['pairwise_target']}.",
        f"- Original R2 note: {summary['feature_target_definition']['not_cross_validated_original_r2']}.",
        "",
        "Important: a global role-label permutation leaves JS distances unchanged; "
        "the permutation baseline here uses record shuffling and independent per-record "
        "role-column shuffling to break semantic alignment.",
        "",
        "## Phase Summary",
        "",
    ]
    for phase in ("all", "prefill", "decode"):
        item = summary["phase_summaries"][phase]
        lines.extend(
            [
                f"### {phase}",
                "",
                f"- in-sample median R2: `{_fmt(item['median_in_sample_r2'])}`",
                f"- random pair CV median R2: `{_fmt(item['median_random_pair_cv_r2'])}`",
                f"- leave-task-out pair CV median R2: `{_fmt(item['median_leave_task_out_pair_cv_r2'])}`",
                f"- train same-task / test cross-task median R2: `{_fmt(item['median_train_same_test_cross_r2'])}`",
                f"- train cross-task / test same-task median R2: `{_fmt(item['median_train_cross_test_same_r2'])}`",
                f"- record-shuffle median R2: `{_fmt(item['median_record_shuffle_r2'])}`; max `{_fmt(item['max_record_shuffle_r2'])}`",
                f"- within-record role-shuffle median R2: `{_fmt(item['median_within_record_role_shuffle_r2'])}`; max `{_fmt(item['max_within_record_role_shuffle_r2'])}`",
                f"- direct system-share -> system-attention-mass median R2: `{_fmt(item['median_system_share_to_attention_mass_r2'])}`",
                f"- direct system-share random pair CV median R2: `{_fmt(item['median_system_share_to_attention_mass_random_pair_cv_r2'])}`",
                "",
            ]
        )
    lines.extend(_verdict_lines(summary))
    return "\n".join(lines)


def _verdict_lines(summary: dict[str, Any]) -> list[str]:
    prefill = summary["phase_summaries"]["prefill"]
    lines = ["## Verdict", ""]
    if prefill["median_record_shuffle_r2"] < 0.1 and prefill["median_within_record_role_shuffle_r2"] < 0.1:
        lines.append(
            "- Prefill pairwise R2 is not explained by trivial leakage that survives "
            "record/role shuffling; both permutation baselines collapse."
        )
    else:
        lines.append(
            "- Prefill permutation baselines do not fully collapse; treat the high R2 "
            "as potentially confounded until investigated further."
        )
    if prefill["median_random_pair_cv_r2"] > 0.9:
        lines.append("- Random pair CV remains high, so the linear relationship generalizes across held-out pairs.")
    else:
        lines.append("- Random pair CV does not remain high; the in-sample R2 is not stable under pair holdout.")
    if prefill["median_leave_task_out_pair_cv_r2"] > 0.8:
        lines.append("- Leave-task-out pair CV remains high, reducing concern that only task boundaries explain the fit.")
    else:
        lines.append("- Leave-task-out pair CV drops materially; task identity/boundaries may be part of the fit.")
    lines.append(
        "- Even if validated, the R2 remains a composition-control result, not an "
        "independent predictor of model behavior."
    )
    return lines


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
