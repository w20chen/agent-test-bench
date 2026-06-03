"""Plot pairwise iteration JS-distance matrices for recordings."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))

from metrics import pairwise_js
from recording_loader import (
    LayerDistributionSet,
    IterationRecord,
    load_attention_distributions,
    load_iteration_records,
    load_moe_distributions,
    parse_layer_selection,
    task_boundaries,
)


def save_iter_distance_figure(
    dataset: LayerDistributionSet,
    output_path: Path,
    *,
    layers: Sequence[int] | None = None,
    max_cols: int = 8,
) -> dict[str, float | None]:
    """Save faceted layer-wise iteration distance matrices."""
    matrices, finite_values = compute_iter_distance_matrices(dataset, layers=layers)
    selected_layers = list(layers or dataset.layers)
    if not selected_layers:
        raise ValueError("no layers selected for distance plot")
    n_layers = len(selected_layers)
    n_cols = min(max_cols, n_layers)
    n_rows = int(math.ceil(n_layers / n_cols))
    fig_width = max(8.0, 2.2 * n_cols)
    fig_height = max(5.5, 2.2 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)

    vmax = float(np.nanpercentile(finite_values, 99)) if finite_values else 1.0
    if vmax <= 0:
        vmax = 1.0
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#eeeeee")

    for ax, layer in zip(axes.ravel(), selected_layers):
        image = ax.imshow(matrices[layer], cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest")
        ax.set_title(f"L{layer}", fontsize=9)
        _decorate_task_boundaries(ax, dataset.records)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[n_layers:]:
        ax.axis("off")

    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.018, pad=0.01)
    colorbar.set_label("JS divergence (bits)")
    title = (
        f"Plot 1: {dataset.modality} pairwise iter distance by layer\n"
        "Low values imply steady behavior; block structure implies phase-like behavior; gradients imply drift."
    )
    fig.suptitle(title, fontsize=12)
    fig.text(0.01, 0.01, _task_caption(dataset.records), fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return _distance_summary(dataset, matrices)


def compute_iter_distance_matrices(
    dataset: LayerDistributionSet,
    *,
    layers: Sequence[int] | None = None,
) -> tuple[dict[int, np.ndarray], list[float]]:
    """Compute layer-wise pairwise JS matrices."""
    selected_layers = list(layers or dataset.layers)
    finite_values: list[float] = []
    matrices: dict[int, np.ndarray] = {}
    n_records = len(dataset.records)
    for layer in selected_layers:
        matrix = dataset.distributions[layer]
        obs = dataset.observation_counts[layer]
        valid = obs > 0
        distances = np.full((n_records, n_records), np.nan)
        if int(valid.sum()) >= 1:
            valid_distances = pairwise_js(matrix[valid])
            valid_indices = np.flatnonzero(valid)
            distances[np.ix_(valid_indices, valid_indices)] = valid_distances
            finite_values.extend(
                float(value) for value in distances[np.isfinite(distances)].ravel()
            )
        matrices[layer] = distances
    return matrices, finite_values


def build_iter_distance_figures(
    records: Sequence[IterationRecord],
    output_dir: Path,
    *,
    layers: Sequence[int] | None = None,
    phase: str = "all",
    attention: LayerDistributionSet | None = None,
    moe: LayerDistributionSet | None = None,
) -> dict[str, dict[str, float | None]]:
    """Build attention and MoE Plot 1 figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    attention = attention or load_attention_distributions(records, phase=phase)
    moe = moe or load_moe_distributions(records)
    attn_layers = _resolve_layers(attention, layers)
    moe_layers = _resolve_layers(moe, layers)
    summaries = {
        "attention": save_iter_distance_figure(
            attention,
            output_dir / "plot1_iter_distance_attention.pdf",
            layers=attn_layers,
        ),
        "moe": save_iter_distance_figure(
            moe,
            output_dir / "plot1_iter_distance_moe.pdf",
            layers=moe_layers,
        ),
    }
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="attempt, task, or run directories")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", help="comma/range layer selector, e.g. 0,8,16-20")
    parser.add_argument("--phase", choices=("all", "prefill", "decode"), default="all")
    parser.add_argument("--include-orphans", action="store_true")
    parser.add_argument("--max-iters", type=int)
    args = parser.parse_args()

    records = load_iteration_records(
        args.inputs,
        include_orphans=args.include_orphans,
        max_iters=args.max_iters,
    )
    attention = load_attention_distributions(records, phase=args.phase)
    moe = load_moe_distributions(records)
    layer_selection = parse_layer_selection(args.layers, attention.layers) if args.layers else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    attn_summary = save_iter_distance_figure(
        attention,
        args.output_dir / "plot1_iter_distance_attention.pdf",
        layers=layer_selection,
    )
    moe_layers = parse_layer_selection(args.layers, moe.layers) if args.layers else None
    moe_summary = save_iter_distance_figure(
        moe,
        args.output_dir / "plot1_iter_distance_moe.pdf",
        layers=moe_layers,
    )
    print(f"wrote {args.output_dir / 'plot1_iter_distance_attention.pdf'}")
    print(f"wrote {args.output_dir / 'plot1_iter_distance_moe.pdf'}")
    print({"attention": attn_summary, "moe": moe_summary})


def _resolve_layers(
    dataset: LayerDistributionSet, layers: Sequence[int] | None
) -> list[int] | None:
    if layers is None:
        return None
    return [layer for layer in layers if layer in set(dataset.layers)]


def _decorate_task_boundaries(ax: plt.Axes, records: Sequence[IterationRecord]) -> None:
    for start, _task in task_boundaries(records)[1:]:
        pos = start - 0.5
        ax.axvline(pos, color="white", linewidth=0.6, alpha=0.8)
        ax.axhline(pos, color="white", linewidth=0.6, alpha=0.8)


def _task_caption(records: Sequence[IterationRecord]) -> str:
    chunks = [f"{idx}:{task}" for idx, task in task_boundaries(records)]
    return "Task starts: " + ", ".join(chunks)


def _distance_summary(
    dataset: LayerDistributionSet, matrices: dict[int, np.ndarray]
) -> dict[str, float | None]:
    all_values: list[float] = []
    adjacent_values: list[float] = []
    for matrix in matrices.values():
        upper = matrix[np.triu_indices(matrix.shape[0], k=1)]
        finite = upper[np.isfinite(upper)]
        all_values.extend(float(value) for value in finite)
        for idx in range(matrix.shape[0] - 1):
            value = matrix[idx, idx + 1]
            if np.isfinite(value):
                adjacent_values.append(float(value))
    mean_js = float(np.mean(all_values)) if all_values else None
    adjacent_js = float(np.mean(adjacent_values)) if adjacent_values else None
    return {
        "n_iters": float(len(dataset.records)),
        "n_layers": float(len(matrices)),
        "mean_pairwise_js": mean_js,
        "mean_adjacent_js": adjacent_js,
    }


if __name__ == "__main__":
    main()
