"""Plot attention-vs-MoE layer specialization alignment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))

from metrics import pearson_corr, specialization_score
from recording_loader import (
    IterationRecord,
    LayerDistributionSet,
    average_layer_matrix,
    load_attention_distributions,
    load_iteration_records,
    load_moe_distributions,
    parse_layer_selection,
)


def save_alignment_scatter(
    records: Sequence[IterationRecord],
    output_path: Path,
    *,
    layers: Sequence[int] | None = None,
    phase: str = "all",
    attention: LayerDistributionSet | None = None,
    moe: LayerDistributionSet | None = None,
) -> dict[str, float]:
    """Save Plot 3: cross-modality layer specialization scatter."""
    attention = attention or load_attention_distributions(records, phase=phase)
    moe = moe or load_moe_distributions(records)
    shared_layers = sorted(set(attention.layers).intersection(moe.layers))
    if layers is not None:
        wanted = set(layers)
        shared_layers = [layer for layer in shared_layers if layer in wanted]
    if not shared_layers:
        raise ValueError("attention and MoE recordings have no shared layers")

    attn_layers, attn_matrix, attn_counts = average_layer_matrix(
        attention,
        layers=shared_layers,
        equal_iter_weight=True,
    )
    moe_layers, moe_matrix, moe_counts = average_layer_matrix(
        moe,
        layers=shared_layers,
        equal_iter_weight=True,
    )
    if attn_layers != moe_layers:
        raise RuntimeError("internal layer ordering mismatch")

    attn_scores = np.asarray([specialization_score(row) for row in attn_matrix])
    moe_scores = np.asarray([specialization_score(row) for row in moe_matrix])
    layers_arr = np.asarray(shared_layers, dtype=np.float64)
    sizes = _point_sizes(attn_counts + moe_counts)
    corr = pearson_corr(attn_scores, moe_scores)

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    scatter = ax.scatter(
        attn_scores,
        moe_scores,
        c=layers_arr,
        s=sizes,
        cmap="viridis",
        alpha=0.86,
        edgecolors="black",
        linewidths=0.35,
    )
    for layer, x, y in zip(shared_layers, attn_scores, moe_scores):
        if layer == shared_layers[0] or layer == shared_layers[-1] or layer % 8 == 0:
            ax.annotate(str(layer), (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("attention specialization score (1 - normalized role entropy)")
    ax.set_ylabel("MoE specialization score (1 - normalized expert entropy)")
    ax.set_title(
        "Plot 3: Cross-modality layer specialization alignment\n"
        f"Pearson r = {corr:.3f} across {len(shared_layers)} shared layers"
    )
    ax.grid(alpha=0.25)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("layer index")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_iters": float(len(records)),
        "n_layers": float(len(shared_layers)),
        "pearson_r": float(corr),
        "attention_score_mean": float(np.mean(attn_scores)),
        "moe_score_mean": float(np.mean(moe_scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="attempt, task, or run directories")
    parser.add_argument("--output", type=Path, required=True)
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
    layers = None
    if args.layers:
        attention = load_attention_distributions(records, phase=args.phase)
        layers = parse_layer_selection(args.layers, attention.layers)
    summary = save_alignment_scatter(records, args.output, layers=layers, phase=args.phase)
    print(f"wrote {args.output}")
    print(summary)


def _point_sizes(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or float(np.nanmax(arr)) <= 0.0:
        return np.full(arr.shape, 60.0)
    scaled = np.sqrt(arr / float(np.nanmax(arr)))
    return 50.0 + 230.0 * scaled


if __name__ == "__main__":
    main()
