"""Plot layer specialization maps for attention roles and MoE experts."""

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

from recording_loader import (
    IterationRecord,
    LayerDistributionSet,
    average_layer_matrix,
    load_attention_distributions,
    load_iteration_records,
    load_moe_distributions,
    parse_layer_selection,
)


def save_layer_specialization_figure(
    records: Sequence[IterationRecord],
    output_path: Path,
    *,
    layers: Sequence[int] | None = None,
    phase: str = "all",
    top_experts: int = 64,
    attention: LayerDistributionSet | None = None,
    moe: LayerDistributionSet | None = None,
) -> dict[str, float]:
    """Save Plot 2: attention and MoE specialization side by side."""
    attention = attention or load_attention_distributions(records, phase=phase)
    moe = moe or load_moe_distributions(records)
    attn_layers = _select_layers(attention.layers, layers)
    moe_layers = _select_layers(moe.layers, layers)

    attn_layers, attn_matrix, _attn_counts = average_layer_matrix(
        attention,
        layers=attn_layers,
        equal_iter_weight=True,
    )
    moe_layers, moe_matrix, _moe_counts = average_layer_matrix(
        moe,
        layers=moe_layers,
        equal_iter_weight=True,
    )
    expert_order = np.argsort(moe_matrix.sum(axis=0))[::-1]
    expert_order = expert_order[: min(top_experts, expert_order.size)]
    moe_sorted = moe_matrix[:, expert_order]
    expert_labels = [moe.axis_labels[idx] for idx in expert_order]

    rank_width = min(top_experts, moe_matrix.shape[1])
    moe_rank = -np.sort(-moe_matrix, axis=1)[:, :rank_width]

    fig_width = 19.0
    fig_height = max(6.0, 0.22 * max(len(attn_layers), len(moe_layers)))
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(fig_width, fig_height),
        gridspec_kw={"width_ratios": [1.0, 1.35, 2.2]},
    )

    attn_image = axes[0].imshow(attn_matrix, aspect="auto", cmap="viridis", vmin=0.0)
    axes[0].set_title("Attention mass by segment role")
    axes[0].set_xlabel("segment role")
    axes[0].set_ylabel("layer")
    axes[0].set_xticks(range(len(attention.axis_labels)))
    axes[0].set_xticklabels(attention.axis_labels, rotation=45, ha="right")
    axes[0].set_yticks(range(len(attn_layers)))
    axes[0].set_yticklabels([str(layer) for layer in attn_layers])
    fig.colorbar(attn_image, ax=axes[0], fraction=0.046, pad=0.02)

    rank_vmax = float(np.nanpercentile(moe_rank, 99.5)) if moe_rank.size else 1.0
    rank_image = axes[1].imshow(
        moe_rank,
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=rank_vmax if rank_vmax > 0 else None,
    )
    axes[1].set_title(f"MoE load by per-layer expert rank (top {rank_width})")
    axes[1].set_xlabel("expert rank within layer")
    axes[1].set_yticks(range(len(moe_layers)))
    axes[1].set_yticklabels([str(layer) for layer in moe_layers])
    rank_tick_count = min(12, rank_width)
    if rank_tick_count:
        rank_ticks = np.linspace(0, rank_width - 1, rank_tick_count, dtype=int)
        axes[1].set_xticks(rank_ticks)
        axes[1].set_xticklabels([str(idx + 1) for idx in rank_ticks], rotation=90)
    fig.colorbar(rank_image, ax=axes[1], fraction=0.035, pad=0.02)

    identity_vmax = float(np.nanpercentile(moe_sorted, 99.5)) if moe_sorted.size else 1.0
    moe_image = axes[2].imshow(
        moe_sorted,
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=identity_vmax if identity_vmax > 0 else None,
    )
    axes[2].set_title(f"MoE expert identity by layer (top {len(expert_order)} experts)")
    axes[2].set_xlabel("experts sorted by global load")
    axes[2].set_yticks(range(len(moe_layers)))
    axes[2].set_yticklabels([str(layer) for layer in moe_layers])
    tick_count = min(16, len(expert_labels))
    if tick_count:
        tick_positions = np.linspace(0, len(expert_labels) - 1, tick_count, dtype=int)
        axes[2].set_xticks(tick_positions)
        axes[2].set_xticklabels([expert_labels[idx] for idx in tick_positions], rotation=90)
    fig.colorbar(moe_image, ax=axes[2], fraction=0.025, pad=0.02)

    fig.suptitle(
        "Plot 2: Layer specialization map\n"
        "Left shows role-level attention; middle shows per-layer MoE concentration; "
        "right preserves expert identity.",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_iters": float(len(records)),
        "n_attention_layers": float(len(attn_layers)),
        "n_moe_layers": float(len(moe_layers)),
        "n_roles": float(len(attention.axis_labels)),
        "n_ranked_experts": float(rank_width),
        "n_plotted_experts": float(len(expert_order)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="attempt, task, or run directories")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", help="comma/range layer selector, e.g. 0,8,16-20")
    parser.add_argument("--phase", choices=("all", "prefill", "decode"), default="all")
    parser.add_argument("--top-experts", type=int, default=64)
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
    summary = save_layer_specialization_figure(
        records,
        args.output,
        layers=layers,
        phase=args.phase,
        top_experts=args.top_experts,
    )
    print(f"wrote {args.output}")
    print(summary)


def _select_layers(available_layers: Sequence[int], layers: Sequence[int] | None) -> list[int]:
    if layers is None:
        return list(available_layers)
    available = set(available_layers)
    return [layer for layer in layers if layer in available]


if __name__ == "__main__":
    main()
