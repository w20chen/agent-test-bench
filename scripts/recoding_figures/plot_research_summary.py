"""Plot durable summary figures from agent attention/MoE research JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageOps


COLORS = {
    "all": "#6f7f8f",
    "prefill": "#4c78a8",
    "decode": "#b2796f",
    "global": "#8c8c8c",
    "layer": "#4c78a8",
    "adjacent": "#5f9e6e",
    "residual": "#b2796f",
    "share": "#b8b8b8",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    apply_academic_style()

    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pngs = [
        plot_phase_residual_control(summary, args.output_dir),
        plot_role_kv_value(summary, args.output_dir),
        plot_moe_cache_coverage(summary, args.output_dir),
        plot_attention_moe_coupling(summary, args.output_dir),
    ]
    make_contact_sheet(pngs, args.output_dir / "figure_contact_sheet.png")
    for path in pngs:
        print(path)
    print(args.output_dir / "figure_contact_sheet.png")


def plot_phase_residual_control(summary: dict[str, Any], output_dir: Path) -> Path:
    phases = ["all", "prefill", "decode"]
    residuals = summary["phase1_measurement_residuals"]
    r2_values = [
        residuals[phase]["median_r2_attention_explained_by_visible_key_role_js"]
        for phase in phases
    ]
    residual_values = [residuals[phase]["mean_abs_residual"] for phase in phases]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    axes[0].bar(phases, r2_values, color=[COLORS[phase] for phase in phases], width=0.62)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("median R2")
    axes[0].set_title("Visible-key role explains prefill")
    axes[0].grid(axis="y", alpha=0.25)
    for idx, value in enumerate(r2_values):
        axes[0].text(idx, value + 0.025, f"{value:.2f}", ha="center", fontsize=8)

    axes[1].bar(
        phases,
        residual_values,
        color=[COLORS[phase] for phase in phases],
        width=0.62,
    )
    axes[1].set_ylabel("mean abs residual JS")
    axes[1].set_title("Decode keeps residual structure")
    axes[1].grid(axis="y", alpha=0.25)
    for idx, value in enumerate(residual_values):
        axes[1].text(idx, value + 0.0015, f"{value:.3f}", ha="center", fontsize=8)

    for ax in axes:
        _clean_axes(ax)
    fig.suptitle("Attention distance after visible-key role control", y=1.05, fontsize=12)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "fig1_phase_residual_control")


def plot_role_kv_value(summary: dict[str, Any], output_dir: Path) -> Path:
    roles = summary["role_labels"]
    phase2 = summary["phase2_role_kv_value"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    for ax, phase in zip(axes, ["prefill", "decode"], strict=True):
        rows = {row["role"]: row for row in phase2[phase]["role_rows"]}
        y_positions = np.arange(len(roles))
        key_share = np.asarray([rows[role]["row_weighted_key_role_share"] for role in roles])
        attn_mass = np.asarray(
            [rows[role]["row_weighted_attention_mass"] for role in roles]
        )
        ax.barh(
            y_positions + 0.18,
            key_share,
            height=0.32,
            color=COLORS["share"],
            label="visible-key share",
        )
        ax.barh(
            y_positions - 0.18,
            attn_mass,
            height=0.32,
            color=COLORS[phase],
            label="attention mass",
        )
        ax.set_yticks(y_positions)
        ax.set_yticklabels(roles)
        ax.invert_yaxis()
        ax.set_xlim(0.0, max(0.72, float(max(key_share.max(), attn_mass.max())) * 1.15))
        ax.set_xlabel("fraction")
        ax.set_title(phase)
        ax.grid(axis="x", alpha=0.25)
        _clean_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle("Role-level KV value: attention mass vs visible key share", y=1.08)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    return _save_figure(fig, output_dir, "fig2_role_kv_value_prefill_decode")


def plot_moe_cache_coverage(summary: dict[str, Any], output_dir: Path) -> Path:
    rows = summary["phase3_moe_cacheability"]["coverage_rows"]
    ks = np.asarray([int(row["k"]) for row in rows])
    global_cov = np.asarray([row["static_global_coverage"] for row in rows])
    layer_cov = np.asarray([row["static_layer_coverage"] for row in rows])
    adjacent_cov = np.asarray([row["adjacent_prev_iter_coverage"] for row in rows])

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.plot(ks, global_cov, marker="o", color=COLORS["global"], label="global static")
    ax.plot(ks, layer_cov, marker="o", color=COLORS["layer"], label="layer static")
    ax.plot(
        ks,
        adjacent_cov,
        marker="o",
        color=COLORS["adjacent"],
        label="previous iter",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("cached experts per layer budget k")
    ax.set_ylabel("expert-load coverage")
    ax.set_title("Layer-aware and adjacent MoE caches beat global hotsets")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    _clean_axes(ax)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "fig3_moe_cache_coverage")


def plot_attention_moe_coupling(summary: dict[str, Any], output_dir: Path) -> Path:
    phases = ["all", "prefill", "decode"]
    coupling = summary["phase4_attention_moe_coupling"]["phase_summary"]
    raw = np.asarray([coupling[phase]["mean_corr_attention_js_vs_moe_js"] for phase in phases])
    residual = np.asarray(
        [coupling[phase]["mean_corr_attention_residual_vs_moe_js"] for phase in phases]
    )
    x_positions = np.arange(len(phases))
    width = 0.34
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    ax.bar(
        x_positions - width / 2,
        raw,
        width=width,
        color=COLORS["layer"],
        label="raw attention JS",
    )
    ax.bar(
        x_positions + width / 2,
        residual,
        width=width,
        color=COLORS["residual"],
        label="visible-key residual",
    )
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(phases)
    ax.set_ylabel("mean Pearson r with all-token MoE JS")
    ax.set_title("Attention distance is not a reliable MoE proxy")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    _clean_axes(ax)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "fig4_attention_moe_coupling")


def make_contact_sheet(pngs: list[Path], output_path: Path) -> None:
    thumbs: list[tuple[str, Image.Image]] = []
    for path in pngs:
        image = Image.open(path).convert("RGB")
        image.thumbnail((640, 420), Image.Resampling.LANCZOS)
        thumbs.append((path.name, ImageOps.expand(image, border=12, fill="white")))

    cell_width = max(image.width for _name, image in thumbs)
    cell_height = max(image.height for _name, image in thumbs) + 28
    sheet = Image.new("RGB", (cell_width * 2, cell_height * 2), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (name, image) in enumerate(thumbs):
        row, col = divmod(idx, 2)
        x_offset = col * cell_width + (cell_width - image.width) // 2
        y_offset = row * cell_height + 22
        sheet.paste(image, (x_offset, y_offset))
        draw.text((col * cell_width + 12, row * cell_height + 4), name, fill=(40, 40, 40))
    sheet.save(output_path)


def apply_academic_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D8D1C7",
            "axes.labelcolor": "#4B5563",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.color": "#6B7280",
            "ytick.color": "#6B7280",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "grid.color": "#E7E5E4",
            "grid.linewidth": 0.8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> Path:
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return png_path


def _clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


if __name__ == "__main__":
    main()
