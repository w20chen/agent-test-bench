"""Generate the core recording figures from existing artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))

from plot_alignment_scatter import save_alignment_scatter
from plot_iter_distance import build_iter_distance_figures
from plot_layer_specialization import save_layer_specialization_figure
from recording_loader import (
    load_attention_distributions,
    load_iteration_records,
    load_moe_distributions,
    parse_layer_selection,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="attempt, task, or run directories")
    parser.add_argument("--output-dir", type=Path, required=True)
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
    attention = None
    if args.layers:
        attention = load_attention_distributions(records, phase=args.phase)
        layers = parse_layer_selection(args.layers, attention.layers)
    if attention is None:
        attention = load_attention_distributions(records, phase=args.phase)
    moe = load_moe_distributions(records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    summaries["plot1"] = build_iter_distance_figures(
        records,
        args.output_dir,
        layers=layers,
        phase=args.phase,
        attention=attention,
        moe=moe,
    )
    summaries["plot2"] = save_layer_specialization_figure(
        records,
        args.output_dir / "plot2_layer_specialization.pdf",
        layers=layers,
        phase=args.phase,
        top_experts=args.top_experts,
        attention=attention,
        moe=moe,
    )
    summaries["plot3"] = save_alignment_scatter(
        records,
        args.output_dir / "plot3_alignment_scatter.pdf",
        layers=layers,
        phase=args.phase,
        attention=attention,
        moe=moe,
    )
    summaries["inputs"] = [str(path) for path in args.inputs]
    summaries["n_records"] = len(records)
    summaries["phase"] = args.phase

    summary_path = args.output_dir / "figure_summary.json"
    safe_summaries = _json_ready(summaries)
    summary_path.write_text(json.dumps(safe_summaries, indent=2, allow_nan=False) + "\n")
    md_path = args.output_dir / "figure_summary.md"
    md_path.write_text(_summary_markdown(safe_summaries), encoding="utf-8")
    print(f"wrote {args.output_dir}")
    print(f"summary: {summary_path}")


def _summary_markdown(summaries: dict[str, Any]) -> str:
    plot1_attention = summaries["plot1"]["attention"]
    plot1_moe = summaries["plot1"]["moe"]
    plot3 = summaries["plot3"]
    lines = [
        "# Recording Figure Summary",
        "",
        f"- Inputs: `{', '.join(summaries['inputs'])}`",
        f"- Records: `{summaries['n_records']}`; attention phase: `{summaries['phase']}`",
        "",
        "## Plot 1",
        (
            "- Attention iter-distance mean JS: "
            f"`{_fmt(plot1_attention['mean_pairwise_js'])}`; adjacent mean JS: "
            f"`{_fmt(plot1_attention['mean_adjacent_js'])}`."
        ),
        (
            "- MoE iter-distance mean JS: "
            f"`{_fmt(plot1_moe['mean_pairwise_js'])}`; adjacent mean JS: "
            f"`{_fmt(plot1_moe['mean_adjacent_js'])}`."
        ),
        "",
        "## Plot 2",
        (
            "- Layer specialization maps compare role-level attention mass against "
            "globally frequent MoE experts. Inspect whether concentrated attention "
            "layers line up with concentrated routing layers."
        ),
        "",
        "## Plot 3",
        (
            "- Cross-modality specialization Pearson r: "
            f"`{_fmt(plot3['pearson_r'])}` across `{plot3['n_layers']:.0f}` layers."
        ),
        "",
    ]
    return "\n".join(lines)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            return None
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
