"""Modal residual-explanation exploration for agent attention recordings."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import modal


APP_NAME = "asb-agent-attention-residual-exploration"
VOLUME_NAME = "asb-terminal-recordings"
VOLUME_ROOT = Path("/data")
EXTRACT_DIR = VOLUME_ROOT / "extracted" / "curated14"
OUTPUT_DIR = VOLUME_ROOT / "outputs" / "agent_attention_residual_exploration_20260510"

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
def run_residual_exploration() -> dict[str, Any]:
    """Run residual-explanation feature leaderboard over full curated-14."""
    sys.path.insert(0, "/opt/recoding_figures")

    from followup_metrics import (
        distribution_component_leaderboard,
        load_attention_context_group_distributions,
        load_attention_decode_step_distributions,
        load_attention_distance_bucket_distributions,
        load_attention_head_role_distributions,
        load_attention_rank_bucket_distributions,
        load_attention_segment_recency_distributions,
        record_pair_feature_matrices,
        record_scalar_feature_arrays,
        residual_explanation_leaderboard,
    )
    from recording_loader import (
        collect_role_labels,
        load_attention_distributions,
        load_attention_key_role_distributions,
        load_iteration_records,
        load_moe_distributions,
    )

    attempts = _attempt_dirs()
    if not attempts:
        raise FileNotFoundError(f"no extracted attempts under {EXTRACT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_iteration_records(attempts)
    role_labels = collect_role_labels(records)

    attention_decode = load_attention_distributions(
        records,
        role_labels=role_labels,
        phase="decode",
    )
    key_role_decode = load_attention_key_role_distributions(
        records,
        role_labels=role_labels,
        phase="decode",
    )
    distribution_features = {
        "attention_distance_buckets": load_attention_distance_bucket_distributions(
            records,
            phase="decode",
        ),
        "attention_context_groups": load_attention_context_group_distributions(
            records,
            role_labels=role_labels,
            phase="decode",
            recent_token_window=256,
        ),
        "attention_segment_recency": load_attention_segment_recency_distributions(
            records,
            role_labels=role_labels,
            phase="decode",
        ),
        "attention_decode_step": load_attention_decode_step_distributions(
            records,
            role_labels=role_labels,
            phase="decode",
        ),
        "attention_rank_buckets": load_attention_rank_bucket_distributions(
            records,
            phase="decode",
        ),
        "attention_head_role_profile": load_attention_head_role_distributions(
            records,
            role_labels=role_labels,
            phase="decode",
        ),
        "moe_decode_expert_distribution": load_moe_distributions(records, phase="decode"),
    }
    scalar_features = record_scalar_feature_arrays(records, role_labels=role_labels)
    pair_features = record_pair_feature_matrices(records)

    leaderboard = residual_explanation_leaderboard(
        attention_decode,
        key_role_decode,
        distribution_features=distribution_features,
        scalar_features=scalar_features,
        pair_features=pair_features,
    )
    component_leaderboards = {
        name: distribution_component_leaderboard(
            attention_decode,
            key_role_decode,
            distribution_features[name],
        )
        for name in (
            "attention_context_groups",
            "attention_segment_recency",
            "attention_head_role_profile",
        )
    }
    feature_metadata = {
        name: {
            "modality": dataset.modality,
            "n_layers": float(len(dataset.layers)),
            "n_axis_labels": float(len(dataset.axis_labels)),
            "axis_labels": (
                dataset.axis_labels
                if len(dataset.axis_labels) <= 24
                else dataset.axis_labels[:24] + ["..."]
            ),
        }
        for name, dataset in distribution_features.items()
    }
    summary = {
        "n_records": len(records),
        "n_tasks": len({record.task for record in records}),
        "role_labels": role_labels,
        "candidate_angles": [
            "task/conversation progress",
            "decode step shape",
            "segment/message recency",
            "attention concentration/rank shape",
            "layer/head role profile",
            "MoE decode expert coupling",
            "evidence volume/noise",
            "unobserved lexical semantics schema gap",
        ],
        "feature_metadata": feature_metadata,
        "leaderboard": leaderboard,
        "component_leaderboards": component_leaderboards,
        "schema_gap": {
            "lexical_token_semantics_available": False,
            "reason": (
                "attention artifacts expose query positions, segment roles, "
                "attention top-k indices/weights, and query_heads, but do not "
                "store token ids or token text."
            ),
        },
    }
    clean_summary = _json_ready(summary)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(clean_summary, indent=2) + "\n")
    (OUTPUT_DIR / "summary.md").write_text(_summary_markdown(clean_summary), encoding="utf-8")

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
def main(background: bool = False) -> None:
    """Run the residual explanation exploration."""
    if background:
        call = run_residual_exploration.spawn()
        print(f"spawned residual exploration: {call.object_id}")
        print(call.get_dashboard_url())
        return
    result = run_residual_exploration.remote()
    summary = result["summary"]
    payload = {
        "output_dir": result["output_dir"],
        "output_tar": result["output_tar"],
        "output_tar_bytes": result["output_tar_bytes"],
        "top_features": summary["leaderboard"]["rows"][:10],
        "top_context_components": summary["component_leaderboards"][
            "attention_context_groups"
        ]["rows"][:8],
        "top_segment_components": summary["component_leaderboards"][
            "attention_segment_recency"
        ]["rows"][:8],
    }
    print(json.dumps(payload, indent=2))


def _attempt_dirs() -> list[Path]:
    return sorted(EXTRACT_DIR.glob("*/attempt_1"))


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _summary_markdown(summary: dict[str, Any]) -> str:
    leaderboard = summary["leaderboard"]
    components = summary.get("component_leaderboards", {})
    rows = leaderboard["rows"][:15]
    lines = [
        "# Agent Attention Residual Explanation Exploration",
        "",
        f"- Records: `{summary['n_records']}` across `{summary['n_tasks']}` tasks.",
        "- Residual: decode attention-role JS after linear control for visible-key role JS.",
        f"- Mean absolute residual: `{_fmt(leaderboard['residual_baseline']['mean_abs_residual'])}`.",
        "",
        "## Candidate Angles",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["candidate_angles"])
    lines.extend(
        [
            "",
            "## Feature Leaderboard",
            "",
            "| Rank | Feature | Mean abs corr | Mean corr | Layers | Direction |",
            "| ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for rank, row in enumerate(rows, start=1):
        direction = _direction(row)
        lines.append(
            "| "
            f"{rank} | `{row['feature']}` | "
            f"{_fmt(row['mean_abs_corr'])} | {_fmt(row['mean_corr'])} | "
            f"{_fmt(row['n_layers_scored'])} | {direction} |"
        )
    for title, key in (
        ("Context Components", "attention_context_groups"),
        ("Segment Recency Components", "attention_segment_recency"),
        ("Head-Role Components", "attention_head_role_profile"),
    ):
        if key not in components:
            continue
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Rank | Component | Mean abs corr | Mean corr | Layers |",
                "| ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for rank, row in enumerate(components[key]["rows"][:12], start=1):
            lines.append(
                "| "
                f"{rank} | `{row['feature']}` | {_fmt(row['mean_abs_corr'])} | "
                f"{_fmt(row['mean_corr'])} | {_fmt(row['n_layers_scored'])} |"
            )
    lines.extend(
        [
            "",
            "## Schema Gap",
            "",
            "- Lexical token semantics are unavailable in the current artifacts; "
            "token ids and token text are not recorded.",
            "",
        ]
    )
    return "\n".join(lines)


def _direction(row: dict[str, Any]) -> str:
    positive = float(row.get("positive_layer_fraction") or 0.0)
    negative = float(row.get("negative_layer_fraction") or 0.0)
    if positive >= 0.75:
        return "mostly positive"
    if negative >= 0.75:
        return "mostly negative"
    return "mixed"


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "nan"
    return f"{number:.4f}"


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
