"""Modal follow-up analyses B1-B4 for agent attention/MoE recordings."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import modal


APP_NAME = "asb-agent-attention-followup"
VOLUME_NAME = "asb-terminal-recordings"
VOLUME_ROOT = Path("/data")
EXTRACT_DIR = VOLUME_ROOT / "extracted" / "curated14"
OUTPUT_DIR = VOLUME_ROOT / "outputs" / "agent_attention_followup_20260510"

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
def run_followup() -> dict[str, Any]:
    """Run B1-B4 offline follow-up analyses over full curated-14."""
    sys.path.insert(0, "/opt/recoding_figures")

    from followup_metrics import (
        alpha_blend_summary,
        context_role_cache_summary,
        decode_residual_closure_summary,
        load_attention_context_group_distributions,
        load_attention_distance_bucket_distributions,
        load_attention_query_role_distributions,
        routed_token_role_load_audit,
        sliding_window_detection_summary,
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

    attention = {
        phase: load_attention_distributions(records, role_labels=role_labels, phase=phase)
        for phase in ("all", "prefill", "decode")
    }
    key_role_decode = load_attention_key_role_distributions(
        records,
        role_labels=role_labels,
        phase="decode",
    )
    moe = {
        phase: load_moe_distributions(records, phase=phase)
        for phase in ("all", "prefill", "decode")
    }

    b1 = {
        "attention_all": sliding_window_detection_summary(attention["all"]),
        "attention_prefill": sliding_window_detection_summary(attention["prefill"]),
        "attention_decode": sliding_window_detection_summary(attention["decode"]),
        "moe_all": sliding_window_detection_summary(moe["all"]),
        "moe_prefill": sliding_window_detection_summary(moe["prefill"]),
        "moe_decode": sliding_window_detection_summary(moe["decode"]),
    }

    b2 = {
        "decode_context_role_cache": context_role_cache_summary(
            moe["decode"],
            load_attention_context_group_distributions(
                records,
                role_labels=role_labels,
                phase="decode",
                recent_token_window=256,
            ),
            role_groups={
                "system": ("system",),
                "tool": ("tool",),
                "recent_gen": ("recent_gen",),
            },
        ),
        "decode_routed_token_role_audit": routed_token_role_load_audit(
            records,
            role_labels,
            phase="decode",
        ),
    }

    b3 = {
        "prefill_alpha_blend": alpha_blend_summary(moe["prefill"]),
    }

    distance_decode = load_attention_distance_bucket_distributions(records, phase="decode")
    query_role_decode = load_attention_query_role_distributions(
        records,
        role_labels=role_labels,
        phase="decode",
    )
    b4 = {
        "decode_residual_closure": decode_residual_closure_summary(
            attention["decode"],
            key_role_decode,
            distance_decode,
            query_role_decode,
        ),
    }

    summary = {
        "n_records": len(records),
        "n_tasks": len({record.task for record in records}),
        "role_labels": role_labels,
        "b1_task_change_detection": b1,
        "b2_decode_role_aware_cache": b2,
        "b3_prefill_alpha_sensitivity": b3,
        "b4_a1_decode_residual_closure": b4,
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
    """Run the B1-B4 follow-up analysis."""
    if background:
        call = run_followup.spawn()
        print(f"spawned followup: {call.object_id}")
        print(call.get_dashboard_url())
        return
    result = run_followup.remote()
    print(json.dumps(result["summary"], indent=2))


def _attempt_dirs() -> list[Path]:
    return sorted(EXTRACT_DIR.glob("*/attempt_1"))


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _summary_markdown(summary: dict[str, Any]) -> str:
    b1 = summary["b1_task_change_detection"]
    b2 = summary["b2_decode_role_aware_cache"]
    b3 = summary["b3_prefill_alpha_sensitivity"]
    b4 = summary["b4_a1_decode_residual_closure"]

    lines = [
        "# Agent Attention Follow-up B1-B4",
        "",
        f"- Records: `{summary['n_records']}` across `{summary['n_tasks']}` tasks.",
        "- Offline analysis only: no inference rerun and no benchmark code changes.",
        "",
        "## B1 - Task-Change Detection",
        "",
    ]
    for name, item in b1.items():
        row = _row_for_window(item["rows"], 4)
        lines.append(
            "- "
            f"{name}: top-budget detection `{_fmt(row['rank_detection_rate'])}`, "
            f"precision `{_fmt(row['rank_precision'])}`, "
            f"rolling-MAD detection `{_fmt(row['rolling_mad_detection_rate'])}`."
        )
    lines.extend(
        [
            "",
            "## B2 - Decode Role-Aware Expert Hotsets",
            "",
        ]
    )
    b2_row = _row_for_k(b2["decode_context_role_cache"]["rows"], 32)
    lines.append(
        "- Top-32 layer-static decode coverage "
        f"`{_fmt(b2_row['layer_static_coverage'])}`; dominant-context role-aware "
        f"`{_fmt(b2_row['dominant_context_role_coverage'])}`; attention-mixture "
        f"`{_fmt(b2_row['attention_mixture_role_coverage'])}`."
    )
    lines.append(
        "- Routed-token role shares: "
        + ", ".join(
            f"{role}={_fmt(value)}"
            for role, value in b2["decode_routed_token_role_audit"]["group_share"].items()
        )
        + "."
    )
    routed_support = b2["decode_routed_token_role_audit"][
        "routed_token_system_tool_hotsets_supported"
    ]
    lines.append(
        "- Routed-token system/tool expert hotsets supported by current schema: "
        f"`{routed_support}`; system+tool routed-token share "
        f"`{_fmt(b2['decode_routed_token_role_audit']['routed_token_system_tool_share'])}`."
    )

    lines.extend(["", "## B3 - Prefill Alpha Sensitivity", ""])
    alpha_rows = [
        row
        for row in b3["prefill_alpha_blend"]["rows"]
        if int(row["k"]) == 32
    ]
    for row in alpha_rows:
        lines.append(
            "- "
            f"alpha `{_fmt(row['alpha'])}`: overall `{_fmt(row['overall_coverage'])}`, "
            f"same-task `{_fmt(row['same_task_coverage'])}`, synthetic cross-splice "
            f"`{_fmt(row['synthetic_cross_task_splice_coverage'])}`."
        )

    closure = b4["decode_residual_closure"]
    query = closure["query_token_semantic_type"]
    lines.extend(
        [
            "",
            "## B4 - A1 Decode Residual Closure",
            "",
            "- Distance-decay correlation with abs decode residual: "
            f"`{_fmt(closure['distance_decay']['mean_corr_abs_residual_vs_distance_js'])}`.",
            "- Query semantic type availability: "
            f"`{query['available_semantics']}`; lexical token semantics available: "
            f"`{query['lexical_token_semantics_available']}`.",
            "- Dominant decode query role: "
            f"`{query['dominant_role']}` with share `{_fmt(query['dominant_role_share'])}`.",
            "- Query-role variation can explain residual: "
            f"`{query['query_role_variation_can_explain_residual']}`.",
            "",
            "## Guardrails",
            "",
            "- B1 boundaries are synthetic task-order splices, not real chronological task switches.",
            "- B2 role-aware hotsets are attention-context diagnostics; routed-token role audit is reported separately.",
            "- B4 query token semantic type is limited by current recording schema to segment roles.",
            "",
        ]
    )
    return "\n".join(lines)


def _row_for_window(rows: list[dict[str, Any]], window: int) -> dict[str, Any]:
    for row in rows:
        if int(row["window"]) == window:
            return row
    raise KeyError(f"missing window {window}")


def _row_for_k(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    for row in rows:
        if int(row["k"]) == k:
            return row
    raise KeyError(f"missing k {k}")


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
