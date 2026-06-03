"""Modal CPU analysis for curated Terminal-Bench recording internals."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import modal


APP_NAME = "asb-curated14-recording-analysis"
VOLUME_NAME = "asb-terminal-recordings"
SECRET_NAME = "asb-gdrive-rclone"
ARCHIVE_NAME = "curated-14-recording-internals-20260509T062036Z.tar.zst"
DRIVE_DIR = (
    "asb_gdrive:agent-sched-bench-backups/terminal-bench-qwen3-coder/"
    "curated-14-recording-internals-20260509T062036Z"
)
VOLUME_ROOT = Path("/data")
ARCHIVE_DIR = VOLUME_ROOT / "archives"
EXTRACT_DIR = VOLUME_ROOT / "extracted" / "curated14"
OUTPUT_DIR = VOLUME_ROOT / "outputs" / "curated14_phase_artifact_analysis"

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
    .apt_install("ca-certificates", "curl", "rclone", "zstd")
    .pip_install("matplotlib", "numpy")
    .add_local_dir(RECODING_FIGURES, remote_path="/opt/recoding_figures", copy=True)
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
secret = modal.Secret.from_name(SECRET_NAME, required_keys=["RCLONE_CONFIG_CONTENT"])
app = modal.App(APP_NAME)


@app.function(
    image=image,
    volumes={VOLUME_ROOT: volume},
    secrets=[secret],
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 4,
)
def prepare_data(force_download: bool = False, force_extract: bool = False) -> dict[str, Any]:
    """Download, verify, and extract the curated archive into the Modal Volume."""
    _write_rclone_config()
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archive = ARCHIVE_DIR / ARCHIVE_NAME
    sha = ARCHIVE_DIR / f"{ARCHIVE_NAME}.sha256"
    if force_download or not archive.exists() or not sha.exists():
        print("downloading archive and checksum from Google Drive", flush=True)
        _rclone_copy(f"{DRIVE_DIR}/{ARCHIVE_NAME}", ARCHIVE_DIR)
        _rclone_copy(f"{DRIVE_DIR}/{ARCHIVE_NAME}.sha256", ARCHIVE_DIR)
    print("verifying archive checksum", flush=True)
    expected_hash = sha.read_text(encoding="utf-8").split()[0]
    _run(["sha256sum", "-c", "-"], cwd=ARCHIVE_DIR, input_text=f"{expected_hash}  {ARCHIVE_NAME}\n")

    marker = EXTRACT_DIR / ".complete"
    if force_extract or not marker.exists():
        print("extracting archive into Modal Volume", flush=True)
        if EXTRACT_DIR.exists():
            shutil.rmtree(EXTRACT_DIR)
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        _run(["tar", "-I", "zstd", "-xf", str(archive), "-C", str(EXTRACT_DIR)])
        marker.write_text("ok\n", encoding="utf-8")

    stats = {
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "extract_dir": str(EXTRACT_DIR),
        "attempt_dirs": len(_attempt_dirs()),
        "recording_iters": len(list(EXTRACT_DIR.glob("*/attempt_1/recordings/iter_*"))),
    }
    volume.commit()
    print("prepare_data complete", flush=True)
    return stats


@app.function(
    image=image,
    volumes={VOLUME_ROOT: volume},
    cpu=16,
    memory=65536,
    timeout=60 * 60 * 4,
)
def run_analysis() -> dict[str, Any]:
    """Run phase and measurement-artifact analyses over extracted recordings."""
    sys.path.insert(0, "/opt/recoding_figures")
    import matplotlib

    matplotlib.use("Agg")

    from plot_iter_distance import compute_iter_distance_matrices
    from recording_loader import (
        ROLE_ORDER,
        average_layer_matrix,
        load_attention_distributions,
        load_iteration_records,
    )

    attempts = _attempt_dirs()
    if not attempts:
        raise FileNotFoundError(f"no extracted attempts under {EXTRACT_DIR}")

    print("loading recordings and computing phase/artifact summaries", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_iteration_records(attempts)
    role_labels = _role_labels_from_records(records, ROLE_ORDER)
    token_matrix = _token_role_matrix(records, role_labels)
    token_js = _pairwise_js_matrix(token_matrix)

    phase_summaries: dict[str, Any] = {}
    artifact_rows: list[dict[str, float]] = []
    role_profiles: dict[str, dict[str, float]] = {}
    for phase in ("all", "prefill", "decode"):
        dataset = load_attention_distributions(records, role_labels=role_labels, phase=phase)
        matrices, _ = compute_iter_distance_matrices(dataset)
        phase_summaries[phase] = _distance_summary(records, matrices)
        layers, matrix, _counts = average_layer_matrix(dataset, equal_iter_weight=True)
        role_profiles[phase] = {
            label: float(matrix[:, idx].mean()) for idx, label in enumerate(role_labels)
        }
        for layer, distance_matrix in matrices.items():
            artifact_rows.append(
                _artifact_row(records, layer, phase, distance_matrix, token_js)
            )

    prefill_decode = _prefill_decode_summary(records, role_labels)
    summary = {
        "n_records": len(records),
        "n_tasks": len({record.task for record in records}),
        "task_counts": {
            task: sum(record.task == task for record in records)
            for task in sorted({record.task for record in records})
        },
        "role_labels": role_labels,
        "token_role_mean": {
            label: float(token_matrix[:, idx].mean()) for idx, label in enumerate(role_labels)
        },
        "token_role_pairwise_js": _distance_summary(records, {0: token_js}),
        "phase_summaries": phase_summaries,
        "role_profiles": role_profiles,
        "artifact_correlations": artifact_rows,
        "artifact_correlation_summary": _artifact_summary(artifact_rows),
        "prefill_decode_same_iter": prefill_decode,
    }

    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUTPUT_DIR / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    _plot_phase_bars(summary, OUTPUT_DIR / "phase_distance_summary.pdf")
    _plot_role_profiles(role_labels, role_profiles, OUTPUT_DIR / "phase_role_profiles.pdf")
    _plot_artifact_correlations(artifact_rows, OUTPUT_DIR / "artifact_correlation_by_layer.pdf")
    _plot_prefill_decode(prefill_decode, OUTPUT_DIR / "prefill_decode_layer_js.pdf")

    tar_path = OUTPUT_DIR.with_suffix(".tar.zst")
    if tar_path.exists():
        tar_path.unlink()
    _run(["tar", "-I", "zstd -T0 -3", "-cf", str(tar_path), "-C", str(OUTPUT_DIR.parent), OUTPUT_DIR.name])
    volume.commit()
    print("run_analysis complete", flush=True)
    return {
        "output_dir": str(OUTPUT_DIR),
        "output_tar": str(tar_path),
        "output_tar_bytes": tar_path.stat().st_size,
        "summary": summary,
    }


@app.local_entrypoint()
def main(
    action: str = "all",
    force_download: bool = False,
    force_extract: bool = False,
    background: bool = False,
) -> None:
    """Run Modal preparation and/or analysis."""
    if background:
        if action not in {"prepare", "analysis"}:
            raise ValueError("--background supports action=prepare or action=analysis")
        function = prepare_data if action == "prepare" else run_analysis
        args = (force_download, force_extract) if action == "prepare" else ()
        call = function.spawn(*args)
        print(f"spawned {action}: {call.object_id}")
        print(call.get_dashboard_url())
        return
    if action in {"all", "prepare"}:
        print(json.dumps(prepare_data.remote(force_download, force_extract), indent=2))
    if action in {"all", "analysis"}:
        result = run_analysis.remote()
        print(json.dumps(result["summary"], indent=2))


def _write_rclone_config() -> None:
    config = os.environ.get("RCLONE_CONFIG_CONTENT")
    if not config:
        raise RuntimeError("RCLONE_CONFIG_CONTENT secret is missing")
    config_dir = Path("/root/.config/rclone")
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "rclone.conf"
    path.write_text(config, encoding="utf-8")
    path.chmod(0o600)


def _run(cmd: list[str], *, cwd: Path | None = None, input_text: str | None = None) -> None:
    subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        input=input_text,
        text=input_text is not None,
        check=True,
    )


def _rclone_copy(source: str, destination: Path) -> None:
    _run(
        [
            "rclone",
            "copy",
            source,
            str(destination),
            "--transfers",
            "1",
            "--checkers",
            "4",
            "--stats",
            "30s",
            "--stats-one-line",
            "--log-level",
            "INFO",
        ]
    )


def _attempt_dirs() -> list[Path]:
    return sorted(EXTRACT_DIR.glob("*/attempt_1"))


def _role_labels_from_records(records: list[Any], role_order: list[str]) -> list[str]:
    seen: set[str] = set()
    for record in records:
        payload = json.loads((record.iter_dir / "segments.json").read_text(encoding="utf-8"))
        for segment in payload.get("segments", []):
            seen.add(_normalize_role(segment))
    labels = [role for role in role_order if role in seen]
    labels.extend(sorted(seen.difference(labels)))
    return labels


def _token_role_matrix(records: list[Any], role_labels: list[str]) -> Any:
    import numpy as np

    role_index = {role: idx for idx, role in enumerate(role_labels)}
    matrix = np.zeros((len(records), len(role_labels)), dtype=np.float64)
    for row_idx, record in enumerate(records):
        payload = json.loads((record.iter_dir / "segments.json").read_text(encoding="utf-8"))
        total = 0
        for segment in payload.get("segments", []):
            start = int(segment.get("token_start", 0) or 0)
            end = int(segment.get("token_end", start) or start)
            length = max(0, end - start)
            matrix[row_idx, role_index[_normalize_role(segment)]] += length
            total += length
        if total > 0:
            matrix[row_idx] /= float(total)
    return matrix


def _normalize_role(segment: dict[str, Any]) -> str:
    role = str(segment.get("role") or "other")
    if role == "assistant" and bool(segment.get("has_tool_calls")):
        return "assistant_call"
    if role == "assistant":
        return "assistant_message"
    if role in {"tool", "tool_result"}:
        return "tool_result"
    if role in {
        "system",
        "user",
        "assistant_message",
        "assistant_call",
        "tool_result",
        "gen_prompt",
        "generation",
        "meta",
        "other",
    }:
        return role
    return "other"


def _pairwise_js_matrix(distributions: Any) -> Any:
    sys.path.insert(0, "/opt/recoding_figures")
    from metrics import pairwise_js

    return pairwise_js(distributions)


def _distance_summary(records: list[Any], matrices: dict[int, Any]) -> dict[str, float]:
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
    same_mean = float(np.mean(same_task_values)) if same_task_values else float("nan")
    cross_mean = float(np.mean(cross_task_values)) if cross_task_values else float("nan")
    return {
        "mean_pairwise_js": float(np.mean(all_values)) if all_values else float("nan"),
        "mean_adjacent_js": float(np.mean(adjacent_values)) if adjacent_values else float("nan"),
        "mean_same_task_js": same_mean,
        "mean_cross_task_js": cross_mean,
        "cross_over_same_ratio": float(cross_mean / same_mean) if same_mean > 0 else float("nan"),
        "n_pairs": float(len(all_values)),
    }


def _artifact_row(records: list[Any], layer: int, phase: str, attention_js: Any, token_js: Any) -> dict[str, float]:
    import numpy as np

    sys.path.insert(0, "/opt/recoding_figures")
    from metrics import pearson_corr

    valid = np.isfinite(attention_js) & np.isfinite(token_js)
    upper = np.triu_indices(attention_js.shape[0], k=1)
    valid_upper = valid[upper]
    attn_values = attention_js[upper][valid_upper]
    token_values = token_js[upper][valid_upper]
    same_mask = np.asarray(
        [records[i].task == records[j].task for i, j in zip(upper[0][valid_upper], upper[1][valid_upper])],
        dtype=bool,
    )
    same_mean = float(np.mean(attn_values[same_mask])) if bool(same_mask.any()) else float("nan")
    cross_mean = float(np.mean(attn_values[~same_mask])) if bool((~same_mask).any()) else float("nan")
    return {
        "layer": float(layer),
        "phase": phase,
        "corr_attention_vs_token_role_js": float(pearson_corr(attn_values, token_values)),
        "attention_mean_js": float(np.mean(attn_values)) if attn_values.size else float("nan"),
        "token_role_mean_js": float(np.mean(token_values)) if token_values.size else float("nan"),
        "same_task_attention_js": same_mean,
        "cross_task_attention_js": cross_mean,
    }


def _artifact_summary(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    import numpy as np

    out: dict[str, dict[str, float]] = {}
    for phase in sorted({str(row["phase"]) for row in rows}):
        phase_rows = [row for row in rows if row["phase"] == phase]
        correlations = np.asarray([row["corr_attention_vs_token_role_js"] for row in phase_rows], dtype=np.float64)
        correlations = correlations[np.isfinite(correlations)]
        out[phase] = {
            "mean_corr_attention_vs_token_role_js": float(np.mean(correlations)) if correlations.size else float("nan"),
            "median_corr_attention_vs_token_role_js": float(np.median(correlations)) if correlations.size else float("nan"),
            "min_corr_attention_vs_token_role_js": float(np.min(correlations)) if correlations.size else float("nan"),
            "max_corr_attention_vs_token_role_js": float(np.max(correlations)) if correlations.size else float("nan"),
        }
    return out


def _prefill_decode_summary(records: list[Any], role_labels: list[str]) -> dict[str, Any]:
    import numpy as np

    sys.path.insert(0, "/opt/recoding_figures")
    from metrics import js_divergence
    from recording_loader import load_attention_distributions

    prefill = load_attention_distributions(records, role_labels=role_labels, phase="prefill")
    decode = load_attention_distributions(records, role_labels=role_labels, phase="decode")
    layer_rows: list[dict[str, float]] = []
    all_values: list[float] = []
    for layer in sorted(set(prefill.layers).intersection(decode.layers)):
        values: list[float] = []
        for idx in range(len(records)):
            if prefill.observation_counts[layer][idx] <= 0 or decode.observation_counts[layer][idx] <= 0:
                continue
            value = js_divergence(prefill.distributions[layer][idx], decode.distributions[layer][idx])
            values.append(float(value))
            all_values.append(float(value))
        if values:
            layer_rows.append(
                {
                    "layer": float(layer),
                    "mean_js": float(np.mean(values)),
                    "median_js": float(np.median(values)),
                    "p90_js": float(np.percentile(values, 90)),
                    "n_records": float(len(values)),
                }
            )
    values_arr = np.asarray(all_values, dtype=np.float64)
    return {
        "mean_js": float(np.mean(values_arr)) if values_arr.size else float("nan"),
        "median_js": float(np.median(values_arr)) if values_arr.size else float("nan"),
        "p90_js": float(np.percentile(values_arr, 90)) if values_arr.size else float("nan"),
        "max_js": float(np.max(values_arr)) if values_arr.size else float("nan"),
        "layer_rows": layer_rows,
    }


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Curated 14 Phase and Artifact Analysis",
        "",
        f"- Records: `{summary['n_records']}` across `{summary['n_tasks']}` tasks.",
        "",
        "## Attention Phase Distance",
    ]
    for phase, item in summary["phase_summaries"].items():
        lines.append(
            f"- `{phase}`: pairwise `{item['mean_pairwise_js']:.4f}`, "
            f"adjacent `{item['mean_adjacent_js']:.4f}`, same-task "
            f"`{item['mean_same_task_js']:.4f}`, cross-task "
            f"`{item['mean_cross_task_js']:.4f}`."
        )
    pd = summary["prefill_decode_same_iter"]
    lines.extend(
        [
            "",
            "## Prefill vs Decode",
            (
                f"- Same-iteration prefill/decode JS mean `{pd['mean_js']:.4f}`, "
                f"median `{pd['median_js']:.4f}`, p90 `{pd['p90_js']:.4f}`."
            ),
            "",
            "## Segment-Composition Artifact Check",
        ]
    )
    for phase, item in summary["artifact_correlation_summary"].items():
        lines.append(
            f"- `{phase}`: mean corr(attention JS, token-role JS) "
            f"`{item['mean_corr_attention_vs_token_role_js']:.4f}`, median "
            f"`{item['median_corr_attention_vs_token_role_js']:.4f}`."
        )
    return "\n".join(lines) + "\n"


def _plot_phase_bars(summary: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    phases = ["all", "prefill", "decode"]
    metrics = ["mean_same_task_js", "mean_cross_task_js", "mean_adjacent_js"]
    labels = ["same task", "cross task", "adjacent"]
    x = np.arange(len(phases))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for idx, metric in enumerate(metrics):
        values = [summary["phase_summaries"][phase][metric] for phase in phases]
        ax.bar(x + (idx - 1) * width, values, width, label=labels[idx])
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.set_ylabel("JS divergence (bits)")
    ax.set_title("Attention role-distance by phase")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_role_profiles(role_labels: list[str], profiles: dict[str, dict[str, float]], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    phases = ["all", "prefill", "decode"]
    x = np.arange(len(role_labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for idx, phase in enumerate(phases):
        values = [profiles[phase].get(role, 0.0) for role in role_labels]
        ax.bar(x + (idx - 1) * width, values, width, label=phase)
    ax.set_xticks(x)
    ax.set_xticklabels(role_labels, rotation=35, ha="right")
    ax.set_ylabel("mean attention mass")
    ax.set_title("Role-level attention profile by phase")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_artifact_correlations(rows: list[dict[str, float]], output: Path) -> None:
    import matplotlib.pyplot as plt

    phases = ["all", "prefill", "decode"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for phase in phases:
        phase_rows = [row for row in rows if row["phase"] == phase]
        ax.plot(
            [row["layer"] for row in phase_rows],
            [row["corr_attention_vs_token_role_js"] for row in phase_rows],
            marker="o",
            markersize=3,
            linewidth=1.1,
            label=phase,
        )
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_xlabel("layer")
    ax.set_ylabel("corr(attention JS, token-role JS)")
    ax.set_title("How much do stripes follow segment composition?")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_prefill_decode(summary: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    rows = summary["layer_rows"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(
        [row["layer"] for row in rows],
        [row["mean_js"] for row in rows],
        marker="o",
        markersize=3,
        linewidth=1.2,
    )
    ax.set_xlabel("layer")
    ax.set_ylabel("same-iteration prefill/decode JS")
    ax.set_title("Prefill vs decode attention role profile gap")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
