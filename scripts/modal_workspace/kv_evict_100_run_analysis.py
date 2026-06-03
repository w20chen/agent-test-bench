"""Modal CPU analysis for the kv-evict-openclaw-qwen-coder-100-run capstone.

Three traces are pulled from `gdrive:agent-sched-bench-backups/
kv-evict-openclaw-qwen-coder-100-run/` (baseline-none / h2o-b1024-fail /
h2o-b4096-success, 4.726 GiB / 178 files uncompressed) into the shared
`asb-terminal-recordings` Modal volume and compared along four axes:
eviction profile, role survival, trace divergence, attention shift.

See `~/.claude/plans/local-trace-collect-streamingllm-h2o-ra-crispy-goose.md`
for the per-analysis spec.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import modal


# Defaults used by attention sink/recent analysis. Match the h2o configs we
# shipped: sink=4 prefix tokens, recent_window matched to the largest budget
# we tested (4096) so the band covers everything b4096 still keeps. The plan
# can override at run time via `--sink/--recent` flags if needed.
_DEFAULT_SINK = 4
_DEFAULT_RECENT = 256


APP_NAME = "asb-kv-evict-100-run"
VOLUME_NAME = "asb-terminal-recordings"
SECRET_NAME = "asb-gdrive-rclone"
DRIVE_DIR = (
    "asb_gdrive:agent-sched-bench-backups/kv-evict-openclaw-qwen-coder-100-run"
)
# All labels living in EXTRACT_DIR. prepare_data iterates these.
RUN_LABELS = ("baseline-none", "h2o-b1024-fail", "h2o-b4096-success")
# Subset analyses operate on. b1024-fail produced a single iter (model
# emitted empty tool_calls and openclaw exited), which is the legitimate
# failure signal but yields too thin a sample for the aggregate plots.
# Treat it as an anecdote referenced in the writeup, not a comparator.
ANALYSIS_LABELS = ("baseline-none", "h2o-b4096-success")
BASELINE_LABEL = "baseline-none"

VOLUME_ROOT = Path("/data")
EXTRACT_DIR = VOLUME_ROOT / "extracted" / "kv-evict-100-run"
OUTPUT_DIR = VOLUME_ROOT / "outputs" / "kv-evict-100-run"

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
    .apt_install("ca-certificates", "curl", "rclone")
    .pip_install("matplotlib", "numpy")
    .add_local_dir(RECODING_FIGURES, remote_path="/opt/recoding_figures", copy=True)
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
secret = modal.Secret.from_name(SECRET_NAME, required_keys=["RCLONE_CONFIG_CONTENT"])
app = modal.App(APP_NAME)


# ---------------------------------------------------------------------------
# prepare_data: rclone uncompressed gdrive tree into EXTRACT_DIR/<label>/
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    volumes={VOLUME_ROOT: volume},
    secrets=[secret],
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 4,
)
def prepare_data(force: bool = False) -> dict[str, Any]:
    """Pull the three trace directories into the Modal Volume via rclone.

    Idempotent: skips any `EXTRACT_DIR/<label>/.complete` markers unless
    `force=True`. After each label completes, scans for `kv_eviction.npz`
    files under that label so we can spot Risk #1 from the plan (a stale
    b1024 run with empty recordings).
    """
    _write_rclone_config()
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    per_run_stats: list[dict[str, Any]] = []
    for label in RUN_LABELS:
        dest = EXTRACT_DIR / label
        marker = dest / ".complete"
        if marker.exists() and not force:
            print(f"[{label}] already complete, skipping rclone copy", flush=True)
        else:
            dest.mkdir(parents=True, exist_ok=True)
            print(f"[{label}] rclone copy from {DRIVE_DIR}/{label}/", flush=True)
            _rclone_copy(f"{DRIVE_DIR}/{label}", dest)
            marker.write_text("ok\n", encoding="utf-8")
        per_run_stats.append(_scan_run(label, dest))

    volume.commit()
    summary = {
        "extract_dir": str(EXTRACT_DIR),
        "runs": per_run_stats,
        "total_iter_dirs": sum(r["iter_dirs"] for r in per_run_stats),
        "total_kv_eviction_npz": sum(r["kv_eviction_npz"] for r in per_run_stats),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _scan_run(label: str, dest: Path) -> dict[str, Any]:
    iter_dirs = list(dest.rglob("recordings/iter_*"))
    iter_dirs = [p for p in iter_dirs if p.is_dir()]
    kv_files = [p for p in dest.rglob("recordings/iter_*/kv_eviction.npz") if p.is_file()]
    attn_files = [p for p in dest.rglob("recordings/iter_*/attention.npz") if p.is_file()]
    trace_files = list(dest.rglob("attempt_*/trace.jsonl"))
    return {
        "label": label,
        "iter_dirs": len(iter_dirs),
        "attention_npz": len(attn_files),
        "kv_eviction_npz": len(kv_files),
        "trace_jsonl": len(trace_files),
        "bytes": sum(p.stat().st_size for p in dest.rglob("*") if p.is_file()),
    }


# ---------------------------------------------------------------------------
# analyze_eviction: Section A from the plan
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    volumes={VOLUME_ROOT: volume},
    cpu=16,
    memory=65536,
    timeout=60 * 60 * 4,
)
def analyze_eviction() -> dict[str, Any]:
    """Phase × budget eviction distribution + per-call pre_len trajectory."""
    sys.path.insert(0, "/opt/recoding_figures")
    import matplotlib

    matplotlib.use("Agg")

    from kv_eviction_metrics import (
        compute_eviction_profile_rows,
        compute_phase_distribution,
    )
    from recording_loader import find_attempt_dirs, load_iteration_records, load_kv_eviction

    out_dir = OUTPUT_DIR / "eviction_profile"
    out_dir.mkdir(parents=True, exist_ok=True)

    profile_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    per_run_stats: dict[str, dict[str, Any]] = {}

    for label in ANALYSIS_LABELS:
        run_root = EXTRACT_DIR / label
        if not run_root.exists():
            per_run_stats[label] = {"status": "missing"}
            continue
        attempts = find_attempt_dirs([run_root])
        if not attempts:
            per_run_stats[label] = {"status": "no_attempts"}
            continue
        records = load_iteration_records(attempts)
        frame = load_kv_eviction(records)
        per_run_stats[label] = {
            "n_records": len(records),
            "n_eviction_rows": frame.n_rows,
        }
        if frame.is_empty:
            print(f"[{label}] no kv_eviction.npz — baseline or stale ingest", flush=True)
            continue
        profile_rows.extend(compute_eviction_profile_rows(frame, run_label=label))
        phase_rows.extend(compute_phase_distribution(frame, run_label=label))

    _write_csv(
        out_dir / "eviction_profile.csv",
        profile_rows,
        ["run", "task", "call_idx", "layer", "step", "phase",
         "pre_len", "post_len", "budget", "n_evicted", "reason"],
    )
    # phase distribution has nested reasons dict; flatten to long format
    flat_phase_rows: list[dict[str, Any]] = []
    for row in phase_rows:
        for reason, count in row["reasons"].items():
            flat_phase_rows.append(
                {
                    "run": row["run"],
                    "phase": row["phase"],
                    "budget": row["budget"],
                    "reason": reason,
                    "n_decisions_with_reason": count,
                    "n_decisions_total": row["n_decisions"],
                    "n_decisions_with_evict": row["n_decisions_with_evict"],
                    "n_evicted_total": row["n_evicted_total"],
                }
            )
    _write_csv(
        out_dir / "phase_distribution.csv",
        flat_phase_rows,
        ["run", "phase", "budget", "reason", "n_decisions_with_reason",
         "n_decisions_total", "n_decisions_with_evict", "n_evicted_total"],
    )

    _plot_phase_distribution(phase_rows, out_dir / "fig_eviction_phase_distribution.png")
    _plot_eviction_onset_per_call(profile_rows, out_dir / "fig_eviction_onset_per_call.png")

    volume.commit()
    return {
        "output_dir": str(out_dir),
        "n_profile_rows": len(profile_rows),
        "n_phase_rows": len(phase_rows),
        "per_run": per_run_stats,
    }


# ---------------------------------------------------------------------------
# analyze_role_survival: Section B
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    volumes={VOLUME_ROOT: volume},
    cpu=16,
    memory=65536,
    timeout=60 * 60 * 4,
)
def analyze_role_survival() -> dict[str, Any]:
    """Per-role kept-vs-total survival rate across every eviction decision."""
    sys.path.insert(0, "/opt/recoding_figures")
    import matplotlib

    matplotlib.use("Agg")

    from kv_eviction_metrics import compute_role_survival_rows, load_segments_by_iter_dir
    from recording_loader import (
        ROLE_ORDER,
        collect_role_labels,
        find_attempt_dirs,
        load_iteration_records,
        load_kv_eviction,
    )

    out_dir = OUTPUT_DIR / "role_survival"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    per_run_stats: dict[str, dict[str, Any]] = {}

    # Pin role_labels to the canonical ROLE_ORDER so columns line up across runs.
    role_labels = list(ROLE_ORDER)

    for label in ANALYSIS_LABELS:
        run_root = EXTRACT_DIR / label
        if not run_root.exists():
            per_run_stats[label] = {"status": "missing"}
            continue
        attempts = find_attempt_dirs([run_root])
        if not attempts:
            per_run_stats[label] = {"status": "no_attempts"}
            continue
        records = load_iteration_records(attempts)
        frame = load_kv_eviction(records)
        # Refresh role_labels with any new roles observed; keep ROLE_ORDER prefix.
        observed_extra = [r for r in collect_role_labels(records) if r not in role_labels]
        role_labels.extend(observed_extra)
        per_run_stats[label] = {
            "n_records": len(records),
            "n_eviction_rows": frame.n_rows,
        }
        if frame.is_empty:
            continue
        segments_payload = load_segments_by_iter_dir(records)
        rows = compute_role_survival_rows(
            frame, segments_payload, role_labels=role_labels, run_label=label
        )
        all_rows.extend(rows)

    _write_csv(
        out_dir / "role_survival.csv",
        all_rows,
        ["run", "task", "call_idx", "layer", "step", "phase",
         "role", "total_tokens", "kept_tokens", "survival_rate"],
    )

    _plot_role_survival_heatmap(all_rows, role_labels, out_dir / "fig_role_survival_heatmap.png")
    _plot_role_survival_diff(all_rows, role_labels, out_dir / "fig_role_survival_diff.png")

    volume.commit()
    return {
        "output_dir": str(out_dir),
        "n_rows": len(all_rows),
        "role_labels": role_labels,
        "per_run": per_run_stats,
    }


# ---------------------------------------------------------------------------
# analyze_trace_divergence: Section C
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    volumes={VOLUME_ROOT: volume},
    cpu=4,
    memory=16384,
    timeout=60 * 60,
)
def analyze_trace_divergence() -> dict[str, Any]:
    """Align llm_call action sequences across the three runs."""
    sys.path.insert(0, "/opt/recoding_figures")
    import matplotlib

    matplotlib.use("Agg")

    out_dir = OUTPUT_DIR / "trace_divergence"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run_actions: dict[str, list[dict[str, Any]]] = {}
    per_run_stats: dict[str, dict[str, Any]] = {}
    for label in ANALYSIS_LABELS:
        run_root = EXTRACT_DIR / label
        trace_files = sorted(run_root.rglob("attempt_*/trace.jsonl"))
        if not trace_files:
            per_run_stats[label] = {"status": "no_trace"}
            continue
        # One task per attempt; concatenate all action lists.
        actions: list[dict[str, Any]] = []
        for trace in trace_files:
            actions.extend(_extract_llm_call_actions(trace))
        per_run_actions[label] = actions
        per_run_stats[label] = {
            "n_llm_calls": len(actions),
            "trace_files": [str(t) for t in trace_files],
        }

    n_calls_per_run = {label: len(acts) for label, acts in per_run_actions.items()}
    max_calls = max(n_calls_per_run.values(), default=0)

    aligned_rows: list[dict[str, Any]] = []
    divergence_point: dict[str, int | None] = {}
    for call_idx in range(max_calls):
        row: dict[str, Any] = {"call_idx": call_idx}
        baseline_hash = None
        for label in ANALYSIS_LABELS:
            acts = per_run_actions.get(label, [])
            if call_idx < len(acts):
                tool_summary = _summarize_tool_calls(acts[call_idx]["tool_calls"])
                tool_hash = _short_hash(tool_summary)
                row[f"{label}__summary"] = tool_summary[:80]
                row[f"{label}__hash"] = tool_hash
                if label == BASELINE_LABEL:
                    baseline_hash = tool_hash
            else:
                row[f"{label}__summary"] = ""
                row[f"{label}__hash"] = ""
        aligned_rows.append(row)
        # Record first call_idx where each variant diverges from baseline.
        if baseline_hash:
            for label in ANALYSIS_LABELS:
                if label == BASELINE_LABEL:
                    continue
                if label in divergence_point:
                    continue
                variant_hash = row.get(f"{label}__hash") or ""
                if not variant_hash or variant_hash != baseline_hash:
                    divergence_point[label] = call_idx

    columns = ["call_idx"]
    for label in ANALYSIS_LABELS:
        columns.extend([f"{label}__summary", f"{label}__hash"])
    _write_csv(out_dir / "trace_action_table.csv", aligned_rows, columns)

    _plot_action_alignment(
        per_run_actions,
        divergence_point,
        out_dir / "fig_action_alignment.png",
    )

    summary = {
        "output_dir": str(out_dir),
        "n_calls_per_run": n_calls_per_run,
        "divergence_point": divergence_point,
        "per_run": per_run_stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    volume.commit()
    return summary


# ---------------------------------------------------------------------------
# analyze_attention_shift: Section D
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    volumes={VOLUME_ROOT: volume},
    cpu=16,
    memory=65536,
    timeout=60 * 60 * 4,
)
def analyze_attention_shift(sink: int = _DEFAULT_SINK, recent: int = _DEFAULT_RECENT) -> dict[str, Any]:
    """Aggregate attention distributions across the whole run, compare vs baseline."""
    sys.path.insert(0, "/opt/recoding_figures")
    import matplotlib

    matplotlib.use("Agg")
    import numpy as np

    from kv_eviction_metrics import (
        aggregate_attention_role_per_layer,
        aggregate_heavy_hitters_per_layer,
        aggregate_sink_recent_share_per_layer,
        compute_attention_js_per_layer,
        compute_heavy_jaccard_per_layer,
    )
    from recording_loader import (
        ROLE_ORDER,
        collect_role_labels,
        find_attempt_dirs,
        load_iteration_records,
    )

    out_dir = OUTPUT_DIR / "attention_shift"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run_records: dict[str, list] = {}
    per_run_stats: dict[str, dict[str, Any]] = {}
    for label in ANALYSIS_LABELS:
        run_root = EXTRACT_DIR / label
        if not run_root.exists():
            per_run_stats[label] = {"status": "missing"}
            continue
        attempts = find_attempt_dirs([run_root])
        if not attempts:
            per_run_stats[label] = {"status": "no_attempts"}
            continue
        records = load_iteration_records(attempts)
        per_run_records[label] = records
        per_run_stats[label] = {"n_records": len(records)}

    if BASELINE_LABEL not in per_run_records:
        raise RuntimeError(f"baseline run {BASELINE_LABEL!r} not ingested")

    # Pin role_labels to a stable union before any aggregation so per-run
    # matrices share columns and JS divergence is meaningful.
    role_labels = list(ROLE_ORDER)
    for records in per_run_records.values():
        for r in collect_role_labels(records):
            if r not in role_labels:
                role_labels.append(r)

    per_run_layer_dist: dict[str, dict[int, np.ndarray]] = {}
    per_run_heavy: dict[str, dict[int, set[int]]] = {}
    per_run_sink_recent: dict[str, dict[int, dict[str, float]]] = {}
    role_distribution_rows: list[dict[str, Any]] = []

    for label, records in per_run_records.items():
        _, per_layer = aggregate_attention_role_per_layer(records, role_labels=role_labels)
        per_run_layer_dist[label] = per_layer
        per_run_heavy[label] = aggregate_heavy_hitters_per_layer(records)
        per_run_sink_recent[label] = aggregate_sink_recent_share_per_layer(
            records, sink=sink, recent=recent
        )
        for layer, dist in per_layer.items():
            for role_idx, role in enumerate(role_labels):
                role_distribution_rows.append(
                    {
                        "run": label,
                        "layer": layer,
                        "role": role,
                        "mass": float(dist[role_idx]),
                    }
                )

    js_rows = compute_attention_js_per_layer(per_run_layer_dist, baseline_label=BASELINE_LABEL)
    jaccard_rows = compute_heavy_jaccard_per_layer(per_run_heavy, baseline_label=BASELINE_LABEL)

    sink_rows: list[dict[str, Any]] = []
    for label, layer_map in per_run_sink_recent.items():
        for layer, shares in layer_map.items():
            sink_rows.append({"run": label, "layer": layer, **shares})

    _write_csv(
        out_dir / "attention_role_distribution.csv",
        role_distribution_rows,
        ["run", "layer", "role", "mass"],
    )
    _write_csv(out_dir / "attention_js_divergence.csv", js_rows, ["layer", "run_pair", "js"])
    _write_csv(
        out_dir / "attention_heavy_jaccard.csv",
        jaccard_rows,
        ["layer", "run_pair", "jaccard", "baseline_size", "variant_size"],
    )
    _write_csv(
        out_dir / "attention_sink_recent_share.csv",
        sink_rows,
        ["run", "layer", "sink_share", "recent_share", "middle_share"],
    )

    _plot_js_per_layer(js_rows, out_dir / "fig_js_divergence_per_layer.png")
    _plot_heavy_jaccard(jaccard_rows, out_dir / "fig_heavy_jaccard_per_layer.png")
    _plot_sink_recent_share(sink_rows, out_dir / "fig_sink_recent_share.png")
    _plot_role_distribution_heatmap(
        per_run_layer_dist, role_labels, out_dir / "fig_role_distribution_heatmap.png"
    )

    volume.commit()
    return {
        "output_dir": str(out_dir),
        "sink": sink,
        "recent": recent,
        "role_labels": role_labels,
        "per_run": per_run_stats,
    }


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    action: str = "prepare",
    force: bool = False,
    background: bool = False,
    sink: int = _DEFAULT_SINK,
    recent: int = _DEFAULT_RECENT,
) -> None:
    """Dispatch Modal jobs.

    actions: prepare | eviction | role | trace | attention | analyze-all
    """
    jobs = {
        "prepare": (prepare_data, (force,)),
        "eviction": (analyze_eviction, ()),
        "role": (analyze_role_survival, ()),
        "trace": (analyze_trace_divergence, ()),
        "attention": (analyze_attention_shift, (sink, recent)),
    }
    if action == "analyze-all":
        for key in ("eviction", "role", "trace", "attention"):
            fn, args = jobs[key]
            print(f"--- {key} ---", flush=True)
            print(json.dumps(fn.remote(*args), indent=2, default=str))
        return
    if action not in jobs:
        raise ValueError(f"unknown action {action!r}; choose one of {list(jobs) + ['analyze-all']}")
    fn, args = jobs[action]
    if background:
        call = fn.spawn(*args)
        print(f"spawned {action}: {call.object_id}")
        print(call.get_dashboard_url())
        return
    print(json.dumps(fn.remote(*args), indent=2, default=str))


# ---------------------------------------------------------------------------
# Helpers (mirror curated14_analysis.py)
# ---------------------------------------------------------------------------


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
            "8",
            "--checkers",
            "16",
            "--stats",
            "30s",
            "--stats-one-line",
            "--log-level",
            "INFO",
        ]
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def _extract_llm_call_actions(trace_path: Path) -> list[dict[str, Any]]:
    """Pull `(call_idx, tool_calls)` rows from one trace.jsonl."""
    actions: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("action_type") != "llm_call":
                continue
            data = rec.get("data") or {}
            rr = data.get("raw_response") or {}
            choices = rr.get("choices") if isinstance(rr, dict) else None
            tool_calls: list[dict[str, str]] = []
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message") if isinstance(choices[0], dict) else None
                if isinstance(msg, dict):
                    for tc in msg.get("tool_calls") or []:
                        fn = tc.get("function") if isinstance(tc, dict) else None
                        if isinstance(fn, dict):
                            tool_calls.append(
                                {
                                    "name": str(fn.get("name") or ""),
                                    "arguments": str(fn.get("arguments") or ""),
                                }
                            )
            actions.append(
                {
                    "call_idx": int(rec.get("iteration") or 0),
                    "tool_calls": tool_calls,
                }
            )
    return actions


def _summarize_tool_calls(tool_calls: list[dict[str, str]]) -> str:
    if not tool_calls:
        return "<empty>"
    parts = []
    for tc in tool_calls:
        args = tc["arguments"]
        if len(args) > 60:
            args = args[:57] + "..."
        parts.append(f"{tc['name']}({args})")
    return " | ".join(parts)


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _plot_phase_distribution(phase_rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    if not phase_rows:
        _placeholder(path, "no eviction rows")
        return
    runs = sorted({r["run"] for r in phase_rows})
    phases = ["prefill", "decode"]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.35
    xs = list(range(len(runs)))
    for i, phase in enumerate(phases):
        values = []
        for run in runs:
            match = [r for r in phase_rows if r["run"] == run and r["phase"] == phase]
            values.append(sum(r["n_decisions_with_evict"] for r in match))
        ax.bar([x + width * (i - 0.5) for x in xs], values, width=width, label=phase)
    ax.set_xticks(xs)
    ax.set_xticklabels(runs, rotation=15)
    ax.set_ylabel("# decisions with eviction")
    ax.set_title("Eviction decisions by phase")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_eviction_onset_per_call(profile_rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    runs = sorted({r["run"] for r in profile_rows})
    if not runs:
        _placeholder(path, "no eviction profile rows")
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 4), sharey=True)
    if len(runs) == 1:
        axes = [axes]
    for ax, run in zip(axes, runs):
        rows = [r for r in profile_rows if r["run"] == run]
        if not rows:
            ax.set_title(f"{run} — no rows")
            continue
        by_call: dict[int, list[int]] = {}
        for r in rows:
            by_call.setdefault(r["call_idx"], []).append(r["pre_len"])
        xs = sorted(by_call)
        means = [float(np.mean(by_call[c])) for c in xs]
        maxes = [float(np.max(by_call[c])) for c in xs]
        ax.plot(xs, means, marker="o", label="mean pre_len")
        ax.plot(xs, maxes, marker="x", linestyle="--", label="max pre_len")
        if rows:
            budget = rows[0]["budget"]
            ax.axhline(budget, color="red", linestyle=":", label=f"budget={budget}")
        ax.set_title(run)
        ax.set_xlabel("call_idx")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("pre_len")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_role_survival_heatmap(rows: list[dict[str, Any]], role_labels: list[str], path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    runs = sorted({r["run"] for r in rows if r["run"] != BASELINE_LABEL})
    if not runs or not rows:
        _placeholder(path, "no survival rows")
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 4), sharey=True)
    if len(runs) == 1:
        axes = [axes]
    role_index = {role: i for i, role in enumerate(role_labels)}
    for ax, run in zip(axes, runs):
        run_rows = [r for r in rows if r["run"] == run]
        calls = sorted({r["call_idx"] for r in run_rows})
        if not calls:
            ax.set_title(f"{run} — empty")
            continue
        call_index = {c: i for i, c in enumerate(calls)}
        matrix = np.full((len(role_labels), len(calls)), np.nan, dtype=np.float64)
        counts = np.zeros_like(matrix)
        for r in run_rows:
            i = role_index.get(r["role"])
            j = call_index.get(r["call_idx"])
            if i is None or j is None:
                continue
            if np.isnan(matrix[i, j]):
                matrix[i, j] = 0.0
            matrix[i, j] += r["survival_rate"]
            counts[i, j] += 1
        with np.errstate(invalid="ignore"):
            matrix = matrix / np.where(counts > 0, counts, 1)
        im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(calls)))
        ax.set_xticklabels(calls, fontsize=7)
        ax.set_yticks(range(len(role_labels)))
        ax.set_yticklabels(role_labels, fontsize=7)
        ax.set_title(f"{run} survival")
    fig.colorbar(im, ax=axes, fraction=0.03)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_role_survival_diff(rows: list[dict[str, Any]], role_labels: list[str], path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    # difference of average per-role survival across calls between two h2o variants.
    runs = sorted({r["run"] for r in rows if r["run"] != BASELINE_LABEL})
    if len(runs) < 2:
        _placeholder(path, "fewer than 2 h2o runs; diff skipped")
        return
    high, low = runs[-1], runs[0]  # higher-budget label last alphabetically (b4096 > b1024)
    role_means: dict[str, dict[str, float]] = {high: {}, low: {}}
    for run in (high, low):
        for role in role_labels:
            vals = [r["survival_rate"] for r in rows if r["run"] == run and r["role"] == role]
            role_means[run][role] = float(np.mean(vals)) if vals else 0.0
    diff = np.array([role_means[high][r] - role_means[low][r] for r in role_labels])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(role_labels, diff)
    ax.set_xlabel(f"survival_rate diff ({high} − {low})")
    ax.set_title("Role survival difference")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_action_alignment(
    per_run_actions: dict[str, list[dict[str, Any]]],
    divergence_point: dict[str, int | None],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    if not per_run_actions:
        _placeholder(path, "no trace actions")
        return
    runs = list(ANALYSIS_LABELS)
    max_calls = max(len(per_run_actions.get(r, [])) for r in runs)
    if max_calls == 0:
        _placeholder(path, "all traces empty")
        return
    fig, ax = plt.subplots(figsize=(max(10, max_calls * 0.3), 2.5))
    for i, run in enumerate(runs):
        acts = per_run_actions.get(run, [])
        ax.broken_barh(
            [(c, 1) for c in range(len(acts))],
            (i - 0.4, 0.8),
            facecolors=[_action_color(acts[c]["tool_calls"]) for c in range(len(acts))],
        )
        ax.text(-0.5, i, run, ha="right", va="center", fontsize=8)
        div = divergence_point.get(run)
        if div is not None:
            ax.axvline(div, color="red", linewidth=1.5, alpha=0.6)
    ax.set_yticks([])
    ax.set_xlim(-1, max_calls + 1)
    ax.set_xlabel("call_idx")
    ax.set_title("Action alignment (color = first tool name; red line = divergence point)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


_TOOL_PALETTE: dict[str, str] = {}
_TOOL_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _action_color(tool_calls: list[dict[str, str]]) -> str:
    name = tool_calls[0]["name"] if tool_calls else "<empty>"
    if name not in _TOOL_PALETTE:
        _TOOL_PALETTE[name] = _TOOL_COLORS[len(_TOOL_PALETTE) % len(_TOOL_COLORS)]
    return _TOOL_PALETTE[name]


def _plot_js_per_layer(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        _placeholder(path, "no JS rows")
        return
    pairs = sorted({r["run_pair"] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 4))
    for pair in pairs:
        pair_rows = sorted([r for r in rows if r["run_pair"] == pair], key=lambda r: r["layer"])
        xs = [r["layer"] for r in pair_rows]
        ys = [r["js"] for r in pair_rows]
        ax.plot(xs, ys, marker="o", label=pair)
    ax.set_xlabel("layer")
    ax.set_ylabel("JS divergence (nats)")
    ax.set_title(f"Attention role-distribution JS vs {BASELINE_LABEL}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_heavy_jaccard(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        _placeholder(path, "no jaccard rows")
        return
    pairs = sorted({r["run_pair"] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 4))
    for pair in pairs:
        pair_rows = sorted([r for r in rows if r["run_pair"] == pair], key=lambda r: r["layer"])
        xs = [r["layer"] for r in pair_rows]
        ys = [r["jaccard"] for r in pair_rows]
        ax.plot(xs, ys, marker="o", label=pair)
    ax.set_xlabel("layer")
    ax.set_ylabel("Jaccard of heavy-hitter sets")
    ax.set_title(f"Heavy-hitter agreement vs {BASELINE_LABEL}")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_sink_recent_share(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    runs = sorted({r["run"] for r in rows})
    if not runs:
        _placeholder(path, "no sink/recent rows")
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 4), sharey=True)
    if len(runs) == 1:
        axes = [axes]
    for ax, run in zip(axes, runs):
        run_rows = sorted([r for r in rows if r["run"] == run], key=lambda r: r["layer"])
        layers = [r["layer"] for r in run_rows]
        sink = np.array([r["sink_share"] for r in run_rows])
        recent = np.array([r["recent_share"] for r in run_rows])
        middle = np.array([r["middle_share"] for r in run_rows])
        ax.bar(layers, sink, label="sink")
        ax.bar(layers, recent, bottom=sink, label="recent")
        ax.bar(layers, middle, bottom=sink + recent, label="middle")
        ax.set_title(run)
        ax.set_xlabel("layer")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("topk attention mass share")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_role_distribution_heatmap(
    per_run_layer_dist: dict[str, dict[int, Any]],
    role_labels: list[str],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    runs = list(per_run_layer_dist)
    if not runs:
        _placeholder(path, "no role distributions")
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 4), sharey=True)
    if len(runs) == 1:
        axes = [axes]
    for ax, run in zip(axes, runs):
        layer_map = per_run_layer_dist[run]
        layers = sorted(layer_map)
        if not layers:
            ax.set_title(f"{run} — empty")
            continue
        matrix = np.vstack([layer_map[layer] for layer in layers])
        im = ax.imshow(matrix.T, aspect="auto", cmap="viridis", vmin=0, vmax=matrix.max() or 1)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, fontsize=6)
        ax.set_yticks(range(len(role_labels)))
        ax.set_yticklabels(role_labels, fontsize=7)
        ax.set_title(run)
        ax.set_xlabel("layer")
    fig.colorbar(im, ax=axes, fraction=0.03)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _placeholder(path: Path, message: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 2))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10)
    ax.axis("off")
    fig.savefig(path, dpi=120)
    plt.close(fig)
