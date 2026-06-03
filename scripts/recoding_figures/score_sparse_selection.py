"""Score sparse-attention selection quality against full-attention ground truth.

Joins `attention.npz` (real per-query top-K from a full-attention run) with
`sparse_attention.npz` (observe-only "would have selected" record) and the
attempt-level `meta.json["sparse_attention"]` block, then computes per-query:

- `mass_in_keep_set`: sum of true top-K attention weights whose key positions
  fall inside the sparse method's keep set.
- `recall_at_k` for k in (8, 32, 128): of the first k true heavy hitters (the
  CSR is sorted DESC by weight), what fraction are in the keep set.

The whole comparison is meaningful only when the trace was collected with
`--sparse-attn-observe-only` so the recorded `attention.npz` is the real,
unmasked attention distribution — enforced sparse traces have hard zeros
exactly where the sparse mask cut, which would make this trivially perfect.

Dynamic decode methods reconstruct their keep set from attempt-level
sink/recent config plus per-row `extras_json["selected_middle_indices"]`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import polars as pl

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.recoding_figures.recording_loader import (  # noqa: E402
    find_attempt_dirs,
    load_iteration_records,
)
from scripts.recoding_figures.sparse_keep_sets import (  # noqa: E402
    SparseParams,
    reconstruct_keep_set,
)


def _read_meta_sparse_block(attempt_dir: Path) -> dict[str, object]:
    meta_path = attempt_dir / "recordings" / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing meta.json: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    sparse_block = meta.get("sparse_attention")
    if sparse_block is None:
        raise ValueError(
            f"{meta_path} has no 'sparse_attention' block — was this attempt "
            "collected with --sparse-attn?"
        )
    return sparse_block


def _sparse_params_from_meta(meta_block: dict[str, object]) -> SparseParams:
    return SparseParams(
        sink_size=int(meta_block["sink_size"]),
        recent_window=int(meta_block["recent_window"]),
    )


def _build_sparse_key_lookup(
    sparse_npz: dict[str, np.ndarray],
) -> dict[tuple[int, str, int], int]:
    """Map `(record_layer, record_phase, record_decode_step)` -> row index."""
    layer = sparse_npz["record_layer"].astype(np.int32)
    phase = sparse_npz["record_phase"]
    dstep = sparse_npz["record_decode_step"].astype(np.int32)
    lookup: dict[tuple[int, str, int], int] = {}
    for i in range(layer.shape[0]):
        key = (int(layer[i]), str(phase[i]), int(dstep[i]))
        if key in lookup:
            raise ValueError(
                f"duplicate sparse_attention.npz key {key} at row {i}; "
                "row alignment is broken — the recorder must produce at most "
                "one row per (layer, phase, decode_step)."
            )
        lookup[key] = i
    return lookup


def _per_query_rows_for_iter(
    *,
    task: str,
    call_idx: int,
    iter_dir: Path,
    method_name: str,
    method_params: SparseParams,
    recall_ks: tuple[int, ...],
) -> Iterable[dict[str, object]]:
    att_path = iter_dir / "attention.npz"
    sp_path = iter_dir / "sparse_attention.npz"
    if not att_path.is_file() or not sp_path.is_file():
        return ()

    with np.load(att_path, allow_pickle=True) as att, np.load(
        sp_path, allow_pickle=True
    ) as sp:
        sp_lookup = _build_sparse_key_lookup({k: sp[k] for k in sp.files})
        sp_key_len = sp["key_len"].astype(np.int32)
        sp_kept_count = sp["kept_count"].astype(np.int32)
        sp_extras = sp["extras_json"]

        record_layer = att["record_layer"].astype(np.int32)
        record_phase = att["record_phase"]
        record_decode_step = att["record_decode_step"].astype(np.int32)
        query_row_offsets = att["query_row_offsets"].astype(np.int64)
        topk_offsets = att["topk_csr_offsets"].astype(np.int64)
        topk_indices = att["topk_csr_indices"].astype(np.int32)
        topk_weights = att["topk_csr_weights"].astype(np.float32)
        query_positions = att["query_positions"].astype(np.int32)
        query_heads = att["query_heads"].astype(np.int32)

        for r in range(record_layer.shape[0]):
            layer = int(record_layer[r])
            phase = str(record_phase[r])
            dstep = int(record_decode_step[r])
            sp_idx = sp_lookup.get((layer, phase, dstep))
            if sp_idx is None:
                # attention.npz sampled this (layer, phase, dstep) but
                # sparse recorder didn't see it. This is a real
                # inconsistency in observe-only mode (every layer should
                # fire both hooks per call); surface loudly.
                raise ValueError(
                    f"{iter_dir.name}: attention.npz has record "
                    f"(layer={layer}, phase={phase}, dstep={dstep}) but no "
                    "matching row in sparse_attention.npz"
                )
            key_len = int(sp_key_len[sp_idx])
            extras = json.loads(str(sp_extras[sp_idx]))
            keep_arr = reconstruct_keep_set(
                method_name=method_name,
                method_params=method_params,
                key_len=key_len,
                extras=extras,
            )
            recorded_kept = int(sp_kept_count[sp_idx])
            if int(keep_arr.shape[0]) != recorded_kept:
                # Reconstruct disagreeing with the runtime-recorded kept_count
                # means meta.json config and the runtime sparse method drifted,
                # OR `reconstruct_keep_set` and the method's own `kept_count`
                # have inconsistent semantics for this case. Either silently
                # produces wrong mass/recall numbers; refuse to score.
                raise ValueError(
                    f"{iter_dir / 'sparse_attention.npz'} row {sp_idx}: "
                    f"reconstructed keep_set size {int(keep_arr.shape[0])} != "
                    f"recorded kept_count {recorded_kept} at "
                    f"(layer={layer}, phase={phase}, dstep={dstep}, key_len={key_len}). "
                    "Likely cause: meta.json sparse_attention block diverged from "
                    "the method config used at runtime, or extras_json schema mismatch."
                )
            keep_set = set(int(x) for x in keep_arr)

            q_lo = int(query_row_offsets[r])
            q_hi = int(query_row_offsets[r + 1])
            for qi in range(q_lo, q_hi):
                csr_lo = int(topk_offsets[qi])
                csr_hi = int(topk_offsets[qi + 1])
                if csr_hi <= csr_lo:
                    continue
                row_indices = topk_indices[csr_lo:csr_hi]
                row_weights = topk_weights[csr_lo:csr_hi]
                in_keep = np.fromiter(
                    (int(k) in keep_set for k in row_indices),
                    dtype=bool,
                    count=row_indices.shape[0],
                )
                mass_in_keep = float(row_weights[in_keep].sum())
                mass_total_topk = float(row_weights.sum())
                row = {
                    "task": task,
                    "call_idx": call_idx,
                    "iter_dir": str(iter_dir),
                    "layer": layer,
                    "phase": phase,
                    "decode_step": dstep,
                    "query_idx": int(qi - q_lo),
                    "query_position": int(query_positions[qi]),
                    "query_head": int(query_heads[qi]),
                    "key_len": key_len,
                    "mass_in_keep_set": mass_in_keep,
                    "mass_in_topk": mass_total_topk,
                    "n_topk": int(row_indices.shape[0]),
                    "keep_set_size": int(keep_arr.shape[0]),
                }
                for k in recall_ks:
                    k_eff = min(k, int(row_indices.shape[0]))
                    if k_eff == 0:
                        row[f"recall_at_{k}"] = 0.0
                    else:
                        row[f"recall_at_{k}"] = float(in_keep[:k_eff].sum()) / k_eff
                yield row


def score_attempts(
    *,
    attempt_dirs: list[Path],
    recall_ks: tuple[int, ...],
) -> pl.DataFrame:
    """Return a long-form DataFrame: one row per (iter, query)."""
    records = load_iteration_records(attempt_dirs)
    if not records:
        raise ValueError(f"no iteration records under {attempt_dirs}")

    # Group records by attempt for per-attempt meta.json lookup.
    by_attempt: dict[Path, list] = {}
    for rec in records:
        by_attempt.setdefault(rec.attempt_dir, []).append(rec)

    rows: list[dict[str, object]] = []
    for attempt_dir, recs in by_attempt.items():
        meta_block = _read_meta_sparse_block(attempt_dir)
        method_name = str(meta_block.get("method", "sliding"))
        method_params = _sparse_params_from_meta(meta_block)
        for rec in recs:
            rows.extend(
                _per_query_rows_for_iter(
                    task=rec.task,
                    call_idx=rec.call_idx,
                    iter_dir=rec.iter_dir,
                    method_name=method_name,
                    method_params=method_params,
                    recall_ks=recall_ks,
                )
            )
    if not rows:
        raise ValueError(
            "no per-query rows produced — attention.npz / sparse_attention.npz "
            "either missing or empty across all iterations"
        )
    return pl.DataFrame(rows)


def summarize(
    df: pl.DataFrame, recall_ks: tuple[int, ...]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (per_layer_phase_summary, global_summary)."""
    agg_exprs = [
        pl.col("mass_in_keep_set").mean().alias("mass_in_keep_set_mean"),
        pl.col("mass_in_keep_set").median().alias("mass_in_keep_set_median"),
        pl.col("mass_in_topk").mean().alias("mass_in_topk_mean"),
        pl.len().alias("n_queries"),
    ]
    for k in recall_ks:
        agg_exprs.append(pl.col(f"recall_at_{k}").mean().alias(f"recall_at_{k}_mean"))

    per_layer_phase = (
        df.group_by(["layer", "phase"])
        .agg(agg_exprs)
        .sort(["phase", "layer"])
    )
    global_row = df.select(agg_exprs)
    return per_layer_phase, global_row


def _render_markdown(
    *,
    per_layer_phase: pl.DataFrame,
    global_row: pl.DataFrame,
    recall_ks: tuple[int, ...],
    method_name: str,
    method_params: SparseParams,
) -> str:
    lines: list[str] = []
    lines.append(f"# Sparse Selection Scoring — `{method_name}`")
    lines.append("")
    lines.append(
        f"Method params: `sink_size={method_params.sink_size}`, "
        f"`recent_window={method_params.recent_window}`."
    )
    lines.append("")
    lines.append("## Global")
    lines.append("")
    g = global_row.row(0, named=True)
    lines.append(
        f"- queries: **{int(g['n_queries'])}**"
    )
    lines.append(
        f"- mass_in_keep_set (mean / median): "
        f"**{g['mass_in_keep_set_mean']:.4f}** / "
        f"**{g['mass_in_keep_set_median']:.4f}**"
    )
    lines.append(
        f"- mass_in_topk (mean): **{g['mass_in_topk_mean']:.4f}** "
        "(upper bound — recall ceiling for any sparse method)"
    )
    for k in recall_ks:
        lines.append(f"- recall@{k}: **{g[f'recall_at_{k}_mean']:.4f}**")
    lines.append("")
    lines.append("## Per (layer, phase)")
    lines.append("")
    headers = ["layer", "phase", "n", "mass_keep_mean", "mass_topk_mean"]
    headers.extend(f"recall@{k}" for k in recall_ks)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in per_layer_phase.iter_rows(named=True):
        cells = [
            str(int(row["layer"])),
            str(row["phase"]),
            str(int(row["n_queries"])),
            f"{row['mass_in_keep_set_mean']:.4f}",
            f"{row['mass_in_topk_mean']:.4f}",
        ]
        cells.extend(f"{row[f'recall_at_{k}_mean']:.4f}" for k in recall_ks)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _parse_recall_ks(value: str) -> tuple[int, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("at least one recall-k value required")
    return tuple(int(p) for p in parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score sparse-attention selection quality against full attention."
    )
    parser.add_argument(
        "--attempt-dir",
        type=Path,
        action="append",
        required=True,
        help="Attempt directory containing recordings/. Pass multiple times to "
        "aggregate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write selection_scores.parquet and selection_summary.md.",
    )
    parser.add_argument(
        "--recall-k",
        type=_parse_recall_ks,
        default=(8, 32, 128),
        help="Comma-separated list of k values for recall@k (default: 8,32,128).",
    )
    args = parser.parse_args(argv)

    attempt_dirs = find_attempt_dirs(args.attempt_dir)
    if not attempt_dirs:
        parser.error(f"no attempt dirs found under {args.attempt_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = score_attempts(attempt_dirs=attempt_dirs, recall_ks=args.recall_k)
    df.write_parquet(args.output_dir / "selection_scores.parquet")

    # Reload one meta block for the summary header (assumes all attempts ran
    # with the same sparse config — typical for a sweep).
    meta_block = _read_meta_sparse_block(attempt_dirs[0])
    method_name = str(meta_block.get("method", "sliding"))
    method_params = _sparse_params_from_meta(meta_block)

    per_layer_phase, global_row = summarize(df, args.recall_k)
    md = _render_markdown(
        per_layer_phase=per_layer_phase,
        global_row=global_row,
        recall_ks=args.recall_k,
        method_name=method_name,
        method_params=method_params,
    )
    (args.output_dir / "selection_summary.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\nWrote {args.output_dir / 'selection_scores.parquet'}")
    print(f"Wrote {args.output_dir / 'selection_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
