#!/usr/bin/env python
"""Load one internal recording iteration and run sanity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_trace_calls(trace_path: Path) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") == "action" and row.get("action_type") == "llm_call":
                calls.append(row)
    return calls


def _iter_entry(recordings_dir: Path, call_idx: int) -> tuple[Path, dict[str, Any]]:
    meta_path = recordings_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for item in meta.get("iters", []):
        if int(item.get("call_idx", -1)) == call_idx:
            return recordings_dir / str(item["dir"]), item
    for item in meta.get("orphan_iters", []):
        if int(item.get("call_idx", -1)) == call_idx:
            raise ValueError(
                f"call_idx {call_idx} has no matching llm_call action in trace"
            )
    raise ValueError(f"call_idx {call_idx} not found in {meta_path}")


def _check_attention(attention: Any) -> dict[str, Any]:
    if int(attention["record_layer"].shape[0]) == 0:
        raise ValueError("attention.npz contains no attention records")
    segment_mass = attention["segment_mass"]
    if segment_mass.size:
        row_sums = segment_mass.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-3):
            worst = float(np.max(np.abs(row_sums - 1.0)))
            raise ValueError(f"segment_mass row sums differ from 1.0; max={worst}")
    return {
        "attention_records": int(attention["record_layer"].shape[0]),
        "attention_rows": int(segment_mass.shape[0]),
        "n_segments": int(attention["n_segments"]),
    }


def _check_routing(routing: Any) -> dict[str, Any]:
    choices = routing["expert_choice"]
    n_experts = int(routing["n_experts"])
    if choices.size and n_experts > 0:
        valid = choices >= 0
        if valid.any() and int(choices[valid].max()) >= n_experts:
            raise ValueError("routing expert index exceeds n_experts")
    return {
        "routing_records": int(routing["record_layer"].shape[0]),
        "n_experts": n_experts,
        "top_k_experts": int(routing["top_k_experts"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--call-idx", type=int, required=True)
    args = parser.parse_args()

    recordings_dir = args.attempt_dir / "recordings"
    call_dir, iter_meta = _iter_entry(recordings_dir, args.call_idx)
    segments = json.loads((call_dir / "segments.json").read_text(encoding="utf-8"))
    token_segment_id = segments.get("token_segment_id") or []
    total_tokens = int(segments["total_tokens"])
    if len(token_segment_id) != total_tokens:
        raise ValueError(
            f"token_segment_id length {len(token_segment_id)} != total_tokens {total_tokens}"
        )

    calls = _load_trace_calls(args.attempt_dir / "trace.jsonl")
    trace_action_id = iter_meta.get("trace_action_id")
    trace_iteration = iter_meta.get("trace_iteration")
    if trace_action_id is not None:
        if not any(call.get("action_id") == trace_action_id for call in calls):
            raise ValueError(f"trace action {trace_action_id!r} not found")
    elif trace_iteration is not None:
        if not any(call.get("iteration") == trace_iteration for call in calls):
            raise ValueError(f"trace iteration {trace_iteration!r} not found")
    elif args.call_idx >= len(calls):
        raise ValueError(
            f"trace has {len(calls)} llm_call actions, no call_idx {args.call_idx}"
        )
    else:
        trace_action_id = calls[args.call_idx].get("action_id")
        trace_iteration = calls[args.call_idx].get("iteration")

    with np.load(call_dir / "attention.npz") as attention:
        attention_summary = _check_attention(attention)
    with np.load(call_dir / "routing.npz") as routing:
        routing_summary = _check_routing(routing)

    summary = {
        "attempt_dir": str(args.attempt_dir),
        "call_idx": args.call_idx,
        "trace_action_id": trace_action_id,
        "trace_iteration": trace_iteration,
        "input_tokens": int(segments["input_tokens"]),
        "output_tokens": int(segments["output_tokens"]),
        "total_tokens": total_tokens,
        **attention_summary,
        **routing_summary,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
