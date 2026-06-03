"""End-to-end smoke for session-preserved KV cache against a real HF model.

Runs two consecutive `HFRecordingProvider.chat()` calls with H2O eviction
and verifies the strict-prefix delta path triggers on call 2, the cache
state grows monotonically, and both calls produce non-empty output.

Lives outside `tests/` because it requires a multi-GB checkpoint and a
GPU (no model cap; defaults to Qwen3-Coder-30B-A3B-Instruct). The slow
e2e under `tests/test_hf_session_cache_e2e.py` covers the same surface
with Qwen3-0.6B for CI.

Usage:
    PYTHONPATH=src python scripts/probes/session_cache_smoke.py \
        --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
        --budget 1024 \
        --out-dir /tmp/probe-session-cache-output

The script writes `report.json` to `--out-dir` and returns non-zero if
any check fails. Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` to avoid
spurious Hub round-trips when running on locked-down clusters.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    p.add_argument("--budget", type=int, default=1024)
    p.add_argument("--sink", type=int, default=4)
    p.add_argument("--recent", type=int, default=256)
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/probe-session-cache-output"))
    p.add_argument("--max-tokens", type=int, default=64)
    return p.parse_args()


async def main(args: argparse.Namespace) -> int:
    from serving.kv_policies.base import EvictionPolicyConfig
    from serving.recording.backend_hf import HFRecordingProvider
    from serving.recording.recording import RecordingConfig

    args.out_dir.mkdir(parents=True, exist_ok=True)

    eviction = EvictionPolicyConfig(
        name="h2o",
        budget=args.budget,
        sink_size=args.sink,
        recent_window=args.recent,
        heavy_ratio=0.5,
        aggregate="sum",
        seed=0,
        record=False,
        prefill_mode="sampled",
    )
    rec = RecordingConfig(
        attention_top_k=8,
        decode_window=4,
        max_prefill_queries=8,
    )

    t0 = time.time()
    print(f"[t+{time.time() - t0:.1f}s] loading provider + model", flush=True)
    provider = HFRecordingProvider(
        default_model=args.model,
        config=rec,
        eviction_config=eviction,
    )
    print(f"[t+{time.time() - t0:.1f}s] provider built", flush=True)
    provider.start_attempt(args.out_dir / "recordings")

    msgs_1 = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Briefly: what is 2+2?"},
    ]
    print(f"[t+{time.time() - t0:.1f}s] chat 1...", flush=True)
    resp_1 = await provider.chat(msgs_1, max_tokens=args.max_tokens, temperature=0.0)
    seq_after_1 = provider._session_cache.get_seq_length(0)
    print(
        f"[t+{time.time() - t0:.1f}s] call 1 done; "
        f"content={resp_1.content[:80]!r}; seq_after_1={seq_after_1}",
        flush=True,
    )

    msgs_2 = msgs_1 + [
        {"role": "assistant", "content": resp_1.content or ""},
        {"role": "user", "content": "Now what is 3+3?"},
    ]
    print(f"[t+{time.time() - t0:.1f}s] chat 2...", flush=True)
    resp_2 = await provider.chat(msgs_2, max_tokens=args.max_tokens, temperature=0.0)
    seq_after_2 = provider._session_cache.get_seq_length(0)
    history = list(provider._session_history)
    print(
        f"[t+{time.time() - t0:.1f}s] call 2 done; "
        f"content={resp_2.content[:80]!r}; seq_after_2={seq_after_2}",
        flush=True,
    )

    checks = {
        "call_1_used_session_cache": history[0]["used_session_cache"],
        "call_1_lcp_zero": history[0]["lcp"] == 0,
        "call_2_used_session_cache": history[1]["used_session_cache"],
        "call_2_strict_prefix": (
            history[1]["lcp"] == history[1]["cached_len_before"]
            and history[1]["lcp"] > 0
        ),
        "call_2_delta_smaller_than_full": (
            (history[1]["new_len"] - history[1]["lcp"]) < history[1]["new_len"]
        ),
        "seq_grew": seq_after_2 > seq_after_1,
        "both_non_empty": (
            bool((resp_1.content or "").strip()) and bool((resp_2.content or "").strip())
        ),
    }
    report = {
        "model": args.model,
        "h2o_budget": args.budget,
        "elapsed_s": time.time() - t0,
        "session_history": history,
        "resp_1_content": resp_1.content,
        "resp_2_content": resp_2.content,
        "seq_after_1": seq_after_1,
        "seq_after_2": seq_after_2,
        "checks": checks,
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))
    print(f"ALL CHECKS PASS: {all(checks.values())}")
    provider.close()
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(_parse_args())))
