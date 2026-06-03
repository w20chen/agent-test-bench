"""Step 4 smoke: HFRecordingProvider + StreamingLLMCache end-to-end.

Drives `HFRecordingProvider.chat()` directly (bypasses the trace_collect CLI
which needs docker + an MCP config) to confirm a `kv_eviction.npz` lands on
disk with `policy_name == "streaming"` and that every decision row matches
the StreamingLLM keep pattern (`[0..sink-1] ∪ [tail-recent..tail-1]`) — or
records no eviction when the prefix-only call stays under budget.

Run:
    conda run -n ML python scripts/spikes/step4_streaming_smoke.py

Generation-quality note
-----------------------

This is the **naive** StreamingLLM variant — no RoPE re-rotation. The script
prints the first 200 chars of generated text so a human can eyeball whether
the implementation produces coherent output. With a 24-token horizon and
budget=16 the eviction only triggers if the prompt + generation crosses 16
tokens; at small horizons the smoke effectively only sanity-checks the
plumbing, while at larger horizons (or longer prompts) we'd start seeing
the naive RoPE behaviour. Expect coherence here because the cache only
actually evicts a handful of tokens past the threshold and Qwen3 is robust
to small evictions; longer-horizon coherence is a future-work question.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from serving.kv_policies.base import EvictionPolicyConfig  # noqa: E402
from serving.recording.backend_hf import HFRecordingProvider  # noqa: E402


# Same Qwen3-0.6B choice as step3: it is the smallest in-family model with
# the q_norm/k_norm layers LayerCapturer requires.
MODEL = "Qwen/Qwen3-0.6B"
BUDGET = 16
SINK_SIZE = 4
RECENT_WINDOW = BUDGET - SINK_SIZE
MAX_NEW_TOKENS = 24


async def _run() -> None:
    eviction_config = EvictionPolicyConfig(
        name="streaming",
        budget=BUDGET,
        sink_size=SINK_SIZE,
        recent_window=RECENT_WINDOW,
    )
    provider = HFRecordingProvider(
        default_model=MODEL,
        eviction_config=eviction_config,
    )
    num_layers = int(provider.model.config.num_hidden_layers)

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        provider.start_attempt(tmp)
        response = await provider.chat(
            messages=[{"role": "user", "content": "Hello, who are you?"}],
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
        )
        provider.finish_attempt()

        iter_dir = tmp / "iter_0000"
        npz_path = iter_dir / "kv_eviction.npz"
        assert npz_path.exists(), f"missing {npz_path}; iter dir: {sorted(iter_dir.iterdir())}"

        with np.load(npz_path) as data:
            policy_name = str(data["policy_name"])
            n_records = int(data["record_step"].shape[0])
            unique_layers = sorted(set(int(x) for x in data["record_layer"].tolist()))
            phases = sorted(set(str(x) for x in data["record_phase"].tolist()))
            unique_steps = sorted(set(int(x) for x in data["record_step"].tolist()))
            evicted_total = int(data["evicted_indices"].shape[0])
            kept_total = int(data["kept_indices"].shape[0])
            budget_value = int(data["budget"][0])

            kept_offsets = data["kept_offsets"]
            evicted_offsets = data["evicted_offsets"]
            kept_flat = data["kept_indices"]
            evicted_flat = data["evicted_indices"]
            evict_reasons = data["evict_reason"]
            pre_lens = data["pre_len"]

            # Per-row keep-pattern audit.
            evict_rows = 0
            for i in range(n_records):
                kept = kept_flat[
                    int(kept_offsets[i]) : int(kept_offsets[i + 1])
                ].tolist()
                evicted = evicted_flat[
                    int(evicted_offsets[i]) : int(evicted_offsets[i + 1])
                ].tolist()
                pre_len = int(pre_lens[i])
                reason = str(evict_reasons[i])
                if reason == "none":
                    assert evicted == [], f"row {i} reason=none but evicted={evicted}"
                    assert kept == list(range(pre_len)), f"row {i} keep mismatch"
                    continue
                if reason == "over_budget":
                    evict_rows += 1
                    expected_keep = list(range(SINK_SIZE)) + list(
                        range(pre_len - RECENT_WINDOW, pre_len)
                    )
                    expected_evict = list(range(SINK_SIZE, pre_len - RECENT_WINDOW))
                    assert kept == expected_keep, (
                        f"row {i} (pre_len={pre_len}) keep={kept[:6]}... "
                        f"!= expected={expected_keep[:6]}..."
                    )
                    assert evicted == expected_evict, (
                        f"row {i} (pre_len={pre_len}) evict={evicted[:6]}... "
                        f"!= expected={expected_evict[:6]}..."
                    )

    completion_tokens = response.usage.get("completion_tokens", 0)
    text = response.content or ""
    print(f"[smoke] response.content[:200] = {text[:200]!r}")
    print(f"[smoke] completion_tokens     = {completion_tokens}")
    print(f"[smoke] model num_hidden_layers = {num_layers}")
    print(f"[smoke] npz path             = {npz_path}")
    print(f"[smoke] policy_name          = {policy_name}")
    print(f"[smoke] budget / sink / recent = {budget_value} / {SINK_SIZE} / {RECENT_WINDOW}")
    print(f"[smoke] record rows          = {n_records}")
    print(f"[smoke] phases               = {phases}")
    print(f"[smoke] unique record_step   = {unique_steps[:5]}... (len={len(unique_steps)})")
    print(f"[smoke] unique record_layer  = first/last = {unique_layers[0]}/{unique_layers[-1]}")
    print(f"[smoke] kept_indices total   = {kept_total}")
    print(f"[smoke] evicted_indices total= {evicted_total}")
    print(f"[smoke] over_budget rows     = {evict_rows} / {n_records}")

    # Same fusion convention as step3: HF generate fuses prefill + first
    # decode, so for max_new_tokens=N the cache sees N forward passes per
    # layer.
    expected_rows = num_layers * completion_tokens
    assert policy_name == "streaming", policy_name
    assert n_records == expected_rows, (n_records, expected_rows)
    assert budget_value == BUDGET
    assert unique_layers == list(range(num_layers))
    print("[smoke] OK -- inspect generated text above for coherence")


if __name__ == "__main__":
    asyncio.run(_run())
