"""Step 6 smoke: HFRecordingProvider + H2OCache end-to-end.

Drives `HFRecordingProvider.chat()` directly (no docker / no MCP) to confirm:

  1. `kv_eviction.npz` lands on disk with `policy_name == "h2o"`.
  2. The score_topk_index/value columns are *populated* (not all -1 / NaN
     sentinels) on rows where eviction actually triggered.
  3. Generated text is at least nominally coherent (printed for human
     inspection — H2O on a 24-token budget is a plumbing-level smoke, not
     a quality bar).
  4. The H2O score buffer for one layer shows a sensible distribution at
     the end of the run (sink-region high, tail high, middle uneven).

Run:
    conda run -n ML python scripts/spikes/step6_h2o_smoke.py
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


# Same Qwen3-0.6B as steps 3-5: smallest in-family model with q_norm/k_norm.
MODEL = "Qwen/Qwen3-0.6B"
BUDGET = 16
SINK_SIZE = 4
RECENT_WINDOW = 8
MAX_NEW_TOKENS = 24


async def _run() -> None:
    eviction_config = EvictionPolicyConfig(
        name="h2o",
        budget=BUDGET,
        sink_size=SINK_SIZE,
        recent_window=RECENT_WINDOW,
        aggregate="sum",
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
            evict_reasons = data["evict_reason"]
            score_topk_index = data["score_topk_index"]
            score_topk_value = data["score_topk_value"]

            kept_offsets = data["kept_offsets"]
            evicted_offsets = data["evicted_offsets"]
            kept_flat = data["kept_indices"]
            evicted_flat = data["evicted_indices"]
            pre_lens = data["pre_len"]

            n_evict_rows = 0
            sample_layer_topk: tuple[int, list[int], list[float]] | None = None
            sample_layer = 0  # picked first non-trivial layer for the diagnostic
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
                    assert evicted == [], f"row {i}: reason=none but evicted={evicted}"
                    assert kept == list(range(pre_len)), f"row {i}: keep mismatch"
                    continue
                if reason == "over_budget":
                    n_evict_rows += 1
                    # H2O guarantees post-eviction length == budget.
                    assert len(kept) == BUDGET, f"row {i}: len(keep)={len(kept)} != budget={BUDGET}"
                    # Sink + recent always kept.
                    sink_set = set(range(SINK_SIZE))
                    recent_set = set(range(pre_len - RECENT_WINDOW, pre_len))
                    assert sink_set.issubset(set(kept)), f"row {i}: missing sink"
                    assert recent_set.issubset(set(kept)), f"row {i}: missing recent"
                    # score_topk for this row should be non-sentinel for the
                    # heavy-hitter slots that fit.
                    n_heavy = BUDGET - SINK_SIZE - RECENT_WINDOW
                    if n_heavy > 0:
                        row_idx = score_topk_index[i, :n_heavy]
                        row_val = score_topk_value[i, :n_heavy]
                        assert int(row_idx.min()) >= 0, (
                            f"row {i}: score_topk_index has -1 sentinel inside heavy slots"
                        )
                        assert not np.isnan(row_val).any(), (
                            f"row {i}: score_topk_value has NaN sentinel inside heavy slots"
                        )
                        # Capture the first eviction we see on a sample layer
                        # for the human-readable diagnostic below.
                        if (
                            sample_layer_topk is None
                            and int(data["record_layer"][i]) == sample_layer
                        ):
                            sample_layer_topk = (
                                int(i),
                                [int(x) for x in row_idx.tolist()],
                                [float(x) for x in row_val.tolist()],
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
    print(f"[smoke] over_budget rows     = {n_evict_rows} / {n_records}")
    if sample_layer_topk is not None:
        row_i, idx_list, val_list = sample_layer_topk
        print(
            f"[smoke] sample layer {sample_layer} row {row_i} score topk: "
            f"index={idx_list}, value={[round(v, 3) for v in val_list]}"
        )

    # Same fusion convention as step3/4: HF generate fuses prefill + first
    # decode, so for max_new_tokens=N the cache sees N forward passes per
    # layer.
    expected_rows = num_layers * completion_tokens
    assert policy_name == "h2o", policy_name
    assert n_records == expected_rows, (n_records, expected_rows)
    assert budget_value == BUDGET
    assert unique_layers == list(range(num_layers))
    print("[smoke] OK -- inspect generated text above for coherence")


if __name__ == "__main__":
    asyncio.run(_run())
