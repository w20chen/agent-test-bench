"""Step 3 smoke: HFRecordingProvider + RandomEvictCache end-to-end.

Bypasses the trace_collect CLI (which needs docker, an MCP config, etc.) and
drives `HFRecordingProvider.chat()` directly to confirm a `kv_eviction.npz`
lands on disk with the right policy_name and per-layer × per-step row count.

Run:
    conda run -n ML python scripts/spikes/step3_random_smoke.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

import numpy as np

# In-tree import.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from serving.kv_policies.base import EvictionPolicyConfig  # noqa: E402
from serving.recording.backend_hf import HFRecordingProvider  # noqa: E402


# LayerCapturer's _project_query_states/_project_key_states call
# `module.q_norm` / `module.k_norm`, which only exist on Qwen3-style
# attention. SmolLM (LLaMA-style) lacks q_norm and crashes the recording
# hook. Qwen3-0.6B is the smallest in-family model that ships with both a
# chat_template and the q_norm/k_norm layers the capturer requires.
MODEL = "Qwen/Qwen3-0.6B"
BUDGET = 16
MAX_NEW_TOKENS = 24


async def _run() -> None:
    eviction_config = EvictionPolicyConfig(name="random", budget=BUDGET, seed=0)
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

    completion_tokens = response.usage.get("completion_tokens", 0)
    print(f"[smoke] response.content[:80] = {(response.content or '')[:80]!r}")
    print(f"[smoke] completion_tokens     = {completion_tokens}")
    print(f"[smoke] model num_hidden_layers = {num_layers}")
    print(f"[smoke] npz path             = {npz_path}")
    print(f"[smoke] policy_name          = {policy_name}")
    print(f"[smoke] budget               = {budget_value}")
    print(f"[smoke] record rows          = {n_records}")
    print(f"[smoke] phases               = {phases}")
    print(f"[smoke] unique record_step   = {unique_steps[:5]}... (len={len(unique_steps)})")
    print(f"[smoke] unique record_layer  = first/last = {unique_layers[0]}/{unique_layers[-1]}")
    print(f"[smoke] kept_indices total   = {kept_total}")
    print(f"[smoke] evicted_indices total= {evicted_total}")

    # Plan §Spike Results: HF generate fuses prefill + first decode, so for
    # max_new_tokens=N the cache sees N forward passes per layer.
    expected_rows = num_layers * completion_tokens
    assert policy_name == "random", policy_name
    assert n_records == expected_rows, (n_records, expected_rows)
    assert budget_value == BUDGET
    assert unique_layers == list(range(num_layers))
    print("[smoke] OK")


if __name__ == "__main__":
    asyncio.run(_run())
