"""End-to-end smoke for KV eviction policies + figure-loader integration.

Plan G17: drive `HFRecordingProvider.chat()` through three policies in
sequence (random/streaming/h2o), assert that each iter_dir has a
`kv_eviction.npz` with the expected schema, that `meta.json.kv_policy.name`
matches, and that `recording_loader.load_kv_eviction(...)` decodes the
flattened audit frame correctly.

Marked `slow` because each policy spins up Qwen3-0.6B once; total runtime
is roughly 1-3 minutes on CPU. Run with `pytest -m slow`.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Same Qwen3-0.6B as scripts/spikes/step{3,4,6}_*_smoke.py: smallest in-family
# model with chat_template + q_norm/k_norm. Smaller models (e.g. SmolLM-135M)
# diverge from Qwen3 attention layout and trip the LayerCapturer reshape.
MODEL = "Qwen/Qwen3-0.6B"
BUDGET = 16
SINK_SIZE = 4
RECENT_WINDOW = BUDGET - SINK_SIZE
MAX_NEW_TOKENS = 24


# Run all three policies under one provider instantiation isolation block, but
# use module-level fixture so test functions stay readable.


def _build_eviction_config(policy_name: str):
    from serving.kv_policies.base import EvictionPolicyConfig

    return EvictionPolicyConfig(
        name=policy_name,
        budget=BUDGET,
        sink_size=SINK_SIZE,
        recent_window=RECENT_WINDOW,
        seed=0,
        aggregate="sum",
    )


async def _run_attempt(provider, recordings_dir: Path) -> int:
    """Drive one chat call; returns completion_tokens."""
    provider.start_attempt(recordings_dir)
    response = await provider.chat(
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
    )
    provider.finish_attempt()
    return int(response.usage.get("completion_tokens", 0))


def _attempt_for_policy(policy_name: str | None, root: Path) -> tuple[Path, int, int]:
    """Build a fresh provider for `policy_name` and run one chat against it.

    Returns (attempt_dir, num_layers, completion_tokens). `policy_name=None`
    drives the no-eviction path (default DynamicCache). The returned
    `attempt_dir` is the parent of `recordings/`, matching the layout
    `find_attempt_dirs` expects.
    """
    from serving.recording.backend_hf import HFRecordingProvider

    attempt_dir = root / (policy_name or "none")
    recordings_dir = attempt_dir / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    eviction_config = (
        _build_eviction_config(policy_name) if policy_name is not None else None
    )
    provider = HFRecordingProvider(
        default_model=MODEL,
        eviction_config=eviction_config,
    )
    num_layers = int(provider.model.config.num_hidden_layers)
    completion_tokens = asyncio.run(_run_attempt(provider, recordings_dir))
    # Free GPU/CPU before the next provider builds; transformers retains
    # cached weights across providers otherwise.
    del provider
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return attempt_dir, num_layers, completion_tokens


_KV_EVICTION_NPZ_KEYS = {
    "call_idx",
    "policy_name",
    "record_step",
    "record_layer",
    "record_phase",
    "pre_len",
    "post_len",
    "budget",
    "kept_offsets",
    "kept_indices",
    "evicted_offsets",
    "evicted_indices",
    "evict_reason",
    "score_topk_index",
    "score_topk_value",
    "score_evicted_offsets",
    "score_evicted_index",
    "score_evicted_value",
}


@pytest.mark.slow
def test_kv_eviction_e2e_three_policies(tmp_path):
    """Run random/streaming/h2o sequentially under HFRecordingProvider and
    verify the recording_loader decode path round-trips for each policy.
    """
    from scripts.recoding_figures.recording_loader import (
        find_attempt_dirs,
        load_iteration_records,
        load_kv_eviction,
    )

    # Sequential, not parallel: loading Qwen3-0.6B three times concurrently
    # would oversubscribe both GPU memory and the LayerCapturer's hook
    # registry (capturer state is per-provider-instance).
    for policy in ("random", "streaming", "h2o"):
        attempt_dir, num_layers, completion_tokens = _attempt_for_policy(
            policy, tmp_path
        )
        assert completion_tokens > 0, f"{policy}: no tokens generated"

        iter_dir = attempt_dir / "recordings" / "iter_0000"
        npz_path = iter_dir / "kv_eviction.npz"
        assert npz_path.exists(), f"{policy}: missing {npz_path}"

        # Schema check on the raw npz first — this is the writer contract,
        # so a missing key here points at a recorder regression rather than
        # at the loader.
        with np.load(npz_path) as data:
            assert _KV_EVICTION_NPZ_KEYS.issubset(set(data.keys())), (
                f"{policy}: npz missing keys "
                f"{_KV_EVICTION_NPZ_KEYS - set(data.keys())}"
            )
            assert str(data["policy_name"]) == policy

        meta = json.loads(
            (attempt_dir / "recordings" / "meta.json").read_text(encoding="utf-8")
        )
        assert meta.get("kv_policy", {}).get("name") == policy, (
            f"{policy}: meta.json kv_policy.name = "
            f"{meta.get('kv_policy', {}).get('name')!r}"
        )

        attempts = find_attempt_dirs([attempt_dir])
        assert attempts == [attempt_dir], f"{policy}: find_attempt_dirs={attempts}"
        records = load_iteration_records([attempt_dir])
        assert len(records) == 1, f"{policy}: expected 1 iter, got {len(records)}"

        frame = load_kv_eviction(records)
        # HF generate fuses prefill + first decode into one forward pass per
        # layer, so for max_new_tokens=N the cache sees N updates per layer.
        # See scripts/spikes/step6_h2o_smoke.py for the same convention.
        expected_rows = num_layers * completion_tokens
        assert frame.n_rows == expected_rows, (
            f"{policy}: frame.n_rows={frame.n_rows} != {num_layers}*"
            f"{completion_tokens}={expected_rows}"
        )
        # Every record should carry the same policy_name and call_idx=0.
        assert set(frame.policy_name.tolist()) == {policy}
        assert set(int(x) for x in frame.call_idx.tolist()) == {0}
        # CSR per-row decode: lengths align with offsets.
        assert len(frame.kept_per_row) == frame.n_rows
        assert len(frame.evicted_per_row) == frame.n_rows
        for r in range(frame.n_rows):
            assert (
                int(frame.kept_offsets[r + 1]) - int(frame.kept_offsets[r])
                == frame.kept_per_row[r].shape[0]
            )
            assert (
                int(frame.evicted_offsets[r + 1]) - int(frame.evicted_offsets[r])
                == frame.evicted_per_row[r].shape[0]
            )
            # Eviction recorder invariant: pre_len - post_len == #evicted.
            assert (
                int(frame.pre_len[r]) - int(frame.post_len[r])
                == frame.evicted_per_row[r].shape[0]
            )


@pytest.mark.slow
def test_load_kv_eviction_skipped_for_none_policy_recording(tmp_path):
    """Recording with no eviction policy must not produce kv_eviction.npz,
    must still register as a valid recording, and `load_kv_eviction()` must
    return an empty frame instead of raising.
    """
    from scripts.recoding_figures.recording_loader import (
        find_attempt_dirs,
        load_iteration_records,
        load_kv_eviction,
    )

    attempt_dir, _num_layers, _tokens = _attempt_for_policy(None, tmp_path)
    iter_dir = attempt_dir / "recordings" / "iter_0000"
    assert not (iter_dir / "kv_eviction.npz").exists(), (
        "no eviction config => no kv_eviction.npz on disk"
    )

    # Wave 4 completion contract: attention/routing/segments plus .done must
    # register the iter as valid even when kv_eviction.npz is absent.
    assert (iter_dir / ".done").is_file()
    attempts = find_attempt_dirs([attempt_dir])
    assert attempts == [attempt_dir]
    records = load_iteration_records([attempt_dir])
    assert len(records) == 1

    # Must return empty frame, not raise.
    frame = load_kv_eviction(records)
    assert frame.is_empty
    assert frame.n_rows == 0
    assert frame.task.shape == (0,)
    assert frame.kept_offsets.shape == (1,) and int(frame.kept_offsets[0]) == 0
    assert frame.evicted_offsets.shape == (1,) and int(frame.evicted_offsets[0]) == 0
    assert frame.kept_indices.shape == (0,)
    assert frame.evicted_indices.shape == (0,)
    assert frame.score_topk_index.shape == (0, 0)
    assert frame.score_topk_value.shape == (0, 0)
    assert frame.score_evicted_offsets.shape == (1,)
    assert frame.score_evicted_index.shape == (0,)
    assert frame.score_evicted_value.shape == (0,)
