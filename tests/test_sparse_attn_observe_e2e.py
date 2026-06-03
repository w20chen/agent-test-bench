"""End-to-end smoke for observe-only sparse attention.

The observe-only mode's hard guarantee: it MUST NOT change the attention
computation. The cleanest possible evidence is that running the same prompt
under (a) no sparse config and (b) sparse observe-only sliding produces
**identical** generated text under greedy decoding.

Also asserts the side-channel recording fires correctly: sparse_attention.npz
exists, meta.json reflects observe_only=True, integrity counters line up.

Marked slow because it loads Qwen3-0.6B twice; matches the pattern in
test_sparse_attn_e2e.py.
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


MODEL = "Qwen/Qwen3-0.6B"
SINK_SIZE = 4
RECENT_WINDOW = 8
MAX_NEW_TOKENS = 16
# Long enough that sliding mask, if enforced, would alter logits; we use this
# to make the equality assertion non-trivial. ~360 chars / >60 tokens.
LONG_PROMPT = (
    "Explain in detail how transformer attention works, including the "
    "roles of query, key, and value projections, the scaled dot-product, "
    "the softmax normalization step, why we divide by the square root of "
    "the head dimension, and how multi-head attention combines several "
    "parallel projections into a single output."
)


def _build_observe_config():
    from serving.sparse_attention.base import SparseAttentionConfig

    return SparseAttentionConfig(
        name="sliding",
        sink_size=SINK_SIZE,
        recent_window=RECENT_WINDOW,
        record=True,
        observe_only=True,
    )


async def _run_attempt(provider, recordings_dir, prompt):
    provider.start_attempt(recordings_dir)
    response = await provider.chat(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,  # greedy
    )
    provider.finish_attempt()
    return int(response.usage.get("completion_tokens", 0)), str(response.content or "")


def _attempt(use_observe, root, prompt=LONG_PROMPT):
    from serving.recording.backend_hf import HFRecordingProvider

    label = "observe" if use_observe else "baseline"
    attempt_dir = root / label
    recordings_dir = attempt_dir / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    sparse_config = _build_observe_config() if use_observe else None
    provider = HFRecordingProvider(
        default_model=MODEL,
        sparse_attention_config=sparse_config,
    )
    num_layers = int(provider.model.config.num_hidden_layers)
    completion, content = asyncio.run(_run_attempt(provider, recordings_dir, prompt))
    del provider
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return attempt_dir, num_layers, completion, content


@pytest.mark.slow
def test_observe_only_does_not_change_generation(tmp_path):
    """Greedy generation must match baseline exactly when observe-only is on."""
    baseline_dir, baseline_layers, baseline_completion, baseline_text = _attempt(
        use_observe=False, root=tmp_path
    )
    assert baseline_completion > 0, "baseline produced no tokens"

    observe_dir, observe_layers, observe_completion, observe_text = _attempt(
        use_observe=True, root=tmp_path
    )
    assert observe_completion > 0, "observe-only produced no tokens"
    assert observe_layers == baseline_layers, (
        "layer count drift between runs invalidates the comparison"
    )

    # THE central invariant. observe-only must not perturb attention.
    assert observe_text == baseline_text, (
        "observe-only changed the generated text — pre-hook is not a no-op.\n"
        f"baseline={baseline_text!r}\n"
        f"observe ={observe_text!r}"
    )

    # Side-channel recording landed correctly.
    iter_dir = observe_dir / "recordings" / "iter_0000"
    npz_path = iter_dir / "sparse_attention.npz"
    assert npz_path.exists(), f"missing {npz_path}"
    with np.load(npz_path, allow_pickle=True) as data:
        assert str(data["method_name"]) == "sliding"
        assert int(data["record_layer"].shape[0]) >= observe_layers, (
            "sparse recorder must have at least one row per layer (prefill); "
            f"got {int(data['record_layer'].shape[0])} for {observe_layers} layers"
        )
        # Sliding (sink=4, recent=8) on a ~360-char prompt must mask middle
        # tokens — otherwise sliding degenerates to dense and the
        # baseline==observe equality below proves nothing about observe-only's
        # no-op contract (it would hold trivially for any no-op sparse mask).
        densities = data["density"].astype(np.float32)
        assert float(densities.min()) < 1.0, (
            f"all rows had density == 1.0; sliding had no opportunity to mask "
            f"(min density={float(densities.min())}). Either the prompt was "
            "too short or sink+recent covered the full key range — the e2e "
            "guarantee then degenerates to a tautology."
        )

    meta_path = observe_dir / "recordings" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    sparse_block = meta.get("sparse_attention", {})
    assert sparse_block.get("method") == "sliding"
    assert sparse_block.get("observe_only") is True, (
        f"meta.json must record observe_only=True; got {sparse_block!r}"
    )

    integrity = meta["iters"][0]["recording_integrity"]
    assert integrity["sparse_attention_recording_enabled"] is True
    assert integrity["sparse_attention_observe_only"] is True
    assert integrity["sparse_attention_records"] > 0
    assert (
        integrity["sparse_attention_records"]
        == integrity["sparse_attention_expected_records"]
    )
    assert integrity["sparse_attention_records_match_expected"] is True
    assert integrity["sparse_attention_hooks_balanced"] is True
