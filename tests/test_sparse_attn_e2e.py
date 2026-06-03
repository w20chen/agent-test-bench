"""End-to-end smoke for sparse attention + loader integration.

Drive `HFRecordingProvider.chat()` once with `--sparse-attn sliding` and
once with `--sparse-attn none`. Assert that the sliding run produces a
schema-valid `sparse_attention.npz`, that the loader decodes it, that
density < 1.0 on at least one row (mask did real work), and that the
baseline run leaves no artifact behind.

Marked `slow` because it loads Qwen3-0.6B; matches the kv_eviction e2e.
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


# Same Qwen3-0.6B as test_kv_eviction_e2e.py — smallest in-family model
# with chat_template + q_norm/k_norm.
MODEL = "Qwen/Qwen3-0.6B"
SINK_SIZE = 4
RECENT_WINDOW = 8
MAX_NEW_TOKENS = 16
# Prompt long enough that sliding (sink=4, recent_window=8) genuinely
# masks middle tokens — otherwise mask reduces to dense and we cannot
# detect divergence vs the baseline. ~250 chars / >50 tokens after
# Qwen3 chat templating + system tokens.
LONG_PROMPT = (
    "Explain in detail how transformer attention works, including the "
    "roles of query, key, and value projections, the scaled dot-product, "
    "the softmax normalization step, why we divide by the square root of "
    "the head dimension, and how multi-head attention combines several "
    "parallel projections into a single output."
)


def _build_sparse_config():
    from serving.sparse_attention.base import SparseAttentionConfig

    return SparseAttentionConfig(
        name="sliding",
        sink_size=SINK_SIZE,
        recent_window=RECENT_WINDOW,
        record=True,
    )


async def _run_attempt(
    provider, recordings_dir: Path, prompt: str
) -> tuple[int, str]:
    provider.start_attempt(recordings_dir)
    response = await provider.chat(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_NEW_TOKENS,
        # temperature=0 -> do_sample=False (greedy), making the two runs
        # deterministic so any output divergence is attributable solely
        # to the sparse mask, not to RNG.
        temperature=0.0,
    )
    provider.finish_attempt()
    return int(response.usage.get("completion_tokens", 0)), str(response.content or "")


def _attempt(
    use_sparse: bool, root: Path, prompt: str = LONG_PROMPT
) -> tuple[Path, int, int, str]:
    """Build a fresh provider and run one chat.

    Returns (attempt_dir, n_layers, completion_tokens, generated_text).
    """
    from serving.recording.backend_hf import HFRecordingProvider

    label = "sliding" if use_sparse else "none"
    attempt_dir = root / label
    recordings_dir = attempt_dir / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    sparse_config = _build_sparse_config() if use_sparse else None
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


_SPARSE_NPZ_KEYS = {
    "call_idx",
    "method_name",
    "record_step",
    "record_layer",
    "record_phase",
    "record_decode_step",
    "query_len",
    "key_len",
    "kept_count",
    "density",
    "extras_json",
}


@pytest.mark.slow
def test_sparse_attn_sliding_e2e(tmp_path):
    """Sliding sparse attention writes a schema-valid sparse_attention.npz
    AND produces generation that diverges from the no-sparsity baseline."""
    from scripts.recoding_figures.recording_loader import (
        find_attempt_dirs,
        load_iteration_records,
        load_sparse_attention,
    )

    # Run baseline first (no sparse) and sparse second on the SAME prompt
    # with greedy decoding (temperature=0 -> do_sample=False). Any output
    # divergence is attributable solely to the sliding mask.
    baseline_dir, _b_layers, baseline_completion, baseline_text = _attempt(
        use_sparse=False, root=tmp_path
    )
    assert baseline_completion > 0, "baseline: no tokens generated"

    attempt_dir, num_layers, completion, sparse_text = _attempt(
        use_sparse=True, root=tmp_path
    )
    assert completion > 0, "sliding: no tokens generated"

    iter_dir = attempt_dir / "recordings" / "iter_0000"
    npz_path = iter_dir / "sparse_attention.npz"
    assert npz_path.exists(), f"missing {npz_path}"

    with np.load(npz_path, allow_pickle=True) as data:
        missing = _SPARSE_NPZ_KEYS - set(data.keys())
        assert not missing, f"npz missing keys {missing}"
        assert str(data["method_name"]) == "sliding"
        # Hook fires at least once per layer (prefill); decode adds more rows.
        assert int(data["record_layer"].shape[0]) >= num_layers, (
            f"expected >= {num_layers} rows, got {int(data['record_layer'].shape[0])}"
        )
        # Every layer index in [0, num_layers) should appear.
        observed_layers = set(int(x) for x in data["record_layer"])
        assert observed_layers >= set(range(num_layers)), (
            f"layers {set(range(num_layers)) - observed_layers} never recorded"
        )

    meta_path = attempt_dir / "recordings" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta.get("sparse_attention", {}).get("method") == "sliding"
    integrity = meta["iters"][0]["recording_integrity"]
    assert integrity["sparse_attention_recording_enabled"] is True
    assert integrity["sparse_attention_records"] > 0
    assert (
        integrity["sparse_attention_records"]
        == integrity["sparse_attention_expected_records"]
    )
    assert integrity["sparse_attention_records_match_expected"] is True
    assert integrity["sparse_attention_expected_layers"] == num_layers
    assert integrity["sparse_attention_observed_layers"] == num_layers
    assert (
        integrity["sparse_attention_hook_invocations"]
        == integrity["sparse_attention_records"]
    )
    assert integrity["sparse_attention_hooks_per_layer_min"] > 0
    assert integrity["sparse_attention_hooks_per_layer_max"] >= (
        integrity["sparse_attention_hooks_per_layer_min"]
    )
    assert integrity["sparse_attention_hooks_balanced"] is True

    attempts = find_attempt_dirs([attempt_dir])
    assert attempts == [attempt_dir]
    records = load_iteration_records([attempt_dir])
    assert len(records) == 1

    frame = load_sparse_attention(records)
    assert not frame.is_empty
    assert frame.n_rows >= num_layers
    assert set(frame.method_name.tolist()) == {"sliding"}
    # Sliding with sink+recent < key_len somewhere along the decode tail must
    # produce at least one density < 1.0 row; otherwise the mask is a no-op
    # and we have not exercised the sparsity path.
    densities = frame.density.astype(np.float32)
    assert float(densities.min()) < 1.0, (
        f"all rows had density == 1.0; mask was inactive (min={float(densities.min())})"
    )
    # extras_per_row round-trips compact effective-mask metadata. The static
    # sliding knobs still live once in meta.json; per-row extras carry only
    # query/key-dependent summaries.
    assert "effective_kept_count_sum" in frame.extras_per_row[0]
    assert "effective_density" in frame.extras_per_row[0]

    # Generation divergence: the sliding mask must actually reach the kernel
    # and change the logits enough to flip at least one decoded token vs the
    # greedy baseline on the same prompt. Identical text means the mask was
    # silently a no-op (e.g. mask never applied, mask never causal-cut at
    # prefill, or window covered the whole prompt). Either failure mode is a
    # bug worth surfacing here.
    assert sparse_text != baseline_text, (
        "sparse and baseline runs produced identical generated text on a "
        f"prompt of {len(LONG_PROMPT)} chars; mask appears inactive. "
        f"text={sparse_text!r}"
    )


@pytest.mark.slow
def test_sparse_attn_none_baseline_unchanged(tmp_path):
    """`--sparse-attn none` must not emit sparse_attention.npz or block recording."""
    from scripts.recoding_figures.recording_loader import (
        find_attempt_dirs,
        load_iteration_records,
        load_sparse_attention,
    )

    attempt_dir, _num_layers, completion, _content = _attempt(
        use_sparse=False, root=tmp_path
    )
    assert completion > 0, "baseline: no tokens generated"

    iter_dir = attempt_dir / "recordings" / "iter_0000"
    assert not (iter_dir / "sparse_attention.npz").exists()
    assert (iter_dir / ".done").is_file()

    meta = json.loads(
        (attempt_dir / "recordings" / "meta.json").read_text(encoding="utf-8")
    )
    assert "sparse_attention" not in meta

    attempts = find_attempt_dirs([attempt_dir])
    assert attempts == [attempt_dir]
    records = load_iteration_records([attempt_dir])
    assert len(records) == 1

    frame = load_sparse_attention(records)
    assert frame.is_empty
    assert frame.n_rows == 0
