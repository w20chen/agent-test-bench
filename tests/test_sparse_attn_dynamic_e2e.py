"""End-to-end smoke for the three dynamic sparse-attention methods.

Pins three things per method (heavy_hitter / block_topk / quest) in observe-only mode:

1. **No-op contract** — greedy generation with observe-only matches the no-sparse
   baseline byte-for-byte. (Per the central observe-only invariant.)

2. **Mask had real work to do** — at least one recorded row has density < 1.0.
   Without this, observe-only is trivially "correct" against a degenerate
   all-dense mask and the no-op contract proves nothing about the method.

3. **Selection actually fired** — the majority of decode rows record
   `selection_reason == "selected"`, NOT a degenerate fallback
   (`sink_recent_no_scores` for heavy_hitter, `phase_dense` / `prefill_dense`
   for any decode_only-scoped method). This is the regression test that
   would have caught the heavy_hitter prefill_observe_mode="sampled" bug
   from reviewer round-2 — without full-prefill scores, every decode row's
   selection_reason was "sink_recent_no_scores" and the method silently
   degraded to sink+recent.

Marked slow: loads Qwen3-0.6B four times (baseline once + three methods).
Matches the cost/scope envelope of `test_sparse_attn_e2e.py`.
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
BUDGET = 16
MAX_NEW_TOKENS = 16
# Long enough that sink+recent (4+8=12) covers a small fraction; the
# methods then have nontrivial "middle band" key space to rank/select
# from, and decode rows should record `selection_reason == "selected"`
# rather than collapsing to sink+recent fallback.
LONG_PROMPT = (
    "Explain in detail how transformer attention works, including the "
    "roles of query, key, and value projections, the scaled dot-product, "
    "the softmax normalization step, why we divide by the square root of "
    "the head dimension, and how multi-head attention combines several "
    "parallel projections into a single output."
)


def _build_sparse_config(method_name: str):
    from serving.sparse_attention.base import SparseAttentionConfig

    return SparseAttentionConfig(
        name=method_name,
        sink_size=SINK_SIZE,
        recent_window=RECENT_WINDOW,
        budget=BUDGET,
        record=True,
        observe_only=True,
    )


async def _run_chat(provider, recordings_dir: Path, prompt: str) -> tuple[int, str]:
    provider.start_attempt(recordings_dir)
    response = await provider.chat(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,  # greedy -> deterministic
    )
    provider.finish_attempt()
    return int(response.usage.get("completion_tokens", 0)), str(response.content or "")


def _attempt(sparse_config, root: Path, label: str) -> tuple[Path, int, int, str]:
    """Build a fresh provider; return (attempt_dir, num_layers, completion, content)."""
    from serving.recording.backend_hf import HFRecordingProvider

    attempt_dir = root / label
    recordings_dir = attempt_dir / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    provider = HFRecordingProvider(
        default_model=MODEL,
        sparse_attention_config=sparse_config,
    )
    num_layers = int(provider.model.config.num_hidden_layers)
    completion, content = asyncio.run(_run_chat(provider, recordings_dir, LONG_PROMPT))
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


@pytest.fixture(scope="module")
def baseline_text(tmp_path_factory) -> str:
    """Greedy generation of LONG_PROMPT with no sparse config; computed once."""
    root = tmp_path_factory.mktemp("dynamic_e2e_baseline")
    _attempt_dir, _layers, completion, content = _attempt(
        sparse_config=None, root=root, label="baseline"
    )
    assert completion > 0, "baseline produced no tokens — bad test setup"
    return content


@pytest.mark.slow
@pytest.mark.parametrize("method_name", ["heavy_hitter", "block_topk", "quest"])
def test_dynamic_method_observe_only_e2e(tmp_path, baseline_text, method_name):
    sparse_config = _build_sparse_config(method_name)
    attempt_dir, num_layers, completion, observed_text = _attempt(
        sparse_config=sparse_config, root=tmp_path, label=method_name
    )
    assert completion > 0, f"{method_name}: produced no tokens"

    # Contract 1: observe-only must not perturb attention -> identical text.
    assert observed_text == baseline_text, (
        f"{method_name}: observe-only changed generated text — pre-hook is not a no-op.\n"
        f"baseline={baseline_text!r}\n"
        f"observed={observed_text!r}"
    )

    iter_dir = attempt_dir / "recordings" / "iter_0000"
    npz_path = iter_dir / "sparse_attention.npz"
    assert npz_path.exists(), f"{method_name}: missing {npz_path}"

    with np.load(npz_path, allow_pickle=True) as data:
        assert str(data["method_name"]) == method_name
        assert int(data["record_layer"].shape[0]) >= num_layers, (
            f"{method_name}: expected >=1 row per layer, got "
            f"{int(data['record_layer'].shape[0])} for {num_layers} layers"
        )

        # Contract 2: at least one row's sparse mask did real work
        # (density < 1.0). Otherwise observe-only's text-equality proves
        # nothing about the no-op contract.
        densities = data["density"].astype(np.float32)
        assert float(densities.min()) < 1.0, (
            f"{method_name}: all rows had density == 1.0; mask had no opportunity "
            f"to disagree from dense (min={float(densities.min())})"
        )

        # Contract 3: majority of decode rows must record selection_reason
        # in {"selected"}, not in fallback set. This is the regression test
        # that would have caught the heavy_hitter prefill_observe_mode="sampled"
        # bug (every decode row would have been "sink_recent_no_scores").
        record_phase = data["record_phase"]
        decode_mask = np.asarray([str(p) == "decode" for p in record_phase])
        if not bool(decode_mask.any()):
            pytest.skip(f"{method_name}: no decode rows recorded (prompt too short)")

        decode_extras = [
            json.loads(str(data["extras_json"][i]))
            for i in range(len(record_phase))
            if decode_mask[i]
        ]
        decode_reasons = [str(e.get("selection_reason", "")) for e in decode_extras]
        n_selected = sum(1 for r in decode_reasons if r == "selected")
        n_decode = len(decode_reasons)
        fraction_selected = n_selected / max(n_decode, 1)
        fallback_reasons = {
            "sink_recent_no_scores",  # heavy_hitter no-score fallback
            "phase_dense",            # decode_only scope misapplied at prefill
            "prefill_dense",          # Q>1 fallback at decode_only methods
            "empty",                  # key_len <= 0
        }
        n_fallback = sum(1 for r in decode_reasons if r in fallback_reasons)
        assert fraction_selected > 0.5, (
            f"{method_name}: only {n_selected}/{n_decode} decode rows had "
            f"selection_reason='selected'; remaining reasons: "
            f"{[r for r in decode_reasons if r != 'selected'][:5]}... "
            f"(fallback count={n_fallback}). The method degraded to fallback — "
            "either the prefill score path didn't accumulate enough state by "
            "decode start, or the method config is wrong for this prompt."
        )

    # meta.json + integrity sanity (one row per layer per call, observe_only=True).
    meta = json.loads(
        (attempt_dir / "recordings" / "meta.json").read_text(encoding="utf-8")
    )
    sparse_block = meta.get("sparse_attention", {})
    assert sparse_block.get("method") == method_name
    assert sparse_block.get("observe_only") is True

    integrity = meta["iters"][0]["recording_integrity"]
    assert integrity["sparse_attention_recording_enabled"] is True
    assert integrity["sparse_attention_observe_only"] is True
    assert integrity["sparse_attention_records_match_expected"] is True
    assert integrity["sparse_attention_hooks_balanced"] is True
