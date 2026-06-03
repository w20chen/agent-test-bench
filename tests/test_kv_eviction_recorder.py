"""Roundtrip tests for `KVEvictionRecorder` npz schema.

Step 2 scaffolding only — exercises the recorder in isolation; eviction
policy subclasses arrive in step 3+.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from serving.kv_policies.recorder import KVEvictionRecorder


def _append_synthetic(recorder: KVEvictionRecorder, n_layers: int, n_steps: int) -> int:
    """Populate `recorder` with `n_layers * n_steps` rows of mixed-shape data.

    Returns total rows appended for the caller to assert against.
    """
    total = 0
    pre_len_base = 32
    budget = 16
    for step in range(n_steps):
        phase = "prefill" if step == 0 else "decode"
        for layer in range(n_layers):
            # Vary the kept/evicted lengths per row so CSR offsets exercise
            # non-uniform widths.
            n_kept = budget if step > 0 else pre_len_base
            n_evicted = max(0, pre_len_base + step - n_kept)
            kept = list(range(n_kept))
            evicted = list(range(n_kept, n_kept + n_evicted))
            pre_len = n_kept + n_evicted
            post_len = pre_len - n_evicted
            # Sprinkle h2o-style scores on every other row so we cover both
            # the present and absent paths.
            include_score = (step + layer) % 2 == 0
            score_index = [layer + step, layer + step + 1] if include_score else None
            score_value = [0.9, 0.1] if include_score else None
            score_evicted_index = evicted[:2] if include_score else None
            score_evicted_value = (
                [0.3 + float(layer), 0.2 + float(step)][: len(score_evicted_index)]
                if score_evicted_index is not None
                else None
            )
            recorder.append(
                step=-1 if step == 0 else step - 1,
                layer=layer,
                phase=phase,
                pre_len=pre_len,
                post_len=post_len,
                budget=budget,
                kept_indices=kept,
                evicted_indices=evicted,
                evict_reason="prefill_compaction" if step == 0 else "over_budget",
                score_topk_index=score_index,
                score_topk_value=score_value,
                score_evicted_index=score_evicted_index,
                score_evicted_value=score_evicted_value,
            )
            total += 1
    return total


def test_recorder_roundtrip(tmp_path) -> None:
    recorder = KVEvictionRecorder(call_idx=7, policy_name="random")
    n_layers, n_steps = 3, 5
    expected_rows = _append_synthetic(recorder, n_layers, n_steps)
    assert recorder.n_records() == expected_rows == 15

    npz_path = tmp_path / "kv_eviction.npz"
    recorder.write(npz_path)

    with np.load(npz_path) as data:
        keys = set(data.keys())
        expected_keys = {
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
        assert expected_keys.issubset(keys), f"missing keys: {expected_keys - keys}"

        assert int(data["call_idx"]) == 7
        assert str(data["policy_name"]) == "random"

        assert data["record_step"].shape == (expected_rows,)
        assert data["record_layer"].shape == (expected_rows,)
        assert data["record_phase"].shape == (expected_rows,)
        assert data["pre_len"].shape == (expected_rows,)
        assert data["post_len"].shape == (expected_rows,)
        assert data["budget"].shape == (expected_rows,)
        assert data["evict_reason"].shape == (expected_rows,)

        assert data["record_step"].dtype == np.int32
        assert data["record_layer"].dtype == np.int32
        assert data["pre_len"].dtype == np.int32
        assert data["post_len"].dtype == np.int32
        assert data["budget"].dtype == np.int32
        assert data["kept_offsets"].dtype == np.int64
        assert data["evicted_offsets"].dtype == np.int64
        assert data["kept_indices"].dtype == np.int32
        assert data["evicted_indices"].dtype == np.int32
        assert data["score_topk_index"].dtype == np.int32
        assert data["score_topk_value"].dtype == np.float32
        assert data["score_evicted_offsets"].dtype == np.int64
        assert data["score_evicted_index"].dtype == np.int32
        assert data["score_evicted_value"].dtype == np.float32
        assert data["record_phase"].dtype == np.dtype("U7")
        assert data["evict_reason"].dtype == np.dtype("U16")

        kept_offsets = data["kept_offsets"]
        evicted_offsets = data["evicted_offsets"]
        assert kept_offsets.shape == (expected_rows + 1,)
        assert evicted_offsets.shape == (expected_rows + 1,)
        assert int(kept_offsets[0]) == 0
        assert int(evicted_offsets[0]) == 0
        assert int(kept_offsets[-1]) == data["kept_indices"].shape[0]
        assert int(evicted_offsets[-1]) == data["evicted_indices"].shape[0]

        pre_len = data["pre_len"]
        post_len = data["post_len"]
        for i in range(expected_rows):
            assert (
                int(pre_len[i]) - int(post_len[i])
                == int(evicted_offsets[i + 1]) - int(evicted_offsets[i])
            ), f"invariant broken at row {i}"

        # h2o sentinel: non-h2o rows are filled with -1 / NaN.
        idx = data["score_topk_index"]
        val = data["score_topk_value"]
        assert idx.shape[0] == expected_rows
        assert val.shape[0] == expected_rows
        # We sprinkled scores on (step + layer) even rows; check at least one
        # missing row stays at sentinels and one present row stays as set.
        present_row = 0  # step=0 layer=0 -> include_score True
        absent_row = 1  # step=0 layer=1 -> include_score False
        assert int(idx[present_row, 0]) == 0
        assert math.isclose(float(val[present_row, 0]), 0.9, rel_tol=1e-6)
        assert int(idx[absent_row, 0]) == -1
        assert math.isnan(float(val[absent_row, 0]))

        score_offsets = data["score_evicted_offsets"]
        assert score_offsets.shape == (expected_rows + 1,)
        assert int(score_offsets[0]) == 0
        assert int(score_offsets[-1]) == data["score_evicted_index"].shape[0]
        assert data["score_evicted_index"].shape == data["score_evicted_value"].shape


def test_recorder_empty_write_behavior(tmp_path) -> None:
    """Empty buffer -> raise (chosen over silent return: an empty
    kv_eviction.npz on disk would be silently misleading downstream)."""
    recorder = KVEvictionRecorder(call_idx=0, policy_name="none")
    assert recorder.n_records() == 0
    with pytest.raises(RuntimeError, match="no records"):
        recorder.write(tmp_path / "kv_eviction.npz")
