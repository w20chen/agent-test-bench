"""Roundtrip tests for `SparseAttentionRecorder` npz schema."""

from __future__ import annotations

import json

import numpy as np
import pytest

from serving.sparse_attention.recorder import SparseAttentionRecorder


def _append_synthetic(recorder: SparseAttentionRecorder, n_layers: int, n_steps: int) -> int:
    total = 0
    for step in range(n_steps):
        phase = "prefill" if step == 0 else "decode"
        decode_step = -1 if step == 0 else step - 1
        for layer in range(n_layers):
            key_len = 32 + step
            kept = min(key_len, 4 + step)
            recorder.append(
                step=total,
                layer=layer,
                phase=phase,
                decode_step=decode_step,
                query_len=key_len if step == 0 else 1,
                key_len=key_len,
                kept_count=kept,
                extras={
                    "sink_size": 4,
                    "recent_window": kept - 4,
                    "layer_tag": f"L{layer}",
                },
            )
            total += 1
    return total


def test_recorder_roundtrip(tmp_path) -> None:
    recorder = SparseAttentionRecorder(call_idx=3, method_name="sliding")
    n_layers, n_steps = 4, 3
    expected_rows = _append_synthetic(recorder, n_layers, n_steps)
    assert recorder.n_records() == expected_rows == 12

    npz_path = tmp_path / "sparse_attention.npz"
    recorder.write(npz_path)

    with np.load(npz_path, allow_pickle=True) as data:
        expected_keys = {
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
        assert expected_keys.issubset(set(data.keys()))

        assert int(data["call_idx"]) == 3
        assert str(data["method_name"]) == "sliding"

        assert data["record_step"].shape == (expected_rows,)
        assert data["record_layer"].dtype == np.int32
        assert data["record_phase"].dtype == np.dtype("U7")
        assert data["record_decode_step"].dtype == np.int32
        assert data["query_len"].dtype == np.int32
        assert data["key_len"].dtype == np.int32
        assert data["kept_count"].dtype == np.int32
        assert data["density"].dtype == np.float16

        # Density invariant: kept/key matches.
        kl = data["key_len"].astype(np.float32)
        kc = data["kept_count"].astype(np.float32)
        expected_density = np.where(kl > 0, kc / kl, np.float32(0.0))
        assert np.allclose(
            data["density"].astype(np.float32),
            expected_density.astype(np.float16).astype(np.float32),
        )

        # Extras round-trip via json.loads.
        extras_json = data["extras_json"]
        assert extras_json.shape == (expected_rows,)
        first = json.loads(str(extras_json[0]))
        assert first["sink_size"] == 4
        assert "recent_window" in first
        assert first["layer_tag"] == "L0"


def test_recorder_empty_write_raises(tmp_path) -> None:
    recorder = SparseAttentionRecorder(call_idx=0, method_name="sliding")
    with pytest.raises(RuntimeError, match="no records"):
        recorder.write(tmp_path / "sparse_attention.npz")


def test_recorder_rejects_kept_count_above_key_len() -> None:
    recorder = SparseAttentionRecorder(call_idx=0, method_name="sliding")
    with pytest.raises(ValueError, match="kept_count"):
        recorder.append(
            step=0,
            layer=0,
            phase="prefill",
            decode_step=-1,
            query_len=4,
            key_len=4,
            kept_count=8,
        )


def test_recorder_zero_key_len_density_zero() -> None:
    recorder = SparseAttentionRecorder(call_idx=0, method_name="sliding")
    recorder.append(
        step=0,
        layer=0,
        phase="prefill",
        decode_step=-1,
        query_len=1,
        key_len=0,
        kept_count=0,
    )
    assert recorder.n_records() == 1
