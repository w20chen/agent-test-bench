from __future__ import annotations

import numpy as np
import pytest

from scripts.recoding_figures.recording_loader import decode_attention_topk


def test_decode_attention_topk_decodes_csr_only_schema() -> None:
    attention = {
        "top_k": np.asarray(3, dtype=np.int32),
        "topk_csr_offsets": np.asarray([0, 2, 2, 5], dtype=np.int64),
        "topk_csr_indices": np.asarray([4, 1, 9, 8, 7], dtype=np.int32),
        "topk_csr_weights": np.asarray([0.7, 0.3, 0.5, 0.4, 0.1], dtype=np.float32),
    }

    indices, weights = decode_attention_topk(attention)

    np.testing.assert_array_equal(
        indices,
        np.asarray([[4, 1, -1], [-1, -1, -1], [9, 8, 7]], dtype=np.int32),
    )
    np.testing.assert_allclose(
        weights,
        np.asarray([[0.7, 0.3, 0.0], [0.0, 0.0, 0.0], [0.5, 0.4, 0.1]]),
    )


def test_decode_attention_topk_rejects_wide_csr_row() -> None:
    attention = {
        "top_k": np.asarray(1, dtype=np.int32),
        "topk_csr_offsets": np.asarray([0, 2], dtype=np.int64),
        "topk_csr_indices": np.asarray([4, 1], dtype=np.int32),
        "topk_csr_weights": np.asarray([0.7, 0.3], dtype=np.float32),
    }

    with pytest.raises(ValueError, match="exceeds top_k"):
        decode_attention_topk(attention)


def test_decode_attention_topk_rejects_invalid_csr_values() -> None:
    attention = {
        "top_k": np.asarray(2, dtype=np.int32),
        "topk_csr_offsets": np.asarray([0, 2], dtype=np.int64),
        "topk_csr_indices": np.asarray([-1, 4], dtype=np.int32),
        "topk_csr_weights": np.asarray([0.7, 0.3], dtype=np.float32),
    }
    with pytest.raises(ValueError, match="non-negative"):
        decode_attention_topk(attention)

    attention = {
        "top_k": np.asarray(2, dtype=np.int32),
        "topk_csr_offsets": np.asarray([0, 2], dtype=np.int64),
        "topk_csr_indices": np.asarray([1, 4], dtype=np.int32),
        "topk_csr_weights": np.asarray([0.7, np.nan], dtype=np.float32),
    }
    with pytest.raises(ValueError, match="finite"):
        decode_attention_topk(attention)


def test_decode_attention_topk_accepts_fp16_csr_weights() -> None:
    original_weights = np.asarray(
        [0.7, 0.3, 0.5, 0.4, 0.1], dtype=np.float32
    )
    attention = {
        "top_k": np.asarray(3, dtype=np.int32),
        "topk_csr_offsets": np.asarray([0, 2, 2, 5], dtype=np.int64),
        "topk_csr_indices": np.asarray([4, 1, 9, 8, 7], dtype=np.int32),
        "topk_csr_weights": original_weights.astype(np.float16),
    }

    indices, weights = decode_attention_topk(attention)

    np.testing.assert_array_equal(
        indices,
        np.asarray([[4, 1, -1], [-1, -1, -1], [9, 8, 7]], dtype=np.int32),
    )
    assert weights.dtype == np.float32
    expected = np.asarray(
        [[0.7, 0.3, 0.0], [0.0, 0.0, 0.0], [0.5, 0.4, 0.1]], dtype=np.float32
    )
    np.testing.assert_allclose(weights, expected, atol=1e-3)
