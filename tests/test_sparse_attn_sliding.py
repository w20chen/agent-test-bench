"""Mask-correctness tests for `SlidingWindowSparseAttention`.

For each (Q, K, sink, recent) we verify:
- mask shape broadcasts to [B, H, Q, K] (key-uniform `[1,1,1,K]` at decode,
  per-row causal `[1,1,Q,K]` at prefill)
- positions inside the sink prefix or recent tail are 0 (modulo causal cut)
- positions outside are -inf
- kept_count helper matches the count of unmasked positions
- the all-attend regime (`sink + recent >= key_len`) yields zero -inf entries
  at decode
- prefill rows are strictly causal: row q has -inf for every k >
  (key_len - query_len) + q
"""

from __future__ import annotations

import pytest
import torch

from serving.sparse_attention.sliding import SlidingWindowSparseAttention


_DEVICE = torch.device("cpu")
_DTYPE = torch.float32


def _kept_positions(sink: int, recent: int, key_len: int) -> set[int]:
    kept: set[int] = set()
    if sink > 0:
        kept.update(range(0, min(sink, key_len)))
    if recent > 0:
        kept.update(range(max(0, key_len - recent), key_len))
    return kept


def _check_mask(method: SlidingWindowSparseAttention, query_len: int, key_len: int) -> None:
    mask = method.build_additive_mask(
        layer_idx=0,
        query_len=query_len,
        key_len=key_len,
        phase="prefill" if query_len > 1 else "decode",
        device=_DEVICE,
        dtype=_DTYPE,
    )
    kept = _kept_positions(method.sink_size, method.recent_window, key_len)
    neg_inf = torch.finfo(_DTYPE).min
    if query_len == 1:
        # Decode: key-uniform `[1,1,1,K]` shape (causal trivially satisfied).
        assert mask.shape == (1, 1, 1, key_len)
        flat = mask[0, 0, 0]
        for k in range(key_len):
            if k in kept:
                assert float(flat[k]) == 0.0, f"k={k} should be kept, got {float(flat[k])}"
            else:
                assert float(flat[k]) == pytest.approx(neg_inf), (
                    f"k={k} should be masked, got {float(flat[k])}"
                )
    else:
        # Prefill: per-row causal `[1,1,Q,K]`. Kept iff (sparsity-kept) AND
        # (k <= absolute query position).
        assert mask.shape == (1, 1, query_len, key_len)
        offset = key_len - query_len
        for q in range(query_len):
            abs_q = offset + q
            for k in range(key_len):
                cell = float(mask[0, 0, q, k])
                if k in kept and k <= abs_q:
                    assert cell == 0.0, (
                        f"q={q}, k={k} (abs_q={abs_q}) should be kept, got {cell}"
                    )
                else:
                    assert cell == pytest.approx(neg_inf), (
                        f"q={q}, k={k} (abs_q={abs_q}) should be masked, got {cell}"
                    )
    assert method.kept_count(key_len) == len(kept)


def test_prefill_basic() -> None:
    method = SlidingWindowSparseAttention(sink_size=4, recent_window=8)
    _check_mask(method, query_len=16, key_len=16)


def test_decode_single_query() -> None:
    method = SlidingWindowSparseAttention(sink_size=2, recent_window=4)
    _check_mask(method, query_len=1, key_len=10)


def test_partial_overlap_long_sequence() -> None:
    method = SlidingWindowSparseAttention(sink_size=4, recent_window=4)
    _check_mask(method, query_len=1, key_len=32)


def test_all_attend_when_window_covers_key() -> None:
    method = SlidingWindowSparseAttention(sink_size=4, recent_window=8)
    key_len = 10  # sink + recent = 12 >= 10
    mask = method.build_additive_mask(
        layer_idx=0,
        query_len=1,
        key_len=key_len,
        phase="decode",
        device=_DEVICE,
        dtype=_DTYPE,
    )
    assert torch.all(mask == 0).item()
    assert method.kept_count(key_len) == key_len


def test_sink_only_no_recent_window() -> None:
    method = SlidingWindowSparseAttention(sink_size=4, recent_window=0)
    _check_mask(method, query_len=1, key_len=16)


def test_recent_only_no_sink() -> None:
    method = SlidingWindowSparseAttention(sink_size=0, recent_window=8)
    _check_mask(method, query_len=1, key_len=16)


def test_constructor_rejects_negative_sink() -> None:
    with pytest.raises(ValueError, match="sink_size >= 0"):
        SlidingWindowSparseAttention(sink_size=-1, recent_window=8)


def test_constructor_rejects_negative_window() -> None:
    with pytest.raises(ValueError, match="recent_window >= 0"):
        SlidingWindowSparseAttention(sink_size=4, recent_window=-1)


def test_constructor_rejects_zero_zero() -> None:
    with pytest.raises(ValueError, match="sink_size \\+ recent_window > 0"):
        SlidingWindowSparseAttention(sink_size=0, recent_window=0)


def test_record_metadata_is_empty() -> None:
    # sink_size / recent_window already live in attempt-level meta.json;
    # per-row duplication in extras_json is pure bloat.
    method = SlidingWindowSparseAttention(sink_size=4, recent_window=16)
    meta = method.record_metadata(layer_idx=3, phase="decode", decode_step=5)
    assert meta == {}


def test_effective_density_decode_matches_key_uniform_density() -> None:
    method = SlidingWindowSparseAttention(sink_size=2, recent_window=4)
    key_len = 10
    expected_kept = method.kept_count(key_len)

    assert method.effective_kept_count_sum(query_len=1, key_len=key_len) == expected_kept
    assert method.effective_density(query_len=1, key_len=key_len) == pytest.approx(
        expected_kept / key_len
    )


def test_effective_density_prefill_includes_causal_cut() -> None:
    method = SlidingWindowSparseAttention(sink_size=2, recent_window=4)
    query_len = key_len = 8
    # Sparse keep set is {0, 1, 4, 5, 6, 7}. Causal rows see:
    # q0: 1, q1: 2, q2: 2, q3: 2, q4: 3, q5: 4, q6: 5, q7: 6.
    expected_sum = 25

    assert method.kept_count(key_len) == 6
    assert method.effective_kept_count_sum(
        query_len=query_len, key_len=key_len
    ) == expected_sum
    assert method.effective_density(
        query_len=query_len, key_len=key_len
    ) == pytest.approx(expected_sum / (query_len * key_len))
    assert method.effective_density(query_len=query_len, key_len=key_len) < (
        method.kept_count(key_len) / key_len
    )


def test_effective_density_all_attend_prefill_still_causal() -> None:
    method = SlidingWindowSparseAttention(sink_size=4, recent_window=8)
    query_len = key_len = 10
    expected_lower_triangle = key_len * (key_len + 1) // 2

    assert method.kept_count(key_len) == key_len
    assert method.effective_kept_count_sum(
        query_len=query_len, key_len=key_len
    ) == expected_lower_triangle
    assert method.effective_density(
        query_len=query_len, key_len=key_len
    ) == pytest.approx(expected_lower_triangle / (query_len * key_len))
    assert method.effective_density(query_len=query_len, key_len=key_len) < 1.0


def test_sliding_mask_is_strictly_causal_at_prefill() -> None:
    """At prefill (Q>1) the additive mask MUST be strictly causal per-row.

    Setting `kwargs["attention_mask"]` non-None disables HF SDPA's implicit
    causal shortcut, so the sparsity mask itself must carry the
    upper-triangular -inf cut. Without it, query row 0 would see the
    `recent_window` future tail — a hindsight leak.
    """
    method = SlidingWindowSparseAttention(sink_size=2, recent_window=4)
    query_len = 8
    key_len = 8
    mask = method.build_additive_mask(
        layer_idx=0,
        query_len=query_len,
        key_len=key_len,
        phase="prefill",
        device=_DEVICE,
        dtype=_DTYPE,
    )
    assert mask.shape == (1, 1, query_len, key_len)
    neg_inf = torch.finfo(_DTYPE).min

    offset = key_len - query_len  # absolute position of query row 0
    for q in range(query_len):
        abs_q = offset + q
        for k in range(key_len):
            cell = float(mask[0, 0, q, k])
            if k > abs_q:
                assert cell == pytest.approx(neg_inf), (
                    f"causality violation: q={q} (abs={abs_q}) attends to "
                    f"future k={k}; cell={cell} (expected -inf)"
                )

    # Row q=0 (abs position offset+0 = 0): only k=0 is both within the sink
    # prefix [0, 2) AND <= 0, so it is the only unmasked cell at row 0.
    row0 = mask[0, 0, 0]
    assert float(row0[0]) == 0.0, "row 0 should keep k=0 (sink ∩ causal)"
    for k in range(1, key_len):
        assert float(row0[k]) == pytest.approx(neg_inf), (
            f"row 0 should mask k={k}; got {float(row0[k])}"
        )


def test_sliding_mask_decode_shape_unchanged() -> None:
    """Q==1 decode keeps the cheap key-uniform `[1,1,1,K]` shape."""
    method = SlidingWindowSparseAttention(sink_size=2, recent_window=4)
    mask = method.build_additive_mask(
        layer_idx=0,
        query_len=1,
        key_len=10,
        phase="decode",
        device=_DEVICE,
        dtype=_DTYPE,
    )
    assert mask.shape == (1, 1, 1, 10)
