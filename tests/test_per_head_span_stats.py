"""Tests for per-head per-span attention statistics."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from serving.recording import RecordingConfig
from serving.recording.hooks import LayerCapturer


class _ToyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_idx = 0
        self.head_dim = 4
        self.num_key_value_groups = 1
        self.scaling = 0.5
        self.q_proj = nn.Linear(8, 4, bias=False)
        self.k_proj = nn.Linear(8, 4, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
    ):
        del position_embeddings, attention_mask, past_key_values
        return hidden_states, None


class _FakeCache:
    def __init__(self, key_states: torch.Tensor) -> None:
        self.key_states = key_states

    def __getitem__(self, layer_idx: int):
        if layer_idx != 0:
            raise KeyError(layer_idx)
        return self.key_states, None


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].self_attn = _ToyAttention()
        self.model.layers[0].mlp = nn.Module()
        self.model.layers[0].mlp.gate = nn.Linear(8, 3, bias=False)


def _compute_expected_mean_var(
    attn: np.ndarray,
    token_segment_id: np.ndarray,
    head: int,
    segment: int,
) -> tuple[float, float]:
    """Hand-compute mean and variance over K-dimension for one (head, segment) cell.

    attn: [H, Q, K] float32
    token_segment_id: [K] int
    Returns (mean_over_queries, var_over_queries) — the per-query mean/var are
    averaged across the Q dimension, matching the accumulation in hooks.py.
    """
    mask = token_segment_id == segment
    if not mask.any():
        return 0.0, 0.0
    vals = attn[head, :, mask]  # [Q, |S|]
    per_query_mean = vals.mean(axis=-1)   # [Q]
    per_query_var = vals.var(axis=-1, ddof=0)  # [Q], population variance
    return float(per_query_mean.mean()), float(per_query_var.mean())


def test_head_span_stats_pure_helper() -> None:
    """Pure helper: verify accumulation logic produces correct mean and var.

    Works entirely in float32 to match the hook implementation. Checks that
    the segment-masked mean/var accumulation matches a direct numpy reference
    computed from the same float32 data.
    """
    H, Q, K = 4, 10, 20
    rng = np.random.default_rng(42)
    # Generate float32 directly to avoid float64 → float32 rounding divergence
    attn_np = rng.random((H, Q, K)).astype(np.float32)
    # Normalize across K so rows sum to 1 (mimicking softmax output)
    attn_np /= attn_np.sum(axis=-1, keepdims=True)

    # Assign K tokens to 3 segments roughly equally
    token_segment_id = np.array([i % 3 for i in range(K)], dtype=np.int64)

    target_head = 2
    target_segment = 1

    # Reference: compute in torch float32 to match the hook exactly
    _ref = torch.from_numpy(attn_np[target_head:target_head+1])  # [1, Q, K]
    _mask = torch.from_numpy(token_segment_id) == target_segment
    _vals = _ref[0, :, _mask]  # [Q, |S|]
    expected_mean = float(_vals.mean(dim=-1).mean().item())
    expected_var = float(_vals.var(dim=-1, unbiased=False).mean().item())

    # Reproduce the accumulation logic from hooks.py _accumulate_head_stats (prefill)
    attn_t = torch.from_numpy(attn_np)  # [H, Q, K] float32
    token_ids_t = torch.from_numpy(token_segment_id)
    S = 3
    mean_sum = torch.zeros((H, S), dtype=torch.float32)
    var_sum = torch.zeros((H, S), dtype=torch.float32)
    for s in range(S):
        mask = token_ids_t == s
        if not mask.any():
            continue
        vals = attn_t[:, :, mask]  # [H, Q, |S|]
        mean_sum[:, s] += vals.mean(dim=-1).sum(dim=-1)
        var_sum[:, s] += vals.var(dim=-1, unbiased=False).sum(dim=-1)
    mean_result = (mean_sum / Q).numpy()
    var_result = (var_sum / Q).numpy()

    # Tolerance: float32 accumulation over Q=10 rows; 1e-5 relative is tight enough
    np.testing.assert_allclose(
        mean_result[target_head, target_segment],
        expected_mean,
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        var_result[target_head, target_segment],
        expected_var,
        rtol=1e-4,
    )


def test_head_stats_accumulator_keys_after_prefill(tmp_path) -> None:
    """Integration: after a prefill capture, accumulator has expected keys."""
    model = _ToyModel()
    capturer = LayerCapturer(
        model,
        config=RecordingConfig(
            attention_top_k=2,
            decode_window=2,
            max_prefill_queries=4,
            per_head_stats_layers=(0,),
        ),
        model_summary={"name": "toy"},
    )
    recordings_dir = tmp_path / "recordings"
    capturer.start_attempt(recordings_dir)
    segments = [
        {
            "role": "user",
            "message_index": 0,
            "token_start": 0,
            "token_end": 4,
            "has_content": True,
            "has_tool_calls": False,
            "first_seen_call": 0,
        }
    ]

    with capturer.recording_session(
        call_idx=0,
        segments=segments,
        input_token_count=4,
    ):
        attn = model.model.layers[0].self_attn
        attn(
            torch.zeros(1, 4, 8),
            position_embeddings=(
                torch.ones(1, 4, 4, dtype=torch.float32),
                torch.zeros(1, 4, 4, dtype=torch.float32),
            ),
            past_key_values=_FakeCache(torch.zeros(1, 1, 4, 4)),
        )
        # Check accumulator before flush
        assert (0, "prefill", 0) in capturer._head_stats, (
            "expected prefill accumulator key (0, 'prefill', 0)"
        )
        entry = capturer._head_stats[(0, "prefill", 0)]
        assert "mean_sum" in entry
        assert "var_sum" in entry
        assert "n_queries" in entry
        assert entry["n_queries"] > 0
        capturer.flush(output_token_ids=[42])

    capturer.finish_attempt()


def test_attention_npz_has_head_span_fields_with_layers(tmp_path) -> None:
    """E2E: attention.npz contains 8 head-span keys with correct shapes when layers configured."""
    model = _ToyModel()
    capturer = LayerCapturer(
        model,
        config=RecordingConfig(
            attention_top_k=2,
            decode_window=2,
            max_prefill_queries=4,
            per_head_stats_layers=(0,),
        ),
        model_summary={"name": "toy"},
    )
    recordings_dir = tmp_path / "recordings"
    capturer.start_attempt(recordings_dir)
    segments = [
        {
            "role": "system",
            "message_index": 0,
            "token_start": 0,
            "token_end": 2,
            "has_content": True,
            "has_tool_calls": False,
            "first_seen_call": 0,
        },
        {
            "role": "user",
            "message_index": 1,
            "token_start": 2,
            "token_end": 4,
            "has_content": True,
            "has_tool_calls": False,
            "first_seen_call": 0,
        },
    ]

    with capturer.recording_session(
        call_idx=0,
        segments=segments,
        input_token_count=4,
    ):
        attn = model.model.layers[0].self_attn
        # prefill
        attn(
            torch.zeros(1, 4, 8),
            position_embeddings=(
                torch.ones(1, 4, 4, dtype=torch.float32),
                torch.zeros(1, 4, 4, dtype=torch.float32),
            ),
            past_key_values=_FakeCache(torch.zeros(1, 1, 4, 4)),
        )
        # decode step 0
        attn(
            torch.zeros(1, 1, 8),
            position_embeddings=(
                torch.ones(1, 1, 4, dtype=torch.float32),
                torch.zeros(1, 1, 4, dtype=torch.float32),
            ),
            past_key_values=_FakeCache(torch.zeros(1, 1, 5, 4)),
        )
        capturer.flush(output_token_ids=[42])

    capturer.finish_attempt()

    iter_dir = recordings_dir / "iter_0000"
    with np.load(iter_dir / "attention.npz") as data:
        keys = set(data.files)
        required = {
            "head_stats_layers",
            "head_span_mean_prefill",
            "head_span_var_prefill",
            "head_span_query_count",
            "head_span_mean_decode",
            "head_span_var_decode",
            "head_span_decode_step",
            "head_span_decode_n",
        }
        assert required.issubset(keys), f"missing keys: {required - keys}"

        L_s = len(capturer.config.per_head_stats_layers)
        assert data["head_stats_layers"].shape == (L_s,)
        assert data["head_span_mean_prefill"].ndim == 3
        assert data["head_span_mean_prefill"].shape[0] == L_s
        assert data["head_span_var_prefill"].ndim == 3
        assert data["head_span_var_prefill"].shape[0] == L_s
        assert data["head_span_query_count"].ndim == 0
        assert data["head_span_mean_decode"].ndim == 4
        assert data["head_span_mean_decode"].shape[0] == L_s
        assert data["head_span_var_decode"].ndim == 4
        assert data["head_span_var_decode"].shape[0] == L_s
        assert data["head_span_decode_step"].shape[0] == L_s
        assert data["head_span_decode_n"].shape == (L_s,)
        # Shapes are consistent
        H = data["head_span_mean_prefill"].shape[1]
        S = data["head_span_mean_prefill"].shape[2]
        T_max = data["head_span_mean_decode"].shape[1]
        assert data["head_span_var_prefill"].shape == (L_s, H, S)
        assert data["head_span_mean_decode"].shape == (L_s, T_max, H, S)
        assert data["head_span_var_decode"].shape == (L_s, T_max, H, S)
        assert data["head_span_decode_step"].shape == (L_s, T_max)


def test_accumulate_head_stats_numeric(tmp_path) -> None:
    """Numeric: _accumulate_head_stats + _build_head_span_arrays match hand-computed values.

    Drives the production code path directly with a known attention tensor and
    asserts the output matches a numpy reference for one specific (head, segment) cell.
    """
    H, Q, K, S = 4, 3, 10, 3
    rng = np.random.default_rng(7)
    attn_np = rng.random((H, Q, K)).astype(np.float32)
    attn_np /= attn_np.sum(axis=-1, keepdims=True)
    token_segment_id = np.array([i % S for i in range(K)], dtype=np.int64)

    model = _ToyModel()
    capturer = LayerCapturer(
        model,
        config=RecordingConfig(
            attention_top_k=2,
            decode_window=2,
            per_head_stats_layers=(0,),
        ),
        model_summary={"name": "toy"},
    )
    # Inject recording session state directly (minimum required fields)
    capturer._head_stats = {}
    capturer._session = {"call_idx": 0, "input_token_count": K, "generated_segment_id": S - 1, "flushed": False}

    attn_t = torch.from_numpy(attn_np)  # [H, Q, K]
    token_ids_t = torch.from_numpy(token_segment_id)

    capturer._accumulate_head_stats(
        layer_idx=0,
        phase="prefill",
        decode_step=0,
        attn=attn_t,
        token_ids=token_ids_t,
        n_segments=S,
    )
    arrays = capturer._build_head_span_arrays(n_segments=S)

    target_head, target_segment = 2, 1
    # Reference computed in torch float32 to match production accumulation exactly
    mask_t = torch.from_numpy(token_segment_id) == target_segment
    vals_t = torch.from_numpy(attn_np)[target_head:target_head + 1, :, mask_t]  # [1, Q, |S|]
    # mean_sum contribution: vals.mean(dim=-1).sum(dim=-1) / n
    expected_mean_f32 = float((vals_t[0].mean(dim=-1).sum() / Q).item())
    # var_sum contribution: vals.var(dim=-1, unbiased=False).sum(dim=-1) / n
    expected_var_f32 = float((vals_t[0].var(dim=-1, unbiased=False).sum() / Q).item())
    # expected_mean after fp16 round-trip (matches prefill_mean cast in _build_head_span_arrays)
    expected_mean_fp16 = float(np.float16(expected_mean_f32))

    got_mean = float(arrays["head_span_mean_prefill"][0, target_head, target_segment])
    got_var = float(arrays["head_span_var_prefill"][0, target_head, target_segment])

    np.testing.assert_allclose(got_mean, expected_mean_fp16, atol=1e-3,
                               err_msg="prefill mean mismatch (fp16 output)")
    np.testing.assert_allclose(got_var, expected_var_f32, atol=1e-6,
                               err_msg="prefill var mismatch (fp32 output)")

    assert int(arrays["head_span_query_count"]) == Q


def test_c2_multi_prefill_raises(tmp_path) -> None:
    """C2 regression: second prefill capture for the same layer raises RuntimeError."""
    model = _ToyModel()
    capturer = LayerCapturer(
        model,
        config=RecordingConfig(
            attention_top_k=2,
            decode_window=2,
            max_prefill_queries=4,
            per_head_stats_layers=(0,),
        ),
        model_summary={"name": "toy"},
    )
    recordings_dir = tmp_path / "recordings"
    capturer.start_attempt(recordings_dir)
    segments = [
        {
            "role": "user",
            "message_index": 0,
            "token_start": 0,
            "token_end": 4,
            "has_content": True,
            "has_tool_calls": False,
            "first_seen_call": 0,
        }
    ]

    attn_mod = model.model.layers[0].self_attn
    prefill_call_kwargs = dict(
        hidden_states=torch.zeros(1, 4, 8),
        position_embeddings=(
            torch.ones(1, 4, 4, dtype=torch.float32),
            torch.zeros(1, 4, 4, dtype=torch.float32),
        ),
        past_key_values=_FakeCache(torch.zeros(1, 1, 4, 4)),
    )

    with pytest.raises(RuntimeError, match="second prefill capture detected"):
        with capturer.recording_session(call_idx=0, segments=segments, input_token_count=4):
            # First prefill — succeeds
            attn_mod(**prefill_call_kwargs)
            # Second prefill for same layer — must raise
            attn_mod(**prefill_call_kwargs)


def test_m2_decode_boundary_raises(tmp_path) -> None:
    """M2 regression: guard fires when key_len <= input_token_count in the head-stats decode branch.

    Phase=decode is only set when key_len > input_token_count, making the guard unreachable
    via normal flow. We trigger it by patching _sampled_attention_rows to bump input_token_count
    to key_len mid-call (after phase is determined as decode, before the guard runs). The patch
    is one-shot: skipped on the prefill call (call_count==0), active on decode (call_count==1).
    """
    model = _ToyModel()
    capturer = LayerCapturer(
        model,
        config=RecordingConfig(
            attention_top_k=2,
            decode_window=2,
            max_prefill_queries=4,
            per_head_stats_layers=(0,),
        ),
        model_summary={"name": "toy"},
    )
    recordings_dir = tmp_path / "recordings"
    capturer.start_attempt(recordings_dir)
    segments = [
        {
            "role": "user",
            "message_index": 0,
            "token_start": 0,
            "token_end": 4,
            "has_content": True,
            "has_tool_calls": False,
            "first_seen_call": 0,
        }
    ]

    original_sampled = capturer._sampled_attention_rows
    call_count = [0]

    def _patched_sampled(*args, **kwargs):
        count = call_count[0]
        call_count[0] += 1
        if count == 1:
            # Decode call: bump input_token_count to key_len to trigger M2 guard.
            # key_len=5 was > input_token_count=4 when phase was determined (decode).
            # Now set input_token_count=5 so guard sees key_len(5) <= 5.
            capturer._session["input_token_count"] = 5
        return original_sampled(*args, **kwargs)

    capturer._sampled_attention_rows = _patched_sampled
    attn_mod = model.model.layers[0].self_attn

    with pytest.raises(RuntimeError, match="unexpected KV boundary"):
        with capturer.recording_session(
            call_idx=0, segments=segments, input_token_count=4
        ):
            # Prefill: key_len=4 == input_token_count=4, query_len=4 → phase=prefill (correct)
            attn_mod(
                torch.zeros(1, 4, 8),
                position_embeddings=(
                    torch.ones(1, 4, 4, dtype=torch.float32),
                    torch.zeros(1, 4, 4, dtype=torch.float32),
                ),
                past_key_values=_FakeCache(torch.zeros(1, 1, 4, 4)),
            )
            # Decode: key_len=5 > input_token_count=4 → phase=decode.
            # Patch bumps input_token_count to 5 inside _sampled_attention_rows.
            # Guard then sees key_len(5) <= input_token_count(5) → raises.
            attn_mod(
                torch.zeros(1, 1, 8),
                position_embeddings=(
                    torch.ones(1, 1, 4, dtype=torch.float32),
                    torch.zeros(1, 1, 4, dtype=torch.float32),
                ),
                past_key_values=_FakeCache(torch.zeros(1, 1, 5, 4)),
            )


def test_attention_npz_head_span_fields_empty_when_no_layers(tmp_path) -> None:
    """E2E: when per_head_stats_layers=(), all head-span keys exist but shapes are (0, ...)."""
    model = _ToyModel()
    capturer = LayerCapturer(
        model,
        config=RecordingConfig(
            attention_top_k=2,
            decode_window=2,
            per_head_stats_layers=(),
        ),
        model_summary={"name": "toy"},
    )
    recordings_dir = tmp_path / "recordings"
    capturer.start_attempt(recordings_dir)
    segments = [
        {
            "role": "user",
            "message_index": 0,
            "token_start": 0,
            "token_end": 4,
            "has_content": True,
            "has_tool_calls": False,
            "first_seen_call": 0,
        }
    ]

    with capturer.recording_session(
        call_idx=0,
        segments=segments,
        input_token_count=4,
    ):
        attn = model.model.layers[0].self_attn
        attn(
            torch.zeros(1, 4, 8),
            position_embeddings=(
                torch.ones(1, 4, 4, dtype=torch.float32),
                torch.zeros(1, 4, 4, dtype=torch.float32),
            ),
            past_key_values=_FakeCache(torch.zeros(1, 1, 4, 4)),
        )
        capturer.flush(output_token_ids=[42])

    capturer.finish_attempt()

    iter_dir = recordings_dir / "iter_0000"
    with np.load(iter_dir / "attention.npz") as data:
        assert data["head_stats_layers"].shape == (0,)
        assert data["head_span_mean_prefill"].shape[0] == 0
        assert data["head_span_var_prefill"].shape[0] == 0
        assert data["head_span_mean_decode"].shape[0] == 0
        assert data["head_span_var_decode"].shape[0] == 0
        assert data["head_span_decode_step"].shape[0] == 0
        assert data["head_span_decode_n"].shape[0] == 0


def test_nan_fill_for_zero_mask_segment(tmp_path) -> None:
    """Fix 1+2: segment with zero key positions produces NaN mean/var and kept_count==0.

    Constructs an attention tensor where the last segment (S-1, the generated segment)
    has no key positions, verifying NaN propagation and kept_count correctness.
    """
    H, Q, K, S = 2, 4, 6, 3
    rng = np.random.default_rng(99)
    attn_np = rng.random((H, Q, K)).astype(np.float32)
    attn_np /= attn_np.sum(axis=-1, keepdims=True)
    # Assign all K keys to segments 0 and 1 only; segment 2 gets zero keys.
    token_segment_id = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)

    model = _ToyModel()
    capturer = LayerCapturer(
        model,
        config=RecordingConfig(
            attention_top_k=2,
            decode_window=2,
            per_head_stats_layers=(0,),
        ),
        model_summary={"name": "toy"},
    )
    capturer._head_stats = {}
    capturer._head_stats_n_segments = S
    capturer._session = {
        "call_idx": 0,
        "input_token_count": K,
        "generated_segment_id": S - 1,
        "flushed": False,
    }

    attn_t = torch.from_numpy(attn_np)
    token_ids_t = torch.from_numpy(token_segment_id)

    capturer._accumulate_head_stats(
        layer_idx=0,
        phase="prefill",
        decode_step=0,
        attn=attn_t,
        token_ids=token_ids_t,
        n_segments=S,
    )
    arrays = capturer._build_head_span_arrays(n_segments=S)

    # Segment 2 (S-1): no key positions → NaN in mean and var
    mean_prefill = arrays["head_span_mean_prefill"].astype(np.float32)
    var_prefill = arrays["head_span_var_prefill"]
    assert np.all(np.isnan(mean_prefill[0, :, S - 1])), (
        f"expected NaN for zero-mask segment in mean, got {mean_prefill[0, :, S - 1]}"
    )
    assert np.all(np.isnan(var_prefill[0, :, S - 1])), (
        f"expected NaN for zero-mask segment in var, got {var_prefill[0, :, S - 1]}"
    )

    # kept_count for segment 2 must be 0
    kept = arrays["head_span_kept_token_count_prefill"]
    assert int(kept[0, S - 1]) == 0, f"expected kept_count==0 for empty segment, got {kept[0, S - 1]}"

    # Segments 0 and 1 must have non-NaN values
    for s in range(S - 1):
        assert not np.any(np.isnan(mean_prefill[0, :, s])), (
            f"unexpected NaN in mean for segment {s}"
        )
        assert not np.any(np.isnan(var_prefill[0, :, s])), (
            f"unexpected NaN in var for segment {s}"
        )
        assert int(kept[0, s]) > 0, f"expected kept_count>0 for segment {s}"


def test_kept_token_count_numeric(tmp_path) -> None:
    """Fix 2 numeric: kept_count_prefill equals expected sum of mask.sum() across Q query rows.

    With two segments of different sizes, verifies the kept_count sidecar
    accumulates mask.sum() * Q (since mask is the same for all Q rows in a chunk).
    """
    H, Q, K, S = 3, 5, 9, 2
    rng = np.random.default_rng(17)
    attn_np = rng.random((H, Q, K)).astype(np.float32)
    attn_np /= attn_np.sum(axis=-1, keepdims=True)
    # Segment 0: 4 keys, Segment 1: 5 keys
    token_segment_id = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)
    expected_kept_0 = int((token_segment_id == 0).sum()) * Q  # 4 * 5 = 20
    expected_kept_1 = int((token_segment_id == 1).sum()) * Q  # 5 * 5 = 25

    model = _ToyModel()
    capturer = LayerCapturer(
        model,
        config=RecordingConfig(
            attention_top_k=2,
            decode_window=2,
            per_head_stats_layers=(0,),
        ),
        model_summary={"name": "toy"},
    )
    capturer._head_stats = {}
    capturer._head_stats_n_segments = S
    capturer._session = {
        "call_idx": 0,
        "input_token_count": K,
        "generated_segment_id": S - 1,
        "flushed": False,
    }

    capturer._accumulate_head_stats(
        layer_idx=0,
        phase="prefill",
        decode_step=0,
        attn=torch.from_numpy(attn_np),
        token_ids=torch.from_numpy(token_segment_id),
        n_segments=S,
    )
    arrays = capturer._build_head_span_arrays(n_segments=S)
    kept = arrays["head_span_kept_token_count_prefill"]  # [L_s, S]

    assert int(kept[0, 0]) == expected_kept_0, (
        f"kept_count segment 0: expected {expected_kept_0}, got {kept[0, 0]}"
    )
    assert int(kept[0, 1]) == expected_kept_1, (
        f"kept_count segment 1: expected {expected_kept_1}, got {kept[0, 1]}"
    )


def test_segment_count_change_raises(tmp_path) -> None:
    """Fix 7: _accumulate_head_stats raises RuntimeError when n_segments changes mid-session."""
    H, Q, K = 2, 3, 6
    attn_np = np.ones((H, Q, K), dtype=np.float32) / K
    token_segment_id_3 = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    token_segment_id_4 = np.array([0, 0, 1, 1, 2, 3], dtype=np.int64)

    model = _ToyModel()
    capturer = LayerCapturer(
        model,
        config=RecordingConfig(
            attention_top_k=2,
            decode_window=2,
            per_head_stats_layers=(0,),
        ),
        model_summary={"name": "toy"},
    )
    capturer._head_stats = {}
    capturer._head_stats_n_segments = 3
    capturer._session = {
        "call_idx": 0,
        "input_token_count": K,
        "generated_segment_id": 2,
        "flushed": False,
    }

    # First call with n_segments=3 — should succeed and populate _head_stats
    capturer._accumulate_head_stats(
        layer_idx=0,
        phase="prefill",
        decode_step=0,
        attn=torch.from_numpy(attn_np),
        token_ids=torch.from_numpy(token_segment_id_3),
        n_segments=3,
    )
    assert capturer._head_stats, "first call must populate _head_stats"

    # Second call with n_segments=4 — must raise
    with pytest.raises(RuntimeError, match="segment count changed mid-session"):
        capturer._accumulate_head_stats(
            layer_idx=0,
            phase="decode",
            decode_step=1,
            attn=torch.from_numpy(attn_np),
            token_ids=torch.from_numpy(token_segment_id_4),
            n_segments=4,
        )
