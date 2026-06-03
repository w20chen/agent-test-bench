from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from serving.recording import RecordingConfig
from serving.recording.hooks import LayerCapturer
from serving.recording.recording import DecodeRingBuffer, segment_bucket
from scripts.recoding_figures.recording_loader import decode_attention_topk


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


class _LayerOnlyCache:
    def __init__(self, key_states: torch.Tensor) -> None:
        self.layers = [SimpleNamespace(keys=key_states)]


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].self_attn = _ToyAttention()
        self.model.layers[0].mlp = nn.Module()
        self.model.layers[0].mlp.gate = nn.Linear(8, 3, bias=False)


def test_decode_ring_buffer_counts_dropped_records() -> None:
    buffer = DecodeRingBuffer(maxlen=2)

    buffer.append({"idx": 0})
    buffer.append({"idx": 1})
    buffer.append({"idx": 2})

    assert buffer.dropped_count() == 1
    assert [item["idx"] for item in buffer.to_list()] == [1, 2]
    buffer.clear()
    assert buffer.dropped_count() == 0
    assert buffer.to_list() == []


def test_layer_capturer_writes_attention_routing_and_segments(tmp_path) -> None:
    model = _ToyModel()
    capturer = LayerCapturer(
        model,
        config=RecordingConfig(attention_top_k=2, decode_window=2, max_prefill_queries=4),
        model_summary={
            "name": "toy",
            "num_experts_per_tok": 2,
        },
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
        generation={
            "seed": 123,
            "do_sample": True,
            "temperature": 0.1,
            "top_p": None,
            "top_k": None,
            "repetition_penalty": None,
        },
    ):
        attn = model.model.layers[0].self_attn
        cos4 = torch.ones(1, 4, 4, dtype=torch.float32)
        sin4 = torch.zeros(1, 4, 4, dtype=torch.float32)
        non_contiguous_hidden = torch.zeros(1, 8, 4).transpose(1, 2)
        attn(
            non_contiguous_hidden,
            position_embeddings=(cos4, sin4),
            past_key_values=_FakeCache(torch.zeros(1, 1, 4, 4)),
        )
        attn(
            torch.zeros(1, 1, 8),
            position_embeddings=(
                torch.ones(1, 1, 4, dtype=torch.float32),
                torch.zeros(1, 1, 4, dtype=torch.float32),
            ),
            past_key_values=_FakeCache(torch.zeros(1, 1, 5, 4)),
        )
        router_logits = (
            torch.tensor(
                [
                    [4.0, 1.0, 0.0],
                    [1.0, 4.0, 0.0],
                    [0.0, 1.0, 4.0],
                    [4.0, 0.0, 1.0],
                    [1.0, 0.0, 4.0],
                ],
                dtype=torch.float32,
            ),
        )
        capturer.record_router_logits(
            SimpleNamespace(router_logits=router_logits),
            total_tokens=5,
        )
        capturer.flush(output_token_ids=[42])
    capturer.set_attempt_extra_meta(
        {
            "session_history": [
                {
                    "call_idx": 0,
                    # Post-`perf(backend_hf)`: session KV cache is default-on
                    # for all configs (sparse / baseline / eviction), so the
                    # first-call audit log records True. Downstream loaders
                    # that previously used this flag to distinguish
                    # eviction-only traces must instead inspect the
                    # `sparse_attention` / `kv_policy` blocks in meta.json or
                    # the presence of `kv_eviction.npz` artifacts.
                    "used_session_cache": True,
                    "lcp": 0,
                    "cached_len_before": 0,
                    "new_len": 4,
                    "delta_len": 4,
                    "diverged": False,
                }
            ]
        }
    )
    capturer.finish_attempt()

    call_dir = recordings_dir / "iter_0000"
    assert (call_dir / "attention.npz").exists()
    assert (call_dir / "routing.npz").exists()
    assert (call_dir / "segments.json").exists()
    assert (call_dir / ".done").exists()

    segments_payload = json.loads((call_dir / "segments.json").read_text())
    assert segments_payload["call_idx"] == 0
    assert segments_payload["complete"] is True
    assert segments_payload["total_tokens"] == 5
    assert len(segments_payload["token_segment_id"]) == 5
    assert segments_payload["segments"][-1]["role"] == "generation"
    assert [segment["first_seen_call"] for segment in segments_payload["segments"]] == [
        0,
        0,
        0,
    ]
    assert [
        segment["first_seen_call_inferred"]
        for segment in segments_payload["segments"]
    ] == [False, False, False]

    with np.load(call_dir / "attention.npz") as attention:
        assert int(attention["call_idx"]) == 0
        assert attention["segment_mass"].shape[1] == 3
        assert attention["segment_mass"].dtype == np.float16
        assert attention["query_heads"].shape == attention["query_positions"].shape
        assert np.allclose(
            attention["segment_mass"].astype(np.float32).sum(axis=1),
            1.0,
            atol=1e-3,
        )
        assert "topk_indices" not in attention.files
        assert "topk_weights" not in attention.files
        assert attention["topk_csr_indices"].dtype == np.int32
        assert attention["topk_csr_weights"].dtype == np.float16
        decoded_indices, decoded_weights = decode_attention_topk(attention)
        assert decoded_indices.shape[1] == 2
        assert decoded_weights.dtype == np.float32
        assert attention["topk_csr_offsets"].shape[0] == int(attention["n_query_rows"]) + 1
        span_span = attention["span_span_matrix"]
        span_span_raw = attention["span_span_matrix_raw"]
        span_row_sums = attention["span_span_row_sums"]
        span_counts = attention["span_span_query_counts"]
        assert span_span.shape == (2, 3, 3)
        assert span_span_raw.shape == (2, 3, 3)
        assert span_row_sums.shape == (2, 3)
        assert span_counts.shape == (2, 3)
        active_rows = span_counts > 0
        assert np.allclose(span_span.sum(axis=-1)[active_rows], 1.0, atol=1e-6)
        assert np.allclose(span_span_raw.sum(axis=-1), span_row_sums, atol=1e-6)
        assert np.any(span_span_raw[active_rows] > span_span[active_rows])
        # head-span fields present with L_s == len(per_head_stats_layers)
        L_s = len(capturer.config.per_head_stats_layers)
        assert attention["head_stats_layers"].shape == (L_s,)
        assert attention["head_span_mean_prefill"].shape[0] == L_s
        assert attention["head_span_var_prefill"].shape[0] == L_s
        assert attention["head_span_mean_decode"].shape[0] == L_s
        assert attention["head_span_var_decode"].shape[0] == L_s
        assert attention["head_span_decode_step"].shape[0] == L_s
        assert attention["head_span_decode_n"].shape == (L_s,)
        assert attention["head_span_query_count"].ndim == 0
        # Fix 2: kept-count sidecar fields have shape (L_s, S) and (L_s, T_max, S)
        kc_prefill = attention["head_span_kept_token_count_prefill"]
        kc_decode = attention["head_span_kept_token_count_decode"]
        assert kc_prefill.ndim == 2 and kc_prefill.shape[0] == L_s
        assert kc_decode.ndim == 3 and kc_decode.shape[0] == L_s
        # S dimension must be consistent across all head-span arrays
        assert kc_prefill.shape[1] == attention["head_span_mean_prefill"].shape[-1]
        assert kc_decode.shape[2] == attention["head_span_mean_prefill"].shape[-1]
        assert kc_decode.shape[1] == attention["head_span_mean_decode"].shape[1]

    with np.load(call_dir / "routing.npz") as routing:
        assert int(routing["n_experts"]) == 3
        assert routing["expert_choice"].shape == (5, 2)
        assert routing["expert_weight"].dtype == np.float16
        assert routing["expert_load"].shape == (1, 3, 3)
        assert routing["record_phase"].tolist() == ["mixed"]
        assert routing["record_decode_step"].tolist() == [-1]
        assert routing["expert_token_count"].shape == (1, 3, 3)
        assert int(routing["expert_token_count"].sum()) == 10
        assert routing["expert_token_count_unit"].tolist() == ["topk_assignments"]
        assert routing["expert_capacity"].tolist() == [4]
        assert routing["drop_signal_mode"].tolist() == ["expected_uniform_capacity"]
        assert int(routing["expected_dropped_token_count"][0]) == int(
            routing["expert_expected_overflow_count"][0].sum()
        )

    meta = json.loads((recordings_dir / "meta.json").read_text())
    assert meta["iters"][0]["call_idx"] == 0
    assert meta["iters"][0]["attention_records"] == 2
    assert meta["iters"][0]["decode_records_dropped_count"] == 0
    # Fix 6: per_head_stats_layers echoed into meta
    assert meta["iters"][0]["per_head_stats_layers"] == list(capturer.config.per_head_stats_layers)
    assert meta["iters"][0]["recording_integrity"] == {
        "complete": True,
        "done_sentinel": True,
        "decode_records_dropped_count": 0,
    }
    assert meta["iters"][0]["generation"] == {
        "seed": 123,
        "do_sample": True,
        "temperature": 0.1,
        "top_p": None,
        "top_k": None,
        "repetition_penalty": None,
    }
    assert meta["session_history"][0]["call_idx"] == 0


def test_layer_capturer_rejects_model_without_attention_modules() -> None:
    model = nn.Linear(2, 2)

    import pytest

    with pytest.raises(ValueError, match="no attention modules"):
        LayerCapturer(
            model,
            config=RecordingConfig(),
            model_summary={"name": "no-attn"},
        )


def test_layer_capturer_records_router_gate_hook_without_second_forward(tmp_path) -> None:
    model = _ToyModel()
    capturer = LayerCapturer(
        model,
        config=RecordingConfig(attention_top_k=2, decode_window=2),
        model_summary={
            "name": "toy",
            "num_experts_per_tok": 2,
        },
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
        model.model.layers[0].mlp.gate(torch.zeros(1, 4, 8))
        model.model.layers[0].mlp.gate(torch.zeros(1, 1, 8))
        capturer.flush(output_token_ids=[42])
    capturer.finish_attempt()

    segments_payload = json.loads(
        (recordings_dir / "iter_0000" / "segments.json").read_text()
    )
    assert [
        segment["first_seen_call_inferred"]
        for segment in segments_payload["segments"]
    ] == [True, False]

    with np.load(recordings_dir / "iter_0000" / "routing.npz") as routing:
        assert routing["record_path"].tolist() == ["gate", "gate"]
        assert routing["record_layer"].tolist() == [0, 0]
        assert routing["record_phase"].tolist() == ["prefill", "decode"]
        assert routing["record_decode_step"].tolist() == [-1, 0]
        assert routing["expert_choice"].shape == (5, 2)
        assert routing["expert_load"].shape == (2, 2, 3)
        assert routing["expert_token_count"].shape == (2, 2, 3)
        assert routing["expert_token_count"].sum(axis=(1, 2)).tolist() == [8, 2]
        assert routing["expert_token_count_unit"].tolist() == [
            "topk_assignments",
            "topk_assignments",
        ]
        assert routing["drop_signal_mode"].tolist() == [
            "expected_uniform_capacity",
            "expected_uniform_capacity",
        ]

    meta = json.loads((recordings_dir / "meta.json").read_text())
    assert meta["model"]["router_capture_mode"] == "gate_forward_hook"


def test_recording_session_persists_generation_metrics_added_before_flush(tmp_path) -> None:
    model = _ToyModel()
    capturer = LayerCapturer(
        model,
        config=RecordingConfig(attention_top_k=2, decode_window=2),
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
        }
    ]
    generation = {
        "seed": 123,
        "do_sample": False,
        "temperature": 0.0,
    }

    with capturer.recording_session(
        call_idx=0,
        segments=segments,
        input_token_count=4,
        generation=generation,
    ):
        model.model.layers[0].self_attn(
            torch.zeros(1, 4, 8),
            position_embeddings=(
                torch.ones(1, 4, 4, dtype=torch.float32),
                torch.zeros(1, 4, 4, dtype=torch.float32),
            ),
            past_key_values=_FakeCache(torch.zeros(1, 1, 4, 4)),
        )
        generation["generate_wall_ms"] = 12.5
        generation["cuda_event_elapsed_ms"] = None
        capturer.flush(output_token_ids=[42])
    capturer.finish_attempt()

    meta = json.loads((recordings_dir / "meta.json").read_text())
    assert meta["iters"][0]["generation"]["generate_wall_ms"] == 12.5
    assert meta["iters"][0]["generation"]["cuda_event_elapsed_ms"] is None
    assert meta["iters"][0]["attention_sampling"] == {
        "prefill_query_sampler": "all_rows",
        "prefill_query_seed": None,
        "configured_max_prefill_queries": 80,
        "effective_max_prefill_queries": 80,
        "unbounded_prefill_queries": False,
        "prefill_query_count": 4,
        "sampled_prefill_queries": 4,
    }


def test_recording_session_persists_seeded_sampling_metadata(tmp_path) -> None:
    model = _ToyModel()
    capturer = LayerCapturer(
        model,
        config=RecordingConfig(
            attention_top_k=2,
            decode_window=2,
            max_prefill_queries=4,
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
            "token_end": 8,
            "has_content": True,
            "has_tool_calls": False,
        }
    ]

    with capturer.recording_session(
        call_idx=3,
        segments=segments,
        input_token_count=8,
    ):
        model.model.layers[0].self_attn(
            torch.zeros(1, 8, 8),
            position_embeddings=(
                torch.ones(1, 8, 4, dtype=torch.float32),
                torch.zeros(1, 8, 4, dtype=torch.float32),
            ),
            past_key_values=_FakeCache(torch.zeros(1, 1, 8, 4)),
        )
        capturer.flush(output_token_ids=[42])
    capturer.finish_attempt()

    meta = json.loads((recordings_dir / "meta.json").read_text())
    assert meta["iters"][0]["attention_sampling"] == {
        "prefill_query_sampler": "stratified_seeded_jitter",
        "prefill_query_seed": "0:3",
        "configured_max_prefill_queries": 4,
        "effective_max_prefill_queries": 4,
        "unbounded_prefill_queries": False,
        "prefill_query_count": 8,
        "sampled_prefill_queries": 4,
    }


def test_recording_session_persists_unbounded_sampling_metadata(tmp_path) -> None:
    model = _ToyModel()
    capturer = LayerCapturer(
        model,
        config=RecordingConfig(
            attention_top_k=2,
            decode_window=2,
            max_prefill_queries=4,
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
            "token_end": 8,
            "has_content": True,
            "has_tool_calls": False,
        }
    ]

    with capturer.recording_session(
        call_idx=3,
        segments=segments,
        input_token_count=8,
    ):
        with capturer.unbounded_prefill_queries():
            model.model.layers[0].self_attn(
                torch.zeros(1, 8, 8),
                position_embeddings=(
                    torch.ones(1, 8, 4, dtype=torch.float32),
                    torch.zeros(1, 8, 4, dtype=torch.float32),
                ),
                past_key_values=_FakeCache(torch.zeros(1, 1, 8, 4)),
            )
        capturer.flush(output_token_ids=[42])
    capturer.finish_attempt()

    meta = json.loads((recordings_dir / "meta.json").read_text())
    assert meta["iters"][0]["attention_sampling"] == {
        "prefill_query_sampler": "all_rows",
        "prefill_query_seed": None,
        "configured_max_prefill_queries": 4,
        "effective_max_prefill_queries": None,
        "unbounded_prefill_queries": True,
        "prefill_query_count": 8,
        "sampled_prefill_queries": 8,
    }


def test_layer_capturer_maps_cached_prefill_gate_rows_to_input_suffix(tmp_path) -> None:
    model = _ToyModel()
    capturer = LayerCapturer(
        model,
        config=RecordingConfig(attention_top_k=2, decode_window=2),
        model_summary={
            "name": "toy",
            "num_experts_per_tok": 2,
        },
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
        },
        {
            "role": "tool_result",
            "message_index": 1,
            "token_start": 2,
            "token_end": 4,
            "has_content": True,
            "has_tool_calls": False,
        },
        {
            "role": "user",
            "message_index": 2,
            "token_start": 4,
            "token_end": 6,
            "has_content": True,
            "has_tool_calls": False,
        },
    ]

    with capturer.recording_session(
        call_idx=0,
        segments=segments,
        input_token_count=6,
    ):
        model.model.layers[0].self_attn(
            torch.zeros(1, 1, 8),
            position_embeddings=(
                torch.ones(1, 1, 4, dtype=torch.float32),
                torch.zeros(1, 1, 4, dtype=torch.float32),
            ),
            past_key_values=_FakeCache(torch.zeros(1, 1, 6, 4)),
        )
        model.model.layers[0].mlp.gate(torch.zeros(1, 1, 8))
        capturer.flush(output_token_ids=[42])
    capturer.finish_attempt()

    with np.load(recordings_dir / "iter_0000" / "routing.npz") as routing:
        assert routing["record_phase"].tolist() == ["prefill"]
        assert routing["record_decode_step"].tolist() == [-1]
        assert routing["expert_token_count"][0].sum(axis=1).tolist() == [0, 0, 2, 0]


def test_record_router_logits_maps_single_cached_prefill_row_to_input_suffix(
    tmp_path,
) -> None:
    model = _ToyModel()
    capturer = LayerCapturer(
        model,
        config=RecordingConfig(attention_top_k=2, decode_window=2),
        model_summary={
            "name": "toy",
            "num_experts_per_tok": 2,
        },
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
        },
        {
            "role": "tool_result",
            "message_index": 1,
            "token_start": 2,
            "token_end": 4,
            "has_content": True,
            "has_tool_calls": False,
        },
        {
            "role": "user",
            "message_index": 2,
            "token_start": 4,
            "token_end": 6,
            "has_content": True,
            "has_tool_calls": False,
        },
    ]

    with capturer.recording_session(
        call_idx=0,
        segments=segments,
        input_token_count=6,
    ):
        model.model.layers[0].self_attn(
            torch.zeros(1, 1, 8),
            position_embeddings=(
                torch.ones(1, 1, 4, dtype=torch.float32),
                torch.zeros(1, 1, 4, dtype=torch.float32),
            ),
            past_key_values=_FakeCache(torch.zeros(1, 1, 6, 4)),
        )
        capturer.record_router_logits(
            SimpleNamespace(router_logits=((torch.zeros(1, 3),),)),
            total_tokens=7,
        )
        capturer.flush(output_token_ids=[42])
    capturer.finish_attempt()

    with np.load(recordings_dir / "iter_0000" / "routing.npz") as routing:
        assert routing["record_phase"].tolist() == ["prefill"]
        assert routing["record_decode_step"].tolist() == [-1]
        assert routing["expert_token_count"][0].sum(axis=1).tolist() == [0, 0, 2, 0]


def test_layer_capturer_clears_records_after_failed_session(tmp_path) -> None:
    model = _ToyModel()
    capturer = LayerCapturer(
        model,
        config=RecordingConfig(attention_top_k=2, decode_window=2),
        model_summary={"name": "toy"},
    )
    capturer.start_attempt(tmp_path / "recordings")
    segments = [
        {
            "role": "user",
            "message_index": 0,
            "token_start": 0,
            "token_end": 4,
            "has_content": True,
            "has_tool_calls": False,
        }
    ]

    import pytest

    with pytest.raises(RuntimeError, match="boom"):
        with capturer.recording_session(
            call_idx=0,
            segments=segments,
            input_token_count=4,
        ):
            model.model.layers[0].self_attn(
                torch.zeros(1, 4, 8),
                position_embeddings=(
                    torch.ones(1, 4, 4, dtype=torch.float32),
                    torch.zeros(1, 4, 4, dtype=torch.float32),
                ),
                past_key_values=_FakeCache(torch.zeros(1, 1, 4, 4)),
            )
            assert capturer._prefill_records
            raise RuntimeError("boom")

    assert capturer._prefill_records == []
    assert len(capturer._decode_records) == 0
    assert capturer._routing_records == []


def test_layer_capturer_aligns_meta_iters_to_trace_actions(tmp_path) -> None:
    model = _ToyModel()
    capturer = LayerCapturer(
        model,
        config=RecordingConfig(attention_top_k=2, decode_window=2),
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
        }
    ]
    attn = model.model.layers[0].self_attn

    for call_idx in range(2):
        with capturer.recording_session(
            call_idx=call_idx,
            segments=segments,
            input_token_count=4,
        ):
            attn(
                torch.zeros(1, 4, 8),
                position_embeddings=(
                    torch.ones(1, 4, 4, dtype=torch.float32),
                    torch.zeros(1, 4, 4, dtype=torch.float32),
                ),
                past_key_values=_FakeCache(torch.zeros(1, 1, 4, 4)),
            )
            capturer.flush(output_token_ids=[42])

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps({"type": "trace_metadata"}) + "\n"
        + json.dumps(
            {
                "type": "action",
                "action_type": "llm_call",
                "action_id": "llm_1",
                "iteration": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    capturer.finish_attempt(trace_path=trace_path)

    meta = json.loads((recordings_dir / "meta.json").read_text())
    assert meta["alignment"] == {
        "trace_path": str(trace_path),
        "trace_llm_actions": 1,
        "recording_iters": 2,
        "aligned_iters": 1,
        "orphan_iters": 1,
        "missing_recording_iters": 0,
        "alignment_source": "iteration",
    }
    assert [item["call_idx"] for item in meta["iters"]] == [0]
    assert meta["iters"][0]["trace_action_id"] == "llm_1"
    assert [item["call_idx"] for item in meta["orphan_iters"]] == [1]
    assert meta["orphan_iters"][0]["trace_alignment_status"] == "orphan_no_llm_action"
    assert (recordings_dir / "iter_0001" / "attention.npz").exists()


def test_layer_capturer_uses_trace_indices_for_non_tail_gaps(tmp_path) -> None:
    model = _ToyModel()
    capturer = LayerCapturer(
        model,
        config=RecordingConfig(attention_top_k=2, decode_window=2),
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
        }
    ]
    attn = model.model.layers[0].self_attn

    for call_idx in range(3):
        with capturer.recording_session(
            call_idx=call_idx,
            segments=segments,
            input_token_count=4,
        ):
            attn(
                torch.zeros(1, 4, 8),
                position_embeddings=(
                    torch.ones(1, 4, 4, dtype=torch.float32),
                    torch.zeros(1, 4, 4, dtype=torch.float32),
                ),
                past_key_values=_FakeCache(torch.zeros(1, 1, 4, 4)),
            )
            capturer.flush(output_token_ids=[42])

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps({"type": "trace_metadata"}),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": "llm_0",
                        "iteration": 0,
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": "llm_2",
                        "iteration": 2,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    capturer.finish_attempt(trace_path=trace_path)

    meta = json.loads((recordings_dir / "meta.json").read_text())
    assert [item["call_idx"] for item in meta["iters"]] == [0, 2]
    assert [item["trace_action_id"] for item in meta["iters"]] == ["llm_0", "llm_2"]
    assert [item["call_idx"] for item in meta["orphan_iters"]] == [1]
    assert meta["alignment"]["aligned_iters"] == 2
    assert meta["alignment"]["orphan_iters"] == 1
    assert meta["alignment"]["missing_recording_iters"] == 0


def test_layer_capturer_flags_trace_calls_without_recording_iter(tmp_path) -> None:
    model = _ToyModel()
    capturer = LayerCapturer(
        model,
        config=RecordingConfig(attention_top_k=2, decode_window=2),
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
        }
    ]
    attn = model.model.layers[0].self_attn

    with capturer.recording_session(
        call_idx=0,
        segments=segments,
        input_token_count=4,
    ):
        attn(
            torch.zeros(1, 4, 8),
            position_embeddings=(
                torch.ones(1, 4, 4, dtype=torch.float32),
                torch.zeros(1, 4, 4, dtype=torch.float32),
            ),
            past_key_values=_FakeCache(torch.zeros(1, 1, 4, 4)),
        )
        capturer.flush(output_token_ids=[42])

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps({"type": "trace_metadata"}),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": "llm_0",
                        "iteration": 0,
                    }
                ),
                json.dumps(
                    {
                        "type": "action",
                        "action_type": "llm_call",
                        "action_id": "llm_1",
                        "iteration": 1,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    capturer.finish_attempt(trace_path=trace_path)

    meta = json.loads((recordings_dir / "meta.json").read_text())
    assert meta["alignment"]["missing_recording_iters"] == 1
    assert meta["missing_recording_iters"] == [
        {
            "call_idx": 1,
            "trace_action_id": "llm_1",
            "trace_iteration": 1,
            "trace_alignment_status": "missing_recording_iter",
        }
    ]


def test_segment_bucket_normalizes_low_precision_rows() -> None:
    attn_rows = torch.tensor([[0.2500, 0.2490, 0.2500, 0.2490]], dtype=torch.bfloat16)
    token_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    segment_mass = segment_bucket(attn_rows, token_ids, n_segments=2)

    assert torch.allclose(segment_mass.sum(dim=1), torch.ones(1), atol=1e-6)


def test_layer_capturer_reads_cache_layer_keys_for_decode(tmp_path) -> None:
    model = _ToyModel()
    capturer = LayerCapturer(
        model,
        config=RecordingConfig(attention_top_k=2, decode_window=2),
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
        }
    ]

    with capturer.recording_session(
        call_idx=0,
        segments=segments,
        input_token_count=4,
    ):
        attn = model.model.layers[0].self_attn
        attn(
            torch.zeros(1, 1, 8),
            position_embeddings=(
                torch.ones(1, 1, 4, dtype=torch.float32),
                torch.zeros(1, 1, 4, dtype=torch.float32),
            ),
            past_key_values=_LayerOnlyCache(torch.zeros(1, 1, 6, 4)),
        )
        capturer.flush(output_token_ids=[42, 43])
    capturer.finish_attempt()

    with np.load(recordings_dir / "iter_0000" / "attention.npz") as attention:
        assert attention["record_phase"].tolist() == ["decode"]
        assert attention["query_positions"].tolist() == [5]
        assert int(attention["n_query_rows"]) == 1
