"""Session-shared KV cache regression tests for `HFRecordingProvider`.

The provider keeps a single `BaseEvictionCache` alive for its lifetime so
H2O score buffers, streaming-LLM windows, and eviction state accumulate
across the agent's chat() loop. Three contract tests cover:

1. Legacy parity — one chat() call still produces the same output IDs the
   pre-session-cache path would have, given identical inputs.
2. Cross-call KV reuse — after two chat() calls, the cache contains the
   cumulative token sequence and the second call's delta is short.
3. H2O score accumulation — the score buffer grows monotonically across
   calls; no per-call reset.

We avoid spinning up a real HF model (LayerCapturer hard-wires Qwen3
q_norm/k_norm) and instead exercise the changed seams directly:
`_prepare_session_cache`, `_extend_session_tokens`, `notify_new_call`, plus
the H2OCache's bus observe() path.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from serving.kv_policies.base import EvictionPolicyConfig
from serving.kv_policies.h2o import H2OCache
from serving.recording.attention_bus import AttentionBus
from serving.recording.backend_hf import (
    HFRecordingProvider,
    _generation_metadata,
    _generation_seed,
    _longest_common_prefix,
    _synchronize_cuda_devices,
)
from serving.recording.recording import RecordingConfig
from agents.openclaw.providers.base import LLMResponse


# ---------------------------------------------------------------------------
# Fixtures: build an HFRecordingProvider without loading transformers.
# ---------------------------------------------------------------------------


class _StubModelConfig:
    num_hidden_layers = 2
    max_position_embeddings = 64


class _StubModel:
    def __init__(self) -> None:
        self.config = _StubModelConfig()


class _StubTokenizer:
    """Identity tokenizer: decode then encode round-trips token IDs unchanged."""

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        # Encode each id as a space-separated decimal string.
        return " ".join(str(i) for i in ids)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if not text.strip():
            return []
        return [int(tok) for tok in text.split()]


class _StubCapturer:
    def __init__(self) -> None:
        self.started: list[Path] = []
        self.finish_calls: list[Path | None] = []
        self.closed = False

    def start_attempt(self, recordings_dir: Path) -> None:
        self.started.append(recordings_dir)

    def set_attempt_extra_meta(self, meta: dict) -> None:
        self.extra_meta = dict(meta)

    def finish_attempt(self, trace_path: Path | None = None) -> None:
        self.finish_calls.append(trace_path)

    def close(self) -> None:
        self.closed = True


def _build_provider(eviction_config: EvictionPolicyConfig | None) -> HFRecordingProvider:
    """Hand-construct a provider so we don't load HF weights."""
    provider = HFRecordingProvider.__new__(HFRecordingProvider)
    provider._eviction_config = eviction_config
    provider._sparse_attention_config = None
    provider._sparse_attention = None
    provider._session_cache = None
    provider._session_token_ids = None
    provider._session_history = []
    provider._message_first_seen = []
    provider._attention_bus = AttentionBus()
    provider.model = _StubModel()
    provider.tokenizer = _StubTokenizer()
    provider._torch = torch
    provider.capturer = _StubCapturer()
    return provider


# ---------------------------------------------------------------------------
# LCP primitive.
# ---------------------------------------------------------------------------


def test_lcp_strict_prefix() -> None:
    a = torch.tensor([10, 20, 30], dtype=torch.long)
    b = torch.tensor([10, 20, 30, 40, 50], dtype=torch.long)
    assert _longest_common_prefix(a, b) == 3


def test_generation_seed_is_stable_per_call() -> None:
    assert _generation_seed(100, 0) == 100
    assert _generation_seed(100, 7) == 107


def test_synchronize_cuda_devices_targets_each_visible_device() -> None:
    class _FakeCuda:
        def __init__(self) -> None:
            self.synced: list[int] = []

        def device_count(self) -> int:
            return 2

        def synchronize(self, device_idx: int) -> None:
            self.synced.append(device_idx)

    fake_torch = SimpleNamespace(cuda=_FakeCuda())

    _synchronize_cuda_devices(fake_torch)

    assert fake_torch.cuda.synced == [0, 1]


def test_generation_metadata_records_resolved_sampling_params() -> None:
    meta = _generation_metadata(
        seed=123,
        temperature=0.1,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.05,
    )

    assert meta == {
        "seed": 123,
        "do_sample": True,
        "temperature": 0.1,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.05,
    }


def test_model_summary_records_runtime_versions() -> None:
    provider = _build_provider(None)
    provider.default_model = "stub-model"
    provider.config = RecordingConfig()
    provider._captures_router_logits = False

    summary = provider._model_summary()

    assert summary["name"] == "stub-model"
    assert "torch_version" in summary
    assert "transformers_version" in summary
    assert "accelerate_version" in summary
    assert "torch_cuda_version" in summary
    assert "cuda_available" in summary
    assert "nvidia_driver_version" in summary
    assert "hf_model_commit_hash" in summary


def test_lcp_divergent() -> None:
    a = torch.tensor([10, 20, 99, 40], dtype=torch.long)
    b = torch.tensor([10, 20, 30, 40], dtype=torch.long)
    assert _longest_common_prefix(a, b) == 2


def test_lcp_empty() -> None:
    a = torch.zeros(0, dtype=torch.long)
    b = torch.tensor([1, 2, 3], dtype=torch.long)
    assert _longest_common_prefix(a, b) == 0


# ---------------------------------------------------------------------------
# Test 1: legacy parity — first call passes the full prompt to generate.
# ---------------------------------------------------------------------------


def test_single_chat_equivalent_to_legacy() -> None:
    """First call: prepare returns (full_prompt, True), cache freshly built.

    Pre-session-cache behavior was to build a fresh cache and pass the full
    prompt; the new path matches that on the first call. Regression gate
    against accidentally treating an empty cache as "has prefix" and
    truncating.
    """
    cfg = EvictionPolicyConfig(name="random", budget=8, seed=0)
    provider = _build_provider(cfg)
    prompt = torch.tensor([[5, 6, 7, 8, 9]], dtype=torch.long)
    delta, used = provider._prepare_session_cache(prompt_ids=prompt, call_idx=0)

    assert used is True
    torch.testing.assert_close(delta, prompt)
    # Cache built, token state mirrors the prompt verbatim.
    assert provider._session_cache is not None
    assert provider._session_token_ids is not None
    torch.testing.assert_close(provider._session_token_ids, prompt)
    assert provider._session_history == [
        {
            "call_idx": 0,
            "used_session_cache": True,
            "lcp": 0,
            "cached_len_before": 0,
            "new_len": 5,
            "delta_len": 5,
            "diverged": False,
        }
    ]


def test_no_eviction_still_builds_session_cache() -> None:
    """`eviction_config is None` now also gets a (plain DynamicCache) session
    cache so consecutive chat() calls can resume past_key_values via LCP delta
    prefill. This is the post-`perf(backend_hf)` decoupling — sparse and
    baseline runs benefit from the same cross-call KV reuse as eviction runs.
    """
    from transformers import DynamicCache

    from serving.kv_policies.base import BaseEvictionCache

    provider = _build_provider(eviction_config=None)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    delta, used = provider._prepare_session_cache(prompt_ids=prompt, call_idx=0)
    assert used is True
    torch.testing.assert_close(delta, prompt)
    assert isinstance(provider._session_cache, DynamicCache)
    # Plain DynamicCache — not an eviction subclass.
    assert not isinstance(provider._session_cache, BaseEvictionCache)
    assert provider._session_history == [
        {
            "call_idx": 0,
            "used_session_cache": True,
            "lcp": 0,
            "cached_len_before": 0,
            "new_len": 3,
            "delta_len": 3,
            "diverged": False,
        }
    ]


def test_env_var_disables_session_cache(monkeypatch) -> None:
    """OMC_DISABLE_SESSION_CACHE=1 short-circuits to the legacy path even with
    an eviction policy configured. This is the byte-equality validation escape
    hatch — default is enabled.
    """
    monkeypatch.setenv("OMC_DISABLE_SESSION_CACHE", "1")
    cfg = EvictionPolicyConfig(name="random", budget=8, seed=0)
    provider = _build_provider(cfg)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    delta, used = provider._prepare_session_cache(prompt_ids=prompt, call_idx=0)
    assert used is False
    torch.testing.assert_close(delta, prompt)
    assert provider._session_cache is None
    assert provider._session_history[-1]["used_session_cache"] is False


# ---------------------------------------------------------------------------
# Test 2: consecutive chats share KV state — delta on the second call.
# ---------------------------------------------------------------------------


def test_consecutive_chats_share_kv_state() -> None:
    """Two prepare()s. After call 1 ingests (prompt_1 + generated_1) the
    second call must pass only the delta beyond that cumulative sequence.
    """
    cfg = EvictionPolicyConfig(name="streaming", budget=16, sink_size=2, recent_window=14)
    provider = _build_provider(cfg)

    prompt_1 = torch.tensor([[1, 2, 3]], dtype=torch.long)
    delta_1, used_1 = provider._prepare_session_cache(prompt_ids=prompt_1, call_idx=0)
    assert used_1 is True
    torch.testing.assert_close(delta_1, prompt_1)

    # Simulate post-generate side effect from `_chat_locked`.
    generated_1 = [10, 11]
    provider._extend_session_tokens(prompt_ids=prompt_1, output_ids=generated_1)
    expected_state = torch.tensor([[1, 2, 3, 10, 11]], dtype=torch.long)
    torch.testing.assert_close(provider._session_token_ids, expected_state)

    # Second call: the agent re-includes prior assistant turn + new user turn,
    # so prompt_2 = prompt_1 + generated_1 + new tokens.
    prompt_2 = torch.tensor([[1, 2, 3, 10, 11, 20, 21, 22]], dtype=torch.long)
    delta_2, used_2 = provider._prepare_session_cache(prompt_ids=prompt_2, call_idx=1)
    assert used_2 is True
    # delta = prompt_2[lcp:] where lcp = len(expected_state) = 5
    torch.testing.assert_close(
        delta_2, torch.tensor([[20, 21, 22]], dtype=torch.long)
    )
    assert provider._session_history[-1] == {
        "call_idx": 1,
        "used_session_cache": True,
        "lcp": 5,
        "cached_len_before": 5,
        "new_len": 8,
        "delta_len": 3,
        "diverged": False,
    }
    # Cache instance is the SAME object — not rebuilt.
    assert provider._session_cache is not None
    cache_after = provider._session_cache
    delta_2_again, _ = provider._prepare_session_cache(
        prompt_ids=prompt_2, call_idx=1
    )
    assert provider._session_cache is cache_after
    torch.testing.assert_close(delta_2_again, delta_2)


def test_divergent_prompt_triggers_rebuild() -> None:
    """When LCP < cached_len, drop the cache and rebuild on the new prompt."""
    cfg = EvictionPolicyConfig(name="random", budget=8, seed=0)
    provider = _build_provider(cfg)
    prompt_1 = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    provider._prepare_session_cache(prompt_ids=prompt_1, call_idx=0)
    cache_before = provider._session_cache
    assert cache_before is not None

    # Divergence at position 2.
    prompt_2 = torch.tensor([[1, 2, 99, 5]], dtype=torch.long)
    delta, used = provider._prepare_session_cache(prompt_ids=prompt_2, call_idx=1)
    assert used is True
    torch.testing.assert_close(delta, prompt_2)
    # Fresh cache instance; old one dropped.
    assert provider._session_cache is not None
    assert provider._session_cache is not cache_before
    torch.testing.assert_close(provider._session_token_ids, prompt_2)
    assert provider._session_history[-1]["diverged"] is True
    assert provider._session_history[-1]["lcp"] == 2


def test_start_attempt_resets_session_cache_between_attempts(tmp_path: Path) -> None:
    """A provider can span tasks, but KV state must not."""
    cfg = EvictionPolicyConfig(name="streaming", budget=8, sink_size=1, recent_window=7)
    provider = _build_provider(cfg)
    provider.capturer = _StubCapturer()

    provider.start_attempt(tmp_path / "attempt_1" / "recordings")
    prompt_1 = torch.tensor([[1, 2, 3]], dtype=torch.long)
    provider._prepare_session_cache(prompt_ids=prompt_1, call_idx=0)
    provider._extend_session_tokens(prompt_ids=prompt_1, output_ids=[10])
    assert provider._session_cache is not None
    assert provider._session_token_ids is not None

    provider.start_attempt(tmp_path / "attempt_2" / "recordings")
    assert provider._session_cache is None
    assert provider._session_token_ids is None
    assert provider._session_history == []
    assert provider._message_first_seen == []

    prompt_2 = torch.tensor([[1, 2, 3, 10, 11]], dtype=torch.long)
    delta, used = provider._prepare_session_cache(prompt_ids=prompt_2, call_idx=0)

    assert used is True
    torch.testing.assert_close(delta, prompt_2)
    assert provider._session_history == [
        {
            "call_idx": 0,
            "used_session_cache": True,
            "lcp": 0,
            "cached_len_before": 0,
            "new_len": 5,
            "delta_len": 5,
            "diverged": False,
        }
    ]


def test_provider_exit_defensively_finishes_attempt() -> None:
    provider = _build_provider(None)
    capturer = provider.capturer

    provider.__exit__(None, None, None)

    assert capturer.finish_calls == [None]
    assert capturer.closed is True


def test_hf_recording_provider_rejects_concurrent_chat() -> None:
    async def drive() -> None:
        provider = HFRecordingProvider.__new__(HFRecordingProvider)
        provider._chat_lock = threading.Lock()
        provider.generation = SimpleNamespace(
            top_p=None,
            top_k=None,
            repetition_penalty=None,
        )
        entered = asyncio.Event()
        release = asyncio.Event()

        async def fake_chat_locked(**_kwargs):
            entered.set()
            await release.wait()
            return LLMResponse(content="ok", finish_reason="stop")

        provider._chat_locked = fake_chat_locked

        first = asyncio.create_task(provider.chat(messages=[]))
        await entered.wait()
        second = await provider.chat(messages=[])
        release.set()
        first_response = await first

        assert first_response.content == "ok"
        assert second.finish_reason == "error"
        assert second.extra["error_type"] == "concurrent_request"

    asyncio.run(drive())


def test_hf_recording_provider_wait_until_idle_timeout() -> None:
    provider = HFRecordingProvider.__new__(HFRecordingProvider)
    provider._chat_lock = threading.Lock()
    provider._chat_lock.acquire()

    try:
        with pytest.raises(
            TimeoutError,
            match="timed out waiting for HF recording provider to become idle",
        ):
            provider.wait_until_idle(timeout_s=0.001)
    finally:
        provider._chat_lock.release()


def test_hf_recording_provider_wait_until_idle_blocks_until_release() -> None:
    class _ObservedLock:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self.acquire_entered = threading.Event()

        def hold(self) -> None:
            self._lock.acquire()

        def acquire(self, *args, **kwargs) -> bool:
            self.acquire_entered.set()
            return self._lock.acquire(*args, **kwargs)

        def release(self) -> None:
            self._lock.release()

    provider = HFRecordingProvider.__new__(HFRecordingProvider)
    observed_lock = _ObservedLock()
    observed_lock.hold()
    provider._chat_lock = observed_lock
    done = threading.Event()
    errors: list[BaseException] = []

    def wait_for_idle() -> None:
        try:
            provider.wait_until_idle(timeout_s=1.0)
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    waiter = threading.Thread(target=wait_for_idle)
    waiter.start()
    assert observed_lock.acquire_entered.wait(timeout=1.0)
    assert not done.is_set()
    time.sleep(0.01)
    assert not done.is_set()

    observed_lock.release()
    waiter.join(timeout=1.0)

    assert done.is_set()
    assert errors == []


def test_message_first_seen_tracks_appends_and_replacements() -> None:
    """Message provenance survives appended turns and resets changed turns."""
    provider = _build_provider(eviction_config=None)

    first = provider._first_seen_calls_for_messages(
        [{"role": "user", "content": "a"}],
        call_idx=0,
    )
    second = provider._first_seen_calls_for_messages(
        [
            {"role": "user", "content": "a"},
            {"role": "tool", "content": "result"},
        ],
        call_idx=1,
    )
    replaced = provider._first_seen_calls_for_messages(
        [{"role": "user", "content": "changed"}],
        call_idx=2,
    )

    assert first == {0: 0}
    assert second == {0: 0, 1: 1}
    assert replaced == {0: 2}


def test_message_first_seen_survives_context_window_snip() -> None:
    provider = _build_provider(eviction_config=None)

    provider._first_seen_calls_for_messages(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "a"},
            {"role": "tool", "content": "result"},
        ],
        call_idx=1,
    )
    snipped = provider._first_seen_calls_for_messages(
        [
            {"role": "system", "content": "sys"},
            {"role": "tool", "content": "result"},
        ],
        call_idx=3,
    )

    assert snipped == {0: 1, 1: 1}


def test_message_first_seen_distinguishes_repeated_messages() -> None:
    provider = _build_provider(eviction_config=None)

    provider._first_seen_calls_for_messages(
        [{"role": "user", "content": "continue"}],
        call_idx=0,
    )
    repeated = provider._first_seen_calls_for_messages(
        [
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "continue"},
        ],
        call_idx=5,
    )
    snipped_latest = provider._first_seen_calls_for_messages(
        [{"role": "user", "content": "continue"}],
        call_idx=6,
    )

    assert repeated == {0: 0, 1: 5, 2: 5}
    assert snipped_latest == {0: 5}


def test_message_first_seen_uses_openclaw_message_ids_for_duplicate_snips() -> None:
    provider = _build_provider(eviction_config=None)

    provider._first_seen_calls_for_messages(
        [{"role": "user", "content": "continue", "_openclaw_message_id": "m0"}],
        call_idx=0,
    )
    full = provider._first_seen_calls_for_messages(
        [
            {"role": "user", "content": "continue", "_openclaw_message_id": "m0"},
            {"role": "assistant", "content": "ok", "_openclaw_message_id": "m1"},
            {"role": "user", "content": "continue", "_openclaw_message_id": "m2"},
            {"role": "assistant", "content": "hmm", "_openclaw_message_id": "m3"},
            {"role": "user", "content": "continue", "_openclaw_message_id": "m4"},
        ],
        call_idx=5,
    )
    snipped = provider._first_seen_calls_for_messages(
        [
            {"role": "assistant", "content": "ok", "_openclaw_message_id": "m1"},
            {"role": "user", "content": "continue", "_openclaw_message_id": "m2"},
        ],
        call_idx=6,
    )

    assert full == {0: 0, 1: 5, 2: 5, 3: 5, 4: 5}
    assert snipped == {0: 5, 1: 5}


def test_context_manager_releases_provider_refs() -> None:
    provider = _build_provider(eviction_config=None)

    with provider as active:
        assert active is provider

    assert provider.model is None
    assert provider.tokenizer is None


# ---------------------------------------------------------------------------
# Test 3: H2O score buffer accumulates across calls.
# ---------------------------------------------------------------------------


def _attn_with_peak(*, key_len: int, peak_pos: int, value: float = 1.0) -> torch.Tensor:
    """(B=1, H=2, Q=1, K=key_len) attn tensor with mass concentrated at peak."""
    attn = torch.zeros(1, 2, 1, key_len, dtype=torch.float32)
    attn[..., peak_pos] = value
    return attn


def test_h2o_score_buffer_accumulates_across_calls() -> None:
    """Two simulated chat() calls observe attention into the same H2OCache.

    `notify_new_call` resets per-call step counters but the score buffer
    persists. After call 2, every observed key position has score ≥ its
    snapshot from end-of-call-1.
    """
    cfg = EvictionPolicyConfig(
        name="h2o",
        budget=8,
        sink_size=1,
        recent_window=2,
        aggregate="sum",
        prefill_mode="sampled",
    )
    bus = AttentionBus()
    cache = H2OCache(
        cfg,
        num_layers=1,
        attention_bus=bus,
        max_position_embeddings=64,
    )

    # ---- Call 1: prefill across positions 0..3, then decode at 4..5 -------
    cache.notify_new_call(call_idx=0)
    for pos in range(4):
        bus.publish(
            layer=0,
            attn=_attn_with_peak(key_len=pos + 1, peak_pos=pos),
            query_positions=torch.tensor([pos], dtype=torch.long),
            key_len=pos + 1,
            phase="prefill",
            suspended=False,
        )
    snapshot_1 = cache._scores[0].clone()
    assert snapshot_1 is not None
    # Each of positions 0..3 saw mass=1 once; 4..63 still zero.
    assert float(snapshot_1[0]) > 0
    assert float(snapshot_1[3]) > 0
    assert float(snapshot_1[4]) == 0.0

    # ---- Call 2: notify boundary, then more observations -----------------
    cache.notify_new_call(call_idx=1)
    # Counter state cleared, score buffer untouched.
    assert cache._step_counter == {}
    assert cache._seen_prefill == set()
    # Observe new positions 4..5 with strong peaks.
    for pos in range(4, 6):
        bus.publish(
            layer=0,
            attn=_attn_with_peak(key_len=pos + 1, peak_pos=pos, value=5.0),
            query_positions=torch.tensor([pos], dtype=torch.long),
            key_len=pos + 1,
            phase="prefill",
            suspended=False,
        )
    snapshot_2 = cache._scores[0].clone()
    # Live prefix at least covers up to position 5.
    assert cache._score_lengths[0] >= 6
    # Monotone non-decrease vs snapshot_1 across positions 0..5.
    for pos in range(6):
        assert float(snapshot_2[pos]) >= float(snapshot_1[pos]), (
            f"score at pos {pos} regressed: "
            f"{float(snapshot_2[pos])} < {float(snapshot_1[pos])}"
        )
    # Strict growth at the newly-observed positions.
    assert float(snapshot_2[4]) > float(snapshot_1[4])
    assert float(snapshot_2[5]) > float(snapshot_1[5])


def test_h2o_notify_new_call_does_not_unsubscribe() -> None:
    """notify_new_call leaves the bus subscription intact (subscription is
    a provider-lifetime concern, not a per-call one).
    """
    cfg = EvictionPolicyConfig(name="h2o", budget=4, sink_size=1, recent_window=1)
    bus = AttentionBus()
    cache = H2OCache(cfg, num_layers=1, attention_bus=bus, max_position_embeddings=16)
    assert bus.n_consumers() == 1
    cache.notify_new_call(call_idx=42)
    assert bus.n_consumers() == 1


# ---------------------------------------------------------------------------
# Provider lifecycle: close() / __del__ drop the bus subscription.
# ---------------------------------------------------------------------------


def test_close_unsubscribes_h2o_session_cache() -> None:
    cfg = EvictionPolicyConfig(name="h2o", budget=4, sink_size=1, recent_window=1)
    provider = _build_provider(cfg)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    provider._prepare_session_cache(prompt_ids=prompt, call_idx=0)
    assert provider._attention_bus.n_consumers() == 1

    provider.close()
    assert provider._attention_bus.n_consumers() == 0
    assert provider._session_cache is None
    assert provider._session_token_ids is None


def test_close_idempotent() -> None:
    cfg = EvictionPolicyConfig(name="random", budget=4, seed=0)
    provider = _build_provider(cfg)
    provider.close()  # no cache built yet — must not raise.
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    provider._prepare_session_cache(prompt_ids=prompt, call_idx=0)
    provider.close()
    provider.close()  # second close is a no-op.


# ---------------------------------------------------------------------------
# _extend_session_tokens stores raw output_ids (thinking disabled).
# ---------------------------------------------------------------------------


def test_session_extends_with_raw_output_ids() -> None:
    """_extend_session_tokens stores prompt_ids + raw output_ids.

    The raw output_ids are stored directly (no decode→re-encode round-trip).
    For models with thinking tokens (e.g. Qwen3), the stored state may diverge
    from the next call's prompt prefix — _prepare_session_cache handles this
    by rebuilding the cache on divergence. The simpler storage avoids a
    template-specific re-render whose behavior is model-dependent.
    """
    cfg = EvictionPolicyConfig(name="random", budget=8, seed=0)
    provider = _build_provider(cfg)

    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    provider._prepare_session_cache(prompt_ids=prompt, call_idx=0)

    output_ids = [40, 41, 42]
    provider._extend_session_tokens(prompt_ids=prompt, output_ids=output_ids)

    assert provider._session_token_ids is not None
    expected = torch.tensor([[1, 2, 3, 40, 41, 42]], dtype=torch.long)
    torch.testing.assert_close(provider._session_token_ids, expected)
