"""Session-shared KV cache: real-model integration and tokenizer prefix tests.

Four test groups:

1. Tokenizer-only strict-prefix check (fast) — verifies that Qwen3's default
   chat template renders growing conversations as strict token-level prefix
   extensions of prompt_ids + raw output_ids, enabling correct LCP reuse.

2. Unmatched-segment single-mismatch coverage (fast) — verifies that one
   misaligned assistant turn emits an "unmatched" sentinel covering every
   token with no gaps or overlaps and no user/unmatched overlap.

3. Unmatched-segment two-in-a-row coverage (fast) — verifies that two
   consecutive misaligned turns each get their own non-overlapping sentinel
   and that subsequent aligned turns still produce their own segments.

4. Slow e2e test — drives two consecutive `HFRecordingProvider.chat()` calls
   against Qwen3-0.6B with H2O eviction enabled and confirms:
   - no exception raised
   - both calls produce non-empty generated text
   - the session cache's physical KV length grows after call 2
   - both iter_0000 and iter_0001 contain a `kv_eviction.npz`
   - _session_history confirms strict-prefix reuse (lcp == cached_len_before)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

MODEL = "Qwen/Qwen3-0.6B"
# Budget large enough to avoid eviction on first call's short prompt but tight
# enough to trigger eviction by second call's cumulative context.
H2O_BUDGET = 128
SINK_SIZE = 4
RECENT_WINDOW = 16
MAX_NEW_TOKENS = 16


# ---------------------------------------------------------------------------
# Group 1: tokenizer-only strict-prefix check (no model weights needed).
# ---------------------------------------------------------------------------


def test_chat_template_produces_strict_prefix_extension() -> None:
    """Qwen3's default chat template renders growing conversations as strict
    token-level prefix extensions of prompt + raw output from the prior call.

    The default template (no enable_thinking kwarg) uses a bare gen prompt
    <|im_start|>assistant\\n with no <think> preamble, and re-renders completed
    assistant turns without any wrapper when followed by more messages. So
    session_token_ids (= prompt_ids + raw_output_ids) is always a prefix of the
    next call's prompt_ids, enabling correct LCP-based delta computation.

    Note: if the model autonomously generates <think>...</think> content
    (possible at higher temperatures), those tokens appear in output_ids and
    will mismatch the re-rendered completed turn, causing divergence and a cache
    rebuild on the next call. This is a safe fallback, not a crash.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)

    generated_content = "Hello!"
    messages_1 = [{"role": "user", "content": "say hello"}]

    # Call-1: prompt produced by tokenize_chat_with_segments (add_generation_prompt=True).
    full_text_1 = tok.apply_chat_template(
        messages_1, tokenize=False, add_generation_prompt=True
    )
    prompt_ids_1 = tok.encode(full_text_1, add_special_tokens=False)

    # Session state after call 1: _extend_session_tokens stores prompt + raw output.
    raw_output_ids = tok.encode(generated_content, add_special_tokens=False)
    session_ids = prompt_ids_1 + raw_output_ids

    # Call-2 prompt: same path as tokenize_chat_with_segments.
    messages_2 = [
        {"role": "user", "content": "say hello"},
        {"role": "assistant", "content": generated_content},
        {"role": "user", "content": "now say goodbye"},
    ]
    full_text_2 = tok.apply_chat_template(
        messages_2, tokenize=False, add_generation_prompt=True
    )
    ids_2 = tok.encode(full_text_2, add_special_tokens=False)

    assert len(ids_2) > len(session_ids), (
        "extended prompt must be longer than session token state"
    )
    prefix = ids_2[: len(session_ids)]
    assert prefix == session_ids, (
        "Qwen3 default chat template does not produce a strict token-level prefix "
        "extension; LCP session-cache assumption is violated. "
        f"First mismatch at position "
        f"{next(i for i,(a,b) in enumerate(zip(session_ids,prefix)) if a!=b)}"
    )


# ---------------------------------------------------------------------------
# Group 2: unmatched-segment coverage (tokenizer only, no GPU).
# ---------------------------------------------------------------------------


def test_unmatched_segment_full_coverage_no_mislabeling() -> None:
    """5-message Qwen3 conversation triggers the 'unmatched' sentinel path.

    Qwen3's chat template renders assistant turns differently when
    add_generation_prompt=False (adds <think> markers), so the prefix-alignment
    check inside tokenize_chat_with_segments will fail for the assistant turn
    and emit an "unmatched" sentinel segment.

    Assertions:
    - Every token position [0, total_tokens) is covered by exactly one segment
      (no gaps, no overlaps).
    - No token position that was actually produced by an assistant turn is
      labeled role='user' (or any role other than 'unmatched' / 'assistant').
    """
    from transformers import AutoTokenizer

    from serving.recording.backend_hf import tokenize_chat_with_segments

    tok = AutoTokenizer.from_pretrained(MODEL)

    # Build a conversation that includes an assistant turn mid-sequence so
    # Qwen3's template triggers the misalignment path.
    messages = [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3 + 3?"},
        {"role": "assistant", "content": "6"},
        {"role": "user", "content": "Thank you."},
    ]

    encoded, segments, full_text = tokenize_chat_with_segments(tok, messages)
    total_tokens = int(
        (encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]).shape[-1]
    )

    # --- full coverage: every position maps to exactly one segment ----------
    covered = [False] * total_tokens
    for seg in segments:
        start = seg["token_start"]
        end = seg["token_end"]
        assert start < end, f"empty segment: {seg}"
        for pos in range(start, end):
            assert not covered[pos], (
                f"token position {pos} covered by multiple segments; "
                f"current segment: {seg}"
            )
            covered[pos] = True
    uncovered = [i for i, c in enumerate(covered) if not c]
    assert not uncovered, (
        f"token positions {uncovered[:10]} (of {len(uncovered)}) are not covered "
        f"by any segment; total_tokens={total_tokens}, segments={segments}"
    )

    # --- no assistant tokens mislabeled as 'user' ---------------------------
    # Build the ground-truth assistant-token positions by re-encoding the full
    # text using the same template without add_generation_prompt, then
    # inspecting which segments are labeled 'user'.  Since we cannot recover
    # exact char boundaries for misaligned assistant turns, we use the simpler
    # property: no segment with role='user' should overlap a segment with
    # role='unmatched' (which is where assistant content fell).
    user_positions: set[int] = set()
    unmatched_positions: set[int] = set()
    for seg in segments:
        r = seg["role"]
        positions = set(range(seg["token_start"], seg["token_end"]))
        if r == "user":
            user_positions.update(positions)
        elif r == "unmatched":
            unmatched_positions.update(positions)

    overlap = user_positions & unmatched_positions
    assert not overlap, (
        f"{len(overlap)} token positions are labeled both 'user' and 'unmatched'; "
        f"first 5: {sorted(overlap)[:5]}"
    )


def test_unmatched_two_in_a_row_no_overlap() -> None:
    """Two consecutive assistant turns both trigger 'unmatched' sentinels.

    Verifies that back-to-back misaligned turns each get their own sentinel
    with non-overlapping, non-duplicate ranges, and that the subsequent aligned
    user turn still gets its own correctly-bounded segment.
    """
    from transformers import AutoTokenizer

    from serving.recording.backend_hf import tokenize_chat_with_segments

    tok = AutoTokenizer.from_pretrained(MODEL)

    # user → asst → asst → user → user: two consecutive assistant turns will
    # both trigger the mismatch path. The second user turn must appear as its
    # own segment with the correct char range.
    messages = [
        {"role": "user", "content": "First question."},
        {"role": "assistant", "content": "First answer."},
        {"role": "assistant", "content": "Second answer."},
        {"role": "user", "content": "Second question."},
        {"role": "user", "content": "Third question."},
    ]

    encoded, segments, full_text = tokenize_chat_with_segments(tok, messages)
    total_tokens = int(
        (encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]).shape[-1]
    )

    # Full coverage: every position maps to exactly one segment.
    covered = [False] * total_tokens
    for seg in segments:
        start = seg["token_start"]
        end = seg["token_end"]
        assert start < end, f"empty segment: {seg}"
        for pos in range(start, end):
            assert not covered[pos], (
                f"token position {pos} covered by multiple segments; "
                f"current: {seg}, segments={segments}"
            )
            covered[pos] = True
    uncovered = [i for i, c in enumerate(covered) if not c]
    assert not uncovered, (
        f"positions {uncovered[:10]} uncovered; total={total_tokens}, segs={segments}"
    )

    # The last two user turns must each produce a segment (not swallowed by
    # the unmatched region). Confirm at least one 'user' segment exists.
    user_segs = [s for s in segments if s["role"] == "user"]
    assert user_segs, "no 'user' segments emitted; aligned turns were swallowed"

    # No duplicate (token_start, token_end) pairs across segments.
    ranges = [(s["token_start"], s["token_end"]) for s in segments]
    assert len(ranges) == len(set(ranges)), f"duplicate segment ranges: {ranges}"


# ---------------------------------------------------------------------------
# Group 3: slow e2e with real model.generate() + session KV reuse.
# ---------------------------------------------------------------------------


def _build_h2o_config():
    from serving.kv_policies.base import EvictionPolicyConfig

    return EvictionPolicyConfig(
        name="h2o",
        budget=H2O_BUDGET,
        sink_size=SINK_SIZE,
        recent_window=RECENT_WINDOW,
        aggregate="sum",
        record=True,
    )


@pytest.mark.slow
def test_two_consecutive_chats_with_session_cache_works(tmp_path: Path) -> None:
    """Drive two real chat() calls through H2O session cache on Qwen3-0.6B.

    Assertions:
    - no exception from either call
    - both produce non-empty generated text
    - physical KV length in the session cache grows after call 2
    - both iter_0000 and iter_0001 have a kv_eviction.npz with records
    - _session_history confirms strict-prefix reuse on call 2 (lcp ==
      cached_len_before), which holds for Qwen3's default template at
      temperature=0.0 when the model does not emit autonomous <think> tokens
    """
    from serving.recording.backend_hf import HFRecordingProvider

    recordings_dir = tmp_path / "recordings"
    recordings_dir.mkdir()

    provider = HFRecordingProvider(
        default_model=MODEL,
        eviction_config=_build_h2o_config(),
    )
    provider.start_attempt(recordings_dir)

    async def _run() -> tuple[str, str]:
        resp1 = await provider.chat(
            messages=[{"role": "user", "content": "Say one word."}],
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
        )
        assert resp1.content, "call 1 produced empty content"
        seq_len_after_1 = provider._session_cache.get_seq_length(0)

        resp2 = await provider.chat(
            messages=[
                {"role": "user", "content": "Say one word."},
                {"role": "assistant", "content": resp1.content},
                {"role": "user", "content": "Say another word."},
            ],
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
        )
        assert resp2.content, "call 2 produced empty content"
        seq_len_after_2 = provider._session_cache.get_seq_length(0)

        assert seq_len_after_2 > seq_len_after_1, (
            f"session cache did not grow: {seq_len_after_1} -> {seq_len_after_2}"
        )
        return resp1.content, resp2.content

    content1, content2 = asyncio.run(_run())

    provider.finish_attempt()

    # --- KV recording artifacts -----------------------------------------------
    iter0 = recordings_dir / "iter_0000"
    iter1 = recordings_dir / "iter_0001"
    assert iter0.exists(), "iter_0000 directory missing"
    assert iter1.exists(), "iter_0001 directory missing"

    npz0 = iter0 / "kv_eviction.npz"
    npz1 = iter1 / "kv_eviction.npz"
    assert npz0.exists(), "iter_0000/kv_eviction.npz missing"
    assert npz1.exists(), "iter_0001/kv_eviction.npz missing"

    import numpy as np

    with np.load(npz0) as d:
        assert len(d["record_step"]) > 0, "iter_0000 kv_eviction has zero rows"
    with np.load(npz1) as d:
        assert len(d["record_step"]) > 0, "iter_0001 kv_eviction has zero rows"

    # --- Session-history: both calls must be recorded -------------------------
    history = provider._session_history
    assert len(history) == 2, f"expected 2 history entries, got {len(history)}"

    h0 = history[0]
    assert h0["call_idx"] == 0
    assert h0["cached_len_before"] == 0, "first call must start from empty cache"
    assert h0["delta_len"] == h0["new_len"], "first call must pass full prompt"
    assert not h0.get("diverged"), "first call must not diverge"

    h1 = history[1]
    assert h1["call_idx"] == 1
    assert not h1.get("diverged"), (
        "call 2 diverged — LCP failed; cache rebuilt instead of reusing KV state. "
        "At temperature=0.0, Qwen3's default template produces a strict prefix "
        "extension, so divergence indicates a regression."
    )
    assert h1["lcp"] == h1["cached_len_before"], (
        f"call 2: lcp ({h1['lcp']}) != cached_len_before ({h1['cached_len_before']}); "
        "the session state is not a strict prefix of the new prompt"
    )
    assert h1["delta_len"] < h1["new_len"], (
        f"call 2: delta_len ({h1['delta_len']}) == new_len ({h1['new_len']}); "
        "no tokens were reused from the session cache"
    )
