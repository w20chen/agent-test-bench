"""KV cache subclass technical spike.

Validates three hypotheses for the StreamingLLM / H2O / Random eviction
integration plan (see local-trace-collect-streamingllm-h2o-ra-crispy-goose.md):

  Q1. Can a custom subclass of `transformers.cache_utils.Cache` be passed
      via `model.generate(past_key_values=...)` and produce correct output?
  Q2. Does the `attn_implementation="sdpa"` path actually invoke
      `Cache.update()` on every layer, every step (prefill + decode)?
  Q3. Is `transformers.cache_utils.SinkCache` directly subclass-friendly
      in the installed transformers version?

This is a spike — no production code is touched, no `src/` imports.
"""

from __future__ import annotations

import inspect
import sys
import traceback
from collections import defaultdict
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache, DynamicCache, SinkCache


PRIMARY_MODEL = "HuggingFaceTB/SmolLM-135M"
FALLBACK_MODEL = "sshleifer/tiny-gpt2"

PREFILL_TOKENS = 16
DECODE_TOKENS = 8


def _print_header(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n{title}\n{bar}", flush=True)


def _load_model() -> tuple[Any, Any, str]:
    """Load primary model; fall back loudly if unavailable."""
    last_err: Exception | None = None
    for name in (PRIMARY_MODEL, FALLBACK_MODEL):
        try:
            tok = AutoTokenizer.from_pretrained(name)
            mdl = AutoModelForCausalLM.from_pretrained(
                name,
                attn_implementation="sdpa",
                torch_dtype=torch.float32,
            ).eval()
            print(f"[load] using model: {name}", flush=True)
            print(f"[load] num_hidden_layers = {mdl.config.num_hidden_layers}", flush=True)
            print(f"[load] attn_implementation = {getattr(mdl.config, '_attn_implementation', '?')}", flush=True)
            return tok, mdl, name
        except Exception as exc:  # noqa: BLE001
            print(f"[load] FAILED for {name}: {type(exc).__name__}: {exc}", flush=True)
            last_err = exc
    raise RuntimeError(f"could not load any spike model; last error: {last_err}")


# ---------------------------------------------------------------------------
# Counting cache (used for Q1 + Q2)
# ---------------------------------------------------------------------------
class CountingDynamicCache(DynamicCache):
    """DynamicCache subclass that records per-layer update() call counts."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Use object.__setattr__ to dodge any pydantic-style guards
        object.__setattr__(self, "_per_layer_calls", defaultdict(int))
        object.__setattr__(self, "_call_log", [])  # (layer_idx, k_seq_len)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):  # type: ignore[override]
        self._per_layer_calls[int(layer_idx)] += 1
        self._call_log.append((int(layer_idx), int(key_states.shape[-2])))
        return super().update(key_states, value_states, layer_idx, cache_kwargs)


# ---------------------------------------------------------------------------
# Q1 + Q2: subclass acceptance + per-layer / per-step update() coverage
# ---------------------------------------------------------------------------
def test_q1_q2_dynamic_subclass(tok: Any, mdl: Any) -> dict[str, Any]:
    _print_header("Q1+Q2: DynamicCache subclass + SDPA update() coverage")
    out: dict[str, Any] = {"q1": "unknown", "q2": "unknown"}
    try:
        prompt_ids = tok("The quick brown fox jumps over the lazy", return_tensors="pt").input_ids
        # Pad / truncate to PREFILL_TOKENS so counts are predictable
        if prompt_ids.shape[-1] < PREFILL_TOKENS:
            pad = torch.full(
                (1, PREFILL_TOKENS - prompt_ids.shape[-1]),
                tok.eos_token_id or 0,
                dtype=prompt_ids.dtype,
            )
            prompt_ids = torch.cat([pad, prompt_ids], dim=-1)
        else:
            prompt_ids = prompt_ids[:, :PREFILL_TOKENS]
        attn_mask = torch.ones_like(prompt_ids)

        cache = CountingDynamicCache()

        gen = mdl.generate(
            input_ids=prompt_ids,
            attention_mask=attn_mask,
            past_key_values=cache,
            max_new_tokens=DECODE_TOKENS,
            min_new_tokens=DECODE_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tok.eos_token_id or 0,
        )
        new_tokens = gen.shape[-1] - prompt_ids.shape[-1]
        decoded = tok.decode(gen[0], skip_special_tokens=True)
        print(f"[q1] generate() returned shape={tuple(gen.shape)}, new_tokens={new_tokens}", flush=True)
        print(f"[q1] decoded: {decoded!r}", flush=True)
        out["q1"] = "pass" if new_tokens == DECODE_TOKENS else "partial"
        out["q1_evidence"] = f"new_tokens={new_tokens}, expected={DECODE_TOKENS}"

        # Q2: per-layer call counts
        nlayers = mdl.config.num_hidden_layers
        expected_per_layer = 1 + DECODE_TOKENS  # 1 prefill + N decode
        per_layer = dict(cache._per_layer_calls)
        total = sum(per_layer.values())
        print(f"[q2] num_hidden_layers={nlayers}", flush=True)
        print(f"[q2] expected per-layer calls = {expected_per_layer}", flush=True)
        print(f"[q2] expected total calls = {nlayers * expected_per_layer}", flush=True)
        print(f"[q2] observed per-layer calls = {per_layer}", flush=True)
        print(f"[q2] observed total = {total}", flush=True)

        all_layers_present = set(per_layer.keys()) == set(range(nlayers))
        all_correct_count = all(per_layer.get(i, 0) == expected_per_layer for i in range(nlayers))
        if all_layers_present and all_correct_count:
            out["q2"] = "pass"
            out["q2_evidence"] = (
                f"all {nlayers} layers each saw exactly {expected_per_layer} update() calls"
            )
        elif all_layers_present:
            out["q2"] = "partial"
            mismatches = {i: per_layer[i] for i in range(nlayers) if per_layer[i] != expected_per_layer}
            out["q2_evidence"] = f"layers present but counts differ: {mismatches}"
        else:
            missing = sorted(set(range(nlayers)) - set(per_layer.keys()))
            out["q2"] = "fail"
            out["q2_evidence"] = f"missing layers in update() log: {missing}"
    except Exception as exc:  # noqa: BLE001
        out["q1"] = out["q1"] if out["q1"] != "unknown" else "fail"
        out["q2"] = "fail"
        out["q1_evidence"] = out.get("q1_evidence", f"{type(exc).__name__}: {exc}")
        out["q2_evidence"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    return out


# ---------------------------------------------------------------------------
# Q3: SinkCache subclass viability
# ---------------------------------------------------------------------------
def test_q3_sinkcache_subclass(tok: Any, mdl: Any) -> dict[str, Any]:
    _print_header("Q3: SinkCache subclass viability")
    out: dict[str, Any] = {"q3": "unknown"}
    try:
        out["sinkcache_init_sig"] = str(inspect.signature(SinkCache.__init__))
        out["sinkcache_update_sig"] = str(inspect.signature(SinkCache.update))
        print(f"[q3] SinkCache.__init__ signature: {out['sinkcache_init_sig']}", flush=True)
        print(f"[q3] SinkCache.update   signature: {out['sinkcache_update_sig']}", flush=True)

        # Probe: is SinkCache flagged deprecated / requires special args?
        # The 4.57 signature is `**kwargs`, suggesting heavy refactor — try
        # the historical (window_length, num_sink_tokens) call first, then
        # fall back to the legacy positional form.
        sink: SinkCache | None = None
        init_attempts: list[tuple[str, str]] = []
        for label, kwargs in (
            ("window_length+num_sink_tokens", dict(window_length=64, num_sink_tokens=4)),
            ("only num_sink_tokens", dict(num_sink_tokens=4)),
            ("empty kwargs", {}),
        ):
            try:
                sink = SinkCache(**kwargs)
                init_attempts.append((label, "ok"))
                print(f"[q3] SinkCache({label}) -> ok", flush=True)
                break
            except Exception as exc:  # noqa: BLE001
                init_attempts.append((label, f"{type(exc).__name__}: {exc}"))
                print(f"[q3] SinkCache({label}) -> {type(exc).__name__}: {exc}", flush=True)
        out["sinkcache_init_attempts"] = init_attempts

        if sink is None:
            out["q3"] = "fail"
            out["q3_evidence"] = "SinkCache could not be instantiated with any tried kwargs"
            return out

        # Try defining a trivial subclass that overrides update()
        class MarkedSinkCache(SinkCache):
            def __init__(self, *a: Any, **kw: Any) -> None:
                super().__init__(*a, **kw)
                object.__setattr__(self, "_marker_calls", 0)

            def update(self, key_states, value_states, layer_idx, cache_kwargs=None):  # type: ignore[override]
                self._marker_calls += 1
                return super().update(key_states, value_states, layer_idx, cache_kwargs)

        # Find a kwargs combo that worked above
        working_kwargs = next(
            (kw for label, kw in [
                ("window_length+num_sink_tokens", dict(window_length=64, num_sink_tokens=4)),
                ("only num_sink_tokens", dict(num_sink_tokens=4)),
                ("empty kwargs", {}),
            ] if init_attempts and any(a[0] == label and a[1] == "ok" for a in init_attempts)),
            {},
        )

        try:
            sub = MarkedSinkCache(**working_kwargs)
            print(f"[q3] MarkedSinkCache instantiated with kwargs={working_kwargs}", flush=True)
        except Exception as exc:  # noqa: BLE001
            out["q3"] = "fail"
            out["q3_evidence"] = f"subclass instantiation failed: {type(exc).__name__}: {exc}"
            traceback.print_exc()
            return out

        # Try generate() with the subclassed SinkCache
        try:
            prompt_ids = tok("The quick brown fox", return_tensors="pt").input_ids
            attn_mask = torch.ones_like(prompt_ids)
            gen = mdl.generate(
                input_ids=prompt_ids,
                attention_mask=attn_mask,
                past_key_values=sub,
                max_new_tokens=DECODE_TOKENS,
                min_new_tokens=DECODE_TOKENS,
                do_sample=False,
                use_cache=True,
                pad_token_id=tok.eos_token_id or 0,
            )
            new_tokens = gen.shape[-1] - prompt_ids.shape[-1]
            calls = sub._marker_calls
            print(f"[q3] generate() with MarkedSinkCache: new_tokens={new_tokens}, marker_calls={calls}", flush=True)
            if new_tokens == DECODE_TOKENS and calls > 0:
                out["q3"] = "pass"
                out["q3_evidence"] = f"subclass generate ok: new_tokens={new_tokens}, update_calls={calls}"
            elif new_tokens == DECODE_TOKENS:
                out["q3"] = "partial"
                out["q3_evidence"] = f"generate ok but subclass update() never called: calls={calls}"
            else:
                out["q3"] = "fail"
                out["q3_evidence"] = f"generate produced wrong token count: {new_tokens}"
        except Exception as exc:  # noqa: BLE001
            out["q3"] = "fail"
            out["q3_evidence"] = f"generate with subclass raised: {type(exc).__name__}: {exc}"
            traceback.print_exc()
    except Exception as exc:  # noqa: BLE001
        out["q3"] = "fail"
        out["q3_evidence"] = f"outer error: {type(exc).__name__}: {exc}"
        traceback.print_exc()
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    print(f"transformers: {transformers.__version__}", flush=True)
    print(f"torch:        {torch.__version__}", flush=True)
    print(f"python:       {sys.version.split()[0]}", flush=True)

    try:
        tok, mdl, model_name = _load_model()
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: {exc}", flush=True)
        return 2

    summary: dict[str, Any] = {"model": model_name}
    summary.update(test_q1_q2_dynamic_subclass(tok, mdl))
    summary.update(test_q3_sinkcache_subclass(tok, mdl))

    _print_header("SUMMARY")
    for key in ("model", "q1", "q1_evidence", "q2", "q2_evidence", "q3", "q3_evidence",
                "sinkcache_init_sig", "sinkcache_update_sig", "sinkcache_init_attempts"):
        if key in summary:
            print(f"{key}: {summary[key]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
