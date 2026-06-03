# KV Cache Subclass Spike Results

**Date:** 2026-05-11
**Spike script:** `scripts/spikes/kv_cache_subclass_spike.py`
**Raw output:** `/tmp/kv_spike_output.txt`
**Plan under validation:** `local-trace-collect-streamingllm-h2o-ra-crispy-goose.md`

## Environment

| Component         | Version |
| ----------------- | ------- |
| Python            | 3.13.5  |
| `transformers`    | 4.57.6  |
| `torch`           | 2.10.0  |
| Conda env         | `ML`    |
| Device            | CPU (Darwin) |
| Model used        | `HuggingFaceTB/SmolLM-135M` (primary, no fallback needed) |
| `num_hidden_layers` | 30 |
| `attn_implementation` | `sdpa` |

## Hypothesis Verdicts

### Q1. Custom `Cache` subclass accepted by `model.generate()` — ✅ PASS

`CountingDynamicCache(DynamicCache)` was passed via `past_key_values=` to
`model.generate(...)`. `generate()` ran cleanly, returned `shape=(1, 24)` from
a 16-token prompt with `max_new_tokens=8`, decoded to coherent text:
`"The quick brown fox jumps over the lazy dog.\nThe quick brown fox is"`.
**Evidence:** `new_tokens=8, expected=8`.

### Q2. SDPA path invokes `Cache.update()` for every layer, every step — ✅ PASS

After running 16-token prefill + 8 decoded tokens with the counting cache, every
one of the 30 layers showed exactly 8 `update()` calls — total = 240. Initially
this looked like a "partial" (expectation was `1+8=9` per layer) but per-layer
inspection of the `key_states` shape sequence resolved it:

```
layer 0  K-seqlens: [8, 1, 1, 1, 1, 1, 1, 1]
layer 29 K-seqlens: [8, 1, 1, 1, 1, 1, 1, 1]
```

The first call has `K seq_len=8` (the full prefill in one pass), followed by 7
decode calls of `K seq_len=1`. HF `generate()` fuses the prefill forward pass
with the first generated token, so for `max_new_tokens=N` you get `N` total
forward passes (`1 prefill + (N-1) decode`), not `1 + N`. Every layer is hit on
every pass — **SDPA does not bypass the `Cache.update()` interface**, which is
the actual property the plan needs.

**Evidence:** `observed total = 240 = 30 layers × (1 prefill + 7 decode)`,
K-seqlen pattern `[8, 1, 1, 1, 1, 1, 1, 1]` confirms prefill+decode behavior.

### Q3. `SinkCache` directly subclassable — ❌ FAIL

In `transformers==4.57.6`, `SinkCache` has been **removed from the in-tree
implementation** and relocated to a `custom_generate` Hub repository:

- `SinkCache.__init__` signature is `(self, **kwargs) -> None` — a stub.
- `SinkCache.update` signature still looks normal:
  `(self, key_states, value_states, layer_idx, cache_kwargs=None) -> tuple[Tensor, Tensor]`
- Any instantiation (`SinkCache()`, `SinkCache(num_sink_tokens=4)`,
  `SinkCache(window_length=64, num_sink_tokens=4)`) raises:

  ```
  NotImplementedError: `SinkCache` has been moved as a `custom_generate`
  repository on the Hub: https://huggingface.co/transformers-community/sink_cache.
  See the repository for usage examples.
  ```

Because instantiation never returns, no subclass test could proceed. **Direct
subclassing of the in-tree `SinkCache` is not viable on 4.57.x.**

## Recommended Adjustments

1. **Do not subclass `transformers.cache_utils.SinkCache` for StreamingLLM.**
   Either:
   - Implement StreamingLLM eviction directly on top of `DynamicCache` (the
     subclass route works and is fully exercised by SDPA — Q1+Q2). The eviction
     logic (sink tokens + sliding window) is small enough to re-implement
     against the `update()` contract, which is what the plan ultimately needs
     for H2O and Random anyway.
   - Or pull the `transformers-community/sink_cache` `custom_generate` repo as
     a reference implementation and adapt it.
   The first option is cleaner: a single `BaseEvictionCache(DynamicCache)`
   parent with `StreamingLLM`, `H2O`, `Random` siblings, all using the same
   override surface verified in Q2.

2. **Pin transformers carefully.** `SinkCache` got deprecated/relocated between
   minor versions in the 4.4x→4.5x range. Recommend pinning to
   `transformers>=4.55,<4.58` for the plan's lifetime, and only using the
   `Cache` / `DynamicCache` API surface (which has been stable across this
   range). Do **not** depend on `SinkCache` itself.

3. **Update the plan's expected-call-count test fixture** to use the formula
   `total_update_calls == num_hidden_layers × max_new_tokens` (with K-seqlen
   pattern `[prefill_len, 1, 1, ...]` per layer), not `num_layers × (1 + N)`.
   This is just a fixture-level note for the milestone-2 unit tests.

## Overall Assessment

The two load-bearing assumptions for the plan (Q1: subclass acceptance; Q2:
SDPA touches `update()` on every layer/step) **both hold cleanly**. The third
assumption (subclassing `SinkCache`) does not hold in 4.57, but it isn't
load-bearing — the plan can implement StreamingLLM directly via
`DynamicCache` subclassing, which is the exact pattern already validated.

**Recommendation: plan can proceed with one modification — drop the
`SinkCache` dependency and base all three eviction caches on a custom
`DynamicCache` subclass.**
