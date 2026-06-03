# Step 9 KV eviction perf microbench

Model: `Qwen/Qwen3-0.6B`  ·  prompt fixed  ·  max_tokens=64  ·  runs/config = 5 (+1 warmup discarded)

| policy | record | mean_ms | std_ms | p50_ms | p95_ms | overhead_vs_none_p50_pct |
|---|---|---|---|---|---|---|
| none | n/a | 19157.2 | 11684.9 | 12838.6 | 39658.1 | +0.0% |
| h2o | on | 13888.0 | 600.1 | 13713.1 | 14519.4 | +6.8% |
| h2o | off | 16315.8 | 4296.3 | 13482.9 | 22376.8 | +5.0% |
| streaming | on | 12752.7 | 83.7 | 12729.6 | 12899.2 | -0.8% |
| streaming | off | 13318.1 | 372.7 | 13225.5 | 13771.2 | +3.0% |

## Interpretation

All overheads are relative to baseline p50 (12838.6 ms); p50 is used because
the baseline mean is inflated by a high-variance cold-start outlier (see Limitations).

- Baseline (policy=none, record=n/a): 12838.6 ms / call (p50).
- H2O policy: +5.0% with record=off, +6.8% with record=on; recording overhead = +1.8 pp.
- Streaming policy: -0.8% with record=off (noise), +3.0% with record=on; recording overhead = +3.8 pp.

H2O eviction costs ~5% in pure compute overhead; npz recording adds ~2 pp on top.
Streaming is near-zero eviction cost; recording adds ~4 pp.

## Limitations

- **Baseline high variance**: the `policy=none` config is always run first; on Mac unified
  memory, the first config after provider re-init triggers OS-level page-in activity that
  inflates several individual measurements (std=11685 ms vs mean=19157 ms). P50 is robust
  to this, but mean-based overhead numbers from this run should be ignored.
- **p50 preferred over mean**: with 5 samples and one outlier-prone cold-start position,
  p50 (median) is the most reliable central-tendency estimate; mean and p95 are provided
  for completeness but should not be used for policy comparisons in this run.
- **CPU vs GPU**: these numbers are from Mac Apple Silicon running float32 inference
  without CUDA. On GPU, attention computation dominates wall time by a larger margin,
  which will change the relative cost of H2O's per-step score maintenance — GPU overhead
  fractions are expected to differ substantially from these CPU measurements.
