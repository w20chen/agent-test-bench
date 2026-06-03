"""Step 9 perf microbench: isolate eviction overhead from recording overhead.

Runs five HFRecordingProvider configurations sequentially against Qwen3-0.6B
on a fixed prompt, capturing wall time per `provider.chat(...)` call:

  1. policy=none, record=on  (baseline — stock DynamicCache + LayerCapturer)
  2. policy=h2o, record=on
  3. policy=h2o, record=off
  4. policy=streaming, record=on
  5. policy=streaming, record=off

Each config runs N+1 times: the first iteration is discarded as warmup
(weights cold, allocator unprimed, prompt cache empty), the next N feed
the statistics. Results land in `scripts/spikes/step9_perf_results.md`
(rendered markdown table) and are also echoed to stdout.

Run:
    conda run -n ML python scripts/spikes/step9_perf_microbench.py

Knobs (CLI flags, all optional):
    --runs N           Number of measured runs per config (default 5)
    --max-tokens N     max_new_tokens per chat call (default 64)
    --output PATH      Override the markdown output path

Sequential by design — running multiple configs in parallel would
oversubscribe the GPU and pollute the timing distribution.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from serving.kv_policies.base import EvictionPolicyConfig  # noqa: E402


MODEL = "Qwen/Qwen3-0.6B"
PROMPT = (
    "Briefly explain why a heavy-hitter KV cache eviction policy can keep "
    "generation quality high even when the cache budget is much smaller "
    "than the prompt."
)
DEFAULT_RUNS = 5
DEFAULT_MAX_TOKENS = 64
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "step9_perf_results.md"


@dataclass
class PerfRow:
    policy: str
    record: str
    samples_ms: list[float]

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.samples_ms)

    @property
    def std_ms(self) -> float:
        if len(self.samples_ms) < 2:
            return 0.0
        return statistics.stdev(self.samples_ms)

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.samples_ms)

    @property
    def p95_ms(self) -> float:
        if len(self.samples_ms) < 2:
            return self.samples_ms[0]
        # Manual nearest-rank p95 to avoid pulling numpy just for percentile.
        ordered = sorted(self.samples_ms)
        idx = max(0, min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))))
        return ordered[idx]


def _build_config(policy: str, record: bool) -> EvictionPolicyConfig | None:
    if policy == "none":
        return None
    recent_window = 252 if policy == "streaming" else 64
    return EvictionPolicyConfig(
        name=policy,
        budget=256,
        sink_size=4,
        recent_window=recent_window,
        aggregate="sum",
        record=record,
    )


async def _bench_one_config(
    policy: str,
    record: bool,
    *,
    runs: int,
    max_tokens: int,
    workdir: Path,
) -> PerfRow:
    """Build a fresh provider, do 1 warmup + `runs` measured chat calls."""
    from serving.recording.backend_hf import HFRecordingProvider

    eviction_config = _build_config(policy, record)
    provider = HFRecordingProvider(
        default_model=MODEL,
        eviction_config=eviction_config,
    )

    record_label = "n/a" if policy == "none" else ("on" if record else "off")
    samples: list[float] = []

    attempt_dir = workdir / f"{policy}_{record_label}"
    attempt_dir.mkdir(parents=True, exist_ok=True)

    # Total iterations = warmup (1) + measured (`runs`).
    for iteration in range(runs + 1):
        # Each call gets its own recordings dir so artifact writes don't pile
        # up across iterations (we want to time the call, not later cleanup).
        iter_recordings = attempt_dir / f"iter_{iteration}"
        iter_recordings.mkdir(exist_ok=True)
        provider.start_attempt(iter_recordings)

        t0 = time.perf_counter()
        await provider.chat(
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        t1 = time.perf_counter()

        provider.finish_attempt()

        if iteration == 0:
            continue  # warmup discarded
        samples.append((t1 - t0) * 1000.0)

    # Tear the provider down before the next config so GPU/CPU memory does
    # not creep across runs (transformers caches embedding tables otherwise).
    del provider
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    return PerfRow(policy=policy, record=record_label, samples_ms=samples)


def _format_markdown(rows: list[PerfRow], baseline_p50: float) -> str:
    headers = [
        "policy",
        "record",
        "mean_ms",
        "std_ms",
        "p50_ms",
        "p95_ms",
        "overhead_vs_none_p50_pct",
    ]
    body_lines = []
    for r in rows:
        overhead_pct = (
            ((r.p50_ms - baseline_p50) / baseline_p50 * 100.0)
            if baseline_p50 > 0
            else 0.0
        )
        body_lines.append(
            "| {policy} | {record} | {mean:.1f} | {std:.1f} | {p50:.1f} | {p95:.1f} | {ovr:+.1f}% |".format(
                policy=r.policy,
                record=r.record,
                mean=r.mean_ms,
                std=r.std_ms,
                p50=r.p50_ms,
                p95=r.p95_ms,
                ovr=overhead_pct,
            )
        )

    table = (
        "| " + " | ".join(headers) + " |\n"
        + "|" + "|".join(["---"] * len(headers)) + "|\n"
        + "\n".join(body_lines)
    )
    return table


def _interpret(rows: list[PerfRow], baseline_p50: float) -> str:
    by_key = {(r.policy, r.record): r for r in rows}

    def overhead(policy: str, record: str) -> float:
        r = by_key.get((policy, record))
        if r is None or baseline_p50 <= 0:
            return 0.0
        return (r.p50_ms - baseline_p50) / baseline_p50 * 100.0

    h2o_on = overhead("h2o", "on")
    h2o_off = overhead("h2o", "off")
    streaming_on = overhead("streaming", "on")
    streaming_off = overhead("streaming", "off")

    record_overhead_h2o = h2o_on - h2o_off
    record_overhead_streaming = streaming_on - streaming_off

    return (
        f"- Baseline (policy=none, record=n/a): {baseline_p50:.1f} ms / call (p50).\n"
        f"- H2O policy: {h2o_off:+.1f}% with record=off, "
        f"{h2o_on:+.1f}% with record=on; "
        f"recording overhead = {record_overhead_h2o:+.1f} pp.\n"
        f"- Streaming policy: {streaming_off:+.1f}% with record=off, "
        f"{streaming_on:+.1f}% with record=on; "
        f"recording overhead = {record_overhead_streaming:+.1f} pp.\n"
    )


async def _main_async(args: argparse.Namespace) -> None:
    workdir = Path(args.workdir) if args.workdir else Path("/tmp/step9_perf_workdir")
    workdir.mkdir(parents=True, exist_ok=True)

    plan = [
        ("none", True),
        ("h2o", True),
        ("h2o", False),
        ("streaming", True),
        ("streaming", False),
    ]
    rows: list[PerfRow] = []
    for policy, record in plan:
        print(
            f"[step9] running policy={policy} record={record} "
            f"({args.runs} measured + 1 warmup) ..."
        )
        row = await _bench_one_config(
            policy,
            record,
            runs=args.runs,
            max_tokens=args.max_tokens,
            workdir=workdir,
        )
        rows.append(row)
        print(
            f"  mean={row.mean_ms:.1f} ms  std={row.std_ms:.1f} "
            f"p50={row.p50_ms:.1f}  p95={row.p95_ms:.1f}"
        )

    baseline = next((r for r in rows if r.policy == "none"), None)
    if baseline is None:
        raise RuntimeError("no baseline policy=none row produced")
    table = _format_markdown(rows, baseline.p50_ms)
    interpretation = _interpret(rows, baseline.p50_ms)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "# Step 9 KV eviction perf microbench\n\n"
        f"Model: `{MODEL}`  ·  prompt fixed  ·  max_tokens={args.max_tokens}  "
        f"·  runs/config = {args.runs} (+1 warmup discarded)\n\n"
        + table
        + "\n\n## Interpretation\n\n"
        + interpretation,
        encoding="utf-8",
    )
    print()
    print(table)
    print()
    print(interpretation)
    print(f"[step9] wrote {output_path}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--workdir",
        type=str,
        default=None,
        help="Where iter_*/recordings land; defaults to /tmp/step9_perf_workdir.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
