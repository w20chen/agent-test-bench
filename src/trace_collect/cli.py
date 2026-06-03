"""CLI entry point for trace collection, import, replay, and viewer helpers."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from llm_call import add_llm_config_arguments, resolve_llm_config
from llm_call.config import (
    nonnegative_float_arg,
    positive_float_arg,
    positive_int_arg,
    top_p_arg,
)


def parse_collect_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect SWE-Bench agent traces using an external LLM API.",
    )
    add_llm_config_arguments(parser)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum agent iterations per task.",
    )
    parser.add_argument(
        "--temperature",
        type=nonnegative_float_arg,
        default=None,
        help=(
            "Optional agent sampling temperature. When omitted, the scaffold "
            "default is used."
        ),
    )
    parser.add_argument(
        "--top-p",
        type=top_p_arg,
        default=None,
        help="Optional agent nucleus sampling top_p value.",
    )
    parser.add_argument(
        "--top-k",
        type=positive_int_arg,
        default=None,
        help="Optional agent top_k sampling value for compatible providers.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=positive_float_arg,
        default=None,
        help="Optional agent repetition penalty for compatible providers.",
    )
    parser.add_argument(
        "--benchmark",
        default="swe-bench-verified",
        help=(
            "Benchmark slug (e.g. 'swe-bench-verified', 'swe-rebench'). "
            "Loads configs/benchmarks/<slug>.yaml and constructs the plugin."
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only run the first N tasks (for testing).",
    )
    parser.add_argument(
        "--instance-ids",
        default=None,
        help="Comma-separated list of instance IDs to run (e.g., 'django__django-12345,sympy__sympy-67890').",
    )
    parser.add_argument(
        "--scaffold",
        choices=["openclaw", "tongyi-deepresearch"],
        default="openclaw",
        help="Agent scaffold to use.",
    )
    parser.add_argument(
        "--container",
        choices=["docker", "podman"],
        default=None,
        help="Container CLI executable for benchmark collection runtime.",
    )
    parser.add_argument(
        "--mcp-config",
        default=None,
        help=(
            "MCP server configuration. Required when --scaffold=openclaw. "
            "Accepts a YAML path (e.g. configs/mcp/context7.yaml) OR the "
            "literal string 'none' for an affirmative MCP-less run. The "
            "trace header records the chosen value under "
            "metadata.run_config.mcp_config so analysis can distinguish "
            "explicit 'none' from a legacy MCP-less default."
        ),
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=256_000,
        help="Sliding window token budget for context management.",
    )
    parser.add_argument(
        "--prompt-template",
        default=None,
        help=(
            "Optional prompt template override; resolved as "
            "configs/prompts/<benchmark_slug>/<name>.md (hyphens converted to underscores). "
            "When omitted, uses the benchmark config default "
            "(e.g. swe-rebench -> cc_aligned, terminal-bench -> default)."
        ),
    )
    parser.add_argument(
        "--min-free-disk-gb",
        type=float,
        default=30.0,
        help="Abort per-task run if free disk falls below this threshold (GB).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Resume an interrupted run by passing its existing run directory path.",
    )
    parser.add_argument(
        "--record-internals",
        action="store_true",
        help=(
            "Record per-call attention aggregates and MoE routing with a "
            "host-side HuggingFace SDPA backend. Forces OpenClaw model request "
            "concurrency to 1 for the run."
        ),
    )
    parser.add_argument(
        "--kv-policy",
        choices=["none", "random", "streaming", "h2o"],
        default="none",
        help=(
            "KV cache eviction policy used by the HF recording backend. "
            "`none` (default) keeps stock DynamicCache behaviour. `random` "
            "evicts uniformly when key_len > budget. `streaming` keeps "
            "`--kv-sink-size` head + `--kv-recent-window` tail tokens "
            "(StreamingLLM, naive variant — no RoPE re-rotation). `h2o` "
            "(Heavy-Hitter Oracle, arXiv:2306.14048) keeps sink + recent + "
            "top-k positions ranked by accumulated post-softmax attention; "
            "subscribes to the per-provider AttentionBus to share the "
            "softmax tensor with LayerCapturer (no double-softmax)."
        ),
    )
    parser.add_argument(
        "--kv-budget",
        type=int,
        default=None,
        help=(
            "Per-layer KV budget (kept tokens). Required when --kv-policy is "
            "not `none`. For `streaming` and `h2o`, acts as the trigger "
            "threshold and must be >= --kv-sink-size + --kv-recent-window. "
            "For `h2o`, post-eviction layer length is exactly `budget`; the "
            "middle slot count is `budget - sink_size - recent_window`."
        ),
    )
    parser.add_argument(
        "--kv-sink-size",
        type=int,
        default=4,
        help=(
            "Sink-prefix length (head tokens kept) for `streaming` and `h2o`. "
            "Default 4 matches StreamingLLM's §3 ablation. Ignored by "
            "`random`."
        ),
    )
    parser.add_argument(
        "--kv-recent-window",
        type=int,
        default=256,
        help=(
            "Recent-window length (tail tokens kept) for `streaming` and "
            "`h2o`. Default 256. Ignored by `random`."
        ),
    )
    parser.add_argument(
        "--kv-aggregate",
        choices=["sum", "mean", "ema"],
        default="sum",
        help=(
            "H2O score aggregation across queries. `sum` (paper default) "
            "accumulates raw mass; `mean` divides by the number of observed "
            "queries per position; `ema` uses an exponential moving average "
            "with `--kv-ema-decay` (yaml only). Ignored by `random` and "
            "`streaming`."
        ),
    )
    parser.add_argument(
        "--kv-config",
        type=str,
        default=None,
        help=(
            "Optional YAML file under e.g. configs/kv_policies/ containing a "
            "flat map of EvictionPolicyConfig fields (name, budget, sink_size, "
            "recent_window, heavy_ratio, aggregate, ema_decay, seed, record, "
            "prefill_mode). When set, the YAML supplies the base config and "
            "any explicitly-passed --kv-* flag overrides the corresponding "
            "yaml value. Mutually compatible with --kv-policy: yaml-name and "
            "an explicit --kv-policy must agree (CLI wins on explicit set)."
        ),
    )
    parser.add_argument(
        "--kv-record",
        choices=["on", "off"],
        default="on",
        help=(
            "Whether to write `kv_eviction.npz` recordings. Default `on` "
            "preserves the audit trail. `off` runs the policy but skips the "
            "per-call recorder allocation and npz write — used by step 9 "
            "perf microbench to isolate eviction overhead from recording "
            "overhead. Meaningful only when --kv-policy != none."
        ),
    )
    parser.add_argument(
        "--sparse-attn",
        choices=["none", "sliding", "streaming", "heavy_hitter", "block_topk", "quest"],
        default="none",
        help=(
            "Sparse attention method used by the HF recording backend. "
            "`none` (default) leaves attention dense. `sliding`/`streaming` "
            "keep the first `--sparse-attn-sink-size` tokens plus the last "
            "`--sparse-attn-recent-window` tokens. Dynamic methods require "
            "`--sparse-attn-budget` or YAML `budget:`. "
            "Mutually exclusive with --kv-policy and requires --record-internals."
        ),
    )
    parser.add_argument(
        "--sparse-attn-sink-size",
        type=int,
        default=4,
        help=(
            "Sink-prefix length kept attended for `sliding`. Default 4. "
            "Ignored when --sparse-attn != sliding."
        ),
    )
    parser.add_argument(
        "--sparse-attn-recent-window",
        type=int,
        default=256,
        help=(
            "Recent-window length kept attended for `sliding`. Default 256. "
            "Ignored when --sparse-attn != sliding."
        ),
    )
    parser.add_argument(
        "--sparse-attn-config",
        type=str,
        default=None,
        help=(
            "Optional YAML file (e.g. configs/sparse_attention/sliding.yaml) "
            "carrying a flat map of SparseAttentionConfig fields. CLI flags "
            "overlay yaml using the same rules as --kv-config."
        ),
    )
    parser.add_argument(
        "--sparse-attn-record",
        choices=["on", "off"],
        default="on",
        help=(
            "Whether to write `sparse_attention.npz` recordings. Default "
            "`on`. `off` runs the method but skips the per-call recorder "
            "allocation and npz write. Meaningful only when --sparse-attn "
            "!= none."
        ),
    )
    parser.add_argument(
        "--sparse-attn-observe-only",
        action="store_true",
        help="Record sparse selection without enforcing it (compatible with --kv-policy).",
    )
    parser.add_argument(
        "--sparse-attn-budget",
        type=int,
        default=None,
        help=(
            "Token budget for dynamic sparse attention methods "
            "(`heavy_hitter`, `block_topk`, `quest`)."
        ),
    )
    parser.add_argument(
        "--sparse-attn-block-size",
        type=int,
        default=16,
        help="Block/page size for `block_topk` and `quest`. Default 16.",
    )
    parser.add_argument(
        "--sparse-attn-score-reduction",
        choices=["max", "mean"],
        default="max",
        help="How to reduce token scores into block/page scores. Default max.",
    )
    parser.add_argument(
        "--sparse-attn-phase-scope",
        choices=["decode_only"],
        default="decode_only",
        help=(
            "Where dynamic methods enforce sparse masks. Only decode_only is "
            "currently supported; prefill remains dense causal."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args(argv)


def parse_simulate_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a trace with local model timing (TTFT/TPOT).",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source-trace",
        help="Path to the source API trace JSONL file.",
    )
    source_group.add_argument(
        "--trace-manifest",
        default=None,
        help=(
            "JSON manifest describing one or more replay traces. "
            "Each entry must contain source_trace and may override task_source."
        ),
    )
    add_llm_config_arguments(parser)
    parser.add_argument(
        "--mode",
        choices=["local_model", "cloud_model"],
        default="local_model",
        help=(
            "Simulation mode. local_model replays one trace through a local "
            "OpenAI-compatible model; cloud_model replays one or more traces "
            "using source-trace timing without issuing any LLM requests."
        ),
    )
    parser.add_argument(
        "--task-source",
        default="data/swe-rebench/tasks.json",
        help="Path to tasks JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default="traces/simulate",
        help="Output directory for the simulate trace.",
    )
    parser.add_argument(
        "--container",
        default=None,
        choices=["docker", "podman"],
        help="Container executable for container-mode trace replay.",
    )
    parser.add_argument(
        "--network-mode",
        default="host",
        help="Container network mode (default: host). Use 'none' for isolated replay.",
    )
    parser.add_argument(
        "--command-timeout",
        type=float,
        default=600.0,
        help=(
            "Fallback timeout in seconds for replayed shell commands when the "
            "source trace does not carry a tool-specific timeout."
        ),
    )
    parser.add_argument(
        "--metrics-url",
        default=None,
        help=(
            "vLLM Prometheus /metrics endpoint URL. When set, the simulator "
            "snapshots scheduler metrics (PreemptionSnapshot) per iteration "
            "and stores them under TraceAction.data.sim_metrics. When unset, "
            "the simulator records empty (all-None) snapshots — the explicit "
            "opt-out path used for local runs without a vLLM server."
        ),
    )
    parser.add_argument(
        "--warmup-skip-iterations",
        type=int,
        default=0,
        help=(
            "Tag the first N replay iterations with sim_metrics.warmup=true "
            "for analysis-time exclusion. Iterations are still measured at "
            "collection time; the flag controls analysis treatment only. "
            "Default 0 (no warmup tagging). Opt in only when first-iteration "
            "latency variance is empirically >20%% vs steady-state."
        ),
    )
    parser.add_argument(
        "--replay-speed",
        type=float,
        default=1.0,
        help=(
            "Wall-clock acceleration factor for cloud_model replay. "
            "Example: --replay-speed 50 replays source timing at 50x."
        ),
    )
    parser.add_argument(
        "--arrival-mode",
        default="closed_loop",
        choices=["closed_loop", "poisson"],
        help="Task arrival pattern for cloud_model replay (default: closed_loop).",
    )
    parser.add_argument(
        "--arrival-rate-per-s",
        type=float,
        default=None,
        help="Poisson arrival rate (tasks/sec). Required when --arrival-mode=poisson.",
    )
    parser.add_argument(
        "--arrival-seed",
        type=int,
        default=None,
        help="RNG seed for Poisson arrival offsets (for reproducibility).",
    )
    parser.add_argument(
        "--gpu-tracking",
        choices=["on", "off"],
        default="off",
        help=(
            "Enable GPU memory tracking. When 'on', requires --metrics-url, "
            "--vllm-pid, and --vllm-startup-log. Forbidden in cloud_model mode."
        ),
    )
    parser.add_argument(
        "--gpu-sample-hz",
        type=float,
        default=10.0,
        help="GPU memory sampling rate in Hz (default: 10.0). Used only when --gpu-tracking on.",
    )
    parser.add_argument(
        "--vllm-pid",
        type=int,
        default=None,
        help="PID of the vLLM server process. Required when --gpu-tracking on.",
    )
    parser.add_argument(
        "--vllm-startup-log",
        type=Path,
        default=None,
        help=(
            "Path to vLLM startup stderr log. Required when --gpu-tracking on. "
            "Used to extract GPU baseline (weights MiB, KV cache MiB)."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args(argv)


def parse_import_claude_code_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Claude Code session JSONL to a canonical trace for the Gantt "
            "viewer. Post-hoc, read-only — no collection, no simulation. "
            "Rich Claude Code fields (cache tokens, thinking blocks, "
            "toolUseResult sidecar) are backfilled into additive data.* and "
            "metadata.run_config.* slots in the canonical trace schema."
        ),
    )
    parser.add_argument(
        "--session",
        required=True,
        help=(
            "Path to the Claude Code session JSONL "
            "(typically ~/.claude/projects/<slug>/<session-uuid>.jsonl)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="traces",
        help=(
            "Root output directory. Final file lands at "
            "<output-dir>/claude-code-import/<session-uuid>/<session-uuid>.jsonl."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=("Optional explicit run directory suffix (default: session uuid)."),
    )
    parser.add_argument(
        "--no-sidechains",
        dest="include_sidechains",
        action="store_false",
        default=True,
        help=(
            "Skip folding <session-dir>/<session-uuid>/subagents/agent-*.jsonl "
            "into the output. Default: include them as distinct agent_id lanes."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    sub = sys.argv[1] if len(sys.argv) > 1 else None
    if sub == "simulate":
        _run_simulate(parse_simulate_args(sys.argv[2:]))
    elif sub == "import-claude-code":
        _run_import_claude_code(parse_import_claude_code_args(sys.argv[2:]))
    elif sub == "inspect":
        _run_inspect(sys.argv[2:])
    elif sub == "gantt-serve":
        from demo.gantt_viewer.backend.dev import main as run_gantt_server

        run_gantt_server(sys.argv[2:])
    elif sub == "gantt-export":
        from demo.gantt_viewer.backend.static_export import (
            build_parser as build_gantt_export_parser,
            export_from_args,
        )

        result = export_from_args(build_gantt_export_parser().parse_args(sys.argv[2:]))
        print(json.dumps(result, indent=2, sort_keys=True))
    elif sub == "profile-gpu":
        from trace_collect.profile_gpu import main as run_profile_gpu

        sys.exit(run_profile_gpu(sys.argv[2:]))
    else:
        _run_collect(parse_collect_args())


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_collect(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # --mcp-config is MANDATORY for openclaw runs: a forgotten flag would
    # silently produce an MCP-less trace. Opt-out is the literal "none".
    if args.scaffold == "openclaw" and args.mcp_config is None:
        print(
            "ERROR: MCP config is required for openclaw; pass "
            "--mcp-config configs/mcp/context7.yaml or --mcp-config none "
            "to acknowledge running without MCP",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.record_internals and args.scaffold != "openclaw":
        print(
            "ERROR: --record-internals currently supports --scaffold openclaw only.",
            file=sys.stderr,
        )
        sys.exit(2)
    # KV eviction policies live in the HF recording path; they are meaningless
    # without --record-internals (no HF backend = no Cache subclass injection
    # site). Either an explicit --kv-policy != none OR a --kv-config yaml that
    # supplies a non-none `name` activates eviction; both must imply
    # --record-internals.
    kv_policy_active = args.kv_policy != "none" or args.kv_config is not None
    if kv_policy_active and not args.record_internals:
        print(
            "ERROR: --kv-policy / --kv-config requires --record-internals "
            "(KV eviction only applies to the HF recording backend).",
            file=sys.stderr,
        )
        sys.exit(2)
    sparse_attn_active = (
        args.sparse_attn != "none" or args.sparse_attn_config is not None
    )
    if sparse_attn_active and not args.record_internals:
        print(
            "ERROR: --sparse-attn / --sparse-attn-config requires "
            "--record-internals (sparse attention only applies to the HF "
            "recording backend).",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.record_internals:
        os.environ["NANOBOT_MAX_CONCURRENT_REQUESTS"] = "1"

    try:
        provider_config = resolve_llm_config(
            provider=args.provider,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            environ=os.environ,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    if not provider_config.api_key:
        print(
            f"ERROR: Set {provider_config.env_key} or pass --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    from agents.benchmarks import get_benchmark_class
    from agents.benchmarks.base import BenchmarkConfig
    from serving.kv_policies.config import load_eviction_config
    from serving.sparse_attention.config import (
        load_sparse_attention_config,
        validate_attention_method_exclusivity,
    )
    from trace_collect.collector import collect_traces

    eviction_config = load_eviction_config(args)
    try:
        sparse_attention_config = load_sparse_attention_config(args)
        validate_attention_method_exclusivity(eviction_config, sparse_attention_config)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    benchmark_yaml = REPO_ROOT / "configs" / "benchmarks" / f"{args.benchmark}.yaml"
    if not benchmark_yaml.exists():
        print(f"ERROR: No benchmark config at {benchmark_yaml}", file=sys.stderr)
        sys.exit(1)
    config = BenchmarkConfig.from_yaml(benchmark_yaml)
    plugin_cls = get_benchmark_class(config.slug)
    benchmark = plugin_cls(config)

    run_dir = asyncio.run(
        collect_traces(
            scaffold=args.scaffold,
            container_executable=args.container,
            provider_name=provider_config.name,
            env_key=provider_config.env_key,
            api_base=provider_config.api_base,
            api_key=provider_config.api_key,
            model=provider_config.model,
            benchmark=benchmark,
            max_iterations=args.max_iterations,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            sample=args.sample,
            instance_ids=args.instance_ids.split(",") if args.instance_ids else None,
            run_id=args.run_id,
            max_context_tokens=args.max_context_tokens,
            mcp_config=args.mcp_config,
            prompt_template=args.prompt_template,
            min_free_disk_gb=args.min_free_disk_gb,
            record_internals=args.record_internals,
            eviction_config=eviction_config,
            sparse_attention_config=sparse_attention_config,
        )
    )
    print(f"Traces written to: {run_dir}/")
    results_path = run_dir / "results.jsonl"
    if results_path.exists():
        print(f"Results written to: {results_path}")


def _resolve_simulate_output_dir(args: argparse.Namespace) -> Path:
    """Return the base dir; structured subpath is resolved by simulator.simulate() when at default."""
    return Path(args.output_dir)


def _run_simulate(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from trace_collect.simulator import simulate, validate_gpu_tracking_args

    try:
        validate_gpu_tracking_args(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    simulate_kwargs = {
        "source_trace": Path(args.source_trace) if args.source_trace else None,
        "trace_manifest": Path(args.trace_manifest) if args.trace_manifest else None,
        "task_source": Path(args.task_source),
        "output_dir": _resolve_simulate_output_dir(args),
        "mode": args.mode,
        "container_executable": args.container,
        "network_mode": args.network_mode,
        "command_timeout_s": args.command_timeout,
        "warmup_skip_iterations": args.warmup_skip_iterations,
        "replay_speed": args.replay_speed,
        "arrival_mode": args.arrival_mode,
        "arrival_rate_per_s": args.arrival_rate_per_s,
        "arrival_seed": args.arrival_seed,
        "structured_output": args.output_dir == "traces/simulate",
    }

    if args.mode == "cloud_model":
        if args.metrics_url:
            print(
                "ERROR: cloud_model replay does not support --metrics-url.",
                file=sys.stderr,
            )
            sys.exit(2)
        trace_file = asyncio.run(simulate(**simulate_kwargs))
        print(f"Simulate trace written to: {trace_file}")
        return

    if args.trace_manifest:
        print(
            "ERROR: local_model mode accepts only --source-trace, not --trace-manifest.",
            file=sys.stderr,
        )
        sys.exit(2)
    if not args.api_base:
        print(
            "ERROR: local_model simulate requires --api-base for the target "
            "OpenAI-compatible endpoint.",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        llm_config = resolve_llm_config(
            provider=args.provider,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            environ=os.environ,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    if not llm_config.api_key:
        print(
            f"ERROR: Set {llm_config.env_key} or pass --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    gpu_tracking_kwargs: dict = {}
    if getattr(args, "gpu_tracking", "off") == "on":
        from harness.vllm_startup_parser import parse_startup_log_file

        gpu_baseline = parse_startup_log_file(args.vllm_startup_log)
        if gpu_baseline is None:
            print(
                f"ERROR: Failed to parse vLLM startup log at {args.vllm_startup_log}; "
                "check log file content and vLLM version (supported 0.5–0.7)",
                file=sys.stderr,
            )
            sys.exit(2)
        gpu_tracking_kwargs = {
            "gpu_baseline": gpu_baseline,
            "vllm_pid": args.vllm_pid,
            "gpu_sample_hz": args.gpu_sample_hz,
        }

    trace_file = asyncio.run(
        simulate(
            **simulate_kwargs,
            api_base=llm_config.api_base,
            api_key=llm_config.api_key,
            model=llm_config.model,
            metrics_url=args.metrics_url,
            **gpu_tracking_kwargs,
        )
    )
    print(f"Simulate trace written to: {trace_file}")


def _run_import_claude_code(args: argparse.Namespace) -> None:
    """Convert a Claude Code session JSONL into a canonical trace for the Gantt viewer."""
    from trace_collect.claude_code_import import import_claude_code_session

    trace_file = import_claude_code_session(
        session_path=Path(args.session),
        output_dir=Path(args.output_dir),
        include_sidechains=args.include_sidechains,
        run_id=args.run_id,
    )
    print(f"Claude Code trace written to: {trace_file}")
    print(
        "Start the dynamic Gantt viewer with: "
        "python -m trace_collect.cli gantt-serve --dev"
    )


def _run_inspect(argv: list[str]) -> None:
    import argparse as _argparse
    from trace_collect.trace_inspector import (
        TraceData,
        cmd_overview,
        cmd_step,
        cmd_messages,
        cmd_response,
        cmd_events,
        cmd_tools,
        cmd_search,
        cmd_timeline,
    )

    parser = _argparse.ArgumentParser(
        prog="python -m trace_collect.cli inspect",
        description="Inspect an OpenClaw JSONL trace file.",
        epilog="""commands:
  overview   Summary stats: steps, tokens, tool counts, elapsed time
  step N     Full details of step N (0-indexed): LLM stats, tool call, result
  messages N Show messages_in (prompt list) for step N
  response N Show raw_response (LLM output) for step N
  events     List fine-grained events (SCHEDULING, SESSION, TOOL, LLM, ...)
  tools      Tool usage breakdown: name, count, total duration, success rate
  search P   Regex search through llm_output fields across all steps
  timeline   Concise per-step timeline with icons, relative timestamps, durations

examples:
  %(prog)s trace.jsonl overview
  %(prog)s trace.jsonl step 3 --full
  %(prog)s trace.jsonl messages 0 --role user
  %(prog)s trace.jsonl response 5 --truncate 500
  %(prog)s trace.jsonl events --category SCHEDULING
  %(prog)s trace.jsonl events --category TOOL --iteration 2
  %(prog)s trace.jsonl tools
  %(prog)s trace.jsonl search "def main"
  %(prog)s trace.jsonl overview --json
  %(prog)s trace.jsonl step 0 --agent django
  %(prog)s trace.jsonl timeline
  %(prog)s trace.jsonl timeline --agent django""",
        formatter_class=_argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("trace", help="Path to the JSONL trace file.")
    parser.add_argument(
        "command",
        choices=[
            "overview",
            "step",
            "messages",
            "response",
            "events",
            "tools",
            "search",
            "timeline",
        ],
        help="Inspection command (see above).",
    )
    parser.add_argument(
        "args",
        nargs="*",
        help="Command argument: step index (for step/messages/response) or regex pattern (for search).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output as JSON for machine consumption.",
    )
    parser.add_argument(
        "--truncate",
        type=int,
        default=2000,
        help="Truncate long fields to N chars (default: 2000, 0=no truncation).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Disable truncation (show complete content).",
    )
    parser.add_argument(
        "--agent",
        default=None,
        help="Filter records by agent_id substring.",
    )
    parser.add_argument(
        "--role",
        default=None,
        help="Filter messages by role (system/user/assistant/tool). Used with 'messages' command.",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Filter events by category (SCHEDULING/SESSION/CONTEXT/TOOL/LLM/MCP/MEMORY/SUBAGENT). Used with 'events' command.",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Filter events by iteration number. Used with 'events' command.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Filter tool stats by step index. Used with 'tools' command.",
    )
    parsed = parser.parse_args(argv)

    truncate = 0 if parsed.full else parsed.truncate
    data = TraceData.load(Path(parsed.trace), agent_filter=parsed.agent)

    def _parse_step_idx(args: list[str]) -> int:
        if not args:
            return 0
        try:
            return int(args[0])
        except ValueError:
            parser.error(f"step index must be an integer, got: {args[0]!r}")

    cmd = parsed.command
    if cmd == "overview":
        cmd_overview(data, as_json=parsed.as_json)
    elif cmd == "step":
        step_n = _parse_step_idx(parsed.args)
        cmd_step(data, step_n, truncate=truncate, as_json=parsed.as_json)
    elif cmd == "messages":
        step_n = _parse_step_idx(parsed.args)
        cmd_messages(
            data,
            step_n,
            role_filter=parsed.role,
            truncate=truncate,
            as_json=parsed.as_json,
        )
    elif cmd == "response":
        step_n = _parse_step_idx(parsed.args)
        cmd_response(data, step_n, truncate=truncate, as_json=parsed.as_json)
    elif cmd == "events":
        cmd_events(
            data,
            category=parsed.category,
            iteration=parsed.iteration,
            as_json=parsed.as_json,
        )
    elif cmd == "tools":
        cmd_tools(data, step_idx=parsed.step, as_json=parsed.as_json)
    elif cmd == "search":
        pattern = parsed.args[0] if parsed.args else ""
        cmd_search(data, pattern, truncate=truncate, as_json=parsed.as_json)
    elif cmd == "timeline":
        if parsed.as_json:
            print(json.dumps({"error": "timeline does not support --json output"}))
            return
        cmd_timeline(data)


if __name__ == "__main__":
    main()
