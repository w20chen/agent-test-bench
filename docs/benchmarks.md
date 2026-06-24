# Supported Benchmarks

> This document is part of the [Agent Sched Bench manual](../README.md).
> For running benchmarks, see [Trace Collect](trace-collect.md).

The repo ships with plugin-based benchmark support. Each benchmark is defined
by a YAML config in `configs/benchmarks/` and a Python plugin in
`src/agents/benchmarks/`.

---

## Table of Contents

- [Benchmark Catalog](#benchmark-catalog)
- [BFCL Function Calling Benchmarks](#bfcl-function-calling-benchmarks)
- [Plugin Architecture](#plugin-architecture)

---

## Benchmark Catalog

| Slug | `task_shape` | Dataset | Split | Docker | Scaffolds | Scoring |
|---|---|---|---|---|---|---|
| `swe-bench-verified` | `swe_patch` | `princeton-nlp/SWE-bench_Verified` | `test` | `swebench/sweb.eval.x86.*` (namespace-prefixed) | openclaw | harness (pytest in container) |
| `swe-rebench` | `swe_patch` | `nebius/SWE-rebench` | `filtered` | `swerebench/sweb.eval.x86_64.*` (fully qualified) | openclaw | harness (pytest in container) |
| `terminal-bench` | `terminal_task` | `terminal-bench-core` (or local task dir) | N/A | Terminal-Bench-managed | openclaw (phase 1) | Terminal-Bench harness + imported OpenClaw trace |
| `deep-research-bench` | `research_qa` | configured in YAML | `test` | host | openclaw, tongyi-deepresearch | reference-answer comparison |
| `browsecomp` | `browse_qa` | configured in YAML | `test` | host | openclaw, tongyi-deepresearch | reference-answer comparison |

Terminal-Bench requires Python 3.12+ (upstream `tb` CLI dependency) and only
supports `--scaffold openclaw` with the Docker runtime in phase 1.

---

## BFCL Function Calling Benchmarks

Four BFCL (Berkeley Function Calling Leaderboard) datasets are wired as
host-mode benchmarks driven by the OpenClaw scaffold. BFCL runs as a read-only
external environment: its dataset loader, tool-doc → schema converter, stateful
backend instantiation, and backend implementations (simulated filesystem,
booking, web search, vector/kv/rec_sum memory) are reused unmodified; OpenClaw
provides the agent loop that interacts with these backends via the BFCL
function-calling protocol.

| Slug | BFCL Version | Multi-turn | Mode |
|------|-------------|------------|------|
| `bfcl-v3` | v3 | No | host |
| `bfcl-v3-multi` | v3 | Yes | host |
| `bfcl-v4` | v4 | No | host |
| `bfcl-v4-multi` | v4 | Yes | host |

```bash
# Example: run BFCL v3 single-turn
PYTHONPATH=src python -m trace_collect.cli \
    --provider deepseek \
    --model deepseek-chat \
    --benchmark bfcl-v3 \
    --scaffold openclaw \
    --sample 1 \
    --mcp-config none
```

See `configs/benchmarks/bfcl-*.yaml` for full benchmark configurations.

---

## Plugin Architecture

All benchmarks MUST be added via the plugin layer in `src/agents/benchmarks/`
and `configs/benchmarks/<slug>.yaml`. To add a new benchmark, create a YAML
config and a Python class inheriting from `agents.benchmarks.base.Benchmark`.

Hardcoding dataset names or adding per-benchmark CLI flags is forbidden —
benchmark specifics belong in the YAML config. The canonical entry point for
creating evaluation tasks is `EvalTask.from_benchmark_instance(row,
workspace_base, benchmark=<plugin>)`.

See `AGENTS.md` for the full rules governing benchmark plugins.
