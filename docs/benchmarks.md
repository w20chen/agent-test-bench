# Supported Benchmarks

> This document is part of the [Agent Sched Bench manual](../README.md).
> For running benchmarks, see [Trace Collect](trace-collect.md).

The repo ships with plugin-based benchmark support. Each benchmark is defined
by a YAML config in `configs/benchmarks/` and a Python plugin in
`src/agents/benchmarks/`.

---

## Table of Contents

- [Benchmark Catalog](#benchmark-catalog)
- [SWE-Bench Verified](#swe-bench-verified)
- [SWE-rebench](#swe-rebench)
- [Terminal-Bench](#terminal-bench)
- [Deep Research Bench & BrowseComp](#deep-research-bench--browsecomp)
- [BFCL Function Calling Benchmarks](#bfcl-function-calling-benchmarks)
- [Plugin Architecture](#plugin-architecture)

---

## Benchmark Catalog

The table below lists every registered benchmark, its task shape, data
source, runtime environment, and supported agent scaffolds.

| Slug | `task_shape` | Dataset | Split | Docker | Scaffolds | Scoring |
|---|---|---|---|---|---|---|
| `swe-bench-verified` | `swe_patch` | `princeton-nlp/SWE-bench_Verified` | `test` | `swebench/sweb.eval.x86.*` (namespace-prefixed) | openclaw | harness (pytest in container) |
| `swe-rebench` | `swe_patch` | `nebius/SWE-rebench` | `filtered` | `swerebench/sweb.eval.x86_64.*` (fully qualified) | openclaw | harness (pytest in container) |
| `terminal-bench` | `terminal_task` | `terminal-bench-core` (or local task dir) | N/A | Terminal-Bench-managed | openclaw (phase 1) | Terminal-Bench harness + imported OpenClaw trace |
| `deep-research-bench` | `research_qa` | configured in YAML | `test` | host | openclaw, tongyi-deepresearch | reference-answer comparison |
| `browsecomp` | `browse_qa` | configured in YAML | `test` | host | openclaw, tongyi-deepresearch | reference-answer comparison |
| `bfcl-multi-turn-base` | BFCL tool use | BFCL `multi_turn_base` | N/A | host | openclaw | completion status + trace |
| `bfcl-multi-turn-long-context` | BFCL tool use | BFCL `multi_turn_long_context` | N/A | host | openclaw | completion status + trace |
| `bfcl-memory` | BFCL memory | BFCL `memory_vector`, `memory_kv`, `memory_rec_sum` | N/A | host | openclaw | completion status + trace |
| `bfcl-web-search` | BFCL web search | BFCL `web_search_base`, `web_search_no_snippet` | N/A | host | openclaw | completion status + trace |

Terminal-Bench requires Python 3.12+ (upstream `tb` CLI dependency) and only
supports `--scaffold openclaw` with the Docker runtime in phase 1.

---

## SWE-Bench Verified

Container-mode benchmark. Prepare tasks and repositories once:

```bash
conda activate ML
make download-swebench-verified
make setup-swebench-repos
```

Run with Docker (replace `--sample 1` with `--instance-ids <id>` to select a
specific case):

```bash
PYTHONPATH=src python -m trace_collect.cli \
    --provider deepseek --model deepseek-chat \
    --benchmark swe-bench-verified --scaffold openclaw \
    --container docker --mcp-config none \
    --sample 1
```

---

## SWE-rebench

Container-mode benchmark on the filtered SWE-rebench split. Prepare once:

```bash
conda activate ML
make setup-swe-rebench
```

Run:

```bash
PYTHONPATH=src python -m trace_collect.cli \
    --provider deepseek --model deepseek-chat \
    --benchmark swe-rebench --scaffold openclaw \
    --container docker --mcp-config none \
    --sample 1
```

Task images are pulled on demand. Prefetch with `make pull-swe-rebench-images`.
On ARM hosts, see the QEMU or native setup in
[Getting Started](getting-started.md).

---

## Terminal-Bench

Requires Python 3.12+, the project dependencies (including the `terminal-bench`
package), and a working Docker daemon. Its pinned `terminal-bench-core` dataset
is resolved through `configs/benchmarks/terminal_bench_registry.json`.

```bash
PYTHONPATH=src python -m trace_collect.cli \
    --provider deepseek --model deepseek-chat \
    --benchmark terminal-bench --scaffold openclaw \
    --container docker --mcp-config none \
    --sample 1
```

For the repository's known smoke case, use `--instance-ids fix-git` instead of
`--sample 1`.

---

## Deep Research Bench & BrowseComp

Host-mode benchmarks — no Docker containers required. See the
[Trace Collect](trace-collect.md#deep-research-bench) CLI reference for
invocation examples. Deep Research Bench supports `--prompt-template`
switching between `default` (spawn subagents) and `no_spawn` (single-agent).

---

## BFCL Function Calling Benchmarks

In addition to the SWE-patch and QA benchmarks above, the repo also supports
function-calling evaluation through the BFCL suite. These benchmarks test an
agent's ability to use tools correctly across single-turn and multi-turn
interactions.

Four BFCL (Berkeley Function Calling Leaderboard) plugin groups are wired as
host-mode benchmarks driven by the OpenClaw scaffold. BFCL runs as a read-only
external environment: its dataset loader, tool-doc → schema converter, stateful
backend instantiation, and backend implementations (simulated filesystem,
booking, web search, vector/kv/rec_sum memory) are reused unmodified; OpenClaw
provides the agent loop that interacts with these backends via the BFCL
function-calling protocol.

| Slug | BFCL categories | Additional requirements |
|------|-----------------|-------------------------|
| `bfcl-multi-turn-base` | `multi_turn_base` | BFCL base runtime dependencies |
| `bfcl-multi-turn-long-context` | `multi_turn_long_context` | BFCL base runtime dependencies |
| `bfcl-memory` | `memory_vector`, `memory_kv`, `memory_rec_sum` | `faiss-cpu`, `sentence-transformers`, and `rank_bm25` |
| `bfcl-web-search` | `web_search_base`, `web_search_no_snippet` | Supported web-search key, for example `SERPAPI_API_KEY` |

BFCL is not vendored by this project. Clone the Gorilla repository separately
and set `BFCL_REPO_PATH` to its root. The benchmark loader expects the Python
package at
`$BFCL_REPO_PATH/berkeley-function-call-leaderboard/bfcl_eval`.

```bash
git clone https://github.com/ShishirPatil/gorilla.git /path/to/gorilla
git -C /path/to/gorilla checkout <tested-gorilla-commit>
git -C /path/to/gorilla rev-parse HEAD
export BFCL_REPO_PATH=/path/to/gorilla
```

Install the BFCL dependencies needed by the selected category from that
checkout. The checkout is treated as read-only by this project. This repository
does not currently pin a known-compatible Gorilla commit, so each experiment
must choose and record `<tested-gorilla-commit>` rather than relying on a
moving branch.

```bash
# Example: run one BFCL multi-turn base task
DEEPSEEK_API_KEY=... BFCL_REPO_PATH=/path/to/gorilla \
PYTHONPATH=src python -m trace_collect.cli \
    --provider deepseek \
    --model deepseek-chat \
    --benchmark bfcl-multi-turn-base \
    --scaffold openclaw \
    --sample 1 \
    --mcp-config none
```

Switch `--benchmark` to `bfcl-multi-turn-long-context` or
`bfcl-web-search` to sample those plugins. The YAML files under
`configs/benchmarks/bfcl-*.yaml` document iteration limits and plugin-specific
requirements. Their `selection_n` and `selection_seed` fields are not currently
applied by the trace-collection path; task selection is controlled only by
`--sample` and `--instance-ids`.

The current runner marks whether the OpenClaw conversation completed and
records the full trace, timings, tool calls, and metadata. It does not call
BFCL's separate official evaluator or emit its prediction-file format.
Official BFCL scoring is therefore not currently supported by this collection
CLI, and completion status must not be reported as leaderboard accuracy.

### BFCL Memory Selection

`bfcl-memory` is not safe to run with an arbitrary `--sample N`. Its dataset
contains ordered prerequisite entries that populate persisted memory before
later question entries. The current collection path slices tasks directly and
does not invoke the plugin's chain-preserving `select_subset` method.

Run the full loaded memory dataset sequentially by omitting both task-selection
flags:

```bash
DEEPSEEK_API_KEY=... BFCL_REPO_PATH=/path/to/gorilla \
PYTHONPATH=src python -m trace_collect.cli \
    --provider deepseek \
    --model deepseek-chat \
    --benchmark bfcl-memory \
    --scaffold openclaw \
    --mcp-config none
```

For a partial run, pass `--instance-ids` only when the list contains the
complete prerequisite/question chain in prerequisite-first order. Do not
derive a memory evaluation subset with `--sample`.

---

## Plugin Architecture

New benchmarks are added through a plugin system rather than by modifying
core harness code. This keeps benchmark-specific logic isolated and makes
the harness reusable across different evaluation domains.

All benchmarks MUST be added via the plugin layer in `src/agents/benchmarks/`
and `configs/benchmarks/<slug>.yaml`. To add a new benchmark, create a YAML
config and a Python class inheriting from `agents.benchmarks.base.Benchmark`.

Hardcoding dataset names or adding per-benchmark CLI flags is forbidden —
benchmark specifics belong in the YAML config. The canonical entry point for
creating evaluation tasks is `EvalTask.from_benchmark_instance(row,
workspace_base, benchmark=<plugin>)`.

See `AGENTS.md` for the full rules governing benchmark plugins.
