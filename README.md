# Agent Sched Bench

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Docs-6C5CE7.svg)](https://deepwiki.com/w20chen/agent-test-bench)

Benchmark environment for studying agent scheduling and KV-cache management on
multi-step LLM workloads. The repo ships three top-level capabilities:

1. **Trace collect** - run agent scaffolds on benchmark tasks inside
   containers and record canonical JSONL traces (`python -m trace_collect.cli`).
2. **Trace simulate** - replay collected traces at scale to measure
   scheduling-sensitive timing under controlled arrival patterns. Uses
   `--workers` to distribute agents across independent asyncio event loops, a
   cross-process barrier to synchronize replay start, and `--prep-concurrency`
   to rate-limit container preparation.
   See [Trace Collect](docs/trace-collect.md#simulate-trace-replay).
3. **HTML trace visualization** - inspect traces with
   `src/trace_collect/html_viz.py`. The older FastAPI + Solid.js Gantt viewer
   remains under `demo/gantt_viewer/`, but is deprecated.

`AGENTS.md` and `CLAUDE.md` define research-integrity and process rules.

## Quick Start

```bash
conda activate ML
make test    # run pytest
make lint    # run ruff
```

On a fresh server, run `bash scripts/setup/bootstrap.sh` once. It installs
Miniconda, creates the `ML` environment, and installs project dependencies.

## Environment Variables

The project uses `python-dotenv` to load environment variables from a `.env`
file at startup. This file is never committed. Copy the template and edit it:

```bash
cp .env.example .env
# edit .env with DEEPSEEK_API_KEY, TAVILY_API_KEY, etc.
```

Shell environment variables take priority over `.env`. The `.env` file acts as
a fallback for keys that are not explicitly exported.

Without a `.env` file or shell exports, pass credentials inline:

```bash
DEEPSEEK_API_KEY=sk-... PYTHONPATH=src python -m trace_collect.cli ...
```

Other LLM providers such as OpenRouter and DashScope, and web search providers
such as Brave and DuckDuckGo, are also supported.

## Repository Layout

```text
agent-sched-bench/
|-- configs/            # benchmark, system, trace_collect, and sweep YAMLs
|-- demo/gantt_viewer/  # deprecated FastAPI + Solid.js viewer
|-- docs/               # documentation and current plans
|-- scripts/            # setup, smoke, experiment, and analysis utilities
|-- src/
|   |-- agents/         # scaffolds and benchmark plugins
|   |-- harness/        # runner, samplers, metrics, trace logger
|   |-- llm_call/       # provider registry and OpenAI-compatible client
|   |-- serving/        # serving launchers, KV policies, sparse attention
|   `-- trace_collect/  # collect, simulate, import, inspect, visualization
`-- tests/
```

For a detailed map of `scripts/`, see [Script Inventory](docs/scripts.md).

## Entry Points

1. **Inspect benchmark cases** without running an agent:

   ```bash
   python scripts/inspect_swebench.py --benchmark swe-bench-verified list
   ```

   See [Case Inspection](docs/case-inspection.md).

2. **Run an agent interactively** with OpenClaw:

   ```bash
   PYTHONPATH=src python -m agents.openclaw \
       --prompt "Write a Python script to download web page and parse title" \
       --provider deepseek --model deepseek-chat --workspace ./workspace
   ```

   See [Getting Started](docs/getting-started.md#end-to-end-walkthrough-arm-server).

3. **Run a benchmark** with container orchestration and trace collection:

   ```bash
   PYTHONPATH=src python -m trace_collect.cli \
       --provider dashscope --model qwen-plus-latest \
       --benchmark swe-rebench --scaffold openclaw \
       --mcp-config none --sample 2
   ```

   See [Trace Collect](docs/trace-collect.md).

4. **Replay collected traces at scale**:

   ```bash
   PYTHONPATH=src python -m trace_collect.cli simulate \
       --trace-manifest manifest.json --mode cloud_model \
       --container docker --replay-speed 50 \
       --workers 320 --prep-concurrency 64 \
       --arrival-mode poisson --arrival-rate-per-s 0.5
   ```

   See [Simulate Trace Replay](docs/trace-collect.md#simulate-trace-replay).

## Running Benchmarks

All benchmarks use the same collection entry point. Replace the provider,
model, benchmark slug, and credentials as needed. OpenClaw requires
`--mcp-config`; pass `none` to explicitly run without MCP.

```bash
PYTHONPATH=src python -m trace_collect.cli \
    --provider deepseek --model deepseek-chat \
    --benchmark <slug> --scaffold openclaw \
    --container docker --mcp-config none \
    --sample 1
```

Traces land under the benchmark's `trace_root`, normally `traces/<benchmark>/`,
grouped by model and timestamp.

Each benchmark has its own setup requirements. See
[Benchmarks](docs/benchmarks.md) for SWE-Bench Verified, SWE-rebench,
Terminal-Bench, Deep Research Bench, BrowseComp, and BFCL.

For task selection, concurrency, resuming, monitoring controls, and per-tool
profiling, see [Trace Collect](docs/trace-collect.md). For simulate mode, see
the [simulate section](docs/trace-collect.md#simulate-trace-replay).

## Documentation

| Document | What it covers |
|---|---|
| [Getting Started](docs/getting-started.md) | Dev environment, ARM walkthrough, QEMU setup, troubleshooting |
| [Trace Collect](docs/trace-collect.md) | CLI reference, concurrent execution, recording internals, ksys metrics |
| [Task Container Environment](docs/task-container-environment.md) | Container startup, OpenClaw bootstrap, pip shims, and benchmark boundaries |
| [Case Inspection](docs/case-inspection.md) | Browsing SWE-bench tasks without running an agent |
| [Benchmarks](docs/benchmarks.md) | Registered benchmarks, BFCL, plugin architecture |
| [Resource Measurement](docs/resource-measurement.md) | CPU/memory/disk/network/PMU sampling, per-tool profiler |
| [VTune Profiling](docs/vtune-profiling.md) | Per-tool VTune setup, architecture, and output format |
| [Script Inventory](docs/scripts.md) | Purpose and stability of scripts under `scripts/` |
| [Kunpeng LLC Experiments](docs/kunpeng-llc-experiments.md) | LLC-related experiments on Kunpeng |
| [Runtime Prediction](docs/runtime-prediction.md) | Predicting execution time for pip install, Python script, and pytest commands |

## Supported Benchmarks

| Slug | Type | Runtime | Scaffolds |
|---|---|---|---|
| `swe-bench-verified` | SWE patch | Docker | openclaw |
| `swe-rebench` | SWE patch | Docker | openclaw |
| `terminal-bench` | Terminal task | Docker | openclaw |
| `deep-research-bench` | Research QA | Host | openclaw, tongyi-deepresearch |
| `browsecomp` | Browse QA | Host | openclaw, tongyi-deepresearch |
| `bfcl-multi-turn-base` | Function calling | Host | openclaw |
| `bfcl-multi-turn-long-context` | Function calling | Host | openclaw |
| `bfcl-memory` | Function calling | Host | openclaw |
| `bfcl-web-search` | Function calling | Host | openclaw |

Full details are in [Benchmarks](docs/benchmarks.md).
