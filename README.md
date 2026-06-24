# Agent Sched Bench

Benchmark environment for studying agent scheduling and KV-cache management on
multi-step LLM workloads. The repo ships three top-level capabilities:

1. **Trace collect** — run agent scaffolds on benchmark tasks inside containers
   and record canonical JSONL traces (`python -m trace_collect.cli`).
2. **Trace simulate** — replay collected traces under new arrival patterns or
   against a local serving stack to measure scheduling-sensitive timing
   (`python -m trace_collect.cli simulate`).
3. **Gantt viewer demo** — an interactive FastAPI + Solid.js viewer under
   `demo/gantt_viewer/` for inspecting traces as multi-lane Gantt charts with
   resource overlays.

`AGENTS.md` and `CLAUDE.md` define research-integrity and process rules.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Environment Variables (.env)](#environment-variables-env)
- [Repository Layout](#repository-layout)
- [Entry Points](#entry-points)
- [Running Benchmarks](#running-benchmarks)
- [Documentation](#documentation)
- [Supported Benchmarks](#supported-benchmarks)

---

## Quick Start

```bash
conda activate ML
make test    # run pytest
make lint    # ruff
```

On a fresh server, run `bash scripts/setup/bootstrap.sh` once — it installs
miniconda, creates env ML, and installs all deps.

---

## Environment Variables (.env)

The project uses `python-dotenv` to load environment variables from a `.env`
file at startup.  This file is **never committed** (it is in `.gitignore`).
Copy the template and edit it with your keys:

```bash
cp .env.example .env
# edit .env — fill in DEEPSEEK_API_KEY, TAVILY_API_KEY, etc.
```

**Precedence:** shell environment variables always take priority over `.env`.
If you `export DEEPSEEK_API_KEY=...` in your shell, the `.env` value is
ignored.  `.env` acts as a **fallback** for keys you haven't explicitly set.

The template includes placeholders for:
- `DEEPSEEK_API_KEY` — DeepSeek API key (used by `--provider deepseek`)
- `TAVILY_API_KEY` — Tavily search API key (used by the web search tool)
- `MODEL_PATH`, `VLLM_*`, `CONTINUUM_*`, `THUNDERAGENT_*` — serving stack configuration

Without a `.env` file or shell exports, you must pass credentials inline:

```bash
DEEPSEEK_API_KEY=sk-... PYTHONPATH=src python -m trace_collect.cli ...
```

---

## Repository Layout

The codebase is organised around a few top-level directories. Here is how
they map to the capabilities described above:

```text
agent-sched-bench/
├── configs/            # benchmark / system / trace_collect / sweep YAMLs
├── demo/gantt_viewer/  # FastAPI backend + Solid.js frontend
├── docs/               # detailed manual (you are reading the index)
├── scripts/            # setup, download, smoke, and runner shells
├── src/
│   ├── agents/         # scaffolds + benchmark plugins
│   ├── harness/        # runner, samplers, metrics, trace logger
│   ├── llm_call/       # provider registry + OpenAI-compatible client
│   └── trace_collect/  # CLI: collect / simulate / import / inspect / gantt-serve
└── tests/
```

---

## Entry Points

If you are new to the project, the quickest way to get oriented is to work
through the three entry points below — from passive inspection to full
benchmark runs:

Three progressively deeper ways to interact with this repo:

1. **Inspect cases** — browse benchmark tasks without running an agent.

   ```bash
   python scripts/inspect_swebench.py --benchmark swe-bench-verified list
   ```

   → [Case Inspection](docs/case-inspection.md)

2. **Run an agent interactively** — send a one-shot prompt to OpenClaw:

   ```bash
   PYTHONPATH=src python -m agents.openclaw \
       --prompt "Write a Python script to download web page and parse title" \
       --provider deepseek --model deepseek-chat --workspace ./workspace
   ```

   → [Getting Started](docs/getting-started.md#end-to-end-walkthrough-arm-server)

3. **Run a full benchmark** — execute the agent on many tasks with container
   orchestration and trace collection:

   ```bash
   PYTHONPATH=src python -m trace_collect.cli \
       --provider dashscope --model qwen-plus-latest \
       --benchmark swe-rebench --scaffold openclaw \
       --mcp-config none --sample 2
   ```

   → [Trace Collect](docs/trace-collect.md)

---

## Running Benchmarks

All benchmarks use the same collection entry point. Replace the provider, model,
and credentials. OpenClaw requires `--mcp-config`; pass `none` to acknowledge
running without MCP. Traces land under the benchmark's `trace_root` (normally
`traces/<benchmark>/`), grouped by model and timestamp.

```bash
# Template — replace benchmark, provider, model
PYTHONPATH=src python -m trace_collect.cli \
    --provider deepseek --model deepseek-chat \
    --benchmark <slug> --scaffold openclaw \
    --container docker --mcp-config none \
    --sample 1
```

Each benchmark has its own setup requirements (data download, repo cloning,
image pulling). See [Benchmarks](docs/benchmarks.md) for per-benchmark setup
commands, invocation examples, and runtime notes for SWE-Bench Verified,
SWE-rebench, Terminal-Bench, Deep Research Bench, BrowseComp, and BFCL.

For task selection (`--sample`, `--instance-ids`), concurrency, resuming, and
monitoring controls, see [Trace Collect](docs/trace-collect.md).

---

## Documentation

Each entry point above links to a dedicated page that goes into depth. The
full set of documentation pages is listed below — pick the one that matches
your current task:

Detailed documentation lives under `docs/`:

| Document | What it covers |
|----------|---------------|
| [Getting Started](docs/getting-started.md) | Dev environment, ARM server walkthrough, QEMU setup, troubleshooting |
| [Trace Collect](docs/trace-collect.md) | CLI reference, concurrent execution, recording internals, ksys metrics |
| [Case Inspection](docs/case-inspection.md) | Browsing SWE-bench tasks without running an agent |
| [Benchmarks](docs/benchmarks.md) | Registered benchmarks, BFCL, plugin architecture |
| [Resource Measurement](docs/resource-measurement.md) | CPU/memory/disk/network/PMU sampling architecture |

---

## Supported Benchmarks

The following benchmarks are currently registered. Each benchmark has a
corresponding YAML config and Python plugin — see the [Benchmarks
documentation](docs/benchmarks.md) for full details including scoring
methodology and runtime requirements.

| Slug | Type | Runtime | Scaffolds |
|------|------|---------|-----------|
| `swe-bench-verified` | SWE patch | Docker | openclaw |
| `swe-rebench` | SWE patch | Docker | openclaw |
| `terminal-bench` | Terminal task | Docker | openclaw |
| `deep-research-bench` | Research QA | Host | openclaw, tongyi-deepresearch |
| `browsecomp` | Browse QA | Host | openclaw, tongyi-deepresearch |
| `bfcl-multi-turn-base` | Function calling | Host | openclaw |
| `bfcl-multi-turn-long-context` | Function calling | Host | openclaw |
| `bfcl-memory` | Function calling | Host | openclaw |
| `bfcl-web-search` | Function calling | Host | openclaw |

Full details → [Benchmarks](docs/benchmarks.md)
