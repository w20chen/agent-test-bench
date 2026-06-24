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

All benchmark runs use the same collection entry point. Replace the provider,
model, and credentials in these examples with the LLM endpoint under test.
OpenClaw requires an explicit `--mcp-config`; pass `none` to record that MCP is
intentionally disabled. Traces are written below the benchmark's configured
`trace_root` (normally `traces/<benchmark>/`), grouped by model and timestamp.

### SWE-Bench Verified

Prepare the selected tasks and their repositories once:

```bash
conda activate ML
make download-swebench-verified
make setup-swebench-repos
```

Run a sampled task with Docker (replace `--sample 1` with
`--instance-ids <id>` to select an exact case):

```bash
DEEPSEEK_API_KEY=... PYTHONPATH=src python -m trace_collect.cli \
    --provider deepseek \
    --model deepseek-chat \
    --benchmark swe-bench-verified \
    --scaffold openclaw \
    --container docker \
    --mcp-config none \
    --sample 1
```

### SWE-rebench

Prepare the filtered split and repositories once:

```bash
conda activate ML
make setup-swe-rebench
```

Run:

```bash
DEEPSEEK_API_KEY=... PYTHONPATH=src python -m trace_collect.cli \
    --provider deepseek \
    --model deepseek-chat \
    --benchmark swe-rebench \
    --scaffold openclaw \
    --container docker \
    --mcp-config none \
    --sample 1
```

SWE-rebench task images are pulled on demand. They can be prefetched with
`make pull-swe-rebench-images`; on ARM hosts, follow the QEMU or native setup
in [Getting Started](docs/getting-started.md).

### Terminal-Bench

Terminal-Bench requires Python 3.12+, the project dependencies (including the
`terminal-bench` package), and a working Docker daemon. Its pinned
`terminal-bench-core` dataset is resolved automatically through
`configs/benchmarks/terminal_bench_registry.json`.

```bash
DEEPSEEK_API_KEY=... PYTHONPATH=src python -m trace_collect.cli \
    --provider deepseek \
    --model deepseek-chat \
    --benchmark terminal-bench \
    --scaffold openclaw \
    --container docker \
    --mcp-config none \
    --sample 1
```

For the repository's known smoke case, use
`--instance-ids fix-git` instead of `--sample 1`.

### BFCL

BFCL runs on the host rather than in a task container. Clone the upstream
Gorilla repository separately, keep it read-only, and point this project at
the checkout:

```bash
git clone https://github.com/ShishirPatil/gorilla.git /path/to/gorilla
git -C /path/to/gorilla checkout <tested-gorilla-commit>
git -C /path/to/gorilla rev-parse HEAD
export BFCL_REPO_PATH=/path/to/gorilla
```

Install the BFCL runtime dependencies required by the chosen category, then
run one of the registered slugs. This repository does not currently pin a
known-compatible Gorilla commit, so the experiment configuration must supply
and record `<tested-gorilla-commit>`.

```bash
DEEPSEEK_API_KEY=... BFCL_REPO_PATH=/path/to/gorilla \
PYTHONPATH=src python -m trace_collect.cli \
    --provider deepseek \
    --model deepseek-chat \
    --benchmark bfcl-multi-turn-base \
    --scaffold openclaw \
    --mcp-config none \
    --sample 1
```

Available BFCL slugs are `bfcl-multi-turn-base`,
`bfcl-multi-turn-long-context`, `bfcl-memory`, and `bfcl-web-search`.
`bfcl-memory` additionally needs the dependencies for its vector, KV, and
recursive-summary backends; `bfcl-web-search` needs a supported search API key
such as `SERPAPI_API_KEY`.

The current integration executes BFCL tasks and records OpenClaw traces. It
does not emit BFCL prediction files or invoke the official scoring pipeline,
so official BFCL scoring is not currently supported by this collection CLI.

Do not use the sampled command above for `bfcl-memory`. Its entries form
ordered prerequisite/question chains that share persisted state. Run the full
loaded dataset sequentially by omitting `--sample` and `--instance-ids`, or
provide a complete prerequisite-first chain of IDs:

```bash
DEEPSEEK_API_KEY=... BFCL_REPO_PATH=/path/to/gorilla \
PYTHONPATH=src python -m trace_collect.cli \
    --provider deepseek \
    --model deepseek-chat \
    --benchmark bfcl-memory \
    --scaffold openclaw \
    --mcp-config none
```

See [Supported Benchmarks](docs/benchmarks.md) for runtime and scoring details,
and [Trace Collect](docs/trace-collect.md) for task selection, resuming, and
concurrency options.

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
