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

## Quick Start

```bash
conda activate ML
make test    # run pytest
make lint    # ruff
```

On a fresh server, run `bash scripts/setup/bootstrap.sh` once — it installs
miniconda, creates env ML, and installs all deps.

---

## Repository Layout

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

## Manual

Detailed documentation lives under `docs/`.  Pick your entry point:

| Document | What it covers |
|----------|---------------|
| [Getting Started](docs/getting-started.md) | Dev environment, ARM server walkthrough, QEMU setup, troubleshooting |
| [Trace Collect](docs/trace-collect.md) | CLI reference, concurrent execution, recording internals, ksys metrics |
| [Case Inspection](docs/case-inspection.md) | Browsing SWE-bench tasks without running an agent |
| [Benchmarks](docs/benchmarks.md) | Registered benchmarks, BFCL, plugin architecture |
| [Resource Measurement](docs/resource-measurement.md) | CPU/memory/disk/network/PMU sampling architecture |

### Three Entry Points

1. **Inspect cases** — browse benchmark tasks without running an agent.
   → [Case Inspection](docs/case-inspection.md)

2. **Run an agent interactively** — send a one-shot prompt to OpenClaw:
   ```bash
   PYTHONPATH=src python -m agents.openclaw \
       --prompt "Write a Python script to download web page and parse title" \
       --provider deepseek --model deepseek-chat --workspace ./workspace
   ```
   → [Getting Started](docs/getting-started.md#quick-test-end-to-end-walkthrough-arm-server)

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

## Supported Benchmarks (at a glance)

| Slug | Type | Runtime | Scaffolds |
|------|------|---------|-----------|
| `swe-bench-verified` | SWE patch | Docker | openclaw |
| `swe-rebench` | SWE patch | Docker | openclaw |
| `terminal-bench` | Terminal task | Docker | openclaw |
| `deep-research-bench` | Research QA | Host | openclaw, tongyi-deepresearch |
| `browsecomp` | Browse QA | Host | openclaw, tongyi-deepresearch |
| `bfcl-v3` / `bfcl-v4` | Function calling | Host | openclaw |

Full details → [Benchmarks](docs/benchmarks.md)
