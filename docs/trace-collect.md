# Trace Collect CLI

> This document is part of the [Agent Sched Bench manual](../README.md).
> For benchmark descriptions, see [Benchmarks](benchmarks.md).
> For environment setup, see [Getting Started](getting-started.md).

Run an agent scaffold on a benchmark and record a canonical v5 JSONL trace per
task. The CLI requires an explicit `--provider` and `--model` and loads
benchmark specifics from `configs/benchmarks/<slug>.yaml`.

---

## Table of Contents

- [Basic Usage](#basic-usage)
  - [CLI Flags Reference](#cli-flags-reference)
  - [OpenClaw Standalone CLI](#openclaw-standalone-cli)
  - [Deep Research Bench](#deep-research-bench)
- [Concurrent Execution](#concurrent-execution)
  - [Resource-monitoring defaults](#resource-monitoring-defaults)
- [Simulate: Trace Replay](#simulate-trace-replay)
  - [Two Simulation Modes](#two-simulation-modes)
  - [Trace Sources](#trace-sources)
  - [Simulate CLI Flags](#simulate-cli-flags)
  - [Arrival Modes](#arrival-modes)
- [N:M Trace-to-Agent Mapping](#nm-trace-to-agent-mapping)
- [CPU Core Limiting](#cpu-core-limiting)
- [N:M Simulation Sweep](#nm-simulation-sweep)
- [Recording Internals](#recording-internals)
- [Ksys System Metrics](#ksys-system-metrics)

---

## Basic Usage

The command below downloads benchmark data, sets up task repositories, and
runs the agent scaffold on two sampled tasks. Output traces are written under
`traces/<benchmark>/<model>/<timestamp>/`.

```bash
conda activate ML
make download-swe-rebench         # or download-swebench-verified
make setup-swe-rebench-repos      # or setup-swebench-repos
PYTHONPATH=src python -m trace_collect.cli \
    --provider dashscope \
    --model qwen-plus-latest \
    --benchmark swe-rebench \
    --scaffold openclaw \
    --mcp-config none \
    --sample 2
```

### CLI Flags Reference

The table below lists the most commonly used flags. For the complete
reference, see `src/trace_collect/CLAUDE.md`.

| Flag | Description |
|------|-------------|
| `--benchmark <slug>` | Benchmark to run (default `swe-bench-verified`) |
| `--scaffold` | `openclaw` or `tongyi-deepresearch` |
| `--mcp-config` | Required for `openclaw`; YAML path or the literal `none` |
| `--sample N` | Run only the first N tasks |
| `--instance-ids a,b,c` | Run only specified instance(s) |
| `--run-id <path>` | Resume an interrupted run |
| `--prompt-template <name>` | Override the benchmark default prompt |
| `--resource-monitoring auto\|on\|off` | Control CPU, memory, disk, network, context-switch, and host memory bandwidth sampling |
| `--pmu-monitoring auto\|on\|off` | Control PMU sampling; concurrent `on` is forbidden |
| `--ksys-monitoring auto\|on\|off` | Control Huawei Kunpeng system metrics |
| `--ksys` | Compatibility alias for `--ksys-monitoring on` |
| `--concurrency N` | Spawn N agent instances per task |
| `--provider` | LLM provider name |
| `--model` | Model identifier |

See `src/trace_collect/CLAUDE.md` for the complete flag reference, provider
registry, checkpointing behavior, and trace schema v5 layout.

### OpenClaw Standalone CLI

The repo also ships a standalone CLI for one-shot prompts without the harness:

```bash
PYTHONPATH=src python -m agents.openclaw \
    --prompt "Write a Python script to download web page and parse title" \
    --provider deepseek \
    --model deepseek-chat \
    --workspace ./workspace
```

Use `--async` for background runs and `--status --session-id <id>` to check
progress (see `openclaw --help`).

### Deep Research Bench

The deep research benchmark is a host-mode benchmark — it does not require
Docker containers. The following example demonstrates a typical invocation:

```bash
PYTHONPATH=src python -m trace_collect.cli \
    --provider deepseek \
    --model deepseek-v4-flash \
    --benchmark deep-research-bench \
    --scaffold openclaw \
    --sample 1 \
    --mcp-config none \
    --verbose
```

Deep research bench ships two prompt templates under
`configs/prompts/deep_research_bench/`:

| Template | Behavior |
|----------|-----------|
| `default` | Agent uses the `spawn` tool to launch 2–4 parallel subagents that decompose and research independent facets, then synthesizes their findings. |
| `no_spawn` | Pure single-agent mode — no subagent spawning. The agent searches, reads, and answers on its own. |

Switch with `--prompt-template <name>`, e.g. `--prompt-template no_spawn`.

---

## Concurrent Execution

Beyond single-task runs, the CLI supports spawning multiple agent instances
in parallel. This is primarily useful for stress-testing hardware under
realistic multi-agent workloads.

`--concurrency N` (default `1`) spawns **N agent instances simultaneously**
for each benchmark task.  This is designed for hardware stress-testing:
measure system-level performance (via `ksys`) under multi-agent load.

### Relationship with `--sample` and `--instance-ids`

These three flags are **orthogonal** and compose freely:

| Flag | Role | Default |
|------|------|---------|
| `--sample N` | Limits to the first **N tasks** from the benchmark | all tasks |
| `--instance-ids a,b,c` | Runs only the **specified instance(s)** | all tasks |
| `--concurrency N` | Spawns **N parallel attempts** per task | 1 (sequential) |

- `--sample` and `--instance-ids` control **which tasks** run.
  When both are given, `--instance-ids` selects first, then `--sample`
  truncates the result.
- `--concurrency` controls **how many attempts per task**.

> **Guard:** `--concurrency > 1` **requires** `--instance-ids` or
> `--sample`.  Running *all* benchmark instances with high concurrency
> is blocked — it would spawn hundreds of parallel containers and
> overwhelm the host.

**Examples of valid combinations:**

```bash
# 4 concurrent attempts on one specific instance
--concurrency 4 --instance-ids "django__django-12345"

# 4 concurrent attempts on each of 3 sampled instances (12 total)
--concurrency 4 --sample 3

# 4 concurrent attempts on each of 2 specific instances (8 total)
--concurrency 4 --instance-ids "django__django-12345,sympy__sympy-67890"

# Single attempt on each of 10 sampled instances (no concurrency)
--sample 10
```

### Resource-monitoring defaults

The CLI exposes three independent tri-state switches for resource monitoring:

| Flag | Controls | Default |
|------|----------|---------|
| `--resource-monitoring auto\|on\|off` | CPU, memory, disk I/O, network I/O, context switches, and host memory bandwidth | `auto` |
| `--pmu-monitoring auto\|on\|off` | CPU micro-architecture PMU counters (cache, branch, instructions) | `auto` |
| `--ksys-monitoring auto\|on\|off` | Huawei Kunpeng ksys system-level telemetry | `auto` |
| `--ksys` | Compatibility alias for `--ksys-monitoring on` | — |

Host memory bandwidth does **not** have its own CLI switch. It is
automatically enabled whenever built-in resource monitoring is active
in a non-concurrent execution mode (see table below). It cannot be
enabled independently.

**`auto` resolution matrix:**

| Scenario | Built-in (CPU/Mem/Disk/Net/CTX) | PMU | Mem BW | ksys |
|---|---:|---:|---:|---:|
| Serial collection (container) | on | on | on | off |
| Serial collection (host) | on | on | on | off |
| Concurrent collection | off | off | off | off |
| Serial container simulation | on | on | on | off |
| Concurrent container simulation | on | off | off | off |
| Host simulation | off | off | off | off |

**Hard constraints (explicit `on` is rejected with a clear error before work starts):**

| Condition | Error |
|-----------|-------|
| `--pmu-monitoring on` with `--concurrency > 1` | PMU uses system-level `perf`; cannot isolate per-attempt measurements |
| `--pmu-monitoring on` with concurrent simulation | Same reason |
| `--resource-monitoring on` with host simulation | Host agent has no isolated process PID |
| `--resource-monitoring on` with concurrent host collection | Attempts cannot be isolated by PID |
| `--pmu-monitoring on` with `--resource-monitoring off` | PMU requires base resource sampling |
| `--ksys` + `--ksys-monitoring off` | Conflicting flags |

**Where monitoring policy is recorded:**

| Artifact | Collection | Simulation |
|----------|:---:|:---:|
| `run_manifest.json` → `runtime.monitoring` | ✅ | ❌ (not written) |
| `resources.json` → `monitoring` (full policy + `status` field) | ✅ | ✅ |
| `trace.jsonl` → trace metadata `monitoring` field | ✅ | ✅ |

The `resources.json` `status` field distinguishes three states:
- `"collected"` — monitoring was enabled and samples were captured
- `"enabled_no_samples"` — monitoring was enabled but the sampler yielded no data (e.g., container not inspectable, unsupported hardware)
- `"disabled"` — monitoring was explicitly turned off by policy

### Practical usage examples

Given a typical serial collection command:

```bash
ARM_IMAGE_MODE=qemu PYTHONPATH=src python -m trace_collect.cli \
    --provider deepseek --model deepseek-v4-flash \
    --benchmark swe-rebench --scaffold openclaw \
    --instance-ids "12rambau__sepal_ui-411" \
    --mcp-config none --verbose --container docker
```

**Default (no flags):** all built-in resources + PMU + MemBW are on;
ksys is off.  This matches the "Serial collection (container)" row in
the `auto` matrix above.

| Goal | Flags to add |
|------|-------------|
| Disable everything | `--resource-monitoring off --pmu-monitoring off` |
| CPU/Mem/Disk/Net/CTX + MemBW only, no PMU | `--resource-monitoring on --pmu-monitoring off` |
| CPU/Mem/Disk/Net/CTX only, no PMU or MemBW | (not possible — MemBW has no independent switch; use `--resource-monitoring off` and accept losing base metrics) |
| All built-in + PMU + ksys (Kunpeng) | `--resource-monitoring on --pmu-monitoring on --ksys-monitoring on` |
| ksys only, no built-in (Kunpeng stress test) | `--resource-monitoring off --pmu-monitoring off --ksys` |
| Concurrent stress test with base metrics only | `--concurrency 3 --resource-monitoring on` |
| Concurrent with everything off | `--concurrency 3` (all `auto` → off) |

**Verifying the resolved policy:** after a run, inspect
`<instance_id>/attempt_N/resources.json` — the `"monitoring"` key
contains the full resolved policy and `"status"` tells you whether
samples were actually collected.

### PMU and memory bandwidth prerequisites

PMU and host memory bandwidth both rely on Linux `perf`.  On most
systems you must lower the kernel's perf-event paranoia level:

```bash
sudo sysctl -w kernel.perf_event_paranoid=-1
```

Without this, `perf` returns zeros silently — the samplers still run,
but `resources.json` will show `"status": "enabled_no_samples"` and
the sample arrays will be empty.

ksys is only available on Huawei Kunpeng hardware.  On other platforms
it degrades gracefully: a warning is logged and collection proceeds
without ksys metrics.

### Behavior With `--concurrency > 1`

- Each task kicks off **N concurrent `run_attempt()` coroutines** via
  `asyncio.gather()`.  Every instance runs in its own container (or host
  process) with an independent `attempt_N/` output directory.
- With `--resource-monitoring auto`, per-attempt resource monitoring is
  disabled. It may be explicitly enabled for isolated container attempts.
- PMU and host memory-bandwidth monitoring are always disabled because their
  system-wide collectors cannot produce isolated concurrent measurements.
- **`--ksys` runs once** for the entire concurrent batch (not per-task),
  writing `ksys_stdout.txt` / `ksys_stderr.txt` to the trace root
  directory (alongside `results.jsonl`).
- **`--record-internals` is blocked** — the HF recording backend does not
  support concurrent attempts (the CLI exits with an error if both are set).
- Results from all N attempts are collected and written to `results.jsonl`;
  per-attempt HTML visualization is generated for each.

```bash
# Stress-test: 3 agents concurrently on a single SWE-rebench case
ARM_IMAGE_MODE=qemu DEEPSEEK_API_KEY=sk-... PYTHONPATH=src python -m trace_collect.cli \
    --provider deepseek \
    --model deepseek-v4-flash \
    --benchmark swe-rebench \
    --scaffold openclaw \
    --instance-ids "12rambau__sepal_ui-411" \
    --mcp-config none \
    --container docker \
    --ksys \
    --concurrency 3
```

### Output Layout (Concurrent Mode)

`--concurrency 3 --instance-ids ...`:

```text
traces/<model>/<ts>/
├── results.jsonl                    # consolidated results from all attempts
├── ksys_stdout.txt                  # ksys output (single copy for the batch)
├── ksys_stderr.txt
└── <instance_id>/
    ├── ksys_stdout.txt              # per-instance ksys logs (serial mode only)
    ├── ksys_stderr.txt
    ├── attempt_1/                   # agent instance 1
    │   ├── trace.jsonl
    │   ├── run_manifest.json
    │   ├── resources.json           # {"monitoring": {"status": "disabled", ...}, ...}
    │   └── trace_viz.html
    ├── attempt_2/                   # agent instance 2
    │   └── ...
    └── attempt_3/                   # agent instance 3
        └── ...
```

**Serial mode (`--concurrency 1`, default):** behavior is unchanged —
one attempt per task, full resource monitoring enabled, `--ksys`
per instance (output to `<instance_id>/`).

---

## Simulate: Trace Replay

The `simulate` subcommand replays pre-collected traces to measure system
behavior under different concurrency and resource constraints.  It does not
call real LLM APIs — instead it replays the timing and tool executions from
source traces.

### Two Simulation Modes

| Mode | LLM calls | Tool execution | Multi-trace | Use case |
|------|-----------|----------------|-------------|----------|
| `cloud_model` | Replayed from source timing (no API calls) | Re-executed in containers, MCP tools replayed | Yes | "What if N agents run concurrently with these traces?" |
| `local_model` | Sent to a real local vLLM endpoint | Re-executed in containers | No (single trace only) | "What is the real TTFT/TPOT on local hardware?" |

### Minimal Examples

```bash
# Cloud model: replay 3 traces concurrently at 50x speed, Poisson arrival
PYTHONPATH=src python -m trace_collect.cli simulate \
    --trace-manifest manifest.json \
    --mode cloud_model \
    --container docker \
    --replay-speed 50 \
    --arrival-mode poisson --arrival-rate-per-s 0.5 --arrival-seed 42

# Local model: measure real TTFT/TPOT against local vLLM
PYTHONPATH=src python -m trace_collect.cli simulate \
    --source-trace traces/.../trace.jsonl \
    --mode local_model \
    --provider openai --api-base http://localhost:8000/v1 \
    --api-key dummy --model Qwen/Qwen3-32B \
    --container docker
```

### Trace Sources

Exactly one of the following must be provided:

| Flag | Description |
|------|-------------|
| `--source-trace <path>` | Replay a single trace JSONL file |
| `--source-dir <path>` | Recursively discover all `trace.jsonl` files under a directory |
| `--trace-manifest <path>` | JSON array with per-trace entries (see below) |

**Trace manifest format:**

```json
[
  {"source_trace": "path/to/trace-a.jsonl"},
  {"source_trace": "path/to/trace-b.jsonl", "docker_image": "custom:latest"},
  {"source_trace": "path/to/trace-c.jsonl", "task_source": "other-tasks.json"}
]
```

Each entry supports:
- `source_trace` (required): path to a `trace.jsonl` file, resolved relative to manifest directory
- `docker_image` (optional): override the Docker image for this trace
- `task_source` (optional): override the tasks JSON file for this trace

### Simulate CLI Flags

| Flag | Required | Default | Notes |
|------|----------|---------|-------|
| `--mode` | no | `local_model` | `local_model` or `cloud_model` |
| `--container` | container traces | — | `docker` or `podman` |
| `--network-mode` | no | `host` | Container network mode |
| `--task-source` | no | `data/swe-rebench/tasks.json` | Path to tasks JSON |
| `--output-dir` | no | `traces/simulate` | Output directory |
| `--command-timeout` | no | `600.0` | Seconds per tool command |
| `--replay-speed` | no | `1.0` | Wall-clock acceleration (cloud_model only) |
| `--warmup-skip-iterations` | no | `0` | Tag first N iterations as warmup |
| `--serial` | no | off | Replay traces one at a time instead of concurrently |
| `--arrival-mode` | no | `closed_loop` | `closed_loop` or `poisson` |
| `--arrival-rate-per-s` | no | — | Required when `--arrival-mode=poisson` |
| `--arrival-seed` | no | — | RNG seed for reproducible Poisson arrival |
| `--num-agents` | no | `0` | Number of agents to spawn (0 = 1:1 with traces) |
| `--trace-assignment` | no | `manifest` | `manifest` or `random` |
| `--trace-assignment-seed` | no | — | RNG seed for `random` trace assignment |
| `--cpu-limit` | no | — | CPU core limit for the entire simulate run |

LLM flags (`--provider`, `--api-base`, `--api-key`, `--model`) are required
for `local_model` mode only.

### Arrival Modes

| Mode | Behavior |
|------|----------|
| `closed_loop` (default) | All agents start simultaneously at t=0, competing for resources |
| `poisson` | Inter-arrival times drawn from Exponential(rate). Seeded RNG for reproducibility |

Arrival offsets are generated by `harness.runner.build_arrival_offsets()`.

---

## N:M Trace-to-Agent Mapping

When `--num-agents N` is set (N > 0), the simulator creates **exactly N agents**
regardless of how many source traces exist.  This enables:

- **Stress testing** with more agents than traces (e.g., 40 traces → 80 agents)
- **Repetition** of a single trace across many agents
- **Randomized** trace-to-agent assignment for Monte Carlo experiments

### `--num-agents N`

| Value | Behavior |
|-------|----------|
| `0` (default) | One agent per input trace — classic 1:1 mapping |
| `> 0` | Create exactly N agents, using `--trace-assignment` to determine which trace each agent replays |

> **Guard:** `--num-agents > 1` is rejected in `local_model` mode (which
> inherently supports only a single trace).  Use `cloud_model` for multi-agent runs.
>
> **Guard:** `--num-agents < 0` is rejected — the count must be non-negative.
>
> **Guard:** `--trace-assignment random` **requires** `--num-agents > 0`.

### `--trace-assignment` Strategies

**`manifest` (default):** Cycle through the input trace list.

```
Input: traces [A, B, C], --num-agents 7
Agent 0 → trace A    Agent 3 → trace A    Agent 6 → trace A
Agent 1 → trace B    Agent 4 → trace B
Agent 2 → trace C    Agent 5 → trace C
```

Each trace is used ⌈N/M⌉ times (ceiling division).  When N < M, only the
first N traces are used.  Order follows the input list exactly.

**`random`:** Each agent independently picks a trace uniformly at random
from the pool (sampling **with replacement**).

```
Input: traces [A, B, C], --num-agents 5, --trace-assignment-seed 42
Agent 0 → trace B
Agent 1 → trace A
Agent 2 → trace C
Agent 3 → trace B    # B picked twice (random, with replacement)
Agent 4 → trace A
```

Use `--trace-assignment-seed` for reproducible random assignment.

### Agent ID Deduplication

When multiple agents replay the same trace, they would otherwise share the
same `agent_id` from the source trace.  The simulator automatically suffixes
duplicate agent_ids with `--a{N}` (where N is the zero-based agent index):

```
Source trace agent_id: "django__django-12345"

Agent 0 → "django__django-12345"        # first occurrence: keeps original
Agent 1 → "django__django-12345--a1"    # duplicate: suffixed
Agent 2 → "django__django-12345--a2"    # duplicate: suffixed
```

This ensures every agent writes to a unique output directory without manual
intervention.  A `WARNING`-level log message is emitted for each rename so
that unexpected duplication is visible in the run output.

### Examples: N:M Mapping

```bash
# 80 agents from 40 traces, cycling through the manifest
PYTHONPATH=src python -m trace_collect.cli simulate \
    --trace-manifest 40_traces.json \
    --mode cloud_model --container docker \
    --num-agents 80 --trace-assignment manifest

# 50 agents all replaying the same single trace
PYTHONPATH=src python -m trace_collect.cli simulate \
    --source-trace traces/.../trace.jsonl \
    --mode cloud_model --container docker \
    --num-agents 50 --trace-assignment manifest

# 100 agents with random trace assignment, deterministic seed
PYTHONPATH=src python -m trace_collect.cli simulate \
    --trace-manifest 40_traces.json \
    --mode cloud_model --container docker \
    --num-agents 100 --trace-assignment random --trace-assignment-seed 42

# Poisson arrival with N:M mapping (agents arrive over time)
PYTHONPATH=src python -m trace_collect.cli simulate \
    --trace-manifest 40_traces.json \
    --mode cloud_model --container docker \
    --num-agents 80 --trace-assignment manifest \
    --arrival-mode poisson --arrival-rate-per-s 0.5 --arrival-seed 42
```

### Combined with `--serial`

`--serial` replays traces sequentially (one at a time).  When combined with
`--num-agents`, each agent still runs one-by-one, but you get N independent
runs instead of M:

```bash
# 80 consecutive replays, cycling through 40 traces
PYTHONPATH=src python -m trace_collect.cli simulate \
    --trace-manifest 40_traces.json \
    --mode cloud_model --container docker \
    --num-agents 80 --trace-assignment manifest --serial
```

---

## CPU Core Limiting

`--cpu-limit N` caps the CPU cores available to the entire simulate run.

### Container-mode traces

Passes `--cpus=N` to `docker run` / `podman run`.  Docker uses the Linux
CFS bandwidth controller (`cpu.cfs_quota_us` / `cpu.cfs_period_us`) to
enforce the limit at the cgroup level.  The value may be fractional
(e.g., `--cpu-limit 1.5` limits to 1.5 cores).

```bash
# Each container gets --cpus=4
PYTHONPATH=src python -m trace_collect.cli simulate \
    --trace-manifest traces.json \
    --mode cloud_model --container docker \
    --cpu-limit 4
```

> **Guard:** `--cpu-limit <= 0` is rejected — the limit must be positive.

### Host-mode traces

Sets CPU affinity on the current process via `psutil.Process().cpu_affinity()`.
Pins the process to cores `[0, 1, ..., ceil(N)-1]`.  Fractional values are
rounded **up** to ensure at least 1 core is pinned (e.g., `--cpu-limit 0.5`
pins to 1 core; `--cpu-limit 3.2` pins to cores 0, 1, 2, 3).

```bash
# Pin simulate process to cores 0-3
PYTHONPATH=src python -m trace_collect.cli simulate \
    --source-trace host_trace.jsonl \
    --mode cloud_model \
    --cpu-limit 4
```

### Mixed host + container sessions

When some sessions are host-mode and some are container-mode, the host
process gets CPU affinity and each container gets `--cpus=N`.

### `cpu_limit` in trace metadata

When `--cpu-limit` is set, the combined `trace.jsonl` metadata records:

```json
{
  "type": "trace_metadata",
  ...
  "cpu_limit": 4.0
}
```

---

## N:M Simulation Sweep

When benchmarking system behavior under scaled multi-agent workloads, a
single `simulate` run answers "what happens with N agents?" — but resource
contention curves require **multiple N values** with consistent methodology.
This section documents the sweep workflow and its supporting scripts.

### Motivation

Running N:M simulation at a single concurrency level shows point-in-time
resource usage.  Sweeping across N ∈ {40, 80, 160, 320} reveals:

- **CPU contention curve** — how aggregate CPU utilization scales with agent count
- **Memory pressure threshold** — at what N does the system begin swapping
- **Tail latency** — p95/p99 agent completion time vs. agent count
- **Docker daemon saturation** — container startup and teardown overhead

Each config runs **independently** with clean output directories, so results
are directly comparable across N values.

### Architecture

```
┌───────────────────────────────────────────────────┐
│  run_simulate_sweep.sh (orchestrator)             │
│                                                   │
│  for N in 40 80 160 320:                          │
│    ┌──────────────────────────────────────┐       │
│    │ 1. system_resource_monitor.py (1 Hz) │       │
│    │    └→ whole-host CPU/mem/disk/net    │       │
│    │                                      │       │
│    │ 2. trace_collect.cli simulate        │       │
│    │    --num-agents $N --cpu-limit 1     │       │
│    │    └→ per-container resources.json   │       │
│    │                                      │       │
│    │ 3. extract_agent_timeline.py         │       │
│    │    └→ per-agent start/end/elapsed    │       │
│    └──────────────────────────────────────┘       │
└───────────────────────────────────────────────────┘
```

### Quick Start

```bash
# 1. Point to your pre-collected traces (40 SWE-rebench traces)
export SOURCE_TRACES_DIR=/path/to/traces/swe-rebench/model/timestamp

# 2. (Optional) override defaults
export SWEEP_VALUES="40 80 160 320"   # default
export CPU_LIMIT=1                    # default: 1 core per agent
export CONTAINER_EXE=docker           # default

# 3. Run the sweep
bash scripts/run_simulate_sweep.sh
```

The script validates prerequisites (trace count, Docker availability, host
cores/memory) before starting, then runs each N sequentially.  On interrupt
(Ctrl+C), it cleanly stops the active system monitor and exits.

### Output Per Experiment

Each N produces a self-contained output directory:

```text
traces/simulate/swe-rebench/sweep_${N}a_1cpu/
├── system_resources.jsonl       # whole-host resource timeline (new)
├── system_viz.html              # interactive system resource charts (new)
├── agent_timeline.jsonl         # per-agent lifecycle log (new)
├── simulate_cloud_model_*.jsonl # combined trace
├── simulate.log                 # full simulate stdout/stderr
├── monitor.log                  # system monitor debug log
└── <agent_id>--a*/attempt_1/
    ├── trace.jsonl              # per-agent canonical trace
    ├── resources.json           # per-container CPU/mem/disk/net
    └── trace_viz.html           # interactive Gantt + charts
```

**`system_resources.jsonl`** — one JSON record per second:

```json
{"ts": 1719000000.123, "cpu_percent": 45.2, "cpu_count": 320,
 "mem_percent": 62.3, "mem_used_gb": 800.4, "mem_total_gb": 1008.0,
 "disk_read_mb": 123456.7, "disk_write_mb": 78901.2,
 "net_rx_mb": 5000.1, "net_tx_mb": 3000.4,
 "container_count": 320,
 "load_1m": 180.5, "load_5m": 150.2, "load_15m": 120.8}
```

**`agent_timeline.jsonl`** — one JSON record per agent, sorted by start time:

```json
{"agent_id": "django__django-12345--a7",
 "start_ts": 1719000000.123, "end_ts": 1719000120.456,
 "elapsed_s": 120.333,
 "n_actions": 42, "n_llm_calls": 21, "n_tool_execs": 19,
 "source_trace": "/path/to/original/trace.jsonl",
 "source_agent_id": "django__django-12345"}
```

A summary is printed to stdout after each N completes:

```
Total agents:          320
Agents with actions:   320
Experiment wall time:  847.2s (14.1 min)
Agent elapsed (mean):  234.5s
Agent elapsed (min):   45.2s
Agent elapsed (max):   812.3s
Agent elapsed (p50):   210.0s
Agent elapsed (p95):   650.1s
Agent elapsed (p99):   780.4s
Total actions:         13440 (llm=6720, tool=6400)
```

A global sweep summary is written to
`traces/simulate/swe-rebench/sweep_summary_<ts>.txt`.

### Script Reference

| Script | Purpose | CLI |
|--------|---------|-----|
| `scripts/system_resource_monitor.py` | Background process that samples system-wide CPU%, memory%, disk I/O, network I/O, Docker container count, and load averages at 1 Hz via `psutil`. Writes JSONL. Stops on SIGTERM/SIGINT or when a stop-file appears. Overhead is negligible (&lt;&lt; 0.1% of one core) — dominated by `sleep(1)`. | `--output <path> [--interval 1.0] [--stop-file <path>] [--verbose]` |
| `scripts/plot_system_resources.py` | Generates a self-contained interactive HTML page from `system_resources.jsonl`. Renders 5 Chart.js time-series panels: CPU + Load, Memory, Container Count, Network I/O rate, Disk I/O rate. No dependencies beyond stdlib (Chart.js loaded from CDN). | `--input <jsonl> --output <html>` |
| `scripts/extract_agent_timeline.py` | Post-processes a simulation output directory. Scans `*/attempt_*/trace.jsonl`, extracts first/last action wall-clock timestamp per agent, and writes an agent lifecycle JSONL with summary statistics. | `--input-dir <dir> --output <path> [--verbose]` |
| `scripts/run_simulate_sweep.sh` | Orchestrator that loops over N values, starts the system monitor, runs `simulate` with `--cpu-limit`, `--resource-monitoring on`, `--pmu-monitoring off`, `--ksys-monitoring off`, stops the monitor, extracts the agent timeline, and auto-generates `system_viz.html`. All parameters are configurable via environment variables. | `SOURCE_TRACES_DIR=<dir> bash scripts/run_simulate_sweep.sh` |

All three scripts are committed to the repository and ready to use on the
target machine.  See `docs/EXPERIMENT_PLAN_arm_sweep.md` for a detailed
design document covering rationale, risk assessment, and implementation
notes.

---

## Recording Internals

In addition to trace-level observability, the harness can capture model
internals — attention patterns and MoE routing decisions — for offline
analysis. This feature uses a host-side HuggingFace backend with custom
forward hooks.

`--record-internals` switches OpenClaw model calls to a host-side HuggingFace
backend and records reduced attention/MoE artifacts beside each attempt.
It currently supports `--scaffold openclaw` only.
For task-container benchmarks, the container talks to a temporary local
OpenAI-compatible proxy backed by that host model, so benchmark tools still run
inside the task container. Docker task containers use `172.17.0.1` as the
default host gateway for that proxy; set `HF_RECORDING_PUBLIC_HOST` if the
server uses a different bridge address.

```bash
PYTHONPATH=src python -m trace_collect.cli \
    --provider openai --api-key hf-recording \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --benchmark swe-rebench \
    --scaffold openclaw \
    --container docker \
    --mcp-config none \
    --sample 1 \
    --record-internals
```

Artifacts are written under
`<attempt_dir>/recordings/iter_0000/{attention.npz,routing.npz,segments.json}`
plus `<attempt_dir>/recordings/meta.json`. `call_idx` is 0-based and aligns to
the nth `action_type="llm_call"` record in `trace.jsonl`.

Sanity-check one call:

```bash
python scripts/load_recording.py --attempt-dir <attempt_dir> --call-idx 0
```

Recording uses `attn_implementation="sdpa"` for the model path and computes
only sampled attention rows inside hooks. It forces
`NANOBOT_MAX_CONCURRENT_REQUESTS=1` and is intended for data collection, not
production throughput.

---

## Ksys System Metrics

For Huawei Kunpeng hardware, `--ksys-monitoring on` enables chip-level telemetry
that complements the standard resource samplers. Ksys captures metrics that
are not available through generic Linux PMU counters.

The legacy `--ksys` alias, or `--ksys-monitoring on`, starts
`ksys collect -o <dir>` as a background process alongside
the agent and stops it (SIGINT) when the agent finishes.

- **Default: off.** Pass `--ksys-monitoring on` or `--ksys` to enable.
- **No-op when `ksys` is not installed** on the host (graceful degradation).
- **Timeline alignment:** ksys starts at the same point as other resource
  samplers, so its data shares the Gantt chart's t0 (time origin).
- **Serial mode** (`--concurrency 1`): ksys runs **per instance** with
  output (`-o`) to `<instance_id>/` (one level above `attempt_N/`).
  The raw stdout/stderr are captured to `ksys_stdout.txt` /
  `ksys_stderr.txt` in the attempt directory.
- **Concurrent mode** (`--concurrency > 1`): ksys runs **once** for the
  entire batch (not per-task, not per-attempt), with output (`-o`) to
  `<run_dir>/` (alongside `results.jsonl`).  All other resource samplers
  are disabled in this mode — ksys is the sole monitoring channel.

```bash
PYTHONPATH=src python -m trace_collect.cli \
    --provider openai \
    --model deepseek-chat \
    --benchmark swe-rebench \
    --scaffold openclaw \
    --container docker \
    --mcp-config none \
    --sample 1 \
    --ksys
```
