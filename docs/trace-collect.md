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
