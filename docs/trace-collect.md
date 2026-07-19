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
- [Resuming Interrupted Runs](#resuming-interrupted-runs)
  - [Resume Judgment Logic](#resume-judgment-logic)
  - [Status Decision Table](#status-decision-table)
  - [Practical Resume Example](#practical-resume-example)
- [Collect Concurrency: `--concurrency`](#collect-concurrency---concurrency)
  - [Resource-monitoring defaults](#resource-monitoring-defaults)
  - [Tool Runtime Artifacts](#tool-runtime-artifacts)
- [Simulate: Trace Replay](#simulate-trace-replay)
  - [Two Simulation Modes](#two-simulation-modes)
  - [Trace Sources](#trace-sources)
  - [Simulate CLI Flags](#simulate-cli-flags)
  - [Arrival Modes](#arrival-modes)
- [Concurrency Models: Simulate vs Collect](#concurrency-models-simulate-vs-collect)
  - [Architecture Comparison](#architecture-comparison)
  - [Execution Flow Diagrams](#execution-flow-diagrams)
  - [Why Collect Doesn't Need `--workers`](#why-collect-doesnt-need---workers)
  - [Choosing the Right Setting](#choosing-the-right-setting)
  - [Worker Architecture & Event-Loop Contention](#worker-architecture--event-loop-contention)
  - [Worker Failure Isolation](#worker-failure-isolation)
- [Timing & Chronometry](#timing--chronometry)
  - [Conceptual Time Windows](#conceptual-time-windows)
  - [Where To Find Timing Data](#where-to-find-timing-data)
  - [Collect vs Simulate: Why `elapsed_s` Differs](#collect-vs-simulate-why-elapsed_s-differs)
- [N:M Trace-to-Agent Mapping](#nm-trace-to-agent-mapping)
- [CPU Core Limiting](#cpu-core-limiting)
  - [Multi-Thread Container Fairness](#multi-thread-container-fairness)
- [N:M Simulation Sweep](#nm-simulation-sweep)
- [Stress Test Analysis](#stress-test-analysis)
- [Recording Internals](#recording-internals)
- [Ksys System Metrics](#ksys-system-metrics)
- [Container Environment Reproducibility](#container-environment-reproducibility)

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
| `--skip N` | Skip the first N tasks before applying `--sample` (e.g. `--skip 10 --sample 5` runs tasks 11–15) |
| `--instance-ids a,b,c` | Run only specified instance(s) |
| `--run-id <path>` | Resume an interrupted run |
| `--prompt-template <name>` | Override the benchmark default prompt |
| `--resource-monitoring auto\|on\|off` | Control CPU, memory, disk, network, context-switch, and host memory bandwidth sampling |
| `--pmu-monitoring auto\|on\|off` | Control PMU sampling; concurrent `on` is forbidden; auto-disabled when `--tool-profiling` is active |
| `--ksys-monitoring auto\|on\|off` | Control Huawei Kunpeng ksys system-level telemetry |
| `--tool-profiling off\|vtune\|ksys` | Wrap each matching tool invocation with a platform profiler (VTune for Intel x86, ksys for Kunpeng) |
| `--tool-profiling-tools <list>` | Comma-separated tool names to profile (default: `exec-pytest`) |
| `--capture-pytest-scripts` / `--no-capture-pytest-scripts` | Save pytest scripts referenced by `exec-pytest` calls (default: on) |
| `--capture-pytest-runtime` / `--no-capture-pytest-runtime` | Enable pytest node timing collection and prediction artifacts (default: on) |
| `--capture-pip-runtime` / `--no-capture-pip-runtime` | Enable compact `pip install` duration prediction artifacts (default: on) |
| `--concurrency N` | Spawn N agent instances per task |
| `--provider` | LLM provider name |
| `--model` | Model identifier |

See `src/trace_collect/CLAUDE.md` for the complete flag reference, provider
registry, checkpointing behavior, and trace schema v5 layout.

### Task Selection Rules

`--instance-ids`, `--skip`, and `--sample` combine in a fixed pipeline order.
Each flag narrows the list left-to-right:

```
Benchmark tasks (all)
    │
    ▼  --instance-ids  (optional)
  Filter to explicit IDs, preserving the order you wrote them.
    │
    ▼  --skip N  (optional)
  Drop the first N tasks.
    │
    ▼  --sample N  (optional)
  Keep only the first N remaining tasks.
    │
    ▼
  Final task list
```

Examples:

```bash
# Run tasks 11-15 of the benchmark (skip first 10, take next 5)
--skip 10 --sample 5

# Run 5 specific instances in a chosen order
--instance-ids django__django-11133,sympy__sympy-12345,scikit-learn__scikit-learn-67890 --sample 5

# Skip the first 3 matching instances, take the next 2
--instance-ids A,B,C,D,E --skip 3 --sample 2
# → runs [D, E]

# --instance-ids with more IDs than --sample: only the first N run
--instance-ids A,B,C,D,E,F,G,H --sample 3
# → runs [A, B, C]
```

Key rules:

- `--instance-ids` order is preserved — the first ID you write gets priority.
- `--skip` runs **after** `--instance-ids` filtering, **before** `--sample`.
- If `--sample` is larger than the remaining task count, all remaining tasks
  run (no error).
- `--concurrency > 1` requires at least one of `--instance-ids`, `--skip`,
  or `--sample` to prevent accidentally running the entire benchmark with
  high parallelism.

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

## Resuming Interrupted Runs

When a long-running collection is interrupted (network failure, OOM, manual
stop, etc.), pass `--run-id <path>` to the **same** `trace_collect.cli`
command to resume from where it left off.  Re-specify all other flags exactly
as in the original invocation (provider, model, benchmark, scaffold,
mcp-config, etc.).

```bash
PYTHONPATH=src python -m trace_collect.cli \
    --provider deepseek \
    --model deepseek-v4-flash \
    --benchmark swe-rebench \
    --scaffold openclaw \
    --mcp-config none \
    --run-id traces/swe-rebench/deepseek-v4-flash/20260624T180504
```

Key behaviors on resume:

- The run directory is **reused** (no new timestamp subdirectory is created).
- Tasks already completed are **skipped** (see judgment logic below).
- Tasks not yet completed are re-run, creating a new `attempt_N/` subdirectory
  (e.g., `attempt_2/`, `attempt_3/`, …).  Previous attempts are preserved.
- Results are **appended** to the existing `results.jsonl`.

### Resume Judgment Logic

At startup, the collector calls `load_completed_ids(run_dir)`, which scans
every `{run_dir}/{instance_id}/attempt_*/run_manifest.json` and reads the
`"status"` field.

**Only two status values mark a task as "done" (will be skipped):**

| `status` | Meaning |
|----------|---------|
| `"completed"` | Agent finished successfully (`success=True`). |
| `"exhausted"` | Agent reached `--max-iterations` without succeeding. Treated as a legitimate terminal state — re-running would waste compute. |

**Everything else will be re-executed:**

- `"error"` — the task failed (see decision table below).
- Missing `run_manifest.json` — the task never ran or was interrupted before
  writing the manifest.
- `run_manifest.json` is corrupted / unreadable JSON.

Once an instance has at least one attempt with status `"completed"` or
`"exhausted"`, that instance is permanently skipped for this run directory.

### Status Decision Table

The manifest `status` is determined in `src/trace_collect/attempt_pipeline.py`.

| Condition | `status` | Resume behavior |
|-----------|----------|-----------------|
| No error, `success=True`, exit_status OK | `"completed"` | **Skipped** |
| `exit_status == "max_iterations"` | `"exhausted"` | **Skipped** (terminal) |
| `inner_error is not None` (unhandled exception crash) | `"error"` | **Re-run** |
| `result.success == False` | `"error"` | **Re-run** |
| `exit_status` is one of `"error"`, `"tool_error"`, `"empty_final_response"`, `"timeout"`, `"failed"` | `"error"` | **Re-run** |
| No `run_manifest.json` at all | N/A | **Re-run** |

> **Note:** The five exit statuses that force `"error"` (`error`, `tool_error`,
> `empty_final_response`, `timeout`, `failed`) are defined in the constant
> `_NONCOMPLETED_EXIT_STATUSES`.  Even if `success=True`, these exit statuses
> override the manifest status to `"error"`, ensuring the task is retried.

### Practical Resume Example

A full production resume command, typically run under `nohup` for long sessions:

```bash
nohup env ARM_IMAGE_MODE=qemu PYTHONPATH=src python -m trace_collect.cli \
    --provider deepseek \
    --model deepseek-v4-flash \
    --benchmark swe-rebench \
    --scaffold openclaw \
    --mcp-config none \
    --container docker \
    --run-id traces/swe-rebench/deepseek-v4-flash/20260624T180504 \
    --resource-monitoring off \
    --pmu-monitoring off \
    --ksys-monitoring off \
    > trace_collect.log 2>&1 &
```

To verify which tasks will be re-run vs skipped before starting, inspect the
run directory:

```bash
# Count completed/error tasks in the run directory
for dir in traces/swe-rebench/deepseek-v4-flash/20260624T180504/*/; do
    instance=$(basename "$dir")
    latest=$(ls -d "$dir"attempt_*/run_manifest.json 2>/dev/null | tail -1)
    if [ -n "$latest" ]; then
        status=$(python3 -c "import json; print(json.load(open('$latest')).get('status','MISSING'))")
        echo "$instance -> $status"
    else
        echo "$instance -> NO_MANIFEST"
    fi
done
```

---

## Collect Concurrency: `--concurrency`

> **This section covers collect mode only.** For simulate mode concurrency,
> see [Worker Architecture & Event-Loop Contention](#worker-architecture--event-loop-contention) and
> [Concurrency Models: Simulate vs Collect](#concurrency-models-simulate-vs-collect).

Beyond single-task runs, the collect CLI supports spawning multiple agent
instances in parallel **for a single benchmark task**.  This is primarily
useful for stress-testing hardware under realistic multi-agent workloads.

`--concurrency N` (default `1`) spawns **N agent instances simultaneously**
for **each** benchmark task.  Tasks themselves are iterated **sequentially** —
different tasks never run concurrently.  Within a single task, all N attempts
run at once, each in its own OS thread via `asyncio.to_thread()`.

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

The CLI exposes independent tri-state switches for resource monitoring,
plus a per-tool profiling selector:

| Flag | Controls | Default |
|------|----------|---------|
| `--resource-monitoring auto\|on\|off` | CPU, memory, disk I/O, network I/O, context switches, and host memory bandwidth | `auto` |
| `--pmu-monitoring auto\|on\|off` | CPU micro-architecture PMU counters (cache, branch, instructions) | `auto` |
| `--ksys-monitoring auto\|on\|off` | Huawei Kunpeng ksys system-level telemetry | `auto` |
| `--tool-profiling off\|vtune\|ksys` | Per-tool platform profiler (VTune x86 / ksys Kunpeng) | `off` |
| `--tool-profiling-tools <list>` | Tool names to profile (comma-separated) | `exec-pytest` |

Host memory bandwidth does **not** have its own CLI switch. It is
automatically enabled whenever built-in resource monitoring is active
in a non-concurrent execution mode (see table below). It cannot be
enabled independently.

Per-tool profiling (`--tool-profiling`) wraps individual tool invocations
(e.g. `pytest`) with a hardware profiler.  When `vtune` is selected, the
host-side PMU monitoring is automatically disabled to avoid PMU counter
contention.  Coarse system metrics (CPU/mem/disk/net/ctx) for each profiled
tool invocation come from an in-container `/proc` proc-tree sampler, giving
accurate per-invocation data even with concurrent tool calls.  See
[VTune Profiling](vtune-profiling.md) for setup and architecture details.

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
| `--tool-profiling vtune` with `runtime_mode = host_controller` | VTune wraps in-container commands only |
| `--tool-profiling vtune` + `--pmu-monitoring on` | PMU auto-disabled (no error — VTune gets exclusive PMU access) |
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

### Tool Runtime Artifacts

OpenClaw collect mode writes additional tool-specific artifacts for selected
commands. These artifacts are passive: unless a subsection says otherwise, they
do not change the executed command or the tool environment.

#### `pytest` Tool Invocation

For SWE-style OpenClaw collection, each pytest tool invocation also writes a
`pytest_scripts/` artifact directory under the attempt.  Each
`iter_<N>_exec-pytest_<tool_call_id>/` subdirectory contains the original
`command.sh`, copied pytest `.py` files under `files/`, and `manifest.json`
with the trace timing fields (`duration_ms`, `ts_start`, `ts_end`, `success`,
`action_id`).  The attempt-level `pytest_scripts/index.jsonl` provides one
row per captured pytest invocation for downstream timing analysis.  This is
enabled by default in collect mode; pass `--no-capture-pytest-scripts` to
disable it.

The same SWE-style OpenClaw collect path also records a minimal pytest runtime
prediction prototype under `pytest_runtime/`.  Matching `exec-pytest` commands
first run an explicit pre-execution `pytest --collect-only` command with a
temporary pytest plugin to identify the nodeids available at prediction time.
The collect-only invocation is run as an argument vector in the tool working
directory using the same runtime environment preparation as the OpenClaw shell
tool, including task-container `PYTHONUSERBASE` and PATH handling. It redirects
pytest cache into the prediction artifact directory and removes known
output-producing/reporting flags such as `--junitxml`, `--html`, `--cov`, and
`--cov-report` to avoid perturbing the target workspace. The
artifact cache override is placed after retained pytest args so user
`cache_dir` overrides cannot redirect collect-only cache writes back into the
target workspace. Cache-dependent selectors such as `--lf` / `--last-failed`
are treated as unsupported for pre-execution nodeid prediction because dropping
or redirecting their cache would change the selected test set. Shell `timeout`
prefixes are enforced by the collect-only subprocess timeout. The
original pytest tool command still runs afterward as requested by the agent and
loads the same kind of temporary plugin to write per-node `nodeid`,
`duration_s`, and `outcome` data. After each pytest finishes, collect mode
immediately prints a concise line like:

```text
[pytest-predict] iter=17 tests=32 actual=220.40s collect_overhead=4.20s last=205.10s last_err=6.9% count=143.80s count_err=34.8% per_test=215.70s per_test_err=2.1% unknown=218.40s unknown_err=0.9% recommended=per_test:215.70s rec_err=2.1% reliability=high
```

Artifacts are written as:

```text
attempt_N/pytest_runtime/
  history.json
  predictions.jsonl
  iter_0017_exec-pytest_<tool_call_id>/
    pending.json
    collect_only/
      pytest_collect_only.json
    pytest_runtime.json
    prediction.json
    instrumentation.json
```

`pending.json` snapshots the history-derived prediction before the pytest tool
command executes, including the collect-only command metadata, pre-execution
nodeids, and `prediction_overhead_s`. `prediction.json` then preserves the
actual wall duration, test collection, per-test durations/outcomes, the
pre-execution prediction baselines, absolute / relative errors, and
`pytest_output.text`, which is the captured shell-tool output for that pytest
invocation. `predictions.jsonl` appends compact per-run rows and omits the full
pytest output to keep aggregate scans cheap.

The extra prediction cost is reported explicitly:

```text
collect_only_duration_s                 wall time spent by pytest --collect-only
prediction_overhead_s                   same value, carried as prediction overhead
total_duration_with_prediction_overhead_s actual pytest runtime plus collect-only overhead
```

The analyzer prints average collect-only overhead across all finalized pytest
rows with overhead metadata, including cold-start rows where no prediction was
available. CSV export also includes those rows so prediction accuracy and
prediction cost can be audited separately.

Future unknown-test calibration is based on the same pre-execution history
snapshot saved in `pending.json`, not on history changes observed after the
tool starts.

`prediction_last_run_s` is keyed by `normalized_command`.  Starting with
`schema_version=4`, this key strips interpreter-path differences and stdout
post-processing such as `| head`, `| tail`, and `| grep`, while preserving
timeout values, test targets, `-k` / `-m` selectors, plugin/config flags, and
selection-changing flags such as `--ignore`.  Order-independent targets and
ignore/deselect flags are sorted before the key is written; this keys by the
selected test set, not by pytest execution order.  Command-level Last Run
history is updated only when the runtime plugin observes at least one test
node, so early failures that produce no pytest runtime JSON do not pollute
later command-duration predictions.  Starting with
`schema_version=2`, `prediction_per_test_s` is an
overhead-adjusted per-test estimate: the sum of historical node/file/project
test durations plus the recent median pytest overhead, where overhead is
`actual_duration_s - sum(observed test duration_s)`.  The unadjusted sum is
kept as `prediction_per_test_without_overhead_s`, and the added overhead is
kept as `prediction_per_test_overhead_s`.  Starting with `schema_version=3`,
`prediction_unknown_test_fallback_s` adds a fourth baseline for cold-start
tests: node/file predictions are reused when available, but completely unseen
tests use the median duration of earlier tests that were also unknown at
prediction time, then add the same overhead term.

Starting with `schema_version=5`, each row also includes an agent-aware
recommended prediction and reliability explanation:

```json
{
  "prediction_recommended_s": 215.7,
  "prediction_recommended_method": "last_run",
  "prediction_reliability": {
    "level": "high",
    "known_node_ratio": 0.92,
    "file_fallback_ratio": 0.08,
    "project_fallback_ratio": 0.0,
    "unknown_fallback_ratio": 0.0,
    "repeated_command": true,
    "previous_collected_count": 32,
    "collected_count_delta_ratio": 0.0,
    "reasons": [
      "same_normalized_command_seen_before",
      "collected_count_stable"
    ]
  }
}
```

The recommended prediction is not a weighted ensemble.  It is a small
rule-based selector designed to expose when history is trustworthy:

```text
if Last Run is available and the previous same-command collected_count is known and changed by <=10%:
    use Last Run, reliability=high
elif Last Run is available but the current test set is not known before execution:
    use Last Run, reliability=medium
elif Last Run is available but previous same-command collected_count is unavailable:
    use Last Run, reliability=medium
elif Per-Test is available and at least 80% of tests have exact node history:
    use Per-Test, reliability=high
elif Per-Test is available and at least 80% of tests have node-or-file history:
    use Per-Test, reliability=medium
elif Unknown-Test Fallback is available:
    use Unknown-Test Fallback, reliability=low
elif pytest collected one or more test nodes but no pre-execution history can
predict them yet:
    mark as coldstart
else:
    mark as error
```

This layer is meant to reflect Code Agent behavior: repeated local pytest
commands are usually best predicted by the previous same command, while changed
test sets rely more on per-node history.  Cold-start tests are still predicted
when possible, but are explicitly marked low reliability instead of being
treated as equally trustworthy.  The command history stores both `durations`
and `collected_counts`, and both are updated only for runs where at least one
real test node was observed.

Commands that explicitly assign or export `PYTHONPATH` are skipped by this
prototype's runtime instrumentation, because that shell assignment can hide the
temporary pytest plugin from pytest while still inheriting plugin-related
environment variables.  Runtime prediction is enabled by default in collect
mode; pass `--no-capture-pytest-runtime` to disable plugin injection,
prediction artifacts, and realtime `[pytest-predict]` stdout lines.

Summarize a run directory with:

```bash
PYTHONPATH=src python scripts/analyze_pytest_prediction.py traces/swe-rebench/<model>/<run>
```

The analyzer reports all baseline methods plus `Recommended`, then prints
`high` / `medium` / `low` / `coldstart` / `error` reliability buckets.  Legacy
rows without reliability metadata may still appear as `unavailable`.  With
`--csv`, the exported rows include the recommended method, reliability level,
fallback ratios, collected-count delta, and all absolute/relative errors.

#### `pip install` Tool Invocation

The OpenClaw collect path also records compact runtime prediction artifacts for
supported `pip install` commands under `pip_runtime/`. This is a passive
trace-collection feature: it does not wrap the command, inject environment
variables, modify tool arguments, or change package installation semantics.
It recognizes common forms such as `pip install ...`, `pip3 install ...`, and
`python -m pip install ...`.

Pip prediction learns from an instance-scoped history database under the run
directory, so later attempts for the same task instance can use successful
installs observed by earlier attempts without mixing unrelated images or repos.
Per-attempt directories still hold the invocation artifacts needed for audit.
In task-container mode, the host seeds each attempt's local history from the
shared database before the container starts, then merges successful prediction
rows back after the attempt finishes; this keeps the shared database
host-visible without adding broader container mounts.

After each supported invocation finishes, collect mode prints one summary line
with all simple baselines and their relative errors:

```text
[pip-predict] iter=4 packages=2 actual=10.00s last=9.00s last_err=10.0% package_count=11.00s package_count_err=10.0% global=8.00s global_err=20.0% recommended=package_count:11.00s rec_err=10.0% reliability=medium
```

Artifacts are written as:

```text
pip_runtime_db/
  <instance_scope>/
    history.json
attempt_N/pip_runtime/
  predictions.jsonl
  iter_0004_exec-pip_<tool_call_id>/
    pending.json
    prediction.json
```

`<instance_scope>` is a filesystem-safe instance label with a short hash suffix
to avoid collisions between different raw `instance_id` values.

`pending.json` snapshots the normalized command and predictions before the pip
command executes, including the history path used for that prediction.
`prediction.json` then adds the observed wall duration, exit code, success
metadata, absolute / relative errors, and whether the run was admitted into
history. `predictions.jsonl` appends the same compact per-run rows for
aggregate scans.

The predictor intentionally keeps only three simple baselines:

```text
if the same normalized pip command has prior successful history:
    use Last Run, reliability=high
elif package_count is available and prior per-package history exists:
    use Package Count, reliability=medium
elif any prior successful pip install duration exists:
    use Global Median, reliability=low
else:
    prediction is unavailable
```

`normalized_command` sorts explicit package arguments and hashes readable
requirement files, so `python -m pip install -r requirements.txt` is keyed by
the dependency file content rather than by the raw shell spelling. The shared
history is updated only for successful installs. Commands with nonzero
`Exit code` do not enter history, and commands containing `||` fallback chains
are recorded but excluded from history updates because the final shell status
can mask a failed pip invocation.

Pip runtime prediction is enabled by default in collect mode; pass
`--no-capture-pip-runtime` to disable `pip_runtime/` artifacts and realtime
`[pip-predict]` stdout lines.

#### Python Script Tool Invocation

Collect mode also records compact runtime prediction artifacts for supported
`python *.py` script commands under `python_script_runtime/`. This feature is
passive: it does not wrap commands, inject environment variables, edit tool
arguments, or change Python execution semantics. It intentionally excludes
`python -c`, heredoc scripts, and `python -m ...`; only file-backed script
invocations such as `python eval.py`, `python3 -u /app/explorer.py`, and
`timeout 20 python3 script.py` are admitted.

Python script prediction uses same-instance history shared across attempts.
In task-container mode, the host seeds each attempt's local history mirror
before the container starts and merges successful prediction rows back after
the attempt finishes, matching the pip runtime history pattern.

Artifacts are written as:

```text
python_script_runtime_db/
  <instance_scope>/
    history.json
attempt_N/python_script_runtime/
  predictions.jsonl
  iter_0004_python-script_<tool_call_id>/
    pending.json
    prediction.json
```

The predictor uses only pre-execution history:

```text
if the same normalized script invocation has prior successful history:
    use Last Run, reliability=high
elif the same script path has prior successful history:
    use Script Path Median, reliability=medium
elif the same script basename has prior successful history:
    use Basename Median, reliability=low
elif any prior successful python-script duration exists for this instance:
    use Global Median, reliability=low
else:
    prediction is unavailable
```

History is updated only for successful commands. Commands with nonzero
`Exit code` and commands containing `||` fallback chains are recorded but do
not enter history, because the final shell status may mask a failed script run.

Python script runtime prediction is enabled by default in collect mode; pass
`--no-capture-python-script-runtime` to disable `python_script_runtime/`
artifacts and realtime `[python-script-predict]` stdout lines.

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

When `--task-source` is omitted, simulate reads the trace metadata
`benchmark` field and selects the canonical local task cache, for example
`data/swe-rebench/tasks.json` or `data/swebench_verified/tasks.json`. Pass
`--task-source` only when replaying against a custom or non-canonical tasks
file.

### Simulate CLI Flags

| Flag | Required | Default | Notes |
|------|----------|---------|-------|
| `--mode` | no | `local_model` | `local_model` or `cloud_model` |
| `--container` | container traces | — | `docker` or `podman` |
| `--network-mode` | no | `host` | Container network mode |
| `--task-source` | no | auto from trace metadata | Optional tasks JSON override |
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
| `--workers` | no | `1` | Number of worker processes for concurrent `cloud_model` replay. Each worker runs its own asyncio event loop, splitting N agents across W processes (N/W per loop). Use `min(num_agents, os.cpu_count())` for large replay runs. `cloud_model` only. |
| `--prep-concurrency` | no | `0` (auto) | System-wide maximum concurrent container preparations (Docker run + Python bootstrap), shared across the main process and all worker processes. `0` uses the default limit of 20. This controls warm-up load; replay is released separately by a global all-ready barrier. |

LLM flags (`--provider`, `--api-base`, `--api-key`, `--model`) are required
for `local_model` mode only.

### Arrival Modes

| Mode | Behavior |
|------|----------|
| `closed_loop` (default) | All agents start simultaneously at t=0, competing for resources |
| `poisson` | Inter-arrival times drawn from Exponential(rate). Seeded RNG for reproducibility |

Arrival offsets are generated by `harness.runner.build_arrival_offsets()`.

## Concurrency Models: Simulate vs Collect

Simulate and collect support different concurrency models with different
isolation guarantees.  Understanding the distinction is critical for
interpreting experimental results.

### Architecture Comparison

| Property | Simulate `--workers` | Collect `--concurrency` |
|----------|---------------------|------------------------|
| **What runs concurrently** | N **different** agents (different traces/tasks) | N **identical** attempts of the **same** task |
| **Cross-task concurrency** | ✅ All agents start simultaneously | ❌ Tasks iterate sequentially (for loop) |
| **Isolation unit** | OS **process** (`ProcessPoolExecutor`) | OS **thread** (`asyncio.to_thread()`) |
| **Event loop per unit** | Each process has its own asyncio event loop | Each thread runs `asyncio.run()` (independent loop) |
| **CPU core distribution** | ✅ Processes pinned to different cores by OS | ✅ Threads can span cores (GIL released during `subprocess.run()`) |
| **Shared state** | Nothing shared (separate address spaces) | Same Python process memory, same Docker daemon socket |
| **Trace output** | Per-agent `trace.jsonl` files in shared directory | Per-attempt `attempt_N/` subdirectories under `instance_id/` |
| **Truly independent** | ✅ Full process isolation | ⚠️ Thread-level isolation |

### Execution Flow Diagrams

**Simulate `--workers 3`:**

```
Main Process                    Worker-1 Process             Worker-2 Process
  │                                │                             │
  │─ chunk[0]: agent 1..K          │─ chunk[1]: agent K+1..2K    │─ chunk[2]: agent 2K+1..N
  │  own event loop                │  own event loop             │  own event loop
  │  own containers                │  own containers             │  own containers
  │  own trace files               │  own trace files            │  own trace files
  │                                │                             │
  └─ (runs IN PARALLEL) ───────────┴─ (runs IN PARALLEL) ────────┴─ ...
```

**Collect `--concurrency 3`:**

```
Single Process
  │
  ├─ Task 1 (instance_id="django__django-12345")
  │    ├─ OS Thread 1: attempt_1  → asyncio.run() → subprocess (container)
  │    ├─ OS Thread 2: attempt_2  → asyncio.run() → subprocess (container)
  │    └─ OS Thread 3: attempt_3  → asyncio.run() → subprocess (container)
  │    ↓ all 3 attempts complete
  │
  ├─ Task 2 (instance_id="sympy__sympy-67890")   ← starts AFTER Task 1
  │    ├─ OS Thread 1: attempt_1  → ...
  │    ├─ OS Thread 2: attempt_2  → ...
  │    └─ OS Thread 3: attempt_3  → ...
  │    ↓ all 3 attempts complete
  │
  └─ ... (sequential task iteration)
```

### Why Collect Doesn't Need `--workers`

Collect mode's `--concurrency` uses `asyncio.to_thread()` for each attempt
because scaffold code calls blocking `subprocess.run()`.  Each thread has
its own `asyncio.run()` event loop, so there is no coroutine contention.

The bottleneck in collect mode is not the Python event loop — it's the
Docker daemon and system resources (CPU, memory, I/O).

### Choosing the Right Setting

**For simulate (`--workers`):**

| Host | N agents | Recommended `--workers` | Agents per loop |
|------|---------|------------------------|----------------|
| ARM 320 vCPU | 40 | 40 | 1 |
| ARM 320 vCPU | 80 | 80 | 1 |
| ARM 320 vCPU | 160 | 160 | 1 |
| ARM 320 vCPU | 320 | 320 | 1 |
| ARM 320 vCPU | 640 | 320 | 2 |
| 16-core desktop | 40 | 16 | ~3 |
| 16-core desktop | 320 | 16 | 20 (expect some congestion) |

Rule of thumb: `--workers = min(num_agents, os.cpu_count())`.  This is the
default in `run_simulate_sweep.sh`.

**For simulate warm-up load (`--prep-concurrency`):**

| Scenario | Recommended `--prep-concurrency` | Effect |
|----------|-------------------------------|--------|
| Default | 0 (auto) | At most 20 preparations system-wide |
| Faster warm-up on a validated host | 64 | At most 64 preparations system-wide |
| Conservative Docker daemon load | 10 | At most 10 preparations system-wide |

The limit does not multiply by `--workers`. For example,
`--workers 320 --prep-concurrency 20` still permits only 20 simultaneous
container preparations across the whole run.

**For collect (`--concurrency`):**

| Scenario | Recommended `--concurrency` | Reason |
|----------|---------------------------|--------|
| Normal trace collection | 1 (default) | One clean attempt per task |
| Multi-sample per task | 3-5 | Statistical significance |
| Stress testing | 10-20 (with `--sample 1`) | Push Docker daemon limits |

> Collect `--concurrency > 1` requires `--instance-ids` or `--sample`.
> Running all benchmark instances with high concurrency is blocked.

---

### Worker Architecture & Event-Loop Contention

> This section covers the **simulate** `--workers` mechanism in detail.
> For the architectural comparison with collect mode, see
> [Concurrency Models: Simulate vs Collect](#concurrency-models-simulate-vs-collect) above.

`--workers` is primarily a measurement-isolation control. It splits replay
sessions across processes so each worker has its own asyncio loop, reducing
Python event-loop contention in timing-sensitive runs. It does not change the
trace workload, command semantics, or shared machine resources.

When many agents share a single asyncio event loop (`--workers 1`), two
measurement distortions arise that affect timing accuracy:

| Distortion | Mechanism | Affected measurements |
|------------|-----------|----------------------|
| **Sleep drift** | `asyncio.sleep()` callbacks queue behind other coroutines; wake-up is delayed until the loop reaches the scheduled callback | LLM replay duration (`source_duration_s / replay_speed`), MCP tool replay duration |
| **Await-callback delay** | Docker SDK `_exec_tool()` completions land as callbacks that must wait their turn before `record_ts_end = time.time()` executes | Non-MCP tool execution `duration_ms` |

Both distortions grow with agent count. At 640 agents on a single event loop,
measured timing can diverge dramatically from ground truth.  The `--workers`
flag eliminates this measurement noise by splitting agents across independent
processes, each with its own event loop:

```
Real latency = container resource contention (stress-test target) + event-loop scheduling latency (measurement noise)

More workers → scheduling noise → 0

More workers → container resource contention unchanged (Docker daemon remains the bottleneck)
```

**Recommended worker counts for a 320-core host (640 agents):**

| Workers | Agents/Worker | Event-loop congestion | Timing accuracy | Memory cost |
|---------|--------------|----------------------|-----------------|-------------|
| 1 | 640 | Severe | ❌ Unusable | Minimal |
| 64 | 10 | Mild | ✅ Acceptable | Low |
| 160 | 4 | Very low | ✅ Good | Moderate |
| **320** | **2** | Near-zero | **✅ Near-perfect** | Moderate (320 Python processes) |
| 640 | 1 | Zero | ✅ Perfect | High (640 Python processes) |

`--workers 320` (2 agents per event loop) is the sweet spot: scheduling delay
is negligible while memory overhead stays manageable.  `--workers 160`
(4 agents per loop) is a conservative alternative that still produces
accurate results.

**Relationship to pressure testing:** workers eliminate *measurement noise*,
not *resource competition*.  Containers still compete for CPU, memory, and
I/O through the Docker daemon regardless of worker count.  The pressure test
remains valid — workers just ensure the timing numbers you record reflect
that competition rather than Python bookkeeping overhead.

**Implementation:** in multi-worker mode the main process handles one chunk
of sessions directly; remaining chunks are dispatched to subprocess workers
via `concurrent.futures.ProcessPoolExecutor`.  Each worker independently
loads, prepares, replays, and tears down its sessions, writing per-agent
`trace.jsonl` files into the shared output directory.  HTML visualization
auto-discovers worker-written directories.

### Worker Failure Isolation

Workers process independent chunks and write per-agent artifacts separately.
Session failures are collected after each worker batch settles, so one failed
session does not close shared outputs while other sessions are still writing.
If an unrecovered worker exception occurs, the run still reports the failure,
and artifacts from completed sessions remain available for inspection.

### Warm-Up Phase: Container Preparation

Before agents can replay traces, each container must be prepared — this
involves `docker run`, Python interpreter probing, dependency bootstrapping,
and starting a persistent replay agent process inside the container.

In early versions, container preparation was rate-limited to **20 concurrent
operations** (hardcoded `asyncio.Semaphore(20)`) and worker processes
prepared containers **sequentially** (one at a time).  With 640 agents this
created visible "waves": the first 20 containers would start, then the next
20, and so on — with agents waiting idle until every container was ready.

`--prep-concurrency` (added 2026-06-28) controls preparation load:

- **System-wide limit:** one shared semaphore covers the main process and all
  worker processes; `0` uses the default limit of 20.
- **Concurrent worker preparation:** sessions inside each worker prepare
  concurrently while respecting that shared limit.
- **Global replay barrier:** every process waits until every session is
  prepared before replay is released.

For `--workers 320 --prep-concurrency 64`, no more than 64 of the 640
containers prepare at once. After all 640 are ready, the global barrier
releases every worker into the replay phase.

**Flow with `--prep-concurrency` enabled:**

```text
                    Preparation Phase                     Replay Phase
                    (rate-limited by                      (fully concurrent)
                    --prep-concurrency)
                    ┌────────────────────┐               ┌──────────────────┐
  t=0               │ Worker 1: agent A  │               │                  │
                    │ Worker 1: agent B  │               │ All N agents     │
                    │ Worker 2: agent C  │  ── all ──→   │ replay traces    │
                    │ Worker 2: agent D  │   prepared    │ simultaneously   │
                    │ ...                │               │                  │
                    │ Worker 320: agent  │               │                  │
                    └────────────────────┘               └──────────────────┘
```

> **Note:** Preparation and replay are separate phases. In `closed_loop` mode,
> all agents are released from one global barrier after the last preparation
> finishes; normal OS scheduling still introduces small start-time skew. In
> `poisson` mode, the same barrier establishes one global time zero and agents
> then follow offsets generated once for the complete N-agent population.

---

## Timing & Chronometry

This section documents every timing field produced by `collect` and `simulate`,
what interval each covers, and where to find the numbers.

### Conceptual Time Windows

For SWE-bench-style benchmarks that use Docker containers, a single task
run is split into three phases:

```
  start_time
  │
  ├─ [SETUP] ───────────────────────────────────────────┤ image_ready_time
  │   disk preflight, ensure_fixed_image(),             │
  │   start_task_container(), container bootstrap       │
  │
  ├─ [AGENT EXECUTION] ─────────────────────────────────┤ agent_end_time
  │   inner(ctx): agent loop (LLM calls + tool execs)   │
  │   │ first action ts_start                           │
  │   │ ...                                             │
  │   │ last action ts_end                              │
  │
  ├─ [TEARDOWN] ────────────────────────────────────────┤ end_time
  │   stop samplers, stop recording, stop container,    │
  │   write manifest, write resources.json              │
```

The **agent execution** window is what the HTML visualization and Gantt chart
display.  The setup and teardown windows are overhead — they are recorded
separately so they can be analysed or excluded as needed.

### Where To Find Timing Data

#### Collect Mode

| Artifact | Field(s) | What It Covers |
|----------|----------|----------------|
| `run_manifest.json` → `timing.wall_total_s` | `start_time` → `end_time` | Full wall clock: setup + agent + teardown |
| `run_manifest.json` → `timing.setup_s` | `start_time` → agent start | Disk preflight, image fix, container bootstrap |
| `run_manifest.json` → `timing.agent_exec_s` | `agent_start_time` → `agent_end_time` | Pure agent execution loop |
| `run_manifest.json` → `timing.teardown_s` | `agent_end_time` → `end_time` | Sampler stop, container teardown, manifest write |
| `run_manifest.json` → `timing.permission_fix_s` | subset of `setup_s` | Image permission fix (`ensure_fixed_image`) |
| `run_manifest.json` → `result_summary.active_time` | LLM time only | Sum of all LLM call latencies (seconds) |
| `run_manifest.json` → `result_summary.tool_time` | Tool time only | Sum of all tool execution latencies (seconds) |
| `results.jsonl` → `elapsed_s` | `t0` (before `run_attempt`) → after `run_attempt` | Slightly wider than `wall_total_s` — includes image prefetch wait and context creation (~ms) |
| `results.jsonl` → `permission_fix_time` | same as `permission_fix_s` | Image permission fix only |
| `trace.jsonl` → action `ts_start` / `ts_end` | per-action epoch seconds | Individual LLM call or tool execution span |
| `resources.json` → sample `epoch` | per-sample epoch seconds | Resource sampling timestamps |

#### Simulate Mode

| Artifact | Field(s) | What It Covers |
|----------|----------|----------------|
| `trace.jsonl` → summary `elapsed_s` | `wall_start` → `wall_end` | Pure replay loop (LLM calls + tool executions) |
| `trace.jsonl` → summary `timing.agent_exec_s` | same as `elapsed_s` | Same — pure replay loop |
| `trace.jsonl` → summary `timing.container_setup_s` | container prep start → agent.start() | Image fix + container start + agent bootstrap |
| `trace.jsonl` → summary `total_llm_ms` | — | Sum of all simulated LLM call latencies (ms) |
| `trace.jsonl` → summary `total_tool_ms` | — | Sum of all tool execution latencies (ms) |
| `resources.json` → sample `epoch` | per-sample epoch seconds | Resource sampling timestamps |

> **Note:** Simulate does **not** write a `run_manifest.json`.  All timing
> fields live in the combined trace JSONL's `summary` record for each agent.

#### HTML Visualization (`trace_viz.html`)

| Display | Source | Meaning |
|---------|--------|---------|
| **Agent Time** stat box | `max(ts_end) - min(ts_start)` across all actions | Pure agent execution span — from first action start to last action end |
| **Wall Clock** stat box (when present) | `run_manifest.json` → `timing.wall_total_s` | Full wall clock including setup + teardown. Only shown when `run_manifest.json` has a `timing` section (collect mode). |
| Gantt chart time axis | same as Agent Time | Only actions are plotted; no blank setup/teardown space |
| Resource chart time axis | same as Agent Time | Resource samples outside the agent execution window are filtered out |
| Header `ts_start → ts_end` | first action start → last action end | Same window as Agent Time |

### Collect vs Simulate: Why `elapsed_s` Differs

| Aspect | Collect | Simulate |
|--------|---------|----------|
| What `elapsed_s` includes | Setup + agent + teardown | Agent replay loop only |
| Container startup | Included (in `setup_s`) | Excluded (container is pre-started; time in `container_setup_s`) |
| Image prep | Included (in `setup_s` / `permission_fix_s`) | Included in `container_setup_s` |
| Teardown | Included (in `teardown_s`) | Excluded (teardown runs after `wall_end`) |

If you replay a collect trace with simulate and compare `elapsed_s`, the
simulate value will be **smaller** because it excludes setup and teardown.
To compare the actual agent work, use `agent_exec_s` from both sides.

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
**This flag is optional** — when omitted, no CPU limit is applied and
containers use native Linux CFS scheduling without quota enforcement.

### When to use --cpu-limit

| Scenario | Recommendation |
|----------|---------------|
| cloud_model single-trace replay (I/O-bound) | **Omit** — CFS throttle adds overhead with no benefit |
| cloud_model sweep / over-subscription stress test | **Set to 1** — fair per-agent allocation, reproducible contention curves |
| local_model replay (real LLM calls, CPU-bound) | Set to avoid noisy-neighbor contention |
| Over-subscription experiment (> host cores) | Set to model resource-constrained deployments |
| Reproducible benchmarking | Set for consistent per-container allocation |

### CFS semantics (container mode)

When `--cpu-limit N` is set, Docker passes `--cpus=N` which writes to the
cgroup CFS bandwidth controller:

```
cpu.cfs_period_us = 100000   (100 ms period)
cpu.cfs_quota_us  = N × 100000
```

Each container may consume at most N cores per 100 ms period.  Unused quota
is **not** transferred to other containers.  If all containers are CPU-bound
simultaneously, this produces a sawtooth utilization pattern (all containers
throttled after exhausting quota, then idle until the next period).

**In practice**, cloud_model replay is I/O-bound (containers spend most time
in `sleep` waiting for the next source-trace action or in `docker exec`
waiting for shell commands).  Under these conditions CFS throttle has
negligible effect — but it also provides no benefit over native scheduling.

### Multi-thread container fairness

The `--cpus` flag matters most when a container runs **multi-threaded tools**.
CFS counts CPU time across **all threads** in the cgroup:

```
Without --cpus (no upper limit):
  Container A: 1 thread  → 1 scheduling entity → ~0.5 core (640 containers / 320 cores)
  Container B: 4 threads → 4 scheduling entities → ~2.0 cores  ← Steals CPU from other containers!

With --cpus=1 (hard cap):
  Container A: 1 thread  → cap 1 core → actual ~0.5 core (unchanged)
  Container B: 4 threads → 4 threads share 1 core → ~0.25 core per thread (fair)
```

For SWE-rebench scenarios, most tools are single-threaded:

| Tool type | Threads | Affected by `--cpus=1`? |
|-----------|---------|------------------------|
| `git diff`, `git checkout` | Single | No |
| `bash`, `sed`, `grep` | Single | No |
| `python script.py` | Single (typically) | No |
| `pip install` (compiling extensions) | Multi (`-j`)  | **Yes** |
| `pytest` | Single (default) | Usually no |
| `make -j` | Multi-process | **Yes** |

With 640 containers where 10 run `pip install` with parallel compilation:

| | No `--cpus` | `--cpus=1` |
|---|---|---|
| 10 multi-thread containers | Each may grab 2-4 cores | Each strictly ≤1 core |
| 630 single-thread containers | Shares squeezed by multi-thread | Unaffected |
| Reproducibility | Low (depends on timing of parallel builds) | High |

**Conclusion:** `--cpus=1` costs nothing for single-thread tools but prevents
a small number of multi-thread tools from skewing the pressure test.  It is a
defensive default with zero downside for I/O-bound replay workloads.

### Container-mode traces

Passes `--cpus=N` to `docker run` / `podman run`.  The value may be
fractional (e.g., `--cpu-limit 0.5` limits to 0.5 cores).

```bash
# Each container gets --cpus=4
PYTHONPATH=src python -m trace_collect.cli simulate \
    --trace-manifest traces.json \
    --mode cloud_model --container docker \
    --cpu-limit 4

# No CPU limit — native Linux scheduling, no CFS throttle
PYTHONPATH=src python -m trace_collect.cli simulate \
    --trace-manifest traces.json \
    --mode cloud_model --container docker
    # (no --cpu-limit flag)
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

When omitted, the `cpu_limit` key is absent from metadata, indicating
native Linux scheduling was used.

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
│  for N in ${SWEEP_VALUES}:                        │
│    ┌──────────────────────────────────────┐       │
│    │ 1. system_resource_monitor.py (1 Hz) │       │
│    │    └→ whole-host CPU/mem/disk/net    │       │
│    │                                      │       │
│    │ 2. trace_collect.cli simulate        │       │
│    │    --num-agents $N                   │       │
│    │    [--cpu-limit $CPU_LIMIT]          │       │
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
export CPU_LIMIT=1                    # default when set: 1 core per agent
#  export CPU_LIMIT=0.5              # fractional: 0.5 core per agent
#  unset CPU_LIMIT                   # no CFS limit — native Linux scheduling
export WORKERS=320                    # default: os.cpu_count(). Controls per-process event-loop load
export PREP_CONCURRENCY=0             # default: 0 (auto=20 system-wide)
export CONTAINER_EXE=docker           # default

# 3. Run the sweep
bash scripts/run_simulate_sweep.sh
```

The script validates prerequisites (trace count, Docker availability, host
cores/memory) before starting, then runs each N sequentially.  On interrupt
(Ctrl+C), it cleanly stops the active system monitor and exits.

### CPU_LIMIT behavior

| `CPU_LIMIT` | Docker flag | Scheduling | Output dir suffix |
|-------------|-------------|------------|-------------------|
| `unset` | *(none)* | Native CFS, no throttle | `sweep_${N}a_nolimit` |
| `0.5` | `--cpus=0.5` | CFS quota 50ms/100ms | `sweep_${N}a_0.5cpu` |
| `1` | `--cpus=1` | CFS quota 100ms/100ms | `sweep_${N}a_1cpu` |
| `2` | `--cpus=2` | CFS quota 200ms/100ms | `sweep_${N}a_2cpu` |

> **Recommendation for single-trace cloud_model replay:** omit `CPU_LIMIT` (unset).
> Cloud replay is I/O-bound — containers spend most wall time in `sleep`
> waiting for source-trace timing or in `docker exec` waiting for shell
> commands.  CFS throttle provides no benefit but adds scheduling overhead.
>
> **Recommendation for sweep / over-subscription stress testing:** set `CPU_LIMIT=1`.
> When N agents share M host cores (N ≫ M), `--cpus=1` enforces fair per-agent
> allocation via Docker CFS quotas, producing clean, reproducible contention
> curves.  Without a limit, the kernel scheduler may allocate CPU unevenly,
> adding noise and reducing cross-experiment comparability.

### Output Per Experiment

Each N produces a self-contained output directory.  The suffix reflects
the `CPU_LIMIT` setting:

```text
traces/simulate/swe-rebench/sweep_${N}a_${CPU_LIMIT}cpu/   # with CPU limit
traces/simulate/swe-rebench/sweep_${N}a_nolimit/           # without CPU limit

├── system_resources.jsonl       # whole-host resource timeline
├── system_viz.html              # interactive system resource charts
├── agent_timeline.jsonl         # per-agent lifecycle log
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
| `scripts/run_mixed_scheduling_sweep.sh` | Orchestrator that compares sequential vs interleaved scheduling for two SWE-rebench replay workloads under the same total CPU budget. Both workloads are container-mode and use resource monitoring. | `SOURCE_TRACES_DIR_A=<dir> SOURCE_TRACES_DIR_B=<dir> bash scripts/run_mixed_scheduling_sweep.sh` |

#### Environment Variables Reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `SOURCE_TRACES_DIR` | *(required)* | Path to pre-collected trace directories |
| `SWEEP_VALUES` | `40 80 160 320` | Space-separated list of N (agent count) values |
| `CPU_LIMIT` | `1` | Per-container CPU quota via Docker `--cpus=N`. Unset for native scheduling |
| `WORKERS` | `os.cpu_count()` | Number of worker processes — splits agents across independent event loops |
| `PREP_CONCURRENCY` | `0` (auto) | System-wide concurrent container preparation limit shared across all workers. `0` uses the default limit of 20 |
| `REPLAY_SPEED` | `1` | Wall-clock acceleration multiplier |
| `PYTHON_BIN` | `python3` | Python interpreter |
| `CONTAINER_EXE` | `docker` | Container runtime |
| `TASK_CONTAINER_NO_PROXY` | *(unset)* | Set to `1` to prevent host proxy env vars (`HTTP_PROXY`, `ALL_PROXY`, etc.) from leaking into task containers. Useful when the host proxy (e.g. a SOCKS5 proxy that breaks HTTPS) should not affect in-container pip operations. |
| `BASE_OUTPUT_DIR` | `traces/simulate/swe-rebench` | Root output directory |
| `STRICT_SYSTEM_MONITOR` | `1` in `run_mixed_scheduling_sweep.sh` | Fail a mixed-scheduling run if the outer system monitor cannot start. Use `0` only for exploratory debugging. |
| `CPU_CORE_LIST` | *(generated)* | Optional explicit list of CPU ids for mixed scheduling. Overrides `CPU_CORE_START`/`CPU_CORE_STRIDE`. |
| `CPU_CORE_START` | `0` | First CPU id when generating the mixed-scheduling core list. |
| `CPU_CORE_STRIDE` | `1` | Step between generated CPU ids. Set `2` for layouts such as `0,2,4,6`, `8,10,12,14`. |
| `LLC_CLUSTER_SIZE` | `4` | Number of configured core ids per LLC-slice cluster for interleaved A/B balancing. |

All sweep support scripts are committed to the repository and ready to use on the
target machine.  See `docs/EXPERIMENT_PLAN_arm_sweep.md` for a detailed
design document covering rationale, risk assessment, and implementation
notes.

### Rebench-vs-Rebench Mixed Scheduling Sweep

Use `scripts/run_mixed_scheduling_sweep.sh` when the experiment needs two
stable, container-mode SWE-rebench workloads rather than a SWE-rebench plus
Deep Research mix. This avoids network API variability from Deep Research
traces while preserving the sequential vs interleaved scheduling comparison.

The script expects two source directories, each containing one or more
`trace.jsonl` files. With `--trace-assignment manifest`, a single case
directory is valid: replay assigns the same trace to multiple agents and
suffixes duplicate agent IDs.

Every replay agent is pinned to exactly one CPU core with repeated
`--agent-cpuset <core>` arguments. Sequential phases draw single-core agent
placements from the full configured core list. Interleaved phases split the
configured core list inside each LLC cluster, alternating A/B assignments so a
cluster such as `0,2,4,6` receives two A cores and two B cores.

Recommended inspected cases from `C:\Users\29068\Desktop\agent_datasets`:

| Role | Case | Observed duration | Rationale |
|---|---|---:|---|
| CPU/memory-heavy | `swe-rebench/AI4S2S__lilio-49/attempt_1` | ~943 s | High resource pressure among duration-matched candidates: avg CPU ~109%, peak memory ~8.9 GB, substantial pytest/tool execution. |
| LLM/context-heavy | `swe-rebench/Azure__azure-cli-2955/attempt_1` | ~1058 s | Similar duration, larger trace/context footprint (~29.7 MB), heavier read/context activity, and lower memory peak than the CPU/memory-heavy case. |

Example:

```bash
export SOURCE_TRACES_DIR_A=/mnt/c/Users/29068/Desktop/agent_datasets/swe-rebench/AI4S2S__lilio-49/attempt_1
export SOURCE_TRACES_DIR_B=/mnt/c/Users/29068/Desktop/agent_datasets/swe-rebench/Azure__azure-cli-2955/attempt_1
export WORKLOAD_A_LABEL=cpu-memory-heavy
export WORKLOAD_B_LABEL=llm-heavy
export SWEEP_VALUES="40 80 160 320"
export TOTAL_CORES=160
export CPU_CORE_START=0
export CPU_CORE_STRIDE=2
export LLC_CLUSTER_SIZE=4
export CPU_LIMIT=1
export STRICT_SYSTEM_MONITOR=1

bash scripts/run_mixed_scheduling_sweep.sh
```

On Git Bash or another shell, use that shell's equivalent path form. The
script requires `taskset`, so WSL/Linux is the expected runtime environment.

For each N, the script runs:

- `sequential_N<N>/workload_a` then `sequential_N<N>/workload_b`, each with
  single-core agent placements drawn from the full configured core list.
- `interleaved_N<N>/workload_a` and `interleaved_N<N>/workload_b` in parallel,
  using disjoint A/B core lists balanced within each LLC cluster.

The summary file records total wall time plus per-workload elapsed seconds and
success flags. Do not change case selection, N values, replay speed, or CPU
budget for result-producing runs without documenting the rationale.

Preflight checks require both source directories to contain SWE-rebench trace
metadata, both task-source JSON files to exist, and every `SWEEP_VALUES` entry
to be a positive even integer. `STRICT_SYSTEM_MONITOR=1` is the default: if the
outer system monitor cannot start, the affected simulate run fails instead of
silently producing a result without system-level resource data. Set it to `0`
only for exploratory debugging, not result-producing runs.

For a 160-core experiment on a machine where LLC clusters are represented as
`0,2,4,6`, then `8,10,12,14`, set `CPU_CORE_STRIDE=2` and
`LLC_CLUSTER_SIZE=4`. If the machine uses a different topology, set
`CPU_CORE_LIST` explicitly to the ordered list of cores grouped by LLC slice.

### Interpreting Sweep Output

This section explains how to read the metrics produced by each sweep run,
what their baseline values mean, and where to find throughput data.

#### system_viz.html Metrics

The HTML page renders 5 time-series charts with a summary stats header.

**Summary Cards:**

| Card | Meaning |
|------|---------|
| CPU Avg (max X%) | Whole-host CPU utilization averaged over the experiment. 320 cores fully saturated = 100%. |
| Mem Avg (max X GB) | Average and peak used memory (includes page cache, not just process RSS). |
| Max Containers | Peak number of concurrently running Docker containers. |
| Duration | Time span from the monitor's first sample to its last sample (slightly longer than simulate wall time — see below). |

**Chart 1 — CPU Utilization & System Load:**

- **CPU %** (blue): Whole-host CPU utilization per sample.
- **Load 1m / 5m / 15m** (orange/red/brown; 5m/15m hidden by default — toggle in legend): Linux load average — the exponentially-weighted moving average of processes in runnable (R) or uninterruptible sleep (D) state over 1, 5, and 15 minutes.

> **load vs cpu_count:** `load ≈ cpu_count` means CPU is saturated with no queueing. `load >> cpu_count` means severe CPU contention — processes are waiting.
>
> **load vs CPU%:** CPU% is instantaneous utilization (like a speedometer). Load average is a smoothed queue-depth metric (like a congestion index). Both should be consulted together.

**Chart 2 — Memory Usage:**

- **Mem Used (GB)** (green): `psutil.virtual_memory().used` — includes page cache. This is NOT process-exclusive memory.
- **Mem Total (GB)** (grey): Total physical RAM.

> **Why memory starts high at t=0:** The system monitor starts ~1 second before `simulate`. The t=0 reading reflects the host's pre-existing memory usage (OS page cache, Docker daemon, other users' processes, etc.). SWE-bench containers are lightweight (tens of MB each) and Docker uses copy-on-write image layers, so 320 containers add relatively little incremental memory.

**Chart 3 — Container Count:**

- **Containers** (purple): Tracks the container lifecycle — ramp-up (rate-limited system-wide by `--prep-concurrency`, default auto: 20), steady state, and teardown.

**Chart 4 — Network I/O Rate (MB/s):**

- **Net RX** (cyan) / **Net TX** (teal): Per-second deltas computed from cumulative `/proc/net/dev` counters.

> **Why SWE-bench coding tasks have network IO:** Even in `cloud_model` mode (no real LLM API calls), network traffic comes from Docker daemon communication (`docker exec` stdout/stderr streaming), container image pulls, SSH session traffic, and NFS/distributed storage if traces or outputs reside on network mounts. ~20 MB/s RX is expected with 320 concurrent containers.

**Chart 5 — Disk I/O Rate (MB/s):**

- **Disk Read / Write**: Per-second deltas from cumulative `/proc/diskstats` counters. Peaks typically align with container startup (image layer reads) and teardown.

#### Measurement Methodology

All system-level metrics are collected by `scripts/system_resource_monitor.py` using:

| Metric | Python API | Underlying Source | Accuracy |
|--------|-----------|-------------------|----------|
| `cpu_percent` | `psutil.cpu_percent(interval=None)` | `/proc/stat` | Same as `top`/`htop`. Non-blocking call returns utilization since last call. 1 Hz sampling may miss sub-second spikes. |
| `mem_used_gb` | `psutil.virtual_memory()` | `/proc/meminfo` | Byte-accurate. **Includes page cache** — not process-exclusive RSS. |
| `load_1m/5m/15m` | `os.getloadavg()` | `/proc/loadavg` | Kernel-maintained EMA. Not available on Windows (falls back to 0). |
| `disk_read_mb` / `disk_write_mb` | `psutil.disk_io_counters()` | `/proc/diskstats` | Cumulative bytes since first sample. Accurate for whole-disk I/O. |
| `net_rx_mb` / `net_tx_mb` | `psutil.net_io_counters()` | `/proc/net/dev` | Cumulative bytes across ALL interfaces (including loopback). Does NOT isolate per-container traffic. |
| `container_count` | `subprocess.run([docker, ps, -q])` | Docker CLI | Accurate count of running containers. |

**Key caveats:**
- `cpu_percent(interval=None)` is non-blocking — it compares against the previous call. The very first reading may be unreliable; discard if needed.
- `mem_used_gb` includes OS page cache. For process-exclusive memory, subtract cached/buffered.
- Network IO aggregates ALL interfaces (eth0, docker0, lo, etc.). Container-level network is available in per-container `resources.json` instead.

#### Duration vs Wall Time

There are **two distinct duration values**:

| Metric | Source | Scope |
|--------|--------|-------|
| **system_viz.html Duration** | `last_sample_ts - first_sample_ts` from monitor JSONL | Monitor startup (~1s before simulate) through monitor shutdown (after simulate exits). Includes everything. |
| **Throughput Wall Time** | `RUN_END - RUN_START` from shell script | Container preparation + all replays + container teardown + trace split + HTML viz generation. Does NOT include monitor startup/teardown or post-processing scripts. |

For throughput analysis, use the **shell Wall Time** (reported in the sweep summary table and `sweep_summary_*.txt`).

#### Throughput Data Locations

| What | Where |
|------|-------|
| **Per-sweep summary** | `traces/simulate/swe-rebench/sweep_summary_<ts>.txt` — lists each N with wall time, exit code, monitor samples |
| **Terminal output** | After all N complete, the script prints a `Throughput Summary` table with `Agents/s` and `Agents/min` |
| **Agent-level timing** | `agent_timeline.jsonl` — per-agent `start_ts`, `end_ts`, `elapsed_s` for P50/P95/P99 analysis |
| **System resource timeline** | `system_resources.jsonl` — per-second samples for CPU saturation analysis |
| **Per-agent trace** | `<agent_id>--aN/attempt_1/trace.jsonl` — full action sequence with per-action timing |

---

## Stress Test Analysis

When running a sweep across N ∈ {40, 80, 160, 320, 640}, the following
metrics reveal where system bottlenecks emerge as concurrency grows.

### Analysis Checklist

For each N, inspect:

- [ ] **`max(agent_exec_s)`** — batch wall time. Under ideal scaling this
      stays flat as N grows. Rising values indicate CPU/IO/event-loop
      congestion. With sufficient `--workers`, the remaining rise reflects
      Docker daemon and system resource contention.
- [ ] **LLM sleep drift** (`llm_latency_ms` vs expected sleep time) —
      `asyncio.sleep()` wake-up delay. With `--workers` ≥ N/4 this should
      approach zero. If still significant, increase `--workers`.
- [ ] **Tool overhead** (`(ts_end - ts_start) * 1000 - duration_ms` for
      actually-executed tools) — Docker pipe transport + event-loop
      scheduling cost. With high `--workers`, the remaining value reflects
      pure system-call and pipe overhead.
- [ ] **Tool slowdown** (`duration_ms` vs `source_duration_ms`) — actual
      tool execution time inflation due to CPU contention. When N ≫ host
      cores, single-threaded tools may slow down as they compete for
      scheduling slices.
- [ ] **System resource curves** (`system_resources.jsonl`) — CPU
      utilization, memory pressure, context-switch rate, and disk/network
      I/O. Which resource saturates first at which N?

### Practical Run Commands

The following commands are tailored for a **320 vCPU ARM host** with
N ∈ {160, 320, 640}.  All parameters follow the recommendations from
the [Choosing the Right Setting](#choosing-the-right-setting) section.

#### Recommended Sweep (N ∈ {160, 320, 640})

```bash
export SOURCE_TRACES_DIR=/path/to/40-traces
export SWEEP_VALUES="160 320 640"
export WORKERS=320
bash scripts/run_simulate_sweep.sh
```

This minimal command already captures all the signal.  The three parameters
below have **marginal impact** on replay timing at this scale — they are
defensive defaults whose effect plateaus once the host is correctly
configured:

| Parameter | Default | Impact on replay results | Keep it? |
|-----------|---------|--------------------------|----------|
| `CPU_LIMIT` | `1` (script default) | Near-zero — I/O-bound replay rarely triggers CFS throttle. Only matters when a task runs `make -j` or `pip install`. | ✅ Safety net, zero cost |
| `WORKERS` | `os.cpu_count()` (320) | Already at the sweet spot — 2 agents/worker at N=640, event-loop congestion near-zero. Raising to 640 wastes memory. | ✅ Optimal by default |
| `PREP_CONCURRENCY` | `0` (auto=20) | **Zero** — only speeds up warm-up, does not touch replay timing. `64` makes prep finish ~3× faster. | 🟡 Purely convenience |

#### Comparison: No CPU Limit

```bash
export SOURCE_TRACES_DIR=/path/to/40-traces
export SWEEP_VALUES="160 320 640"
export WORKERS=320
CPU_LIMIT="" bash scripts/run_simulate_sweep.sh
# Output dir suffix: sweep_${N}a_nolimit
```

#### Faster Warm-Up

If preparation time matters (e.g., 640 containers × 20 concurrent = slow ramp):

```bash
export PREP_CONCURRENCY=64
bash scripts/run_simulate_sweep.sh
# Cuts prep phase to ~1/3 of default. Does not affect replay.
```

#### Quick Smoke Test

```bash
export SOURCE_TRACES_DIR=/path/to/40-traces
export SWEEP_VALUES="40 80"
export WORKERS=320
bash scripts/run_simulate_sweep.sh
# Each N completes in ~2-5 min on a 320-core host with REPLAY_SPEED=1
```

### Key Signals

| Signal | What to look for | Diagnosis |
|--------|-----------------|-----------|
| Batch wall time | Linear with N | Expected: 40a 2min, 640a 32min |
| Batch wall time | Super-linear with N | Event-loop contention or Docker daemon saturation — increase `--workers` |
| LLM sleep drift | N/workers > ~4 | Event-loop loading too high — increase `--workers` |
| Tool duration | Increases with N | CPU or I/O contention — check `system_resources.jsonl` |
| CPU % | Plateaus below 100% | Another bottleneck (memory bandwidth, disk I/O) is limiting throughput |
| Context switches | Spike at high N | Kernel scheduler overhead — may need `--cpu-limit` |
| `load >> cpu_count` | Load average much higher than core count | Severe CPU queueing — processes waiting for CPU time |

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

The `--ksys` alias, or `--ksys-monitoring on`, starts
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

---

## Container Environment Reproducibility

For container-mode benchmarks (e.g., swe-rebench), **collect** and
**simulate** must produce identical tool-execution outputs for the same
tool call.  If the container environment differs between the two modes
(e.g., different pre-installed Python packages), the agent in simulate
may see different results and diverge from the original trace.

### Architecture

```
                          HOST FILESYSTEM
 ┌──────────────────────────────────────────────────────────────────┐
 │  $HOME/.cache/task-container-bootstrap/                          │
 │  ├── pydeps/                   ← pip install --target (bootstrap)│
 │  │   └── .bootstrap-ready.json ← marker: requirements + manifest │
 │  └── .pyuserbase/              ← get-pip.py --user               │
 │      ├── bin/pip                                                 │
 │      └── lib/python3.12/site-packages/                           │
 │                                                                  │
 │  Mounted into every container via -v $HOME:$HOME                 │
 └──────────────────────────────────────────────────────────────────┘

              CONTAINER (ephemeral)         CONTAINER (ephemeral)
              ┌──────────────────┐         ┌──────────────────┐
              │ /testbed         │         │ /testbed         │
              │ site-packages/   │         │ site-packages/   │
              │ (non-persistent) │         │ (non-persistent) │
              └──────────────────┘         └──────────────────┘
```

### Two Caches, Two Risks

The bootstrap process maintains **two** shared directories on the host
filesystem, both mounted into every container:

| Directory | Purpose | Populated by |
|-----------|---------|-------------|
| `pydeps/` | Agent runtime dependencies (openai, pydantic, tiktoken, …) | `pip install --target` during bootstrap |
| `.pyuserbase/lib/.../site-packages/` | pip itself (installed by `get-pip.py --user`) + its transitive dependencies | `get-pip.py --user` during bootstrap |

The container has `PYTHONUSERBASE` set, which causes Python to add
`.pyuserbase/lib/python3.12/site-packages/` to `sys.path`.  Any
packages installed there by a **previous run** (collect or simulate)
will persist and be importable by all subsequent containers.

### How Contamination Happens

SWE-bench containers run as **non-root** (via `--user uid:gid`, see
`container_run_user_args` in `container_runtime.py`).  The conda
environment at `/opt/conda/` is installed by root during image build,
so its `site-packages/` is **not writable** by the container user.

As a result, every agent `pip install` follows this fallback chain:

1. pip tries `/opt/conda/lib/python3.12/site-packages/` → **Permission denied**
2. `PIP_BREAK_SYSTEM_PACKAGES=1` bypasses the EXTERNALLY-MANAGED marker
   but cannot override filesystem permissions.
3. pip falls back to `--user` installation → writes to `PYTHONUSERBASE`
   (`~/.cache/task-container-bootstrap/.pyuserbase/`).

This is **expected and correct** for the benchmark — non-root containers
are a security best practice and match the official SWE-bench evaluation
environment.  Giving root would deviate from the reference setup and
introduce reproducibility risks (agent could modify system files).

The contamination chain across runs:

```
collect run A → agent pip install -e . → .pyuserbase/ (non-root, must user-install)
                                              ↓
                    .pyuserbase/ now contains traitlets, xarray_leaflet, ...
                                              ↓
simulate run B → PYTHONUSERBASE set → Python finds stale packages
                                              ↓
              → simulate agent sees DIFFERENT environment than collect agent
```

> **Note:** Within a single run, agent packages installed to `.pyuserbase/`
> are fully functional — `PYTHONUSERBASE` is on `sys.path`, so imports and
> tests work correctly.  The problem is only *cross-run* leakage.

### Three-Layer Defense

The codebase implements three complementary defenses:

| Layer | Location | Mechanism |
|-------|----------|-----------|
| **Source prevention** | `_REPLAY_AGENT_SCRIPT._exec_command` in `openclaw_tools.py` | Strips `PYTHONUSERBASE` from the subprocess environment before executing agent tool commands (pip install, pytest, etc.). Agent tools now install to the container's ephemeral site-packages. |
| **Cache-hit verification** | `_bootstrap_marker_matches` in `task_container.py` | On every bootstrap cache hit, compares the actual `.dist-info` directories in both `pydeps/` and `.pyuserbase/site-packages/` against the package manifest recorded in `.bootstrap-ready.json`. |
| **Auto-rebuild on contamination** | `bootstrap_task_container_python` in `task_container.py` | When contamination is detected (extra or missing packages vs. the manifest), the stale directories are removed and the bootstrap is rebuilt from scratch. |

The marker JSON (`.bootstrap-ready.json`) records:

```json
{
  "requirements": ["openai>=2.0,<3.0", "httpx>=0.27,<1.0", "..."],
  "arch": "amd64",
  "packages": ["annotated_types", "anyio", "certifi", "..."],
  "userbase_packages": ["pip", "setuptools", "wheel"]
}
```

- `packages` — packages installed to `pydeps/` via `pip install --target`
- `userbase_packages` — packages installed to `.pyuserbase/site-packages/`
  via `get-pip.py --user`

**Legacy markers** (without `packages` / `userbase_packages` keys) are
treated as stale and trigger an automatic rebuild on the next run.

### Design Rationale

- **Bootstrap packages are cached by design** — they are fixed
  (`openai`, `pydantic`, etc.) and identical across all benchmark
  instances.  Caching avoids reinstalling these on every run.

- **Agent-installed packages should NOT be cached** — they are
  benchmark-specific and should be isolated per run.  The
  `PYTHONUSERBASE` stripping in `_exec_command` (simulate path)
  ensures pip does not discover stale packages from previous runs.

- **Collect path tolerates userbase installation** — because the
  container runs as non-root and conda site-packages is root-owned,
  agent `pip install` unavoidably falls back to `--user`.  This is
  harmless *within* a single run (the packages are functional) and
  the contamination is cleaned by the bootstrap check before the
  next run starts.

- **Non-root is intentional** — matches the official SWE-bench
  evaluation environment.  Granting root would deviate from the
  reference setup and risk agent modifications to system state.

- **The contamination check is a safety net** — it catches any
  residual contamination that may have occurred before these
  defenses were deployed, or from manual intervention on the host.

### Checking for Contamination

If you suspect environment drift, check the bootstrap log:

```bash
# Look for contamination messages in simulate output
grep "contamination" trace_output.log
# Example output:
# [bootstrap] userbase cache contamination detected in .../site-packages:
#   extra=['traitlets', 'xarray_leaflet', ...]
```

Or inspect the shared cache directly:

```bash
ls ~/.cache/task-container-bootstrap/pydeps/*.dist-info/
ls ~/.cache/task-container-bootstrap/.pyuserbase/lib/python*/site-packages/*.dist-info/
cat ~/.cache/task-container-bootstrap/pydeps/.bootstrap-ready.json | python -m json.tool
```

To force a clean rebuild for the next run:

```bash
rm -rf ~/.cache/task-container-bootstrap/pydeps
rm -rf ~/.cache/task-container-bootstrap/.pyuserbase/lib/python*/site-packages
```
