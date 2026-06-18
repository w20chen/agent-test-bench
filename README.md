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

## Repository Layout

```text
agent-sched-bench/
├── configs/            # benchmark / system / trace_collect / sweep YAMLs
├── demo/gantt_viewer/  # FastAPI backend + Solid.js frontend
├── docs/               # specs and plans
├── scripts/            # setup, download, smoke, and runner shells
├── src/
│   ├── agents/         # scaffolds + benchmark plugins
│   ├── harness/        # runner, samplers, metrics, trace logger
│   ├── llm_call/       # provider registry + OpenAI-compatible client
│   └── trace_collect/  # CLI: collect / simulate / import / inspect / gantt-serve
└── tests/
```

## Resource Measurement Architecture

The harness layer (`src/harness/`) provides a comprehensive, cross-platform
resource observability stack. Every benchmark attempt records time-series
samples of CPU, memory, disk I/O, network I/O, context switches, host memory
bandwidth, and CPU micro-architecture (PMU) metrics. Two sampler backends
cover both containerised and host-process workloads.

### Sampler Topology

```
                    ┌───────────────────────────────┐
                    │     attempt_pipeline.py       │
                    │  (orchestrates both samplers) │
                    └──────┬────────────┬───────────┘
                           │            │
              ┌────────────▼───┐  ┌─────▼────────────────┐
              │ ContainerStats │  │ ProcessStatsSampler  │
              │   Sampler      │  │ (host-process mode)  │
              │ (Docker/Podman)│  │                      │
              └──────┬─────────┘  └──────┬───────────────┘
                     │                   │
      ┌──────────────┼───────────────────┼──────────────┐
      │              ▼                   ▼              │
      │   ┌─────────────────┐  ┌─────────────────┐      │
      │   │ attach_host_    │  │ attach_micro_   │      │
      │   │ memory_bandwidth│  │ arch            │      │
      │   └────────┬────────┘  └────────┬────────┘      │
      │            │                    │               │
      │   ┌────────▼────────────────────▼─────────┐     │
      │   │         Linux perf_event subsystem    │     │
      │   │   (/sys/bus/event_source/devices/)    │     │
      │   └───────────────────────────────────────┘     │
      └─────────────────────────────────────────────────┘
```

Each sample dict emitted by either sampler carries a union of all metric
families; unavailable metrics are omitted rather than zeroed so downstream
consumers (HTML viz, summary aggregation) can distinguish "absent" from
"zero".

---

### 1. CPU & Memory (per-sample, always available)

| Metric | Container Mode | Host-Process Mode |
|---|---|---|
| **CPU %** | cgroup v2 `cpu.stat` (`usage_usec` delta / wall-clock / nproc) | `psutil.Process.cpu_percent()` summed over process tree |
| **Memory RSS** | cgroup v2 `memory.current` (bytes) | `psutil.Process.memory_info().rss` summed over process tree |
| **Memory Limit** | cgroup v2 `memory.max` (bytes) | N/A |

**Container fallback path:** When cgroup v2 is unavailable (e.g. cgroup v1
hosts), `ContainerStatsSampler` falls back to `docker stats --no-stream
--format` and parses the pipe-delimited output (`MemUsage|MemPerc|CPUPerc|NetIO`).

**Host-process fallback path:** When `psutil` is not installed,
`ProcessStatsSampler` falls back to running `ps -o %cpu= -o rss= -p <pid>`
and parsing its output.

---

### 2. Disk I/O (per-sample)

| Mode | Data Source | Method |
|---|---|---|
| **Container (cgroup)** | `/sys/fs/cgroup/<path>/io.stat` | Read `rbytes=` and `wbytes=` fields aggregated across all block devices. Monotonic cumulative counters, includes exited processes. |
| **Container (exec fallback)** | `docker exec <id> python3 -c "..."` | Reads cgroup `io.stat` inside the container namespace; falls back to summing `/proc/*/io` if cgroup unavailable inside. Uses high-water marks because `/proc/*/io` is non-monotonic. |
| **Host-process** | `psutil.Process.io_counters()` | `read_bytes` / `write_bytes` summed over process tree. Per-process high-water marks keyed by `(pid, create_time)` prevent counter drops when children exit. Falls back to `/proc/<pid>/io` (`read_bytes` / `write_bytes` from block layer) for the root PID only. |

**High-water mark consolidation:** Both samplers periodically fold per-PID /
per-process high-water marks into consolidated running totals (every 200
samples), preventing unbounded dict growth on long-running agents that spawn
many short-lived child processes.

---

### 3. Network I/O (per-sample)

| Mode | Data Source | Method |
|---|---|---|
| **Container** | `/proc/<container_pid>/net/dev` | Reads cumulative RX/TX bytes from all non-loopback interfaces. Parsed from the proc file directly (columns 0 and 8). |
| **Host-process** | `/proc/<pid>/net/dev` | Same method as container mode, scoped to the agent root PID. |

**Alternative container path:** The `docker stats` pipe-delimited output
includes a `NetIO` field (e.g. `1.5kB / 2.3MB`), parsed for RX/TX bytes
with support for SI and binary unit suffixes (`kB`, `KiB`, `MB`, `MiB`,
`GB`, `GiB`, `TB`, `TiB`, `B`).

---

### 4. Context Switches (per-sample)

| Mode | Data Source | Method |
|---|---|---|
| **Container (cgroup)** | `/proc/<pid>/status` for each PID in `cgroup.procs` | Sums `voluntary_ctxt_switches` + `nonvoluntary_ctxt_switches` across all container PIDs. Per-PID high-water marks keyed by `(pid, starttime)` for stable identity across PID reuse. |
| **Container (exec fallback)** | `docker exec <id> python3 -c "..."` | Sums `ctxt_switches` from `/proc/*/status` inside the container. Uses high-water mark (non-monotonic source). |
| **Host-process** | `psutil.Process.num_ctx_switches()` | `voluntary + involuntary` summed over process tree. Per-process high-water marks keyed by `(pid, create_time)`. Falls back to `/proc/<pid>/status` for the root PID. |

---

### 5. Host Memory Bandwidth (per-sample)

**Tool:** Linux `perf stat` (requires `linux-tools-common` or equivalent).
**Privilege:** `sudo sysctl -w kernel.perf_event_paranoid=-1` (otherwise
reads 0 without error).

The `HostMemoryBandwidthCollector` is a module-level singleton daemon thread
that runs `perf stat -a -e <events> -- sleep <interval>` on a loop. Each
sample attaches the latest reading via `attach_host_memory_bandwidth()`.

#### Backend Detection Chain (in priority order)

| Priority | Backend Kind | Detection Method | Platforms |
|---|---|---|---|
| 1 | `intel_imc_cas` | Scans `/sys/bus/event_source/devices/uncore_imc_*` for CAS count events | Intel Xeon (Haswell through Emerald Rapids) |
| 2 | `explicit_byte_events` | Scans ALL devices in `event_source/devices/` for named `read_bytes` / `write_bytes` (or `bytes_read` / `bytes_write`, etc.) events | ARM DDRC with named events, other PMUs exposing byte counters |
| 3 | `arm_ddrc` (speculative) | Scans for `hisi_ddrc*` / `arm_ddrc*` devices WITHOUT named events; uses raw event codes | HiSilicon Kunpeng 920, generic ARM DDRC (⚠️ not validated on hardware) |

**Intel IMC CAS Count method:**
- Named events: `uncore_imc_<N>/cas_count_read/`, `uncore_imc_<N>/cas_count_write/`
- Raw fallback (Sapphire/Emerald Rapids without `events/` dir): `event=0x04,umask=0x03/` (read), `event=0x04,umask=0x0c/` (write)
- Conversion: each CAS (Column Access Strobe) = 64 bytes (one cache line)

**Explicit byte events method:**
- Searches for event aliases: `read_bytes`, `bytes_read`, `data_read_bytes`, `data_read` (and corresponding write variants)
- `bytes_per_count = 1.0` (the event directly reports bytes)

**ARM DDRC raw method (LAST RESORT, speculative):**
- HiSilicon DDRC: `event=0x02` (flux_rcmd, read commands), `event=0x03` (flux_wcmd, write commands)
- ARM DDRC: `event=0x00` (read), `event=0x01` (write)
- Assumes 64 bytes per command (may be inaccurate; logged with warning)

**Output metrics:** `memory_total_mb_s`, `memory_read_mb_s`, `memory_write_mb_s`
(total/read/write bandwidth in MiB/s).

**Scope:** Always system-wide (`perf stat -a`). Host memory bandwidth is a
shared resource; per-container filtering is not meaningful for DRAM
bandwidth measured at the memory controller level.

---

### 6. Micro-Architecture PMU Metrics (per-sample)

**Tool:** Linux `perf stat` (requires `linux-tools-common` or equivalent).
**Privilege:** `sudo sysctl -w kernel.perf_event_paranoid=-1`.

The `MicroArchCollector` is a module-level singleton daemon thread that
alternates between event groups to avoid PMU counter multiplexing. Each
`perf stat` call samples one group for the sampling interval, then the
collector rotates to the next group.

#### Event Group Design

Core PMUs have limited programmable counters (4 on x86, 6 on ARMv8). To
avoid multiplexing (which introduces error on bursty LLM workloads), events
are partitioned into groups that each fit within the counter budget:

| Group | Contents | Counters Used (x86) | Counters Used (ARMv8) |
|---|---|---|---|
| **cache** | cycles, instructions, L1D access, L1D miss (+ L1I on ARMv8) | 4 | 6 |
| **icache** | cycles, instructions, L1I access, L1I miss | 4 (x86 only) | N/A |
| **branch** | cycles, instructions, branch inst, branch miss (+ bus_access on ARMv8) | 4 | 5 |

On x86, L1D and L1I are sampled in separate groups (cache + icache) to
stay within the 4-counter limit. On ARMv8, L1D and L1I are combined in
the cache group (6 counters available).

The collector cycles through groups in order: cache → (icache) → branch →
cache → ... . Each sample receives the latest reading from each group.

#### Platform Detection & Event Codes

Auto-detection probes `/sys/bus/event_source/devices/` and
`/proc/cpuinfo`, then selects one of four platform specs:

| Platform Spec | Detection Trigger | L1D Events | L1I Events | Branch Events | Bus Access |
|---|---|---|---|---|---|
| **`armv8-raw`** | `armv8_pmu*` or `armv8_cortex*` in sysfs, OR ARM implementer ID in cpuinfo (`0x41`, `0x48` HiSilicon/Kunpeng, etc.) | `r04` (L1D_CACHE), `r03` (L1D_CACHE_REFILL) | `r14` (L1I_CACHE), `r01` (L1I_CACHE_REFILL) | `r12` (BR_PRED), `r10` (BR_MIS_PRED) | `r19` (BUS_ACCESS) |
| **`x86-intel`** | `cpu`/`intel` in sysfs + cpuinfo `GenuineIntel` | `L1-dcache-loads`, `L1-dcache-load-misses` | `L1-icache-loads`, `L1-icache-load-misses` | `branch-instructions`, `branch-misses` | None |
| **`x86-amd`** | `cpu`/`amd` in sysfs + cpuinfo `AuthenticAMD` | `L1-dcache-loads`, `L1-dcache-load-misses` | `cpu/event=0x80/`, `cpu/event=0x81/` (raw) | `branch-instructions`, `branch-misses` | None |
| **`generic`** | Fallback (always succeeds) | `cache-references`, `cache-misses` (LLC-level, NOT L1 — approximation) | None | `branch-instructions`, `branch-misses` | None |

**ARMv8 raw event codes** are architectural per the ARM ARM — they work on
all ARMv8-A cores including Cortex-A53/57/72/76, Neoverse N1, and HiSilicon
TSV110 (Kunpeng 920).

**AMD L1I raw events:** `cpu/event=0x80/` (L1I access), `cpu/event=0x81/`
(L1I miss). These bypass the kernel's generic `PERF_TYPE_HW_CACHE` mapping,
which is unreliable for L1I on AMD Zen 1-4 (verified against
`arch/x86/events/amd/core.c`). L1D uses standard named events which work
correctly on AMD.

**Generic fallback:** `cache-references` / `cache-misses` count at the
last-level cache (LLC), not L1. L1 hit rate derived from these is an
approximation, but the metrics are available on ANY CPU with a Linux PMU
driver (x86, ARM, POWER, RISC-V).

#### Perf Scoping

The sampler attempts three scoping modes in priority order:

| Priority | Scope | Perf Args | Requirement |
|---|---|---|---|
| 1 | `cgroup` | `--cgroup <path>` | `perf` compiled with cgroup support; cgroup v2 path available |
| 2 | `process` | `-p <container_pid>` | Container init PID known |
| 3 | `system_wide` | `-a` | Always available |

Cgroup-scoped sampling isolates PMU counts to the container's cgroup,
excluding noise from other workloads on the host.

#### Derived Metrics

| Metric | Formula | Available On |
|---|---|---|
| **IPC** | `instructions / cycles` | All platforms |
| **Instructions/s** | `instructions / interval_s` | All platforms |
| **L1D Hit Rate** | `1.0 - l1d_miss / l1d_access` | `armv8-raw`, `x86-intel`, `x86-amd` (L1-level); `generic` (LLC-level) |
| **L1I Hit Rate** | `1.0 - l1i_miss / l1i_access` | `armv8-raw`, `x86-intel`, `x86-amd` |
| **L1I Hit Rate (fallback)** | `1.0 - l1i_miss / instructions` | When L1I access event is unsupported (conservative proxy) |
| **Branch Miss Rate** | `branch_miss / branch_inst` | All platforms |
| **Bus Access/s** | `bus_access / interval_s` | `armv8-raw` only (memory traffic proxy) |

#### Perf Output Parsing

`perf stat -x,` produces CSV output where the third field is the canonical
event name. The parser matches on this field exactly (not substring scan)
to avoid false matches on description substrings like `insn per cycle`.
Events reporting `<not counted>` or `<not supported>` are silently skipped
so remaining metrics in the group are still computed.

#### Failure Modes

| Reason | Cause |
|---|---|
| `perf_missing` | `perf` executable not found in `$PATH` |
| `permission_denied` | `kernel.perf_event_paranoid` > 0 and not running as root |
| `pmu_unsupported` | No suitable PMU detected (e.g. virtualised environment without PMU passthrough) |
| `unsupported_platform` | Non-Linux OS (macOS, Windows) |
| `cgroup_unsupported` | `perf` binary lacks `--cgroup` support |

---

### 7. Metric Aggregation (Summary)

Each sampler's `stop()` method returns raw sample lists. The
`summarize_samples()` function in `container_stats_sampler.py` computes
per-metric `{min, max, avg}` across all samples, plus `delta` (last − first)
for cumulative counters (disk I/O, network I/O, context switches).

The summary is written to `resources.json` beside the trace file, and is
consumed by the HTML Gantt viewer for resource overlay charts.

---

### 8. Architecture Support Matrix

| Resource Metric | x86 Intel | x86 AMD | ARMv8 (Kunpeng 920) | Generic Linux | macOS/Windows |
|---|---|---|---|---|---|
| CPU / Memory | ✅ cgroup or docker stats | ✅ cgroup or docker stats | ✅ cgroup or docker stats | ✅ docker stats fallback | ❌ |
| Disk I/O | ✅ cgroup io.stat | ✅ cgroup io.stat | ✅ cgroup io.stat | ✅ exec fallback | ❌ |
| Network I/O | ✅ /proc/net/dev | ✅ /proc/net/dev | ✅ /proc/net/dev | ✅ /proc/net/dev | ❌ |
| Context Switches | ✅ /proc/status | ✅ /proc/status | ✅ /proc/status | ✅ /proc/status | ❌ |
| Memory Bandwidth | ✅ Intel IMC CAS | ✅ Intel IMC CAS | ⚠️ ARM DDRC (speculative) | ❌ | ❌ |
| IPC | ✅ named events | ✅ named events | ✅ raw events (r08/r11) | ✅ generic | ❌ |
| L1D Hit Rate | ✅ L1-level | ✅ L1-level | ✅ L1-level | ⚠️ LLC-level approx | ❌ |
| L1I Hit Rate | ✅ named events | ✅ raw events (0x80/0x81) | ✅ raw events (r14/r01) | ❌ | ❌ |
| Branch Miss Rate | ✅ named events | ✅ named events | ✅ raw events (r12/r10) | ✅ generic | ❌ |
| Bus Access | ❌ | ❌ | ✅ r19 | ❌ | ❌ |

> **Note on Kunpeng 920 (HiSilicon TSV110):** The ARMv8 architectural raw
> event codes are mandated by the ARM ARM and work on all ARMv8-A cores.
> Auto-detection recognises HiSilicon implementer ID `0x48` from
> `/proc/cpuinfo`. Memory bandwidth via ARM DDRC raw events is marked
> speculative and has not been validated on physical Kunpeng hardware.

---

### 9. HTML Visualization

The Gantt viewer (`src/trace_collect/html_viz.py`) renders all resource
metrics as time-series charts below the Gantt chart:

- **CPU & Memory** — dual-axis chart (CPU % left, memory MiB right)
- **Memory Bandwidth** — total / read / write MiB/s (hidden with reason if unavailable)
- **Micro-Architecture (PMU)** — L1D hit rate, L1I hit rate, branch miss rate (0-1 scale), IPC (right axis). Uses `spanGaps: true` so the alternating group sampling pattern does not produce misleading zero-values.
- **Network I/O** — RX/TX rate in MiB/s (computed from cumulative byte deltas)
- **Disk I/O** — read/write rate in MiB/s (computed from cumulative byte deltas)
- **Context Switches** — rate in switches/s
- **Ksys (Huawei Kunpeng)** — when `--ksys` is passed, `ksys collect` runs
  alongside the agent and its stdout/stderr are captured in the attempt
  directory (see **Ksys System Metrics** below).

Each chart gracefully degrades when its data source is unavailable, showing
a descriptive error message with the specific reason and remediation hint.

---

## Development Workflow

All Python invocations run inside conda env "ML" (Python 3.12). On a fresh
server, run `bash scripts/setup/bootstrap.sh` once — it installs miniconda,
creates env ML, installs deps, and runs a 1-task terminal-bench smoke. Do
not create `.venv` or `pip install` ad hoc.

```bash
conda activate ML
make help    # list all targets
make test    # run pytest
make lint    # ruff
```

## Using the Repo: Three Entry Points

This repo supports three progressively deeper ways to interact with benchmarks.
The sections below walk through each:

- **Inspect cases** — browse benchmark tasks without running an agent (next section).
- **Run an agent interactively** — send a one-shot prompt to the OpenClaw agent
  via CLI (see [Quick Test](#quick-test)).
- **Run a full benchmark** — execute the agent on many tasks with container
  orchestration, trace collection, and result aggregation (see
  [Trace Collect](#trace-collect) onwards).

## SWE-bench Case Inspection (No Agent Required / Review Cases Directly)

`scripts/inspect_swebench.py` is a **standalone script** for quickly inspecting
SWE-bench Verified and SWE-rebench test cases. No agent system integration
needed — just Docker + HuggingFace datasets.

**Use cases:** Reviewers who want to see what benchmark cases look like —
repo structure, problem statements, and test details.

### Prerequisites

```bash
pip install datasets docker

export HF_ENDPOINT=https://hf-mirror.com
```

### Usage Examples

```bash
# List the first 20 SWE-bench Verified tasks
python scripts/inspect_swebench.py --benchmark swe-bench-verified list

# Search by keyword
python scripts/inspect_swebench.py --benchmark swe-bench-verified list -k django

# View full task details (problem statement, test command, image name)
python scripts/inspect_swebench.py --benchmark swe-bench-verified info django__django-10097

# Pull the Docker image (~2 GB, may take a few minutes)
python scripts/inspect_swebench.py --benchmark swe-bench-verified pull django__django-10097

# List files under /testbed inside the container
python scripts/inspect_swebench.py --benchmark swe-bench-verified ls django__django-10097

# View a specific file inside the container
python scripts/inspect_swebench.py --benchmark swe-bench-verified cat django__django-10097 /testbed/setup.py

# View the gold fix patch (what the agent is expected to produce)
python scripts/inspect_swebench.py --benchmark swe-bench-verified diff django__django-10097

# Live git diff inside the container (after making manual edits in a shell)
python scripts/inspect_swebench.py --benchmark swe-bench-verified diff django__django-10097 --container

# Show FAIL_TO_PASS tests grouped by source file (understand what needs fixing)
python scripts/inspect_swebench.py --benchmark swe-bench-verified tests django__django-10097

# View a specific test file
python scripts/inspect_swebench.py --benchmark swe-bench-verified tests django__django-10097 -f tests/auth_tests/test_validators.py

# Export the entire /testbed to a local directory
python scripts/inspect_swebench.py --benchmark swe-bench-verified export django__django-10097 /testbed ./export_django/

# Enter an interactive bash shell in the container (most flexible)
python scripts/inspect_swebench.py --benchmark swe-bench-verified shell django__django-10097
```

### SWE-rebench Works the Same Way

```bash
# SWE-rebench
python scripts/inspect_swebench.py --benchmark swe-rebench list
python scripts/inspect_swebench.py --benchmark swe-rebench info 12rambau__sepal_ui-411
python scripts/inspect_swebench.py --benchmark swe-rebench pull 12rambau__sepal_ui-411
python scripts/inspect_swebench.py --benchmark swe-rebench shell 12rambau__sepal_ui-411
```

### Use Local Cache to Skip HF Download

If you've already downloaded data via `make download-swebench-verified`,
use the local `tasks.json` cache to skip the HuggingFace download:

```bash
python scripts/inspect_swebench.py \
    --benchmark swe-bench-verified \
    --cache-file data/swebench_verified/tasks.json \
    list
```

### Common Workflows

**Workflow 1: Quickly browse a few cases**

```bash
# 1. List some tasks
python scripts/inspect_swebench.py -b swe-bench-verified list -n 5

# 2. Pick one and view details
python scripts/inspect_swebench.py -b swe-bench-verified info sympy__sympy-12481

# 3. Pull image + enter shell to explore code
python scripts/inspect_swebench.py -b swe-bench-verified pull sympy__sympy-12481
python scripts/inspect_swebench.py -b swe-bench-verified shell sympy__sympy-12481
# Inside container: ls /testbed, cat /testbed/setup.py, git log, etc.
```

**Workflow 2: Export all files for offline analysis**

```bash
python scripts/inspect_swebench.py -b swe-bench-verified pull astropy__astropy-12907
python scripts/inspect_swebench.py -b swe-bench-verified export astropy__astropy-12907 /testbed ./case_astropy_12907/
# Then open ./case_astropy_12907/ in your local IDE
```

**Workflow 3: See the gold fix patch (the expected solution)**

```bash
# Shows both the code patch and the test patch from the dataset
python scripts/inspect_swebench.py -b swe-bench-verified diff django__django-10097
```

**Workflow 4: Understand what tests need to pass (FAIL_TO_PASS)**

```bash
# See which test files are involved and how many tests per file
python scripts/inspect_swebench.py -b swe-bench-verified tests django__django-10097

# Then view a specific test file
python scripts/inspect_swebench.py -b swe-bench-verified tests django__django-10097 -f tests/auth_tests/test_validators.py
```

The inspection script is read-only — it helps you understand what a benchmark
case looks like. The next section covers actually running an agent to solve
these cases.

## Trace Collect

Run an agent scaffold on a benchmark and record a canonical v5 JSONL trace per
task. The CLI requires an explicit `--provider` and `--model` and loads
benchmark specifics from `configs/benchmarks/<slug>.yaml`.

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

Key flags: `--benchmark <slug>` (default `swe-bench-verified`),
`--scaffold openclaw|tongyi-deepresearch`, `--mcp-config` (required for
`openclaw`; YAML path or the literal `none`), `--sample N`,
`--instance-ids a,b,c`, `--run-id <path>` (resume an interrupted run),
`--prompt-template <name>` (override the benchmark default).

See `src/trace_collect/CLAUDE.md` for the complete flag reference, provider
registry, checkpointing behaviour, and trace schema v5 layout.

To run the deep research bench:

```bash
PYTHONPATH=src python -m trace_collect.cli \
    --provider deepseek \
    --model deepseek-v4-flash \
    --benchmark deep-research-bench \
    --scaffold openclaw \
    --sample 1 \    # only run the first case
    --mcp-config none \
    --verbose
```

Deep research bench ships two prompt templates under
`configs/prompts/deep_research_bench/`:

| Template | Behaviour |
|----------|-----------|
| `default` | Agent uses the `spawn` tool to launch 2–4 parallel subagents that decompose and research independent facets, then synthesises their findings. |
| `no_spawn` | Pure single-agent mode — no subagent spawning. The agent searches, reads, and answers on its own. |

Switch with `--prompt-template <name>`, e.g. `--prompt-template no_spawn`.

The commands above cover the general pattern. The next section is a concrete,
step-by-step walkthrough for running SWE-rebench end-to-end on an ARM server,
including environment setup, image preparation, execution, and troubleshooting.

In addition to the benchmark runner, the repo also ships a **standalone
OpenClaw CLI** for sending one-shot prompts without the benchmark harness:

```bash
PYTHONPATH=src python -m agents.openclaw \
    --prompt "Write a Python script to download web page and parse title" \
    --provider deepseek \
    --model deepseek-chat \
    --workspace ./workspace
```

This is the quickest way to test that your LLM provider is wired correctly
before running a full benchmark.  Use `--async` for background runs and
`--status --session-id <id>` to check progress (see `openclaw --help`).

## Quick Test

End-to-end walkthrough for running a single SWE-rebench task on an ARM server.

Prerequisites: ARM server + DeepSeek API + Docker

The harness supports **two ARM image modes**, controlled by the
``ARM_IMAGE_MODE`` environment variable:

| Mode | Env | Behaviour |
|------|-----|-----------|
| **native** (default) | ``ARM_IMAGE_MODE=native`` or unset | Build a shared ARM base image once; clone each task's repo from a local bare mirror at runtime. No QEMU needed. |
| **qemu** | ``ARM_IMAGE_MODE=qemu`` | Pull the official x86_64 per-task Docker images and run them via QEMU binfmt emulation. Requires ``make setup-arm-host`` first. |

Choose the mode that fits your environment before proceeding with Step 1.

### Step 0

```bash
export HF_ENDPOINT=https://hf-mirror.com

sudo tee /etc/docker/daemon.json <<'EOF'
{
  "registry-mirrors": [
    "https://docker.1panel.live",
    "https://dockerpull.org"
  ]
}
EOF
sudo systemctl restart docker

export KEEP_IMAGES_ABOVE_GB=30

sudo sysctl -w kernel.perf_event_paranoid=-1

export WEB_SEARCH_PROVIDER=tavily

export TASK_CONTAINER_PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

### Step 1 — One-time environment setup

**Native mode (default):**

```bash
# Build the ARM-native base image and download SWE-rebench data + repos
make setup-arm-native

# Activate the conda environment
conda activate ML
```

ARM hosts auto-detect and use the native ``swe-arm-base`` image with local
repo mirrors — no QEMU emulation needed.

**QEMU mode:**

```bash
# Install QEMU binfmt handlers so Docker can run x86_64 images on ARM
make setup-arm-host

# Download the dataset metadata (task list) — images are pulled on demand
make download-swe-rebench

# Activate the conda environment
conda activate ML

# Then set ARM_IMAGE_MODE=qemu when running (see Step 2).
```

In QEMU mode the official ``swerebench/sweb.eval.x86_64.<task>`` images are
pulled and executed via QEMU user-mode emulation.  The ARM base image
(``make setup-arm-native``) is not needed.

### Step 1b — Pre-pull images (recommended for QEMU mode)

In **QEMU mode** each SWE-rebench task uses its own ~2 GB Docker image
(``swerebench/sweb.eval.x86_64.<task>:latest``).  Pulling them ahead of
time avoids network stalls during the run.  (In native mode there is only
one shared base image; pre-pulling is unnecessary.)

```bash
# Pull images for the first 16 tasks (match --sample 16)
make pull-swe-rebench-images PULL_SAMPLE=16

# Pull for specific tasks
./scripts/setup/pull_swe_rebench_images.sh \
    --instance-ids "12rambau__sepal_ui-411,0b01001001__spectree-64"

# Concurrent pulls (4 at a time)
make pull-swe-rebench-images PULL_SAMPLE=16 PULL_PARALLEL=4

# Pull everything (6,500+ images — use with care!)
make pull-swe-rebench-images
```

Already-pulled images are re-used across runs and only removed when disk
runs low (set ``KEEP_IMAGES_ABOVE_GB`` to raise the threshold; see Step 0).

### Step 2 — Run

Check valid `instance_id`s.

```bash
python -c "
import json
tasks = json.load(open('data/swe-rebench/tasks.json'))
for t in tasks[:20]:
    print(t['instance_id'], '|', t.get('repo',''))
print(f'... ({len(tasks)} total)')
"
```

Run a specific test case.

**Native mode (default):**

```bash
DEEPSEEK_API_KEY=sk-deepseek-api-key PYTHONPATH=src python -m trace_collect.cli \
    --provider deepseek \
    --model deepseek-v4-flash \
    --benchmark swe-rebench \
    --scaffold openclaw \
    --instance-ids "12rambau__sepal_ui-411" \
    --mcp-config none \
    --verbose \
    --container docker
```

The first run on a task builds a cached ARM derivative image
(``swe-arm-fixed-<instance_id>``).  Subsequent runs skip the build step
and start immediately.

**QEMU mode:**

```bash
ARM_IMAGE_MODE=qemu DEEPSEEK_API_KEY=sk-deepseek-api-key PYTHONPATH=src python -m trace_collect.cli \
    --provider deepseek \
    --model deepseek-v4-flash \
    --benchmark swe-rebench \
    --scaffold openclaw \
    --instance-ids "12rambau__sepal_ui-411" \
    --mcp-config none \
    --verbose \
    --container docker
```

The official x86_64 task image is pulled on first use; a writable
derivative (``swebench-fixed-*``) is cached for subsequent runs.  Docker
transparently handles the x86_64 → ARM emulation via QEMU.

### Step 2b — Replay

```bash
PYTHONPATH=src python -c "
import json
from pathlib import Path
from agents.benchmarks.swe_bench_verified import SweBenchVerified
from agents.benchmarks.base import BenchmarkConfig

config = BenchmarkConfig.from_yaml(Path('configs/benchmarks/swe-bench-verified.yaml'))
plugin = SweBenchVerified(config)
tasks = plugin.load_tasks()  # load all 500 tasks without select_subset
Path('data/swebench_verified/tasks_full.json').write_text(
    json.dumps(tasks, indent=2, ensure_ascii=False, default=str) + '\n',
    encoding='utf-8',
)
print(f'Wrote {len(tasks)} tasks')
"
```

```bash
PYTHONPATH=src python -m trace_collect.cli simulate \
    --source-trace traces/swebench_verified/deepseek-v4-flash/20260605T182234/astropy__astropy-12907/attempt_1/trace.jsonl \
    --mode cloud_model \
    --replay-speed 1 \
    --task-source data/swebench_verified/tasks_full.json \
    --container docker
```

```bash
PYTHONPATH=src python -m trace_collect.cli simulate \
    --source-dir /home/weitian/agent-test-bench/traces/swebench_verified/deepseek-v4-flash/20260605T182234 \
    --mode cloud_model \
    --replay-speed 1 \
    --task-source data/swebench_verified/tasks_full.json \
    --container docker
    --serial
```

### Troubleshooting

```bash
docker ps
# Check whether the task container is running
ls -lt traces/swe-rebench/deepseek-chat/<run-timestamp>/12rambau__sepal_ui-411/attempt_1/_task_container_runtime/
# Check run progress
curl -s https://api.deepseek.com/v1/models -H "Authorization: Bearer sk-your-key" | tail -1
# Verify API connectivity
```

### Visualise results

```bash
PYTHONPATH=src python -m trace_collect.html_viz traces/swe-rebench/deepseek-chat/20260603T030206/12rambau__sepal_ui-411/attempt_1
```

### Recording Internals

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

### Ksys System Metrics

`--ksys` starts `ksys collect` as a background process alongside the agent
and stops it (SIGINT) when the agent finishes.  The raw stdout/stderr are
written to `ksys_stdout.txt` / `ksys_stderr.txt` in the attempt directory.

- **Default: off.**  Pass `--ksys` to enable.
- **No-op when `ksys` is not installed** on the host (graceful degradation).
- **Timeline alignment:** ksys starts at the same point as other resource
  samplers, so its data shares the Gantt chart's t0 (time origin).

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

## Supported Benchmarks

The repo ships with plugin-based benchmark support. Each benchmark is defined
by a YAML config in `configs/benchmarks/` and a Python plugin in
`src/agents/benchmarks/`. The table below lists all registered benchmarks,
their task shape, data source, runtime environment, and supported scaffolds.

To add a new benchmark, follow the plugin architecture: create a YAML config
and a Python class inheriting from `agents.benchmarks.base.Benchmark`.

### Registered Benchmarks

| Slug | `task_shape` | Dataset | Split | Docker | Scaffolds | Scoring |
|---|---|---|---|---|---|---|
| `swe-bench-verified` | `swe_patch` | `princeton-nlp/SWE-bench_Verified` | `test` | `swebench/sweb.eval.x86.*` (namespace-prefixed) | openclaw | harness (pytest in container) |
| `swe-rebench` | `swe_patch` | `nebius/SWE-rebench` | `filtered` | `swerebench/sweb.eval.x86_64.*` (fully qualified) | openclaw | harness (pytest in container) |
| `terminal-bench` | `terminal_task` | `terminal-bench-core` (or local task dir) | n/a | Terminal-Bench-managed | openclaw (phase 1) | Terminal-Bench harness + imported OpenClaw trace |
| `deep-research-bench` | `research_qa` | configured in YAML | `test` | host | openclaw, tongyi-deepresearch | reference-answer comparison |
| `browsecomp` | `browse_qa` | configured in YAML | `test` | host | openclaw, tongyi-deepresearch | reference-answer comparison |

Terminal-Bench requires Python 3.12+ (upstream `tb` CLI dependency) and only
supports `--scaffold openclaw` with the Docker runtime in phase 1.

In addition to the SWE-style and QA benchmarks above, the repo also integrates
with the **BFCL (Berkeley Function Calling Leaderboard)** datasets as host-mode
benchmarks. Unlike SWE-bench tasks which run inside Docker containers, BFCL
runs directly on the host using OpenClaw's multi-turn loop.

## BFCL via OpenClaw (host-mode)

Four BFCL (Berkeley Function Calling Leaderboard) datasets are wired as
host-mode benchmarks driven by the OpenClaw scaffold. BFCL runs as a read-only
external environment: its dataset loader, tool-doc → schema converter, stateful
backend instantiation, and backend implementations (simulated filesystem,
booking, web search, vector/kv/rec_sum memory) are reused unmodified; OpenClaw
issues the LLM calls and runs the multi-turn loop. Each BFCL conversation turn
is delivered one at a time, so the agent only ever sees the current and past
turns. Traces, `resources.json`, and a per-task HTML report come from the
standard host pipeline — BFCL's own evaluation/scoring is **not** run.

| Slug | BFCL categories | Notes |
|---|---|---|
| `bfcl-multi-turn-base` | `multi_turn_base` | stateful file/booking/etc. tools |
| `bfcl-multi-turn-long-context` | `multi_turn_long_context` | large injected backend state |
| `bfcl-web-search` | `web_search_base`, `web_search_no_snippet` | needs a web-search API key |
| `bfcl-memory` | `memory_vector`, `memory_kv`, `memory_rec_sum` | prereq conversations run first |

### Full run on a fresh machine

**1. Get both repos.** Clone this repo and the BFCL (`gorilla`) repo anywhere:

```bash
git clone <this-repo> agent-test-bench
git clone https://github.com/ShishirPatil/gorilla.git gorilla
```

**2. Create the Python env** (uv, Python 3.12) at this repo root:

```bash
cd agent-test-bench
uv venv --python 3.12
uv pip install -e ".[dev]"
```

**3. Install the BFCL runtime deps** these four datasets need (pins match
BFCL's `pyproject.toml`):

```bash
uv pip install \
  "tree_sitter==0.21.3" "tree-sitter-java==0.21.0" "tree-sitter-javascript==0.21.4" \
  "faiss-cpu==1.11.0" "sentence-transformers" "rank_bm25" "overrides" \
  "google-search-results" "html2text" "beautifulsoup4"
```

**4. Point at the BFCL checkout** (no machine-specific default is baked in):

```bash
export BFCL_REPO_PATH=/abs/path/to/gorilla   # dir containing berkeley-function-call-leaderboard/
```

**5. Provide API keys / model assets:**

- LLM provider key for `--provider/--model` (e.g. `DASHSCOPE_API_KEY`,
  `OPENAI_API_KEY`, `OPENROUTER_API_KEY`).
- `bfcl-web-search`: a SerpAPI key — `export SERPAPI_API_KEY=...`.
- `bfcl-memory` (vector): first run downloads `all-MiniLM-L6-v2` from
  HuggingFace. If a global `HF_ENDPOINT` mirror is set but its model-config API
  path fails, force the real host for the run with
  `HF_ENDPOINT=https://huggingface.co` (works through an HTTP/SOCKS proxy).
- For host memory-bandwidth sampling: `sudo sysctl -w kernel.perf_event_paranoid=-1`
  (otherwise the bandwidth fields read 0; everything else still works).

**6. Run the full test set** (omit `--sample` for all entries; `--mcp-config none`
disables MCP). Examples — repeat per slug:

```bash
PYTHONPATH=src python -m trace_collect.cli \
    --provider <provider> --model <model> \
    --benchmark bfcl-multi-turn-base \
    --scaffold openclaw --mcp-config none

PYTHONPATH=src python -m trace_collect.cli --provider <provider> --model <model> \
    --benchmark bfcl-multi-turn-long-context --scaffold openclaw --mcp-config none

PYTHONPATH=src python -m trace_collect.cli --provider <provider> --model <model> \
    --benchmark bfcl-web-search --scaffold openclaw --mcp-config none      # SERPAPI_API_KEY

PYTHONPATH=src python -m trace_collect.cli --provider <provider> --model <model> \
    --benchmark bfcl-memory --scaffold openclaw --mcp-config none          # faiss + ST
```

Smoke first with `--sample 1` (or `--instance-ids <id>`) before a full run.
Per-task artifacts land under
`traces/<slug>/<safe-model>/<timestamp>/<instance_id>/attempt_1/`:
`trace.jsonl`, `resources.json`, and the HTML report
`<slug>__<instance_id>.html` (named so a full run's HTMLs are uniquely
identifiable and collectible into one folder, e.g.
`find traces -name '<slug>__*.html'`).

> Note on `bfcl-memory`: each scenario's prerequisite "memory write"
> conversations are emitted before its question entries and must run in that
> order (the plugin preserves load order instead of sorting by id), so memory
> state is populated before the questions. Sampling a subset may split a
> scenario — prefer a full run, or select whole scenarios via `--instance-ids`.

## Resource Monitoring

The harness ships five resource monitors that activate depending on the
runtime mode (container, host process, local GPU serving, or deep-profile).
Each sample carries a UTC timestamp and an epoch; monitors run on background
threads or asyncio tasks at a configurable interval.

### Monitors by Runtime Mode

| Runtime Mode | Active Monitor(s) | Key Source Files |
|---|---|---|
| **container** (Docker/Podman) | `ContainerStatsSampler` | `src/harness/container_stats_sampler.py` |
| **host** (no container) | `ProcessStatsSampler` | `src/harness/process_stats_sampler.py` |
| **local_model** (vLLM GPU replay) | `VLLMMetricsClient` + `GpuResourceSampler` | `src/harness/metrics_client.py`, `src/harness/gpu_resource_sampler.py` |
| **deep-profile** (PyTorch hooks) | `ComponentMemoryProfiler` | `src/harness/component_memory_profiler.py` |

### 1. ContainerStatsSampler — container CPU / memory / I/O

Reads Linux cgroup v2 counters directly (sub-millisecond overhead, no Docker
daemon round-trip). Falls back to `docker stats --no-stream` +
`docker exec python3` on non-cgroup-v2 hosts.

| Metric | Source | Isolation |
|--------|--------|-----------|
| CPU % | cgroup v2 `cpu.stat` `usage_usec` delta between consecutive samples, divided by `ncpus` | ✅ container-scoped |
| Memory | cgroup v2 `memory.current` / `memory.max` | ✅ container-scoped |
| Disk I/O (read/write bytes) | cgroup v2 `io.stat` aggregated across all devices; fallback: sum `/proc/*/io` inside container via `docker exec` | ✅ container-scoped |
| Context switches | sum of `voluntary_ctxt_switches` + `nonvoluntary_ctxt_switches` across all PIDs in the container's `cgroup.procs`, with per-(pid, starttime) high-water marks so exited processes are not lost | ✅ container-scoped |
| Network I/O (rx/tx bytes) | `/proc/<pid>/net/dev` of the container init PID (loopback excluded) | ✅ container-scoped |
| Memory bandwidth | `perf stat -a` reading Intel IMC CAS counters (system-wide); requires `kernel.perf_event_paranoid=-1` | ⚠️ host-wide |

### 2. ProcessStatsSampler — host process CPU / memory / I/O

Uses `psutil` (fallback: `ps`) to sample the target PID and its recursive
children. Complements with `/proc/<pid>/{io,status,net/dev}` reads where
available.

| Metric | Source | Isolation |
|--------|--------|-----------|
| CPU % | `psutil.Process.cpu_percent()` + recursive children; fallback: `ps -o %cpu=` | ✅ process-tree scoped |
| Memory (RSS) | `psutil.Process.memory_info().rss` + children RSS summed; fallback: `ps -o rss=` | ✅ process-tree scoped |
| Disk I/O | `psutil.Process.io_counters()` (read_bytes, write_bytes) + children; also reads `/proc/<pid>/io` | ✅ process-tree scoped |
| Context switches | `/proc/<pid>/status` (`voluntary_ctxt_switches` + `nonvoluntary_ctxt_switches`) | ✅ process-tree scoped |
| Network I/O | `/proc/<pid>/net/dev` (loopback excluded) | ✅ process-tree scoped |
| Memory bandwidth | same `perf stat -a` as ContainerStatsSampler | ⚠️ host-wide |

### 3. VLLMMetricsClient — vLLM scheduler Prometheus metrics

Polls the vLLM Prometheus endpoint (`/metrics`) and parses gauge/counter
values. All metrics are vLLM-internal — naturally isolated to the vLLM
process.

| Metric | Prometheus Gauge |
|--------|-----------------|
| Running / waiting requests | `vllm:num_requests_running`, `vllm:num_requests_waiting` |
| GPU / CPU KV cache usage | `vllm:gpu_cache_usage_perc` (or `vllm:kv_cache_usage_perc` in vLLM 0.10+), `vllm:cpu_cache_usage_perc` |
| Preemption count | `vllm:num_preemptions_total` |
| Prefix cache hit rates | `vllm:gpu_prefix_cache_hit_rate`, `vllm:cpu_prefix_cache_hit_rate` |
| Throughput | `vllm:avg_prompt_throughput_toks_per_s`, `vllm:avg_generation_throughput_toks_per_s` |
| Latency | `vllm:e2e_request_latency_seconds` (sum + count), `vllm:time_to_first_token_seconds` |

### 4. GpuResourceSampler — GPU memory breakdown (local_model only)

Combines three data sources at a configurable rate (default 10 Hz) to produce
a `GpuMemoryBreakdown` time series:

| Component | Source | Description |
|-----------|--------|-------------|
| `weights_mib` | `GpuBaseline` parsed from vLLM startup log | One-shot; model weights resident in GPU memory |
| `kv_cache_total_mib` | `GpuBaseline` parsed from vLLM startup log | Total GPU memory budget for KV cache |
| `kv_cache_used_mib` | vLLM Prometheus `gpu_cache_usage_perc` × `kv_cache_total_mib` | How much of the KV budget is currently occupied |
| `total_pid_mib` | `nvidia-smi --query-compute-apps=pid,gpu_serial,used_memory` filtered by vLLM PID | Total GPU memory held by the vLLM process |
| `activations_mib` | residual: `total_pid_mib - weights_mib - kv_cache_used_mib` | Estimated activation memory (clamped ≥ 0) |

All GPU memory readings are per-PID, so other users' GPU processes do not
affect them. Fails fast with `GpuPidNotFoundError` if the vLLM PID disappears
mid-run.

### 5. ComponentMemoryProfiler — per-layer activation memory (deep-profile)

Attaches PyTorch forward hooks to attention and MLP submodules (detected by
class-name pattern matching: `attention|attn`, `mlp|feedforward|ffn`). Uses
`torch.cuda.memory_allocated()` pre/post delta to estimate per-component
activation memory. Outputs `GpuComponentBreakdown` records per generate step.
This is entirely in-process and not affected by other users.

### Output Format

All monitors emit samples that feed into `summarize_samples()` producing a
`resources.json` with min/max/avg for every metric plus deltas for monotonic
counters (disk, network, context switches). The HTML viz renders these as
time-series overlays on the Gantt chart.

### Multi-Tenant Safety Summary

| Monitoring Layer | Affected by other users on the same machine? |
|---|---|
| Container cgroup (CPU, memory, disk I/O, ctxt switches) | ❌ No |
| Container network I/O (init PID `/proc/net/dev`) | ❌ No |
| Process-level (psutil, `/proc/<pid>/*`, ps) | ❌ No |
| GPU memory per PID (`nvidia-smi --query-compute-apps`) | ❌ No |
| vLLM Prometheus internal metrics | ❌ No |
| PyTorch `torch.cuda.memory_allocated()` | ❌ No |
| **Host memory bandwidth (`perf stat -a`)** | ⚠️ **Yes — system-wide PMC counters** |

> **Note on memory bandwidth:** `perf stat -a` monitors all CPUs and IMCs on
> the host. If other users are running memory-intensive workloads on the same
> physical machine, `memory_total_mb_s` / `memory_read_mb_s` /
> `memory_write_mb_s` in the output will include their traffic. For clean
> measurements, run on a dedicated node or treat these as upper-bound
> estimates. Set `sudo sysctl -w kernel.perf_event_paranoid=-1` before
> starting the harness, otherwise bandwidth sampling will report
> `permission_denied` and all bandwidth fields will be 0.

### Adding a New Benchmark

Benchmarks live in `src/agents/benchmarks/<slug>.py` with a matching
`configs/benchmarks/<slug>.yaml`. See `src/agents/benchmarks/base.py` for the
`Benchmark` ABC and `BenchmarkConfig` fields, plus `CLAUDE.md §Benchmark Plugin
Architecture` for the enforcement rules. Short version:

1. Implement a `Benchmark` subclass in `src/agents/benchmarks/<slug>.py`.
2. Author `configs/benchmarks/<slug>.yaml` with the `BenchmarkConfig` fields.
3. Register the class in `src/agents/benchmarks/__init__.py::REGISTRY`.
4. Add `tests/test_<slug>_plugin.py` covering `normalize_task`.
5. Add `make download-<slug>` / `make setup-<slug>-repos` Makefile targets.

Dataset names, image namespaces, and CLI-visible defaults must live in YAML —
never hardcode them in `collector.py`, `cli.py`, or scaffold code.

## Trace Simulate

Replay a collected trace under new infrastructure assumptions. Two modes:

| Mode | LLM calls | Timing source | Multi-trace | Use case |
|---|---|---|---|---|
| `cloud_model` | replayed from source trace (no API call) | `ts_start`/`ts_end` × `--replay-speed` | yes (`--trace-manifest`) | "what if N agents arrive concurrently?" |
| `local_model` | sent to a real OpenAI-compatible endpoint | live TTFT + TPOT | single (`--source-trace`) | "what if we self-host on local vLLM?" |

### cloud_model — arrival-pattern sweep

```bash
PYTHONPATH=src python -m trace_collect.cli simulate \
    --trace-manifest configs/trace_collect/simulate.yaml \
    --mode cloud_model \
    --container docker \
    --replay-speed 50 \
    --arrival-mode poisson \
    --arrival-rate-per-s 0.5 \
    --arrival-seed 42
```

### local_model — self-hosted serving

```bash
PYTHONPATH=src python -m trace_collect.cli simulate \
    --source-trace traces/.../trace.jsonl \
    --mode local_model \
    --provider openai --api-base http://localhost:8000/v1 \
    --api-key dummy --model Qwen/Qwen3-32B \
    --container docker \
    --metrics-url http://localhost:8000/metrics
```

`--metrics-url` snapshots vLLM Prometheus counters per iteration
(`num_preemptions_total`, `gpu_cache_usage_perc`, `*_prefix_cache_hit_rate`) into
`TraceAction.data.sim_metrics`. Container resource usage is sampled at 1 Hz by
`ContainerStatsSampler` and written to `resources.json`.

**GPU memory tracking** (`--gpu-tracking on`): add `--vllm-pid`, `--vllm-startup-log`,
and `--gpu-sample-hz` to capture a full GPU memory breakdown time-series (weights,
KV cache, activations) sampled in the background and written to `gpu_resources.json`.
For per-component (attn/mlp) deep profiling without a separate server, use the
`profile-gpu` subcommand (requires `pip install -e .[profile]`; GPU + vLLM only).

See `src/trace_collect/CLAUDE.md` §Simulate for the full flag table, manifest
format, output directory layout, and simulation-specific fields in `action.data`.

## Importing Claude Code Sessions

Convert a raw Claude Code session JSONL to canonical trace format:

```bash
PYTHONPATH=src python -m trace_collect.cli import-claude-code \
    --session ~/.claude/projects/<slug>/<uuid>.jsonl \
    --output-dir traces
```

Sidechains under `subagents/` are folded in by default (pass `--no-sidechains`
to skip). The Gantt viewer's `/api/traces/register` and `/api/traces/upload`
endpoints auto-invoke this importer when they detect raw CC JSONL.

## Inspecting Traces

```bash
PYTHONPATH=src python -m trace_collect.cli inspect traces/.../trace.jsonl overview
PYTHONPATH=src python -m trace_collect.cli inspect traces/.../trace.jsonl timeline --json
PYTHONPATH=src python -m trace_collect.cli inspect traces/.../trace.jsonl tools --agent <instance-id>
```

Subcommands: `overview`, `step`, `messages`, `response`, `events`, `tools`,
`search`, `timeline`. Filters: `--agent`, `--role`, `--category`, `--iteration`.

## Demo: Gantt Viewer

Interactive multi-lane Gantt visualisation for collected and simulated traces.
FastAPI backend (`:8765`) + Solid.js / Vite frontend rendered on Canvas 2D.

```bash
make gantt-viewer-install   # one-time: npm install
make gantt-viewer-dev       # dev mode (Vite HMR on :5173)
make gantt-viewer-build     # production bundle into frontend/dist
make gantt-viewer-test      # backend pytest + frontend vitest
make gantt-viewer-smoke     # headless browser smoke check
```

Equivalent CLI:

```bash
PYTHONPATH=src:. python -m trace_collect.cli gantt-serve --dev
PYTHONPATH=src:. python -m trace_collect.cli gantt-serve \
    --config demo/gantt_viewer/configs/example.yaml
```

Discovery config globs accept canonical trace JSONL only. At runtime,
`POST /api/traces/register` and `POST /api/traces/upload` auto-import raw
Claude Code sessions through `import-claude-code` before registration.

Smoke-only subsets belong in dedicated `*_smoke.yaml` workload configs; default
workload configs should describe the full benchmark dataset path.

See `demo/gantt_viewer/README.md` for the full acceptance workflow and
`demo/gantt_viewer/AGENT_INTERFACE.md` for the REST-driven agent interface.
