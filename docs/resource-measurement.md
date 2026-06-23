# Resource Measurement Architecture

> This document is part of the [Agent Sched Bench manual](../README.md).
> For getting started, see [Getting Started](getting-started.md).
> For CLI usage, see [Trace Collect](trace-collect.md).

The harness layer (`src/harness/`) provides a comprehensive, cross-platform
resource observability stack. Every benchmark attempt records time-series
samples of CPU, memory, disk I/O, network I/O, context switches, host memory
bandwidth, and CPU micro-architecture (PMU) metrics. Two sampler backends
cover both containerised and host-process workloads.

## Sampler Topology

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

## 1. CPU & Memory (per-sample, always available)

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

## 2. Disk I/O (per-sample)

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

## 3. Network I/O (per-sample)

| Mode | Data Source | Method |
|---|---|---|
| **Container** | `/proc/<container_pid>/net/dev` | Reads cumulative RX/TX bytes from all non-loopback interfaces. Parsed from the proc file directly (columns 0 and 8). |
| **Host-process** | `/proc/<pid>/net/dev` | Same method as container mode, scoped to the agent root PID. |

**Alternative container path:** The `docker stats` pipe-delimited output
includes a `NetIO` field (e.g. `1.5kB / 2.3MB`), parsed for RX/TX bytes
with support for SI and binary unit suffixes (`kB`, `KiB`, `MB`, `MiB`,
`GB`, `GiB`, `TB`, `TiB`, `B`).

---

## 4. Context Switches (per-sample)

| Mode | Data Source | Method |
|---|---|---|
| **Container (cgroup)** | `/proc/<pid>/status` for each PID in `cgroup.procs` | Sums `voluntary_ctxt_switches` + `nonvoluntary_ctxt_switches` across all container PIDs. Per-PID high-water marks keyed by `(pid, starttime)` for stable identity across PID reuse. |
| **Container (exec fallback)** | `docker exec <id> python3 -c "..."` | Sums `ctxt_switches` from `/proc/*/status` inside the container. Uses high-water mark (non-monotonic source). |
| **Host-process** | `psutil.Process.num_ctx_switches()` | `voluntary + involuntary` summed over process tree. Per-process high-water marks keyed by `(pid, create_time)`. Falls back to `/proc/<pid>/status` for the root PID. |

---

## 5. Host Memory Bandwidth (per-sample)

**Tool:** Linux `perf stat` (requires `linux-tools-common` or equivalent).
**Privilege:** `sudo sysctl -w kernel.perf_event_paranoid=-1` (otherwise
reads 0 without error).

The `HostMemoryBandwidthCollector` is a module-level singleton daemon thread
that runs `perf stat -a -e <events> -- sleep <interval>` on a loop. Each
sample attaches the latest reading via `attach_host_memory_bandwidth()`.

### Backend Detection Chain (in priority order)

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

## 6. Micro-Architecture PMU Metrics (per-sample)

**Tool:** Linux `perf stat` (requires `linux-tools-common` or equivalent).
**Privilege:** `sudo sysctl -w kernel.perf_event_paranoid=-1`.

The `MicroArchCollector` is a module-level singleton daemon thread that
alternates between event groups to avoid PMU counter multiplexing. Each
`perf stat` call samples one group for the sampling interval, then the
collector rotates to the next group.

### Event Group Design

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

### Platform Detection & Event Codes

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

### Perf Scoping

The sampler attempts three scoping modes in priority order:

| Priority | Scope | Perf Args | Requirement |
|---|---|---|---|
| 1 | `cgroup` | `--cgroup <path>` | `perf` compiled with cgroup support; cgroup v2 path available |
| 2 | `process` | `-p <container_pid>` | Container init PID known |
| 3 | `system_wide` | `-a` | Always available |

Cgroup-scoped sampling isolates PMU counts to the container's cgroup,
excluding noise from other workloads on the host.

### Derived Metrics

| Metric | Formula | Available On |
|---|---|---|
| **IPC** | `instructions / cycles` | All platforms |
| **Instructions/s** | `instructions / interval_s` | All platforms |
| **L1D Hit Rate** | `1.0 - l1d_miss / l1d_access` | `armv8-raw`, `x86-intel`, `x86-amd` (L1-level); `generic` (LLC-level) |
| **L1I Hit Rate** | `1.0 - l1i_miss / l1i_access` | `armv8-raw`, `x86-intel`, `x86-amd` |
| **L1I Hit Rate (fallback)** | `1.0 - l1i_miss / instructions` | When L1I access event is unsupported (conservative proxy) |
| **Branch Miss Rate** | `branch_miss / branch_inst` | All platforms |
| **Bus Access/s** | `bus_access / interval_s` | `armv8-raw` only (memory traffic proxy) |

### Perf Output Parsing

`perf stat -x,` produces CSV output where the third field is the canonical
event name. The parser matches on this field exactly (not substring scan)
to avoid false matches on description substrings like `insn per cycle`.
Events reporting `<not counted>` or `<not supported>` are silently skipped
so remaining metrics in the group are still computed.

### Failure Modes

| Reason | Cause |
|---|---|
| `perf_missing` | `perf` executable not found in `$PATH` |
| `permission_denied` | `kernel.perf_event_paranoid` > 0 and not running as root |
| `pmu_unsupported` | No suitable PMU detected (e.g. virtualised environment without PMU passthrough) |
| `unsupported_platform` | Non-Linux OS (macOS, Windows) |
| `cgroup_unsupported` | `perf` binary lacks `--cgroup` support |

---

## 7. Metric Aggregation (Summary)

Each sampler's `stop()` method returns raw sample lists. The
`summarize_samples()` function in `container_stats_sampler.py` computes
per-metric `{min, max, avg}` across all samples, plus `delta` (last − first)
for cumulative counters (disk I/O, network I/O, context switches).

The summary is written to `resources.json` beside the trace file, and is
consumed by the HTML Gantt viewer for resource overlay charts.

---

## 8. Architecture Support Matrix

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

## 9. HTML Visualization

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
