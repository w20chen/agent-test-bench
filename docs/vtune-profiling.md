# Per-Tool Profiling (VTune / Ksys)

> **Added**: `feat/hyf` branch, ported to `main` on 2026-06-27.
> **Renamed**: `--vtune` → `--tool-profiling vtune` on 2026-06-29.
> **Scope**: Container-mode benchmarks only (`runtime_mode = task_container_agent`).

## Overview

The `--tool-profiling` flag wraps each matching tool invocation (e.g.
`pytest`) inside a benchmark task container with a platform-specific
hardware profiler.  Currently supported backends:

| `--tool-profiling` | Hardware | Profiler | Data Source |
|---|---|---|---|
| `vtune` | Intel x86_64 | Intel VTune Profiler (uarch-exploration) | VTune PMU + in-container `/proc` sampler |
| `ksys` | Huawei Kunpeng ARM64 | Huawei ksys | (not yet implemented) |
| `off` | — | — | No per-tool profiling (default) |

When active, each profiled tool invocation produces **three layers** of
per-invocation metrics:

| Output File | Contents | Data Source |
|---|---|---|
| `summary.json` | Command, wall-clock duration, return code, sample count, coarse source label | `window.json` metadata |
| `coarse.json` | CPU%, memory, disk I/O, network, context switches | In-container `/proc` proc-tree sampler (`per_tool_samples.jsonl`) — accurate even with concurrent invocations |
| `fine.json` | VTune hotspots summary — CPI, instructions, clockticks, branch mispredict, cache metrics | VTune `-report summary` CSV parsing |

Results land under `<attempt_dir>/vtune/pytest_<timestamp>_<microsecond>_<pid>/`.

## Quick Start

```bash
# Profile every pytest invocation with VTune (coarse + fine always on)
PYTHONPATH=src python3 -m trace_collect.cli collect \
  --scaffold openclaw \
  --benchmark swe-bench-verified \
  --container docker \
  --sample 1 \
  --mcp-config none \
  --provider deepseek \
  --model deepseek-v4-flash \
  --tool-profiling vtune

# Profile multiple tool types
PYTHONPATH=src python3 -m trace_collect.cli collect \
  ... \
  --tool-profiling vtune \
  --tool-profiling-tools exec-pytest,exec-make

# Future: Kunpeng per-tool profiling
# --tool-profiling ksys
```

## Installation & Setup

### Step 1 — Check Your CPU

```bash
# Must be Intel (GenuineIntel), not AMD or ARM.
grep -m1 'vendor_id' /proc/cpuinfo
# Expected: GenuineIntel

# Must be x86_64, not aarch64.
uname -m
# Expected: x86_64
```

### Step 2 — Install Intel oneAPI Base Toolkit

The `vtune` binary ships inside the **Intel oneAPI Base Toolkit**. Pick one
of the following methods:

#### Option A: System Package Manager (recommended for persistent servers)

```bash
# Ubuntu / Debian
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor | sudo tee /usr/share/keyrings/intel-oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/intel-oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install intel-basekit
```

```bash
# RHEL / CentOS / Fedora
sudo tee /etc/yum.repos.d/oneAPI.repo << EOF
[oneAPI]
name=Intel oneAPI repository
baseurl=https://yum.repos.intel.com/oneapi
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
EOF
sudo yum install intel-basekit
```

#### Option B: Offline / Custom-Path Install

Download the installer from
<https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html>,
then:

```bash
chmod +x l_BaseKit_p_*.sh
sudo ./l_BaseKit_p_*.sh -a --silent --eula accept --install-dir /opt/intel/oneapi
```

> **Minimum install**: only the "Intel VTune Profiler" component is needed
> (~2 GB). You can deselect everything else in the GUI installer.

#### Option C: pip (lightweight, only CLI)

```bash
pip install vtune
```

This installs a minimal `vtune` CLI. However, the full `.so`-based collectors
and kernel drivers are only available in the system packages above. The pip
variant may have limited collection capabilities.

### Step 3 — Verify Installation

```bash
# Check if vtune binary exists
ls -la /opt/intel/oneapi/vtune/latest/bin64/vtune

# Source the environment (must be done in every new shell, or add to ~/.bashrc)
source /opt/intel/oneapi/setvars.sh --force
which vtune
vtune --version
```

### Step 4 — Set Environment Variables

The collection code resolves VTune via two environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `VTUNE_BIN` | auto-detect via `which vtune` | Absolute path to the `vtune` binary |
| `VTUNE_ROOT` | `/opt/intel/oneapi` | Root directory bind-mounted into containers |

If you installed to a non-default location, you **must** set both:

```bash
export VTUNE_BIN=/opt/intel/oneapi/vtune/latest/bin64/vtune
export VTUNE_ROOT=/opt/intel/oneapi
```

To persist across sessions, add these to your `~/.bashrc` (or `~/.zshrc`):

```bash
# Intel VTune
source /opt/intel/oneapi/setvars.sh --force 2>/dev/null
export VTUNE_BIN=/opt/intel/oneapi/vtune/latest/bin64/vtune
export VTUNE_ROOT=/opt/intel/oneapi
```

Or, if you prefer to avoid sourcing the heavy `setvars.sh` every shell start:

```bash
export VTUNE_BIN=/opt/intel/oneapi/vtune/latest/bin64/vtune
export VTUNE_ROOT=/opt/intel/oneapi
export PATH="$VTUNE_BIN:$PATH"
```

### Step 5 — Verify the Host → Container Path

VTune is bind-mounted into containers at the **same absolute path** as on the
host. This means:

- Host: `/opt/intel/oneapi/` → Container: `/opt/intel/oneapi/` (read-only)
- `VTUNE_BIN` must resolve to the same path in both environments

Quick sanity check:

```bash
# Should print the same path inside and outside:
docker run --rm -v /opt/intel/oneapi:/opt/intel/oneapi:ro \
  alpine ls /opt/intel/oneapi/vtune/latest/bin64/vtune
```

### Step 6 — Grant PMU Access (One-Time)

VTune needs access to the CPU Performance Monitoring Unit. The collector
automatically adds `--cap-add PERFMON` and `--cap-add SYS_ADMIN` to the
Docker run command. However, the **host kernel** may restrict this further.

Check and fix:

```bash
# Check current value (0 = unrestricted, 1 = restricted, 2 = only pid 0)
cat /proc/sys/kernel/perf_event_paranoid

# If it's 2 or higher, relax it temporarily:
sudo sysctl -w kernel.perf_event_paranoid=0

# To persist across reboots:
echo "kernel.perf_event_paranoid=0" | sudo tee -a /etc/sysctl.d/99-vtune.conf
```

If you see `permission denied` errors in VTune output, this is the fix.

### Summary Checklist

- [ ] Intel x86_64 CPU
- [ ] `intel-basekit` (or equivalent) installed
- [ ] `vtune` binary at `/opt/intel/oneapi/vtune/latest/bin64/vtune`
- [ ] `VTUNE_BIN` and `VTUNE_ROOT` set in `~/.bashrc`
- [ ] `kernel.perf_event_paranoid = 0`
- [ ] VTune path resolves inside a test container

## Prerequisites

1. **Intel x86 native** — no QEMU emulation (VTune relies on hardware PMU
   counters).  For Kunpeng, use `--tool-profiling ksys` (future).

2. **Container-mode benchmark** — only benchmarks with
   `runtime_mode = task_container_agent` are supported.  Attempting
   `--tool-profiling vtune` with `runtime_mode = host_controller` raises a
   clear error.

## Architecture

```
┌─ Host ───────────────────────────────────────────────────────────┐
│                                                                  │
│  CLI (--tool-profiling vtune)                                    │
│    │                                                             │
│    ▼                                                             │
│  collector.py                                                    │
│    │  if tool_profiling == "vtune":                              │
│    │    vtune_report.vtune_container_run_args()                  │
│    │    → mount VTune into container (read-only)                 │
│    │    → grant CAP_PERFMON, CAP_SYS_ADMIN                       │
│    │    → set VTUNE_PROFILE=1 env var                            │
│    │    → disable host-side PMU (perf stat) to avoid counter     │
│    │      contention with VTune                                  │
│    │                                                             │
│    │  After agent finishes:                                      │
│    │    vtune_report.finalize_vtune()                            │
│    │    → read per_tool_samples.jsonl (in-container /proc)       │
│    │    → fall back to ContainerStatsSampler samples if absent   │
│    │    → emit summary.json + coarse.json + fine.json            │
│    │                                                             │
├─────── Container boundary ───────────────────────────────────────┤
│    │                                                             │
│    ▼                                                             │
│  shell.py (ExecTool)                                             │
│    │  if VTUNE_PROFILE=1 and tool matches VTUNE_TOOLS:           │
│    │    wrap: vtune -collect hotspots -- bash ...              │
│    │    start per-tool /proc sampler thread (0.5 s interval)     │
│    │    record ts_start / ts_end / returncode → window.json      │
│    │                                                             │
│    ▼                                                             │
│  Per-invocation output directory:                                │
│    pytest_<ts>_<us>_<pid>/                                       │
│      ├── window.json               timing + exit code            │
│      ├── result/                    VTune raw PMU data           │
│      └── per_tool_samples.jsonl     /proc proc-tree samples      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Output Structure

```
<run_dir>/<instance_id>/<attempt>/
└── vtune/
    ├── pytest_20260627T120000_123456_42/
    │   ├── window.json               {"cmd": "...", "ts_start": ..., "ts_end": ..., "returncode": 0}
    │   ├── result/                    VTune raw data (for vtune -report)
    │   ├── per_tool_samples.jsonl     In-container /proc proc-tree samples
    │   ├── summary.json               Per-invocation summary + coarse_source label
    │   ├── coarse.json                CPU/mem/disk/net/ctx per invocation
    │   └── fine.json                  VTune TMA + perf counters
    └── pytest_20260627T120142_654321_42/
        └── ...
```

`coarse.json` metrics are sourced from the in-container `/proc` proc-tree
sampler (`per_tool_samples.jsonl`) when available, which gives accurate
per-process-tree data even when multiple pytest invocations overlap in time.
When the file is absent (e.g. traces collected before this feature), the
host-side `ContainerStatsSampler` samples are time-sliced as a fallback.
The `coarse_source` field in `summary.json` records which source was used
(`"per_tool_proc"` or `"container_cgroup"`).

## Design Decisions

- **Lazy imports**: `vtune_report` is only imported when
  `--tool-profiling vtune` is active.  Zero overhead otherwise.
- **Opt-in by default**: `--tool-profiling` defaults to `off`.
  Existing workflows are completely unaffected.
- **hotspots mode**: VTune ``hotspots`` collection uses ~6 basic PMU events
  (cycles, instructions, branches, cache) that fit within the per-core PMU
  counter budget on any core count, avoiding the multi-core event distribution
  issue that plagues ``uarch-exploration`` mode.
- **Per-tool-window slicing**: Rather than profiling the entire agent run,
  each matching tool invocation is profiled independently.
- **In-container `/proc` sampler**: CPU, memory, disk I/O, and context
  switches are sampled from `/proc/<pid>` inside the container, scoped to
  the exact pytest process tree.  This avoids cgroup-level contamination
  when multiple tool invocations overlap.
- **Classifier-based matching**: Exec tool detection reuses the same
  `exec_classifier` logic that produces trace tool names (``exec-pytest``,
  ``exec-pip``, etc.).  ``--tool-profiling-tools`` accepts any tool name
  (exec or non-exec) for forward compatibility.
- **PMU conflict avoidance**: When `--tool-profiling vtune` is active, the
  host-side `--pmu-monitoring` is automatically disabled to prevent PMU
  counter contention between the host `perf stat` and in-container VTune.
- **No synthetic/mock data**: VTune runs against the real containerized
  tool execution.  If VTune is not installed, the feature fails early with
  a clear error message.

## Analysis

Aggregated analysis across many pytest invocations is provided by
`scripts/analyze_vtune_aggregate.py`:

```bash
python scripts/analyze_vtune_aggregate.py --input traces/my_sweep/
# Produces: vtune_aggregate.csv, vtune_aggregate_summary.txt, vtune_aggregate_topn.txt
# Add --plot for KDE distribution charts (requires matplotlib + scipy)
```

## Limitations

- Intel x86 native only (no ARM, no QEMU) for `--tool-profiling vtune`.
- Container-mode benchmarks only.
- Adds modest overhead: VTune hotspots collects PMU counters during
  the pytest run.  Expect 2–5% runtime increase per profiled test
  (significantly lower than uarch-exploration's 5-15%).
- The in-container `/proc` sampler adds negligible overhead (~60 syscalls
  per 0.5 s interval for a typical pytest process tree).
