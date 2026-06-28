# VTune Profiling for In-Container Pytest

> **Added**: `feat/hyf` branch, ported to `main` on 2026-06-27.
> **Scope**: Container-mode benchmarks only (`runtime_mode = task_container_agent`).

## Overview

The `--vtune` feature wraps each `pytest` invocation inside a benchmark task
container with **Intel VTune Profiler** (uarch-exploration), producing
per-test-window performance reports. Combined with the built-in
`ContainerStatsSampler`, it yields three layers of metric granularity:

| Output File | Contents |
|---|---|
| `summary.json` | Command, wall-clock duration, return code, sample count |
| `coarse.json` | CPU%, memory, disk I/O, network, context switches |
| `fine.json` | VTune Top-down Microarchitecture Analysis (TMA) + perf IPC, L1I hit rate, branch miss rate |

Results land under `<attempt_dir>/vtune/pytest_<timestamp>_<pid>/`.

## Quick Start

Basic: wrap every in-container pytest with VTune
```bash
PYTHONPATH=src python3 -m trace_collect.cli collect \
  --scaffold openclaw \
  --benchmark swe-bench-verified \
  --container docker \
  --sample 1 \
  --mcp-config none \
  --provider deepseek \
  --model deepseek-v4-flash \
  --vtune
```

With coarse system metrics per test window
```bash
PYTHONPATH=src python3 -m trace_collect.cli collect \
  --scaffold openclaw \
  --benchmark swe-bench-verified \
  --container docker \
  --sample 1 \
  --mcp-config none \
  --provider deepseek \
  --model deepseek-v4-flash \
  --vtune --vtune-coarse
```

With full TMA breakdown
```bash
PYTHONPATH=src python3 -m trace_collect.cli collect \
  --scaffold openclaw \
  --benchmark swe-bench-verified \
  --container docker \
  --sample 1 \
  --mcp-config none \
  --provider deepseek \
  --model deepseek-v4-flash \
  --vtune --vtune-coarse --vtune-fine
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
   counters).

2. **Container-mode benchmark** — only benchmarks with
   `runtime_mode = task_container_agent` are supported. Attempting `--vtune`
   with `runtime_mode = host_controller` raises a clear error.

## Architecture

```
┌─ Host ───────────────────────────────────────────────────────┐
│                                                              │
│  CLI (--vtune)                                               │
│    │                                                         │
│    ▼                                                         │
│  collector.py                                                │
│    │  if vtune:                                              │
│    │    vtune_report.vtune_container_run_args()              │
│    │    → mount VTune into container (read-only)             │
│    │    → grant CAP_PERFMON, CAP_SYS_ADMIN                   │
│    │    → set VTUNE_PROFILE=1 env var                        │
│    │                                                         │
│    │  After agent finishes:                                  │
│    │    vtune_report.finalize_vtune()                        │
│    │    → slice ContainerStatsSampler samples per test       │
│    │    → emit summary.json / coarse.json / fine.json        │
│    │                                                         │
├─────── Container boundary ───────────────────────────────────┤
│    │                                                         │
│    ▼                                                         │
│  shell.py (ExecTool)                                         │
│    │  if VTUNE_PROFILE=1 and command contains "pytest":      │
│    │    wrap: vtune -collect uarch-exploration -- bash ...   │
│    │    record ts_start / ts_end / returncode → window.json  │
│    │                                                         │
│    ▼                                                         │
│  VTune result dir (per pytest invocation)                    │
│    + window.json                                             │
│    + result/          ← raw VTune data                       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Output Structure

```
<run_dir>/<instance_id>/<attempt>/
└── vtune/
    ├── pytest_20260627T120000_42/
    │   ├── window.json       {"cmd": "...", "ts_start": ..., "ts_end": ..., "returncode": 0}
    │   ├── result/            VTune raw data (for -report)
    │   ├── summary.json       Per-test summary
    │   ├── coarse.json        System metrics (if --vtune-coarse)
    │   └── fine.json          TMA + perf counters (if --vtune-fine)
    └── pytest_20260627T120142_57/
        └── ...
```

## Design Decisions

- **Lazy imports**: `vtune_report` is only imported when `--vtune` is active.
  Zero overhead for non-VTune runs.
- **Opt-in by default**: `--vtune` defaults to `False`. Existing workflows
  are completely unaffected.
- **No synthetic/mock data**: VTune runs against the real containerized
  pytest. If VTune is not installed, the feature fails early with a clear
  error message rather than silently producing empty results.
- **Per-test-window slicing**: Rather than profiling the entire agent run,
  each pytest invocation is profiled independently. The host-side
  `finalize_vtune` function correlates `ContainerStatsSampler` samples
  with each test's `window.json` to produce per-test metrics.

## Limitations

- Intel x86 native only (no ARM, no QEMU).
- Container-mode benchmarks only.
- Adds modest overhead: VTune uarch-exploration collects PMU counters during
  the pytest run. Expect 5–15% runtime increase per profiled test.
