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

```bash
# Basic: wrap every in-container pytest with VTune
python -m trace_collect.cli collect \
  --scaffold openclaw \
  --benchmark swe_bench_verified \
  --container docker \
  --vtune

# With coarse system metrics per test window
python -m trace_collect.cli collect \
  --scaffold openclaw \
  --benchmark swe_bench_verified \
  --container docker \
  --vtune --vtune-coarse

# With full TMA breakdown
python -m trace_collect.cli collect \
  --scaffold openclaw \
  --benchmark swe_bench_verified \
  --container docker \
  --vtune --vtune-coarse --vtune-fine
```

## Prerequisites

1. **Intel VTune installed** on the host. Set `VTUNE_BIN` to the `vtune`
   binary path, or source the oneAPI `setvars.sh` so `vtune` is on `PATH`.
   Default `VTUNE_ROOT` is `/opt/intel/oneapi`.

2. **Intel x86 native** — no QEMU emulation (VTune relies on hardware PMU
   counters).

3. **Container-mode benchmark** — only benchmarks with
   `runtime_mode = task_container_agent` are supported. Attempting `--vtune`
   with `runtime_mode = host_controller` raises a clear error.

## Architecture

```
┌─ Host ──────────────────────────────────────────────────────┐
│                                                              │
│  CLI (--vtune)                                               │
│    │                                                         │
│    ▼                                                         │
│  collector.py                                                │
│    │  if vtune:                                              │
│    │    vtune_report.vtune_container_run_args()              │
│    │    → mount VTune into container (read-only)            │
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
