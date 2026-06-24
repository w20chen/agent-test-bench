# Current Plan: Resource monitoring controls

## Goal

Audit every resource-monitoring path, document when it is enabled or disabled,
and add explicit controls without changing existing experiment defaults
silently.

## Read-only audit findings

### Trace collection (`trace_collect.cli` collect path)

- Serial collection (`--concurrency 1`, the default):
  - Container attempts start `ContainerStatsSampler` by default.
  - Host attempts start `ProcessStatsSampler` by default.
  - Both samplers collect CPU, RSS memory, disk I/O, network I/O, and context
    switches.
  - Both samplers also call `attach_host_memory_bandwidth()` and
    `attach_micro_arch()`, which lazily start module-level `perf` collectors.
    These two metric families are therefore implicitly enabled whenever the
    base sampler is enabled, although unsupported hardware/permissions produce
    unavailable markers rather than samples.
- Concurrent collection (`--concurrency > 1`):
  - `collector._run_scaffold_tasks()` hardcodes
    `disable_resource_monitoring=True` for every attempt.
  - Container/process sampling, host memory bandwidth, and PMU sampling are all
    skipped.
  - `--ksys`, if explicitly passed, runs once for the whole batch.
- `--ksys` is an independent opt-in flag and defaults to off.
- There is no user-facing switch for the built-in sampler stack.

### Simulation (`trace_collect.cli simulate`)

- Container replay always starts `ContainerStatsSampler`; there is no switch.
- Host replay writes an empty `resources.json` and does not start
  `ProcessStatsSampler`.
- vLLM scheduler/Prometheus snapshots are collected when `--metrics-url` is
  provided; otherwise explicit empty snapshots are recorded.
- GPU memory tracking has its own `--gpu-tracking on|off` switch and defaults
  to off. It is supported only for `local_model` mode and requires the vLLM
  metrics URL, PID, and startup log.

### Dedicated GPU profiling

- `profile-gpu` is a dedicated command whose purpose is component-level GPU
  memory profiling. Running the command always attaches attention/MLP forward
  hooks.
- It is separate from ordinary collection and simulation resource sampling.

### Standalone OpenClaw

- The standalone `python -m agents.openclaw` path does not use the harness
  resource samplers.

## Correctness issue found during audit

`run_attempt()` currently writes the same
`{"monitoring_disabled": true, "sample_count": 0}` summary both when monitoring
was explicitly disabled and when monitoring was enabled but yielded no sample.
That conflates policy with collection failure/unavailability and should be
corrected as part of the control work.

## Human-confirmed requirements

- Provide three independent switches:
  1. PMU micro-architecture monitoring.
  2. Other built-in project resource monitoring.
  3. ksys monitoring.
- Keep GPU monitoring unchanged and outside this work.
- PMU monitoring must be forbidden whenever execution is concurrent, for both
  real collection and simulation. An explicit request to enable PMU in a
  concurrent run must fail before work starts; it must never silently run.
- Host memory-bandwidth monitoring must also be forbidden whenever execution
  is concurrent. It uses a system-wide `perf` singleton and cannot produce
  correctly isolated per-attempt concurrent measurements.
- ksys controls must be supported by both collection and simulation.
- Preserve the legacy `--ksys` flag as a compatibility alias for
  `--ksys-monitoring on`.
- Simulation concurrency means the default multi-trace `cloud_model` path
  without `--serial`. `local_model` is single-session, while cloud replay with
  `--serial` is serial.

## Approved design

1. Add independent tri-state switches to both `collect` and `simulate`:
   - `--pmu-monitoring {auto,on,off}`
   - `--resource-monitoring {auto,on,off}`
   - `--ksys-monitoring {auto,on,off}`
2. Resolve PMU policy before starting work:
   - Concurrent execution always resolves PMU to off.
   - Explicit `on` in concurrent execution raises a clear configuration error.
   - Serial `auto` preserves the existing PMU behavior.
3. Decouple PMU attachment from `ContainerStatsSampler` and
   `ProcessStatsSampler`, so base resource sampling can run without starting
   the module-level PMU singleton.
4. Keep GPU behavior and `--gpu-tracking` unchanged.
5. Record requested and resolved monitoring policy in trace/run metadata and
   `resources.json`, distinguishing:
   - disabled by policy;
   - enabled with samples;
   - enabled but unavailable/no samples.
6. Update CLI help and `docs/trace-collect.md` /
   `docs/resource-measurement.md` with a scenario/default matrix.

### Exact `auto` semantics

- PMU:
  - Serial container/host collection: on when base resource monitoring is on.
  - Serial container simulation: on when base resource monitoring is on.
  - Every concurrent mode and host simulation: off.
- Host memory bandwidth:
  - On with base resource monitoring only in serial collection/container
    simulation.
  - Always off in concurrent modes and host simulation.
- Other built-in resources:
  - Collection: on in serial mode, off in concurrent mode.
  - Simulation: on for container sessions, off for host sessions.
- ksys: off everywhere unless explicitly enabled.
- Concurrent container collection may explicitly enable other built-in
  resources, while PMU and host memory bandwidth remain prohibited.
- Explicit base resource monitoring in host simulation is rejected because
  there is no isolated agent PID.
- Explicit PMU monitoring requires base resource monitoring.

## Planned implementation after approval

1. Introduce a small typed three-channel monitoring-policy resolver shared by
   CLI and runtime code.
2. Thread the resolved policies through collection and simulation without
   benchmark-specific branches.
3. Make PMU attachment an explicit sampler option and enforce the concurrency
   prohibition before any attempt/session starts.
4. Fix resource artifact status metadata and preserve the existing sample
   schema.
5. Add CLI, serial/container, serial/host, concurrent, disabled, and no-sample
   regression tests.
6. Update documentation and changelog if present.
7. Run focused tests, then the relevant trace-collection test group.
8. Spawn a fresh strict reviewer sub-agent because this touches the evaluation
   pipeline; fix all findings and record the review audit here.

## Checkpoints

- [x] Read-only source, test, CLI, and documentation audit
- [x] Persist findings and proposed design
- [x] Human confirmation of switch scope and semantics
- [x] Implementation
- [x] Mandatory independent review (2026-06-24)
- [ ] Focused and regression verification
- [ ] Final diff and scope audit

---

## Implementation Summary (2026-06-24)

### Completed

1. **`src/trace_collect/monitoring.py`** — `MonitoringPolicy` frozen dataclass
   with three-channel resolver functions (`resolve_collect_monitoring`,
   `resolve_simulate_monitoring`, `resolve_ksys_request`).

2. **`src/trace_collect/cli.py`** — CLI switches `--resource-monitoring`,
   `--pmu-monitoring`, `--ksys-monitoring` (all `auto|on|off`) added to both
   `collect` and `simulate` parsers via `_add_monitoring_arguments()`. Legacy
   `--ksys` preserved as compatibility alias.

3. **`src/trace_collect/simulator.py`** — Policy threaded through `simulate()`
   → `_setup_one()` → `ContainerStatsSampler(enable_pmu=..., enable_memory_bandwidth=...)`
   → `_teardown_one()` writing `resources.json` with three-way status
   (`"collected"` / `"enabled_no_samples"` / `"disabled"`).

4. **`src/trace_collect/collector.py`** — Policy resolved in `collect_traces()`
   and threaded to `_run_scaffold_tasks()` → `run_attempt()`. Concurrent path
   passes `enable_ksys=False` per-attempt with one batch-level ksys process.

5. **`src/trace_collect/attempt_pipeline.py`** — `run_attempt()` accepts
   `disable_resource_monitoring`, `enable_pmu_monitoring`,
   `enable_memory_bandwidth_monitoring` parameters and propagates them to
   `ContainerStatsSampler`/`ProcessStatsSampler`.

6. **`src/harness/container_stats_sampler.py`** — Added `enable_pmu` and
   `enable_memory_bandwidth` parameters.

7. **`src/harness/process_stats_sampler.py`** — Added `enable_pmu` and
   `enable_memory_bandwidth` parameters.

### Bugs Fixed (Independent Review Findings)

| # | Severity | File | Issue | Fix |
|---|----------|------|-------|-----|
| 1 | 🔴 | `cli.py` `_run_simulate` | `ValueError` from `resolve_simulate_monitoring` not caught | Wrapped `asyncio.run(simulate(...))` in `try/except ValueError` |
| 2 | 🔴 | `cli.py` `_run_collect` | `enable_ksys=False` hardcoded alongside resolved `ksys_monitoring` | Removed redundant `enable_ksys=False`; `collect_traces` no longer takes `enable_ksys` |
| 3 | 🔴 | `collector.py` `collect_traces` | Redundant `resolve_ksys_request` re-resolution | Removed `enable_ksys` parameter; ksys resolved once in `_run_collect` |
| 4 | 🟠 | `_run_scaffold_tasks` + `run_attempt` | Ksys startup duplicated across 3 locations with raw `subprocess.Popen` | Replaced with `KsysSession` from `harness/ksys.py`; removed `_stop_ksys` |
| 5 | 🟠 | `attempt_pipeline.py` `AttemptContext` | Boolean defaults (True for PMU/MemBW) could silently override resolved policy | Changed defaults to safe values (all `False`) |
| 6 | 🟡 | `_run_scaffold_tasks` | Dead code fallback `if monitoring_policy is None: resolve_collect_monitoring(...)` | Made `monitoring_policy` a required parameter |
| 7 | 🟡 | `monitoring.py` | `resolve_ksys_request` error message vague | Added actual requested value to error message |

### Correctness Issue Resolved

The original audit found that `run_attempt()` wrote the same summary for
"disabled" vs "enabled_no_samples". This was fixed at both sites:

- **`attempt_pipeline.py` `run_attempt()`**: `resources.json` now uses
  `"monitoring": {"status": "collected"|"enabled_no_samples"|"disabled"}`
  plus the full `monitoring_policy` dict.

- **`simulator.py` `_teardown_one()`**: Same three-way status distinction.

### Documentation Updated

- `docs/resource-measurement.md` — Added note that host memory bandwidth has
  no independent CLI switch; it follows `resource_enabled && !concurrent`.
- `docs/trace-collect.md` — Updated CLI flag table to mention host memory
  bandwidth under `--resource-monitoring`.

### Pending

- Run focused regression tests (test_collector_*, test_monitoring_*, etc.)
- Final diff review and scope audit
- `run_manifest.json` in simulation path (awaiting user decision)

## Scope guard

- Do not change benchmark plugin behavior or benchmark-specific YAML.
- Do not add dependencies.
- Do not run experiments.
- Preserve current defaults under `auto`.
- Do not enable unsupported concurrent per-attempt PMU measurements.
