# Current Plan: Kunpeng QEMU LLC Placement Experiment

## Goal

Design and implement a small, reproducible script layer for the key experiment:
8 concurrent replayed OpenClaw agents on a Kunpeng ARM host in QEMU mode,
constraining Docker containers to either one LLC's CPU set or a CPU set spread
across LLCs, and comparing both against OS default placement.

The experiment should use an existing real trace, preserve real tool execution
through `trace_collect.cli simulate --mode cloud_model`, avoid live LLM API
calls, and collect enough topology/performance evidence to explain performance
changes through shared LLC / memory-bandwidth contention.

## Research Integrity Guardrails

- Do not introduce synthetic/mocked agent results as final evidence.
- Do not tune the selected task for a favorable result. The selected trace must
  be documented as a fixed representative workload before running the placement
  comparison.
- Do not add per-benchmark CLI flags to trace collection. Benchmark settings
  stay in benchmark YAML/plugin code.
- Do not silently reduce concurrency, sample count, iterations, tool work, or
  QEMU usage because a run is slow.
- QEMU mode is explicit: use `ARM_IMAGE_MODE=qemu`, after validating binfmt.
- Replay experiments must not pass provider/model/API-key arguments and must
  use `cloud_model`, which replays source LLM timing without issuing requests.
- Scripts must record the exact core lists, topology snapshot, environment, and
  command line for every run.

## Proposed Script Set

### 1. Topology Probe

Path: `scripts/experiments/probe_llc_topology.py`

Responsibilities:

- Parse Linux topology from:
  - `/sys/devices/system/cpu/cpu*/cache/index*/`
  - `/sys/devices/system/cpu/cpu*/topology/`
  - `/sys/devices/system/cpu/cpu*/node*`
- Identify the last-level cache index per CPU by highest cache level.
- Emit machine-readable topology:
  - `cpu`
  - `core_id`
  - `socket_id` / physical package id
  - `numa_node`
  - `llc_id`
  - `llc_level`
  - `llc_shared_cpu_list`
- Generate placement candidates:
  - `same_llc_8`: 8 CPUs from one LLC sharing group.
  - `spread_llc_8`: 8 CPUs round-robin across distinct LLC groups where possible.
  - `os_default_8`: no affinity list.
- Fail clearly if the host cannot provide a valid 8-core same-LLC or spread
  placement. Do not silently substitute fewer cores.

Outputs:

- `topology.json`
- `placements.json`
- `topology.txt`

### 2. Live Placement Runner

Path: `scripts/experiments/run_kunpeng_llc_agent_case.py`

Responsibilities:

- Validate QEMU support with existing `scripts/setup/arm_setup.sh status/check`
  guidance, but do not install anything automatically.
- Run one fixed real benchmark task through existing `trace_collect.cli collect`.
- Use `swe-rebench + openclaw` as the default because this repo already has
  ARM/QEMU image handling for that path.
- Launch three placement conditions:
  - `os_default`
  - `same_llc`
  - `spread_llc`
- For each condition:
  - set `ARM_IMAGE_MODE=qemu`
  - set `PYTHONPATH=src`
  - run `--benchmark swe-rebench --scaffold openclaw`
  - run `--concurrency 8`
  - run `--instance-ids <fixed_case>`
  - set `--resource-monitoring on`
  - keep `--pmu-monitoring off` because current concurrent collection forbids
    PMU per attempt
  - optionally set `--ksys-monitoring on` only if requested by user/config
  - apply process-level affinity using `taskset -c <core-list>` for same/spread
    placement
- Record the full command and environment in `run_config.json`.
- Do not change benchmark plugin architecture or add benchmark-specific flags.

Important limitation:

- `taskset` on the parent process constrains the whole concurrent run to the
  chosen 8 CPUs. It does not by itself guarantee one agent process per named
  core. This is still a valid placement comparison at the CPU-set level, but if
  we need strict one-agent-one-core binding, we need a deeper change in the
  concurrent scaffold launcher to assign per-attempt affinity. That touches the
  evaluation pipeline and requires the mandatory reviewer gate.

This live runner is retained as a convenience path, but it is not the preferred
LLC experiment because it calls live provider APIs and `taskset` does not
strictly constrain Docker containers started by the daemon.

### 2b. Replay Placement Runner

Path: `scripts/experiments/run_kunpeng_llc_replay.py`

Responsibilities:

- Replay one fixed existing trace as `N=8` identical cloud-model agents.
- Use:
  - `trace_collect.cli simulate`
  - `--mode cloud_model`
  - `--source-trace <trace.jsonl>`
  - `--num-agents 8`
  - `--trace-assignment manifest`
  - `--arrival-mode closed_loop`
  - `--container docker`
  - `--network-mode none` by default
  - `--cpu-limit 1` per container
- Apply Docker-level placement via the new generic simulate flag
  `--cpuset-cpus <cpu-list>`, not parent-process `taskset`.
- Record full commands and selected CPU/LLC lists in `run_config.json`.
- Avoid live provider/API-key arguments entirely.

Recommended source trace:

- First choice: `django__django-10880` from the existing case-study traces.
- Rationale: roughly 58s recorded tool time, with about 54s in `exec-pytest`;
  this makes the replay CPU/QEMU/tool-heavy and less dominated by LLM/API
  latency.
- Fallback: any existing SWE/SWE-rebench trace with high `exec-pytest` or
  `exec-python` tool time, selected before looking at placement results.

### 3. Perf / Ksys Wrapper

Path: `scripts/experiments/run_with_perf_stat.sh`

Responsibilities:

- Wrap each placement run with system-level `perf stat` events that work on
  ARM/Kunpeng:
  - `cycles`
  - `instructions`
  - `cache-references`
  - `cache-misses`
  - ARM raw events where supported through the existing harness docs:
    `r04,r03,r14,r01,r12,r10,r19`
- Write raw perf output per condition.
- If `perf` permissions are unavailable, fail with a diagnostic instead of
  producing partial results as if they were valid.

Alternative:

- When Kunpeng `ksys` is available and explicitly requested, use existing
  `--ksys-monitoring on` rather than inventing a separate collection path.

### 4. Result Summarizer

Path: `scripts/experiments/summarize_llc_placement_runs.py`

Responsibilities:

- Read the trace outputs and resource summaries from each condition.
- Preserve all intermediate trace fields; derive summaries in a separate file.
- Compute:
  - agent completion mean / p50 / p95
  - tool latency mean / p50 / p95
  - tool count per agent
  - LLM time and tool time if trace events provide enough phase data
  - resource sample availability and memory-bandwidth fields when present
  - perf LLC/cache counters when available
- Emit:
  - `summary.csv`
  - `summary.json`
  - `README.md` describing methodology and caveats

## Default Representative Case

Preferred default:

- Benchmark: `swe-rebench`
- Scaffold: `openclaw`
- Case selection: first explicitly available instance from existing local
  `data/swe-rebench/tasks.json`, unless the user provides a specific
  `--instance-id`.

Rationale:

- It is a real agent/code-editing workload with containerized tools.
- It exercises QEMU path on Kunpeng through existing repo support.
- It avoids synthetic final results.
- It keeps benchmark-specific details inside the existing plugin/YAML.

Open question for user:

- If the mentor expects the exact same task as previous trace-case slides, use
  that fixed instance ID instead of the first local SWE-rebench task.

## Implementation Checkpoints

- [x] Read current project instructions and existing ARM/QEMU/profiling code.
- [x] Persist this plan to disk.
- [x] Human approval of script scope: CPU-set placement is sufficient.
- [x] Implement topology probe and live placement runner.
- [x] Implement replay placement runner with Docker `--cpuset-cpus`.
- [x] Implement perf wrapper and summarizer.
- [x] Add focused tests for topology parsing / placement selection using
      temporary sysfs-like fixtures.
- [x] Run focused tests.
- [ ] If strict one-agent-one-core binding is requested, implement in the
      evaluation/concurrent launcher only after mandatory independent review.

## Commands The Runner Should Produce

Preferred replay same-LLC run shape:

```bash
ARM_IMAGE_MODE=qemu PYTHONPATH=src:. python -m trace_collect.cli simulate \
  --source-trace <trace.jsonl> \
  --task-source data/swe-rebench/tasks.json \
  --output-dir traces/experiments/kunpeng_llc_replay/<timestamp>/same_llc \
  --mode cloud_model \
  --container docker \
  --network-mode none \
  --num-agents 8 \
  --trace-assignment manifest \
  --arrival-mode closed_loop \
  --replay-speed 1 \
  --cpu-limit 1 \
  --cpuset-cpus 0,1,2,3,4,5,6,7 \
  --resource-monitoring on \
  --pmu-monitoring off \
  --ksys-monitoring off
```

Preferred replay spread-LLC run shape:

```bash
ARM_IMAGE_MODE=qemu PYTHONPATH=src:. python -m trace_collect.cli simulate \
  --source-trace <trace.jsonl> \
  --task-source data/swe-rebench/tasks.json \
  --output-dir traces/experiments/kunpeng_llc_replay/<timestamp>/spread_llc \
  --mode cloud_model \
  --container docker \
  --network-mode none \
  --num-agents 8 \
  --trace-assignment manifest \
  --arrival-mode closed_loop \
  --replay-speed 1 \
  --cpu-limit 1 \
  --cpuset-cpus 0,80,160,240,1,81,161,241 \
  --resource-monitoring on \
  --pmu-monitoring off \
  --ksys-monitoring off
```

The actual CPU lists must come from the topology probe, not from hardcoded
numbers.
