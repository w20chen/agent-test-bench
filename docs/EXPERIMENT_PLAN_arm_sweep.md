# Experiment Plan: SWE-rebench N:M Simulation Sweep on ARM (320 vCPUs)

## Objective

Simulate 40 SWE-rebench traces with N=40, 80, 160, 320 agent instances
(cyclic allocation, 1 CPU per agent) and collect:
- Per-container resource metrics (`resources.json`)
- System-level resource timeline (CPU, memory, disk, network over time)
- Agent lifecycle log (start/end wall-clock time per agent)

---

## 1. Feasibility Assessment

### What the system already supports

| Requirement | Support | Notes |
|-------------|---------|-------|
| `--source-dir` to discover 40 traces | ✅ Yes | `_discover_traces()` finds all `**/trace.jsonl` |
| `--num-agents 40/80/160/320` | ✅ Yes | `_expand_trace_inputs()` with `manifest` (cyclic) |
| `--trace-assignment manifest` | ✅ Yes | Default — cycles through input traces |
| `--cpu-limit 1` per container | ✅ Yes | `--cpus=1` passed to Docker |
| `--resource-monitoring on` | ✅ Yes | `ContainerStatsSampler` per container |
| `--pmu-monitoring off` | ✅ Yes | Explicit disable |
| `--ksys-monitoring off` | ✅ Yes | Explicit disable |
| `--mode cloud_model` concurrent | ✅ Yes | All agents replay concurrently; `--workers` distributes across independent asyncio event loops to eliminate single-loop scheduling bottleneck |
| System-level resource monitor | ❌ **Not yet** | Only per-container stats exist |
| Agent lifecycle log (start/end) | ❌ **Not yet** | Can extract from trace, but no structured log |

### Key risks

1. **Docker daemon pressure**: 320 concurrent containers may overwhelm Docker.
   Mitigation: `--prep-concurrency` uses one system-wide semaphore (auto
   preserves 20), and a global all-ready barrier delays replay until every
   container is prepared.

2. **Memory**: Each SWE container typically consumes 200-500 MB (idle).
   320 containers × 300 MB ≈ 96 GB. The ARM machine must have ≥128 GB RAM.

3. **Disk I/O**: All containers may write to disk simultaneously during
   bash command replay. Recommend using SSD/NVMe.

4. **CPU contention measurement**: Docker `--cpus=1` is a soft limit (CPU
   shares). Containers can burst above 1 CPU if the system has slack. This
   is expected and part of what we want to measure.

---

## 2. What Needs to Be Built

### 2a. System Resource Monitor (`scripts/system_resource_monitor.py`)

A standalone background process that samples system-wide metrics via
`psutil` and writes a JSONL log:

```
Fields per sample:
  ts          : float   # wall-clock timestamp (time.time())
  cpu_percent : float   # system-wide CPU utilization %
  cpu_count   : int     # logical core count
  mem_percent : float   # system memory utilization %
  mem_used_gb : float   # memory used
  mem_total_gb: float   # memory total
  disk_read_mb: float   # cumulative disk read (all disks)
  disk_write_mb: float  # cumulative disk write
  net_rx_mb   : float   # cumulative network received
  net_tx_mb   : float   # cumulative network sent
  container_count: int  # docker ps -q | wc -l
  load_1m     : float   # system load average 1min
  load_5m     : float
  load_15m    : float
```

Runs at 1 Hz. Writes to `<output_dir>/system_resources.jsonl`.

### 2b. Agent Lifecycle Extractor (`scripts/extract_agent_timeline.py`)

Post-processes per-agent `trace.jsonl` files to produce an agent lifecycle
log:

```
Fields per agent:
  agent_id     : str
  start_ts     : float   # first action ts_start
  end_ts       : float   # last action ts_end
  elapsed_s    : float
  n_actions    : int
  n_llm_calls  : int
  n_tool_execs : int
```

Writes to `<output_dir>/agent_timeline.jsonl`.

### 2c. Experiment Runner (`scripts/run_simulate_sweep.sh`)

Orchestrates the 4-config sweep:

```bash
for N in 40 80 160 320; do
    # 1. Start system monitor in background
    # 2. Run simulate command
    # 3. Stop system monitor
    # 4. Extract agent timeline
    # 5. Print summary
done
```

---

## 3. The Simulate Command

For each N, the core command is:

```bash
python -m trace_collect.cli simulate \
    --source-dir "${SOURCE_TRACES_DIR}" \
    --mode cloud_model \
    --container docker \
    --num-agents ${N} \
    --trace-assignment manifest \
    --cpu-limit 1 \
    --resource-monitoring on \
    --pmu-monitoring off \
    --ksys-monitoring off \
    --replay-speed 999999 \
    --output-dir "traces/simulate/swe-rebench/sweep_${N}a_1cpu"
```

### Parameter rationale

- `--source-dir`: Points to directory containing the 40 trace directories
- `--mode cloud_model`: Replays LLM timing from traces; re-executes tools in containers
- `--container docker`: Uses Docker for tool execution isolation
- `--num-agents N`: Creates N agents cycling through 40 traces
- `--trace-assignment manifest`: Cycles agents through traces (agent i → trace[i%40])
- `--cpu-limit 1`: Each container gets `--cpus=1`
- `--resource-monitoring on`: Enables `ContainerStatsSampler` for each container
- `--pmu-monitoring off`: No micro-architecture sampling
- `--ksys-monitoring off`: No Kunpeng ksys collection
- `--replay-speed 999999`: LLM/MCP sleep times become near-zero; tool execution is the bottleneck
- `--output-dir`: Separate output dir per experiment for clean comparison

---

## 4. Output Layout Per Experiment

```
traces/simulate/swe-rebench/sweep_${N}a_1cpu/
├── system_resources.jsonl       # System-wide resource timeline (new)
├── agent_timeline.jsonl         # Agent lifecycle log (new)
├── simulate_cloud_model_<ts>.jsonl  # Combined trace
├── <agent_id>--a0/
│   └── attempt_1/
│       ├── trace.jsonl
│       ├── resources.json       # Per-container stats
│       └── trace_viz.html
├── <agent_id>--a1/
│   └── attempt_1/
│       └── ...
└── ...
```

---

## 5. Implementation Plan

### Phase 1: System Resource Monitor (new file)
- `scripts/system_resource_monitor.py`: standalone background sampler
- Uses `psutil` for system-wide metrics
- Start: `python scripts/system_resource_monitor.py --output <path> &`
- Stop: send SIGTERM, or create a stop-file

### Phase 2: Agent Timeline Extractor (new file)
- `scripts/extract_agent_timeline.py`: post-processes trace files
- Reads all `*/attempt_*/trace.jsonl` under output dir
- Extracts first/last action timestamps per agent

### Phase 3: Experiment Runner (new file)
- Create `scripts/run_simulate_sweep.sh` that:
  - Validates prerequisites (source traces exist, Docker available)
  - Runs the 4-config sweep
  - Starts/stops system monitor
  - Extracts agent timeline after each run
  - Prints summary statistics

### Phase 4: Execute
- Run the sweep script on the ARM machine
- Monitor progress
- Collect results

---

## Status: ✅ Implemented

All three files have been created:

| File | Status |
|------|--------|
| `scripts/system_resource_monitor.py` | ✅ Done |
| `scripts/extract_agent_timeline.py` | ✅ Done |
| `scripts/run_simulate_sweep.sh` | ✅ Done |
