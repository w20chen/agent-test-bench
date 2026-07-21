# Tool Profiler — Lightweight Online Profiling Prototype

A minimal-viable prototype for the research question:

> Can we form a useful invocation-level resource profile by observing only the
> first 2 seconds of a tool's process tree, without any knowledge of its
> internal code, threading model, or subprocess architecture?

## Design

The profiler wraps any external command, samples the **entire process tree**
(root + all recursive children) every 200 ms by default, and computes:

- **effective_cpu_cores** = ΔCPU_time / Δwall_time (actual parallelism, not thread count)
- **early profile** after a configurable warmup window (default 2 s)
- **final profile** after the tool exits
- **early-final comparison** to test whether the early window predicts the full run

The profiler **does NOT** pause, restart, or modify the tool process in any way.
It is read-only observation.

## Quick Start

```bash
# Profile any command
python -m prototype.tool_profiler -- python my_script.py

# With custom warmup window
python -m prototype.tool_profiler --warmup-seconds 3 -- make -j8

# Save raw samples (for detailed analysis)
python -m prototype.tool_profiler --save-samples -- pytest -q

# Run the full demo suite
bash run_demo.sh

# Summarize results
python summarize_profiles.py demo_profiles.jsonl
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--warmup-seconds` | `2.0` | Early profile observation window (seconds) |
| `--sample-interval` | `0.2` | Time between consecutive samples (seconds) |
| `--output` | `tool_profiles.jsonl` | JSONL output file path |
| `--verbose` | `false` | Print per-sample summaries to stderr |
| `--save-samples` | `false` | Include all raw sample data in JSONL |

## Architecture

```
prototype/tool_profiler/
├── __init__.py      # Package metadata
├── cli.py           # Argument parsing and entry point
├── runner.py        # Tool lifecycle: launch, monitor, collect, output
├── sampler.py       # psutil-based process tree sampling
├── metrics.py       # Delta computation, effective cores, percentiles
├── classifier.py    # Weak behavior label classification
└── README.md        # This file
```

### Module Responsibilities

- **`cli.py`**: Parses `--warmup-seconds`, `--sample-interval`, `--output`, `--`, and delegates to `runner.run_tool()`.
- **`runner.py`**: Launches argv commands directly (`shell=False`) and uses `--shell-command` only for existing shell command strings, runs a daemon sampling thread, waits for tool exit, handles termination by cleaning the profiled process tree, and writes JSONL output.
- **`sampler.py`**: Uses `psutil.Process(pid)` to enumerate the process tree (`children(recursive=True)`) and aggregates CPU times, RSS, VMS, I/O counters, context switches, and page faults across all live processes.
- **`metrics.py`**: Computes per-window deltas, `effective_cpu_cores`, percentiles (linear interpolation), coefficient of variation, and aggregates into `AggregatedMetrics`.
- **`classifier.py`**: Applies simple threshold-based rules to assign a weak behavior label (`cpu_parallel`, `cpu_serial`, `io_active`, `mixed`, `idle_or_waiting`, `unknown`).

## Metrics Collected

### Per-Sample Window
```
timestamp_s, elapsed_s, root_pid
process_count, thread_count
cpu_user_time_s, cpu_system_time_s, cpu_total_time_s
cpu_time_delta_s, wall_time_delta_s, effective_cpu_cores
rss_bytes, vms_bytes
read_bytes, write_bytes, read_count, write_count
voluntary_context_switches, involuntary_context_switches
minor_page_faults, major_page_faults
```

### Behavior Labels

| Label | Condition |
|-------|-----------|
| `cpu_parallel` | avg_effective_cores ≥ 2.0 AND cpu_time_ratio ≥ 1.5 |
| `cpu_serial` | 0.6 ≤ avg_effective_cores < 2.0 AND I/O not significant |
| `io_active` | cpu_time_ratio < 0.5 AND I/O bytes > 1 MiB |
| `idle_or_waiting` | cpu_time_ratio < 0.2 AND I/O bytes < 1 MiB |
| `mixed` | Both CPU and I/O present |
| `unknown` | Insufficient data, too short, or conflicting |

**IMPORTANT**: `preliminary_behavior` is a quick profile result, NOT a rigorous
performance bottleneck classification. Without DRAM bandwidth or interference
experiments, we cannot reliably claim a tool is memory-bandwidth-bound.

## JSONL Output Format

Each invocation writes one JSON line:
```json
{
  "schema_version": 1,
  "invocation_id": "uuid",
  "command": ["python3", "run.py"],
  "command_string": "python3 run.py",
  "cwd": "/path/to/workdir",
  "root_pid": 12345,
  "start_time": "2026-07-21T...",
  "warmup_seconds": 2.0,
  "sample_interval": 0.2,
  "exit_code": 0,
  "early_profile": { "available": true, ... },
  "early_final_comparison": {
    "effective_cores_relative_error": 0.03,
    "behavior_changed": false,
    "stability_changed": false
  },
  "final_profile": {
    "total_wall_time_s": 46.3,
    "short_tool": false,
    ...
  }
}
```

## Known Limitations

1. **Short-lived subprocesses may be missed**: If a child process is created and
   destroyed between two 200 ms samples, it will not appear in the profile.
   This is an inherent limitation of polling-based observation.

2. **Platform-dependent metrics**: `psutil` may not expose page fault details or
   I/O counters on all platforms (particularly macOS). Missing fields are set to
   `null` in the output.

3. **Cumulative I/O counters**: `psutil` reports cumulative I/O from process
   start, not per-window I/O. We report the cumulative values at the last sample.

4. **Behavior labels are weak**: The classifier uses simple thresholds and does
   not distinguish CPU-bound from memory-bandwidth-bound. No DRAM bandwidth
   measurement is performed.

5. **No cgroup/namespace isolation**: The profiler observes the tool's process
   tree within the same cgroup/namespace. If the tool spawns processes outside
   the profiler's visibility scope, they won't be captured.

6. **Thread-level metrics**: We count threads but do not attribute CPU time to
   individual threads within a process.

## Future Integration with OpenClaw

The profiler could be integrated into the OpenClaw agent scaffold as a hook:

```
# Minimal integration sketch
from prototype.tool_profiler.runner import run_tool

# In OpenClaw tool execution path:
exit_code = run_tool(
    command=["bash", "-c", tool_command],
    warmup_seconds=2.0,
    output_path=f"traces/tool_profiles/{invocation_id}.jsonl",
)
```

The JSONL output could then be consumed by the trace collector for downstream
analysis without modifying the agent execution pipeline.

## Test Workloads

```
workloads/
├── cpu_serial.py      # Single-threaded CPU burn (default 5 s)
├── cpu_parallel.py    # Multi-process CPU burn (default 4 workers, 5 s)
├── io_worker.py       # Sustained read/write to temp file
└── process_tree.py    # Parent + N child processes
```
