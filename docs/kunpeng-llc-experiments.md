# Kunpeng LLC Experiment Workflow

This note defines the current LLC-related experiment stack. The main rule is
that CPU ids and sub-LLC cluster sizes must come from topology probes or
hardware validation, not from hand-written host-specific assumptions.

## 1. Topology Probe

Use `scripts/experiments/probe_llc_topology.py` to read Linux sysfs CPU/cache
topology and generate placement candidates.

Example:

```bash
python scripts/experiments/probe_llc_topology.py \
  --output-dir traces/experiments/kunpeng_llc/topology \
  --agent-count 8 \
  --cluster-size 4
```

Outputs:

- `topology.json`: raw CPU/core/socket/NUMA/LLC records from sysfs.
- `placements.json`: generated agent placement candidates.
- `topology.txt`: human-readable summary.

Important caveat: `--cluster-size 4` is an inferred sub-LLC grouping used for
placement generation. It is not treated as a verified LLC-slice fact until a
separate pointer-chase validation supports it on the target host.

## 2. Pointer-Chase LLC Slice Validation

Use `scripts/experiments/probe_kunpeng_llc_slices.py` to validate candidate
sub-LLC cluster sizes with real memory-interference measurements. This script
is the structured version of manual commands such as:

```bash
numactl --membind=0 taskset -c 0 ./chase 3 8
```

Build the `chase` binary on the target Linux host:

```bash
gcc -O3 -std=c11 -Wall -Wextra -o chase scripts/experiments/chase.c
```

Quick smoke test:

```bash
./chase 3 1
```

Expected output shape:

```text
ns_per_access 20.000 accesses 4096000
```

The exact values are hardware-dependent. CPU affinity and NUMA memory binding
stay outside the binary and are controlled by `taskset` / `numactl` in the
probe script.

Example:

```bash
python scripts/experiments/probe_kunpeng_llc_slices.py \
  --chase-bin ./chase \
  --candidate-cluster-sizes 2,4,8 \
  --runs 3 \
  --aggressors-per-run 3 \
  --victim-mb 3 \
  --victim-sec 6 \
  --aggressor-mb 16 \
  --aggressor-sec 9 \
  --membind 0
```

For each candidate cluster size, the script derives physical-core
representative CPUs from sysfs, then compares:

- `baseline`: victim pointer chase without aggressors.
- `same_candidate_cluster`: aggressors in the same inferred candidate cluster.
- `other_candidate_cluster_same_llc`: aggressors in a different candidate
  cluster but the same Linux LLC domain.
- `other_llc_domain`: aggressors in another Linux LLC domain when available.

`--aggressors-per-run` is a maximum. For a 2-core candidate cluster, the
victim has only one non-victim same-cluster core, so the script uses one
same-cluster aggressor and one matched other-cluster aggressor when available.
For a 4-core candidate cluster, the default naturally uses three aggressors.

Outputs:

- `measurements.csv`: compact table for plotting and manual checks.
- `manifest.json`: full commands, raw outputs, topology, and summary.
- `summary.txt`: quick verdict per candidate cluster size.

The summary verdict is descriptive. The script first normalizes each victim
CPU's interference medians to that victim's own baseline, then summarizes
those per-victim effects. By default, a candidate size is marked
`supported_by_interference` only when same-cluster latency increases over the
victim baseline and exceeds the other-cluster same-LLC normalized effect by at
least `--support-margin-ratio` (default `0.05`). Treat this as validation
evidence, not a publishable benchmark result by itself. Report raw
distributions and limitations in analysis.

Use `--dry-run` first to inspect the derived CPU bindings without running the
workload:

```bash
python scripts/experiments/probe_kunpeng_llc_slices.py \
  --chase-bin ./chase \
  --candidate-cluster-sizes 2,4,8 \
  --dry-run
```

## 3. Replay Placement Experiments

Use `scripts/experiments/run_kunpeng_llc_replay.py` for API-free replay under
one agent count. It reuses `probe_llc_topology.py` and passes one
`--agent-cpuset` per replay agent to the simulate CLI.

Example:

```bash
python scripts/experiments/run_kunpeng_llc_replay.py \
  --source-trace traces/source.jsonl \
  --task-source data/swe-rebench/tasks.json \
  --num-agents 8 \
  --placements compact_cluster,spread_clusters_same_llc \
  --cluster-size 4 \
  --dry-run
```

Only use cluster-based placements as LLC-slice evidence after the
pointer-chase validation above supports the chosen `--cluster-size` on the
target machine.

## 4. Scaling Experiments

Use `scripts/experiments/run_kunpeng_llc_scaling.py` for the topology-derived
1/2/4/8 agent-count matrix. It replaces the older hardcoded scaling runner and
keeps CPU ids derived from sysfs.

Example:

```bash
python scripts/experiments/run_kunpeng_llc_scaling.py \
  --source-trace traces/source.jsonl \
  --task-source data/swe-rebench/tasks.json \
  --agent-counts 1,2,4,8 \
  --placements compact_cluster,spread_clusters_same_llc \
  --cluster-size 4 \
  --dry-run
```

`scripts/experiments/run_scaling_hardcoded.py` is legacy analysis support. It
requires `--allow-hardcoded-placement` and should not be used for new results.

## 5. Summaries And Analysis

Use these scripts after replay/scaling runs finish:

- `scripts/experiments/summarize_llc_placement_runs.py`: summarize run
  manifests and per-run artifacts.
- `scripts/experiments/analyze_llc_replay_results.py`: generate
  report-oriented tables and plots from LLC replay outputs.
- `scripts/experiments/run_with_perf_stat.sh`: optional wrapper for system
  `perf stat` counters around placement runs.

Analysis must keep topology-derived placement metadata with the results:
agent count, placement name, per-agent cpuset, Linux LLC ids, inferred
cluster ids, `cluster_size`, and whether pointer-chase validation supported
that cluster size on the host.

## Research Integrity Notes

- Do not tune cluster size or CPU ids to make one benchmark look better.
- Do not use benchmark outcome data to decide topology assumptions.
- Do not claim a 4-core LLC-slice mapping unless the target host validation
  supports it.
- Preserve raw pointer-chase outputs and replay manifests with any reported
  result.
- If a candidate mapping is ambiguous, report it as ambiguous and run placement
  experiments as exploratory rather than confirmatory evidence.
