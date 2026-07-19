# Stable Parallel K-Means NUMA Plan

Objective: run a reproducible NUMA placement experiment for a
`stable-parallel-kmeans`-style workload inspired by the Terminal-Bench case on
the Xeon Gold 6530 host.

This is a synthetic proxy placement experiment, not a Terminal-Bench score. It
keeps the stable-kmeans algorithmic shape fixed and changes only external NUMA
placement for the default fair comparison.

## Hardware Assumption

Use the provided `lscpu` topology:

- 128 logical CPUs, 2 sockets, 32 cores per socket, 2 threads per core.
- 4 NUMA nodes.
- node0 CPUs: `0-15,64-79`
- node1 CPUs: `16-31,80-95`
- node2 CPUs: `32-47,96-111`
- node3 CPUs: `48-63,112-127`

The default fair experiment uses matched parallelism (`n_jobs=16`) and first
hardware threads only:

- same NUMA: CPUs `0-15`, memory node `0`
- cross NUMA remote memory: CPUs `16-63`, memory node `0`
- cross NUMA interleave: CPUs `16-63`, memory interleaved over nodes `0-3`

The remote-memory condition avoids node0 CPUs, so worker processes are not
local to the bound memory node. For deliberately extreme scaling-plus-placement
runs, use the explicit `cross_numa_remote_mem_extreme` strategy; do not compare
that strategy directly against the matched-`n_jobs` fair strategies without
labeling the extra parallelism confound.

For the strictest mask-width control, add
`cross_numa_remote_mem_matched_mask`. It uses CPUs `16-31`, memory node `0`,
and `n_jobs=16`, matching the same-NUMA CPU-mask width while moving compute to
node1 and memory to node0.

## Method

Use `scripts/experiments/run_tb_stable_kmeans_numa.py`.

The runner keeps the algorithmic workload fixed and changes only the external
placement policy through `numactl` for the default strategies. It records:

- command line and strategy metadata
- `lscpu` and `numactl --hardware` output when available
- wall time, CPU time, max RSS, exit code
- workload JSON emitted by the child process
- raw stdout/stderr for each run

Before non-dry-run execution, the runner validates the host NUMA cpulists
against the expected topology from the supplied `lscpu` output. Use
`--allow-topology-mismatch` only for exploratory work on other hosts.

Default smoke command:

```bash
python scripts/experiments/run_tb_stable_kmeans_numa.py --dry-run
```

Default measurement command:

```bash
python scripts/experiments/run_tb_stable_kmeans_numa.py \
  --runs 5 \
  --samples 200000 \
  --features 64 \
  --centers 16 \
  --k-max 24 \
  --subsamples 32 \
  --percent-subsampling 0.7 \
  --output-dir traces/experiments/tb_stable_kmeans_numa
```

Strict matched-mask comparison:

```bash
python scripts/experiments/run_tb_stable_kmeans_numa.py \
  --strategies same_numa,cross_numa_remote_mem_matched_mask \
  --runs 5 \
  --samples 200000 \
  --features 64 \
  --centers 16 \
  --k-max 24 \
  --subsamples 32 \
  --percent-subsampling 0.7 \
  --output-dir traces/experiments/tb_stable_kmeans_numa_matched_mask
```

More extreme command for the same fair strategy set:

```bash
python scripts/experiments/run_tb_stable_kmeans_numa.py \
  --runs 5 \
  --samples 500000 \
  --features 128 \
  --centers 32 \
  --k-max 40 \
  --subsamples 64 \
  --percent-subsampling 0.7 \
  --output-dir traces/experiments/tb_stable_kmeans_numa_extreme
```

Extreme scaling-plus-placement remote-memory command:

```bash
python scripts/experiments/run_tb_stable_kmeans_numa.py \
  --strategies cross_numa_remote_mem_extreme \
  --runs 5 \
  --samples 500000 \
  --features 128 \
  --centers 32 \
  --k-max 40 \
  --subsamples 64 \
  --percent-subsampling 0.7 \
  --output-dir traces/experiments/tb_stable_kmeans_numa_remote_extreme
```

## Integrity Rules

- Do not change dataset size, `k_max`, subsamples, random seed, or `n_jobs`
  after seeing preliminary results unless recording a separate follow-up
  experiment.
- Do not select CPU IDs from results. CPU IDs come from the supplied topology or
  from sysfs/lscpu before the benchmark runs.
- Report all strategies, including failed or slow runs.
- Treat this as a placement sensitivity experiment, not a new Terminal-Bench
  score.
- Do not reuse an output directory unless resuming compatible runs with
  `--resume`; otherwise old and new measurements must stay separated.
