# Agent Bench

Benchmark environment for studying agent scheduling and KV-cache management on
multi-step LLM workloads. The repo ships three top-level capabilities:

1. **Trace collect** — run agent scaffolds on benchmark tasks inside containers
   and record canonical JSONL traces (`python -m trace_collect.cli`).
2. **Trace simulate** — replay collected traces under new arrival patterns or
   against a local serving stack to measure scheduling-sensitive timing
   (`python -m trace_collect.cli simulate`).
3. **Gantt viewer demo** — an interactive FastAPI + Solid.js viewer under
   `demo/gantt_viewer/` for inspecting traces as multi-lane Gantt charts with
   resource overlays.

`AGENTS.md` and `CLAUDE.md` define research-integrity and process rules.

## Repository Layout

```text
agent-sched-bench/
├── configs/            # benchmark / system / trace_collect / sweep YAMLs
├── demo/gantt_viewer/  # FastAPI backend + Solid.js frontend
├── docs/               # specs and plans
├── scripts/            # setup, download, smoke, and runner shells
├── src/
│   ├── agents/         # scaffolds + benchmark plugins
│   ├── harness/        # runner, samplers, metrics, trace logger
│   ├── llm_call/       # provider registry + OpenAI-compatible client
│   └── trace_collect/  # CLI: collect / simulate / import / inspect / gantt-serve
└── tests/
```

## Development Workflow

All Python invocations run inside conda env "ML" (Python 3.12). On a fresh
server, run `bash scripts/setup/bootstrap.sh` once — it installs miniconda,
creates env ML, installs deps, and runs a 1-task terminal-bench smoke. Do
not create `.venv` or `pip install` ad hoc.

```bash
conda activate ML
make help    # list all targets
make test    # run pytest
make lint    # ruff
```

## SWE-bench Case Inspection (No Agent Required / Review Cases Directly)

`scripts/inspect_swebench.py` is a **standalone script** for quickly inspecting
SWE-bench Verified and SWE-rebench test cases. No agent system integration
needed — just Docker + HuggingFace datasets.

**Use cases:** Reviewers who want to see what benchmark cases look like —
repo structure, problem statements, and test details.

### Prerequisites

```bash
pip install datasets docker

export HF_ENDPOINT=https://hf-mirror.com
```

### Usage Examples

```bash
# List the first 20 SWE-bench Verified tasks
python scripts/inspect_swebench.py --benchmark swe-bench-verified list

# Search by keyword
python scripts/inspect_swebench.py --benchmark swe-bench-verified list -k django

# View full task details (problem statement, test command, image name)
python scripts/inspect_swebench.py --benchmark swe-bench-verified info django__django-10097

# Pull the Docker image (~2 GB, may take a few minutes)
python scripts/inspect_swebench.py --benchmark swe-bench-verified pull django__django-10097

# List files under /testbed inside the container
python scripts/inspect_swebench.py --benchmark swe-bench-verified ls django__django-10097

# View a specific file inside the container
python scripts/inspect_swebench.py --benchmark swe-bench-verified cat django__django-10097 /testbed/setup.py

# View the gold fix patch (what the agent is expected to produce)
python scripts/inspect_swebench.py --benchmark swe-bench-verified diff django__django-10097

# Live git diff inside the container (after making manual edits in a shell)
python scripts/inspect_swebench.py --benchmark swe-bench-verified diff django__django-10097 --container

# Show FAIL_TO_PASS tests grouped by source file (understand what needs fixing)
python scripts/inspect_swebench.py --benchmark swe-bench-verified tests django__django-10097

# View a specific test file
python scripts/inspect_swebench.py --benchmark swe-bench-verified tests django__django-10097 -f tests/auth_tests/test_validators.py

# Export the entire /testbed to a local directory
python scripts/inspect_swebench.py --benchmark swe-bench-verified export django__django-10097 /testbed ./export_django/

# Enter an interactive bash shell in the container (most flexible)
python scripts/inspect_swebench.py --benchmark swe-bench-verified shell django__django-10097
```

### SWE-rebench Works the Same Way

```bash
# SWE-rebench
python scripts/inspect_swebench.py --benchmark swe-rebench list
python scripts/inspect_swebench.py --benchmark swe-rebench info 12rambau__sepal_ui-411
python scripts/inspect_swebench.py --benchmark swe-rebench pull 12rambau__sepal_ui-411
python scripts/inspect_swebench.py --benchmark swe-rebench shell 12rambau__sepal_ui-411
```

### Use Local Cache to Skip HF Download

If you've already downloaded data via `make download-swebench-verified`,
use the local `tasks.json` cache to skip the HuggingFace download:

```bash
python scripts/inspect_swebench.py \
    --benchmark swe-bench-verified \
    --cache-file data/swebench_verified/tasks.json \
    list
```

### Common Workflows

**Workflow 1: Quickly browse a few cases**

```bash
# 1. List some tasks
python scripts/inspect_swebench.py -b swe-bench-verified list -n 5

# 2. Pick one and view details
python scripts/inspect_swebench.py -b swe-bench-verified info sympy__sympy-12481

# 3. Pull image + enter shell to explore code
python scripts/inspect_swebench.py -b swe-bench-verified pull sympy__sympy-12481
python scripts/inspect_swebench.py -b swe-bench-verified shell sympy__sympy-12481
# Inside container: ls /testbed, cat /testbed/setup.py, git log, etc.
```

**Workflow 2: Export all files for offline analysis**

```bash
python scripts/inspect_swebench.py -b swe-bench-verified pull astropy__astropy-12907
python scripts/inspect_swebench.py -b swe-bench-verified export astropy__astropy-12907 /testbed ./case_astropy_12907/
# Then open ./case_astropy_12907/ in your local IDE
```

**Workflow 3: See the gold fix patch (the expected solution)**

```bash
# Shows both the code patch and the test patch from the dataset
python scripts/inspect_swebench.py -b swe-bench-verified diff django__django-10097
```

**Workflow 4: Understand what tests need to pass (FAIL_TO_PASS)**

```bash
# See which test files are involved and how many tests per file
python scripts/inspect_swebench.py -b swe-bench-verified tests django__django-10097

# Then view a specific test file
python scripts/inspect_swebench.py -b swe-bench-verified tests django__django-10097 -f tests/auth_tests/test_validators.py
```

## Trace Collect

Run an agent scaffold on a benchmark and record a canonical v5 JSONL trace per
task. The CLI requires an explicit `--provider` and `--model` and loads
benchmark specifics from `configs/benchmarks/<slug>.yaml`.

```bash
conda activate ML
make download-swe-rebench         # or download-swebench-verified
make setup-swe-rebench-repos      # or setup-swebench-repos
PYTHONPATH=src python -m trace_collect.cli \
    --provider dashscope \
    --model qwen-plus-latest \
    --benchmark swe-rebench \
    --scaffold openclaw \
    --mcp-config none \
    --sample 2
```

Key flags: `--benchmark <slug>` (default `swe-bench-verified`),
`--scaffold openclaw|tongyi-deepresearch`, `--mcp-config` (required for
`openclaw`; YAML path or the literal `none`), `--sample N`,
`--instance-ids a,b,c`, `--run-id <path>` (resume an interrupted run),
`--prompt-template <name>` (override the benchmark default).

See `src/trace_collect/CLAUDE.md` for the complete flag reference, provider
registry, checkpointing behaviour, and trace schema v5 layout.

To run the deep research bench:

```bash
PYTHONPATH=src python -m trace_collect.cli \
    --provider deepseek \
    --model deepseek-v4-flash \
    --benchmark deep-research-bench \
    --scaffold openclaw \
    --sample 1 \    # only run the first case
    --mcp-config none \
    --verbose
```

## Quick Test

End-to-end walkthrough for running a single SWE-rebench task.

Prerequisites: ARM server + DeepSeek API + Docker

### Step 0

```bash
export HF_ENDPOINT=https://hf-mirror.com

sudo tee /etc/docker/daemon.json <<'EOF'
{
  "registry-mirrors": [
    "https://docker.1panel.live",
    "https://dockerpull.org"
  ]
}
EOF
sudo systemctl restart docker

export KEEP_IMAGES_ABOVE_GB=30

sudo sysctl -w kernel.perf_event_paranoid=-1

export WEB_SEARCH_PROVIDER=tavily
```

### Step 1 — One-time environment setup

```bash
# Build the ARM-native base image and download SWE-rebench data + repos
make setup-arm-native

# Activate the conda environment
conda activate ML
```

ARM hosts auto-detect and use the native `swe-arm-base` image with local
repo mirrors — no QEMU emulation needed.  The legacy `make setup-arm-host`
target still exists for x86_64-on-ARM QEMU emulation if required.

### Step 1b — Pre-pull images (recommended)

Each SWE-rebench task uses its own ~2 GB Docker image
(``swerebench/sweb.eval.x86_64.<task>:latest``).  Pulling them ahead of
time avoids network stalls during the run.

```bash
# Pull images for the first 16 tasks (match --sample 16)
make pull-swe-rebench-images PULL_SAMPLE=16

# Pull for specific tasks
./scripts/setup/pull_swe_rebench_images.sh \
    --instance-ids "12rambau__sepal_ui-411,0b01001001__spectree-64"

# Concurrent pulls (4 at a time)
make pull-swe-rebench-images PULL_SAMPLE=16 PULL_PARALLEL=4

# Pull everything (6,500+ images — use with care!)
make pull-swe-rebench-images
```

Already-pulled images are re-used across runs and only removed when disk
runs low (set ``KEEP_IMAGES_ABOVE_GB`` to raise the threshold; see Step 0).

### Step 2 — Run

Check valid `instance_id`s.

```bash
python -c "
import json
tasks = json.load(open('data/swe-rebench/tasks.json'))
for t in tasks[:20]:
    print(t['instance_id'], '|', t.get('repo',''))
print(f'... ({len(tasks)} total)')
"
```

Run a specific test case.

```bash
DEEPSEEK_API_KEY=sk-deepseek-api-key PYTHONPATH=src python -m trace_collect.cli \
    --provider deepseek \
    --model deepseek-v4-flash \
    --benchmark swe-rebench \
    --scaffold openclaw \
    --instance-ids "12rambau__sepal_ui-411" \
    --mcp-config none \
    --verbose \
    --container docker
```

The first run on a task builds a cached ARM derivative image
(``swe-arm-fixed-<instance_id>``).  Subsequent runs skip the build step
and start immediately.

### Step 2b — Replay

```bash
PYTHONPATH=src python -c "
import json
from pathlib import Path
from agents.benchmarks.swe_bench_verified import SweBenchVerified
from agents.benchmarks.base import BenchmarkConfig

config = BenchmarkConfig.from_yaml(Path('configs/benchmarks/swe-bench-verified.yaml'))
plugin = SweBenchVerified(config)
tasks = plugin.load_tasks()  # load all 500 tasks without select_subset
Path('data/swebench_verified/tasks_full.json').write_text(
    json.dumps(tasks, indent=2, ensure_ascii=False, default=str) + '\n',
    encoding='utf-8',
)
print(f'Wrote {len(tasks)} tasks')
"
```

```bash
PYTHONPATH=src python -m trace_collect.cli simulate \
    --source-trace traces/swebench_verified/deepseek-v4-flash/20260605T182234/astropy__astropy-12907/attempt_1/trace.jsonl \
    --mode cloud_model \
    --replay-speed 1 \
    --task-source data/swebench_verified/tasks_full.json \
    --container docker
```

```bash
PYTHONPATH=src python -m trace_collect.cli simulate \
    --source-dir /home/weitian/agent-test-bench/traces/swebench_verified/deepseek-v4-flash/20260605T182234 \
    --mode cloud_model \
    --replay-speed 1 \
    --task-source data/swebench_verified/tasks_full.json \
    --container docker
    --serial
```

### Troubleshooting

```bash
docker ps
# Check whether the task container is running
ls -lt traces/swe-rebench/deepseek-chat/<run-timestamp>/12rambau__sepal_ui-411/attempt_1/_task_container_runtime/
# Check run progress
curl -s https://api.deepseek.com/v1/models -H "Authorization: Bearer sk-your-key" | tail -1
# Verify API connectivity
```

### Visualise results

```bash
PYTHONPATH=src python -m trace_collect.html_viz traces/swe-rebench/deepseek-chat/20260603T030206/12rambau__sepal_ui-411/attempt_1
```

### Recording Internals

`--record-internals` switches OpenClaw model calls to a host-side HuggingFace
backend and records reduced attention/MoE artifacts beside each attempt.
It currently supports `--scaffold openclaw` only.
For task-container benchmarks, the container talks to a temporary local
OpenAI-compatible proxy backed by that host model, so benchmark tools still run
inside the task container. Docker task containers use `172.17.0.1` as the
default host gateway for that proxy; set `HF_RECORDING_PUBLIC_HOST` if the
server uses a different bridge address.

```bash
PYTHONPATH=src python -m trace_collect.cli \
    --provider openai --api-key hf-recording \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --benchmark swe-rebench \
    --scaffold openclaw \
    --container docker \
    --mcp-config none \
    --sample 1 \
    --record-internals
```

Artifacts are written under
`<attempt_dir>/recordings/iter_0000/{attention.npz,routing.npz,segments.json}`
plus `<attempt_dir>/recordings/meta.json`. `call_idx` is 0-based and aligns to
the nth `action_type="llm_call"` record in `trace.jsonl`.

Sanity-check one call:

```bash
python scripts/load_recording.py --attempt-dir <attempt_dir> --call-idx 0
```

Recording uses `attn_implementation="sdpa"` for the model path and computes
only sampled attention rows inside hooks. It forces
`NANOBOT_MAX_CONCURRENT_REQUESTS=1` and is intended for data collection, not
production throughput.

### Registered Benchmarks

| Slug | `task_shape` | Dataset | Split | Docker | Scaffolds | Scoring |
|---|---|---|---|---|---|---|
| `swe-bench-verified` | `swe_patch` | `princeton-nlp/SWE-bench_Verified` | `test` | `swebench/sweb.eval.x86.*` (namespace-prefixed) | openclaw | harness (pytest in container) |
| `swe-rebench` | `swe_patch` | `nebius/SWE-rebench` | `filtered` | `swerebench/sweb.eval.x86_64.*` (fully qualified) | openclaw | harness (pytest in container) |
| `terminal-bench` | `terminal_task` | `terminal-bench-core` (or local task dir) | n/a | Terminal-Bench-managed | openclaw (phase 1) | Terminal-Bench harness + imported OpenClaw trace |
| `deep-research-bench` | `research_qa` | configured in YAML | `test` | host | openclaw, tongyi-deepresearch | reference-answer comparison |
| `browsecomp` | `browse_qa` | configured in YAML | `test` | host | openclaw, tongyi-deepresearch | reference-answer comparison |

Terminal-Bench requires Python 3.12+ (upstream `tb` CLI dependency) and only
supports `--scaffold openclaw` with the Docker runtime in phase 1.

## BFCL via OpenClaw (host-mode)

Four BFCL (Berkeley Function Calling Leaderboard) datasets are wired as
host-mode benchmarks driven by the OpenClaw scaffold. BFCL runs as a read-only
external environment: its dataset loader, tool-doc → schema converter, stateful
backend instantiation, and backend implementations (simulated filesystem,
booking, web search, vector/kv/rec_sum memory) are reused unmodified; OpenClaw
issues the LLM calls and runs the multi-turn loop. Each BFCL conversation turn
is delivered one at a time, so the agent only ever sees the current and past
turns. Traces, `resources.json`, and a per-task HTML report come from the
standard host pipeline — BFCL's own evaluation/scoring is **not** run.

| Slug | BFCL categories | Notes |
|---|---|---|
| `bfcl-multi-turn-base` | `multi_turn_base` | stateful file/booking/etc. tools |
| `bfcl-multi-turn-long-context` | `multi_turn_long_context` | large injected backend state |
| `bfcl-web-search` | `web_search_base`, `web_search_no_snippet` | needs a web-search API key |
| `bfcl-memory` | `memory_vector`, `memory_kv`, `memory_rec_sum` | prereq conversations run first |

### Full run on a fresh machine

**1. Get both repos.** Clone this repo and the BFCL (`gorilla`) repo anywhere:

```bash
git clone <this-repo> agent-test-bench
git clone https://github.com/ShishirPatil/gorilla.git gorilla
```

**2. Create the Python env** (uv, Python 3.12) at this repo root:

```bash
cd agent-test-bench
uv venv --python 3.12
uv pip install -e ".[dev]"
```

**3. Install the BFCL runtime deps** these four datasets need (pins match
BFCL's `pyproject.toml`):

```bash
uv pip install \
  "tree_sitter==0.21.3" "tree-sitter-java==0.21.0" "tree-sitter-javascript==0.21.4" \
  "faiss-cpu==1.11.0" "sentence-transformers" "rank_bm25" "overrides" \
  "google-search-results" "html2text" "beautifulsoup4"
```

**4. Point at the BFCL checkout** (no machine-specific default is baked in):

```bash
export BFCL_REPO_PATH=/abs/path/to/gorilla   # dir containing berkeley-function-call-leaderboard/
```

**5. Provide API keys / model assets:**

- LLM provider key for `--provider/--model` (e.g. `DASHSCOPE_API_KEY`,
  `OPENAI_API_KEY`, `OPENROUTER_API_KEY`).
- `bfcl-web-search`: a SerpAPI key — `export SERPAPI_API_KEY=...`.
- `bfcl-memory` (vector): first run downloads `all-MiniLM-L6-v2` from
  HuggingFace. If a global `HF_ENDPOINT` mirror is set but its model-config API
  path fails, force the real host for the run with
  `HF_ENDPOINT=https://huggingface.co` (works through an HTTP/SOCKS proxy).
- For host memory-bandwidth sampling: `sudo sysctl -w kernel.perf_event_paranoid=-1`
  (otherwise the bandwidth fields read 0; everything else still works).

**6. Run the full test set** (omit `--sample` for all entries; `--mcp-config none`
disables MCP). Examples — repeat per slug:

```bash
PYTHONPATH=src python -m trace_collect.cli \
    --provider <provider> --model <model> \
    --benchmark bfcl-multi-turn-base \
    --scaffold openclaw --mcp-config none

PYTHONPATH=src python -m trace_collect.cli --provider <provider> --model <model> \
    --benchmark bfcl-multi-turn-long-context --scaffold openclaw --mcp-config none

PYTHONPATH=src python -m trace_collect.cli --provider <provider> --model <model> \
    --benchmark bfcl-web-search --scaffold openclaw --mcp-config none      # SERPAPI_API_KEY

PYTHONPATH=src python -m trace_collect.cli --provider <provider> --model <model> \
    --benchmark bfcl-memory --scaffold openclaw --mcp-config none          # faiss + ST
```

Smoke first with `--sample 1` (or `--instance-ids <id>`) before a full run.
Per-task artifacts land under
`traces/<slug>/<safe-model>/<timestamp>/<instance_id>/attempt_1/`:
`trace.jsonl`, `resources.json`, and the HTML report
`<slug>__<instance_id>.html` (named so a full run's HTMLs are uniquely
identifiable and collectible into one folder, e.g.
`find traces -name '<slug>__*.html'`).

> Note on `bfcl-memory`: each scenario's prerequisite "memory write"
> conversations are emitted before its question entries and must run in that
> order (the plugin preserves load order instead of sorting by id), so memory
> state is populated before the questions. Sampling a subset may split a
> scenario — prefer a full run, or select whole scenarios via `--instance-ids`.

## Resource Monitoring

The harness ships five resource monitors that activate depending on the
runtime mode (container, host process, local GPU serving, or deep-profile).
Each sample carries a UTC timestamp and an epoch; monitors run on background
threads or asyncio tasks at a configurable interval.

### Monitors by Runtime Mode

| Runtime Mode | Active Monitor(s) | Key Source Files |
|---|---|---|
| **container** (Docker/Podman) | `ContainerStatsSampler` | `src/harness/container_stats_sampler.py` |
| **host** (no container) | `ProcessStatsSampler` | `src/harness/process_stats_sampler.py` |
| **local_model** (vLLM GPU replay) | `VLLMMetricsClient` + `GpuResourceSampler` | `src/harness/metrics_client.py`, `src/harness/gpu_resource_sampler.py` |
| **deep-profile** (PyTorch hooks) | `ComponentMemoryProfiler` | `src/harness/component_memory_profiler.py` |

### 1. ContainerStatsSampler — container CPU / memory / I/O

Reads Linux cgroup v2 counters directly (sub-millisecond overhead, no Docker
daemon round-trip). Falls back to `docker stats --no-stream` +
`docker exec python3` on non-cgroup-v2 hosts.

| Metric | Source | Isolation |
|--------|--------|-----------|
| CPU % | cgroup v2 `cpu.stat` `usage_usec` delta between consecutive samples, divided by `ncpus` | ✅ container-scoped |
| Memory | cgroup v2 `memory.current` / `memory.max` | ✅ container-scoped |
| Disk I/O (read/write bytes) | cgroup v2 `io.stat` aggregated across all devices; fallback: sum `/proc/*/io` inside container via `docker exec` | ✅ container-scoped |
| Context switches | sum of `voluntary_ctxt_switches` + `nonvoluntary_ctxt_switches` across all PIDs in the container's `cgroup.procs`, with per-(pid, starttime) high-water marks so exited processes are not lost | ✅ container-scoped |
| Network I/O (rx/tx bytes) | `/proc/<pid>/net/dev` of the container init PID (loopback excluded) | ✅ container-scoped |
| Memory bandwidth | `perf stat -a` reading Intel IMC CAS counters (system-wide); requires `kernel.perf_event_paranoid=-1` | ⚠️ host-wide |

### 2. ProcessStatsSampler — host process CPU / memory / I/O

Uses `psutil` (fallback: `ps`) to sample the target PID and its recursive
children. Complements with `/proc/<pid>/{io,status,net/dev}` reads where
available.

| Metric | Source | Isolation |
|--------|--------|-----------|
| CPU % | `psutil.Process.cpu_percent()` + recursive children; fallback: `ps -o %cpu=` | ✅ process-tree scoped |
| Memory (RSS) | `psutil.Process.memory_info().rss` + children RSS summed; fallback: `ps -o rss=` | ✅ process-tree scoped |
| Disk I/O | `psutil.Process.io_counters()` (read_bytes, write_bytes) + children; also reads `/proc/<pid>/io` | ✅ process-tree scoped |
| Context switches | `/proc/<pid>/status` (`voluntary_ctxt_switches` + `nonvoluntary_ctxt_switches`) | ✅ process-tree scoped |
| Network I/O | `/proc/<pid>/net/dev` (loopback excluded) | ✅ process-tree scoped |
| Memory bandwidth | same `perf stat -a` as ContainerStatsSampler | ⚠️ host-wide |

### 3. VLLMMetricsClient — vLLM scheduler Prometheus metrics

Polls the vLLM Prometheus endpoint (`/metrics`) and parses gauge/counter
values. All metrics are vLLM-internal — naturally isolated to the vLLM
process.

| Metric | Prometheus Gauge |
|--------|-----------------|
| Running / waiting requests | `vllm:num_requests_running`, `vllm:num_requests_waiting` |
| GPU / CPU KV cache usage | `vllm:gpu_cache_usage_perc` (or `vllm:kv_cache_usage_perc` in vLLM 0.10+), `vllm:cpu_cache_usage_perc` |
| Preemption count | `vllm:num_preemptions_total` |
| Prefix cache hit rates | `vllm:gpu_prefix_cache_hit_rate`, `vllm:cpu_prefix_cache_hit_rate` |
| Throughput | `vllm:avg_prompt_throughput_toks_per_s`, `vllm:avg_generation_throughput_toks_per_s` |
| Latency | `vllm:e2e_request_latency_seconds` (sum + count), `vllm:time_to_first_token_seconds` |

### 4. GpuResourceSampler — GPU memory breakdown (local_model only)

Combines three data sources at a configurable rate (default 10 Hz) to produce
a `GpuMemoryBreakdown` time series:

| Component | Source | Description |
|-----------|--------|-------------|
| `weights_mib` | `GpuBaseline` parsed from vLLM startup log | One-shot; model weights resident in GPU memory |
| `kv_cache_total_mib` | `GpuBaseline` parsed from vLLM startup log | Total GPU memory budget for KV cache |
| `kv_cache_used_mib` | vLLM Prometheus `gpu_cache_usage_perc` × `kv_cache_total_mib` | How much of the KV budget is currently occupied |
| `total_pid_mib` | `nvidia-smi --query-compute-apps=pid,gpu_serial,used_memory` filtered by vLLM PID | Total GPU memory held by the vLLM process |
| `activations_mib` | residual: `total_pid_mib - weights_mib - kv_cache_used_mib` | Estimated activation memory (clamped ≥ 0) |

All GPU memory readings are per-PID, so other users' GPU processes do not
affect them. Fails fast with `GpuPidNotFoundError` if the vLLM PID disappears
mid-run.

### 5. ComponentMemoryProfiler — per-layer activation memory (deep-profile)

Attaches PyTorch forward hooks to attention and MLP submodules (detected by
class-name pattern matching: `attention|attn`, `mlp|feedforward|ffn`). Uses
`torch.cuda.memory_allocated()` pre/post delta to estimate per-component
activation memory. Outputs `GpuComponentBreakdown` records per generate step.
This is entirely in-process and not affected by other users.

### Output Format

All monitors emit samples that feed into `summarize_samples()` producing a
`resources.json` with min/max/avg for every metric plus deltas for monotonic
counters (disk, network, context switches). The HTML viz renders these as
time-series overlays on the Gantt chart.

### Multi-Tenant Safety Summary

| Monitoring Layer | Affected by other users on the same machine? |
|---|---|
| Container cgroup (CPU, memory, disk I/O, ctxt switches) | ❌ No |
| Container network I/O (init PID `/proc/net/dev`) | ❌ No |
| Process-level (psutil, `/proc/<pid>/*`, ps) | ❌ No |
| GPU memory per PID (`nvidia-smi --query-compute-apps`) | ❌ No |
| vLLM Prometheus internal metrics | ❌ No |
| PyTorch `torch.cuda.memory_allocated()` | ❌ No |
| **Host memory bandwidth (`perf stat -a`)** | ⚠️ **Yes — system-wide PMC counters** |

> **Note on memory bandwidth:** `perf stat -a` monitors all CPUs and IMCs on
> the host. If other users are running memory-intensive workloads on the same
> physical machine, `memory_total_mb_s` / `memory_read_mb_s` /
> `memory_write_mb_s` in the output will include their traffic. For clean
> measurements, run on a dedicated node or treat these as upper-bound
> estimates. Set `sudo sysctl -w kernel.perf_event_paranoid=-1` before
> starting the harness, otherwise bandwidth sampling will report
> `permission_denied` and all bandwidth fields will be 0.

### Adding a New Benchmark

Benchmarks live in `src/agents/benchmarks/<slug>.py` with a matching
`configs/benchmarks/<slug>.yaml`. See `src/agents/benchmarks/base.py` for the
`Benchmark` ABC and `BenchmarkConfig` fields, plus `CLAUDE.md §Benchmark Plugin
Architecture` for the enforcement rules. Short version:

1. Implement a `Benchmark` subclass in `src/agents/benchmarks/<slug>.py`.
2. Author `configs/benchmarks/<slug>.yaml` with the `BenchmarkConfig` fields.
3. Register the class in `src/agents/benchmarks/__init__.py::REGISTRY`.
4. Add `tests/test_<slug>_plugin.py` covering `normalize_task`.
5. Add `make download-<slug>` / `make setup-<slug>-repos` Makefile targets.

Dataset names, image namespaces, and CLI-visible defaults must live in YAML —
never hardcode them in `collector.py`, `cli.py`, or scaffold code.

## Trace Simulate

Replay a collected trace under new infrastructure assumptions. Two modes:

| Mode | LLM calls | Timing source | Multi-trace | Use case |
|---|---|---|---|---|
| `cloud_model` | replayed from source trace (no API call) | `ts_start`/`ts_end` × `--replay-speed` | yes (`--trace-manifest`) | "what if N agents arrive concurrently?" |
| `local_model` | sent to a real OpenAI-compatible endpoint | live TTFT + TPOT | single (`--source-trace`) | "what if we self-host on local vLLM?" |

### cloud_model — arrival-pattern sweep

```bash
PYTHONPATH=src python -m trace_collect.cli simulate \
    --trace-manifest configs/trace_collect/simulate.yaml \
    --mode cloud_model \
    --container docker \
    --replay-speed 50 \
    --arrival-mode poisson \
    --arrival-rate-per-s 0.5 \
    --arrival-seed 42
```

### local_model — self-hosted serving

```bash
PYTHONPATH=src python -m trace_collect.cli simulate \
    --source-trace traces/.../trace.jsonl \
    --mode local_model \
    --provider openai --api-base http://localhost:8000/v1 \
    --api-key dummy --model Qwen/Qwen3-32B \
    --container docker \
    --metrics-url http://localhost:8000/metrics
```

`--metrics-url` snapshots vLLM Prometheus counters per iteration
(`num_preemptions_total`, `gpu_cache_usage_perc`, `*_prefix_cache_hit_rate`) into
`TraceAction.data.sim_metrics`. Container resource usage is sampled at 1 Hz by
`ContainerStatsSampler` and written to `resources.json`.

**GPU memory tracking** (`--gpu-tracking on`): add `--vllm-pid`, `--vllm-startup-log`,
and `--gpu-sample-hz` to capture a full GPU memory breakdown time-series (weights,
KV cache, activations) sampled in the background and written to `gpu_resources.json`.
For per-component (attn/mlp) deep profiling without a separate server, use the
`profile-gpu` subcommand (requires `pip install -e .[profile]`; GPU + vLLM only).

See `src/trace_collect/CLAUDE.md` §Simulate for the full flag table, manifest
format, output directory layout, and simulation-specific fields in `action.data`.

## Importing Claude Code Sessions

Convert a raw Claude Code session JSONL to canonical trace format:

```bash
PYTHONPATH=src python -m trace_collect.cli import-claude-code \
    --session ~/.claude/projects/<slug>/<uuid>.jsonl \
    --output-dir traces
```

Sidechains under `subagents/` are folded in by default (pass `--no-sidechains`
to skip). The Gantt viewer's `/api/traces/register` and `/api/traces/upload`
endpoints auto-invoke this importer when they detect raw CC JSONL.

## Inspecting Traces

```bash
PYTHONPATH=src python -m trace_collect.cli inspect traces/.../trace.jsonl overview
PYTHONPATH=src python -m trace_collect.cli inspect traces/.../trace.jsonl timeline --json
PYTHONPATH=src python -m trace_collect.cli inspect traces/.../trace.jsonl tools --agent <instance-id>
```

Subcommands: `overview`, `step`, `messages`, `response`, `events`, `tools`,
`search`, `timeline`. Filters: `--agent`, `--role`, `--category`, `--iteration`.

## Demo: Gantt Viewer

Interactive multi-lane Gantt visualisation for collected and simulated traces.
FastAPI backend (`:8765`) + Solid.js / Vite frontend rendered on Canvas 2D.

```bash
make gantt-viewer-install   # one-time: npm install
make gantt-viewer-dev       # dev mode (Vite HMR on :5173)
make gantt-viewer-build     # production bundle into frontend/dist
make gantt-viewer-test      # backend pytest + frontend vitest
make gantt-viewer-smoke     # headless browser smoke check
```

Equivalent CLI:

```bash
PYTHONPATH=src:. python -m trace_collect.cli gantt-serve --dev
PYTHONPATH=src:. python -m trace_collect.cli gantt-serve \
    --config demo/gantt_viewer/configs/example.yaml
```

Discovery config globs accept canonical trace JSONL only. At runtime,
`POST /api/traces/register` and `POST /api/traces/upload` auto-import raw
Claude Code sessions through `import-claude-code` before registration.

Smoke-only subsets belong in dedicated `*_smoke.yaml` workload configs; default
workload configs should describe the full benchmark dataset path.

See `demo/gantt_viewer/README.md` for the full acceptance workflow and
`demo/gantt_viewer/AGENT_INTERFACE.md` for the REST-driven agent interface.
