# AT-Bench

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

### Step 2 — Replay

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
