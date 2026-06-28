# Getting Started

> This document is part of the [Agent Sched Bench manual](../README.md).
> For CLI reference, see [Trace Collect](trace-collect.md).
> For benchmark descriptions, see [Benchmarks](benchmarks.md).

---

## Table of Contents

- [Development Workflow](#development-workflow)
- [End-to-End Walkthrough (ARM Server)](#end-to-end-walkthrough-arm-server)
  - [Step 0 — Global Configuration](#step-0--global-configuration)
  - [Step 1 — One-Time Environment Setup](#step-1--one-time-environment-setup)
  - [Step 1b — Pre-Pull Images (QEMU Mode)](#step-1b--pre-pull-images-qemu-mode)
  - [Step 2 — Run](#step-2--run)
  - [Step 2b — Replay](#step-2b--replay)
- [ARM QEMU Architecture Details](#arm-qemu-architecture-details)
  - [Bootstrap Timeline](#bootstrap-timeline)
  - [Fixed Images](#fixed-images)
  - [QEMU Binfmt Loss](#qemu-binfmt-loss)
  - [Pre-Flight Checklist](#pre-flight-checklist)
- [Troubleshooting](#troubleshooting)

---

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

---

## End-to-End Walkthrough (ARM Server)

The following walkthrough takes you from a fresh ARM server to a completed
benchmark run. It covers both native ARM mode and QEMU-emulated x86_64 mode.
If you are on an x86_64 host, skip the QEMU-specific steps.

Step-by-step guide for running a single SWE-rebench task on an ARM server.

**Prerequisites:** ARM server + DeepSeek API + Docker

The harness supports **two ARM image modes**, controlled by the
`ARM_IMAGE_MODE` environment variable:

| Mode | Env | Behavior |
|------|-----|-----------|
| **native** (default) | `ARM_IMAGE_MODE=native` or unset | Build a shared ARM base image once; clone each task's repo from a local bare mirror at runtime. No QEMU needed. |
| **qemu** | `ARM_IMAGE_MODE=qemu` | Pull the official x86_64 per-task Docker images and run them via QEMU binfmt emulation. Requires `make setup-arm-host` first. |

Choose the mode that fits your environment before proceeding.

### Step 0 — Global Configuration

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

export TASK_CONTAINER_PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

sudo chmod +x scripts/setup/*.sh
```

### Step 1 — One-Time Environment Setup

**Native mode (default, yet not recommended):**

```bash
# Build the ARM-native base image and download SWE-rebench data + repos
make setup-arm-native

# Activate the conda environment
conda activate ML
```

ARM hosts auto-detect and use the native `swe-arm-base` image with local
repo mirrors — no QEMU emulation needed.

**QEMU mode:**

```bash
# Install QEMU binfmt handlers so Docker can run x86_64 images on ARM
make setup-arm-host

# Download the dataset metadata (task list) — images are pulled on demand
make download-swe-rebench

# Activate the conda environment
conda activate ML

# Then set ARM_IMAGE_MODE=qemu when running (see Step 2).
```

In QEMU mode the official `swerebench/sweb.eval.x86_64.<task>` images are
pulled and executed via QEMU user-mode emulation.  The ARM base image
(`make setup-arm-native`) is not needed.

### Step 1b — Pre-Pull Images (QEMU Mode)

In **QEMU mode** each SWE-rebench task uses its own ~2 GB Docker image
(`swerebench/sweb.eval.x86_64.<task>:latest`).  Pulling them ahead of
time avoids network stalls during the run.  (In native mode there is only
one shared base image; pre-pulling is unnecessary.)

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
runs low (set `KEEP_IMAGES_ABOVE_GB` to raise the threshold; see Step 0).

### Step 2 — Run

Check valid `instance_id`s:

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

**Native mode (default):**

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
(`swe-arm-fixed-<instance_id>`).  Subsequent runs skip the build step
and start immediately.

**QEMU mode:**

```bash
ARM_IMAGE_MODE=qemu DEEPSEEK_API_KEY=sk-deepseek-api-key PYTHONPATH=src python -m trace_collect.cli \
    --provider deepseek \
    --model deepseek-v4-flash \
    --benchmark swe-rebench \
    --scaffold openclaw \
    --instance-ids "12rambau__sepal_ui-411" \
    --mcp-config none \
    --verbose \
    --container docker
```

The official x86_64 task image is pulled on first use; a writable
derivative (`swebench-fixed-*`) is cached for subsequent runs.  Docker
transparently handles the x86_64 → ARM emulation via QEMU.

### Step 2b — Replay

Once you have collected a trace, you can replay it under different arrival
patterns or against a local serving stack. This is useful for measuring
scheduling-sensitive timing without re-running the expensive agent loop.

**Single trace replay:**

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
    --source-dir traces/swebench_verified/deepseek-v4-flash/20260605T182234 \
    --mode cloud_model \
    --replay-speed 1 \
    --task-source data/swebench_verified/tasks_full.json \
    --container docker \
    --serial
```

**Large-scale concurrent simulation:**

For stress-testing with hundreds of concurrent agents, use a trace manifest
and distribute agents across multiple worker processes:

```bash
PYTHONPATH=src python -m trace_collect.cli simulate \
    --trace-manifest manifest.json \
    --mode cloud_model \
    --container docker \
    --replay-speed 50 \
    --workers 320 \
    --prep-concurrency 64 \
    --arrival-mode poisson --arrival-rate-per-s 0.5 --arrival-seed 42
```

- `--workers` splits agents across independent OS processes, each with its
  own asyncio event loop, eliminating scheduling congestion.
- `--prep-concurrency` limits system-wide container preparations (default
  `0` = auto, preserving the historical limit of 20).  After the last
  container is ready, a global barrier releases all workers simultaneously
  into the replay phase.
- See [Trace Collect: Simulate](docs/trace-collect.md#simulate-trace-replay)
  for the full CLI reference, arrival modes, and N:M mapping.

### Visualize Results

After a run completes, generate an interactive HTML Gantt chart with resource
overlays to inspect the trace timeline:

```bash
PYTHONPATH=src python -m trace_collect.html_viz traces/swe-rebench/deepseek-chat/20260603T030206/12rambau__sepal_ui-411/attempt_1
```

---

## ARM QEMU Architecture Details

The sections below explain the internals of QEMU-mode execution. They are
useful when diagnosing performance anomalies or debugging container startup
failures. If you are running in native ARM mode, you can skip this section.

When running on an ARM server in QEMU mode, each task container goes through
a **bootstrap phase** before the agent sends its first LLM request.  During
this phase you will observe rising memory, CPU activity, and network I/O —
this is normal and expected.

### Bootstrap Timeline

```
Container Start     Bootstrap Phase (~30–120s)          Agent Actions
    │                                                       │
    ├─ sleep infinity (container alive)                     ├─ First LLM call
    ├─ ① Python probe: find /usr/bin/python3 (instant)      ├─ Tool calls
    ├─ ② pip bootstrap: download get-pip.py →       │       ├─ ...
    │      install pip inside container             │ Memory
    ├─ ③ pip install: openai, httpx, PyYAML,        │ rising
    │      json-repair, loguru, pydantic,           │
    │      socksio, tiktoken (~8 packages)          │
    └─ ④ Preflight: import-verify all modules       ┘
```

| Step | What | Resource Signature |
|------|------|-------------------|
| ① Python probe | `docker exec` to locate Python ≥3.11 | Instant, no visible impact |
| ② pip bootstrap | Download `get-pip.py`, install pip | Brief CPU + network spike |
| ③ pip install | Install 8 runtime dependencies into container | **Memory gradually rises** (download + extract wheels), sustained CPU & network I/O, the dominant cost |
| ④ Preflight | Import-verify `trace_collect`, `openclaw`, `harness` | Short CPU burst, small memory bump from module loading |

**Subsequent runs of the same task skip the entire bootstrap** — a marker file
`.bootstrap-ready.json` is written after the first successful bootstrap, and
the pipeline checks it before re-running any of steps ①–④.  You will see:

```
bootstrap runtime: reuse existing site-packages
```

### Fixed Images

The pipeline uses **two layers of images** for each task:

| Image | Source | Typical Name | Lifespan |
|-------|--------|-------------|----------|
| **Source image** | `docker pull` from registry | `swerebench/sweb.eval.x86_64.<task>` | Persistent; shared across runs |
| **Fixed image** | `docker commit` built locally | `swebench-fixed-docker.io_swerebench_sweb.eval.x86_64.<task>` | Cached per task; rebuilt if deleted |

The fixed image is a lightweight derivative (~0 extra disk) created by
`chown`-ing `/testbed` to your host UID/GID so the agent can write code
inside the container.  Building it takes a few seconds per task.

**If fixed images already exist from a previous run,** `ensure_fixed_image`
reuses them instantly (`elapsed=0.00s`).  However, stale fixed images built
under a different UID/GID or on a different host will cause the container
to exit immediately.  **Symptom:**

```
Error response from daemon: No such container: <container_id>
```

**Fix:** Delete all fixed images and let the pipeline rebuild them:

```bash
docker images --format '{{.Repository}}:{{.Tag}}' | grep '^swebench-fixed-' | xargs docker rmi -f
```

Source images are NOT affected and do NOT need re-pulling.

### QEMU Binfmt Loss

The x86_64 → ARM emulation depends on the kernel's `binfmt_misc` handler
registered via `tonistiigi/binfmt`.  This registration can be lost after a
host reboot, kernel update, or Docker daemon restart.

**Symptom:** EVERY container (even `sleep infinity`) exits immediately,
and `docker exec` returns `exec format error`:

```
exec /usr/bin/uname: exec format error
```

**Diagnose:**

```bash
docker run --rm --platform linux/amd64 <any-x86_64-image> uname -m
# Expected: x86_64
# If you see "exec format error", QEMU binfmt is gone.
```

**Fix:**

```bash
make setup-arm-host
```

This runs `docker run --privileged --rm tonistiigi/binfmt --install amd64`.
The `tonistiigi/binfmt` image may take a minute to pull on first use.

### Pre-Flight Checklist

```bash
# 1. Verify QEMU binfmt is alive
docker run --rm --platform linux/amd64 <any-x86_64-image> uname -m
# Must print: x86_64

# 2. Ensure fixed images are fresh (delete if unsure)
docker images --format '{{.Repository}}:{{.Tag}}' | grep '^swebench-fixed-' | xargs docker rmi -f

# 3. (Optional) Pre-pull source images to avoid network stalls during the run
make pull-swe-rebench-images PULL_SAMPLE=128

# 4. Run
ARM_IMAGE_MODE=qemu PYTHONPATH=src python -m trace_collect.cli ...
```

---

## Troubleshooting

The commands below help diagnose common issues during benchmark runs.

```bash
docker ps
# Check whether the task container is running

ls -lt traces/swe-rebench/deepseek-chat/<run-timestamp>/12rambau__sepal_ui-411/attempt_1/_task_container_runtime/
# Check run progress

curl -s https://api.deepseek.com/v1/models -H "Authorization: Bearer sk-your-key" | tail -1
# Verify API connectivity
```

### "docker compose: unknown command" or Docker exit code 125

Terminal-Bench requires the Docker Compose **V2 plugin** (`docker compose` with
a space), not the legacy standalone `docker-compose` (with a hyphen). If your
system only has the legacy binary, `tb run` will fail with exit code 125 and
the trace file will not be produced.

**Symptoms:**

- `docker compose version` prints "docker: unknown command: docker compose"
- Terminal-Bench run errors show `docker compose ... build` returning exit
  status 125
- `agent-logs/` directory exists but is empty (no `openclaw-trace.jsonl`)

**Fix — install the Compose plugin:**

```bash
# Ubuntu / Debian (preferred)
sudo apt-get update && sudo apt-get install docker-compose-plugin

# Or download standalone plugin binary (any Linux x86_64):
DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
mkdir -p $DOCKER_CONFIG/cli-plugins
curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" \
  -o $DOCKER_CONFIG/cli-plugins/docker-compose
chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose

# Verify
docker compose version
```
