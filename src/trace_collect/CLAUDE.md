# trace_collect — Agent Guide

Module for collecting, simulating, importing, and inspecting LLM agent traces
on SWE-style benchmarks.

## CLI Entry Point

```
python -m trace_collect.cli <subcommand> [OPTIONS]
```

Subcommands: (none) = **collect**, `simulate`, `import-claude-code`, `inspect`, `gantt-serve`, `profile-gpu`.

---

## Part 1 — Collect

Run agents on benchmark tasks inside containers; record canonical JSONL traces.

### Minimal Example

```bash
OPENROUTER_API_KEY=sk-... python -m trace_collect.cli \
  --provider openrouter --model anthropic/claude-sonnet-4-20250514 \
  --scaffold openclaw --container docker \
  --benchmark swe-rebench \
  --mcp-config configs/mcp/context7.yaml \
  --max-iterations 80
```

### CLI Flags (collect)

| Flag | Required | Default | Notes |
|------|----------|---------|-------|
| `--provider` | yes | — | `openrouter`, `dashscope`, `openai`, `siliconflow` |
| `--model` | yes | — | Full model ID (e.g. `anthropic/claude-sonnet-4-20250514`) |
| `--container` | container benchmarks only | — | `docker` or `podman` |
| `--scaffold` | no | `openclaw` | `openclaw` (structured tools) |
| `--benchmark` | no | `swe-bench-verified` | Slug from `configs/benchmarks/<slug>.yaml` |
| `--mcp-config` | **yes for openclaw** | `None` | YAML path or literal `none` |
| `--max-iterations` | no | `100` | Max agent loop iterations per task |
| `--sample` | no | all | Run only first N tasks |
| `--instance-ids` | no | all | Comma-separated instance IDs |
| `--max-context-tokens` | no | `256000` | Sliding window token budget |
| `--prompt-template` | no | from YAML | Override; resolved as `configs/prompts/<benchmark_slug>/<name>.md` (hyphens → underscores) |
| `--min-free-disk-gb` | no | `30.0` | Disk space preflight |
| `--run-id` | no | auto | Resume an interrupted run (pass existing run dir) |
| `--api-base` | no | from provider | Override API base URL |
| `--api-key` | no | from env | Override API key |
| `--record-internals` | no | off | OpenClaw-only: record sampled HF attention/MoE artifacts under each attempt's `recordings/`; forces model request concurrency to 1 |
| `--kv-policy` | no | `none` | KV cache eviction policy for the HF recording backend. `none` (default) = stock `DynamicCache`. `random` = uniform random over-budget eviction (step 3 baseline). `streaming` = StreamingLLM (sink prefix + recent window, naive variant — no RoPE re-rotation). `h2o` = Heavy-Hitter Oracle (arXiv:2306.14048): keep sink + recent + top-k middle positions ranked by accumulated post-softmax attention; subscribes to the `AttentionBus` to share LayerCapturer's softmax. Requires `--record-internals`. |
| `--kv-budget` | when `--kv-policy != none` | — | Per-layer KV budget in tokens. Required and must be `> 0` whenever `--kv-policy` is set. For `streaming`, this is the fixed cache capacity and must equal `sink_size + recent_window`. For `h2o`, post-eviction layer length is exactly `budget`; the heavy-hitter slot count is `budget - sink_size - recent_window`. Each call writes a `kv_eviction.npz` under `recordings/iter_<call>/` with the keep/evict audit. |
| `--kv-sink-size` | no | `4` | Sink-prefix length (head tokens preserved). Used by `streaming` and `h2o`. Ignored by `random`. |
| `--kv-recent-window` | no | `256` | Recent-window length (tail tokens preserved). Used by `streaming` and `h2o`. Ignored by `random`. |
| `--kv-aggregate` | no | `sum` | H2O score aggregation across queries. `sum` = cumulative mass (paper default); `mean` = sum / observation count per position (parallel int32 counter); `ema` = exponential moving average with `ema_decay` (yaml-only field, default 0.9, must be in (0, 1)). Ignored by `random` and `streaming`. |
| `--kv-config` | no | `None` | Optional YAML file (e.g. `configs/kv_policies/h2o_b1024.yaml`) carrying a flat map of `EvictionPolicyConfig` fields. CLI flags overlay yaml: an explicitly-passed `--kv-*` overrides the corresponding yaml value, while CLI flags left at their argparse default fall back to the yaml value. `--kv-policy none` plus a yaml `name:` activates the yaml's policy; passing `--kv-policy <other>` overrides yaml's `name`. Either an active `--kv-policy` OR a `--kv-config` requires `--record-internals`. |
| `--kv-record` | no | `on` | Whether to write `kv_eviction.npz` recordings. `on` (default) preserves the audit trail. `off` runs the configured policy but skips both the per-call `KVEvictionRecorder` allocation and the `kv_eviction.npz` write — used by the step 9 perf microbench (`scripts/spikes/step9_perf_microbench.py`) to isolate eviction overhead from recording overhead. Meaningful only when `--kv-policy != none` (or yaml supplies a name). |
| `--sparse-attn` | no | `none` | HF sparse attention mask method. `sliding`/`streaming` keep sink+recent. `heavy_hitter` uses historical attention scores. `block_topk` uses current pre-softmax QK block scores. `quest` uses Quest-style page min/max bounds. Requires `--record-internals`; enforce mode is mutually exclusive with `--kv-policy`. |
| `--sparse-attn-budget` | dynamic methods | — | Total keep-token budget target for `heavy_hitter`, `block_topk`, and `quest`; must be `>= sink_size + recent_window`. |
| `--sparse-attn-config` | no | `None` | Optional YAML under `configs/sparse_attention/`; CLI flags overlay YAML with the same default-preserving semantics as `--kv-config`. |

Internal notes:
- `LayerCapturer` publishes the post-softmax attention tensor on a per-provider `AttentionBus` (`src/serving/recording/attention_bus.py`). With no subscribers (e.g. `--kv-policy none` / `streaming` / `random`), publish is a no-op and `attention.npz` is byte-identical to the pre-bus path.
- `h2o` subscribes to that bus at construction time, consumes the same softmax tensor (no re-softmax), accumulates a head-mean cumulative score per layer, and unsubscribes in a `finally` around `model.generate(...)` to avoid leaking subscribers across calls. The score buffer is pre-allocated to `model.config.max_position_embeddings` per layer in fp32 on the attn device.
- `meta.json` gains an attempt-level `kv_policy` block reflecting the active config. The `prefill_score_bias` field is `true` when `h2o` runs with `prefill_mode="sampled"` (the bus only sees LayerCapturer-sampled prefill rows, so the score accumulator is biased toward those rows). Always `false` for `random` / `streaming` / `none` AND for `h2o` with `prefill_mode="full"`.
- `meta.json.session_history` records one row per HF chat call with `{call_idx, used_session_cache, lcp, cached_len_before, new_len, delta_len, diverged}`. Use this to distinguish cold prompts, strict-prefix resume prompts, and divergence rebuilds. Each `segments.json` segment also carries `first_seen_call`; provider-generated segments use append-only message-index provenance, while direct capturer callers may get a best-effort fallback marked by `first_seen_call_inferred=true`.

### KV Cache Eviction Policies

Three policies are wired into the HF recording path; choose one with
`--kv-policy` plus `--kv-budget`, OR point `--kv-config` at a YAML under
`configs/kv_policies/`. The YAML schema is a flat map mirroring
`src/serving/kv_policies/base.py:EvictionPolicyConfig` field-for-field:

```yaml
# configs/kv_policies/h2o_b1024.yaml
name: h2o            # one of: random, streaming, h2o
budget: 1024         # required, > 0
sink_size: 4         # streaming + h2o
recent_window: 256   # streaming + h2o
heavy_ratio: 0.5     # h2o (currently informational; eviction uses
                     #      budget - sink - recent for the heavy slot count)
aggregate: sum       # h2o: sum | mean | ema
ema_decay: 0.9       # h2o, when aggregate=ema (must be in (0, 1))
seed: 0              # random
record: true         # write kv_eviction.npz; set false for perf isolation
prefill_mode: full     # h2o: sampled | full
```

| Policy | What it keeps |
|--------|---------------|
| `random` | Uniform-random over `[0, key_len)`; no recency carve-out. Determinism via `seed`. |
| `streaming` | StreamingLLM (Xiao 2023): first `sink_size` + last `recent_window` tokens; naive variant — no RoPE re-rotation. |
| `h2o` | Heavy-Hitter Oracle (Zhang 2023): sink + recent + top-`(budget - sink - recent)` middle positions ranked by accumulated post-softmax attention. |

CLI / YAML resolution order (see `src/serving/kv_policies/config.py`):

1. If `--kv-config PATH` is set, load that YAML as the base map.
2. Each `--kv-*` flag explicitly different from its argparse default
   overrides the corresponding yaml value. Defaults left untouched fall
   through to yaml.
3. `--kv-policy none` (the default) does NOT clobber a yaml-supplied
   `name:`; any other explicit `--kv-policy` does.
4. The merged map must produce a non-`none` `name` and a positive `budget`,
   otherwise the loader exits with `argparse.ArgumentTypeError`.

`prefill_mode` (h2o only):

| Mode | Behaviour |
|------|-----------|
| `sampled` | Bus sees only LayerCapturer-sampled prefill query rows (cap = `RecordingConfig.max_prefill_queries`, default 80). Cheap approximation; meta.json marks `prefill_score_bias=true`. |
| `full` | LayerCapturer streams every prefill query row to H2O in bounded chunks of at most `RecordingConfig.max_prefill_queries`, while `attention.npz` remains sampled. This avoids materializing a full QxK recording tensor; `prefill_score_bias=false`. |

Perf isolation (`--kv-record off`): policy still runs, no
`KVEvictionRecorder` allocated, no `kv_eviction.npz` written. Used by
`scripts/spikes/step9_perf_microbench.py` to separate eviction overhead
from recording overhead.

### Provider System

Defined in `src/llm_call/providers.py`. Resolution: CLI `--api-key` > env var > error.

| Provider | Env Var | API Base |
|----------|---------|----------|
| `openrouter` | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` |
| `dashscope` | `DASHSCOPE_API_KEY` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `openai` | `OPENAI_API_KEY` | `https://api.openai.com/v1` |
| `siliconflow` | `SILICONFLOW_API_KEY` | `https://api.siliconflow.com/v1` |

### Scaffold System

| Scaffold | Description | MCP | Tools |
|----------|-------------|-----|-------|
| `openclaw` | Structured tools agent (nanobot) | **mandatory** | filesystem, shell, web, MCP, memory, skills |
| `tongyi-deepresearch` | Vendored ReAct scaffold from Alibaba-NLP/DeepResearch (pinned SHA `f72f75d8`, Apache-2.0) | no | search (Serper), visit (Jina Reader), google_scholar (Serper), parse_file (DashScope docmind), PythonInterpreter (sandbox_fusion) |

For openclaw, `--mcp-config` is enforced: pass a YAML path or the literal `none`.
MCP YAML lives in `configs/mcp/` (e.g. `context7.yaml`).

### Benchmark Plugin Architecture

Benchmarks live in `src/agents/benchmarks/` with YAML config at `configs/benchmarks/<slug>.yaml`.

| Slug | Dataset | Default Split | Default Iterations |
|------|---------|---------------|-------------------|
| `swe-bench-verified` | `princeton-nlp/SWE-bench_Verified` | `test` | 100 |
| `swe-rebench` | `nebius/SWE-rebench` | `filtered` | 100 |
| `terminal-bench` | (custom) | — | 100 |
| `deep-research-bench` | configured in YAML | `test` | 100 |
| `browsecomp` | configured in YAML | `test` | 100 |

BenchmarkConfig fields: `slug`, `display_name`, `harness_dataset`, `harness_split`,
`data_root`, `repos_root`, `trace_root`, `default_max_iterations`,
`selection_n`, `selection_seed`, `default_prompt_template`, `exclude_lite`.

**FORBIDDEN**: hardcoding dataset names in collector/cli/scaffold code. All
benchmark specifics go through the plugin + YAML.

### Container & ARM/x86

Each task runs in a fresh Docker/Podman container. Image reference comes from
the benchmark plugin's `image_name_for(task)`.

- Architecture normalization: `amd64`/`x86_64` → `amd64`, `arm64`/`aarch64` → `arm64`
- SWE-rebench ships fully-qualified x86_64 image URIs; ARM Macs use QEMU emulation
- Image prep: source image → fixed image (permission corrections, cached)
- Environment passthrough: `HTTP_PROXY`, `HTTPS_PROXY`, `PIP_INDEX_URL`, etc.
- Network mode: defaults to host; override via `--network-mode`

### Checkpointing / Resume

```bash
python -m trace_collect.cli --run-id traces/swe-rebench/model/20260414T120000 ...
```

- `load_completed_ids(run_dir)` scans `attempt_*/run_manifest.json` for `"status": "completed"`
- Completed tasks are skipped; failed/missing tasks re-run
- Each re-run increments the attempt number (`attempt_1`, `attempt_2`, ...)
- `run_manifest.json` status: `"completed"` or `"error"`

### Execution Model

Tasks run **sequentially** (one container at a time). A `ThreadPoolExecutor`
prefetches the *next* task's Docker image while the current task executes.

### Retry / Error Handling

- LLM retries: `(2, 4, 8)` second exponential backoff on HTTP 429/500/502/503/504
- Exit statuses: `success`, `error`, `tool_error`, `empty_final_response`, `max_iterations`, `timeout`, `failed`
- Disk space preflight at `--min-free-disk-gb` (default 30 GB)
- Container start timeout: 180s; stop timeout: 30s

---

## Part 2 — Simulate

Replay collected traces to measure timing under different infrastructure.

### Two Modes

| Aspect | `cloud_model` | `local_model` |
|--------|---------------|---------------|
| **LLM calls** | Replayed from source timing (no API calls) | Sent to real local OpenAI-compatible endpoint |
| **Timing source** | Source trace `ts_start`/`ts_end` × `replay_speed` | Actual TTFT + TPOT from live model |
| **Tool execution** | MCP tools (`mcp_*`) replayed from trace; others re-executed in container | All tools re-executed in container |
| **Multi-trace** | Yes (via `--trace-manifest`) | Single trace only (`--source-trace`) |
| **LLM client** | None created (forbidden) | Requires `--api-base`, `--model`, `--api-key` |
| **Use case** | "What if N agents run concurrently?" | "What if we self-host on local vLLM?" |

### Minimal Examples

```bash
# Cloud model: replay 3 traces at 50x speed, Poisson arrival
python -m trace_collect.cli simulate \
  --trace-manifest manifest.json \
  --mode cloud_model \
  --container docker \
  --replay-speed 50 \
  --arrival-mode poisson --arrival-rate-per-s 0.5 --arrival-seed 42

# Local model: measure real TTFT/TPOT against local vLLM
python -m trace_collect.cli simulate \
  --source-trace traces/.../trace.jsonl \
  --mode local_model \
  --provider openai --api-base http://localhost:8000/v1 \
  --api-key dummy --model Qwen/Qwen3-32B \
  --container docker \
  --metrics-url http://localhost:8000/metrics
```

### CLI Flags (simulate)

| Flag | Required | Default | Notes |
|------|----------|---------|-------|
| `--source-trace` | one of | — | Mutually exclusive with `--trace-manifest` |
| `--trace-manifest` | one of | — | JSON array of `{source_trace, task_source?, docker_image?}` |
| `--mode` | no | `local_model` | `local_model` or `cloud_model` |
| `--task-source` | no | `data/swe-rebench/tasks.json` | Path to tasks JSON |
| `--output-dir` | no | `traces/simulate` | Output directory |
| `--container` | no | `docker` | `docker` or `podman` |
| `--network-mode` | no | `host` | Container network mode |
| `--command-timeout` | no | `600.0` | Seconds per command |
| `--replay-speed` | no | `1.0` | Wall-clock acceleration (cloud_model only) |
| `--warmup-skip-iterations` | no | `0` | Tag first N iterations as warmup |
| `--arrival-mode` | no | `closed_loop` | `closed_loop` or `poisson` |
| `--arrival-rate-per-s` | no | — | Required for poisson mode |
| `--arrival-seed` | no | — | RNG seed for reproducibility |
| `--metrics-url` | no | — | vLLM Prometheus endpoint (local_model only; forbidden for cloud_model) |

LLM flags (`--provider`, `--api-base`, `--api-key`, `--model`) required for `local_model` only.

### Arrival Modes

| Mode | Behavior |
|------|----------|
| `closed_loop` | All tasks arrive at t=0, compete for resources |
| `poisson` | Inter-arrival times ~ Exp(rate). Seeded RNG for reproducibility |

Implementation: `harness.runner.build_arrival_offsets()` generates offsets;
cloud_model sessions `asyncio.gather` with per-session delay.

### Trace Manifest Format

```json
[
  {"source_trace": "path/to/trace-a.jsonl"},
  {"source_trace": "path/to/trace-b.jsonl", "docker_image": "custom:latest"},
  {"source_trace": "path/to/trace-c.jsonl", "task_source": "other-tasks.json"}
]
```

Paths resolved relative to manifest directory.

### vLLM Metrics Integration

When `--metrics-url` is set, the simulator snapshots Prometheus metrics per
iteration and stores them in `TraceAction.data.sim_metrics`:

- `PreemptionSnapshot`: `num_preemptions_total`, `gpu_cache_usage_perc`,
  `cpu_cache_usage_perc`, `gpu_prefix_cache_hit_rate`, `cpu_prefix_cache_hit_rate`
- When unset: empty (all-None) snapshots recorded as explicit opt-out

#### GPU Memory Tracking (US-6)

Full GPU memory breakdown time-series, sampled continuously in the background
throughout the simulation run. Requires `--gpu-tracking on` plus three additional
flags (all mutually required):

| Flag | Default | Notes |
|------|---------|-------|
| `--gpu-tracking` | `off` | `on` or `off`. When `on`, the three flags below are required. Forbidden in `cloud_model` mode. |
| `--gpu-sample-hz` | `10.0` | Sampling rate in Hz for `GpuResourceSampler`. |
| `--vllm-pid` | — | PID of the vLLM server process (used by nvidia-smi per-PID query). |
| `--vllm-startup-log` | — | Path to vLLM startup stderr log. Parsed to extract `GpuBaseline` (weights MiB, KV cache MiB). |

Validation is performed eagerly before any work begins (CLAUDE.md no-silent-fallback
rule). If `parse_startup_log_file()` returns `None`, the CLI exits with code 2 and
an explicit message naming the log path and supported vLLM versions (0.5–0.7).

Output: `<attempt-dir>/gpu_resources.json` with keys `gpu_baseline`, `gpu_samples`,
`summary`. Written atomically by `GpuResourceSampler.stop()`.

Per-iteration snapshots (`TraceAction.data.sim_metrics.vllm_scheduler_snapshot`) also
gain a `gpu_memory_breakdown` field when GPU tracking is enabled.

**Complete example:**

```bash
python -m trace_collect.cli simulate \
  --source-trace traces/.../trace.jsonl \
  --mode local_model \
  --provider openai --api-base http://localhost:8000/v1 \
  --api-key dummy --model Qwen/Qwen3-32B \
  --container docker \
  --metrics-url http://localhost:8000/metrics \
  --gpu-tracking on \
  --vllm-pid 12345 \
  --vllm-startup-log /tmp/vllm_startup.log \
  --gpu-sample-hz 10.0
```

#### Deep Profile Mode (profile-gpu subcommand)

The `profile-gpu` subcommand runs vLLM **in-process** (no separate server process)
and attaches PyTorch forward hooks on attention and MLP submodules to measure
per-step GPU memory consumption split by component type.

**Required setup:**

```bash
pip install -e .[profile]   # installs vllm + torch extras
```

**What it does:**

1. Loads the model via `InProcessEngine` (wraps `vllm.LLM`).
2. Walks the module tree, identifies `attn`/`mlp` submodules by class-name pattern.
3. Attaches `register_forward_pre_hook` + `register_forward_hook` on each module.
4. Replays LLM calls from the source trace (up to `--max-iterations`), calls
   `profiler.record_step()` after each `engine.generate()`.
5. Writes one output record per step with `sim_metrics.gpu_component_breakdown`.

**Restriction:** `tensor_parallel_size=1` only. Multi-GPU TP is out of scope;
the hook architecture assumes a single-process model graph.

**User-contribution callback:**

`src/harness/component_memory_profiler.py` — `default_memory_measurement()`.
The `# TODO(user):` block is the measurement strategy — replace with whichever
policy your experiment requires:

| Strategy | API | Notes |
|----------|-----|-------|
| `memory_allocated_delta` *(default)* | `torch.cuda.memory_allocated()` pre/post delta | Actual live tensors; misses caching-allocator headroom |
| `memory_reserved_delta` | `torch.cuda.memory_reserved()` pre/post delta | Matches nvidia-smi; includes allocator slack |
| Peak tracking | `reset_peak_memory_stats()` + `max_memory_allocated()` | Catches spikes, needs reset per step |

Keep the returned dict shape `{"value_mib": float, "measurement_kind": str}` stable;
the rest of the profiler scaffolding depends on it.

**Example invocation:**

```bash
python -m trace_collect.cli profile-gpu \
  --source-trace traces/.../trace.jsonl \
  --model Qwen/Qwen3-1.7B \
  --max-iterations 5
```

Optional flags: `--dtype float16` (default), `--max-model-len 4096` (default),
`--output-dir traces/profile_gpu` (default).

**Output:**

`traces/profile_gpu/profile_gpu_<ts>.jsonl` — v5 JSONL trace. Each `action`
record has:

```json
{
  "data": {
    "sim_metrics": {
      "gpu_component_breakdown": {
        "step_index": 0,
        "attn_mib": 42.3,
        "mlp_mib": 18.7,
        "other_activations_mib": 0.0,
        "per_module": [{"module_path": "...", "kind": "attn", "value_mib": 42.3}],
        "measurement_kind": "memory_allocated_delta"
      }
    }
  }
}
```

### Container Resource Sampling

`ContainerStatsSampler` runs a background thread at 1s intervals:
- Metrics: CPU %, memory MB, disk I/O MB, network I/O MB, context switches
- Prefers cgroup v2 host-side reads; falls back to `docker exec`-based aggregation
- Output: `resources.json` with `{samples: [...], summary: {...}}`

### Key Dataclasses (simulator.py)

- `LoadedTraceSession` — parsed source trace: actions, iterations, metadata, task
- `PreparedContainer` — container_id + ContainerAgent handle
- `PreparedTraceSession` — loaded session + container + sampler

---

## Part 3 — import-claude-code

Convert Claude Code session JSONL to canonical trace format.

```bash
python -m trace_collect.cli import-claude-code \
  --session ~/.claude/projects/<slug>/<uuid>.jsonl \
  --output-dir traces
```

- `--no-sidechains` skips folding `subagents/agent-*.jsonl` into output
- Output: `<output-dir>/claude-code-import/<uuid>/<uuid>.jsonl`
- Handles the `toolUseResult` string-vs-dict gotcha in CC JSONL format

---

## Part 4 — inspect

Post-hoc CLI for querying traces.

```bash
python -m trace_collect.cli inspect traces/.../trace.jsonl overview
python -m trace_collect.cli inspect traces/.../trace.jsonl timeline --json
python -m trace_collect.cli inspect traces/.../trace.jsonl tools --agent django__django-12345
```

Subcommands: `overview`, `step`, `messages`, `response`, `events`, `tools`, `search`, `timeline`.
Filters: `--agent`, `--role`, `--category`, `--iteration`. Output: `--json`.

---

## Trace Schema (v5)

Format: JSONL, one JSON record per line. Four record types:

### trace_metadata (first line)

```json
{
  "type": "trace_metadata",
  "trace_format_version": 5,
  "scaffold": "openclaw",
  "benchmark": "swe-rebench",
  "model": "openrouter/anthropic/claude-sonnet-4-20250514",
  "api_base": "https://openrouter.ai/api/v1",
  "instance_id": "django__django-12345",
  "max_iterations": 100,
  "prompt_template": "cc_aligned",
  "agent_runtime_mode": "host_controller",
  "scaffold_capabilities": {"tools": ["bash"], "memory": false, "skills": false},
  "run_config": {"mcp_config": "context7.yaml"}
}
```

### action (LLM call or tool execution)

```json
{
  "type": "action",
  "action_type": "llm_call",
  "action_id": "llm_0",
  "agent_id": "django__django-12345",
  "iteration": 0,
  "ts_start": 1234567890.123,
  "ts_end": 1234567895.456,
  "data": {
    "llm_wall_latency_ms": 5000.0,
    "llm_call_time_ms": 4800.0,
    "prompt_tokens": 1200,
    "completion_tokens": 350
  }
}
```

`action_type`: `llm_call` | `tool_exec`.
Timing fields in `data`: `llm_wall_latency_ms` (local wall), `llm_call_time_ms`
(preferred), `openrouter_generation_time_ms` (provider-side).

### event

```json
{
  "type": "event",
  "category": "SCHEDULING",
  "ts": 1234567890.123,
  "iteration": 0,
  "description": "..."
}
```

Categories: `SCHEDULING`, `SESSION`, `CONTEXT`, `TOOL`, `LLM`, `MCP`, `MEMORY`, `SUBAGENT`.

### summary (per-agent, at end)

```json
{
  "type": "summary",
  "agent_id": "django__django-12345",
  "success": true,
  "n_iterations": 15,
  "elapsed_s": 120.5
}
```

### Simulation-specific fields in action.data

When produced by `simulate`:
- `sim_metrics.timing`: `{ttft_ms, tpot_ms, total_ms}` (local_model)
- `sim_metrics.warmup`: bool
- `sim_metrics.source`: `"replayed_from_trace"` or `"executed_in_container"`
- `replay_mode`, `replay_speed` (cloud_model)
- `source_llm_latency_ms`, `source_duration_ms`

---

## Output Directory Layout

### collect

```
traces/<benchmark>/<safe-model>/<timestamp>/        # run_dir
  results.jsonl                                     # per-task summary
  <instance_id>/
    attempt_1/                                      # (or attempt_2, ...)
      run_manifest.json
      trace.jsonl           # canonical trace
      results.json
      resources.json        # container stats
      tool_calls.json       # flattened tool calls
      container_stdout.txt
```

### simulate

```
traces/simulate/<benchmark>/<safe-model>/<scaffold>/<arrival_tag>/
  <instance_id>/
    attempt_<N>/                    # increments per rerun
      trace.jsonl
      resources.json
  simulate_<model-or-mode>_<timestamp>.jsonl   # combined
```

`<benchmark>`, `<safe-model>`, `<scaffold>` come from the first source trace's `trace_metadata`. Attempt N auto-increments. Explicit `--output-dir` override is honored verbatim without subdir injection.

### Artifact Files

| File | Description |
|------|-------------|
| `run_manifest.json` | Status + metadata + artifact pointers (schema v1) |
| `trace.jsonl` | Canonical trace (v5 JSONL) |
| `results.json` | Task result summary |
| `resources.json` | `{samples: [...], summary: {...}}` |
| `tool_calls.json` | Flattened `{timestamp, tool, id, input, duration_ms, result_preview}` |
| `container_stdout.txt` | Raw container logs |
| `results.jsonl` | Run-level summary (one line per task) |

---

## Key Source Files

| File | Purpose |
|------|---------|
| `cli.py` | CLI entry point (argparse, subcommand dispatch) |
| `collector.py` | Collection orchestration, image prefetch, sequential dispatch |
| `simulator.py` | Simulation: cloud/local model replay, container prep, arrival offsets |
| `attempt_pipeline.py` | Per-attempt lifecycle: container start → agent run → artifact write |
| `attempt_layout.py` | Canonical artifact filenames and writers |
| `claude_code_import.py` | CC session → canonical trace converter |
| `runtime/task_container.py` | In-container entrypoint and runtime bootstrap |
| `runtime/entrypoint.py` | Container-side JSON stdin/stdout protocol |
| `src/llm_call/providers.py` | Provider registry |
| `src/llm_call/config.py` | `resolve_llm_config()` |
| `src/agents/benchmarks/base.py` | `Benchmark` ABC + `BenchmarkConfig` |
| `src/agents/base.py` | `TraceAction`, `LLMCallResult`, retry logic |
| `src/harness/runner.py` | `build_arrival_offsets()` |
| `src/harness/container_stats_sampler.py` | `ContainerStatsSampler` |
| `src/harness/metrics_client.py` | `VLLMMetricsClient` |
| `src/harness/trace_logger.py` | `TraceLogger` (JSONL writer) |
