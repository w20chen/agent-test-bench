# Script Inventory

This page explains what lives under `scripts/` and how to decide whether a
script is a public entry point, experiment support, analysis utility, or
research scratch space.

The cleanup rule is conservative: do not delete scripts just because they are
not called from `Makefile`. Many scripts are intentionally run by hand for
analysis or figure generation, and several are imported directly by tests.

## Status Labels

| Label | Meaning |
|---|---|
| Public | Documented entry point or Makefile target. Keep stable. |
| Supported | Used by docs, tests, or active experiment plans. Keep stable enough to rerun. |
| Analysis | Reads existing traces/recordings and produces derived tables or figures. Does not create benchmark results. |
| Research workspace | Ad-hoc or Modal-backed research code. Keep, but do not treat as a stable CLI. |
| Spike | Exploratory validation code. Keep with its notes unless explicitly retired. |
| Pending review | Untracked or encoding-damaged script that needs human confirmation before promoting or deleting. |

## Top-Level Scripts

| Script | Status | Purpose | References / notes |
|---|---|---|---|
| `inspect_swebench.py` | Public | Inspect SWE-Bench Verified and SWE-rebench tasks without running an agent. | Documented in `README.md` and `docs/case-inspection.md`. |
| `system_resource_monitor.py` | Public | Background sampler for system CPU, memory, disk, network, load, and container count. | Used by simulation sweep docs. |
| `plot_system_resources.py` | Public | Convert `system_resources.jsonl` into a self-contained HTML dashboard. | Documented in `docs/trace-collect.md` and `docs/resource-measurement.md`. |
| `extract_agent_timeline.py` | Public | Extract per-agent lifecycle timing from simulation outputs. | Used by `run_simulate_sweep.sh`. |
| `run_simulate_sweep.sh` | Public | Orchestrate trace replay sweeps with system monitoring and timeline extraction. | Documented in `docs/trace-collect.md`. |
| `run_mixed_scheduling_sweep.sh` | Supported | Run mixed scheduling replay sweeps. | Similar role to `run_simulate_sweep.sh`; keep with sweep tooling. |
| `run_smoke.sh` | Public | Run current infrastructure smoke checks. | Exposed by `make run-smoke`. |
| `run_sweep.sh` | Public | Run harness sweep command. | Exposed by `make run-sweep`. |
| `collect_results.sh` | Public | Pull result artifacts back from a remote machine. | Exposed by `make collect-results`. |
| `serve_vllm.sh` | Public | Launch raw vLLM serving stack. | Exposed by `make serve-vllm`. |
| `smoke_gantt_viewer.sh` | Public | Browser smoke test for the Gantt viewer demo. | Exposed by `make gantt-viewer-smoke`. |
| `pull_repo.sh` | Public | Fast-forward pull helper. | Exposed by `make pull`. |
| `load_recording.py` | Supported | Inspect an attempt recording by call index. | Documented in `docs/trace-collect.md`. |
| `summarize_trace_cases.py` | Supported | Summarize selected trace cases for report/deck inputs. | Referenced by `docs/CURRENT_PLAN_trace_case_ppt.md`. |
| `analyze_vtune_aggregate.py` | Public | Aggregate VTune `summary.json`, `coarse.json`, and `fine.json` windows across traces. | Documented in `docs/vtune-profiling.md` and `docs/resource-measurement.md`. |
| `launch_kv_capstone.sh` | Supported | Convenience launcher for KV-cache capstone experiments. | Keep with KV experiment tooling. |
| `analyze_vtune.py` | Pending tracking | Untracked VTune aggregation script with overlapping purpose. | Prefer tracked `analyze_vtune_aggregate.py` for the documented path unless this narrower CSV/report flow is intentionally promoted. |
| `generate_report.py` | Pending tracking | Untracked Markdown report generator for `vtune_runs.csv`. | Repaired CLI: use `--input` and `--output`; depends on CSV from `analyze_vtune.py`. |

## Setup Scripts

`scripts/setup/` contains environment, data, and container setup helpers.
These are public operational entry points and should not be removed casually.

| Script | Purpose |
|---|---|
| `bootstrap.sh` | Fresh-server bootstrap for Miniconda/env/dependencies. |
| `install_deps.sh` | Dependency installation helper. |
| `configure_env.sh` | Environment configuration helper. |
| `download_model.sh` | Model download helper. |
| `clone_repos.sh` | Clone benchmark repositories referenced by task manifests. |
| `swebench_data.sh` | Download/prepare SWE-Bench Verified data. |
| `swe_rebench_data.sh` | Download/prepare SWE-rebench data. |
| `build_images.sh` | Build SWE container images. |
| `pull_swe_rebench_images.sh` | Pre-pull SWE-rebench Docker images. |
| `arm_setup.sh` | ARM/QEMU binfmt setup and checks. |
| `install_podman_vastai.sh` | Podman install helper for Vast.ai hosts. |
| `start_podman_socket.sh` | Podman socket helper. |
| `terminal_bench_server.sh` | Terminal-Bench server setup helper. |

## Experiment Support

`scripts/experiments/` contains active experiment support code, not disposable
scratch files. The LLC replay scripts are covered by focused tests. See
`docs/kunpeng-llc-experiments.md` for the full Kunpeng LLC validation and
replay workflow.

| Script | Status | Purpose |
|---|---|---|
| `chase.c` | Supported | Pointer-chase memory latency microbenchmark source used by the Kunpeng LLC slice validation probe. |
| `probe_llc_topology.py` | Supported | Probe Linux CPU/cache topology and generate placement plans. |
| `probe_kunpeng_llc_slices.py` | Supported | Validate inferred Kunpeng sub-LLC cluster sizes with real pointer-chase interference measurements. |
| `run_kunpeng_llc_replay.py` | Supported | Run replay-based Kunpeng LLC placement experiments with per-agent cpusets. |
| `run_kunpeng_llc_scaling.py` | Supported | Run topology-derived 1/2/4/8 scaling replay matrix. |
| `run_kunpeng_llc_agent_case.py` | Supported | Convenience live-agent placement runner; less preferred than replay because it can call live APIs. |
| `analyze_llc_replay_results.py` | Supported | Analyze LLC replay experiment outputs. |
| `summarize_llc_placement_runs.py` | Supported | Summarize placement runs into JSON/CSV/README artifacts. |
| `run_with_perf_stat.sh` | Supported | Wrap placement runs with system-level `perf stat`. |
| `run_scaling_hardcoded.py` | Legacy analysis | Older scaling runner with hardcoded CPU ids. | Refuses to run unless `--allow-hardcoded-placement` is passed. Use `run_kunpeng_llc_scaling.py` or `run_kunpeng_llc_replay.py` for topology-derived placement experiments. |

## Figure And Recording Analysis

`scripts/figures/` and `scripts/recoding_figures/` read existing traces or
recordings and generate figures or derived metrics. They are analysis utilities,
not benchmark execution paths.

Important tested modules in `scripts/recoding_figures/`:

| Script | Purpose |
|---|---|
| `recording_loader.py` | Load attention/MoE recording artifacts. |
| `metrics.py` | Shared metric helpers such as pairwise distances. |
| `followup_metrics.py` | Follow-up metrics for attention/MoE analyses. |
| `modal_followup_metrics.py` | Metrics used by Modal follow-up scripts. |
| `kv_eviction_metrics.py` | KV eviction analysis metrics. |
| `expert_cache_metrics.py` | Expert-cache analysis metrics. |
| `score_sparse_selection.py` | Score sparse attention keep-set selection. Requires `polars`. |
| `plot_sparse_segment_grid.py` | Plot sparse segment grids; imported by tests. |

Figure drivers:

| Script | Purpose |
|---|---|
| `make_figures.py` | Generate the base recording figures. |
| `make_curated14_figures.py` | Generate curated-14 figure set. |
| `plot_iter_distance.py` | Iteration distance plots. |
| `plot_layer_specialization.py` | Layer specialization plots. |
| `plot_alignment_scatter.py` | Attention/MoE alignment scatter. |
| `plot_research_summary.py` | Research summary figure. |
| `moe_phase_audit.py` | MoE phase audit helper. |
| `sparse_keep_sets.py` | Sparse keep-set helpers. |

`scripts/figures/` contains trace-level plotting utilities:

| Script | Purpose |
|---|---|
| `_real_trace_metrics.py` | Shared real-trace metric extraction. |
| `plot_execution_profile.py` | Plot execution profiles from synthetic or summarized inputs. |
| `plot_execution_profile_real.py` | Plot execution profiles from real traces. |
| `plot_resource_phase_alignment.py` | Plot resource/phase alignment. |
| `plot_tool_ratio_progress.py` | Plot tool ratio progress. |
| `plot_tool_ratio_progress_real.py` | Real-trace variant of tool ratio progress. |

Additional attention probe helpers at the top level:

| Script | Purpose |
|---|---|
| `plot_attention_fullseq_downsample.py` | Attention plotting helper with sample validation. |
| `probe_attention_maps.py` | Qwen/GLM chat rendering helpers for attention-map probing. |

## Modal Research Workspace

`scripts/modal_workspace/` contains Modal-backed or one-off research scripts.
They are useful, but they are not stable public CLIs. Tests should gate these
on optional dependencies such as `modal`.

Current files:

- `agent_attention_modal_followup.py`
- `agent_attention_followup.py`
- `agent_attention_moe_research.py`
- `agent_attention_moe_phase_denominator_audit.py`
- `agent_attention_residual_exploration.py`
- `agent_attention_a1_a5_posthoc.py`
- `curated14_analysis.py`
- `h2o_causal_failure_analysis.py`
- `kv_evict_100_run_analysis.py`
- `r2_leakage_sanity.py`

## Spikes

`scripts/spikes/` is exploratory validation code. Keep the Markdown notes with
the corresponding scripts because they record rationale and measured outcomes.

| Script | Purpose |
|---|---|
| `kv_cache_subclass_spike.py` / `.md` | Validate the Transformers KV-cache subclass contract. |
| `step3_random_smoke.py` | Random eviction smoke path. |
| `step4_streaming_smoke.py` | Streaming eviction smoke path. |
| `step6_h2o_smoke.py` | H2O eviction smoke path. |
| `step9_perf_microbench.py` / `step9_perf_results.md` | KV policy performance microbenchmark and results. |

## Test Notes

The repository is Linux-first for container execution. On Windows hosts,
collection-only pytest can still expose useful import problems, but tests that
exercise container execution, POSIX file locking, Modal, Tongyi vendor tools,
or optional analysis dependencies may be skipped unless the corresponding
runtime dependency is installed.
