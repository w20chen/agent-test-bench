# Runtime Prediction

Runtime prediction is framed as a knowledge-base problem: every tool execution
is described by a tool identity, workload features, duration distribution, and
whole-run resource profile. Tool-specific predictors may use different
algorithms, but they read and write through a shared Personal/Common knowledge
model.

The system currently supports `pytest`, `pip install`, and direct Python script
commands. It never runs an extra pre-execution probe such as
`pytest --collect-only`; predictions use only information already available
before the agent-selected command runs, plus history from prior real executions.

## Knowledge Model

There are two knowledge bases.

### Common KB

Common KB is a read-only cross-repo prior. It is assumed to be populated
offline and provided at runtime by setting `TOOL_RUNTIME_COMMON_KB` to a JSON
file. Online calibration never writes to Common KB.

Recommended fields for each Common prior:

```json
{
  "priors": {
    "pytest/run_tests/101-500-tests": {
      "tool_name": "pytest",
      "tool_family": "test_runner",
      "operation": "run_tests",
      "workload_bucket": "101-500-tests",
      "environment_bucket": {
        "os": "linux",
        "arch": "x86_64",
        "cpu_class": "generic",
        "memory_bucket": "8-32gb"
      },
      "duration": {
        "p50_s": 40.0,
        "p75_s": 65.0,
        "p90_s": 90.0,
        "p95_s": 120.0,
        "mean_s": 52.0,
        "std_s": 20.0,
        "sample_count": 1000
      },
      "resources": {
        "load_class": "cpu_parallel",
        "expected_cores": 3.2,
        "peak_cores_p90": 6.0,
        "peak_memory_mb": 1024.0,
        "disk_read_mb_p90": 128.0,
        "disk_write_mb_p90": 64.0,
        "io_class": "light"
      },
      "quality": {
        "min_samples": 50,
        "outlier_policy": "winsorized_p01_p99",
        "source_version": "common-v1",
        "privacy_level": "aggregate",
        "confidence": "medium"
      }
    }
  }
}
```

Lookup is progressive:

```text
tool_name / operation / workload_bucket
-> tool_name / operation
-> tool_family / operation / workload_bucket
-> tool_family / operation
-> generic_process / operation
-> generic_process
```

### Personal KB

Personal KB is private run/repo knowledge written only after a real tool
execution completes. It stores exact normalized command history and tool/family
aggregates:

```text
run_dir/
  runtime_knowledge_db/<instance>/personal_runtime_knowledge.json
```

The exact-command entries are eligible for duration prediction. Tool/family
aggregates are retained for resource analysis and future scheduling signals,
but they do not override Common KB for a new normalized command. This preserves
the intended cold-start behavior: if Personal does not match the command, use
Common.

The Personal KB stores:

- `repo_id`
- `normalized_command`
- tool identity: `tool_name`, `tool_family`, `operation`
- bounded duration samples
- last command features, such as package count, script path, or test count
- whole-run resource summaries when profiler/scheduler data is available
- timestamps and sample counts

## Prediction Flow

All predictors follow the same high-level flow:

```text
Tool Call
  -> tool-specific recognizer and feature extractor
  -> tool-specific personal history
  -> repo-family history
  -> unified Personal KB exact command
  -> read-only Common KB
  -> prediction payload
  -> real tool execution
  -> Personal KB update from actual duration and whole-run resources
```

The unified prediction payload exposes:

- `prediction_recommended_s`
- `prediction_recommended_method`
- `prediction_reliability`
- `prediction_knowledge_p50_s`
- `prediction_knowledge_p90_s`
- `prediction_common_p50_s`
- `prediction_common_p90_s`
- `runtime_knowledge_prediction`

`runtime_knowledge_prediction` uses the shared shape:

```text
duration_p50_s
duration_p90_s
load_class
expected_cores
peak_memory_mb
confidence
prediction_source
sample_count
```

## Tool Predictors

Tool-specific predictors remain intentionally different where the tool gives
useful non-probing features.

### `pip install`

Source: `src/trace_collect/package_runtime_prediction.py`

Recognized forms include `pip install`, `pip3 install`, and
`python -m pip install`. The predictor normalizes package order and requirement
inputs where possible.

Priority:

1. Last run for the same normalized command.
2. Repo-family last run for the same normalized command.
3. Package-count estimate from personal per-package history.
4. Unified Personal KB exact command.
5. Common KB package-install prior.
6. Tool global median.

Successful standalone installs update the existing pip history and Personal KB.
Compound shell commands, `||` chains, prefix work, and follow-up segments do not
update pip history.

### Python Script

Source: `src/trace_collect/python_script_runtime_prediction.py`

Recognized forms include `python script.py`, `python3 -u script.py`, and
`timeout 20 python3 script.py`. The predictor excludes `python -c`, heredocs,
and `python -m ...` because those are different execution modes.

Priority:

1. Last run for the same normalized command.
2. Repo-family last run for the same normalized command.
3. Median for the same script path.
4. Median for the same script basename.
5. Unified Personal KB exact command.
6. Common KB script-execution prior.
7. Tool global median.

Successful standalone script runs update the existing script history and
Personal KB.

### `pytest`

Source: `src/trace_collect/pytest_runtime_prediction.py`

`pytest` has the most specialized predictor because test nodeids are meaningful
units of historical work. The predictor still does not probe.

Allowed pre-execution inputs:

- normalized pytest command text
- instance and repo-family history
- pytest nodeids explicitly written in the shell command, such as
  `pytest tests/test_a.py::test_one`

Forbidden pre-execution inputs:

- `pytest --collect-only`
- importing test modules to discover tests
- walking or expanding files to infer hidden pytest collection
- any extra tool execution before the agent-selected command

Explicit nodeid handling:

| Coverage | Meaning |
|----------|---------|
| `unknown` | no explicit nodeids in the command text |
| `explicit_only` | all positional selection tokens contain `::` |
| `partial` | explicit nodeids are mixed with files, directories, or selector flags |

Explicit selectors are conservative. A token such as
`tests/test_a.py::TestClass` can expand to multiple tests, and parametrized
tests can expand too. Therefore:

- `pre_execution_test_set_known` is true only when explicit tokens exactly
  match historical collected nodeids.
- partial or non-exact explicit matches are reported as
  `prediction_explicit_nodeid_lower_bound_s`, not as a full-command
  recommended prediction.

Priority:

1. Same normalized command with stable historical collected count.
2. Same normalized command with unknown previous count.
3. Repo-family last run for the same normalized command.
4. Exact explicit nodeids with sufficient exact nodeid history.
5. Non-probing nodeids with sufficient node/file history.
6. Unknown-test fallback when it represents the full known non-probing set.
7. Unified Personal/Common runtime knowledge only for full-command cold start.
8. No recommended prediction.

Per-test prediction cascades:

```text
exact nodeid history
-> same-file median
-> project-wide median
-> unavailable
```

The unknown-test variant inserts the median of previously unseen tests between
same-file and project-wide medians.

## Resource Updates

Resource calibration is based on the whole tool execution, not the first
1-2 seconds. When `tool_profiler` or `tool_scheduler` is enabled, their
`final_profile` records are converted into Personal KB resource summaries:

- wall time
- average, P50, P90, and peak effective cores when available
- peak RSS in MiB
- disk read/write in MiB when available
- load class from the final whole-run profile
- profiler sample count

Failed profiler rows are skipped and do not count as Personal KB updates.
Common KB is not modified.

## Artifacts

Tool-specific artifacts remain under each attempt:

```text
attempt_N/
  pip_runtime/
    history.json
    predictions.jsonl
    iter_0001_exec-pip_<id>/pending.json
    iter_0001_exec-pip_<id>/prediction.json
  python_script_runtime/
    history.json
    predictions.jsonl
  pytest_runtime/
    history.json
    predictions.jsonl
    iter_0017_exec-pytest_<id>/
      pending.json
      prediction.json
      pytest_runtime.json
      instrumentation.json
  tool_profiles_summary.json
  tool_scheduler_summary.json
```

Shared history and knowledge live under the run directory:

```text
run_dir/
  pip_runtime_db/<instance>/history.json
  python_script_runtime_db/<instance>/history.json
  pytest_runtime_db/<instance>/history.json
  runtime_knowledge_db/<instance>/personal_runtime_knowledge.json
```

## Reusing Knowledge

Use `--run-id` to continue writing to an existing run directory, or copy the
`*_runtime_db` and `runtime_knowledge_db` directories into a new run directory
before starting a new experiment.

Instance history is task-specific. Repo-family history helps sibling instances
from the same repository. Common KB remains external, read-only, and suitable
for cold starts.
