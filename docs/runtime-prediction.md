# Runtime Prediction

Predicts execution time for `pip install`, Python script, and `pytest`
commands using bounded history collected from prior real tool executions.

## History Sharing

Prediction history lives under the run directory:

```text
run_dir/
  pip_runtime_db/<instance>/history.json
  python_script_runtime_db/<instance>/history.json
  pytest_runtime_db/<instance>/history.json
  runtime_kb/
    repo/<repo>/
      pip/history.json
      python_script/history.json
      pytest/history.json
```

Each tool has instance-level history for the same task and repo-family history
for sibling tasks in the same repository. Histories are bounded to 5 entries
per key and protected by file locks.

## `pip install` Prediction

Source: `src/trace_collect/package_runtime_prediction.py`

Recognized commands include `pip install`, `pip3 install`, and
`python -m pip install`. Successful standalone installs update history.

Prediction methods:

| Method | Uses |
|--------|------|
| Last Run | Most recent duration for the same normalized command |
| Family Last Run | Same command from sibling instances in the same repo |
| Package Count | Package count multiplied by historical per-package median |
| Global Median | Median of successful pip install durations |

## Python Script Prediction

Source: `src/trace_collect/python_script_runtime_prediction.py`

Recognized commands include `python script.py`, `python3 -u script.py`, and
`timeout 20 python3 script.py`. It excludes `python -c`, heredocs, and
`python -m ...`.

Prediction methods:

| Method | Uses |
|--------|------|
| Last Run | Most recent duration for the same normalized command |
| Family Last Run | Same command from sibling instances in the same repo |
| Script Path Median | Median for the same script path |
| Basename Median | Median for the same filename |
| Global Median | Median of successful Python script runs |

## `pytest` Prediction

Source: `src/trace_collect/pytest_runtime_prediction.py`

`pytest` prediction does not run `pytest --collect-only` or any other
pre-execution probe. The pre-tool path only parses the command text and reads
bounded history. A temporary pytest plugin observes the real pytest invocation
that the agent already chose to run; only successful real runs update future
history.

### Pre-Execution Inputs

Allowed inputs:

- normalized pytest command text
- instance and repo-family history
- pytest nodeids explicitly written in the shell command, such as
  `pytest tests/test_a.py::test_one`

Forbidden inputs:

- `pytest --collect-only`
- importing test modules to discover tests
- walking/expanding files to infer hidden pytest collection
- any extra tool execution before the agent-selected command

### Explicit Nodeids

The parser extracts only positional pytest arguments containing `::`. It skips
known option values such as `--ignore`, `--deselect`, `--basetemp`, `-k`, and
`-m`.

Coverage is recorded explicitly:

| Coverage | Meaning |
|----------|---------|
| `unknown` | no explicit nodeids in the command text |
| `explicit_only` | all positional selection tokens contain `::` |
| `partial` | command mixes explicit nodeids with broader selectors, files, directories, or selector flags |

Explicit selectors are not automatically treated as collected pytest items:
`tests/test_a.py::TestClass` can expand to multiple tests, and parametrized
functions can expand as well. Therefore:

- `pre_execution_test_set_known` is true only when explicit tokens also match
  exact historical collected nodeids.
- partial or non-exact explicit matches are exposed as
  `prediction_explicit_nodeid_lower_bound_s`, not as the recommended total
  runtime.

### Prediction Methods

| Method | Uses |
|--------|------|
| Last Run | Most recent duration for the same normalized pytest command |
| Family Last Run | Same command from sibling instances in the same repo |
| Test Count | `len(nodeids) * project_test_median` only when nodeids are known from non-probing information |
| Per-Test | sum of historical per-nodeid predictions plus overhead, only for exact historical item matches |
| Explicit Nodeid Lower Bound | partial/non-exact explicit-nodeid sum; not a total runtime prediction |
| Unknown Fallback | cold-start per-test fallback when nodeids are available from a non-probing source |

Per-test prediction cascades:

```text
exact nodeid history -> same-file median -> project-wide median -> unavailable
```

The unknown-test variant inserts the median of previously unseen tests between
same-file and project-wide medians.

### Recommendation Rules

The recommended prediction is intended to estimate full command runtime.
Priority is:

1. Same normalized command with stable historical collected count: Last Run,
   high reliability.
2. Same normalized command with unknown previous count: Last Run, medium.
3. Same command from repo-family history: Family Last Run, medium.
4. Exact explicit nodeids with sufficient exact nodeid history: Per-Test.
5. Non-probing nodeids with sufficient node/file history: Per-Test.
6. Unknown fallback when it represents the full known non-probing set.
7. Otherwise no recommended prediction (`coldstart` or `error`).

Partial explicit-nodeid sums are recorded as lower bounds and are not selected
as `prediction_recommended_s`.

### Artifacts

```text
attempt_N/pytest_runtime/
  history.json
  predictions.jsonl
  iter_0017_exec-pytest_<id>/
    pending.json
    prediction.json
    pytest_runtime.json
    instrumentation.json
```

Useful fields include:

- `pre_execution_nodeid_source`
- `pre_execution_nodeid_coverage`
- `pre_execution_test_set_known`
- `pre_execution_explicit_nodeid_count`
- `prediction_nodeid_coverage`
- `prediction_explicit_nodeid_lower_bound_s`
- `prediction_recommended_s`
- `prediction_recommended_method`
- `prediction_reliability`

Realtime output does not include collect-only/probe overhead:

```text
[pytest-predict] #17 tests=32 | 220.4s actual ->last=205.1s(+6.9%) fam=210.0s(+5.0%) count=?(?) per=?(?) unk=?(?) | medium
```

Analyze a run:

```bash
PYTHONPATH=src python scripts/analyze_pytest_prediction.py traces/swe-rebench/<model>/<run>
```

The analyzer may still report legacy collect-only overhead for old traces; new
policy traces should leave that value unset.

## Reusing Knowledge Across Runs

Use `--run-id` to continue writing to an existing run directory, or copy the
`*_runtime_db` directories into a new run directory before starting a new
experiment. History transfer is limited by scope: instance history is
task-specific, while repo-family history can help sibling instances from the
same repository.
