# Current Plan: Pytest Runtime Prediction Prototype

## Objective

Build a minimal trace-collect-only prototype for pytest tool calls that records
per-test runtimes and evaluates whether summing historical per-test medians
predicts the next pytest total better than simple baselines.

## Source Findings

- OpenClaw tool execution enters through `AgentRunner._execute_tools()` and
  `ExecTool.execute()`.
- `exec` commands are classified by `trace_collect.exec_classifier`; common
  `pytest` and `python -m pytest` forms become `exec-pytest`.
- Canonical trace records are written by `TraceCollectorHook` in
  `src/agents/openclaw/_session_runner.py`.
- Each `tool_exec` action currently includes `tool_name`, `tool_call_id`,
  serialized `tool_args`, `tool_result`, `duration_ms`, `success`, and wall
  timing fields.
- The previous pytest script capture work writes attempt-local artifacts under
  `attempt_N/pytest_scripts/` and already has parser/helpers for common pytest
  commands.
- Tool output contains an `Exit code: N` trailer, so exit status can be parsed
  without changing pytest semantics.

## Design Choice

Use offline replay validation first:

1. Let the pytest tool command run normally.
2. Add a temporary pytest plugin via environment/command wrapping only for
   matching pytest invocations so it emits per-node runtime JSON.
3. After execution, compute predictions using the history as it existed before
   this run.
4. Save actuals, predictions, errors, and test details.
5. Update the bounded history with this run's per-test durations.

This avoids a separate collect-only pass and keeps benchmark semantics close to
the current execution path. If instrumentation fails, the original pytest
command must still run and the trace collector must not fail the benchmark.

## Planned Files

- `src/trace_collect/pytest_runtime_prediction.py`
  - command recognition/wrapping;
  - lightweight pytest plugin source;
  - per-test result parsing;
  - history medians/fallbacks;
  - prediction and artifact writing.
- `src/agents/openclaw/tools/shell.py`
  - enable trace-collect pytest instrumentation for classified `exec-pytest`
    commands when configured by environment.
- `src/agents/openclaw/_session_runner.py`
  - finalize prediction artifacts after the tool action timing is known.
- `src/agents/openclaw/eval/runner.py`, `src/trace_collect/runtime/entrypoint.py`,
  `src/trace_collect/collector.py`
  - pass attempt-local pytest prediction directories in trace collect mode.
- `scripts/analyze_pytest_prediction.py`
  - aggregate MAE/MAPE and long-run metrics.
- focused tests under `tests/`.
- `docs/trace-collect.md`
  - document artifact layout and analysis command.

## Artifact Layout

```text
attempt_N/pytest_runtime/
  history.json
  predictions.jsonl
  iter_0017_exec-pytest_<tool_call_id>/
    pytest_runtime.json
    prediction.json
    instrumentation.json
```

## Review Gate

Because this touches the evaluation/trace collection path, run an independent
reviewer after implementation and fix any critical or major findings before
finalizing.

## Progress

- Added `trace_collect.pytest_runtime_prediction` for instrumentation command
  wrapping, temporary pytest plugin creation, history maintenance, prediction
  computation, and artifact writing.
- Wired OpenClaw trace collect so `exec-pytest` invocations receive an
  internal attempt-local runtime artifact directory.
- Added realtime collect output lines with actual duration, three predictions,
  and per-test relative error.
- Added `scripts/analyze_pytest_prediction.py` for MAE/MAPE summaries and
  optional CSV export.
- Added focused tests, including a small project run twice to confirm history
  updates and second-run predictions.
- Validation so far:
  - `python -m py_compile ...` for modified Python files passed.
  - `python -m pytest tests\test_pytest_runtime_prediction.py
    tests\test_pytest_script_capture.py tests\test_session_runner_actions.py
    tests\test_openclaw_eval_runner.py tests\test_collector_task_container_runtime.py
    -q -p no:cacheprovider --basetemp .tmp-tests\pytest-runtime-focused`
    passed with 26 tests.
  - `python scripts\analyze_pytest_prediction.py
    .tmp-tests\pytest-runtime-focused` produced one valid smoke row.
  - `git diff --check` passed, with line-ending warnings only.

## Review Findings And Fixes

- Reviewer found two major issues:
  - Runtime instrumentation failure could affect the original pytest command
    because the first implementation wrapped the command with a shell wrapper.
  - Collected but not executed tests (`outcome="notrun"`) could be added to
    history with `duration_s=0.0`, polluting future predictions.
- Fixes applied:
  - Removed command wrapping. `ExecTool` now runs the original command string
    and only adds pytest plugin environment variables to the subprocess
    environment.
  - The generated pytest plugin catches its own hook/write failures, and the
    plugin source is compiled before injection; prepare failures disable
    instrumentation for that run.
  - `update_pytest_history()` skips `outcome="notrun"` tests while preserving
    them in the current run artifact.
  - The analysis script now prints per-method `N`, and docs call out exact
    command matching plus wall-time vs per-report duration limitations.
- Re-validation:
  - Modified-file `py_compile` passed.
  - Focused pytest passed with 29 tests.
  - `git diff --check` passed, with line-ending warnings only.
- Re-review found one remaining major issue: commands that set `PYTHONPATH`
  could hide the injected plugin module while still inheriting `PYTEST_PLUGINS`.
- Fixed conservatively by disabling runtime instrumentation for pytest
  commands that explicitly assign/export `PYTHONPATH`; focused pytest and
  `git diff --check` still pass.
- Third review found no critical or major issues. Remaining minor limitation:
  the `PYTHONPATH` detector is conservative and may skip some harmless commands,
  which reduces collection coverage but does not alter benchmark semantics.

## Reliability Update

- Added schema v5 reliability and recommended prediction fields.
- Added command `collected_counts` history, updated only for runs with observed
  test nodes.
- Added a rule-based selector: stable same-command Last Run, high-coverage
  Per-Test, medium node-or-file Per-Test, then low-confidence Unknown fallback.
- Updated realtime stdout, prediction artifacts, analyzer output, CSV export,
  and trace-collect docs.
- Validation so far:
  - `python -m py_compile src\trace_collect\pytest_runtime_prediction.py
    scripts\analyze_pytest_prediction.py tests\test_pytest_runtime_prediction.py
    tests\test_analyze_pytest_prediction.py`
  - `python -m pytest tests\test_pytest_runtime_prediction.py
    tests\test_pytest_script_capture.py tests\test_session_runner_actions.py
    tests\test_openclaw_eval_runner.py tests\test_analyze_pytest_prediction.py
    -q -p no:cacheprovider --basetemp .tmp-tests\pytest-reliability-final`
    passed with 43 tests.
- Review found one major issue: old command history without collected counts
  made Last Run reliability too confident. Fixed by downgrading that path to
  `medium`.
- Reviewer minor findings fixed:
  - missing reliability in old schema rows is counted as `unavailable`;
  - realtime stdout docs include recommended/reliability;
  - added tests for old history, medium file fallback, old-schema analyzer
    buckets, and JSONL recommended fields.

## Pip Runtime Prediction Update

Objective: add a minimal trace-collect-only runtime predictor for `pip install`
commands, focused only on wall-clock duration.

Scope:

- Add `src/trace_collect/package_runtime_prediction.py` for command recognition,
  conservative normalization, bounded history, and prediction artifact writing.
- Support `pip install ...`, `pip3 install ...`, and `python -m pip install ...`.
- Use only simple predictors: same-command Last Run, package-count baseline, and
  tool-global median fallback.
- Record compact attempt-local artifacts under `attempt_N/pip_runtime/`.
- Wire a thin optional hook through OpenClaw trace collection without modifying
  `ExecTool` or changing command execution semantics.
- Add focused unit tests and run a targeted pytest suite.

Review gate: because the trace/evaluation collection path is touched, run a
fresh reviewer before finalizing and fix any critical or major findings.

Progress:

- Added compact pip runtime prediction artifacts with pre-execution prediction
  snapshots and post-execution actual/error finalization.
- Wired optional `capture_pip_runtime` through CLI, collector, task-container
  entrypoint, and OpenClaw session runner.
- Kept `ExecTool` and command execution semantics unchanged.
- Reviewer found major issues around failed installs entering history and
  hindsight-prone post-execution prediction. Fixed by:
  - updating history only when shell success is consistent with `Exit code: 0`
    and the command has no `||` fallback chain;
  - snapshotting predictions in `pending.json` before execution;
  - disabling post-execution recomputation when the pending snapshot is missing;
  - preserving per-call `working_dir` instead of overwriting it with project
    root.
- Validation:
  - `python -m py_compile src\trace_collect\package_runtime_prediction.py
    src\agents\openclaw\_session_runner.py tests\test_package_runtime_prediction.py
    tests\test_session_runner_actions.py`
  - `python -m pytest tests\test_package_runtime_prediction.py
    tests\test_session_runner_actions.py tests\test_collector_task_container_runtime.py
    tests\test_openclaw_runtime_selection.py -q -p no:cacheprovider
    --basetemp .tmp-tests\pip-runtime-expanded` passed with 30 tests.
  - `git diff --check` passed with Windows line-ending warnings only.

## Pip Shared History Update

Objective: make pip runtime prediction useful across attempts by storing the
learned history in a run-level database while preserving per-attempt artifacts.

Plan:

- Keep `attempt_N/pip_runtime/` as the local prediction artifact directory.
- Add a shared instance-scoped history directory such as
  `run_dir/pip_runtime_db/<instance_id>/`.
- Read predictions from the shared history before each pip invocation.
- Update the shared history only after successful, unmasked pip installs.
- Record the history path in `pending.json` and `prediction.json` for audit.
- Add focused tests for cross-attempt reuse and disabled capture behavior.

Reviewer follow-up:

- Avoided mounting `run_dir` into task containers. Task-container attempts use
  an attempt-local history mirror; the host seeds it before execution and
  merges successful prediction rows back afterward.
- Made JSON writes atomic and added stale lock recovery for shared history.
- Scoped collect-level shared history by `instance_id` to avoid mixing unrelated
  task images/repos.
- Added a short hash suffix to the instance scope directory to avoid sanitized
  `instance_id` collisions.

## Pytest Collect-Only Prediction Update

Objective: restore live per-test pytest prediction without hindsight leakage by
collecting nodeids before the original pytest tool command runs.

Plan:

- Run a side-effect-minimized `pytest --collect-only` command during pytest
  runtime prediction preparation, scoped to the tool working directory.
- Use the pre-execution collected nodeids to compute test-count, per-test, and
  unknown-test-fallback predictions before tool execution.
- Record collect-only overhead in `pending.json`, `prediction.json`, compact
  `predictions.jsonl`, realtime stdout, analyzer stdout, and analyzer CSV
  output.
- Keep the original pytest tool command itself unchanged; record the
  collect-only phase as measured prediction overhead.
- Run targeted tests and mandatory independent review because this touches the
  trace/evaluation collection path.

## Python Script Runtime Prediction Update

Objective: add an instance-scoped runtime predictor for `python *.py` tool
commands, with history shared across attempts for the same benchmark instance.

Plan:

- Add `src/trace_collect/python_script_runtime_prediction.py`.
- Recognize Python script invocations such as `python foo.py`,
  `python3 -u eval.py`, `timeout 20 python3 /path/script.py`, and commands with
  preceding `cd` / environment / venv activation segments.
- Keep original command execution unchanged.
- Write attempt-local artifacts under `attempt_N/python_script_runtime/`.
- Store shared history under `run_dir/python_script_runtime_db/<instance_id>/`,
  seeded into each attempt and merged back after successful prediction rows.
- Predict from same instance history only, with fallbacks:
  - same normalized invocation last run;
  - same script path median;
  - same script basename median;
  - instance-global python-script median.
- Update history only for successful commands without `||` fallback chains.
- Add focused unit and wiring tests, then run mandatory independent review
  because this touches trace/evaluation collection.

Progress:

- Added `trace_collect.python_script_runtime_prediction` with conservative
  `python *.py` recognition, pre-execution prediction snapshots, bounded
  history, shared-history seed/merge helpers, and realtime summary output.
- Wired OpenClaw TraceCollectorHook, EvalRunner, task-container entrypoint,
  collector, CLI, and trace metadata.
- Added docs for artifact layout and CLI disable flag.
- Added focused tests for recognition, fallback order, failed/or-chain history
  exclusion, cross-attempt history reuse, trace hook artifact writing, and
  task-container seed/merge wiring.
- Validation so far:
  - `python -m py_compile` for modified Python files passed.
  - `python -m pytest tests\test_python_script_runtime_prediction.py
    tests\test_session_runner_actions.py
    tests\test_collector_task_container_runtime.py -q -p no:cacheprovider
    --basetemp .tmp-tests\python-script-runtime-focused` passed with 19 tests.
  - `python -m pytest tests\test_openclaw_runtime_selection.py
    tests\test_collector_openclaw_metadata.py
    tests\test_package_runtime_prediction.py -q -p no:cacheprovider
    --basetemp .tmp-tests\python-script-runtime-expanded` passed with 29 tests.

Review findings and fixes:

- Reviewer found one major issue: compound shell commands could contaminate
  python-script history by recording full tool wall time for `python script.py`
  plus follow-up commands, or by masking failure through `; true` / pipelines.
- Fixed by preserving artifacts but excluding commands with follow-up shell
  segments from history updates; commands with `||` remain excluded.
- Reviewer minor findings fixed:
  - `python -V foo.py` / `python --version foo.py` are no longer recognized as
    script runs;
  - shared-history merge skips rows already finalized directly against shared
    history to avoid double-add helper misuse.
- Re-review found the same contamination class for nontrivial prefix work such
  as `make data && python eval.py`. Fixed by tracking prefix work separately:
  only `cd`, venv activation, comments/blanks, and pure env-assignment prefix
  segments are considered history-safe.
- Final re-review found unquoted newlines and single `&` were not treated as
  shell separators. Fixed by splitting on newline and background `&`, with
  regression tests for newline follow-ups and background commands.
- Re-validation:
  - `python -m pytest tests\test_python_script_runtime_prediction.py -q
    -p no:cacheprovider --basetemp
    .tmp-tests\python-script-runtime-regression3` passed with 16 tests.
  - `python -m pytest tests\test_python_script_runtime_prediction.py
    tests\test_session_runner_actions.py
    tests\test_collector_task_container_runtime.py
    tests\test_openclaw_runtime_selection.py
    tests\test_collector_openclaw_metadata.py
    tests\test_package_runtime_prediction.py -q -p no:cacheprovider
    --basetemp .tmp-tests\python-script-runtime-final` passed with 58 tests.
  - Modified-file `py_compile` passed.
  - `git diff --check` passed with Windows line-ending warnings only.

## Rerun Completed Attempts Update

Objective: allow an existing run directory to append new attempts for instances
that already have a completed attempt, while preserving the default resume
behavior.

Plan:

- Add an explicit CLI flag `--rerun-completed`.
- Default behavior remains unchanged: completed/exhausted instances are skipped
  on resume.
- When `--rerun-completed` is set, selected instances run again and receive the
  next `attempt_N` directory from `next_attempt_number()`.
- Keep incomplete/error-only instances runnable under both modes.
- Document the flag with the instance-scoped runtime DB workflow.
- Add focused tests for default skip behavior and opt-in rerun behavior.

Progress:

- Added `--rerun-completed` to the collect CLI and threaded it through
  `collect_traces()` / `_run_scaffold_tasks()`.
- Default resume behavior remains unchanged.
- Rerun mode leaves completed/exhausted instances eligible, allocates the next
  `attempt_N`, records `rerun_completed` in trace run config, and keeps image
  prefetch aligned with the rerun eligibility set.
- Added tests for default completed skip and explicit rerun of both
  `completed` and `exhausted` terminal statuses.
- Independent reviewer found no critical/major issues; minor findings were
  addressed by fixing rerun-mode prefetch, adding exhausted coverage, and
  clarifying CLI/docs wording.
- Validation:
  - `python -m py_compile src\trace_collect\cli.py src\trace_collect\collector.py
    tests\test_collector_runtime_mode.py`
  - `python -m pytest tests\test_collector_runtime_mode.py
    tests\test_collector_task_container_runtime.py
    tests\test_collector_openclaw_metadata.py -q -p no:cacheprovider
    --basetemp .tmp-tests\rerun-completed-final` passed with 33 tests.
  - CLI parser smoke for default and enabled `rerun_completed` passed.
  - `git diff --check` passed with Windows line-ending warnings only.

## Trace Collect Parameter Documentation Update

Objective: clarify how `--instance-ids`, `--skip`, `--sample`, `--run-id`,
`--rerun-completed`, and `--concurrency` compose, without adding excessive
implementation detail.

Progress:

- Added a concise "Run Directory and Attempt Rules" section to
  `docs/trace-collect.md`.
- Documented the three-step mental model:
  task selection first, resume/rerun eligibility second, per-task attempt count
  third.
- Shortened the resume section to a small status table and linked it back to
  the combination rules.
- Replaced duplicated concurrency/selection explanation with a pointer to the
  new combination rules.
