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
