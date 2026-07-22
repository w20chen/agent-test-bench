Objective: Remove pre-execution pytest probing from runtime prediction.

User requirement:
- collect-only overhead is unacceptable.
- Running a tool in advance to discover concrete test state is unacceptable.
- Collecting history from the real pytest invocation is acceptable.
- Predictions must be based on historical information only.

Scope:
- `src/trace_collect/pytest_runtime_prediction.py`
- pytest runtime prediction tests/docs that describe collect-only behavior.
- Keep real-run pytest plugin instrumentation because it only observes the command
  the agent already chose to run.

Plan:
1. Remove `pytest --collect-only` execution from the pre-tool preparation path. - completed
2. Compute pre-run predictions only from bounded history plus normalized command. - completed
   - `last_run` and `family_last_run` remain valid.
   - Command-level median fallback may use historical command durations.
   - Node/test-count predictions require no pre-run nodeids and should stay null
     before execution unless future non-probing metadata is added.
3. Preserve post-run history collection from the injected pytest plugin. - completed
4. Update artifacts so collect-only fields are explicit null/disabled, not measured. - completed
5. Update tests and documentation to enforce that preparation does not spawn
   subprocesses. - completed
6. Run focused tests. - completed
7. Use a fresh reviewer sub-agent before treating the evaluation-path change as done. - completed

Review notes:
- Fresh reviewer found one major artifact-semantics issue: unknown
  pre-execution test count was represented as 0.
- Fixed by writing `pre_execution_test_set_known: false` and null counts.
- Reviewer also noted a misplaced docs sentence and stale plan status; both were
  fixed.

Verification:
- `python -m pytest tests\test_pytest_runtime_prediction.py tests\test_analyze_pytest_prediction.py --basetemp .pytest-tmp-root`
  passed: 37 tests.
- `python -m py_compile src\trace_collect\pytest_runtime_prediction.py tests\test_pytest_runtime_prediction.py scripts\analyze_pytest_prediction.py`
  passed.

Research integrity notes:
- No benchmark-specific cases.
- No oracle/test-label leakage.
- No hidden pre-execution workload.
- Successful real pytest invocations remain the only source of new per-node
  runtime history.

---

Objective: Use explicit pytest nodeids from command text without probing.

User requirement:
- If the agent's shell command explicitly names pytest nodeids, those nodeids
  are inference-time information and may be used for prediction.
- Still must not run collect-only, inspect pytest collection, expand files, or
  infer hidden tests by probing.
- Partial matches are allowed but must be marked as partial so they are not
  mistaken for complete command runtime coverage.

Plan:
1. Add a pure command-text parser that extracts positional pytest args
   containing `::`. - completed
2. Mark explicit-nodeid coverage as `explicit_only` only when all positional
   selection args are nodeid-like command tokens; broad selectors mark
   coverage as `partial`. - completed
3. Feed explicit nodeids into historical per-test prediction. - completed
4. Downgrade partial and non-exact explicit-nodeid predictions so they do not
   overclaim total command runtime. - completed
5. Add tests for exact explicit nodeids, mixed partial nodeids, class selector
   conservatism, `--basetemp`, and no subprocess probing. - completed
6. Update docs and run a fresh reviewer plus focused tests. - completed

Review notes:
- Fresh reviewer found that `::` selectors can expand to multiple pytest items.
  Fixed by distinguishing explicit command tokens from known collected items.
- Fresh reviewer found partial sums were exported as recommended total runtime.
  Fixed by exposing `prediction_explicit_nodeid_lower_bound_s` and leaving
  `prediction_recommended_s` unset for partial/non-exact command selectors.
- Added option-value handling for `--basetemp` and related pytest flags.

Verification:
- `python -m pytest tests\test_pytest_runtime_prediction.py tests\test_analyze_pytest_prediction.py --basetemp .pytest-tmp-root`
  passed: 44 tests.
- `python -m py_compile src\trace_collect\pytest_runtime_prediction.py tests\test_pytest_runtime_prediction.py tests\test_analyze_pytest_prediction.py scripts\analyze_pytest_prediction.py`
  passed.
- `python -m ruff check src\trace_collect\pytest_runtime_prediction.py tests\test_pytest_runtime_prediction.py tests\test_analyze_pytest_prediction.py scripts\analyze_pytest_prediction.py`
  passed.

---

Objective: Design common/personal runtime knowledge bases for tool time and resources.

User requirement:
- Treat tool duration and resource usage as knowledge-base entries.
- Assume a populated common KB will be provided later; plan the schema fields
  now.
- Implement the personal KB after plan approval.
- Resource/load statistics must summarize the whole tool execution, not only
  the first 1-2 seconds.

Common KB planning:
1. Define shared dimensions for all tools:
   - tool identity: `tool_name`, `tool_family`, `operation`, `workload_bucket`.
   - environment bucket: OS/container family, architecture, CPU class, memory
     bucket, cache/network availability when known.
   - duration distribution: `duration_p50_s`, `duration_p75_s`,
     `duration_p90_s`, `duration_p95_s`, `duration_mean_s`, `duration_std_s`,
     `sample_count`, `last_updated_at`.
   - whole-run resource distribution: `cpu_core_seconds_p50`,
     `avg_cores_p50`, `peak_cores_p90`, `peak_rss_mb_p90`,
     `disk_read_mb_p90`, `disk_write_mb_p90`, `net_rx_mb_p90`,
     `net_tx_mb_p90`, `io_wait_or_io_class`, `load_class`.
   - quality controls: `min_samples`, `outlier_policy`, `source_version`,
     `privacy_level`, `confidence`.
2. Define tool-specific feature fields:
   - pytest/test runners: explicit nodeid count, historical collected count
     bucket, selector type, file/dir/nodeid selection mode, xdist workers,
     known plugin flags, per-test history availability.
   - pip/package install: package count bucket, requirement-file hash bucket,
     wheel/cache hint, source-build hint, editable/install mode.
   - python script: script basename/path bucket, argv complexity bucket,
     input-size bucket, timeout wrapper, module/script mode.
   - build tools/make: target bucket, parallelism `-j`, changed-file count
     bucket, source tree size bucket.
   - generic process: command basename, argv count bucket, input path type,
     input size bucket, parallelism hints, network/file-processing hints.
3. Define fallback lookup order:
   - exact tool operation/workload bucket.
   - tool family bucket.
   - generic tool bucket.
   - no common prior.

Personal KB implementation plan:
1. Add a unified schema module for observations and predictions.
2. Add a personal KB storage layer over JSON files under the existing run
   directory, reusing current instance/repo-family history locations where
   possible.
3. Store complete private details only in personal KB:
   - repo id, normalized command, command features, explicit nodeids or script
     identity, observed duration, exit status, full-run resource summary, and
     timestamps.
4. Add adapters for existing pytest, pip, and python-script predictors rather
   than rewriting their algorithms first.
5. Emit a unified prediction payload:
   - `duration_p50_s`, `duration_p90_s`, `load_class`, `expected_cores`,
     `peak_memory_mb`, `confidence`, `prediction_source`, `warnings`.
6. Update personal KB after the real tool run completes, using full-run resource
   summaries from existing profiler/scheduler/resource artifacts when present.
7. Keep common KB read-only initially; define the interface for later
   aggregate/de-identified updates but do not upload or mutate common data yet.
8. Add tests for schema validation, personal KB append/load/merge, fallback
   selection, and compatibility with current pytest/pip/python-script history.

Checkpoint:
- Stop after this plan for user approval before implementation.

Implementation progress after user approval:
1. Added `src/trace_collect/runtime_knowledge.py` for unified prediction,
   whole-run resource summaries, read-only Common lookup, and bounded Personal
   KB updates. - completed
2. Added Common fallback to pip, Python-script, and pytest prediction chains.
   Tool-specific and repo-family personal history remain higher priority. -
   completed
3. Added unified Personal KB updates in pip, Python-script, and pytest
   finalizers after successful real tool completion. - completed
4. Added profiler/scheduler final-profile ingestion into Personal KB summaries
   in `trace_collect.collector`. - completed
5. Updated runtime prediction documentation with Common fields and lookup
   order. - completed
6. Added focused tests in `tests/test_runtime_knowledge.py`. - completed
7. Run reviewer gate and fix findings. - completed
8. Run final focused verification. - completed

Reviewer findings resolved:
- Unified Personal duration prediction now requires an exact normalized
  command, so tool/family aggregates do not suppress Common cold-start
  fallback.
- Profiler and scheduler final profiles now write to the same run-level
  Personal KB path that predictors read.
- Failed profile rows no longer count as Personal KB updates.
- Common prior parsing now preserves valid zero values instead of using
  truthiness.

Verification:
- `python -m py_compile src\trace_collect\runtime_knowledge.py src\trace_collect\package_runtime_prediction.py src\trace_collect\python_script_runtime_prediction.py src\trace_collect\pytest_runtime_prediction.py src\trace_collect\collector.py tests\test_runtime_knowledge.py`
  passed.
- `python -m ruff check src\trace_collect\runtime_knowledge.py src\trace_collect\package_runtime_prediction.py src\trace_collect\python_script_runtime_prediction.py src\trace_collect\pytest_runtime_prediction.py src\trace_collect\collector.py tests\test_runtime_knowledge.py`
  passed.
- `python -m pytest tests\test_runtime_knowledge.py tests\test_package_runtime_prediction.py tests\test_python_script_runtime_prediction.py tests\test_pytest_runtime_prediction.py --basetemp .pytest-tmp-root`
  passed: 95 tests.
