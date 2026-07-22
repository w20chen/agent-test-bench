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

Objective: Make pytest runtime instrumentation invisible to the agent.

User requirement:
- Agent should not be able to perceive or disable runtime prediction/capture
  machinery through ordinary shell environment inspection.
- Real pytest invocations may still be observed, but the mechanism must stay
  outside the agent-visible command surface.

Plan:
1. Stop injecting pytest runtime variables into the whole shell environment. -
   completed
2. Inject the runtime variables only at the pytest process boundary so commands
   such as `env` do not expose them and `unset PYTEST_PLUGINS && pytest ...`
   cannot disable capture. - superseded
3. Add regression tests for agent-invisible environment and unset resilience. -
   completed
4. Run focused tests and a fresh reviewer because this touches the evaluation
   runtime path. - completed

Review outcome:
- Review found that both direct process-boundary env injection and PATH shims
  remain agent-visible or disableable through ordinary shell/pytest mechanisms.
- Final direction: do not inject pytest plugins, `PYTEST_PLUGINS`, or PATH
  shims into agent tool commands. Use outer tool-call duration for successful
  pytest command-level history, and use node-level history only when it already
  exists from prior artifacts.
- Follow-up review found stale injection helper APIs and misleading Personal KB
  update status. The helper APIs/tests were removed, missing node timing is now
  recorded as `runtime_observation_status: outer_tool_timing_only`, and
  Personal KB updates now reflect command-level duration learning.
- Final review found no remaining major or critical issues. Residual risk:
  prediction quality is coarser without node-level online capture, by design.

Verification so far:
- `python -m pytest tests\test_runtime_knowledge.py tests\test_package_runtime_prediction.py tests\test_python_script_runtime_prediction.py tests\test_pytest_runtime_prediction.py tests\test_shell_pytest_runtime_invisibility.py --basetemp .pytest-tmp-root`
  passed: 96 passed.
- `python -m py_compile src\agents\openclaw\tools\shell.py src\trace_collect\pytest_runtime_prediction.py tests\test_shell_pytest_runtime_invisibility.py tests\test_pytest_runtime_prediction.py`
  passed.
- `python -m ruff check src\agents\openclaw\tools\shell.py src\trace_collect\pytest_runtime_prediction.py tests\test_shell_pytest_runtime_invisibility.py tests\test_pytest_runtime_prediction.py`
  could not run in the current Python environment: `No module named ruff`.

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

---

Objective: Build a Common runtime KB from SWE-rebench P1 traces on SSH host 5090.

User requirement:
- Source data lives on host `5090` at
  `/data/share/datasets/agent_datasets/swe-rebench-p1`.
- Build Common KB from existing historical data only.
- Do not run cases, do not run pre-execution probes, and do not mutate Common
  from online calibration.
- Keep private repo/path/command details out of Common output.

Plan:
1. Inspect the remote directory structure and identify available trace/runtime
   artifacts. - completed
2. Add a reproducible builder script that reads local or remote-copied trace
   artifacts and emits Common KB JSON. - completed
3. Aggregate only de-identified fields:
   - tool name, tool family, operation, workload bucket
   - duration P50/P75/P90/P95/mean/std/sample count
   - whole-run resource summaries from final profiler/scheduler profiles when
     present
   - confidence/quality metadata - completed
4. Add tests with small synthetic artifact fixtures for the builder logic. -
   completed
5. Run reviewer gate because this touches evaluation/prediction artifacts. -
   completed
6. Run focused verification and report the generated Common KB path. -
   completed

Remote data notes:
- SSH source: `weitian@202.120.39.13:17722`.
- Remote root:
  `/data/share/datasets/agent_datasets/swe-rebench-p1`.
- Downloaded existing artifacts only; no case execution and no pre-run probes.
- The dataset contains `tool_calls.json`, `resources.json`, `trace.jsonl`,
  and related attempt artifacts. No per-tool `profile.jsonl` files were found,
  so resource priors are reconstructed by matching each `tool_calls.json`
  timestamp/end-timestamp interval against the sibling `resources.json` samples.
- Local generated output:
  `artifacts/runtime_common_kb_swe_rebench_p1.json`.
- Current generated KB has 53 prior buckets from 197 `tool_calls.json` files.
- Reviewer found missing family/generic fallback buckets, possible multi-source
  duplicate duration counting, missing failed-profile filtering, root name
  leakage, and coarse pytest workload buckets.
- Builder was fixed to emit tool/family/generic buckets, deduplicate by
  attempt/id where possible, use profile duration only for attempts without
  prediction/tool_call duration artifacts, skip failed tool calls/profiles,
  omit root names, and parse pytest counts from historical result previews.
- Follow-up review found remaining issues in source-priority deduplication,
  structured failure filtering, global generic fallback, and test coverage.
- Builder was fixed again to suppress tool-call duration when a prediction
  duration exists for the same attempt/tool/operation, preserve repeated
  same-tool tool calls when no higher-priority source exists, filter structured
  failure fields, and emit the final `generic_process` fallback prior.
- Builder now adds whole-interval resource/load statistics from timestamped
  `resources.json` samples while leaving the existing duration extraction
  strategy unchanged. Resource samples use the strict inclusive tool interval
  `[timestamp, end_timestamp]`.
- Resource output includes CPU cores, peak RSS, disk I/O, network I/O, context
  switches, and selected CPU micro-architecture counters. Memory bandwidth
  fields are intentionally ignored because those measurements are not reliable.
- Reviewer found that duration suppression could accidentally suppress resource
  extraction, profile resources could double-count tool-call interval
  resources, and padded resource windows could contaminate adjacent tool calls.
  Fixed by separating duration suppression from resource extraction, using
  profile resources only as a fallback when no tool-call resource interval
  exists for the same attempt/tool/operation, and removing the window padding.
- Follow-up review found that shared prediction/tool-call IDs still suppressed
  tool-call resources, and that repeated same-tool calls needed less coarse
  profile fallback handling. Fixed by allowing shared-ID tool calls to continue
  resource extraction while suppressing only duration, and by suppressing
  profile resources exactly by shared observation ID or coarsely only when a
  single successful tool call exists for that attempt/tool/operation.
- Final review found no remaining major or medium issues. Residual low-risk
  boundary: repeated same-tool calls without shared profile IDs may retain
  profile fallback resources to avoid dropping an unmatched invocation.
- Final regenerated Common KB has 92 prior buckets from 197 `tool_calls.json`
  files; 85 buckets include resource samples from strict matched tool
  intervals.

Verification:
- `python -m py_compile scripts\build_runtime_common_kb.py tests\test_build_runtime_common_kb.py`
  passed.
- `python -m ruff check scripts\build_runtime_common_kb.py tests\test_build_runtime_common_kb.py`
  passed.
- `python -m pytest tests\test_build_runtime_common_kb.py --basetemp .pytest-tmp-root`
  passed: 12 tests.
