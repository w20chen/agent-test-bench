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
