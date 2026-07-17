# Current Plan: Capture Pytest Test Scripts

## Objective

During trace collection, whenever an OpenClaw `exec` tool call classifies as
`exec-pytest`, save the pytest test script files referenced by that invocation
as first-class attempt artifacts for later timing analysis.

## Scope

- Support trace collect mode only.
- Store artifacts under each attempt directory, next to `trace.jsonl`.
- Preserve the agent's executed pytest command and per-tool timing metadata.
- Do not alter benchmark execution semantics, pytest commands, or task images.
- Do not use benchmark labels or oracle fields to decide what to capture.

## Artifact Layout

For each pytest invocation:

```text
attempt_N/pytest_scripts/
  iter_0006_exec-pytest_<tool_call_id>/
    command.sh
    manifest.json
    files/
      tests/test_example.py
```

The manifest records the command, iteration, tool_call_id, resolved target
paths, copied file metadata, capture warnings, and final tool timing fields
(`duration_ms`, `ts_start`, `ts_end`, `success`, `action_id`).

## Implementation Steps

1. Add a focused `trace_collect.pytest_script_capture` helper:
   - classify tool calls with the existing exec classifier;
   - parse pytest command positional path targets;
   - resolve paths relative to the command working directory;
   - copy discoverable pytest `.py` scripts into a stable artifact directory;
   - write/update manifest files and a JSONL index.
2. Wire `TraceCollectorHook.before_execute_tools()` to capture before execution
   so file contents reflect the pytest invocation point.
3. Wire `TraceCollectorHook.after_iteration()` to update timing metadata after
   the tool completes.
4. Enable the capture directory from `SessionRunner` as
   `trace_file.parent / "pytest_scripts"` with project root equal to the
   OpenClaw project/tool workspace (`/testbed` in task-container collection).
5. Add focused tests for command parsing, file capture, manifest timing update,
   and hook integration.
6. Run focused tests.
7. Run the mandatory independent review gate because this touches trace
   collection/evaluation artifacts.

## Checkpoints

- Do not run real benchmark collection in this pass.
- If pytest target parsing is ambiguous, preserve the command and warnings in
  the manifest rather than guessing from benchmark ground truth.

## Progress

- Added `trace_collect.pytest_script_capture` for command parsing, file copying,
  manifests, and index rows.
- Wired OpenClaw trace collection so SWE benchmark collect runs pass
  `attempt_N/pytest_scripts` explicitly, including task-container mode where
  the raw trace lives under `_task_container_runtime/openclaw`.
- Captures occur before tool execution; final manifests are updated with
  per-tool trace timing after execution.
- Capture and finalize errors now leave failure manifests/index rows instead of
  silently disappearing.
- Commands classified as `exec-pytest` without a literal pytest executable
  (for example Django test runner commands) produce an empty manifest with a
  warning rather than discovering unrelated files.

## Review Gate

- Independent reviewer found two major issues:
  - task-container mode wrote artifacts under the runtime raw-trace directory;
  - capture/finalize failures were swallowed silently.
- Both major issues were fixed.
- Reviewer also noted Django-test classification as a minor ambiguity; fixed by
  warning and capturing no files unless a pytest executable is present.
- Re-review found one remaining major issue: `capture_failed` manifests could
  be overwritten as `complete` during timing finalization.
- Fixed by preserving `capture_failed` while adding timing fields and writing a
  single indexed failure row with `duration_ms`.
- Final independent re-review found no critical or major issues remaining.
