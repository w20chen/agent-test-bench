# Current Plan

Objective: Audit and fix correctness bugs in `tool_profiler` and its OpenClaw integration.

Status: completed

## Scope

- Keep `tool_profiler` as a profiling/measurement wrapper; do not add scheduling behavior.
- Preserve the profile JSONL schema unless a narrow diagnostic addition is required.
- Fix only confirmed bugs in command semantics, output preservation, process cleanup, and measurement validity.
- Avoid benchmark-specific behavior or workload shortcuts.

## Steps

1. Inspect `prototype/tool_profiler` runner/CLI/sampler/output and OpenClaw wrapping. - completed
2. Identify confirmed bugs and choose minimal general fixes. - completed
3. Implement fixes. - completed
4. Add regression tests for fixed behavior. - completed
5. Run focused tests with workspace-local pytest temp directories. - completed
6. Run independent strict review because this touches tool execution paths. - completed
7. Address review findings and summarize final state. - completed
