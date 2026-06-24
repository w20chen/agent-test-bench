# Current Plan: Preserve DeepSeek reasoning content

## Goal

Prevent OpenClaw runs from failing when a thinking-mode provider returns an
assistant tool-call message whose `reasoning_content` field is present but is
the empty string.

## Evidence and root cause

- Reproducing trace:
  `C:\Users\user\Desktop\20260624T053526\12rambau__sepal_ui-411\attempt_1`
- Iterations 0-9 returned non-empty reasoning content with tool calls.
- Iteration 10 returned a tool call with `reasoning_content: ""`.
- The message was preserved in OpenClaw history, but
  `LLMProvider._sanitize_request_messages` removed the empty field before the
  iteration 11 DeepSeek request.
- DeepSeek rejected that request with HTTP 400:
  `The reasoning_content in the thinking mode must be passed back to the API.`

## Planned changes

1. Make empty `reasoning_content` handling provider-aware instead of deleting
   it unconditionally in the shared sanitizer.
2. Preserve an explicitly returned empty value for DeepSeek request history,
   especially assistant messages containing tool calls.
3. Retain the existing compatibility behavior for providers that require empty
   reasoning fields to be omitted.
4. Add focused regression tests that cover both policies and the exact
   assistant-tool-call shape observed in the trace.

## Verification

1. Run the focused provider serialization regression tests.
2. Run the relevant OpenClaw provider and runner test files.
3. Run formatting/lint checks for modified Python files if available.
4. Spawn a fresh reviewer sub-agent before treating the evaluation-pipeline
   change as complete.
5. Fix every review finding and re-run the affected tests.

## Checkpoints

- [x] Read-only trace and source diagnosis
- [x] Human approval to implement
- [x] Implementation and focused tests added
- [ ] Focused tests executed (blocked: no local Python runtime)
- [x] Mandatory independent review completed
- [x] Final verification and diff audit

## Scope guard

- Do not modify the external trace.
- Do not modify the user's existing `docs/getting-started.md` worktree change.
- Do not run a new benchmark experiment before the mandatory review gate.

## Review audit

- Reviewer: fresh-context sub-agent `Lorentz`
- First pass:
  - Major: streaming parsing collapsed explicit empty reasoning content to
    `None`.
  - Minor: direct DeepSeek endpoint detection did not accept a trailing DNS
    dot.
  - Minor: plan checkpoint state was stale.
- Resolution:
  - Track whether streaming reasoning content was explicitly present, keeping
    `""` distinct from an absent field.
  - Normalize a trailing DNS dot while retaining exact hostname matching.
  - Add non-streaming and streaming parse-to-replay regression tests.
  - Add endpoint acceptance and lookalike-host rejection tests.
- Second pass:
  - Minor: add explicit coverage for lookalike hostname rejection.
- Resolution:
  - Added the requested regression test.
- Final pass: CLEAN.

## Verification status

- `git diff --check`: passed.
- Independent static code review: CLEAN.
- Pytest: not run because this Windows host exposes only Microsoft Store
  Python placeholders and has no project Python, Conda, Docker, or WSL runtime.
