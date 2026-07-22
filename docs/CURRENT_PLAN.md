Objective: Fix runtime prediction knowledge-base correctness and scheduler signal bugs.

Scope:
- Runtime prediction history for pip, Python scripts, and pytest.
- Repo/family-scoped knowledge sharing and instance retry history.
- Tool scheduler resource-signal correctness where it affects predictions.
- Focused tests for the fixed behavior.

Plan:
1. Confirm knowledge-base scope design and keep repo/family as the sharing boundary. - completed
2. Fix pip prediction history pollution from compound commands and duplicate shared merges. - completed
3. Fix pytest prediction history pollution and collect-only environment-prefix handling. - completed
4. Fix tool_scheduler PMU stderr parsing, idle snapshot consistency, and history persistence semantics. - completed
5. Align summary tests/docs and run focused tests with an in-workspace pytest basetemp. - completed
6. Run strict independent review before finalizing. - completed

Review follow-up:
- Moved scheduler history from attempt-local default to repo-level runtime KB when launched by the collector.
- Clarified scheduler as a dry-run placement recommender, not an affinity-applying scheduler.
- Strengthened tests for instance/family merge coverage and scheduler history persistence.

Notes:
- Do not introduce benchmark-specific heuristics.
- Only write complete, inference-time-available runtime observations into predictive history.
- Compound commands may still emit artifacts, but must not update tool-specific duration history when elapsed time includes unrelated work.
