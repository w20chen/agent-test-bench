# Current Plan

Objective: Implement conservative cross-instance runtime-history sharing for pip, python-script, and pytest prediction without changing agent decision logic.

Steps:
1. Inspect current prediction history formats and runner/collector wiring. - completed
2. Add family-level history buckets alongside existing instance-level buckets. - completed
3. Seed attempts conservatively from instance history when available, otherwise family history; merge new successful observations into both. - completed
4. Extend pytest prediction to support shared history roots if needed. - completed
5. Run focused tests and a strict review pass before finalizing. - pending

