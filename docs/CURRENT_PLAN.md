# Current Plan: Document benchmark run commands

## Goal

Document how to prepare and run SWE-Bench Verified, SWE-rebench,
Terminal-Bench, and the currently registered BFCL benchmarks.

## Verified current state

- All benchmarks run through `PYTHONPATH=src python -m trace_collect.cli`.
- OpenClaw runs require an explicit `--mcp-config`; use `none` to opt out.
- SWE-Bench Verified and SWE-rebench use task containers and have Make targets
  for dataset/repository preparation.
- Terminal-Bench requires Python 3.12+, the `terminal-bench` package, Docker,
  and uses the pinned local registry in
  `configs/benchmarks/terminal_bench_registry.json`.
- BFCL runs in host mode against an external read-only Gorilla checkout
  selected by `BFCL_REPO_PATH`.
- The registered BFCL slugs are:
  `bfcl-multi-turn-base`, `bfcl-multi-turn-long-context`, `bfcl-memory`, and
  `bfcl-web-search`.
- Before this change, README and benchmark documentation advertised obsolete
  `bfcl-v3` / `bfcl-v4` slugs.

## Planned changes

1. Add a README section with prerequisites and copy-paste commands for:
   SWE-Bench Verified, SWE-rebench, Terminal-Bench, and all four BFCL plugins.
2. Correct the README supported-benchmark table to list actual registered BFCL
   slugs.
3. Update `docs/benchmarks.md` so its catalog, BFCL descriptions, setup, and
   examples match the current configs and plugin registry.
4. Preserve benchmark-specific settings in YAML rather than introducing new
   CLI flags or hardcoded dataset details in runtime code.

## Verification

1. Cross-check every documented slug against `agents.benchmarks.REGISTRY`.
2. Cross-check setup commands against the Makefile, configs, and plugin
   validation logic.
3. Run Markdown/static consistency checks available in the repository and
   `git diff --check`.
4. Spawn a fresh reviewer sub-agent and resolve every finding before completion.

## Checkpoints

- [x] Read-only source/config/CLI audit
- [x] Human approval to edit README and benchmark documentation
- [x] Documentation implementation
- [x] Static verification
- [x] Mandatory independent review
- [x] Final diff audit

## Scope guard

- Modify only `README.md`, `docs/benchmarks.md`, and this plan file.
- Do not alter benchmark code, configs, datasets, traces, or the user's
  unrelated working-tree changes.
- Do not run benchmark experiments as part of this documentation-only task.

## Review audit

- First independent pass:
  - Major: BFCL YAML selection fields were described as active even though the
    collection path slices tasks directly.
  - Major: the generic `--sample 1` example was unsafe for BFCL memory chains.
  - Major: the docs implied collected traces were ready for official BFCL
    scoring even though no prediction export or evaluator integration exists.
  - Minor: the BFCL clone example did not show commit pinning/recording.
- Resolutions:
  - Documented that `selection_n` and `selection_seed` are not applied by the
    collection path.
  - Added a separate chain-safe `bfcl-memory` command and partial-run warning.
  - Explicitly stated that official BFCL scoring is unsupported by this CLI.
  - Added checkout and `rev-parse` steps while noting that the repository does
    not yet define a known-compatible Gorilla commit.
- Second independent pass:
  - All substantive checks passed.
  - Minor: checkpoint text still said fixes were in progress.
- Resolution:
  - Updated the checkpoint and recorded this audit.
- Final independent pass: CLEAN.

## Verification status

- Documented benchmark slugs match `agents.benchmarks.REGISTRY`.
- Make targets, CLI flags, runtime prerequisites, and BFCL limitations were
  cross-checked against source and configuration.
- `git diff --check`: passed.
- Final independent review: CLEAN.
- No benchmark experiment was run because this is a documentation-only change.
