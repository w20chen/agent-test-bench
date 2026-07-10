# Current Plan: Backward-Compatible Exec Classification

## Goal

Improve single-winner `exec-*` classification for previously unseen terminal
tools without changing the trace schema or breaking existing known-command
classification.

## Guardrails

- Keep the single `tool_name = exec-<winner>` representation.
- Preserve every existing known-command classification and priority unless a
  failing regression test demonstrates that it is incorrect.
- Preserve preclassified `exec-*` names; repeated classification must be
  idempotent.
- Derive unknown labels only from the executable position, never from arbitrary
  arguments.
- Normalize unknown executables to a safe basename and strict lowercase slug;
  otherwise retain plain `exec`.
- Do not add benchmark-specific commands or tune priorities using held-out
  benchmark outcomes.
- Do not introduce a new dependency without explicit approval.

## Implementation Phases

1. Add regression tests for legacy behavior, safe unknown executable labels,
   invalid tokens, command chains/pipelines, and idempotence.
2. Make the smallest classifier changes needed to satisfy those tests, while
   retaining the existing public functions and trace schema.
3. Run the focused classifier tests and relevant trace logger/import/rewrite
   tests.
4. Pass the mandatory independent review gate; fix all critical, major, and
   minor findings and re-run verification.
5. Replace/remove the experimental v2 file only if explicitly in scope and
   safe after review; otherwise leave it untracked for the user to compare.

## Checkpoint

- Phase 1 complete: added compatibility, open-world executable, malformed
  input, compound-command, and idempotence tests.
- Phase 2 complete: implemented conservative command-position parsing, safe
  unknown executable slugs, general database/data-analysis categories, and
  preservation of existing `exec-*` names.
- Phase 3 complete:
  - `python -m py_compile src/trace_collect/exec_classifier.py tests/test_exec_classifier.py`
    passed.
  - Final focused classifier suite passed: 172 tests.
  - Trace logger, Claude Code import, and attempt pipeline integration suite
    passed: 43 tests, with 1 existing skip.
- Phase 4 complete: independent review and all re-review iterations are clean.
- Phase 5 complete: the untracked experimental `exec_classifier_v2.py` was
  deliberately left untouched for the user; it is not imported by the code.

## Independent Review Gate

- Initial review found wrapper/`xargs` operand parsing, `command -v/-V`, and
  shell-control keyword issues; all were fixed with regression tests.
- Re-review found missing documented wrapper aliases and incorrect GNU `xargs`
  optional-argument boundaries; all were fixed and tested across bare,
  attached, equals, and whitespace-separated forms.
- Final independent re-review found no critical, major, or minor issues.
- Research-integrity review found no benchmark-specific tuning, oracle
  leakage, or unexplained dataset coupling.
- Final `git diff --check` passed with line-ending warnings only.

---

# Archived Plan: Kunpeng LLC Slice Validation Scripts

## Goal

Add a low-level Kunpeng LLC/cluster validation script that can reproduce the
manual pointer-chase checks without hardcoding CPU ids. The script must verify
candidate sub-LLC cluster sizes from measured interference rather than assuming
that each LLC slice always maps to four physical cores.

## Guardrails

- Do not run long or real hardware experiments in this pass.
- Do not introduce dataset-specific or result-tuned behavior.
- Do not hardcode CPU ids such as `0,2,4,...` in the scripts.
- Do not treat inferred 4-core groups as validated hardware facts unless the
  pointer-chase probe supports that mapping.
- Preserve the existing benchmark plugin architecture and simulate CLI.
- Reuse existing topology and replay command builders where possible.

## Implemented

- Added `scripts/experiments/probe_kunpeng_llc_slices.py`.
- The script:
  - derives physical-core representative CPUs from sysfs topology;
  - checks candidate cluster sizes with real pointer-chase interference;
  - defaults to `--candidate-cluster-sizes 2,4,8`;
  - treats `--aggressors-per-run` as a maximum so 2-core candidates can still
    be measured with matched one-aggressor controls;
  - preserves full commands, raw stdout/stderr, topology, CSV measurements,
    and JSON manifest outputs;
  - computes verdicts from per-victim normalized deltas rather than pooled raw
    medians across heterogeneous cores.
- Added `tests/test_kunpeng_llc_slice_probe.py`.
- Added `docs/kunpeng-llc-experiments.md`.
- Updated `docs/scripts.md` with the new supported script and doc link.

## Verification

- `python -m py_compile scripts\experiments\probe_kunpeng_llc_slices.py tests\test_kunpeng_llc_slice_probe.py`
  passed.
- `python -m pytest tests\test_kunpeng_llc_slice_probe.py -q -p no:cacheprovider --basetemp .tmp-tests\pytest-llc-slice-probe`
  passed 9 tests after final reviewer fix.
- `python -m py_compile scripts\experiments\probe_kunpeng_llc_slices.py scripts\experiments\probe_llc_topology.py scripts\experiments\run_kunpeng_llc_scaling.py tests\test_kunpeng_llc_slice_probe.py tests\test_llc_topology_probe.py`
  passed.
- `python -m pytest tests\test_kunpeng_llc_slice_probe.py tests\test_llc_topology_probe.py -q -p no:cacheprovider --basetemp .tmp-tests\pytest-llc-all`
  passed 26 tests after reviewer fixes.
- `python scripts\experiments\probe_kunpeng_llc_slices.py --help` passed after
  reviewer fixes.
- `python -m pytest tests\test_llc_replay_analysis.py -q -p no:cacheprovider --basetemp .tmp-tests\pytest-llc-analysis`
  passed 1 test.

## Review Gate

- Independent reviewer found no critical issues.
- Reviewer found major issues:
  - `Path("./chase")` lost the `./` prefix in command construction;
  - verdicts could claim support without same-cluster slowdown over baseline;
  - verdicts pooled raw medians across victims instead of normalizing per
    victim;
  - defaults did not check smaller-than-4 candidates.
- Fixes implemented:
  - command construction now preserves relative `./chase` execution;
  - verdicts require same-cluster slowdown over baseline and a normalized gap
    over other-cluster same-LLC controls;
  - summaries include per-victim effects;
  - default candidate sizes are `2,4,8`.
- Re-review found one additional major issue:
  - failed aggressor processes were not invalidating the trial.
- Final fix implemented:
  - measurements now record `aggressor_returncodes`;
  - victim or aggressor nonzero exits mark the measurement invalid/skipped so
    it cannot enter the summary.
- Final re-review completed with no critical or major issues remaining.

## Remaining Notes

- This pass will not run the real pointer-chase sweep locally because the
  current workspace is Windows and the target commands require Linux
  `numactl`, `taskset`, and the compiled `chase` binary.

## Follow-up: Chase Microbenchmark Source

Goal: add the missing `chase` source code used by
`probe_kunpeng_llc_slices.py`, so the pointer-chase workload is reproducible
from the repository rather than copied from shell history.

Guardrails:

- Keep `chase` a real memory-latency microbenchmark, not a stub or simulated
  output generator.
- Preserve the exact output contract consumed by the Python probe:
  `ns_per_access <float> accesses <integer>`.
- Keep CPU and NUMA binding outside the binary; the experiment scripts should
  continue to use `taskset` and `numactl`.
- Do not run long hardware experiments in this pass.

Implemented:

- Added `scripts/experiments/chase.c`.
- The binary:
  - allocates one cache-line-sized node per pointer-chase entry;
  - builds a deterministic shuffled ring to limit prefetch-friendly access;
  - runs for the requested duration;
  - emits `ns_per_access <float> accesses <integer>`.
- Documented build and smoke-test commands in
  `docs/kunpeng-llc-experiments.md`.
- Registered the source file in `docs/scripts.md`.
- Added a lightweight source-contract test in
  `tests/test_kunpeng_llc_slice_probe.py`.

Verification:

- `python -m pytest tests\test_kunpeng_llc_slice_probe.py -q -p no:cacheprovider --basetemp .tmp-tests\pytest-chase-source`
  passed 10 tests.
- `gcc -O3 -std=c11 -Wall -Wextra -o .tmp-tests\chase.exe scripts\experiments\chase.c`
  passed on the local Windows toolchain using the portable aligned-allocation
  fallback.
- `.tmp-tests\chase.exe 1 1` printed `ns_per_access ... accesses ...`.

Review:

- Independent reviewer found no critical or major issues.
- Minor issues fixed:
  - added a lightweight compiler barrier in the pointer-chase loop;
  - added access-counter overflow protection;
  - malformed successful `chase` output is now recorded as an invalid/skipped
    measurement instead of aborting the whole probe;
  - added an optional gcc compile test when a compiler is available;
  - changed the documentation's example `accesses` value to a multiple of the
    benchmark batch size.
