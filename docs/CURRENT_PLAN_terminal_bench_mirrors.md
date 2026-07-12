# Terminal-Bench Mirror Switch Plan

## Goal

Add an explicit Terminal-Bench mirror switch for environments where direct
access to GHCR or Hugging Face is unstable, while preserving the pinned
Terminal-Bench dataset cache and recording the mirror configuration in traces.

## Checkpoints

- [x] Implement mirror configuration in `TerminalBenchRunner`.
- [x] Materialize a per-attempt task copy and rewrite only that copy's
      Dockerfile when mirrors are enabled.
- [x] Record mirror configuration in attempt summary and trace metadata.
- [x] Add focused tests for default-off behavior, URL rewriting, and metadata.
- [x] Update Terminal-Bench documentation.
- [x] Run targeted tests.
- [ ] Run mandatory evaluation-pipeline code review with a fresh sub-agent.
