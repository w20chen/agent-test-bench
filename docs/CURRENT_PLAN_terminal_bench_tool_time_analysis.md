# Terminal-Bench Tool Time Analysis Plan

## Objective

Analyze Terminal-Bench trace files under
`/data/share/datasets/agent_datasets/terminal-bench-p2` and quantify which tools
consume the most wall-clock time as a share of end-to-end trace duration.

## Access Status

- `5090` resolves via SSH config to `weitian@202.120.39.13:17722`.
- Key authentication failed in this environment.
- The user provided a password for SSH login.

## Analysis Scope

- Read traces only; do not mutate dataset files.
- Locate canonical trace JSONL files and run-level manifests.
- Use per-action `tool_exec` records with `data.duration_ms` as tool wall time.
- Use available trace span timestamps or manifest fields for end-to-end time.
- Aggregate by normalized tool name and report:
  - total tool wall time;
  - share of end-to-end time;
  - call count;
  - median, p90, and p95 per-call duration;
  - number of attempts/tasks containing the tool;
  - per-attempt share distribution.

## Intended Output

- A concise table ranking tools by total end-to-end share.
- A second table ranking tools by median and p95 duration to find consistently
  slow tools.
- CSV or JSON artifacts under `analysis_outputs/` if local data access becomes
  available.

## Next Step

Use non-interactive SSH password authentication or ask the user to run/enable a
working SSH session, then execute a read-only remote summarization command.

## Progress

- Installed `paramiko` into the current Python environment for non-interactive
  password SSH; no project dependency files were changed.
- Added `scripts/analyze_terminal_bench_tool_times_remote.py`.
- Ran read-only full scans of the remote dataset.
- Fixed the analysis to:
  - count only real `tool_exec` action records rather than auxiliary tool
    events;
  - deduplicate multiple trace copies by selecting one trace per `attempt_N`;
  - split generic `exec` commands by command string, including comment-prefixed
    shell blocks and common wrappers like `timeout`.
- Wrote local artifacts:
  - `analysis_outputs/terminal_bench_p2_tool_times.json`
  - `analysis_outputs/terminal_bench_p2_tool_times_tools.csv`

## Findings

- Scanned 561 candidate trace JSONL files and selected 187 canonical trace
  files after per-attempt deduplication.
- Found 172 attempts with tool records.
- Aggregate trace span time is 15.80 hours.
- Summed tool duration is 54.6% of aggregate trace span time.
- Remaining generic `exec` is only 8 calls / 0.64 seconds / 0.0011% of E2E.
- Top contributors by aggregate end-to-end share:
  - `exec-python`: 26.19%
  - `exec-pip`: 2.99%
  - `web_fetch`: 2.82%
  - `exec-tail`: 2.55%
  - `exec-curl`: 1.94%
  - `exec-apt`: 1.80%
  - `exec-conda`: 1.42%
  - `exec-git`: 1.35%
  - `exec-r`: 1.28%
  - `exec-make`: 1.12%

## Prediction Targets

- First priority: `exec-python`, because it dominates total tool time and has
  broad coverage.
- High-value specific targets: `web_fetch`, `exec-pip`, `exec-apt`,
  `exec-curl`, `exec-conda`, and `exec-make`.
- Outlier-heavy but less broad targets: `exec-tail`, `exec-r`, `exec-bash`,
  `exec-git`, `exec-systemctl`.
