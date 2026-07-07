# Current Plan: Trace Case PPT

## Goal

Create a concise, editable PowerPoint deck that summarizes representative
benchmark traces as visual LLM/tool-call timelines, including tool and LLM
timing and short annotations of what each case is doing.

## Source Trace Directories

- `C:\Users\29068\Desktop\agent-tool-predictor\swe-bench-verified\django__django-10880\attempt_1`
- `C:\Users\29068\Desktop\agent-tool-predictor\swe-bench-verified\astropy__astropy-7336\attempt_1`
- `C:\Users\29068\Desktop\agent-tool-predictor\swe-rebench\12rambau__sepal_ui-411\attempt_1`
- `C:\Users\29068\Desktop\agent-tool-predictor\swe-rebench\AzureAD__microsoft-authentication-library-for-python-77\attempt_1`
- `C:\Users\29068\Desktop\agent-tool-predictor\terminal-bench\causal-inference-r\attempt_1`
- `C:\Users\29068\Desktop\agent-tool-predictor\terminal-bench\query-optimize\attempt_1`
- `C:\Users\29068\Desktop\agent-tool-predictor\deep-research-bench\51\attempt_1`
- `C:\Users\29068\Desktop\agent-tool-predictor\deep-research-bench\66\attempt_1`

## Planned Workflow

1. Inspect each attempt directory structure and identify the canonical trace
   artifacts, HTML outputs, timing metadata, tool-call logs, and final result
   files.
2. Extract a compact per-case summary:
   - task/workload identity;
   - major LLM turns;
   - important tool calls;
   - LLM/tool durations where available;
   - concise "what happened" narrative.
3. Design a reusable trace visual grammar suitable for PPT:
   - LLM as one lane;
   - tools as grouped lane/events;
   - width or labels encode duration;
   - highlighted calls show semantically important operations;
   - each case gets a plain-language caption.
4. Build an editable PPTX deck with a cover, overview, and one slide per case
   or paired cases if density demands it.
5. Render/inspect previews, check for legibility/collisions, and revise.

## Checkpoints

- [x] Persist plan without overwriting unrelated active plan.
- [x] Get permission to read trace artifacts outside the current repository.
- [x] Complete read-only artifact audit.
- [x] Confirm slide structure if the trace artifacts do not expose enough
      timing/detail for the requested visuals.
- [x] Generate editable PPTX and rendered previews.
- [x] Run visual QA and report paths, commands, and unresolved limitations.

## Read-Only Audit Summary

All eight attempts include `trace.jsonl`, `tool_calls.json`, `resources.json`,
`results.json`, and `run_manifest.json`. Six also include `trace_viz.html`;
the two Terminal-Bench cases do not, but their canonical trace artifacts are
available.

Timing is available from canonical `trace_format_version=5` action spans.
The deck should use recorded LLM/tool timing only:

| Case | Actions | Elapsed (s) | LLM (s) | Tool (s) | Notable tools |
|---|---:|---:|---:|---:|---|
| `django__django-10880` | 64 | 143.4 | 80.5 | 58.3 | `exec-python`, `exec-grep`, `read_file`, `write_file`, `exec-pytest` |
| `astropy__astropy-7336` | 57 | 117.0 | 85.5 | 27.1 | `exec-python`, `read_file`, `exec-grep`, `edit_file`, `exec-pip` |
| `12rambau__sepal_ui-411` | 56 | 405.8 | 171.0 | 176.9 | `read_file`, `exec-python`, `exec-grep`, `edit_file`, `exec-git`, `exec-pytest` |
| `AzureAD__microsoft-authentication-library-for-python-77` | 129 | 576.7 | 259.2 | 291.3 | `read_file`, `exec-python`, `edit_file`, `web_fetch`, `web_search`, `exec-curl` |
| `causal-inference-r` | 87 | 730.1 | 172.3 | 659.3 | `exec-R`, `exec-cat`, `exec-git`, `exec-apt` |
| `query-optimize` | 24 | 687.1 | 137.3 | 547.4 | `exec-sqlite3`, `read_file`, `write_file`, `exec-head` |
| `deep-research-bench/51` | 46 | 446.0 | 112.0 | 383.9 | `web_search`, `web_fetch`, `read_file`, `write_file` |
| `deep-research-bench/66` | 48 | 346.0 | 88.3 | 395.1 | `web_search`, `web_fetch` |

Proposed deck grammar:

- one cover slide;
- one overview slide comparing the eight cases by LLM/tool time share;
- one slide per case, with a compressed horizontal trace timeline;
- highlights for semantically important tools: code execution/tests/search/fetch/edit/read;
- short "what happened" caption based on actual tool sequence and result
  metadata.

## Scope Guard

- Do not modify trace source directories.
- Do not run benchmark experiments.
- Do not fake timing data; if timing is missing, label it as unavailable or
  derive only from recorded timestamps with documented assumptions.
- Do not add project dependencies.

## Implementation Summary

Generated artifacts:

- Data extraction script: `scripts/summarize_trace_cases.py`
- Deck builder: `reports/trace_case_ppt/build_trace_case_deck.py`
- Extracted data: `reports/trace_case_ppt/scratch/case_summary.json`
- Editable PPTX: `reports/trace_case_ppt/output/trace_case_studies.pptx`
- Lightweight PNG previews:
  `reports/trace_case_ppt/scratch/previews/slide_*.png`

Additional requested update:

- Replaced the original Terminal-Bench examples with `causal-inference-r` and
  `query-optimize`.
- Both selected Terminal-Bench traces are successful, multi-action/multi-turn
  traces with recorded tool time greater than recorded LLM time.
- Added an all-case atlas slide that places every case on one page with a
  mini LLM/tool timeline, timing, important tools, and short case action
  description.

Verification completed:

- PPTX has 12 slides.
- PPTX contains no embedded media files, so slide visuals are native editable
  PowerPoint shapes/text, not screenshots.
- Internal shape boundary check found and fixed one overview-slide overflow.
- Final boundary check reports zero off-slide shapes and zero placeholder text.
- Preview count is 12.

Reviewer gate:

- First reviewer sub-agent failed due to stream disconnection before producing
  findings.
- Second independent reviewer sub-agent started for the same strict review.
- Second reviewer sub-agent also failed due to stream disconnection before
  producing findings. Manual structural/visual checks above were completed,
  but the independent review gate did not successfully return findings.
