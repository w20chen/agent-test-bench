# Vendor Notes: Tongyi-DeepResearch

Vendored implementation of the Alibaba-NLP/DeepResearch ReAct scaffold,
pinned to support Ralplan R3 (see `docs/CURRENT_PLAN.md`).

## Upstream pin

- **Upstream**: https://github.com/Alibaba-NLP/DeepResearch
- **Pinned commit SHA**: `f72f75d8c3eb842f2bbbab096a12206ff66e270f`
- **Pinned date**: 2026-02-27 (upstream commit date, "fix bug")
- **License**: Apache-2.0 (full text at `vendor/NOTICE`)
- **Local clone used for vendoring**: `/private/tmp/deepresearch-recon/DeepResearch`

## Freeze protocol

Pin is **frozen** from Phase A through Phase F completion (R3 Pre-mortem #4).
Re-pin is allowed only for upstream blocker bugs during Phase C/D/E, and
requires re-running the patch-bucket audit and re-applying patches. Post-Phase-F
upstream bumps are handled as an independent follow-up PR, not within R3 scope.

## Vendored files

**All** upstream inference files required by the ReAct scaffold are vendored
after the Phase C+F course correction (see "Re-vendor pass" section below).
Original R3 Principle #5 ("bounded tool surface; only 2 tools") is
superseded — user directive 2026-04-16: "python 也加上（没有 infra 也加！）".

Only `run_multi_react.py` and `run_react_infer.sh` (batch drivers) stay
unvendored because the Runner adapter replaces their responsibility.

| File | Upstream path | SHA256 | Patch status |
|------|---------------|--------|-------------|
| `vendor/react_agent.py` | `inference/react_agent.py` | patched | Bucket D (6 package-relative imports) |
| `vendor/prompt.py`      | `inference/prompt.py`      | `6e4262c05e44a4104b349226e6d009aa442d536a011ae3abe2f48c3a4a222315` | verbatim |
| `vendor/tool_search.py` | `inference/tool_search.py` | `b2cf78dd5d6766bcfa39971e4ed220efccdc6e16f1b3d43769a3ac74df1614f7` | verbatim |
| `vendor/tool_visit.py`  | `inference/tool_visit.py`  | patched | Bucket D (1 import) |
| `vendor/tool_scholar.py` | `inference/tool_scholar.py` | `429716a1376c3884544e8e53f77e91012aab53aad66ee501d587d60150001eec` | verbatim |
| `vendor/tool_file.py`   | `inference/tool_file.py`   | patched | Bucket D (2 imports) |
| `vendor/tool_python.py` | `inference/tool_python.py` | `eda289fbf17c4d9759869b6b2e45521a737534287ae776f73ef635cf4be60999` | verbatim |
| `vendor/file_tools/file_parser.py` | `inference/file_tools/file_parser.py` | patched | Bucket D (2 imports) |
| `vendor/file_tools/idp.py` | `inference/file_tools/idp.py` | `38ccd5ab0171e23b01d907e4083e440e90b2aee0df9369e27f9e2f69306f4e54` | verbatim |
| `vendor/file_tools/utils.py` | `inference/file_tools/utils.py` | `3ab90cfd911b8c288f7d80745acd8c77fb2001c8410df5959b9adee459125f18` | verbatim |
| `vendor/file_tools/video_agent.py` | `inference/file_tools/video_agent.py` | patched | Bucket D (1 import) |
| `vendor/file_tools/video_analysis.py` | `inference/file_tools/video_analysis.py` | `02b498bc40e0e360e64317e6ff8bc3c2a9b8e22cbfa53823b4cc5b13832324af` | verbatim |

Byte-for-byte identity verified against the upstream clone at pinned SHA
`f72f75d8c3eb842f2bbbab096a12206ff66e270f`.
Recompute via `shasum -a 256 vendor/*.py vendor/file_tools/*.py`.

### Re-vendor pass (post Phase F)

Original Phase C dropped 3 tools (`scholar`, `file_parser`, `python`) citing
"smaller is better". On review this was a misjudgment:
- `scholar` uses the same `SERPER_API_KEY` already in use for `search` — zero
  new infra.
- `file_parser` uses `DASHSCOPE_API_KEY` which the operator already had.
- `python` needs `sandbox_fusion` pip package plus a `SANDBOX_FUSION_ENDPOINT`
  at **call time**. The pip client imports cleanly without the endpoint, so
  vendoring doesn't break module load. A runtime tool call will fail when no
  endpoint is configured — that's acceptable per user directive ("没有 infra
  也加") and produces a legitimate scheduling-trace span (retry / error).

All 3 tool files + the transitive `file_tools/` package (5 files: file_parser,
idp, utils, video_agent, video_analysis) are now vendored. `TOOL_CLASS`
restored to the 5-tool upstream default. `_run` and `custom_call_tool`'s
original `python` + `parse_file` dispatch branches restored.

New pip deps added for vendored-code-import viability (all client SDKs; none
are server infra):
- `sandbox-fusion==0.3.7`
- `alibabacloud-docmind-api20220711==1.4.11` (pulled in via idp.py)

## Patch buckets (actual LOC after Phase C + post-F re-vendor)

Tracked for audit. Bucket design was refined during Phase C kickoff — see
"Adapter-side zero-patch strategy" section below for why A + B are at zero.
Bucket C was zeroed out during the post-F re-vendor pass (all tools restored).

| Bucket | Description | Actual LOC (+/-) | Applied in |
|--------|-------------|------------------|------------|
| A: trace-hook emit        | TraceAction emits at LLM + tool call sites | **0 / 0** | Adapter-side (zero vendor patch; adapter monkey-patches `vendor.OpenAI` and `vendor.TOOL_CLASS` with traced equivalents) |
| B: streaming shim         | `call_server` streaming + TTFT/TPOT capture | **0 / 0** | Adapter-side (internal to TracedStreamingOpenAI) |
| C: TOOL_CLASS+imports prune + dead code | (REVERTED post-F) all 5 tools are vendored; `_run`/`custom_call_tool` branches restored | **0 / 0** | Post-F re-vendor pass |
| D: package-import fix     | File-local `from prompt import *` / `from tool_X import *` / `from file_tools.X import ...` converted to package-relative across 5 touched files | **12 / 12** | Phase C + post-F re-vendor |

Total vendor patch footprint after post-F re-vendor: **12 import lines modified** across 5 files (`react_agent.py`, `tool_visit.py`, `tool_file.py`, `file_tools/file_parser.py`, `file_tools/video_agent.py`). No numerical LOC hard limit per user directive; recorded here for audit.

Note: `logical_turn_id` is **not** a patch bucket — per R3 Principle #2, it
is generated in the runner adapter (Phase D), not injected into vendor code.
Vendor source stays turn-semantics-agnostic.

### Adapter-side zero-patch strategy (Phase C design refinement)

R3 originally budgeted Bucket A + B as vendor patches. During Phase C kickoff
it was realized `vendor.OpenAI` and `vendor.TOOL_CLASS` are **module-level
attributes**, so the adapter (Phase D) can monkey-patch them with traced
equivalents. This eliminates ~70 LOC of vendor patch and moves trace logic
entirely into the adapter layer, maximizing fidelity preservation. Vendor
source only changes for the two concerns that cannot live in the adapter:
(C) runtime-unresolvable imports of unvendored tool files, and (D) package
layout mismatch (file-local imports).

### Known trace coverage gap: Visit tool's summarization LLM

`vendor/tool_visit.py` defines its own `Visit.call_server()` method that makes
a **separate LLM call** to a summarization model via the env-configured
`API_KEY` / `API_BASE` / `SUMMARY_MODEL_NAME` endpoint, using a raw
`openai.OpenAI()` client (not via our monkey-patched `TracedStreamingOpenAI`).
These extractor calls happen every time the Visit tool successfully fetches a
page through Jina Reader, to distill webpage content into evidence.

**Implication**: `summary.total_llm_ms` and `n_turns` under-report by the
wall-ms and call count of these extractor LLM calls. The Visit tool's total
wall time (including the sub-call) is correctly captured in the tool_exec
TraceAction's `duration_ms`, so the upstream-facing latency is right, but the
LLM-call breakdown inside the tool is opaque.

Why not trace it: the extractor is an implementation detail of the Visit tool,
not a ReAct turn, and tracing it would require a second vendor patch. For
scheduling analysis at the ReAct-turn granularity, this is acceptable (the
wall time is still attributed). If future research needs visibility into the
extractor latency distribution, the adapter can patch vendor_tool_visit's
inner `OpenAI` import the same way the main ReAct loop patches it.

## Import smoke (conda env `ML`, command `PYTHONPATH=src conda run -n ML python -c "import <module>"`)

| Module | Phase B (pre-patch) | Phase C (post-patch) |
|--------|---------------------|----------------------|
| `agents.tongyi_deepresearch`                    | OK | OK |
| `agents.tongyi_deepresearch.vendor`             | OK | OK |
| `agents.tongyi_deepresearch.vendor.prompt`      | OK | OK |
| `agents.tongyi_deepresearch.vendor.tool_search` | OK | OK |
| `agents.tongyi_deepresearch.vendor.react_agent` | FAIL (`ModuleNotFoundError: prompt`) | **OK** (after Bucket D) |
| `agents.tongyi_deepresearch.vendor.tool_visit`  | FAIL (`ModuleNotFoundError: prompt`) | **OK** (after Bucket D) |

### Qwen-agent dep status (resolved in Phase C)

`qwen-agent==0.0.34` is installed in conda env `ML` (newer than R3's original
`==0.0.26` pin; API-compatible based on successful vendor import). Pre-mortem
scenario #1 (dep conflict) did not fire. No install/pin adjustments needed
for downstream phases.

## Phase E smoke log (2026-04-16, UTC 09:45:06)

First end-to-end invocation of `TongyiDeepResearchRunner` against a real
cloud backend. Invocation:
`python scripts/smoke_tongyi_deepresearch.py --provider dashscope --model qwen-plus-latest --max-iterations 4`.
Synthetic task: "What is the capital of France?".

| Field | Value |
|---|---|
| provider | dashscope |
| model | qwen-plus-latest |
| api_base | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| exit_status | `completed` |
| vendor_termination | `answer` |
| n_turns | 1 |
| total_llm_ms | 2156.8 |
| total_tool_ms | 0.0 |
| total_tokens | 960 |
| transport_retry_count | 0 |
| llm_call actions | 1 (non-retry) |
| tool_exec actions | 0 |
| transport_retry spans | 0 |
| final_answer | `"Paris"` |

AC#4 invariants: every llm_call `TraceAction` has non-None `ttft_ms` + `tpot_ms`
on the DashScope streaming endpoint — **PASS**. Non-Tongyi model
(`qwen-plus-latest`, not the 30B-A3B) still produced valid `<think>...<answer>`
output on a trivial prompt, so the ReAct prompt format is not strictly Tongyi-
model-specific for simple cases (ACCURACY IS NOT A CONSTRAINT here — this is a
scheduling-trace validation).

## Phase I smoke log (2026-04-16, paid 1+1 via CLI)

Full `trace_collect.cli` pipeline against DashScope, 1 task each on
deep-research-bench + browsecomp, `--max-iterations 8`. Two bugs surfaced
and fixed during this phase:

1. **action_id counter reset per round**: vendor constructs a fresh `OpenAI`
   client inside every `call_server` invocation, so the per-instance counter
   started at 0 each round — every `llm_call` got `action_id="llm_1"`. Fix:
   runner now injects a shared `call_counter` + `retry_state` into the bound
   factory so IDs stay monotonic across vendor's per-round rebuilds.
2. **tool_exec not serialised to trace file**: vendor's `custom_call_tool`
   aliases `tool_args["params"] = tool_args` (self-reference), breaking
   `json.dumps` inside `log_trace_action`. Actions were captured in memory
   (so summary saw them) but never reached disk. Fix: tool wrapper strips
   the self-reference before recording `tool_args` in the TraceAction.
3. **Env-var name mismatch**: vendor reads `SERPER_KEY_ID` / `JINA_API_KEYS`,
   our repo convention is `SERPER_API_KEY` / `JINA_API_KEY`. Runner now
   aliases on demand at `run_task` entry.

### DRB: instance 51 (Japan elderly population + consumption, `qwen-plus-latest`)

| Field | Value |
|---|---|
| llm_call actions | 7 (`llm_1` .. `llm_7`) |
| tool_exec actions | 6 (5 search + 1 visit) |
| transport_retry_count | 0 |
| total_llm_ms | 74 515 |
| total_tool_ms | 55 579 (search ~4–9 s each; visit 22 s) |
| total_tokens | 68 883 |
| ttft_ms range | 740 – 3 244 |
| tpot_ms | ~22.0 (constant) |
| vendor_termination | `answer` |
| exit_status | `completed` |
| AC#2 React triplet | **PASS** (`<tool_call>` in LLM output + 6 tool_execs fired) |
| AC#4 ttft/tpot non-None | **PASS** (7/7) |

### browsecomp: instance 0 (named-person retrieval, `qwen-plus-latest`)

| Field | Value |
|---|---|
| llm_call actions | 1 (`llm_1`) |
| tool_exec actions | 0 (model answered directly from training memory) |
| transport_retry_count | 0 |
| total_llm_ms | 116 635 |
| total_tool_ms | 0 |
| total_tokens | 28 748 |
| ttft_ms | ~2 000 |
| tpot_ms | ~22 |
| vendor_termination | `answer` |
| exit_status | `completed` |
| AC#4 ttft/tpot non-None | **PASS** (1/1) |

Note: browsecomp task did not trigger tool use — the model's training memory
had the answer directly. Per R3 AC#2 "**at least one** produced trace contains
a complete ReAct triplet", the DRB trace satisfies this (browsecomp trace
does not, but the AC is over the set of Phase I traces, not per-task). The
browsecomp trace is still a valid scheduling-analysis artifact (one long
LLM turn with real TTFT/TPOT).

Total smoke cost: ~97k tokens on `qwen-plus-latest` ≈ **$0.14** (well under
R3's $2 paid-smoke budget).

### 5-tool re-smoke (2026-04-16, post post-F re-vendor, same tasks/model)

After the post-F re-vendor pass restored the full upstream tool surface
(`scholar`, `parse_file`, `python` added), the same 1+1 paid smoke was
re-run to observe whether `qwen-plus-latest` reaches for the new tools on
these benchmark tasks.

| Task | Turns | Tools invoked | Scholar / parse_file / python | Outcome |
|------|------:|--------------:|------------------------------:|---------|
| DRB #51 | 8 (hit max) | 6 × `search` + 2 × `visit` | **0 calls** | `exceed available llm calls`; success=False |
| browsecomp #0 | 1 | 0 | **0 calls** | `answer`; success=True |

Tokens: ~95k total, cost ≈ $0.15.

**Observations**:
- Expanded tool surface did not change the DRB tool-dispatch profile — model
  stayed with `search` + `visit`. Scheduling analysis of tool-latency
  distributions is stable across the 2-tool vs 5-tool scaffold for these
  tasks on this model.
- DRB hit `max_iterations=8` limit with the 5-tool scaffold, whereas on the
  first (2-tool) run it completed in 7 turns with an `<answer>`. The extra
  `visit` call (40s) pushed wall time up; not an inherent regression — the
  CLI `--max-iterations` cap is the operational parameter.
- Neither task exercised `scholar`, `parse_file`, or `python`. For larger
  or code-heavy task suites, having those tools available is still the
  faithful upstream setup; this run just shows the operational cost is
  zero when they go unused.

## Deprecation tracker

- **Homegrown multi-phase scaffold**: **DELETED 2026-04-16 in Phase F**, within
  the R3 Principle #3 deadline of 3 calendar days post Phase I green. The
  `src/agents/<deprecated>/` directory and its 2 test files are gone; all
  registration surfaces and doc references have been swept.
- **Interim shim**: env-flag `OMCBENCH_ALLOW_DEPRECATED_SCAFFOLD=1` was
  proposed in R3 Principle #3 but NOT implemented — Phase F's same-day
  deletion made the gap-period shim unnecessary.
