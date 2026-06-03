# CLAUDE.md - Python MLSys Research Project

> This file provides authoritative instructions for AI agents working on this codebase.
> All rules here take precedence over default agent behaviors.

---

## Project Overview

This is an **academic research project** in Machine Learning Systems (MLSys).
The goal is to produce **publishable, reproducible, and scientifically rigorous** results.

**Primary constraints:**
- Results must be obtained through legitimate, generalizable methods
- Code must be production-quality, not prototype/toy implementations
- All experiments must be reproducible with documented configurations

---

## Research Integrity & Taste (CRITICAL)

> These principles are NON-NEGOTIABLE. Violating them compromises the scientific validity of our work.

### 1. No Benchmark Gaming

- **MUST NOT** add tricks that only work on specific datasets
- **MUST NOT** tune hyperparameters to overfit evaluation benchmarks
- **MUST NOT** cherry-pick evaluation metrics or subsets
- **MUST NOT** use dataset-specific priors disguised as "general" methods
- If a technique requires knowing properties of the test set, it is INVALID

```python
# ❌ FORBIDDEN: Dataset-specific magic numbers
if dataset_name == "HotpotQA":
    threshold = 0.73  # "tuned" to this specific benchmark
    
# ✅ CORRECT: Generalizable approach
threshold = config.get("threshold", 0.5)  # documented, configurable
```

### 2. No Hindsight Contamination

- **MUST NOT** use information that would not be available at inference time
- **MUST NOT** leak ground truth labels into feature engineering
- **MUST NOT** use "oracle" signals in the actual workflow (only for analysis)
- **MUST NOT** design methods around post-hoc observations of test data

```python
# ❌ FORBIDDEN: Using future/oracle information
def extract_features(query, ground_truth_answer):  # GT leaked!
    similarity_to_answer = compute_sim(query, ground_truth_answer)
    
# ✅ CORRECT: Only use available information
def extract_features(query, retrieved_context):
    # Only information available at inference time
```

### 3. No Unjustified Complexity

- **MUST NOT** add hyperparameters without clear justification
- **MUST NOT** hardcode values that should be configurable
- **MUST NOT** add components "just in case" or "for future use"
- Every design choice must have a documented rationale
- Prefer simple baselines that work over complex methods that barely beat them

```python
# ❌ FORBIDDEN: Unexplained magic
alpha = 0.7823  # Where does this come from?
beta = 1.2 if len(x) > 100 else 0.8  # Why these thresholds?

# ✅ CORRECT: Justified and documented
# Alpha controls exploration-exploitation tradeoff (see Section 3.2 of paper)
alpha = config.exploration_weight  # Default: 0.5, tuned on validation set
```

### 4. Real Workloads, Real Results, No Synthetic 

- **MUST** use realistic data scales and distributions
- **MUST** test on held-out data that was never seen during development
- **MUST** include failure cases and limitations in analysis
- **MUST** not introduce mocks, simulations, stubs, or bypasses to avoid running real operations — even "temporarily" or "for testing." If a component is slow, expensive, or inconvenient to run, that is not a justification for faking it.plan and wait for approval before implementing.
- Toy examples are for debugging only, never for final evaluation

### 5. Completeness Over Shortcuts

- **MUST** implement full pipelines, not hacky shortcuts
- **MUST** handle edge cases properly (empty inputs, missing data, etc.)
- **MUST** preserve all relevant information in data structures
- If something is "too slow," optimize it properly, don't skip it

```python
# ❌ FORBIDDEN: Lossy shortcut
def process_trace(trace):
    return {"score": trace["final_score"]}  # Discards everything else!

# ✅ CORRECT: Preserve information
def process_trace(trace):
    return {
        "score": trace["final_score"],
        "intermediate_steps": trace["steps"],
        "metadata": trace["metadata"],
        "timing": trace["timing"],
        # Preserve fields for downstream analysis
    }
```

### 6. Use Established Tools

- **MUST** use mature, well-tested libraries for standard operations
- **MUST NOT** reimplement standard algorithms without justification
- For agent tracing: use established frameworks (LangSmith, Weights & Biases, etc.)
- For experiment tracking: use proper tools (MLflow, Hydra, etc.)
- For data processing: use battle-tested libraries (pandas, polars, etc.)

```python
# ❌ FORBIDDEN: Reinventing the wheel
def my_json_parser(text):  # Why not use json.loads?
    ...

# ✅ CORRECT: Use established tools
import json
from langsmith import traceable

@traceable  # Proper tracing with established tool
def run_agent(query):
    ...
```

---

## Code Quality Standards

### 1. Correctness First

- Code must produce correct results before any optimization
- All assumptions must be validated with assertions or checks
- Edge cases must panic explicitly or be handled by designed fallback mechanisms, not silently ignored
- Type hints are required for all function signatures

```python
def compute_metric(predictions: list[float], labels: list[float]) -> float:
    """Compute evaluation metric.
    
    Args:
        predictions: Model predictions, must be same length as labels
        labels: Ground truth labels
        
    Returns:
        Metric value in range [0, 1]
        
    Raises:
        ValueError: If inputs have mismatched lengths or are empty
    """
    if len(predictions) != len(labels):
        raise ValueError(f"Length mismatch: {len(predictions)} vs {len(labels)}")
    if not predictions:
        raise ValueError("Empty input")
    ...
```

### 2. Simplicity

- Generated code must stay simple and readable
- Comments must be concise and add value (not repeat the code)
- Avoid over-complex abstractions; prefer explicit over implicit
- One function should do one thing

```python
# ❌ Over-commented
# This function adds two numbers together by taking the first number
# and the second number and using the + operator to add them
def add(a, b):
    return a + b  # Return the sum of a and b

# ✅ Appropriately commented
def compute_f1(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
```

### 3. No Code Duplication

- **MUST NOT** duplicate same/similar logic across files
- If you find yourself copying code, extract to a shared utility
- Before implementing, check if similar functionality already exists
- Generalize existing implementations when extending functionality

### 4. Configuration Management

- All configurable values in config files, not hardcoded
- Use hierarchical configuration (Hydra, OmegaConf, or similar)
- Separate: data config, model config, experiment config
- Log full config with every experiment run

```python
# ❌ FORBIDDEN
learning_rate = 0.001
batch_size = 32

# ✅ CORRECT
@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    batch_size: int = 32
    
config = TrainingConfig(**load_yaml("configs/training.yaml"))
```

---
## Be Patient — Do Not Alter Scope Due to Runtime

**Long-running operations are expected in MLSys research.** Do not let execution time influence your decisions.

**MUST NOT** take any of the following actions without explicit human approval:

- Cancel a download/process because it's "taking too long"
- Substitute a smaller dataset, model, or subset "to save time"  
- Reduce epochs, iterations, or sample size for "quick testing"
- Skip preprocessing steps that seem "expensive"
- Use cached/stale results instead of recomputing
- Switch to a "lighter" alternative (e.g., smaller model, fewer features)

**Expected timescales you should wait for:**

| Operation | Normal Duration |
|-----------|-----------------|
| Dataset download | Minutes to hours |
| Preprocessing/feature extraction | Minutes to hours |
| Model training | Minutes to days |
| Full evaluation pipeline | Minutes to hours |
| Hyperparameter search | Hours to days |

**If you believe the runtime is genuinely problematic:**

1. Report the expected duration to the human
2. Explain why you think it may be an issue
3. **Wait for explicit approval** before changing anything
4. Never silently make substitutions

---

## Agent Trace Standards

When working with LLM agents or multi-step pipelines:

- **MUST** preserve all intermediate outputs, not just final results
- **MUST** log timing information for each step
- **MUST** capture model responses in full (not truncated)
- **MUST** record all metadata (model version, parameters, etc.)

---

## Benchmark Plugin Architecture

All benchmarks MUST be added via the plugin layer in `src/agents/benchmarks/`
and `configs/benchmarks/<slug>.yaml`.

**FORBIDDEN:**
- Hardcoding dataset names (`princeton-nlp/SWE-bench_Verified`,
  `nebius/SWE-rebench`, etc.) in `src/trace_collect/collector.py`,
  `src/trace_collect/cli.py`, or any scaffold module.
- Adding `--harness-dataset` / `--harness-split` / `--harness-namespace`
  or similar "per-benchmark" CLI flags — those belong in the YAML.
- Adding a `from_<benchmark>_instance()` factory method on `EvalTask`.
  The canonical entry point is `EvalTask.from_benchmark_instance(row,
  workspace_base, benchmark=<plugin>)` which delegates to the plugin's
  `normalize_task` for benchmark-specific quirks.

---

## Mandatory Review Gate for Vibe Coding

Before proceeding to any of the following milestones, you **MUST** spawn a separate sub-agent to conduct a rigorous code review:

**Trigger points:**
- Completing a major module or feature
- Before committing a significant refactor
- Before running any experiment that will produce results for analysis
- Before any code that touches the evaluation pipeline

**Review process:**

1. **Spawn a dedicated reviewer sub-agent** with a fresh context (not the one that wrote the code — the author is blind to their own mistakes) and ordered to be strict and have good research taste.

2. **The reviewer must check:**
   - Correctness: Does the logic actually do what it claims?
   - Research integrity: Any hindsight leakage? Benchmark-specific tricks? Unjustified magic numbers?
   - Completeness: Are all fields preserved? Edge cases handled?
   - Consistency: Does it match existing conventions and documentation?

3. **Iterate until clean:**
   - If the reviewer finds 🔴 critical or 🟠 major issues → fix and re-review
   - If only 🟡 minor issues remain → may proceed (but still fix them)
   - If clean → proceed to experiment/commit

4. **Document the review:**
   - Log what was reviewed
   - Log issues found and how they were resolved
   - This creates an audit trail

**Why this matters:**

The agent that writes code develops "tunnel vision" — it becomes convinced its implementation is correct because it wrote it with that intent. A fresh sub-agent has no such bias. It reads the code as-is and catches:
- Logic errors the author was blind to
- Subtle hindsight contamination that "felt natural" while writing
- Hardcoded values that the author "meant to make configurable later"
- Missing edge cases the author didn't think of

**Non-negotiable rule:**

No experiment results are valid if the code that produced them was not reviewed through this gate. Running experiments on unreviewed code is wasting compute on potentially meaningless results.

---

## Done Criteria (Pre-Commit Checklist)

Before finalizing any change:

### Scope Verification
- [ ] Modified files stay within requested scope
- [ ] No accidental edits to data/artifact directories
- [ ] No changes to unrelated modules

### Functional Verification
- [ ] Run relevant test command(s) - report what was run
- [ ] Verify at least one representative output is generated
- [ ] Check no regressions in existing functionality

### Code Quality
- [ ] Type hints on all new functions
- [ ] No code duplication introduced
- [ ] No hardcoded values that should be configurable
- [ ] Comments are helpful and not excessive

### Research Integrity
- [ ] No benchmark-specific tricks introduced
- [ ] No hindsight/oracle information leakage
- [ ] All design choices are justified
- [ ] Generalizable to other datasets/settings

### Documentation
- [ ] Changelog updated if behavior changed
- [ ] Docstrings for new public functions
- [ ] Config changes documented

### Commit Message Format
```
[type] Brief description (max 50 chars)

- Detailed point 1
- Detailed point 2

Types: feat, fix, refactor, docs, test, config
```

---

## Agent Behavioral Rules

### DO
- Verify by reading actual source code before making claims
- Check existing code for conventions before asking
- Run the minimal test covering your changes
- Ask for clarification when requirements are ambiguous
- Preserve existing functionality unless explicitly told otherwise
- Persist multi-step plans to a file (e.g., docs/CURRENT_PLAN.md). Complex tasks with checkpoints, TODOs, or dependencies must be written to disk — not held in conversation context. After any context compaction or session resume, re-read the plan file immediately before continuing. The plan file is the source of truth.
- MUST pause at checkpoints for human review — this is the default mode. Before advancing to the next implementing phase, modifying new files, or running significant operations, stop and wait for explicit approval. Autonomous "keep going" behavior is only permitted when the human explicitly requests autopilot mode (e.g., "run to completion", "autopilot", "全自动", "持续推进"). Even then, pause on errors, ambiguity, or judgment calls. Default behavior: stop and check. Not: charge ahead and hope.

### DO NOT
- Guess or hallucinate about project internals
- Introduce new dependencies without explicit approval
- Modify files outside the scope of current task
- Make "improvements" that weren't requested
- Simplify by removing functionality (simplify implementation, not behavior)

### When Uncertain
1. First: Check existing code for precedent
2. Second: Check documentation
3. Third: Ask the human explicitly