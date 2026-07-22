# Runtime Prediction

> Predicts execution time for pip install, Python script, and pytest commands
> using bounded history collected from prior attempts.

---

## History Sharing

Prediction history lives in per-run databases under `run_dir`:

```text
run_dir/
  pip_runtime_db/<instance>/history.json            # pip retry history
  python_script_runtime_db/<instance>/history.json  # Python script retry history
  pytest_runtime_db/<instance>/history.json         # pytest retry history
  runtime_kb/
    repo/<repo>/
      pip/history.json
      python_script/history.json
      pytest/history.json
```

Each database has **two bucket levels**:

| Bucket | Location | Scope | Used when |
|--------|----------|-------|-----------|
| **Instance** | `<tool>_runtime_db/<instance>/history.json` | This exact task instance | Attempt N+1 reads attempt 1..N |
| **Family** | `runtime_kb/repo/<repo>/<tool>/history.json` | All tasks sharing the same repository | Cross-instance cold start |

**Seed (before each attempt):** instance history is always seeded into `history.json`.  Family history is independently seeded into `family_history.json` when available — the two signals are stored separately and used side-by-side in prediction.

**Merge (after each attempt):** successful prediction rows write to **both** instance and family buckets.

Repo-family history is organized repo-first: one repository-level knowledge base
contains separate tool libraries (`pip`, `python_script`, and `pytest`).

Family scope is intentionally minimal: **only repo**.  Image family, Python version, and install config are deliberately excluded so that all instances of the same repository share a single history bucket — maximising the chance of a cold-start hit.

All history containers are bounded to **5 entries** per key (FIFO).  File-based exclusivity locks (`O_CREAT | O_EXCL`) protect concurrent access with 600 s stale-lock cleanup. Compound commands whose elapsed time includes unrelated work still emit artifacts, but they do not update predictive history.

---

## Pip Install Prediction

**Source:** `src/trace_collect/package_runtime_prediction.py`

**Recognised commands:** `pip install`, `pip3 install`, `python -m pip install`.

**Command normalization:** strips output and boolean flags, sorts package specs, hashes requirement-file contents.  Commands with `||`, prefix work, follow-up shell segments, or pipes are recorded but excluded from history.

**Four baselines** (cascading priority):

| Priority | Method | Reliability | Uses |
|----------|--------|-------------|------|
| 1 | **Last Run** | `high` | Most recent duration for the same normalised command (this instance) |
| 2 | **Family Last Run** | `medium` | Most recent duration for the same command from sibling instances (same repo) |
| 3 | **Package Count** | `medium` | `package_count × per_package_median` |
| 4 | **Global Median** | `low` | Median of all successful pip install durations |

**CLI flags:** `--capture-pip-runtime` / `--no-capture-pip-runtime` (default: on).

**Realtime output:**
```text
[pip-predict] #4 pkgs=2 | 10.0s actual →last=9.0s(+10.0%) fam=9.5s(+5.3%) pkgs=11.0s(-9.1%) glob=12.0s(-16.7%) | high
```
All strategies are displayed with their relative errors; `→` marks the recommended (selected) strategy.

**Artifacts:**
```text
attempt_N/pip_runtime/
  predictions.jsonl
  iter_0004_exec-pip_<id>/
    pending.json     # pre-execution snapshot
    prediction.json  # post-execution comparison
```

---

## Python Script Prediction

**Source:** `src/trace_collect/python_script_runtime_prediction.py`

**Recognised commands:** `python <script.py>`, `python3 -u /app/script.py`, `timeout 20 python3 script.py`.  Excludes `python -c`, heredocs, and `python -m ...`.

**Command normalization:** strips interpreter flags (`-b`, `-O`, `-W all`), environment variables, and `cd` preludes.  Extracts `script_path` and `script_basename`.

**Five baselines** (cascading priority):

| Priority | Method | Reliability | Uses |
|----------|--------|-------------|------|
| 1 | **Last Run** | `high` | Most recent duration for the same normalised command (this instance) |
| 2 | **Family Last Run** | `medium` | Most recent duration for the same command from sibling instances (same repo) |
| 3 | **Script Path Median** | `medium` | Median of all runs of the same script path |
| 4 | **Basename Median** | `low` | Median of all runs sharing the same filename |
| 5 | **Global Median** | `low` | Median of all successful Python script runs |

Only successful, standalone runs (not `or`-chains, prefix work, follow-up commands, or pipes) enter history.

**CLI flags:** `--capture-python-script-runtime` / `--no-capture-python-script-runtime` (default: on).

**Artifacts:**
```text
attempt_N/python_script_runtime/
  predictions.jsonl
  iter_0004_python-script_<id>/
    pending.json
    prediction.json
```

---

## Pytest Prediction

**Source:** `src/trace_collect/pytest_runtime_prediction.py`

The most sophisticated of the three.  Before the actual pytest runs, a
lightweight `pytest --collect-only` discovers all test nodeids without
executing tests.  A temporary pytest plugin records per-node timing data
after the real run. Only successful pytest invocations update predictive
history; failed, interrupted, or partial runs keep artifacts for audit but are
not used as future runtime knowledge.

### Pre-execution: collect-only

- Runs `pytest --collect-only` with a temporary plugin in the tool's working directory
- Strips side-effect flags (`--junitxml`, `--html`, `--cov`, `--cov-report`)
- Rejects cache-dependent selectors (`--lf`, `--last-failed`)
- Records nodeid list and collection overhead time

### Per-test prediction

For each nodeid discovered, prediction cascades:

```
exact nodeid history  →  same-file median  →  project-wide median  →  unavailable
```

A cold-start variant inserts `"unknown"` (median of previously-unseen tests) between file and project levels.

### Aggregated prediction methods

| Method | Formula |
|--------|---------|
| **Last Run** | Most recent duration of the same normalised pytest command |
| **Test Count** | `len(nodeids) × project_test_median` |
| **Per-Test** | `Σ(per-nodeid predictions) + historical_overhead_median` |
| **Unknown Fallback** | Same as Per-Test, but uses the cold-start variant with `"unknown"` fallback |

### Reliability assessment

Each test's prediction source is classified (nodeid/file/project/unknown/unavailable).  The recommended method is chosen by rule priority:

| Priority | Condition | Method | Reliability |
|----------|-----------|--------|-------------|
| 1 | Same command + collected-count delta ≤ 10% | Last Run | `high` |
| 2 | Same command + previous count unknown | Last Run | `medium` |
| 3 | Same command available from sibling instance (same repo) | Family Last Run | `medium` |
| 4 | No tests collected + command was seen before | Last Run | `medium` |
| 5 | ≥ 80% tests have exact nodeid history | Per-Test | `high` |
| 6 | ≥ 80% tests have nodeid or same-file history | Per-Test | `medium` |
| 7 | Cold-start unknown-test fallback available | Unknown | `low` |
| 8 | No prediction possible | none | `coldstart` / `error` |

Key thresholds: `KNOWN_NODE_HIGH_RATIO = 0.80`, `KNOWN_OR_FILE_MEDIUM_RATIO = 0.80`, `COMMAND_COUNT_STABLE_REL_DELTA = 0.10`.

### CLI flags and analysis

| Flag | Default | Effect |
|------|---------|--------|
| `--capture-pytest-runtime` | on | Enable plugin injection, prediction artifacts, realtime output |
| `--no-capture-pytest-runtime` | — | Disable all pytest prediction |

**Analyse a run:**
```bash
PYTHONPATH=src python scripts/analyze_pytest_prediction.py traces/swe-rebench/<model>/<run>
```

Reports all baseline methods plus `Recommended`, then prints `high` / `medium` / `low` / `coldstart` / `error` reliability buckets.  Use `--csv` for export.

**Realtime output:**
```text
[pytest-predict] #17 tests=32 | 220.4s actual (collect 4.2s) →last=205.1s(+6.9%) fam=210.0s(+5.0%) count=230.0s(-4.2%) per=215.7s(+2.1%) unk=240.0s(-8.2%) | high
```
All strategies are displayed with their relative errors; `→` marks the recommended (selected) strategy.

**Artifacts:**
```text
attempt_N/pytest_runtime/
  history.json        # attempt-local history (seeded from shared)
  predictions.jsonl   # compact per-run rows (no full output)
  iter_0017_exec-pytest_<id>/
    pending.json       # pre-execution snapshot
    prediction.json    # post-execution comparison + full pytest output
    pytest_runtime.json
    pytest_collect_only.json
    instrumentation.json
```

---

## Reusing Knowledge Across Runs

History databases are **per-run**: each `collect` invocation with a new `run_dir`
starts from an empty database.  To carry prediction knowledge forward:

### Method 1: `--run-id` (recommended)

Point new runs at an existing `run_dir`.  All accumulated history is retained
and new attempts contribute to it.

```bash
# All experiments share one persistent run_dir
python -m trace_collect.cli ... --run-id traces/swe-rebench/my-experiment

# Rerun with new parameters — history from previous attempts feeds predictions
python -m trace_collect.cli ... --run-id traces/swe-rebench/my-experiment \
  --prompt-template new_prompt --rerun-completed
```

Constraints: same benchmark + same model.  `--rerun-completed` lets completed
instances run again while reading their earlier history.

### Method 2: Copy history directories

Copy the three `*_runtime_db/` directories from an old run into a new one:

```bash
OLD="traces/swe-rebench/model/20260720T093000"
NEW="traces/swe-rebench/model/20260721T140000"

mkdir -p "$NEW"
for db in pip_runtime_db python_script_runtime_db pytest_runtime_db; do
  cp -r "$OLD/$db" "$NEW/$db"
done

python -m trace_collect.cli ... --run-id "$NEW"
```

Useful when switching models or when you want a clean trace directory while
keeping the prediction knowledge.

### Method 3: Maintain a seed database

After a full benchmark run, package the history directories as a reusable seed:

```bash
# Create seed
tar -czf seed_history.tar.gz -C traces/swe-rebench/model/full-run \
  pip_runtime_db python_script_runtime_db pytest_runtime_db

# Use seed for new experiments
tar -xzf seed_history.tar.gz -C traces/swe-rebench/model/new-run/
```

### Quick decision table

| Scenario | Approach |
|----------|----------|
| Resume interrupted run | `--run-id <existing>` |
| Same model, tweak prompt/params | `--run-id <existing> --rerun-completed` |
| Different model, same benchmark | Copy `*_runtime_db/` dirs |
| Different machine | scp the `*_runtime_db/` dirs |
| Different benchmark | Limited — only family buckets for overlapping repos may transfer |
