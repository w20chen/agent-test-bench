# Runtime Prediction

> Predicts execution time for pip install, Python script, and pytest commands
> using bounded history collected from prior attempts.

---

## History Sharing

Prediction history lives in per-run databases under `run_dir`:

```text
run_dir/
  pip_runtime_db/           # pip prediction history
  python_script_runtime_db/ # python script prediction history
  pytest_runtime_db/        # pytest prediction history
```

Each database has **two bucket levels**:

| Bucket | Location | Scope | Used when |
|--------|----------|-------|-----------|
| **Instance** | `<tool>_runtime_db/<instance>/history.json` | This exact task instance | Attempt N+1 reads attempt 1..N |
| **Family** | `<tool>_runtime_db/family/<family>/history.json` | Tasks sharing repo, image family, Python version, install config | No instance history exists yet (cold start) |

**Seed (before each attempt):** instance history if available → family history if available → empty (pure cold start).

**Merge (after each attempt):** successful prediction rows write to **both** instance and family buckets.

Family scope is built from task metadata: `repo`, `image` (family label with version suffix stripped), `python_version`, and a SHA1 hash of `install_config`.  When metadata is sparse the scope degrades gracefully to per-instance, preventing incorrect cross-task sharing.

All history containers are bounded to **5 entries** per key (FIFO).  File-based exclusivity locks (`O_CREAT | O_EXCL`) protect concurrent access with 600 s stale-lock cleanup.

---

## Pip Install Prediction

**Source:** `src/trace_collect/package_runtime_prediction.py`

**Recognised commands:** `pip install`, `pip3 install`, `python -m pip install`.

**Command normalization:** strips output and boolean flags, sorts package specs, hashes requirement-file contents.  Commands with `||` chains are recorded but excluded from history.

**Three baselines** (cascading priority):

| Priority | Method | Uses |
|----------|--------|------|
| 1 | **Last Run** (reliability `high`) | Most recent duration for the same normalised command |
| 2 | **Package Count** (`medium`) | `package_count × per_package_median` |
| 3 | **Global Median** (`low`) | Median of all successful pip install durations |

**CLI flags:** `--capture-pip-runtime` / `--no-capture-pip-runtime` (default: on).

**Realtime output:**
```text
[pip-predict] iter=4 packages=2 actual=10.00s last=9.00s last_err=10.0% ...
```

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

**Four baselines** (cascading priority):

| Priority | Method | Uses |
|----------|--------|------|
| 1 | **Last Run** (`high`) | Most recent duration for the same normalised command |
| 2 | **Script Path Median** (`medium`) | Median of all runs of the same script path |
| 3 | **Basename Median** (`low`) | Median of all runs sharing the same filename |
| 4 | **Global Median** (`low`) | Median of all successful Python script runs |

Only successful, standalone runs (not `or`-chains, not pipes) enter history.

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
after the real run.

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
| 3 | No tests collected + command was seen before | Last Run | `medium` |
| 4 | ≥ 80% tests have exact nodeid history | Per-Test | `high` |
| 5 | ≥ 80% tests have nodeid or same-file history | Per-Test | `medium` |
| 6 | Cold-start unknown-test fallback available | Unknown | `low` |
| 7 | No prediction possible | none | `coldstart` / `error` |

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
[pytest-predict] iter=17 tests=32 actual=220.40s collect_overhead=4.20s \
  last=205.10s last_err=6.9% ... recommended=per_test:215.70s rec_err=2.1% reliability=high
```

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
