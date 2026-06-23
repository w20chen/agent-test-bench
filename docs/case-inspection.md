# Case Inspection (SWE-bench)

> This document is part of the [Agent Sched Bench manual](../README.md).
> For running benchmarks, see [Trace Collect](trace-collect.md).
> For setup instructions, see [Getting Started](getting-started.md).

`scripts/inspect_swebench.py` is a **standalone script** for quickly inspecting
SWE-bench Verified and SWE-rebench test cases. No agent system integration
needed — just Docker + HuggingFace datasets.

**Use cases:** Reviewers who want to see what benchmark cases look like —
repo structure, problem statements, and test details.

## Prerequisites

```bash
pip install datasets docker

export HF_ENDPOINT=https://hf-mirror.com
```

## Usage Examples

```bash
# List the first 20 SWE-bench Verified tasks
python scripts/inspect_swebench.py --benchmark swe-bench-verified list

# Search by keyword
python scripts/inspect_swebench.py --benchmark swe-bench-verified list -k django

# View full task details (problem statement, test command, image name)
python scripts/inspect_swebench.py --benchmark swe-bench-verified info django__django-10097

# Pull the Docker image (~2 GB, may take a few minutes)
python scripts/inspect_swebench.py --benchmark swe-bench-verified pull django__django-10097

# List files under /testbed inside the container
python scripts/inspect_swebench.py --benchmark swe-bench-verified ls django__django-10097

# View a specific file inside the container
python scripts/inspect_swebench.py --benchmark swe-bench-verified cat django__django-10097 /testbed/setup.py

# View the gold fix patch (what the agent is expected to produce)
python scripts/inspect_swebench.py --benchmark swe-bench-verified diff django__django-10097

# Live git diff inside the container (after making manual edits in a shell)
python scripts/inspect_swebench.py --benchmark swe-bench-verified diff django__django-10097 --container

# Show FAIL_TO_PASS tests grouped by source file (understand what needs fixing)
python scripts/inspect_swebench.py --benchmark swe-bench-verified tests django__django-10097

# View a specific test file
python scripts/inspect_swebench.py --benchmark swe-bench-verified tests django__django-10097 -f tests/auth_tests/test_validators.py

# Export the entire /testbed to a local directory
python scripts/inspect_swebench.py --benchmark swe-bench-verified export django__django-10097 /testbed ./export_django/

# Enter an interactive bash shell in the container (most flexible)
python scripts/inspect_swebench.py --benchmark swe-bench-verified shell django__django-10097
```

## SWE-rebench Works the Same Way

```bash
# SWE-rebench
python scripts/inspect_swebench.py --benchmark swe-rebench list
python scripts/inspect_swebench.py --benchmark swe-rebench info 12rambau__sepal_ui-411
python scripts/inspect_swebench.py --benchmark swe-rebench pull 12rambau__sepal_ui-411
python scripts/inspect_swebench.py --benchmark swe-rebench shell 12rambau__sepal_ui-411
```

## Use Local Cache to Skip HF Download

If you've already downloaded data via `make download-swebench-verified`,
use the local `tasks.json` cache to skip the HuggingFace download:

```bash
python scripts/inspect_swebench.py \
    --benchmark swe-bench-verified \
    --cache-file data/swebench_verified/tasks.json \
    list
```

## Common Workflows

### Workflow 1: Quickly browse a few cases

```bash
# 1. List some tasks
python scripts/inspect_swebench.py -b swe-bench-verified list -n 5

# 2. Pick one and view details
python scripts/inspect_swebench.py -b swe-bench-verified info sympy__sympy-12481

# 3. Pull image + enter shell to explore code
python scripts/inspect_swebench.py -b swe-bench-verified pull sympy__sympy-12481
python scripts/inspect_swebench.py -b swe-bench-verified shell sympy__sympy-12481
# Inside container: ls /testbed, cat /testbed/setup.py, git log, etc.
```

### Workflow 2: Export all files for offline analysis

```bash
python scripts/inspect_swebench.py -b swe-bench-verified pull astropy__astropy-12907
python scripts/inspect_swebench.py -b swe-bench-verified export astropy__astropy-12907 /testbed ./case_astropy_12907/
# Then open ./case_astropy_12907/ in your local IDE
```

### Workflow 3: See the gold fix patch (the expected solution)

```bash
# Shows both the code patch and the test patch from the dataset
python scripts/inspect_swebench.py -b swe-bench-verified diff django__django-10097
```

### Workflow 4: Understand what tests need to pass (FAIL_TO_PASS)

```bash
# See which test files are involved and how many tests per file
python scripts/inspect_swebench.py -b swe-bench-verified tests django__django-10097

# Then view a specific test file
python scripts/inspect_swebench.py -b swe-bench-verified tests django__django-10097 -f tests/auth_tests/test_validators.py
```

The inspection script is read-only — it helps you understand what a benchmark
case looks like. To actually run an agent to solve these cases, see
[Trace Collect](trace-collect.md).
