# Case Inspection

> This document is part of the [Agent Sched Bench manual](../README.md).
> For running benchmarks, see [Trace Collect](trace-collect.md).
> For setup instructions, see [Getting Started](getting-started.md).

`scripts/inspect_swebench.py` is a **standalone script** for quickly inspecting
SWE-bench Verified and SWE-rebench test cases. No agent system integration
needed — just Docker + HuggingFace datasets.

**Use cases:** Reviewers who want to see what benchmark cases look like —
repo structure, problem statements, and test details.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Usage Examples](#usage-examples)
- [SWE-rebench Usage](#swe-rebench-usage)
- [Using a Local Cache](#using-a-local-cache)
- [Common Workflows](#common-workflows)

---

## Prerequisites

```bash
pip install datasets docker

export HF_ENDPOINT=https://hf-mirror.com
```

## Usage Examples

The script supports several subcommands — `list`, `info`, `pull`, `shell`,
`diff`, `tests`, `export`, and more. Examples below use `--benchmark
swe-bench-verified`; all subcommands work identically for `swe-rebench`.

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

## SWE-rebench Usage

The same script works for SWE-rebench with identical subcommands — just
change the `--benchmark` argument:

```bash
python scripts/inspect_swebench.py --benchmark swe-rebench list
python scripts/inspect_swebench.py --benchmark swe-rebench info 12rambau__sepal_ui-411
python scripts/inspect_swebench.py --benchmark swe-rebench pull 12rambau__sepal_ui-411
python scripts/inspect_swebench.py --benchmark swe-rebench shell 12rambau__sepal_ui-411
```

## Using a Local Cache

Downloading task metadata from HuggingFace on every invocation can be slow.
If you have already run `make download-swebench-verified`, point the script
at the local cache to skip the network round-trip:

```bash
python scripts/inspect_swebench.py \
    --benchmark swe-bench-verified \
    --cache-file data/swebench_verified/tasks.json \
    list
```

## Common Workflows

The subcommands above can be combined into typical inspection workflows.
Each workflow below assumes you have Docker running and the benchmark data
downloaded.

### Workflow 1: Quickly Browse a Few Cases

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

### Workflow 2: Export All Files for Offline Analysis

```bash
python scripts/inspect_swebench.py -b swe-bench-verified pull astropy__astropy-12907
python scripts/inspect_swebench.py -b swe-bench-verified export astropy__astropy-12907 /testbed ./case_astropy_12907/
# Then open ./case_astropy_12907/ in your local IDE
```

### Workflow 3: View the Gold Fix Patch

```bash
# Shows both the code patch and the test patch from the dataset
python scripts/inspect_swebench.py -b swe-bench-verified diff django__django-10097
```

### Workflow 4: Inspect FAIL_TO_PASS Tests

```bash
# See which test files are involved and how many tests per file
python scripts/inspect_swebench.py -b swe-bench-verified tests django__django-10097

# Then view a specific test file
python scripts/inspect_swebench.py -b swe-bench-verified tests django__django-10097 -f tests/auth_tests/test_validators.py
```

The inspection script is read-only — it helps you understand what a benchmark
case looks like. To actually run an agent to solve these cases, see
[Trace Collect](trace-collect.md).
