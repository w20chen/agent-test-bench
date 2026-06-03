#!/usr/bin/env bash
# Clone benchmark repos locally for fast container-internal cloning.
#
# Usage:
#   ./scripts/setup/clone_repos.sh [tasks.json] [repos_root]
#
# Arguments:
#   tasks.json  — path to a tasks JSON file; only repos referenced there
#                 are cloned. Default: data/swebench_verified/tasks.json
#                 (SWE-Bench Verified, preserved for backward compat).
#   repos_root  — target directory for the cloned mirrors. Default:
#                 data/swebench_repos (SWE-Bench Verified legacy path).
#
# For SWE-rebench:
#   ./scripts/setup/clone_repos.sh data/swe-rebench/tasks.json data/swe-rebench/repos
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TASKS_FILE="${1:-${PROJECT_ROOT}/data/swebench_verified/tasks.json}"
REPOS_ROOT="${2:-$PROJECT_ROOT/data/swebench_repos}"

if [[ -d "$REPOS_ROOT" ]] && [[ -n "$(ls -A "$REPOS_ROOT" 2>/dev/null)" ]]; then
    echo "[setup] SKIP clone_repos: $REPOS_ROOT is non-empty"
    exit 0
fi

mkdir -p "$REPOS_ROOT"

if [[ -f "$TASKS_FILE" ]]; then
    REPOS=$(python3 -c "
import json, sys
tasks = json.load(open(sys.argv[1]))
repos = sorted(set(t['repo'] for t in tasks))
for r in repos:
    print(r)
" "$TASKS_FILE")
else
    REPOS="django/django
sympy/sympy
scikit-learn/scikit-learn
matplotlib/matplotlib
sphinx-doc/sphinx
pytest-dev/pytest
astropy/astropy
pydata/xarray
psf/requests
pallets/flask
pylint-dev/pylint
mwaskom/seaborn"
fi

echo "[setup] Cloning SWE-bench repos to $REPOS_ROOT"

while IFS= read -r repo; do
    owner="${repo%%/*}"
    name="${repo##*/}"
    dir_name="${owner}__${name}"
    target="$REPOS_ROOT/$dir_name"

    if [[ -d "$target" ]]; then
        echo "[setup] SKIP: $dir_name (already exists)"
        continue
    fi

    echo "[setup] CLONE: $repo → $dir_name"
    git clone --quiet "https://github.com/${repo}.git" "$target"
    echo "[setup]   done ($(du -sh "$target" | cut -f1))"
done <<< "$REPOS"

echo "[setup] clone_repos done"
du -sh "$REPOS_ROOT"
