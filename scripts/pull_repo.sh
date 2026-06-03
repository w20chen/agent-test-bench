#!/usr/bin/env bash
set -euo pipefail

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Refusing to pull with tracked local changes present." >&2
  exit 1
fi

git fetch origin "${CURRENT_BRANCH}"
git pull --ff-only origin "${CURRENT_BRANCH}"
