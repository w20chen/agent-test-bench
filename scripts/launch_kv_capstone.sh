#!/usr/bin/env bash
# Launch a KV-eviction capstone trace on a terminal-bench task with
# Qwen3-Coder-30B + openclaw + HF recording backend (session-shared KV cache).
#
# The KV policy is driven entirely by the yaml's `name:` field (random | streaming | h2o).
# No --kv-policy flag is passed so the CLI default ("none") does not clobber it.
# Pass the literal string "none" as <kv-config-yaml> to disable eviction (baseline run).
#
# Usage:
#   ./scripts/launch_kv_capstone.sh configs/kv_policies/h2o_b1024.yaml [label-suffix]
#   ./scripts/launch_kv_capstone.sh none baseline
#   INSTANCE_ID=dna-insert ./scripts/launch_kv_capstone.sh configs/kv_policies/h2o_b4096.yaml h2o_b4096_dna
#
# Env (override defaults if your host differs):
#   REPO        - repo root (default: $(pwd))
#   INSTANCE_ID - terminal-bench task id (default: jsonl-aggregator)
#   MODEL_ID    - HF model id/path (default: Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8;
#                 FP8 fits on 80 GB / 48 GB cards — bf16 30B-A3B needs 96 GB)
#   ENV_BIN     - conda env bin dir holding python + the `tb` console script
#                 (default: dir of `which python`)
#   HF_HOME     - HuggingFace cache dir (default: $HOME/hf_cache)
#   HF_RECORDING_MAX_GPU_MEMORY_GIB - cap for backend_hf (default: 90)
#   OPENCLAW_TEMPERATURE / OPENCLAW_TOP_P / OPENCLAW_TOP_K /
#   OPENCLAW_REPETITION_PENALTY - generation controls. Defaults match the
#                 Qwen3-Coder model card for this Qwen-focused launcher.
#
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 <kv-config-yaml> [label-suffix]" >&2
  exit 2
fi

KV_CONFIG="$1"
INSTANCE_ID="${INSTANCE_ID:-jsonl-aggregator}"
if [ "$KV_CONFIG" = "none" ]; then
  DEFAULT_LABEL="baseline-${INSTANCE_ID}"
else
  DEFAULT_LABEL="$(basename "$KV_CONFIG" .yaml)-${INSTANCE_ID}"
fi
LABEL_SUFFIX="${2:-$DEFAULT_LABEL}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="logs/capstone-${LABEL_SUFFIX}-${TS}.log"

REPO="${REPO:-$(pwd)}"
cd "$REPO"
mkdir -p logs

export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_RECORDING_MAX_GPU_MEMORY_GIB="${HF_RECORDING_MAX_GPU_MEMORY_GIB:-90}"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8}"
OPENCLAW_TEMPERATURE="${OPENCLAW_TEMPERATURE:-0.7}"
OPENCLAW_TOP_P="${OPENCLAW_TOP_P:-0.8}"
OPENCLAW_TOP_K="${OPENCLAW_TOP_K:-20}"
OPENCLAW_REPETITION_PENALTY="${OPENCLAW_REPETITION_PENALTY:-1.05}"

# Conda env bin must be on PATH so terminal-bench's `tb` console script
# resolves (TerminalBenchRunner._preflight greps PATH for it). Calling
# `${ENV_BIN}/python` directly skips activation, so PATH must be patched.
ENV_BIN="${ENV_BIN:-$(dirname "$(command -v python)")}"
export PATH="$ENV_BIN:${PATH}"

PY="$ENV_BIN/python"

echo "=== launch_kv_capstone.sh ==="
echo "ts=$TS"
echo "kv_config=$KV_CONFIG"
echo "instance_id=$INSTANCE_ID"
echo "model_id=$MODEL_ID"
echo "temperature=$OPENCLAW_TEMPERATURE"
echo "top_p=$OPENCLAW_TOP_P"
echo "top_k=$OPENCLAW_TOP_K"
echo "repetition_penalty=$OPENCLAW_REPETITION_PENALTY"
echo "label=$LABEL_SUFFIX"
echo "log=$LOG"
echo "head=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
echo "================================"

CMD=("$PY" -m trace_collect.cli
  --provider openai
  --model "$MODEL_ID"
  --benchmark terminal-bench
  --scaffold openclaw
  --container docker
  --mcp-config none
  --instance-ids "$INSTANCE_ID"
  --record-internals
  --max-iterations 100
  --temperature "$OPENCLAW_TEMPERATURE"
  --top-p "$OPENCLAW_TOP_P"
  --top-k "$OPENCLAW_TOP_K"
  --repetition-penalty "$OPENCLAW_REPETITION_PENALTY")
if [ "$KV_CONFIG" != "none" ]; then
  CMD+=(--kv-config "$KV_CONFIG")
fi
exec "${CMD[@]}"
