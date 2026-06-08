#!/usr/bin/env bash
# Convenience wrapper for late-fusion eval only.
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash scripts/sh/run_late_fusion_reason.sh zeroshot Valid 20
#   CUDA_VISIBLE_DEVICES=1 bash scripts/sh/run_late_fusion_reason.sh finetuned Valid 0 outputs/.../best_checkpoint
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-zeroshot}"
SPLIT="${2:-Valid}"
LIMIT="${3:-20}"
ADAPTER_PATH="${4:-outputs/qlora_cosmos8b_view_expanded_reason/best_checkpoint}"

MODEL_ID="nvidia/Cosmos-Reason2-8B"
DATA_ROOT="data/SoccerNet/mvfouls"
DEVICE="cuda:0"
NUM_FRAMES=16
MAX_NEW_TOKENS=256
MAX_VIEWS=0
FUSION_RULE="main_first"

OUT_DIR="outputs/late_fusion_reason_${MODE}"
ARGS=(
  --model-id "$MODEL_ID"
  --data-root "$DATA_ROOT"
  --split "$SPLIT"
  --limit "$LIMIT"
  --device "$DEVICE"
  --num-frames "$NUM_FRAMES"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --max-views "$MAX_VIEWS"
  --fusion-rule "$FUSION_RULE"
  --out-dir "$OUT_DIR"
)

if [[ "$MODE" == "finetuned" ]]; then
  ARGS+=(--adapter-path "$ADAPTER_PATH")
fi

python scripts/eval/eval_late_fusion_reason.py "${ARGS[@]}"
