#!/usr/bin/env bash
# ============================================================
# SoccerNet-MVFoul — View-expanded SV Reasoning QLoRA + Late Fusion
# ============================================================
# Usage from repo root:
#   CUDA_VISIBLE_DEVICES=1 bash scripts/sh/run_view_expanded_reason.sh [smoke|train|eval|zeroshot]
# ============================================================
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-smoke}"

MODEL_ID="nvidia/Cosmos-Reason2-8B"
DATA_ROOT="data/SoccerNet/mvfouls"
OUTPUT_DIR="outputs/qlora_cosmos8b_view_expanded_reason"
SMOKE_OUTPUT_DIR="${OUTPUT_DIR}_smoke"
ADAPTER_PATH="${OUTPUT_DIR}/best_checkpoint"
EVAL_DIR="outputs/late_fusion_view_expanded_reason"
ZS_DIR="outputs/zero_shot_late_fusion_reason"

DEVICE="cuda:0"
NUM_FRAMES=32
NUM_EPOCHS=3
GRAD_ACCUM=8
LR="5e-5"
LORA_R=128
LORA_ALPHA=256
SEED=42
MAX_NEW_TOKENS=256
FUSION_RULE="main_first"
MAX_TRAIN_VIEWS=0   # 0 = all views per Train action
MAX_EVAL_VIEWS=0    # 0 = all views per eval action

LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/view_expanded_reason_${MODE}_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log: $LOG_FILE"

if [[ "$MODE" == "smoke" ]]; then
    python scripts/train/train_view_expanded_reason.py \
        --model-id "$MODEL_ID" \
        --data-root "$DATA_ROOT" \
        --output-dir "$SMOKE_OUTPUT_DIR" \
        --device "$DEVICE" \
        --num-frames "$NUM_FRAMES" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --max-train-views "$MAX_TRAIN_VIEWS" \
        --max-eval-views "$MAX_EVAL_VIEWS" \
        --fusion-rule "$FUSION_RULE" \
        --seed "$SEED" \
        --smoke-test
fi

if [[ "$MODE" == "train" ]]; then
    python scripts/train/train_view_expanded_reason.py \
        --model-id "$MODEL_ID" \
        --data-root "$DATA_ROOT" \
        --output-dir "$OUTPUT_DIR" \
        --device "$DEVICE" \
        --num-epochs "$NUM_EPOCHS" \
        --num-frames "$NUM_FRAMES" \
        --grad-accum "$GRAD_ACCUM" \
        --lr "$LR" \
        --lora-r "$LORA_R" \
        --lora-alpha "$LORA_ALPHA" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --max-train-views "$MAX_TRAIN_VIEWS" \
        --max-eval-views "$MAX_EVAL_VIEWS" \
        --fusion-rule "$FUSION_RULE" \
        --balanced-sampling \
        --seed "$SEED"
fi

if [[ "$MODE" == "eval" ]]; then
    if [[ ! -d "$ADAPTER_PATH" ]]; then
        echo "ERROR: adapter not found at $ADAPTER_PATH. Run train first."
        exit 1
    fi
    for SPLIT in Valid Test; do
        python scripts/eval/eval_late_fusion_reason.py \
            --model-id "$MODEL_ID" \
            --adapter-path "$ADAPTER_PATH" \
            --data-root "$DATA_ROOT" \
            --split "$SPLIT" \
            --device "$DEVICE" \
            --num-frames "$NUM_FRAMES" \
            --max-new-tokens "$MAX_NEW_TOKENS" \
            --max-views "$MAX_EVAL_VIEWS" \
            --fusion-rule "$FUSION_RULE" \
            --out-dir "$EVAL_DIR"
    done
fi

if [[ "$MODE" == "zeroshot" ]]; then
    for SPLIT in Valid Test; do
        python scripts/eval/eval_late_fusion_reason.py \
            --model-id "$MODEL_ID" \
            --data-root "$DATA_ROOT" \
            --split "$SPLIT" \
            --device "$DEVICE" \
            --num-frames "$NUM_FRAMES" \
            --max-new-tokens "$MAX_NEW_TOKENS" \
            --max-views "$MAX_EVAL_VIEWS" \
            --fusion-rule "$FUSION_RULE" \
            --out-dir "$ZS_DIR"
    done
fi
