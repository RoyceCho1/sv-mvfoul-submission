#!/usr/bin/env bash
# ============================================================
# SoccerNet-MVFoul — Reasoning QLoRA (answer-only + weighted)
# Cosmos-Reason2-8B
# ============================================================
# Usage (from repo root):
#   bash scripts/sh/run_reason_answer.sh [smoke|train|eval]
#   Default: smoke
#
# Ablation flags (append to python command):
#   --no-answer-only    → use full think+answer loss (reverts to train_reason)
#   --no-weighted-loss  → disable class weighting
# ============================================================
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-smoke}"

MODEL_ID="nvidia/Cosmos-Reason2-8B"
DATA_ROOT="data/SoccerNet/mvfouls"
OUTPUT_DIR="outputs/qlora_cosmos8b_reason_answer"
ADAPTER_PATH="${OUTPUT_DIR}/best_checkpoint"
EVAL_DIR="outputs/qlora_cosmos8b_reason_answer_eval"

DEVICE="cuda:0"
NUM_FRAMES=16
NUM_EPOCHS=3
GRAD_ACCUM=8
LR="2e-4"
LORA_R=16
LORA_ALPHA=32
SEED=42
MAX_NEW_TOKENS=2048
MAX_CLASS_WEIGHT=10.0

LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/reason_answer_${MODE}_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log: $LOG_FILE"

# ---- (a) SMOKE TEST ---------------------------------------------------------
if [[ "$MODE" == "smoke" ]]; then
    echo "========================================"
    echo "  SMOKE TEST — answer-only + weighted loss"
    echo "========================================"
    python scripts/train/train_reason_answer.py \
        --model-id   "$MODEL_ID" \
        --data-root  "$DATA_ROOT" \
        --output-dir "$OUTPUT_DIR" \
        --device     "$DEVICE" \
        --num-frames "$NUM_FRAMES" \
        --seed       "$SEED" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --max-class-weight "$MAX_CLASS_WEIGHT" \
        --smoke-test
    echo ""
    echo ">>> Smoke test complete. Proceed with: bash scripts/sh/run_reason_answer.sh train"
fi

# ---- (b) FULL TRAINING ------------------------------------------------------
if [[ "$MODE" == "train" ]]; then
    echo "========================================"
    echo "  FULL TRAINING — answer-only + weighted (${NUM_EPOCHS} epochs)"
    echo "========================================"
    python scripts/train/train_reason_answer.py \
        --model-id    "$MODEL_ID" \
        --data-root   "$DATA_ROOT" \
        --output-dir  "$OUTPUT_DIR" \
        --device      "$DEVICE" \
        --num-epochs  "$NUM_EPOCHS" \
        --num-frames  "$NUM_FRAMES" \
        --grad-accum  "$GRAD_ACCUM" \
        --lr          "$LR" \
        --lora-r      "$LORA_R" \
        --lora-alpha  "$LORA_ALPHA" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --max-class-weight "$MAX_CLASS_WEIGHT" \
        --seed        "$SEED"
    echo ""
    echo ">>> Training complete. Best checkpoint: $ADAPTER_PATH"
    echo ">>> Now run: bash scripts/sh/run_reason_answer.sh eval"
fi

# ---- (c) EVALUATION ---------------------------------------------------------
if [[ "$MODE" == "eval" ]]; then
    echo "========================================"
    echo "  EVALUATION — answer-only + weighted (Valid + Test)"
    echo "========================================"
    if [[ ! -d "$ADAPTER_PATH" ]]; then
        echo "ERROR: adapter not found at $ADAPTER_PATH. Run training first."
        exit 1
    fi
    for SPLIT in Valid Test; do
        echo ""
        echo "--- Split: $SPLIT ---"
        python scripts/eval/eval_finetuned_reason.py \
            --model-id      "$MODEL_ID" \
            --adapter-path  "$ADAPTER_PATH" \
            --data-root     "$DATA_ROOT" \
            --split         "$SPLIT" \
            --device        "$DEVICE" \
            --num-frames    "$NUM_FRAMES" \
            --max-new-tokens "$MAX_NEW_TOKENS" \
            --out-dir       "$EVAL_DIR"
    done
    echo ""
    echo ">>> Evaluation complete. Results in: $EVAL_DIR"
fi
