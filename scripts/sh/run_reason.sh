#!/usr/bin/env bash
# ============================================================
# SoccerNet-MVFoul — Reasoning QLoRA, Cosmos-Reason2-8B
# ============================================================
# Usage (from repo root):
#   bash scripts/run_reason.sh [smoke|zeroshot|train|eval]
#   Default: smoke
# ============================================================
set -euo pipefail

# Prevent CUDA memory fragmentation OOM (epoch 2+ accumulates fragmentation)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-smoke}"

MODEL_ID="nvidia/Cosmos-Reason2-8B"
DATA_ROOT="data/SoccerNet/mvfouls"
OUTPUT_DIR="outputs/qlora_cosmos8b_reason_2"
ADAPTER_PATH="${OUTPUT_DIR}/best_checkpoint"
EVAL_DIR="outputs/qlora_cosmos8b_reason_eval_2"
ZS_DIR="outputs/zero_shot_reason"

DEVICE="cuda:0"
NUM_FRAMES=16
NUM_EPOCHS=5          # 기존 3ep에서 안정 수렴 위해 연장
GRAD_ACCUM=8
LR="1e-4"             # 2e-4에서 불안정 → 절반으로
LORA_R=16
LORA_ALPHA=32
SEED=42
MAX_NEW_TOKENS=512
MAX_CLASS_WEIGHT=3.0  # balanced_sampling 사용 시 cap

LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/reason_${MODE}_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log: $LOG_FILE"

# ---- (a) SMOKE TEST ---------------------------------------------------------
if [[ "$MODE" == "smoke" ]]; then
    echo "========================================"
    echo "  SMOKE TEST — reasoning mode"
    echo "========================================"
    python scripts/train/train_reason.py \
        --model-id   "$MODEL_ID" \
        --data-root  "$DATA_ROOT" \
        --output-dir "$OUTPUT_DIR" \
        --device     "$DEVICE" \
        --num-frames "$NUM_FRAMES" \
        --seed       "$SEED" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --smoke-test
    echo ""
    echo ">>> Smoke test complete. Proceed with: bash scripts/run_reason.sh train"
fi

# ---- (b) ZERO-SHOT EVAL (no fine-tuning, reasoning prompt) ------------------
if [[ "$MODE" == "zeroshot" ]]; then
    echo "========================================"
    echo "  ZERO-SHOT EVAL — reasoning prompt"
    echo "========================================"
    for SPLIT in Valid Test; do
        echo "--- Split: $SPLIT ---"
        python scripts/zero_shot/eval_zeroshot_reason.py \
            --model-id   "$MODEL_ID" \
            --data-root  "$DATA_ROOT" \
            --split      "$SPLIT" \
            --limit      0 \
            --num-frames "$NUM_FRAMES" \
            --max-new-tokens "$MAX_NEW_TOKENS" \
            --device     "$DEVICE" \
            --out-dir    "$ZS_DIR"
    done
    echo ""
    echo ">>> Zero-shot reason eval complete. Results in: $ZS_DIR"
fi

# ---- (c) FULL TRAINING ------------------------------------------------------
if [[ "$MODE" == "train" ]]; then
    echo "========================================"
    echo "  FULL TRAINING — reasoning (${NUM_EPOCHS} epochs)"
    echo "========================================"
    python scripts/train/train_reason.py \
        --model-id         "$MODEL_ID" \
        --data-root        "$DATA_ROOT" \
        --output-dir       "$OUTPUT_DIR" \
        --device           "$DEVICE" \
        --num-epochs       "$NUM_EPOCHS" \
        --num-frames       "$NUM_FRAMES" \
        --grad-accum       "$GRAD_ACCUM" \
        --lr               "$LR" \
        --lora-r           "$LORA_R" \
        --lora-alpha       "$LORA_ALPHA" \
        --max-new-tokens   "$MAX_NEW_TOKENS" \
        --max-class-weight "$MAX_CLASS_WEIGHT" \
        --balanced-sampling \
        --seed             "$SEED"
    echo ""
    echo ">>> Training complete. Best checkpoint: $ADAPTER_PATH"
    echo ">>> Now run: bash scripts/run_reason.sh eval"
fi

# ---- (d) EVALUATION (Valid + Test, 4-row comparison table) ------------------
if [[ "$MODE" == "eval" ]]; then
    echo "========================================"
    echo "  EVALUATION — reasoning (Valid + Test)"
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
