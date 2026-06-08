#!/usr/bin/env bash
# ============================================================
# SoccerNet-MVFoul — Baseline (no-reason) QLoRA, Cosmos-Reason2-8B
# ============================================================
# Usage (from repo root):
#   bash scripts/run_baseline.sh [smoke|zeroshot|train|eval]
#   Default: smoke
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-smoke}"

MODEL_ID="nvidia/Cosmos-Reason2-8B"
DATA_ROOT="data/SoccerNet/mvfouls"
OUTPUT_DIR="outputs/qlora_cosmos8b"
ADAPTER_PATH="${OUTPUT_DIR}/best_checkpoint"
EVAL_DIR="outputs/finetuned_eval"
ZS_DIR="outputs/zero_shot_baseline"

DEVICE="cuda:0"
NUM_FRAMES=8        # must be even; reduce to 4 if OOM
NUM_EPOCHS=3
GRAD_ACCUM=8        # effective batch = 8
LR="2e-4"
LORA_R=16
LORA_ALPHA=32
SEED=42
MAX_NEW_TOKENS=64

LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/baseline_${MODE}_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log: $LOG_FILE"

# ---- (a) SMOKE TEST ---------------------------------------------------------
if [[ "$MODE" == "smoke" ]]; then
    echo "========================================"
    echo "  SMOKE TEST — baseline (no-reason)"
    echo "========================================"
    python scripts/train/train_baseline.py \
        --model-id   "$MODEL_ID" \
        --data-root  "$DATA_ROOT" \
        --output-dir "$OUTPUT_DIR" \
        --device     "$DEVICE" \
        --num-frames "$NUM_FRAMES" \
        --seed       "$SEED" \
        --smoke-test
    echo ""
    echo ">>> Smoke test complete. Proceed with: bash scripts/run_baseline.sh train"
fi

# ---- (b) ZERO-SHOT EVAL (no fine-tuning) ------------------------------------
if [[ "$MODE" == "zeroshot" ]]; then
    echo "========================================"
    echo "  ZERO-SHOT EVAL — baseline prompt"
    echo "========================================"
    for SPLIT in Valid Test; do
        echo "--- Split: $SPLIT ---"
        python scripts/zero_shot/zero_shot_eval.py \
            --model-id   "$MODEL_ID" \
            --data-root  "$DATA_ROOT" \
            --split      "$SPLIT" \
            --limit      0 \
            --num-frames "$NUM_FRAMES" \
            --device     "$DEVICE" \
            --out-dir    "$ZS_DIR"
    done
    echo ""
    echo ">>> Zero-shot eval complete. Results in: $ZS_DIR"
fi

# ---- (c) FULL TRAINING ------------------------------------------------------
if [[ "$MODE" == "train" ]]; then
    echo "========================================"
    echo "  FULL TRAINING — baseline (${NUM_EPOCHS} epochs)"
    echo "========================================"
    python scripts/train/train_baseline.py \
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
        --seed        "$SEED"
    echo ""
    echo ">>> Training complete. Best checkpoint: $ADAPTER_PATH"
    echo ">>> Now run: bash scripts/run_baseline.sh eval"
fi

# ---- (d) EVALUATION (Valid + Test) ------------------------------------------
if [[ "$MODE" == "eval" ]]; then
    echo "========================================"
    echo "  EVALUATION — baseline (Valid + Test)"
    echo "========================================"
    if [[ ! -d "$ADAPTER_PATH" ]]; then
        echo "ERROR: adapter not found at $ADAPTER_PATH. Run training first."
        exit 1
    fi
    for SPLIT in Valid Test; do
        echo ""
        echo "--- Split: $SPLIT ---"
        python scripts/eval/eval_finetuned_baseline.py \
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
