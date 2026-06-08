#!/usr/bin/env bash
# ============================================================
# SoccerNet-MVFoul — Multi-view Reasoning QLoRA, Cosmos-Reason2-2B
# ============================================================
# Usage (from repo root):
#   bash scripts/sh/run_multiview_reason.sh [smoke|train|eval]
#   Default: smoke
# ============================================================
set -euo pipefail

# Prevent CUDA memory fragmentation OOM (same fix as run_reason.sh)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-smoke}"

MODEL_ID="nvidia/Cosmos-Reason2-2B"
DATA_ROOT="data/SoccerNet/mvfouls"
OUTPUT_DIR="outputs/qlora_cosmos2b_multiview_reason_f32_p10m_r32"
ADAPTER_PATH="${OUTPUT_DIR}/best_checkpoint"
EVAL_DIR="outputs/multiview_reason_eval"

DEVICE="cuda:0"   # CUDA_VISIBLE_DEVICES=1로 실행하면 물리 GPU1이 cuda:0으로 보임
NUM_FRAMES=32     # frames per view; 2 fixed views => 64 sampled frames total
NUM_EPOCHS=3
GRAD_ACCUM=8
LR="2e-4"
LORA_R=32
LORA_ALPHA=64
SEED=42
MAX_NEW_TOKENS=512
MAX_PIXELS=10000000  # smoke-tested on RTX 5090 32GB with NUM_FRAMES=32, LoRA r=32

LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/multiview_reason_${MODE}_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log: $LOG_FILE"

# ---- (a) SMOKE TEST ---------------------------------------------------------
if [[ "$MODE" == "smoke" ]]; then
    echo "========================================"
    echo "  SMOKE TEST — multi-view reasoning"
    echo "  (view composition + masking + 2 steps + 2 valid)"
    echo "========================================"
    python scripts/train/train_multiview_reason.py \
        --model-id   "$MODEL_ID" \
        --data-root  "$DATA_ROOT" \
        --output-dir "$OUTPUT_DIR" \
        --device     "$DEVICE" \
        --num-frames "$NUM_FRAMES" \
        --seed       "$SEED" \
        --lora-r     "$LORA_R" \
        --lora-alpha "$LORA_ALPHA" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --max-pixels   "$MAX_PIXELS" \
        --smoke-test
    echo ""
    echo ">>> Smoke test complete. Proceed with: bash scripts/sh/run_multiview_reason.sh train"
fi

# ---- (b) FULL TRAINING ------------------------------------------------------
if [[ "$MODE" == "train" ]]; then
    echo "========================================"
    echo "  FULL TRAINING — multi-view reasoning (${NUM_EPOCHS} epochs)"
    echo "========================================"
    python scripts/train/train_multiview_reason.py \
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
        --max-pixels   "$MAX_PIXELS" \
        --seed        "$SEED"
    echo ""
    echo ">>> Training complete. Best checkpoint: $ADAPTER_PATH"
    echo ">>> Now run: bash scripts/sh/run_multiview_reason.sh eval"
fi

# ---- (c) EVALUATION ---------------------------------------------------------
if [[ "$MODE" == "eval" ]]; then
    echo "========================================"
    echo "  EVALUATION — multi-view reasoning (Valid + Test)"
    echo "========================================"
    if [[ ! -d "$ADAPTER_PATH" ]]; then
        echo "ERROR: adapter not found at $ADAPTER_PATH. Run training first."
        exit 1
    fi
    for SPLIT in Valid Test; do
        echo ""
        echo "--- Split: $SPLIT ---"
        python scripts/eval/eval_finetuned_multiview_reason.py \
            --model-id     "$MODEL_ID" \
            --adapter-path "$ADAPTER_PATH" \
            --data-root    "$DATA_ROOT" \
            --split        "$SPLIT" \
            --device       "$DEVICE" \
            --num-frames   "$NUM_FRAMES" \
            --max-new-tokens "$MAX_NEW_TOKENS" \
            --max-pixels   "$MAX_PIXELS" \
            --out-dir      "$EVAL_DIR"
    done
    echo ""
    echo ">>> Evaluation complete. Results in: $EVAL_DIR"
fi
