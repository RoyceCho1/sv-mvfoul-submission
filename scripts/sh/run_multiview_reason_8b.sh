#!/usr/bin/env bash
# ============================================================
# SoccerNet-MVFoul — Multi-view Reasoning QLoRA, Cosmos-Reason2-8B
# ============================================================
# Usage (from repo root):
#   bash scripts/sh/run_multiview_reason_8b.sh [smoke|train|eval]
#   Default: smoke
#
# Smoke sweep protocol (escalate until VRAM > 28 GB or OOM):
#   Step 1: F8  / P3M  / r16  (conservative, seq_len ~1300, ~14–17 GB)
#   Step 2: F16 / P5M  / r16  (seq_len ~2000, ~18–22 GB)
#   Step 3: F16 / P10M / r16  (seq_len ~3800, ~24–28 GB)
#   Step 4: F32 / P10M / r16  (same seq_len as step 3, +temporal coverage)
#
# Key: at P10M, pixel budget dominates → seq_len ≈ constant regardless
# of frame count. So F16→F32 at same pixel budget costs no extra VRAM.
# ============================================================
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-smoke}"

MODEL_ID="nvidia/Cosmos-Reason2-8B"
DATA_ROOT="data/SoccerNet/mvfouls"
OUTPUT_DIR="outputs/qlora_cosmos8b_multiview_reason_1"
ADAPTER_PATH="${OUTPUT_DIR}/best_checkpoint"
EVAL_DIR="outputs/qlora_cosmos8b_multiview_reason_eval_1"

DEVICE="cuda:0"       # CUDA_VISIBLE_DEVICES=0 => physical GPU0 is cuda:0
NUM_FRAMES=32         # frames per view; 2 fixed views => 64 sampled frames total
MAX_PIXELS=10000000   # no downscale for 720p clips; smoke-tested on RTX 5090
LORA_R=32             # smoke-tested with F32/P10M on 8B
LORA_ALPHA=64
NUM_EPOCHS=5          # epoch2→3 하락 방지: 더 긴 학습으로 안정 수렴
GRAD_ACCUM=8
LR="1e-4"             # 2e-4에서 epoch3 불안정 → 절반으로
SEED=42
MAX_NEW_TOKENS=1024  # 8B 모델은 2B보다 더 긴 시퀀스 처리 가능; smoke-tested up to ~3800 with F16/P10M
MAX_CLASS_WEIGHT=3.0  # balanced_sampling과 함께 사용 시 double-correction 방지

LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/multiview_reason_8b_${MODE}_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log: $LOG_FILE"

# ---- (a) SMOKE TEST ---------------------------------------------------------
if [[ "$MODE" == "smoke" ]]; then
    echo "========================================"
    echo "  SMOKE TEST — 8B multi-view reasoning"
    echo "  F${NUM_FRAMES} / P${MAX_PIXELS} / r${LORA_R}"
    echo "  (view composition + masking + 2 steps + 2 valid)"
    echo "========================================"
    python scripts/train/train_multiview_reason.py \
        --model-id       "$MODEL_ID" \
        --data-root      "$DATA_ROOT" \
        --output-dir     "$OUTPUT_DIR" \
        --device         "$DEVICE" \
        --num-frames     "$NUM_FRAMES" \
        --max-pixels     "$MAX_PIXELS" \
        --lora-r         "$LORA_R" \
        --lora-alpha     "$LORA_ALPHA" \
        --seed           "$SEED" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --smoke-test
    echo ""
    echo ">>> Smoke test complete."
    echo ">>> Check 'reserved' VRAM in log. If < 26 GB, escalate settings."
    echo ">>> Next: bash scripts/sh/run_multiview_reason_8b.sh train"
fi

# ---- (b) FULL TRAINING ------------------------------------------------------
if [[ "$MODE" == "train" ]]; then
    echo "========================================"
    echo "  FULL TRAINING — 8B multi-view reasoning (${NUM_EPOCHS} epochs)"
    echo "  F${NUM_FRAMES} / P${MAX_PIXELS} / r${LORA_R} / alpha${LORA_ALPHA}"
    echo "========================================"
    python scripts/train/train_multiview_reason.py \
        --model-id          "$MODEL_ID" \
        --data-root         "$DATA_ROOT" \
        --output-dir        "$OUTPUT_DIR" \
        --device            "$DEVICE" \
        --num-epochs        "$NUM_EPOCHS" \
        --num-frames        "$NUM_FRAMES" \
        --max-pixels        "$MAX_PIXELS" \
        --lora-r            "$LORA_R" \
        --lora-alpha        "$LORA_ALPHA" \
        --grad-accum        "$GRAD_ACCUM" \
        --lr                "$LR" \
        --max-new-tokens    "$MAX_NEW_TOKENS" \
        --max-class-weight  "$MAX_CLASS_WEIGHT" \
        --balanced-sampling \
        --seed              "$SEED"
    echo ""
    echo ">>> Training complete. Best checkpoint: $ADAPTER_PATH"
    echo ">>> Now run: bash scripts/sh/run_multiview_reason_8b.sh eval"
fi

# ---- (c) EVALUATION ---------------------------------------------------------
if [[ "$MODE" == "eval" ]]; then
    echo "========================================"
    echo "  EVALUATION — 8B multi-view reasoning (Valid + Test)"
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
            --max-pixels   "$MAX_PIXELS" \
            --max-new-tokens "$MAX_NEW_TOKENS" \
            --out-dir      "$EVAL_DIR"
    done
    echo ""
    echo ">>> Evaluation complete. Results in: $EVAL_DIR"
fi
