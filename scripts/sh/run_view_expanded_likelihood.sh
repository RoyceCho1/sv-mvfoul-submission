#!/usr/bin/env bash
# ============================================================
# SoccerNet-MVFoul — View-expanded Likelihood QLoRA
# Cosmos-Reason2-8B
# ============================================================
# Usage (from repo root):
#   CUDA_VISIBLE_DEVICES=1 bash scripts/sh/run_view_expanded_likelihood.sh [smoke|train|eval|zeroshot]
#   Default: smoke
#
# Notes:
#   - Training saves every epoch checkpoint and skips epoch-end eval.
#   - Run eval separately by setting ADAPTER_PATH to an epoch checkpoint.
# ============================================================
set -euo pipefail

# Prevent CUDA memory fragmentation OOM.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-smoke}"

MODEL_ID="${MODEL_ID:-nvidia/Cosmos-Reason2-8B}"
DATA_ROOT="${DATA_ROOT:-data/SoccerNet/mvfouls}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qlora_cosmos8b_view_expanded_likelihood}"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-${OUTPUT_DIR}_smoke}"
ADAPTER_PATH="${ADAPTER_PATH:-${OUTPUT_DIR}/best_checkpoint}"
EVAL_DIR="${EVAL_DIR:-outputs/likelihood_view_expanded_likelihood_eval}"
ZS_DIR="${ZS_DIR:-outputs/zero_shot_likelihood_full_valid}"

DEVICE="${DEVICE:-cuda:0}"
NUM_FRAMES="${NUM_FRAMES:-32}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-5e-5}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
SEED="${SEED:-42}"

TARGET_FORMAT="${TARGET_FORMAT:-json}"       # json | answer_tag
SCORE_REDUCTION="${SCORE_REDUCTION:-mean}"   # mean | sum
FUSION_METHODS="${FUSION_METHODS:-score_mean,score_sum,clip0,clip1,weighted_clip1}"
PRIOR_ALPHAS="${PRIOR_ALPHAS:-0,0.005,0.01,0.02,0.03,0.05,0.075}"
MAX_TRAIN_VIEWS="${MAX_TRAIN_VIEWS:-0}"      # 0 = all views per Train action
MAX_EVAL_VIEWS="${MAX_EVAL_VIEWS:-0}"        # 0 = all views per eval action

LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/view_expanded_likelihood_${MODE}_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log: $LOG_FILE"

# ---- (a) SMOKE TEST ---------------------------------------------------------
if [[ "$MODE" == "smoke" ]]; then
    echo "========================================"
    echo "  SMOKE TEST — view-expanded likelihood"
    echo "========================================"
    python scripts/train/train_view_expanded_likelihood.py \
        --model-id "$MODEL_ID" \
        --data-root "$DATA_ROOT" \
        --output-dir "$SMOKE_OUTPUT_DIR" \
        --device "$DEVICE" \
        --num-frames "$NUM_FRAMES" \
        --target-format "$TARGET_FORMAT" \
        --score-reduction "$SCORE_REDUCTION" \
        --fusion-methods "$FUSION_METHODS" \
        --prior-alphas "$PRIOR_ALPHAS" \
        --max-train-views "$MAX_TRAIN_VIEWS" \
        --max-eval-views "$MAX_EVAL_VIEWS" \
        --seed "$SEED" \
        --smoke-test
    echo ""
    echo ">>> Smoke test complete. Proceed with: bash scripts/sh/run_view_expanded_likelihood.sh train"
fi

# ---- (b) ZERO-SHOT EVAL (likelihood scoring, no fine-tuning) ---------------
if [[ "$MODE" == "zeroshot" ]]; then
    echo "========================================"
    echo "  ZERO-SHOT EVAL — likelihood scoring"
    echo "========================================"
    for SPLIT in Valid Test; do
        echo ""
        echo "--- Split: $SPLIT ---"
        python scripts/eval/eval_late_fusion_likelihood.py \
            --model-id "$MODEL_ID" \
            --data-root "$DATA_ROOT" \
            --split "$SPLIT" \
            --device "$DEVICE" \
            --num-frames "$NUM_FRAMES" \
            --max-views "$MAX_EVAL_VIEWS" \
            --candidate-format "$TARGET_FORMAT" \
            --score-reduction "$SCORE_REDUCTION" \
            --fusion-methods "$FUSION_METHODS" \
            --prior-alphas "$PRIOR_ALPHAS" \
            --out-dir "$ZS_DIR" \
            --output-prefix "${SPLIT,,}_likelihood" \
            --log-every 20 \
            --save-every 10
    done
    echo ""
    echo ">>> Zero-shot likelihood eval complete. Results in: $ZS_DIR"
fi

# ---- (c) FULL TRAINING ------------------------------------------------------
if [[ "$MODE" == "train" ]]; then
    echo "========================================"
    echo "  FULL TRAINING — likelihood (${NUM_EPOCHS} epochs)"
    echo "========================================"
    python scripts/train/train_view_expanded_likelihood.py \
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
        --lora-dropout "$LORA_DROPOUT" \
        --target-format "$TARGET_FORMAT" \
        --score-reduction "$SCORE_REDUCTION" \
        --fusion-methods "$FUSION_METHODS" \
        --prior-alphas "$PRIOR_ALPHAS" \
        --max-train-views "$MAX_TRAIN_VIEWS" \
        --max-eval-views "$MAX_EVAL_VIEWS" \
        --balanced-sampling \
        --save-every-epoch \
        --no-epoch-eval \
        --seed "$SEED"
    echo ""
    echo ">>> Training complete. Epoch checkpoints: ${OUTPUT_DIR}/epoch_1 ... ${OUTPUT_DIR}/epoch_${NUM_EPOCHS}"
    echo ">>> Eval one epoch with: ADAPTER_PATH=${OUTPUT_DIR}/epoch_1 bash scripts/sh/run_view_expanded_likelihood.sh eval"
fi

# ---- (d) EVALUATION (Valid + Test, likelihood scoring) ----------------------
if [[ "$MODE" == "eval" ]]; then
    echo "========================================"
    echo "  EVALUATION — likelihood scoring"
    echo "========================================"
    if [[ ! -d "$ADAPTER_PATH" ]]; then
        echo "ERROR: adapter not found at $ADAPTER_PATH. Run training first or set ADAPTER_PATH."
        exit 1
    fi
    for SPLIT in Valid Test; do
        echo ""
        echo "--- Split: $SPLIT ---"
        python scripts/eval/eval_late_fusion_likelihood.py \
            --model-id "$MODEL_ID" \
            --adapter-path "$ADAPTER_PATH" \
            --data-root "$DATA_ROOT" \
            --split "$SPLIT" \
            --device "$DEVICE" \
            --num-frames "$NUM_FRAMES" \
            --max-views "$MAX_EVAL_VIEWS" \
            --candidate-format "$TARGET_FORMAT" \
            --score-reduction "$SCORE_REDUCTION" \
            --fusion-methods "$FUSION_METHODS" \
            --prior-alphas "$PRIOR_ALPHAS" \
            --out-dir "$EVAL_DIR" \
            --output-prefix "${SPLIT,,}_likelihood" \
            --log-every 20 \
            --save-every 10
    done
    echo ""
    echo ">>> Likelihood eval complete. Results in: $EVAL_DIR"
fi
