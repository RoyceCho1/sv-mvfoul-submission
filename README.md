# SoccerNet-MVFoul Severity Classification with Cosmos-Reason2 QLoRA

This repository contains the code and experiment documentation for a final project on SoccerNet-MVFoul foul severity classification. The goal is to solve the SoccerNet-MVFoul offence severity task with a reasoning vision-language model instead of the original VARS MViT-based video classifier.

## 1. Project Summary

We fine-tune `nvidia/Cosmos-Reason2-8B`, a reasoning VLM based on `Qwen3VLForConditionalGeneration`, with QLoRA for football foul severity classification.

Current task:

| Target | Classes |
|---|---|
| `offence_severity` | `No offence`, `Offence + No card`, `Offence + Yellow card`, `Offence + Red card` |

The original two-task setup also included `action_class` 8-way classification. During development, the project was narrowed to the severity task because it is the main VAR-style decision target and the action-class task caused strong mode collapse under the available data and compute budget.

Evaluation metrics:

- Accuracy
- Balanced accuracy over seen classes

Main baseline for comparison:

- VARS / SoccerNet-MVFoul benchmark, Task2 balanced accuracy around 43.0%.

## 2. Reproducibility Scope

Full training reproduction is resource-intensive because it requires:

- Access to the SoccerNet-MVFoul video dataset
- Hugging Face access to `nvidia/Cosmos-Reason2-8B`
- An RTX 5090-class GPU with 32GB VRAM
- CUDA 13 / Blackwell-compatible PyTorch and bitsandbytes
- Several hours of QLoRA training and validation

Therefore, the primary reproducibility target for the final submission is:

1. Run zero-shot late-fusion evaluation from the base model.
2. Run fine-tuned late-fusion evaluation from the submitted LoRA adapter.
3. Reproduce the reported validation/test metrics from saved adapter weights and evaluation scripts.

Training scripts and commands are included for completeness. Full training can be rerun on a compatible GPU, but exact bit-level reproducibility is not guaranteed due to video decoding, CUDA kernels, and autoregressive generation.

## 3. Repository Contents

```text
.
├── README.md
├── DEVELOPMENT_HISTORY.md
├── EXPERIMENT_LOG.md
├── download_mvfoul_720p.py
├── scripts/
│   ├── train/
│   ├── eval/
│   ├── zero_shot/
│   └── sh/
└── .gitignore
```

Important scripts:

| File | Purpose |
|---|---|
| `scripts/train/train_view_expanded_reason.py` | View-expanded single-view QLoRA training |
| `scripts/eval/eval_late_fusion_reason.py` | Per-view inference and action-level late fusion |
| `scripts/eval/refuse_late_fusion_rows.py` | Offline re-fusion using saved per-view rows |
| `scripts/train/frame_utils.py` | Foul-anchored frame sampling utilities |
| `scripts/sh/run_view_expanded_reason.sh` | Wrapper for view-expanded experiments |
| `scripts/zero_shot/zero_shot_eval.py` | Shared label/parser/metric utilities |

## 4. Environment

The experiments were run with the following hardware and software stack:

```text
GPU: 2 x NVIDIA GeForce RTX 5090, 32607 MiB each
Driver: 580.159.03
CUDA shown by nvidia-smi: 13.0
PyTorch CUDA: 13.0
GPU capability: sm_120
Conda environment: mvfoul
Python: 3.10.20
```

Key package versions:

```text
torch==2.11.0+cu130
torchvision==0.26.0+cu130
transformers==5.9.0
accelerate==1.13.0
peft==0.19.1
bitsandbytes==0.49.2
qwen-vl-utils==0.0.14
torchcodec==0.13.0
decord==0.6.0
numpy==2.2.6
```

Before final submission, the exact environment files should be added under `env/`:

```bash
mkdir -p env
conda env export -n mvfoul > env/mvfoul_full.yml
conda env export -n mvfoul --from-history > env/mvfoul_from_history.yml
conda run -n mvfoul pip freeze > env/pip_freeze.txt
nvidia-smi > env/nvidia_smi_5090_cuda13.txt
```

## 5. Data Setup

The dataset is not included in this repository because SoccerNet-MVFoul is a large licensed dataset.

Expected data root:

```text
data/SoccerNet/mvfouls/
```

Expected structure:

```text
data/SoccerNet/mvfouls/
  Train/
    annotations.json
    action_{id}/clip_{idx}.mp4
  Valid/
    annotations.json
    action_{id}/clip_{idx}.mp4
  Test/
    annotations.json
    action_{id}/clip_{idx}.mp4
```

The code uses the official-target filter:

```text
action_class != "Dont know"
Offence != "Between"
Severity not in {"2.0", "4.0"}
```

Expected filtered counts:

| Split | Count |
|---|---:|
| Train | 2,319 official-target actions / 5,277 view samples |
| Valid | 321 actions |
| Test | 247 actions |

## 6. Model Weights

Base model:

```text
nvidia/Cosmos-Reason2-8B
```

The final submission should include either:

1. The fine-tuned LoRA adapter directory, or
2. A publicly accessible download link to the adapter.

Expected adapter directory format:

```text
weights/best_checkpoint/
  adapter_config.json
  adapter_model.safetensors
  tokenizer.json
  tokenizer_config.json
  processor_config.json
  chat_template.jinja
```

Current working output path during development:

```text
outputs/qlora_cosmos8b_view_expanded_reason_clean/best_checkpoint/
```

If the adapter is too large for GitHub, do not commit it. Put the download link in `weights_or_links.txt` and include that file in the LMS zip.

## 7. Zero-shot Evaluation

Zero-shot evaluation uses the base Cosmos-Reason2-8B model without any LoRA adapter. Each view is evaluated independently and then fused at the action level.

Run full Valid zero-shot late-fusion evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_late_fusion_reason.py \
  --model-id nvidia/Cosmos-Reason2-8B \
  --split Valid \
  --limit 0 \
  --num-frames 32 \
  --max-views 0 \
  --max-new-tokens 256 \
  --fusion-rule main_first \
  --log-every 20 \
  --save-every 10 \
  --out-dir outputs/zero_shot_late_fusion_reason_full_valid \
  --output-prefix valid_base_views
```

Resume if interrupted:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_late_fusion_reason.py \
  --model-id nvidia/Cosmos-Reason2-8B \
  --split Valid \
  --limit 0 \
  --num-frames 32 \
  --max-views 0 \
  --max-new-tokens 256 \
  --fusion-rule main_first \
  --log-every 20 \
  --save-every 10 \
  --resume \
  --out-dir outputs/zero_shot_late_fusion_reason_full_valid \
  --output-prefix valid_base_views
```

Outputs:

```text
outputs/zero_shot_late_fusion_reason_full_valid/
  valid_base_views_rows.jsonl
  valid_base_views_rows.json
  valid_base_views_predictions.json
  valid_base_views_metrics.json
```

## 8. Fine-tuned Evaluation

After the LoRA adapter is available, run fine-tuned late-fusion evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_late_fusion_reason.py \
  --model-id nvidia/Cosmos-Reason2-8B \
  --adapter-path weights/best_checkpoint \
  --split Valid \
  --limit 0 \
  --num-frames 32 \
  --max-views 0 \
  --max-new-tokens 256 \
  --fusion-rule main_first \
  --log-every 20 \
  --save-every 10 \
  --out-dir outputs/late_fusion_view_expanded_reason \
  --output-prefix valid_base_views
```

If using the development output path directly:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_late_fusion_reason.py \
  --model-id nvidia/Cosmos-Reason2-8B \
  --adapter-path outputs/qlora_cosmos8b_view_expanded_reason_clean/best_checkpoint \
  --split Valid \
  --limit 0 \
  --num-frames 32 \
  --max-views 0 \
  --max-new-tokens 256 \
  --fusion-rule main_first \
  --log-every 20 \
  --save-every 10 \
  --out-dir outputs/late_fusion_view_expanded_reason \
  --output-prefix valid_base_views
```

## 9. Offline Fusion Evaluation

After `valid_base_views_rows.json` is produced, evaluate several fusion rules without rerunning the model:

```bash
for RULE in main_first clip1_first majority_vote majority_clip1_tiebreak conservative_card; do
  python scripts/eval/refuse_late_fusion_rows.py \
    --rows outputs/late_fusion_view_expanded_reason/valid_base_views_rows.json \
    --fusion-rule "$RULE" \
    --annotations data/SoccerNet/mvfouls/Valid/annotations.json \
    --out-dir outputs/late_fusion_view_expanded_reason \
    --output-prefix "valid_refuse_${RULE}"
done
```

Summarize metrics:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/late_fusion_view_expanded_reason")
for p in sorted(root.glob("valid_refuse_*_metrics.json")):
    m = json.loads(p.read_text())
    print(
        p.name,
        "acc=", m["accuracy_offence_severity"],
        "ba=", m["balanced_accuracy_offence_severity_seen_classes"],
        "view_parse_errors=", m["view_level"]["view_parse_errors"],
    )
PY
```

## 10. Optional Training Command

Full training is optional for reproduction because it is expensive. The current clean training command is:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train/train_view_expanded_reason.py \
  --model-id nvidia/Cosmos-Reason2-8B \
  --data-root data/SoccerNet/mvfouls \
  --output-dir outputs/qlora_cosmos8b_view_expanded_reason_clean \
  --device cuda:0 \
  --num-epochs 3 \
  --num-frames 32 \
  --grad-accum 8 \
  --lr 5e-5 \
  --lora-r 128 \
  --lora-alpha 256 \
  --max-new-tokens 256 \
  --max-train-views 0 \
  --max-eval-views 0 \
  --fusion-rule main_first \
  --balanced-sampling \
  --seed 42
```

Checkpoint behavior:

- Validation runs after each epoch.
- `best_checkpoint/` is saved only after validation completes.
- The saved checkpoint is a LoRA adapter for evaluation, not a full optimizer-state resume checkpoint.

## 11. Current Known Results

Completed historical results are summarized in `EXPERIMENT_LOG.md`.

Current zero-shot late-fusion full-valid evaluation was still running at the time of this README draft. A partial result at 90 Valid actions was:

| Setting | Samples | Accuracy | Balanced Accuracy | View Parse Errors |
|---|---:|---:|---:|---:|
| Zero-shot late fusion, `main_first` | 90 | 18.89 | 14.89 | 18 / 207 views |

Observed zero-shot issues:

- Missing or malformed `<answer>` tags
- CJK drift in some raw outputs
- Overly long reasoning outputs
- Underprediction of `Offence + No card`
- Overprediction of Yellow/Red severity

The fine-tuned results should be added after the running clean training/evaluation finishes.

## 12. Baselines and Existing Code

Baseline source:

- VARS / SoccerNet-MVFoul benchmark. The reported reference number used here is the VARS Task2 balanced accuracy, approximately 43.0%.

The full VARS baseline code is not included in this submission. Only the source and reported benchmark numbers are cited for comparison.

## 13. AI / Coding Tools Used

AI and coding-agent tools were used for implementation support.

Tools used:

- OpenAI ChatGPT / Codex-style coding agent

How they were used:

- Data-structure inspection and experiment planning
- Script generation and debugging
- Parser and evaluation utility development
- QLoRA training script organization
- Documentation and reproducibility checklist drafting

The final project decisions, experiment execution, result interpretation, and presentation preparation were performed by the student/team.

## 14. Final Submission Checklist

Before creating the LMS zip, add or update the following items:

- [ ] `env/mvfoul_full.yml`
- [ ] `env/mvfoul_from_history.yml`
- [ ] `env/pip_freeze.txt`
- [ ] `env/nvidia_smi_5090_cuda13.txt`
- [ ] `data_access.md` with SoccerNet-MVFoul download/access instructions
- [ ] `weights_or_links.txt` with a working adapter download link, or include `weights/best_checkpoint/`
- [ ] Final zero-shot metrics copied to `results/`
- [ ] Final fine-tuned validation metrics copied to `results/`
- [ ] Test metrics, if available
- [ ] Presentation slides PDF
- [ ] Final README result table updated

Suggested final zip layout:

```text
sn-mvfoul-submission/
  README.md
  EXPERIMENT_LOG.md
  DEVELOPMENT_HISTORY.md
  data_access.md
  weights_or_links.txt
  env/
  results/
  scripts/
  slides.pdf
```
