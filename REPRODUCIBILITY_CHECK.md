# Reproducibility Check Guide

This guide describes how to check the submission from the TA/reviewer point of view. It is split into checks that can be run now and checks that should be run after the final LoRA adapter and fine-tuned metrics are added.

## 0. Scope

Full training reproduction is resource-intensive for this project because it requires:

- SoccerNet-MVFoul video data access
- Hugging Face access to `nvidia/Cosmos-Reason2-8B`
- A CUDA 13 / RTX 5090-class GPU environment
- Several hours of QLoRA training and validation

Therefore, the final reproduction target is:

1. Verify the submitted code and environment specification.
2. Reproduce included zero-shot offline fusion metrics from saved per-view rows.
3. Reproduce fine-tuned Valid/Test metrics from the submitted LoRA adapter once the adapter is added.
4. Keep full training commands as optional, resource-dependent reproduction.

## 1. Fresh Submission Check

Run these commands from the root of the unzipped submission directory.

Unless a command explicitly says otherwise, Python commands below assume that a clean reproduction environment is active:

```bash
conda activate mvfoul
```

```bash
pwd
find . -maxdepth 2 -type f | sort
```

Expected important files/directories:

```text
README.md
DEVELOPMENT_HISTORY.md
EXPERIMENT_LOG.md
REPRODUCIBILITY_CHECK.md
data_access.md
weights_or_links.txt
download_mvfoul_720p.py
requirements.txt
results/
scripts/
```

Check that large local-only artifacts are not accidentally included:

```bash
find . \
  -path './.git' -prune -o \
  -name '__pycache__' -o \
  -name '*.pyc' -o \
  -path './data/*' -o \
  -path './outputs/*' -o \
  -path './checkpoints/*' -print
```

For the final zip, this command should print nothing or only intentional files documented in the README.

## 2. Clean Environment Rebuild

This section checks reproducibility from a fresh conda environment instead of relying on the existing development environment.

The package specification should exist at the repository root:

```bash
ls -lh requirements.txt
```

Expected file:

```text
requirements.txt
```

### 2.1 Create a fresh env from minimal conda packages

The recommended rebuild path is to create a minimal conda environment first, then install packages from the included `requirements.txt`. That file includes the PyTorch CUDA 13 wheel index and pinned Python package versions.

If an environment named `mvfoul` already exists, either choose a different temporary name for local testing or remove the old environment only after confirming it is safe to delete:

```bash
conda env list
# Optional, only if you intentionally want to replace the old env:
# conda env remove -n mvfoul -y
```

```bash
cd /path/to/sn-mvfoul-submission
eval "$(conda shell.bash hook)"
conda create -n mvfoul python=3.10 ffmpeg pip -c conda-forge -y
conda activate mvfoul
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `conda activate` is not available in a non-interactive shell, run `eval "$(conda shell.bash hook)"` first or use `conda run -n mvfoul ...` for individual commands.

After installation, verify the interpreter:

```bash
which python
python --version
```

Expected Python version:

```text
Python 3.10.x
```

### 2.2 Key package versions

The key versions to match are listed in `README.md`, especially:

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
```

### 2.3 Package and CUDA sanity check

Run this after activating `mvfoul`:

```bash
python - <<'PY'
import torch
import transformers
import accelerate
import peft
import bitsandbytes as bnb
import numpy as np

print('python import ok')
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('torch cuda:', torch.version.cuda)
print('gpu count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('gpu 0:', torch.cuda.get_device_name(0))
print('transformers:', transformers.__version__)
print('accelerate:', accelerate.__version__)
print('peft:', peft.__version__)
print('bitsandbytes:', getattr(bnb, '__version__', 'unknown'))
print('numpy:', np.__version__)
PY
```

Expected result on the development server:

```text
cuda available: True
torch cuda: 13.0
bitsandbytes import succeeds
```

Check GPU state:

```bash
nvidia-smi
```

### 2.4 Code syntax and CLI smoke check

```bash
python -m compileall scripts
```

This does not validate model inference, but it catches syntax errors in the submitted scripts.

Also confirm that the main CLIs load:

```bash
python scripts/eval/eval_late_fusion_reason.py --help
python scripts/eval/refuse_late_fusion_rows.py --help
python scripts/train/train_view_expanded_reason.py --help
```

## 3. Data Layout Check

The dataset is not included directly. After downloading SoccerNet-MVFoul, place it at:

```text
data/SoccerNet/mvfouls/
```

Check the required files:

```bash
ls data/SoccerNet/mvfouls/Train/annotations.json
ls data/SoccerNet/mvfouls/Valid/annotations.json
ls data/SoccerNet/mvfouls/Test/annotations.json
```

Optional quick structure check:

```bash
python -c "from pathlib import Path; root=Path('data/SoccerNet/mvfouls'); assert (root/'Train/annotations.json').exists(); assert (root/'Valid/annotations.json').exists(); assert (root/'Test/annotations.json').exists(); print('dataset annotations found')"
```

## 4. Current No-GPU Result Reproduction

The included zero-shot late-fusion result can be checked without loading Cosmos-Reason2 and without GPU inference. This uses saved per-view rows and reruns only the offline fusion rule.

This still requires the Python dependencies from the project environment because the evaluation utilities import the shared project modules.

Run conservative-card fusion from the saved rows:

```bash
python scripts/eval/refuse_late_fusion_rows.py \
  --rows results/zero_shot_late_fusion_reason_full_valid/valid_base_views_rows.json \
  --fusion-rule conservative_card \
  --out-dir /tmp/mvfoul_check \
  --output-prefix zero_shot_conservative_card_check
```

Expected metrics:

```text
num_samples: 321
accuracy_offence_severity: 22.118380062305295
balanced_accuracy_offence_severity_seen_classes: 18.612637362637365
view_level.total_views: 763
view_level.view_parse_errors: 70
```

Compare the regenerated metric file with the included reference:

```bash
python -c "import json, math; a=json.load(open('/tmp/mvfoul_check/zero_shot_conservative_card_check_metrics.json')); b=json.load(open('results/zero_shot_late_fusion_reason_full_valid/valid_refuse_conservative_card_metrics.json')); keys=['num_samples','accuracy_offence_severity','balanced_accuracy_offence_severity_seen_classes']; assert all(math.isclose(float(a[k]), float(b[k]), rel_tol=0, abs_tol=1e-9) for k in keys); assert a['view_level']['view_parse_errors']==b['view_level']['view_parse_errors']; print('offline fusion metrics match')"
```

Repeat for all fusion rules if desired:

```bash
for RULE in main_first clip1_first majority_vote majority_clip1_tiebreak conservative_card; do
  python scripts/eval/refuse_late_fusion_rows.py \
    --rows results/zero_shot_late_fusion_reason_full_valid/valid_base_views_rows.json \
    --fusion-rule "$RULE" \
    --out-dir /tmp/mvfoul_check \
    --output-prefix "zero_shot_${RULE}_check"
done
```

## 5. Zero-Shot Full Inference Check

This check requires the SoccerNet-MVFoul Valid split, Hugging Face model access, and a CUDA GPU.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_late_fusion_reason.py \
  --model-id nvidia/Cosmos-Reason2-8B \
  --data-root data/SoccerNet/mvfouls \
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

Expected output files:

```text
outputs/zero_shot_late_fusion_reason_full_valid/valid_base_views_rows.jsonl
outputs/zero_shot_late_fusion_reason_full_valid/valid_base_views_rows.json
outputs/zero_shot_late_fusion_reason_full_valid/valid_base_views_predictions.json
outputs/zero_shot_late_fusion_reason_full_valid/valid_base_views_metrics.json
```

This full inference may not be bit-identical because generation, video decoding, and CUDA kernels can vary. The included saved rows are the deterministic reference for the submitted zero-shot metrics.

## 6. Final Fine-Tuned Adapter Check

Run this after `weights/best_checkpoint/` is added or after the adapter is downloaded from the link in `weights_or_links.txt`.

Check adapter files:

```bash
ls weights/best_checkpoint/adapter_config.json
ls weights/best_checkpoint/adapter_model.safetensors
```

Run a small smoke evaluation first:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_late_fusion_reason.py \
  --model-id nvidia/Cosmos-Reason2-8B \
  --adapter-path weights/best_checkpoint \
  --data-root data/SoccerNet/mvfouls \
  --split Valid \
  --limit 2 \
  --num-frames 32 \
  --max-views 2 \
  --max-new-tokens 256 \
  --fusion-rule main_first \
  --out-dir /tmp/mvfoul_adapter_smoke \
  --output-prefix valid2_adapter_smoke
```

Expected smoke output:

```text
/tmp/mvfoul_adapter_smoke/valid2_adapter_smoke_rows.json
/tmp/mvfoul_adapter_smoke/valid2_adapter_smoke_metrics.json
```

Then run full Valid evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_late_fusion_reason.py \
  --model-id nvidia/Cosmos-Reason2-8B \
  --adapter-path weights/best_checkpoint \
  --data-root data/SoccerNet/mvfouls \
  --split Valid \
  --limit 0 \
  --num-frames 32 \
  --max-views 0 \
  --max-new-tokens 256 \
  --fusion-rule main_first \
  --log-every 20 \
  --save-every 10 \
  --out-dir outputs/late_fusion_view_expanded_reason_valid \
  --output-prefix valid_finetuned_views
```

After full Valid inference, run offline fusion comparisons:

```bash
for RULE in main_first clip1_first majority_vote majority_clip1_tiebreak conservative_card; do
  python scripts/eval/refuse_late_fusion_rows.py \
    --rows outputs/late_fusion_view_expanded_reason_valid/valid_finetuned_views_rows.json \
    --fusion-rule "$RULE" \
    --annotations data/SoccerNet/mvfouls/Valid/annotations.json \
    --out-dir outputs/late_fusion_view_expanded_reason_valid \
    --output-prefix "valid_finetuned_refuse_${RULE}"
done
```

Copy the final selected metrics to `results/` and update `README.md` and `EXPERIMENT_LOG.md`.

## 7. Optional Training Smoke Test

This is not the primary reproduction target, but it checks that the training pipeline can start on a compatible machine.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train/train_view_expanded_reason.py \
  --model-id nvidia/Cosmos-Reason2-8B \
  --data-root data/SoccerNet/mvfouls \
  --output-dir /tmp/mvfoul_train_smoke \
  --device cuda:0 \
  --num-frames 32 \
  --lora-r 16 \
  --lora-alpha 32 \
  --max-new-tokens 128 \
  --smoke-test \
  --balanced-sampling
```

Expected behavior:

- Loads two view-level training samples.
- Runs a minimal training loop.
- Runs validation on two Valid actions.
- Writes smoke outputs under `/tmp/mvfoul_train_smoke`.

## 8. Final Pre-Zip Checklist

Before creating the LMS zip:

```bash
git status --short
find scripts -type d -name __pycache__
find scripts -type f -name '*.pyc'
du -sh .
```

Expected final state:

- README contains final fine-tuned Valid/Test metrics.
- `weights_or_links.txt` contains a real adapter link, or `weights/best_checkpoint/` is included intentionally.
- `results/` contains zero-shot and fine-tuned metrics/predictions/rows needed to reproduce reported tables.
- `slides.pdf` is present.
- No dataset, raw `outputs/`, cache directories, or accidental large checkpoints are included unless explicitly documented.
