# SoccerNet-MVFoul Severity Classification with Cosmos-Reason2 QLoRA

This repository contains the code, final LoRA adapter, evaluation artifacts, and reproducibility instructions for our SoccerNet-MVFoul offence severity classification project.

The final system uses `nvidia/Cosmos-Reason2-8B` with QLoRA and evaluates each video incident by fixed-label likelihood scoring instead of free-form text generation.

## 1. Task

Target task:

| Target | Classes |
|---|---|
| `offence_severity` | `No offence`, `Offence + No card`, `Offence + Yellow card`, `Offence + Red card` |

Metrics:

- Accuracy
- Balanced accuracy over seen classes

Baseline reference:

- VARS / SoccerNet-MVFoul benchmark, Task2 balanced accuracy approximately 43.0%.

We originally explored both `action_class` and `offence_severity`, but the final submitted system focuses on `offence_severity` because it is the main VAR-style decision target and was more stable under the available data and compute budget.

## 2. Final Method

Early experiments used free-form reasoning generation. The model generated `<think>` and `<answer>` text, then the predicted JSON was parsed. This caused avoidable evaluation noise:

- Missing or malformed answer tags
- Incomplete JSON
- CJK/output drift
- Long repetitive outputs
- Sensitivity to `max_new_tokens`

The final method treats Cosmos-Reason2 as a fixed-label likelihood scorer. For each video view, the evaluator scores the four allowed candidate answers:

```json
{"offence_severity": "No offence"}
{"offence_severity": "Offence + No card"}
{"offence_severity": "Offence + Yellow card"}
{"offence_severity": "Offence + Red card"}
```

For each candidate answer, the score is the average token log likelihood:

```text
score(label) = -NLL(candidate_answer) / token_count(candidate_answer)
```

The highest scoring label is selected. Multi-view action-level prediction is produced with score-level fusion. The final main setting uses:

```text
fusion_method = clip1
prior_alpha = 0.005
```

A small class-prior correction is applied as:

```text
adjusted_score(label) = score(label) - alpha * log(train_prior(label))
```

This removes parsing failures because the model no longer generates open-ended answer text during classification.

## 3. Reproducibility Scope

Full training reproduction is resource-intensive because it requires:

- Access to the SoccerNet-MVFoul video dataset
- Hugging Face access to `nvidia/Cosmos-Reason2-8B`
- An RTX 5090-class GPU with 32GB VRAM
- CUDA 13 / Blackwell-compatible PyTorch and bitsandbytes
- Several hours of QLoRA training and validation

The intended reproducibility target for this submission is:

1. Rebuild the Python environment from `requirements.txt`.
2. Place SoccerNet-MVFoul under the documented local data layout.
3. Run the provided final LoRA adapter with likelihood-scoring evaluation.
4. Reproduce the reported Valid/Test metrics from the included code, adapter, and evaluation data.

Training scripts are included for completeness. Exact bit-level training reproduction is not guaranteed because video decoding, CUDA kernels, and large-model training can vary across machines.

## 4. Repository Layout

Recommended final package layout for this project component:

```text
sn-mvfoul-submission/
  README.md
  requirements.txt
  weights_or_links.txt
  download_mvfoul_720p.py
  scripts/
  results/
  weights/
```

Main paths:

| Path | Purpose |
|---|---|
| `README.md` | Main submission and reproduction guide |
| `requirements.txt` | Python package specification |
| `weights_or_links.txt` | Adapter location and large-weight note |
| `download_mvfoul_720p.py` | Helper script for SoccerNet-MVFoul download setup |
| `scripts/eval/eval_late_fusion_likelihood.py` | Final likelihood-scoring evaluation |
| `scripts/train/train_view_expanded_likelihood.py` | Likelihood-aligned QLoRA training |
| `scripts/eval/eval_late_fusion_reason.py` | Historical free-generation evaluation |
| `scripts/eval/refuse_late_fusion_rows.py` | Offline re-fusion for saved generation rows |
| `results/` | Included Valid/Test result artifacts |
| `weights/final_likelihood_epoch3/` | Final LoRA adapter |

Only `README.md` is required as a markdown document for the final integrated team repository. Auxiliary development notes were moved out of the submission surface and should not be included.

## 5. Environment

Experiments were run with:

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

Create a fresh conda environment:

```bash
cd /path/to/sn-mvfoul-submission
eval "$(conda shell.bash hook)"
conda create -n mvfoul python=3.10 ffmpeg pip -c conda-forge -y
conda activate mvfoul
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check CUDA/PyTorch availability:

```bash
python - <<'PY'
import torch
import transformers
import accelerate
import peft
import bitsandbytes as bnb

print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('gpu count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('gpu 0:', torch.cuda.get_device_name(0))
print('transformers:', transformers.__version__)
print('accelerate:', accelerate.__version__)
print('peft:', peft.__version__)
print('bitsandbytes:', getattr(bnb, '__version__', 'unknown'))
PY
```

Expected on a compatible GPU server:

```text
cuda available: True
torch cuda: 13.0
bitsandbytes import succeeds
```

## 6. Data Setup

This project uses the SoccerNet-MVFoul video dataset. The dataset is not included in this repository because it is large and distributed under SoccerNet's data access terms.

Expected local path:

```text
data/SoccerNet/mvfouls/
```

Expected directory layout:

```text
data/SoccerNet/mvfouls/
  Train/
    annotations.json
    action_{id}/
      clip_0.mp4
      clip_1.mp4
      ...
  Valid/
    annotations.json
    action_{id}/
      clip_0.mp4
      clip_1.mp4
      ...
  Test/
    annotations.json
    action_{id}/
      clip_0.mp4
      clip_1.mp4
      ...
```

The scripts assume this path by default:

```bash
--data-root data/SoccerNet/mvfouls
```

Download procedure:

1. Create or use a SoccerNet account with access to SoccerNet-MVFoul.
2. Make sure the project environment is active and dependencies are installed from `requirements.txt`.
3. Run the included helper script:

```bash
python download_mvfoul_720p.py
```

The script uses `SoccerNet.Downloader.SoccerNetDownloader`, asks for the SoccerNet password interactively, and downloads the 720p MVFoul data under `data/SoccerNet/`.

If the dataset already exists elsewhere, copy or symlink it so that the final path is:

```text
data/SoccerNet/mvfouls/
```

Depending on SoccerNet account permissions and token/password setup, the reviewer may need to authenticate through the official SoccerNet instructions before the download succeeds.

Official-target filtering used for the submitted experiments:

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

## 7. Model Weights

Base model:

```text
nvidia/Cosmos-Reason2-8B
```

Hugging Face access is required to load the base model. Before running evaluation or training, log in with an account that can access `nvidia/Cosmos-Reason2-8B`:

```bash
huggingface-cli login
```

Alternatively, set a token in the shell before running the scripts:

```bash
export HF_TOKEN=your_huggingface_token
```

If the model is gated for the account, accept/request access on the Hugging Face model page first. The submitted LoRA adapter is included locally, but the base model weights are still loaded from Hugging Face unless already cached.

Final submitted LoRA adapter:

```text
weights/final_likelihood_epoch3/
  adapter_config.json
  adapter_model.safetensors
  tokenizer.json
  tokenizer_config.json
  processor_config.json
  chat_template.jinja
```

The adapter is not a full copy of the base model. `adapter_model.safetensors` is approximately 333 MiB. It is included in the local LMS zip package. If this project is submitted through GitHub instead of a zip, upload the adapter separately and provide a public download link in `weights_or_links.txt`.

## 8. Final Results

Final model setting:

```text
Base model: nvidia/Cosmos-Reason2-8B
Adapter: weights/final_likelihood_epoch3
Training: view-expanded QLoRA with short JSON likelihood-aligned targets
Inference: fixed-label likelihood scoring
Frames per view: 32
Fusion: clip1
Prior alpha: 0.005
```

Main Valid/Test result:

| Split | Samples | Fusion | Prior alpha | Accuracy | Balanced Accuracy | Parse Errors |
|---|---:|---|---:|---:|---:|---:|
| Valid | 321 | `clip1` | 0.005 | 57.63 | 31.70 | 0 |
| Test | 247 | `clip1` | 0.005 | 57.49 | 29.44 | 0 |

Additional Valid candidates:

| Candidate | Fusion | Prior alpha | Valid Accuracy | Valid Balanced Accuracy | Notes |
|---|---|---:|---:|---:|---|
| Main | `clip1` | 0.005 | 57.63 | 31.70 | Best Acc/BA tradeoff |
| BA-focused | `weighted_clip1` | 0.05 | 50.16 | 33.81 | Highest usable balanced accuracy |
| All-view | `score_mean` | 0.03 | 56.07 | 31.42 | Uses all views |

Comparison with earlier generation-based experiments:

| Setting | Valid Accuracy | Valid Balanced Accuracy | Parse Errors |
|---|---:|---:|---:|
| Zero-shot generation, best listed fusion | 22.12 | 18.61 | 70 / 763 views |
| Fine-tuned generation, `main_first` | 38.94 | 23.71 | Present |
| Fine-tuned generation, `clip1_first` | 38.01 | 25.62 | Present |
| Reason-clean adapter + likelihood scoring | 53.58 | 26.69 | 0 |
| Likelihood-aligned epoch 3, main | 57.63 | 31.70 | 0 |
| Likelihood-aligned epoch 3, BA-focused | 50.16 | 33.81 | 0 |

Included result artifacts:

```text
results/likelihood_final_valid_selection/
results/likelihood_epoch3_valid/
results/likelihood_final_test_main/
results/zero_shot_late_fusion_reason_full_valid/
```

Main Test result files:

```text
results/likelihood_final_test_main/
  test_likelihood_best_metrics.json
  test_likelihood_best_predictions.json
  test_likelihood_best_rows.json
  test_likelihood_metrics_grid.json
  test_likelihood_score_rows.json
  test_likelihood_score_rows.jsonl
```

## 9. Reproduce Final Evaluation

Run syntax and CLI checks:

```bash
python -m compileall scripts
python scripts/eval/eval_late_fusion_likelihood.py --help
python scripts/train/train_view_expanded_likelihood.py --help
```

Check adapter files:

```bash
ls weights/final_likelihood_epoch3/adapter_config.json
ls weights/final_likelihood_epoch3/adapter_model.safetensors
ls weights/final_likelihood_epoch3/tokenizer_config.json
ls weights/final_likelihood_epoch3/processor_config.json
```

Run a small Valid smoke evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_late_fusion_likelihood.py \
  --model-id nvidia/Cosmos-Reason2-8B \
  --adapter-path weights/final_likelihood_epoch3 \
  --data-root data/SoccerNet/mvfouls \
  --split Valid \
  --limit 2 \
  --num-frames 32 \
  --max-views 2 \
  --candidate-format json \
  --score-reduction mean \
  --fusion-methods clip1 \
  --prior-alphas 0.005 \
  --out-dir /tmp/mvfoul_likelihood_smoke \
  --output-prefix valid2_likelihood_smoke
```

Expected smoke outputs:

```text
/tmp/mvfoul_likelihood_smoke/valid2_likelihood_smoke_score_rows.json
/tmp/mvfoul_likelihood_smoke/valid2_likelihood_smoke_metrics_grid.json
/tmp/mvfoul_likelihood_smoke/valid2_likelihood_smoke_best_metrics.json
/tmp/mvfoul_likelihood_smoke/valid2_likelihood_smoke_best_predictions.json
```

Run the final Valid evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_late_fusion_likelihood.py \
  --model-id nvidia/Cosmos-Reason2-8B \
  --adapter-path weights/final_likelihood_epoch3 \
  --data-root data/SoccerNet/mvfouls \
  --split Valid \
  --limit 0 \
  --num-frames 32 \
  --max-views 0 \
  --candidate-format json \
  --score-reduction mean \
  --fusion-methods clip1 \
  --prior-alphas 0.005 \
  --out-dir outputs/likelihood_final_valid_main \
  --output-prefix valid_likelihood \
  --log-every 20 \
  --save-every 10
```

Expected Valid metrics:

```text
accuracy_offence_severity: 57.63239875389408
balanced_accuracy_offence_severity_seen_classes: 31.69642857142857
```

Run the final Test evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_late_fusion_likelihood.py \
  --model-id nvidia/Cosmos-Reason2-8B \
  --adapter-path weights/final_likelihood_epoch3 \
  --data-root data/SoccerNet/mvfouls \
  --split Test \
  --limit 0 \
  --num-frames 32 \
  --max-views 0 \
  --candidate-format json \
  --score-reduction mean \
  --fusion-methods clip1 \
  --prior-alphas 0.005 \
  --out-dir outputs/likelihood_final_test_main \
  --output-prefix test_likelihood \
  --log-every 20 \
  --save-every 10
```

Expected Test metrics:

```text
accuracy_offence_severity: 57.48987854251012
balanced_accuracy_offence_severity_seen_classes: 29.44035582456382
```

Compare a generated Valid run against the included reference metrics:

```bash
python - <<'PY'
import json, math
new = json.load(open('outputs/likelihood_final_valid_main/valid_likelihood_metrics_grid.json'))
ref = json.load(open('results/likelihood_final_valid_selection/epoch3_acc_ba_tradeoff_valid_metrics.json'))
grid = new['grid'] if isinstance(new, dict) and 'grid' in new else new
row = next(x for x in grid if x.get('fusion_method') == 'clip1' and abs(float(x.get('prior_alpha')) - 0.005) < 1e-12)
metrics = row.get('metrics', row)
for key in ['accuracy_offence_severity', 'balanced_accuracy_offence_severity_seen_classes']:
    assert math.isclose(float(metrics[key]), float(ref[key]), rel_tol=0, abs_tol=1e-9), (key, metrics[key], ref[key])
print('main likelihood Valid metrics match')
PY
```

## 10. Optional Training Command

Training is not the primary reproduction requirement. The final likelihood-aligned training command was:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train/train_view_expanded_likelihood.py \
  --model-id nvidia/Cosmos-Reason2-8B \
  --data-root data/SoccerNet/mvfouls \
  --output-dir outputs/qlora_cosmos8b_view_expanded_likelihood \
  --device cuda:0 \
  --num-epochs 3 \
  --num-frames 32 \
  --grad-accum 8 \
  --lr 5e-5 \
  --lora-r 32 \
  --lora-alpha 64 \
  --target-format json \
  --score-reduction mean \
  --fusion-methods score_mean,score_sum,clip0,clip1,weighted_clip1 \
  --prior-alphas 0,0.001,0.0025,0.005,0.01,0.02,0.03,0.05,0.075 \
  --max-train-views 0 \
  --max-eval-views 0 \
  --balanced-sampling \
  --save-every-epoch \
  --no-epoch-eval \
  --seed 42
```

A smaller smoke run can be used to verify that the training pipeline starts:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train/train_view_expanded_likelihood.py \
  --model-id nvidia/Cosmos-Reason2-8B \
  --data-root data/SoccerNet/mvfouls \
  --output-dir /tmp/mvfoul_train_smoke \
  --device cuda:0 \
  --num-frames 32 \
  --lora-r 16 \
  --lora-alpha 32 \
  --train-limit 4 \
  --valid-limit 2 \
  --max-train-views 1 \
  --max-eval-views 1 \
  --balanced-sampling \
  --smoke-test
```

## 11. Baselines and Existing Code

Baseline source:

- VARS / SoccerNet-MVFoul benchmark. The reported reference number used here is the VARS Task2 balanced accuracy, approximately 43.0%.

The full VARS baseline code is not included in this submission. Only the source and reported benchmark number are cited for comparison.

## 12. AI / Coding Tools Used

AI and coding-agent tools were used for implementation support.

Tools used:

- OpenAI ChatGPT / Codex-style coding agent

How they were used:

- Data-structure inspection and experiment planning
- Script generation and debugging
- Parser and evaluation utility development
- QLoRA training/evaluation script organization
- Documentation and reproducibility checklist drafting

The final project decisions, experiment execution, result interpretation, and presentation preparation were performed by the student/team.

## 13. Final Submission Notes

For integration into a larger team repository, this project component only needs one markdown document: this `README.md`. Do not include auxiliary markdown notes. The following files/directories are the reproducibility package for this component:

```text
README.md
requirements.txt
weights_or_links.txt
download_mvfoul_720p.py
scripts/
results/
weights/final_likelihood_epoch3/
```

The dataset itself is not included. It must be downloaded separately through SoccerNet and placed under `data/SoccerNet/mvfouls/`.
