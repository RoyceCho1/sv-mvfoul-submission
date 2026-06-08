# SoccerNet MVFoul — Experiment Log

> 기준일: 2026-06-08  
> 목표: SoccerNet-MVFoul에서 VLM 기반 축구 파울 심각도(`offence_severity`) 분류

---

## 1. 현재 태스크 정의

초기에는 VARS와 동일하게 두 태스크를 동시에 다뤘다.

| 태스크 | 분류 대상 | 클래스 수 | 현재 사용 여부 |
|---|---|---:|---|
| Task 1 | `action_class` | 8 | 보류 |
| Task 2 | `offence_severity` | 4 | **현재 메인** |

현재 실험은 `offence_severity`만 예측한다.

| Label | 의미 |
|---|---|
| `No offence` | 파울 아님 |
| `Offence + No card` | 파울이지만 카드 없음 |
| `Offence + Yellow card` | 옐로 카드 수준 |
| `Offence + Red card` | 레드 카드 수준 |

**평가 지표**: accuracy + balanced accuracy (seen classes)  
**비교 기준**: VARS 논문 Task2 balanced accuracy 약 **43.0%**

---

## 2. 데이터셋 요약

```text
data/SoccerNet/mvfouls/
  Train/
  Valid/
  Test/
    action_{id}/
      clip_0.mp4       # main camera
      clip_1.mp4 ...   # replay / alternate views
    annotations.json
```

공식 annotation에서 다음 케이스는 제외한다.

```text
action_class == "Dont know"
Offence == "Between"
Severity in {"2.0", "4.0"}
```

| Split | 원본 Actions | official target actions |
|---|---:|---:|
| Train | 2,916 | 2,319 actions / 5,277 view samples (view-expanded 기준) |
| Valid | 411 | 321 actions |
| Test | 301 | 247 actions |

View-expanded train 분포:

| View | Count |
|---|---:|
| clip_0 | 2,319 |
| clip_1 | 2,319 |
| clip_2 | 538 |
| clip_3 | 101 |

Severity view-counts:

| Label | Train view count |
|---|---:|
| `Offence + No card` | 2,925 |
| `Offence + Yellow card` | 1,598 |
| `No offence` | 678 |
| `Offence + Red card` | 76 |

---

## 3. 모델과 학습 방식

### Base Model

```text
nvidia/Cosmos-Reason2-8B
Qwen3VLForConditionalGeneration 기반 VLM
```

### QLoRA

```text
4-bit NF4 quantization
Vision encoder freeze
LoRA target: LLM attention + MLP modules
Optimizer: PagedAdamW8bit
Gradient accumulation: 8
```

현재 view-expanded clean run:

```text
OUTPUT_DIR      = outputs/qlora_cosmos8b_view_expanded_reason_clean
MODEL_ID        = nvidia/Cosmos-Reason2-8B
NUM_EPOCHS      = 3
NUM_FRAMES      = 32
LR              = 5e-5
LORA_R          = 128
LORA_ALPHA      = 256
MAX_TRAIN_VIEWS = 0  # all views
MAX_EVAL_VIEWS  = 0  # all views
FUSION_RULE     = main_first
```

현재 실행 중인 프로세스:

```text
GPU 1: train_view_expanded_reason.py
```

---

## 4. Reasoning Prompt

현재 출력은 severity-only다.

```text
<think>
contact: yes/no/unclear
ball_play: yes/no/unclear
force: low/medium/high/unclear
risk: low/medium/high/unclear
offence: yes/no
card_threshold: no_card/yellow/red/none
decision: one short reason
</think>
<answer>{"offence_severity": "Offence + Yellow card"}</answer>
```

평가는 `<answer>` 안의 JSON 또는 parser가 복구한 `offence_severity`만 사용한다.

---

## 5. 멀티뷰 전략

### 폐기한 방향: Early Fusion

```text
[clip_0 video] + [clip_1 video] + prompt -> one answer
```

문제:

- 긴 입력에서 reasoning 형식이 더 쉽게 붕괴
- 2B 모델은 JSON schema 오류와 CJK drift가 큼
- 8B MV도 mode collapse 발생
- 어떤 view가 문제인지 분석하기 어려움

### 현재 방향: View-expanded SV + Late Fusion

학습은 view 단위로 한다.

```text
action_i/clip_0.mp4 -> same offence_severity
action_i/clip_1.mp4 -> same offence_severity
action_i/clip_2.mp4 -> same offence_severity
```

추론은 view별로 독립 실행 후 action 단위로 fusion한다.

```text
clip_0 -> pred_0
clip_1 -> pred_1
clip_2 -> pred_2
fusion(pred_0, pred_1, pred_2) -> final prediction
```

Fusion rules:

| Rule | 설명 |
|---|---|
| `main_first` | clip_0 우선 |
| `clip1_first` | clip_1 우선 |
| `majority_vote` | 전체 view 다수결, tie는 clip_0 |
| `majority_clip1_tiebreak` | 전체 view 다수결, tie는 clip_1 |
| `conservative_card` | 단독 Red 예측을 보수적으로 낮춤 |

---

## 6. 지금까지의 주요 결과

### 과거 baseline / 실패 실험

| 실험 | 모델 | 설정 | Valid Task1 | Valid Task2 | Avg | 판단 |
|---|---|---|---:|---:|---:|---|
| VARS 논문 | MViT | multi-view | 47.0 | 43.0 | 45.0 | 비교 기준 |
| Cosmos-8B zero-shot JSON | 8B | single-view | 11.8 | 25.0 | 18.4 | 낮음 |
| Cosmos-8B zero-shot think+answer | 8B | single-view | 11.1 | 19.5 | 15.3 | 낮음 |
| SV Reason v1 | 8B | 16 frames, LR=2e-4 | 21.5 | 25.2 | 23.3 | 이전 최고 |
| SV Reason Answer | 8B | answer-only masking | 5.8 | 15.1 | 10.4 | 폐기 |
| 2B MV Reason | 2B | 2 views | 4.8 | 1.9 | 3.4 | 실패 |
| 8B MV Reason v1 | 8B | early fusion | 10.6 | 9.3 | 10.0 | mode collapse |

### 현재 zero-shot late fusion partial

경로:

```text
outputs/zero_shot_late_fusion_reason_full_valid/
```

실행 상태:

```text
GPU 0: eval_late_fusion_reason.py
```

중간 저장 파일:

```text
valid_base_views_rows.jsonl        # action 1개마다 append
valid_base_views_rows.json         # --save-every마다 갱신
valid_base_views_metrics.json      # --save-every마다 갱신
valid_base_views_predictions.json  # --save-every마다 갱신
```

90 samples 기준 partial metrics:

| Metric | Value |
|---|---:|
| Accuracy | 18.89 |
| Balanced accuracy | 14.89 |
| View parse errors | 18 / 207 views |

Support:

| Label | Count |
|---|---:|
| `No offence` | 11 |
| `Offence + No card` | 50 |
| `Offence + Yellow card` | 27 |
| `Offence + Red card` | 2 |

관찰:

- zero-shot은 `<answer>` 태그를 잘 지키지 못함
- CJK drift 발생
- 출력이 지나치게 길어지는 경우 많음
- `Offence + No card`를 거의 예측하지 못함
- Yellow/Red 과대예측 경향

이 결과는 fine-tuning 필요성을 보여주는 baseline으로 사용한다.

---

## 7. 현재 실행 중인 실험

| 실험 | GPU | 상태 | 출력 |
|---|---:|---|---|
| View-expanded SV QLoRA clean train | 1 | 실행 중 | `outputs/qlora_cosmos8b_view_expanded_reason_clean` |
| Zero-shot late fusion full Valid | 0 | 실행 중 | `outputs/zero_shot_late_fusion_reason_full_valid` |

이전 깨진 view-expanded run:

```text
outputs/archive_20260608/training_checkpoints/qlora_cosmos8b_view_expanded_reason
```

원인:

```text
TypeError: Object of type set is not JSON serializable
```

해결:

- `make_jsonable()` 추가
- train/eval/offline fusion 저장부에 적용
- 새 clean run은 별도 output dir에서 재시작

---

## 8. 코드와 스크립트

| 파일 | 역할 |
|---|---|
| `scripts/train/train_reason.py` | single-view reasoning QLoRA |
| `scripts/train/train_multiview_reason.py` | early-fusion multiview reasoning QLoRA |
| `scripts/train/train_view_expanded_reason.py` | **view-expanded SV 학습** |
| `scripts/eval/eval_late_fusion_reason.py` | **view별 inference + action-level fusion** |
| `scripts/eval/refuse_late_fusion_rows.py` | 저장된 rows로 offline fusion 재평가 |
| `scripts/train/frame_utils.py` | foul-anchored frame sampling |
| `scripts/sh/run_view_expanded_reason.sh` | view-expanded 실험 wrapper |

`eval_late_fusion_reason.py`의 최신 기능:

```text
--save-every N   # N actions마다 partial rows/metrics/predictions 저장
--resume         # 기존 rows.jsonl/json에서 이어서 실행
```

---

## 9. Outputs 정리

현재 최상위 구조:

```text
outputs/
  archive_20260608/
    diagnostics/
    eval_results/
    logs/
    training_checkpoints/
    zero_shot/
  logs/
  qlora_cosmos8b_view_expanded_reason_clean/
  zero_shot_late_fusion_reason_full_valid/
```

보존 정책:

- 과거 실험 결과는 `archive_20260608/` 아래로 이동
- 현재 진행 중인 clean train과 zero-shot full valid만 최상위 유지
- `.gitignore`에서 `data/`, `outputs/`, checkpoint, cache 제외

---

## 10. 다음 할 일

1. zero-shot late fusion full Valid 완료 확인
2. offline fusion rules 비교
   - `main_first`
   - `clip1_first`
   - `majority_vote`
   - `majority_clip1_tiebreak`
   - `conservative_card`
3. view-expanded clean train 3 epoch 완료 확인
4. fine-tuned adapter로 Valid late fusion 평가
5. fine-tuned 결과에서 다음 항목 비교
   - balanced accuracy
   - parse error 감소 여부
   - CJK drift 감소 여부
   - `Offence + No card` 회복 여부
6. best fusion rule로 Test 평가
7. 보고서에는 zero-shot instability와 fine-tuning 개선을 함께 제시
