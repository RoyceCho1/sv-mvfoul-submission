# SoccerNet MVFoul — 모델 개발 과정 기록

> 이 문서는 실험을 진행하면서 어떤 문제를 만났고, 어떻게 해결했는지  
> 단계별로 기록한 개발 히스토리입니다.

---

## Step 0. 문제 정의와 데이터 이해

### 태스크
5초짜리 축구 파울 영상을 보고 두 가지를 분류:
- **Task 1**: action_class (8종류 — Standing tackling, High leg 등)
- **Task 2**: offence_severity (4종류 — No offence, Yellow card 등)

### 원본 데이터 구조 파악

```
annotations.json 예시:
{
  "action_42": {
    "Contact": "With contact",
    "Bodypart": "Under body",
    "Try to play": "Yes",
    "Touch ball": "No",
    "Offence": "Offence",
    "Severity": "3.0"          ← Yellow card
  }
}
```

**문제**: `Offence`와 `Severity`가 분리되어 있고, 불확실한 값들이 많음  
**해결**: `official_target()` 함수로 필터링 및 통합

```python
# 제외 케이스
action_class == "Dont know"      # 심판도 모르는 케이스
Offence == "Between"             # 경계선 케이스
Severity in {"2.0", "4.0"}      # 불확실한 심각도 (1.0=노카드, 3.0=옐로, 5.0=레드)

# 레이블 통합
Offence=Offence + Severity=3.0  →  "Offence + Yellow card"
Offence=No offence              →  "No offence"
```

필터링 후 Train 6,621개 / Valid 321개 / Test 247개

### 클래스 불균형 발견

```
Standing tackling: 45%   ← 지배적
Tackling:         19%
Challenge:        11%
Holding:          15%
High leg:          4%
Pushing:           4%
Elbowing:          4%
Dive:              1.2%  ← 극소수
```

→ 이후 balanced sampling으로 해결 (Step 4)

---

## Step 1. 베이스라인 확인 — Zero-shot

### 시도
학습 없이 Cosmos-Reason2-8B로 바로 추론

**결과**:
```
Valid Task1 (action):   11.8%   (VARS 논문: 47.0%)
Valid Task2 (severity): 25.0%   (VARS 논문: 43.0%)
```

→ 파인튜닝이 반드시 필요함을 확인

---

## Step 2. 모델 선택과 메모리 절감 방법

### 왜 Cosmos-Reason2인가
- NVIDIA가 멀티뷰 비디오 이해를 목적으로 만든 모델
- 비디오 + 텍스트 동시 처리 (Qwen3VLForConditionalGeneration 기반)
- 8B 파라미터: 성능과 VRAM 사이 현실적인 균형점

### 메모리 문제와 해결: QLoRA

8B 모델을 그대로 학습하면 **약 60GB VRAM** 필요 → RTX 5090 (32GB) 1장으로 불가능

**단계별 메모리 절감**:

#### ① 4-bit NF4 양자화 (가장 큰 절감)
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # Normal Float 4: 정보 손실 최소화
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,   # 양자화 상수도 양자화 → 추가 절감
)
```
→ 8B 모델 (16GB) → **약 4.5GB**로 축소

#### ② Vision Encoder 완전 동결
```python
for param in model.visual.parameters():
    param.requires_grad = False
```
→ 비전 인코더는 이미 좋은 표현을 학습 → 동결해도 품질 유지  
→ **gradient 계산 불필요 → activation memory 대폭 절감**

#### ③ LoRA (Low-Rank Adaptation)
```python
LoraConfig(
    r=16,           # rank: 저랭크 행렬 크기
    lora_alpha=32,  # 스케일링 계수
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
)
```

원래 가중치 W를 건드리지 않고, 작은 행렬 A, B만 학습:
```
W' = W + (alpha/r) * B @ A
파라미터 수: 전체 8B → 학습 대상 ~20M (0.25%)
```
→ optimizer state (Adam: 파라미터당 2배) 메모리도 0.25%로 축소

#### ④ Gradient Accumulation
```python
GRAD_ACCUM = 8  # 8 스텝마다 한 번 업데이트
```
→ 실질적 batch_size = 1×8 = 8, 실제 VRAM은 batch=1 수준 유지

#### 최종 VRAM 사용량
```
SV (16프레임):  ~17.3GB reserved
MV (32프레임 × 2뷰): ~23.2GB reserved
```
→ 32GB GPU에서 여유 있게 학습 가능

---

## Step 3. 학습 방식 설계 — Think + Answer

### 처음 방식: 단순 JSON 출력
```
입력: [비디오]
출력: {"action_class": "Standing tackling", "offence_severity": "Offence + Yellow card"}
```

**문제**: 베이스라인은 되지만, 모델이 왜 그 결론을 냈는지 설명 없음.  
Cosmos-Reason2는 원래 reasoning 모델 — 이 능력을 활용하지 않는 낭비

### 발전: Think + Answer (Chain-of-Thought)
```
입력: [비디오] + 프롬프트
출력:
<think>
Analyzing the video clip:
Contact: There is clear physical contact between the players.
Body part: The challenge involves the lower body (legs/feet).
Ball interaction: The player attempts to play the ball but does not make contact.
A lower body challenge from a standing position...
The card decision depends on the force of impact...
</think>
<answer>{"action_class": "Standing tackling", "offence_severity": "Offence + Yellow card"}</answer>
```

**핵심 아이디어**: 먼저 추론하고(`<think>`), 그 결론을 답으로 내기(`<answer>`)  
→ 모델이 "왜 이 판정인지"를 고려하도록 유도

### Reasoning Trace 합성 (`synthesize_think`)

실제 사람이 쓴 reasoning 데이터 없음 → **annotation 속성에서 deterministic하게 생성**

```python
# Contact → 문장 A
"Contact: There is clear physical contact between the players."

# Bodypart + Upper body part → 문장 B
"Body part: The challenge involves the lower body (legs/feet)."

# Try to play + Touch ball → 문장 C
"Ball interaction: The player attempts to play the ball but does not make contact."

# action_class별 고정 reasoning → 문장 E
_SENTENCE_E = {
    "Standing tackling": "A lower body challenge from a standing position...",
    "High leg": "Raising the foot or leg to a height that endangers...",
    ...
}

# severity 판단 기준 → 문장 F (verdict 없이 기준만)
"The card decision depends on the force of impact, degree of risk..."
```

**설계 원칙**: Sentence F는 "어떤 카드인지" 선언하지 않음  
→ 모델이 영상을 보고 직접 판단하도록 유도  
→ annotation의 카드 결정이 영상에서 실제로 보이는 것을 반영한다는 가정 하에

---

## Step 4. 데이터 불균형 문제와 해결

### 문제 발생 (8B MV Reason v1)
```
학습 후 Valid 예측 분포:
  "Standing tackling": 95%   ← 모든 걸 이걸로 예측
  나머지: 5%
```
→ **Mode Collapse**: 가장 많은 클래스만 예측하면 loss가 최소화되는 함정

Standing tackling이 45%이므로, 항상 이걸 예측해도 accuracy는 45%  
→ 모델이 영상을 전혀 보지 않고 "기본값" 출력

### 해결: Balanced Sampling (sqrt-inverse-frequency)

```python
# 단순 inverse-frequency는 너무 극단적 → sqrt로 완화
weights = [1.0 / sqrt(count(action_class)) for each sample]

# 효과:
# Standing tackling: cnt=2974 → w=0.018 (적게 샘플링)
# Dive:              cnt=80   → w=0.112 (많이 샘플링)
# → 에포크당 분포: 45% → ~27%로 완화

sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
```

**왜 sqrt인가**: 완전 inverse-frequency(`1/cnt`)를 쓰면 Dive 같은 극소수 클래스가  
과도하게 반복되어 과적합 → sqrt로 적당한 수준 유지  
`max_class_weight=3.0` 캡을 추가로 두어 double-correction 방지

---

## Step 5. Answer-Only Masking 시도 — 실패 및 폐기

### 시도 배경
"think 블록은 합성된 가짜 reasoning이니, answer만 학습하면 되지 않을까?"

```python
# answer-only masking: <answer>...</answer>만 loss 계산
# <think>...</think>는 -100으로 마스킹
```

### 결과 (Valid)
```
Task1: 5.8%   (zero-shot 11.8%보다 낮음)
Task2: 15.1%
```

### 실패 원인 분석

**핵심 문제**: 모델이 `<think>` 블록을 언제 끝내야 하는지 학습하지 못함

추론 시 동작:
```
<think>
The player approaches... the player approaches... the player approaches...
[무한 반복 — max_new_tokens 512 소진]
→ <answer> 태그에 도달 못 함 (38.9%가 <answer> 없음)
```

Cosmos-Reason2는 pre-training에서 이미 think 블록을 길게 생성하는 경향  
→ answer-only masking으로는 이 성향을 덮어쓰지 못함

**해결**: think + answer 전체를 loss에 포함 (현재 방식)  
→ 모델이 `</think>` 태그를 올바른 위치에 생성하는 법을 학습

---

## Step 6. 비디오 프레임 샘플링 개선

### 처음: 균일 샘플링 (nframes 방식)
```python
{"type": "video", "video": path, "nframes": 16}
# → processor가 균일하게 16프레임 선택
```

**문제**: VARS 스펙상 파울은 t=3.0s (5초 클립의 60% 지점)에 발생  
균일 샘플링 시 충돌 순간이 16프레임 중 하나일 뿐 → 포착 확률 낮음

### 개선: Foul-Anchored Sampling

```python
# T_FOUL=3.0s 를 항상 포함하고, 앞뒤에 비례 배분
def foul_anchored_timestamps(n=16, t_start=0.5, t_end=4.5, t_foul=3.0):
    ratio = (3.0 - 0.5) / (4.5 - 0.5) = 0.625
    n_before = round(0.625 * 15) = 10   # t_foul 이전 프레임
    n_after  = 5                         # t_foul 이후 프레임
    # → t=3.0s가 항상 10번째 프레임에 포함
```

torchcodec으로 정확한 타임스탬프에서 프레임 추출 후 PIL 리스트로 반환

### 추가: Frame Hint 프롬프트

```
"Note: Frame 10 of 16 in the clip captures the actual foul contact moment —
pay close attention to it when judging the action type and severity."
```

→ 모델에게 어떤 프레임이 중요한지 명시적으로 알려줌  
→ 16프레임 중 어디를 집중적으로 봐야 하는지 가이드

---

## Step 7. Repetition Penalty 추가

### 문제
평가 스크립트에서 한 샘플당 111초 소요 (정상: 5~10초)

로그 확인:
```
<think>
The player approaches the opponent... the player approaches the opponent...
the player approaches the opponent... [수백 토큰 반복]
```

### 원인
`model.generate()`에 반복 방지 장치 없음 → 같은 문구 무한 생성

### 해결
```python
model.generate(
    **inputs,
    max_new_tokens=512,
    repetition_penalty=1.5,   # 추가: 이미 생성한 토큰의 logit에 패널티
)
```

`repetition_penalty > 1.0`: 이미 생성한 토큰 재사용 시 확률 낮춤  
→ 한 샘플 5~10초로 정상화

---

## Step 8. Pixel Budget과 멀티뷰 학습

### 멀티뷰의 의미
같은 파울을 **여러 카메라 각도**에서 동시에 보여주면 더 정확한 판단 가능  
→ clip_0 (정면) + clip_1 (측면) 동시 입력

### Pixel Budget 개념

Qwen3VL은 `max_pixels` 파라미터로 프레임당 픽셀 수를 제한:

```
max_pixels = 10,000,000  (10M)

비디오 토큰 수는 프레임 수가 아닌 총 픽셀 예산에 비례
→ F16/P10M ≈ F32/P10M (sequence length 동일)
→ 프레임을 32장으로 늘려도 VRAM 추가 없음
   (각 프레임이 더 작게 smart_resize됨)
```

이 덕분에 **프레임을 늘려도 temporal coverage만 증가하고 VRAM은 일정**

### 멀티뷰 입력 구조

```
[Clip 1 — Main Camera]
[비디오 16프레임]
[Clip 2 — Side Camera]
[비디오 16프레임]
Note: Frame 10 of 16 in each clip captures the foul contact moment...
```

### 2B MV → 실패 원인

2B 모델로 멀티뷰 시도 시:
```
- CJK drift: 중국어로 전환되어 생성
- Think 붕괴: 78.8% 샘플에서 동일한 boilerplate 반복
- JSON 오류: 95.3%가 파싱 불가
```

→ 2B 용량으로는 멀티뷰 reasoning + 구조화된 JSON 동시 처리 불가  
→ 8B로 전환

### 8B MV v1 → Mode Collapse
→ Step 4에서 해결 (balanced sampling)

---

## Step 9. LR과 학습 안정성

### 문제 (8B MV v1)
```
Epoch 1 → Epoch 2: Valid accuracy 상승
Epoch 2 → Epoch 3: Valid accuracy 하락
```

→ LR=2e-4가 너무 커서 epoch 3에서 발산

### 해결
```
LR: 2e-4 → 1e-4   (절반으로)
Epochs: 3 → 5     (낮은 LR이니 더 많은 epoch 필요)
```

PagedAdamW8bit 옵티마이저 사용 (bitsandbytes):  
→ optimizer state도 8-bit 양자화 → VRAM 추가 절감

---

## Step 10. 실험 모니터링 개선

### 문제
eval 실행 시 수 시간 후에야 결과 확인 가능 — 중간 상황을 알 수 없음

### 해결: JSONL 스트리밍

```python
jsonl_file = (out_dir / f"{prefix}_rows.jsonl").open("w")

for sample in samples:
    result = run_one(...)
    jsonl_file.write(json.dumps(result) + "\n")
    jsonl_file.flush()   # 즉시 디스크 기록
```

→ 별도 터미널에서 실시간 모니터링:
```bash
tail -f outputs/reason_eval_2/*_rows.jsonl
```

각 샘플이 완료될 때마다 즉시 예측 결과와 정답 비교 가능

---

## Step 11. Skeleton 데이터 탐색 (별도 실험)

### 시도 배경
YOLO11로 추출한 skeleton (관절 좌표) 데이터 추가 활용 가능성 검토

### 시도 1: 별도 Skeleton 분류기
- 관절 좌표 (12프레임 × 2인물 × 17관절 × 3) → MLP → 파울 분류
- 결과: Valid action_acc = **8.4%** (랜덤 12.5%보다 낮음)
- 폐기 이유: skeleton만으로 파울 종류 구분은 인간도 어려운 과제

### 시도 2: Late Fusion
- VLM 예측 + skeleton 분류기 → 가중합
- 폐기 이유: 8.4% 분류기를 섞으면 VLM 성능을 오히려 깎음

### 시도 3: Skeleton Overlay (VLM 입력)
- skeleton stick figure를 비디오 프레임에 렌더링 → VLM 입력
- 폐기 이유: 축구 경기에서 10명 검출 시 파울 당사자 2명 식별 불가
  → 엉뚱한 선수 overlay → 노이즈 학습 리스크

### 결론
Skeleton 데이터는 현재 방식으로는 유의미한 기여 어려움  
평가 시 skeleton text hint A/B 테스트는 여전히 가능 (재학습 불필요)

---

---

## Step 12. 태스크 단순화: action_class 제거, offence_severity만 예측

### 변경 배경
초기 목표는 VARS와 동일하게 두 태스크를 동시에 푸는 것이었다.

- Task 1: `action_class` 8분류
- Task 2: `offence_severity` 4분류

하지만 실험이 진행되면서 다음 문제가 반복적으로 나타났다.

1. `action_class`가 매우 불균형함
2. 모델이 `Standing tackling` 같은 다수 클래스로 collapse함
3. severity 판단보다 action type 판단이 먼저 흔들리면서 전체 JSON 출력이 불안정해짐
4. 최종 프로젝트 메시지는 "foul severity 판정" 쪽이 더 명확함

따라서 이후 실험은 **offence_severity 4분류만 예측**하도록 단순화했다.

```json
{"offence_severity": "Offence + Yellow card"}
```

### 효과
- 출력 스키마가 단순해짐
- parser 실패 가능성이 줄어듦
- VAR/심판 판정 관점에서 설명이 명확해짐
- 다만 VARS의 Task1과 직접 비교는 하지 않고, Task2 중심 비교로 전환해야 함

---

## Step 13. Reasoning Prompt 재설계

### 기존 문제
초기 reasoning prompt는 자유 서술형에 가까웠다.

```text
Analyze the video and explain the foul type and severity...
```

Zero-shot과 일부 fine-tuned 모델에서 다음 문제가 발생했다.

- `<answer>` 태그 미생성
- `<think>` 무한 반복
- 중국어/CJK drift
- 긴 일반 축구 해설문 생성
- `Offence + Red card` 과대예측

### 개선된 slot-based reasoning
모델이 길게 자유 서술하지 않고, 고정된 판단 항목을 채우도록 변경했다.

```text
contact: yes/no/unclear
ball_play: yes/no/unclear
force: low/medium/high/unclear
risk: low/medium/high/unclear
offence: yes/no
card_threshold: no_card/yellow/red/none
decision: one short reason
```

최종 출력은 여전히 `<answer>` 안의 JSON만 평가한다.

```text
<think>
contact: yes
ball_play: unclear
force: medium
risk: medium
offence: yes
card_threshold: yellow
decision: contact with risk above normal football contact.
</think>
<answer>{"offence_severity": "Offence + Yellow card"}</answer>
```

### 중요한 해석
`think`는 사람이 직접 라벨링한 정답이 아니다. annotation 속성에서 deterministic하게 합성한 supervision이다.
따라서 think 자체의 정확도를 평가하는 것이 아니라, 모델이 **짧고 안정적인 중간 판단 형식**을 학습하도록 하는 보조 신호로 사용한다.

---

## Step 14. Early Fusion에서 Late Fusion으로 전환

### Early Fusion 문제
멀티뷰를 한 번에 모델 입력으로 넣는 early fusion을 시도했다.

```text
[clip_0 video] + [clip_1 video] + prompt -> one answer
```

하지만 zero-shot과 fine-tuning 모두에서 다음 문제가 있었다.

- 입력이 길어지면 reasoning이 더 불안정해짐
- view 간 정보가 섞이며 출력 형식이 깨짐
- 2B 모델은 용량 부족으로 JSON schema 오류가 많음
- 8B MV도 mode collapse가 반복됨

### Late Fusion 설계
각 view를 single-view 모델로 독립 추론한 뒤, action 단위에서 예측을 합친다.

```text
clip_0 -> pred_0
clip_1 -> pred_1
clip_2 -> pred_2
fusion(pred_0, pred_1, pred_2) -> final action prediction
```

장점:
- 학습은 single-view로 단순하게 유지
- inference 때 여러 view를 모두 활용 가능
- view별 성능과 오류를 따로 분석 가능
- fusion rule을 offline으로 바꿔가며 비교 가능

현재 구현된 fusion rule:

| Rule | 의미 |
|---|---|
| `main_first` | clip_0 예측 우선, 없으면 다음 valid view |
| `clip1_first` | clip_1 예측 우선, 없으면 clip_0 |
| `majority_vote` | 전체 view 다수결, tie는 clip_0 |
| `majority_clip1_tiebreak` | 전체 view 다수결, tie는 clip_1 우선 |
| `conservative_card` | Red 단독 예측을 보수적으로 낮춤 |

---

## Step 15. View-expanded SV Training

### 배경
기존 SV 학습은 `clip_0`만 사용했다.
Late fusion을 하려면 clip_1, clip_2 같은 replay view도 모델이 제대로 처리해야 한다.

### 변경
학습 sample 단위를 action이 아니라 **view**로 확장했다.

```text
action_0/clip_0.mp4 -> same offence_severity label
action_0/clip_1.mp4 -> same offence_severity label
action_0/clip_2.mp4 -> same offence_severity label
```

현재 clean run 설정:

```text
MODEL_ID       = nvidia/Cosmos-Reason2-8B
OUTPUT_DIR     = outputs/qlora_cosmos8b_view_expanded_reason_clean
NUM_EPOCHS     = 3
NUM_FRAMES     = 32
LR             = 5e-5
LORA_R         = 128
LORA_ALPHA     = 256
GRAD_ACCUM     = 8
MAX_TRAIN_VIEWS = 0  # all views
MAX_EVAL_VIEWS  = 0  # all views
FUSION_RULE    = main_first
--balanced-sampling
```

학습 데이터 구성:

```text
view_samples = 5,277
actions      = 2,319
clip_0       = 2,319
clip_1       = 2,319
clip_2       =   538
clip_3       =   101
```

이전 `outputs/qlora_cosmos8b_view_expanded_reason` run은 epoch 1 validation 후 JSON serialization 문제로 중단되었다.
해당 문제는 `make_jsonable()` 저장 유틸로 해결했고, clean run은 별도 디렉토리에 다시 시작했다.

---

## Step 16. Resumable Zero-shot Late Fusion Evaluation

### 문제
전체 Valid zero-shot late fusion은 action마다 여러 view를 생성해야 해서 오래 걸린다.
중간에 GPU를 끊으면 결과가 모두 사라지는 문제가 있었다.

### 해결
`eval_late_fusion_reason.py`에 다음 옵션을 추가했다.

```text
--save-every N   # N actions마다 rows/metrics/predictions 저장
--resume         # 기존 rows.jsonl 또는 rows.json에서 완료 action을 읽고 건너뜀
```

저장 파일:

```text
outputs/zero_shot_late_fusion_reason_full_valid/
  valid_base_views_rows.jsonl        # action 1개 완료마다 append
  valid_base_views_rows.json         # save-every마다 갱신
  valid_base_views_metrics.json      # save-every마다 갱신
  valid_base_views_predictions.json  # save-every마다 갱신
```

현재 partial 결과는 zero-shot baseline 수집 중이다.
90 samples 기준 중간 결과:

```text
accuracy_offence_severity = 18.89%
balanced_accuracy_seen    = 14.89%
view_parse_errors         = 18 / 207 views
```

관찰된 zero-shot 문제:

- `<answer>` 태그를 거의 지키지 않음
- CJK drift 발생
- 출력이 1200자 이상 길어지는 경우 많음
- `Offence + No card`를 거의 예측하지 못함
- 접촉이 보이면 Yellow/Red로 과대 판정하는 경향

이 결과는 fine-tuning 필요성을 보여주는 baseline으로 사용한다.

---

## Step 17. Outputs 정리와 Git Ignore

`outputs/`가 너무 커져서 실험 보존용 archive로 정리했다.

```text
outputs/
  archive_20260608/
    training_checkpoints/
    eval_results/
    zero_shot/
    diagnostics/
    logs/
  logs/
  qlora_cosmos8b_view_expanded_reason_clean/
  zero_shot_late_fusion_reason_full_valid/
```

`.gitignore`도 업데이트했다.

주요 ignore 대상:

```text
data/
outputs/
*.safetensors
*.pth
*.pth.tar
__pycache__/
VARS model/models/
VARS model/pretrained/
```

결과물과 가중치는 로컬에 보존하되, git에는 코드와 실험 문서만 올리는 구조로 정리했다.

---

## 현재 상태 (2026-06-08)

| 항목 | 상태 |
|---|---|
| 태스크 | `offence_severity` 4분류 중심으로 전환 |
| 출력 형식 | `<think>` slot reasoning + `<answer>{"offence_severity": ...}</answer>` |
| 멀티뷰 전략 | early fusion 대신 view-expanded SV 학습 + late fusion |
| zero-shot full Valid | GPU 0에서 진행 중, JSONL 중간 저장/resume 가능 |
| view-expanded clean training | GPU 1에서 `outputs/qlora_cosmos8b_view_expanded_reason_clean`로 진행 중 |
| 이전 깨진 run | `archive_20260608/training_checkpoints/qlora_cosmos8b_view_expanded_reason`에 보존 |
| outputs 정리 | 완료 |
| gitignore 정리 | 완료 |

**현재 가장 중요한 다음 확인사항**:

1. zero-shot late fusion full Valid 완료 후 fusion rule별 offline 평가
2. view-expanded clean training 3 epoch 완료 여부 확인
3. fine-tuned adapter로 Valid late fusion 재평가
4. zero-shot 대비 parse error / CJK drift / balanced accuracy 개선 여부 확인
5. best fusion rule을 선택해 Test 평가

**현재 판단**:

- zero-shot은 baseline으로 충분히 의미가 있지만, 출력 형식과 severity calibration이 불안정하다.
- fine-tuning은 단순 accuracy뿐 아니라 JSON 형식 준수, CJK drift 감소, `Offence + No card` 회복 여부를 같이 평가해야 한다.
- late fusion은 early fusion보다 분석과 ablation이 쉬워 현재 프로젝트 방향에 더 적합하다.

