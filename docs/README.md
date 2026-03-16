# Floorplan-LLM

좌표 기반 평면도 자동 생성 LLM 구축 프로젝트.

사용자가 "방 3개짜리 집, 거실 오른쪽에 침실"과 같은 자연어 조건을 입력하면, 모델이 각 방의 정확한 꼭지점 좌표를 포함한 평면도를 토큰 시퀀스로 생성한다.

---

## 목차

- [프로젝트 구조](#프로젝트-구조)
- [핵심 개념](#핵심-개념)
- [설치](#설치)
- [전체 파이프라인 워크플로우](#전체-파이프라인-워크플로우)
- [스크립트 사용법](#스크립트-사용법)
- [설정 파일](#설정-파일)
- [데이터 저장 형식](#데이터-저장-형식)
- [구현 현황](#구현-현황)

---

## 프로젝트 구조

```
floorplan-llm/
├── config/                         # Hydra 설정 파일
│   ├── build_dataset/
│   │   ├── rplan2json/             # PNG → JSONL 추출 설정
│   │   │   ├── pipeline.yaml
│   │   │   └── room_type_merge.json
│   │   ├── json2arrow/             # JSONL → Arrow 변환 설정
│   │   │   └── pipeline.yaml
│   │   └── visualize_json/         # 시각화 설정
│   │       └── color_map.yaml
│   ├── build_model/
│   │   └── tokenization/           # 어휘(Vocabulary) 빌드 설정
│   │       └── pipeline.yaml
│   └── training/
│       ├── augmentation/           # 데이터 증강 프리셋 (Hydra config group)
│       │   ├── pre_stage.yaml      # Pre-Stage용 증강 설정 → cfg.augmentation으로 병합
│       │   └── validate_augmentation/  # validate_augmentation.py 실행 설정
│       │       ├── pipeline.yaml       # 스크립트 전반 설정 (model, data, validate)
│       │       └── augmentation.yaml   # 검증에 사용할 증강 파라미터
│       └── pre_stage/              # Pre-Stage 훈련 설정
│           └── pipeline.yaml       # defaults로 training/augmentation: pre_stage 합성
│
├── src/                            # 핵심 모듈 (uv 패키지로 설치)
│   ├── build_dataset/
│   │   ├── rplan2json/             # RPLAN PNG 파싱 엔진
│   │   │   ├── channel_parser.py   # BGRA 채널 분리
│   │   │   ├── room_extractor.py   # 방 분리 + 직교 폴리곤 추출
│   │   │   ├── door_extractor.py   # 현관문/인테리어 문 추출
│   │   │   ├── edge_builder.py     # 방 간 연결관계(Edge) 구성
│   │   │   ├── spatial_calculator.py # 8방위 공간관계 계산
│   │   │   └── serializer.py       # JSONL 직렬화
│   │   ├── json2arrow/             # JSONL → Arrow 변환기
│   │   │   ├── schema.py           # Arrow 스키마 정의
│   │   │   ├── converter.py        # 변환 로직
│   │   │   └── validator.py        # 변환 결과 검증
│   │   └── visualize_json/         # JSONL 시각화 렌더러
│   │       ├── visualizer.py
│   │       ├── renderer.py
│   │       └── loader.py
│   ├── build_model/
│   │   └── tokenization/           # 커스텀 어휘 빌더
│   │       ├── token_definitions.py # 토큰 목록 정의
│   │       └── vocab_builder.py    # HuggingFace 토크나이저 확장
│   └── training/
│       ├── augmentation/           # 데이터 증강 파이프라인
│       │   ├── pipeline.py         # 증강 파이프라인 오케스트레이터
│       │   ├── strategies.py       # 15+ 증강 전략 구현
│       │   ├── tokenizer.py        # 조건/정답 토큰 시퀀스 생성
│       │   └── decoder.py          # 토큰 → 텍스트 디코딩
│       └── pre_stage/              # Pre-Stage 훈련 모듈
│           ├── model_loader.py     # 4bit 양자화 로드 + PartialEmbedding/PartialLMHead
│           ├── dataset.py          # Arrow 로드 + 증강 + Chat Template
│           ├── collator.py         # Dynamic padding + label 마스킹
│           └── trainer.py          # TrainingArguments + Trainer 빌드
│
├── scripts/                        # CLI 실행 진입점
│   ├── build_dataset/
│   │   ├── rplan2json/
│   │   │   ├── run_extraction.py   # PNG 배치 처리
│   │   │   └── validate_jsonl.py   # JSONL 검증
│   │   ├── json2arrow/
│   │   │   └── run_conversion.py   # Arrow 변환 실행
│   │   └── visualize_json/
│   │       └── run_visualization.py # 평면도 시각화
│   ├── build_model/
│   │   └── tokenization/
│   │       └── build_vocab.py      # 어휘 빌드 실행
│   └── training/
│       ├── augmentation/
│       │   └── validate_augmentation.py # 증강 결과 검증
│       └── run_pre_stage.py        # Pre-Stage 훈련 실행
│
├── data/                           # 데이터 저장소 (Git 추적 제외)
│   ├── dataset/
│   │   ├── raw_dataset/rplan/dataset/          # 원본 PNG 입력
│   │   └── processed_dataset/rplan/
│   │       ├── jsonl/                          # Step 1 출력
│   │       ├── arrow/                          # Step 3 출력 (train/val/test)
│   │       └── validation_result/
│   └── models/
│       └── {model.name}/                       # 모델명별 독립 저장 (예: Qwen2.5-Coder-7B)
│           ├── tokenization/                   # 확장된 토크나이저 + vocab
│           └── checkpoints/
│               └── pre_stage/                  # Pre-Stage 체크포인트 + 최종 모델
│                   ├── checkpoint-*/           # 에폭별 자동 저장 체크포인트
│                   └── final/                  # 최종 병합 모델 (다음 Stage 입력)
│
├── outputs/                        # Hydra 실행 로그 + 설정 스냅샷
│   └── training/
│       └── pre_stage/              # Pre-Stage 실행 로그 (scripts/training/pre_stage 계층과 동일)
│           └── YYYY-MM-DD/HH-MM-SS/
└── docs/                           # 문서
    ├── README.md                   # 이 파일
    └── Docs.md                     # 상세 설계 문서
```

---

## 핵심 개념

### 평면도 = 토큰 시퀀스

평면도를 이미지 대신 커스텀 토큰 시퀀스로 표현한다.

```
<ROOM> <RID:1> <TYPE:livingroom> <X:100> <Y:200> <X:200> <Y:200> <X:200> <Y:300> <X:100> <Y:300> <END_ROOM>
```

좌표는 `<ROOM>` ~ `<END_ROOM>` 사이에 직접 나열되며, 별도의 `<COORDS>` 래퍼 없이 표현된다. 이 표현 방식 덕분에 평면도 생성이 LLM의 자연스러운 토큰 생성 문제가 된다.

### GPT 스타일 조건부 생성

모델은 조건(입력)을 받아 전체 평면도(출력)를 처음부터 끝까지 자동회귀적으로 생성한다.

- **입력 조건**: 방 종류/개수, 일부 방의 좌표, 방 간 연결관계, 방 간 위치관계
- **출력**: 전체 방의 종류 + 꼭지점 좌표 + 문 정보 (FRONT_DOOR + 인테리어 DOORs)

---

## 설치

```bash
# uv 기반 의존성 설치
uv sync
```

**주요 의존성:**

| 라이브러리 | 용도 |
|-----------|------|
| `torch >= 2.6.0` | 딥러닝 프레임워크 |
| `transformers >= 4.51.0` | HuggingFace 모델 + 토크나이저 |
| `datasets >= 4.6.1` | Arrow 데이터셋 I/O |
| `peft >= 0.15.0` | kbit 훈련 준비 (`prepare_model_for_kbit_training`) |
| `bitsandbytes >= 0.45.0` | 4bit 양자화 (`BitsAndBytesConfig`) |
| `accelerate >= 1.6.0` | 분산 학습 + 혼합 정밀도 |
| `wandb >= 0.19.0` | 실험 추적 |
| `opencv-python-headless >= 4.13` | 이미지 처리 (PNG 파싱) |
| `hydra-core >= 1.3.2` | 설정 관리 |
| `omegaconf >= 2.3.0` | YAML 파싱 + 보간 |
| `orjson >= 3.11.7` | 고속 JSON 직렬화 |

---

## 전체 파이프라인 워크플로우

```
PNG (RPLAN 데이터셋)
        │
        ▼
[Step 1] rplan2json          → JSONL 원본 데이터
        │
        ▼
[Step 2] build_vocab         → vocab_extension.json + 확장된 토크나이저
        │
        ▼
[Step 3] json2arrow          → Arrow DatasetDict (train / val / test)
        │
        ▼
[Step 4] augmentation        → (condition_tokens, output_tokens) 쌍
        │
        ▼
[Pre-Stage] 새 토큰 Embedding 워밍업 → 커스텀 토큰 embedding 안착
        │
        ▼
[Step 5] LLM 학습 (구현 예정) → 평면도 생성 모델
        │
        ▼
[Step 6] 추론 + 시각화 (예정) → 평면도 이미지
```

> **현재 구현 완료 범위:** Step 1 ~ Step 4, Pre-Stage

---

## 스크립트 사용법

모든 스크립트는 `uv run python scripts/...` 형태로 실행하며, Hydra CLI 오버라이드를 통해 설정을 변경할 수 있다.

### Step 1: PNG → JSONL 추출

RPLAN PNG 이미지에서 방 정보, 문 정보, 공간관계를 추출하여 JSONL로 저장한다.

```bash
# 전체 배치 처리 (기본값)
uv run python scripts/build_dataset/rplan2json/run_extraction.py

# 워커 수 조정
uv run python scripts/build_dataset/rplan2json/run_extraction.py \
    batch.num_workers=16

# 단일 파일 디버깅
uv run python scripts/build_dataset/rplan2json/run_extraction.py \
    mode=single target_file=0.png
```

**입력:** `data/dataset/raw_dataset/rplan/dataset/*.png`
**출력:** `data/dataset/processed_dataset/rplan/jsonl/floorplans_*.jsonl`

---

### Step 1 검증: JSONL 유효성 검사

```bash
uv run python scripts/build_dataset/rplan2json/validate_jsonl.py \
    data/dataset/processed_dataset/rplan/jsonl/floorplans_0000.jsonl

# 리포트 파일 저장
uv run python scripts/build_dataset/rplan2json/validate_jsonl.py \
    data/dataset/processed_dataset/rplan/jsonl/floorplans_0000.jsonl \
    -o report.txt
```

---

### Step 1 시각화: JSONL 시각화

```bash
uv run python scripts/build_dataset/visualize_json/run_visualization.py
```

---

### Step 2: Vocabulary 빌드

Pretrained LLM의 토크나이저에 평면도 전용 커스텀 토큰을 추가하고 저장한다.

```bash
# 기본 (config 기본값: Qwen/Qwen2.5-Coder-7B)
uv run python scripts/build_model/tokenization/build_vocab.py

# 다른 베이스 모델 사용 (user와 name 모두 지정)
uv run python scripts/build_model/tokenization/build_vocab.py \
    model.user=meta-llama model.name=Llama-3.1-8B
```

**출력:** `data/models/{model.name}/tokenization/`
- `vocab_extension.json` — 커스텀 토큰 목록 및 ID 매핑
- `tokenizer.json`, `tokenizer_config.json` — 확장된 토크나이저

---

### Step 3: JSONL → Arrow 변환

JSONL 원본을 HuggingFace `datasets` 라이브러리의 Arrow 포맷으로 변환하고 train/val/test로 분리한다.

```bash
# 기본 변환 (검증 포함)
uv run python scripts/build_dataset/json2arrow/run_conversion.py

# Split 비율 조정
uv run python scripts/build_dataset/json2arrow/run_conversion.py \
    split.val_ratio=0.05 split.test_ratio=0.10

# Split 없이 전체를 하나로
uv run python scripts/build_dataset/json2arrow/run_conversion.py \
    split.enabled=false
```

**출력:** `data/dataset/processed_dataset/rplan/arrow/{train,validation,test}/`

---

### Step 4: 증강 파이프라인 검증

증강 파이프라인이 올바르게 동작하는지 샘플 데이터로 확인한다.

**설정 파일:**
- `config/training/augmentation/validate_augmentation/pipeline.yaml` — 스크립트 전반 설정 (model, data 경로, 샘플 수 등)
- `config/training/augmentation/validate_augmentation/augmentation.yaml` — 검증에 사용할 증강 파라미터 (`data.pipeline_config`로 경로 지정)

```bash
# 기본 검증 (20개 샘플)
uv run python scripts/training/augmentation/validate_augmentation.py

# 샘플 수 변경
uv run python scripts/training/augmentation/validate_augmentation.py \
    validate.num_samples=50

# validation split 사용
uv run python scripts/training/augmentation/validate_augmentation.py \
    data.split=validation
```

---

### Pre-Stage: 새 토큰 Embedding 워밍업

커스텀 토큰의 embedding과 lm_head 행만 훈련하여 기존 Pretrained 파라미터 공간에 안착시킨다.

```bash
# 기본 실행
uv run python scripts/training/run_pre_stage.py

# 디버그 (10 step만 실행, W&B 비활성화)
uv run python scripts/training/run_pre_stage.py \
    training.max_steps=10 training.report_to=none

# 하이퍼파라미터 오버라이드
uv run python scripts/training/run_pre_stage.py \
    training.learning_rate=1e-3 training.num_train_epochs=3

# 다른 모델 사용
uv run python scripts/training/run_pre_stage.py \
    model.user=meta-llama model.name=Llama-3.1-8B

# 계속 훈련: 최신 체크포인트 자동 탐색 후 재개
uv run python scripts/training/run_pre_stage.py \
    resume.enabled=true

# 계속 훈련: 특정 체크포인트 지정
uv run python scripts/training/run_pre_stage.py \
    resume.enabled=true \
    resume.checkpoint_path=data/models/Qwen2.5-Coder-7B/checkpoints/pre_stage/checkpoint-500
```

**출력:**
- `data/models/{model.name}/checkpoints/pre_stage/checkpoint-*/` — 에폭별 자동 저장 (최대 `save_total_limit`개 보존)
- `data/models/{model.name}/checkpoints/pre_stage/final/` — 최종 병합 모델 (표준 HuggingFace 형식)

---

## 설정 파일

모든 설정은 `config/` 디렉토리의 YAML 파일로 관리된다. 실행 시 Hydra가 설정을 `outputs/` 디렉토리에 자동으로 스냅샷 저장하여 재현성을 보장한다.

### 공통: 모델 설정 구조

모든 LLM 모델 정보를 사용하는 config는 아래 구조를 따른다. `hub_id`, `tokenizer_dir`, `vocab_extension`은 OmegaConf 보간으로 자동 파생되므로 `user`와 `name` 2개만 수정하면 된다.

```yaml
model:
  user: "Qwen"                   # HuggingFace Hub 사용자(조직)명
  name: "Qwen2.5-Coder-7B"      # 모델명 (로컬 저장 디렉토리명으로도 사용)
  hub_id: "${model.user}/${model.name}"               # 자동 파생
  tokenizer_dir: "data/models/${model.name}/tokenization"        # 자동 파생
  vocab_extension: "${model.tokenizer_dir}/vocab_extension.json" # 자동 파생
```

### `config/build_dataset/rplan2json/pipeline.yaml`

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `processing.min_room_area` | `30` | 최소 방 면적(px). 이하 제외 |
| `processing.min_door_pixels` | `5` | 최소 문 픽셀 수 |
| `processing.door_dilation_kernel` | `5` | 문-방 경계 매칭 팽창 커널 크기 |
| `batch.num_workers` | `8` | 병렬 처리 워커 수 |
| `batch.output_shard_size` | `10000` | JSONL 파일당 레코드 수 |
| `mode` | `batch` | `batch` \| `single` |

### `config/build_dataset/json2arrow/pipeline.yaml`

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `split.val_ratio` | `0.001` | Validation 비율 (0.1%) |
| `split.test_ratio` | `0.005` | Test 비율 (0.5%) |
| `split.seed` | `42` | 분리 랜덤 시드 |
| `validation.enabled` | `true` | 변환 후 검증 여부 |
| `validation.num_samples` | `10` | 검증 샘플 수 |

### `config/training/augmentation/pre_stage.yaml`

훈련 단계별로 독립된 증강 프리셋을 관리한다. Hydra **config group** 방식으로 `config/training/pre_stage/pipeline.yaml`의 `defaults`에서 합성되어 `cfg.augmentation`으로 접근된다.
추후 SFT, DPO, GRPO 등 각 단계별로 `sft.yaml`, `dpo.yaml` 등을 추가하여 독립 관리한다.

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `shuffle.rid` | `true` | 방 ID 재배정 증강 활성화 |
| `shuffle.vertex_order` | `true` | 꼭지점 순서 셔플 활성화 |
| `transform.flip` | `true` | 뒤집기 증강 활성화 |
| `transform.zoom_min/max` | `0.7 / 1.3` | 줌 배율 범위 |
| `transform.scale_aspect_min/max` | `0.7 / 1.3` | 종횡비 변형 범위 |
| `noise.p_noise` | `0.50` | 노이즈 적용 확률 |
| `noise.noise_sigma` | `3.0` | 가우시안 노이즈 표준편차 (px) |
| `drop.p_drop_block` | `0.5` | 방 블록 삭제 확률 |
| `drop.p_drop_coords` | `0.20` | 방 좌표 삭제 확률 |
| `drop.p_drop_spatial` | `0.80` | Spatial 관계 삭제 확률 |
| `room_summary.p_drop_total` | `0.50` | `<TOTAL>` + 숫자 쌍 삭제 확률 |
| `room_summary.p_drop_type` | `0.60` | 개별 타입별 `<COUNT>` + 숫자 쌍 삭제 확률 |

### `config/training/pre_stage/pipeline.yaml`

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `model.user` | `"Qwen"` | HuggingFace Hub 사용자(조직)명 |
| `model.name` | `"Qwen2.5-Coder-7B"` | 모델명 + 로컬 저장 디렉토리명 |
| `quantization.load_in_4bit` | `true` | 4bit 양자화 활성화 |
| `quantization.bnb_4bit_quant_type` | `"nf4"` | 양자화 방식 |
| `quantization.bnb_4bit_use_double_quant` | `true` | Double quantization |
| `data.max_length` | `4096` | 최대 시퀀스 길이 |
| `training.output_dir` | `data/models/${model.name}/checkpoints/pre_stage` | 체크포인트 저장 경로 |
| `training.learning_rate` | `5e-4` | 학습률 (공격적 설정) |
| `training.num_train_epochs` | `2` | 훈련 에폭 수 |
| `training.per_device_train_batch_size` | `2` | GPU당 배치 크기 |
| `training.gradient_accumulation_steps` | `1` | 그래디언트 누적 steps |
| `training.bf16` | `true` | 혼합 정밀도 (AMP) |
| `training.save_total_limit` | `3` | 보존할 체크포인트 최대 수 (오래된 순 삭제) |
| `training.load_best_model_at_end` | `true` | 훈련 종료 시 eval_loss 최고 체크포인트 복원 |
| `training.max_steps` | `0` | 디버그용 step 제한 (0=비활성) |
| `resume.enabled` | `false` | 계속 훈련 활성화 여부 |
| `resume.checkpoint_path` | `null` | 특정 체크포인트 경로 지정 (null이면 자동 탐색) |
| `resume.auto_find_latest` | `true` | output_dir에서 최신 체크포인트 자동 탐색 |

---

## 데이터 저장 형식

### JSONL 레코드 예시

```json
{
  "plan_id": "fp_00123",
  "rooms": [
    {"rid": 0, "type": "outline",    "coords": [80,30, 80,220, 210,220, 210,30]},
    {"rid": 1, "type": "livingroom", "coords": [100,200, 100,300, 200,300, 200,200]},
    {"rid": 2, "type": "bedroom",    "coords": [200,200, 200,300, 300,300, 300,200]}
  ],
  "edges": [
    {"pair": [1,2], "doors": [{"x": 200, "y": 250, "w": 2, "h": 10}]},
    {"pair": [0,1], "doors": []}
  ],
  "front_door": {"x": 128, "y": 32, "w": 8, "h": 2},
  "spatial": [[1, 2, "right"]]
}
```

### 토큰 시퀀스 예시 (증강 후)

`<INPUT>` ~ `<END_INPUT>`, `<OUTPUT>` ~ `<END_OUTPUT>` 사이의 토큰은 줄바꿈·공백 없이 이어붙인다 (아래는 가독성을 위해 줄바꿈 표기).

**입력 (조건):**
```
<INPUT>
  <ROOM_SUMMARY> <TOTAL> 2 <TYPE:bedroom> <COUNT> 1 <TYPE:livingroom> <COUNT> 1 <END_ROOM_SUMMARY>
  <ROOM> <RID:1> <TYPE:livingroom> <X:100> <Y:200> <X:100> <Y:300> <X:200> <Y:300> <X:200> <Y:200> <END_ROOM>
  <ROOM> <RID:2> <TYPE:bedroom> <END_ROOM>
  <EDGE> <RID:1> <RID:2> <DOOR> <SEP_DOOR> <END_DOOR> <END_EDGE>
<END_INPUT>
```

**출력 (정답 — 항상 완전한 정보):**
```
<OUTPUT>
  <FRONT_DOOR> <X:128> <Y:32> <SEP_DOOR> <X:8> <Y:2> <END_DOOR>
  <ROOM> <TYPE:livingroom> <X:100> <Y:200> <X:100> <Y:300> <X:200> <Y:300> <X:200> <Y:200> <END_ROOM>
  <ROOM> <TYPE:bedroom> <X:200> <Y:200> <X:200> <Y:300> <X:300> <Y:300> <X:300> <Y:200> <END_ROOM>
  <DOOR> <X:200> <Y:250> <SEP_DOOR> <X:2> <Y:10> <END_DOOR>
<END_OUTPUT>
```

**Chat Template (Qwen2.5 형식 기준):**
```
<|im_start|>system
You are a floor plan generator. Given room conditions, generate complete floorplan coordinates.<|im_end|>
<|im_start|>user
<INPUT>...<END_INPUT><|im_end|>
<|im_start|>assistant
<OUTPUT>...<END_OUTPUT><|im_end|>
```

---

## 구현 현황

| 단계 | 내용 | 상태 |
|------|------|------|
| Step 1 | 평면도 PNG → JSONL 추출 | ✅ 완료 |
| Step 2 | 커스텀 Vocabulary 빌드 | ✅ 완료 |
| Step 3 | JSONL → Arrow 변환 | ✅ 완료 |
| Step 4 | 데이터 증강 + 토크나이징 | ✅ 완료 |
| Pre-Stage | 새 토큰 Embedding 워밍업 훈련 | ✅ 완료 |
| Step 5 | SFT → DPO → GRPO Fine-tuning | 🔜 구현 예정 |
| Step 6 | 추론 + 시각화 | 🔜 구현 예정 |

자세한 설계 내용은 [Docs.md](Docs.md)를 참고.
