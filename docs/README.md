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
│   ├── training/
│   │   ├── augmentation/           # 데이터 증강 프리셋 (Hydra config group)
│   │   │   ├── embed_align.yaml      # Embedding Alignment용 증강 설정 → cfg.augmentation으로 병합
│   │   │   ├── sft.yaml            # SFT용 증강 설정 → cfg.augmentation으로 병합
│   │   │   └── validate_augmentation/  # validate_augmentation.py 실행 설정
│   │   │       ├── pipeline.yaml       # 스크립트 전반 설정 (model, data, validate)
│   │   │       └── augmentation.yaml   # 검증에 사용할 증강 파라미터
│   │   ├── embed_align/              # Embedding Alignment 훈련 설정
│   │   │   └── pipeline.yaml       # defaults로 training/augmentation: embed_align 합성
│   │   ├── sft/                    # SFT 훈련 설정
│   │   │   └── pipeline.yaml       # LoRA, 학습률, hub_id/model_dir (final_checkpoints/embed_align) 등
│   │   └── rl/                     # RL(GRPO) 훈련 설정
│   │       └── pipeline.yaml       # GDPO, 보상함수, vLLM colocate, DDP 설정
│   └── inference/                  # 추론 설정
│       └── pipeline.yaml           # 모델 로드 모드, 입력 소스, 생성 파라미터, 출력 설정
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
│   ├── training/
│   │   ├── augmentation/           # 데이터 증강 파이프라인
│   │   │   ├── pipeline.py         # 증강 파이프라인 오케스트레이터
│   │   │   ├── strategies.py       # 15+ 증강 전략 구현
│   │   │   ├── tokenizer.py        # 조건/정답 토큰 시퀀스 생성
│   │   │   └── decoder.py          # 토큰 → 텍스트 디코딩
│   │   ├── embed_align/              # Embedding Alignment 훈련 모듈
│   │   │   ├── model_loader.py     # 4bit 양자화 로드 + PartialEmbedding/PartialLMHead
│   │   │   ├── dataset.py          # Arrow 로드 + 증강 + Chat Template
│   │   │   ├── collator.py         # Dynamic padding + label 마스킹
│   │   │   └── trainer.py          # TrainingArguments + Trainer 빌드
│   │   ├── sft/                    # SFT 훈련 모듈
│   │   │   ├── model_loader.py     # HF Hub base model + partial_state.pt 주입 + LoRA 적용 (공개 API: load_base_model_with_partial_state, build_lora_config)
│   │   │   └── trainer.py          # TrainingArguments + 표준 Trainer 빌드
│   │   └── rl/                     # RL(GRPO) 훈련 모듈
│   │       ├── __init__.py
│   │       ├── model_loader.py     # SFT(frozen) + RL(trainable) 멀티어댑터 구성 + vllm_base bf16 저장
│   │       ├── dataset.py          # RLPromptDataset (프롬프트 + 모델 시점 metadata, drop 데이터에 반영)
│   │       ├── advantage.py        # GDPO 정규화 + 토큰 신용할당 + 배치 정규화
│   │       ├── trainer.py          # RLTrainer (GRPOTrainer 서브클래스)
│   │       └── rewards/            # 11개 규칙 기반 보상함수
│   │           ├── __init__.py     # compute_all_rewards 공개 API
│   │           ├── parser.py       # 생성 토큰 파싱 (ParsedFloorplan, front_door_token_indices 포함)
│   │           ├── format_reward.py
│   │           ├── geometry_reward.py
│   │           ├── room_in_outline_reward.py    # 방 + front door의 outline 포함 검증 (케이스 A)
│   │           ├── outline_in_room_reward.py    # outline 꼭짓점이 방 내부에 포함되는지 (케이스 B)
│   │           ├── coverage_reward.py           # outline 내 빈공간 비율 (room_in_outline 쌍대)
│   │           ├── connectivity_reward.py       # 헝가리안 + 후보 기반 satisfiability
│   │           ├── count_reward.py
│   │           ├── spatial_reward.py            # 후보 기반 satisfiability
│   │           ├── input_consistency_reward.py  # 입력 앵커 방 무게중심 일관성
│   │           └── credit_assignment.py  # 토큰 수준 신용할당
│   ├── inference/                  # 추론 모듈
│   │   ├── model_loader.py         # Hub NF4 + partial_state.pt 주입 + LoRA adapter 스태킹
│   │   ├── condition_builder.py    # 입력 소스별 샘플 로드 + 증강 적용 + condition 토큰 빌드
│   │   ├── generator.py            # Chat Template 적용 + model.generate() 호출
│   │   ├── output_parser.py        # 생성 토큰 → 구조화 딕셔너리 역변환
│   │   └── result_saver.py         # 결과 JSON / 이미지 / 토큰 저장
│   └── utils/                      # 범용 유틸리티
│       └── extract_partial_state.py  # merged model.safetensors → partial_state.pt 추출
│
├── scripts/                        # CLI 실행 진입점
│   ├── build_dataset/
│   │   ├── rplan2json/
│   │   │   └── run_extraction.py   # PNG 배치 처리
│   │   └── json2arrow/
│   │       └── run_conversion.py   # Arrow 변환 실행
│   ├── build_model/
│   │   └── tokenization/
│   │       └── build_vocab.py      # 어휘 빌드 실행
│   ├── training/
│   │   ├── augmentation/
│   │   │   └── validate_augmentation.py # 증강 결과 검증
│   │   ├── run_embed_align.py        # Embedding Alignment 훈련 실행
│   │   ├── run_sft.py              # SFT 훈련 실행 (HF Hub + partial_state.pt + LoRA adapter 저장)
│   │   └── run_rl.py               # RL(GRPO) 훈련 실행 (vLLM colocate + DDP 자동 전환)
│   ├── inference/
│   │   └── run_inference.py        # 추론 실행 (입력 소스 선택, 다중 출력, 결과 저장)
│   └── utils/
│       └── extract_partial_state.py  # merged model.safetensors → partial_state.pt 추출 CLI
│
├── tests/                          # 검증 및 시각화 스크립트 (핵심 파이프라인 외)
│   ├── build_dataset/
│   │   └── rplan2json/
│   │       ├── validate_jsonl.py   # JSONL 스키마 무결성 검증
│   │       └── visualize_jsonl.py  # 평면도 JSONL 시각화
│   ├── training/
│   │   ├── embed_align/
│   │   │   ├── validate_resume.py          # Resume 체크포인트 복원 검증
│   │   │   └── validate_save_and_load.py   # 저장/로드 후 optimizer 업데이트 정상 동작 검증
│   │   ├── sft/
│   │   │   └── validate_sft.py             # SFT 통합 검증 (로드·LoRA구조·훈련·저장·Resume)
│   │   └── rl/
│   │       ├── __init__.py
│   │       ├── validate_rl.py              # RL 통합 검증 (파일존재·어댑터구조·훈련갱신·보상+생성)
│   │       └── verification/               # 보상함수·어드밴티지·손실 격리 검증 도구 모음
│   │           ├── _common.py              # vocab 로더, 토큰 fixture 빌더, reward_cfg 빌더, asserts
│   │           ├── group1_preprocessing/   # 변형/drop 후 metadata 추출 검증 (2개)
│   │           ├── group2_rewards/         # 11개 보상함수 의도 격리 검증
│   │           ├── group3_advantage/       # GDPO·token credit·batch_norm·micro_step (4개)
│   │           ├── run_all.py              # 일괄 실행 오케스트레이터
│   │           └── findings.md             # 트랙 A(스크립트) + B(코드 정독) 통합 보고서
│   ├── inference/
│   │   └── validate_inference.py           # 추론 통합 검증 (import·모델 로드·토큰 생성·파싱)
│   └── utils/
│       └── test_extract_partial_state.py   # extract_partial_state 단위/통합 검증
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
│           ├── final_checkpoints/              # 수동 관리 최종 버전 (훈련 run final과 분리)
│           │   └── embed_align/                  # Embedding Alignment 완료 후 수동 복사 (SFT 입력)
│           ├── merged_checkpoints/             # merge_lora 유틸로 생성한 standalone full model (merged 로드 모드용)
│           └── checkpoints/
│               ├── embed_align/                  # Embedding Alignment 훈련 run 체크포인트
│               │   └── {run_name}/             # run_name별 독립 저장 (기본: floorplan-embed-align)
│               │       ├── checkpoint-*/       # 에폭별 자동 저장 체크포인트
│               │       └── final/              # 훈련 run 최종 체크포인트 (partial_state.pt)
│               ├── sft/                        # SFT 훈련 run 체크포인트
│               │   └── {run_name}/             # run_name별 독립 저장 (기본: floorplan-sft)
│               │       ├── checkpoint-*/       # 에폭별 자동 저장 (adapter_model.safetensors)
│               │       └── final/              # 훈련 run 최종 체크포인트 (adapter + optimizer)
│               └── rl/                         # RL(GRPO) 훈련 run 체크포인트
│                   └── {run_name}/             # run_name별 독립 저장 (기본: floorplan-rl)
│                       ├── checkpoint-*/       # step별 자동 저장 (adapter_model.safetensors)
│                       └── final/              # 훈련 run 최종 체크포인트 (RL adapter + optimizer)
│
├── outputs/                        # Hydra 실행 로그 + 추론 결과
│   ├── training/
│   │   ├── embed_align/              # Embedding Alignment 실행 로그
│   │   │   └── YYYY-MM-DD/HH-MM-SS/
│   │   └── sft/                    # SFT 실행 로그
│   │       └── YYYY-MM-DD/HH-MM-SS/
│   └── inference/
│       └── {model.name}/{training_stage}/{YYYY-MM-DD}/{HH-MM-SS}/  # 날짜별 실행 디렉토리 (Hydra 로그 포함)
│           ├── .hydra/             # Hydra 설정 스냅샷 (config.yaml, overrides.yaml 등)
│           ├── run_inference.log   # 실행 로그
│           └── {plan_id}/          # 추론 결과 (입력 조건 + 출력 평면도)
│               ├── input/              # condition.json, tokens.txt, floorplan.png
│               ├── output/             # floorplan.json, tokens.txt, floorplan.png (num_outputs=1)
│               ├── output_0/ output_1/ # num_outputs>1 시 인덱스별 서브디렉토리
│               └── meta.json           # plan_id, 토큰 수, 소요 시간, 파싱 성공 여부
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

# 시스템 의존성 설치 (triton이 런타임에 C 코드를 컴파일하므로 필요)
sudo apt-get update && sudo apt-get install -y gcc python3.12-dev
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
| `trl >= 0.29.0` | RL(GRPO) 훈련 |
| `vllm >= 0.19.0` | RL rollout 생성 (colocate) |

> **WSL2 + PyTorch 2.10 + NCCL 2.27.5 환경 주의:** vLLM 의존성으로 인해 PyTorch가 2.10(cu128)으로 끌려오면, NCCL의 P2P/SHM 통신 회귀 버그로 DDP 초기화 단계에서 `ncclUnhandledCudaError "out of memory"`가 발생한다. `scripts/training/run_*.py`에서 `NCCL_P2P_DISABLE=1` / `NCCL_SHM_DISABLE=1` / `NCCL_IB_DISABLE=1` 및 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`를 자동 설정하여 SOCKET 통신과 가상 주소 점진 확장으로 우회한다 (성능 손실 무시 가능). 자세한 내용은 [Docs.md의 DDP 섹션](Docs.md#ddp-data-parallel-지원)을 참고.

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
[Embedding Alignment] 새 토큰 Embedding 워밍업 → 커스텀 토큰 embedding 안착
        │
        ▼
[SFT] LoRA Fine-tuning         → attention/MLP 전 레이어 학습
        │
        ▼
[Step 5] GRPO (GDPO) 강화학습 → 평면도 생성 모델
        │
        ▼
[Step 6] 추론 + 시각화 → 평면도 이미지
```

> **현재 구현 완료 범위:** Step 1 ~ Step 4, Embedding Alignment, SFT, GRPO(GDPO), Step 6 (추론)

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
uv run python tests/build_dataset/rplan2json/validate_jsonl.py \
    data/dataset/processed_dataset/rplan/jsonl/floorplans_0000.jsonl

# 리포트 파일 저장
uv run python tests/build_dataset/rplan2json/validate_jsonl.py \
    data/dataset/processed_dataset/rplan/jsonl/floorplans_0000.jsonl \
    -o report.txt
```

---

### Step 1 시각화: JSONL 시각화

```bash
# 특정 plan_id 시각화
uv run python tests/build_dataset/rplan2json/visualize_jsonl.py --plan_id 0 1 5

# 전체 시각화
uv run python tests/build_dataset/rplan2json/visualize_jsonl.py --all
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

### Embedding Alignment: 새 토큰 Embedding 워밍업

커스텀 토큰의 embedding과 lm_head 행만 훈련하여 기존 Pretrained 파라미터 공간에 안착시킨다.

```bash
# 기본 실행
uv run python scripts/training/run_embed_align.py

# 디버그 (10 step만 실행, W&B 비활성화)
uv run python scripts/training/run_embed_align.py \
    training.max_steps=10 training.report_to=none

# 하이퍼파라미터 오버라이드
uv run python scripts/training/run_embed_align.py \
    training.learning_rate=1e-3 training.num_train_epochs=3

# 다른 모델 사용
uv run python scripts/training/run_embed_align.py \
    model.user=meta-llama model.name=Llama-3.1-8B

# DDP 멀티 GPU: nproc_per_node를 config에서 설정하거나 override로 지정
uv run python scripts/training/run_embed_align.py \
    distributed.nproc_per_node=2

# 계속 훈련: 최신 체크포인트 자동 탐색 후 재개
uv run python scripts/training/run_embed_align.py \
    resume.enabled=true

# 계속 훈련: 특정 체크포인트 지정
uv run python scripts/training/run_embed_align.py \
    resume.enabled=true \
    resume.checkpoint_path=data/models/Qwen2.5-Coder-7B/checkpoints/embed_align/floorplan-embed-align/checkpoint-500
```

### SFT: LoRA Fine-tuning

HF Hub에서 base model을 로드하고 Embedding Alignment에서 훈련된 커스텀 토큰 가중치(`final_checkpoints/embed_align/partial_state.pt`)를 주입한 뒤 LoRA를 적용하여 attention/MLP 전 레이어를 fine-tuning한다.

```bash
# 기본 실행
uv run python scripts/training/run_sft.py

# 디버그 (10 step만 실행, W&B 비활성화)
uv run python scripts/training/run_sft.py \
    training.max_steps=10 training.report_to=none

# 하이퍼파라미터 오버라이드
uv run python scripts/training/run_sft.py \
    training.learning_rate=1e-4 lora.r=16

# DDP 멀티 GPU: nproc_per_node를 config에서 설정하거나 override로 지정
uv run python scripts/training/run_sft.py \
    distributed.nproc_per_node=2

# 계속 훈련: 최신 체크포인트 자동 탐색 후 재개
uv run python scripts/training/run_sft.py \
    resume.enabled=true

# 계속 훈련: 특정 체크포인트 지정
uv run python scripts/training/run_sft.py \
    resume.enabled=true \
    resume.checkpoint_path=data/models/Qwen2.5-Coder-7B/checkpoints/sft/floorplan-sft/checkpoint-500
```

**SFT 체크포인트 출력 구조:**
```
data/models/{model.name}/checkpoints/sft/{run_name}/
├── checkpoint-{step}/          # 에폭별 자동 저장 (최대 save_total_limit개 보존)
│   ├── adapter_model.safetensors  # LoRA adapter 가중치
│   ├── adapter_config.json        # LoRA 설정 (use_dora: false)
│   ├── optimizer.pt               # AdamW state
│   └── trainer_state.json
└── final/                      # 훈련 run 최종 체크포인트 (중간 체크포인트와 동일 구조)
    ├── adapter_model.safetensors
    ├── adapter_config.json
    ├── optimizer.pt
    ├── scheduler.pt
    ├── trainer_state.json
    └── tokenizer.json 등
```

---

### Step 6: 추론 실행

훈련된 모델을 로드하고 입력 조건에 대해 평면도 토큰 시퀀스를 생성한다. 결과는 JSON, 텍스트 토큰, 이미지로 저장된다.

**모델 로드 모드:**
- `adapters` (권장): HF Hub에서 base model 로드 → `partial_state.pt` 커스텀 토큰 주입 → LoRA adapter 스태킹
- `merged`: 사전 병합된 standalone full model 직접 로드 (`merged_checkpoints/` 디렉토리)

**입력 소스:**
- `jsonl_file`: 단일 JSONL 파일에서 샘플 로드
- `jsonl_dir`: JSONL 디렉토리 전체 일괄 처리
- `arrow`: HuggingFace Arrow 데이터셋에서 특정 split 사용
- `txt_dir`: 사전 증강된 토큰 텍스트 파일 디렉토리 (파일 1개 = 입력 조건 1개)

```bash
# 기본 실행 (JSONL 파일 30개, 증강 적용, do_sample=true)
uv run python scripts/inference/run_inference.py

# embed-align base model (adapters 없이, adapter 목록 비워둠)
uv run python scripts/inference/run_inference.py \
    model.training_stage=embed_align

# SFT adapter 적용
uv run python scripts/inference/run_inference.py \
    model.training_stage=sft \
    "inference.adapters=[{path: data/models/Qwen2.5-Coder-7B/final_checkpoints/sft, name: sft}]"

# Arrow test split, 10개, 3개 출력 (sampling 모드)
uv run python scripts/inference/run_inference.py \
    input.mode=arrow input.max_samples=10 \
    generation.num_outputs=3 \
    generation.do_sample=true generation.temperature=0.8

# 특정 plan_id만 처리
uv run python scripts/inference/run_inference.py \
    'input.plan_ids=[fp_00001,fp_00005]'

# 텍스트 파일 입력 모드 (증강 미적용)
uv run python scripts/inference/run_inference.py input.mode=txt_dir

# greedy 생성 (do_sample=false)
uv run python scripts/inference/run_inference.py \
    generation.do_sample=false generation.num_beams=1

# 증강 비활성화 (full condition 그대로 사용)
uv run python scripts/inference/run_inference.py augmentation.enabled=false
```

**출력 디렉토리 구조 (`outputs/inference/{model.name}/{training_stage}/{YYYY-MM-DD}/{HH-MM-SS}/`):**

Hydra 실행 로그·설정 스냅샷과 추론 결과가 동일한 날짜/시간 폴더 아래 저장된다.

```
outputs/inference/{model.name}/{training_stage}/{YYYY-MM-DD}/{HH-MM-SS}/
├── .hydra/             # Hydra 설정 스냅샷 (config.yaml, overrides.yaml 등)
├── run_inference.log   # 실행 로그
└── {plan_id}/
    ├── input/
    │   ├── tokens.txt          # 조건 토큰 텍스트
    │   ├── condition.json      # 조건 구조화 JSON
    │   └── floorplan.png       # 입력 조건 시각화
    ├── output/                 # num_outputs=1
    │   ├── tokens.txt          # 생성 토큰 텍스트
    │   ├── floorplan.json      # 역변환된 평면도 JSON
    │   └── floorplan.png       # 생성 결과 시각화
    └── meta.json               # plan_id, 토큰 수, 소요 시간, 파싱 성공 여부
```

> `num_outputs>1`이면 `output_0/`, `output_1/`, … 형태로 인덱스별 저장된다.

---

### 유틸리티: merged model.safetensors → partial_state.pt 추출

Embedding Alignment 저장 방식 변경 전(구 `merge_and_restore` 방식)에 저장된 `model.safetensors`에서
커스텀 토큰 가중치만 분리하여 현재 코드와 호환되는 `partial_state.pt`를 생성한다.

```bash
# 기본 실행 (출력: {checkpoint_dir}/partial_state_extracted.pt, 기존 파일 덮어쓰기 방지)
uv run python scripts/utils/extract_partial_state.py \
    --checkpoint_dir data/models/Qwen2.5-Coder-7B/checkpoints/embed_align/final \
    --model_name Qwen2.5-Coder-7B

# SFT 입력 경로(final_checkpoints/embed_align)로 직접 추출 + bfloat16 변환
uv run python scripts/utils/extract_partial_state.py \
    --checkpoint_dir data/models/Qwen2.5-Coder-7B/checkpoints/embed_align/final \
    --model_name Qwen2.5-Coder-7B \
    --output_path data/models/Qwen2.5-Coder-7B/final_checkpoints/embed_align/partial_state.pt \
    --dtype bfloat16

# vocab_extension.json 경로 직접 지정
uv run python scripts/utils/extract_partial_state.py \
    --checkpoint_dir /path/to/checkpoint \
    --vocab_extension_path /path/to/vocab_extension.json \
    --output_path /path/to/partial_state.pt
```

추출 결과 검증:
```bash
uv run python tests/utils/test_extract_partial_state.py
```

---

### SFT 검증: 통합 검증 스크립트

partial_state.pt 가중치 로드 및 커스텀 토큰 주입, LoRA 구조, 훈련 중 파라미터 갱신, 저장/Resume을 4단계로 통합 검증한다.

**검증 단계:**
- **Phase 0:** 파일 존재 확인 (partial_state.pt, tokenizer.json, tokenizer_config.json, vocab_extension.json)
- **Phase 1:** 모델 로드 + vocab_size 일치 + 커스텀 토큰 확인 + LoRA 구조 확인 (lora_A/lora_B 생성 여부, 7개 target_modules 전부 커버, base weight frozen)
- **Phase 2:** N step 훈련 전후 lora_A/lora_B 갱신 확인 + frozen 파라미터 불변 확인
- **Phase 3a:** 체크포인트 저장 확인 (adapter_model.safetensors, use_dora:false, optimizer.pt)
- **Phase 3b:** Resume 후 adapter 가중치 복원 + 추가 훈련 갱신 + global_step 연속성 확인

```bash
uv run python tests/training/sft/validate_sft.py

# 특정 model_dir 지정
uv run python tests/training/sft/validate_sft.py \
    --model_dir data/models/Qwen2.5-Coder-7B/final_checkpoints/embed_align
```

> 모든 Phase가 `[PASS]`가 출력되어야 정상.

---

### RL (GRPO): GDPO 강화학습

HF Hub base model + partial_state.pt + SFT adapter(frozen) + RL adapter(trainable) 멀티어댑터 구조로 GDPO + 토큰 수준 신용할당 강화학습을 수행한다. 롤아웃은 기본적으로 HF generate로 생성한다.

> **vLLM colocate 비활성 사유:** NF4 base + LoRA 환경에서 PEFT의 sync 단계 4bit merge round-trip 손실이 누적되어 vLLM 정책과 훈련 정책 logprob 차이가 폭증하고, IS ratio 붕괴 + KL 추정자 폭주로 학습이 발산한다 (자세한 내용은 [Docs.md의 vLLM 섹션](Docs.md#vllm-colocate-통합)). bf16 base + 긴 시퀀스 + 멀티 GPU 환경에서만 `rl.use_vllm=true`로 켜는 것을 권장한다.

```bash
# 단일 GPU 또는 DDP 자동 (distributed.nproc_per_node로 제어)
uv run python scripts/training/run_rl.py

# DDP 명시 (torchrun 직접 호출)
uv run torchrun --nproc_per_node=2 scripts/training/run_rl.py

# 디버그 (10 step, W&B 비활성화)
uv run python scripts/training/run_rl.py \
    training.max_steps=10 training.report_to=none

# vLLM colocate 강제 활성화 (bf16 base 환경에서만 권장)
uv run python scripts/training/run_rl.py rl.use_vllm=true

# 신용할당 비활성화 (균등 broadcast 모드)
uv run python scripts/training/run_rl.py \
    advantage.use_token_credit_assignment=false

# W&B 비활성화
uv run python scripts/training/run_rl.py training.report_to=none
```

**RL 체크포인트 출력 구조:**
```
data/models/{model.name}/checkpoints/rl/{run_name}/
├── checkpoint-{step}/
│   ├── adapter_model.safetensors  # RL LoRA adapter 가중치
│   ├── adapter_config.json
│   ├── optimizer.pt
│   └── trainer_state.json
└── final/
    ├── adapter_model.safetensors
    ├── adapter_config.json
    ├── optimizer.pt
    ├── scheduler.pt
    └── trainer_state.json
```

---

### RL 검증: 통합 검증 스크립트

모델 로드, 어댑터 구조, 훈련 파라미터 갱신, 보상함수, vLLM/HF 생성을 4단계로 통합 검증한다.

**검증 단계:**
- **Phase 0:** 파일 존재 확인 (partial_state.pt, SFT adapter, tokenizer, vocab_extension)
- **Phase 1:** 모델 로드 + vocab_size 일치 + 멀티어댑터 구조 확인 (sft frozen, rl trainable)
- **Phase 2:** N step 훈련 전후 rl 파라미터 갱신 + sft 파라미터 불변 확인
- **Phase 3:** 보상함수 9개 계산 + vLLM 또는 HF generate 통합 생성 검증

```bash
# HF generate 모드 (vLLM 없이)
uv run python tests/training/rl/validate_rl.py

# vLLM colocate 모드
uv run python tests/training/rl/validate_rl.py --use_vllm
```

> 모든 Phase가 `[PASS]`가 출력되어야 정상.

---

### RL 검증: 보상함수·어드밴티지·손실 격리 검증 (verification 도구 모음)

`validate_rl.py`(통합 4-phase)와 별개로 **보상함수와 어드밴티지/손실 흐름을 의도 격리 단위로 검증**하는 도구 세트. challenging 엣지케이스 fixture로 각 보상의 책임 범위와 신용할당 토큰 위치까지 직접 단언한다.

**구성 (3 그룹, 17 verifier, 100+ 케이스):**
- **Group 1 — 전처리**: 변형(flip/scale/translate/zoom) 후 좌표가 `_extract_metadata()`에 정확히 반영되는지, 8가지 drop이 metadata 필드별로 올바르게 마스킹되는지 검증
- **Group 2 — 보상별**: 11개 보상함수(format / count_total / count_type / orthogonality / no_overlap / room_in_outline / outline_in_room / coverage / connectivity / spatial / input_consistency) 각각에 대해 의도-위배 케이스 + 회귀 가드 케이스로 PASS/FAIL 단언
- **Group 3 — 어드밴티지/손실**: `gdpo_group_normalize`, `compute_token_advantages`, `_batch_normalize` mock 검증 + 실제 모델 1 micro-step E2E (advantages shape, RL adapter trainable / SFT frozen, 캐시 일관성)

```bash
# 전체 일괄 (mock 기반만 — 빠름)
uv run python tests/training/rl/verification/run_all.py --skip-microstep

# 전체 (실제 모델 1 micro-step 포함, GPU + SFT adapter 필요)
uv run python tests/training/rl/verification/run_all.py

# 특정 그룹만
uv run python tests/training/rl/verification/run_all.py --only group2

# 단일 verifier 직접 실행
uv run python tests/training/rl/verification/group2_rewards/verify_format_reward.py
```

**산출물**: 모든 verifier가 PASS 출력. 트랙 A(스크립트 실행) + 트랙 B(직접 코드 정독)로 발견된 의심점은 [tests/training/rl/verification/findings.md](../tests/training/rl/verification/findings.md)에 통합 정리.

---

### Embedding Alignment 검증: Resume 체크포인트 확인

체크포인트의 `partial_state.pt`가 올바르게 저장되어 있는지, Resume 시 new_embed/new_lm_head 복원이 가능한지 확인한다.

```bash
# 최신 체크포인트 자동 탐색 검증
uv run python tests/training/embed_align/validate_resume.py

# 특정 체크포인트 지정 검증
uv run python tests/training/embed_align/validate_resume.py \
    --checkpoint data/models/Qwen2.5-Coder-7B/checkpoints/embed_align/floorplan-embed-align/checkpoint-80304
```

---

### Embedding Alignment 검증: 저장/로드 후 optimizer 업데이트 검증

체크포인트 저장 후 optimizer의 Parameter 참조가 유지되어 훈련이 정상적으로 계속되는지 검증한다.

**검증 시나리오:**
- **Case 1 (연속 훈련):** 체크포인트 저장 후에도 new_embed가 계속 업데이트되는지 확인 (저장 전후 파라미터가 달라야 함)
- **Case 2 (Resume):** Phase 1 훈련 → 체크포인트 저장 → 새 모델 로드 → Resume → Phase 2 훈련이 정상적으로 이어지는지 확인

```bash
uv run python tests/training/embed_align/validate_save_and_load.py
```

> 임시 출력 디렉토리 (`data/temp/validate_save_load`)가 자동 생성/삭제된다.
> 두 케이스 모두 `PASS`가 출력되어야 정상.

**Embedding Alignment 체크포인트 출력 구조:**
```
data/models/{model.name}/checkpoints/embed_align/{run_name}/
├── checkpoint-{step}/          # 에폭별 자동 저장 (최대 save_total_limit개 보존)
│   ├── partial_state.pt        # new_embed / new_lm_head 가중치 (model.safetensors 없음)
│   ├── optimizer.pt            # AdamW state (~16MB)
│   └── trainer_state.json
└── final/                      # 훈련 run 최종 체크포인트 (중간 체크포인트와 동일 구조)
    ├── partial_state.pt        # new_embed / new_lm_head 가중치
    ├── optimizer.pt
    ├── scheduler.pt
    ├── trainer_state.json
    └── tokenizer.json 등
```

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

### `config/training/augmentation/embed_align.yaml` / `sft.yaml`

훈련 단계별로 독립된 증강 프리셋을 관리한다. Hydra **config group** 방식으로 각 파이프라인 yaml에서 합성되어 `cfg.augmentation`으로 접근된다.
현재 `embed_align.yaml`과 `sft.yaml`이 동일한 증강 파라미터를 사용하며, 추후 GRPO 등 각 단계별로 독립 관리한다.

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

### `config/training/rl/pipeline.yaml`

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `model.sft_adapter_dir` | `data/models/.../checkpoints/sft/final` | SFT adapter 경로 (frozen으로 사용) |
| `rl.use_vllm` | `false` | vLLM colocate 활성화 (NF4 환경에서는 발산하므로 비활성. bf16 base 전환 시에만 `true` 권장) |
| `rl.vllm_mode` | `"colocate"` | `"colocate"` (DDP 각 rank 내장) \| `"server"` (별도 GPU) |
| `rl.vllm_gpu_memory_utilization` | `0.45` | vLLM KV cache 비율 (24GB 기준, OOM 시 0.4로 낮춤) |
| `rl.num_generations` | `4` | 프롬프트당 rollout 생성 수 (G) |
| `rl.temperature` | `0.7` | 생성 온도 |
| `rl.kl_coeff` | `0.05` | KL 페널티 계수 (β) |
| `rl.clip_range` | `0.2` | PPO 클리핑 엡실론 |
| `advantage.use_token_credit_assignment` | `true` | 토큰 수준 신용할당 전역 토글 |
| `rewards.format.hard_gate` | `true` | R_format=0이면 모든 보상 0으로 강제 |
| `rewards.no_overlap.weight` | `2.0` | 최고 가중치 보상 (겹침 없음) |
| `rewards.room_in_outline.weight` | `1.5` | 비-outline 방 + front door가 outline 경계 내 포함되는지 (케이스 A, 신용할당 ON) |
| `rewards.outline_in_room.weight` | `1.0` | outline 꼭짓점이 방 내부에 포함되는지 (케이스 B, 신용할당 ON) |
| `rewards.coverage.weight` | `1.5` | outline 내 빈공간 없는지 (room_in_outline 쌍대, sequence-level) |
| `rewards.input_consistency.weight` | `1.5` | 입력에 좌표 명시된 방(앵커+drop_type)이 출력에 일관되게 존재하는지 |
| `rewards.input_consistency.threshold` | `15.0` | 무게중심 거리 임계값(px). 노이즈 3σ=9px + 모델 오차 마진 (transform 증강은 상대 오차 없음) |
| `training.learning_rate` | `5e-6` | RL adapter 학습률 |
| `training.optim` | `"paged_adamw_32bit"` | GPU OOM 방지 (momentum을 CPU RAM에 페이징) |
| `data.max_completion_length` | `512` | 최대 completion 토큰 수 |

### `config/training/sft/pipeline.yaml`

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `hydra.run.dir` | `outputs/training/sft/${now:%Y-%m-%d}/${now:%H-%M-%S}` | Hydra 로그·설정 스냅샷 저장 경로 |
| `model.hub_id` | `${model.user}/${model.name}` | HF Hub에서 base model 로드에 사용 |
| `model.model_dir` | `data/models/${model.name}/final_checkpoints/embed_align` | partial_state.pt 위치 (수동 관리 최종 버전) |
| `model.tokenizer_dir` | `data/models/${model.name}/tokenization` | 토크나이저 경로 (훈련 단계와 무관한 공통 경로) |
| `lora.r` | `32` | LoRA rank (adapter 표현력) |
| `lora.lora_alpha` | `64` | LoRA scaling factor (alpha/r=2, 실효 LR 스케일) |
| `lora.lora_dropout` | `0.05` | adapter dropout |
| `lora.target_modules` | `q/k/v/o_proj, gate/up/down_proj` | LoRA 적용 레이어 (attention + MLP 전부) |
| `training.output_dir` | `data/models/${model.name}/checkpoints/sft/${training.run_name}` | 체크포인트 저장 경로 |
| `training.run_name` | `"floorplan-sft"` | W&B run 이름 + 체크포인트 저장 서브디렉토리명 |
| `training.learning_rate` | `2e-4` | adapter 학습률 |
| `training.num_train_epochs` | `3` | 훈련 에폭 수 |
| `training.gradient_accumulation_steps` | `1` | 그래디언트 누적 steps |
| `training.save_total_limit` | `null` | 보존할 체크포인트 최대 수 (`null` = 제한 없이 전체 보존) |
| `training.load_best_model_at_end` | `true` | 훈련 종료 시 eval_loss 최고 체크포인트 복원 |
| `training.max_steps` | `0` | 디버그용 step 제한 (0=비활성) |
| `resume.enabled` | `false` | 계속 훈련 활성화 여부 |

### `config/training/embed_align/pipeline.yaml`

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `model.user` | `"Qwen"` | HuggingFace Hub 사용자(조직)명 |
| `model.name` | `"Qwen2.5-Coder-7B"` | 모델명 + 로컬 저장 디렉토리명 |
| `quantization.load_in_4bit` | `true` | 4bit 양자화 활성화 |
| `quantization.bnb_4bit_quant_type` | `"nf4"` | 양자화 방식 |
| `quantization.bnb_4bit_use_double_quant` | `true` | Double quantization |
| `data.max_length` | `4096` | 최대 시퀀스 길이 |
| `training.output_dir` | `data/models/${model.name}/checkpoints/embed_align/${training.run_name}` | 체크포인트 저장 경로 |
| `training.run_name` | `"floorplan-embed-align"` | W&B run 이름 + 체크포인트 저장 서브디렉토리명 |
| `training.learning_rate` | `5e-4` | 학습률 (공격적 설정) |
| `training.num_train_epochs` | `5` | 훈련 에폭 수 |
| `training.per_device_train_batch_size` | `2` | GPU당 배치 크기 |
| `training.gradient_accumulation_steps` | `1` | 그래디언트 누적 steps |
| `training.bf16` | `true` | 혼합 정밀도 (AMP) |
| `training.save_total_limit` | `3` | 보존할 체크포인트 최대 수 (오래된 순 삭제) |
| `training.load_best_model_at_end` | `true` | 훈련 종료 시 eval_loss 최고 체크포인트 복원 |
| `training.max_steps` | `0` | 디버그용 step 제한 (0=비활성) |
| `resume.enabled` | `false` | 계속 훈련 활성화 여부 |
| `resume.checkpoint_path` | `null` | 특정 체크포인트 경로 지정 (null이면 자동 탐색) |
| `resume.auto_find_latest` | `true` | output_dir에서 최신 체크포인트 자동 탐색 |

### `config/inference/pipeline.yaml`

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `hydra.run.dir` | `outputs/inference/${model.name}/${model.training_stage}/${now:%Y-%m-%d}/${now:%H-%M-%S}` | Hydra 로그·설정 스냅샷 + 추론 결과 저장 기준 경로 |
| `inference.load_mode` | `"adapters"` | `"adapters"`: Hub NF4 + partial_state.pt + adapter 스태킹. `"merged"`: standalone full model 직접 로드 |
| `inference.adapters` | `[]` | 적재할 adapter 목록 (`{path, name}` 형태). 비어있으면 embed-align base model로 추론 |
| `input.mode` | `"jsonl_file"` | 입력 소스: `jsonl_file` / `jsonl_dir` / `arrow` / `txt_dir` |
| `input.max_samples` | `30` | 처리할 최대 샘플 수 (`null`이면 전체) |
| `input.plan_ids` | `null` | 처리할 plan_id 리스트 (`null`이면 전체) |
| `augmentation.enabled` | `true` | 증강 활성화 여부 |
| `generation.max_new_tokens` | `2048` | 최대 생성 토큰 수 |
| `generation.do_sample` | `true` | 샘플링 여부 (false=greedy) |
| `generation.num_outputs` | `2` | 동일 조건에 대한 출력 수 |
| `model.training_stage` | `"sft"` | 출력 경로 레이블 (`outputs/inference/{model.name}/{training_stage}/{날짜}/{시간}/`) |

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
| Embedding Alignment | 새 토큰 Embedding 워밍업 훈련 | ✅ 완료 |
| SFT | LoRA Fine-tuning (attention/MLP 전 레이어) | ✅ 완료 |
| ~~Stage 2~~ | ~~DPO Fine-tuning~~ | ❌ 미진행 (계획 취소) |
| Stage 2 | GRPO (GDPO) 강화학습 + vllm colocate | ✅ 완료 |
| Step 6 | 추론 + 시각화 (adapters/merged 모드, 4개 입력 소스) | ✅ 완료 |

자세한 설계 내용은 [Docs.md](Docs.md)를 참고.
