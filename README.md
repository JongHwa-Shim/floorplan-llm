# FloorplanLLM

방 종류·개수, 일부 방 좌표, 방 연결 및 위치 관계를 조건으로 받아 평면도 토큰 시퀀스를 생성합니다. 입력과 출력은 프로젝트의 구조화된 토큰 형식을 사용합니다.

## 공개 범위

- `src/`: 데이터 전처리, 토큰화, 증강, 훈련, 추론 및 학습용 보상 계산
- `config/`: Hydra/OmegaConf 설정
- `scripts/`: 전처리·훈련·추론 실행 진입점과 모델 상태 추출 도구

데이터셋, 체크포인트, 실행 로그, 측정 결과, 연구 문서, 실험·성능 평가·표 생성 도구 및 개발 검증 스크립트는 이 공개 배포에 포함하지 않습니다. 이 저장소만으로 논문의 평가 표 전체를 재현할 수는 없습니다. 실행에 필요한 데이터와 체크포인트를 준비하고 설정의 경로를 자신의 환경에 맞게 지정해야 합니다.

## 설치

Python 3.11 이상과 `uv`를 사용합니다. 모델 훈련·추론에는 PyTorch, bitsandbytes 등 의존성과 호환되는 CUDA 환경이 필요합니다. 의존성은 `pyproject.toml`과 `uv.lock`에서 관리합니다.

저장소 루트에서 실행합니다.

```bash
uv sync
```

## 데이터 준비

RPLAN 원본 PNG를 준비하고 `config/build_dataset/rplan2json/pipeline.yaml`의 입력 경로를 지정합니다. 토크나이저의 기반 모델과 출력 위치는 `config/build_model/tokenization/pipeline.yaml`에서 설정합니다.

```bash
uv run python scripts/build_dataset/rplan2json/run_extraction.py
uv run python scripts/build_model/tokenization/build_vocab.py
uv run python scripts/build_dataset/json2arrow/run_conversion.py
```

분할은 `config/build_dataset/json2arrow/pipeline.yaml`에서 설정합니다. `split.held_out_room_count=null`은 모든 방 개수를 대상으로 무작위 분할합니다. 특정 개수를 지정하면 그 개수의 평면도 전체를 학습·검증 후보에서 제외하고 시험 표본을 해당 개수에서 선택합니다. 분할 시드와 선택된 평면도 ID는 데이터 디렉토리의 `split_manifest.json`에 저장됩니다.

공간 방향은 폴리곤 면적 중심점을 기준으로 계산합니다. 좌표 변환 후에는 입력 전용 노이즈를 추가하기 전에 방향을 갱신합니다.

## 훈련

각 단계의 데이터·모델·출력 경로를 `config/training/`에서 지정합니다. SFT는 embedding alignment 산출물을 사용하고, RL은 준비된 SFT 어댑터를 사용합니다.

```bash
uv run python scripts/training/run_embed_align.py
uv run python scripts/training/run_sft.py
uv run python scripts/training/run_rl.py
```

방 개수 제외 분할을 사용하는 경우 세 단계 모두 같은 데이터 경로와 `data.held_out_room_count`를 지정합니다. 데이터 로더는 선언된 제외 개수가 학습·검증 데이터에 포함되어 있으면 중단합니다. 각 단계에 연결하는 체크포인트도 동일한 분할로 학습한 산출물이어야 합니다.

Hydra 설정은 명령행에서 재정의할 수 있습니다.

```bash
uv run python scripts/training/run_rl.py training.max_steps=10 training.report_to=none
```

학습용 보상과 가중치는 `config/training/rl/pipeline.yaml`에서 설정합니다. 연결과 공간 방향은 각 보상 안에서 모든 조건을 만족하는 일대일 방 대응을 검사합니다. 두 보상은 독립적으로 계산합니다. Format 하드게이트는 훈련에 적용합니다.

최종 토큰 advantage 정규화는 시퀀스 평균의 표준편차에 `advantage.batch_eps`를 더한 값으로 나눕니다. 기본값은 `0.1`이며, 보상별 그룹 정규화의 `advantage.eps=1e-8`과 구분합니다.

## 추론

`config/inference/pipeline.yaml`에서 기반 모델, 확장 토크나이저, embedding alignment 산출물, 어댑터 및 입력 데이터 경로를 지정합니다.

```bash
uv run python scripts/inference/run_inference.py
```

기본 `inference.model_variant=full`은 SFT와 RL 어댑터를 모두 적재하고 활성화합니다. 필요한 어댑터가 빠져 있으면 오류로 처리합니다. 단계별 모델은 다음처럼 선택합니다.

```bash
uv run python scripts/inference/run_inference.py inference.model_variant=sft
uv run python scripts/inference/run_inference.py inference.model_variant=embed_align
```

`custom`은 설정의 어댑터 목록을 사용합니다. 이 선택은 `inference.load_mode=adapters`에 적용됩니다. `merged` 모드는 `model.model_dir`에 지정된 완성 모델을 직접 불러옵니다.

입력 조건의 변형·노이즈·삭제는 `augmentation.config_path`의 설정을 따릅니다. 이미 완성된 입력 토큰을 사용할 때는 `input.mode=txt_dir`을 지정합니다. 생성된 토큰, 평면도 JSON, 이미지는 설정에 지정된 로컬 출력 디렉토리에 저장됩니다.

실행 산출물과 로컬 연구 자료는 `.gitignore`로 추적에서 제외합니다.
