# 논문 실험 재현 가이드

본 문서는 `experiment_implementation_guide.md` 의 Exp 1 ~ 10 을 본 저장소에서 처음부터 끝까지
재현하기 위한 단계별 가이드다. 가이드만 보고도 동일하게 재현할 수 있도록 절대경로·환경
가정·발견된 시행착오까지 명시한다.

관련 문서:
- 실험 진행 이력·시행착오 : [`docs/EXPERIMENT_PROGRESS.md`](EXPERIMENT_PROGRESS.md)
- 측정 결과 요약 (정량값) : [`docs/EXPERIMENT_SUMMARY.md`](EXPERIMENT_SUMMARY.md)

## 0. 환경 가정

- **codebase root**: `/home/jhs/app/floorplan-llm/` (이하 모든 명령은 이 디렉토리에서 실행).
  경로가 다르다면 `cd` 만 바꿔서 그대로 사용 가능 (모든 명령은 codebase 기준 상대경로).
- **OS**: Linux (WSL2 Ubuntu).
- **GPU**: RTX 3090 ×2 가 표준 (DDP). 단일 GPU 도 추론은 가능, 학습은 시간이 2 배 든다.
- **Python**: 3.11+ (codebase). Baseline 들은 별도 `.venv` (HouseDiffusion=3.8, 그 외 3.10).
- **Package manager**: `uv` (project 루트의 `pyproject.toml` + `uv.lock` 사용).
- **HF Token (선택)**: Hub rate limit 회피용. 없어도 동작 (smoke 로 확인됨).

## 0.1 작업 디렉토리 레이아웃

```
/home/jhs/app/floorplan-llm/        # 본 연구 codebase (.git 추적)
├── baselines/                     # (Git 추적 제외) 외부 baseline repo clone 위치
│   ├── house_diffusion/{.venv, ckpts/exp/, datasets/rplan/, ...}
│   ├── gsdiff/{.venv, outputs/, ...}
│   └── ds2d/{.venv, models/{5R,6R,7R,8R}/, ...}
├── experiments/                   # (Git 추적 제외) 실험 산출물
│   ├── testset_unified.json       # 통일 test set plan_id (329 plans)
│   ├── testset_smoke.json         # 3 plan smoke 테스트셋 (dry-run 용)
│   ├── cited_baselines.yaml       # 직접 재현하지 않는 baseline 의 published 수치
│   ├── figure_selections.yaml     # Figures 6/7/8/11 case 선택 (사용자가 채움)
│   ├── generations/{model}/       # 공통 스키마 generation JSON
│   ├── renders/{model}/           # 256×256 raster PNG (통일 protocol)
│   ├── metrics/                   # 메트릭 CSV
│   ├── tables_figures/            # 논문 산출물 (Tables 1~9, Figures 6~11)
│   ├── user_study/                # responses.csv + aggregated.csv
│   ├── figure_candidates/         # 후보 plan_id 미리보기 PDF
│   └── failure_candidates/        # 실패 case 미리보기 PDF
├── data/models/Qwen2.5-Coder-7B/
│   ├── final_checkpoints/
│   │   ├── embed_align/partial_state.pt          # EA 결과 (필수)
│   │   ├── sft/active/                           # SFT adapter (symlink) → 한국어 폴더의 checkpoint-105399
│   │   └── rl/active/                            # RL adapter (symlink) → rl-token-credit-... /checkpoint-3300
│   └── tokenization/                             # 확장 토크나이저 + vocab_extension.json
├── data/dataset/processed_dataset/rplan/arrow/
│   ├── train/ validation/ test/                  # 원본 split (val=404, test=81)
│   └── eval_pool/                                # val + test concat (485 rows) — 통일 추론용
└── scripts/
    ├── utils/                     # testset 선정, GT dump
    ├── normalize/                 # 공통 스키마 변환
    ├── render/                    # 통일 raster 렌더
    ├── metrics/                   # 메트릭 측정
    ├── baselines/                 # baseline setup·데이터변환·정규화
    ├── experiments/               # 본 연구 추론 orchestrator + ablation shell
    │   └── ablations/             # exp4~7, exp10_gdpo_vs_grpo shell + README
    ├── figures/                   # Figures 6~11 빌더 + preview
    ├── tables/                    # Tables 1~9 빌더 + csv→LaTeX
    └── user_study/                # Streamlit 앱 + aggregate
```

## 0.2 통일 test set 분포

`experiments/testset_unified.json` 의 `_metadata.selected_counts`:

| 방 개수 | 선택 수 | 풀 수 |
|---|---|---|
| 5 | 29 | 29 (전부 사용, 100 미달) |
| 6 | 100 | 173 |
| 7 | 100 | 181 |
| 8 | 100 | 100 |
| **합계** | **329** | 485 |

> RPLAN Arrow split 비율 yaml 은 val=0.1%·test=0.5% 명시이나 실제 결과는 val=404·test=81로
> swap 적용됐다. 본 연구 학습은 train(80,303 plans)만 사용하므로 val·test 모두 학습 미사용 →
> 둘을 합쳐 평가 풀로 쓰는 것이 안전. eval_pool 은 `select_unified_testset.py` 가 concat 으로 생성.

## 0.3 실험별 입력·중간·최종 산출물 매핑

가이드 §1 ~ §10 의 명령은 다음 데이터 흐름을 구현한다. **모든 경로는 codebase root
(`/home/jhs/app/floorplan-llm/`) 기준 상대경로**. 한 실험에 필요한 입력·중간·최종 산출물을
한눈에 확인하려면 이 표를 먼저 보고 §5~§8 의 명령으로 이동하라.

### 공통 입력 (모든 Exp 가 참조)

| 자원 | 절대경로 | 용도 | 생성 단계 |
|---|---|---|---|
| RPLAN PNG 원본 | `data/dataset/raw_dataset/rplan/dataset/*.png` (80,788 장) | 모든 baseline 의 preprocessing 입력 | 사용자 외부 다운로드 |
| JSONL 원본 | `data/dataset/processed_dataset/rplan/jsonl/floorplans_*.jsonl` | 본 연구 인풋, HouseGAN++ JSON 변환 입력 | `scripts/build_dataset/rplan2json/` |
| Arrow split | `data/dataset/processed_dataset/rplan/arrow/{train,validation,test}/` | 본 연구 학습·검증 | `scripts/build_dataset/json2arrow/` |
| **eval_pool Arrow** | `data/dataset/processed_dataset/rplan/arrow/eval_pool/` (val+test concat, 485 rows) | 모든 평가의 GT 풀 | `scripts/utils/select_unified_testset.py` |
| **testset_unified.json** | `experiments/testset_unified.json` (329 plan_id) | Exp 1~3, Exp 8~10 의 plan_id 필터 | `scripts/utils/select_unified_testset.py` |
| 토크나이저 / vocab | `data/models/Qwen2.5-Coder-7B/tokenization/` | 추론 시 토큰화 | `scripts/build_model/tokenization/build_vocab.py` |
| EA partial_state | `data/models/Qwen2.5-Coder-7B/final_checkpoints/embed_align/partial_state.pt` | 본 연구·ablation 추론 기반 | EA 학습 |
| SFT adapter (active) | `data/models/Qwen2.5-Coder-7B/final_checkpoints/sft/active/` (symlink) | 본 연구 추론 1차 어댑터 | SFT 학습 |
| RL adapter (active) | `data/models/Qwen2.5-Coder-7B/final_checkpoints/rl/active/` (symlink) | 본 연구 추론 2차 어댑터 | RL 학습 |

### Exp 별 데이터 흐름

| Exp (§) | 모델/조건 | 입력 | 중간 산출물 | 최종 산출물 |
|---|---|---|---|---|
| **Exp 1** Comprehensive Eval (§5) | ours + 3 baseline (HD/GS/DS2D) + cited | (a) `experiments/generations/{model}/` (b) `experiments/renders/{model,gt_test}/` (c) `experiments/cited_baselines.yaml` (d) `experiments/user_study/responses.csv` | `experiments/metrics/exp1_compatibility_{model}.csv`, `experiments/metrics/exp1_fid.csv`, `experiments/metrics/exp1_self_{model}.csv` | `experiments/tables_figures/table_1.csv` |
| **Exp 2** Robustness (§6) | ours_{full,bubble_only,partial,sparse} + DS2D 호환 조건 | `experiments/generations/ours_{cond}/`, `experiments/generations/ds2d_{cond}/` | `experiments/metrics/exp2_novel_{model}_{cond}.csv` | `experiments/tables_figures/table_2.csv` |
| **Exp 3** Geometric Quality (§7) | ours + 3 baseline | `experiments/generations/{model}/` | `experiments/metrics/exp3_novel_{model}.csv` | `experiments/tables_figures/table_3.csv` |
| **Exp 4** Stage Ablation (§8.1) | ours_full / no_ea / no_sft / no_rl / pretrained | (a) EA partial_state (variant 별 skip 또는 normal) (b) SFT adapter (variant 별 학습 산출물) (c) RL adapter (variant 별 학습 산출물) | variant 별 `outputs/inference/.../{variant}/...`, `experiments/generations/{variant}/`, `experiments/renders/{variant}/`, `experiments/metrics/exp4_novel_{variant}.csv` | `experiments/tables_figures/table_4.csv` |
| **Exp 5** Reward Ablation (§8.2) | 6 variants — `no_geometry`, `no_connectivity`, ..., `no_outline_group` | EA partial_state + SFT (main) + RL (variant: reward weight=0 재학습) | `data/models/.../checkpoints/rl/ablation_{variant}/final/`, `experiments/generations/{variant}/`, `experiments/metrics/exp5_novel_{variant}.csv` | `experiments/tables_figures/table_5.csv` |
| **Exp 6** Token CA Ablation (§8.3) | 3 variants — `option_f`, `uniform`, `no_ca` | RL 재학습 (advantage 토글) | `data/models/.../checkpoints/rl/ablation_{variant}/final/`, `experiments/metrics/exp6_novel_{variant}.csv`, W&B reward history | `experiments/tables_figures/table_6.csv`, `experiments/tables_figures/figure_5.pdf` |
| **Exp 7** Augmentation Ablation (§8.4) | 4 variants — `no_transform`, `no_drop`, `no_shuffle`, `no_noise` | SFT 재학습 (augmentation toggle) → RL 재학습 | variant 별 SFT/RL checkpoint, `experiments/generations/{variant}/`, `experiments/metrics/exp7_novel_{variant}.csv` | `experiments/tables_figures/table_7.csv` |
| **Exp 8** Qualitative (§9) | ours + 3 baseline + GT | (a) `experiments/renders/{model,gt_test}/{plan_id}.png` (b) `experiments/figure_selections.yaml` | `experiments/figure_candidates/bucket_{5,6,7,8}.pdf` | `experiments/tables_figures/figure_{6,7,8}.pdf` (+ .png) |
| **Exp 9** User Study (§6) | ours / HD / DS2D / GS | `experiments/renders/{model,gt_test}/*.png` | `experiments/user_study/responses.csv`, `experiments/user_study/aggregated.csv` | Table 1 의 Realism 컬럼 통합 |
| **Exp 10** Analysis (§10) | (1) main run reward trajectories (2) GDPO vs Standard GRPO (3) Failure mode (4) Efficiency | (a) W&B history CSV (`experiments/wandb_rewards.csv` 등) (b) `experiments/metrics/exp10_novel_{gdpo,grpo}.csv` (c) `experiments/figure_selections.yaml::figure_11` (d) `experiments/metrics/ours_efficiency.csv` | `experiments/metrics/exp10_efficiency_{model}.csv` | `experiments/tables_figures/figure_{9,10,11}.{pdf,png}`, `experiments/tables_figures/table_{8,9}.csv` |

### 산출물 디렉토리 한눈에

| 디렉토리 | 무엇 | 누가 만드는가 |
|---|---|---|
| `outputs/inference/{model}/{stage}/{date}/{time}/` | 추론 원본 (tokens.txt / floorplan.json / floorplan.png / meta.json) | `scripts/inference/run_inference.py` (Hydra 자동) |
| `experiments/generations/{model}/{plan_id}_{idx}.json` | 공통 스키마 JSON (best-of-K 인덱스 포함) | `scripts/normalize/normalize_ours.py` 또는 `scripts/baselines/normalize_*.py` |
| `experiments/renders/{model}/{plan_id}_{idx}.png` (GT 만 suffix 없음) | 256×256 통일 raster | `scripts/render/render_unified.py` |
| `experiments/metrics/{exp_id}_{metric}_{model}.csv` | per-plan 메트릭 측정 결과 | `scripts/metrics/compute_*.py` |
| `experiments/tables_figures/table_{N}.csv` | 논문 표 (모델 × 메트릭 행렬) | `scripts/tables/build_table_{N}.py` 또는 `build_ablation_table.py` |
| `experiments/tables_figures/figure_{N}.{pdf,png}` | 논문 그림 | `scripts/figures/build_figure_{N}.py` |
| `experiments/figure_candidates/`, `experiments/failure_candidates/` | 후보 plan_id 미리보기 PDF | `scripts/figures/preview_candidates.py` |
| `experiments/user_study/responses.csv`, `aggregated.csv` | 응답 + 집계 | `scripts/user_study/app.py`, `aggregate.py` |
| `data/models/Qwen2.5-Coder-7B/checkpoints/{stage}/ablation_{variant}/` | ablation 학습 산출물 | `scripts/training/run_{embed_align,sft,rl}.py` |

## 1. 사전 준비 (자동 수행 완료된 항목)

```bash
# 1) 워크스페이스 디렉토리 (이미 생성됨)
mkdir -p baselines experiments/{generations,renders,metrics,tables_figures}

# 2) codebase env 에 메트릭 라이브러리 추가
uv add clean-fid networkx pandas matplotlib seaborn
# 검증: uv run python -c "from cleanfid import fid; import networkx; import pandas; import matplotlib; import seaborn; print('OK')"

# 3) 학습 체크포인트 symlink (영문·공백 없는 가이드 호환 경로)
#    data/models/Qwen2.5-Coder-7B/final_checkpoints/sft/active   →  '10epoch additional training using sft-lora-0424-first-begin(20epoch)/checkpoint-105399'
#    data/models/Qwen2.5-Coder-7B/final_checkpoints/rl/active    →  'rl-token-level-credit-assignment-max-step-10000-constant-lr-2e-5-real/checkpoint-3300'

# 4) 통일 test set + eval_pool 생성
uv run python scripts/utils/select_unified_testset.py
#   → experiments/testset_unified.json (329 plans)
#   → data/dataset/processed_dataset/rplan/arrow/eval_pool/ (val + test concat, 485 rows)
```

## 2. 본 연구 추론 (Ours)

### 2.1 Smoke run (먼저 1 분 검증 권장)

```bash
# 통일 testset 의 첫 3 plan 으로 smoke test (experiments/testset_smoke.json 이용)
uv run python scripts/inference/run_inference.py \
    model.training_stage=rl \
    "inference.adapters=[{path: data/models/Qwen2.5-Coder-7B/final_checkpoints/sft/active, name: sft},{path: data/models/Qwen2.5-Coder-7B/final_checkpoints/rl/active, name: rl}]" \
    input.mode=arrow \
    input.arrow_dir=data/dataset/processed_dataset/rplan/arrow/eval_pool \
    input.plan_ids_file=experiments/testset_smoke.json \
    input.max_samples=null \
    generation.do_sample=true generation.num_outputs=2 \
    augmentation.enabled=true \
    output.draw_labels=false
```

성공 시 로그에 다음이 모두 보여야 한다 (adapter 활성화 검증):
- `[model_loader] adapter 'sft' 로드 완료`
- `[model_loader] adapter 'rl' 로드 완료`
- `[model_loader] 다중 어댑터 동시 활성화 (set_adapter): ['sft', 'rl']` ← **이 줄이 없으면 RL adapter 가 forward 에 반영되지 않음 (2026-05-15 수정 이전 버그)**

### 2.1.b ours 의 3 가지 추론 protocol (baseline 비교용 — 2026-05-25 추가)

본 학습은 단일 protocol (augmented sparse) 이지만, baseline 비교를 fair 하게 만들기 위해 **세
가지 다른 추론 protocol** 을 별도 측정한다. 결과 정량값은 `docs/EXPERIMENT_SUMMARY.md` §2 참조.

| protocol | augmentation 설정 | num_outputs | sample | 매핑 baseline |
|---|---|---|---|---|
| **ours_augmented** | `augmentation.config_path=config/training/augmentation/sft.yaml` (학습과 동일) | 10 | sample (best-of-K) | 본 연구 자체 평가 |
| **ours_bubble** | `augmentation.config_path=config/training/augmentation/ours_bubble.yaml` — 좌표·front_door 전부 drop, transform OFF, noise OFF | 1 | greedy | HD / HouseGAN / HouseGAN++ / ref1 (cited) — **fair 비교** |
| **ours_fullcond** | `augmentation.enabled=false` | 1 | greedy | DStruct2Design `full_prompt` (직접 측정) — **upper bound sanity** |

#### ours_bubble 본 추론 명령

```bash
nohup uv run python scripts/inference/run_inference.py \
    model.training_stage=rl \
    "inference.adapters=[{path: data/models/Qwen2.5-Coder-7B/final_checkpoints/sft/active, name: sft},{path: data/models/Qwen2.5-Coder-7B/final_checkpoints/rl/active, name: rl}]" \
    input.mode=arrow \
    input.arrow_dir=data/dataset/processed_dataset/rplan/arrow/eval_pool \
    input.plan_ids_file=experiments/testset_unified.json \
    input.max_samples=null \
    generation.do_sample=false generation.num_outputs=1 \
    augmentation.enabled=true \
    augmentation.config_path=config/training/augmentation/ours_bubble.yaml \
    output.draw_labels=false \
    > experiments/run_ours_bubble.log 2>&1 &
```

#### ours_fullcond 본 추론 명령

```bash
nohup uv run python scripts/inference/run_inference.py \
    model.training_stage=rl \
    "inference.adapters=[{path: data/models/Qwen2.5-Coder-7B/final_checkpoints/sft/active, name: sft},{path: data/models/Qwen2.5-Coder-7B/final_checkpoints/rl/active, name: rl}]" \
    input.mode=arrow \
    input.arrow_dir=data/dataset/processed_dataset/rplan/arrow/eval_pool \
    input.plan_ids_file=experiments/testset_unified.json \
    input.max_samples=null \
    generation.do_sample=false generation.num_outputs=1 \
    augmentation.enabled=false \
    output.draw_labels=false \
    > experiments/run_ours_fullcond.log 2>&1 &
```

- 둘 다 ~1.3 시간 소요 (단일 RTX 3090, plan 당 ~13 s).
- 정규화 + 렌더링 + 메트릭 측정도 후속 자동 처리 가능 (가이드 §5 참조). 정규화 호출 시
  `--model_name ours_bubble` 또는 `ours_fullcond` 명시.

### 2.2 본 추론 (best-of-10, 329 plans)

```bash
bash scripts/experiments/run_ours_unified.sh 10 ours
```

- **측정값 (2026-05-15 본 추론 실측)**: 단일 RTX 3090 기준 **약 12.2 시간** 소요.
  3,290 generations, 평균 **13.36 s/gen** (smoke 의 10.6 s 보다 다소 느림 — KV cache 누적 영향).
  `parsing_success_rate = 1.0` (모든 generation 이 정상 파싱).
- 백그라운드 실행 권장: `nohup bash scripts/experiments/run_ours_unified.sh 10 ours > experiments/run_ours_unified.log 2>&1 &`
- **진행 모니터링은 반드시 `tail`로**: `tqdm` 진행률이 한 줄에 누적 출력되어 `head` 만 보면
  처음 일부 plan 까지만 보인다. 다음 명령으로 정상 종료 확인:
  ```bash
  tail experiments/run_ours_unified.log   # 끝줄에 [run_ours_unified] DONE 가 보여야 정상
  ls outputs/inference/Qwen2.5-Coder-7B/rl/*/*/ -d | wc -l         # >= 1
  ls outputs/inference/Qwen2.5-Coder-7B/rl/*/*/14-52-26/*/ 2>/dev/null | head  # 329 plan dir
  cat experiments/metrics/ours_efficiency.csv    # n_generations=3290, parsing_success_rate=1.0
  ```

산출물 (자동 생성):
- `outputs/inference/Qwen2.5-Coder-7B/rl/{YYYY-MM-DD}/{HH-MM-SS}/{plan_id}/` — 원본 추론 결과
  (`tokens.txt`, `floorplan.json`, `floorplan.png`, `meta.json`). 추론 시점의 `floorplan.png` 는
  `output.draw_labels` 토글 그대로 (기본 false → 라벨 없음).
- `experiments/generations/ours/{plan_id}_{0..9}.json` — 공통 스키마 (`normalize_ours.py` 결과)
- `experiments/renders/ours/{plan_id}_{0..9}.png` — 256×256 통일 raster (`render_unified.py` 결과,
  `--show_labels false` 기본 → 라벨 없음)
- `experiments/metrics/ours_efficiency.csv`

### 2.3 GT raster (Ours 와 동일 protocol)

```bash
uv run python scripts/render/render_unified.py \
    --gt_arrow data/dataset/processed_dataset/rplan/arrow/eval_pool \
    --plan_ids_file experiments/testset_unified.json \
    --output experiments/renders/gt_test \
    --show_labels false
```

- 출력: `experiments/renders/gt_test/{plan_id}.png` (best-of-K suffix 없음 — ours/baseline 의
  `{plan_id}_{idx}.png` 와 구분되는 점).
- 약 0.5 초 소요 (329 plan, polygon 그리기만).

> **렌더링 라벨 토글 2 종 정리**:
> 1. **추론 시점 PNG** (`outputs/.../floorplan.png`): `config/inference/pipeline.yaml` 의
>    `output.draw_labels` (기본 false). `result_saver.py` 가 `FloorplanVisualizer(show_labels=...)`
>    에 전달.
> 2. **통일 raster PNG** (`experiments/renders/*/*.png`): `scripts/render/render_unified.py` 의
>    `--show_labels` 인자 (기본 false). 둘은 별도 경로이므로 각각 명시해야 안전.
>
> `config/build_dataset/visualize_json/color_map.yaml` 의 `vis_settings.show_labels` 는 위 둘이
> `None` 일 때만 적용되는 base 기본값 (`true`). 디버그·검증용으로 라벨 ON 이 필요하면 두 토글
> 모두 명시적으로 true 로 override 한다.
>
> **SSAA (Super-Sampling Anti-Aliasing)**: 같은 yaml 의 `vis_settings.supersample` (기본 1, 비활성).
> 직교 도형은 ``LINE_8`` 단색 라인 + alpha 블렌딩만으로도 픽셀 단위 단색이 보장되어 SSAA 가
> 도리어 가장자리를 흐린다. 텍스트는 ``cv2.putText`` 의 ``LINE_AA`` 로 자체 안티앨리어싱 처리.
> 텍스트만 더 부드럽게 만들고 싶을 때 ``supersample: 2~4`` 로 켤 수 있다 (메모리·시간 비용 ``supersample²`` 배).

### 2.4 FID 측정 protocol — plan 당 1 sample 권장 (2026-05-25 추가)

ours best-of-10 (3,290 generation) 의 모집단을 그대로 FID 측정에 쓰면 GT (329) 와 비대칭이며
worst-9 까지 포함된 분산이 FID 를 끌어올린다. **plan 당 1 sample** 만 골라 GT 와 동일 모집단으로
비교하는 것이 표준:

```bash
# ours_best1 디렉토리 신설 (output_0 만 추출)
mkdir -p experiments/renders/ours_best1 experiments/generations/ours_best1
uv run python - <<'PY'
import shutil
from pathlib import Path
for f in Path("experiments/renders/ours").glob("*_0.png"):
    shutil.copy(f, Path("experiments/renders/ours_best1") / f"{f.stem[:-2]}.png")
for f in Path("experiments/generations/ours").glob("*_0.json"):
    shutil.copy(f, Path("experiments/generations/ours_best1") / f"{f.stem[:-2]}.json")
PY
# FID 측정
uv run python scripts/metrics/compute_fid.py \
    --gt experiments/renders/gt_test \
    --gen experiments/renders/ours_best1 \
    --names ours \
    --output experiments/metrics/exp1_fid_ours.csv
```

ours_fullcond / ours_bubble 은 이미 num_outputs=1 이라 `{plan_id}.png` 형식 — 별도 추출 불필요.

## 3. Baseline 환경 구축 (🚨 사용자 액션)

### 3.1 Repo clone + venv 생성

```bash
bash scripts/baselines/setup_baselines.sh all
# 또는 개별:
bash scripts/baselines/setup_baselines.sh hd     # HouseDiffusion
bash scripts/baselines/setup_baselines.sh gs     # GSDiff
bash scripts/baselines/setup_baselines.sh ds2d   # DStruct2Design
```

setup 스크립트가 자동으로 처리하는 것:
- 각 baseline 디렉토리에 **shim `pyproject.toml`** 생성 (codebase 루트의 `pyproject.toml` 의
  `requires-python>=3.11` 이 자식 venv 를 오염시키는 문제 회피).
- per-repo `.venv` 생성 (HouseDiffusion=Py3.8, GSDiff/DS2D=Py3.10).
- 한 baseline 실패해도 다음 baseline 진행 (set -e 비활성).

requirements 설치는 **기본 비활성**이다. 이유:
- HouseDiffusion 의 `mpi4py` 는 `apt-get install -y libopenmpi-dev` 필요.
- GSDiff 의 `pyg-lib` / `torch_scatter` 는 torch 버전 lock 필요.
- DS2D 의 `llama-recipes` 는 transformers 4.40 등 큰 의존성.

각자 시간 들여 점검해야 안전하다. 설치 시도 원하면:
```bash
bash scripts/baselines/setup_baselines.sh all --install
# 실패해도 || true 로 다음 baseline 진행.
```

### 3.2 Pretrained weights 다운로드 (수동)

| Baseline | URL | 배치 경로 |
|---|---|---|
| HouseDiffusion | https://drive.google.com/file/d/16zKmtxwY5lF6JE-CJGkRf3-OFoD1TrdR/view | `baselines/house_diffusion/ckpts/exp/model250000.pt` |
| GSDiff (topology-constrained) | https://drive.google.com/file/d/1pk7SmvLZ8ON3OUL3SNxPRu73ndVKru0z/view | `baselines/gsdiff/outputs/` 아래 압축 해제 |
| GSDiff (topology-autoencoder) | https://drive.google.com/file/d/1tExX8LdrFpJfBQH5y2emC6BltBwf9tHx/view | `baselines/gsdiff/outputs/` 아래 압축 해제 |
| DStruct2Design (RPLAN 4 variants) | https://drive.google.com/file/d/1cAYlEupNUGJefNdwkNaaq7fD3X3_P46D/view | `baselines/ds2d/models/{5R,6R,7R,8R}/` 압축 해제 |

> Google Drive 대용량 파일은 gdown 자동 다운로드가 종종 실패한다. 브라우저로 직접 받는 것을 권장.

## 3.3 Baseline 진행 정책 (2026-05-25 갱신)

| Baseline | 결정 | 사유 |
|---|---|---|
| **DStruct2Design** | ✅ 직접 재현 완료 | venv install + 80,788 PNG 데이터 변환 + 4 partial_prompt × 20 plan × 1 sample = 320 generations 추론 + outline 합성 정규화 + 메트릭 측정 모두 완료 |
| **HouseDiffusion** | ⏸ **cited 인용 (P1)** | HouseGAN++ JSON 형식 (벽 segment Nx4 + ed_rm 매핑) 이 본 연구 JSONL 의 polygon 기반 정보와 호환 안 됨. `Housegan-data-reader` 별도 clone + 변환은 reader 자체 setup 복잡. cited published 수치 (paper Table 1) 그대로 인용 |
| **GSDiff** | ⏸ **baseline 비교에서 제외** | preprocessing process2 가 repo 자체 코드 정합성 문제로 깨짐 (`TypeError: list indices must be integers or slices, not tuple`). README 의 `move.py` 도 실제 파일 없음. cited 값도 보고 불완전해 cited_baselines.yaml 에서도 제외 |

cited 값은 `experiments/cited_baselines.yaml` 에 사용자가 paper 직접 조회 후 입력 완료 (HD /
HouseGAN / HouseGAN++ / ref1 / HouseLLM / Ashual & Wolf / Johnson et al.).

## 4. Baseline 데이터 변환 + 추론 + 정규화

### 4.1 HouseDiffusion

```bash
# 4.1.1 HouseGAN++ JSON 입력 변환 (본 연구 JSONL → HouseGAN++ JSON)
uv run python scripts/baselines/convert_to_housegan_json.py \
    --src data/dataset/processed_dataset/rplan/jsonl \
    --plan_ids_file experiments/testset_unified.json \
    --out baselines/house_diffusion/datasets/rplan

# 4.1.2 HouseDiffusion 추론 (각 target_set 별 — 5/6/7/8 rooms)
cd baselines/house_diffusion
for n in 5 6 7 8; do
    uv run python scripts/image_sample.py \
        --dataset rplan \
        --batch_size 32 \
        --set_name eval \
        --target_set $n \
        --model_path ckpts/exp/model250000.pt \
        --num_samples 100
done
cd ../..

# 4.1.3 공통 스키마 정규화 + 통일 raster 렌더링
uv run python scripts/baselines/normalize_housediffusion.py \
    --src baselines/house_diffusion/outputs \
    --out experiments/generations/housediffusion \
    --pattern "*.json"   # 또는 *.npz, save_samples patch 형식에 따름
uv run python scripts/render/render_unified.py \
    --input experiments/generations/housediffusion \
    --output experiments/renders/housediffusion \
    --show_labels false
```

> **HouseDiffusion `save_samples` patch 필요**: 기본 repo 는 visualisation 만 떨구고 polygon
> 좌표를 별도 파일로 저장하지 않는다. `scripts/image_sample.py::save_samples` 끝에 polygon
> vertices 를 `experiments/generations/housediffusion_raw/{plan_id}.json` 등으로 dump 하는 코드를
> 직접 추가해야 한다 (출력 형식은 `normalize_housediffusion.py` 의 두 입력 모드 중 하나에 맞춤).

### 4.2 GSDiff

```bash
# 4.2.1 GSDiff repo 자체의 preprocessing 10단계 (rplan-extract.py ~ process10.py)
cd baselines/gsdiff
uv run python rplan-extract.py
for i in 1 2 3 4 5 6 7 8 9 10; do uv run python rplan-process${i}.py; done
uv run python move.py
cd ../..

# 4.2.2 추론 (topology-constrained mode, 통일 test set 만)
cd baselines/gsdiff
uv run python scripts/test_topology_constrained.py \
    --plan_ids_file ../../experiments/testset_unified.json   # ⚠ patch 필요 (인자 추가)
cd ../..

# 4.2.3 정규화 (GSDiff 출력 → 공통 스키마)
uv run python scripts/baselines/normalize_gsdiff.py \
    --src baselines/gsdiff/outputs/predict_rooms \
    --out experiments/generations/gsdiff
uv run python scripts/render/render_unified.py \
    --input experiments/generations/gsdiff \
    --output experiments/renders/gsdiff \
    --show_labels false
```

> GSDiff 는 자체 test set 757 plan 기준이라 인자 patch 가 필요할 수 있다. plan_id 매핑 검증
> 필수 (RPLAN 원본 PNG 의 plan_id 와 GSDiff structural graph 의 사용자 정의 id 가 같은지).

### 4.3 DStruct2Design

```bash
# 4.3.1 RPLAN → DS2D JSON 변환 (DS2D repo 자체 스크립트)
cd baselines/ds2d
uv run python rplan_dataset_convert.py
cd ../..

# 4.3.2 추론 (4 variants 각각)
cd baselines/ds2d
for v in 5R 6R 7R 8R; do
    uv run python run_generation_rplan.py --exprm $v --num_samples 20
done
cd ../..

# 4.3.3 정규화
uv run python scripts/baselines/normalize_ds2d.py \
    --src baselines/ds2d/generations \
    --out experiments/generations/ds2d
uv run python scripts/render/render_unified.py \
    --input experiments/generations/ds2d \
    --output experiments/renders/ds2d \
    --show_labels false
```

## 5. 메트릭 측정 (Exp 1 ~ 3)

> **ours 단독 메트릭 측정**: baseline 추론을 마치지 않았어도 ours vs GT 비교 (GED / FID /
> self-consistency / novel 11) 는 즉시 측정 가능하다. 본 추론 직후 ours 결과만 우선 확인하고
> 싶을 때 활용. 아래 for 루프에서 ``$m`` 을 ``ours`` 만 두면 된다.


```bash
TESTSET=experiments/testset_unified.json
POOL=data/dataset/processed_dataset/rplan/arrow/eval_pool

# Compatibility (GED)
for m in ours housediffusion gsdiff ds2d; do
    [[ -d experiments/generations/$m ]] || continue
    uv run python scripts/metrics/compute_compatibility.py \
        --gen experiments/generations/$m --gt_pool $POOL \
        --plan_ids_file $TESTSET --model_name $m \
        --output experiments/metrics/exp1_compatibility_${m}.csv \
        --timeout 10
done

# FID (clean-fid legacy_pytorch mode)
uv run python scripts/metrics/compute_fid.py \
    --gt experiments/renders/gt_test \
    --gen experiments/renders/ours experiments/renders/housediffusion experiments/renders/gsdiff experiments/renders/ds2d \
    --names ours housediffusion gsdiff ds2d \
    --output experiments/metrics/exp1_fid.csv

# Self/Prompt Consistency (Overlap/P.Overlap/R.Area MAPE)
for m in ours housediffusion gsdiff ds2d; do
    [[ -d experiments/generations/$m ]] || continue
    uv run python scripts/metrics/compute_self_consistency.py \
        --gen experiments/generations/$m --gt_pool $POOL \
        --plan_ids_file $TESTSET --model_name $m \
        --output experiments/metrics/exp1_self_${m}.csv
done

# Novel metrics 11종 (Exp 3 — Geometric Quality)
for m in ours housediffusion gsdiff ds2d; do
    [[ -d experiments/generations/$m ]] || continue
    uv run python scripts/metrics/compute_novel_metrics.py \
        --gen experiments/generations/$m --gt_pool $POOL \
        --plan_ids_file $TESTSET --model_name $m \
        --output experiments/metrics/exp3_novel_${m}.csv
done
```

### 5.1 Smoke test (GT 자기 vs 자기) — 메트릭 정상성 사전 확인

```bash
# GT 를 generation 으로 dump
uv run python scripts/utils/dump_gt_as_generations.py
# Novel metric 으로 self-check
uv run python scripts/metrics/compute_novel_metrics.py \
    --gen experiments/generations/gt \
    --gt_pool $POOL --plan_ids_file $TESTSET \
    --model_name gt_self \
    --output experiments/metrics/_smoke_novel_gt.csv
# 기대값: format=1.0, orthogonality≈1.0, no_overlap=1.0, count_total=1.0, count_type=1.0
# room_in_outline≈0.96 / outline_in_room=1.0 / coverage≈0.78 / spatial≈0.93 는 RPLAN polygon
# 변환 잔차로 1.0 이 안 나오는 정상 케이스 (smoke test 측정값).
```

## 6. User Study (🚨 사용자 액션 — Exp 9)

### 6.1 사전 조건
- `experiments/renders/{ours,housediffusion,ds2d,gsdiff,gt_test}/` 가 모두 채워져 있어야 함.

### 6.2 실행

```bash
# Streamlit 별도 설치 (학습 환경과 격리)
uv pip install streamlit
# 앱 실행 — `--` 뒤로 우리 옵션 전달 (Streamlit CLI 규약)
uv run streamlit run scripts/user_study/app.py -- \
    --renders_root experiments/renders \
    --models ours housediffusion ds2d gsdiff \
    --trials_per_model 10
```

### 6.3 참가자 진행 흐름
1. 참가자 13 명 (건축 전공 대학원생 또는 실무자 권장; IRB 절차 필요시 사전 동의 확보).
2. 각자 ID 입력 → 좌/우 평면도 (GT 1 장 + Generated 1 장, 위치·모델 무작위) 표시.
3. "A 가 더 사실적 / 동등 / B 가 더 사실적" 중 클릭.
4. 응답은 `experiments/user_study/responses.csv` 에 즉시 append.
5. 모델당 10 trials × 4 모델 = 한 참가자당 40 trials (~5 분).

### 6.4 집계
```bash
uv run python scripts/user_study/aggregate.py
# → experiments/user_study/aggregated.csv  (model, n_responses, n_participants, realism_mean, ci_low, ci_high)
```

Realism 점수 부호: GT=-1, Generated=+1, 동등=0. **0 에 가까울수록 GT 와 구분 불가 (모델 우수)**.

## 7. Ablations (Exp 4 ~ 7, 10) — 코드·명령만 준비됨

학습은 GPU 시간이 매우 크므로 사용자가 가용 시점에 실행. `eval_only` 모드는 학습 산출물이
없는 variant 는 자동 스킵하므로 안전.

```bash
# Stage Ablation (no_ea, no_sft, no_rl, pretrained)
bash scripts/experiments/ablations/exp4.sh train     # 또는 eval_only

# Reward Ablation (6 variants)
bash scripts/experiments/ablations/exp5.sh train

# Token Credit Assignment Ablation (3 variants)
bash scripts/experiments/ablations/exp6.sh train

# Augmentation Ablation (4 variants)
bash scripts/experiments/ablations/exp7.sh train

# GDPO vs Standard GRPO
bash scripts/experiments/ablations/exp10_gdpo_vs_grpo.sh train
```

각 variant 의 학습 / 추론 / 정규화 / 메트릭 측정까지 한 shell 안에서 처리한다. 상세는
`scripts/experiments/ablations/README.md` 참조.

**Ablation 토글 정리 (config override 키)**:
| variant | override |
|---|---|
| w/o EA | `model.skip_partial_state=true` |
| w/o SFT | `model.sft_adapter_dir=null` |
| no token CA | `advantage.use_token_credit_assignment=false` |
| no batch norm | `advantage.use_batch_norm=false` |
| standard GRPO | `advantage.use_gdpo_normalization=false` |
| no reward X | `rewards.X.weight=0.0` |
| no augmentation X | `augmentation.X.*` (전략별; exp7.sh 참조) |

## 8. 표·그림 산출

```bash
# Tables
uv run python scripts/tables/build_table_1.py \
    --metrics_dir experiments/metrics --cited experiments/cited_baselines.yaml \
    --user_study experiments/user_study/responses.csv \
    --output experiments/tables_figures/table_1.csv
uv run python scripts/tables/build_table_2_robustness.py \
    --metrics_dir experiments/metrics --output experiments/tables_figures/table_2.csv
uv run python scripts/tables/build_table_3_geometric.py \
    --metrics_dir experiments/metrics --output experiments/tables_figures/table_3.csv
for n in 4 5 6 7; do
    uv run python scripts/tables/build_ablation_table.py \
        --metrics_dir experiments/metrics --pattern "exp${n}_novel_*.csv" \
        --output experiments/tables_figures/table_${n}.csv
done
uv run python scripts/tables/build_ablation_table.py \
    --metrics_dir experiments/metrics --pattern 'exp10_novel_*.csv' \
    --output experiments/tables_figures/table_8.csv
uv run python scripts/tables/build_table_9_efficiency.py \
    --metrics_dir experiments/metrics --output experiments/tables_figures/table_9.csv

# Figure 후보 미리보기 → 사용자가 figure_selections.yaml 채움
uv run python scripts/figures/preview_candidates.py
# → experiments/figure_candidates/bucket_{5,6,7,8}.pdf 검토 후 plan_id 선정

# Figures (figure_selections.yaml 채운 뒤)
uv run python scripts/figures/build_figure_6.py
uv run python scripts/figures/build_figure_7.py
uv run python scripts/figures/build_figure_8.py
uv run python scripts/figures/build_figure_9.py --wandb_csv experiments/wandb_rewards.csv
uv run python scripts/figures/build_figure_10.py \
    --gdpo_csv experiments/wandb_gdpo.csv --grpo_csv experiments/wandb_grpo.csv
uv run python scripts/figures/build_figure_11.py
```

**W&B history CSV export (Figure 9 / 10 용)**:
```python
import wandb, pandas as pd
api = wandb.Api()
run = api.run("entity/project/<run_id>")
keys = [f"rewards/reward_{k}/mean" for k in [
    "format","count_total","count_type","orthogonality","no_overlap",
    "room_in_outline","outline_in_room","coverage","connectivity","spatial","input_consistency",
]]
hist = run.history(keys=keys, pandas=True)
hist["step"] = hist.get("_step", range(len(hist)))
hist.to_csv("experiments/wandb_rewards.csv", index=False)
```

## 9. LaTeX/docx 통합 (🚨 사용자 액션 — 논문 작성 단계)

```bash
# CSV → LaTeX 변환
uv run python scripts/tables/csv_to_latex.py experiments/tables_figures/table_1.csv \
    --caption "Comprehensive quantitative comparison." --label "tab:t1" \
    > paper/tables/table_1.tex
```

이후 AiC LaTeX 템플릿 (`elsarticle.cls`) 또는 docx 템플릿에 `experiments/tables_figures/` 산출물
삽입 → 캡션 번호·표 정렬 등 서식 조정. 본 가이드는 여기서 마무리.

## 9.5 정성 시각화 실험 (for_paper — 2026-07-06 추가)

논문 figure 용 정성 시각화·비교 실험 5종. 통합 스크립트 `scripts/figures/for_paper_experiments.py`
(`--exp {1..5}`) 로 실행하며 산출물은 `experiments/for_paper/{실험}/` 하위에 저장한다.

> **2026-07-06 재생성 (SFT final 체크포인트 + 20 sample):** SFT 체크포인트를 `checkpoint-110418`
> → `final` (10epoch 추가학습 완료본) 로 교체하고 입력 조건당 sample 을 20개로 증량했다. 이전
> checkpoint-110418 결과는 각 실험의 `sft_110418/` 폴더로 아카이브 보존. 모듈 상수 `N_SAMPLES=20`.
> RL 은 lora_B=0 무효라 여전히 생성 제외.
>
> **2026-07-14 png 전용 (벡터 중단):** 산출물은 **png 만** 저장한다(모듈 상수 `VECTOR_EXTS=()`).
> 2026-07-06 에 pdf·svg 벡터를 병행 추가했으나 사용자 요청으로 중단하고 기존 pdf/svg 를 전부
> 삭제했다. `FloorplanVisualizer.render_floorplan_to_vector()` 메서드는 유지되며, `VECTOR_EXTS` 를
> `("pdf","svg")` 로 되돌리면 벡터 병행 저장이 재활성화된다.
>
> **2026-07-14 door 겹침 블렌딩 제외:** 현관문·interior door 는 겹침 색상 블렌딩에서 제외되어
> solid 원색으로 선명하게 표시된다(`_compute_blend_regions` 가 `is_door` 요소 skip). 방-방 겹침만
> 블렌딩. raster·vector 공통. 상세는 `docs/Docs.md` "평면도 렌더링 방식" 참조.

| # | 실험 (폴더) | 구성 | 목적 |
|---|---|---|---|
| 1 | `input_output_example` | **50** plan (bucket 5/6/7/8 균등, `--n_plans` 로 조정) | 학습-시 증강 입력 조건 + 정답 평면도의 토큰·시각화 예시 (input/·output/ 분리 + 양쪽 bubble diagram) |
| 2 | `generated_floorplan_per_stage` | 20 plan × stage × 20 sample | **동일 입력**에 EA / SFT(final) / **SFT(checkpoint-35133)=sft_old** 비교. 입력=outline+방 2개(bubble 풍부, `coords_keep`). 구 결과 old/·old_old/ (2026-07-14d/e) |
| 3 | `1-to-many-generation` | 10 plan×40 + 신규 40 plan×40 + **sparse/ 10 plan×10** | 중간 밀도 입력 1개 고정 → 다양한 variants. **exp3 항상 spatial 절제**(2026-07-07). `sparse/` = room polygon 실루엣만·counts 다수·connection/spatial 적당(2026-07-14b) |
| 4 | `varying_input_density` | 4 density × 10 plan × **20** sample | 입력 밀도별 (polygons→spatial→counts→connectivity_only) 생성 |
| 5 | `imprecise_and_contradictory_inputs` | contradictory 10 + imprecise 20 plan × 20 sample | contradictory(total<type합 강제) / imprecise(**σ=10px** 노이즈, 2026-07-13 30→10 완화) 입력 |

```bash
# 각 실험 개별 실행 (모델 추론 포함 — exp2~5 는 GPU + SFT adapter 필요)
uv run python scripts/figures/for_paper_experiments.py --exp 1 --n_plans 50  # 추론 없음(CPU) — GPU 작업과 병렬 가능
uv run python scripts/figures/for_paper_experiments.py --exp 2   # EA/SFT stage 비교
uv run python scripts/figures/for_paper_experiments.py --exp 3   # 1-to-many
uv run python scripts/figures/for_paper_experiments.py --exp 4   # varying density
uv run python scripts/figures/for_paper_experiments.py --exp 5   # imprecise/contradictory
```
> **exp1 은 모델을 안 쓰므로**(입력 조건 + GT 시각화만) exp2~5 의 GPU 실행과 **병렬**로 돌려도 안전하다
> (별도 프로세스·다른 출력 디렉토리). `--n_plans` 로 생성 plan 수를 조정한다(기본 10).

**핵심 구현 규칙:**
- 저장 구조 (exp2~5): `{실험}/{plan_id}/input/{input.txt, bubble_diagram.png, rooms.png}` +
  `output/{stage}/{n}.{txt,png}` (exp2 는 `{plan_id}/{stage}/{n}.{txt,png}`). 아카이브는
  `sft_110418/` (exp2), `output/sft_110418/` (exp3~5).
- **exp1 저장 구조 (2026-07-06 개선)**: `{plan_id}/input/{input.txt, rooms.png, bubble_diagram.png}`
  + `{plan_id}/output/{output.txt, floorplan.png, bubble_diagram.png}`. 입력 조건(drop 반영)과
  정답(full) 을 분리 저장하고 양쪽 모두 bubble diagram 을 제공. 입력 bubble 은 drop 반영,
  출력 bubble 은 GT 전체 connectivity (`save_output_artifacts()`).
- **입력 조건은 plan 당 한 번만 증강** 후 모든 stage·sample 에 동일하게 재사용 (매번 증강 시
  입력이 달라지는 것을 방지). exp2 는 EA/SFT 가 완전히 같은 condition 을 받는 것이 핵심. 단
  증강은 seed=None 이라 **실행 시마다 입력 조건이 새로 생성**됨 (아카이브 sft_110418 과는 다른 입력).
- input.txt 는 블록 단위 (`<ROOM_SUMMARY> ... <END_ROOM_SUMMARY>` 한 줄) 로 표시.
- bubble_diagram node = **edge(connectivity) 에 등장하는 방만** (rooms 는 type 색칠 lookup),
  node 색은 `color_map.yaml` room_colors 연동. **(2026-07-14)** node 를 data 좌표 원으로 그리고
  최근접 node 간격이 지름보다 커지도록 좌표를 스케일해 **node 수가 많아도 겹치지 않게** 개선
  (`draw_bubble_diagram`). figure 크기는 layout bbox 에 비례해 렌더 비율 일정.
- **RL stage 는 보류** — RL adapter lora_B=0 무효 (SUMMARY §1.2). 코드의 `EXP2_STAGES`/`GEN_STAGES`
  상수에 `"rl"` 을 추가하면 RL 수정 후 자동 재실행된다. 현재는 EA/SFT 만.
- sample 수는 모듈 상수 `N_SAMPLES=20` (2026-07-06 4~5 → 20 증량). exp4·exp5 는 idempotent skip
  (output png ≥ N_SAMPLES 존재 시 건너뜀) — 중단 후 재실행 시 이어서 완료.
- **png 전용 출력**(2026-07-14): 평면도·bubble diagram 모두 png 만 저장(`VECTOR_EXTS=()`). door 는
  겹침 블렌딩 제외로 solid 표시.
- **exp2~5 순차 재생성 드라이버**: `bash experiments/run_for_paper_exp2to5.sh` (각 exp 를 독립
  프로세스로 순차 실행하여 GPU 반납, 로그 `experiments/run_for_paper_exp{2..5}.log`).
- **exp3 증강(`_DENSITY_MID_YAML`, 2026-07-07)**: spatial 은 항상 절제(방침). counts(total 0.4/
  type 0.5) · connectivity(edge 0.5/pair 0.2) · spatial(0.55) 을 과감하게, block/type/coords/door/
  front_door 도 활성화해 입력 sparsity 를 다양하게 만든다. `run_exp3(n_plans, n_samples, seed,
  exclude_pids, idempotent)` 로 파라미터화됨.
- **exp3 sample 추가/신규 배치 스크립트**:
  - `extend_exp3_samples.py` — 기존 exp3 각 plan 의 **저장된 input.txt 조건 그대로** 재사용해
    21~40 등 추가 sample 을 이어붙임(증강 재실행 없이 동일 입력 보장, 새 seed 로 신규 sample).
  - `add_exp3_batch.py` — for_paper **전 실험 plan_id 를 모두 제외**하고 새 plan N개(현재 40)를
    새 증강 조건으로 뽑아 각 M개(현재 40) 생성. idempotent(완료 plan skip).
  - `add_exp3_sparse.py` — **room polygon 실루엣만**(drop_coords=1.0 + outline 좌표 복원) + counts
    다수·connection/spatial 적당 절제한 sparse 입력으로 10 plan×10. `1-to-many-generation/sparse/`
    에 별도 저장. 핵심: `drop_state.drop_coords.discard(0)` 로 outline(실루엣) 좌표만 복원 후 재토큰화.
- **exp5 imprecise 노이즈(2026-07-13)**: anchor 방 좌표 Gaussian 노이즈 σ 를 30→**10px** 로 완화
  (`_apply_strong_noise_to_anchors` 기본값·`run_exp5` 호출 모두). σ=30 은 방이 과도하게 일그러졌다.
  `redo_imprecise.py` — 기존 imprecise 결과를 `imprecise_input/old/` 로 옮긴 뒤 신규 20 plan 을
  σ=10 으로 재생성(조건당 20 output). 향후 `--exp 5` 도 σ=10 사용.
- **exp2 입력 증강 재조정(2026-07-13→14d)**: bubble 을 풍부하게(`p_drop_edge`·`p_drop_block`
  0.5→0.15) + room polygon 은 sparse 하지 않게. 초기(2026-07-13 `redo_exp2.py`)엔 drop_coords=0.65
  로 폴리곤을 줄였으나 일부 plan 이 outline 도 사라져 "거의 문만 남는" 문제가 생겨, **2026-07-14d
  `redo_exp2_v2.py`** 로 교체: `run_exp2(coords_keep=2)` 가 outline(실루엣)을 항상 표시하고 방
  폴리곤을 정확히 2개만 남긴다(`_limit_room_polygons` 후처리, 0-polygon 방지). `run_exp2` 는
  `n_plans/n_samples/seed/exclude_pids/aug_yaml/idempotent/coords_keep` 파라미터화 + bucket 소진 시
  top-up. 구 결과 아카이브: `old/`(최초 sft.yaml exp2) · `old_old/`(2026-07-13 sparse redo).
- **exp2 SFT 체크포인트 비교(2026-07-14e)**: `build_inference_cfg`/`load_stage_model` 에 `sft_path`
  override 추가. `redo_exp2_sft_old.py` 가 exp2 각 plan 의 저장된 input.txt 조건을 그대로 재사용해
  다른 SFT 체크포인트(`checkpoints/sft/checkpoint-35133`, DoRA)로 20 sample 생성 →
  `{plan_id}/sft_old/`. 같은 입력에 EA / SFT(final)=sft/ / SFT(35133)=sft_old/ 비교.

**⚠️ matplotlib 백엔드 (Tcl 크래시 방지):** 벡터(pdf/svg) 병행으로 figure 생성/close 가 폭증하면
기본 Tk 백엔드가 `Tcl_AsyncDelete: async handler deleted by the wrong thread` 로 프로세스를
크래시시킨다(2026-07-06 exp4 재생성 중단 사례). `for_paper_experiments.py` 상단에서 pyplot import
전에 `matplotlib.use("Agg")` 를 강제해 해결. 신규 렌더 스크립트도 이 모듈을 import 하면 Agg 상속.

**메모리 OOM 주의:** 반복 재실행 시 좀비 프로세스 누적으로 각 모델 (~10GB GPU) 이 겹쳐 RAM 99%
도달. 실행 전 `pkill -9 -f for_paper_experiments` 로 정리. `generate_n_samples` 는 `max_new_tokens`
1800 + 매 sample `torch.cuda.empty_cache()` 로 방어. ※ **주의**: `pkill -f for_paper_experiments`
를 exp 를 띄우는 **같은 명령줄**에서 쓰면 자기 launcher 셸까지 죽여 즉시 종료된다(별도 스텝/스크립트
파일에서 실행할 것).

### 시각화 재생성 (color_map / 렌더링 방식 변경 시)

색 팔레트·렌더링 방식(예: door 블렌딩 제외)을 바꾼 뒤에는 **모델 추론 없이** input.txt / {n}.txt /
output.txt 토큰을 파싱해 모든 png 를 다시 그린다:

```bash
uv run python scripts/figures/regenerate_for_paper_pngs.py
# input/rooms + 생성 sample({n}) + exp1 output/floorplan 재렌더. png 전용(VECTOR_EXTS=()).
# 이미지 1건 실패해도 try/except 로 건너뛰고 계속(로그 "실패: N"). 렌더러는 홀수 좌표(garbage)
# 안전 처리. exp1 output/bubble_diagram 은 output.txt 에 edge 없어 복원 불가라 유지(door 없어 무관).
```
> (참고) `add_missing_vectors.py` 는 pdf/svg 소급용이었으나 벡터 중단(2026-07-14)으로 미사용.

> 렌더링 방식 (solid + 겹침 블렌딩 + 테두리 최상단, door 통합) 상세는 `docs/Docs.md` 의
> "평면도 렌더링 방식" 섹션 참조. rooms.png 재생성 시 input.txt 의 `<EDGE>` door 좌표까지
> 파싱해야 interior door 가 누락되지 않는다.

## 10. 사용자 액션 체크리스트

- [x] **Baseline pretrained weight 다운로드** (3 종, Google Drive) — 2026-05-22 완료
- [x] **본 연구 best-of-10 추론 실행** — 2026-05-15~16 완료 (3,290 generations, parse_success=1.0)
- [ ] Baseline 추론 환경 검증 (`uv pip install -r requirements.txt`, `mpi4py` 등 시스템 의존성 점검)
- [ ] Baseline 별 추론 실행 (HouseDiffusion / GSDiff / DS2D) + 정규화 + 렌더링
- [ ] (선택) Ablation 학습 실행 (variant 당 수 시간 ~ 며칠)
- [ ] User study 13 명 모집·실행·`responses.csv` 수집
- [ ] `figure_selections.yaml` plan_id 선정 (`figure_candidates/` 미리보기 참조)
- [ ] `cited_baselines.yaml` 의 published 수치 채우기 (논문에서 직접 옮겨 적기)
- [ ] LaTeX/docx 최종 통합·서식 조정

## 11. 시행착오 · 자주 발생하는 문제

자세한 시행착오 기록은 [`docs/EXPERIMENT_PROGRESS.md`](EXPERIMENT_PROGRESS.md) 의 "주요 시행착오 ·
수정 기록" 섹션 참조. 핵심 요약:

| 증상 | 원인 | 해결 |
|---|---|---|
| baseline `.venv` 에서 Python 3.11 요구 warning | 부모 `pyproject.toml` 의 `requires-python>=3.11` 가 자식 venv 로 끌려옴 | 각 baseline 디렉토리에 shim `pyproject.toml` 자동 생성 (setup_baselines.sh) |
| HouseDiffusion `requirements.txt` 설치 실패 | `mpi4py`, `tensorflow 2.11`, `torch 2.0.0.dev` 등 시스템 의존성 | `apt-get install -y libopenmpi-dev gcc python3.8-dev` 후 재시도. 안 되면 base inference 만 작동하는 최소 패키지로 직접 추림 |
| `KeyError: split 'eval_pool'` | `input.arrow_split` 는 DatasetDict 의 split. eval_pool 은 별도 디렉토리 | `input.arrow_dir=data/.../arrow/eval_pool` 로 직접 지정 |
| RL adapter 가 추론에 반영 안 됨 | `load_adapter` 후 `set_adapter` 누락 → 첫 어댑터만 active | `model.base_model.set_adapter(["sft","rl"])` 명시 호출 (2026-05-15 패치) |
| 라벨 텍스트가 반투명하게 보임 | 방 A 라벨 그린 직후 방 B alpha 채우기가 텍스트 위에 덮임 | renderer 에서 라벨 그리기를 분리해 모든 도형 그린 뒤 마지막에 라벨만 한꺼번에 그림 |
| 라벨이 방 중앙에 표시되어 다른 방에 가려짐 | 텍스트 위치가 polygon 중심 | `min(xs)` + `mean(ys)` 로 변경 (방의 가장 왼쪽에서 시작) |
| 통일 raster (`experiments/renders/*/`) 가 라벨 ON 으로 그려짐 | `render_unified.py` 가 color_map.yaml `vis_settings.show_labels=true` 기본값 따름 — 추론 config 의 `output.draw_labels` 와 별도 경로 | `--show_labels false` 인자 명시 (논문 figure 표준) |
| 추론 백그라운드 상태 오판 (실제 정상 종료인데 죽었다고 진단) | `head` 로 진행률 로그 앞부분만 봄 | 반드시 `tail experiments/run_ours_unified.log` + `[run_ours_unified] DONE` 라인 + plan_id dir 개수 + efficiency CSV 존재 여부 다중 확인 |
| baseline `uv pip install` 시 `numpy>=2.4.2` override 충돌 | 부모 codebase pyproject.toml `[tool.uv] override-dependencies` 가 자식 venv 까지 영향 | `--no-config` 플래그 추가 |
| HuggingFace `huggingface-cli login` 명령이 없다 | `huggingface_hub` 1.x 부터 CLI 이름이 `hf` 로 변경 | `uv run hf auth login` 또는 Python `from huggingface_hub import login; login(token=...)` |
| GSDiff `from datasets.X import ...` import 충돌 | GSDiff 자체 `datasets/` 폴더 vs HF `datasets` 패키지 동명 | `datasets/` → `gs_datasets/` rename + 50 파일 import sed patch (단 HF `from datasets import` 없는지 grep 사전 검증) |
| DS2D `run_generation_rplan.py` `KeyError: 4` | 변환된 data 가 list-of-dict (`room_type` 문자열) 인데 코드는 list-of-list 의 `[4]` 정수 인덱싱 기대 | `rooms[u][4]` → `rooms[u]['room_type']` patch |
| DS2D 추론 출력이 `generations/rplan/` 아닌 `rplan_greedy/` | `num_samples=1` 시 `out_dir += '_greedy'` 자동 append | 정규화 시 `_greedy` suffix 직접 가리킴 |
| DS2D outline 없음 → novel metric 모두 0 | `common_to_parsed()` success 조건이 `rooms[0].room_type=='outline'` | `normalize_ds2d.py` 에 `_synthesize_outline()` (shapely unary_union + envelope) 추가하여 합성 outline 을 rooms[0] 에 삽입 |
| DS2D plan_id 가 RPLAN 원본 plan_id 와 매핑 안 됨 | DS2D 가 자체 train split 에서 random sample 추출 | (1) plan-self metric (Overlap/FID/format/ortho/no_overlap) 만 측정 — `compute_self_consistency.py` + `compute_novel_metrics.py` 에 plan-self only mode patch 추가 (2) GED 는 `ground_truth.json` 으로 self-paired 측정 |
| HouseDiffusion 데이터 변환 — 우리 wrapper JSON 형식 ≠ HD reader 기대 형식 | HD reader: 벽 segment Nx4 + `ed_rm` 매핑. 우리 wrapper: 방-방 adjacency | cited 인용 정책 (P1) 으로 결정 |
| GSDiff process2 `TypeError: list indices must be integers or slices, not tuple` | process1 산출물 구조와 process2 기대 구조 mismatch (repo 자체 정합성 문제) | 디버그 미완 → cited 인용. README 의 `move.py` 도 실제 파일 없음 |
| ours_fullcond FID 0.21 / GED 0.27 / input_consistency 0.9999 — 너무 완벽 | full cond 입력에 좌표가 포함되어 LLM 이 거의 reconstruction | (1) sanity upper bound 로만 사용 (2) **ours_bubble** protocol (좌표 drop 만, 다른 정보 유지) 별도 측정 → HD cited 와 fair 비교 |
| 통일 raster (`experiments/renders/*/`) 가 라벨 ON 으로 그려짐 | `render_unified.py` 가 color_map.yaml `vis_settings.show_labels=true` 기본값 따름 — 추론 config 의 `output.draw_labels` 와 별도 경로 | `--show_labels false` 인자 명시 (논문 figure 표준) |
| 추론 백그라운드 상태 오판 (실제 정상 종료인데 죽었다고 진단) | `head` 로 진행률 로그 앞부분만 봄 | 반드시 `tail experiments/run_ours_unified.log` + `[run_ours_unified] DONE` 라인 + plan_id dir 개수 + efficiency CSV 존재 여부 다중 확인 |
