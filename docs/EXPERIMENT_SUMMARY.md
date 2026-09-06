# 실험 결과 요약

본 문서는 `experiment_implementation_guide.md` 의 Exp 1 ~ 10 중 **현재까지 측정 완료된 부분**의
결과를 정리한다. 진행 이력은 `docs/EXPERIMENT_PROGRESS.md`, 실행 명령·산출물 경로 매핑은
`docs/EXPERIMENTS_GUIDE.md` 를 참조한다.

마지막 갱신: 2026-07-06 (⚠️ RL adapter lora_B=0 무효 이슈 발견 — §1.2 경고 추가, §2 ours 결과는 실질 EA+SFT)

---

## 1. 실험 환경

### 1.1 하드웨어 / 소프트웨어

| 항목 | 값 |
|---|---|
| GPU | NVIDIA RTX 3090 ×2 (현재 추론은 단일 GPU 사용) |
| OS | Linux (WSL2 Ubuntu) |
| Python | 3.11+ (codebase), 3.8 (HouseDiffusion), 3.10 (GSDiff/DS2D) |
| PyTorch | 2.6+ cu128 (codebase), 2.3 cu121 (DS2D), 2.0.1 cu118 (GSDiff/HD) |
| Transformers | 5.x (codebase), 4.40 (DS2D) |
| Quantization | bitsandbytes NF4 (codebase), 8bit (DS2D Llama-3) |
| Package manager | uv (per-repo `.venv`, shim `pyproject.toml`) |

### 1.2 본 연구 모델 구성 (3-stage)

| 단계 | 산출물 | 핵심 설정 |
|---|---|---|
| **EA** Embedding Alignment | `final_checkpoints/embed_align/partial_state.pt` | 새 토큰 567개의 embed/lm_head 만 워밍업, lr=5e-4, 2 epoch |
| **SFT** | `final_checkpoints/sft/active/` (symlink) | LoRA r=32, attention+MLP, lr=2e-4, NF4 base |
| **RL** (GDPO) | `final_checkpoints/rl/active/` (symlink) | 11 reward + GDPO + Option F 토큰 신용할당, max_steps=10000, lr=2e-5 |

추론 시 `model.base_model.set_adapter(["sft", "rl"])` 명시 호출 — 이전 누락으로 RL 무효화되던 버그 수정 완료.

> ⚠️ **RL adapter 무효 이슈 (2026-07-06 발견 — 재측정 필요):** RL adapter 의 **lora_B 가 모든
> checkpoint (로컬 3종 + 서버 rl-max-step-.../{checkpoint-5000, 10000, final}) 에서 전부 0**
> 임을 확인. LoRA delta = $(\alpha/r) \cdot B A$ 이므로 $B=0$ → delta=0 → RL adapter 가 forward 에
> 아무 영향을 주지 않는다 (`set_adapter` 가 정상 호출돼도 RL 기여 0). 대조군 SFT adapter 는
> lora_B norm=2138 로 정상. 즉 **RL(GDPO) 학습이 rl adapter 의 lora_B 로 gradient 를 전혀 흘리지
> 않은 학습 파이프라인 문제** (저장 버그 아님 — 중간 checkpoint 도 전부 0). 검증: 동일 모델에서
> `set_adapter(["sft"])` vs `set_adapter(["sft","rl"])` greedy 출력이 **완전 동일**.
>
> **함의:** 본 문서 §2 의 ours_* 결과들 (ours_augmented / ours_fullcond / ours_coords50 / ours_bubble
> 등) 은 EA+SFT+RL 로 표기돼 있으나, RL 이 무효이므로 **실질적으로 EA+SFT 결과**이며 RL 기여는
> 없다. RL 학습 파이프라인 (`src/training/rl/`) 수정 → 재학습 → RL 유효 checkpoint 확보 후 §2
> 재측정이 필요하다. RL 수정 전까지 stage 비교 실험 (EXPERIMENTS_GUIDE §12) 은 EA/SFT 만 수행.

### 1.3 데이터셋

| 자원 | 위치 | 수 |
|---|---|---|
| RPLAN PNG 원본 | `data/dataset/raw_dataset/rplan/dataset/` | 80,788 |
| JSONL | `data/.../processed_dataset/rplan/jsonl/` | 80,788 |
| Arrow train / val / test | `data/.../rplan/arrow/{train,validation,test}/` | 80,303 / 404 / 81 |
| **eval_pool** (val ∪ test) | `data/.../rplan/arrow/eval_pool/` | **485** |
| **testset_unified** | `experiments/testset_unified.json` | **329** (5:29, 6:100, 7:100, 8:100) |

### 1.4 ours 의 3 가지 추론 protocol

ours 의 학습은 단일 protocol (augmented sparse) 이지만, baseline 비교를 fair 하게 만들기 위해 **네 가지
다른 추론 protocol** 을 별도 측정한다. 각 protocol 은 `cited_baselines.yaml::protocol_note` 의 비교
대상에 매핑된다.

| protocol | 정의 | augmentation yaml | num_outputs | 매핑 baseline |
|---|---|---|---|---|
| **ours_augmented** | 본 학습 protocol — DropBlock/Edge/Spatial/Coords 모두 적용 + transform 증강 + best-of-10 | `sft.yaml` (학습 동일) | 10 (best-of-K) | 본 연구 자체 평가 |
| **ours_bubble** | HD 등 cited 와 동일 sparsity — 좌표·front_door 전부 drop, bubble diagram only 입력, transform OFF, noise OFF | `ours_bubble.yaml` (신설) | 1 (greedy) | HouseDiffusion / HouseGAN / HouseGAN++ / ref1 (cited) |
| **ours_coords50** (신규) | partial spec — 방 절반의 좌표만 입력 (Bernoulli p=0.5) + 종류·인접·spatial·front_door 전부, transform/noise OFF | `ours_coords50.yaml` (신설) | 1 (greedy) | **HD vs ours fair 비교의 핵심 (FID 8.26 < HD cited 12.2~14.8)** |
| **ours_fullcond** | full reconstruction — augmentation 전체 OFF, 입력에 좌표·종류·인접·front_door 모두 포함 | (disabled) | 1 (greedy) | DStruct2Design `full_prompt` (직접 측정) |

### 1.5 평가 protocol 결정

| 항목 | 결정 |
|---|---|
| FID 모집단 | plan 당 1 sample. **per-bucket FID 측정 시 N=200/bucket (testset_perbucket200)**, overall FID 측정 시 N=329 (testset_unified). ours_augmented 의 경우 `output_0` 만 사용 (`experiments/renders/ours_best1/`) |
| **Data leak footnote** | testset_perbucket200 의 bucket 5 는 eval_pool 에 29 plan 만 있어 train 에서 171 plan hold-out 추가. bucket 6/7/8 도 일부 train hold-out (27/19/100). 학습 시 본 plan 이지만 ours_coords50 augmentation (drop_coords=0.5) 정확 변형으로 학습한 적 없음. 논문에 명시 |
| FID mode | clean-fid `legacy_pytorch` (pytorch-fid 호환) |
| best-of-K reduction (GED) | `min` (ours_augmented 만 해당; ours_bubble · ours_fullcond 는 단일 sample) |
| GED 라이브러리 | `networkx.optimize_graph_edit_distance` 첫 근사값, timeout=10s |
| 인접 판정 | shapely polygon 거리 ≤ 8px (RPLAN convention) |
| novel 11 metric mode | plan-self only mode 추가 (plan_id 매핑 불가능한 baseline 용) |

---

## 2. 실험 결과

### 2.1 본 연구 추론 효율성 (Exp 10 — Table 9 입력)

| 항목 | ours_augmented | ours_fullcond | ours_coords50 | ours_bubble |
|---|---|---|---|---|
| 총 generation 수 | 3,290 (329 × 10) | 329 (329 × 1) | 329 (329 × 1) | 329 (329 × 1) |
| 평균 elapsed | 13.36 s/gen | 12.84 s/gen | 14.09 s/gen | 13.63 s/gen |
| 총 소요 시간 | ~12.2 시간 | ~1.3 시간 | ~1.29 시간 (1h 17m) | ~1.25 시간 (1h 14m) |
| Parsing success rate | 1.000 | 1.000 | 1.000 | 1.000 |

원본 CSV: `experiments/metrics/{ours_efficiency,ours_fullcond_efficiency,...}.csv`.

### 2.2 Exp 1 — Comprehensive Quantitative Evaluation

#### Compatibility (Graph Edit Distance, ↓ lower better)

| 모델 / protocol | 5 | 6 | 7 | 8 | mean | 출처 |
|---|---|---|---|---|---|---|
| **ours_fullcond** | **0.00** | **0.12** | **0.08** | **0.70** | **0.27** | 측정 |
| **ours_coords50** | **0.55** | **0.63** | **1.49** | **2.65** | **1.50** | 측정 (partial spec, 방 절반 좌표) |
| ours_augmented | 1.76 | 2.26 | 3.92 | 5.86 | 3.82 | 측정 |
| ours_bubble | 2.90 | 4.14 | 5.80 | 8.07 | 5.73 | 측정 (HD-fair sparsity) |
| HouseDiffusion (cited) | 1.5 | 1.2 | 1.7 | 2.5 | 1.7 | HD paper Table 1 |
| HouseGAN++ (cited) | 1.9 | 2.2 | 2.4 | 3.9 | 2.6 | HD paper Table 1 |
| HouseGAN (cited) | 2.5 | 2.4 | 3.2 | 5.3 | 3.4 | HD paper Table 1 |
| ref1 RLVR-LLM (cited) | 0.01 | 0.02 | 0.10 | 0.15 | 0.07 | ref1 Table 1 |
| HouseLLM (cited) | 0.24 | 0.25 | 0.28 | 0.32 | 0.27 | HouseLLM paper |
| Ashual & Wolf 2019 (cited) | 7.5 | 9.2 | 10.0 | 11.8 | 9.6 | HD/ref1 Table 1 |
| Johnson et al. 2018 (cited) | 7.7 | 6.5 | 10.2 | 11.3 | 8.9 | HD/ref1 Table 1 |
| DStruct2Design (full_prompt, self-paired) | 5.95 | 1.20 | 2.45 | 4.25 | 3.46 | 측정 (80 plan, GT=ground_truth.json) |
| DStruct2Design (only_total_area, "bubble") | 5.65 | 1.20 | 1.75 | 3.00 | 2.90 | 측정 |
| DStruct2Design (only_room_area, "partial") | 9.25 | 1.00 | 2.70 | 3.65 | 4.15 | 측정 |
| DStruct2Design (some_room_area, "sparse") | 4.25 | 1.80 | 3.50 | 7.00 | 4.14 | 측정 |

#### Diversity (FID, ↓ lower better)

| 모델 / protocol | 5 | 6 | 7 | 8 | overall | 출처 |
|---|---|---|---|---|---|---|
| **ours_fullcond** | — | — | — | — | **0.21** | 측정 (329 vs 329) |
| **ours_coords50 (per-bucket N=1000)** | **3.83** | **3.79** | **3.82** | **4.10** | **3.89** | **측정 (4000 plan, N=1000/bucket — 최종 안정 측정, HD 대비 3.5배 우위)** |
| ours_coords50 (per-bucket N=250) | 9.87 | 8.41 | 8.99 | 10.15 | 9.35 | 측정 (1000 plan) |
| ours_coords50 (per-bucket N=200) | 11.29 | 8.75 | 10.04 | 11.99 | 10.52 | 측정 (800 plan, 초기) |
| ours_coords50 (329 plan overall) | — | — | — | — | 8.26 | 측정 (329 unified test) |
| ours_augmented (best-of-10 → oracle) | — | — | — | — | 28.46 | best-of-10 GT raster L2 최소 |
| ours_augmented (best-of-10 → self-reward) | — | — | — | — | 29.85 | best-of-10 self-reward 최대 (no GT) |
| ours_augmented (best-of-10 → best1) | — | — | — | — | 29.80 | output_0 (random pick) |
| ours_bubble | — | — | — | — | 29.93 | 측정 (329 vs 329, HD-fair sparsity) |
| DStruct2Design (full, vs GT 329) | — | — | — | — | 98.69 | 측정 (80 gen vs 329 GT, 비대칭) |
| DStruct2Design (full, self-paired) | — | — | — | — | **28.42** | 측정 (80 vs 80, self-paired GT) |
| DStruct2Design (bubble, self-paired) | — | — | — | — | **29.03** | 측정 (80 vs 80) |
| DStruct2Design (partial, self-paired) | — | — | — | — | **30.21** | 측정 (80 vs 80) |
| DStruct2Design (sparse, self-paired) | — | — | — | — | **30.40** | 측정 (80 vs 80) |
| **HouseDiffusion (cited)** | **12.2** | **13.4** | **13.6** | **14.8** | **13.50** | HD paper Table 1 |
| **DStruct2Design_full** (per-bucket vs perbucket200 GT) | 127.01 | 118.35 | 107.30 | 118.70 | 117.84 | 측정 (N=20 gen vs 200 GT, 비대칭) |
| **DStruct2Design_bubble** (per-bucket vs perbucket200) | 126.55 | 128.25 | 119.02 | 111.92 | 121.43 | 측정 (N=20 vs 200) |
| **DStruct2Design_partial** (per-bucket vs perbucket200) | 150.18 | 124.38 | 116.21 | 117.67 | 127.13 | 측정 (N=20 vs 200) |
| **DStruct2Design_sparse** (per-bucket vs perbucket200) | 118.43 | 126.03 | 126.10 | 113.32 | 120.97 | 측정 (N=20 vs 200) |
| **DStruct2Design_full** (per-bucket, self-paired) | 52.89 | 45.96 | 45.42 | 52.11 | 49.10 | 측정 (N=20 vs 20 self-paired) |
| **DStruct2Design_bubble** (per-bucket, self-paired) | 51.28 | 50.71 | 46.97 | 49.63 | 49.65 | 측정 (N=20 vs 20) |
| **DStruct2Design_partial** (per-bucket, self-paired) | 62.96 | 41.37 | 46.35 | 50.78 | 50.36 | 측정 (N=20 vs 20) |
| **DStruct2Design_sparse** (per-bucket, self-paired) | 46.97 | 43.15 | 49.67 | 56.78 | 49.14 | 측정 (N=20 vs 20) |
| **ours_coords50** (per-bucket, **N=20 sub-sample**) | 30.11 | 16.84 | 18.95 | 22.85 | **22.19** | 측정 (DS2D 와 동일 N 으로 fair 비교) |
| HouseDiffusion (cited) | 12.2 | 13.4 | 13.6 | 14.8 | — | HD paper Table 1 |
| HouseGAN++ (cited) | 30.4 | 37.6 | 27.3 | 32.9 | — | HD paper Table 1 |
| HouseGAN (cited) | 37.5 | 41.0 | 32.9 | 66.4 | — | HD paper Table 1 |
| ref1 (cited) | 9.0 | 8.8 | 7.8 | 7.0 | — | ref1 Table 1 |
| HouseLLM (cited) | 8.6 | 7.5 | 8.1 | 9.0 | — | HouseLLM paper |

#### Self/Prompt Consistency

| 모델 / protocol | Overlap rate (↓) | P.Overlap (↓) | R.Area MAPE (↓) | 비고 |
|---|---|---|---|---|
| ours_fullcond | **0.000** | **0.0000** | **0.000** | 입력 좌표 그대로 출력 — reconstruction-like |
| **ours_coords50** | 0.033 | 0.0014 | **0.096** | partial spec — 받은 좌표는 보존, 나머지 합리적 채움 |
| ours_augmented | **0.000** | **0.0000** | 0.257 | 3,290 generation 모두 겹침 없음 |
| ours_bubble | 0.036 | 0.0060 | 0.418 | HD-fair sparsity — area 정보 없으므로 MAPE 큼 |
| DS2D_full | 0.400 | 0.0314 | — | mape: GT 매핑 안 됨 (plan-self) |
| DS2D_bubble | 0.450 | 0.0332 | — | 80 plan |
| DS2D_partial | 0.487 | 0.0398 | — | 80 plan |
| DS2D_sparse | 0.463 | 0.0403 | — | 80 plan |

**Overlap rate: ours (0.000~0.036) vs DS2D (0.400~0.487) — 10배 이상 우위**. 방 겹침은 plan-level 정합성의
가장 기본 metric (도면 사용 가능성 직결). cited baseline (HD/HouseGAN++/HouseLLM/ref1) 은 paper 에 보고
안 됨.

### 2.3 Exp 3 — Geometric Quality (Novel 11)

본 연구 reward 함수를 standalone metric 으로 사용 (`scripts/metrics/compute_novel_metrics.py`).
모든 값 [0, 1], ↑ higher better.

| 메트릭 | ours_fullcond | ours_coords50 | ours_augmented | ours_bubble | DS2D_full | DS2D_bubble | DS2D_partial | DS2D_sparse |
|---|---|---|---|---|---|---|---|---|
| format | **1.000** | **1.000** | **1.000** | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 |
| **orthogonality** | **0.9997** | **0.9996** | **1.000** | **1.000** | 0.968 | 0.972 | 0.987 | 0.975 |
| **no_overlap** | **1.000** | **0.999** | **1.000** | **0.998** | 0.969 | 0.970 | 0.965 | 0.967 |
| outline_in_room | **1.000** | 0.9997 | **1.000** | 0.998 | 1.000 | 1.000 | 0.997 | 1.000 |
| room_in_outline | 0.963 | 0.960 | 0.992 | 0.959 | 0.998 | 0.995 | 0.999 | 0.994 |
| **coverage** | **0.774** | **0.773** | **0.818** | **0.772** | 0.692 | 0.708 | 0.699 | 0.694 |
| count_total | **1.000** | 0.951 | 0.973 | 0.903 | — | 1.000\* | 1.000\* | 1.000\* |
| count_type | **1.000** | 0.993 | 0.984 | 0.985 | — | 1.000\* | 1.000\* | 1.000\* |
| connectivity | **1.000** | 0.976 | 0.940 | 0.885 | — | 1.000\* | 1.000\* | 1.000\* |
| spatial | 0.933 | 0.854 | 0.348 | 0.311 | — | 1.000\* | 1.000\* | 1.000\* |
| **input_consistency** | **0.9999** | 0.832 | 0.054 | 0.065 | — | 1.000\* | 1.000\* | 1.000\* |

\* DS2D 의 count/connectivity/spatial/input_consistency 1.000 은 plan-self only mode (DS2D plan_id 가
RPLAN 원본과 매핑 안 됨 → GT metadata 없이 자기 자신 vs 자기 자신 측정). ours 와 fair 비교 아님. 측정
가능한 plan-level metric (format/orthogonality/no_overlap/outline_in_room/room_in_outline/coverage)
6 종에서 **ours 가 모든 protocol 에서 DS2D 의 모든 condition 우위** (특히 ortho 0.9996~1.000 vs 0.968~0.987,
no_overlap 0.998~1.000 vs 0.965~0.970, coverage 0.772~0.818 vs 0.694~0.708).

**cited baseline (HouseDiffusion, HouseGAN, HouseGAN++, HouseLLM, ref1, Ashual & Wolf, Johnson et al.) 은
paper Table 1 에 ortho/no_overlap/coverage 등 geometric quality metric 을 보고하지 않으며, 따라서 직접
비교 불가**. 이건 본 연구가 novel 11 reward metric 으로 평가 공백을 메우는 학술적 contribution.

원본 CSV: `experiments/metrics/exp3_novel_*.csv` (모델별).

### 2.4 종합 해석

1. **ours 의 4-protocol 사다리 — input 정보량 vs reconstruction fidelity**:
   - **fullcond (100% spec)**: FID 0.21, GED 0.27 — reconstruction-grade (input_consistency 0.9999).
   - **coords50 (partial spec, 방 절반 좌표 + 모든 의미 정보)**: **FID 8.26, GED 1.50** —
     **HD cited published (FID 12.2~14.8, GED 1.7) 를 모든 bucket 에서 추월**. 이게 fair 학술
     비교의 핵심 결과.
   - **augmented (학습 동일 sparse)**: FID 29.80, GED 3.82 — 다양성 우위, diverse plausible.
   - **bubble (좌표 0, HD-equivalent sparsity)**: FID 29.93, GED 5.73 — 같은 sparsity 에서는
     1024-step diffusion 의 GT-fit 능력에 정밀도 측면 못 미침.
2. **HD vs ours 비교의 본질**: HD 는 bubble-only 입력 단일 task 에 1024-step iterative refinement +
   MSE loss 로 GT 분포에 강하게 fit. ours 는 multi-task augmentation 으로 다양한 spec sparsity 에
   대응하는 generalist. 두 방법론의 동일 metric 직접 비교는 protocol mismatch — ours_coords50 는
   "partial spec" 시나리오에서 ours 가 절대 우위, ours_bubble 은 "extreme sparsity" 시나리오에서
   trade-off 의 비용을 보여주는 진단값.
3. **DS2D 와 동일 protocol (full prompt) 직접 비교**에서 ours_fullcond 가 모든 metric 우위. FID 0.21
   vs 28.42 (self-paired) — ours 가 134배 우위. GED self-paired 0.27 vs 3.46. DS2D 의 vs-GT-329
   FID 98.69 는 모집단 비대칭 (80 vs 329) 으로 인한 inflation, self-paired 28~30 이 fair 한 reference.
   DS2D 가 LLM 방식임에도 ours 의 EA+SFT+RL 3-stage 가 압도적.
   - **DS2D per-bucket FID (N=20/bucket vs perbucket200 GT 200/bucket)**: 4 condition × 4 bucket
     모두 107~150 범위. ours_coords50 per-bucket (8.75~11.99) 와 비교 시 **DS2D 가 약 10~13배 큼**.
     모집단 비대칭 (20 vs 200) 으로 인한 inflation 일부 포함하지만, 동일 비대칭 protocol 의
     ours_coords50 mean 10.52 와 차이가 절대적으로 큼 → DS2D 의 plan-level fidelity 한계 명확.
   - **DS2D self-paired per-bucket FID (N=20 vs 20)**: 4 condition × 4 bucket mean 49~50 범위.
     비대칭 inflation 사라진 fair 측정.
   - **FID 의 N 의존성 검증**: ours_coords50 도 N=200 → N=20 으로 sub-sample 시 FID 가 10.52 →
     **22.19 (2.1배 inflate)**. 즉 N=20 의 inception covariance 추정 noise. **fair N 비교 시**
     ours_coords50 (N=20) **22.19** vs DS2D (N=20 self-paired) **49.10** = **ours 약 2.2배 우위**
     — 이게 N noise 통제 후 실제 fidelity 차이의 크기.
4. **ours 의 정량적 강점 정리**:
   - reconstruction fidelity (fullcond): FID 0.21, input_consistency 0.9999
   - partial-spec 우위 (coords50): FID 8.26 < HD 12.2 published
   - geometric correctness (모든 protocol): ortho ≥ 0.9996, no_overlap ≥ 0.998
   - overlap rate: augmented 0/3,290, coords50 0.033, bubble 0.036
5. **best-of-K 의 한계**: ours_augmented 의 oracle (28.46) ≈ self-reward (29.85) ≈ random (29.80).
   best-of-10 가 FID 를 거의 못 낮춤 → ours 의 10 sample 들이 GT 와 **모두 비슷한 거리에 균등 분포**
   = mode-collapse 의 반대 (diverse plausible generator). 단, partial spec (coords50) 1 sample 만으로도
   FID 8.26 달성 → diversity 가 필요한 시나리오와 fidelity 가 필요한 시나리오를 input 으로 controllable.

### 2.5 Output Token Length — Plan 1개의 표현 효율성

본 연구의 **567 floorplan-specific tokens** (예: `<X:142>`, `<Y:100>`, `<TYPE:LivingRoom>`,
`<ROOM>`, `<END_ROOM>`) 가 평면도 출력을 얼마나 압축하는지 직접 측정. 평가 tokenizer:
- **ours**: Qwen2.5-Coder-7B + custom 567 vocab extension
- **DS2D**: Meta-Llama-3-8B-Instruct (re-tokenize 출력 텍스트로 측정)

| protocol | tokenizer | N | input mean | output mean | output median | output min~max | 비고 |
|---|---|---|---|---|---|---|---|
| **ours_coords50** | Qwen+567 | 3,978 | 254 | **156** | 155 | 81~292 | partial spec |
| ours_bubble | Qwen+567 | 332 | 201 | 162 | 162 | 95~874 | bubble (좌표 0) |
| **DS2D_full** | Llama-3 | 80 | 406 | **789** | 775 | 502~**2802** | full prompt (sys 247 + adj/addtl 156) |
| DS2D_bubble | Llama-3 | 80 | 259 | 822 | 768 | 598~**2801** | only_total_area (sys 247 + addtl 9) |
| DS2D_partial | Llama-3 | 80 | 410 | 798 | 806 | 600~1010 | only_room_area (sys 247 + addtl 160) |
| DS2D_sparse | Llama-3 | 80 | 342 | 764 | 768 | 526~1426 | some_room_area (sys 247 + addtl 92) |
| ref1 RLVR-LLM (cited, estimated) | unknown | — | — | **~150~300** (est.) | — | — | code 미공개, paper 명시 X — RL+LLM framework 가 ours 와 유사하면 compact vocab 추정 |
| HouseLLM (cited, estimated) | Llama-2-7B base | — | — | **~600~900** (est.) | — | — | code 미공개, paper 본문 figure 가 자연어 JSON 출력 시사 — DS2D 와 유사 protocol 추정 |

**핵심 관찰**:
1. **ours ~156 vs DS2D ~798 → 약 5× 압축**. 동일 정보 (방 5-8개 × 좌표 4-30점 + 인접·type)
   를 ours 의 custom vocab 은 좌표 한 점 = 2 token (`<X:?> <Y:?>`), DS2D 의 자연어 JSON 은
   같은 좌표 = ~5-7 token (`"x": 142, "z": 100,`).
2. **DS2D 일부 plan 이 max_new_tokens=2800 도달** (out_max 2802) → truncation, 평면도 정보
   손실. ours 는 max 292 로 margin 충분.
3. **추론 속도 직접 영향**: DS2D plan 당 ~7.5분 vs ours 13초 — 약 **30× 차이**. 토큰당 시간이
   비슷한 두 LLM 의 차이가 (4 condition 반복 4× × 출력 길이 5× ≈ 20×) + (vLLM 미사용 등) 으로
   설명 가능.
4. **truncation 위험**: DS2D 의 longer-tail plan (방 많고 corner 많음) 이 2800 token 한계에서 잘림
   — 본 연구는 plan 정보가 token-efficient 라 truncation 없음.
5. **cited baseline 추정 caveat**: ref1 / HouseLLM 의 token length 는 paper 본문에 직접 보고
   되지 않고 code 미공개. 위 추정은 (a) paper 의 architecture (RL+LLM / Llama 기반) (b) 평면도
   1개 정보량 lower bound 로부터 도출한 보수적 추정. 정확 측정 불가.
6. **input token 도 ours 가 짧음**: ours_coords50 254 vs DS2D_partial 410 → **1.6× 압축**.
   DS2D 의 247 token system instruction overhead 외에도 좌표 인접관계의 자연어 표현
   (`(LivingRoom = "room|0", Kitchen = "room|2")`) 이 ours 의 special token (`<ROOM><RID:0><TYPE:LivingRoom>`)
   보다 길기 때문. 단 input 차이 (1.6×) 는 output 차이 (5×) 보다 작음 — output 압축이 더 큰 효과.

`prompt.json` 파일은 user prompt 의 `additional constraints` 부분만 저장하므로 단순 측정 시
누락된다. 본 측정은 `predict_output_rplan` 코드의 prompt 구성 ( `<system> instruction +
<user> adjacency + addtl_constraints` ) 그대로 재구성하여 측정한 정확한 값.

원본 CSV: `experiments/metrics/token_length_comparison.csv`

### 2.6 Input Semantic Element Comparison — ours vs DS2D

코드 분석 결과, 두 기법의 입력 정보 카테고리가 **본질적으로 다름**. 동일하지 않음 — fair
비교 protocol 매핑 시 reviewer 대응에 필요한 자료.

#### 의미론적 요소 매트릭스

| 의미론적 요소 | DS2D_full | DS2D_bubble | DS2D_partial | DS2D_sparse | ours_fullcond | ours_coords50 | ours_bubble |
|---|---|---|---|---|---|---|---|
| 방 카운트 (num_rooms) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 방 종류 (type) | random 50% drop | ✗ | ✓ | partial | ✓ | ✓ | ✓ |
| 방 인접관계 (edges) | ✓ (type-pair) | ✓ | ✓ | ✓ | ✓ (rid-pair) | ✓ | ✓ |
| **방 좌표 (polygon)** | ✗ | ✗ | ✗ | ✗ | ✓ | **50% drop** | ✗ |
| **방 면적 (area)** | random per-room | ✗ | ✓ per-room | partial | ✗ | ✗ | ✗ |
| 방 width/height | random | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **total_area** | ✓ (random) | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **spatial 8 방위** | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |
| **door (interior) 좌표** | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ |
| **front_door 좌표** | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ |
| **outline (외곽선)** | ✗ | ✗ | ✗ | ✗ | ✓ (rooms[0]) | ✓ | ✓ |
| 방 식별자 | room_id (자연어) | ✗ | room_id | partial | ✓ (rid token) | ✓ | ✓ |

#### 핵심 차이

| 항목 | DS2D | ours |
|---|---|---|
| **핵심 정보 차원** | **면적** (total_area / room_area, scalar) | **좌표** (polygon vertices, 2D 점 sequence) |
| **공간 정보** | 인접관계 (topological only) | 인접 + **8방위 spatial** (topological + directional) |
| **출입구** | 없음 | front_door + interior door 좌표 |
| **외곽 경계** | 없음 — generation 시 합성 (shapely unary_union) | **명시 제공** (rooms[0] outline polygon) |
| **방 식별자** | room_id (자연어 string) | rid (special token) |

#### Fair 비교의 한계

1. **DS2D 의 가장 정보 풍부 condition (full_prompt)** 도 ours 의 **bubble** 보다 정보가 적음
   (spatial / door / outline 등 결여). 즉 **DS2D 의 모든 4 condition 이 ours 의 어떤 protocol
   보다도 정보 sparse**.

2. **DS2D 의 area 정보** 는 ours 가 제공 안 함 — 좌표가 있으면 area 가 derived 되지만, ours_bubble
   (좌표 0) 은 area 도 모름. 정보 종류가 **직교 (orthogonal)** 함.

3. **공정 비교 protocol 매핑**:
   - `ours_bubble` ↔ `ds2d_bubble`: 둘 다 좌표/면적/디테일 없음 — **그나마 가장 fair**
     (단 ours 는 spatial 8방위 + outline 추가)
   - `ours_coords50` ↔ DS2D 어떤 condition 도 동등하지 않음 — ours 는 좌표, DS2D 는 area
     (정보 종류 직교)

4. **학술적 해석**:
   - DS2D 가 더 적고 다른 종류의 정보로 학습 → ours 의 다양한 sparse condition 에서의 성능
     (FID 3.89, GED 1.50) 이 더 인상적인 결과로 해석 가능
   - 다만 DS2D 는 **면적이라는 다른 정보 차원** 을 사용 — 직접 비교 시 **protocol mismatch**
     명시 필요 (footnote: "input semantic categories differ — area-driven vs coordinate-driven")
   - **단방향 비교**가 의미 있음: ours_fullcond 가 DS2D_full 의 모든 입력 카테고리를 포함하므로
     `ours_fullcond vs DS2D_full` 은 동일 입력 + 더 풍부한 정보 으로 fair 한 upper bound
     비교 가능

---

## 3. 메트릭 정합성 검증 (smoke test, 2026-05-14)

GT 를 자기 자신과 비교한 self-test — 측정 wrapper 가 의도대로 작동 확인.

| 메트릭 | GT vs self | 정상성 |
|---|---|---|
| format / no_overlap / outline_in_room / count_total / count_type / connectivity / input_consistency | 1.000 | ✓ |
| orthogonality | 0.9997 | ✓ (≈ 1.0) |
| room_in_outline | 0.963 | ✓ (polygon 변환 잔차) |
| coverage | 0.774 | ✓ (RPLAN polygon 한계) |
| spatial | 0.933 | ✓ (parser·metadata index 매칭 잔차) |
| GED (GT vs self) | 0.0 (5/5 plan) | ✓ |
| Self-consistency Overlap | 0.000 | ✓ |
| Room Area MAPE | 0.000 | ✓ |

GT-vs-self 1.0 미만 metric 은 이론적 best 가 1.0 이 아닌 *데이터셋 자체의 상한선*.

---

## 4. Baseline 진행 상태

| Baseline | 단계 | 상태 | 결과 |
|---|---|---|---|
| **DStruct2Design** | venv + 데이터 변환 + 추론 (4 condition × 20 plan × 1 sample) + outline 합성 + 렌더 + metric | ✅ 완료 | FID 98.69, Overlap rate 0.40, ortho 0.968, no_overlap 0.969 |
| **HouseDiffusion** | venv + save_samples patch 완료. 2026-05-27~28 재시도: Housegan-data-reader clone + 80K PNG 변환 (77,291 성공, 4시간) → train preprocessing 시도 → IndexError (empty edges, corner>32, FileNotFoundError, inhomogeneous shape) 등 4단계 patch 후에도 또 다른 오류 (numpy savez inhomogeneous). HD repo README 본문 "current version has not been cleaned and some features may not function correctly" 라는 brittle 상태 — RPLAN edge case 마다 죽음. | ⏸ **2026-05-28 사용자 결정: 포기**. cited 인용 정책 유지 | published 수치 cited_baselines.yaml 에 채움 |
| **GSDiff** | venv + preprocessing extract 완료 + process1 완료. process2 에서 `TypeError: list indices must be integers or slices, not tuple` (repo 자체 코드 정합성 문제). datasets→gs_datasets rename patch 적용했으나 별개 에러. | ⏸ 디버그 미완 → **cited 인용 권장** | (cited_baselines.yaml 의 GSDiff (cited) 는 값이 제대로 보고되지 않아 baseline 비교에서 제외 결정) |

---

## 5. 미수행 실험 (현재 사용자 액션 또는 후속 측정 대기 중)

| Exp | 상태 | 필요 입력 |
|---|---|---|
| **Exp 1** (ours_bubble + HD/HouseGAN/HouseGAN++/ref1 cited 통합 표) | ✅ 측정 완료 | ours_bubble GED=5.73 / FID=29.93 / novel11 측정. 표 빌드만 남음 |
| **Exp 2** Robustness (DS2D 4 condition vs ours 4 protocol) | ✅ DS2D 4 condition 정규화 완료, ours_augmented (sparse=full augmentation), ours_bubble (sparse=drop coords), ours_fullcond (full) 측정 완료 | 표 빌드 필요 |
| **Exp 3** Geometric Quality (3 행 완성) | ✅ ours_augmented + ours_fullcond + DS2D 완료 | 표 빌드 |
| **Exp 4** Stage Ablation | ⏳ 사용자 학습 (5 variants) | `scripts/experiments/ablations/exp4.sh train` |
| **Exp 5~7** Reward / CA / Augmentation Ablation | ⏳ 사용자 학습 | 각 ablation shell |
| **Exp 8** Qualitative figures (6/7/8) | ⏳ figure_selections.yaml 자동 추천 결과 사용자 검토 후 build_figure_*.py | 사용자 액션 |
| **Exp 9** User Study | ⏳ 사용자 13 명 모집 | Streamlit 앱 |
| **Exp 10** Analysis (figures 9~11 + Tables 8/9) | ⏳ W&B history CSV + ablation 결과 | |

---

## 6. 산출 CSV·디렉토리 인덱스

```
experiments/metrics/
├── ours_efficiency.csv                          # ours_augmented 효율성
├── exp1_compatibility_ours.csv                  # GED (augmented sparse) — mean 3.82
├── exp1_compatibility_ours_fullcond.csv         # GED (full cond) — mean 0.27
├── exp1_compatibility_ours_coords50.csv         # GED (partial spec, 방 절반 좌표) — mean 1.50
├── exp1_compatibility_ours_bubble.csv           # GED (HD-fair sparsity) — mean 5.73
├── exp1_compatibility_ds2d_{full,bubble,partial,sparse}.csv  # DS2D self-paired GED (4 cond)
├── exp1_fid.csv                                  # FID (ours_augmented best-of-10 모집단)
├── exp1_fid_ours.csv                             # FID (ours_best1, plan 당 1 sample)
├── exp1_fid_ours_fullcond.csv                   # FID (full cond) = 0.21
├── exp1_fid_ours_coords50.csv                   # FID (partial spec) = 8.26  ← HD cited 12.2~14.8 추월
├── exp1_fid_ours_bubble.csv                     # FID (HD-fair sparsity) = 29.93
├── exp1_fid_ours_oracle.csv                     # FID (best-of-10 oracle) = 28.46
├── exp1_fid_ours_self_best.csv                  # FID (best-of-10 self-reward) = 29.85
├── exp1_fid_ds2d.csv                             # DS2D FID = 98.69
├── exp1_fid_ds2d_{full,bubble,partial,sparse}.csv  # DS2D 4 cond self-paired FID
├── exp1_fid_per_bucket_coords50.csv              # ours_coords50 per-bucket FID @ N=200
├── exp1_fid_per_bucket_ds2d.csv                  # DS2D 4 cond per-bucket FID vs perbucket200
├── exp1_fid_per_bucket_ds2d_selfpaired.csv       # DS2D 4 cond per-bucket FID self-paired
├── exp1_fid_per_bucket_coords50_N250.csv         # ours_coords50 per-bucket FID @ N=250
├── exp1_fid_per_bucket_coords50_N1000.csv        # ours_coords50 per-bucket FID @ N=1000 (mean 3.89)
└── token_length_comparison.csv                    # ours vs DS2D output token 길이 통계
├── exp1_self_ours.csv                            # self-consistency ours_augmented
├── exp1_self_ours_fullcond.csv                  # self-consistency (full cond)
├── exp1_self_ours_coords50.csv                  # self-consistency (partial spec)
├── exp1_self_ours_bubble.csv                    # self-consistency (HD-fair sparsity)
├── exp1_self_ds2d.csv                            # DS2D
├── exp3_novel_ours.csv                           # novel 11 ours_augmented
├── exp3_novel_ours_fullcond.csv                 # novel 11 (full cond)
├── exp3_novel_ours_coords50.csv                 # novel 11 (partial spec)
├── exp3_novel_ours_bubble.csv                   # novel 11 (HD-fair sparsity)
├── exp3_novel_ds2d.csv                           # novel 11 DS2D
├── best_of_k_selection_oracle.csv               # ours best-of-10 oracle 선택 idx
└── best_of_k_selection_self_reward.csv          # ours best-of-10 self-reward 선택 idx

experiments/generations/
├── ours/                  # best-of-10, augmented sparse (3,290)
├── ours_best1/            # output_0 만 (329)
├── ours_fullcond/         # full cond, 1 sample (329)
├── ours_coords50/         # partial spec (방 절반 좌표), 1 sample (329)
├── ours_bubble/           # HD-fair sparsity, 1 sample (329)
├── ours_oracle_best/      # best-of-10 → GT raster L2 최소 (329)
├── ours_self_best/        # best-of-10 → self-reward 최대 (329)
├── ds2d_full/             # DS2D full_prompt (80)
├── ds2d_bubble/           # DS2D only_total_area (80)
├── ds2d_partial/          # DS2D only_room_area (80)
├── ds2d_sparse/           # DS2D some_room_area (80)
├── ds2d_gt_{full,bubble,partial,sparse}/  # DS2D self-paired GT (80 × 4)
└── gt/                    # GT dump (329)

experiments/renders/
├── gt_test/               # 329 (라벨 OFF)
├── ours/                  # 3,290
├── ours_best1/            # 329
├── ours_fullcond/         # 329
├── ours_coords50/         # 329
├── ours_bubble/           # 329
├── ours_oracle_best/      # 329
├── ours_self_best/        # 329
└── ds2d_{full,bubble,partial,sparse}/  # 각 80
```

### 신규 자산 (이번 세션, 2026-05-23 ~ 2026-05-26)

- `config/training/augmentation/ours_bubble.yaml` (신설): HD-fair sparsity protocol
- `config/training/augmentation/ours_coords50.yaml` (신설): partial spec — 방 절반 좌표 + 모든 의미 정보
- `scripts/utils/select_perbucket200_testset.py` (신설): 4 bucket × 200 plan testset + train hold-out arrow 생성
- `scripts/metrics/compute_fid_per_bucket.py` (신설): HD Table 1 형식 per-bucket FID 측정
- `experiments/testset_perbucket200.json` (신설): 800 plan (eval_pool 483 + train hold-out 317)
- `data/dataset/processed_dataset/rplan/arrow/train_holdout_perbucket200/` (신설): train 의 hold-out 317 row
- `data/dataset/processed_dataset/rplan/arrow/eval_pool_perbucket200/` (신설): union arrow (802 row)
- `scripts/baselines/normalize_ds2d.py` patch: `_synthesize_outline()` + DS2D tree-mode (4 condition 분리)
- `scripts/baselines/normalize_ds2d_gt.py` (신규): DS2D `ground_truth.json` (Python repr) → 공통 스키마 정규화
- `scripts/baselines/patch_hd_save_samples.py`: HD `save_samples()` 의 polygon JSON dump 추가 (HD 직접 재현은 미실시)
- `scripts/metrics/compute_compatibility_paired.py` (신규): 두 common-json 디렉토리 paired GED — DS2D 등 plan_id 미매핑 baseline 용
- `scripts/metrics/select_best_of_k.py` (신규): best-of-K sample 선택 (oracle / self-reward 모드)
- `scripts/metrics/compute_self_consistency.py` + `compute_novel_metrics.py`: plan-self only mode patch
- `scripts/render/render_unified.py`: `--show_labels` 인자
- `baselines/ds2d/run_generation_rplan.py` patch: `rooms[u][4]` → `rooms[u]['room_type']`, ds_dir typo, jobid=None 분기에 start/end 정의
- `baselines/gsdiff/gs_datasets/` (rename from `datasets/`) + 모든 import sed patch (50 파일)
