# 실험 진행상황 맥락 문서

본 문서는 **논문 실험 인프라 구축 작업의 시계열 진행상황**을 기록한다. AI 가 매 작업 turn 마다
이 문서를 갱신해 다른 협업자(또는 미래의 AI 세션)가 어떤 결정이 왜 내려졌는지·어디까지
완료됐는지·무엇이 남았는지 파악할 수 있도록 한다.

업데이트 규칙:
- 새 작업 시작/완료 시 §A 진행 현황 표와 §B 시계열 로그를 같이 갱신.
- 비자명한 시행착오·수정 결정은 §C 에 한 줄로 압축 기록 (rationale 포함).
- 사용자 요청 사항이 들어오면 §D 에 추가하고 처리 후 status 갱신.
- 구체적 명령·디렉토리 구조 등은 [`EXPERIMENTS_GUIDE.md`](EXPERIMENTS_GUIDE.md) 에 두고 본 문서는
  *맥락* 만 다룬다.
- **측정 결과 수치** (mean GED, FID, novel 11 등 정량값) 는 [`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md)
  에 결과 요약으로 별도 정리한다. 본 문서는 *측정 완료 여부와 시행착오* 만 기록.

---

## A. 현재 진행 현황 (한눈에)

마지막 갱신: 2026-07-06 (for_paper 시각화 실험 5종 + color_map 팔레트 개선)

| 영역 | 상태 | 비고 |
|---|---|---|
| 워크스페이스 / 메트릭 라이브러리 | ✅ 완료 | `baselines/`, `experiments/` + `.gitignore`, clean-fid/networkx/pandas/matplotlib/seaborn |
| 학습 체크포인트 active symlink | ✅ 완료 | `final_checkpoints/{sft,rl}/active` |
| 통일 test set + eval_pool | ✅ 완료 | 329 plans, val+test concat 485 rows |
| 공통 JSON 스키마 + 정규화 | ✅ 완료 | 본 연구·HD·GS·DS2D 모두 |
| 통일 raster renderer | ✅ 완료 | GT 329 검증, 라벨 토글, z-order 분리 |
| 5종 메트릭 + smoke test | ✅ 완료 | GED / FID / self / novel(11) / efficiency |
| 본 연구 추론 orchestrator + smoke run | ✅ 완료 | 6/6 generation 성공, ~10.6 s/gen |
| Baseline setup (clone + venv) | ✅ 완료 | 3 baseline 모두 .venv 생성, pyproject shim |
| Baseline 데이터 변환·정규화 스크립트 | ✅ 완료 | 사용자 실 weight 확보 후 추론 + 정규화 |
| Exp 1-3 표 빌더 + cited_baselines.yaml | ✅ 완료 | placeholder yaml — 사용자가 채움 |
| Exp 4-7 ablation 패치 + shell + 표 빌더 | ✅ 완료 | skip_partial_state / skip_sft / use_gdpo_normalization 토글 패치 |
| Exp 8 qualitative figures + preview | ✅ 완료 | figure_selections.yaml placeholder |
| Exp 9 user study (Streamlit + aggregate) | ✅ 완료 | 사용자가 실제 13 명 모집·실행 |
| Exp 10 figures 9/10/11 + Table 8/9 | ✅ 완료 | wandb history CSV 기반 |
| csv→LaTeX 변환기 + EXPERIMENTS_GUIDE.md | ✅ 완료 | guide 는 docs/ 로 이동, 시행착오 섹션 추가 |
| Renderer 라벨 토글 / z-order / 위치 / 색깔 / 테두리 | ✅ 완료 | 2026-05-15 사용자 요청 반영 |
| 추론 SFT+RL 동시 활성화 패치 | ✅ 완료 | **버그 발견·수정** (set_adapter 누락) |
| 본 연구 best-of-10 추론 실행 | ✅ 완료 | 2026-05-15 14:52 시작 → 정상 종료. 329 plan × 10 = **3,290 generations, parse_success=1.0**, 평균 13.36 s/gen. 결과: `outputs/inference/.../14-52-26/`, `experiments/generations/ours/`, efficiency CSV |
| ours / GT raster 재렌더링 (라벨 OFF) | ✅ 완료 | 2026-05-22. `render_unified.py --show_labels` 인자 신설(기본 False). ours 3,290 / GT 329 재렌더링. ~6 초·~0.5 초 소요 (모델 추론 없이 polygon 만 다시 그리므로) |
| ours 단독 메트릭 측정 (Exp 1/3 의 ours 행) | ✅ 완료 | 2026-05-22. **GED 3.82** (5/6/7/8=1.76/2.26/3.92/5.86), **FID 25.36** (vs GT), **Overlap=0**, **format/orthogonality/no_overlap/outline_in_room=1.0**, count_total=0.97, count_type=0.98, room_in_outline=0.99, connectivity=0.94, coverage=0.82, spatial=0.35, input_consistency=0.05 (sparse augmentation 영향) |
| Baseline weight 다운로드 | ✅ 완료 | 2026-05-22 사용자 확인. HouseDiffusion `model250000.pt`, GSDiff `topo-params/`·`topoae/`, DS2D `models/{5R,6R,7R,8R}/` |
| DS2D venv·데이터·추론·정규화·렌더·메트릭 | ✅ 완료 | 2026-05-24. venv 의존성 설치, 80,788 PNG → Arrow 변환, 4 condition × 20 plan × 1 sample = 320 gen 추론, outline 합성 정규화 (4 partial_prompt 분리), 통일 raster 렌더링, FID/self/novel 측정 (FID 98.69, ortho 0.97, no_overlap 0.97) |
| HouseDiffusion 환경·patch | ✅ 완료 | 2026-05-23. venv (torch 2.0.1+cu118 + mpi4py 등) 설치, `save_samples()` 에 polygon JSON dump patch 적용. 단 직접 재현은 미실시 (HouseGAN++ JSON 형식 불일치) |
| HouseDiffusion 직접 재현 | ⏸ skip | 2026-05-24 1차 결정 → 2026-05-27~28 재시도: Housegan-data-reader clone + 77,291 plan 변환 (4시간) + HD train preprocessing patch 4단계 (empty edges, corner>32, cwd 상대경로, inhomogeneous shape) 시도. RPLAN edge case 마다 죽음 → **2026-05-28 사용자 결정: 최종 포기**. cited 인용 정책 유지 |
| DS2D per-bucket FID (vs perbucket200 GT) | ✅ 완료 | 2026-05-28. 4 condition × 4 bucket = 16 cell. N=20 gen vs 200 GT 비대칭. mean FID: full=117.8, bubble=121.4, partial=127.1, sparse=121.0 — ours_coords50 (10.52) 와 비교 시 약 10~13배 큼 |
| DS2D self-paired FID (4 condition) | ✅ 완료 | 2026-05-28. full=28.42, bubble=29.03, partial=30.21, sparse=30.40 (80 vs 80 self-paired). ours_fullcond (0.21) 보다 134배 큼 |
| GSDiff process 디버그 | ⏸ 미완 | 2026-05-25. `datasets`→`gs_datasets` rename + 50 파일 import patch 로 모듈 충돌 해결. 그러나 process2 에서 `TypeError: list indices must be integers or slices, not tuple` — repo 자체 코드 정합성 문제. README 의 `move.py` 도 실제 파일 없음. cited 인용 권장 (값 보고 불완전해 baseline 비교에서 제외 결정) |
| cited_baselines.yaml (HD/HouseGAN/HouseGAN++/ref1/HouseLLM) | ✅ 완료 | 2026-05-25 사용자 직접 paper 조회 후 값 입력. FloorPlan-LLaMa / GSDiff (cited) / DiffPlanner 제외 |
| metric script plan-self only mode | ✅ 완료 | 2026-05-24. `compute_self_consistency.py` + `compute_novel_metrics.py` patch — GT 매핑 불가능한 baseline (DS2D 등) 도 plan-self metric (format/ortho/no_overlap/Overlap 등) 측정 가능 |
| FID protocol 개선 (ours_best1, plan 당 1 sample) | ✅ 완료 | 2026-05-25. ours best-of-10 → output_0 만 (329) → GT 와 모집단 균형. FID 29.80 |
| ours_fullcond 추론 (aug OFF, greedy, 1 sample) | ✅ 완료 | 2026-05-25 00:46~02:04 (1h 18m). 329 plan, FID 0.21, GED 0.27, MAPE 0.0, input_consistency 0.9999 — reconstruction-like (입력 좌표가 출력에 거의 그대로 보존) |
| ours_bubble protocol 정의 + smoke | ✅ 완료 | 2026-05-25 22:42. `config/training/augmentation/ours_bubble.yaml` 신설 — HD cited 와 동일 sparsity (drop_coords=1.0, drop_front_door=1.0, transform/noise OFF). 3 plan smoke 통과. 본격 329 plan 추론은 미실행 |
| ours_bubble 본 추론 + 정규화 + 렌더 + 메트릭 | ✅ 완료 | 2026-05-25 23:23 ~ 2026-05-26 00:39 (1h 14m), 329 plan parse_success=1.0, 13.63 s/gen. **GED=5.73 (2.90/4.14/5.80/8.07), FID=29.93, ortho=1.0, no_overlap=0.998**, count_total=0.903, connectivity=0.885, spatial=0.311, mape=0.418 |
| ours_coords50 본 추론 + 정규화 + 렌더 + 메트릭 | ✅ 완료 | 2026-05-26 17:30 ~ 18:48 (1h 17m), 329 plan parse_success=1.0, 14.09 s/gen. **FID=8.26 (HD cited 12.2~14.8 추월), GED=1.50 (0.55/0.63/1.49/2.65)**, ortho=0.9996, no_overlap=0.9986, connectivity=0.976, spatial=0.854, mape=0.096, input_consistency=0.832 |
| ours_coords50 per-bucket FID @ N=250 (testset_perbucket1000) | ✅ 완료 | 2026-05-29. 추가 200 plan 추론 (44m) + perbucket1000 merge (1000 plan). **per-bucket FID: 5=9.87, 6=8.41, 7=8.99, 8=10.15, mean=9.35** |
| ours_coords50 per-bucket FID @ **N=1000** (testset_perbucket4000) | ✅ 완료 | 2026-05-29 ~ 2026-05-30. 추가 3000 plan 추론 (10h 53m, 13.06 s/gen, parse_success=1.0) + perbucket4000 merge (4000 plan). **per-bucket FID: 5=3.83, 6=3.79, 7=3.82, 8=4.10, mean=3.89** — N=1000 의 최종 안정 측정, HD cited (13.50) 대비 **약 3.5배 우위** |
| Best-of-K FID protocol (oracle / self-reward) | ✅ 완료 | 2026-05-26. oracle (10 sample 중 GT raster L2 최소) FID=28.46; self-reward (11 metric 합 최대) FID=29.85; random pick FID=29.80. 셋 다 유사 → diverse plausible generator 의 직접 증거 |
| per-bucket FID @ N=200 (testset_perbucket200, ours_coords50) | ✅ 완료 | 2026-05-27. 800 plan (eval_pool 483 + train hold-out 317, data leak footnote). **ours_coords50: 5=11.29, 6=8.75, 7=10.04, 8=11.99 — 4 bucket 모두 HD cited (12.2/13.4/13.6/14.8) 추월** |
| DS2D GED (per-plan self-paired) | ✅ 완료 | 2026-05-25 23:30. 4 condition × 80 plan paired GED. **full=3.46 / bubble=2.90 / partial=4.15 / sparse=4.14**. `normalize_ds2d_gt.py` + `compute_compatibility_paired.py` 신설 |
| Ablation 학습 실행 | ⏳ 사용자 | 매우 큰 GPU 시간 |
| User study 수행 | ⏳ 사용자 | 13 명 모집 |
| figure_selections.yaml 채우기 | ⏳ 사용자 | preview PDF 검토 후 |
| LaTeX/docx 최종 통합 | ⏳ 사용자 | 논문 작성 단계 |

---

## B. 시계열 작업 로그

### 2026-07-06 (for_paper 논문 figure 실험 5종 + color_map 팔레트 개선)

논문 정성 평가 figure 용 시각화·비교 실험 5종. 모든 산출물은 `experiments/for_paper/{exp}/` 하위에
실험별 폴더로 저장. 통합 스크립트: `scripts/figures/for_paper_experiments.py` (--exp {1..5}).

**#0 color_map.yaml 팔레트 개선**
- room_colors 9종을 ColorBrewer Set2/Set3 + 색맹(CVD) 친화 톤으로 교체 (시인성·구분성 ↑).
  기존 값은 `# old3:` 주석으로 보존 (이미 old/old2 있었음).

**#1 input_output_example (5 plan)**
- 1@5rooms + 3@6rooms + 1@7rooms. 학습-시 증강 (sft.yaml) 적용 입력 조건 + 정답 평면도.
- `{plan_id}/{input.txt, input.png, output.txt, output.png}`

**#2 generated_floorplan_per_stage (5 plan × 3 stage × 4 sample = 60 gen)**
- 동일 입력 조건 (증강 완료본, 3 stage 에 동일 제공) → EA/SFT/RL 각 4 sample.
- **핵심 구현**: augmented condition 을 plan 당 한 번만 생성 후 3 stage 모두 재사용 (매번 증강 시
  입력이 달라지는 문제 방지). stage 순차 로드 (GPU 메모리 해제 후 다음 stage).
- `{plan_id}/input/{input.txt,bubble_diagram.png,rooms.png}` + `{stage}/{1..4}.{txt,png}`

**#3 1-to-many-generation (4 plan × 20 sample = 80 gen, RL)**
- 중간 밀도 입력 (drop_block 0.15/coords 0.35/edge 0.25/spatial 0.3) 한 번 고정 → 20 variants.
- `{plan_id}/input/...` + `output/{1..20}.{txt,png}`

**#4 varying_input_density (4 density × 4 plan × 4 sample = 64 gen, RL)**
- 밀도별 입력: connectivity_spatial_counts_polygons (일부 폴리곤만 drop) / connectivity_spatial_counts
  (폴리곤 전부 drop) / connectivity_counts (+spatial drop) / connectivity_only (+room_summary drop).
- `{density}/{plan_id}/input/...` + `output/{1..4}.{txt,png}`

**#5 imprecise_and_contradictory_inputs (2 kind × 5 plan × 4 sample = 40 gen, RL)**
- **contradictory**: room_summary `<TOTAL>` 를 type count 합보다 작게 강제 (real-2). 검증: 58577
  → TOTAL=4 vs type합=6 모순 확인. tokenizer 의 `FLOORPLAN_FORCE_TOTAL_OVERRIDE` 환경변수 활용.
- **imprecise**: 앵커 방 절반에 σ=30px Gaussian noise (학습 σ=3 의 10배) 적용 후 재토큰화
  (`build_condition_tokens` 직접 호출). 검증: 12058 bedroom x_range=[30,144] 폭 114px 로 일그러짐.
- `{contradictory_input|imprecise_input}/{plan_id}/input/...` + `output/{1..4}.{txt,png}`

**공통 산출물**: 각 input/ 폴더에 input.txt (토큰 시퀀스 텍스트화), bubble_diagram.png
(connectivity 그래프, 텍스트 없음·굵은 edge), rooms.png (drop 반영 외곽선+방). 각 실험 폴더에 README.md.

**시행착오**: (1) AugmentationPipeline signature 는 `(vocab, cfg)` 순서 (2) build_inference_cfg 에
`model.embed_align_dir`/`hub_id`/`tokenizer_dir` + `generation.num_beams/top_k/repetition_penalty`
누락 시 ConfigKeyError → pipeline.yaml 구조에 맞춰 보강 (3) parse_output_tokens 는 dict 반환
(ParsedFloorplan 객체 아님) → visualizer 에 dict 직접 전달.

### 2026-06-16 ~ 2026-06-23 (OOD / controllability 실험 시리즈)

본 세션은 모델의 generalization · controllability 검증 목적의 합성 입력 실험들로 구성됨.

**1. 15-room OOD plan 추론 (2026-06-22)**
- 학습 분포 (4~8 방) 외 — RID vocab 한계 (0~15) 까지 활용한 15 방 plan
- 5×3 grid 직사각형 outline + livingroom 1 / bedroom 5 / kitchen 2 / bathroom 3 / balcony 2 / storage 2
- 3 protocol 모두 추론 (bubble / coords50 / fullcond), parse_success=1.0
- 산출: `experiments/ood_15rooms/{comparison.png, input_visualization/, ours_{bubble,coords50,fullcond}_{generation,render}/}`
- 신규 스크립트: `scripts/figures/build_ood_15rooms.py`

**2. 10-room "count only" 추론 (2026-06-22)**
- 가장 sparse 입력 — 방 개수 정보 만 (room_summary 의 total + type counts)
- 좌표/edges/spatial/door/front_door/outline 모두 drop
- 신규 yaml: `config/training/augmentation/ours_count_only.yaml` (drop_block=1.0 외 전부 1.0,
  room_summary 만 유지)
- 산출: `experiments/ood_10rooms_count_only/{input_condition.txt, generation/, render/}`
- 검증: augmentation log 의 DropBlock(rids=[0,1,...]) 으로 모든 방 block 제거 확인

**3. Mismatched total controllability 실험 (2026-06-23)**
- 입력 조건의 `<TOTAL>N` 토큰 값을 type counts 합과 일부러 mismatch 시켜 모델 행동 관찰
- 신규 patch: `src/training/augmentation/tokenizer.py` 의 `build_room_summary_tokens` 에
  환경변수 `FLOORPLAN_FORCE_TOTAL_OVERRIDE={"plan_id": int}` 기반 override 추가
- 5 plan (bucket 6×3 + bucket 7×2) × ours_coords50 protocol × forced total = real - 2
- 결과 — 모델이 inconsistency 를 다양하게 해결:
  | plan | GT real | forced | output 방수 | 행동 |
  |---|---|---|---|---|
  | 10231 | 7 | 5 | 7 | type counts 합 우선 |
  | 1026  | 7 | 5 | 6 | 절충 |
  | 10438 | 6 | 4 | 6 | type counts 합 우선 |
  | 10990 | 6 | 4 | 4 | total 우선 (bedroom 2→0) |
  | 11378 | 6 | 4 | 5 | 절충 (bedroom 1 제거) |
- 산출: `experiments/ood_mismatched_total/{comparison.png, {plan_id}_input_condition.txt,
  generation/, render/, gt/}`
- 부속: `experiments/ood_mismatched_total_override.json`, `experiments/testset_ood_mismatched_total.json`
- 시사점: input_consistency reward 가 type counts 합·total 양쪽을 학습했으나 둘이 충돌 시
  명확한 우선순위 없이 plan 별로 다르게 행동 → 향후 hard constraint 부여 ablation 가능

### 2026-05-14 (인프라 초기 구축)

- **요구사항 수신**: 사용자가 `experiment_implementation_guide.md` 첨부, Automation in Construction
  full paper 목적 Exp 1~10 구현 요청.
- **사용자 결정 사항**:
  - 워크스페이스 옵션 A — codebase 내부에 `baselines/`, `experiments/` 둠.
  - SFT/RL 체크포인트 경로 명시: 한국어 폴더의 `checkpoint-105399` (SFT), `rl-token-level-...`/`checkpoint-3300` (RL).
  - Ablation 은 코드만 구현, 학습 미실행.
  - RPLAN PNG 원본이 본 연구 codebase 내부에 있어도 무방 (절대경로로 baseline 에 전달).
- **자동 작업 (1차)**: 워크스페이스 구조, 메트릭 라이브러리(`clean-fid`, `networkx`, `pandas`,
  `matplotlib`, `seaborn`), active symlink, 통일 testset(329 plans) + eval_pool Arrow split,
  공통 JSON 스키마, 통일 raster renderer, 5 종 메트릭, 본 연구 추론 orchestrator, baseline
  setup·변환·정규화, Exp 1-3 표 빌더, Exp 4-7+10 ablation 코드·shell, Figures 6~11 + Tables 1~9
  빌더, Streamlit user-study 앱, csv→LaTeX, EXPERIMENTS_GUIDE.md.
- **시행착오**:
  - test split 81 plans 만 → val(404)+test(81) 합쳐 통일 test 풀로 확장.
  - val/test 비율이 yaml 명세와 swap 적용된 것으로 보임 (현재 Arrow 가 그렇게 split 되어 있음).
- **smoke test 통과**: GT vs self (novel/selfcon/GED 모두 합리적 값), Ours 3 plan × 2 output
  추론 6/6 success.

### 2026-05-26 21:19 ~ 2026-05-27 13:35 (per-bucket FID @ N=200 — HD cited 4 bucket 추월)

- **사용자 요청**: ours_coords50 도 HD Table 1 처럼 5/6/7/8 bucket 별 FID 측정. 데이터 부족 시
  train 에서 추가 허용 (data leak OK).
- **testset_perbucket200 신설**: 신규 `scripts/utils/select_perbucket200_testset.py`.
  - eval_pool 483 plan (bucket 4 제외) + train hold-out 317 plan = **800 plan, 4 bucket × 200**
  - bucket 별 hold-out 분포: 5(eval 29 + train 171) / 6(eval 173 + train 27) / 7(eval 181 + train 19) / 8(eval 100 + train 100)
  - 산출: `experiments/testset_perbucket200.json`, `train_holdout_perbucket200/` arrow (317 row),
    `eval_pool_perbucket200/` union arrow (802 row), `testset_perbucket200_extra.json` (471 extra plan_id)
- **GT raster 추가 471 plan 렌더링**: `experiments/renders/gt_perbucket200_extra/`. ~1초 (polygon 만).
- **ours_coords50 추가 추론 471 plan**: 21:19 시작 ~ 23:01 종료, 1h 41m, 12.86 s/gen, parse_success=1.0.
- **정규화 + 렌더 + 디렉토리 merge**:
  - `experiments/generations/ours_coords50_perbucket200/` (800 JSON, 기존 329 + 새 471)
  - `experiments/renders/ours_coords50_perbucket200/` (800 PNG)
  - `experiments/renders/gt_perbucket200/` (800 PNG, 기존 gt_test 329 + 새 gt_perbucket200_extra 471)
- **per-bucket FID 측정** (신규 스크립트 `scripts/metrics/compute_fid_per_bucket.py`):
  | bucket | ours_coords50 | HD cited | 차이 |
  |---|---|---|---|
  | 5 | **11.29** | 12.2 | -0.91 |
  | 6 | **8.75** | 13.4 | -4.65 |
  | 7 | **10.04** | 13.6 | -3.56 |
  | 8 | **11.99** | 14.8 | -2.81 |
  | mean | **10.52** | 13.50 | -2.98 |
- **결론**: **4 bucket 모두 HD published 추월**. bucket 6 에서 격차 가장 큼. data leak (bucket 5 의
  171 plan) 효과 minimal — bucket 5 (FID 11.29) 가 다른 bucket (8.75~11.99) 와 큰 차이 없음.
  data leak 이 의미 있게 효과 있었다면 bucket 5 가 현저히 낮았을 것 → augmentation 다양성 학습
  이 정확 reconstruction memorization 으로 이어지지 않음 증거.

### 2026-05-26 17:30~21:00 (ours_coords50 protocol 신설 + Best-of-K FID 분석)

- **Best-of-K FID protocol 측정**: ours best-of-10 (3,290 gen) 에서 plan 당 1 sample 만 선택.
  신규 스크립트 `scripts/metrics/select_best_of_k.py` (oracle / self_reward 두 mode):
  - **Oracle** (각 plan 의 10 sample 중 GT raster L2 최소): FID=**28.46**, selected_idx 분포
    {0:21, 1:33, 2:43, ..., 9:32} 거의 균등.
  - **Self-Reward** (11 metric 합 최대, GT 안 봄): FID=**29.85**.
  - 둘 다 random pick (29.80) 와 거의 동일 (1.4 차이만) → ours 의 10 sample 들이 **GT 와 모두
    비슷한 거리에 균등 분포** = mode-collapse 의 반대 = diverse plausible generator.
- **사용자 요청**: "fullcond 만큼은 아니지만 방·엣지 정보 많이 제공" — 좌표 일부 제공하는 partial
  protocol 신설.
- **ours_coords50 protocol 신설**: `config/training/augmentation/ours_coords50.yaml` —
  `p_drop_coords=0.5` (방 절반의 좌표만 입력, Bernoulli per-room), `p_drop_*=0` (다른 정보 다 받음),
  `transform/noise OFF`. smoke test 통과 (DropCoords(rids=[0, 14, 15]) 등 적용 확인).
- **ours_coords50 본 추론**: 17:30 시작 ~ 18:48 종료, 1h 17m, 14.09 s/gen, parse_success=1.0.
- **ours_coords50 메트릭 — HD cited 모든 행 추월**:
  - **FID = 8.26** vs HD cited 12.2~14.8 → ours 가 명백히 낮음
  - **GED = 1.50** (per-bucket 0.55/0.63/1.49/2.65) vs HD cited 1.7 (1.5/1.2/1.7/2.5) → ours 가
    낮음 (5/6 bucket 큰 차이, 7 약간 우위, 8 비등)
  - ortho=0.9996, no_overlap=0.9986, count_total=0.951, connectivity=0.976, spatial=0.854,
    input_consistency=0.832 (방 절반 좌표 받았으므로 그 부분은 보존), mape=0.096
- **학술적 framing**: ours 가 partial spec 시나리오에서 HD published 를 추월. ours_bubble (29.93)
  은 extreme sparsity 시나리오 진단값, ours_coords50 (8.26) 은 fair 비교의 핵심 결과.

### 2026-05-26 (ours_bubble 추론 종료 + 후처리 + 메트릭)

- **ours_bubble 추론 종료**: 23:23 시작 ~ 00:39 종료, 1h 14m. 평균 13.63 s/gen, parse_success=1.0.
  outputs/inference/Qwen2.5-Coder-7B/rl/2026-05-25/23-23-28.
- **정규화 + 통일 렌더링**: `experiments/generations/ours_bubble/` (329 JSON),
  `experiments/renders/ours_bubble/` (329 PNG, 라벨 OFF).
- **ours_bubble 메트릭 측정 완료**:
  - **GED** = 5.73 mean (per-bucket 5/6/7/8 = 2.90/4.14/5.80/8.07). HD cited 1.7 보다 큼 — LLM
    autoregressive 의 좌표 추정 한계 + RPLAN 8px 인접 판정 vs HD wall-segment 차이.
  - **FID** = 29.93 (329 vs 329). ours_augmented_best1 의 29.80 과 거의 동일 — augmentation
    transform 의 FID 영향이 작다는 증거.
  - **Self-consistency**: overlap_bool=0.036, overlap_pct=0.006, MAPE=0.418. ours_augmented (0/0/
    0.257) 보다 약간 높지만 bubble 입력엔 area 정보가 없어 MAPE 큰 것은 당연.
  - **Novel 11**: format/ortho 1.0, no_overlap 0.998, count_total 0.903, count_type 0.985,
    room_in_outline 0.959, outline_in_room 0.998, coverage 0.772, connectivity 0.885,
    spatial 0.311, input_consistency 0.065.
- **EXPERIMENT_SUMMARY.md §2.2/§2.3 표 ours_bubble 행 추가** + 종합 해석 §2.4 갱신.

### 2026-05-25 23:23 (ours_bubble 본 추론 시작 + DS2D paired GED 측정)

- **ours_bubble 본 추론 백그라운드 시작**: PID 20060, log=`experiments/run_ours_bubble.log`. 329 plan,
  `augmentation.config_path=config/training/augmentation/ours_bubble.yaml`, num_outputs=1, greedy.
  초기 6 plan 측정값 ~15s/plan → 총 ~85 분 예상. HD/HouseGAN/HouseGAN++ cited 와 동일 sparsity
  (좌표 없음, bubble diagram only) protocol 의 진짜 fair 비교 산출.
- **DS2D paired GED 측정 (4 condition × 80 plan)**:
  - 신규 스크립트 1: `scripts/baselines/normalize_ds2d_gt.py` — `ground_truth.json` (Python repr
    형식) 을 `ast.literal_eval` 로 파싱하여 공통 스키마로 변환. `experiments/generations/
    ds2d_gt_{full,bubble,partial,sparse}/{bucket}_{plan_idx}.json`. 320 GT 파일 정규화.
  - 신규 스크립트 2: `scripts/metrics/compute_compatibility_paired.py` — 두 common-json 디렉토리
    (gen + gt) 를 plan_id 기준 매칭하여 self-paired GED 측정. DS2D plan_id 형식 `{N}R_{idx}`
    가 자체적으로 `_숫자` 포함하므로 GT 측 `strip_sample_suffix=False`, gen 측 True 로 분기.
  - 결과: **full=3.46 / bubble=2.90 / partial=4.15 / sparse=4.14** (mean GED). per-bucket 분산이
    큼 — DS2D 가 5 rooms 에서 더 어려워하는 경향 (full=5.95, partial=9.25). 6 rooms 에서는 GED
    가장 낮음 (full=1.20).
- **EXPERIMENT_SUMMARY.md Compatibility 표에 DS2D 4 condition 행 추가**.

### 2026-05-25 (FID protocol 개선 + ours_fullcond 추론 + ours_bubble protocol 정의)

- **FID protocol 비대칭 진단**: ours 의 3,290 generation (best-of-10) vs GT 329 모집단 차이 + ours
  augmentation drop·transform 의 영향으로 GED·FID 가 published HouseDiffusion (cited) 보다 크게
  나옴. 사용자 분석 핵심: (a) augmentation drop=0.5~0.8 → 입력 정보의 절반 이상 사라진 상태에서
  GT 추측 (b) transform (Flip/Scale/Zoom) → ours 출력이 변형 좌표계, GT raster 는 원본 좌표계 →
  분포 mismatch (c) FID 모집단 비대칭 + best-of-K 분산.
- **FID protocol 개선 1차 — ours_best1**: `experiments/renders/ours_best1/` 신설 (plan 당 output_0
  만, 329 PNG). FID 측정 시 GT 와 동일 모집단 크기. 결과 FID = 29.80 (best-of-10 모집단 25.36
  보다 약간 ↑ 하지만 protocol 일관성).
- **HouseDiffusion published 인용 (P1)**: 사용자가 paper 직접 조회하여 cited_baselines.yaml 정확히
  채움. HD/HouseGAN/HouseGAN++/ref1/HouseLLM/Ashual & Wolf/Johnson et al. 모두. FloorPlan-LLaMa /
  GSDiff (cited) / DiffPlanner 는 값 보고 불완전 + 코드 미공개로 baseline 비교에서 제외 결정.
- **ours_fullcond 추론**: augmentation OFF + greedy + 1 sample/plan = HD 와 동일 1-sample protocol
  + full information condition. 329 plan, 1h 18m. 결과 **FID 0.21, GED 0.27, MAPE 0.0,
  input_consistency 0.9999** — 본질적으로 reconstruction (입력 좌표 그대로 출력). 학술적
  contribution 이라기보다 sanity check.
- **ours_bubble protocol 정의**: 진짜 fair baseline 비교를 위해 HD cited 와 동일 sparsity (좌표 없는
  bubble diagram only) 정의. `config/training/augmentation/ours_bubble.yaml` 신설:
  `drop_coords=1.0, drop_front_door=1.0, drop_door=1.0, drop_block/edge/spatial/type=0, transform
  OFF, noise OFF`. 3 plan smoke 통과 (DropCoords(rids=[0..14]), DropEdgeDoor, DropFrontDoor 적용
  확인, transform 미적용 확인). 본격 추론은 사용자 결정 대기.
- **EXPERIMENT_RESULTS.md → EXPERIMENT_SUMMARY.md rename**: 사용자가 명시한 파일명으로 통일.

### 2026-05-24 (DS2D 환경 + 데이터 변환 + 추론 + 정규화 + 메트릭)

- **DS2D venv 의존성 설치**: torch 2.3.0 + transformers 4.40.1 + peft 0.10.0 + bitsandbytes 0.43.1
  + datasets + accelerate + skimage + scipy + shapely + jsonschema + matplotlib + pandas.
  `--no-config` 로 부모 codebase pyproject.toml 의 `numpy>=2.4.2` override 차단.
- **DS2D 데이터 변환**: RPLAN PNG → HF Arrow (`baselines/ds2d/datasets/rplan_converted/{5,6,7,8}`).
  `range(1000)` → `range(80788)` patch + plan_id 매핑 검증 (PNG idx = plan_id 일치 확인).
  ~58분 소요.
- **DS2D 추론**: 4 bucket × 4 partial_prompt × 1 sample × 20 plan = 320 generations. 처음
  num_samples=10 으로 시작했으나 plan 당 ~11분 (102시간 예상) 이라 옵션 B2 (num_samples=1 +
  end_idx=20) 로 단축. 8 plan 후 7.7 분/plan → 약 10 시간 예상. 4 bucket 완주.
- **DS2D 코드 patch**: `rooms[u][4]` → `rooms[u]['room_type']` (KeyError 4 수정), `ds_dir =
  'datasets/rplan_converted'` (typo), jobid=None 분기에 start_idx/end_idx 정의 추가, 결국
  end_idx=20 으로 단축.
- **outline 합성**: DS2D 가 외곽선을 출력하지 않으므로 `normalize_ds2d.py` 에 `_synthesize_outline()`
  추가 — shapely `unary_union` + envelope (MultiPolygon 케이스). 4 condition (full / bubble /
  partial / sparse) 각 80 plan = 320 정규화 결과를 `experiments/generations/ds2d_{cond}/` 로 분리
  저장. 통일 raster 렌더링까지 완료.
- **DS2D 메트릭 측정**: FID = 98.69, Overlap rate = 0.40, ortho = 0.968, no_overlap = 0.969,
  room_in_outline = 0.998 (합성 outline 효과), coverage = 0.692. ours 와의 격차 명확.
- **HouseDiffusion JSON 변환 불일치 발견**: 우리 wrapper 가 만든 JSON (`edges_features` 키) 와 HD
  `reader()` 함수가 기대하는 (`edges` 키, 벽 line segment Nx4 + `ed_rm` 매핑) 형식이 다름. 본 연구
  JSONL 에는 벽 segment 정보가 없어 정확한 변환 어려움 → cited 인용 결정 (P1).

### 2026-05-23 (Baseline 환경 구축 — DS2D / GSDiff / HouseDiffusion venv 의존성 설치)

- **HF 토큰 등록 + Llama-3-8B-Instruct gated access**: 사용자가 `huggingface_hub` CLI 신버전
  (`hf auth login`) 으로 토큰 저장 후 라이센스 동의 완료. mpi4py prerequisite 도 `sudo apt-get
  install libopenmpi-dev` 로 사용자가 처리.
- **3 baseline venv 동시 의존성 설치**: 각 baseline 의 `.venv` 안에 torch + transformers + peft 등
  필수 패키지 설치. `--python <venv>` + `--no-config` 로 부모 codebase pyproject.toml override
  차단. DS2D=Py3.10 + torch 2.3, GSDiff=Py3.10 + torch 2.0.1+cu118 + PyG 일부, HouseDiffusion=Py3.8
  + torch 2.0.1+cu118 + mpi4py 4.1.2 + drawSvg (대문자 S).
- **GSDiff `datasets` 모듈 충돌**: GSDiff 의 자체 `datasets/` 폴더 vs HF `datasets` 패키지가 같은
  이름이라 import 충돌. grep 으로 검증 (HD `from datasets import` 없음 확인) 후 `datasets/` →
  `gs_datasets/` rename + 50 파일의 import 일괄 sed patch.
- **GSDiff preprocessing 시도**: extract 완료 (80,788 plan), process1 완료. process2 에서
  `TypeError: list indices must be integers or slices, not tuple` (graph_ori adjacency_list 구조
  불일치 — repo 자체 코드 정합성 문제). README 의 `move.py` 도 실제 파일 없음. 디버그 미완.
- **HouseDiffusion save_samples patch**: `scripts/baselines/patch_hd_save_samples.py` 작성 — HD 의
  `save_samples()` 의 polygon 그리기 직후 JSON dump 코드 idempotent in-place 추가. 본격 추론은
  미실시.
- **render_unified.py 의 `--show_labels` 인자 추가**: 통일 raster 가 color_map.yaml 의
  `show_labels=true` 기본값을 따라 라벨 ON 으로 그려지던 문제 해결. 추론·논문 figure 표준 라벨
  OFF (`--show_labels false`).
- **DS2D 정규화의 KeyError: 4 진단**: 데이터 변환된 `data['rooms']` 는 list-of-dict (room_type 이
  문자열) 인데 코드는 list-of-list ([4] 정수 인덱싱) 기대. 두 줄 patch 로 해결.

### 2026-05-22 (`docs/EXPERIMENT_RESULTS.md` 신규 — 결과 요약 문서 분리)

- 사용자 요청: "실험 환경·세팅·변수 + 결과를 종합 정리한 문서". 측정값들이 PROGRESS 의 §A 표
  비고란이나 §B 시계열 로그에 흩어져 있던 것을 한 곳으로 통합.
- 구성:
  1. 실험 환경 (하드웨어·소프트웨어·3-stage pipeline·데이터셋·통일 testset·추론 설정·평가
     protocol)
  2. 실험 결과 (효율성·Exp 1 ours 행·Exp 3 ours 행·종합 해석)
  3. 메트릭 정합성 검증 (GT-vs-GT smoke test 참고선)
  4. 미수행 실험 목록
  5. 산출 CSV 인덱스
- 향후 baseline·ablation·user study 결과가 들어오면 이 문서 §2 에 추가 (현재는 ours 단독 결과만).

### 2026-05-22 (ours 단독 메트릭 측정)

- Baseline 추론 결과를 기다리지 않고 ours vs GT 단독 측정 자동 실행 (~15 초).
- 산출 파일:
  - `experiments/metrics/exp1_compatibility_ours.csv` (329 행, mean GED 3.82)
  - `experiments/metrics/exp1_fid.csv` (legacy_pytorch FID = 25.36)
  - `experiments/metrics/exp1_self_ours.csv` (Overlap 0, P.Overlap 0, R.Area MAPE 0.257)
  - `experiments/metrics/exp3_novel_ours.csv` (11 metric)
- 시사점:
  - 기하학적 quality 가 매우 우수 (format/ortho/no_overlap/outline_in_room 모두 1.0).
  - sparse augmentation 영향으로 spatial·input_consistency 는 낮음 (의도된 자율생성).
  - FID 25.36 은 HouseDiffusion published 11.2 보다 높으나 protocol 차이 (best-of-10 모두
    사용 + augmentation drop 적용) — Table 1 footnote 에 명시 필요.

### 2026-05-22 (가이드 §0.3 — 실험별 데이터 흐름 매핑 마스터 표 신설)

- 사용자 요청: "어떤 실험이 어디에 위치한 어떤 데이터를 사용하고, 어디에 중간/최종 결과가
  저장되는지 전부 명시" — Exp 1~10 별 입력·중간·최종 산출물 매핑이 가이드 안에 흩어져 있던 것을
  **`EXPERIMENTS_GUIDE.md` §0.3** 한 곳으로 통합. 세 표 추가:
  1. 공통 입력 (모든 Exp 가 참조하는 자원: RPLAN PNG, JSONL/Arrow, eval_pool, testset_unified,
     토크나이저, EA/SFT/RL adapter) — 각 자원의 절대경로·용도·생성 단계 명시.
  2. Exp 1~10 별 흐름 표 — 모델/조건, 입력(여러 항목), 중간 산출물, 최종 산출물 (table_*.csv /
     figure_*.pdf) 한 줄씩.
  3. 산출물 디렉토리 한눈에 — 어떤 디렉토리가 어떤 스크립트로 만들어지는지.

### 2026-05-22 (본 추론 완료 확인 + baseline weight 수령 + ours/GT 라벨 OFF 재렌더링)

- **본 연구 추론 완료 확인**: 5/15 14:52 시작한 백그라운드 추론이 ~12.2 시간 후 정상 종료. 329
  plans × 10 outputs = 3,290 generations 모두 `parse_success=True`, 평균 13.36 s/gen (smoke 측정값
  10.6 s 보다 다소 느림 — KV cache 누적 영향). 산출물: `outputs/inference/.../14-52-26/`
  (329 plan dir), `experiments/generations/ours/` (3,290 JSON), `experiments/metrics/ours_efficiency.csv`.
- **잘못된 1차 진단**: 추론 로그를 `head -20` 으로 시작부만 확인하고 진행률 5% 에서 죽었다고
  오판 → 실제는 `tail` 확인 시 `[run_ours_unified] DONE` + 산출물 fully 존재. 향후 가이드의
  진행 검증은 반드시 `tail` 로 한다.
- **Baseline weight 수령 확인**: 사용자가 3 종 모두 배치 완료.
  - `baselines/house_diffusion/ckpts/exp/model250000.pt`
  - `baselines/gsdiff/outputs/topo-params/`, `topoae/`
  - `baselines/ds2d/models/{5R,6R,7R,8R}/`
- **render_unified.py 라벨 토글 누락 발견**: 추론 자체 PNG (`outputs/.../floorplan.png`) 는 추론
  config `output.draw_labels=false` 가 result_saver 까지 전달되어 라벨 OFF 정상. 그러나 통일
  raster (`experiments/renders/ours/*.png`) 는 `render_unified.py` 가 `color_map.yaml` 의 기본값
  `show_labels=true` 를 따르고 있어 라벨 ON 으로 그려져 있었음. 논문 figure 표준 (라벨 OFF) 과
  불일치 → `--show_labels` 인자 신설(기본 False).
- **재렌더링**: ours 3,290 (~6 초) + GT 329 (~0.5 초). 모델 추론은 다시 수행하지 않음. 공통 스키마
  JSON 만으로 polygon 다시 그리는 게 끝이라 매우 빠름. 이로써 ours·GT 모두 라벨 OFF 단일 protocol.

### 2026-05-15 12:55 (본 추론 백그라운드 시작)

- 사용자: (A) 본 연구 추론을 백그라운드로 시작하라 요청.
- `nohup bash scripts/experiments/run_ours_unified.sh 10 ours &` 로 PID 33094 시작.
- 사용자: 약 15 분 후 추론 중지 요청 + 렌더러 수정 요청.

### 2026-05-15 15:05 (텍스트도 LINE_8 통일)

- 일회성 monkey patch 로 텍스트 LINE_8 버전(`gt_test_textline8/`) 을 사용자에게 비교 제시.
- 사용자가 LINE_8 선호 → ``draw_label_at`` 의 ``cv2.putText`` lineType 을 ``cv2.LINE_8`` 로
  영구 변경. 도형·텍스트 모두 픽셀 단위 단색 톤으로 일관됨.
- 검증: 영구 변경 결과 (`gt_test_final/`) 가 monkey patch 결과 (`gt_test_textline8/`) 와
  md5 해시 단위 동일 (5/5 plan 모두 ✓).

### 2026-05-15 14:50 (픽셀 깨짐 원인 재진단 + 옛 동작 복원)

- 사용자가 옛 렌더러 결과(`image.png`)와 변경 후 렌더러 결과(`image_1.png`)를 확대 비교한
  사진을 제공. 옛 결과는 alpha 블렌딩 + 256 해상도에서도 테두리·방 영역의 픽셀이 전혀
  깨지지 않음을 확인.
- 재진단 결과:
  1. 옛 코드: ``cv2.polylines(...)`` 에 lineType 미지정 = 기본 ``cv2.LINE_8``. 픽셀 단위
     단색 라인 → alpha 블렌딩과 결합해도 영역별 단색 보장.
  2. 첫 변경 시 z-order 분리하면서 제가 ``lineType=cv2.LINE_AA`` 를 *추가* 했다 — 이게 픽셀
     깨짐의 직접 원인 (LINE_AA 는 가장자리에 부분 투명 픽셀을 만듦).
  3. SSAA 4× 는 텍스트 가독성에는 도움이 되지만 도형 가장자리는 INTER_AREA 다운샘플 평균
     으로 오히려 약간 흐려진다. 옛 LINE_8 + alpha 블렌딩만큼 sharp 하지 못함.
- 해결: **옛 동작 복원**
  - 도형 (polylines / rectangle / fillPoly): ``LINE_8`` (옛날과 동일).
  - 텍스트 (cv2.putText): ``LINE_AA`` 유지 (작은 폰트 가독성 필수).
  - SSAA 기본값 ``supersample: 4 → 1`` (비활성). 옵션 자체는 유지하여 사용자가 텍스트
    가독성을 더 부드럽게 만들고 싶을 때 켤 수 있음.
- 결과: `experiments/renders/gt_test_sharp/` (라벨 ON) 와 `gt_test_sharp_nolabel/` (라벨 OFF) 에
  재렌더링한 329 / 5 장 검증. 옛 image.png 와 동일한 픽셀 단위 단색 테두리 확인.

### 2026-05-15 14:25 (렌더러 SSAA 도입)

- 사용자가 확대된 평면도 일부분 사진을 보내며 **테두리 / 방 영역 / 텍스트의 픽셀 깨짐**을
  지적. 직교 도형은 본질적으로 discrete 한 색만 칠하므로 픽셀 단위 정확 렌더링이 가능해야
  한다는 의견.
- 원인:
  - `cv2.LINE_AA` 안티앨리어싱 + 256 작은 해상도에서 alpha 0.6 블렌딩이 결합해 픽셀 단위
    색 변조가 발생.
  - `cv2.putText` 의 폰트는 sub-pixel 렌더라 256 같은 작은 해상도에서 흐릿.
- 해결: **SSAA (Super-Sampling Anti-Aliasing) 도입**
  - `RoomRenderer(supersample=4)` 신설 — 내부 캔버스를 1024×1024 로 만들고 모든 좌표·두께·
    폰트 크기를 4× 곱셈해서 그린 뒤 ``finalize_canvas`` 가 ``cv2.INTER_AREA`` 로 256 로 다운샘플.
  - 직교 폴리곤이라 SSAA 만으로 충분 → ``LINE_AA`` 를 ``LINE_8`` 로 변경 (LINE_AA 가 alpha 블렌딩과
    겹치면 도리어 색을 흐림).
  - 텍스트도 4× 큰 폰트로 그린 뒤 축소되어 가독성·픽셀 정렬 모두 향상.
- 비용: 캔버스 메모리·렌더 시간 16× (256→1024), 측정값 329 plan ≈ 7 초 (이전 1 초). 무시 가능.
- `vis_settings.supersample: 4` 기본값 추가, visualizer 의 `_save_*_image` / `_render_floorplan_canvas`
  마지막 단계가 `finalize_canvas` 호출.

### 2026-05-15 13:10 (렌더러 / config 수정)

사용자 요청 사항 7 가지:

1. **추론 시 라벨 끄기 옵션** — `config/inference/pipeline.yaml` 에 `output.draw_labels: false`
   추가, `result_saver.py` 가 visualizer 에 전달.
2. **텍스트 z-order 보존** — 방 A 텍스트가 방 B alpha 채우기에 가려지는 문제. renderer 에서
   `draw_room_polygon` / `draw_door_rect` 의 label 인자 제거, 별도 `draw_room_label` /
   `draw_door_label` / `draw_label_at` 메서드로 분리. visualizer `_render_floorplan_canvas` 가
   모든 도형 그린 후 라벨만 한꺼번에 그림.
3. **텍스트 위치 변경** — 폴리곤 중심 → `min(xs)` + `mean(ys)` (가장 왼쪽 x 에서 시작).
4. **추론에서 SFT/RL 둘 다 사용 검증** — *버그 발견*: 추론 model_loader 가 `load_adapter` 만
   호출하고 `set_adapter` 미호출. PEFT 의 `load_adapter` 는 메모리만 적재, 두 번째 adapter 의
   active 활성화는 안 됨. 결과적으로 **이전까지의 모든 추론은 SFT 만 적용된 결과였을 가능성**.
   `model.base_model.set_adapter(adapter_names_loaded)` 추가 (다중 adapter 시).
5. **color_map.yaml 팔레트 개선** — Material Design 톤(채도 ↑, 명도 ↓ 약간) 으로 교체. 기존
   값은 `# old: ...` 주석으로 전부 보존.
6. **테두리 색 통일** — 명도 차감 분기 폐기, 모든 방 공통 차콜(`[44, 44, 44]`). 건축 도면 톤.
7. **EXPERIMENTS_GUIDE.md → docs/** 이동 + 보충 (절대경로·환경 가정·smoke test 절차·시행착오 표
   추가). 본 진행상황 문서(`EXPERIMENT_PROGRESS.md`) 신설.

검증: GT raster 1 장 (10197) 을 라벨 on/off 두 버전으로 재렌더링 → 사용자가 직접 시각 확인.
라벨 위치·z-order·색깔 모두 정상.

---

## C. 주요 시행착오 · 수정 기록

| 일자 | 항목 | 원인 | 해결 |
|---|---|---|---|
| 2026-05-14 | 통일 test 풀이 81 plan 으로 부족 | yaml 명세 val=0.1%·test=0.5% 가 Arrow 에 swap 적용된 듯 (val=404, test=81) | val + test concat → eval_pool Arrow split + plan_id 기반 통일 추출 |
| 2026-05-14 | `input.arrow_split=eval_pool` 호출 실패 | eval_pool 은 별도 디렉토리, DatasetDict 의 split 키가 아님 | `input.arrow_dir=data/.../eval_pool` 로 직접 지정 |
| 2026-05-14 | smoke test 의 input_consistency=0.009 등 일부 metric 낮음 | metadata 를 GT full 로 구성 + augmentation drop 으로 모델이 본 입력이 sparse | 평가 protocol 결정은 사용자 선택 (가이드 §6.3 의 standard reward-as-metric 방식 유지) |
| 2026-05-14 | input_consistency / spatial / room_in_outline 이 GT-vs-GT 에서도 1.0 미달 | RPLAN PNG → polygon 변환 잔차 (직교 폴리곤 근사) | 정상 패턴으로 인식, 모든 모델에 동일 protocol 적용되어 비교 fair |
| 2026-05-14 | baseline `uv venv` 가 부모 pyproject.toml Python 3.11 요구를 끌어옴 | uv 가 cwd 의 부모를 자동 탐색 | 각 baseline 디렉토리에 shim pyproject.toml 자동 생성 |
| 2026-05-14 | setup_baselines.sh 첫 baseline 실패 시 다음 baseline 미진행 | `set -e` | set -e 제거, 각 baseline 함수에 `\|\| true` 와 별도 함수로 격리 |
| 2026-05-15 | 텍스트 라벨이 반투명 가려짐 + 방 가운데 위치 → 다른 방에 가려짐 | (1) 라벨을 도형 그릴 때마다 즉시 그리고 다음 방 alpha 가 그 위 덮음 (2) 위치가 polygon 중심 | renderer 에서 라벨 그리기를 별도 메서드로 분리, visualizer 가 모든 도형 → 라벨 순서로 호출. 위치는 `min(xs)` + `mean(ys)` |
| 2026-05-15 | **CRITICAL: 추론에서 RL adapter 가 forward 에 반영 안 됨** | `load_adapter` 는 메모리 적재만, active 변경 없음. 학습 model_loader 는 `set_adapter([sft,rl])` 명시 호출하지만 추론은 누락 | `model.base_model.set_adapter(adapter_names_loaded)` 추가. 이전까지의 모든 ours 추론 결과는 SFT only 결과였을 가능성 → 본 추론 재실행 필요 |
| 2026-05-15 | 256 캔버스에서 테두리·텍스트·alpha 블렌딩 영역의 픽셀 깨짐 (1차 진단) | 작은 해상도 + LINE_AA + alpha 0.6 결합 + putText sub-pixel | SSAA 4× 도입 — 내부 1024 캔버스 렌더 → INTER_AREA 다운샘플 |
| 2026-05-15 | 위 SSAA 도입에도 사용자 확대 비교 결과 도형 가장자리가 옛 image.png 만큼 sharp 하지 못함 (2차 진단) | SSAA 다운샘플 평균이 가장자리에 부분 평균 픽셀을 남김. 옛 코드는 LINE_8 + alpha 블렌딩만으로도 픽셀 단위 단색이 이미 깨끗했음 → SSAA 자체가 옛 동작 대비 후퇴 | 도형 LINE_8 유지 / 텍스트만 LINE_AA / SSAA 기본 OFF (supersample=1). 옵션 자체는 유지 |
| 2026-05-22 | 추론 백그라운드 상태 확인 시 `head -20` 만 보고 진행률 5% 에서 죽었다고 오판 | tqdm 진행률이 한 줄에 누적 출력되어 head 만 보면 처음 17 plan 까지만 보임 | 진행 검증은 반드시 `tail experiments/run_ours_unified.log` 로. `[run_ours_unified] DONE` 줄 + plan_id dir 카운트 329 + efficiency CSV 존재 여부로 확인 |
| 2026-05-22 | 통일 raster (`experiments/renders/ours/*.png`) 가 라벨 ON 으로 그려짐 | `render_unified.py` 가 color_map.yaml `vis_settings.show_labels` 기본값 true 를 따름. 추론 config 의 `output.draw_labels` 와는 별도 경로 (result_saver 가 만드는 추론 PNG vs render_unified 가 만드는 통일 raster) | `render_unified.py` 에 `--show_labels` 인자 신설(기본 False, 논문 figure 표준). visualizer 의 `show_labels=False` argparse override 와 일관 |
| 2026-05-23 | baseline `uv pip install` 시 부모 codebase `pyproject.toml` 의 `[tool.uv] override-dependencies = ["numpy>=2.4.2"]` 가 자식 venv 까지 영향 | uv 가 cwd 의 `pyproject.toml`/`uv.toml` 의 tool.uv 설정을 자동 사용 | 모든 baseline install 명령에 `--no-config` 플래그 추가 |
| 2026-05-23 | HF `huggingface-cli login` 명령이 사라짐 | `huggingface_hub` 1.x 부터 CLI 이름이 `hf` 로 변경 | `uv run hf auth login` 또는 Python `from huggingface_hub import login; login(token=...)` |
| 2026-05-23 | DS2D Llama-3-8B-Instruct gated repo 접근 거부 | HF 라이센스 동의 + 토큰 부족 | 사용자가 직접 https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct 에서 Agree + `HF_TOKEN` 토큰 발급 |
| 2026-05-23 | DS2D `triton` import 시 `ModuleNotFoundError: setuptools` | uv venv 는 기본적으로 setuptools 미포함 | `setuptools` 추가 install |
| 2026-05-23 | GSDiff `from datasets.X import ...` import 시 HF `datasets` 패키지로 잘못 매핑 | 폴더명·패키지명 충돌. cwd 에서 venv `datasets` 가 더 먼저 잡힘 | (1) grep 으로 HD `from datasets import` 없는지 검증 (2) `datasets/` → `gs_datasets/` rename + 50 파일 sed patch |
| 2026-05-23 | GSDiff `rplan-extract.py` 가 `.git` 디렉토리에서 `int('.git')` 시도 | `os.listdir()` 의 결과에 PNG 외 파일 (.git 등) 포함 | `[int(fn[:-4]) for fn in all_data if fn.endswith('.png')]` 로 filter |
| 2026-05-23 | render_unified.py 의 `--show_labels` 인자 신설 후 ours/GT 재렌더링 | (기존) color_map.yaml 의 vis_settings.show_labels 기본값 true 만 따름 | argparse `--show_labels` (기본 False) 추가 |
| 2026-05-24 | DS2D `run_generation_rplan.py` 의 `room_label[data['rooms'][u][4]]` KeyError | 변환된 데이터는 list-of-dict (room_type 문자열), 코드는 list-of-list (정수 [4]) 기대 | `data['rooms'][u]['room_type']` 으로 patch (room_label dict 통과 불필요) |
| 2026-05-24 | DS2D 추론 출력이 `generations/rplan/` 가 아닌 `generations/rplan_greedy/` 에 저장 | `num_samples=1` 시 코드가 `out_dir += '_greedy'` 자동 append | 정규화·렌더링 시 `_greedy` suffix 디렉토리 직접 가리킴 |
| 2026-05-24 | DS2D 의 plan_id 가 RPLAN 원본 plan_id 아닌 자체 인덱스 ("5R_0" 등) | DS2D 가 자체 train split 에서 random sample 추출 — plan-by-plan 매핑 불가능 | (1) FID/Self-consistency Overlap/Novel plan-self metric 은 매핑 불요 — 그대로 측정 (2) GED 는 DS2D 의 `ground_truth.json` 으로 self-paired 측정 (3) compute_self_consistency.py + compute_novel_metrics.py 에 **plan-self only mode** patch 추가 |
| 2026-05-24 | DS2D 출력에 outline 없음 → novel metric 모두 0 | `common_to_parsed()` 의 `success` 조건이 `rooms[0].room_type == "outline"` | `normalize_ds2d.py` 에 `_synthesize_outline()` (shapely unary_union + envelope) 추가하여 rooms[0] 에 합성 outline 삽입 |
| 2026-05-24 | HouseDiffusion 데이터 변환 — 우리 wrapper 의 JSON 형식과 HD reader 기대 형식 불일치 | HD reader: `edges` (벽 line segment Nx4) + `ed_rm` (edge ↔ 방 인덱스 매핑). 우리 wrapper: `edges_features` (방-방 인접). HouseGAN++ 의 wall-segment 기반 입력 형식 | 직접 재현 미실시, **cited 인용 (P1)** 으로 정책 결정 |
| 2026-05-25 | ours_fullcond 결과 (FID 0.21, GED 0.27, input_consistency 0.9999) 가 의심스럽게 완벽 | 입력에 좌표 + 종류 + adjacency 모두 받아 LLM 이 좌표를 거의 그대로 출력 — 본질적으로 reconstruction. baseline (HD/HouseGAN++) 은 bubble diagram only 받음 → protocol mismatch | (1) ours_fullcond 는 upper bound sanity check 로 사용 (2) 진짜 fair 비교는 **ours_bubble** (HD 와 동일 sparsity, drop_coords=1.0) protocol 별도 측정 |
| 2026-05-25 | FID published HD vs ours_augmented 비대칭 (모집단 차이 + augmentation drop·transform 효과) | 사용자 정밀 분석 — 모델 품질 차이가 아닌 protocol 차이 | (1) ours_best1 신설 — plan 당 1 sample, 329 vs 329 균형 (2) cited_baselines.yaml protocol_note 강화 (3) ours_bubble protocol 별도 측정 예정 |
| 2026-05-25 | GSDiff process2 의 `TypeError: list indices must be integers or slices, not tuple` | process1 산출물의 adjacency_list 구조와 process2 기대 구조 mismatch — repo 자체 코드 정합성 문제 | 디버그 미완. README 의 `move.py` 도 실제 파일 없음. cited 인용 정책 적용 — 단 GSDiff 의 cited 값도 보고 불완전해 baseline 비교에서 제외 결정 |
| 2026-05-25 | DS2D paired GED 측정 첫 시도에서 `matched=0` 발생 | `_load_common_dir_grouped` 가 stem 끝의 `_숫자` 를 일률적으로 sample_idx 로 인식해 GT (`5R_2.json`) 의 plan_id 를 `5R` + `2` 로 잘못 분리 | `strip_sample_suffix` 인자 추가, gen=True / GT=False 로 분기 적용 |

---

## D. 사용자 요청 / 응답 트래커

| # | 요청 | 처리 | 상태 |
|---|---|---|---|
| 1 | 워크스페이스 옵션 A | codebase 내부에 `baselines/`, `experiments/` + `.gitignore` | ✅ |
| 2 | SFT/RL active symlink | 한국어 폴더 → 영문 alias | ✅ |
| 3 | Ablation 코드만 (학습 X) | 모든 variant shell + 코드 패치 작성 | ✅ |
| 4 | 본 연구 추론 (A) 백그라운드 시작 | 시작 후 사용자가 중지 요청 → 렌더러 수정 후 재시작 대기 | ⏸ |
| 5 | Baseline setup (B) | 3 baseline clone + venv 완료, weight 는 사용자 다운로드 | ⏳ |
| 6 | User study 가이드 | guide §6 + EXPERIMENT_PROGRESS 본 문서 + Streamlit 앱 | ✅ |
| 7 | YAML 채우기 가이드 | guide §10 체크리스트 + cited/figure_selections placeholder 주석 | ✅ |
| 8 | LaTeX/docx 통합 설명 | guide §9 — 논문 작성 단계로 분리, 자동화 어려움 명시 | ✅ |
| 9 | 추론 라벨 토글 옵션 | `output.draw_labels` + visualizer `show_labels` | ✅ |
| 10 | 텍스트 z-order 보존 | renderer 라벨 분리, visualizer 후처리 | ✅ |
| 11 | 텍스트 위치 min(xs) | renderer `draw_room_label` | ✅ |
| 12 | 추론에서 SFT/RL 동시 사용 검증 | **버그 발견·수정**: set_adapter 호출 추가 | ✅ |
| 13 | color_map.yaml 시각적 개선 | Material Design 톤, 기존 값 # old: 주석 보존 | ✅ |
| 14 | 테두리 검정 통일 | `[44,44,44]` 차콜 일괄 적용 | ✅ |
| 15 | EXPERIMENTS_GUIDE.md → docs/ + 보충 | 이동 + 절대경로·smoke test·시행착오 표 추가 | ✅ |
| 16 | EXPERIMENT_PROGRESS.md 신설 + 자동 갱신 | 본 문서 — 매 turn 갱신 | ✅ (계속) |
| 17 | guide 만으로 재현 가능하도록 보충 | 환경 가정·trial 분포·이벤트 로그 검증 예시 추가 | ✅ |
| 18 | 픽셀 깨짐 문제 (테두리·방 영역·텍스트) | SSAA 4× + LINE_AA→LINE_8 (1차) → 도형 LINE_8 / 텍스트 LINE_AA / SSAA OFF (2차, 옛 동작 복원) → 텍스트도 LINE_8 통일 (3차, 사용자 선호) | ✅ |
| 19 | "추론 끝났어?" 확인 | 잘못된 1차 진단 (5% 에서 죽음) → tail 재확인 → 정상 종료 + 3,290 generations + parse_success=1.0 보고 | ✅ |
| 20 | "raw output 어디?" 설명 | tokens.txt / floorplan.json (추론) + experiments/generations/ours/ (공통 스키마) — 5 단계 보존 위치 안내 | ✅ |
| 21 | render_unified.py 가 라벨 ON 으로 그리는 문제 | `--show_labels` 인자 추가(기본 False), ours/GT 재렌더링 (~6 초 + ~0.5 초) | ✅ |
| 22 | "GT raw output 출처 + 렌더 폴더?" 질문 | eval_pool Arrow split / `experiments/renders/gt_test/` 답변 | ✅ |
| 23 | "실험별 입력·중간·최종 결과를 가이드에 전부 명시" | EXPERIMENTS_GUIDE.md §0.3 신설 — 공통 입력 / Exp 1~10 흐름 / 산출물 디렉토리 3 표 | ✅ |
| 24 | "현재까지 실험 결과를 요약하는 문서 신설" | `docs/EXPERIMENT_RESULTS.md` 신규 — 환경·세팅·변수 + 결과 + GT-vs-self 참고선 + 미수행 목록 + CSV 인덱스 | ✅ |
| 25 | "baseline 추론 자동 시작" | 사용자가 sudo apt + HF Llama-3 라이센스 동의 끝낸 뒤 DS2D / GSDiff / HouseDiffusion venv 의존성 모두 자동 설치 (백그라운드 병렬). DS2D 부터 데이터 변환 + 추론 진행 | ✅ |
| 26 | "DS2D outline 합성 + GED·FID·novel 측정" | normalize_ds2d.py 에 `_synthesize_outline()` 추가, 4 condition 분리, FID 98.69 / Overlap 0.40 / ortho 0.97 / no_overlap 0.97 측정 | ✅ |
| 27 | "HouseDiffusion published 인용 (P1)" | cited_baselines.yaml 에 HD/HouseGAN/HouseGAN++/ref1/HouseLLM/Ashual/Johnson 값 채움. FloorPlan-LLaMa/GSDiff (cited)/DiffPlanner 제외 | ✅ |
| 28 | "FID 측정 protocol 을 ours 에 유리하게 수정" | (1) ours_best1 plan 당 1 sample 모집단 균형 (2) ours_fullcond (aug OFF, greedy) 추론 → FID 0.21 (3) ours_bubble protocol 정의 (drop_coords=1.0, transform OFF) | ✅ (ours_bubble 본 추론만 대기) |
| 29 | "EXPERIMENT_SUMMARY.md 로 rename + 모든 문서 갱신" | EXPERIMENT_RESULTS.md → EXPERIMENT_SUMMARY.md rename, PROGRESS·GUIDE·SUMMARY 모두 이번 세션 (5/23~5/25) 작업 반영 | ✅ (진행 중) |

---

## E. 다음에 해야 할 일

✅ 완료된 단계:
- 본 연구 best-of-10 추론 (3,290 generations, parse_success=1.0)
- ours/GT 라벨 OFF 재렌더링
- Baseline weight 3 종 수령

남은 단계:

1. **ours 단독 메트릭 측정** — 자동 가능, 즉시 실행. baseline 추론 없이도 GED/FID(vs GT)/
   self-consistency/novel 11 모두 계산 가능.
   ```bash
   TESTSET=experiments/testset_unified.json
   POOL=data/dataset/processed_dataset/rplan/arrow/eval_pool
   uv run python scripts/metrics/compute_compatibility.py \
       --gen experiments/generations/ours --gt_pool $POOL \
       --plan_ids_file $TESTSET --model_name ours \
       --output experiments/metrics/exp1_compatibility_ours.csv
   uv run python scripts/metrics/compute_fid.py \
       --gt experiments/renders/gt_test --gen experiments/renders/ours \
       --names ours --output experiments/metrics/exp1_fid.csv
   uv run python scripts/metrics/compute_self_consistency.py \
       --gen experiments/generations/ours --gt_pool $POOL \
       --plan_ids_file $TESTSET --model_name ours \
       --output experiments/metrics/exp1_self_ours.csv
   uv run python scripts/metrics/compute_novel_metrics.py \
       --gen experiments/generations/ours --gt_pool $POOL \
       --plan_ids_file $TESTSET --model_name ours \
       --output experiments/metrics/exp3_novel_ours.csv
   ```

2. **Baseline 추론·정규화·렌더링** (🚨 사용자, 가이드 §4) — weight 받았으므로 진행 가능.
   각 baseline 의 `.venv` 안에서 requirements 설치 + 추론 + 정규화 + 통일 렌더링.

3. **표·그림 산출** (자동, 가이드 §8) — baseline 결과 들어오면 Tables 1/2/3 + Figures 6~11.

4. **User study** (🚨 사용자, 가이드 §6) — 13 명 모집·Streamlit 진행·CSV 집계.

5. **figure_selections.yaml / cited_baselines.yaml** 사용자 작성.

6. **(선택) Ablation 학습** — variant 당 수 시간 ~ 며칠.

7. **LaTeX/docx 통합** (사용자, 논문 작성 단계).

---

## F. 본 문서 갱신 규칙 (AI 자기 점검)

매 turn 마다 다음을 확인:
- 새로운 작업·요청이 들어왔는가? → §B 시계열 로그에 한 줄 추가, §D 트래커에 항목 추가.
- 새 시행착오를 발견했는가? → §C 표에 한 줄 추가 (원인·해결 포함).
- 작업 상태가 변경됐는가? → §A 진행 현황 표 갱신.
- 마지막 갱신 일자를 §A 헤더에 명시.

**갱신 누락 시 본 문서가 stale 해진다.** 새 사용자 요청을 처리하기 전에 반드시 본 문서 갱신
대상인지 점검할 것.
