# RL 보상함수 + 어드밴티지/손실 흐름 검증 결과 보고서

본 보고서는 두 트랙 검증 결과 + 사용자 결정에 따른 수정 결과를 통합한다.
- **트랙 A — 스크립트 실행 검증**: 17 verifier (101+ 케이스), 모두 PASS
- **트랙 B — 직접 코드 정독 분석**: 11개 보상 + parser + advantage + dataset + trainer 정합성 점검
- **수정 결과**: 사용자 지시에 따라 7개 항목 수정, 6개 항목 유지(의도된 동작 또는 영향 미미)

---

## ✅ 수정 완료 항목

### F-1. `outline_in_room`이 `RLTrainer.reward_order`에 포함됨 [수정 완료]

**위치**: [src/training/rl/trainer.py:168-173](../../../../src/training/rl/trainer.py#L168-L173)

**Before**: `outline_in_room` 누락 → 학습에 weight=0으로 적용
**After**: `room_in_outline` 다음에 `outline_in_room` 추가

**검증**:
- `verify_compute_token_advantages.py / case_outline_in_room_in_order`: outline_in_room mask가 advantage 가중합에 정상 반영 (PASS)
- `verify_micro_step_loss_flow.py`: `_cached_rewards_per_func.shape = (2, 11)` (이전 10 → 11)
- W&B 로그에 `rewards/reward_outline_in_room/mean` 메트릭 새로 등장

---

### F-2. `format_reward`가 outline 부재/위치 잘못 시 페널티 부여 [수정 완료]

**위치**: [src/training/rl/rewards/format_reward.py:46-58](../../../../src/training/rl/rewards/format_reward.py#L46-L58)

**Before**: `len(parsed.rooms) < 2`만 검사 → outline 없이 방 2개만 출력해도 통과
**After**: 다음 조건 중 하나라도 위반하면 0.0:
- 방 개수 < 2
- 첫 번째 방이 outline이 아님
- outline이 두 번 이상 등장 (rooms[1:]에 outline)

**검증**:
- `verify_format_reward.py / case_no_outline_two_rooms`: outline 없이 bedroom+kitchen → 0.0 (이전 1.0)
- `verify_format_reward.py / case_outline_not_first`: outline이 두 번째 위치 → 0.0
- 회귀 가드로 추가됨

---

### F-3. `_parse_single_room`의 Y 토큰 누락 시 무한 루프 가드 [수정 완료]

**위치**: [src/training/rl/rewards/parser.py:322-330](../../../../src/training/rl/rewards/parser.py#L322-L330)

**Before**: `y is None` 시 `continue`만 하여 pos 미진행 → X-X-X… 패턴에서 같은 위치를 다시 검사
**After**: `y is None` 시 `self.pos += 1`로 강제 진행 (Mod Record 주석 명시)

**검증**: 기존 verifier들이 모두 PASS (회귀 없음).

---

### F-4. `connectivity._has_door_between` 깔끔한 boundary 거리 기반 재작성 [수정 완료]

**위치**: [src/training/rl/rewards/connectivity_reward.py:318-379](../../../../src/training/rl/rewards/connectivity_reward.py#L318-L379)

**Before**: 자체 작성한 `_min_distance_to_polygon` 사용. 점이 폴리곤 내부에 있어도 경계까지 거리만 반환 → door가 한 방 내부 깊숙이 있으면 dist가 작아 통과되는 false positive
**After**: shapely `polygon.boundary.distance(point)` 사용. 점이 내부에 있어도 boundary까지 양수 거리 반환하므로 깊이에 비례. door가 두 방 모두의 boundary 근방(≤20px)에 있을 때만 통과
**제거**: 더 이상 사용 안 하는 헬퍼 `_min_distance_to_polygon`, `_point_to_segment_distance` 삭제

**검증**:
- `verify_connectivity_reward.py`: 7/7 PASS (정상 케이스, drop_pair, drop_coords satisfiability 모두)
- `verify_micro_step_loss_flow.py`: 실제 1 step에서 `connectivity` 보상 메트릭 정상 산출

---

### F-10. `_vector_to_direction` [0, 360) 정규화로 부등호 일관 사용 [수정 완료]

**위치**: [src/training/rl/rewards/spatial_reward.py:147-188](../../../../src/training/rl/rewards/spatial_reward.py#L147-L188)

**Before**: `[-180, 180]` 범위 직접 사용. left 분기만 `>=` 사용 (다른 분기는 `<`). 정확히 angle=±157.5에서 분류 비대칭
**After**: atan2 결과를 `[0, 360)`으로 정규화한 뒤 모든 분기를 `low <= x < high` 형태로 통일. wrap-around은 right만 (0°/360° 경계)

**검증**: `verify_spatial_reward.py`: 5+1 PASS. 22.5° 경계 분류는 동일하게 "right-below" (부동소수점 오차로 22.5보다 약간 큼).

---

### F-11. `input_consistency` docstring threshold 표기 수정 [수정 완료]

**위치**: [src/training/rl/rewards/input_consistency_reward.py:60-62](../../../../src/training/rl/rewards/input_consistency_reward.py#L60-L62)

**Before**: `기본 30px` 표기 (실제 코드는 `_ANCHOR_DISTANCE_THRESHOLD = 15.0`)
**After**: `기본 _ANCHOR_DISTANCE_THRESHOLD = 15px (좌표 노이즈 3σ=9px + 모델 오차 마진)` 명시

---

### F-13. `parser.py` EOS 비교에서 `0 or -9999` falsy 단락 버그 수정 [수정 완료]

**위치**: [src/training/rl/rewards/parser.py:371-377](../../../../src/training/rl/rewards/parser.py#L371-L377)

**Before**: `(self.vocab.eos_token_id or -9999)` — eos가 정수 0인 경우 `0 or -9999 == -9999`라 EOS 비교 영구 false
**After**: `is not None` 명시 체크로 교체. Mod Record 주석 추가.

---

## ⛔ 수정하지 않은 항목 (사용자 결정 또는 영향 미미)

### F-5. `count_type`이 hallucinated/drop type을 silent allow [유지 — 의도된 동작]

**사용자 결정 사유**:
- ROOM_SUMMARY는 positive constraint이지 negative constraint가 아님
- `drop_room_summary_types` 적용 시 그 type이 입력에서 사라지는데 출력에 등장하는 건 자연스러운 시나리오
- 현재 동작이 의도된 데이터 증강과 정합

→ G 항목으로 분류 (회귀 가드).

---

### F-6. `count_reward.py:95-98` 분기 중복 [유지 — 무해]

`if exp == 0 and act == 0: continue`와 `if max_val == 0: continue`가 동치. 안전한 가드 중복으로 무해.

---

### F-7. `no_overlap`이 self-intersecting polygon에 페널티 부여 안 함 [유지 — 책임 범위 분리]

**사용자 결정 사유**:
- `no_overlap`은 docstring 그대로 "방끼리 겹침" 책임만
- self-intersecting은 `orthogonality`에서 비직각 꼭짓점으로 잡힘 (책임 분리)
- SFT까지 학습된 모델이 self-intersecting을 출력할 가능성은 매우 낮음

---

### F-8. `front_door` w=0/음수 가드 [유지 — 사실 결함 아님]

**판단**: 토큰 좌표는 0~255 범위이므로 음수는 발생 불가. w=0/h=0은 면적 0이라 채점 무의미하므로 skip이 정당. 가드 자체가 안전한 동작.

---

### F-9. same-type 다중 앵커 헝가리안 ambiguity [유지 — 영향 미미]

**판단**: 헝가리안이 거리 최소화로 매핑하므로 모델이 잘못 그리면 input_consistency 등에서 결국 잡힘. 이론적 결함이지만 실제 학습에 큰 영향 없음.

---

### F-12. `_batch_normalize` B=1 시 1/eps 폭주 [유지 — 실무 미발생]

**판단**: 실배치 = `per_device_train_batch_size × num_generations`. RL은 항상 G≥2이므로 B_local=1이 발생 거의 없음.

---

## ✓ 회귀 가드 (검증으로 정상 동작 확인됨)

### G-1. metadata 추출이 변형 후 좌표 + drop 반영 모두 수행
- `verify_metadata_after_transform.py`: 4/4 PASS
- `verify_metadata_after_drops.py`: 16/16 PASS

### G-2. `total_rooms` vs `len(metadata.rooms)` 비대칭 의도대로 작동
- drop_block 시 `total_rooms`는 ROOM_SUMMARY 노출값(불변), `metadata.rooms` 길이는 visible 방 기준

### G-3. count_type이 노출된 type만 채점 (drop된 type 무시 + hallucinated 무시는 의도된 동작)
- `case_drop_type_ignored`, `case_halluc_type_silent_allow`: 모두 1.0 PASS

### G-4. no_overlap 공유 벽 false positive 가드 (`contains` exclusive)
- `case_shared_wall_no_false_positive`, `case_single_corner_share`: 1.0 PASS

### G-5. room_in_outline의 front_door 4꼭짓점 책임 분리
- `case_front_door_rb_outside`: w/h만 error, cx,cy 마킹 금지 PASS

### G-6. outline_in_room 케이스 B 검출 + 경계 false positive 가드
- `case_case_b_l_concavity`, `case_boundary_no_false_positive`: PASS

### G-7. coverage와 room_in_outline의 dual 관계
- `case_quarter_filled`: room_in=1.0이지만 coverage=0.25 PASS

### G-8. connectivity/spatial 후보 satisfiability
- `case_drop_coords_satisfiable_match` 등: PASS

### G-9. input_consistency 앵커 + drop_type 매칭
- 7/7 PASS

### G-10. GDPO 정규화 NaN 방어 + z-score 정확성
- 6/6 PASS

### G-11. compute_token_advantages 신용할당 ON/OFF + 가중합 + 전역 토글 + outline_in_room 회귀 가드
- 6/6 PASS

### G-12. `_batch_normalize` 패딩 영역 평균 제외 (`completion_lengths` 사용)
- `case_padding_excluded` PASS

### G-13. RL adapter trainable + SFT adapter frozen + advantages 2D 변환 + outline_in_room 포함
- micro_step_loss_flow: `(B, T) = (2, 128)`, `K = 11` 확인

---

## 검증 스크립트 실행 결과 요약 (트랙 A, 수정 후)

| Group | 스크립트 | 케이스 | 결과 |
|---|---|---|---|
| 1 | verify_metadata_after_transform | 4 | 4/4 PASS |
| 1 | verify_metadata_after_drops | 16 | 16/16 PASS |
| 2 | verify_format_reward | 8 | 8/8 PASS (F-2 회귀 가드 추가) |
| 2 | verify_count_total_reward | 5 | 5/5 PASS |
| 2 | verify_count_type_reward | 7 | 7/7 PASS |
| 2 | verify_orthogonality_reward | 4 | 4/4 PASS |
| 2 | verify_no_overlap_reward | 6 | 6/6 PASS |
| 2 | verify_room_in_outline_reward | 5 | 5/5 PASS |
| 2 | verify_outline_in_room_reward | 3+1 | 4/4 PASS (직접 호출 격리 검증 추가) |
| 2 | verify_coverage_reward | 6 | 6/6 PASS |
| 2 | verify_connectivity_reward | 7 | 7/7 PASS |
| 2 | verify_spatial_reward | 5+1 | 6/6 PASS |
| 2 | verify_input_consistency_reward | 7 | 7/7 PASS |
| 3 | verify_gdpo_group_normalize | 6 | 6/6 PASS |
| 3 | verify_compute_token_advantages | 6 | 6/6 PASS (outline_in_room 회귀 가드) |
| 3 | verify_batch_normalize | 5 | 5/5 PASS |
| 3 | verify_micro_step_loss_flow | E2E | PASS (K=11, advantages=(2,128), grad 분리, loss=0.3515) |
| **총계** | | **103+** | **103+/103+ PASS** |

---

## 수정한 코드 파일 요약

| 파일 | 변경 내용 |
|---|---|
| [src/training/rl/trainer.py](../../../../src/training/rl/trainer.py) | F-1: reward_order에 `outline_in_room` 추가 |
| [src/training/rl/rewards/format_reward.py](../../../../src/training/rl/rewards/format_reward.py) | F-2: outline 첫 번째 위치 + 중복 검증 |
| [src/training/rl/rewards/parser.py](../../../../src/training/rl/rewards/parser.py) | F-3: Y 누락 시 pos += 1 / F-13: EOS None 명시 체크 |
| [src/training/rl/rewards/connectivity_reward.py](../../../../src/training/rl/rewards/connectivity_reward.py) | F-4: shapely boundary.distance 기반 재작성 + 미사용 헬퍼 제거 |
| [src/training/rl/rewards/spatial_reward.py](../../../../src/training/rl/rewards/spatial_reward.py) | F-10: [0, 360) 정규화로 부등호 일관 사용 |
| [src/training/rl/rewards/input_consistency_reward.py](../../../../src/training/rl/rewards/input_consistency_reward.py) | F-11: docstring threshold 30→15 |
