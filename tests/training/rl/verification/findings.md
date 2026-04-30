# RL 보상함수 + 어드밴티지/손실 흐름 검증 결과 보고서

본 보고서는 두 트랙으로 진행된 검증 결과를 통합한다.
- **트랙 A — 스크립트 실행 검증**: Group 1 (전처리) 20 케이스, Group 2 (보상별) 56 케이스, Group 3 mock (어드밴티지) 17 케이스, Group 3 실제 모델 1 micro-step → 모두 PASS.
- **트랙 B — 직접 코드 정독 분석**: 11개 보상함수 + parser + advantage + dataset + trainer 소스를 한 줄씩 분석하여 의도-구현 정합성 점검.

각 finding은 `[A]`, `[B]`, `[A+B]` 출처를 표기한다.

---

## ★★★ 핵심 결함 (즉시 수정 권장)

### F-1. `outline_in_room`이 `RLTrainer.reward_order`에서 누락 [A+B]

**위치**: [src/training/rl/trainer.py:168-172](../../../../src/training/rl/trainer.py#L168-L172)

**현재 코드**:
```python
reward_order = [
    "format", "count_total", "count_type",
    "orthogonality", "no_overlap", "room_in_outline", "coverage",
    "connectivity", "spatial", "input_consistency",
]
```
`outline_in_room`이 빠져 10개만 등록된다.

**영향**:
- `compute_all_rewards()`는 `outline_in_room` 보상을 계산하지만 cache only 효과
- TRL `reward_funcs` callable이 만들어지지 않아 `rewards_per_func` 행렬에 미반영 (열 K=10이 11이어야 함)
- `_error_masks_buffer`에 `outline_in_room` mask가 저장되어도 `compute_token_advantages`는
  `reward_names`만 순회하므로 mask가 가중합에 반영 안 됨 → 신용할당 무력화
- 이 보상의 가중치(현재 1.0 가정)가 GDPO 학습 신호에서 0에 해당

**경험 검증 (트랙 A)**:
- `verify_compute_token_advantages.py / case_outline_in_room_miss`: mask 단독 제공 시 advantage에 반영 안 됨 PASS
- `verify_micro_step_loss_flow.py`: 실제 1 step에서 `_cached_rewards_per_func.shape == (2, 10)` 관측 (11이어야 함)

**권장**: `reward_order` 리스트에 `"outline_in_room"`을 `"room_in_outline"` 다음에 추가.

---

### F-2. `R_format`이 outline 부재 케이스를 통과시킴 [A+B]

**위치**: [src/training/rl/rewards/format_reward.py:50-53](../../../../src/training/rl/rewards/format_reward.py#L50-L53)

**현재 코드**:
```python
# 최소 outline + 1개 방 필요
if len(parsed.rooms) < 2:
    all_room_indices = _collect_room_token_indices(parsed)
    return 0.0, all_room_indices

return 1.0, []
```

**문제**: `len(parsed.rooms) < 2`만 검사. `parsed.rooms`은 `<TYPE:outline>`이 첫 번째인지 보장하지 않는다. `parser.py`의 `_parse_rooms()`는 `<ROOM>` 토큰을 순서대로 파싱할 뿐 outline 위치 강제 없음.

**경험 검증 (트랙 A)**:
- `verify_format_reward.py / case_no_outline_two_rooms`: outline 없이 bedroom + kitchen 2개 출력 → format=1.0 통과 PASS

**영향**: docstring(line 50: "최소 outline + 1개 방 필요")과 모순. 다른 보상(`R_room_in_outline`, `R_coverage`)이 outline 부재를 0점으로 처리하므로 hard gate를 우회한 비정상 출력이 다른 보상에서 페널티 받지만, format이 통과시켜 `hard_gate_pass=True`가 되어 다른 보상 계산 자체는 허용된다. 모델이 outline을 생략하는 것이 학습 신호상 유리할 가능성 (다른 보상의 페널티가 더 작을 때).

**권장**: format 검사에 `parsed.rooms[0].room_type == "outline"` 또는 `any(r.room_type == "outline" for r in parsed.rooms)` 단언 추가.

---

### F-3. `_parse_single_room`에서 잠재적 무한 루프 가능성 [B]

**위치**: [src/training/rl/rewards/parser.py:303-326](../../../../src/training/rl/rewards/parser.py#L303-L326)

**현재 코드** (요약):
```python
while self.pos < self.n and not self._peek(self.END_ROOM):
    if self._is_block_start_token(self.ids[self.pos]):
        self.error_indices.append(self.pos); break
    x_pos = self.pos
    x = self._parse_x()
    if x is None:
        self.error_indices.append(self.pos); self.pos += 1; continue
    y = self._parse_y()
    if y is None:
        self.error_indices.append(self.pos - 1); continue   # ← pos 미증가
    coords.append((x, y))
    coord_token_indices.append(x_pos)
```

**문제**: `_parse_x` 성공 후 `_parse_y` 실패 시 `pos`가 X 토큰 다음을 가리키지만 `_parse_y`는 `pos`를 안 옮긴다. `continue`로 인해 같은 위치로 다시 분기. 그 위치 토큰이 X도 block-start도 아니면 `_parse_x` returns None → 다음 iter에서 `pos += 1`로 빠져나감 (안전). 단, 그 위치가 또 다른 X 토큰이라면 `(x, y=None)`이 반복되어 **무한 루프 유사 동작** 가능.

**영향 범위**: 모델이 X-X-Y-X-X-Y... 패턴 출력 시. 실제 발생 확률은 낮으나 가드 부재.

**권장**: `if y is None` 분기에서 `self.pos += 1` 추가하여 강제 진행.

---

### F-4. `connectivity_reward._has_door_between`의 false positive [B]

**위치**: [src/training/rl/rewards/connectivity_reward.py:355-386](../../../../src/training/rl/rewards/connectivity_reward.py#L355-L386)

**현재 코드**: `_min_distance_to_polygon`이 점이 폴리곤 **내부**에 있어도 경계까지 거리를 반환. door 중심이 두 방의 경계 근방(≤20px)이면 통과인데, door가 방 A 내부 깊숙이 있을 경우 dist_a가 (경계까지 거리) >0이 나오지만 작을 수 있어 통과하기 쉽다.

**문제**: door 의도는 "두 방 사이 벽에 위치"인데, 한 방의 내부 깊숙이 있는 door도 통과시킬 가능성. 정확한 의도 검증을 위해서는 door 중심이 두 방의 **공유 경계** 근방인지 확인해야 함.

**권장**: door 중심이 두 방의 합집합의 경계 또는 둘의 교집합/경계 둘레 근방인지 더 엄격하게 검사.

---

## ☆ 의도 vs 구현 불일치 (검토 필요)

### F-5. `count_type`이 hallucinated/drop된 type을 silent allow [A]

**위치**: [src/training/rl/rewards/count_reward.py:79-106](../../../../src/training/rl/rewards/count_reward.py#L79-L106)

**현재 코드**: `expected_counts.keys()`만 순회하므로 출력에 포함된 새로운 type(metadata에 없는)은 무시됨.

**경험 검증 (트랙 A)**:
- `verify_count_type_reward.py / case_halluc_type_silent_allow`: `expected={bedroom:2}`, 출력에 `bedroom×2 + storage×1` → reward=1.0 PASS
- `verify_count_type_reward.py / case_drop_type_ignored`: drop된 type을 출력해도 1.0 PASS

**의도**: 수정 시 의도는 "drop된 type 부당 감점 방지"였으나 부수효과로 hallucinated type도 무시됨. 사용자 의도 확인 필요.

**권장**: 의도가 "드롭된 type만 무시"라면 drop_state를 metadata에 명시하고 hallucinated type은 페널티 부여. 의도가 "노출된 type만 채점"이라면 현재 동작이 정확.

---

### F-6. `count_reward.py:95-98` 분기 중복 [B]

**위치**: [src/training/rl/rewards/count_reward.py:95-98](../../../../src/training/rl/rewards/count_reward.py#L95-L98)

**현재 코드**:
```python
if exp == 0 and act == 0: continue
max_val = max(exp, act)
if max_val == 0: continue
```

`exp == 0 and act == 0` 인 경우 `max_val == 0` 도 동시에 참이므로 두 번째 검사는 redundant. 무해하나 의도적 안전 가드인지 정리 필요.

---

### F-7. `no_overlap`이 self-intersecting polygon에 페널티 부여 안 함 [A+B]

**위치**: [src/training/rl/rewards/geometry_reward.py:127-135](../../../../src/training/rl/rewards/geometry_reward.py#L127-L135)

**현재 코드**: invalid polygon은 `polys[i] = None`으로 처리되며, `total_area` 계산에서 제외. 토큰 마킹도 안 됨.

**경험 검증 (트랙 A)**:
- `verify_no_overlap_reward.py / case_self_intersecting_bowtie`: bowtie polygon (자기교차) → reward=1.0 (페널티 없음) PASS

**영향**: 모델이 self-intersecting polygon 출력 시 no_overlap에서 페널티 없음 (orthogonality에서는 비직각으로 잡힘). 일관된 페널티 부여 부재.

**권장**: invalid polygon에 대해 명시적 페널티 (예: 해당 방의 모든 꼭짓점 토큰 마킹 + reward 감점).

---

### F-8. `room_in_outline`에서 front_door 음수/0 w/h 처리 [B]

**위치**: [src/training/rl/rewards/room_in_outline_reward.py:140-141](../../../../src/training/rl/rewards/room_in_outline_reward.py#L140-L141)

**현재 코드**: `if w > 0 and h > 0:` 가드. 음수도 skip된다.

**의견**: 좋은 방어. 다만 음수 w/h는 모델 출력에서 발생할 수 없을 것 같지만 (커스텀 좌표 토큰 0~255), 정확히는 토큰값 자체가 음수가 될 수 없음. 단 cx+w가 256을 넘으면 outline 폴리곤 좌표계 밖이지만 shapely는 처리 가능.

---

### F-9. `connectivity` same-type 다중 앵커 ambiguity [B]

**위치**: [src/training/rl/rewards/connectivity_reward.py:200-283](../../../../src/training/rl/rewards/connectivity_reward.py#L200-L283) (`_hungarian_match`)

**의도 vs 구현 갭**:
- 사용자 의도(Mod Record 주석): drop된 RID는 satisfiability, 앵커는 결정 매핑
- 코드: 같은 type 내에서 무게중심 거리 헝가리안 → 같은 type 다중 앵커가 swap된 위치에 출력되면 헝가리안이 swap을 매핑할 수 있어 false positive 가능. 반대로 모델이 두 방을 같은 위치에 그리면 매칭 1-1이 결정적이라 false negative 가능.

**권장**: 동일 type 다중 앵커도 후보 satisfiability로 처리하는 것이 의도와 일치. 또는 결정 매핑이 의도라면 명시 주석 추가.

---

### F-10. `spatial` 8방위 분기의 부동소수점 경계각 [A+B]

**위치**: [src/training/rl/rewards/spatial_reward.py:169-184](../../../../src/training/rl/rewards/spatial_reward.py#L169-L184)

**경험 검증 (트랙 A)**:
- `verify_spatial_reward.py`: 22.5° 경계 입력 → "right-below"로 분류 (코드의 `<` 분기 동작 그대로). 부동소수점 오차로 실제는 22.5보다 약간 큼.

**영향**: 의도된 동작 (>=22.5 → right-below). 그러나 line 177-178: `angle_deg >= 157.5 or angle_deg < -157.5`만 `>=` 사용. 일관성이 깨져 ±157.5에서 분류 차이 발생.

**권장**: `<` 분기 일관 사용 또는 명시 주석 추가.

---

### F-11. `input_consistency` docstring threshold 표기 오류 [B]

**위치**: [src/training/rl/rewards/input_consistency_reward.py:60](../../../../src/training/rl/rewards/input_consistency_reward.py#L60)

**현재 docstring**: `threshold: 무게중심 거리 임계값 (px). 기본 30px.`

**실제 기본값**: `_ANCHOR_DISTANCE_THRESHOLD = 15.0` (line 38).

**권장**: docstring을 "기본 15px"로 수정.

---

### F-12. `_batch_normalize` B=1 시 1/eps 폭주 [B]

**위치**: [src/training/rl/advantage.py:213-220](../../../../src/training/rl/advantage.py#L213-L220)

**현재 코드**: `batch_std = seq_means.std() if B > 1 else torch.zeros(...)`. B=1이면 std=0 → eps만으로 분할 → token_advantages가 1e8배로 폭주.

**경험 검증 (트랙 A)**: `verify_batch_normalize.py / case_b_one_batch_std_zero`: 차등 부호 보존되지만 절대값 폭주.

**영향**: PPO clip 후 loss는 NaN/Inf 안 되도록 처리되나, gradient norm이 매우 커져 학습 불안정 가능. 현실적으로 B_local≥2이면 발생 안 함.

**권장**: B=1 시 batch_std fallback (예: max(|seq_means|))을 도입하거나 B=1 케이스 별도 처리.

---

### F-13. `parser.py`의 EOS 비교 버그 (`0 or -9999` 단락) [B]

**위치**: [src/training/rl/rewards/parser.py:371](../../../../src/training/rl/rewards/parser.py#L371)

**현재 코드**: `self.ids[self.pos] == (self.vocab.eos_token_id or -9999)`

**문제**: `eos_token_id`가 `0`이면 `0 or -9999 == -9999`. EOS 비교가 항상 false. Qwen2.5는 EOS=151643이므로 영향 없으나 다른 LLM에서 잠재 결함.

**권장**: `eos_token_id if eos_token_id is not None else -9999` 형태로 수정.

---

## ✓ 정상 동작 확인 항목 (회귀 가드)

다음 항목들은 검증 결과 의도대로 정확히 작동함을 확인. 향후 코드 수정 시 회귀 방지를 위한 가드.

### G-1. metadata 추출이 변형 후 좌표 + drop 반영 모두 수행 [A]
- `verify_metadata_after_transform.py`: flip("H"/"V"), translate, pipeline e2e 모두 metadata 좌표가 변형 후 좌표와 일치
- `verify_metadata_after_drops.py`: 16개 케이스 모두 PASS (8가지 drop + 비대칭 + 조합)

### G-2. `total_rooms` vs `len(metadata.rooms)` 비대칭 의도대로 작동 [A]
- `verify_count_total_reward.py / case_drop_block_asymmetry`: drop_block 후 total_rooms=4 유지, len(visible)=2 → 출력 3개일 때 total_rooms 기준 1.0

### G-3. `count_type`의 drop된 type 무시 (이전 버그 회귀 방지) [A]
- `verify_count_type_reward.py / case_drop_type_ignored`: PASS

### G-4. `no_overlap` 공유 벽 false positive 가드 (`contains` exclusive) [A]
- `verify_no_overlap_reward.py / case_shared_wall_no_false_positive`: 한 변 공유 → 1.0
- `verify_no_overlap_reward.py / case_single_corner_share`: 한 점 공유 → 1.0

### G-5. `room_in_outline`의 front_door 4꼭짓점 책임 분리 [A]
- `verify_room_in_outline_reward.py / case_front_door_rb_outside`: rb만 외부 → w_idx, h_idx만 error, cx,cy 마킹 금지

### G-6. `outline_in_room` 케이스 B 검출 정확성 [A]
- `verify_outline_in_room_reward.py / case_case_b_l_concavity`: L자 outline + 사각방 → reward<1, reflex 꼭짓점 트랩 검출
- `verify_outline_in_room_reward.py / case_boundary_no_false_positive`: 경계 통과 false positive 가드 정상

### G-7. `coverage`와 `room_in_outline`의 dual 관계 [A]
- `verify_coverage_reward.py / case_quarter_filled`: room_in_outline=1.0이지만 coverage=0.25 → dual reward 강조

### G-8. `connectivity`/`spatial` 후보 satisfiability [A]
- `verify_connectivity_reward.py / case_drop_coords_satisfiable_match`: drop_coords 후보 매칭 PASS
- `verify_spatial_reward.py / case_drop_coords_satisfiability`: 같은 type 후보 satisfiability PASS

### G-9. `input_consistency` 앵커 + drop_type 매칭 [A]
- `verify_input_consistency_reward.py`: 7/7 PASS (앵커 거리 채점, drop_type 매칭, type 미스매치, 채점 비활성)

### G-10. GDPO 정규화 NaN 방어 + 그룹별 z-score [A]
- `verify_gdpo_group_normalize.py`: 6/6 PASS

### G-11. compute_token_advantages 신용할당 ON/OFF + 가중합 + 전역 토글 [A]
- `verify_compute_token_advantages.py`: 6/6 PASS (CA mask 적용, OFF 균등, 전역 토글, 다중 가중합, outline_in_room 누락 검출)

### G-12. `_batch_normalize` 패딩 영역 평균 제외 [A]
- `verify_batch_normalize.py / case_padding_excluded`: completion_lengths를 사용한 평균 정확

### G-13. RL adapter trainable + SFT adapter frozen + advantages 2D 변환 [A]
- `verify_micro_step_loss_flow.py`: 실제 1 step에서 advantages.shape=(2, 128), RL 392 trainable, SFT 392 frozen, loss=0.3561 finite

---

## 우선순위 요약

**즉시 수정 권장 (구현 결함)**:
1. ★★★ F-1: `outline_in_room` reward_order 누락 — 해당 보상이 학습에 0 weight로 적용되는 심각한 결함
2. ★★ F-2: format이 outline 부재 통과 — docstring 의도와 모순
3. ★ F-3: parser의 잠재 무한 루프 — 가드 추가 1줄
4. ★ F-13: EOS `or -9999` 단락 — 토큰 ID 0인 LLM에서 잠재 버그

**의도 확인 후 결정**:
5. F-4: connectivity false positive (door 내부 위치)
6. F-5: count_type hallucinated 무시 (silent allow)
7. F-9: connectivity same-type 다중 앵커 ambiguity

**문서/코드 정합성**:
8. F-11: input_consistency docstring 30px → 15px
9. F-6: count_reward.py:95-98 분기 중복

**관찰성/안정성**:
10. F-7: self-intersecting polygon 페널티 부재
11. F-10: spatial 8방위 부동소수점 일관성
12. F-12: B=1 batch_std=0 폭주

---

## 검증 스크립트 실행 결과 요약 (트랙 A)

| Group | 스크립트 | 케이스 | 결과 |
|---|---|---|---|
| 1 | verify_metadata_after_transform | 4 | 4/4 PASS |
| 1 | verify_metadata_after_drops | 16 | 16/16 PASS |
| 2 | verify_format_reward | 7 | 7/7 PASS |
| 2 | verify_count_total_reward | 5 | 5/5 PASS |
| 2 | verify_count_type_reward | 7 | 7/7 PASS |
| 2 | verify_orthogonality_reward | 4 | 4/4 PASS |
| 2 | verify_no_overlap_reward | 6 | 6/6 PASS |
| 2 | verify_room_in_outline_reward | 5 | 5/5 PASS |
| 2 | verify_outline_in_room_reward | 4 | 4/4 PASS |
| 2 | verify_coverage_reward | 6 | 6/6 PASS |
| 2 | verify_connectivity_reward | 7 | 7/7 PASS |
| 2 | verify_spatial_reward | 5+1 | 6/6 PASS |
| 2 | verify_input_consistency_reward | 7 | 7/7 PASS |
| 3 | verify_gdpo_group_normalize | 6 | 6/6 PASS |
| 3 | verify_compute_token_advantages | 6 | 6/6 PASS |
| 3 | verify_batch_normalize | 5 | 5/5 PASS |
| 3 | verify_micro_step_loss_flow | 1 (E2E) | PASS |
| **총계** | | **101+** | **101+/101+ PASS** |

---

## 핵심 메시지

전반적으로 보상함수와 어드밴티지 흐름은 의도대로 작동하며, **모델 시점 metadata 추출 (이전 버그 수정 사항)**, **GDPO + 토큰 신용할당**, **PPO loss → backward 흐름** 모두 정상이다.

다만 **F-1 (outline_in_room 누락)**은 새로 추가한 보상함수가 학습에 사실상 0 weight로 적용되는 심각한 결함이며, 1줄 수정으로 해결 가능하다. 이외에 F-2, F-3, F-13 등 명백한 결함과 F-5, F-9 같은 의도-구현 불일치가 존재하므로 우선순위에 따라 검토 후 수정할 것을 권장한다.
