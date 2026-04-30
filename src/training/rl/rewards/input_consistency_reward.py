"""R_input_consistency 보상함수 모듈.

입력 프롬프트에 좌표가 명시된 방이 출력에도 일관되게 존재하는지 평가한다.

평가 대상 방 두 종류:
    1. 앵커 방 (type+coords 모두 visible): 헝가리안(타입 그룹별) → 타입 일치 검증 + 거리 채점
    2. drop_type 방 (coords visible, type=""): 앵커 매핑 후 남은 출력 방들을 대상으로
       타입 무관 헝가리안(좌표 거리 기반) → 거리 채점만

채점 절차:
    1. metadata.rooms에서 앵커 방 / drop_type 방 분리
    2. 앵커 방: 헝가리안(타입 그룹별) → rid_to_output_idx
    3. drop_type 방: 앵커에서 사용된 출력 인덱스 제외 후 헝가리안(타입 무관)
    4. 각 방에 대해 매칭된 출력 방의 무게중심 거리로 점수 산출
       score_i = max(0, 1 - dist / threshold)
    5. 전체 평균 점수 반환

threshold:
    좌표 노이즈(σ=3px) + 변형 증강 후 잔차 + 모델 생성 오차 마진을 합산한 보수적 값.
    기본 30px. 이내일수록 점수 1에 가까움, 이상이면 0.

신용할당: 없음 (sequence-level 보상).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.training.rl.rewards.parser import ParsedFloorplan

logger = logging.getLogger(__name__)

# 앵커/drop_type 방 무게중심 거리 임계값 (px). 이 거리에서 점수 0, 0px에서 점수 1.
# 노이즈 3σ=9px + 모델 오차 마진. transform 증강은 입력/출력에 동일 적용되어 상대 오차 없음.
_ANCHOR_DISTANCE_THRESHOLD = 15.0


def compute_input_consistency_reward(
    parsed: "ParsedFloorplan",
    metadata: dict,
    threshold: float = _ANCHOR_DISTANCE_THRESHOLD,
) -> float:
    """입력 조건의 좌표 명시 방이 출력에 일관되게 포함되는 정도를 반환한다.

    각 평가 대상 방 $r$에 대해 매칭된 출력 방의 무게중심 거리 $d_r$로부터
    점수 $s_r = \\max(0, 1 - d_r / \\tau)$ 를 산출하고, 전체 평균을 반환한다.

    앵커 방은 타입 그룹별 헝가리안으로 결정 매핑하고,
    drop_type 방은 앵커 매핑 후 잔여 출력 방을 대상으로 타입 무관 헝가리안을 수행한다.

    Args:
        parsed: parse_output_tokens()의 반환값.
        metadata: 모델 시점 메타데이터. 사용 키:
            - rooms (list[dict]): visible 방 정보.
                앵커 = type과 coords 모두 비어있지 않음.
                drop_type = type="" AND coords 비어있지 않음.
        threshold: 무게중심 거리 임계값 (px). 기본 30px.

    Returns:
        [0, 1] 범위. 평가 대상 방의 평균 일관성 점수. 대상 없으면 1.0 (채점 비활성).
    """
    if not parsed.success or not parsed.rooms:
        return 0.0

    rooms = metadata.get("rooms", [])

    # 앵커: type+coords 모두 visible
    anchors = [
        r for r in rooms
        if r.get("type") not in ("outline", "")
        and r.get("coords")
    ]
    # drop_type: coords visible이나 type 마스킹됨
    drop_type_rooms = [
        r for r in rooms
        if r.get("type") == ""
        and r.get("coords")
    ]

    if not anchors and not drop_type_rooms:
        return 1.0  # 평가 대상 없으면 채점 비활성

    from src.training.rl.rewards.connectivity_reward import (
        _hungarian_match,
        _compute_centroid_from_raw,
        _compute_centroid_from_parsed,
    )

    output_rooms = [r for r in parsed.rooms if r.room_type != "outline"]
    if not output_rooms:
        return 0.0

    if threshold <= 0:
        threshold = _ANCHOR_DISTANCE_THRESHOLD

    # 앵커 방: 타입 그룹별 헝가리안
    rid_to_output_idx = _hungarian_match(parsed, metadata)
    used_output_indices: set[int] = set(rid_to_output_idx.values())

    # drop_type 방: 앵커 미사용 출력 방 대상 타입 무관 헝가리안
    free_rid_to_output_idx = _match_drop_type_rooms(
        drop_type_rooms, output_rooms, used_output_indices,
        _compute_centroid_from_raw, _compute_centroid_from_parsed,
    )

    scores: list[float] = []

    # 앵커 채점
    for anchor in anchors:
        rid = anchor.get("rid")
        out_idx = rid_to_output_idx.get(rid)

        if out_idx is None or out_idx >= len(output_rooms):
            scores.append(0.0)
            continue

        out_room = output_rooms[out_idx]

        # 타입 안전 확인 (헝가리안이 타입 그룹별이므로 거의 보장되지만 명시적으로)
        if out_room.room_type != anchor.get("type"):
            scores.append(0.0)
            continue

        in_cx, in_cy = _compute_centroid_from_raw(anchor["coords"])
        out_cx, out_cy = _compute_centroid_from_parsed(out_room)
        dist = math.sqrt((in_cx - out_cx) ** 2 + (in_cy - out_cy) ** 2)
        scores.append(max(0.0, 1.0 - dist / threshold))

    # drop_type 방 채점 (타입 불일치 검증 없음 — 타입 정보가 없으므로)
    for free_room in drop_type_rooms:
        rid = free_room.get("rid")
        out_idx = free_rid_to_output_idx.get(rid)

        if out_idx is None or out_idx >= len(output_rooms):
            scores.append(0.0)
            continue

        out_room = output_rooms[out_idx]
        in_cx, in_cy = _compute_centroid_from_raw(free_room["coords"])
        out_cx, out_cy = _compute_centroid_from_parsed(out_room)
        dist = math.sqrt((in_cx - out_cx) ** 2 + (in_cy - out_cy) ** 2)
        scores.append(max(0.0, 1.0 - dist / threshold))

    return sum(scores) / len(scores) if scores else 1.0


def _match_drop_type_rooms(
    drop_type_rooms: list[dict],
    output_rooms: list,
    used_output_indices: set[int],
    centroid_from_raw,
    centroid_from_parsed,
) -> dict[int, int]:
    """drop_type 방(coords만 visible)을 잔여 출력 방에 타입 무관 헝가리안으로 매핑한다.

    앵커 헝가리안에서 사용된 출력 인덱스를 제외한 나머지 출력 방 중
    좌표 거리 기반으로 최적 할당을 수행한다.

    Args:
        drop_type_rooms: type="" AND coords 비어있지 않은 방 리스트.
        output_rooms: outline 제외 출력 방 리스트.
        used_output_indices: 앵커 매핑에서 이미 사용된 출력 인덱스 집합.
        centroid_from_raw: flat coords 리스트 → (cx, cy) 함수.
        centroid_from_parsed: ParsedRoom → (cx, cy) 함수.

    Returns:
        {rid: output_room_index} 딕셔너리. 매핑 실패 시 빈 딕셔너리.
    """
    if not drop_type_rooms:
        return {}

    # 앵커에서 사용되지 않은 출력 방만 후보
    available = [(i, r) for i, r in enumerate(output_rooms) if i not in used_output_indices]
    if not available:
        return {}

    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as np
    except ImportError:
        logger.warning("scipy 미설치. drop_type 방 매칭 불가.")
        return {}

    n_free = len(drop_type_rooms)
    n_avail = len(available)

    # 비용 행렬: drop_type 방 × 잔여 출력 방 무게중심 거리
    cost = np.zeros((n_free, n_avail))
    free_centroids = [centroid_from_raw(r["coords"]) for r in drop_type_rooms]
    avail_centroids = [centroid_from_parsed(r) for _, r in available]

    for i, (fcx, fcy) in enumerate(free_centroids):
        for j, (acx, acy) in enumerate(avail_centroids):
            dx = fcx - acx
            dy = fcy - acy
            cost[i, j] = math.sqrt(dx * dx + dy * dy)

    row_ind, col_ind = linear_sum_assignment(cost)

    result: dict[int, int] = {}
    for row, col in zip(row_ind, col_ind):
        rid = drop_type_rooms[row].get("rid")
        out_idx = available[col][0]
        result[rid] = out_idx

    return result
