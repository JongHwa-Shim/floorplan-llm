"""R_spatial 보상함수 모듈.

공간 관계(8방위 방향) 조건 충족도를 측정하는 보상.

두 방의 무게중심 벡터를 8방위 각도로 분류하고,
입력 조건의 SP(Spatial Relation) 방향과 일치하는 비율을 반환한다.

헝가리안 매칭은 connectivity_reward와 동일한 함수를 재사용한다.
신용할당: 없음 (sequence-level 보상).

8방위 정의 (token_definitions.py SPATIAL_DIRECTIONS):
    right, right-below, below, left-below,
    left, left-above, above, right-above
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.training.grpo.rewards.parser import ParsedFloorplan

logger = logging.getLogger(__name__)

# 8방위 각도 범위 (degrees, 반시계 방향, East=0)
# 각 방위는 45° 씩 분할
_DIRECTIONS = [
    "right",        # -22.5 ~ 22.5
    "right-below",  # -67.5 ~ -22.5 (또는 292.5 ~ 337.5)
    "below",        # -112.5 ~ -67.5 (또는 247.5 ~ 292.5)
    "left-below",   # -157.5 ~ -112.5 (또는 202.5 ~ 247.5)
    "left",         # ±157.5 ~ ±180
    "left-above",   # 112.5 ~ 157.5
    "above",        # 67.5 ~ 112.5
    "right-above",  # 22.5 ~ 67.5
]


def compute_spatial_reward(
    parsed: "ParsedFloorplan",
    metadata: dict,
) -> float:
    """공간 관계 조건 충족도를 반환한다.

    헝가리안 매칭(connectivity_reward와 동일)으로 입력-출력 방을 연결하고,
    spatial 조건의 방향과 실제 무게중심 방향을 비교한다.

    Args:
        parsed: parse_output_tokens()의 반환값.
        metadata: 입력 조건 메타데이터. 키:
            - spatial (list[dict]): 공간 관계 조건.
                [{rid_a, rid_b, direction}]
            - rooms (list[dict]): 입력 방 정보.
            - type_counts (dict[str, int]): 타입별 방 개수.

    Returns:
        [0, 1] 범위. 방향 조건 충족 비율.
    """
    if not parsed.success or not parsed.rooms:
        return 0.0

    spatial_conditions = metadata.get("spatial", [])
    if not spatial_conditions:
        return 1.0  # 공간 조건 없으면 만점

    # 헝가리안 매칭 재사용 (connectivity_reward와 동일)
    from src.training.grpo.rewards.connectivity_reward import (
        _hungarian_match,
        _compute_centroid_from_parsed,
    )

    rid_to_room_idx = _hungarian_match(parsed, metadata)
    if not rid_to_room_idx:
        return 0.0

    # outline 제외 출력 방 리스트
    output_rooms = [r for r in parsed.rooms if r.room_type != "outline"]

    satisfied = 0
    total = 0

    for sp in spatial_conditions:
        rid_a = sp.get("rid_a")
        rid_b = sp.get("rid_b")
        expected_dir = sp.get("direction", "")

        if rid_a is None or rid_b is None:
            continue

        room_idx_a = rid_to_room_idx.get(rid_a)
        room_idx_b = rid_to_room_idx.get(rid_b)

        if room_idx_a is None or room_idx_b is None:
            total += 1
            continue

        if room_idx_a >= len(output_rooms) or room_idx_b >= len(output_rooms):
            total += 1
            continue

        room_a = output_rooms[room_idx_a]
        room_b = output_rooms[room_idx_b]

        cx_a, cy_a = _compute_centroid_from_parsed(room_a)
        cx_b, cy_b = _compute_centroid_from_parsed(room_b)

        # A → B 방향 벡터
        dx = cx_b - cx_a
        dy = cy_b - cy_a

        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            total += 1
            continue

        # 실제 방향 분류
        # 이미지 좌표계 (y 아래로 증가)에서 아래 방향이 below
        # atan2(dy, dx): dy>0이면 아래 방향
        actual_dir = _vector_to_direction(dx, dy)
        total += 1

        if actual_dir == expected_dir:
            satisfied += 1

    if total == 0:
        return 1.0

    return satisfied / total


def _vector_to_direction(dx: float, dy: float) -> str:
    """방향 벡터를 8방위 문자열로 변환한다.

    이미지 좌표계 기준 (y축이 아래로 증가):
        dy > 0 → 아래 방향 (below)
        dy < 0 → 위 방향 (above)

    Args:
        dx: X 방향 성분.
        dy: Y 방향 성분 (양수면 아래).

    Returns:
        8방위 문자열 중 하나.
    """
    # atan2로 각도 계산 (라디안, -π ~ π)
    # 표준 수학 좌표계이지만 y 방향을 뒤집어서 이미지 좌표계로 처리
    angle_rad = math.atan2(dy, dx)  # dy > 0이면 아래(-y 방향 포함)
    angle_deg = math.degrees(angle_rad)

    # -180 ~ 180 범위를 8방위로 분류
    # Right=0°, 반시계: above(90°), left(±180°), below(-90°)
    # 이미지 좌표계: below는 dy>0 즉 angle > 0
    if -22.5 <= angle_deg < 22.5:
        return "right"
    elif 22.5 <= angle_deg < 67.5:
        return "right-below"   # dy>0이므로 아래-오른쪽
    elif 67.5 <= angle_deg < 112.5:
        return "below"
    elif 112.5 <= angle_deg < 157.5:
        return "left-below"
    elif angle_deg >= 157.5 or angle_deg < -157.5:
        return "left"
    elif -157.5 <= angle_deg < -112.5:
        return "left-above"
    elif -112.5 <= angle_deg < -67.5:
        return "above"
    else:  # -67.5 <= angle_deg < -22.5
        return "right-above"
