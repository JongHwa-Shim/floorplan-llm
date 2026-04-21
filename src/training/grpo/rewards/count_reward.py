"""R_count 보상함수 모듈.

전체 방 개수 및 타입별 방 개수 일치도를 측정하는 보상.

R_count_total: 전체 방 개수 일치 여부 (이진: 0 or 1)
R_count_type:  타입별 방 개수 편차 기반 연속 점수 [0, 1]

신용할당: 없음 (sequence-level 보상).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.training.grpo.rewards.parser import ParsedFloorplan

logger = logging.getLogger(__name__)


def compute_count_total_reward(
    parsed: "ParsedFloorplan",
    metadata: dict,
) -> float:
    """전체 방 개수 일치 여부를 반환한다.

    outline을 제외한 방 수와 metadata의 total_rooms를 비교한다.

    Args:
        parsed: parse_output_tokens()의 반환값.
        metadata: 입력 조건 메타데이터. 키:
            - total_rooms (int): outline 제외 전체 방 개수.

    Returns:
        1.0 (일치) 또는 0.0 (불일치).
    """
    if not parsed.success:
        return 0.0

    # outline 제외 방 수 집계
    actual_count = sum(1 for r in parsed.rooms if r.room_type != "outline")
    expected_count = metadata.get("total_rooms", 0)

    return 1.0 if actual_count == expected_count else 0.0


def compute_count_type_reward(
    parsed: "ParsedFloorplan",
    metadata: dict,
) -> float:
    """타입별 방 개수 편차를 기반으로 연속 점수를 반환한다.

    각 타입에 대해 |actual - expected| / max(expected, 1)를 계산하고,
    전체 타입의 평균 정확도를 반환한다.

    Args:
        parsed: parse_output_tokens()의 반환값.
        metadata: 입력 조건 메타데이터. 키:
            - type_counts (dict[str, int]): 타입별 기대 방 개수.

    Returns:
        [0, 1] 범위의 연속 점수.
        expected_counts가 비어있으면 parsed.success에 따라 1.0 또는 0.0.
    """
    if not parsed.success:
        return 0.0

    expected_counts: dict[str, int] = metadata.get("type_counts", {})
    if not expected_counts:
        return 1.0  # 기대 타입 없으면 만점

    # outline 제외 실제 타입별 개수 집계
    actual_counts: dict[str, int] = {}
    for room in parsed.rooms:
        if room.room_type == "outline":
            continue
        actual_counts[room.room_type] = actual_counts.get(room.room_type, 0) + 1

    # 모든 기대 타입에 대해 정확도 계산
    accuracy_per_type: list[float] = []
    for room_type, expected_n in expected_counts.items():
        actual_n = actual_counts.get(room_type, 0)
        deviation = abs(actual_n - expected_n) / max(expected_n, 1)
        accuracy = max(0.0, 1.0 - deviation)
        accuracy_per_type.append(accuracy)

    # 예상치 않은 추가 타입 페널티 (기대에 없는 타입이 생성된 경우)
    for room_type, actual_n in actual_counts.items():
        if room_type not in expected_counts and actual_n > 0:
            accuracy_per_type.append(0.0)

    if not accuracy_per_type:
        return 1.0

    return sum(accuracy_per_type) / len(accuracy_per_type)
