"""R_count_total + R_count_type 보상함수 모듈.

방 개수 조건 충족도를 측정하는 보상.

R_count_total:
    출력 방 전체 개수(outline 제외)가 조건과 일치하는지 이진 판정.
    신용할당: 없음 (sequence-level 보상).

R_count_type:
    타입별 방 개수 정확도의 평균.
    각 타입에 대해 min(출력수, 조건수) / max(출력수, 조건수) 비율로 계산.
    신용할당: 없음 (sequence-level 보상).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.training.rl.rewards.parser import ParsedFloorplan

logger = logging.getLogger(__name__)


def compute_count_total_reward(
    parsed: "ParsedFloorplan",
    metadata: dict,
) -> float:
    """전체 방 개수 일치 여부를 이진값으로 반환한다.

    outline을 제외한 방 개수가 metadata.total_rooms와 일치하면 1.0.

    Args:
        parsed: parse_output_tokens()의 반환값.
        metadata: 입력 조건 메타데이터. 키:
            - total_rooms (int): outline 제외 전체 방 개수.

    Returns:
        1.0 (일치) 또는 0.0 (불일치).
    """
    if not parsed.success or not parsed.rooms:
        return 0.0

    expected_total = metadata.get("total_rooms", 0)
    actual_total = sum(1 for r in parsed.rooms if r.room_type != "outline")

    return 1.0 if actual_total == expected_total else 0.0


def compute_count_type_reward(
    parsed: "ParsedFloorplan",
    metadata: dict,
) -> float:
    """타입별 방 개수 정확도 평균을 반환한다.

    각 타입에 대해 min(출력, 조건) / max(출력, 조건) 비율을 계산하고 평균한다.
    조건에 없는 타입이 출력에 있으면 해당 타입 점수 0.

    Args:
        parsed: parse_output_tokens()의 반환값.
        metadata: 입력 조건 메타데이터. 키:
            - type_counts (dict[str, int]): 타입별 방 개수.

    Returns:
        [0, 1] 범위. 타입별 정확도 평균.
    """
    if not parsed.success or not parsed.rooms:
        return 0.0

    expected_counts: dict[str, int] = metadata.get("type_counts", {})
    if not expected_counts:
        return 1.0  # 타입 조건 없으면 만점

    # 출력 방 타입별 집계 (outline 제외)
    actual_counts: dict[str, int] = {}
    for room in parsed.rooms:
        if room.room_type == "outline":
            continue
        actual_counts[room.room_type] = actual_counts.get(room.room_type, 0) + 1

    # 조건에 있는 모든 타입에 대해 정확도 계산
    scores: list[float] = []
    all_types = set(expected_counts.keys()) | set(actual_counts.keys())

    for t in all_types:
        exp = expected_counts.get(t, 0)
        act = actual_counts.get(t, 0)

        if exp == 0 and act == 0:
            continue
        max_val = max(exp, act)
        if max_val == 0:
            continue

        scores.append(min(exp, act) / max_val)

    if not scores:
        return 1.0

    return sum(scores) / len(scores)
