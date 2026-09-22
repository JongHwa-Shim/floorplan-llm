"""R_count_total + R_count_type 보상함수 모듈.

방 개수 조건 충족도를 측정하는 보상.

R_count_total:
    출력 방 전체 개수(outline 제외)가 조건과 일치하는지 이진 판정.
    신용할당: 없음 (sequence-level 보상).

R_count_type:
    지정된 모든 타입별 방 개수의 일치 여부.
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

    Mod Record: metadata가 모델 시점으로 재구성되면서 ROOM_SUMMARY의 <TOTAL>이
    drop된 경우(drop_room_summary_total) total_rooms=None이 들어온다. 이 경우
    모델은 N에 대한 신호를 받지 못했으므로 채점하지 않고 만점(1.0)을 반환한다.

    Args:
        parsed: parse_output_tokens()의 반환값.
        metadata: 입력 조건 메타데이터. 키:
            - total_rooms (int|None): outline 제외 전체 방 개수.
                None이면 ROOM_SUMMARY의 <TOTAL>이 drop되어 채점 비활성.

    Returns:
        1.0 (일치 또는 채점 비활성) 또는 0.0 (불일치).

    Raises:
        없음.
    """
    # Mod Record: 형식 오류와 방 개수는 독립적으로 평가한다.
    if not parsed.rooms:
        return 0.0

    expected_total = metadata.get("total_rooms")
    if expected_total is None:
        # 입력 프롬프트에 <TOTAL>이 없었으므로 채점 대상 아님
        return 1.0

    actual_total = sum(1 for r in parsed.rooms if r.room_type != "outline")
    return 1.0 if actual_total == expected_total else 0.0


def compute_count_type_reward(
    parsed: "ParsedFloorplan",
    metadata: dict,
) -> float:
    """지정된 모든 타입별 방 개수가 맞을 때만 1을 반환한다.

    Mod Record: metadata.type_counts는 모델 시점으로 재구성되어 ROOM_SUMMARY에서
    drop된 타입은 키에서 제외된다. 따라서 expected_counts에 없는 타입은 모델이 못 본
    조건이므로 채점 대상에서 제외한다(이전 구현은 모델이 출력한 타입까지 합집합으로
    순회해서 drop된 타입을 출력하면 부당하게 0점을 부여했다).

    Args:
        parsed: parse_output_tokens()의 반환값.
        metadata: 입력 조건 메타데이터. 키:
            - type_counts (dict[str, int]): ROOM_SUMMARY에 노출된 타입별 방 개수.

    Returns:
        모든 노출 타입의 개수가 맞으면 1.0, 하나라도 다르면 0.0. 조건이 없으면 1.0.

    Raises:
        없음.
    """
    # Mod Record: 형식 오류와 종류별 방 개수는 독립적으로 평가한다.
    if not parsed.rooms:
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

    # 생략된 종류는 채점하지 않고, 명시된 0개 조건도 엄격히 검사한다.
    return float(all(actual_counts.get(room_type, 0) == count
                     for room_type, count in expected_counts.items()))
