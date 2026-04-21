"""R_format 보상함수 모듈.

출력 시퀀스의 구조적 올바름을 검증하는 Hard Gate 보상.
R_format < 1.0이면 모든 다른 보상도 0으로 강제된다.

검증 항목:
    Level 0: <OUTPUT>...<END_OUTPUT> 래퍼 존재
    Level 1: FRONT_DOOR 블록 형식
    Level 2: ROOM 블록 (TYPE + X/Y 교대 + 짝수쌍 + 최소 4쌍)
    Level 3: DOOR 블록 형식

오류 토큰 신용할당:
    X/Y 교대가 깨진 위치, ROOM 미닫힘, SEP_DOOR 누락 등의 인덱스를 반환.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.training.grpo.rewards.parser import ParsedFloorplan

logger = logging.getLogger(__name__)


def compute_format_reward(
    parsed: "ParsedFloorplan",
) -> tuple[float, list[int]]:
    """출력 포맷 올바름 보상을 계산한다.

    Hard gate: 포맷 오류가 있으면 0.0 반환.
    성공 조건:
        - level == 3 (완전 파싱 성공)
        - 최소 1개의 방 블록 존재 (outline 포함)
        - error_indices 없음

    Args:
        parsed: parse_output_tokens()의 반환값.

    Returns:
        tuple:
            - reward: 1.0 (성공) 또는 0.0 (실패).
            - error_indices: 오류 토큰 인덱스 리스트 (신용할당용).
    """
    # 파싱 실패 (level 0, 1, 2)는 모두 포맷 오류
    if not parsed.success or parsed.level < 3:
        return 0.0, parsed.error_indices

    # 최소 outline + 1개 방 필요
    if len(parsed.rooms) < 2:
        all_room_indices = _collect_room_token_indices(parsed)
        return 0.0, all_room_indices

    return 1.0, []


def _collect_room_token_indices(parsed: "ParsedFloorplan") -> list[int]:
    """모든 방 블록의 토큰 인덱스를 수집한다.

    Args:
        parsed: ParsedFloorplan 인스턴스.

    Returns:
        모든 방 블록 토큰 인덱스 리스트.
    """
    indices: list[int] = []
    for room in parsed.rooms:
        indices.extend(range(room.block_start, room.block_end + 1))
    return indices
