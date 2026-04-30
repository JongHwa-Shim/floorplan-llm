"""R_coverage 보상함수 모듈.

outline 내부의 빈공간(어떤 방도 차지하지 않은 영역) 비율을 페널티로 환산하는 보상.

R_coverage는 R_room_in_outline의 쌍대(dual) 보상이다:
    - R_room_in_outline: 방 → outline 포함 (방이 outline 밖으로 삐져나가는지)
    - R_coverage:        outline → 방들 합집합 포함 (outline 내 빈공간이 있는지)
두 보상이 모두 1.0이어야 비로소 "$\\text{outline} = \\bigsqcup_i \\text{room}_i$"라는
평면도의 본질 제약이 강제된다. 단독 사용 시 reward hacking 여지가 남는다.

수식:
    $$R_{\\text{coverage}} = 1 - \\frac{\\text{area}(O \\setminus \\bigcup_i R_i)}{\\text{area}(O)}$$
    - $O$: outline 폴리곤
    - $R_i$: 비-outline 방 폴리곤

신용할당: 없음 (sequence-level 보상).
    빈공간 발생 책임 소재가 본질적으로 모호하다. 좌표가 잘못된 방이 원인일 수도,
    방 개수가 부족해서일 수도 있다. 또한 입력 조건에 좌표 노이즈가 들어가 있어
    "어느 좌표가 정답인지" 자체가 모호하므로 토큰 단위 페널티는 잘못된 시그널을
    줄 위험이 크다. 따라서 sequence-level만 사용한다.

의존성: shapely>=2.0.0
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.training.rl.rewards.parser import ParsedFloorplan

logger = logging.getLogger(__name__)


def compute_coverage_reward(
    parsed: "ParsedFloorplan",
) -> float:
    """outline 내부 빈공간 비율의 보수값(1 - empty_ratio)을 반환한다.

    outline 폴리곤에서 모든 비-outline 방의 합집합을 차집합으로 빼낸 영역(빈공간)의
    면적 비율을 계산하고, 이를 1에서 뺀 값을 보상으로 반환한다. 1.0이면 빈공간 없음
    (방들이 outline을 완전히 채움), 0.0이면 outline 전체가 빈공간(방이 없거나
    모든 방이 outline 밖에 있음).

    Args:
        parsed: parse_output_tokens()의 반환값.

    Returns:
        $[0, 1]$ 범위 스칼라. 빈공간이 적을수록 1에 가깝다.

    Raises:
        없음 (모든 예외는 내부에서 처리하며 보수적 반환값 사용).
    """
    if not parsed.success or not parsed.rooms:
        return 0.0

    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
    except ImportError:
        logger.warning("shapely 미설치. coverage 보상 계산 불가.")
        return 1.0

    # outline 폴리곤 생성 (rooms[0]이 outline이지만 명시 검색)
    outline_room = next((r for r in parsed.rooms if r.room_type == "outline"), None)
    if outline_room is None or len(outline_room.coords) < 3:
        return 0.0

    try:
        outline_poly = Polygon(outline_room.coords)
        if not outline_poly.is_valid:
            outline_poly = outline_poly.buffer(0)        # self-intersection 등 정리
        if outline_poly.is_empty or outline_poly.area <= 0:
            return 0.0
    except Exception:
        return 0.0

    # 비-outline 방 폴리곤 생성
    non_outline_rooms = [
        r for r in parsed.rooms
        if r.room_type != "outline" and len(r.coords) >= 3
    ]
    if not non_outline_rooms:
        # format_reward가 최소 outline + 1개 방 조건이지만 안전 가드
        return 0.0

    room_polys = []
    for room in non_outline_rooms:
        try:
            poly = Polygon(room.coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_empty and poly.area > 0:
                room_polys.append(poly)
        except Exception:
            continue

    if not room_polys:
        return 0.0

    # 모든 방의 합집합과 outline 차집합으로 빈공간 면적 계산
    try:
        union_rooms = unary_union(room_polys)
        empty = outline_poly.difference(union_rooms)
    except Exception:
        # 기하 연산 실패 시 보수적으로 0점 (페널티 발생)
        return 0.0

    empty_area = max(0.0, float(empty.area))            # 수치 오차로 음수 방지
    empty_ratio = empty_area / outline_poly.area
    empty_ratio = max(0.0, min(1.0, empty_ratio))       # [0, 1] 클램핑

    return 1.0 - empty_ratio
