"""방 합집합의 외곽선 내부 피복 비율을 임계값과 비교하는 이진 보상."""

from __future__ import annotations

from typing import TYPE_CHECKING

from shapely.geometry import Polygon
from shapely.ops import unary_union

if TYPE_CHECKING:
    from src.training.rl.rewards.parser import ParsedFloorplan


def compute_coverage_reward(parsed: "ParsedFloorplan", threshold: float = 0.774) -> float:
    """외곽선 내부의 피복 비율이 임계값 이상이면 1을 반환한다.

    Mod Record: 원시 면적 비율을 보상으로 사용하던 방식을 논문의 이진 기준으로
    바꾼다. 기존 폴리곤 합집합 연산을 유지하며, 래스터화나 최적화 루프는 없다.

    Args:
        parsed: 파싱된 생성 평면도.
        threshold: 실평면도 평균에서 정한 피복 비율 기준.

    Returns:
        임계값 충족 시 1.0, 미충족 또는 유효하지 않은 폴리곤이면 0.0.

    Raises:
        ValueError: threshold가 0과 1 사이가 아닐 때.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("coverage threshold는 0과 1 사이여야 합니다.")
    # Mod Record: 형식 실패와 별개로 복원된 외곽선과 방의 피복을 검사한다.
    outline_room = next((room for room in parsed.rooms if room.room_type == "outline"), None)
    if outline_room is None or len(outline_room.coords) < 3:
        return 0.0
    outline = Polygon(outline_room.coords)
    if not outline.is_valid or outline.area <= 0:
        return 0.0
    polygons = []
    for room in parsed.rooms:
        if room.room_type == "outline":
            continue
        if len(room.coords) < 3:
            return 0.0
        polygon = Polygon(room.coords)
        if not polygon.is_valid or polygon.area <= 0:
            return 0.0
        polygons.append(polygon)
    if not polygons:
        return 0.0
    covered_area = outline.intersection(unary_union(polygons)).area
    return float(covered_area / outline.area >= threshold)
