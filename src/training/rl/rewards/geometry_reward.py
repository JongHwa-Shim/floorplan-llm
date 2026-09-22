"""전체 꼭짓점 직교성과 비외곽선 방 사이의 내부 겹침을 평가한다."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from shapely.geometry import Point, Polygon

if TYPE_CHECKING:
    from src.training.rl.rewards.parser import ParsedFloorplan

_ORTHOGONALITY_TOL = 1e-3


def compute_orthogonality_reward(parsed: "ParsedFloorplan") -> tuple[float, list[int]]:
    """외곽선을 포함한 모든 꼭짓점이 직각일 때만 1을 반환한다.

    Args:
        parsed: 파싱된 생성 평면도.

    Returns:
        이진 보상과 직각이 아닌 꼭짓점의 X/Y 토큰 인덱스.

    Raises:
        없음.
    """
    # Mod Record: 형식 오류는 직교성과 독립적이며, 훈련 게이트는 호출부에서 적용한다.
    if not parsed.rooms:
        return 0.0, []
    satisfied = True
    errors = []
    for room in parsed.rooms:
        count = len(room.coords)
        if count < 3:
            satisfied = False
            continue
        for i, current in enumerate(room.coords):
            previous, following = room.coords[(i - 1) % count], room.coords[(i + 1) % count]
            ax, ay = previous[0] - current[0], previous[1] - current[1]
            bx, by = following[0] - current[0], following[1] - current[1]
            length_product = math.hypot(ax, ay) * math.hypot(bx, by)
            # 길이 0인 변도 직각을 이루지 못하므로 실패시킨다.
            if length_product == 0 or abs(ax * bx + ay * by) > _ORTHOGONALITY_TOL * length_product:
                satisfied = False
                if i < len(room.coord_token_indices):
                    errors.extend((room.coord_token_indices[i], room.coord_token_indices[i] + 1))
    return float(satisfied), sorted(set(errors))


def compute_no_overlap_reward(parsed: "ParsedFloorplan") -> tuple[float, list[int]]:
    """모든 비외곽선 방 쌍의 내부 겹침이 없을 때만 1을 반환한다.

    Mod Record: 면적 비율 대신 이진 판정을 사용한다. 경계 공유는 허용하며,
    다른 방 내부에 있는 꼭짓점만 마스킹하는 기존 책임 범위를 유지한다.

    Args:
        parsed: 파싱된 생성 평면도.

    Returns:
        이진 보상과 다른 방 내부에 있는 꼭짓점의 X/Y 토큰 인덱스.

    Raises:
        없음. 유효하지 않은 폴리곤은 실패로 처리한다.
    """
    # Mod Record: 복원된 방의 겹침을 형식 보상과 독립적으로 검사한다.
    if not parsed.rooms:
        return 0.0, []
    rooms = [room for room in parsed.rooms if room.room_type != "outline"]
    polygons = []
    satisfied = True
    for room in rooms:
        try:
            polygon = Polygon(room.coords)
            valid = polygon.is_valid and polygon.area > 0
        except (ValueError, TypeError):
            valid = False
        polygons.append(polygon if valid else None)
        satisfied = satisfied and valid
    errors = []
    for i, first in enumerate(polygons):
        if first is None:
            continue
        for j in range(i + 1, len(polygons)):
            second = polygons[j]
            if second is None or first.intersection(second).area <= 0:
                continue
            satisfied = False
            for room, other in ((rooms[i], second), (rooms[j], first)):
                for vertex, coord in enumerate(room.coords):
                    if vertex < len(room.coord_token_indices) and other.contains(Point(coord)):
                        index = room.coord_token_indices[vertex]
                        errors.extend((index, index + 1))
    return float(satisfied), sorted(set(errors))
