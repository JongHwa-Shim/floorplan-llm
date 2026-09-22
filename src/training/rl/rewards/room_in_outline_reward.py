"""방과 현관문 폴리곤 전체가 외곽선에 포함되는지 평가한다."""

from __future__ import annotations

from typing import TYPE_CHECKING
from shapely.geometry import Point, Polygon, box

if TYPE_CHECKING:
    from src.training.rl.rewards.parser import ParsedFloorplan


def compute_room_in_outline_reward(parsed: "ParsedFloorplan") -> tuple[float, list[int]]:
    """모든 방과 현관문이 외곽선 내부 또는 경계에 있으면 1을 반환한다.

    Mod Record: 면적 비율 평균을 포함 여부로 대체한다. 폴리곤 전체의 covers
    검사로 오목한 외곽선을 가로지르는 변도 검출한다. 마스크는 논문대로
    외부 꼭짓점에만 부여하며, 별도 outline_in_room 보상을 요구하지 않는다.
    현관문의 cx/cy는 중심 좌표, w/h는 크기로 해석한다.

    Args:
        parsed: 파싱된 생성 평면도.

    Returns:
        이진 보상과 외곽선 밖 방 꼭짓점/현관문 위치·크기의 좌표 토큰 인덱스.

    Raises:
        없음. 유효하지 않은 폴리곤은 실패로 처리한다.
    """
    # Mod Record: 전체 형식 성공 여부 대신 필요한 외곽선·방·문 기하를 검사한다.
    outline_room = next((room for room in parsed.rooms if room.room_type == "outline"), None)
    if outline_room is None or len(outline_room.coords) < 3:
        return 0.0, []
    outline = Polygon(outline_room.coords)
    if not outline.is_valid or outline.area <= 0:
        return 0.0, []
    satisfied = True
    errors = []
    for room in parsed.rooms:
        if room.room_type == "outline":
            continue
        if len(room.coords) < 3:
            satisfied = False
            continue
        polygon = Polygon(room.coords)
        contained = polygon.is_valid and polygon.area > 0 and outline.covers(polygon)
        satisfied = satisfied and contained
        if not contained:
            for vertex, coord in enumerate(room.coords):
                if vertex < len(room.coord_token_indices) and not outline.covers(Point(coord)):
                    index = room.coord_token_indices[vertex]
                    errors.extend((index, index + 1))
    door = parsed.front_door
    if door is not None:
        cx, cy, width, height = (door[key] for key in ("cx", "cy", "w", "h"))
        corners = ((cx - width / 2, cy - height / 2),
                   (cx + width / 2, cy - height / 2),
                   (cx + width / 2, cy + height / 2),
                   (cx - width / 2, cy + height / 2))
        contained = width > 0 and height > 0 and outline.covers(box(*corners[0], *corners[2]))
        satisfied = satisfied and contained
        if not contained:
            indices = parsed.front_door_token_indices
            if not outline.covers(Point(cx, cy)):
                errors.extend(indices[:2])
            if width <= 0 or height <= 0 or any(not outline.covers(Point(corner)) for corner in corners):
                errors.extend(indices[2:4])
    return float(satisfied), sorted(set(errors))
