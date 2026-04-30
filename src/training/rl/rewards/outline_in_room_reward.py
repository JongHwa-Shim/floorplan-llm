"""R_outline_in_room 보상함수 모듈.

outline 폴리곤의 꼭짓점이 비-outline 방의 내부에 포함되는지 평가한다.

R_room_in_outline의 쌍대적 검사로, 케이스 B를 검출한다.

케이스 B: 방 꼭짓점은 모두 outline 내부에 위치하지만 방 edge가 outline 경계를
가로질러 방이 실질적으로 outline을 벗어나는 경우. 이때 outline의 오목한(reflex)
꼭짓점이 방 폴리곤 내부에 포함되는 기하학적 특징이 나타난다.

R_outline_in_room:
    각 outline 꼭짓점에 대해 비-outline 방 폴리곤이 해당 꼭짓점을 엄격하게
    포함(strict interior)하는지 확인한다.
    포함된 경우 해당 방에서 outline 꼭짓점에 가장 가까운 방 꼭짓점을 오류로 마킹.

    보상 = 1 - (방에 포함된 outline 꼭짓점 수 / 전체 outline 꼭짓점 수)

의존성: shapely>=2.0.0
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.training.rl.rewards.parser import ParsedFloorplan

logger = logging.getLogger(__name__)


def compute_outline_in_room_reward(
    parsed: "ParsedFloorplan",
) -> tuple[float, list[int]]:
    """outline 꼭짓점이 방 내부에 포함되는 비율에 기반한 보상을 반환한다.

    각 outline 꼭짓점에 대해 비-outline 방 폴리곤이 해당 꼭짓점을 엄격하게
    포함하는지 확인한다. 포함된 경우 방 폴리곤에서 해당 outline 꼭짓점에
    가장 가까운 방 꼭짓점을 오류로 마킹한다.

    Args:
        parsed: parse_output_tokens()의 반환값.

    Returns:
        tuple:
            - reward: [0, 1] 범위.
                1.0이면 모든 outline 꼭짓점이 방 외부에 위치 (정상).
                값이 낮을수록 더 많은 outline 꼭짓점이 방 내부에 포함됨.
            - error_indices: outline 꼭짓점을 포함하는 방의 해당 좌표 토큰 인덱스 리스트.

    Raises:
        없음 (모든 예외는 내부에서 처리하며 보수적 반환값 사용).
    """
    if not parsed.success or not parsed.rooms:
        return 1.0, []

    try:
        from shapely.geometry import Point, Polygon
    except ImportError:
        logger.warning("shapely 미설치. outline_in_room 보상 계산 불가.")
        return 1.0, []

    outline_room = next((r for r in parsed.rooms if r.room_type == "outline"), None)
    if outline_room is None or len(outline_room.coords) < 3:
        return 1.0, []

    non_outline_rooms = [r for r in parsed.rooms if r.room_type != "outline"]
    if not non_outline_rooms:
        return 1.0, []

    # 방 폴리곤 사전 생성 (반복 연산 방지)
    room_poly_pairs: list[tuple] = []
    for room in non_outline_rooms:
        if len(room.coords) < 3:
            room_poly_pairs.append((room, None))
            continue
        try:
            poly = Polygon(room.coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            room_poly_pairs.append((room, poly if not poly.is_empty and poly.area > 0 else None))
        except Exception:
            room_poly_pairs.append((room, None))

    total_verts = len(outline_room.coords)
    trapped_count = 0
    error_indices: list[int] = []

    for outline_coord in outline_room.coords:
        pt = Point(outline_coord)
        for room, room_poly in room_poly_pairs:
            if room_poly is None:
                continue
            try:
                # contains()는 경계를 제외한 엄격한 내부만 검사
                # → 방 경계와 outline 꼭짓점이 맞닿는 정상 케이스는 false positive 없음
                if room_poly.contains(pt):
                    trapped_count += 1
                    nearest_idx = _find_nearest_coord_index(room.coords, outline_coord)
                    if 0 <= nearest_idx < len(room.coord_token_indices):
                        tok_idx = room.coord_token_indices[nearest_idx]
                        error_indices.append(tok_idx)
                        error_indices.append(tok_idx + 1)  # Y 토큰
                    break  # 한 방에 포함됨을 확인했으면 다음 outline 꼭짓점으로
            except Exception:
                continue

    if total_verts == 0:
        return 1.0, []

    reward = 1.0 - trapped_count / total_verts
    return reward, sorted(set(error_indices))


def _find_nearest_coord_index(
    coords: list[tuple[int, int]],
    point: tuple[float, float],
) -> int:
    """point에 가장 가까운 꼭짓점 인덱스를 반환한다.

    Args:
        coords: 꼭짓점 좌표 리스트.
        point: 비교할 점 (x, y).

    Returns:
        가장 가까운 꼭짓점의 인덱스. coords가 비어있으면 -1.
    """
    if not coords:
        return -1

    min_dist_sq = float("inf")
    nearest = 0
    px, py = point

    for i, (cx, cy) in enumerate(coords):
        dist_sq = (cx - px) ** 2 + (cy - py) ** 2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            nearest = i

    return nearest
