"""R_orthogonality + R_no_overlap 보상함수 모듈.

직각도 및 방 겹침 여부를 평가하는 기하학 기반 보상.

R_orthogonality:
    각 방 폴리곤의 꼭짓점이 직각(내적 ≈ 0)인 비율.
    비직각 꼭짓점의 좌표 토큰을 오류로 표시 (신용할당).

R_no_overlap:
    전체 방 면적 대비 겹치는 면적 비율.
    겹치는 교집합 폴리곤의 꼭짓점에 가장 가까운 원본 좌표 토큰을 오류로 표시 (신용할당).
    양쪽 방 모두에 오류 마킹.

의존성: shapely>=2.0.0
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.training.rl.rewards.parser import ParsedFloorplan, ParsedRoom

logger = logging.getLogger(__name__)

# 직각 판정 허용 오차 (정수 좌표 기반)
_ORTHOGONALITY_TOL = 1e-3


def compute_orthogonality_reward(
    parsed: "ParsedFloorplan",
) -> tuple[float, list[int]]:
    """각 방 꼭짓점의 직각도 비율을 반환한다.

    각 꼭짓점에서 인접 두 변의 벡터 내적이 0에 가까우면 직각으로 판정한다.
    outline 포함 모든 방을 평가한다.

    Args:
        parsed: parse_output_tokens()의 반환값.

    Returns:
        tuple:
            - reward: [0, 1] 범위. 전체 꼭짓점 중 직각 비율.
            - error_indices: 비직각 꼭짓점의 X 토큰 인덱스 리스트.
    """
    if not parsed.success or not parsed.rooms:
        return 0.0, []

    total_vertices = 0
    right_angle_count = 0
    error_indices: list[int] = []

    for room in parsed.rooms:
        if len(room.coords) < 3:
            continue

        n = len(room.coords)
        for i in range(n):
            # 현재 꼭짓점과 전후 꼭짓점
            prev_v = room.coords[(i - 1) % n]
            curr_v = room.coords[i]
            next_v = room.coords[(i + 1) % n]

            # 두 인접 벡터
            v1 = (prev_v[0] - curr_v[0], prev_v[1] - curr_v[1])
            v2 = (next_v[0] - curr_v[0], next_v[1] - curr_v[1])

            # 영벡터 방어
            len1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
            len2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
            if len1 < 1e-9 or len2 < 1e-9:
                continue

            # 정규화 후 내적 계산 (직각이면 0)
            dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2)
            total_vertices += 1

            if abs(dot) <= _ORTHOGONALITY_TOL:
                right_angle_count += 1
            else:
                # 비직각: 현재 꼭짓점(i)의 X 토큰 인덱스를 오류로 기록
                if i < len(room.coord_token_indices):
                    error_indices.append(room.coord_token_indices[i])
                    error_indices.append(room.coord_token_indices[i] + 1)  # Y 토큰

    if total_vertices == 0:
        return 0.0, []

    reward = right_angle_count / total_vertices
    return reward, error_indices


def compute_no_overlap_reward(
    parsed: "ParsedFloorplan",
) -> tuple[float, list[int]]:
    """방 간 겹침 면적 비율 기반 보상을 반환한다.

    outline을 제외한 방들 사이에 shapely Polygon intersection으로 겹침 면적을 계산한다.
    겹치는 영역의 꼭짓점에 가장 가까운 원본 방 좌표 토큰을 오류로 마킹한다.

    Args:
        parsed: parse_output_tokens()의 반환값.

    Returns:
        tuple:
            - reward: [0, 1] 범위. 1.0이면 겹침 없음.
            - error_indices: 겹침 관련 오류 토큰 인덱스 리스트.
    """
    if not parsed.success:
        return 0.0, []

    try:
        from shapely.geometry import Polygon
    except ImportError:
        logger.warning("shapely 미설치. no_overlap 보상 계산 불가.")
        return 1.0, []

    # outline 제외 방들의 Polygon 생성
    non_outline_rooms = [r for r in parsed.rooms if r.room_type != "outline"]
    if len(non_outline_rooms) < 2:
        return 1.0, []

    # shapely Polygon 리스트 생성 (3개 이상의 유효한 꼭짓점 필요)
    polys: list[Polygon | None] = []
    for room in non_outline_rooms:
        if len(room.coords) >= 3:
            try:
                poly = Polygon(room.coords)
                polys.append(poly if poly.is_valid and poly.area > 0 else None)
            except Exception:
                polys.append(None)
        else:
            polys.append(None)

    total_area = sum(p.area for p in polys if p is not None)
    if total_area <= 0:
        return 1.0, []

    overlap_area = 0.0
    error_indices: list[int] = []

    # 모든 방 쌍에 대해 겹침 계산
    n = len(non_outline_rooms)
    for i in range(n):
        for j in range(i + 1, n):
            if polys[i] is None or polys[j] is None:
                continue
            try:
                intersection = polys[i].intersection(polys[j])
            except Exception:
                continue

            if intersection.is_empty or intersection.area <= 0:
                continue

            overlap_area += intersection.area

            # 겹치는 영역의 꼭짓점에 가장 가까운 원본 좌표 토큰 인덱스를 오류로 마킹
            # 양쪽 방(i, j) 모두에 대해 처리
            try:
                inter_coords = list(intersection.exterior.coords)
                for room_idx in (i, j):
                    room = non_outline_rooms[room_idx]
                    for inter_pt in inter_coords:
                        nearest_idx = _find_nearest_coord_index(room.coords, inter_pt)
                        if nearest_idx >= 0 and nearest_idx < len(room.coord_token_indices):
                            tok_idx = room.coord_token_indices[nearest_idx]
                            error_indices.append(tok_idx)
                            error_indices.append(tok_idx + 1)  # Y 토큰
            except Exception:
                pass

    # 겹침 비율로 보상 계산
    overlap_ratio = min(overlap_area / total_area, 1.0)
    reward = 1.0 - overlap_ratio

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
