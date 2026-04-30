"""R_room_in_outline 보상함수 모듈.

모든 비-outline 방 및 front door가 outline 폴리곤 경계 내에 포함되는지 평가한다.

R_room_in_outline:
    각 비-outline 방과 front door에 대해 intersection(target, outline).area / target.area
    비율을 계산하고 전체 평균을 보상으로 반환한다.

    front door 폴리곤 구성:
        토큰 인코딩 {cx, cy, w, h}에서 (cx, cy)를 left-top, (cx+w, cy+h)를 right-bottom으로
        하는 직사각형 4개 꼭짓점을 구성한다.

    신용할당 (케이스 A만 처리):
        outline 경계를 벗어난 방 꼭짓점을 직접 확인하여 해당 좌표 토큰을 오류로 마킹.
        front door는 left-top (cx, cy)와 right-bottom (cx+w, cy+h) 두 꼭짓점을 검사하여
        해당 토큰 쌍(cx/cy, w/h)을 마킹한다.
        케이스 B (꼭짓점은 모두 outline 내부이나 edge가 oultline을 가로지르는 경우)는
        발생 빈도가 낮으므로 여기서는 처리하지 않는다. 케이스 B 검출은
        outline_in_room_reward.py (R_outline_in_room)가 담당한다.

의존성: shapely>=2.0.0
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.training.rl.rewards.parser import ParsedFloorplan

logger = logging.getLogger(__name__)

# outline 포함 판정 허용 오차 (정수 좌표 기반, 경계 터치 등 수치 오차 허용)
_CONTAINMENT_TOL = 1e-4


def compute_room_in_outline_reward(
    parsed: "ParsedFloorplan",
) -> tuple[float, list[int]]:
    """모든 비-outline 방과 front door가 outline 경계 내에 포함되는 비율을 반환한다.

    outline을 제외한 각 방과 front door에 대해 outline 폴리곤과의 교집합 면적 비율
    $\\frac{\\text{area}(target \\cap outline)}{\\text{area}(target)}$ 을 계산하고 평균을 반환한다.

    front door는 토큰 {cx, cy, w, h}에서 4개 꼭짓점을 구성하여 검증한다:
    left-top $(cx, cy)$, right-top $(cx+w, cy)$, right-bottom $(cx+w, cy+h)$, left-bottom $(cx, cy+h)$.

    신용할당: outline 경계 밖에 위치한 방 꼭짓점을 직접 검사하여 마킹한다.
    front door는 left-top (cx, cy)와 right-bottom (cx+w, cy+h) 두 꼭짓점을 검사하여
    해당 토큰 쌍을 마킹한다. 경계 위(covers) 꼭짓점은 유효로 판정하여 false positive를 방지한다.

    Args:
        parsed: parse_output_tokens()의 반환값.

    Returns:
        tuple:
            - reward: [0, 1] 범위. 1.0이면 모든 방과 front door가 outline 내 완전 포함.
            - error_indices: outline 경계 밖에 위치한 꼭짓점의 좌표 토큰 인덱스 리스트.

    Raises:
        없음 (모든 예외는 내부에서 처리하며 보수적 반환값 사용).
    """
    if not parsed.success or not parsed.rooms:
        return 0.0, []

    try:
        from shapely.geometry import Point, Polygon
    except ImportError:
        logger.warning("shapely 미설치. room_in_outline 보상 계산 불가.")
        return 1.0, []

    # outline 방 찾기 (항상 rooms[0]이지만 명시적으로 검색)
    outline_room = next((r for r in parsed.rooms if r.room_type == "outline"), None)
    if outline_room is None or len(outline_room.coords) < 3:
        return 0.0, []

    # outline Polygon 생성
    try:
        outline_poly = Polygon(outline_room.coords)
        if not outline_poly.is_valid:
            outline_poly = outline_poly.buffer(0)
        if outline_poly.is_empty or outline_poly.area <= 0:
            return 0.0, []
    except Exception:
        return 0.0, []

    # 비-outline 방 목록
    non_outline_rooms = [r for r in parsed.rooms if r.room_type != "outline"]

    containment_scores: list[float] = []
    error_indices: list[int] = []

    for room in non_outline_rooms:
        if len(room.coords) < 3:
            containment_scores.append(0.0)
            error_indices.extend(room.coord_token_indices)
            continue

        try:
            room_poly = Polygon(room.coords)
            if not room_poly.is_valid:
                room_poly = room_poly.buffer(0)
            if room_poly.is_empty or room_poly.area <= 0:
                containment_scores.append(0.0)
                continue
        except Exception:
            containment_scores.append(0.0)
            continue

        # 포함 비율 계산
        try:
            intersection = outline_poly.intersection(room_poly)
            containment_ratio = float(intersection.area / room_poly.area)
        except Exception:
            containment_scores.append(0.0)
            error_indices.extend(room.coord_token_indices)
            continue

        containment_scores.append(containment_ratio)

        # 신용할당: outline 경계 밖에 위치한 꼭짓점을 직접 마킹 (케이스 A)
        # covers()는 경계 위 꼭짓점을 포함으로 판정 → 벽에 닿는 꼭짓점 false positive 방지
        if containment_ratio < 1.0 - _CONTAINMENT_TOL:
            for i, coord in enumerate(room.coords):
                if i >= len(room.coord_token_indices):
                    break
                try:
                    if not outline_poly.covers(Point(coord)):
                        tok_idx = room.coord_token_indices[i]
                        error_indices.append(tok_idx)
                        error_indices.append(tok_idx + 1)  # Y 토큰
                except Exception:
                    continue

    # front door 포함 검증
    # {cx, cy, w, h}에서 (cx,cy)=left-top, (cx+w,cy+h)=right-bottom으로 직사각형 구성
    fd = parsed.front_door
    if fd is not None:
        cx, cy, w, h = fd["cx"], fd["cy"], fd["w"], fd["h"]
        if w > 0 and h > 0:
            fd_coords = [(cx, cy), (cx + w, cy), (cx + w, cy + h), (cx, cy + h)]
            try:
                fd_poly = Polygon(fd_coords)
                if not fd_poly.is_valid:
                    fd_poly = fd_poly.buffer(0)
                if not fd_poly.is_empty and fd_poly.area > 0:
                    intersection = outline_poly.intersection(fd_poly)
                    fd_containment = float(intersection.area / fd_poly.area)
                    containment_scores.append(fd_containment)

                    # 신용할당: left-top (cx,cy)와 right-bottom (cx+w,cy+h) 검사
                    if fd_containment < 1.0 - _CONTAINMENT_TOL:
                        fd_tok = parsed.front_door_token_indices  # [cx_idx, cy_idx, w_idx, h_idx]
                        try:
                            if len(fd_tok) >= 2 and not outline_poly.covers(Point(cx, cy)):
                                error_indices.extend([fd_tok[0], fd_tok[1]])
                        except Exception:
                            pass
                        try:
                            if len(fd_tok) >= 4 and not outline_poly.covers(Point(cx + w, cy + h)):
                                error_indices.extend([fd_tok[2], fd_tok[3]])
                        except Exception:
                            pass
            except Exception:
                containment_scores.append(0.0)

    if not containment_scores:
        return 1.0, []

    reward = float(sum(containment_scores) / len(containment_scores))
    return reward, sorted(set(error_indices))
