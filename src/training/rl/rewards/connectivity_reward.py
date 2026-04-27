"""R_connectivity 보상함수 모듈.

헝가리안 알고리즘으로 출력 방과 입력 조건 방을 매핑하고,
각 EDGE 조건의 두 방 사이에 DOOR가 존재하는지 검증하는 보상.

신용할당: 없음 (sequence-level 보상).

알고리즘:
    1. 같은 타입 내에서 무게중심 거리 기반 scipy.optimize.linear_sum_assignment 수행
    2. 매핑된 방 쌍에 대해 EDGE 조건의 DOOR 존재 여부 기하학적 확인
       - DOOR 중심점이 두 방의 경계 근방에 위치하면 연결로 판정

의존성: scipy>=1.14.0
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.training.rl.rewards.parser import ParsedFloorplan, ParsedRoom

logger = logging.getLogger(__name__)

# DOOR가 두 방 경계에 존재하는지 판정하는 거리 허용 오차 (좌표 단위)
_DOOR_PROXIMITY_THRESHOLD = 20.0


def compute_connectivity_reward(
    parsed: "ParsedFloorplan",
    metadata: dict,
) -> float:
    """연결성(방 간 문 존재) 보상을 계산한다.

    헝가리안 매칭으로 출력 방과 입력 방을 연결하고,
    입력의 EDGE 조건에 따른 DOOR 존재 여부를 검증한다.

    Args:
        parsed: parse_output_tokens()의 반환값.
        metadata: 입력 조건 메타데이터. 키:
            - edges (list[dict]): 엣지 조건. 각 항목:
                {pair: [rid_a, rid_b], door: list[{x,y,w,h}]}
            - rooms (list[dict]): 입력 방 정보.
                [{rid, type, coords: list[int]}]
            - type_counts (dict[str, int]): 타입별 방 개수.

    Returns:
        [0, 1] 범위. DOOR 조건 충족 비율.
    """
    if not parsed.success or not parsed.rooms:
        return 0.0

    edges = metadata.get("edges", [])
    if not edges:
        return 1.0  # 연결 조건 없으면 만점

    # 헝가리안 매칭: 입력 RID → 출력 방 인덱스
    rid_to_room_idx = _hungarian_match(parsed, metadata)
    if not rid_to_room_idx:
        return 0.0

    # DOOR 존재 여부 검증
    satisfied = 0
    total_with_door = 0

    for edge in edges:
        if not edge.get("door"):
            continue  # 문 없는 엣지는 건너뜀
        total_with_door += 1

        pair = edge["pair"]
        if len(pair) < 2:
            continue

        rid_a, rid_b = pair[0], pair[1]
        room_idx_a = rid_to_room_idx.get(rid_a)
        room_idx_b = rid_to_room_idx.get(rid_b)

        if room_idx_a is None or room_idx_b is None:
            continue

        non_outline = [r for r in parsed.rooms if r.room_type != "outline"]
        if room_idx_a >= len(non_outline) or room_idx_b >= len(non_outline):
            continue

        room_a = non_outline[room_idx_a]
        room_b = non_outline[room_idx_b]

        # 두 방 사이에 DOOR가 존재하는지 확인
        if _has_door_between(room_a, room_b, parsed.doors):
            satisfied += 1

    if total_with_door == 0:
        return 1.0

    return satisfied / total_with_door


# ---------------------------------------------------------------------------
# 내부 헬퍼 함수
# ---------------------------------------------------------------------------

def _hungarian_match(
    parsed: "ParsedFloorplan",
    metadata: dict,
) -> dict[int, int]:
    """헝가리안 알고리즘으로 입력 RID → 출력 방 인덱스를 매핑한다.

    같은 타입 내에서 무게중심 거리 기반으로 최적 할당을 수행한다.
    출력 방에는 RID가 없으므로 타입 분류 후 매핑한다.

    Args:
        parsed: ParsedFloorplan 인스턴스.
        metadata: 입력 조건 메타데이터.

    Returns:
        {rid: output_room_index} 딕셔너리 (outline 제외 인덱스).
        매핑 실패 시 빈 딕셔너리.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        logger.warning("scipy 미설치. connectivity 보상 계산 불가.")
        return {}

    input_rooms = metadata.get("rooms", [])
    if not input_rooms:
        return {}

    # outline 제외 출력 방 리스트
    output_rooms = [r for r in parsed.rooms if r.room_type != "outline"]
    if not output_rooms:
        return {}

    # 입력 방 타입별 분류 (outline 제외)
    input_by_type: dict[str, list[dict]] = {}
    for room in input_rooms:
        if room.get("type") == "outline":
            continue
        t = room["type"]
        input_by_type.setdefault(t, []).append(room)

    # 출력 방 타입별 분류
    output_by_type: dict[str, list[tuple[int, "ParsedRoom"]]] = {}
    for idx, room in enumerate(output_rooms):
        t = room.room_type
        output_by_type.setdefault(t, []).append((idx, room))

    rid_to_output_idx: dict[int, int] = {}

    for room_type, in_rooms in input_by_type.items():
        out_rooms = output_by_type.get(room_type, [])
        if not out_rooms:
            continue

        n_in = len(in_rooms)
        n_out = len(out_rooms)

        # 비용 행렬: 입력 × 출력 무게중심 거리
        cost = [[0.0] * n_out for _ in range(n_in)]
        in_centroids = [_compute_centroid_from_raw(r["coords"]) for r in in_rooms]
        out_centroids = [_compute_centroid_from_parsed(out_rooms[j][1]) for j in range(n_out)]

        for i in range(n_in):
            for j in range(n_out):
                dx = in_centroids[i][0] - out_centroids[j][0]
                dy = in_centroids[i][1] - out_centroids[j][1]
                cost[i][j] = math.sqrt(dx * dx + dy * dy)

        # 헝가리안 알고리즘
        import numpy as np
        cost_np = np.array(cost)
        row_ind, col_ind = linear_sum_assignment(cost_np)

        for row, col in zip(row_ind, col_ind):
            rid = in_rooms[row]["rid"]
            out_idx = out_rooms[col][0]
            rid_to_output_idx[rid] = out_idx

    return rid_to_output_idx


def _compute_centroid_from_raw(coords: list[int]) -> tuple[float, float]:
    """flat 좌표 리스트 [x1,y1,x2,y2,...] 에서 무게중심을 계산한다.

    Args:
        coords: flat 정수 좌표 리스트.

    Returns:
        (cx, cy) 무게중심.
    """
    if not coords:
        return (0.0, 0.0)
    xs = coords[0::2]
    ys = coords[1::2]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _compute_centroid_from_parsed(room: "ParsedRoom") -> tuple[float, float]:
    """ParsedRoom 꼭짓점에서 무게중심을 계산한다.

    Args:
        room: ParsedRoom 인스턴스.

    Returns:
        (cx, cy) 무게중심.
    """
    if not room.coords:
        return (0.0, 0.0)
    xs = [c[0] for c in room.coords]
    ys = [c[1] for c in room.coords]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _has_door_between(
    room_a: "ParsedRoom",
    room_b: "ParsedRoom",
    doors: list,
) -> bool:
    """두 방 사이에 DOOR가 존재하는지 확인한다.

    DOOR 중심점이 두 방의 경계 근방에 위치하고,
    두 방의 폴리곤 모두와 가까운 경우 연결로 판정한다.

    Args:
        room_a: 첫 번째 방.
        room_b: 두 번째 방.
        doors: ParsedDoor 리스트.

    Returns:
        DOOR가 두 방 사이에 위치하면 True.
    """
    if not doors or not room_a.coords or not room_b.coords:
        return False

    for door in doors:
        if not door.is_valid:
            continue

        door_cx, door_cy = door.cx, door.cy

        # 문 중심점이 두 방의 경계 근방에 있는지 확인
        dist_a = _min_distance_to_polygon(door_cx, door_cy, room_a.coords)
        dist_b = _min_distance_to_polygon(door_cx, door_cy, room_b.coords)

        if dist_a <= _DOOR_PROXIMITY_THRESHOLD and dist_b <= _DOOR_PROXIMITY_THRESHOLD:
            return True

    return False


def _min_distance_to_polygon(
    px: float,
    py: float,
    coords: list[tuple[int, int]],
) -> float:
    """점 (px, py)에서 폴리곤 경계까지의 최소 거리를 계산한다.

    점이 폴리곤 내부에 있으면 경계까지 거리 0으로 처리한다.

    Args:
        px: 점의 X 좌표.
        py: 점의 Y 좌표.
        coords: 폴리곤 꼭짓점 리스트.

    Returns:
        폴리곤 경계까지의 최소 거리.
    """
    if not coords:
        return float("inf")

    min_dist = float("inf")
    n = len(coords)

    for i in range(n):
        ax, ay = coords[i]
        bx, by = coords[(i + 1) % n]

        # 점에서 선분까지 거리 계산
        dist = _point_to_segment_distance(px, py, ax, ay, bx, by)
        min_dist = min(min_dist, dist)

    return min_dist


def _point_to_segment_distance(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> float:
    """점 P에서 선분 AB까지의 최소 거리를 계산한다.

    Args:
        px, py: 점 P 좌표.
        ax, ay: 선분 시작점 A 좌표.
        bx, by: 선분 끝점 B 좌표.

    Returns:
        최소 거리.
    """
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy

    if length_sq < 1e-10:
        # 선분이 점인 경우
        return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)

    # 투영 비율 t ∈ [0, 1]
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    nearest_x = ax + t * dx
    nearest_y = ay + t * dy

    return math.sqrt((px - nearest_x) ** 2 + (py - nearest_y) ** 2)
