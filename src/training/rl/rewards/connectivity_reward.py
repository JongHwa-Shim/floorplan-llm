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

# drop_type 방(좌표만 visible)을 출력 방에 매칭할 때 무게중심 거리 임계값
# 좌표 노이즈 σ=3px + 모델 생성 오차 + 변형 증강 후 잔차를 고려한 보수적 값.
_FREE_ROOM_COORD_THRESHOLD = 30.0


def compute_connectivity_reward(
    parsed: "ParsedFloorplan",
    metadata: dict,
) -> float:
    """연결성(방 간 문 존재) 보상을 계산한다.

    Mod Record: metadata가 모델 시점으로 재구성된 후 헝가리안 매칭은 앵커 방
    (type+coords 모두 visible)에 대해서만 결정적 1:1 매칭을 만든다. drop_coords
    /drop_type 방(자유 방)은 매칭 모호하므로 직접 매칭하지 않고, 제약 검사 시점에
    후보 출력 방 집합으로 확장한 뒤 어떤 (a, b) 쌍에서든 제약이 만족되면 통과로
    채점한다(satisfiability-based 채점).

    Args:
        parsed: parse_output_tokens()의 반환값.
        metadata: 모델 시점 메타데이터. 키:
            - edges (list[dict]): visible 엣지 조건. 각 항목:
                {pair: [rid_a, rid_b], door: list[{x,y,w,h}]}
            - rooms (list[dict]): visible 방 정보 (자유 방은 type 또는 coords 마스킹).

    Returns:
        [0, 1] 범위. DOOR 조건 충족 비율.
    """
    if not parsed.success or not parsed.rooms:
        return 0.0

    edges = metadata.get("edges", [])
    if not edges:
        return 1.0  # 연결 조건 없으면 만점

    # 앵커 방의 결정적 1:1 매칭
    rid_to_room_idx = _hungarian_match(parsed, metadata)

    # outline 제외 출력 방 (한 번만 계산)
    non_outline = [r for r in parsed.rooms if r.room_type != "outline"]
    if not non_outline:
        return 0.0

    input_rooms = metadata.get("rooms", [])

    # DOOR 존재 여부 검증
    satisfied = 0
    total_with_door = 0

    for edge in edges:
        if not edge.get("door"):
            continue  # 문 없는 엣지는 건너뜀

        pair = edge.get("pair", [])
        if len(pair) < 2:
            # drop_pair("both"/"one")으로 마스킹된 엣지는 채점 불가 → 분모에서도 제외
            continue
        total_with_door += 1

        rid_a, rid_b = pair[0], pair[1]
        cands_a = _get_candidate_output_indices(rid_a, input_rooms, non_outline, rid_to_room_idx)
        cands_b = _get_candidate_output_indices(rid_b, input_rooms, non_outline, rid_to_room_idx)

        if not cands_a or not cands_b:
            continue

        # 어떤 (a, b) 조합에서든 door가 두 방 경계에 존재하면 만족
        if _exists_door_pair(cands_a, cands_b, non_outline, parsed.doors):
            satisfied += 1

    if total_with_door == 0:
        return 1.0

    return satisfied / total_with_door


def _exists_door_pair(
    cands_a: list[int],
    cands_b: list[int],
    output_rooms: list,
    doors: list,
) -> bool:
    """후보 출력 방 인덱스 집합 두 개에서 door로 연결된 (a, b) 쌍이 존재하는지 확인한다.

    Args:
        cands_a: 첫 번째 RID의 후보 출력 방 인덱스 리스트.
        cands_b: 두 번째 RID의 후보 출력 방 인덱스 리스트.
        output_rooms: outline 제외 출력 방 리스트.
        doors: parsed.doors.

    Returns:
        어떤 (a_idx, b_idx) 조합에서든 door가 존재하면 True.
    """
    for a_idx in cands_a:
        for b_idx in cands_b:
            if a_idx == b_idx:
                continue
            if _has_door_between(output_rooms[a_idx], output_rooms[b_idx], doors):
                return True
    return False


# ---------------------------------------------------------------------------
# 내부 헬퍼 함수
# ---------------------------------------------------------------------------

def _get_candidate_output_indices(
    rid: int,
    input_rooms: list[dict],
    output_rooms: list,
    rid_to_output_idx: dict[int, int],
    coord_threshold: float = _FREE_ROOM_COORD_THRESHOLD,
) -> list[int]:
    """입력 RID에 대응 가능한 출력 방 인덱스 후보 리스트를 반환한다.

    metadata가 모델 시점으로 재구성된 후, 입력 RID는 다음 4가지 분류 중 하나이다:
        1. 앵커: type+coords 모두 visible → rid_to_output_idx에 1:1 매핑됨 → 단일 후보
        2. drop_coords (자유): type만 visible → 같은 type의 모든 출력 방이 후보
        3. drop_type (자유): coords만 visible → 좌표 근접 (centroid 거리 ≤ threshold)
            출력 방이 후보
        4. drop_block: metadata에 없으므로 빈 리스트 반환 (호출이 발생하면 안 되는 케이스)

    이 후보 집합으로 connectivity/spatial 제약을 satisfiability 기반으로 채점하면,
    모델이 식별 가능한 정보 범위 내에서 공정한 평가가 가능해진다.

    Args:
        rid: 입력 측 RID.
        input_rooms: metadata.rooms (자유 방은 type 또는 coords가 마스킹된 상태).
        output_rooms: outline 제외 출력 방 리스트 (parsed.rooms 기반).
        rid_to_output_idx: 앵커 방의 결정적 매핑 (_hungarian_match 결과).
        coord_threshold: drop_type 방 매칭 시 무게중심 거리 임계값 (px).

    Returns:
        출력 방 인덱스 후보 리스트. 빈 리스트면 어떤 후보도 없음(드물게 drop_block).
    """
    # 앵커: 결정적 매핑 사용
    if rid in rid_to_output_idx:
        idx = rid_to_output_idx[rid]
        return [idx] if 0 <= idx < len(output_rooms) else []

    # 메타데이터에서 해당 RID의 자유 방 정보 조회
    meta_room = next((r for r in input_rooms if r.get("rid") == rid), None)
    if meta_room is None:
        return []  # drop_block 또는 알 수 없는 RID

    room_type = meta_room.get("type", "")
    coords = meta_room.get("coords", [])

    has_type = bool(room_type) and room_type != "outline"
    has_coords = bool(coords)

    if has_type and not has_coords:
        # drop_coords: 같은 type 출력 방 모두
        return [i for i, r in enumerate(output_rooms) if r.room_type == room_type]

    if has_coords and not has_type:
        # drop_type: 좌표 근접 출력 방
        in_cx, in_cy = _compute_centroid_from_raw(coords)
        thresh_sq = coord_threshold ** 2
        candidates: list[int] = []
        for i, r in enumerate(output_rooms):
            out_cx, out_cy = _compute_centroid_from_parsed(r)
            dist_sq = (in_cx - out_cx) ** 2 + (in_cy - out_cy) ** 2
            if dist_sq <= thresh_sq:
                candidates.append(i)
        return candidates

    # 양쪽 모두 마스킹된 상태는 정상 흐름에선 발생하지 않음 (drop_block은 metadata에서 제거됨)
    return []


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
    # Mod Record: metadata가 모델 시점으로 재구성된 후 drop_type 방은 type=""로,
    # drop_coords 방은 coords=[]로 마스킹된다. 두 경우 모두 모델이 RID를 식별할
    # 단서가 부족하므로 헝가리안 매칭 후보에서 제외한다 (자유 방 — 매칭 모호).
    # 이 처리가 없으면 빈 coords가 (0,0)으로 계산되어 잘못된 매칭이 발생한다.
    input_by_type: dict[str, list[dict]] = {}
    for room in input_rooms:
        room_type = room.get("type", "")
        if room_type == "outline" or room_type == "":
            continue
        if not room.get("coords"):
            continue
        input_by_type.setdefault(room_type, []).append(room)

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
