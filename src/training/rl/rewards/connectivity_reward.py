"""R_connectivity 보상함수 모듈.

헝가리안 알고리즘으로 출력 방과 입력 조건 방을 매핑하고,
각 EDGE 조건의 두 방 사이에 DOOR가 존재하는지 검증하는 보상.

신용할당: 없음 (sequence-level 보상).

알고리즘:
    1. 같은 타입 내에서 무게중심 거리 기반 scipy.optimize.linear_sum_assignment 수행
    2. 확장된 문 사각형과 겹침 면적이 가장 큰 두 출력 방을 문마다 선택
       - 두 면적의 작은 값/큰 값 비율이 기준 이상이면 연결 쌍으로 등록
    3. 입력 EDGE의 후보 방 쌍이 등록된 연결에 포함되는지 확인

의존성: scipy>=1.14.0
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from shapely.errors import GEOSException
from shapely.geometry import Polygon, box

from src.utils.spatial import polygon_centroid
from src.training.rl.rewards.room_assignment import has_consistent_assignment

if TYPE_CHECKING:
    from src.training.rl.rewards.parser import ParsedDoor, ParsedFloorplan, ParsedRoom

logger = logging.getLogger(__name__)

# drop_type 방(좌표만 visible)을 출력 방에 매칭할 때 무게중심 거리 임계값
# 좌표 노이즈 σ=3px + 모델 생성 오차 + 변형 증강 후 잔차를 고려한 보수적 값.
_FREE_ROOM_COORD_THRESHOLD = 30.0


def compute_connectivity_reward(
    parsed: "ParsedFloorplan",
    metadata: dict,
    *,
    door_expansion: float = 2.0,
    min_overlap_balance: float = 0.15,
) -> float:
    """연결성(방 간 문 존재) 보상을 계산한다.

    Mod Record: metadata가 모델 시점으로 재구성된 후 헝가리안 매칭은 앵커 방
    (type+coords 모두 visible)에 대해서만 결정적 1:1 매칭을 만든다. drop_coords
    /drop_type 방(자유 방)은 후보 집합을 구성한 뒤 모든 연결 조건을 동시에
    만족하는 일대일 대응이 있는지 검사한다.
    Mod Record: 중심점의 20px 경계 근접 검사를 문 영역의 겹침 면적으로 교체한다.
    전처리의 5×5 확장에 대응하는 사방 2px 확장과 면적 균형 기준을 사용한다.
    모든 출력 방에 대해 문당 연결 쌍을 한 번 정하여 후보별 중복 배정을 방지한다.

    Args:
        parsed: parse_output_tokens()의 반환값.
        metadata: 모델 시점 메타데이터. 키:
            - edges (list[dict]): visible 엣지 조건. 각 항목:
                {pair: [rid_a, rid_b], door: list[{x,y,w,h}]}
            - rooms (list[dict]): visible 방 정보 (자유 방은 type 또는 coords 마스킹).
        door_expansion: 문 사각형의 각 방향 확장 거리(px). 유한한 0 이상.
        min_overlap_balance: 작은 겹침 면적/큰 겹침 면적의 최솟값. 0 이상 1 이하.

    Returns:
        모든 지정 문 연결이 충족되면 1.0, 하나라도 미충족이면 0.0.

    Raises:
        ValueError: 문 확장 거리나 겹침 균형 기준이 유효 범위를 벗어날 때.
    """
    # Mod Record: 형식 오류와 독립적으로 복원된 방·문 연결을 평가한다.
    if not parsed.rooms:
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

    connected_pairs = _door_connected_pairs(
        non_outline, parsed.doors, door_expansion=door_expansion,
        min_overlap_balance=min_overlap_balance,
    )
    input_rooms = metadata.get("rooms", [])

    # Mod Record: 같은 RID가 연결 조건마다 다른 출력 방으로 바뀌지 않게 한다.
    allowed = connected_pairs | {(b, a) for a, b in connected_pairs}
    constraints = []

    for edge in edges:
        if not edge.get("has_door", bool(edge.get("door"))):
            continue  # 문 없는 엣지는 건너뜀

        pair = edge.get("pair", [])
        if len(pair) < 2:
            # drop_pair("both"/"one")으로 마스킹된 엣지는 채점 불가 → 분모에서도 제외
            continue
        constraints.append((pair[0], pair[1], allowed))

    if not constraints:
        return 1.0
    candidates = _assignment_candidates(constraints, input_rooms, non_outline, rid_to_room_idx)
    return float(has_consistent_assignment(candidates, constraints))


def _assignment_candidates(constraints, input_rooms, output_rooms, anchors):
    """관계에 사용된 방과 고정된 앵커의 후보를 모은다.

    Args:
        constraints: RID 쌍과 허용 출력 쌍 목록.
        input_rooms: 입력에 노출된 방.
        output_rooms: 외곽선을 제외한 생성 방.
        anchors: 좌표·종류가 주어진 방의 고정 매칭.
    Returns:
        RID별 후보. 앵커가 차지한 출력 방도 예약한다.
    Raises:
        없음.
    """
    required = {rid for a, b, _ in constraints for rid in (a, b)} | set(anchors)
    return {rid: _get_candidate_output_indices(rid, input_rooms, output_rooms, anchors)
            for rid in required}


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
        try:
            in_cx, in_cy = _compute_centroid_from_raw(coords)
        except (ValueError, TypeError):
            return []
        thresh_sq = coord_threshold ** 2
        candidates: list[int] = []
        for i, r in enumerate(output_rooms):
            try:
                out_cx, out_cy = _compute_centroid_from_parsed(r)
            except (ValueError, TypeError):
                continue
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
        try:
            _compute_centroid_from_raw(room["coords"])
        except (ValueError, TypeError):
            continue
        input_by_type.setdefault(room_type, []).append(room)

    # 출력 방 타입별 분류
    output_by_type: dict[str, list[tuple[int, "ParsedRoom"]]] = {}
    for idx, room in enumerate(output_rooms):
        try:
            _compute_centroid_from_parsed(room)
        except (ValueError, TypeError):
            continue
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
    return polygon_centroid(coords)


def _compute_centroid_from_parsed(room: "ParsedRoom") -> tuple[float, float]:
    """ParsedRoom 꼭짓점에서 무게중심을 계산한다.

    Args:
        room: ParsedRoom 인스턴스.

    Returns:
        (cx, cy) 무게중심.
    """
    return polygon_centroid(room.coords)


def _door_connected_pairs(
    output_rooms: list["ParsedRoom"],
    doors: list["ParsedDoor"],
    *,
    door_expansion: float = 2.0,
    min_overlap_balance: float = 0.15,
) -> set[tuple[int, int]]:
    """문 영역과 가장 많이 겹치는 두 방의 인덱스 쌍을 반환한다.

    Mod Record: 전처리의 문 마스크 확장·상위 두 방 선택·면적 균형 원리를
    연속 폴리곤 면적으로 적용한다. 래스터 변환 없이 방 폴리곤은 한 번만 만들고,
    각 문을 전체 출력 방과 비교한다. 점·선 접촉은 면적 0이므로 제외한다.
    같은 면적이면 출력 방 순서로 선택하며, 잘못된 기하를 자동 수리하지 않는다.

    Args:
        output_rooms: outline을 제외한 전체 출력 방 리스트.
        doors: 파싱된 인테리어 문 리스트.
        door_expansion: 문 사각형의 각 방향 확장 거리(px). 유한한 0 이상.
        min_overlap_balance: 작은 겹침 면적/큰 겹침 면적의 최솟값. 0 이상 1 이하.

    Returns:
        오름차순 방 인덱스 쌍의 집합. 유효한 문 하나당 최대 한 쌍을 추가한다.

    Raises:
        ValueError: 문 확장 거리나 겹침 균형 기준이 유효 범위를 벗어날 때.
    """
    if not math.isfinite(door_expansion) or door_expansion < 0:
        raise ValueError("door_expansion은 유한한 0 이상의 값이어야 합니다.")
    if not math.isfinite(min_overlap_balance) or not 0 <= min_overlap_balance <= 1:
        raise ValueError("min_overlap_balance는 유한한 0 이상 1 이하의 값이어야 합니다.")
    if not doors or len(output_rooms) < 2:
        return set()

    polygons = []
    for index, room in enumerate(output_rooms):
        if len(room.coords) < 3 or not all(math.isfinite(v) for xy in room.coords for v in xy):
            continue
        try:
            polygon = Polygon(room.coords)
        except (ValueError, TypeError, GEOSException):
            continue
        if polygon.is_valid and not polygon.is_empty and polygon.area > 0:
            polygons.append((index, polygon))

    connected_pairs = set()
    for door in doors:
        if (not door.is_valid or not all(math.isfinite(v) for v in (door.cx, door.cy, door.w, door.h))
                or door.w <= 0 or door.h <= 0):
            continue
        expanded_door = box(
            door.cx - door.w / 2 - door_expansion,
            door.cy - door.h / 2 - door_expansion,
            door.cx + door.w / 2 + door_expansion,
            door.cy + door.h / 2 + door_expansion,
        )
        overlaps = []
        for index, polygon in polygons:
            try:
                area = polygon.intersection(expanded_door).area
            except GEOSException:
                continue
            if area > 0:
                overlaps.append((area, index))
        if len(overlaps) < 2:
            continue
        overlaps.sort(key=lambda item: item[0], reverse=True)
        (larger, first), (smaller, second) = overlaps[:2]
        if smaller / larger >= min_overlap_balance:
            connected_pairs.add((min(first, second), max(first, second)))
    return connected_pairs
