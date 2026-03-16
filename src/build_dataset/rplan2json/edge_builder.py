"""Edge 구성 모듈 (Step 8).

두 가지 조건으로 방 간 연결 쌍(edge)을 탐지한다:
  1. 직접 픽셀 인접: 두 방의 영역 픽셀이 1px 이내로 맞닿아 있는 경우.
  2. 문 연결: 두 방의 경계 영역에 인테리어 문이 존재하는 경우.

두 조건의 합집합이 edge가 된다. 두 조건 모두 해당하는 쌍은
door 정보가 등록된 edge로 저장된다.

exterior_wall(rid=0)은 edge 탐색 대상에서 제외.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np

from src.build_dataset.rplan2json.door_extractor import DoorInstance
from src.build_dataset.rplan2json.room_extractor import RoomInstance


@dataclass
class EdgeRecord:
    """방 간 연결 관계 + 문 정보.

    Attributes:
        pair: 연결된 방 ID 쌍 (rid_a, rid_b), rid_a < rid_b.
        doors: 문 정보 리스트 (각 {"x","y","w","h"}). 문 없으면 빈 리스트.
    """

    pair: tuple[int, int]
    doors: list[dict[str, int]]


def build_edges(
    room_instances: list[RoomInstance],
    door_instances: list[DoorInstance],
    door_dilation_kernel: int,
) -> list[EdgeRecord]:
    """Step 8: 방 간 Edge 구성.

    두 방이 아래 조건 중 하나 이상을 만족하면 edge로 등록한다:
      - 조건 1 (직접 인접): 두 방의 픽셀 영역이 1px 이내로 맞닿아 있음.
      - 조건 2 (문 연결): 두 방의 경계 영역에 인테리어 문이 존재함.
    두 조건 모두 해당하는 경우 door 정보를 포함한 edge로 등록된다.

    Args:
        room_instances: 방 인스턴스 리스트 (rid 배정 완료 상태).
        door_instances: 인테리어 문 인스턴스 리스트.
        door_dilation_kernel: 문 마스크 확장 커널 크기 (방 탐색용).

    Returns:
        EdgeRecord 리스트, pair 오름차순 정렬.
    """
    # outline(rid=0) 제외: 건물 외곽선은 edge 탐색 대상이 아님
    rooms = [r for r in room_instances if r.type_name != "outline"]
    if len(rooms) < 2:
        return []

    # 각 방의 바이너리 마스크 사전 계산
    raw_masks: dict[int, np.ndarray] = {}
    for room in rooms:
        raw_masks[room.rid] = (room.mask > 0).astype(np.uint8)

    # --- Step A: 직접 픽셀 인접 쌍 탐지 ---
    direct_pairs = _find_direct_adjacent_pairs(rooms, raw_masks)

    # --- Step B: 문 → 방 쌍 매핑 구성 ---
    pair_to_door = _map_doors_to_pairs(
        rooms, raw_masks, door_instances, door_dilation_kernel
    )

    # --- Step C: edge 등록 (두 조건의 합집합) ---
    all_pairs = direct_pairs | set(pair_to_door.keys())

    edges: list[EdgeRecord] = []
    for pair in all_pairs:
        edges.append(EdgeRecord(pair=pair, doors=pair_to_door.get(pair, [])))

    # pair 기준 정렬 (canonical 순서)
    edges.sort(key=lambda e: e.pair)
    return edges


def _find_direct_adjacent_pairs(
    rooms: list[RoomInstance],
    raw_masks: dict[int, np.ndarray],
) -> set[tuple[int, int]]:
    """조건 1: 방 픽셀이 직접 맞닿는 쌍을 탐지.

    1px dilation(3x3 kernel) 후 상대방 원본 마스크와 overlap이 있으면
    두 방이 픽셀 수준에서 직접 인접해 있다고 판단한다.

    Args:
        rooms: 방 인스턴스 리스트.
        raw_masks: rid → 바이너리 마스크 딕셔너리.

    Returns:
        직접 인접한 (rid_a, rid_b) 쌍의 집합 (rid_a < rid_b).
    """
    # 1px 확장 커널 (8-connectivity 방향으로 맞닿음 감지)
    kernel_1px = np.ones((3, 3), dtype=np.uint8)
    direct_pairs: set[tuple[int, int]] = set()

    for room_a, room_b in combinations(rooms, 2):
        # room_a를 1px 확장 후 room_b 원본과 overlap 확인
        dilated_a = cv2.dilate(raw_masks[room_a.rid], kernel_1px, iterations=1)
        if (dilated_a & raw_masks[room_b.rid]).sum() > 0:
            rid_a, rid_b = sorted([room_a.rid, room_b.rid])
            direct_pairs.add((rid_a, rid_b))

    return direct_pairs


def _map_doors_to_pairs(
    rooms: list[RoomInstance],
    raw_masks: dict[int, np.ndarray],
    door_instances: list[DoorInstance],
    door_dilation_kernel: int,
    min_balance: float = 0.15,
) -> dict[tuple[int, int], list[dict[str, int]]]:
    """조건 2: 각 문(door)에 인접한 방 쌍을 찾아 pair → door 리스트 매핑 구성.

    각 door 마스크를 door_dilation_kernel 크기로 확장하여
    어떤 방들이 문 주변에 위치하는지 탐지한다.
    인접 방이 정확히 2개인 쌍에만 door를 매칭한다.
    동일 pair에 여러 문이 있을 수 있으며 모두 리스트로 저장한다.

    Args:
        rooms: 방 인스턴스 리스트.
        raw_masks: rid → 바이너리 마스크 딕셔너리.
        door_instances: 인테리어 문 인스턴스 리스트.
        door_dilation_kernel: 문 마스크 확장 커널 크기.
        min_balance: overlap 균형도 최솟값. 이 미만인 문은 아티팩트로 간주하여 제외.

    Returns:
        (rid_a, rid_b) → door bbox 리스트 딕셔너리 (rid_a < rid_b).
    """
    search_kernel = np.ones(
        (door_dilation_kernel, door_dilation_kernel), dtype=np.uint8
    )

    pair_to_doors: dict[tuple[int, int], list[dict[str, int]]] = {}

    for door in door_instances:
        door_binary = (door.mask > 0).astype(np.uint8)

        # 문 마스크를 확장하여 주변 방 탐색 (벽 너머 방에 도달)
        expanded_door = cv2.dilate(door_binary, search_kernel, iterations=1)

        # 확장 문 영역과 overlap이 있는 방들을 overlap 크기 기준으로 정렬
        overlapping: list[tuple[int, int]] = []  # (overlap_count, rid)
        for room in rooms:
            overlap_count = int((expanded_door & raw_masks[room.rid]).sum())
            if overlap_count > 0:
                overlapping.append((overlap_count, room.rid))

        if len(overlapping) < 2:
            # 인접 방이 1개 이하이면 방 간 문으로 볼 수 없음
            continue

        # overlap이 가장 큰 2개의 방을 선택 (문이 두 방 경계에 위치해야 함)
        overlapping.sort(key=lambda x: x[0], reverse=True)
        rid_top1 = overlapping[0][1]
        rid_top2 = overlapping[1][1]
        rid_a, rid_b = sorted([rid_top1, rid_top2])
        pair = (rid_a, rid_b)

        # overlap 균형도: 두 방에 대한 overlap이 균등할수록 실제 경계 문에 가까움
        # min_balance 미만인 문은 한쪽 방에만 치우친 아티팩트로 간주하여 제외
        top1_overlap = overlapping[0][0]
        top2_overlap = overlapping[1][0]
        balance = min(top1_overlap, top2_overlap) / max(top1_overlap, top2_overlap)
        if balance < min_balance:
            continue

        pair_to_doors.setdefault(pair, []).append(door.bbox.copy())

    return pair_to_doors
