"""JSONL 직렬화 모듈 (Step 10).

추출된 방, 엣지, 문, 공간 관계 정보를 JSONL 형식으로 조립하고 저장한다.
방 순서는 canonical raster scan (centroid 기준 y→x 오름차순)으로 정렬하고 rid를 재배정한다.
rid=0은 항상 outline(건물 외곽선)에 고정.
"""

from __future__ import annotations

import os
from pathlib import Path

import orjson

from src.build_dataset.rplan2json.edge_builder import EdgeRecord
from src.build_dataset.rplan2json.room_extractor import RoomInstance


def sort_rooms_raster_order(
    room_instances: list[RoomInstance],
) -> list[RoomInstance]:
    """방 인스턴스를 raster scan 순서로 정렬하고 rid 재배정.

    centroid 기준 y 오름차순 → x 오름차순으로 정렬.
    rid=1부터 순차적으로 부여한다. (outline은 이 함수의 대상이 아님)

    Args:
        room_instances: outline 제외한 방 인스턴스 리스트.

    Returns:
        rid가 재배정된 RoomInstance 리스트.
    """
    # raster scan: y 우선 → x
    sorted_rooms = sorted(room_instances, key=lambda r: (r.centroid[1], r.centroid[0]))

    # rid 재배정 (1부터)
    for i, room in enumerate(sorted_rooms):
        room.rid = i + 1

    return sorted_rooms


def build_plan_record(
    plan_id: str,
    exterior_wall_coords: list[int],
    room_instances: list[RoomInstance],
    edges: list[EdgeRecord],
    front_door: dict[str, int] | None,
    spatial_relations: list[list],
) -> dict:
    """전체 추출 결과를 JSONL 레코드 dict로 조립.

    Args:
        plan_id: 평면도 고유 ID (파일명 기반).
        exterior_wall_coords: 외곽선 좌표 flat 리스트.
        room_instances: rid 배정 완료된 방 리스트.
        edges: EdgeRecord 리스트.
        front_door: 현관문 정보 또는 None.
        spatial_relations: [[rid_a, rid_b, direction], ...].

    Returns:
        JSONL 한 줄에 해당하는 dict.
    """
    rooms = [
        {"rid": 0, "type": "outline", "coords": exterior_wall_coords}
    ]
    for room in room_instances:
        rooms.append({
            "rid": room.rid,
            "type": room.type_name,
            "coords": room.coords,
        })

    edge_list = []
    for e in edges:
        edge_list.append({
            "pair": list(e.pair),
            "doors": e.doors,
        })

    return {
        "plan_id": plan_id,
        "rooms": rooms,
        "edges": edge_list,
        "front_door": front_door,
        "spatial": spatial_relations,
    }


def serialize_to_jsonl(records: list[dict], output_path: str) -> None:
    """레코드 리스트를 JSONL 파일로 저장.

    Args:
        records: dict 리스트.
        output_path: 출력 JSONL 파일 경로.
    """
    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, "wb") as f:
        for record in records:
            f.write(orjson.dumps(record))
            f.write(b"\n")


def append_to_jsonl(record: dict, output_path: str) -> None:
    """단일 레코드를 JSONL 파일에 추가.

    Args:
        record: JSONL 레코드 dict.
        output_path: 출력 JSONL 파일 경로.
    """
    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, "ab") as f:
        f.write(orjson.dumps(record))
        f.write(b"\n")
