"""JSONL → Arrow 변환 핵심 로직 모듈.

JSONL 샤드 파일들을 스트리밍 방식으로 파싱하여 HuggingFace Arrow 데이터셋으로 변환한다.
80k+ 레코드를 메모리에 한 번에 올리지 않도록 Dataset.from_generator()를 사용한다.

Notes:
    Arrow 스키마 변환 시 다음 정규화가 적용된다:
    - front_door: null → [] (길이 0 리스트)
    - edges[].door: null → [] (길이 0 리스트)
    - spatial: [[int, int, str], ...] → [{"rid_a":..., "rid_b":..., "direction":...}, ...]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

import datasets
import orjson

from src.build_dataset.json2arrow.schema import get_floorplan_features

log = logging.getLogger(__name__)


def _normalize_door(door_value: list | dict | None) -> list[dict]:
    """door 값을 Sequence[dict] 형태로 정규화.

    JSONL에서 door는 null, 단일 dict, 또는 list of dicts로 올 수 있다.
    Arrow 스키마의 Sequence 타입에 맞게 항상 list[dict]로 정규화한다.

    Args:
        door_value: 원본 door 값 (null/dict/list).

    Returns:
        list[dict]: 정규화된 door 리스트. null이면 빈 리스트.
    """
    if door_value is None:
        return []
    if isinstance(door_value, dict):
        return [
            {
                "x": float(door_value["x"]),
                "y": float(door_value["y"]),
                "w": float(door_value["w"]),
                "h": float(door_value["h"]),
            }
        ]
    # list of dicts
    return [
        {
            "x": float(d["x"]),
            "y": float(d["y"]),
            "w": float(d["w"]),
            "h": float(d["h"]),
        }
        for d in door_value
    ]


def normalize_record(record: dict) -> dict:
    """JSONL 레코드를 Arrow 스키마에 맞게 정규화.

    HuggingFace datasets의 Sequence(dict) 타입은 columnar 형식(dict of lists)으로
    데이터를 제공해야 한다. list of dicts가 아닌 dict of lists로 변환한다.

    Args:
        record: 원본 JSONL 레코드 dict.

    Returns:
        dict: Arrow 스키마에 부합하는 columnar 형식 레코드.

    Raises:
        KeyError: 필수 필드(`plan_id`, `rooms`, `edges`, `spatial`)가 없을 때.
    """
    raw_rooms = record["rooms"]
    raw_edges = record["edges"]
    raw_spatial = record["spatial"]

    # rooms: list of dicts → dict of lists (columnar)
    rooms = {
        "rid": [int(r["rid"]) for r in raw_rooms],
        "type": [str(r["type"]) for r in raw_rooms],
        "coords": [[int(c) for c in r["coords"]] for r in raw_rooms],
    }

    # edges: Sequence({"pair": Sequence(int32), "door": Sequence(door_struct)}) 스키마에 맞게
    # 외부 edges는 columnar(dict of lists), 내부 door는 각 edge별 columnar dict 리스트
    edge_pairs = [[int(e["pair"][0]), int(e["pair"][1])] for e in raw_edges]
    edge_doors_list = [_normalize_door(e.get("doors") or e.get("door")) for e in raw_edges]
    # 각 edge의 door: list of door_dicts → 해당 edge의 columnar dict (Sequence(door_struct))
    edge_doors = [
        {
            "x": [d["x"] for d in doors],
            "y": [d["y"] for d in doors],
            "w": [d["w"] for d in doors],
            "h": [d["h"] for d in doors],
        }
        for doors in edge_doors_list
    ]
    edges = {
        "pair": edge_pairs,
        "door": edge_doors,  # list (per edge) of columnar door dicts
    }

    # front_door: null → 빈 리스트, dict → 길이 1 리스트 → columnar
    fd_list = _normalize_door(record.get("front_door"))
    front_door = {
        "x": [d["x"] for d in fd_list],
        "y": [d["y"] for d in fd_list],
        "w": [d["w"] for d in fd_list],
        "h": [d["h"] for d in fd_list],
    }

    # spatial: [[rid_a, rid_b, direction], ...] → columnar dict
    spatial = {
        "rid_a": [int(s[0]) for s in raw_spatial],
        "rid_b": [int(s[1]) for s in raw_spatial],
        "direction": [str(s[2]) for s in raw_spatial],
    }

    return {
        "plan_id": str(record["plan_id"]),
        "rooms": rooms,
        "edges": edges,
        "front_door": front_door,
        "spatial": spatial,
    }


def record_generator(
    jsonl_paths: list[str],
) -> Generator[dict, None, None]:
    """JSONL 파일들을 한 줄씩 스트리밍 파싱하는 Generator.

    대용량 데이터셋을 메모리에 한 번에 올리지 않도록 generator 방식을 사용한다.
    파싱 실패 레코드는 경고 로그만 남기고 건너뛴다.

    Args:
        jsonl_paths: 처리할 JSONL 파일 경로 리스트.

    Yields:
        dict: 정규화된 Arrow 스키마 레코드.
    """
    for path in jsonl_paths:
        log.info("파싱 중: %s", path)
        with open(path, "rb") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = orjson.loads(line)
                    yield normalize_record(record)
                except Exception as e:  # noqa: BLE001
                    log.warning(
                        "레코드 파싱 실패 [%s:%d]: %s", path, line_no, e
                    )


def convert_to_arrow(
    jsonl_paths: list[str],
    output_dir: str,
    features: datasets.Features | None = None,
) -> datasets.Dataset:
    """JSONL 파일 리스트를 Arrow 데이터셋으로 변환하고 디스크에 저장.

    Args:
        jsonl_paths: 처리할 JSONL 파일 경로 리스트.
        output_dir: Arrow 데이터셋 저장 경로.
        features: Arrow 스키마. None이면 get_floorplan_features() 사용.

    Returns:
        datasets.Dataset: 저장 완료된 Arrow 데이터셋.
    """
    if features is None:
        features = get_floorplan_features()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log.info("Arrow 변환 시작. 총 %d개 JSONL 파일.", len(jsonl_paths))

    # from_generator로 스트리밍 파싱 → 명시적 스키마 적용
    # HF datasets 캐시를 비활성화하여 항상 fresh하게 변환
    datasets.disable_caching()
    dataset = datasets.Dataset.from_generator(
        generator=record_generator,
        gen_kwargs={"jsonl_paths": jsonl_paths},
        features=features,
    )
    datasets.enable_caching()

    log.info("Arrow 변환 완료. 총 %d개 레코드.", len(dataset))
    log.info("디스크 저장 중: %s", output_dir)

    dataset.save_to_disk(str(output_path))

    log.info("저장 완료: %s", output_dir)
    return dataset
