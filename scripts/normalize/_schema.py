"""공통 JSON 스키마 정의 + 변환 헬퍼.

논문 실험 가이드 §3.3 의 공통 스키마:

    {
        "plan_id": str,
        "model": str,              # "ours" | "housediffusion" | "gsdiff" | "ds2d"
        "rooms": [
            {"type": str, "polygon": [[x, y], ...], "rid": int (optional)}
        ],
        "front_door": {"x": float, "y": float, "w": float, "h": float} | None,
        "doors": [{"x": float, "y": float, "w": float, "h": float}, ...]   # 인테리어 문
    }

모든 baseline · 본 연구의 추론 결과를 이 스키마로 떨어뜨려 후속 metric / renderer 가 한 형식으로 처리한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# 본 연구 출력(`outputs/inference/.../floorplan.json`) 스키마:
#   {"rooms":[{"rid","type","coords":[x1,y1,x2,y2,...]}], "edges":[{"pair","doors":[...]}],
#    "front_door":{"x","y","w","h"} | None, "spatial":[]}


def coords_flat_to_polygon(coords: list[int | float]) -> list[list[float]]:
    """[x1,y1,x2,y2,...] flat array → [[x,y], ...] list of pairs."""
    return [[float(coords[i]), float(coords[i + 1])] for i in range(0, len(coords), 2)]


def from_ours_floorplan_json(
    raw: dict[str, Any],
    plan_id: str,
    model: str = "ours",
) -> dict[str, Any]:
    """본 연구 `floorplan.json` (output_parser.parse_output_tokens 결과)을 공통 스키마로 변환.

    Args:
        raw: 본 연구 floorplan.json 의 파싱 결과 dict.
        plan_id: 결과에 포함될 plan_id.
        model: 결과의 model 식별자.

    Returns:
        공통 스키마 dict.
    """
    rooms_out: list[dict[str, Any]] = []
    for r in raw.get("rooms", []):
        if not r.get("coords"):
            continue
        rooms_out.append({
            "type": r.get("type", "unknown"),
            "polygon": coords_flat_to_polygon(r["coords"]),
            "rid": r.get("rid"),
        })

    doors: list[dict[str, Any]] = []
    for e in raw.get("edges", []) or []:
        for d in e.get("doors", []) or []:
            doors.append({
                "x": float(d["x"]),
                "y": float(d["y"]),
                "w": float(d["w"]),
                "h": float(d["h"]),
            })

    fd = raw.get("front_door")
    front_door = (
        {"x": float(fd["x"]), "y": float(fd["y"]), "w": float(fd["w"]), "h": float(fd["h"])}
        if fd
        else None
    )

    return {
        "plan_id": plan_id,
        "model": model,
        "rooms": rooms_out,
        "front_door": front_door,
        "doors": doors,
    }


def to_parsed_floorplan_dict(common: dict[str, Any]) -> dict[str, Any]:
    """공통 스키마 → 본 연구 reward parser (`ParsedFloorplan`) 호환 dict 로 역변환.

    Returns:
        {"rooms": [{"rid","type","coords"}], "edges":[{"pair","doors":[...]}],
         "front_door":{...}|None, "spatial":[]} (output_parser 출력과 동일 구조)
    """
    rooms_back = []
    for i, r in enumerate(common.get("rooms", [])):
        coords_flat: list[float] = []
        for x, y in r.get("polygon", []):
            coords_flat.extend([float(x), float(y)])
        rooms_back.append({
            "rid": r.get("rid") if r.get("rid") is not None else i,
            "type": r.get("type", "unknown"),
            "coords": coords_flat,
        })
    edges_back = [{"pair": [0, 0], "doors": [d]} for d in common.get("doors", []) or []]
    return {
        "rooms": rooms_back,
        "edges": edges_back,
        "front_door": common.get("front_door"),
        "spatial": [],
    }


def write_common_json(common: dict[str, Any], out_path: Path) -> None:
    """공통 스키마 dict 를 디스크에 저장 (UTF-8, 2-space indent)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(common, indent=2, ensure_ascii=False))


def load_common_json(path: Path) -> dict[str, Any]:
    """공통 스키마 JSON 파일 로드."""
    return json.loads(Path(path).read_text())
