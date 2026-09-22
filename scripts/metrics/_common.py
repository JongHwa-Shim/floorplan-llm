"""메트릭 스크립트 공통 헬퍼.

- 공통 스키마 JSON 디렉토리 → 모델/plan_id 별 dict 로 정리
- Arrow GT split row → 공통 스키마 변환
- plan_id별 복수 생성 결과 그룹화 (집계 방식은 각 평가 스크립트에서 결정)
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts" / "normalize"))

from _schema import coords_flat_to_polygon  # noqa: E402


_GENERATION_RE = re.compile(r"^(?P<plan_id>.+?)(?:_(?P<idx>\d+))?$")


def load_common_dir(in_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """디렉토리의 모든 공통 스키마 JSON 을 plan_id 별로 그룹핑.

    파일명 규칙: ``{plan_id}.json`` 또는 ``{plan_id}_{idx}.json`` (best-of-K).

    Returns:
        {plan_id: [generation_dict, ...]} (idx 순 정렬).
    """
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for f in sorted(Path(in_dir).glob("*.json")):
        m = _GENERATION_RE.match(f.stem)
        if m is None:
            continue
        plan_id = m.group("plan_id")
        idx = int(m.group("idx")) if m.group("idx") else 0
        groups[plan_id].append((idx, json.loads(f.read_text())))
    return {pid: [d for _, d in sorted(items)] for pid, items in groups.items()}


def arrow_row_to_common(row: dict[str, Any]) -> dict[str, Any]:
    """Arrow columnar GT row → 공통 스키마 dict.

    Args:
        row: load_from_disk(...)[i] 결과 (rooms/edges/front_door/spatial columnar dict 포함).

    Returns:
        공통 스키마 dict (model="gt").
    """
    rooms_col = row["rooms"]
    rooms: list[dict[str, Any]] = []
    for i, rid in enumerate(rooms_col["rid"]):
        rooms.append({
            "rid": int(rid),
            "type": rooms_col["type"][i],
            "polygon": coords_flat_to_polygon(list(rooms_col["coords"][i])),
        })
    edges_col = row["edges"]
    doors: list[dict[str, Any]] = []
    if edges_col["door"]:
        for door_dict in edges_col["door"]:
            for j in range(len(door_dict["x"])):
                doors.append({
                    "x": float(door_dict["x"][j]),
                    "y": float(door_dict["y"][j]),
                    "w": float(door_dict["w"][j]),
                    "h": float(door_dict["h"][j]),
                })

    fd_col = row.get("front_door") or {"x": [], "y": [], "w": [], "h": []}
    if fd_col.get("x"):
        front_door = {
            "x": float(fd_col["x"][0]),
            "y": float(fd_col["y"][0]),
            "w": float(fd_col["w"][0]),
            "h": float(fd_col["h"][0]),
        }
    else:
        front_door = None

    return {
        "plan_id": row["plan_id"],
        "model": "gt",
        "rooms": rooms,
        "front_door": front_door,
        "doors": doors,
    }


def load_gt_pool(arrow_pool_path: Path) -> dict[str, dict[str, Any]]:
    """eval_pool Arrow split 의 모든 row 를 plan_id → 공통 스키마 dict 로 로드."""
    from datasets import load_from_disk

    ds = load_from_disk(str(arrow_pool_path))
    return {row["plan_id"]: arrow_row_to_common(row) for row in ds}


def load_unified_plan_ids(json_path: Path) -> list[str]:
    """testset_unified.json 에서 "all" 또는 bucket 합집합 plan_id 리스트 로드."""
    data = json.loads(Path(json_path).read_text())
    if isinstance(data, dict):
        if "all" in data:
            return [str(pid) for pid in data["all"]]
        return sorted({
            str(pid)
            for k, v in data.items()
            if not k.startswith("_") and isinstance(v, list)
            for pid in v
        })
    return [str(pid) for pid in data]


def get_room_count_bucket(plan: dict[str, Any]) -> int:
    """비-outline 방 개수 (room-count bucket key)."""
    return sum(1 for r in plan.get("rooms", []) if r.get("type") != "outline")
