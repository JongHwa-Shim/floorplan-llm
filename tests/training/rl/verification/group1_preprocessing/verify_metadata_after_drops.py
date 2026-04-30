"""Group 1: drop 상태 → metadata 마스킹 검증.

`_extract_metadata(augmented_sample, drop_state)`를 직접 호출하여 8가지 drop을
각각 격리 검증한다. pipeline 우회 (RNG / 확률 제거).

검증 항목 (의도 vs 구현):
    - drop_block: rooms에서 RID 제거. total_rooms는 유지 (비대칭).
    - drop_type: rid type=""로 마스킹, coords 보존.
    - drop_coords: coords=[] 마스킹, type 보존.
    - drop_edge: edges에서 idx 제거.
    - drop_pair "both": edges[idx].pair=[].
    - drop_pair "one": edges[idx].pair=[kept_rid].
    - drop_door modes: position/orientation/all 부분 마스킹.
    - drop_spatial: spatial에서 idx 제거.
    - drop_front_door / drop_front_door_coords: front_door 마스킹.
    - drop_room_summary_total: total_rooms=None.
    - drop_room_summary_types: type_counts에서 type 제거.

핵심 비대칭 단언: drop_block 후 total_rooms != len(metadata.rooms)
"""

from __future__ import annotations

import sys
from pathlib import Path

_VERIF_ROOT = Path(__file__).resolve().parents[1]
if str(_VERIF_ROOT) not in sys.path:
    sys.path.insert(0, str(_VERIF_ROOT))

from _common import run_cases, summary_and_exit  # noqa: E402

from src.training.augmentation.strategies import DropState  # noqa: E402
from src.training.rl.dataset import _extract_metadata  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def fixture_sample() -> dict:
    """row-oriented 표준 fixture (변형 적용 완료 가정)."""
    return {
        "rooms": [
            {"rid": 0, "type": "outline",    "coords": [10, 10, 200, 10, 200, 200, 10, 200]},
            {"rid": 1, "type": "livingroom", "coords": [10, 10, 100, 10, 100, 100, 10, 100]},
            {"rid": 2, "type": "bedroom",    "coords": [100, 10, 200, 10, 200, 100, 100, 100]},
            {"rid": 3, "type": "bedroom",    "coords": [100, 100, 200, 100, 200, 200, 100, 200]},
            {"rid": 4, "type": "kitchen",    "coords": [10, 100, 100, 100, 100, 200, 10, 200]},
        ],
        "edges": [
            {"pair": [1, 2], "door": [{"x": 100.0, "y": 50.0, "w": 2.0, "h": 10.0}]},
            {"pair": [1, 4], "door": [{"x": 50.0, "y": 100.0, "w": 10.0, "h": 2.0}]},
            {"pair": [2, 3], "door": [{"x": 150.0, "y": 100.0, "w": 10.0, "h": 2.0}]},
        ],
        "front_door": {"x": 105.0, "y": 10.0, "w": 8.0, "h": 2.0},
        "spatial": [
            {"rid_a": 1, "rid_b": 2, "direction": "right"},
            {"rid_a": 1, "rid_b": 4, "direction": "below"},
            {"rid_a": 2, "rid_b": 3, "direction": "below"},
        ],
    }


def _rid_in(metadata: dict, rid: int) -> dict | None:
    return next((r for r in metadata["rooms"] if r.get("rid") == rid), None)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def case_drop_block():
    """drop_block(rid=2) → metadata.rooms에서 rid=2 제거. total_rooms 유지 (비대칭)."""
    sample = fixture_sample()
    drop_state = DropState(drop_block={2})
    metadata = _extract_metadata(sample, drop_state)

    assert _rid_in(metadata, 2) is None, "drop_block 후에도 rid=2가 남아 있음"
    assert {r["rid"] for r in metadata["rooms"]} == {0, 1, 3, 4}, \
        f"잘못된 rooms: {[r['rid'] for r in metadata['rooms']]}"
    # 비대칭 단언: total_rooms는 ROOM_SUMMARY 노출 (drop_block 포함) 기준
    # outline 제외 4개 (1, 2, 3, 4)
    assert metadata["total_rooms"] == 4, \
        f"total_rooms는 ROOM_SUMMARY 노출값(4)이어야 함: {metadata['total_rooms']}"
    # metadata.rooms 길이 (visible)
    visible_rooms = [r for r in metadata["rooms"] if r["rid"] != 0]
    assert len(visible_rooms) == 3, "drop_block 후 visible 방 3개여야 함"
    # 핵심 비대칭
    assert metadata["total_rooms"] != len(visible_rooms), \
        "total_rooms와 visible 방 개수가 같음 — 비대칭 깨짐"


def case_drop_type():
    """drop_type(rid=3) → rid=3 type=""로 마스킹, coords 보존."""
    sample = fixture_sample()
    drop_state = DropState(drop_type={3})
    metadata = _extract_metadata(sample, drop_state)
    r3 = _rid_in(metadata, 3)
    assert r3 is not None, "drop_type 시 방이 제거되면 안 됨"
    assert r3["type"] == "", f"drop_type 후 type='': {r3['type']!r}"
    assert r3["coords"] == [100, 100, 200, 100, 200, 200, 100, 200], \
        f"drop_type은 coords 보존해야 함: {r3['coords']}"


def case_drop_coords():
    """drop_coords(rid=4) → rid=4 coords=[], type 보존."""
    sample = fixture_sample()
    drop_state = DropState(drop_coords={4})
    metadata = _extract_metadata(sample, drop_state)
    r4 = _rid_in(metadata, 4)
    assert r4 is not None
    assert r4["coords"] == [], f"drop_coords 후 coords=[]: {r4['coords']}"
    assert r4["type"] == "kitchen", f"drop_coords는 type 보존: {r4['type']!r}"


def case_drop_edge():
    """drop_edge(idx=0) → metadata.edges에서 첫 엣지 제거."""
    sample = fixture_sample()
    drop_state = DropState(drop_edge={0})
    metadata = _extract_metadata(sample, drop_state)
    assert len(metadata["edges"]) == 2, f"drop_edge 후 엣지 2개여야 함: {len(metadata['edges'])}"
    # 남은 엣지: [1,4], [2,3]
    assert metadata["edges"][0]["pair"] == [1, 4]
    assert metadata["edges"][1]["pair"] == [2, 3]


def case_drop_pair_both():
    """drop_pair {1: 'both'} → edges[1].pair=[]."""
    sample = fixture_sample()
    drop_state = DropState(drop_pair={1: "both"})
    metadata = _extract_metadata(sample, drop_state)
    assert metadata["edges"][1]["pair"] == [], f"drop_pair both 후 pair=[]: {metadata['edges'][1]['pair']}"
    # 다른 edge는 영향 없음
    assert metadata["edges"][0]["pair"] == [1, 2]


def case_drop_pair_one():
    """drop_pair {1: ('one', 4)} → edges[1].pair=[4]."""
    sample = fixture_sample()
    drop_state = DropState(drop_pair={1: ("one", 4)})
    metadata = _extract_metadata(sample, drop_state)
    assert metadata["edges"][1]["pair"] == [4], \
        f"drop_pair one 후 pair=[4]: {metadata['edges'][1]['pair']}"


def case_drop_door_all():
    """drop_door {2: 'all'} → edges[2].door=[]."""
    sample = fixture_sample()
    drop_state = DropState(drop_door={2: "all"})
    metadata = _extract_metadata(sample, drop_state)
    assert metadata["edges"][2]["door"] == [], f"drop_door all 후 door=[]: {metadata['edges'][2]['door']}"


def case_drop_door_position():
    """drop_door {0: 'position'} → edges[0].door[i].x/y=None, w/h 보존."""
    sample = fixture_sample()
    drop_state = DropState(drop_door={0: "position"})
    metadata = _extract_metadata(sample, drop_state)
    door = metadata["edges"][0]["door"][0]
    assert door["x"] is None and door["y"] is None, f"x/y가 None이어야 함: {door}"
    assert door["w"] == 2.0 and door["h"] == 10.0, f"w/h 보존되어야 함: {door}"


def case_drop_door_orientation():
    """drop_door {1: 'orientation'} → edges[1].door[i].w/h=None, x/y 보존."""
    sample = fixture_sample()
    drop_state = DropState(drop_door={1: "orientation"})
    metadata = _extract_metadata(sample, drop_state)
    door = metadata["edges"][1]["door"][0]
    assert door["w"] is None and door["h"] is None, f"w/h가 None이어야 함: {door}"
    assert door["x"] == 50.0 and door["y"] == 100.0, f"x/y 보존되어야 함: {door}"


def case_drop_spatial():
    """drop_spatial {0, 2} → 1번 spatial만 남음."""
    sample = fixture_sample()
    drop_state = DropState(drop_spatial={0, 2})
    metadata = _extract_metadata(sample, drop_state)
    assert len(metadata["spatial"]) == 1, f"spatial 1개 남아야 함: {len(metadata['spatial'])}"
    assert metadata["spatial"][0]["direction"] == "below"


def case_drop_front_door():
    """drop_front_door=True → front_door=None."""
    sample = fixture_sample()
    drop_state = DropState(drop_front_door=True)
    metadata = _extract_metadata(sample, drop_state)
    assert metadata["front_door"] is None


def case_drop_front_door_coords():
    """drop_front_door_coords=True → front_door.x/y=None, w/h 보존."""
    sample = fixture_sample()
    drop_state = DropState(drop_front_door_coords=True)
    metadata = _extract_metadata(sample, drop_state)
    fd = metadata["front_door"]
    assert fd is not None
    assert fd["x"] is None and fd["y"] is None, f"x/y가 None이어야 함: {fd}"
    assert fd["w"] == 8.0 and fd["h"] == 2.0, f"w/h 보존: {fd}"


def case_drop_room_summary_total():
    """drop_room_summary_total=True → total_rooms=None."""
    sample = fixture_sample()
    drop_state = DropState(drop_room_summary_total=True)
    metadata = _extract_metadata(sample, drop_state)
    assert metadata["total_rooms"] is None, \
        f"drop_room_summary_total 후 total_rooms=None: {metadata['total_rooms']}"
    # type_counts는 영향 없음
    assert metadata["type_counts"] == {"livingroom": 1, "bedroom": 2, "kitchen": 1}


def case_drop_room_summary_types():
    """drop_room_summary_types={'bedroom'} → type_counts에서 bedroom 키 제거."""
    sample = fixture_sample()
    drop_state = DropState(drop_room_summary_types={"bedroom"})
    metadata = _extract_metadata(sample, drop_state)
    assert "bedroom" not in metadata["type_counts"], \
        f"bedroom이 type_counts에 남아 있음: {metadata['type_counts']}"
    assert metadata["type_counts"] == {"livingroom": 1, "kitchen": 1}


def case_drop_block_with_total_intact():
    """★ 핵심 비대칭: drop_block(rid=2) + drop_room_summary_total=False
       → total_rooms=4 (ROOM_SUMMARY 노출), len(visible non-outline)=3."""
    sample = fixture_sample()
    drop_state = DropState(drop_block={2})
    metadata = _extract_metadata(sample, drop_state)
    visible_non_outline = [r for r in metadata["rooms"] if r.get("rid") != 0]
    assert metadata["total_rooms"] == 4, "ROOM_SUMMARY total은 drop_block 영향 안 받음"
    assert len(visible_non_outline) == 3, "visible 방은 drop_block 반영"
    # 비대칭 보장
    assert metadata["total_rooms"] != len(visible_non_outline)


def case_combined_drops():
    """여러 drop 조합: drop_block(2) + drop_type(3) + drop_coords(4)."""
    sample = fixture_sample()
    drop_state = DropState(drop_block={2}, drop_type={3}, drop_coords={4})
    metadata = _extract_metadata(sample, drop_state)
    rids = {r["rid"] for r in metadata["rooms"]}
    assert rids == {0, 1, 3, 4}, f"drop_block 후 rid 집합: {rids}"
    r3 = _rid_in(metadata, 3)
    r4 = _rid_in(metadata, 4)
    assert r3["type"] == "" and r3["coords"], "rid=3 drop_type"
    assert r4["coords"] == [] and r4["type"] == "kitchen", "rid=4 drop_coords"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class _Case:
    def __init__(self, name, intent, fn):
        self.name = name
        self.intent = intent
        self.fn = fn


def main():
    cases = [
        _Case("drop_block",                 "drop_block(rid=2)로 visible rooms에서 제거",                       case_drop_block),
        _Case("drop_type",                  "drop_type(rid=3) 후 type=''/coords 보존",                          case_drop_type),
        _Case("drop_coords",                "drop_coords(rid=4) 후 coords=[]/type 보존",                        case_drop_coords),
        _Case("drop_edge",                  "drop_edge(idx=0) 후 edges 리스트에서 제거",                        case_drop_edge),
        _Case("drop_pair_both",             "drop_pair both 후 edge.pair=[]",                                   case_drop_pair_both),
        _Case("drop_pair_one",              "drop_pair one 후 edge.pair=[kept_rid]",                            case_drop_pair_one),
        _Case("drop_door_all",              "drop_door all 후 door=[]",                                         case_drop_door_all),
        _Case("drop_door_position",         "drop_door position 후 x/y=None, w/h 보존",                          case_drop_door_position),
        _Case("drop_door_orientation",      "drop_door orientation 후 w/h=None, x/y 보존",                       case_drop_door_orientation),
        _Case("drop_spatial",               "drop_spatial 후 해당 idx 제거",                                    case_drop_spatial),
        _Case("drop_front_door",            "drop_front_door=True → front_door=None",                           case_drop_front_door),
        _Case("drop_front_door_coords",     "drop_front_door_coords → x/y=None, w/h 보존",                       case_drop_front_door_coords),
        _Case("drop_room_summary_total",    "drop_room_summary_total → total_rooms=None",                       case_drop_room_summary_total),
        _Case("drop_room_summary_types",    "drop_room_summary_types → type_counts에서 type 제거",              case_drop_room_summary_types),
        _Case("drop_block_total_asymmetric", "★ 비대칭: total_rooms=4 vs visible 방 3개",                       case_drop_block_with_total_intact),
        _Case("combined_drops",             "drop_block + drop_type + drop_coords 조합",                        case_combined_drops),
    ]
    results = run_cases(cases, lambda c: c.fn(), label="Group 1: metadata after drops")
    summary_and_exit(results, label="metadata after drops")


if __name__ == "__main__":
    main()
