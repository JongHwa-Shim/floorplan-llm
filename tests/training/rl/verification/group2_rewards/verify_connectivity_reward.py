"""Group 2: R_connectivity 검증.

의도: 입력 edge.door가 두 방의 경계 근방(≤ 20px)에 존재하는지.
앵커는 헝가리안 결정 매핑, drop_coords/drop_type 자유 방은 후보 satisfiability.

핵심 케이스:
    - 정상: edge에 door 있고 두 방 사이 → 1.0
    - door 누락 (출력에 doors=[]) → 0.0
    - drop_coords 같은 type 후보 satisfiability → 어느 한 쌍이라도 만족 → 1.0
    - drop_type 좌표 근접 후보 satisfiability → 1.0
    - drop_pair "both" → 분모 제외, 1.0
    - **★ same-type 다중 앵커 ambiguity**: 모델이 두 방 위치를 swap해서 출력해도
      헝가리안이 매칭으로 잡아냄. door가 다른 같은 type 방 사이에 있으면 false negative.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_VERIF_ROOT = Path(__file__).resolve().parents[1]
if str(_VERIF_ROOT) not in sys.path:
    sys.path.insert(0, str(_VERIF_ROOT))

from _common import (  # noqa: E402
    RoomSpec, DoorSpec, FrontDoorSpec, flat_coords,
    build_output_token_ids, build_metadata, make_reward_cfg,
    assert_reward_close, get_vocab, run_cases, summary_and_exit,
)

from src.training.rl.rewards import compute_all_rewards  # noqa: E402


REWARD_NAME = "connectivity"


@dataclass
class Case:
    name: str
    intent: str
    rooms: list
    doors: list
    metadata: dict
    expected: float
    tol: float = 1e-3


def runner(case: Case) -> None:
    vocab = get_vocab()
    cfg = make_reward_cfg(enable=["connectivity"])
    fd = FrontDoorSpec(cx=10, cy=5, w=4, h=2)
    token_ids, _ = build_output_token_ids(case.rooms, doors=case.doors, front_door=fd, vocab=vocab)
    result = compute_all_rewards(token_ids, vocab, case.metadata, cfg)
    actual = result["rewards"].get(REWARD_NAME, 0.0)
    assert_reward_close(actual, case.expected, tol=case.tol, name=case.name)


def build_cases() -> list[Case]:
    outline = RoomSpec("outline", [(0, 0), (200, 0), (200, 200), (0, 200)])
    bed1 = RoomSpec("bedroom", [(10, 10), (100, 10), (100, 100), (10, 100)])
    bed2 = RoomSpec("bedroom", [(110, 10), (190, 10), (190, 100), (110, 100)])
    kit  = RoomSpec("kitchen", [(10, 110), (100, 110), (100, 190), (10, 190)])

    # metadata 방 dict는 flat coords
    bed1_meta = {"rid": 1, "type": "bedroom", "coords": flat_coords(bed1.coords)}
    bed2_meta = {"rid": 2, "type": "bedroom", "coords": flat_coords(bed2.coords)}
    kit_meta  = {"rid": 3, "type": "kitchen", "coords": flat_coords(kit.coords)}

    return [
        Case(
            "anchor_pass",
            "★ 앵커 매칭: edge=(1,2), door가 두 방 경계 (x=100) 위 → 1.0",
            rooms=[outline, bed1, bed2],
            doors=[DoorSpec(cx=100, cy=50, w=2, h=10)],  # bedroom1과 bedroom2 경계
            metadata=build_metadata(
                rooms=[bed1_meta, bed2_meta],
                edges=[{"pair": [1, 2], "door": [{"x": 100, "y": 50, "w": 2, "h": 10}]}],
                total_rooms=2, type_counts={"bedroom": 2},
            ),
            expected=1.0,
        ),
        Case(
            "missing_door",
            "edge=(1,2) 조건 있는데 출력 doors=[] → 0.0",
            rooms=[outline, bed1, bed2],
            doors=[],
            metadata=build_metadata(
                rooms=[bed1_meta, bed2_meta],
                edges=[{"pair": [1, 2], "door": [{"x": 100, "y": 50, "w": 2, "h": 10}]}],
                total_rooms=2, type_counts={"bedroom": 2},
            ),
            expected=0.0,
        ),
        Case(
            "drop_pair_both_excluded",
            "★ drop_pair both → pair=[] → 분모 제외 → 1.0",
            rooms=[outline, bed1, bed2],
            doors=[],
            metadata=build_metadata(
                rooms=[bed1_meta, bed2_meta],
                edges=[{"pair": [], "door": [{"x": 100, "y": 50, "w": 2, "h": 10}]}],
                total_rooms=2, type_counts={"bedroom": 2},
            ),
            expected=1.0,
        ),
        Case(
            "no_door_in_edge_skipped",
            "edge.door=[] (drop_door all) → 분모 제외 → 1.0",
            rooms=[outline, bed1, bed2],
            doors=[],
            metadata=build_metadata(
                rooms=[bed1_meta, bed2_meta],
                edges=[{"pair": [1, 2], "door": []}],
                total_rooms=2, type_counts={"bedroom": 2},
            ),
            expected=1.0,
        ),
        Case(
            "drop_coords_satisfiability",
            "★ drop_coords (bedroom 같은 type 두 방 중 어느 하나에라도 door) → 1.0",
            rooms=[outline, bed1, bed2, kit],
            doors=[DoorSpec(cx=100, cy=50, w=2, h=10)],  # bed1-bed2 경계
            metadata=build_metadata(
                rooms=[
                    {"rid": 1, "type": "bedroom", "coords": []},  # drop_coords (같은 type 후보 확장)
                    bed2_meta,
                    kit_meta,
                ],
                edges=[{"pair": [1, 3], "door": [{"x": 100, "y": 50, "w": 2, "h": 10}]}],
                # rid=1 (drop_coords bedroom)과 rid=3 (kitchen) 사이 door 조건
                # 후보 확장: rid=1 → bedroom 출력 모두 (bed1, bed2). rid=3 → kit (앵커, 단 1)
                # bed1과 kit 사이엔 door 없음 (door는 (100,50)에 있음)
                # bed2와 kit 사이도 door 없음
                # 따라서 어느 후보 쌍도 만족 안 함 → 0.0
                total_rooms=3, type_counts={"bedroom": 1, "kitchen": 1},
            ),
            expected=0.0,  # 후보 satisfiability이지만 어느 쌍도 door로 연결 안 됨
        ),
        Case(
            "drop_coords_satisfiable_match",
            "★ drop_coords + door가 후보 사이 → 1.0",
            rooms=[outline, bed1, bed2],
            doors=[DoorSpec(cx=100, cy=50, w=2, h=10)],  # bed1-bed2 경계
            metadata=build_metadata(
                rooms=[
                    {"rid": 1, "type": "bedroom", "coords": []},  # drop_coords
                    bed2_meta,
                ],
                edges=[{"pair": [1, 2], "door": [{"x": 100, "y": 50, "w": 2, "h": 10}]}],
                total_rooms=2, type_counts={"bedroom": 2},
            ),
            expected=1.0,
        ),
        Case(
            "no_edges_returns_one",
            "edges=[] → 1.0 (조건 없음)",
            rooms=[outline, bed1, bed2],
            doors=[],
            metadata=build_metadata(
                rooms=[bed1_meta, bed2_meta], edges=[],
                total_rooms=2, type_counts={"bedroom": 2},
            ),
            expected=1.0,
        ),
    ]


def main():
    cases = build_cases()
    results = run_cases(cases, runner, label=f"Group 2: R_{REWARD_NAME}")
    summary_and_exit(results, label=f"R_{REWARD_NAME}")


if __name__ == "__main__":
    main()
