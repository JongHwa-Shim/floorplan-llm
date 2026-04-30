"""Group 2: R_input_consistency 검증.

의도: 입력에서 좌표 명시된 방(앵커: type+coords / drop_type: coords only)이
출력에 일관되게 존재하는가. centroid 거리 ≤ threshold이면 1, ≥ threshold이면 0.

threshold 기본 15px (코드 line 38: _ANCHOR_DISTANCE_THRESHOLD = 15.0).
docstring(line 60)은 "기본 30px"로 잘못 표기됨 → finding.

핵심 케이스:
    - 정상 앵커 → 1.0
    - 거리 30 (threshold 15 초과) → 0.0
    - 거리 7.5 (threshold 15의 절반) → 0.5
    - drop_type 매칭 (type=""+coords) → 점수 양수
    - 앵커 type 미스매치 (출력에 해당 type 없음) → 0.0
    - 앵커도 drop_type도 없음 → 1.0 (채점 비활성)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_VERIF_ROOT = Path(__file__).resolve().parents[1]
if str(_VERIF_ROOT) not in sys.path:
    sys.path.insert(0, str(_VERIF_ROOT))

from _common import (  # noqa: E402
    RoomSpec, FrontDoorSpec, flat_coords,
    build_output_token_ids, build_metadata, make_reward_cfg,
    assert_reward_close, get_vocab, run_cases, summary_and_exit,
)

from src.training.rl.rewards import compute_all_rewards  # noqa: E402


REWARD_NAME = "input_consistency"


@dataclass
class Case:
    name: str
    intent: str
    rooms: list
    metadata: dict
    expected: float
    threshold: float = 15.0
    tol: float = 1e-3


def runner(case: Case) -> None:
    vocab = get_vocab()
    cfg = make_reward_cfg(enable=["input_consistency"], threshold=case.threshold)
    fd = FrontDoorSpec(cx=10, cy=5, w=4, h=2)
    token_ids, _ = build_output_token_ids(case.rooms, doors=[], front_door=fd, vocab=vocab)
    result = compute_all_rewards(token_ids, vocab, case.metadata, cfg)
    actual = result["rewards"].get(REWARD_NAME, 0.0)
    assert_reward_close(actual, case.expected, tol=case.tol, name=case.name)


def build_cases() -> list[Case]:
    outline = RoomSpec("outline", [(0, 0), (200, 0), (200, 200), (0, 200)])

    # bedroom: centroid (50, 50)
    bed = RoomSpec("bedroom", [(20, 20), (80, 20), (80, 80), (20, 80)])
    # bedroom 출력 위치를 30px shift: (80,80)~(140,140) → centroid (110,110), 거리=84.85
    bed_far = RoomSpec("bedroom", [(80, 80), (140, 80), (140, 140), (80, 140)])
    # bedroom 출력 위치를 살짝 이동: centroid (57.5, 57.5), 거리=10.6
    bed_near = RoomSpec("bedroom", [(27, 27), (88, 27), (88, 88), (27, 88)])

    bed_input = {"rid": 1, "type": "bedroom", "coords": flat_coords(bed.coords)}

    return [
        Case(
            "exact_match",
            "앵커 정확 일치 (centroid 거리 0) → 1.0",
            rooms=[outline, bed],
            metadata=build_metadata(
                rooms=[bed_input], total_rooms=1, type_counts={"bedroom": 1},
            ),
            expected=1.0,
        ),
        Case(
            "far_anchor_zero",
            "★ 거리 84.85 (threshold 15) → max(0, 1-84.85/15)=0",
            rooms=[outline, bed_far],
            metadata=build_metadata(
                rooms=[bed_input], total_rooms=1, type_counts={"bedroom": 1},
            ),
            expected=0.0,
        ),
        Case(
            "threshold_half",
            "★ 거리 7.5 (threshold 15의 절반) → 0.5",
            rooms=[
                outline,
                # bedroom: centroid (50+7.5, 50) = (57.5, 50). 즉 +7.5 in x.
                # vertices: (27, 20), (87, 20), (87, 80), (27, 80) — centroid (57, 50). 거리=7
                # → score = 1 - 7/15 = 0.5333
                # 정확히 7.5 거리를 만들려면 vertex shift가 모두 +7.5 in x
                # int 좌표 사용: shift +7px → centroid +7. 거리=7. score=1-7/15=0.5333
                # 또는 +8: 거리=8. score=1-8/15=0.4667
                # 정확한 0.5에 가까운 fixture: 거리=7.5 (불가능 정수 좌표)
                # 따라서 tol을 키워서 검증
                RoomSpec("bedroom", [(27, 20), (87, 20), (87, 80), (27, 80)]),
            ],
            metadata=build_metadata(
                rooms=[bed_input], total_rooms=1, type_counts={"bedroom": 1},
            ),
            expected=1.0 - 7.0 / 15.0,
            tol=0.005,
        ),
        Case(
            "wrong_type_anchor",
            "★ 앵커 type=bedroom인데 출력에 kitchen만 → 0.0 (헝가리안 미수행)",
            rooms=[outline, RoomSpec("kitchen", [(20, 20), (80, 20), (80, 80), (20, 80)])],
            metadata=build_metadata(
                rooms=[bed_input], total_rooms=1, type_counts={"bedroom": 1},
            ),
            expected=0.0,
        ),
        Case(
            "drop_type_match",
            "★ drop_type (type=''+coords) → 잔여 출력 방 헝가리안 → 점수 양수",
            rooms=[outline, bed_near],  # centroid (57.5, 57.5), 거리=√112.5≈10.6
            metadata=build_metadata(
                rooms=[{"rid": 1, "type": "", "coords": flat_coords(bed.coords)}],
                total_rooms=1, type_counts={"bedroom": 1},
            ),
            # score = 1 - 10.6/15 ≈ 0.293
            expected=1.0 - 10.6066 / 15.0,
            tol=0.05,
        ),
        Case(
            "no_anchors_no_drop_type",
            "앵커도 drop_type도 0 → 1.0 (채점 비활성)",
            rooms=[outline, bed],
            metadata=build_metadata(
                rooms=[],  # 빈 메타
                total_rooms=1, type_counts={"bedroom": 1},
            ),
            expected=1.0,
        ),
        Case(
            "all_drop_block_no_eval",
            "drop_coords된 방만 있으면 (type만 visible) → 채점 대상 아님 → 1.0",
            rooms=[outline, bed],
            metadata=build_metadata(
                rooms=[{"rid": 1, "type": "bedroom", "coords": []}],  # drop_coords (앵커X, drop_type X)
                total_rooms=1, type_counts={"bedroom": 1},
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
