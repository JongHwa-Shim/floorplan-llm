"""Group 2: R_outline_in_room 검증 (케이스 B).

의도: outline 꼭짓점이 비-outline 방 내부에 갇히는지 (case B detection).
방 꼭짓점은 모두 outline 내부지만 방의 edge가 outline 오목부를 가로지르는 경우
outline의 reflex 꼭짓점이 방 내부에 들어갈 수 있다.
신용할당: 갇힌 outline 꼭짓점에 가장 가까운 방 꼭짓점의 X/Y 토큰 마킹.

핵심 케이스:
    - 정상: 모든 outline 꼭짓점이 방 외부 → 1.0
    - 케이스 B: L자 outline + 사각형 방이 오목부 가로지름 → <1, 가장 가까운 방 꼭짓점 error
    - 경계 false positive 가드: 방 경계가 outline 꼭짓점 정확히 통과 → 1.0 (contains exclusive)
    - outline 꼭짓점 0개 (outline 없음) → 1.0
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_VERIF_ROOT = Path(__file__).resolve().parents[1]
if str(_VERIF_ROOT) not in sys.path:
    sys.path.insert(0, str(_VERIF_ROOT))

from _common import (  # noqa: E402
    RoomSpec, FrontDoorSpec,
    build_output_token_ids, build_metadata, make_reward_cfg,
    assert_reward_close, assert_error_indices_contains,
    get_vocab, run_cases, summary_and_exit,
)

from src.training.rl.rewards import compute_all_rewards  # noqa: E402


REWARD_NAME = "outline_in_room"


@dataclass
class Case:
    name: str
    intent: str
    rooms: list
    expected_reward: float
    expected_error_vertices: list = None  # 또는 None
    tol: float = 1e-3


def runner(case: Case) -> None:
    vocab = get_vocab()
    cfg = make_reward_cfg(
        enable=["outline_in_room"],
        credit_assignment={"outline_in_room": True},
        penalty_scale={"outline_in_room": 1.0},
    )
    fd = FrontDoorSpec(cx=10, cy=5, w=4, h=2)
    metadata = build_metadata(total_rooms=len(case.rooms) - 1, type_counts={})

    token_ids, idx_map = build_output_token_ids(case.rooms, doors=[], front_door=fd, vocab=vocab)
    result = compute_all_rewards(token_ids, vocab, metadata, cfg)
    actual = result["rewards"].get(REWARD_NAME, 0.0)
    assert_reward_close(actual, case.expected_reward, tol=case.tol, name=case.name)

    mask = result["error_masks"].get(REWARD_NAME)
    if case.expected_error_vertices is not None:
        expected_positions: list[int] = []
        for r_idx, v_idx in case.expected_error_vertices:
            x_pos = idx_map.room_vertex_x[(r_idx, v_idx)]
            expected_positions.extend([x_pos, x_pos + 1])
        assert_error_indices_contains(mask, expected_positions, name=case.name)


def build_cases() -> list[Case]:
    rect_outline = RoomSpec("outline", [(0, 0), (200, 0), (200, 200), (0, 200)])
    # L자 outline: (0,0)→(100,0)→(100,100)→(200,100)→(200,200)→(0,200)
    # reflex 꼭짓점: (100,100)
    l_outline = RoomSpec("outline", [(0, 0), (100, 0), (100, 100), (200, 100), (200, 200), (0, 200)])

    return [
        Case(
            "all_outside",
            "정상: outline 꼭짓점 모두 방 외부 → 1.0",
            rooms=[
                rect_outline,
                RoomSpec("bedroom", [(20, 20), (90, 20), (90, 90), (20, 90)]),
            ],
            expected_reward=1.0,
            expected_error_vertices=[],
        ),
        Case(
            "case_b_l_concavity",
            "★ 케이스 B: L자 outline + 사각방이 오목부 가로지름 (reflex (100,100) 트랩)",
            rooms=[
                l_outline,
                # 방: (50,50)~(150,150) — 모든 꼭짓점은 outline 내부, 그러나 (100,100) reflex 꼭짓점이 방 내부
                RoomSpec("bedroom", [(50, 50), (150, 50), (150, 150), (50, 150)]),
            ],
            # outline 꼭짓점 6개 중 (100,100)만 trapped
            # reward = 1 - 1/6
            expected_reward=1.0 - 1.0 / 6.0,
            # 가장 가까운 방 꼭짓점 — (50,50)~(150,150) 중 (100,100)에 가장 가까운 것은
            # (50,50) / (150,50) / (150,150) / (50,150) 중 어느 것도 거리 동일 (50, 50)... 사실
            # |a-b| = sqrt((100-x)²+(100-y)²)
            # (50,50): sqrt(2500+2500) = sqrt(5000)
            # (150,50): sqrt(2500+2500) = sqrt(5000)
            # 모두 동일. 첫 번째가 선택될 것.
            # 결정적이지 않으므로 error_vertices 검사 skip
            expected_error_vertices=None,
            tol=0.01,
        ),
        Case(
            "boundary_no_false_positive",
            "★ 방 경계가 outline 꼭짓점에 정확히 닿음 → 1.0 (contains exclusive)",
            rooms=[
                rect_outline,
                # 방이 outline (200,0)에 정확히 닿는 변
                RoomSpec("bedroom", [(150, 0), (200, 0), (200, 100), (150, 100)]),
            ],
            # outline 꼭짓점 (200,0)은 방 경계 위 — contains() 사용으로 false positive 없음
            expected_reward=1.0,
            expected_error_vertices=[],
        ),
    ]


def case_no_outline_direct_call():
    """outline 없는 ParsedFloorplan을 직접 만들어 함수 자체의 동작 검증.

    compute_all_rewards()를 우회 — F-2 수정 후 outline 부재는 format hard gate에서
    막혀 다른 보상이 모두 0으로 강제된다. 여기서는 outline_in_room 함수 자체가
    "outline 없으면 1.0 반환 (채점 비활성)" 동작을 하는지 격리 검증.
    """
    from src.training.rl.rewards.outline_in_room_reward import (
        compute_outline_in_room_reward,
    )
    from src.training.rl.rewards.parser import ParsedRoom, ParsedFloorplan

    parsed = ParsedFloorplan(
        success=True, level=3, front_door=None,
        rooms=[
            ParsedRoom("bedroom", [(20, 20), (100, 20), (100, 100), (20, 100)],
                      coord_token_indices=[10, 12, 14, 16],
                      block_start=0, block_end=20),
            ParsedRoom("kitchen", [(120, 120), (180, 120), (180, 180), (120, 180)],
                      coord_token_indices=[30, 32, 34, 36],
                      block_start=22, block_end=42),
        ],
        doors=[], error_indices=[],
    )
    reward, errors = compute_outline_in_room_reward(parsed)
    assert reward == 1.0, f"outline 없으면 1.0 반환해야 함: {reward}"
    assert errors == [], f"errors=[] 기대: {errors}"


def main():
    cases = build_cases()
    results = run_cases(cases, runner, label=f"Group 2: R_{REWARD_NAME}")
    # 추가 격리 검증 (compute_all_rewards 우회)
    print("\n[추가] outline 없는 ParsedFloorplan 직접 호출:")
    case_no_outline_direct_call()
    print("[PASS] outline 없을 때 1.0 반환 (채점 비활성)")
    summary_and_exit(results, label=f"R_{REWARD_NAME}")


if __name__ == "__main__":
    main()
