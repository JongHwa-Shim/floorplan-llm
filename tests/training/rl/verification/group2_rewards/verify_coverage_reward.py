"""논문 정렬 회귀 검증: 모든 필수 조건을 충족할 때만 1을 반환한다."""

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
    assert_reward_close, assert_error_mask_absent,
    get_vocab, run_cases, summary_and_exit,
)

from src.training.rl.rewards import compute_all_rewards  # noqa: E402


REWARD_NAME = "coverage"


@dataclass
class Case:
    name: str
    intent: str
    rooms: list
    expected_reward: float
    tol: float = 0.01


def runner(case: Case) -> None:
    vocab = get_vocab()
    cfg = make_reward_cfg(
        enable=["coverage"],
        # CA를 의도적으로 True로 두어도 coverage는 무시하는지 검증
        credit_assignment={"coverage": True},
    )
    fd = FrontDoorSpec(cx=10, cy=5, w=4, h=2)
    metadata = build_metadata(total_rooms=len(case.rooms) - 1, type_counts={})
    token_ids, _ = build_output_token_ids(case.rooms, doors=[], front_door=fd, vocab=vocab)
    result = compute_all_rewards(token_ids, vocab, metadata, cfg)
    actual = result["rewards"].get(REWARD_NAME, 0.0)
    assert_reward_close(actual, case.expected_reward, tol=case.tol, name=case.name)
    # 신용할당 OFF 확인 — coverage는 error_masks dict에 키가 없어야 함
    assert_error_mask_absent(result["error_masks"], REWARD_NAME)


def build_cases() -> list[Case]:
    outline = RoomSpec("outline", [(0, 0), (200, 0), (200, 200), (0, 200)])  # area=40000

    return [
        Case(
            "fully_covered",
            "방들이 outline 완전 분할 채움 → 1.0",
            rooms=[
                outline,
                RoomSpec("bedroom", [(0, 0), (100, 0), (100, 200), (0, 200)]),    # left half (20000)
                RoomSpec("kitchen", [(100, 0), (200, 0), (200, 200), (100, 200)]), # right half (20000)
            ],
            expected_reward=1.0,
        ),
        Case(
            "quarter_filled",
            "★ 작은 방 1개로 25% 채움 (50×50/100×100 outline) → 피복 임계값 0.774 미달로 0",
            rooms=[
                RoomSpec("outline", [(0, 0), (100, 0), (100, 100), (0, 100)]),  # area=10000
                RoomSpec("bedroom", [(0, 0), (50, 0), (50, 50), (0, 50)]),      # area=2500
            ],
            # 빈공간 = 7500, ratio=0.75, reward=0.25
            expected_reward=0.0,
            tol=0.01,
        ),
        Case(
            "half_filled",
            "방 1개로 50% 채움 → 임계값 미달로 0",
            rooms=[
                outline,
                RoomSpec("bedroom", [(0, 0), (200, 0), (200, 100), (0, 100)]),  # 20000/40000
            ],
            expected_reward=0.0,
        ),
        Case(
            "outside_outline",
            "방이 outline 밖 (overlap 없음) → 0.0",
            rooms=[
                RoomSpec("outline", [(0, 0), (50, 0), (50, 50), (0, 50)]),
                RoomSpec("bedroom", [(100, 100), (150, 100), (150, 150), (100, 150)]),
            ],
            expected_reward=0.0,
        ),
        Case(
            "no_outline_returns_zero",
            "outline 없음 → 0.0",
            rooms=[
                RoomSpec("bedroom", [(20, 20), (100, 20), (100, 100), (20, 100)]),
                RoomSpec("kitchen", [(120, 120), (180, 120), (180, 180), (120, 180)]),
            ],
            expected_reward=0.0,
        ),
        Case(
            "overlapping_rooms_no_double_count",
            "겹친 방 둘 → unary_union으로 합집합 면적 계산 (이중 계산 X)",
            rooms=[
                RoomSpec("outline", [(0, 0), (100, 0), (100, 100), (0, 100)]),  # 10000
                # 두 방 모두 (0,0)~(60,60), (40,40)~(100,100) 겹침
                RoomSpec("bedroom", [(0, 0), (60, 0), (60, 60), (0, 60)]),    # 3600
                RoomSpec("kitchen", [(40, 40), (100, 40), (100, 100), (40, 100)]),  # 3600
            ],
            # 합집합: 3600 + 3600 - 400 = 6800. 빈공간 = 3200. ratio=0.32. reward=0.68
            expected_reward=0.0,
            tol=0.02,
        ),
    ]


def main():
    cases = build_cases()
    results = run_cases(cases, runner, label=f"Group 2: R_{REWARD_NAME}")
    summary_and_exit(results, label=f"R_{REWARD_NAME}")


if __name__ == "__main__":
    main()
