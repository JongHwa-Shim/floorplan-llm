"""Group 2: R_count_total 검증.

의도: ROOM_SUMMARY로 노출된 N(metadata.total_rooms)과 출력 비-outline 방 개수가
일치하는지 이진 판정. total_rooms=None이면 채점 비활성 (drop_room_summary_total).

핵심 비대칭: total_rooms (ROOM_SUMMARY 노출) ≠ len(metadata.rooms) (drop_block 제외).
count_total은 metadata.total_rooms을 따른다.
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
    assert_reward_close, get_vocab, run_cases, summary_and_exit,
)

from src.training.rl.rewards import compute_all_rewards  # noqa: E402


REWARD_NAME = "count_total"


@dataclass
class Case:
    name: str
    intent: str
    rooms: list
    metadata: dict
    expected: float


def runner(case: Case) -> None:
    vocab = get_vocab()
    cfg = make_reward_cfg(enable=["count_total"])
    fd = FrontDoorSpec(cx=105, cy=10, w=8, h=2)
    token_ids, _ = build_output_token_ids(case.rooms, doors=[], front_door=fd, vocab=vocab)
    result = compute_all_rewards(token_ids, vocab, case.metadata, cfg)
    actual = result["rewards"].get(REWARD_NAME, 0.0)
    assert_reward_close(actual, case.expected, name=case.name)


def build_cases() -> list[Case]:
    outline = RoomSpec("outline", [(10, 10), (200, 10), (200, 200), (10, 200)])
    bedroom1 = RoomSpec("bedroom", [(20, 20), (90, 20), (90, 90), (20, 90)])
    bedroom2 = RoomSpec("bedroom", [(110, 20), (190, 20), (190, 90), (110, 90)])
    kitchen  = RoomSpec("kitchen", [(20, 110), (190, 110), (190, 190), (20, 190)])

    return [
        Case(
            "total_none_skipped",
            "metadata.total_rooms=None → 채점 비활성 (1.0)",
            rooms=[outline, bedroom1],
            metadata=build_metadata(total_rooms=None, type_counts={"bedroom": 1}),
            expected=1.0,
        ),
        Case(
            "exact_match",
            "total=2, 출력 비-outline 2개 → 1.0",
            rooms=[outline, bedroom1, bedroom2],
            metadata=build_metadata(total_rooms=2, type_counts={"bedroom": 2}),
            expected=1.0,
        ),
        Case(
            "off_by_one",
            "total=3, 출력 2개 → 0.0",
            rooms=[outline, bedroom1, bedroom2],
            metadata=build_metadata(total_rooms=3, type_counts={"bedroom": 3}),
            expected=0.0,
        ),
        Case(
            "drop_block_asymmetry",
            "★ total=3, len(rooms)=2 (drop_block), 출력 3 → 1.0 (total_rooms 기준)",
            rooms=[outline, bedroom1, bedroom2, kitchen],
            metadata=build_metadata(
                total_rooms=3,                     # ROOM_SUMMARY 노출 N
                type_counts={"bedroom": 2, "kitchen": 1},
                rooms=[                            # drop_block으로 visible은 2개
                    {"rid": 0, "type": "outline", "coords": [10, 10, 200, 10, 200, 200, 10, 200]},
                    {"rid": 1, "type": "bedroom", "coords": [20, 20, 90, 20, 90, 90, 20, 90]},
                    {"rid": 3, "type": "kitchen", "coords": [20, 110, 190, 110, 190, 190, 20, 190]},
                ],
            ),
            expected=1.0,
        ),
        Case(
            "extra_output",
            "total=2, 출력 3개 (모델이 더 많이 생성) → 0.0",
            rooms=[outline, bedroom1, bedroom2, kitchen],
            metadata=build_metadata(total_rooms=2, type_counts={"bedroom": 2}),
            expected=0.0,
        ),
    ]


def main():
    cases = build_cases()
    results = run_cases(cases, runner, label=f"Group 2: R_{REWARD_NAME}")
    summary_and_exit(results, label=f"R_{REWARD_NAME}")


if __name__ == "__main__":
    main()
