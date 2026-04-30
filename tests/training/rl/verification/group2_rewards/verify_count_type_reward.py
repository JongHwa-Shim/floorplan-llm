"""Group 2: R_count_type 검증.

의도: 노출된 type별 개수 정확도 평균 (drop된 type은 채점 제외).
이전 버그: 출력에 hallucinated/drop된 type 포함 시 부당 0점 — 수정 후 무시.

핵심 케이스:
    - drop된 type을 모델이 출력 → 무시 (채점 제외) → 1.0
    - 부분 일치 (3 expected, 2 actual) → 0.667
    - hallucinated type (expected에 없는 type 출력) → 무시 (silent allow). 이는
      의도일 수 있지만 사용자 의도 확인 필요 → finding 후보.
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


REWARD_NAME = "count_type"


@dataclass
class Case:
    name: str
    intent: str
    rooms: list
    metadata: dict
    expected: float


def runner(case: Case) -> None:
    vocab = get_vocab()
    cfg = make_reward_cfg(enable=["count_type"])
    fd = FrontDoorSpec(cx=105, cy=10, w=8, h=2)
    token_ids, _ = build_output_token_ids(case.rooms, doors=[], front_door=fd, vocab=vocab)
    result = compute_all_rewards(token_ids, vocab, case.metadata, cfg)
    actual = result["rewards"].get(REWARD_NAME, 0.0)
    assert_reward_close(actual, case.expected, name=case.name)


def _make_room(t: str, x: int) -> RoomSpec:
    return RoomSpec(t, [(x, 20), (x + 30, 20), (x + 30, 90), (x, 90)])


def build_cases() -> list[Case]:
    outline = RoomSpec("outline", [(10, 10), (200, 10), (200, 200), (10, 200)])

    return [
        Case(
            "exact_match",
            "expected={bedroom:2, kitchen:1}, 출력 동일 → 1.0",
            rooms=[outline, _make_room("bedroom", 20), _make_room("bedroom", 60), _make_room("kitchen", 100)],
            metadata=build_metadata(total_rooms=3, type_counts={"bedroom": 2, "kitchen": 1}),
            expected=1.0,
        ),
        Case(
            "partial_match_two_thirds",
            "expected bedroom 3, 출력 2 → score=2/3 (단일 타입이므로 평균=2/3)",
            rooms=[outline, _make_room("bedroom", 20), _make_room("bedroom", 60)],
            metadata=build_metadata(total_rooms=3, type_counts={"bedroom": 3}),
            expected=2.0 / 3.0,
        ),
        Case(
            "drop_type_ignored",
            "★ drop_room_summary_types로 kitchen 제외, 출력에 kitchen 1 → 1.0 (이전 버그)",
            rooms=[outline, _make_room("bedroom", 20), _make_room("bedroom", 60), _make_room("kitchen", 100)],
            metadata=build_metadata(total_rooms=3, type_counts={"bedroom": 2}),  # kitchen 키 없음
            expected=1.0,
        ),
        Case(
            "halluc_type_silent_allow",
            "★ expected={bedroom:2}, 출력에 bedroom 2 + storage 1 (hallucinated) → 1.0",
            rooms=[outline, _make_room("bedroom", 20), _make_room("bedroom", 60), _make_room("storage", 100)],
            metadata=build_metadata(total_rooms=2, type_counts={"bedroom": 2}),
            expected=1.0,  # storage가 expected에 없으므로 무시 — finding: silent allow
        ),
        Case(
            "empty_type_counts",
            "type_counts={} → 1.0 (채점 비활성)",
            rooms=[outline, _make_room("bedroom", 20)],
            metadata=build_metadata(total_rooms=1, type_counts={}),
            expected=1.0,
        ),
        Case(
            "missing_expected_type",
            "expected={bedroom:1, kitchen:1}, 출력에 bedroom만 → score=(1+0)/2=0.5",
            rooms=[outline, _make_room("bedroom", 20)],
            metadata=build_metadata(total_rooms=2, type_counts={"bedroom": 1, "kitchen": 1}),
            expected=0.5,
        ),
        Case(
            "expected_zero_actual_zero",
            "expected={bedroom:0}, 출력 0 → 1.0 (channel skip)",
            rooms=[outline, _make_room("kitchen", 20)],
            metadata=build_metadata(total_rooms=1, type_counts={"bedroom": 0}),
            expected=1.0,  # 모든 케이스 skip → scores=[] → return 1.0
        ),
    ]


def main():
    cases = build_cases()
    results = run_cases(cases, runner, label=f"Group 2: R_{REWARD_NAME}")
    summary_and_exit(results, label=f"R_{REWARD_NAME}")


if __name__ == "__main__":
    main()
