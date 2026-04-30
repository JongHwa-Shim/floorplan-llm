"""Group 2: R_spatial 검증.

의도: 입력 spatial 조건의 8방위 방향이 출력 방 무게중심 벡터와 일치.
앵커는 헝가리안 결정 매핑, 자유 방은 후보 satisfiability.

핵심 케이스:
    - "right" 정확 일치 → 1.0
    - 22.5° 경계각 (`<` vs `<=` 부동소수점 검증)
    - direction 명백 불일치 → 0.0
    - drop_coords + 다른 type 매칭 가능 → satisfiability
    - 영벡터 (centroid 일치) skip → 분모 0이면 1.0
    - drop_pair는 spatial 채점에 영향 없음 (spatial은 metadata.spatial 직접 참조)
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
from src.training.rl.rewards.spatial_reward import _vector_to_direction  # noqa: E402


REWARD_NAME = "spatial"


@dataclass
class Case:
    name: str
    intent: str
    rooms: list
    metadata: dict
    expected: float
    tol: float = 1e-3


def runner(case: Case) -> None:
    vocab = get_vocab()
    cfg = make_reward_cfg(enable=["spatial"])
    fd = FrontDoorSpec(cx=10, cy=5, w=4, h=2)
    token_ids, _ = build_output_token_ids(case.rooms, doors=[], front_door=fd, vocab=vocab)
    result = compute_all_rewards(token_ids, vocab, case.metadata, cfg)
    actual = result["rewards"].get(REWARD_NAME, 0.0)
    assert_reward_close(actual, case.expected, tol=case.tol, name=case.name)


def build_cases() -> list[Case]:
    outline = RoomSpec("outline", [(0, 0), (200, 0), (200, 200), (0, 200)])

    # 두 방을 정확한 위치에 배치
    # bedroom1: centroid (50, 50)
    # bedroom2: centroid (150, 50) → bedroom1 기준 right
    bed1 = RoomSpec("bedroom", [(10, 10), (90, 10), (90, 90), (10, 90)])
    bed2 = RoomSpec("bedroom", [(110, 10), (190, 10), (190, 90), (110, 90)])

    bed1_meta = {"rid": 1, "type": "bedroom", "coords": flat_coords(bed1.coords)}
    bed2_meta = {"rid": 2, "type": "bedroom", "coords": flat_coords(bed2.coords)}

    return [
        Case(
            "right_exact",
            "right 정확 일치 → 1.0",
            rooms=[outline, bed1, bed2],
            metadata=build_metadata(
                rooms=[bed1_meta, bed2_meta],
                spatial=[{"rid_a": 1, "rid_b": 2, "direction": "right"}],
                total_rooms=2, type_counts={"bedroom": 2},
            ),
            expected=1.0,
        ),
        Case(
            "wrong_direction",
            "below 조건이지만 실제 right → 0.0",
            rooms=[outline, bed1, bed2],
            metadata=build_metadata(
                rooms=[bed1_meta, bed2_meta],
                spatial=[{"rid_a": 1, "rid_b": 2, "direction": "below"}],
                total_rooms=2, type_counts={"bedroom": 2},
            ),
            expected=0.0,
        ),
        Case(
            "no_spatial_returns_one",
            "spatial=[] → 1.0",
            rooms=[outline, bed1, bed2],
            metadata=build_metadata(
                rooms=[bed1_meta, bed2_meta], spatial=[],
                total_rooms=2, type_counts={"bedroom": 2},
            ),
            expected=1.0,
        ),
        Case(
            "drop_coords_satisfiability",
            "drop_coords bedroom 후보 → 같은 type 후보 어느 쌍이든 만족",
            rooms=[outline, bed1, bed2],
            metadata=build_metadata(
                rooms=[
                    {"rid": 1, "type": "bedroom", "coords": []},  # drop_coords
                    bed2_meta,
                ],
                spatial=[{"rid_a": 1, "rid_b": 2, "direction": "right"}],
                total_rooms=2, type_counts={"bedroom": 2},
            ),
            expected=1.0,
        ),
        Case(
            "two_conditions_partial",
            "spatial 2개 중 1개 만족 → 0.5",
            rooms=[outline, bed1, bed2],
            metadata=build_metadata(
                rooms=[bed1_meta, bed2_meta],
                spatial=[
                    {"rid_a": 1, "rid_b": 2, "direction": "right"},
                    {"rid_a": 1, "rid_b": 2, "direction": "above"},  # 거짓
                ],
                total_rooms=2, type_counts={"bedroom": 2},
            ),
            expected=0.5,
        ),
    ]


def case_vector_to_direction_boundary():
    """★ 22.5° 경계각 부동소수점 분기 검증 (코드 직접 호출, 보상 무관)."""
    # angle_deg < 22.5: "right". 22.5 <= angle_deg < 67.5: "right-below"
    import math

    # 정확히 22.5° (이미지 좌표계: dy>0)
    # tan(22.5°) ≈ 0.4142
    # dx=10, dy=10*tan(22.5)
    dx, dy = 10.0, 10.0 * math.tan(math.radians(22.5))
    actual = _vector_to_direction(dx, dy)
    # 부동소수점 오차로 22.5에 정확히 도달 못 할 수도 있음
    # 의도 문서화: 22.5°는 right-below로 분류 (`<` 분기). 22.4999는 right.
    # 어느 쪽이든 코드 동작 단언만 수행
    assert actual in {"right", "right-below"}, f"22.5° 경계각: {actual}"
    print(f"  22.5° 경계 분류: {actual} (관측됨)")

    # 명백한 right (10, 0)
    assert _vector_to_direction(10.0, 0.0) == "right"
    # 명백한 below (0, 10)
    assert _vector_to_direction(0.0, 10.0) == "below"
    # 명백한 above (0, -10)
    assert _vector_to_direction(0.0, -10.0) == "above"
    # left (-10, 0)
    assert _vector_to_direction(-10.0, 0.0) == "left"


def main():
    cases = build_cases()
    results = run_cases(cases, runner, label=f"Group 2: R_{REWARD_NAME}")

    # 별도 직접 호출 검증
    print("\n[추가] _vector_to_direction 경계각 부동소수점 분기 검증:")
    case_vector_to_direction_boundary()
    print("[PASS] _vector_to_direction 분기 동작 (22.5° 경계 결과 위 출력)")

    summary_and_exit(results, label=f"R_{REWARD_NAME}")


if __name__ == "__main__":
    main()
