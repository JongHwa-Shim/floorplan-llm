"""Group 2: R_orthogonality 검증.

의도: 모든 방의 모든 꼭짓점이 직각이어야 함. 비직각 꼭짓점만 신용할당.
영벡터(중복 꼭짓점)는 skip하여 crash 방지.

핵심 케이스:
    - 정상 직각 → 1.0, error 없음
    - 1px 미세 비직각 → <1, 그 꼭짓점만 X/Y error
    - 영벡터 (중복 꼭짓점) → skip, crash 없음
    - 한 꼭짓점만 비직각 → (n-1)/n, 그 한 점만 error
    - format=1, orthogonality<1 (의도 분리)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_VERIF_ROOT = Path(__file__).resolve().parents[1]
if str(_VERIF_ROOT) not in sys.path:
    sys.path.insert(0, str(_VERIF_ROOT))

from _common import (  # noqa: E402
    RoomSpec, FrontDoorSpec, TokenIndexMap,
    build_output_token_ids, build_metadata, make_reward_cfg,
    assert_reward_close, assert_error_indices_contains, assert_error_indices_excludes,
    get_vocab, run_cases, summary_and_exit,
)

from src.training.rl.rewards import compute_all_rewards  # noqa: E402


REWARD_NAME = "orthogonality"


@dataclass
class Case:
    name: str
    intent: str
    rooms: list
    expected_reward: float
    expected_error_vertices: list  # [(room_idx, vertex_idx), ...]
    forbidden_error_vertices: list = None
    tol: float = 1e-3


def runner(case: Case) -> None:
    vocab = get_vocab()
    cfg = make_reward_cfg(
        enable=["orthogonality"],
        credit_assignment={"orthogonality": True},
        penalty_scale={"orthogonality": 1.0},
    )
    fd = FrontDoorSpec(cx=105, cy=10, w=8, h=2)
    metadata = build_metadata(total_rooms=len(case.rooms) - 1, type_counts={})

    token_ids, idx_map = build_output_token_ids(case.rooms, doors=[], front_door=fd, vocab=vocab)
    result = compute_all_rewards(token_ids, vocab, metadata, cfg)
    actual = result["rewards"].get(REWARD_NAME, 0.0)
    assert_reward_close(actual, case.expected_reward, tol=case.tol, name=case.name)

    # error_mask 검증 (각 꼭짓점은 X/Y 두 토큰 마킹)
    expected_positions: list[int] = []
    for r_idx, v_idx in case.expected_error_vertices:
        x_pos = idx_map.room_vertex_x[(r_idx, v_idx)]
        expected_positions.extend([x_pos, x_pos + 1])

    mask = result["error_masks"].get(REWARD_NAME)
    assert_error_indices_contains(mask, expected_positions, name=case.name)

    if case.forbidden_error_vertices:
        forbidden_positions: list[int] = []
        for r_idx, v_idx in case.forbidden_error_vertices:
            x_pos = idx_map.room_vertex_x[(r_idx, v_idx)]
            forbidden_positions.extend([x_pos, x_pos + 1])
        assert_error_indices_excludes(mask, forbidden_positions, name=case.name)


def build_cases() -> list[Case]:
    outline = RoomSpec("outline", [(10, 10), (200, 10), (200, 200), (10, 200)])
    rect = RoomSpec("bedroom", [(20, 20), (100, 20), (100, 100), (20, 100)])

    return [
        Case(
            "all_orthogonal",
            "직각 outline + 직각 방 → 1.0, error 없음",
            rooms=[outline, rect],
            expected_reward=1.0,
            expected_error_vertices=[],
        ),
        Case(
            "single_vertex_off",
            "★ 한 꼭짓점 1px 어긋남 → 인접 3꼭짓점이 비직각",
            rooms=[outline, RoomSpec("bedroom", [(20, 20), (100, 20), (100, 100), (21, 99)])],
            # vertex 1만 직각, 나머지 3 비직각. outline 4 + bedroom 1 = 5/8 = 0.625
            expected_reward=5.0 / 8.0,
            expected_error_vertices=[(1, 0), (1, 2), (1, 3)],
            forbidden_error_vertices=[(0, 0), (0, 1), (0, 2), (0, 3), (1, 1)],
            tol=0.001,
        ),
        Case(
            "duplicate_vertex_zero_vector",
            "중복 꼭짓점 (영벡터) → 해당 꼭짓점 skip, crash 없음",
            rooms=[outline, RoomSpec("bedroom", [(20, 20), (20, 20), (100, 100), (20, 100)])],
            # vertex 0/1: 영벡터 (skip), vertex 2 (100,100): prev=(20,20), next=(20,100). v1=(-80,-80), v2=(-80,0). dot=6400+0. 비직각
            # vertex 3 (20,100): prev=(100,100), next=(20,20). v1=(80,0), v2=(0,-80). dot=0. 직각
            # outline 4 + bedroom 1 (vertex 3 직각) + bedroom 1 비직각 = 5직 / 6평가 = 5/6
            expected_reward=5.0 / 6.0,
            expected_error_vertices=[(1, 2)],
            tol=0.01,
        ),
        Case(
            "trapezoid_all_off",
            "사다리꼴 (모든 꼭짓점 비직각) → outline 4 + bedroom 0 = 4/8 = 0.5",
            rooms=[outline, RoomSpec("bedroom", [(50, 30), (100, 30), (110, 80), (40, 80)])],
            expected_reward=4.0 / 8.0,
            expected_error_vertices=[(1, 0), (1, 1), (1, 2), (1, 3)],
            tol=0.05,
        ),
    ]


def main():
    cases = build_cases()
    results = run_cases(cases, runner, label=f"Group 2: R_{REWARD_NAME}")
    summary_and_exit(results, label=f"R_{REWARD_NAME}")


if __name__ == "__main__":
    main()
