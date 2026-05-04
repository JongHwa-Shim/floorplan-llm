"""Group 2: R_no_overlap 검증.

의도: 비-outline 방끼리 면적 겹침이 없어야 함. 침범 꼭짓점만 책임 마킹.
shapely contains() 사용 → 경계 위 점은 false → 공유 벽 false positive 방지.

핵심 케이스:
    - 분리된 방 두 개 → 1.0
    - 한 변 공유 (boundary) → 1.0 (false positive 없음)
    - 한 점 공유 (corner) → 1.0
    - 부분 겹침 → 1 - overlap_ratio, 침범 꼭짓점만 error
    - 완전 포함 → 작은 방의 모든 꼭짓점 error
    - **★ Self-intersecting polygon (bowtie)**: invalid → no overlap reward 페널티 없음 (finding 후보)
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
    assert_reward_close, assert_error_indices_contains, assert_error_indices_excludes,
    get_vocab, run_cases, summary_and_exit,
)

from src.training.rl.rewards import compute_all_rewards  # noqa: E402


REWARD_NAME = "no_overlap"


@dataclass
class Case:
    name: str
    intent: str
    rooms: list
    expected_reward: float
    expected_error_vertices: list = None  # [(room_idx, vertex_idx), ...] 또는 None (검사 안 함)
    forbidden_error_vertices: list = None
    tol: float = 1e-3


def runner(case: Case) -> None:
    vocab = get_vocab()
    cfg = make_reward_cfg(
        enable=["no_overlap"],
        credit_assignment={"no_overlap": True},
        penalty_offset={"no_overlap": 1.0},
    )
    fd = FrontDoorSpec(cx=105, cy=10, w=8, h=2)
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
    if case.forbidden_error_vertices:
        forbidden_positions: list[int] = []
        for r_idx, v_idx in case.forbidden_error_vertices:
            x_pos = idx_map.room_vertex_x[(r_idx, v_idx)]
            forbidden_positions.extend([x_pos, x_pos + 1])
        assert_error_indices_excludes(mask, forbidden_positions, name=case.name)


def build_cases() -> list[Case]:
    outline = RoomSpec("outline", [(0, 0), (200, 0), (200, 200), (0, 200)])

    return [
        Case(
            "separated",
            "분리된 두 방 → 1.0",
            rooms=[
                outline,
                RoomSpec("bedroom", [(10, 10), (50, 10), (50, 50), (10, 50)]),
                RoomSpec("kitchen", [(100, 100), (150, 100), (150, 150), (100, 150)]),
            ],
            expected_reward=1.0,
            expected_error_vertices=[],
        ),
        Case(
            "shared_wall_no_false_positive",
            "★ 한 변 공유 (벽) → 1.0, error 없음 (contains exclusive)",
            rooms=[
                outline,
                RoomSpec("bedroom", [(10, 10), (100, 10), (100, 50), (10, 50)]),
                RoomSpec("kitchen", [(100, 10), (190, 10), (190, 50), (100, 50)]),  # x=100 변 공유
            ],
            expected_reward=1.0,
            expected_error_vertices=[],
        ),
        Case(
            "single_corner_share",
            "한 점만 공유 (모서리) → 1.0",
            rooms=[
                outline,
                RoomSpec("bedroom", [(10, 10), (50, 10), (50, 50), (10, 50)]),
                RoomSpec("kitchen", [(50, 50), (100, 50), (100, 100), (50, 100)]),  # (50,50) 한 점 공유
            ],
            expected_reward=1.0,
            expected_error_vertices=[],
        ),
        Case(
            "partial_overlap",
            "★ 25% 부분 겹침 (50×50 area, 25 overlap) → 침범 꼭짓점만 error",
            rooms=[
                outline,
                # bedroom: (10,10)~(60,60)
                RoomSpec("bedroom", [(10, 10), (60, 10), (60, 60), (10, 60)]),
                # kitchen: (35,35)~(85,85). bedroom의 (60,60) 영역에 25×25 침범
                RoomSpec("kitchen", [(35, 35), (85, 35), (85, 85), (35, 85)]),
            ],
            # bedroom area=2500, kitchen area=2500, total=5000, overlap=625
            # reward = 1 - 625/5000 = 0.875
            expected_reward=1.0 - 625.0 / 5000.0,
            # bedroom (60,60)이 kitchen 내부에 포함 → bedroom vertex 2 error
            # kitchen (35,35)이 bedroom 내부에 포함 → kitchen vertex 0 error
            expected_error_vertices=[(1, 2), (2, 0)],
            tol=0.01,
        ),
        Case(
            "full_containment",
            "★ B가 A에 완전 포함 → B의 4꼭짓점 모두 error",
            rooms=[
                outline,
                RoomSpec("bedroom", [(10, 10), (100, 10), (100, 100), (10, 100)]),  # 큰 방
                RoomSpec("kitchen", [(30, 30), (60, 30), (60, 60), (30, 60)]),  # 작은 방, 큰 방 안
            ],
            # bedroom area=8100, kitchen area=900, total=9000, overlap=900 (kitchen 전체)
            # reward = 1 - 900/9000 = 0.9
            expected_reward=1.0 - 900.0 / 9000.0,
            # kitchen 4 꼭짓점 모두 bedroom 내부 → kitchen vertex 0,1,2,3 error
            expected_error_vertices=[(2, 0), (2, 1), (2, 2), (2, 3)],
            tol=0.01,
        ),
        Case(
            "self_intersecting_bowtie",
            "★ Self-intersecting polygon (bowtie) → invalid 처리 후 페널티 없음 (finding)",
            rooms=[
                outline,
                # bowtie: (10,10) → (50,50) → (50,10) → (10,50) → close. 자기교차.
                RoomSpec("bedroom", [(10, 10), (50, 50), (50, 10), (10, 50)]),
                RoomSpec("kitchen", [(80, 80), (100, 80), (100, 100), (80, 100)]),
            ],
            # bowtie는 invalid → buffer(0)로 정리되거나 None 처리. 명확한 보상은 알 수 없으나
            # 일반적으로 1.0 또는 0이 나옴. 검증의 의도는 crash 없음 + 결함 식별.
            expected_reward=1.0,  # 현재 구현은 invalid polygon 보상 미반영 → 페널티 없음 (finding)
            expected_error_vertices=None,  # 검사 안 함
            tol=0.5,  # 매우 관대
        ),
    ]


def main():
    cases = build_cases()
    results = run_cases(cases, runner, label=f"Group 2: R_{REWARD_NAME}")
    summary_and_exit(results, label=f"R_{REWARD_NAME}")


if __name__ == "__main__":
    main()
