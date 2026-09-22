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
    assert_reward_close, assert_error_indices_contains, assert_error_indices_excludes,
    get_vocab, run_cases, summary_and_exit,
)

from src.training.rl.rewards import compute_all_rewards  # noqa: E402


REWARD_NAME = "room_in_outline"


@dataclass
class Case:
    name: str
    intent: str
    rooms: list
    front_door: object
    expected_reward: float
    expected_error_vertices: list = None
    forbidden_error_vertices: list = None
    expected_fd_error_indices: list = None  # [0, 1] (cx,cy) or [2, 3] (w,h) — front_door_indices 인덱스
    forbidden_fd_error_indices: list = None
    tol: float = 1e-3


def runner(case: Case) -> None:
    vocab = get_vocab()
    cfg = make_reward_cfg(
        enable=["room_in_outline"],
        credit_assignment={"room_in_outline": True},
        penalty_offset={"room_in_outline": 1.0},
    )
    metadata = build_metadata(total_rooms=len(case.rooms) - 1, type_counts={})

    token_ids, idx_map = build_output_token_ids(case.rooms, doors=[], front_door=case.front_door, vocab=vocab)
    result = compute_all_rewards(token_ids, vocab, metadata, cfg)
    actual = result["rewards"].get(REWARD_NAME, 0.0)
    assert_reward_close(actual, case.expected_reward, tol=case.tol, name=case.name)

    mask = result["error_masks"].get(REWARD_NAME)
    expected_positions: list[int] = []
    if case.expected_error_vertices:
        for r_idx, v_idx in case.expected_error_vertices:
            x_pos = idx_map.room_vertex_x[(r_idx, v_idx)]
            expected_positions.extend([x_pos, x_pos + 1])
    if case.expected_fd_error_indices:
        for fd_pos in case.expected_fd_error_indices:
            expected_positions.append(idx_map.front_door_indices[fd_pos])
    assert_error_indices_contains(mask, expected_positions, name=case.name)

    forbidden_positions: list[int] = []
    if case.forbidden_error_vertices:
        for r_idx, v_idx in case.forbidden_error_vertices:
            x_pos = idx_map.room_vertex_x[(r_idx, v_idx)]
            forbidden_positions.extend([x_pos, x_pos + 1])
    if case.forbidden_fd_error_indices:
        for fd_pos in case.forbidden_fd_error_indices:
            forbidden_positions.append(idx_map.front_door_indices[fd_pos])
    if forbidden_positions:
        assert_error_indices_excludes(mask, forbidden_positions, name=case.name)


def build_cases() -> list[Case]:
    outline = RoomSpec("outline", [(0, 0), (200, 0), (200, 200), (0, 200)])
    inside = RoomSpec("bedroom", [(20, 20), (100, 20), (100, 100), (20, 100)])

    return [
        Case(
            "all_inside",
            "방 + front_door 모두 outline 내 → 1.0",
            rooms=[outline, inside],
            front_door=FrontDoorSpec(cx=50, cy=10, w=8, h=2),  # outline 안
            expected_reward=1.0,
            expected_error_vertices=[],
        ),
        Case(
            "one_vertex_out",
            "★ 한 꼭짓점만 outline 밖 1px → 그 꼭짓점만 error",
            rooms=[
                outline,
                # vertex 2를 (210, 100)로 → outline 우측(200) 너머 10px
                RoomSpec("bedroom", [(20, 20), (100, 20), (210, 100), (20, 100)]),
            ],
            front_door=FrontDoorSpec(cx=50, cy=10, w=8, h=2),
            expected_reward=0.0,  # 일부라도 외부이면 이진 실패
            expected_error_vertices=[(1, 2)],
            forbidden_error_vertices=[(1, 0), (1, 1), (1, 3)],
            tol=1.0,  # 보상값은 검증 안 함
        ),
        Case(
            "front_door_zero_width",
            "★ front_door w=0 (degenerate) → 유효하지 않은 현관문으로 0",
            rooms=[outline, inside],
            front_door=FrontDoorSpec(cx=50, cy=10, w=0, h=2),
            # 면적 0 현관문은 유효하지 않으므로 포함 보상 실패.
            expected_reward=0.0,
            expected_error_vertices=[],
            expected_fd_error_indices=[2, 3],
        ),
        Case(
            "front_door_rb_outside",
            "★ 중심 기준 현관문 오른쪽·아래쪽 모서리가 outline 밖 → w/h 토큰만 error",
            rooms=[outline, inside],
            # cx=196, cy=196, w=10, h=10 → 우하단=(201,201)은 outline 밖
            # 중심=(196,196)은 outline 안
            front_door=FrontDoorSpec(cx=196, cy=196, w=10, h=10),
            expected_reward=0.0,
            expected_fd_error_indices=[2, 3],   # w_idx, h_idx
            forbidden_fd_error_indices=[0, 1],  # 중심 위치는 outline 안
            tol=1.0,
        ),
        Case(
            "no_outline",
            "outline 없음 (bedroom 둘만) → 0.0",
            rooms=[
                RoomSpec("bedroom", [(20, 20), (100, 20), (100, 100), (20, 100)]),
                RoomSpec("kitchen", [(120, 120), (180, 120), (180, 180), (120, 180)]),
            ],
            front_door=FrontDoorSpec(cx=50, cy=10, w=8, h=2),
            expected_reward=0.0,
            expected_error_vertices=None,  # 검사 안 함
            tol=1e-3,
        ),
    ]


def runner_dispatch(case: Case) -> None:
    if case.expected_reward is None:
        # 보상값 검증 skip, error 인덱스만 검증
        vocab = get_vocab()
        cfg = make_reward_cfg(
            enable=["room_in_outline"],
            credit_assignment={"room_in_outline": True},
        )
        metadata = build_metadata(total_rooms=len(case.rooms) - 1, type_counts={})
        token_ids, idx_map = build_output_token_ids(case.rooms, doors=[], front_door=case.front_door, vocab=vocab)
        result = compute_all_rewards(token_ids, vocab, metadata, cfg)
        actual = result["rewards"].get(REWARD_NAME, 0.0)
        # 1.0 미만은 의도적 violation 케이스에서 기대됨
        assert actual < 1.0, f"{case.name}: 보상이 1.0 미만이어야 하는데 actual={actual}"

        mask = result["error_masks"].get(REWARD_NAME)
        expected_positions: list[int] = []
        if case.expected_error_vertices:
            for r_idx, v_idx in case.expected_error_vertices:
                x_pos = idx_map.room_vertex_x[(r_idx, v_idx)]
                expected_positions.extend([x_pos, x_pos + 1])
        if case.expected_fd_error_indices:
            for fd_pos in case.expected_fd_error_indices:
                expected_positions.append(idx_map.front_door_indices[fd_pos])
        assert_error_indices_contains(mask, expected_positions, name=case.name)
        forbidden_positions: list[int] = []
        if case.forbidden_error_vertices:
            for r_idx, v_idx in case.forbidden_error_vertices:
                x_pos = idx_map.room_vertex_x[(r_idx, v_idx)]
                forbidden_positions.extend([x_pos, x_pos + 1])
        if case.forbidden_fd_error_indices:
            for fd_pos in case.forbidden_fd_error_indices:
                forbidden_positions.append(idx_map.front_door_indices[fd_pos])
        if forbidden_positions:
            assert_error_indices_excludes(mask, forbidden_positions, name=case.name)
    else:
        runner(case)


def main():
    cases = build_cases()
    results = run_cases(cases, runner_dispatch, label=f"Group 2: R_{REWARD_NAME}")
    summary_and_exit(results, label=f"R_{REWARD_NAME}")


if __name__ == "__main__":
    main()
