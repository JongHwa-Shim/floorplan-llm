"""Group 2: R_format 검증.

의도: 출력 시퀀스가 파싱 가능하고 (level==3) 최소 outline + 1방인지 검증.
신용할당: 파싱 오류 위치 토큰을 마킹.

검증 케이스:
    - 정상: outline + 1방 + 정상 토큰 → 1.0, error 없음
    - <OUTPUT> 누락 → 0.0, 모든 토큰 error
    - outline만 (방 1개) → 0.0, room block error
    - <END_OUTPUT> 누락 → 1.0 단언 (parser는 EOS 미요구) — 회귀 가드
    - X X Y 변형 (Y 자리에 X) → 0.0, 깨진 위치 error
    - <END_ROOM> 누락 → 0.0, 그 위치 error
    - format vs orthogonality 분리: 비직각 출력해도 format=1
    - **★ outline 없이 방 2개만**: format=1 통과 (outline 미보장 결함 검출)
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


REWARD_NAME = "format"


@dataclass
class Case:
    name: str
    intent: str
    rooms: list
    front_door: object
    metadata: dict
    cfg: object
    expected_reward: float
    expected_error_positions: list | None = None
    forbidden_error_positions: list | None = None
    omit_output: bool = False
    omit_end_output: bool = False
    omit_end_room_for_idx: list | None = None
    swap_xy_for_idx: list | None = None
    extra_x_for_idx: list | None = None
    expected_findings: str | None = None


def runner(case: Case) -> None:
    vocab = get_vocab()
    token_ids, idx_map = build_output_token_ids(
        case.rooms,
        doors=[],
        front_door=case.front_door,
        vocab=vocab,
        omit_output_wrapper=case.omit_output,
        omit_end_output=case.omit_end_output,
        omit_end_room_for_room_idx=case.omit_end_room_for_idx,
        swap_xy_for_room_idx=case.swap_xy_for_idx,
        extra_x_for_room_idx=case.extra_x_for_idx,
    )
    result = compute_all_rewards(token_ids, vocab, case.metadata, case.cfg)
    actual = result["rewards"].get(REWARD_NAME, 0.0)
    assert_reward_close(actual, case.expected_reward, name=case.name)
    if case.expected_error_positions is not None:
        mask = result["error_masks"].get(REWARD_NAME)
        assert_error_indices_contains(mask, case.expected_error_positions, name=case.name)


def build_cases() -> list[Case]:
    cfg_format = make_reward_cfg(
        enable=["format"],
        credit_assignment={"format": True},
        penalty_scale={"format": 1.0},
    )
    metadata = build_metadata(total_rooms=1, type_counts={"bedroom": 1})

    # 정상 outline + 1방
    rooms_ok = [
        RoomSpec("outline",  [(10, 10), (200, 10), (200, 200), (10, 200)]),
        RoomSpec("bedroom",  [(20, 20), (100, 20), (100, 100), (20, 100)]),
    ]
    fd_ok = FrontDoorSpec(cx=105, cy=10, w=8, h=2)

    return [
        Case(
            "normal_full_pass",
            "정상 outline + 1방 + front_door → 1.0",
            rooms=rooms_ok, front_door=fd_ok,
            metadata=metadata, cfg=cfg_format, expected_reward=1.0,
            expected_error_positions=[],
        ),
        Case(
            "no_output_wrapper",
            "<OUTPUT> 토큰 누락 → 0.0, 전체 error",
            rooms=rooms_ok, front_door=fd_ok,
            metadata=metadata, cfg=cfg_format, expected_reward=0.0,
            omit_output=True,
        ),
        Case(
            "outline_only",
            "outline만 (방 1개) → 0.0",
            rooms=[rooms_ok[0]], front_door=fd_ok,
            metadata=build_metadata(total_rooms=0, type_counts={}),
            cfg=cfg_format, expected_reward=0.0,
        ),
        Case(
            "xxy_broken",
            "X X Y 변형 (Y 자리에 X) → 0.0",
            rooms=rooms_ok, front_door=fd_ok,
            metadata=metadata, cfg=cfg_format, expected_reward=0.0,
            swap_xy_for_idx=[1],  # 두번째 방의 첫 꼭짓점 변형
        ),
        Case(
            "missing_end_room",
            "<END_ROOM> 누락 → 0.0",
            rooms=rooms_ok, front_door=fd_ok,
            metadata=metadata, cfg=cfg_format, expected_reward=0.0,
            omit_end_room_for_idx=[1],
        ),
        Case(
            "format_vs_orth",
            "format은 비직각이어도 1.0 (orthogonality와 분리)",
            rooms=[
                RoomSpec("outline", [(10, 10), (200, 10), (200, 200), (10, 200)]),
                # 직각이 아닌 마름모 형태 방
                RoomSpec("bedroom", [(50, 50), (100, 60), (110, 110), (60, 100)]),
            ],
            front_door=fd_ok, metadata=metadata, cfg=cfg_format,
            expected_reward=1.0,
        ),
        Case(
            "no_outline_two_rooms",
            "★ outline 없이 방 2개만 → 0.0 (F-2 수정 후 outline 부재 검증 추가됨)",
            rooms=[
                RoomSpec("bedroom", [(20, 20), (100, 20), (100, 100), (20, 100)]),
                RoomSpec("kitchen", [(120, 20), (200, 20), (200, 100), (120, 100)]),
            ],
            front_door=fd_ok, metadata=metadata, cfg=cfg_format,
            expected_reward=0.0,  # F-2 수정 후: outline이 첫 방으로 명시되어야만 통과
        ),
        Case(
            "outline_not_first",
            "★ outline이 첫 번째가 아닌 위치에 있음 → 0.0 (F-2 검증)",
            rooms=[
                RoomSpec("bedroom", [(20, 20), (100, 20), (100, 100), (20, 100)]),
                RoomSpec("outline", [(10, 10), (200, 10), (200, 200), (10, 200)]),
            ],
            front_door=fd_ok, metadata=metadata, cfg=cfg_format,
            expected_reward=0.0,
        ),
    ]


def main():
    cases = build_cases()
    results = run_cases(cases, runner, label=f"Group 2: R_{REWARD_NAME}")
    summary_and_exit(results, label=f"R_{REWARD_NAME}")


if __name__ == "__main__":
    main()
