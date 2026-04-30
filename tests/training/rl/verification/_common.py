"""RL 보상함수 검증 공용 헬퍼.

각 verifier에서 사용하는 fixture 빌더, vocab 로더, 단언 함수를 제공한다.

설계 원칙:
    - 의도된 violation을 표현 가능한 검증 전용 토큰 빌더 (정상 케이스는
      tokenizer.build_output_tokens()로 cross-check 가능).
    - 토큰 빌더는 (token_ids, TokenIndexMap)을 반환해 신용할당 토큰 위치를
      테스트가 정확히 단언할 수 있게 한다.
    - vocab은 디스크에서 캐시 로드 (LLM 모델 로드 불필요).
"""

from __future__ import annotations

import functools
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 프로젝트 루트를 sys.path에 추가 (tests/ 하위에서 실행 시 src 임포트 가능)
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from omegaconf import DictConfig, OmegaConf

from src.training.augmentation.tokenizer import Vocab, load_vocab


# ---------------------------------------------------------------------------
# Vocab 로더 (캐시)
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_NAME = "Qwen2.5-Coder-7B"


@functools.lru_cache(maxsize=2)
def get_vocab(model_name: str = _DEFAULT_MODEL_NAME) -> Vocab:
    """프로젝트 디스크에서 Vocab을 로드해 캐시한다.

    Args:
        model_name: data/models/<name>/tokenization/ 디렉토리명.

    Returns:
        Vocab 인스턴스. LLM 모델은 로드하지 않으나 number_to_ids는 채워진다.

    Raises:
        FileNotFoundError: vocab_extension.json이 디스크에 없을 때.
    """
    tokenizer_dir = _REPO_ROOT / "data" / "models" / model_name / "tokenization"
    vocab_path = tokenizer_dir / "vocab_extension.json"
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"vocab_extension.json 없음: {vocab_path}. "
            f"먼저 build_vocab.py를 실행하시오."
        )
    return load_vocab(vocab_path, tokenizer_dir=tokenizer_dir)


# ---------------------------------------------------------------------------
# Fixture 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class RoomSpec:
    """검증용 방 사양 (의도된 violation 표현 가능).

    Attributes:
        room_type: 방 타입 문자열 ("outline", "bedroom" 등).
        coords: 꼭짓점 좌표 (x, y) 리스트. 직각 미보장.
    """
    room_type: str
    coords: list[tuple[int, int]]


@dataclass
class DoorSpec:
    """검증용 DOOR 블록 사양 (인테리어 문)."""
    cx: int
    cy: int
    w: int
    h: int


@dataclass
class FrontDoorSpec:
    """검증용 FRONT_DOOR 사양.

    Attributes:
        cx: 중심 X (left-top 의도). None이면 NO_DOOR로 인코딩.
        cy: 중심 Y. cx가 None이 아니면 None 금지.
        w: 너비.
        h: 높이.
        omit: True면 FRONT_DOOR 블록 자체 생략.
        no_door: True면 <FRONT_DOOR> <NO_DOOR> <END_DOOR> 인코딩.
    """
    cx: int | None = None
    cy: int | None = None
    w: int | None = None
    h: int | None = None
    omit: bool = False
    no_door: bool = False


@dataclass
class TokenIndexMap:
    """build_output_token_ids가 반환하는 토큰 위치 인덱스 맵.

    Attributes:
        room_block_starts: room_idx → <ROOM> 토큰 인덱스.
        room_vertex_x: (room_idx, vertex_idx) → X 토큰 인덱스. (Y는 +1)
        front_door_indices: [cx_idx, cy_idx, w_idx, h_idx] (없으면 빈 리스트).
        door_block_starts: door_idx → <DOOR> 토큰 인덱스.
        door_coord_indices: door_idx → [cx_idx, cy_idx, w_idx, h_idx].
    """
    room_block_starts: dict[int, int] = field(default_factory=dict)
    room_vertex_x: dict[tuple[int, int], int] = field(default_factory=dict)
    front_door_indices: list[int] = field(default_factory=list)
    door_block_starts: dict[int, int] = field(default_factory=dict)
    door_coord_indices: dict[int, list[int]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 출력 토큰 빌더 (의도된 violation 지원)
# ---------------------------------------------------------------------------

def build_output_token_ids(
    rooms: list[RoomSpec],
    doors: list[DoorSpec] | None = None,
    front_door: FrontDoorSpec | None = None,
    *,
    vocab: Vocab | None = None,
    omit_output_wrapper: bool = False,
    omit_end_output: bool = False,
    omit_end_room_for_room_idx: list[int] | None = None,
    swap_xy_for_room_idx: list[int] | None = None,
    extra_x_for_room_idx: list[int] | None = None,
) -> tuple[list[int], TokenIndexMap]:
    """평면도 fixture를 출력 토큰 ID 시퀀스로 변환한다.

    `tokenizer.build_output_tokens`와 호환되는 정상 출력을 만들면서도, 검증을
    위해 의도된 포맷 violation을 주입할 수 있다.

    Args:
        rooms: 방 사양 리스트. 첫 번째가 outline 의도.
        doors: 인테리어 DOOR 블록 리스트. None이면 doors 없음 (NO_DOOR로 인코딩 안 함;
            doors 없으면 그냥 생략).
        front_door: 현관문 사양. None이면 NO_DOOR로 인코딩.
        vocab: Vocab. None이면 get_vocab()로 로드.
        omit_output_wrapper: True면 <OUTPUT> 토큰 자체 생략 (level 0 트리거).
        omit_end_output: True면 <END_OUTPUT> 생략.
        omit_end_room_for_room_idx: 해당 인덱스 방의 <END_ROOM> 토큰을 생략 (포맷 오류 트리거).
        swap_xy_for_room_idx: 해당 방의 첫 꼭짓점 토큰 순서를 X/X/Y로 변형 (X 중복).
        extra_x_for_room_idx: 해당 방의 첫 꼭짓점 후에 추가 X 토큰을 삽입.

    Returns:
        (token_ids, TokenIndexMap).
    """
    if vocab is None:
        vocab = get_vocab()
    if doors is None:
        doors = []
    omit_end_room = set(omit_end_room_for_room_idx or [])
    swap_xy = set(swap_xy_for_room_idx or [])
    extra_x = set(extra_x_for_room_idx or [])

    tokens: list[int] = []
    idx_map = TokenIndexMap()

    if not omit_output_wrapper:
        tokens.append(vocab.get("<OUTPUT>"))

    # FRONT_DOOR
    if front_door is None:
        # NO_DOOR로 인코딩
        tokens.append(vocab.get("<FRONT_DOOR>"))
        tokens.append(vocab.get("<NO_DOOR>"))
        tokens.append(vocab.get("<END_DOOR>"))
    elif front_door.omit:
        pass  # 블록 전체 생략
    elif front_door.no_door:
        tokens.append(vocab.get("<FRONT_DOOR>"))
        tokens.append(vocab.get("<NO_DOOR>"))
        tokens.append(vocab.get("<END_DOOR>"))
    else:
        tokens.append(vocab.get("<FRONT_DOOR>"))
        cx_idx = len(tokens)
        tokens.append(vocab.get(f"<X:{front_door.cx}>"))
        cy_idx = len(tokens)
        tokens.append(vocab.get(f"<Y:{front_door.cy}>"))
        tokens.append(vocab.get("<SEP_DOOR>"))
        w_idx = len(tokens)
        tokens.append(vocab.get(f"<X:{front_door.w}>"))
        h_idx = len(tokens)
        tokens.append(vocab.get(f"<Y:{front_door.h}>"))
        tokens.append(vocab.get("<END_DOOR>"))
        idx_map.front_door_indices = [cx_idx, cy_idx, w_idx, h_idx]

    # ROOMS
    for r_idx, room in enumerate(rooms):
        block_start = len(tokens)
        idx_map.room_block_starts[r_idx] = block_start

        tokens.append(vocab.get("<ROOM>"))
        tokens.append(vocab.get(f"<TYPE:{room.room_type}>"))

        for v_idx, (x, y) in enumerate(room.coords):
            x_pos = len(tokens)
            tokens.append(vocab.get(f"<X:{x}>"))

            if v_idx == 0 and r_idx in swap_xy:
                # X X Y 변형 (Y 자리에 X)
                tokens.append(vocab.get(f"<X:{x}>"))
            else:
                tokens.append(vocab.get(f"<Y:{y}>"))

            idx_map.room_vertex_x[(r_idx, v_idx)] = x_pos

            if v_idx == 0 and r_idx in extra_x:
                # 추가 X 토큰 (Y 누락 시뮬레이션)
                tokens.append(vocab.get(f"<X:{x}>"))

        if r_idx not in omit_end_room:
            tokens.append(vocab.get("<END_ROOM>"))

    # DOORS
    for d_idx, door in enumerate(doors):
        block_start = len(tokens)
        idx_map.door_block_starts[d_idx] = block_start
        tokens.append(vocab.get("<DOOR>"))
        cx_idx = len(tokens)
        tokens.append(vocab.get(f"<X:{door.cx}>"))
        cy_idx = len(tokens)
        tokens.append(vocab.get(f"<Y:{door.cy}>"))
        tokens.append(vocab.get("<SEP_DOOR>"))
        w_idx = len(tokens)
        tokens.append(vocab.get(f"<X:{door.w}>"))
        h_idx = len(tokens)
        tokens.append(vocab.get(f"<Y:{door.h}>"))
        tokens.append(vocab.get("<END_DOOR>"))
        idx_map.door_coord_indices[d_idx] = [cx_idx, cy_idx, w_idx, h_idx]

    if not omit_end_output:
        tokens.append(vocab.get("<END_OUTPUT>"))

    return tokens, idx_map


# ---------------------------------------------------------------------------
# Metadata 빌더
# ---------------------------------------------------------------------------

def build_metadata(
    *,
    rooms: list[dict] | None = None,
    edges: list[dict] | None = None,
    spatial: list[dict] | None = None,
    front_door: dict | None = None,
    total_rooms: int | None = None,
    type_counts: dict[str, int] | None = None,
) -> dict:
    """모델 시점 metadata dict를 만든다 (`_extract_metadata` 출력 호환).

    None을 빈 list/dict로 정규화한다.
    """
    return {
        "rooms": list(rooms or []),
        "edges": list(edges or []),
        "spatial": list(spatial or []),
        "front_door": front_door,
        "total_rooms": total_rooms,
        "type_counts": dict(type_counts or {}),
    }


def flat_coords(pairs: list[tuple[int, int]]) -> list[int]:
    """(x, y) 쌍 리스트를 flat [x1, y1, x2, y2, ...]로 변환."""
    out: list[int] = []
    for x, y in pairs:
        out.extend([int(x), int(y)])
    return out


# ---------------------------------------------------------------------------
# Reward config 빌더
# ---------------------------------------------------------------------------

ALL_REWARD_NAMES = [
    "format", "count_total", "count_type",
    "orthogonality", "no_overlap",
    "room_in_outline", "outline_in_room", "coverage",
    "connectivity", "spatial", "input_consistency",
]


def make_reward_cfg(
    *,
    enable: list[str] | None = None,
    weights: dict[str, float] | None = None,
    credit_assignment: dict[str, bool] | None = None,
    penalty_scale: dict[str, float] | None = None,
    threshold: float | None = None,
    hard_gate: bool = True,
) -> DictConfig:
    """검증용 reward_cfg DictConfig를 만든다.

    기본은 모든 보상 비활성, format은 항상 활성 (Hard Gate 통과 위해).
    enable에 명시한 보상만 추가 활성화. credit_assignment / penalty_scale은
    enable된 보상 중 신용할당 가능한 보상에만 적용.

    Args:
        enable: 활성화할 보상 이름 리스트 (format 자동 포함).
        weights: 보상명 → 가중치.
        credit_assignment: 보상명 → 신용할당 ON/OFF.
        penalty_scale: 보상명 → 페널티 배율.
        threshold: input_consistency 전용 (px).
        hard_gate: format Hard Gate 의미적 표시 (현재 코드는 항상 작동).

    Returns:
        OmegaConf DictConfig.
    """
    enable = list(enable or [])
    if "format" not in enable:
        enable = ["format"] + enable
    weights = weights or {}
    credit_assignment = credit_assignment or {}
    penalty_scale = penalty_scale or {}

    cfg_dict: dict[str, Any] = {}
    for name in ALL_REWARD_NAMES:
        is_enabled = name in enable
        item: dict[str, Any] = {
            "enabled": is_enabled,
            "weight": float(weights.get(name, 1.0)),
            "credit_assignment": bool(credit_assignment.get(name, False)),
            "penalty_scale": float(penalty_scale.get(name, 1.0)),
        }
        if name == "input_consistency" and threshold is not None:
            item["threshold"] = float(threshold)
        cfg_dict[name] = item

    cfg_dict["format"]["hard_gate"] = bool(hard_gate)
    return OmegaConf.create(cfg_dict)


# ---------------------------------------------------------------------------
# Assert 헬퍼
# ---------------------------------------------------------------------------

def assert_reward_close(
    actual: float,
    expected: float,
    *,
    tol: float = 1e-4,
    name: str = "",
) -> None:
    """보상값을 부동소수점 허용 오차 내에서 비교."""
    if not (abs(actual - expected) <= tol):
        raise AssertionError(
            f"reward 불일치{f' ({name})' if name else ''}: "
            f"actual={actual:.6f}, expected={expected:.6f}, tol={tol}"
        )


def assert_error_indices_contains(
    error_mask: torch.Tensor | None,
    expected_positions: list[int],
    *,
    name: str = "",
) -> None:
    """error_mask가 expected_positions를 모두 포함하는지 확인.

    error_mask는 (L,) shape의 float 마스크. 1.0인 위치가 expected_positions를
    모두 포함해야 함. 추가 마스킹은 허용 (subset check).

    expected_positions=[]인 경우 mask가 None이거나 모두 0이어도 PASS (오류 없음).
    """
    if not expected_positions:
        # 오류 없음을 기대 — mask 없거나 all-zero면 OK
        if error_mask is None:
            return
        if (error_mask > 0.5).sum().item() == 0:
            return
        actual_set = {i for i, v in enumerate(error_mask.tolist()) if v > 0.5}
        raise AssertionError(
            f"error 없음을 기대했으나 mask에 마킹 위치 존재{f' ({name})' if name else ''}: "
            f"{sorted(actual_set)}"
        )
    if error_mask is None:
        raise AssertionError(f"error_mask 없음{f' ({name})' if name else ''}")
    actual_set = {i for i, v in enumerate(error_mask.tolist()) if v > 0.5}
    expected_set = set(expected_positions)
    missing = expected_set - actual_set
    if missing:
        raise AssertionError(
            f"error_mask{f' ({name})' if name else ''}에 누락된 위치: {sorted(missing)}. "
            f"actual masked positions: {sorted(actual_set)}"
        )


def assert_error_indices_excludes(
    error_mask: torch.Tensor | None,
    forbidden_positions: list[int],
    *,
    name: str = "",
) -> None:
    """error_mask가 forbidden_positions를 모두 마킹하지 않았는지 확인."""
    if error_mask is None:
        return
    actual_set = {i for i, v in enumerate(error_mask.tolist()) if v > 0.5}
    overlap = actual_set & set(forbidden_positions)
    if overlap:
        raise AssertionError(
            f"error_mask{f' ({name})' if name else ''}에 포함되면 안 되는 위치: "
            f"{sorted(overlap)}"
        )


def assert_error_mask_absent(
    error_masks: dict,
    name: str,
) -> None:
    """특정 보상에 error_mask가 없어야 함을 단언 (신용할당 OFF 보상)."""
    if name in error_masks and error_masks[name] is not None:
        # error_mask가 모두 0인 경우는 OK (사실상 없음)
        if error_masks[name].sum().item() > 0:
            raise AssertionError(
                f"{name}는 신용할당 OFF여야 하나 비활성 토큰을 포함하는 mask가 존재함"
            )


# ---------------------------------------------------------------------------
# Case 실행 골격
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    name: str
    passed: bool
    error: str = ""


def run_cases(
    cases: list,
    runner_fn,
    *,
    label: str = "",
) -> list[CaseResult]:
    """케이스 리스트를 일괄 실행하고 결과를 출력한다.

    Args:
        cases: Case 객체 리스트 (각 verifier가 정의).
        runner_fn: case → None (raise on failure) 함수.
        label: 출력 헤더 레이블.

    Returns:
        CaseResult 리스트.
    """
    if label:
        print(f"=== {label} ===")
    results: list[CaseResult] = []
    for case in cases:
        name = getattr(case, "name", str(case))
        intent = getattr(case, "intent", "")
        try:
            runner_fn(case)
            results.append(CaseResult(name=name, passed=True))
            print(f"[PASS] {name}: {intent}")
        except AssertionError as e:
            results.append(CaseResult(name=name, passed=False, error=str(e)))
            print(f"[FAIL] {name}: {intent}\n       → {e}")
        except Exception as e:
            results.append(CaseResult(
                name=name, passed=False,
                error=f"{type(e).__name__}: {e}"
            ))
            print(f"[ERROR] {name}: {intent}\n       → {type(e).__name__}: {e}")
    return results


def summary_and_exit(results: list[CaseResult], label: str = "") -> None:
    """결과 요약 출력 후 실패 시 sys.exit(1)."""
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\n--- {label} 요약 ---")
    print(f"PASS: {passed}/{total}")
    if passed < total:
        print("FAIL 케이스:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.error}")
        sys.exit(1)
    sys.exit(0)
