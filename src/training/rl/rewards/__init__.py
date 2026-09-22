"""논문에 정의된 10개 이진 보상과 토큰별 위반 마스크를 계산한다."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from src.training.rl.rewards.parser import parse_output_tokens
from src.training.rl.rewards.format_reward import compute_format_reward
from src.training.rl.rewards.count_reward import compute_count_total_reward, compute_count_type_reward
from src.training.rl.rewards.geometry_reward import compute_orthogonality_reward, compute_no_overlap_reward
from src.training.rl.rewards.room_in_outline_reward import compute_room_in_outline_reward
from src.training.rl.rewards.coverage_reward import compute_coverage_reward
from src.training.rl.rewards.connectivity_reward import compute_connectivity_reward
from src.training.rl.rewards.spatial_reward import compute_spatial_reward
from src.training.rl.rewards.polygon_fidelity_reward import compute_polygon_fidelity_reward
from src.training.rl.rewards.credit_assignment import build_error_mask

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from src.training.augmentation.tokenizer import Vocab

REWARD_NAMES = (
    "format", "count_total", "count_type", "orthogonality", "no_overlap",
    "room_in_outline", "coverage", "connectivity", "spatial", "polygon_fidelity",
)
TOKEN_CREDIT_REWARDS = frozenset((
    "format", "orthogonality", "no_overlap", "room_in_outline", "polygon_fidelity",
))


def compute_all_rewards(
    token_ids: list[int], vocab: "Vocab", metadata: dict, reward_cfg: "DictConfig",
    *, apply_format_gate: bool = True,
) -> dict:
    """출력을 한 번 파싱하고 활성 보상과 마스크를 반환한다.

    Mod Record: 위반이 없는 국소화 가능 보상에도 0 마스크를 만든다.
    그래야 Eq. (7)의 정상 토큰 α 조정이 성공 시퀀스에도 적용된다.
    format 보상 제거 실험에서는 gate도 꺼지며, 나머지 실험에서는 유지된다.
    Mod Record: 평가는 apply_format_gate=False로 호출한다. 개별 보상은
    전체 파싱 성공 여부 대신 복원된 데이터에 자신의 조건을 적용한다.

    Args:
        token_ids: completion 토큰 ID.
        vocab: 평면도 어휘.
        metadata: 입력에 실제 노출된 조건.
        reward_cfg: 보상별 활성화·가중치·신용 할당·임계값 설정.
        apply_format_gate: 훈련용 형식 게이트 적용 여부. 평가에서는 False.

    Returns:
        rewards, error_masks, hard_gate_pass, parsed를 포함하는 딕셔너리.
        hard_gate_pass는 게이트 적용 여부와 별개로 Format 통과 여부를 나타낸다.

    Raises:
        ValueError: 보상 임계값이 잘못된 경우.
    """
    parsed = parse_output_tokens(token_ids, vocab)
    format_reward, format_errors = compute_format_reward(parsed)
    format_cfg = reward_cfg.get("format", {})
    gate_failed = (
        apply_format_gate and format_reward == 0 and format_cfg.get("enabled", True)
        and format_cfg.get("hard_gate", True)
    )
    rewards: dict[str, float] = {}
    error_masks: dict[str, torch.Tensor] = {}
    for name in REWARD_NAMES:
        cfg = reward_cfg.get(name, {})
        if not cfg.get("enabled", True):
            continue
        errors = []
        if name == "format":
            value, errors = format_reward, format_errors
        elif gate_failed:
            value = 0.0
        elif name == "count_total":
            value = compute_count_total_reward(parsed, metadata)
        elif name == "count_type":
            value = compute_count_type_reward(parsed, metadata)
        elif name == "orthogonality":
            value, errors = compute_orthogonality_reward(parsed)
        elif name == "no_overlap":
            value, errors = compute_no_overlap_reward(parsed)
        elif name == "room_in_outline":
            value, errors = compute_room_in_outline_reward(parsed)
        elif name == "coverage":
            value = compute_coverage_reward(parsed, threshold=float(cfg.get("threshold", 0.774)))
        elif name == "connectivity":
            value = compute_connectivity_reward(parsed, metadata)
        elif name == "spatial":
            value = compute_spatial_reward(parsed, metadata)
        else:
            value, errors = compute_polygon_fidelity_reward(
                parsed, metadata, tolerance=float(cfg.get("tolerance", 15.0)),
            )
        rewards[name] = value
        if name in TOKEN_CREDIT_REWARDS and cfg.get("credit_assignment", False):
            error_masks[name] = build_error_mask(len(token_ids), errors)
    return {
        "rewards": rewards, "error_masks": error_masks,
        "hard_gate_pass": format_reward == 1.0, "parsed": parsed,
    }
