"""GRPO 보상함수 패키지.

모든 보상 계산을 총괄하는 compute_all_rewards() 함수를 제공한다.

보상 목록:
    - format:       R_format (Hard Gate, 신용할당 ON)
    - count_total:  R_count_total (이진)
    - count_type:   R_count_type (연속)
    - orthogonality: R_orthogonality (직각도, 신용할당 ON)
    - no_overlap:   R_no_overlap (겹침, 신용할당 ON)
    - connectivity: R_connectivity (연결성)
    - spatial:      R_spatial (공간 관계)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from src.training.grpo.rewards.parser import parse_output_tokens
from src.training.grpo.rewards.format_reward import compute_format_reward
from src.training.grpo.rewards.count_reward import (
    compute_count_total_reward,
    compute_count_type_reward,
)
from src.training.grpo.rewards.geometry_reward import (
    compute_orthogonality_reward,
    compute_no_overlap_reward,
)
from src.training.grpo.rewards.connectivity_reward import compute_connectivity_reward
from src.training.grpo.rewards.spatial_reward import compute_spatial_reward
from src.training.grpo.rewards.credit_assignment import build_error_mask

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from src.training.augmentation.tokenizer import Vocab

logger = logging.getLogger(__name__)


def compute_all_rewards(
    token_ids: list[int],
    vocab: "Vocab",
    metadata: dict,
    reward_cfg: "DictConfig",
) -> dict:
    """모든 보상을 계산하고 에러 마스크를 함께 반환한다.

    처리 흐름:
        1. 출력 토큰 파싱 (parse_output_tokens)
        2. R_format 평가 (Hard Gate)
        3. R_format=0이면 모든 보상 0 (Hard Gate 강제)
        4. 나머지 보상 계산
        5. 신용할당 ON인 보상의 error_mask 생성

    Args:
        token_ids: completion 토큰 ID 리스트 (<OUTPUT>부터).
        vocab: Vocab 객체.
        metadata: 입력 조건 메타데이터 딕셔너리.
        reward_cfg: Hydra DictConfig. rewards 섹션.

    Returns:
        dict:
            - rewards (dict[str, float]): 보상명 → 스칼라 보상값.
            - error_masks (dict[str, Tensor]): 보상명 → shape $(L,)$ 마스크.
                신용할당 ON인 보상만 포함.
            - hard_gate_pass (bool): R_format >= 1.0이면 True.
            - parsed: ParsedFloorplan 인스턴스 (디버깅용).
    """
    seq_length = len(token_ids)

    # 파싱
    parsed = parse_output_tokens(token_ids, vocab)

    # R_format (Hard Gate)
    format_reward, format_error_indices = compute_format_reward(parsed)
    hard_gate_pass = format_reward >= 1.0

    rewards: dict[str, float] = {}
    error_masks: dict[str, torch.Tensor] = {}

    # R_format 결과 저장
    cfg_format = reward_cfg.get("format", {})
    if cfg_format.get("enabled", True):
        rewards["format"] = format_reward
        if cfg_format.get("credit_assignment", False) and format_error_indices:
            error_masks["format"] = build_error_mask(seq_length, format_error_indices)

    # Hard Gate: format 실패 시 나머지 보상 모두 0
    if not hard_gate_pass:
        for name in ("count_total", "count_type", "orthogonality", "no_overlap",
                     "connectivity", "spatial"):
            cfg = reward_cfg.get(name, {})
            if cfg.get("enabled", True):
                rewards[name] = 0.0
        return {
            "rewards": rewards,
            "error_masks": error_masks,
            "hard_gate_pass": False,
            "parsed": parsed,
        }

    # R_count_total
    cfg = reward_cfg.get("count_total", {})
    if cfg.get("enabled", True):
        rewards["count_total"] = compute_count_total_reward(parsed, metadata)

    # R_count_type
    cfg = reward_cfg.get("count_type", {})
    if cfg.get("enabled", True):
        rewards["count_type"] = compute_count_type_reward(parsed, metadata)

    # R_orthogonality
    cfg = reward_cfg.get("orthogonality", {})
    if cfg.get("enabled", True):
        orth_reward, orth_errors = compute_orthogonality_reward(parsed)
        rewards["orthogonality"] = orth_reward
        if cfg.get("credit_assignment", False) and orth_errors:
            error_masks["orthogonality"] = build_error_mask(seq_length, orth_errors)

    # R_no_overlap
    cfg = reward_cfg.get("no_overlap", {})
    if cfg.get("enabled", True):
        no_overlap_reward, no_overlap_errors = compute_no_overlap_reward(parsed)
        rewards["no_overlap"] = no_overlap_reward
        if cfg.get("credit_assignment", False) and no_overlap_errors:
            error_masks["no_overlap"] = build_error_mask(seq_length, no_overlap_errors)

    # R_connectivity
    cfg = reward_cfg.get("connectivity", {})
    if cfg.get("enabled", True):
        rewards["connectivity"] = compute_connectivity_reward(parsed, metadata)

    # R_spatial
    cfg = reward_cfg.get("spatial", {})
    if cfg.get("enabled", True):
        rewards["spatial"] = compute_spatial_reward(parsed, metadata)

    return {
        "rewards": rewards,
        "error_masks": error_masks,
        "hard_gate_pass": True,
        "parsed": parsed,
    }
