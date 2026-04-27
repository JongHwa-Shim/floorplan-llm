"""토큰 수준 신용할당 모듈.

오류 토큰 마스크를 생성하고, 시퀀스 수준 어드밴티지를
토큰별 차등 어드밴티지로 변환하는 함수를 제공한다.

신용할당 수식:
    token_A[t] = A × (1 - mask[t]) - |A| × penalty_scale × mask[t]

    - 정상 토큰 (mask=0): A 그대로
    - 오류 토큰 (mask=1): -|A| × penalty_scale (항상 음수, A 부호 무관)
    - A가 양수든 음수든 오류 토큰은 항상 추가 페널티
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def build_error_mask(
    seq_length: int,
    error_indices: list[int],
) -> torch.Tensor:
    """오류 토큰 바이너리 마스크를 생성한다.

    Args:
        seq_length: 시퀀스 길이 (completion 토큰 수).
        error_indices: 오류 토큰 인덱스 리스트.

    Returns:
        바이너리 마스크 텐서. shape: $(L,)$
        오류 토큰=1.0, 정상 토큰=0.0. dtype=float32.
    """
    mask = torch.zeros(seq_length, dtype=torch.float32)
    for idx in error_indices:
        if 0 <= idx < seq_length:
            mask[idx] = 1.0
    return mask


def apply_token_credit_assignment(
    advantage: float,
    error_mask: torch.Tensor,
    penalty_scale: float,
) -> torch.Tensor:
    """시퀀스 수준 어드밴티지를 토큰별 차등 어드밴티지로 변환한다.

    수식:
        token_A[t] = A × (1 - mask[t]) - |A| × penalty_scale × mask[t]

    A가 양수이든 음수이든 오류 토큰(mask=1)은 항상 음수 방향 벌칙을 받는다:
        - A > 0, mask=1: 0 - |A| × penalty < 0 (이득 차단 + 페널티)
        - A < 0, mask=1: 0 - |A| × penalty < A (추가 페널티)

    Args:
        advantage: 시퀀스 수준 스칼라 어드밴티지 A.
        error_mask: 바이너리 마스크. shape: $(L,)$. 오류=1, 정상=0.
        penalty_scale: 오류 토큰 페널티 배율.

    Returns:
        토큰별 어드밴티지 텐서. shape: $(L,)$
    """
    abs_adv = abs(advantage)
    # 정상 토큰: A × (1 - mask)
    normal_part = advantage * (1.0 - error_mask)
    # 오류 토큰: -|A| × penalty_scale × mask
    penalty_part = abs_adv * penalty_scale * error_mask
    return normal_part - penalty_part
