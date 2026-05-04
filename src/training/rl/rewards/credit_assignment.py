"""토큰 수준 신용할당 모듈 (Sign-Asymmetric Credit Assignment with Penalty Offset).

오류 토큰 마스크를 생성하고, 시퀀스 수준 어드밴티지를 토큰별 차등 어드밴티지로
변환하는 함수를 제공한다.

신용할당 공식 (옵션 F):
    a_t = A * [1 + sign(A) * (alpha*(1-m_t) - beta*m_t)] - kappa * m_t

조건별 전개:
    - 정상 토큰 (m_t=0): a_t = A * (1 + alpha * sign(A))
    - 오류 토큰 (m_t=1): a_t = A * (1 - beta * sign(A)) - kappa

설계 의도 (4-cell):
    | A 부호 | 정상 토큰          | 오류 토큰              |
    |--------|--------------------|------------------------|
    | A > 0  | A(1+alpha) ≥ A     | A(1-beta) - kappa < A  |
    | A ≈ 0  | 0                  | -kappa  (보장된 페널티) |
    | A < 0  | A(1-alpha) (가벼움)| A(1+beta) - kappa (강화)|

Mod Record:
    이전 수식 a_t = A(1-m_t) - |A| * penalty * m_t 은 |A| → 0일 때 페널티가
    소실되어 GDPO 그룹 평균 근처 시퀀스에서 신용할당 신호가 사라지는 결함이 있었다.
    옵션 F는 (1) 정상 토큰의 magnitude도 조정하고 (2) advantage와 무관한 절대
    페널티 오프셋(kappa)을 추가하여 |A|=0 케이스를 포함한 4-cell 의도를 모두 충족한다.
"""

from __future__ import annotations

import logging
import math

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
    nominal_gain: float,
    faulty_attenuation: float,
    penalty_offset: float,
) -> torch.Tensor:
    r"""시퀀스 수준 어드밴티지를 토큰별 차등 어드밴티지로 변환한다 (옵션 F).

    Sign-Asymmetric Credit Assignment with Unconditional Penalty Offset.

    수식:
        $$a_t = A \cdot \bigl[1 + \mathrm{sign}(A) \cdot
                  \bigl(\alpha (1 - m_t) - \beta m_t\bigr)\bigr] - \kappa m_t$$

    조건별 전개:
        - 정상 토큰 ($m_t = 0$): $a_t = A (1 + \alpha \cdot \mathrm{sign}(A))$
        - 오류 토큰 ($m_t = 1$): $a_t = A (1 - \beta \cdot \mathrm{sign}(A)) - \kappa$

    의도된 4-cell 동작:
        - $A > 0$, 정상: 더 큰 상 ($A(1+\alpha)$)
        - $A > 0$, 오류: 작은 상 또는 벌 ($A(1-\beta) - \kappa$)
        - $A \approx 0$, 정상: $0$ (변화 없음)
        - $A \approx 0$, 오류: $-\kappa$ (보장된 벌)
        - $A < 0$, 정상: 가벼운 벌 ($A(1-\alpha)$, magnitude 축소)
        - $A < 0$, 오류: 더 센 벌 ($A(1+\beta) - \kappa$, magnitude 증폭)

    Args:
        advantage: 시퀀스 수준 스칼라 어드밴티지 $A$.
        error_mask: 바이너리 마스크. shape: $(L,)$. 오류=1, 정상=0.
        nominal_gain: $\alpha \in [0, 1)$. 정상 토큰 신용 이득 계수.
            $0$이면 정상 토큰의 magnitude 변화 없음.
        faulty_attenuation: $\beta \in [0, 1)$. 오류 토큰 신용 감쇄 계수.
            $A > 0$에서는 magnitude를 줄이고, $A < 0$에서는 magnitude를 키운다.
            $0$이면 오류 토큰의 advantage 부분은 $A$ 그대로.
        penalty_offset: $\kappa \geq 0$. advantage와 무관하게 오류 토큰에 더하는
            절대 페널티. $A = 0$ 케이스에서 페널티 보장 역할.

    Returns:
        토큰별 어드밴티지 텐서. shape: $(L,)$
    """
    # sign(0)=0이므로 A=0일 때는 정상/오류 모두 magnitude 변형 항이 0이 되고,
    # 오류 토큰에는 -kappa만 남는 구조.
    sign_a = math.copysign(1.0, advantage) if advantage != 0.0 else 0.0
    nominal_factor = 1.0 + nominal_gain * sign_a
    faulty_factor = 1.0 - faulty_attenuation * sign_a

    nominal_part = advantage * nominal_factor * (1.0 - error_mask)
    faulty_part = (advantage * faulty_factor - penalty_offset) * error_mask
    return nominal_part + faulty_part
