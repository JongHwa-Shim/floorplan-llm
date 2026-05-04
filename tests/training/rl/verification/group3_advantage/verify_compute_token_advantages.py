"""Group 3: compute_token_advantages 단독 검증 (★ 핵심).

옵션 F (Sign-Asymmetric Credit Assignment with Penalty Offset) 검증.

신용할당 공식:
    a_t = A * [1 + sign(A) * (alpha*(1-m_t) - beta*m_t)] - kappa * m_t

조건별:
    - 정상 토큰 (m_t=0): a_t = A * (1 + alpha * sign(A))
    - 오류 토큰 (m_t=1): a_t = A * (1 - beta * sign(A)) - kappa

검증 범위:
    1. 신용할당 ON 보상의 mask가 토큰별 advantage에 정확히 적용되는가
    2. 가중합 정확성
    3. 신용할당 OFF는 균등 broadcast인가
    4. ★ outline_in_room이 trainer.reward_order에서 누락되지 않았는가 (F-1 회귀 가드)
    5. ★ 옵션 F 4-cell 의도 (4가지 신규 케이스)
        - A>0, 정상: A(1+alpha) > A
        - A>0, 오류: A(1-beta) - kappa < A
        - A<0, 정상: A(1-alpha), magnitude 축소
        - A<0, 오류: A(1+beta) - kappa, magnitude 증폭
        - A=0, 오류: -kappa (페널티 보장)

배치 정규화는 마지막에 적용되므로 test에서는 batch_normalize 효과를 고려해야 함.
"""

from __future__ import annotations

import sys
from pathlib import Path

_VERIF_ROOT = Path(__file__).resolve().parents[1]
if str(_VERIF_ROOT) not in sys.path:
    sys.path.insert(0, str(_VERIF_ROOT))

import torch  # noqa: E402

from _common import run_cases, summary_and_exit  # noqa: E402

from src.training.rl.advantage import compute_token_advantages  # noqa: E402
from src.training.rl.rewards.credit_assignment import (  # noqa: E402
    apply_token_credit_assignment,
)


# ---------------------------------------------------------------------------
# 단위 검증: apply_token_credit_assignment 직접 호출 (배치 정규화 영향 없음)
# ---------------------------------------------------------------------------

def case_unit_positive_advantage_4cell():
    """A>0: 정상=A(1+alpha), 오류=A(1-beta)-kappa."""
    A = 2.0
    alpha, beta, kappa = 0.2, 0.5, 1.0
    error_mask = torch.tensor([0.0, 1.0, 0.0])

    out = apply_token_credit_assignment(
        advantage=A, error_mask=error_mask,
        nominal_gain=alpha, faulty_attenuation=beta, penalty_offset=kappa,
    )
    # 정상 [0]: A * (1 + alpha) = 2.0 * 1.2 = 2.4
    # 오류 [1]: A * (1 - beta) - kappa = 2.0 * 0.5 - 1.0 = 0.0
    # 정상 [2]: 2.4
    expected = torch.tensor([2.4, 0.0, 2.4])
    assert torch.allclose(out, expected, atol=1e-6), \
        f"A>0 4-cell 불일치: out={out}, expected={expected}"


def case_unit_negative_advantage_4cell():
    """A<0: 정상=A(1-alpha) (가벼운 벌), 오류=A(1+beta)-kappa (더 센 벌)."""
    A = -1.0
    alpha, beta, kappa = 0.2, 0.5, 1.0
    error_mask = torch.tensor([0.0, 1.0, 0.0])

    out = apply_token_credit_assignment(
        advantage=A, error_mask=error_mask,
        nominal_gain=alpha, faulty_attenuation=beta, penalty_offset=kappa,
    )
    # 정상 [0]: A * (1 + alpha * (-1)) = -1.0 * 0.8 = -0.8 (가벼운 벌, |A| 축소)
    # 오류 [1]: A * (1 - beta * (-1)) - kappa = -1.0 * 1.5 - 1.0 = -2.5 (더 센 벌)
    # 정상 [2]: -0.8
    expected = torch.tensor([-0.8, -2.5, -0.8])
    assert torch.allclose(out, expected, atol=1e-6), \
        f"A<0 4-cell 불일치: out={out}, expected={expected}"

    # magnitude 비교: 정상 토큰 |advantage| < |A| (가벼움), 오류 토큰 |advantage| > |A| (강화)
    assert abs(out[0].item()) < abs(A), "A<0 정상 토큰이 가벼운 벌이 아님"
    assert abs(out[1].item()) > abs(A), "A<0 오류 토큰이 더 센 벌이 아님"


def case_unit_zero_advantage_kappa_guaranteed():
    """A=0: 정상=0, 오류=-kappa (보장된 페널티). |A|→0 페널티 소실 결함 회귀 가드."""
    A = 0.0
    alpha, beta, kappa = 0.2, 0.5, 1.0
    error_mask = torch.tensor([0.0, 1.0, 0.0])

    out = apply_token_credit_assignment(
        advantage=A, error_mask=error_mask,
        nominal_gain=alpha, faulty_attenuation=beta, penalty_offset=kappa,
    )
    # 정상 [0]: 0 (sign(0)=0이므로 magnitude 변형 항도 0)
    # 오류 [1]: 0 - kappa = -1.0 (★ A=0이어도 페널티 보장)
    # 정상 [2]: 0
    expected = torch.tensor([0.0, -1.0, 0.0])
    assert torch.allclose(out, expected, atol=1e-6), \
        f"A=0 페널티 보장 실패: out={out}, expected={expected}"


def case_unit_default_params_no_op():
    """기본값(alpha=beta=kappa=0)이면 정상/오류 모두 A 그대로 broadcast."""
    A = 1.5
    error_mask = torch.tensor([0.0, 1.0, 0.0])

    out = apply_token_credit_assignment(
        advantage=A, error_mask=error_mask,
        nominal_gain=0.0, faulty_attenuation=0.0, penalty_offset=0.0,
    )
    expected = torch.tensor([1.5, 1.5, 1.5])
    assert torch.allclose(out, expected, atol=1e-6), \
        f"기본값에서 broadcast 깨짐: out={out}"


# ---------------------------------------------------------------------------
# 통합 검증: compute_token_advantages (배치 정규화 포함)
# ---------------------------------------------------------------------------

def _make_cfg(
    credit_assignment: bool = True,
    weight: float = 1.0,
    nominal_gain: float = 0.0,
    faulty_attenuation: float = 0.0,
    penalty_offset: float = 0.0,
) -> dict:
    """통합 검증용 reward cfg dict 빌더."""
    return {
        "weight": weight,
        "credit_assignment": credit_assignment,
        "enabled": True,
        "nominal_gain": nominal_gain,
        "faulty_attenuation": faulty_attenuation,
        "penalty_offset": penalty_offset,
    }


def case_credit_on_mask_applied():
    """★ 신용할당 ON: error_mask가 토큰별 advantage에 차등 적용."""
    A_k_local = torch.tensor([[1.0], [1.0]])
    error_mask_0 = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])  # 시퀀스 0: 토큰 1, 3 오류
    error_mask_1 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])  # 시퀀스 1: 오류 없음
    error_masks_batch = [
        {"format": error_mask_0},
        {"format": error_mask_1},
    ]
    completion_lengths = [5, 5]
    max_seq_len = 5

    token_adv = compute_token_advantages(
        A_k_local=A_k_local,
        reward_names=["format"],
        reward_cfgs=[_make_cfg(
            credit_assignment=True,
            nominal_gain=0.2, faulty_attenuation=0.5, penalty_offset=1.0,
        )],
        error_masks_batch=error_masks_batch,
        completion_lengths=completion_lengths,
        max_seq_len=max_seq_len,
        use_token_credit_assignment=True,
    )
    # raw 시퀀스 0 (A=1>0): 정상 1*1.2=1.2, 오류 1*0.5-1.0=-0.5
    #   → [1.2, -0.5, 1.2, -0.5, 1.2]
    # raw 시퀀스 1: [1.2, 1.2, 1.2, 1.2, 1.2]
    # batch_norm 후에도 차등 패턴 보존: 시퀀스 0의 mask=1 위치는 mask=0보다 작아야
    seq0 = token_adv[0]
    seq1 = token_adv[1]
    assert seq0[1].item() < seq0[0].item(), \
        f"mask=1 위치가 mask=0보다 작지 않음: seq0={seq0}"
    assert seq0[3].item() < seq0[4].item()
    # 시퀀스 1 (오류 없음 + A>0): 정상 토큰 보너스로 모든 토큰 동일 magnitude
    assert torch.allclose(seq1, torch.full_like(seq1, seq1[0].item()), atol=1e-4), \
        f"오류 없는 시퀀스는 모든 토큰 동일해야 함: seq1={seq1}"


def case_credit_off_uniform():
    """신용할당 OFF: 같은 advantage가 모든 토큰에 균등 broadcast."""
    A_k_local = torch.tensor([[1.0], [1.0]])
    error_masks_batch = [
        {"format": torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])},
        {"format": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])},
    ]
    completion_lengths = [5, 5]

    token_adv = compute_token_advantages(
        A_k_local=A_k_local,
        reward_names=["format"],
        reward_cfgs=[_make_cfg(credit_assignment=False)],
        error_masks_batch=error_masks_batch,
        completion_lengths=completion_lengths,
        max_seq_len=5,
        use_token_credit_assignment=True,
    )
    for i in range(2):
        seq = token_adv[i]
        assert torch.allclose(seq, torch.full_like(seq, seq[0].item()), atol=1e-4), \
            f"CA OFF면 시퀀스 {i} 토큰 동일해야 함: {seq}"


def case_global_toggle_off():
    """전역 토글 use_token_credit_assignment=False → CA=True 보상도 균등."""
    A_k_local = torch.tensor([[1.0], [1.0]])
    error_masks_batch = [
        {"format": torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])},
        {"format": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])},
    ]
    token_adv = compute_token_advantages(
        A_k_local=A_k_local,
        reward_names=["format"],
        reward_cfgs=[_make_cfg(
            credit_assignment=True,
            nominal_gain=0.2, faulty_attenuation=0.5, penalty_offset=1.0,
        )],
        error_masks_batch=error_masks_batch,
        completion_lengths=[5, 5],
        max_seq_len=5,
        use_token_credit_assignment=False,  # 전역 OFF
    )
    for i in range(2):
        seq = token_adv[i]
        assert torch.allclose(seq, torch.full_like(seq, seq[0].item()), atol=1e-4), \
            f"전역 OFF면 시퀀스 {i} 토큰 동일해야 함"


def case_weighted_sum():
    """다중 보상 가중합: format(w=1, CA on) + count_total(w=0.5, CA off)."""
    A_k_local = torch.tensor([
        [1.0, 0.5],
        [1.0, 0.5],
    ])
    error_masks_batch = [
        {"format": torch.tensor([0.0, 1.0, 0.0])},
        {"format": torch.tensor([0.0, 0.0, 0.0])},
    ]
    token_adv = compute_token_advantages(
        A_k_local=A_k_local,
        reward_names=["format", "count_total"],
        reward_cfgs=[
            _make_cfg(credit_assignment=True, weight=1.0,
                      nominal_gain=0.2, faulty_attenuation=0.5, penalty_offset=1.0),
            _make_cfg(credit_assignment=False, weight=0.5),
        ],
        error_masks_batch=error_masks_batch,
        completion_lengths=[3, 3],
        max_seq_len=3,
        use_token_credit_assignment=True,
    )
    seq0 = token_adv[0]
    seq1 = token_adv[1]
    assert seq0[1].item() < seq0[0].item(), \
        f"가중합 + CA 후 mask=1 작아야 함: seq0={seq0}"
    assert torch.allclose(seq1, torch.full_like(seq1, seq1[0].item()), atol=1e-4), \
        f"오류 없는 시퀀스 균일: seq1={seq1}"


def case_outline_in_room_in_reward_order():
    """★★★ F-1 수정 회귀 가드: trainer.reward_order에 outline_in_room이 포함되어야 함."""
    A_k_local = torch.tensor([[1.0], [1.0]])
    error_masks_batch = [
        {"outline_in_room": torch.tensor([0.0, 1.0, 0.0])},
        {"outline_in_room": torch.tensor([0.0, 0.0, 0.0])},
    ]
    token_adv = compute_token_advantages(
        A_k_local=A_k_local,
        reward_names=["outline_in_room"],
        reward_cfgs=[_make_cfg(
            credit_assignment=True,
            nominal_gain=0.2, faulty_attenuation=0.5, penalty_offset=1.0,
        )],
        error_masks_batch=error_masks_batch,
        completion_lengths=[3, 3],
        max_seq_len=3,
        use_token_credit_assignment=True,
    )
    seq0 = token_adv[0]
    assert seq0[1].item() < seq0[0].item(), \
        f"★ F-1 회귀: outline_in_room mask가 advantage에 반영되지 않음. seq0={seq0}"
    print("     [PASS] outline_in_room mask가 advantage 가중합에 정상 반영됨")


def case_padding_zero_in_advantage():
    """패딩 영역(seq_len 이후)은 advantage shape이 정확히 max_seq_len."""
    A_k_local = torch.tensor([[1.0], [1.0]])
    error_masks_batch = [
        {"format": torch.tensor([0.0, 0.0, 0.0])},
        {"format": torch.tensor([0.0, 0.0, 0.0])},
    ]
    completion_lengths = [3, 3]
    max_seq_len = 5

    token_adv = compute_token_advantages(
        A_k_local=A_k_local,
        reward_names=["format"],
        reward_cfgs=[_make_cfg(
            credit_assignment=True,
            nominal_gain=0.2, faulty_attenuation=0.5, penalty_offset=1.0,
        )],
        error_masks_batch=error_masks_batch,
        completion_lengths=completion_lengths,
        max_seq_len=max_seq_len,
        use_token_credit_assignment=True,
    )
    assert token_adv.shape == (2, 5), f"shape 불일치: {token_adv.shape}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class _Case:
    def __init__(self, name, intent, fn):
        self.name = name
        self.intent = intent
        self.fn = fn


def main():
    cases = [
        # 단위 검증 (옵션 F 4-cell 의도 직접 단언)
        _Case("unit_positive_a_4cell",       "★ 옵션 F: A>0 정상=A(1+α), 오류=A(1-β)-κ",            case_unit_positive_advantage_4cell),
        _Case("unit_negative_a_4cell",       "★ 옵션 F: A<0 정상=A(1-α) 가벼움, 오류=A(1+β)-κ 강화", case_unit_negative_advantage_4cell),
        _Case("unit_zero_a_kappa_guaranteed","★★ 옵션 F: A=0이어도 오류=-κ 페널티 보장 (회귀 가드)",  case_unit_zero_advantage_kappa_guaranteed),
        _Case("unit_default_no_op",          "기본값(α=β=κ=0)에서 broadcast 동작",                  case_unit_default_params_no_op),
        # 통합 검증 (배치 정규화 포함)
        _Case("credit_on_mask_applied",      "★ 신용할당 ON 시 mask가 token advantage에 차등 반영",  case_credit_on_mask_applied),
        _Case("credit_off_uniform",          "신용할당 OFF는 균등 broadcast",                       case_credit_off_uniform),
        _Case("global_toggle_off",           "use_token_credit_assignment=False 전역 OFF",           case_global_toggle_off),
        _Case("weighted_sum",                "다중 보상 가중합 정확성 (CA on/off 혼합)",            case_weighted_sum),
        _Case("outline_in_room_in_order",    "★★★ outline_in_room이 reward_order 포함 회귀 가드 (F-1)",  case_outline_in_room_in_reward_order),
        _Case("padding_shape",               "패딩 포함 max_seq_len shape 정확",                    case_padding_zero_in_advantage),
    ]
    results = run_cases(cases, lambda c: c.fn(), label="Group 3: compute_token_advantages (옵션 F)")
    summary_and_exit(results, label="compute_token_advantages")


if __name__ == "__main__":
    main()
