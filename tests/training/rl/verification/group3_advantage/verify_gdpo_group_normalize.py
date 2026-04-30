"""Group 3: gdpo_group_normalize 단독 검증.

mock 텐서로 그룹별 z-score 정규화 동작을 검증한다.

검증 케이스:
    - 동일 그룹 (std=0) → A=0 (NaN 방어)
    - 2 그룹 (N=2, G=2) z-score 손계산 일치
    - B_total % G != 0 → warning + 원본 반환
    - num_generations=1 → std=0 강제 → A=0
    - 큰 그룹 평균/표준편차 정확성
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_VERIF_ROOT = Path(__file__).resolve().parents[1]
if str(_VERIF_ROOT) not in sys.path:
    sys.path.insert(0, str(_VERIF_ROOT))

import torch  # noqa: E402

from _common import run_cases, summary_and_exit  # noqa: E402

from src.training.rl.advantage import gdpo_group_normalize  # noqa: E402


def case_identical_group_zero():
    """동일 보상이 그룹 내 모두면 std=0 → A=0."""
    rewards = torch.tensor([
        [1.0, 0.5],
        [1.0, 0.5],
        [1.0, 0.5],
        [1.0, 0.5],
    ])
    A = gdpo_group_normalize(rewards, num_generations=4)
    assert torch.allclose(A, torch.zeros_like(A), atol=1e-5), \
        f"동일 그룹 → A=0 기대, actual={A}"


def case_two_groups_zscore():
    """N=2, G=2 그룹별 z-score 손계산 일치."""
    rewards = torch.tensor([
        [1.0, 5.0],
        [3.0, 5.0],
        [2.0, 6.0],
        [4.0, 6.0],
    ])
    # Group 0: rewards [1,3], [5,5]. mean=[2,5], std=[1,0]
    # Group 1: rewards [2,4], [6,6]. mean=[3,6], std=[1,0]
    A = gdpo_group_normalize(rewards, num_generations=2)
    expected_g0 = torch.tensor([
        [(1.0 - 2.0) / 1.0, 0.0],
        [(3.0 - 2.0) / 1.0, 0.0],
    ])
    expected_g1 = torch.tensor([
        [(2.0 - 3.0) / 1.0, 0.0],
        [(4.0 - 3.0) / 1.0, 0.0],
    ])
    expected = torch.cat([expected_g0, expected_g1], dim=0)
    assert torch.allclose(A, expected, atol=1e-4), f"z-score 불일치: actual={A}, expected={expected}"


def case_size_mismatch_warning():
    """B_total이 num_generations 배수가 아니면 원본 반환."""
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    # B_total=3, num_generations=2 → mismatch
    A = gdpo_group_normalize(rewards, num_generations=2)
    assert torch.allclose(A, rewards), "size mismatch 시 원본 반환되어야 함"


def case_num_gen_one_zero_std():
    """num_generations=1이면 std 강제 0 → A=0."""
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    A = gdpo_group_normalize(rewards, num_generations=1)
    # 각 그룹의 std=0 → (값 - 값) / eps = 0
    assert torch.allclose(A, torch.zeros_like(A), atol=1e-5), \
        f"num_generations=1 → A=0 기대, actual={A}"


def case_nan_safe():
    """NaN 보상 포함 시 nanmean 사용 + std=0 fallback → 결과 NaN 0으로."""
    rewards = torch.tensor([
        [float("nan"), 1.0],
        [2.0, 3.0],
        [4.0, 5.0],
        [6.0, 7.0],
    ])
    A = gdpo_group_normalize(rewards, num_generations=2)
    assert not torch.any(torch.isnan(A)), f"NaN 방어 실패: {A}"


def case_large_group_mean_std():
    """큰 그룹 mean/std 직접 계산 일치 (G=4)."""
    rewards = torch.tensor([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ])
    # mean=2.5, var=((1.5)²+(0.5)²+(0.5)²+(1.5)²)/4 = (2.25+0.25+0.25+2.25)/4 = 1.25
    # std=sqrt(1.25)≈1.118
    A = gdpo_group_normalize(rewards, num_generations=4)
    expected_std = math.sqrt(1.25)
    expected = torch.tensor([
        [(1.0 - 2.5) / expected_std],
        [(2.0 - 2.5) / expected_std],
        [(3.0 - 2.5) / expected_std],
        [(4.0 - 2.5) / expected_std],
    ])
    assert torch.allclose(A, expected, atol=1e-4), f"big group z-score 불일치: actual={A}"


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
        _Case("identical_group_zero",   "그룹 내 동일 보상 → A=0 (std=0 NaN 방어)",          case_identical_group_zero),
        _Case("two_groups_zscore",      "N=2 G=2 z-score 손계산 일치",                       case_two_groups_zscore),
        _Case("size_mismatch_warn",     "B_total % G != 0 → 원본 반환",                      case_size_mismatch_warning),
        _Case("num_gen_one",            "num_generations=1 → std 강제 0 → A=0",              case_num_gen_one_zero_std),
        _Case("nan_safe",               "NaN 보상 포함 → 결과 NaN 0으로 변환",               case_nan_safe),
        _Case("large_group_mean_std",   "G=4 큰 그룹 평균/std 정확성",                       case_large_group_mean_std),
    ]
    results = run_cases(cases, lambda c: c.fn(), label="Group 3: gdpo_group_normalize")
    summary_and_exit(results, label="gdpo_group_normalize")


if __name__ == "__main__":
    main()
