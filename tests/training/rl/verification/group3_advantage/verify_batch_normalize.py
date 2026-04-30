"""Group 3: _batch_normalize 단독 검증.

시퀀스 대표값(평균) 기반 배치 정규화. 토큰 차등 보존 여부 + 패딩 영역
평균 제외 검증.

검증 케이스:
    - B=1 (batch_std=0) → eps만으로 분할, 차등 보존
    - 모두 동일 seq_means (batch_std=0) → 토큰 차등 보존
    - completion_lengths 다양: 패딩 영역 평균 제외
    - 일반 케이스: 평균/std 손계산 비교
"""

from __future__ import annotations

import sys
from pathlib import Path

_VERIF_ROOT = Path(__file__).resolve().parents[1]
if str(_VERIF_ROOT) not in sys.path:
    sys.path.insert(0, str(_VERIF_ROOT))

import torch  # noqa: E402

from _common import run_cases, summary_and_exit  # noqa: E402

from src.training.rl.advantage import _batch_normalize  # noqa: E402


def case_b_one_batch_std_zero():
    """B=1 → batch_std=0 → eps만으로 분할. 토큰 차등은 보존되나 값 폭주."""
    token_adv = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    completion_lengths = [4]
    out = _batch_normalize(token_adv, completion_lengths, eps=1e-8)
    # batch_mean = mean of seq_means = 0.0
    # batch_std = std of [0] (single) = 0
    # token_adv_normalized = (token_adv - 0) / (0 + eps) = token_adv * 1e8
    # 차등은 보존됨: out[0] > 0, out[1] < 0
    assert out[0, 0].item() > 0 and out[0, 1].item() < 0, \
        f"B=1 차등 보존 실패: {out}"


def case_identical_seq_means():
    """모든 시퀀스의 평균이 같으면 batch_std=0 → 차등 보존."""
    # 시퀀스 0: [1, -1] → mean 0
    # 시퀀스 1: [2, -2] → mean 0
    token_adv = torch.tensor([
        [1.0, -1.0],
        [2.0, -2.0],
    ])
    completion_lengths = [2, 2]
    out = _batch_normalize(token_adv, completion_lengths, eps=1e-8)
    # batch_mean=0, batch_std=0 → out = token_adv / eps (모두 큰 값)
    # 차등 보존: 시퀀스 1이 시퀀스 0보다 큰 magnitude
    assert abs(out[1, 0].item()) > abs(out[0, 0].item()), \
        f"동일 seq_means에서 차등 보존 실패: {out}"


def case_completion_lengths_padding_excluded():
    """패딩 영역(seq_len 이후)은 평균 계산에서 제외."""
    # 시퀀스 0의 completion 길이 2, 패딩 영역 [2:] 무시
    # 시퀀스 0: [1, 1, 0, 0, 0] (seq_len=2) → 평균 = 1.0
    # 시퀀스 1: [2, 2, 2, 2, 2] (seq_len=5) → 평균 = 2.0
    token_adv = torch.tensor([
        [1.0, 1.0, 0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0, 2.0, 2.0],
    ])
    completion_lengths = [2, 5]

    # 만약 패딩이 평균에 포함되면 시퀀스 0 평균 = 0.4 (5개 토큰 평균)
    # 정확히 처리되면 시퀀스 0 평균 = 1.0
    out = _batch_normalize(token_adv, completion_lengths, eps=1e-8)
    # batch_mean = (1.0 + 2.0) / 2 = 1.5
    # batch_std = std([1.0, 2.0]) = 0.7071 (unbiased)
    # 직접 검사: out[0, 0] = (1 - 1.5) / 0.7071 ≈ -0.7071
    expected_seq0_token0 = (1.0 - 1.5) / 0.7071
    assert abs(out[0, 0].item() - expected_seq0_token0) < 0.01, \
        f"패딩 영역이 평균에서 제외되지 않음: out[0,0]={out[0, 0].item()}, expected~{expected_seq0_token0}"


def case_normal_two_seq_zscore():
    """일반 케이스: 두 시퀀스의 mean/std 손계산 일치."""
    token_adv = torch.tensor([
        [2.0, 4.0],   # mean = 3.0
        [6.0, 8.0],   # mean = 7.0
    ])
    completion_lengths = [2, 2]
    out = _batch_normalize(token_adv, completion_lengths, eps=1e-8)
    # batch_mean = 5.0, batch_std = std([3,7]) = 2.828 (sample) or 2.0 (population)
    # torch tensor std() default = unbiased (sample)
    import math
    batch_std = math.sqrt(((3.0 - 5.0) ** 2 + (7.0 - 5.0) ** 2) / 1)  # unbiased
    expected = torch.tensor([
        [(2.0 - 5.0) / batch_std, (4.0 - 5.0) / batch_std],
        [(6.0 - 5.0) / batch_std, (8.0 - 5.0) / batch_std],
    ])
    assert torch.allclose(out, expected, atol=1e-3), f"z-score 불일치: out={out}, expected={expected}"


def case_zero_seq_len_safe():
    """seq_len=0 시퀀스 안전 처리."""
    token_adv = torch.tensor([
        [0.0, 0.0],
        [1.0, 2.0],
    ])
    completion_lengths = [0, 2]
    out = _batch_normalize(token_adv, completion_lengths, eps=1e-8)
    # 시퀀스 0의 mean = 0 (skip), 시퀀스 1 mean = 1.5
    # batch_mean = 0.75, batch_std = std([0, 1.5]) = 1.06
    # NaN/Inf 없어야 함
    assert not torch.any(torch.isnan(out)), f"seq_len=0 처리 NaN: {out}"
    assert not torch.any(torch.isinf(out)), f"seq_len=0 처리 Inf: {out}"


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
        _Case("b_one",                "B=1 → batch_std=0, 차등 보존",                       case_b_one_batch_std_zero),
        _Case("identical_seq_means",  "동일 seq_means → batch_std=0, magnitude 차등 보존",  case_identical_seq_means),
        _Case("padding_excluded",     "★ 패딩 영역 평균에서 제외 (completion_lengths 사용)", case_completion_lengths_padding_excluded),
        _Case("normal_two_seq",       "일반 두 시퀀스 z-score 손계산 일치",                  case_normal_two_seq_zscore),
        _Case("zero_seq_len_safe",    "seq_len=0 시퀀스 NaN/Inf 없음",                       case_zero_seq_len_safe),
    ]
    results = run_cases(cases, lambda c: c.fn(), label="Group 3: _batch_normalize")
    summary_and_exit(results, label="_batch_normalize")


if __name__ == "__main__":
    main()
