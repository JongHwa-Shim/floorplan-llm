"""GDPO + 토큰 수준 신용할당 어드밴티지 계산 모듈.

RLTrainer._apply_token_credit_assignment()에서 호출되는
핵심 어드밴티지 계산 파이프라인.

처리 흐름:
    ① 보상별 그룹 내 정규화 (GDPO) — Trainer에서 전처리
       A_k = (r_k - mean_group(r_k)) / (std_group(r_k) + eps)
       ALL-PROCESS 데이터로 수행 → 로컬 슬라이스 추출 후 이 모듈에 전달

    ② 보상별 토큰 advantage 계산 (error_mask 적용)
       신용할당 ON (전역 토글 AND 보상별 설정 모두 True):
           token_A_k = A_k * [1 + sign(A_k) * (alpha*(1-mask_k) - beta*mask_k)]
                       - kappa * mask_k
       신용할당 OFF:
           token_A_k = A_k (broadcast, 모든 토큰 동일)

    ③ 보상 가중합 (토큰 수준)
       token_A_combined[t] = Σ(w_k × token_A_k[t])

    ④ 배치 정규화 (시퀀스 수준 대표값 기반, 토큰 차등 보존)
       (a) 시퀀스별 대표값: seq_mean_i = mean(token_A_combined_i)
       (b) 배치 통계: batch_mean, batch_std from {seq_mean_i}
       (c) 토큰별 정규화: token_A_final[t] = (token_A_combined[t] - batch_mean)
                                              / (batch_std + eps)
"""

from __future__ import annotations

import logging

import torch

from src.training.rl.rewards.credit_assignment import apply_token_credit_assignment

logger = logging.getLogger(__name__)


def gdpo_group_normalize(
    rewards_per_func: torch.Tensor,
    num_generations: int,
    eps: float = 1e-8,
    use_gdpo_normalization: bool = True,
) -> torch.Tensor:
    """보상별 그룹 내 z-score 정규화 (GDPO).

    각 프롬프트 그룹(G개 completion) 내에서 보상별로 독립적으로 정규화한다.
    이를 통해 보상 신호 간 척도 차이로 인한 붕괴를 방지한다.

    ALL-PROCESS 데이터로 호출해야 올바른 그룹 통계를 계산할 수 있다.

    Mod Record: ``use_gdpo_normalization=False`` 시 표준 GRPO (그룹 평균만 빼고 분산 정규화 미적용)
    동작으로 fallback — 가이드 Exp 10 (GDPO vs Standard GRPO) 의 baseline 변형 지원.

    Args:
        rewards_per_func: shape $(B_{total}, K)$
            $B_{total}$ = 전체 프로세스 completion 수 (gather 후).
        num_generations: 그룹 크기 G (프롬프트당 생성 개수).
        eps: 수치 안정성 엡실론.
        use_gdpo_normalization: True 면 보상별 z-score(GDPO). False 면 단순 그룹 평균 차감(GRPO).

    Returns:
        정규화된 보상별 어드밴티지. shape $(B_{total}, K)$
        그룹 내 모든 값이 동일하면 해당 그룹의 어드밴티지=0.
    """
    B_total, K = rewards_per_func.shape

    # B_total이 num_generations의 배수가 아닌 경우 방어
    if B_total % num_generations != 0:
        logger.warning(
            f"B_total({B_total})이 num_generations({num_generations})의 배수가 아님. "
            "GDPO 정규화 건너뜀."
        )
        return rewards_per_func

    N = B_total // num_generations  # 프롬프트 수 (전체)

    # (N, G, K)로 reshape하여 그룹별 통계 계산
    grouped = rewards_per_func.view(N, num_generations, K)  # (N, G, K)
    mean_k = grouped.nanmean(dim=1, keepdim=True)           # (N, 1, K)

    if num_generations > 1:
        # NaN을 0으로 처리하여 std 계산 (unbiased=False, 전체 group 기준)
        diff = grouped - mean_k                                  # (N, G, K)
        diff_clean = torch.where(torch.isnan(diff), torch.zeros_like(diff), diff)
        std_k = torch.sqrt((diff_clean ** 2).mean(dim=1, keepdim=True))  # (N, 1, K)
    else:
        std_k = torch.zeros_like(mean_k)

    if not use_gdpo_normalization:
        # Standard GRPO: 분산 정규화 미적용, 그룹 평균만 빼기 (가이드 Exp 10 baseline)
        A_k = grouped - mean_k                                          # (N, G, K)
        return A_k.view(B_total, K)

    A_k = (grouped - mean_k) / (std_k + eps)  # (N, G, K)
    # NaN 방어 (전체 그룹이 동일값이면 std=0, mean=값 → (값-값)/eps ≈ 0)
    A_k = torch.where(torch.isnan(A_k), torch.zeros_like(A_k), A_k)

    return A_k.view(B_total, K)  # (B_total, K)


def compute_token_advantages(
    A_k_local: torch.Tensor,
    reward_names: list[str],
    reward_cfgs: list[dict],
    error_masks_batch: list[dict[str, torch.Tensor]],
    completion_lengths: list[int],
    max_seq_len: int,
    eps: float = 1e-8,
    use_token_credit_assignment: bool = True,
    gather_fn=None,
) -> torch.Tensor:
    """Eq. (7)–(10)의 신용 할당, 가중합, 전체 배치 정규화를 수행한다.

    Mod Record: 보상별 배치 연산으로 GPU 스칼라 읽기를 제거한다. 위반이 없는
    마스크에도 α를 적용하고, 분산 실행에서는 시퀀스 대표값만 모은다.

    Args:
        A_k_local: 그룹별 정규화가 끝난 로컬 보상 어드밴티지, $(B, K)$.
        reward_names: 보상 열 이름.
        reward_cfgs: 보상 열에 대응하는 설정.
        error_masks_batch: 시퀀스별 위반 마스크.
        completion_lengths: 패딩을 제외한 길이.
        max_seq_len: 패딩을 포함한 길이.
        eps: 정규화 분모의 안정화 상수.
        use_token_credit_assignment: 전체 신용 할당 활성화 여부.
        gather_fn: 전체 프로세스 대표값을 모으는 함수. 단일 프로세스는 None.

    Returns:
        정규화된 토큰 어드밴티지 $(B, T)$. 패딩은 0.

    Raises:
        ValueError: 시퀀스 길이가 텐서 범위 밖이거나 배치 크기가 다를 때.
    """
    batch_size = A_k_local.shape[0]
    if len(completion_lengths) != batch_size or any(
        length < 0 or length > max_seq_len for length in completion_lengths
    ):
        raise ValueError("completion_lengths가 배치 크기/시퀀스 범위와 맞지 않습니다.")
    advantages = A_k_local.new_zeros((batch_size, max_seq_len))
    for k, (name, cfg) in enumerate(zip(reward_names, reward_cfgs)):
        if not cfg.get("enabled", True):
            continue
        scalar = A_k_local[:, k:k + 1]  # (B, 1)
        if use_token_credit_assignment and cfg.get("credit_assignment", False):
            # 파싱으로 생성한 CPU 마스크를 한 번에 전송한다.
            mask = torch.zeros((batch_size, max_seq_len))
            for i, length in enumerate(completion_lengths):
                errors = error_masks_batch[i].get(name) if i < len(error_masks_batch) else None
                if errors is not None:
                    count = min(len(errors), length)
                    mask[i, :count] = errors[:count].detach().cpu()
            mask = mask.to(device=scalar.device, dtype=scalar.dtype)
            token_credit = apply_token_credit_assignment(
                scalar, mask,
                nominal_gain=float(cfg.get("nominal_gain", 0.0)),
                faulty_attenuation=float(cfg.get("faulty_attenuation", 0.0)),
                penalty_offset=float(cfg.get("penalty_offset", 0.0)),
            )
        else:
            token_credit = scalar
        advantages += float(cfg.get("weight", 1.0)) * token_credit
    return _batch_normalize(advantages, completion_lengths, eps, gather_fn=gather_fn)


def _batch_normalize(
    token_advantages: torch.Tensor,
    completion_lengths: list[int],
    eps: float,
    gather_fn=None,
) -> torch.Tensor:
    """시퀀스 대표값을 전체 생성 배치에서 집계하여 토큰을 정규화한다.

    Args:
        token_advantages: 가중합 어드밴티지 $(B, T)$.
        completion_lengths: 패딩을 제외한 각 시퀀스 길이.
        eps: 분모의 안정화 상수.
        gather_fn: 전체 프로세스의 $(B, 2)$ 대표값/유효 여부를 모으는 함수.

    Returns:
        전체 배치 통계를 적용한 로컬 토큰 어드밴티지. 패딩은 0.

    Raises:
        없음.
    """
    lengths = torch.as_tensor(completion_lengths, device=token_advantages.device)
    valid_tokens = torch.arange(token_advantages.shape[1], device=lengths.device)[None, :] < lengths[:, None]
    masked = token_advantages.masked_fill(~valid_tokens, 0)
    means = masked.sum(dim=1) / lengths.clamp_min(1)  # (B,)
    # 길이 0(잘린 completion 전체가 loss에서 제외된 경우)은 통계에서도 제외한다.
    representatives = torch.stack((means, (lengths > 0).to(means.dtype)), dim=1)  # (B, 2)
    if gather_fn is not None:
        representatives = gather_fn(representatives)  # (M*G, 2)
    means_all = representatives[representatives[:, 1] > 0, 0]
    if means_all.numel() == 0:
        return torch.zeros_like(token_advantages)
    batch_mean = means_all.mean()
    # 기존 표본 표준편차 규약을 유지하되 집계 범위를 모든 프로세스로 확장한다.
    batch_std = means_all.std() if means_all.numel() > 1 else means_all.new_zeros(())
    normalized = (masked - batch_mean) / (batch_std + eps)
    return normalized.masked_fill(~valid_tokens, 0)
