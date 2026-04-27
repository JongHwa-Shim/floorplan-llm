"""GDPO + 토큰 수준 신용할당 어드밴티지 계산 모듈.

RLTrainer._apply_token_credit_assignment()에서 호출되는
핵심 어드밴티지 계산 파이프라인.

처리 흐름:
    ① 보상별 그룹 내 정규화 (GDPO) — Trainer에서 전처리
       A_k = (r_k - mean_group(r_k)) / (std_group(r_k) + eps)
       ALL-PROCESS 데이터로 수행 → 로컬 슬라이스 추출 후 이 모듈에 전달

    ② 보상별 토큰 advantage 계산 (error_mask 적용)
       신용할당 ON (전역 토글 AND 보상별 설정 모두 True):
           token_A_k = A_k × (1-mask_k) - |A_k| × penalty × mask_k
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
) -> torch.Tensor:
    """보상별 그룹 내 z-score 정규화 (GDPO).

    각 프롬프트 그룹(G개 completion) 내에서 보상별로 독립적으로 정규화한다.
    이를 통해 보상 신호 간 척도 차이로 인한 붕괴를 방지한다.

    ALL-PROCESS 데이터로 호출해야 올바른 그룹 통계를 계산할 수 있다.

    Args:
        rewards_per_func: shape $(B_{total}, K)$
            $B_{total}$ = 전체 프로세스 completion 수 (gather 후).
        num_generations: 그룹 크기 G (프롬프트당 생성 개수).
        eps: 수치 안정성 엡실론.

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
) -> torch.Tensor:
    """GDPO 정규화된 A_k로 토큰별 어드밴티지를 계산한다.

    Trainer에서 GDPO 정규화 + 로컬 슬라이싱 후 이 함수를 호출한다.

    처리:
        ② 보상별 토큰 advantage (신용할당 ON/OFF)
        ③ 보상 가중합 → token_A_combined
        ④ 배치 정규화 (시퀀스 수준 대표값)

    신용할당 적용 조건: use_token_credit_assignment AND 보상별 credit_assignment 설정 모두 True.

    Args:
        A_k_local: 로컬 프로세스의 정규화된 보상별 어드밴티지. shape: $(B_{local}, K)$
        reward_names: 보상함수 이름 리스트 (K개, reward_cfgs와 순서 일치).
        reward_cfgs: 보상 설정 딕셔너리 리스트 (K개).
            각 항목: {weight, credit_assignment, penalty_scale, enabled}.
        error_masks_batch: 로컬 배치 오류 마스크.
            error_masks_batch[i][reward_name] = shape $(L_i,)$ mask tensor.
        completion_lengths: 각 completion의 실제 토큰 수 리스트.
        max_seq_len: 패딩 포함 최대 시퀀스 길이 T.
        eps: 수치 안정성 엡실론.
        use_token_credit_assignment: False면 모든 신용할당을 비활성화하고
            균등 broadcast만 사용한다. config에서 전역 토글로 제어.

    Returns:
        배치 정규화된 토큰별 최종 어드밴티지. shape: $(B_{local}, T)$
    """
    device = A_k_local.device
    B_local = A_k_local.shape[0]
    T = max_seq_len

    # ② + ③: 보상별 토큰 advantage 계산 및 가중합
    token_advantages = torch.zeros(B_local, T, device=device)

    for i in range(B_local):
        seq_len = completion_lengths[i]
        error_masks_i = error_masks_batch[i] if i < len(error_masks_batch) else {}

        for k, (name, cfg) in enumerate(zip(reward_names, reward_cfgs)):
            if not cfg.get("enabled", True):
                continue

            w_k = float(cfg.get("weight", 1.0))
            A_k_i = A_k_local[i, k].item()  # 스칼라

            token_A_k = torch.zeros(T, device=device)

            # 신용할당 ON/OFF 결정: 전역 토글 AND 보상별 credit_assignment 설정 모두 True
            use_credit = use_token_credit_assignment and cfg.get("credit_assignment", False)
            if use_credit:
                error_mask = error_masks_i.get(name)
                if error_mask is not None and error_mask.sum() > 0:
                    # 완전한 마스크 텐서 구성 (seq_len까지만 유효)
                    mask_len = min(len(error_mask), seq_len)
                    mask_for_credit = torch.zeros(seq_len, device=device)
                    mask_for_credit[:mask_len] = error_mask[:mask_len].to(device)

                    # token_A_k = A × (1-mask) - |A| × penalty × mask
                    token_A_seq = apply_token_credit_assignment(
                        advantage=A_k_i,
                        error_mask=mask_for_credit,
                        penalty_scale=float(cfg.get("penalty_scale", 1.0)),
                    )
                    token_A_k[:seq_len] = token_A_seq
                else:
                    # 오류 마스크 없거나 오류 없음 → 균등 broadcast
                    token_A_k[:seq_len] = A_k_i
            else:
                # 신용할당 OFF → 모든 토큰 동일
                token_A_k[:seq_len] = A_k_i

            token_advantages[i] += w_k * token_A_k

    # ④ 배치 정규화 (시퀀스 수준 대표값 기반, 토큰 차등 보존)
    token_advantages = _batch_normalize(
        token_advantages=token_advantages,
        completion_lengths=completion_lengths,
        eps=eps,
    )

    return token_advantages


def _batch_normalize(
    token_advantages: torch.Tensor,
    completion_lengths: list[int],
    eps: float,
) -> torch.Tensor:
    """배치 정규화 (시퀀스 수준 대표값 기반, 토큰 차등 보존).

    각 시퀀스의 completion 토큰 advantage를 평균하여 대표값을 구하고,
    배치 전체 대표값들의 평균/표준편차로 정규화한다.
    토큰 간 상대적 차이는 보존된다.

    Args:
        token_advantages: 정규화 전 토큰별 어드밴티지. shape $(B, T)$
        completion_lengths: 각 completion의 실제 토큰 수.
        eps: 수치 안정성 엡실론.

    Returns:
        정규화된 토큰별 어드밴티지. shape $(B, T)$
    """
    B = token_advantages.shape[0]
    device = token_advantages.device

    # (a) 시퀀스별 completion 토큰 advantage 평균 계산
    seq_means = torch.zeros(B, device=device)
    for i in range(B):
        seq_len = completion_lengths[i]
        if seq_len > 0:
            seq_means[i] = token_advantages[i, :seq_len].mean()

    # (b) 배치 통계 계산 (시퀀스 대표값 기반)
    batch_mean = seq_means.mean()
    batch_std = seq_means.std() if B > 1 else torch.zeros(1, device=device).squeeze()

    # (c) 모든 토큰에 동일한 배치 통계 적용 (상대적 차이 보존)
    token_advantages = (token_advantages - batch_mean) / (batch_std + eps)

    return token_advantages
