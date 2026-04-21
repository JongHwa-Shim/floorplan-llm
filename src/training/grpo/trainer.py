"""GDPOTrainer 모듈.

TRL GRPOTrainer를 서브클래싱하여 GDPO + 토큰 수준 신용할당을 구현한다.

설계 원칙:
    1. _calculate_rewards() 오버라이드:
       - ALL-PROCESS rewards_per_func 캐싱
       - 로컬 completion_ids 캐싱
       - 보상 계산 결과(스칼라 + error_mask) 버퍼 저장

    2. _generate_and_score_completions() 오버라이드:
       - 부모 호출로 generation + 보상 계산 + 스칼라 advantages 획득
       - _apply_token_credit_assignment()로 토큰별 advantages 변환

    DDP 처리:
       - rewards_per_func: ALL-PROCESS (B_total, K) — gather() 후
       - completion_ids: LOCAL (B_local × G, T)
       - process_slice: trainer가 A_k_local 슬라이싱에 사용
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from trl import GRPOTrainer

from src.training.augmentation.tokenizer import Vocab
from src.training.grpo.rewards import compute_all_rewards
from src.training.grpo.advantage import gdpo_group_normalize, compute_token_advantages

logger = logging.getLogger(__name__)


class GDPOTrainer(GRPOTrainer):
    """GDPO + 토큰 수준 신용할당 Trainer.

    TRL GRPOTrainer를 확장하여 다음을 구현한다:
        1. 6개 Rule-based 보상함수 (format/count/geometry/connectivity/spatial)
        2. 보상 계산 시 error_mask 자동 생성 및 버퍼 저장
        3. ALL-PROCESS rewards_per_func 캐싱
        4. 스칼라 advantages → 토큰별 advantages 변환 (GDPO + 신용할당 + 배치 정규화)

    Args:
        reward_cfg: Hydra DictConfig. rewards 섹션.
        advantage_cfg: Hydra DictConfig. advantage 섹션.
        vocab: Vocab 객체 (토큰 파싱용).
        **kwargs: GRPOTrainer 기본 인자.
    """

    def __init__(
        self,
        reward_cfg: DictConfig,
        advantage_cfg: DictConfig,
        vocab: Vocab,
        **kwargs: Any,
    ) -> None:
        self.reward_cfg = reward_cfg
        self.advantage_cfg = advantage_cfg
        self.vocab = vocab

        # 보상 계산 버퍼 (생성 배치마다 초기화)
        self._error_masks_buffer: list[dict[str, torch.Tensor]] = []
        self._cached_local_completion_ids: list[list[int]] = []
        self._cached_rewards_per_func: torch.Tensor | None = None

        # 보상함수 이름 및 설정 리스트 (순서가 rewards_per_func 열 순서와 일치)
        self._reward_names: list[str] = []
        self._reward_cfgs_list: list[dict] = []

        # 보상 계산 캐시 (동일 step에서 중복 계산 방지)
        self._reward_results_cache: dict[int, dict[str, float]] = {}
        self._reward_cache_step: int = -1

        # 보상함수 callable 생성
        reward_funcs = self._build_reward_funcs()

        # TRL에 reward_weights 전달하지 않음: 이 Trainer가 직접 가중치 적용
        # (TRL의 normalize_then_sum 모드 대신 자체 GDPO + 신용할당 사용)
        super().__init__(reward_funcs=reward_funcs, **kwargs)

    # ---------------------------------------------------------------------------
    # 저장 / 로드 오버라이드
    # ---------------------------------------------------------------------------

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        """GRPO adapter만 저장한다 (SFT adapter와 base model 제외).

        Mod Record: 이전 구조에서는 merge_dora_and_save()로 전체 모델을 저장했음.
        새 구조에서는 GRPO adapter(adapter_name="grpo")만 저장하여 다음 단계(추론/분석)에서
        멀티 어댑터 스태킹으로 복원한다.

        Args:
            output_dir: 저장 대상 디렉토리. None이면 TrainingArguments.output_dir 사용.
            _internal_call: Trainer 내부 호출 여부 (무시).
        """
        save_dir = output_dir or self.args.output_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # accelerator로 DDP 래퍼 언래핑
        raw_model = self.accelerator.unwrap_model(self.model)
        # GRPO adapter만 선택적 저장
        raw_model.save_pretrained(save_dir, selected_adapters=["grpo"])
        if self.processing_class is not None:
            self.processing_class.save_pretrained(save_dir)
        logger.info(f"GRPO adapter 저장 완료 (grpo only): {save_dir}")

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None) -> None:
        """체크포인트에서 GRPO adapter 가중치를 in-place copy로 복원한다.

        Mod Record: load_adapter()는 새 Parameter 객체를 생성하여 optimizer의 Parameter 참조가
        끊어지는 문제가 있다. param.data.copy_()로 in-place 업데이트하면 optimizer 참조가 유지된다.
        (pre_stage/trainer.py의 동일한 패턴 참고)

        Args:
            resume_from_checkpoint: 체크포인트 디렉토리 경로.
            model: 복원 대상 모델. None이면 self.model 사용.
        """
        from safetensors.torch import load_file

        ckpt_path = Path(resume_from_checkpoint)
        adapter_file = ckpt_path / "adapter_model.safetensors"

        if not adapter_file.exists():
            logger.warning(f"adapter_model.safetensors 없음: {adapter_file}. 복원 건너뜀.")
            return

        target = model if model is not None else self.model
        raw_model = self.accelerator.unwrap_model(target)

        checkpoint_state = load_file(str(adapter_file), device="cpu")

        restored_count = 0
        for name, param in raw_model.named_parameters():
            if param.requires_grad and name in checkpoint_state:
                param.data.copy_(checkpoint_state[name].to(param.device))
                restored_count += 1

        logger.info(
            f"GRPO adapter Resume 완료: {restored_count}개 파라미터 복원 "
            f"(step={resume_from_checkpoint.split('-')[-1]})"
        )

    # ---------------------------------------------------------------------------
    # 보상함수 생성
    # ---------------------------------------------------------------------------

    def _build_reward_funcs(self) -> list:
        """GRPO reward function callable 리스트를 생성한다.

        각 보상함수는 TRL 시그니처를 따른다:
            func(prompts, completions, completion_ids, **kwargs) → list[float]

        반환 리스트의 순서가 rewards_per_func 열(K) 순서가 됨.

        Returns:
            TRL 호환 callable 리스트.
        """
        funcs = []
        reward_order = [
            "format", "count_total", "count_type",
            "orthogonality", "no_overlap", "connectivity", "spatial",
        ]

        for name in reward_order:
            cfg_item = self.reward_cfg.get(name, {})
            if not cfg_item.get("enabled", True):
                continue
            self._reward_names.append(name)
            self._reward_cfgs_list.append(dict(cfg_item))
            funcs.append(self._make_reward_func(name))

        logger.info(f"활성화된 보상함수 ({len(funcs)}개): {self._reward_names}")
        return funcs

    def _make_reward_func(self, reward_name: str):
        """특정 보상함수명에 대응하는 TRL 호환 callable을 반환한다.

        첫 번째 보상함수 호출 시 compute_all_rewards()로 모든 보상을 일괄 계산하고
        캐싱하여, 이후 호출은 캐시에서 조회한다.
        error_mask는 Trainer 인스턴스의 버퍼에 저장된다.

        Args:
            reward_name: 보상함수 이름.

        Returns:
            TRL reward function callable.
        """

        def reward_func(
            prompts: list,
            completions: list,
            completion_ids: list[list[int]],
            **kwargs: Any,
        ) -> list[float]:
            # 새 step이면 전체 보상 일괄 계산 + 캐싱
            current_step = getattr(self, "_step", 0)
            if current_step != self._reward_cache_step:
                self._compute_and_cache_rewards(completion_ids, kwargs)
                self._reward_cache_step = current_step

            # 캐시에서 해당 보상값 조회
            rewards_list = []
            for i in range(len(completion_ids)):
                cache = self._reward_results_cache.get(i, {})
                rewards_list.append(float(cache.get(reward_name, 0.0)))

            return rewards_list

        reward_func.__name__ = f"reward_{reward_name}"
        return reward_func

    def _compute_and_cache_rewards(
        self,
        completion_ids: list[list[int]],
        kwargs: dict,
    ) -> None:
        """모든 completion에 대해 보상을 일괄 계산하고 버퍼에 저장한다.

        Args:
            completion_ids: completion 토큰 ID 리스트의 리스트 (LOCAL).
            kwargs: reward function kwargs (metadata 포함).
        """
        n = len(completion_ids)

        # 버퍼 초기화
        self._reward_results_cache = {}
        self._error_masks_buffer = [{} for _ in range(n)]

        # metadata 추출 (TRL이 dataset 추가 컬럼을 kwargs로 전달)
        metadata_list: list[dict] = kwargs.get("metadata", [{}] * n)

        for i, ids in enumerate(completion_ids):
            metadata = metadata_list[i] if i < len(metadata_list) else {}

            result = compute_all_rewards(
                token_ids=ids,
                vocab=self.vocab,
                metadata=metadata,
                reward_cfg=self.reward_cfg,
            )

            self._reward_results_cache[i] = result.get("rewards", {})
            self._error_masks_buffer[i] = result.get("error_masks", {})

        # 로컬 completion_ids 캐싱
        self._cached_local_completion_ids = list(completion_ids)

    # ---------------------------------------------------------------------------
    # TRL 메서드 오버라이드
    # ---------------------------------------------------------------------------

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        """rewards_per_func를 gather 후 캐싱하는 오버라이드.

        Args:
            inputs: 로컬 배치 inputs.
            prompts: 프롬프트 리스트.
            completions: completion 문자열 리스트.
            completion_ids_list: completion 토큰 ID 리스트 (LOCAL).

        Returns:
            gather 후 ALL-PROCESS rewards_per_func. shape $(B_{total}, K)$
        """
        # 버퍼 초기화 (새 생성 배치)
        self._reward_results_cache = {}
        self._error_masks_buffer = []
        self._reward_cache_step = -1  # 강제 재계산 트리거

        result = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        self._cached_rewards_per_func = result  # (B_total, K)
        return result

    def _generate_and_score_completions(
        self,
        inputs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """부모 메서드 호출 후 스칼라 advantages를 토큰별로 변환한다.

        Args:
            inputs: 로컬 배치 inputs.

        Returns:
            수정된 출력 딕셔너리. advantages가 (B_local, T) 형태로 교체됨.
        """
        # 부모 호출: generation + 보상 계산 + 스칼라 advantages 획득
        output = super()._generate_and_score_completions(inputs)

        # 스칼라 advantages → 토큰별 advantages 변환
        output = self._apply_token_credit_assignment(output)
        return output

    def _apply_token_credit_assignment(
        self,
        output: dict[str, Any],
    ) -> dict[str, Any]:
        """스칼라 advantages를 토큰별 advantages로 변환한다.

        처리 흐름:
            1. ALL-PROCESS rewards_per_func로 GDPO 정규화 → A_k_all (B_total, K)
            2. process_slice로 로컬 슬라이스 추출 → A_k_local (B_local, K)
            3. compute_token_advantages()로 토큰별 advantages + 배치 정규화

        Args:
            output: _generate_and_score_completions() 반환값.

        Returns:
            advantages가 (B_local, T) 형태로 교체된 output 딕셔너리.
        """
        if self._cached_rewards_per_func is None:
            logger.warning("rewards_per_func 캐시 없음. 스칼라 advantages 유지.")
            return output

        completion_ids = output["completion_ids"]   # (B_local, T)
        completion_mask = output["completion_mask"]  # (B_local, T)
        B_local, T = completion_ids.shape
        device = completion_ids.device

        # 실제 completion 길이 계산
        completion_lengths = [int(l) for l in completion_mask.sum(dim=1).tolist()]

        # ① GDPO: ALL-PROCESS 데이터로 보상별 그룹 정규화
        rewards_all = self._cached_rewards_per_func.to(device)  # (B_total, K)
        eps = float(self.advantage_cfg.get("eps", 1e-8))

        A_k_all = gdpo_group_normalize(
            rewards_per_func=rewards_all,
            num_generations=self.num_generations,
            eps=eps,
        )  # (B_total, K)

        # 로컬 프로세스 슬라이스
        proc_idx = self.accelerator.process_index
        local_start = proc_idx * B_local
        local_end = local_start + B_local

        if local_end > A_k_all.shape[0]:
            logger.warning(
                f"process_slice 범위 초과: [{local_start}:{local_end}] > {A_k_all.shape[0]}. "
                "스칼라 advantages 유지."
            )
            return output

        A_k_local = A_k_all[local_start:local_end]  # (B_local, K)

        # ② + ③ + ④: 토큰별 advantage + 가중합 + 배치 정규화
        token_advantages = compute_token_advantages(
            A_k_local=A_k_local,
            reward_names=self._reward_names,
            reward_cfgs=self._reward_cfgs_list,
            error_masks_batch=self._error_masks_buffer,
            completion_lengths=completion_lengths,
            max_seq_len=T,
            eps=eps,
        )  # (B_local, T)

        # advantages 교체: 스칼라 (B_local,) → 토큰별 (B_local, T)
        # TRL compute_loss는 advantages.dim() == 2이면 unsqueeze하지 않음
        output["advantages"] = token_advantages
        return output
