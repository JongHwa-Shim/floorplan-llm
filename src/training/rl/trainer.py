"""RLTrainer 모듈.

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

    vLLM use_cache 핵:
       - prepare_model_for_kbit_training(use_gradient_checkpointing=True)가
         model.config.use_cache=False를 세팅한다.
       - HF generate() 경로에서는 KV cache가 필수이므로 생성 구간에서만 True 복원.
       - vLLM 경로(use_vllm=True)에서는 vLLM이 별도로 inference하므로 핵 불필요.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from trl import GRPOTrainer

from src.training.augmentation.tokenizer import Vocab
from src.training.rl.rewards import compute_all_rewards
from src.training.rl.advantage import gdpo_group_normalize, compute_token_advantages

logger = logging.getLogger(__name__)


class RLTrainer(GRPOTrainer):
    """GDPO + 토큰 수준 신용할당 Trainer.

    TRL GRPOTrainer를 확장하여 다음을 구현한다:
        1. 7개 Rule-based 보상함수 (format/count/geometry/connectivity/spatial)
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

        # Mod Record: TRL GRPOTrainer.__init__은 beta != 0.0일 때 ref adapter를 추가한 후
        # `if ".default." in name: ref_param.copy_(default_param)`으로 가중치를 복사한다.
        # 우리 멀티 어댑터(sft/rl) 구조에는 .default. 이름의 파라미터가 없어 TRL의 복사 루프가
        # 0건 매칭되고, ref adapter는 add_adapter()의 무작위 초기화 상태로 남는다.
        # 이대로 KL을 계산하면 reference policy가 의미 없는 분포가 되므로, SFT lora 가중치를
        # ref adapter로 직접 복사하여 reference = base + SFT 정책으로 정확히 정의한다.
        # 동시에 ref adapter 파라미터를 frozen 처리하고, 활성 adapter를 [sft, rl]로 재설정해
        # forward 시 ref adapter가 정책 계산에 끼어들지 않도록 한다.
        if hasattr(self.model, "peft_config") and "ref" in self.model.peft_config:
            raw_model = self.accelerator.unwrap_model(self.model)
            copied = 0
            for name, param in raw_model.named_parameters():
                if ".sft." in name and ("lora_A" in name or "lora_B" in name):
                    ref_name = name.replace(".sft.", ".ref.")
                    try:
                        ref_param = raw_model.get_parameter(ref_name)
                    except AttributeError:
                        continue
                    ref_param.data.copy_(param.data)
                    copied += 1
            # ref adapter 전체 frozen
            for name, param in raw_model.named_parameters():
                if ".ref." in name:
                    param.requires_grad_(False)
            # 활성 어댑터를 [sft, rl]로 재설정 (ref는 KL 계산 시 TRL이 임시 활성화)
            try:
                raw_model.base_model.set_adapter(["sft", "rl"])
                # set_adapter가 SFT를 다시 trainable로 만드는 경우 재차 frozen 강제
                for name, param in raw_model.named_parameters():
                    if ".sft." in name:
                        param.requires_grad_(False)
            except Exception as e:
                logger.warning(f"활성 어댑터 재설정 실패 (무시 가능): {e}")
            logger.info(
                f"TRL ref adapter 가중치 SFT에서 복사 완료: {copied}개 lora 텐서 "
                "(reference policy = base + SFT)"
            )

    # ---------------------------------------------------------------------------
    # 저장 / 로드 오버라이드
    # ---------------------------------------------------------------------------

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        """RL adapter만 저장한다 (SFT adapter와 base model 제외).

        Mod Record: 이전 구조에서는 merge_dora_and_save()로 전체 모델을 저장했음.
        새 구조에서는 RL adapter(adapter_name="rl")만 저장하여 다음 단계(추론/분석)에서
        멀티 어댑터 스태킹으로 복원한다.

        Mod Record (어댑터 평탄화): PEFT의 PeftModel.save_pretrained는 adapter_name이
        "default"가 아닌 경우 항상 ``{save_directory}/{adapter_name}/`` 서브디렉토리에 저장
        하는 표준 컨벤션을 갖는다 (peft/peft_model.py L299 참고). 그 결과 우리의 RL 체크포인트는
        ``checkpoint-{step}/rl/adapter_model.safetensors`` 형태가 되어 (a) optimizer/scheduler/
        trainer_state 등 Trainer가 root에 직접 저장하는 파일들과 디렉토리 레벨이 어긋나고
        (b) ``_load_from_checkpoint``가 root를 가정해 읽으므로 Resume이 동작하지 않는 잠재 버그가
        있었다. 평탄화 후에는 SFT 체크포인트와 동일한 단일-어댑터 root 구조가 되며, PEFT API에
        의존하지 않는 우리 자체 Resume 로직과도 자연스럽게 정합한다.

        Args:
            output_dir: 저장 대상 디렉토리. None이면 TrainingArguments.output_dir 사용.
            _internal_call: Trainer 내부 호출 여부 (무시).
        """
        save_dir = Path(output_dir or self.args.output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # accelerator로 DDP 래퍼 언래핑
        raw_model = self.accelerator.unwrap_model(self.model)
        # RL adapter만 선택적 저장 (PEFT는 save_dir/rl/ 서브디렉토리에 저장)
        # Mod Record (체크포인트 비대화 수정): save_embedding_layers=False로 embed_tokens/lm_head
        # 저장을 억제한다. PEFT는 vocab resize를 감지하면 기본값("auto")으로 이 두 레이어(~2.18GB)를
        # 어댑터에 함께 저장하는데, RL에서는 frozen이고 로드 시 항상 partial_state.pt에서 주입되므로
        # 순전히 중복(저장본과 partial_state.pt 값이 byte-identical 확인)이다. 억제 시 체크포인트가
        # ~2.34GB → ~161MB로 축소되며, 로드/추론/Resume은 partial_state.pt 주입에 의존하므로 무영향.
        raw_model.save_pretrained(
            str(save_dir), selected_adapters=["rl"], save_embedding_layers=False
        )
        if self.processing_class is not None:
            self.processing_class.save_pretrained(str(save_dir))

        # PEFT가 만든 save_dir/rl/ 서브디렉토리의 내용을 root로 평탄화. 단일 어댑터만 저장하므로
        # SFT 체크포인트와 동일한 root 구조가 되며, _load_from_checkpoint의 root 경로 가정과 일치.
        if self.accelerator.is_main_process:
            self._flatten_adapter_subdir(save_dir, adapter_name="rl")

        logger.info(f"RL adapter 저장 완료 (rl only, flattened): {save_dir}")

    @staticmethod
    def _flatten_adapter_subdir(save_dir: Path, adapter_name: str) -> None:
        """PEFT가 생성한 ``save_dir/{adapter_name}/`` 내용을 ``save_dir/`` 루트로 옮긴다.

        루트에 같은 이름 파일이 이미 있으면 (예: PEFT가 만든 README.md vs Trainer가 만든 README.md)
        루트 쪽을 어댑터 본체로 덮어쓴다 — adapter_model.safetensors / adapter_config.json은
        반드시 PEFT가 새로 쓴 것이어야 하기 때문이다. 이동 후 빈 서브디렉토리를 삭제한다.
        호출자는 main process 가드 하에서만 호출해야 한다 (DDP 환경 안전).

        Args:
            save_dir: 체크포인트 root 디렉토리.
            adapter_name: 평탄화 대상 어댑터 이름.
        """
        sub_dir = save_dir / adapter_name
        if not sub_dir.is_dir():
            return

        for entry in sub_dir.iterdir():
            target = save_dir / entry.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(entry), str(target))

        sub_dir.rmdir()

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None) -> None:
        """체크포인트에서 RL adapter 가중치를 in-place copy로 복원한다.

        Mod Record: load_adapter()는 새 Parameter 객체를 생성하여 optimizer의 Parameter 참조가
        끊어지는 문제가 있다. param.data.copy_()로 in-place 업데이트하면 optimizer 참조가 유지된다.

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
            f"RL adapter Resume 완료: {restored_count}개 파라미터 복원 "
            f"(step={resume_from_checkpoint.split('-')[-1]})"
        )

    # ---------------------------------------------------------------------------
    # 보상함수 생성
    # ---------------------------------------------------------------------------

    def _build_reward_funcs(self) -> list:
        """RL reward function callable 리스트를 생성한다.

        각 보상함수는 TRL 시그니처를 따른다:
            func(prompts, completions, completion_ids, **kwargs) → list[float]

        반환 리스트의 순서가 rewards_per_func 열(K) 순서가 됨.

        Returns:
            TRL 호환 callable 리스트.
        """
        funcs = []
        reward_order = [
            "format", "count_total", "count_type",
            "orthogonality", "no_overlap",
            "room_in_outline", "outline_in_room", "coverage",
            "connectivity", "spatial", "input_consistency",
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

        Mod Record: prepare_model_for_kbit_training(use_gradient_checkpointing=True)가
        내부적으로 model.config.use_cache=False를 세팅한다. gradient checkpointing과
        KV cache는 훈련 backward에서 함께 쓸 수 없으므로 이는 정상 동작이다.
        그러나 rollout generation은 autoregressive 순차 생성이므로 KV cache가 필수다.
        use_cache=False인 채로 generate()하면 O(n²) 연산이 발생해 극단적으로 느려진다.
        따라서 HF generate() 경로에서만 생성 구간에 use_cache=True로 복원하고, 이후 되돌린다.
        vLLM 경로(use_vllm=True)에서는 vLLM이 독립 inference하므로 이 핵이 불필요하다.

        Args:
            inputs: 로컬 배치 inputs.

        Returns:
            수정된 출력 딕셔너리. advantages가 (B_local, T) 형태로 교체됨.
        """
        if not self.use_vllm:
            self.model.config.use_cache = True
        try:
            output = super()._generate_and_score_completions(inputs)
        finally:
            if not self.use_vllm:
                # 훈련 backward pass를 위해 반드시 복원 (예외 발생 시에도)
                self.model.config.use_cache = False

        # Mod Record (RL adapter gradient 누락 수정): TRL은 beta != 0일 때 매
        # _generate_and_score_completions 내부에서 use_adapter(model, "ref")로 reference
        # logp를 계산하는데, 그 컨텍스트가 종료될 때 model.set_adapter(previous_adapter)로
        # 복원한다. 이때 previous_adapter는 PeftModel 인스턴스 속성 active_adapter(단일 문자열
        # "sft")이므로, 우리가 base_model.set_adapter(["sft","rl"])로 설정한 다중 활성 상태가
        # "sft" 단독으로 덮어써진다. 결과적으로 RL 어댑터가 비활성화 + requires_grad=False가 되어
        # 이후 compute_loss forward에서 gradient가 흐르지 않고 lora_B가 0 초기값에 고정됐다.
        # 매 generation 배치 직후 [sft, rl] 활성 상태를 재확립하여 학습 forward가 항상 RL
        # 어댑터를 포함하도록 보장한다. (use_adapter는 단일 어댑터만 복원 가능해 라이브러리
        # 레벨에서 다중 활성 상태를 되돌릴 수 없으므로 서브클래스에서 재확립해야 함)
        self._reassert_active_adapters()

        # 스칼라 advantages → 토큰별 advantages 변환
        output = self._apply_token_credit_assignment(output)
        return output

    def _reassert_active_adapters(self) -> None:
        """활성 어댑터를 [sft, rl]로, 학습 대상을 rl로 재확립한다.

        TRL의 reference logp 계산(use_adapter)이 활성 어댑터를 stale 단일 "sft"로 되돌려
        RL 어댑터를 비활성화 + frozen 시키는 문제를 매 generation 배치 직후 복구한다.
        sft/ref 어댑터는 frozen을 유지하고 rl 어댑터만 trainable로 남긴다.

        Note:
            requires_grad_()는 동일 Parameter 객체의 플래그만 in-place 토글하므로, 이미 생성된
            optimizer의 파라미터 참조를 끊지 않는다. beta == 0(ref 어댑터 미생성) 케이스에서도
            ".ref." 매칭이 0건이라 안전하게 동작한다.
        """
        raw_model = self.accelerator.unwrap_model(self.model)
        try:
            raw_model.base_model.set_adapter(["sft", "rl"])
        except Exception as e:  # 어댑터 구성이 예상과 다를 경우 학습을 중단시키지 않고 경고만
            logger.warning(f"활성 어댑터 재확립 실패 (무시 가능): {e}")
            return
        for name, param in raw_model.named_parameters():
            # sft/ref는 frozen 유지, rl만 trainable
            if ".sft." in name or ".ref." in name:
                param.requires_grad_(False)

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
            use_gdpo_normalization=bool(
                self.advantage_cfg.get("use_gdpo_normalization", True)
            ),
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

        # 전역 신용할당 토글 설정
        use_token_credit_assignment = bool(
            self.advantage_cfg.get("use_token_credit_assignment", True)
        )

        # ② + ③ + ④: 토큰별 advantage + 가중합 + 배치 정규화
        token_advantages = compute_token_advantages(
            A_k_local=A_k_local,
            reward_names=self._reward_names,
            reward_cfgs=self._reward_cfgs_list,
            error_masks_batch=self._error_masks_buffer,
            completion_lengths=completion_lengths,
            max_seq_len=T,
            eps=eps,
            use_token_credit_assignment=use_token_credit_assignment,
        )  # (B_local, T)

        # advantages 교체: 스칼라 (B_local,) → 토큰별 (B_local, T)
        # TRL compute_loss는 advantages.dim() == 2이면 unsqueeze하지 않음
        output["advantages"] = token_advantages
        return output
