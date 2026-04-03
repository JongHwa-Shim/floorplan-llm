"""Pre-Stage 훈련 설정 및 Trainer 빌드 모듈.

transformers.Trainer를 기반으로 Pre-Stage 훈련을 구성한다.
PEFT 어댑터(LoRA/DoRA)는 사용하지 않고, PartialEmbedding / PartialLMHead 모듈로
새 토큰 행만 nn.Parameter로 분리하여 optimizer state를 최소화한다.
"""

import logging
import os
from pathlib import Path

import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from torch.utils.data import Dataset

from src.training.pre_stage.collator import PreStageCollator
from src.training.pre_stage.model_loader import (
    PartialEmbedding,
    PartialLMHead,
)

logger = logging.getLogger(__name__)


class PreStageTrainer(Trainer):
    """Pre-Stage 전용 Trainer.

    embed_tokens와 lm_head는 4bit 양자화 대상에서 자동 제외되어 bf16으로 유지되므로
    gradient 기반 학습이 가능하다. 그러나 transformers.Trainer는 PEFT 어댑터 없는
    quantized model을 일괄 거부하므로, 초기화 시 해당 검증을 일시 우회한다.

    체크포인트 저장: partial_state.pt + optimizer/scheduler/trainer_state만 저장.
      model.safetensors는 저장하지 않음 (transformer 레이어는 항상 HF에서 새로 로드,
      new_embed/new_lm_head는 partial_state.pt로 관리).
    체크포인트 로드: partial_state.pt에서 new_embed/new_lm_head만 복원.
      optimizer/scheduler는 Trainer 내부 로직이 자동 처리.

    Attributes:
        _new_token_ids: 새 커스텀 토큰 ID 리스트. partial_state.pt 저장에 사용.
    """

    def __init__(self, *args, new_token_ids: list[int], **kwargs):
        """PreStageTrainer 초기화.

        Args:
            *args: Trainer에 전달될 위치 인자.
            new_token_ids: 새 커스텀 토큰 ID 리스트. partial_state.pt 저장에 사용.
            **kwargs: Trainer에 전달될 키워드 인자.
        """
        self._new_token_ids = new_token_ids

        # Mod Record: Trainer.__init__가 validate_quantization_for_training()을 호출해
        # "adapter 없는 quantized model은 학습 불가" 에러를 발생시킨다.
        # Pre-Stage에서 embed_tokens/lm_head는 비양자화(bf16) 레이어이므로
        # adapter 없이도 훈련 가능하다. 검증 함수를 일시 비활성화하여 우회한다.
        # Mod Record: trainer.py가 validate_quantization_for_training을 from-import로
        # 가져오므로 trainer_utils를 패치해도 무효하다. trainer 모듈 네임스페이스를 직접 패치한다.
        import transformers.trainer as _trainer_module

        _orig_validate = _trainer_module.validate_quantization_for_training
        _trainer_module.validate_quantization_for_training = lambda _model: None
        try:
            super().__init__(*args, **kwargs)
        finally:
            _trainer_module.validate_quantization_for_training = _orig_validate

    def _save_checkpoint(self, model, trial, metrics=None):
        """체크포인트 저장: partial_state.pt + optimizer/scheduler/trainer_state.

        저장 흐름:
        1. partial_state.pt 저장 (현재 훈련된 new_embed/new_lm_head 값)
        2. super()._save_checkpoint: optimizer + scheduler + trainer_state 저장
           (model.safetensors는 저장 안 함 → optimizer Parameter 참조 유지)

        Mod Record: 기존에는 merge_and_restore() → super() → _setup_partial_training() 순서로
        실행했다. 이 방식의 문제: merge_and_restore가 nn.Parameter 객체를 소멸시키고
        _setup_partial_training이 새 객체를 생성하면서 optimizer의 Parameter 참조가 끊어진다.
        이후 optimizer.step()이 소멸된 객체를 업데이트하려 해도 grad=None이므로 no-op이 되어
        체크포인트 저장 이후의 훈련이 완전히 무효가 되는 버그가 있었다.
        수정: merge_and_restore/_setup_partial_training을 중간 체크포인트에서 제거.
        model.safetensors는 저장하지 않고 partial_state.pt만으로 resume을 지원한다.
        merge_and_restore는 최종 저장(run_pre_stage.py)에서만 호출한다.

        Args:
            model: 현재 훈련 중인 모델 (PartialEmbedding/PartialLMHead 적용 상태).
            trial: Optuna/Ray 하이퍼파라미터 탐색 trial (일반 훈련 시 None).
            metrics: 현재 step의 평가 지표 딕셔너리.
        """
        # 1. 체크포인트 저장 경로 계산 (Trainer 내부 로직과 동일)
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)

        # 2. partial_state.pt 저장: 현재 훈련된 new_embed/new_lm_head 값 보존
        # DDP에서는 model이 DistributedDataParallel로 래핑되므로 .module로 실제 모델에 접근
        raw_model = model.module if hasattr(model, "module") else model
        embed = raw_model.model.embed_tokens
        lm_head = raw_model.lm_head
        if isinstance(embed, PartialEmbedding) and isinstance(lm_head, PartialLMHead):
            # DDP에서는 모든 rank가 _save_checkpoint를 호출하므로 rank 0만 저장
            if self.is_world_process_zero():
                torch.save(
                    {
                        "new_embed": embed.new_embed.data.cpu(),
                        "new_lm_head": lm_head.new_lm_head.data.cpu(),
                        "new_token_ids": self._new_token_ids,
                    },
                    os.path.join(output_dir, "partial_state.pt"),
                )
                logger.info(f"partial_state.pt 저장 완료: {output_dir}")

        # 3. optimizer/scheduler/trainer_state 저장 (model.safetensors는 저장 안 함)
        # self.save_model을 일시 no-op으로 교체하여 모델 가중치 저장만 건너뜀
        # transformer 레이어는 항상 HuggingFace에서 새로 로드하므로 저장 불필요
        _orig_save_model = self.save_model
        self.save_model = lambda *args, **kwargs: None
        try:
            super()._save_checkpoint(model, trial)
        finally:
            self.save_model = _orig_save_model

        logger.info("체크포인트 저장 완료 (partial_state.pt + optimizer/scheduler/trainer_state)")

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None):
        """체크포인트에서 new_embed/new_lm_head 복원.

        중간 체크포인트에는 model.safetensors가 없으므로 super()를 호출하지 않는다.
        transformer 레이어는 load_model_and_tokenizer()에서 HuggingFace로부터 이미 로드됨.
        optimizer/scheduler는 Trainer 내부 _load_optimizer_and_scheduler()가 자동 처리.

        Args:
            resume_from_checkpoint: 체크포인트 디렉토리 경로.
            model: 복원 대상 모델. None이면 self.model 사용.

        Raises:
            없음. partial_state.pt 부재 시 경고 후 계속.
        """
        partial_state_path = Path(resume_from_checkpoint) / "partial_state.pt"
        if not partial_state_path.exists():
            logger.warning(
                f"partial_state.pt 없음: {partial_state_path} — new_embed/new_lm_head가 "
                "초기값으로 유지됩니다."
            )
            return

        target_model = model if model is not None else self.model
        # DDP 언래핑: DistributedDataParallel 래퍼 안쪽의 실제 모델 접근
        target_model = target_model.module if hasattr(target_model, "module") else target_model
        embed = target_model.model.embed_tokens
        lm_head = target_model.lm_head

        if not isinstance(embed, PartialEmbedding) or not isinstance(lm_head, PartialLMHead):
            logger.warning("embed_tokens 또는 lm_head가 Partial 모듈이 아님. partial_state 로드 건너뜀.")
            return

        partial_state = torch.load(
            partial_state_path, map_location="cpu", weights_only=True
        )
        embed.new_embed.data.copy_(partial_state["new_embed"].to(embed.new_embed.device))
        lm_head.new_lm_head.data.copy_(partial_state["new_lm_head"].to(lm_head.new_lm_head.device))
        logger.info(
            f"Resume: partial_state.pt 복원 완료 (step={resume_from_checkpoint.split('-')[-1]})"
        )

    def _load_best_model(self):
        """최고 성능 체크포인트에서 partial_state.pt로 new_embed/new_lm_head 복원.

        Trainer 기본 구현은 model.load_state_dict()로 전체 모델을 로드하지만,
        PartialEmbedding 구조에서는 key mismatch가 발생한다.
        대신 partial_state.pt에서 new_embed/new_lm_head만 직접 복사한다.

        Args:
            없음.

        Returns:
            없음.
        """
        best_ckpt = self.state.best_model_checkpoint
        if best_ckpt is None:
            logger.warning("best_model_checkpoint가 없어 최고 모델 로드를 건너뜁니다.")
            return

        partial_state_path = Path(best_ckpt) / "partial_state.pt"
        if not partial_state_path.exists():
            logger.warning(f"partial_state.pt 없음: {partial_state_path}, 로드 건너뜀")
            return

        partial_state = torch.load(
            partial_state_path, map_location="cpu", weights_only=True
        )
        # DDP 언래핑: DistributedDataParallel 래퍼 안쪽의 실제 모델 접근
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        embed = raw_model.model.embed_tokens
        lm_head = raw_model.lm_head

        if isinstance(embed, PartialEmbedding):
            embed.new_embed.data.copy_(
                partial_state["new_embed"].to(embed.new_embed.device)
            )
        if isinstance(lm_head, PartialLMHead):
            lm_head.new_lm_head.data.copy_(
                partial_state["new_lm_head"].to(lm_head.new_lm_head.device)
            )
        logger.info(f"최고 모델 partial_state 로드 완료: {best_ckpt}")


def build_training_arguments(cfg: DictConfig) -> TrainingArguments:
    """Hydra config로부터 TrainingArguments를 생성한다.

    Args:
        cfg: Hydra DictConfig. cfg.training 섹션을 참조한다.

    Returns:
        설정된 TrainingArguments 인스턴스.
    """
    train_cfg = cfg.training

    # max_steps가 0 이하면 num_train_epochs 기반으로 학습 (Trainer 기본 동작)
    max_steps = int(train_cfg.get("max_steps", 0))

    os.environ["WANDB_PROJECT"] = train_cfg.project_name
    kwargs = dict(
        output_dir=train_cfg.output_dir,
        num_train_epochs=train_cfg.num_train_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        warmup_ratio=train_cfg.warmup_ratio,
        weight_decay=train_cfg.weight_decay,
        # 혼합 정밀도 (AMP): forward/backward bf16, optimizer state fp32
        bf16=train_cfg.bf16,
        dataloader_num_workers=train_cfg.dataloader_num_workers,
        save_strategy=train_cfg.save_strategy,
        eval_strategy=train_cfg.eval_strategy,
        logging_steps=train_cfg.logging_steps,
        report_to=train_cfg.report_to,
        run_name=train_cfg.run_name,
        seed=train_cfg.seed,
        save_total_limit=train_cfg.get("save_total_limit", 3),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # gradient checkpointing은 model_loader에서 prepare_model_for_kbit_training으로 이미 활성화
        gradient_checkpointing=False,
        # 데이터셋에 label이 포함되어 있으므로 label_names 명시
        label_names=["labels"],
        # DoRA adapter + PartialEmbedding/PartialLMHead 파라미터는 매 forward에서 gradient 수신
        # DDP unused parameter 탐지를 비활성화하여 불필요한 오버헤드 제거
        ddp_find_unused_parameters=False,
    )

    if max_steps > 0:
        kwargs["max_steps"] = max_steps

    return TrainingArguments(**kwargs)


def build_trainer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    cfg: DictConfig,
    new_token_ids: list[int],
) -> PreStageTrainer:
    """Pre-Stage Trainer를 생성한다.

    Args:
        model: PartialEmbedding / PartialLMHead가 적용된 AutoModelForCausalLM.
        tokenizer: 커스텀 토큰이 포함된 AutoTokenizer.
        train_dataset: 훈련용 PreStageDataset.
        eval_dataset: 검증용 PreStageDataset.
        cfg: Hydra DictConfig. cfg.training, cfg.data 섹션 참조.
        new_token_ids: 새 커스텀 토큰 ID 리스트.
            체크포인트 저장 후 PartialEmbedding 재적용에 사용.

    Returns:
        설정된 PreStageTrainer 인스턴스.
    """
    training_args = build_training_arguments(cfg)
    collator = PreStageCollator(tokenizer=tokenizer, max_length=cfg.data.max_length)

    trainer = PreStageTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        new_token_ids=new_token_ids,
    )

    logger.info("Trainer 생성 완료")
    logger.info(f"  훈련 샘플 수: {len(train_dataset)}")
    logger.info(f"  검증 샘플 수: {len(eval_dataset)}")
    logger.info(f"  실효 배치 크기: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"  출력 디렉토리: {training_args.output_dir}")

    return trainer
