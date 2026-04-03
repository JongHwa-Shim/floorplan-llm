"""SFT 훈련 설정 및 Trainer 빌드 모듈.

표준 transformers.Trainer를 기반으로 SFT 훈련을 구성한다.
PEFT DoRA adapter가 정식 지원되므로 Pre-Stage와 달리 Trainer 패치가 필요 없다.
"""

import logging
import os

from omegaconf import DictConfig
from peft import PeftModel
from transformers import AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset

from src.training.pre_stage.collator import PreStageCollator

logger = logging.getLogger(__name__)


def build_training_arguments(cfg: DictConfig) -> TrainingArguments:
    """Hydra config로부터 SFT용 TrainingArguments를 생성한다.

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
        # PEFT DoRA: prepare_model_for_kbit_training이 이미 gradient checkpointing을 활성화하므로
        # TrainingArguments에서도 명시적으로 true 설정
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # 데이터셋에 label이 포함되어 있으므로 label_names 명시
        label_names=["labels"],
        # DoRA adapter 파라미터는 매 forward에서 gradient 수신
        # DDP unused parameter 탐지를 비활성화하여 불필요한 오버헤드 제거
        ddp_find_unused_parameters=False,
    )

    if max_steps > 0:
        kwargs["max_steps"] = max_steps

    return TrainingArguments(**kwargs)


def build_trainer(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    cfg: DictConfig,
) -> Trainer:
    """SFT Trainer를 생성한다.

    PEFT DoRA adapter는 transformers.Trainer에서 정식 지원되므로
    Pre-Stage의 PreStageTrainer와 달리 표준 Trainer를 사용한다.
    DataCollator는 데이터 포맷이 동일한 PreStageCollator를 재활용한다.

    Args:
        model: DoRA adapter가 적용된 PeftModelForCausalLM.
        tokenizer: 커스텀 토큰이 포함된 AutoTokenizer.
        train_dataset: 훈련용 Dataset (SFTDataset).
        eval_dataset: 검증용 Dataset (SFTDataset).
        cfg: Hydra DictConfig. cfg.training, cfg.data 섹션 참조.

    Returns:
        설정된 Trainer 인스턴스.
    """
    training_args = build_training_arguments(cfg)
    collator = PreStageCollator(tokenizer=tokenizer, max_length=cfg.data.max_length)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    logger.info("SFT Trainer 생성 완료")
    logger.info(f"  훈련 샘플 수: {len(train_dataset)}")
    logger.info(f"  검증 샘플 수: {len(eval_dataset)}")
    logger.info(
        f"  실효 배치 크기: "
        f"{training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}"
    )
    logger.info(f"  출력 디렉토리: {training_args.output_dir}")

    return trainer
