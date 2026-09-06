"""SFT 훈련 설정 및 Trainer 빌드 모듈.

표준 transformers.Trainer를 기반으로 SFT 훈련을 구성한다.
PEFT DoRA adapter가 정식 지원되므로 Embedding Alignment와 달리 Trainer 패치가 필요 없다.
"""

import logging
import os

import torch
from omegaconf import DictConfig
from peft import PeftModel
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer import TRAINING_ARGS_NAME
from torch.utils.data import Dataset

from src.training.embed_align.collator import EmbedAlignCollator

logger = logging.getLogger(__name__)


class SFTAdapterTrainer(Trainer):
    """embed_tokens/lm_head 저장을 억제하는 SFT용 Trainer.

    표준 ``Trainer._save``는 PEFT 모델을 ``self.model.save_pretrained(output_dir, state_dict=...)``로
    저장하는데 ``save_embedding_layers``를 지정하지 않아, PEFT 기본값 ``"auto"``가 vocab resize를 감지해
    resize된 ``embed_tokens``/``lm_head``(~4.36GB, F32)를 어댑터에 함께 저장한다. 이 두 레이어는 SFT에서
    frozen이고 로드 시 항상 ``partial_state.pt``에서 주입되므로 순전히 중복이며(저장본과 partial_state.pt
    값이 byte-identical), 어댑터 크기를 ~14배 부풀린다(323MB 어댑터 → 4.69GB 파일).

    ``_save``를 오버라이드해 ``save_embedding_layers=False``로 순수 LoRA 어댑터만 저장한다. main-process
    가드/FSDP·DeepSpeed 분기는 상위 ``save_model``에 그대로 있으므로 여기서는 PEFT 저장 분기만 재현한다
    (transformers 5.6 ``Trainer._save`` PEFT 분기 + 플래그 추가). 로드/추론/Resume은 partial_state.pt
    주입에 의존하므로 무영향.
    """

    def _save(self, output_dir: str | None = None, state_dict: dict | None = None) -> None:
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # SFT 모델은 항상 PeftModel이므로 표준 _save의 PEFT 분기만 재현 + save_embedding_layers=False
        self.model.save_pretrained(
            output_dir, state_dict=state_dict, save_embedding_layers=False
        )
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


def _parse_save_total_limit(value: int | str | None) -> int | None:
    """save_total_limit 설정값을 안전하게 파싱한다.

    OmegaConf가 YAML null을 Python None 대신 문자열 "null"로 전달하는 경우가 있어
    명시적으로 None으로 변환한다. None이면 rotate_checkpoints가 제한 없이 전체 보존.

    Args:
        value: config에서 읽은 원시값 (None, int, 또는 "null" 문자열).

    Returns:
        None (제한 없음) 또는 int (최대 보존 수).
    """
    if value is None or value == "null":
        return None
    return int(value)


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
        # Mod Record: OmegaConf에서 null이 Python None 대신 문자열 "null"로 전달되는 경우가 있어
        # rotate_checkpoints의 "null" <= 0 비교에서 TypeError 발생. 명시적으로 None으로 변환.
        save_total_limit=_parse_save_total_limit(train_cfg.get("save_total_limit", None)),
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
        # Mod Record: PyTorch 2.10 + bitsandbytes NF4에서 DDP 초기화 시 broadcast_buffers=True(기본값)이면
        # quantized weight buffer까지 NCCL로 rank간 브로드캐스트하여 ~수GB 임시 메모리 할당 → OOM.
        # NF4 buffer는 rank간 동기화 불필요(각 rank가 동일하게 로드)하므로 False로 비활성화.
        ddp_broadcast_buffers=False,
        # Mod Record: paged_adamw_32bit은 momentum/variance fp32 텐서를 CPU RAM에 페이징하여
        # GPU 메모리를 절약한다. LoRA trainable params 80M 기준 ~640MB GPU 절약. 디폴트는
        # 기존 adamw_torch로 두어 외부 호환성 보존, config에서 paged_adamw_32bit로 변경 가능.
        optim=train_cfg.get("optim", "adamw_torch"),
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
    Embedding Alignment의 EmbedAlignTrainer와 달리 표준 Trainer를 사용한다.
    DataCollator는 데이터 포맷이 동일한 EmbedAlignCollator를 재활용한다.

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
    collator = EmbedAlignCollator(tokenizer=tokenizer, max_length=cfg.data.max_length)

    trainer = SFTAdapterTrainer(
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
