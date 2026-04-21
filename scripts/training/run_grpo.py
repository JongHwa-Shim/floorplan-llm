"""GRPO(Group Relative Policy Optimization) 훈련 실행 스크립트.

SFT final 모델에 GDPO + 토큰 수준 신용할당 강화학습을 적용한다.
Rule-based RLVR 보상함수로 6가지 보상을 사용한다.

사용법:
    # 기본 실행 (config/training/grpo/pipeline.yaml 사용)
    uv run python scripts/training/run_grpo.py

    # DDP 멀티 GPU (2개)
    uv run torchrun --nproc_per_node=2 scripts/training/run_grpo.py

    # 하이퍼파라미터 오버라이드
    uv run python scripts/training/run_grpo.py training.learning_rate=5e-6

    # 디버그 모드 (10 step만 실행)
    uv run python scripts/training/run_grpo.py training.max_steps=10 training.report_to=none

    # W&B 비활성화
    uv run python scripts/training/run_grpo.py training.report_to=none

    # Resume
    uv run python scripts/training/run_grpo.py resume.enabled=true
"""

import logging
import os
import random
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

# 프로젝트 루트를 sys.path에 추가 (패키지 임포트 보장)
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.training.grpo import (
    GDPOTrainer,
    GRPOPromptDataset,
    load_model_and_tokenizer,
)
from src.training.augmentation.tokenizer import load_vocab

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """재현성을 위해 모든 랜덤 시드를 고정한다.

    Args:
        seed: 고정할 시드 값.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_checkpoint(cfg: DictConfig) -> str | None:
    """Resume 설정으로부터 사용할 체크포인트 경로를 결정한다.

    Args:
        cfg: Hydra DictConfig. cfg.resume 섹션을 참조한다.

    Returns:
        체크포인트 경로 문자열. Resume 불필요 또는 체크포인트 없으면 None.

    Raises:
        FileNotFoundError: checkpoint_path가 지정됐으나 해당 경로가 없을 경우.
    """
    resume_cfg = cfg.get("resume", {})

    if not resume_cfg.get("enabled", False):
        return None

    checkpoint_path = resume_cfg.get("checkpoint_path")
    if checkpoint_path:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"지정된 체크포인트를 찾을 수 없음: {path}")
        logger.info(f"지정된 체크포인트로 Resume: {path}")
        return str(path)

    if resume_cfg.get("auto_find_latest", True):
        output_dir = Path(cfg.training.output_dir)
        if not output_dir.exists():
            logger.warning(f"output_dir가 없어 Resume 불가: {output_dir}. 처음부터 시작합니다.")
            return None

        checkpoints = sorted(
            [p for p in output_dir.glob("checkpoint-*") if p.is_dir()],
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if checkpoints:
            latest = checkpoints[-1]
            logger.info(f"최신 체크포인트 자동 탐색 성공: {latest}")
            return str(latest)

        logger.warning("체크포인트를 찾을 수 없음. 처음부터 시작합니다.")

    return None


@hydra.main(
    config_path=os.path.join(_PROJECT_ROOT, "config", "training", "grpo"),
    config_name="pipeline",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    """GRPO 훈련 메인 함수.

    Args:
        cfg: Hydra가 주입하는 DictConfig.
    """
    logger.info("=== GRPO 훈련 시작 ===")

    # DDP 재시작: nproc_per_node > 1이고 아직 torchrun 하위 프로세스가 아니면
    # os.execvp로 현재 프로세스를 torchrun으로 교체
    nproc = cfg.get("distributed", {}).get("nproc_per_node", 1)
    if nproc > 1 and "LOCAL_RANK" not in os.environ:
        cmd = ["torchrun", f"--nproc_per_node={nproc}"] + sys.argv
        logger.info(f"DDP 모드: torchrun으로 재시작 (nproc_per_node={nproc})")
        os.execvp("torchrun", cmd)

    # 증강 설정 로드 후 cfg.augmentation으로 병합
    aug_config_path = Path(_PROJECT_ROOT) / cfg.data.aug_pipeline_config
    aug_cfg = OmegaConf.load(aug_config_path)
    OmegaConf.set_struct(cfg, False)
    OmegaConf.update(cfg, "augmentation", aug_cfg, merge=True)
    OmegaConf.set_struct(cfg, True)
    logger.info(f"증강 설정 로드: {aug_config_path}")

    logger.info(f"설정:\n{OmegaConf.to_yaml(cfg)}")

    # 재현성 시드 고정
    seed: int = cfg.training.seed
    set_seed(seed)
    logger.info(f"시드 고정: {seed}")

    # Resume 체크포인트 경로 결정
    resume_checkpoint = _resolve_checkpoint(cfg)

    # 모델 + 토크나이저 로드 (로컬 sft/final + LoRA 멀티 어댑터 스태킹)
    logger.info("모델 및 토크나이저 로드 중...")
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Vocab 로드 (파서 및 데이터셋에서 필요)
    vocab_path = Path(_PROJECT_ROOT) / cfg.model.vocab_extension
    vocab = load_vocab(vocab_path, cfg.model.tokenizer_dir)
    logger.info(f"Vocab 로드 완료: {len(vocab.token_to_id)}개 커스텀 토큰")

    # GRPO 프롬프트 데이터셋 로드 (output 없이 prompt + metadata만)
    logger.info("데이터셋 로드 중...")
    train_dataset = GRPOPromptDataset(cfg, tokenizer, split="train", seed=seed)
    logger.info(f"훈련 데이터셋 크기: {len(train_dataset)}")

    # W&B 프로젝트 설정
    os.environ["WANDB_PROJECT"] = cfg.training.project_name

    # GRPOConfig 생성
    from trl import GRPOConfig

    max_steps = int(cfg.training.get("max_steps", 0))
    grpo_config_kwargs = dict(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        bf16=cfg.training.bf16,
        logging_steps=cfg.training.logging_steps,
        save_strategy=cfg.training.save_strategy,
        save_steps=cfg.training.save_steps,
        report_to=cfg.training.report_to,
        run_name=cfg.training.run_name,
        seed=cfg.training.seed,
        save_total_limit=cfg.training.get("save_total_limit", 3),
        # GRPO 설정
        num_generations=cfg.grpo.num_generations,
        temperature=cfg.grpo.temperature,
        top_p=cfg.grpo.top_p,
        beta=cfg.grpo.kl_coeff,                   # KL 페널티 계수
        epsilon=cfg.grpo.clip_range,              # PPO 클리핑 엡실론
        num_iterations=cfg.grpo.num_iterations,   # roll-out 재활용 횟수
        max_completion_length=cfg.data.max_completion_length,
        # GDPO는 GDPOTrainer에서 직접 처리
        # TRL의 기본 normalize_then_sum 모드 사용 (GDPOTrainer에서 오버라이드)
        multi_objective_aggregation="normalize_then_sum",
        # 그래디언트 체크포인팅
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        optim=cfg.training.get("optim", "paged_adamw_32bit"),
    )

    if max_steps > 0:
        grpo_config_kwargs["max_steps"] = max_steps

    grpo_config = GRPOConfig(**grpo_config_kwargs)

    # GDPOTrainer 생성
    logger.info("GDPOTrainer 생성 중...")
    trainer = GDPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_cfg=cfg.rewards,
        advantage_cfg=cfg.advantage,
        vocab=vocab,
    )

    # 훈련 실행
    logger.info("GRPO 훈련 시작...")
    if resume_checkpoint:
        logger.info(f"Resume from checkpoint: {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # 훈련 완료 로그
    logger.info("훈련 완료")

    # 최종 저장: 중간 체크포인트와 동일한 구조 (adapter_model.safetensors + optimizer.pt + trainer_state.json)
    # Mod Record: 이전 구조에서는 merge_dora_and_save()로 full model을 저장했음.
    # 새 구조에서는 GRPO adapter만 저장. 추론 시 멀티 어댑터 스태킹으로 복원.
    output_dir = Path(cfg.training.output_dir) / "final"
    logger.info(f"최종 체크포인트 저장 중: {output_dir}")
    trainer.save_model(str(output_dir))
    trainer._save_optimizer_and_scheduler(str(output_dir))
    trainer.state.save_to_json(str(output_dir / "trainer_state.json"))

    logger.info("=== GRPO 훈련 완료 ===")


if __name__ == "__main__":
    main()
