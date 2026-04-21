"""Pre-Stage 훈련 실행 스크립트.

새 커스텀 토큰의 embedding과 lm_head 행만 훈련하는 워밍업 단계.
Transformer 레이어 전체는 동결된 상태로, PartialEmbedding / PartialLMHead 모듈을 통해
새 토큰 567행만 nn.Parameter로 분리하여 optimizer state를 최소화한다.

사용법:
    # 기본 실행 (config/training/pre_stage/pipeline.yaml 사용)
    uv run python scripts/training/run_pre_stage.py

    # DDP 멀티 GPU (2개)
    uv run torchrun --nproc_per_node=2 scripts/training/run_pre_stage.py

    # 하이퍼파라미터 오버라이드
    uv run python scripts/training/run_pre_stage.py training.learning_rate=1e-3

    # 디버그 모드 (10 step만 실행)
    uv run python scripts/training/run_pre_stage.py training.max_steps=10

    # W&B 비활성화
    uv run python scripts/training/run_pre_stage.py training.report_to=none

    # Resume: 최신 체크포인트 자동 탐색
    uv run python scripts/training/run_pre_stage.py resume.enabled=true

    # Resume: 특정 체크포인트 지정
    uv run python scripts/training/run_pre_stage.py resume.enabled=true resume.checkpoint_path=data/models/Qwen2.5-Coder-7B/checkpoints/pre_stage/checkpoint-500
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

from src.training.pre_stage import (
    PreStageDataset,
    build_trainer,
    load_model_and_tokenizer,
)

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
    # CuDNN 결정론적 모드 (성능 저하가 있을 수 있으나 재현성 보장)
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

    # 체크포인트 경로 직접 지정
    checkpoint_path = resume_cfg.get("checkpoint_path")
    if checkpoint_path:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"지정된 체크포인트를 찾을 수 없음: {path}")
        logger.info(f"지정된 체크포인트로 Resume: {path}")
        return str(path)

    # 최신 체크포인트 자동 탐색
    if resume_cfg.get("auto_find_latest", True):
        output_dir = Path(cfg.training.output_dir)
        if not output_dir.exists():
            logger.warning(f"output_dir가 없어 Resume 불가: {output_dir}. 처음부터 시작합니다.")
            return None

        # checkpoint-{step} 형식의 디렉토리를 step 번호 기준으로 정렬
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
    config_path=os.path.join(_PROJECT_ROOT, "config", "training", "pre_stage"),
    config_name="pipeline",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    """Pre-Stage 훈련 메인 함수.

    Args:
        cfg: Hydra가 주입하는 DictConfig.
    """
    logger.info("=== Pre-Stage 훈련 시작 ===")

    # DDP 재시작: distributed.nproc_per_node > 1이고 아직 torchrun 하위 프로세스가 아니면
    # os.execvp로 현재 프로세스를 torchrun으로 교체한다.
    # Hydra가 먼저 실행된 뒤 cfg를 읽으므로 커맨드라인 override도 자연스럽게 반영된다.
    # torchrun이 띄운 하위 프로세스는 LOCAL_RANK 환경변수를 가지므로 재귀 재시작은 없다.
    nproc = cfg.get("distributed", {}).get("nproc_per_node", 1)
    if nproc > 1 and "LOCAL_RANK" not in os.environ:
        cmd = ["torchrun", f"--nproc_per_node={nproc}"] + sys.argv
        logger.info(f"DDP 모드: torchrun으로 재시작 (nproc_per_node={nproc})")
        os.execvp("torchrun", cmd)

    # 증강 설정 로드 후 cfg.augmentation으로 병합
    # validate_augmentation.py와 동일한 패턴: _PROJECT_ROOT 기준 상대경로로 resolve
    aug_config_path = Path(_PROJECT_ROOT) / cfg.data.aug_pipeline_config
    aug_cfg = OmegaConf.load(aug_config_path)
    # Hydra DictConfig는 struct 모드라 선언되지 않은 키 추가 불가 → 일시 해제 후 병합
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

    # 모델 + 토크나이저 로드 (4bit 양자화 + 파라미터 동결 + PartialEmbedding 적용)
    # Resume 시에도 항상 새로 로드한 뒤 _setup_partial_training이 체크포인트의
    # 병합된 embed_tokens.weight에서 new_embed 초기값을 자동으로 추출한다
    logger.info("모델 및 토크나이저 로드 중...")
    model, tokenizer, new_token_ids = load_model_and_tokenizer(cfg)

    # 데이터셋 로드 (증강 파이프라인 포함)
    logger.info("데이터셋 로드 중...")
    train_dataset = PreStageDataset(cfg, tokenizer, split="train", seed=seed)
    eval_dataset = PreStageDataset(cfg, tokenizer, split="validation", seed=seed + 1)

    # Trainer 구성
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        cfg=cfg,
        new_token_ids=new_token_ids,
    )

    # 훈련 실행
    # resume_from_checkpoint가 지정되면 Trainer가 step/epoch/optimizer state를 복원한다.
    # optimizer.pt의 param shape이 new_embed(567,3584)과 동일하므로 정상 로드됨
    logger.info("훈련 시작...")
    if resume_checkpoint:
        logger.info(f"Resume from checkpoint: {resume_checkpoint}")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    # 훈련 결과 로그
    logger.info(f"훈련 완료: {train_result.metrics}")
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    # 최종 평가 (merge 이전에 수행해야 PartialLMHead의 학습된 가중치로 평가)
    logger.info("최종 평가 실행 중...")
    eval_metrics = trainer.evaluate()
    logger.info(f"최종 평가 결과: {eval_metrics}")
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # 최종 저장: 중간 체크포인트와 완전히 동일한 구조 (partial_state.pt + optimizer.pt + trainer_state.json)
    # Mod Record: 이전 구조에서는 merge_and_restore() 후 model.safetensors 전체를 저장했음.
    # 다음 stage(SFT/GRPO)도 HF Hub에서 base model을 새로 로드하므로 전체 모델 저장은 불필요.
    # final도 resume 시작점으로 사용될 수 있으므로 중간 체크포인트와 동일하게 저장한다.
    output_dir = Path(cfg.training.output_dir) / "final"
    logger.info(f"최종 체크포인트 저장 중: {output_dir}")
    trainer.save_final_checkpoint(str(output_dir))

    logger.info("=== Pre-Stage 훈련 완료 ===")


if __name__ == "__main__":
    main()
