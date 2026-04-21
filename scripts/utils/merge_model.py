"""모델 병합 실행 스크립트.

Hub NF4 base + partial_state.pt + adapter 체인을 bf16 full model로 병합한다.
병합 결과는 추론 파이프라인의 load_mode="merged" 모드에서 사용한다.

사용법:
    # 기본 실행 (config/utils/merge.yaml 사용)
    uv run python scripts/utils/merge_model.py

    # SFT + GRPO 모두 병합
    uv run python scripts/utils/merge_model.py \\
        merge.output_dir=data/models/Qwen2.5-Coder-7B/merged/grpo \\
        merge.adapters='[{path: data/.../sft/final, name: sft}, {path: data/.../grpo/final, name: grpo}]'
"""

import logging
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils.model_merger import ModelMerger

logger = logging.getLogger(__name__)


@hydra.main(
    config_path=os.path.join(_PROJECT_ROOT, "config", "utils"),
    config_name="merge",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    """모델 병합 메인 함수.

    Args:
        cfg: Hydra가 주입하는 DictConfig.
    """
    logger.info("=== 모델 병합 시작 ===")
    logger.info(f"설정:\n{OmegaConf.to_yaml(cfg)}")

    merger = ModelMerger(cfg)
    merger.merge(output_dir=cfg.merge.output_dir)

    logger.info("=== 모델 병합 완료 ===")


if __name__ == "__main__":
    main()
