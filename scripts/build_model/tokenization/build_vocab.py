"""Vocabulary 빌드 실행 스크립트.

config/vocab/pipeline.yaml 설정을 기반으로 커스텀 토큰을 정의하고
pretrained tokenizer에 추가한 뒤 결과를 저장한다.

실행 예시:
    uv run python scripts/build_model/tokenization/build_vocab.py
    uv run python scripts/build_model/tokenization/build_vocab.py model.hub_id=Qwen/Qwen2.5-Coder-7B-Instruct model.name=Qwen2.5-Coder-7B-Instruct
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

# 프로젝트 루트를 sys.path에 추가
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.build_model.tokenization.vocab_builder import build_vocab  # noqa: E402

log = logging.getLogger(__name__)


@hydra.main(
    config_path=str(Path(_PROJECT_ROOT) / "config" / "build_model" / "tokenization"),
    config_name="pipeline",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    """Vocabulary 빌드 메인 함수.

    Args:
        cfg: Hydra가 로드한 pipeline.yaml 설정.

    Returns:
        None.
    """
    log.info("=== Vocabulary 빌드 시작 ===")
    log.info("모델: %s (저장 경로: data/models/%s)", cfg.model.hub_id, cfg.model.name)

    merge_config_path = Path(cfg.data.room_type_merge_config)
    output_dir = Path(cfg.output.dir)

    if not merge_config_path.exists():
        raise FileNotFoundError(
            f"room_type_merge_config 파일을 찾을 수 없음: {merge_config_path}"
        )

    token_cfg = cfg.get("tokens", {})
    result = build_vocab(
        model_name=cfg.model.hub_id,
        merge_config_path=merge_config_path,
        output_dir=output_dir,
        max_rid=token_cfg.get("max_rid", 15),
        max_coord_x=token_cfg.get("max_coord_x", 255),
        max_coord_y=token_cfg.get("max_coord_y", 255),
    )

    log.info("=== Vocabulary 빌드 완료 ===")
    log.info("기본 vocab 크기  : %d", result["base_vocab_size"])
    log.info("확장 후 vocab 크기: %d", result["new_vocab_size"])
    log.info("추가된 토큰 수   : %d", result["total_added"])
    log.info("카테고리별 토큰 수:")
    for cat, tokens in result["categories"].items():
        log.info("  %-20s: %d개", cat, len(tokens))


if __name__ == "__main__":
    main()
