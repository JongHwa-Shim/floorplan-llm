"""데이터 증강 + 토크나이징 검증 스크립트.

Arrow 데이터셋에서 N개 샘플을 무작위로 뽑아 증강 파이프라인을 적용하고,
각 샘플의 origin / input / output 3섹션을 하나의 텍스트 파일에 저장한다.

샘플 1개당 augment_per_sample회 서로 다른 시드로 증강을 적용하여
증강 결과의 다양성을 확인할 수 있다.

Usage:
    # 기본 실행 (config/augmentation/validate.yaml 사용)
    uv run python scripts/training/augmentation/validate_augmentation.py

    # 샘플 수 오버라이드
    uv run python scripts/training/augmentation/validate_augmentation.py validate.num_samples=10

    # 증강 없이 (전체 정보 입력 + 표현 변형만)
    uv run python scripts/training/augmentation/validate_augmentation.py \\
        augmentation.drop.p_drop_block=0 \\
        augmentation.drop.p_drop_type=0 \\
        augmentation.drop.p_drop_coords=0 \\
        augmentation.drop.p_drop_edge=0 \\
        augmentation.drop.p_drop_pair=0 \\
        augmentation.drop.p_drop_door=0 \\
        augmentation.drop.p_drop_spatial=0 \\
        augmentation.drop.p_drop_front_door=0

    # 특정 split 사용
    uv run python scripts/training/augmentation/validate_augmentation.py data.split=validation
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

import datasets
import hydra
from omegaconf import DictConfig

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.training.augmentation.decoder import (  # noqa: E402
    decode_tokens,
    format_sample_report,
)
from src.training.augmentation.pipeline import (  # noqa: E402
    AugmentationConfig,
    AugmentationPipeline,
    config_from_omegaconf,
)
from src.training.augmentation.strategies import DropState  # noqa: E402
from src.training.augmentation.tokenizer import (  # noqa: E402
    Vocab,
    build_condition_tokens,
    load_vocab,
    to_row_oriented,
)

log = logging.getLogger(__name__)


def _load_dataset(arrow_dir: str, split: str) -> datasets.Dataset:
    """Arrow 데이터셋을 로드한다.

    DatasetDict(train/validation/test)와 단일 Dataset 모두 지원한다.

    Args:
        arrow_dir: Arrow 데이터셋 디렉토리 경로.
        split: 사용할 split 이름.

    Returns:
        로드된 Dataset 객체.

    Raises:
        FileNotFoundError: 경로가 존재하지 않을 때.
        KeyError: split이 DatasetDict에 없을 때.
    """
    ds = datasets.load_from_disk(arrow_dir)
    if isinstance(ds, datasets.DatasetDict):
        if split not in ds:
            available = list(ds.keys())
            raise KeyError(
                f"split '{split}'이 DatasetDict에 없습니다. "
                f"사용 가능한 split: {available}"
            )
        return ds[split]
    return ds


def _sample_indices(dataset_size: int, num_samples: int, seed: int) -> list[int]:
    """데이터셋에서 무작위 인덱스를 추출한다.

    Args:
        dataset_size: 전체 데이터셋 크기.
        num_samples: 추출할 샘플 수.
        seed: 재현성을 위한 시드.

    Returns:
        무작위 인덱스 리스트 (오름차순 정렬).
    """
    rng = random.Random(seed)
    n = min(num_samples, dataset_size)
    indices = rng.sample(range(dataset_size), n)
    return sorted(indices)


@hydra.main(
    version_base=None,
    config_path=os.path.join(_PROJECT_ROOT, "config", "training", "augmentation", "validate_augmentation"),
    config_name="pipeline",
)
def main(cfg: DictConfig) -> None:
    """증강 검증 메인 진입점.

    Args:
        cfg: Hydra DictConfig.
    """
    arrow_dir   = os.path.join(_PROJECT_ROOT, cfg.data.arrow_dir)
    vocab_path  = Path(_PROJECT_ROOT) / cfg.model.vocab_extension
    tokenizer_dir = Path(_PROJECT_ROOT) / cfg.model.tokenizer_dir
    output_file = cfg.validate.output_file
    num_samples = cfg.validate.num_samples
    seed        = cfg.validate.seed
    n_aug       = cfg.validate.augment_per_sample

    # --- vocab 로드 ---
    log.info("vocab 로드 중: %s", vocab_path)
    vocab = load_vocab(vocab_path, tokenizer_dir)
    log.info(
        "vocab 로드 완료 — 커스텀 토큰: %d개, BOS: %s, EOS: %s",
        len(vocab.token_to_id),
        vocab.bos_token_id,
        vocab.eos_token_id,
    )

    # --- 데이터셋 로드 ---
    log.info("데이터셋 로드 중: %s (split=%s)", arrow_dir, cfg.data.split)
    dataset = _load_dataset(arrow_dir, cfg.data.split)
    log.info("데이터셋 크기: %d", len(dataset))

    # --- 샘플 인덱스 추출 ---
    indices = _sample_indices(len(dataset), num_samples, seed)
    log.info("검증 샘플 수: %d", len(indices))

    # --- 증강 설정 (pipeline.yaml을 단일 출처로 로드) ---
    from omegaconf import OmegaConf  # 지연 임포트
    pipeline_yaml = Path(_PROJECT_ROOT) / cfg.data.aug_pipeline_config
    pipeline_cfg = OmegaConf.load(pipeline_yaml)
    log.info("증강 설정 로드: %s", pipeline_yaml)
    aug_cfg: AugmentationConfig = config_from_omegaconf(pipeline_cfg)

    # --- 리포트 생성 ---
    all_reports: list[str] = []

    for sample_no, ds_idx in enumerate(indices, start=1):
        raw_sample = dataset[ds_idx]
        row_sample = to_row_oriented(raw_sample)
        plan_id = row_sample["plan_id"]

        log.info("[%d/%d] plan_id=%s (dataset idx=%d)", sample_no, len(indices), plan_id, ds_idx)

        # origin 토크나이징 — 증강 없이 full condition 토큰만 시각화 (output은 생략)
        no_aug_state = DropState()
        origin_condition_tokens = build_condition_tokens(row_sample, no_aug_state, vocab)
        origin_str = decode_tokens(origin_condition_tokens, vocab)

        # augment_per_sample회 서로 다른 시드로 증강 적용
        for aug_no in range(1, n_aug + 1):
            aug_seed = seed * 1000 + sample_no * 100 + aug_no
            pipeline = AugmentationPipeline(vocab=vocab, cfg=aug_cfg, seed=aug_seed)

            condition_tokens, output_tokens = pipeline(raw_sample)

            condition_str = decode_tokens(condition_tokens, vocab)
            output_str    = decode_tokens(output_tokens, vocab)
            aug_summary   = pipeline.augmented_summary()

            # 두 번째 증강부터는 origin 섹션 생략하고 label로 구분
            if aug_no == 1:
                report = format_sample_report(
                    plan_id=plan_id,
                    sample_idx=sample_no,
                    augmentation_summary=f"[증강 {aug_no}/{n_aug}] {aug_summary}",
                    origin_str=origin_str,
                    condition_str=condition_str,
                    output_str=output_str,
                )
            else:
                # 2번째 이후 — origin 반복 없이 증강 결과만 표시
                lines = [
                    f"{'·' * 60}",
                    f"[증강 {aug_no}/{n_aug}] {aug_summary}",
                    f"{'·' * 60}",
                    "",
                    "[INPUT]",
                    condition_str,
                    "",
                    "─" * 60,
                    "[OUTPUT]",
                    output_str,
                    "",
                ]
                report = "\n".join(lines)

            all_reports.append(report)

    # --- 파일 저장 (Hydra 실행 디렉토리 기준 저장) ---
    output_path = Path(output_file)
    if not output_path.is_absolute():
        output_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_reports))

    log.info(
        "검증 완료. 결과 저장: %s (%d 샘플 × %d 증강 = %d 리포트)",
        output_path.resolve(),
        len(indices),
        n_aug,
        len(all_reports),
    )


if __name__ == "__main__":
    main()
