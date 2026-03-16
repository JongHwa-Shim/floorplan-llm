"""JSONL → Arrow 변환 실행 스크립트.

config/json2arrow/pipeline.yaml 설정을 기반으로 JSONL 샤드 파일들을
HuggingFace Arrow 데이터셋으로 변환하여 디스크에 저장한다.
Hydra 기반 설정 관리로 하이퍼파라미터를 CLI에서 오버라이드 가능.

Usage:
    # 전체 배치 변환 (split 포함, 기본값)
    uv run python scripts/build_dataset/json2arrow/run_conversion.py

    # split 비율 오버라이드
    uv run python scripts/build_dataset/json2arrow/run_conversion.py split.val_ratio=0.05 split.test_ratio=0.1

    # split 비활성화 (단일 Dataset으로 저장)
    uv run python scripts/build_dataset/json2arrow/run_conversion.py split.enabled=false

    # 단일 파일 디버깅
    uv run python scripts/build_dataset/json2arrow/run_conversion.py mode=single target_file=floorplans_0000.jsonl

    # 검증 비활성화
    uv run python scripts/build_dataset/json2arrow/run_conversion.py validation.enabled=false
"""

from __future__ import annotations

import logging
import os
import sys
from glob import glob
from pathlib import Path

import glob as _glob

import datasets
import hydra
from omegaconf import DictConfig

# 프로젝트 루트를 sys.path에 추가
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.build_dataset.json2arrow.converter import convert_to_arrow  # noqa: E402
from src.build_dataset.json2arrow.schema import get_floorplan_features  # noqa: E402
from src.build_dataset.json2arrow.validator import validate_conversion  # noqa: E402

log = logging.getLogger(__name__)


def split_and_save(
    dataset: datasets.Dataset,
    output_dir: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> datasets.DatasetDict:
    """Dataset을 train/validation/test로 분할하여 DatasetDict로 저장.

    분할 순서: 전체 → test 분리 → 나머지에서 validation 분리 → 나머지가 train.
    이 순서 덕분에 test 비율이 전체 기준으로 정확하게 유지된다.

    Args:
        dataset: 분할할 전체 Dataset.
        output_dir: DatasetDict 저장 경로.
        val_ratio: validation 비율 (전체 대비).
        test_ratio: test 비율 (전체 대비).
        seed: 셔플 시드 (재현성 보장).

    Returns:
        datasets.DatasetDict: train/validation/test split이 완료된 DatasetDict.
    """
    total = len(dataset)

    # 전체에서 test 분리
    split1 = dataset.train_test_split(test_size=test_ratio, seed=seed)
    test_ds = split1["test"]
    remainder = split1["train"]

    # 나머지에서 validation 분리
    # remainder 기준 val 비율 = val_ratio / (1 - test_ratio)
    val_ratio_in_remainder = val_ratio / (1.0 - test_ratio)
    split2 = remainder.train_test_split(test_size=val_ratio_in_remainder, seed=seed)
    train_ds = split2["train"]
    val_ds = split2["test"]

    log.info(
        "Split 결과 — train: %d (%.1f%%), validation: %d (%.1f%%), test: %d (%.1f%%)",
        len(train_ds), len(train_ds) / total * 100,
        len(val_ds),   len(val_ds)   / total * 100,
        len(test_ds),  len(test_ds)  / total * 100,
    )

    dataset_dict = datasets.DatasetDict(
        {
            "train": train_ds,
            "validation": val_ds,
            "test": test_ds,
        }
    )
    dataset_dict.save_to_disk(output_dir)

    # DatasetDict 저장 후 루트 레벨 단일 Dataset 파일 제거
    # (convert_to_arrow()가 먼저 저장한 파일들이 load_from_disk()를 혼란시킴)
    for pattern in ["data-*.arrow", "dataset_info.json", "state.json"]:
        for f in _glob.glob(os.path.join(output_dir, pattern)):
            os.remove(f)
            log.debug("루트 레벨 파일 제거: %s", f)

    log.info("DatasetDict 저장 완료: %s", output_dir)
    return dataset_dict


@hydra.main(
    version_base=None,
    config_path=os.path.join(_PROJECT_ROOT, "config", "build_dataset", "json2arrow"),
    config_name="pipeline",
)
def main(cfg: DictConfig) -> None:
    """JSONL → Arrow 변환 메인 진입점.

    Args:
        cfg: Hydra DictConfig.
    """
    input_dir = os.path.join(_PROJECT_ROOT, cfg.data.input_dir)
    output_dir = os.path.join(_PROJECT_ROOT, cfg.data.output_dir)
    features = get_floorplan_features()

    if cfg.mode == "single":
        # 단일 파일 모드 (디버깅용)
        if cfg.target_file is None:
            log.error("single 모드에서는 target_file을 지정해야 합니다.")
            return

        jsonl_path = os.path.join(input_dir, cfg.target_file)
        if not os.path.exists(jsonl_path):
            log.error("파일을 찾을 수 없음: %s", jsonl_path)
            return

        jsonl_paths = [jsonl_path]
        log.info("single 모드: %s", jsonl_path)

    else:
        # 배치 모드: input_pattern 매칭 파일 전체 처리
        pattern = os.path.join(input_dir, cfg.data.input_pattern)
        jsonl_paths = sorted(glob(pattern))

        if not jsonl_paths:
            log.error("패턴 '%s'에 매칭되는 파일이 없습니다.", pattern)
            return

        log.info("batch 모드: %d개 JSONL 파일 발견.", len(jsonl_paths))

    # 변환 실행
    dataset = convert_to_arrow(
        jsonl_paths=jsonl_paths,
        output_dir=output_dir,
        features=features,
    )

    # 검증 실행 (split 전 원본 Dataset 기준)
    if cfg.validation.enabled:
        passed = validate_conversion(
            arrow_dir=output_dir,
            jsonl_paths=jsonl_paths,
            num_samples=cfg.validation.num_samples,
            seed=cfg.validation.seed,
        )
        if not passed:
            log.error("변환 검증 실패. Arrow 데이터셋을 확인하세요.")
            sys.exit(1)
    else:
        log.info("검증 비활성화 (validation.enabled=false).")

    # split 적용
    if cfg.split.enabled:
        log.info(
            "Split 적용 중 (val=%.3f, test=%.3f, seed=%d)...",
            cfg.split.val_ratio,
            cfg.split.test_ratio,
            cfg.split.seed,
        )
        split_and_save(
            dataset=dataset,
            output_dir=output_dir,
            val_ratio=cfg.split.val_ratio,
            test_ratio=cfg.split.test_ratio,
            seed=cfg.split.seed,
        )
    else:
        log.info("Split 비활성화 (split.enabled=false). 단일 Dataset으로 저장됨.")

    log.info("완료. Arrow 데이터셋 위치: %s", output_dir)


if __name__ == "__main__":
    main()
