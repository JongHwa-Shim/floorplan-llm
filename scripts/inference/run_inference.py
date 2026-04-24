"""추론(Inference) 실행 스크립트.

훈련된 LLM 모델에 입력 조건을 주어 평면도 토큰 시퀀스를 생성하고,
결과를 텍스트/JSON/이미지로 저장한다.

사용법:
    # 기본 실행 (JSONL 파일, 증강 적용)
    uv run python scripts/inference/run_inference.py

    # 텍스트 파일 입력 모드 (data/inference/input_txt/ 폴더 사용)
    uv run python scripts/inference/run_inference.py input.mode=txt_dir

    # 단일 JSONL 파일, 특정 plan_id만
    uv run python scripts/inference/run_inference.py \\
        input.mode=jsonl_file \\
        input.jsonl_file=data/dataset/processed_dataset/rplan/jsonl/floorplans_0000.jsonl \\
        'input.plan_ids=[fp_00001,fp_00005]'

    # Arrow test split, 10개, 3개 출력 (sampling 모드)
    uv run python scripts/inference/run_inference.py \\
        input.mode=arrow input.max_samples=10 \\
        generation.num_outputs=3 \\
        generation.do_sample=true generation.temperature=0.8

    # 다른 모델 사용 (GRPO)
    uv run python scripts/inference/run_inference.py \\
        model.training_stage=grpo

출력 디렉토리 구조:
    outputs/inference/{model.name}/{model.training_stage}/{YYYY-MM-DD}/{HH-MM-SS}/
    ├── .hydra/            (Hydra 설정 스냅샷)
    ├── run_inference.log  (실행 로그)
    └── {plan_id}/
        ├── input/
        │   ├── tokens.txt, condition.json, floorplan.png
        ├── output/            (num_outputs=1)
        │   ├── tokens.txt, floorplan.json, floorplan.png
        ├── output_0/          (num_outputs>1)
        ├── output_1/
        └── meta.json
"""

import logging
import os
import random
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# 프로젝트 루트를 sys.path에 추가 (패키지 임포트 보장)
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.inference.condition_builder import (  # noqa: E402
    build_condition_no_aug,
    build_condition_with_augmentation,
    load_samples,
)
from src.inference.generator import generate_floorplan  # noqa: E402
from src.inference.model_loader import load_model_for_inference  # noqa: E402
from src.inference.output_parser import parse_output_tokens  # noqa: E402
from src.inference.result_saver import save_results  # noqa: E402
from src.training.augmentation.pipeline import (  # noqa: E402
    AugmentationPipeline,
    config_from_omegaconf,
)
from src.training.augmentation.tokenizer import load_vocab  # noqa: E402

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


@hydra.main(
    config_path=os.path.join(_PROJECT_ROOT, "config", "inference"),
    config_name="pipeline",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    """추론 메인 함수.

    Args:
        cfg: Hydra가 주입하는 DictConfig.
    """
    logger.info("=== 추론 파이프라인 시작 ===")
    logger.info("설정:\n%s", OmegaConf.to_yaml(cfg))

    # 재현성 시드 고정
    seed = cfg.seed
    set_seed(seed)

    # 모델 + 토크나이저 로드
    logger.info("모델 로드 중...")
    model, tokenizer = load_model_for_inference(cfg)

    # Vocab 로드
    vocab_path = Path(_PROJECT_ROOT) / cfg.model.vocab_extension
    tokenizer_dir = Path(_PROJECT_ROOT) / cfg.model.tokenizer_dir
    vocab = load_vocab(vocab_path, tokenizer_dir)
    logger.info("Vocab 로드 완료 — 커스텀 토큰: %d개", len(vocab.token_to_id))

    # 색상 설정 로드
    color_map_path = Path(_PROJECT_ROOT) / cfg.color_map_path
    color_map_cfg = OmegaConf.load(color_map_path)

    # 증강 파이프라인 초기화 (txt_dir 모드이거나 augmentation.enabled=false 시 미사용)
    is_txt_mode = (cfg.input.mode == "txt_dir")
    pipeline = None
    if not is_txt_mode and cfg.augmentation.enabled:
        aug_config_path = Path(_PROJECT_ROOT) / cfg.augmentation.config_path
        aug_omegacfg = OmegaConf.load(aug_config_path)
        aug_config = config_from_omegaconf(aug_omegacfg)
        pipeline = AugmentationPipeline(vocab, aug_config, seed=seed)
        logger.info("증강 파이프라인 초기화 완료 (config: %s)", aug_config_path)
    elif is_txt_mode:
        logger.info("txt_dir 모드 — 증강 미적용 (이미 완료된 입력)")
    else:
        logger.info("증강 비활성화 — full condition으로 입력 구성")

    # 입력 샘플 로드
    # txt_dir 모드에서는 tokenizer/vocab이 필요 (토큰 텍스트 파싱용)
    logger.info("입력 샘플 로드 중...")
    raw_samples, row_samples = load_samples(
        cfg,
        tokenizer=tokenizer if is_txt_mode else None,
        vocab=vocab if is_txt_mode else None,
    )
    logger.info("처리 대상: %d개 샘플", len(row_samples))

    # 출력 디렉토리: Hydra run.dir 기준 (outputs/inference/{model.name}/{training_stage}/{date}/)
    # Hydra 설정 스냅샷·로그와 추론 결과가 동일한 날짜 폴더 아래 저장됨
    # txt_dir 모드는 별도 서브디렉토리로 분리하여 다른 입력 모드 결과와 혼재 방지
    output_dir = Path(HydraConfig.get().run.dir)
    if is_txt_mode:
        output_dir = output_dir / "txt_input"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1대N 출력 설정
    num_outputs = cfg.generation.get("num_outputs", 1)
    if num_outputs > 1:
        logger.info("1대N 출력 활성화: 샘플당 %d개 생성", num_outputs)

    # 추론 실행
    success_count = 0
    fail_count = 0

    for raw_sample, row_sample in tqdm(
        zip(raw_samples, row_samples), total=len(row_samples), desc="추론 진행"
    ):
        plan_id = row_sample["plan_id"]

        # condition 토큰 빌드
        if is_txt_mode:
            # txt_dir 모드: 텍스트 파일의 토큰 텍스트를 직접 인코딩
            condition_tokens = tokenizer.encode(
                raw_sample["token_text"], add_special_tokens=False
            )
            aug_summary = "pre-augmented (text file)"
            input_sample = row_sample  # parse_input_tokens()로 역변환된 구조화 dict
            drop_state = None
        elif pipeline is not None:
            condition_tokens, _, aug_summary, input_sample, drop_state = build_condition_with_augmentation(
                raw_sample, pipeline
            )
        else:
            condition_tokens = build_condition_no_aug(row_sample, vocab)
            aug_summary = "no augmentation"
            input_sample = row_sample
            drop_state = None

        # N개 출력 생성
        output_results: list[tuple[list[int], dict | None, float]] = []
        for _ in range(num_outputs):
            t_start = time.perf_counter()
            generated_ids = generate_floorplan(
                condition_tokens, model, tokenizer, cfg.generation
            )
            elapsed = time.perf_counter() - t_start
            parsed = parse_output_tokens(generated_ids, vocab)
            output_results.append((generated_ids, parsed, elapsed))

            if parsed is not None:
                success_count += 1
            else:
                fail_count += 1
                logger.warning("파싱 실패: plan_id=%s", plan_id)

        # 결과 저장
        save_results(
            plan_id=plan_id,
            raw_sample=input_sample,
            condition_tokens=condition_tokens,
            output_results=output_results,
            vocab=vocab,
            output_cfg=cfg.output,
            color_map_cfg=color_map_cfg,
            output_dir=output_dir,
            augmentation_summary=aug_summary,
            drop_state=drop_state,
        )

    total_outputs = len(row_samples) * num_outputs
    logger.info(
        "=== 추론 완료 === %d개 샘플 × %d출력 = %d건 처리 (성공: %d, 파싱 실패: %d) → %s",
        len(row_samples),
        num_outputs,
        total_outputs,
        success_count,
        fail_count,
        output_dir,
    )


if __name__ == "__main__":
    main()
