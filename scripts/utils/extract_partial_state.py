"""병합된 model.safetensors에서 partial_state.pt를 추출하는 CLI 스크립트.

Pre-Stage 훈련 방식 변경 전에 저장된 model.safetensors (merge_and_restore 방식)에서
커스텀 토큰 가중치(new_embed, new_lm_head)만 분리하여 현재 코드와 호환되는
partial_state.pt를 생성한다.

사용 예시:
    # 기본 실행 (출력 경로 자동 결정)
    uv run python scripts/utils/extract_partial_state.py \\
        --checkpoint_dir data/models/Qwen2.5-Coder-7B/checkpoints/pre_stage/final \\
        --model_name Qwen2.5-Coder-7B

    # 출력 경로 지정 (final_checkpoints/pre_stage로 직접 추출)
    uv run python scripts/utils/extract_partial_state.py \\
        --checkpoint_dir data/models/Qwen2.5-Coder-7B/checkpoints/pre_stage/final \\
        --model_name Qwen2.5-Coder-7B \\
        --output_path data/models/Qwen2.5-Coder-7B/final_checkpoints/pre_stage/partial_state.pt

    # dtype 변환 (bfloat16으로 저장)
    uv run python scripts/utils/extract_partial_state.py \\
        --checkpoint_dir data/models/Qwen2.5-Coder-7B/checkpoints/pre_stage/final \\
        --model_name Qwen2.5-Coder-7B \\
        --output_path data/models/Qwen2.5-Coder-7B/final_checkpoints/pre_stage/partial_state.pt \\
        --dtype bfloat16

    # vocab_extension.json 경로 직접 지정
    uv run python scripts/utils/extract_partial_state.py \\
        --checkpoint_dir /path/to/checkpoint \\
        --vocab_extension_path /path/to/vocab_extension.json \\
        --output_path /path/to/partial_state.pt
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# src 패키지 경로 추가
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.extract_partial_state import extract_partial_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="병합된 model.safetensors → partial_state.pt 추출",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        required=True,
        help="model.safetensors가 있는 체크포인트 디렉토리",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help=(
            "모델명 (예: Qwen2.5-Coder-7B). "
            "vocab_extension_path를 자동 결정하는 데 사용. "
            "--vocab_extension_path를 직접 지정하면 불필요."
        ),
    )
    parser.add_argument(
        "--vocab_extension_path",
        type=Path,
        default=None,
        help=(
            "vocab_extension.json 경로. "
            "미지정 시 data/models/{model_name}/tokenization/vocab_extension.json 사용."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help=(
            "저장할 partial_state.pt 경로. "
            "미지정 시 {checkpoint_dir}/partial_state_extracted.pt로 저장 "
            "(기존 partial_state.pt 덮어쓰기 방지)."
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=list(_DTYPE_MAP.keys()),
        help="저장 dtype (기본값: safetensors 원본 dtype 유지)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # vocab_extension_path 결정
    vocab_extension_path: Path
    if args.vocab_extension_path is not None:
        vocab_extension_path = args.vocab_extension_path
    elif args.model_name is not None:
        # 프로젝트 루트 기준 상대 경로
        project_root = Path(__file__).resolve().parents[2]
        vocab_extension_path = (
            project_root / "data" / "models" / args.model_name / "tokenization" / "vocab_extension.json"
        )
    else:
        logger.error("--model_name 또는 --vocab_extension_path 중 하나를 지정해야 합니다.")
        sys.exit(1)

    # output_path 결정
    output_path: Path
    if args.output_path is not None:
        output_path = args.output_path
    else:
        output_path = args.checkpoint_dir / "partial_state_extracted.pt"
        logger.info(
            "출력 경로 미지정 → 기본값 사용 (기존 partial_state.pt 보존): %s", output_path
        )

    # dtype 결정
    dtype: torch.dtype | None = None
    if args.dtype is not None:
        dtype = _DTYPE_MAP[args.dtype]

    # 입력값 요약 출력
    logger.info("=== partial_state.pt 추출 시작 ===")
    logger.info("  checkpoint_dir      : %s", args.checkpoint_dir)
    logger.info("  vocab_extension_path: %s", vocab_extension_path)
    logger.info("  output_path         : %s", output_path)
    logger.info("  dtype               : %s", args.dtype or "원본 유지")

    try:
        partial_state = extract_partial_state(
            checkpoint_dir=args.checkpoint_dir,
            vocab_extension_path=vocab_extension_path,
            output_path=output_path,
            dtype=dtype,
        )
    except FileNotFoundError as e:
        logger.error("파일 없음: %s", e)
        sys.exit(1)
    except KeyError as e:
        logger.error("키 오류: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("추출 실패: %s", e, exc_info=True)
        sys.exit(1)

    # 결과 요약
    new_embed = partial_state["new_embed"]
    new_lm_head = partial_state["new_lm_head"]
    new_token_ids = partial_state["new_token_ids"]

    logger.info("=== 추출 완료 ===")
    logger.info("  저장 경로       : %s", output_path)
    logger.info("  new_embed       : shape=%s, dtype=%s", new_embed.shape, new_embed.dtype)
    logger.info("  new_lm_head     : shape=%s, dtype=%s", new_lm_head.shape, new_lm_head.dtype)
    logger.info("  new_token_ids   : %d개, 범위=[%d, %d]", len(new_token_ids), new_token_ids[0], new_token_ids[-1])
    logger.info("  파일 크기       : %.1f MB", output_path.stat().st_size / 1024 / 1024)


if __name__ == "__main__":
    main()
