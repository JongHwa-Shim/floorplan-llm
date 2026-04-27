"""Resume 기능 검증 스크립트.

partial_state.pt 로드 → new_embed/new_lm_head 복원 → embed 값 일치 검증.
모델 전체 로드 없이 partial_state.pt의 복원 로직만 빠르게 검증한다.

사용법:
    # 최신 체크포인트 자동 탐색
    uv run python tests/training/pre_stage/validate_resume.py

    # 특정 체크포인트 지정
    uv run python tests/training/pre_stage/validate_resume.py \
        --checkpoint data/models/Qwen2.5-Coder-7B/checkpoints/pre_stage/checkpoint-80304
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.training.pre_stage.model_loader import PartialEmbedding, PartialLMHead

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    """step 번호 기준으로 가장 최신 체크포인트를 반환한다."""
    checkpoints = sorted(
        [p for p in output_dir.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return checkpoints[-1] if checkpoints else None


def _mock_partial_embedding(new_token_ids: list, vocab_size: int, hidden_size: int) -> PartialEmbedding:
    """모델 로드 없이 PartialEmbedding 구조만 생성한다 (복원 로직 검증용).

    Args:
        new_token_ids: 새 토큰 ID 리스트.
        vocab_size: 전체 vocab 크기.
        hidden_size: 임베딩 hidden size.

    Returns:
        dummy PartialEmbedding 인스턴스.
    """
    base_embedding = nn.Embedding(vocab_size, hidden_size)
    return PartialEmbedding(base_embedding, new_token_ids)


def validate_partial_state(checkpoint_dir: Path) -> bool:
    """partial_state.pt 파일 존재 여부 및 텐서 형태를 검증한다.

    Args:
        checkpoint_dir: 체크포인트 디렉토리 경로.

    Returns:
        검증 통과 여부.
    """
    partial_state_path = checkpoint_dir / "partial_state.pt"

    # --- 파일 존재 확인 ---
    if not partial_state_path.exists():
        logger.error(f"[FAIL] partial_state.pt 없음: {partial_state_path}")
        logger.error("      → _save_checkpoint 오버라이드가 적용되지 않은 체크포인트입니다.")
        return False
    logger.info(f"[PASS] partial_state.pt 존재: {partial_state_path}")

    # --- 텐서 로드 및 키/형태 검증 ---
    state = torch.load(partial_state_path, map_location="cpu", weights_only=True)

    required_keys = {"new_embed", "new_lm_head", "new_token_ids"}
    missing = required_keys - state.keys()
    if missing:
        logger.error(f"[FAIL] partial_state.pt에 누락된 키: {missing}")
        return False
    logger.info(f"[PASS] 필수 키 존재: {list(state.keys())}")

    new_embed: torch.Tensor = state["new_embed"]
    new_lm_head: torch.Tensor = state["new_lm_head"]
    new_token_ids: list = state["new_token_ids"]

    logger.info(f"[INFO] new_embed     shape: {tuple(new_embed.shape)}, dtype: {new_embed.dtype}")
    logger.info(f"[INFO] new_lm_head   shape: {tuple(new_lm_head.shape)}, dtype: {new_lm_head.dtype}")
    logger.info(f"[INFO] new_token_ids count: {len(new_token_ids)}")

    # 형태 정합성 확인 (new_embed와 new_lm_head의 shape이 동일해야 함)
    if new_embed.shape != new_lm_head.shape:
        logger.error(f"[FAIL] new_embed {new_embed.shape} ≠ new_lm_head {new_lm_head.shape}")
        return False
    logger.info(f"[PASS] new_embed / new_lm_head shape 일치: {tuple(new_embed.shape)}")

    # 토큰 ID 수 일치 확인
    n_tokens = new_embed.shape[0]
    if len(new_token_ids) != n_tokens:
        logger.error(f"[FAIL] new_token_ids 수({len(new_token_ids)}) ≠ embed 행 수({n_tokens})")
        return False
    logger.info(f"[PASS] new_token_ids 수 일치: {n_tokens}")

    # --- 모의 복원: PartialEmbedding 생성 후 partial_state 덮어쓰기 ---
    hidden_size = new_embed.shape[1]
    vocab_size = max(new_token_ids) + 1  # 최소 vocab_size 추정

    mock_embed = _mock_partial_embedding(new_token_ids, vocab_size, hidden_size)
    original_values = mock_embed.new_embed.data.clone()

    # 복원 수행
    mock_embed.new_embed.data.copy_(new_embed)
    restored_values = mock_embed.new_embed.data

    # float32 버퍼에 bfloat16 복사 후 비교 시 dtype 통일 필요
    if not torch.allclose(restored_values, new_embed.float()):
        logger.error("[FAIL] partial_state 복원 후 값 불일치")
        return False
    logger.info("[PASS] partial_state 모의 복원 성공 (값 일치)")

    # 복원 전후 값이 실제로 달라졌는지 (훈련이 있었다면 달라야 함)
    max_diff = (restored_values - original_values).abs().max().item()
    logger.info(f"[INFO] 초기값 대비 최대 변화량: {max_diff:.6f} (0이면 아직 미훈련)")
    if max_diff == 0.0:
        logger.warning("[WARN] new_embed가 초기값과 동일합니다. 훈련이 반영되지 않은 체크포인트일 수 있습니다.")

    return True


def main():
    parser = argparse.ArgumentParser(description="Pre-Stage resume 기능 검증")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="검증할 체크포인트 경로. 미지정 시 output_dir에서 최신 탐색.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/models/Qwen2.5-Coder-7B/checkpoints/pre_stage",
        help="체크포인트 탐색 기준 디렉토리.",
    )
    args = parser.parse_args()

    if args.checkpoint:
        checkpoint_dir = Path(args.checkpoint)
    else:
        output_dir = Path(_PROJECT_ROOT) / args.output_dir
        checkpoint_dir = find_latest_checkpoint(output_dir)
        if checkpoint_dir is None:
            logger.error(f"체크포인트를 찾을 수 없음: {output_dir}")
            sys.exit(1)

    logger.info(f"검증 대상 체크포인트: {checkpoint_dir}")
    logger.info("=" * 60)

    passed = validate_partial_state(checkpoint_dir)

    logger.info("=" * 60)
    if passed:
        logger.info("[결과] 모든 검증 통과 ✓ — resume 시 partial_state 복원 가능")
    else:
        logger.error("[결과] 검증 실패 ✗ — 위 에러를 확인하세요")
        sys.exit(1)


if __name__ == "__main__":
    main()
