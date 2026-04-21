"""기존 pre_stage/final/model.safetensors에서 partial_state.pt를 추출하는 스크립트.

새 아키텍처로 전환하기 전에 이미 저장된 pre_stage/final 모델에서
커스텀 토큰 가중치(new_embed/new_lm_head)를 partial_state.pt 형태로 추출한다.

safetensors를 직접 열어 embed_tokens.weight/lm_head.weight의 new_token_ids 행만
추출하므로 전체 모델을 GPU에 올릴 필요가 없다.

사용법:
    uv run python scripts/utils/extract_partial_state.py \\
        --model_path data/models/Qwen2.5-Coder-7B/checkpoints/pre_stage/final \\
        --vocab_extension data/models/Qwen2.5-Coder-7B/tokenization/vocab_extension.json \\
        --output_path data/models/Qwen2.5-Coder-7B/checkpoints/pre_stage/final/partial_state.pt \\
        --dtype bfloat16
"""

import argparse
import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def extract_partial_state(
    model_path: Path,
    vocab_extension_path: Path,
    output_path: Path,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """model.safetensors에서 new_token_ids에 해당하는 embed/lm_head 행을 추출한다.

    safetensors를 직접 열어 embed_tokens.weight와 lm_head.weight의 커스텀 토큰 행만
    읽으므로 전체 7B 모델을 메모리에 올리지 않아도 된다.

    Args:
        model_path: pre_stage/final 디렉토리 경로.
        vocab_extension_path: vocab_extension.json 경로 (new_token_ids 조회용).
        output_path: 저장할 partial_state.pt 경로.
        dtype: 저장 dtype (기본값: bfloat16).

    Raises:
        FileNotFoundError: model_path 또는 vocab_extension_path가 없을 경우.
        KeyError: model.safetensors에 필요한 키가 없을 경우.
    """
    from safetensors import safe_open

    if not model_path.exists():
        raise FileNotFoundError(f"모델 디렉토리를 찾을 수 없음: {model_path}")
    if not vocab_extension_path.exists():
        raise FileNotFoundError(f"vocab_extension.json을 찾을 수 없음: {vocab_extension_path}")

    # vocab_extension에서 new_token_ids 로드
    with open(vocab_extension_path, encoding="utf-8") as f:
        vocab_ext = json.load(f)
    # Mod Record: vocab_extension.json 실제 키는 "token_to_id" (token_id_mapping 아님)
    mapping_key = "token_id_mapping" if "token_id_mapping" in vocab_ext else "token_to_id"
    new_token_ids = sorted(vocab_ext[mapping_key].values())
    logger.info(f"new_token_ids 로드 완료: {len(new_token_ids)}개 커스텀 토큰")

    # sharded 모델인지 확인 (model.safetensors.index.json 존재 여부)
    index_path = model_path / "model.safetensors.index.json"
    single_shard_path = model_path / "model.safetensors"

    if index_path.exists():
        # 샤딩된 모델: index.json으로 embed_tokens/lm_head가 있는 샤드 파일 탐색
        logger.info("샤딩된 모델 감지. index.json에서 shard 파일 탐색 중...")
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)

        weight_map = index["weight_map"]
        embed_shard = model_path / weight_map["model.embed_tokens.weight"]
        lm_head_shard = model_path / weight_map.get("lm_head.weight", weight_map.get("model.embed_tokens.weight"))
    elif single_shard_path.exists():
        embed_shard = single_shard_path
        lm_head_shard = single_shard_path
        logger.info("단일 shard 모델 감지.")
    else:
        raise FileNotFoundError(f"model.safetensors (또는 index.json)를 찾을 수 없음: {model_path}")

    # embed_tokens.weight에서 new_token_ids 행 추출
    logger.info(f"embed_tokens.weight 추출 중: {embed_shard}")
    with safe_open(str(embed_shard), framework="pt", device="cpu") as f:
        embed_weight = f.get_tensor("model.embed_tokens.weight")
    new_embed = embed_weight[new_token_ids].to(dtype)
    logger.info(f"new_embed shape: {new_embed.shape}, dtype: {new_embed.dtype}")

    # lm_head.weight에서 new_token_ids 행 추출
    logger.info(f"lm_head.weight 추출 중: {lm_head_shard}")
    with safe_open(str(lm_head_shard), framework="pt", device="cpu") as f:
        lm_head_key = "lm_head.weight"
        if lm_head_key not in f.keys():
            # tied weights: lm_head는 embed_tokens와 동일한 가중치를 사용하는 경우
            logger.warning("lm_head.weight 없음 (tied weights). embed_tokens.weight로 대체합니다.")
            lm_head_weight = embed_weight
        else:
            lm_head_weight = f.get_tensor(lm_head_key)
    new_lm_head = lm_head_weight[new_token_ids].to(dtype)
    logger.info(f"new_lm_head shape: {new_lm_head.shape}, dtype: {new_lm_head.dtype}")

    # partial_state.pt 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "new_embed": new_embed,
            "new_lm_head": new_lm_head,
            "new_token_ids": new_token_ids,
        },
        str(output_path),
    )
    logger.info(f"partial_state.pt 저장 완료: {output_path}")
    logger.info(f"  new_embed: {new_embed.shape}, {new_embed.dtype}")
    logger.info(f"  new_lm_head: {new_lm_head.shape}, {new_lm_head.dtype}")
    logger.info(f"  new_token_ids: {len(new_token_ids)}개")


def main() -> None:
    """커맨드라인 인터페이스 진입점."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="pre_stage/final/model.safetensors에서 partial_state.pt 추출"
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("data/models/Qwen2.5-Coder-7B/checkpoints/pre_stage/final"),
        help="pre_stage/final 디렉토리 경로",
    )
    parser.add_argument(
        "--vocab_extension",
        type=Path,
        default=Path("data/models/Qwen2.5-Coder-7B/tokenization/vocab_extension.json"),
        help="vocab_extension.json 경로",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("data/models/final_checkpoints/Qwen2.5-Coder-7B/pre_stage/final/partial_state.pt"),
        help="저장할 partial_state.pt 경로",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="저장 dtype (기본값: bfloat16)",
    )

    args = parser.parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    extract_partial_state(
        model_path=args.model_path,
        vocab_extension_path=args.vocab_extension,
        output_path=args.output_path,
        dtype=dtype_map[args.dtype],
    )


if __name__ == "__main__":
    main()
