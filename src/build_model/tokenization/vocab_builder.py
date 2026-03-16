"""Vocabulary 빌더 모듈.

Pretrained tokenizer를 로드하고 커스텀 토큰을 추가한 뒤,
토큰→ID 매핑을 vocab_extension.json으로 저장하고
확장된 tokenizer를 디렉토리에 저장한다.
"""

import json
import logging
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from src.build_model.tokenization.token_definitions import build_token_list, flatten_token_list

logger = logging.getLogger(__name__)


def build_vocab(
    model_name: str,
    merge_config_path: Path,
    output_dir: Path,
    max_rid: int = 15,
    max_coord_x: int = 255,
    max_coord_y: int = 255,
) -> dict:
    """Pretrained tokenizer에 커스텀 토큰을 추가하고 결과를 저장한다.

    Args:
        model_name: HuggingFace Hub 모델 이름.
            예: "Qwen/Qwen2.5-Coder-7B"
        merge_config_path: room_type_merge.json 파일 경로.
        output_dir: 출력 디렉토리.
        max_rid: 방 인스턴스 ID 최대값 (<RID:0>~<RID:max_rid>).
        max_coord_x: X 좌표 최대값 (<X:0>~<X:max_coord_x>).
        max_coord_y: Y 좌표 최대값 (<Y:0>~<Y:max_coord_y>).
            - {output_dir}/vocab_extension.json: 토큰→ID 매핑
            - {output_dir}/tokenizer 파일들: 확장된 tokenizer save_pretrained() 결과
            vocab_extension.json과 tokenizer 파일들이 같은 디렉토리에 저장된다.

    Returns:
        vocab_extension.json에 저장되는 딕셔너리와 동일한 내용.

    Raises:
        FileNotFoundError: merge_config_path가 존재하지 않을 때.
    """
    output_dir = Path(output_dir)
    # vocab_extension.json과 tokenizer 파일 모두 tokenizer/ 디렉토리에 저장
    tokenizer_save_dir = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pretrained tokenizer 로드
    logger.info("tokenizer 로드 중: %s", model_name)
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    base_vocab_size = len(tokenizer)
    logger.info("기본 vocab 크기: %d", base_vocab_size)

    # 2. 카테고리별 토큰 목록 구성
    categories = build_token_list(
        merge_config_path,
        max_rid=max_rid,
        max_coord_x=max_coord_x,
        max_coord_y=max_coord_y,
    )
    all_tokens = flatten_token_list(categories)
    logger.info("추가 예정 커스텀 토큰 수: %d", len(all_tokens))

    # 3. 기존 tokenizer와 중복 토큰 확인 (이미 존재하는 토큰은 제외)
    existing_vocab = tokenizer.get_vocab()
    new_tokens = [t for t in all_tokens if t not in existing_vocab]
    skipped = len(all_tokens) - len(new_tokens)
    if skipped > 0:
        logger.warning(
            "기존 vocab에 이미 존재하는 토큰 %d개 건너뜀", skipped
        )

    # 4. 토큰 추가
    #    - structure_special 카테고리(PAD)는 add_special_tokens()로 등록
    #    - 나머지는 add_tokens()로 등록
    special_tokens = [t for t in categories["structure_special"] if t not in existing_vocab]
    normal_tokens = [
        t for t in new_tokens if t not in categories["structure_special"]
    ]

    added_special = 0
    if special_tokens:
        added_special = tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}
        )
        logger.info("special token 추가: %d개", added_special)

    added_normal = tokenizer.add_tokens(normal_tokens)
    logger.info("일반 커스텀 토큰 추가: %d개", added_normal)

    total_added = added_special + added_normal
    new_vocab_size = len(tokenizer)
    logger.info(
        "vocab 확장 완료: %d → %d (+%d)",
        base_vocab_size,
        new_vocab_size,
        total_added,
    )

    # 5. 토큰 → ID 매핑 추출
    token_to_id: dict[str, int] = {}
    for token in all_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        token_to_id[token] = token_id

    id_to_token: dict[str, str] = {str(v): k for k, v in token_to_id.items()}

    # 6. 카테고리별 토큰 → ID 매핑 구성 (메타정보 포함)
    category_token_ids: dict[str, dict[str, int]] = {}
    for cat_name, tokens in categories.items():
        category_token_ids[cat_name] = {
            t: tokenizer.convert_tokens_to_ids(t) for t in tokens
        }

    # 7. vocab_extension.json 저장
    extension_data = {
        "model_name": model_name,
        "base_vocab_size": base_vocab_size,
        "new_vocab_size": new_vocab_size,
        "total_added": total_added,
        "categories": {cat: tokens for cat, tokens in categories.items()},
        "category_token_ids": category_token_ids,
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
    }

    json_path = output_dir / "vocab_extension.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(extension_data, f, ensure_ascii=False, indent=2)
    logger.info("vocab_extension.json 저장 완료: %s", json_path)

    # 8. 확장된 tokenizer 저장
    tokenizer.save_pretrained(tokenizer_save_dir)
    logger.info("확장된 tokenizer 저장 완료: %s", tokenizer_save_dir)

    return extension_data
