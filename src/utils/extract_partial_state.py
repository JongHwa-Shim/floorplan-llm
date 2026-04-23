"""병합된 model.safetensors에서 커스텀 토큰 가중치만 추출하여 partial_state.pt로 저장한다.

Pre-Stage 훈련 방식 변경 전, 최종 체크포인트는 merge_and_restore() → save_pretrained()를 통해
new_embed + frozen base 가중치를 단일 model.safetensors에 합쳐 저장했다.
이 모듈은 그 합쳐진 파일에서 새 토큰 행(new_token_ids)만 슬라이싱하여
현재 코드와 호환되는 partial_state.pt를 복원한다.
"""

import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _load_new_token_ids(vocab_extension_path: Path) -> list[int]:
    """vocab_extension.json에서 새로 추가된 커스텀 토큰 ID 목록을 추출한다.

    Args:
        vocab_extension_path: vocab_extension.json 파일 경로.

    Returns:
        정렬된 커스텀 토큰 ID 리스트 (base_vocab_size 이상인 ID).

    Raises:
        FileNotFoundError: vocab_extension.json이 없을 때.
        KeyError: 필수 필드가 없을 때.
    """
    if not vocab_extension_path.exists():
        raise FileNotFoundError(f"vocab_extension.json 없음: {vocab_extension_path}")

    with open(vocab_extension_path, encoding="utf-8") as f:
        vocab_ext = json.load(f)

    base_vocab_size: int = vocab_ext["base_vocab_size"]
    token_to_id: dict[str, int] = vocab_ext["token_to_id"]

    new_ids = [tid for tid in token_to_id.values() if tid >= base_vocab_size]
    return sorted(new_ids)


def _find_safetensors_files(
    checkpoint_dir: Path,
) -> tuple[list[Path], dict[str, str] | None]:
    """체크포인트 디렉토리에서 safetensors 파일을 탐색한다.

    단일 파일과 sharded 파일 모두 지원한다.

    Args:
        checkpoint_dir: model.safetensors 또는 shard 파일이 있는 디렉토리.

    Returns:
        (safetensors 파일 목록, shard index 매핑 또는 None)
        shard index 매핑 형식: {"weight_key": "파일명", ...}

    Raises:
        FileNotFoundError: safetensors 파일을 찾을 수 없을 때.
    """
    index_path = checkpoint_dir / "model.safetensors.index.json"
    single_path = checkpoint_dir / "model.safetensors"

    # sharded 포맷 우선 확인
    if index_path.exists():
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        weight_map: dict[str, str] = index["weight_map"]
        shard_files = sorted(set(weight_map.values()))
        paths = [checkpoint_dir / fname for fname in shard_files]
        logger.info("sharded safetensors 감지: %d개 shard", len(paths))
        return paths, weight_map

    # 단일 파일
    if single_path.exists():
        logger.info("단일 model.safetensors 감지: %s", single_path)
        return [single_path], None

    raise FileNotFoundError(
        f"safetensors 파일을 찾을 수 없음: {checkpoint_dir}\n"
        "model.safetensors 또는 model.safetensors.index.json이 필요합니다."
    )


def _read_tensor_from_safetensors(
    safetensors_paths: list[Path],
    weight_map: dict[str, str] | None,
    key: str,
) -> torch.Tensor:
    """safetensors 파일(들)에서 특정 키의 텐서를 읽는다.

    Args:
        safetensors_paths: safetensors 파일 경로 목록.
        weight_map: sharded 포맷의 키→파일명 매핑 (단일 파일이면 None).
        key: 읽을 텐서 키.

    Returns:
        CPU 텐서.

    Raises:
        KeyError: 키가 safetensors 파일에 없을 때.
    """
    from safetensors import safe_open  # 지연 임포트 (선택적 의존성)

    if weight_map is not None:
        # sharded: 해당 키가 있는 shard 파일만 열기
        if key not in weight_map:
            raise KeyError(key)
        target_file = safetensors_paths[0].parent / weight_map[key]
        with safe_open(target_file, framework="pt", device="cpu") as f:
            return f.get_tensor(key)
    else:
        # 단일 파일
        with safe_open(safetensors_paths[0], framework="pt", device="cpu") as f:
            if key not in list(f.keys()):
                raise KeyError(key)
            return f.get_tensor(key)


def extract_partial_state(
    checkpoint_dir: Path,
    vocab_extension_path: Path,
    output_path: Path,
    dtype: torch.dtype | None = None,
) -> dict:
    """병합된 model.safetensors에서 커스텀 토큰 가중치를 추출하여 partial_state.pt로 저장한다.

    merge_and_restore() 후 save_pretrained()로 저장된 model.safetensors에는
    전체 vocab 크기(embed_tokens, lm_head)가 포함되어 있다. 이 함수는
    new_token_ids에 해당하는 행만 슬라이싱하여 SFT 호환 partial_state.pt를 생성한다.

    Args:
        checkpoint_dir: model.safetensors (또는 shard) 파일이 있는 디렉토리.
        vocab_extension_path: vocab_extension.json 경로. new_token_ids 결정에 사용.
        output_path: 저장할 partial_state.pt 경로. 부모 디렉토리가 없으면 자동 생성.
        dtype: 저장 dtype. None이면 safetensors 원본 dtype 그대로 사용.
            예: torch.bfloat16, torch.float32.

    Returns:
        저장된 partial_state dict.
        {"new_embed": Tensor, "new_lm_head": Tensor, "new_token_ids": list[int]}

    Raises:
        FileNotFoundError: safetensors 파일 또는 vocab_extension.json이 없을 때.
        KeyError: embed_tokens / lm_head 키가 safetensors에 없을 때.
        ValueError: 추출된 텐서의 shape이 예상과 다를 때.
    """
    checkpoint_dir = Path(checkpoint_dir)
    vocab_extension_path = Path(vocab_extension_path)
    output_path = Path(output_path)

    # 1. new_token_ids 로드
    new_token_ids = _load_new_token_ids(vocab_extension_path)
    num_new = len(new_token_ids)
    logger.info("커스텀 토큰 %d개 확인: ID 범위 [%d, %d]", num_new, new_token_ids[0], new_token_ids[-1])

    # 2. safetensors 파일 탐색
    safetensors_paths, weight_map = _find_safetensors_files(checkpoint_dir)

    # 사용 가능한 키 목록 (에러 메시지용)
    def _get_all_keys() -> list[str]:
        from safetensors import safe_open
        keys = []
        for p in safetensors_paths:
            with safe_open(p, framework="pt", device="cpu") as f:
                keys.extend(f.keys())
        return keys

    # 3. embed_tokens와 lm_head 텐서 읽기
    embed_key = "model.embed_tokens.weight"
    lm_head_key = "lm_head.weight"

    try:
        full_embed = _read_tensor_from_safetensors(safetensors_paths, weight_map, embed_key)
    except KeyError:
        all_keys = _get_all_keys()
        embed_candidates = [k for k in all_keys if "embed" in k.lower()]
        raise KeyError(
            f"'{embed_key}' 키가 safetensors에 없습니다.\n"
            f"embed 관련 키: {embed_candidates}\n"
            f"전체 키 수: {len(all_keys)}"
        )

    try:
        full_lm_head = _read_tensor_from_safetensors(safetensors_paths, weight_map, lm_head_key)
    except KeyError:
        all_keys = _get_all_keys()
        lm_head_candidates = [k for k in all_keys if "lm_head" in k.lower() or "head" in k.lower()]
        raise KeyError(
            f"'{lm_head_key}' 키가 safetensors에 없습니다.\n"
            f"lm_head 관련 키: {lm_head_candidates}"
        )

    logger.info(
        "로드 완료 — embed: %s %s, lm_head: %s %s",
        full_embed.shape, full_embed.dtype,
        full_lm_head.shape, full_lm_head.dtype,
    )

    # 4. new_token_ids 행 슬라이싱
    new_embed = full_embed[new_token_ids].clone()    # (num_new, hidden)
    new_lm_head = full_lm_head[new_token_ids].clone()  # (num_new, hidden)

    # shape 검증
    if new_embed.shape[0] != num_new:
        raise ValueError(f"new_embed 행 수 불일치: {new_embed.shape[0]} != {num_new}")
    if new_lm_head.shape[0] != num_new:
        raise ValueError(f"new_lm_head 행 수 불일치: {new_lm_head.shape[0]} != {num_new}")
    if new_embed.shape[1] != new_lm_head.shape[1]:
        raise ValueError(
            f"hidden_dim 불일치: embed {new_embed.shape[1]} vs lm_head {new_lm_head.shape[1]}"
        )

    # 5. dtype 변환 (옵션)
    if dtype is not None:
        new_embed = new_embed.to(dtype)
        new_lm_head = new_lm_head.to(dtype)
        logger.info("dtype 변환: %s", dtype)

    logger.info(
        "추출 완료 — new_embed: %s %s, new_lm_head: %s %s",
        new_embed.shape, new_embed.dtype,
        new_lm_head.shape, new_lm_head.dtype,
    )
    logger.info(
        "new_embed 값 범위: [%.4f, %.4f]",
        new_embed.float().min().item(),
        new_embed.float().max().item(),
    )
    logger.info(
        "new_lm_head 값 범위: [%.4f, %.4f]",
        new_lm_head.float().min().item(),
        new_lm_head.float().max().item(),
    )

    # 6. 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_state = {
        "new_embed": new_embed.cpu(),
        "new_lm_head": new_lm_head.cpu(),
        "new_token_ids": new_token_ids,
    }
    torch.save(partial_state, output_path)
    logger.info("partial_state.pt 저장 완료: %s", output_path)

    return partial_state
