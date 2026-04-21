"""GRPO용 프롬프트 Dataset 모듈.

SFT Dataset과 달리 assistant 응답(output_tokens)을 생성하지 않는다.
모델이 GRPO 훈련 중 직접 completion을 생성하므로,
이 Dataset은 프롬프트(조건) + 메타데이터만 반환한다.

메타데이터는 보상 계산에 필요한 정보를 담고 있으며,
TRL reward function에 추가 컬럼으로 자동 전달된다.
"""

import logging
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.training.augmentation.pipeline import AugmentationPipeline, config_from_omegaconf
from src.training.augmentation.tokenizer import Vocab, load_vocab, to_row_oriented

logger = logging.getLogger(__name__)

# 평면도 생성기 시스템 프롬프트 (SFT와 동일)
SYSTEM_PROMPT = (
    "You are a floor plan generator. "
    "Given room conditions, generate complete floorplan coordinates."
)


class GRPOPromptDataset(Dataset):
    """GRPO 훈련용 프롬프트 Dataset.

    Arrow 포맷 데이터셋에서 샘플을 로드하고 증강을 적용한 뒤,
    chat template으로 감싼 프롬프트 문자열과 보상 계산용 메타데이터를 반환한다.

    출력 레이블은 생성하지 않는다 (모델이 GRPO 중 직접 생성).
    증강은 입력 조건에만 적용하며, 메타데이터는 증강 전 원본 샘플에서 추출한다.

    Args:
        cfg: Hydra DictConfig. cfg.data, cfg.augmentation, cfg.model 섹션 참조.
        tokenizer: 커스텀 토큰이 포함된 AutoTokenizer.
        split: 사용할 데이터셋 split ("train", "validation", "test").
        seed: 증강 파이프라인의 랜덤 시드.
    """

    def __init__(
        self,
        cfg: DictConfig,
        tokenizer: AutoTokenizer,
        split: str = "train",
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_prompt_length: int = cfg.data.max_prompt_length
        self.split = split

        # Arrow 데이터셋 로드
        arrow_dir = Path(cfg.data.arrow_dir)
        if not arrow_dir.exists():
            raise FileNotFoundError(f"Arrow 데이터셋 경로를 찾을 수 없음: {arrow_dir}")

        dataset_dict = load_from_disk(str(arrow_dir))
        if split not in dataset_dict:
            raise ValueError(
                f"split '{split}'이 데이터셋에 없음. "
                f"가능한 split: {list(dataset_dict.keys())}"
            )
        self.dataset = dataset_dict[split]
        logger.info(f"[{split}] Arrow 데이터셋 로드 완료: {len(self.dataset)}개 샘플")

        # Vocab 로드 (파서에서 필요)
        self.vocab: Vocab = load_vocab(cfg.model.vocab_extension, cfg.model.tokenizer_dir)

        # 증강 파이프라인 초기화 (입력 조건 증강용)
        aug_config = config_from_omegaconf(cfg.augmentation)
        self.pipeline = AugmentationPipeline(self.vocab, aug_config, seed=seed)

    def __len__(self) -> int:
        """데이터셋 샘플 수를 반환한다.

        Returns:
            전체 샘플 수.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """단일 샘플을 로드하고 증강 + chat template 적용 후 반환한다.

        메타데이터는 증강 전 원본 샘플에서 추출한다.
        TRL GRPOTrainer는 이 dict의 모든 키를 reward function에 전달한다.

        Args:
            idx: 샘플 인덱스.

        Returns:
            dict:
                - prompt (list[dict]): chat template 형식의 프롬프트.
                    TRL이 내부적으로 chat template을 적용한다.
                - metadata (dict): 보상 계산용 메타데이터.
                    - total_rooms (int): outline 제외 전체 방 개수.
                    - type_counts (dict[str, int]): 타입별 방 개수.
                    - edges (list[dict]): 엣지 조건.
                    - spatial (list[dict]): 공간 관계 조건.
                    - rooms (list[dict]): 입력 방 정보 (헝가리안 매칭용).
        """
        raw_sample: dict[str, Any] = self.dataset[idx]

        # row-oriented 변환
        sample = to_row_oriented(raw_sample)

        # 원본 샘플에서 메타데이터 추출 (증강 전)
        metadata = _extract_metadata(sample)

        # 증강 파이프라인 적용 (condition_tokens만 생성, output_tokens 불필요)
        condition_tokens, _ = self.pipeline(raw_sample)

        # 토큰 ID → 문자열 디코딩
        decoded_condition = self.tokenizer.decode(
            condition_tokens, skip_special_tokens=False
        )

        # TRL 형식의 conversational prompt 반환
        # TRL이 내부적으로 apply_chat_template()을 처리함
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": decoded_condition},
        ]

        return {
            "prompt": prompt,
            "metadata": metadata,
        }


def _extract_metadata(sample: dict) -> dict:
    """row-oriented 샘플에서 보상 계산용 메타데이터를 추출한다.

    outline 제외 방 정보 및 edge/spatial 조건을 수집한다.

    Args:
        sample: to_row_oriented()의 반환값.

    Returns:
        메타데이터 딕셔너리:
            - total_rooms (int): outline 제외 전체 방 개수.
            - type_counts (dict[str, int]): 타입별 방 개수.
            - edges (list[dict]): 엣지 조건 (pair, door).
            - spatial (list[dict]): 공간 관계 조건 (rid_a, rid_b, direction).
            - rooms (list[dict]): outline 포함 전체 방 정보 (rid, type, coords).
    """
    rooms = sample.get("rooms", [])
    non_outline = [r for r in rooms if r.get("type") != "outline"]

    # 타입별 개수 집계
    type_counts: dict[str, int] = {}
    for room in non_outline:
        t = room.get("type", "")
        type_counts[t] = type_counts.get(t, 0) + 1

    return {
        "total_rooms": len(non_outline),
        "type_counts": type_counts,
        "edges": sample.get("edges", []),
        "spatial": sample.get("spatial", []),
        "rooms": rooms,  # outline 포함 (rid 매핑에 필요)
    }
