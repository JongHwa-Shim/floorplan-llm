"""Pre-Stage 훈련용 PyTorch Dataset 모듈.

Arrow 데이터셋에서 샘플을 로드하고, 증강 파이프라인을 적용한 뒤
Chat Template으로 감싸 input_ids / labels / attention_mask를 반환한다.

Chat Template 적용 방식:
    condition_tokens / output_tokens (토큰 ID 리스트)를 tokenizer.decode()로
    문자열로 변환한 뒤, apply_chat_template()으로 최종 input_ids를 생성한다.
    label 마스킹을 위해 system+user 파트 길이를 미리 계산한다.
"""

import logging
from pathlib import Path
from typing import Any

import torch
from datasets import load_from_disk
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.training.augmentation.pipeline import AugmentationPipeline, config_from_omegaconf
from src.training.augmentation.tokenizer import load_vocab

logger = logging.getLogger(__name__)

# 평면도 생성기 시스템 프롬프트
SYSTEM_PROMPT = (
    "You are a floor plan generator. "
    "Given room conditions, generate complete floorplan coordinates."
)


class PreStageDataset(Dataset):
    """Pre-Stage 훈련용 Dataset.

    Arrow 포맷의 평면도 데이터셋을 읽어 증강 파이프라인을 적용하고,
    chat template으로 감싼 입력/레이블 텐서를 반환한다.

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
        self.max_length: int = cfg.data.max_length
        self.split = split

        # Arrow 데이터셋 로드
        arrow_dir = Path(cfg.data.arrow_dir)
        if not arrow_dir.exists():
            raise FileNotFoundError(f"Arrow 데이터셋 경로를 찾을 수 없음: {arrow_dir}")

        dataset_dict = load_from_disk(str(arrow_dir))
        if split not in dataset_dict:
            raise ValueError(f"split '{split}'이 데이터셋에 없음. 가능한 split: {list(dataset_dict.keys())}")
        self.dataset = dataset_dict[split]
        logger.info(f"[{split}] Arrow 데이터셋 로드 완료: {len(self.dataset)}개 샘플")

        # Vocab 로드 (pad_token_id 등 필요)
        vocab = load_vocab(cfg.model.vocab_extension, cfg.model.tokenizer_dir)

        # 증강 파이프라인 초기화
        aug_config = config_from_omegaconf(cfg.augmentation)
        self.pipeline = AugmentationPipeline(vocab, aug_config, seed=seed)

        # label 마스킹용: system+user 파트만 있는 prefix의 토큰 길이를 미리 계산
        # apply_chat_template에서 assistant 응답 직전까지의 길이를 기준으로 마스킹
        self._user_prefix_ids = self._compute_user_prefix_ids()

    def _compute_user_prefix_ids(self) -> list[int]:
        """system + user 파트까지의 토큰 ID 접두사를 계산한다.

        dummy content로 template을 적용하고 add_generation_prompt=True 로 설정하면
        assistant 응답 시작 직전까지의 토큰을 얻을 수 있다.
        실제 샘플마다 content가 달라지므로 이 값은 prefix 길이 계산에만 사용한다.

        Returns:
            system + user 메시지 + assistant 턴 시작 토큰까지의 ID 리스트.
        """
        # Mod Record: apply_chat_template(tokenize=True)의 반환 타입이 transformers 버전/
        # 토크나이저 백엔드에 따라 BatchEncoding, tokenizers.Encoding, list 등으로 달라져
        # torch.tensor 변환 시 TypeError가 발생했다.
        # tokenize=False로 렌더링된 문자열을 받은 뒤 encode()로 명시적 토크나이징하여 해결한다.
        # add_generation_prompt=True: assistant 응답 시작 토큰(<|im_start|>assistant\n)까지 포함
        chat_str: str = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "PLACEHOLDER"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix_ids: list[int] = self.tokenizer.encode(chat_str, add_special_tokens=False)
        return prefix_ids

    def __len__(self) -> int:
        """데이터셋 샘플 수를 반환한다.

        Returns:
            전체 샘플 수.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """단일 샘플을 로드하고 증강 + chat template 적용 후 텐서를 반환한다.

        Args:
            idx: 샘플 인덱스.

        Returns:
            dict:
                - input_ids: chat template이 적용된 전체 토큰 ID. shape: $(L,)$
                - labels: assistant 응답 부분만 유효한 label. shape: $(L,)$
                  (system+user 파트는 -100으로 마스킹)
                - attention_mask: 유효 토큰 마스크. shape: $(L,)$
        """
        raw_sample: dict[str, Any] = self.dataset[idx]

        # 증강 파이프라인 적용: 변형 증강 → 삭제 증강 → 토크나이징
        condition_tokens, output_tokens = self.pipeline(raw_sample)

        # 토큰 ID → 문자열 디코딩 (special token 포함)
        decoded_condition = self.tokenizer.decode(
            condition_tokens, skip_special_tokens=False
        )
        decoded_output = self.tokenizer.decode(
            output_tokens, skip_special_tokens=False
        )

        # Chat template 적용 (tokenize=False → encode()로 명시적 토크나이징)
        # add_generation_prompt=False: 훈련용이므로 assistant 응답까지 전부 포함
        full_chat_str: str = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": decoded_condition},
                {"role": "assistant", "content": decoded_output},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        input_ids: list[int] = self.tokenizer.encode(full_chat_str, add_special_tokens=False)

        # max_length 초과 시 잘라냄
        input_ids = input_ids[: self.max_length]

        # label 마스킹: system + user 파트 (add_generation_prompt=True 길이)를 -100으로 처리
        # "PLACEHOLDER" 대신 실제 condition을 넣어 정확한 prefix 길이 계산
        prefix_chat_str: str = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": decoded_condition},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        actual_prefix_ids: list[int] = self.tokenizer.encode(prefix_chat_str, add_special_tokens=False)
        prefix_length = len(actual_prefix_ids)

        labels = [-100] * prefix_length + input_ids[prefix_length:]

        # max_length 초과 시 labels도 동일하게 자름
        labels = labels[: self.max_length]

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
