"""Pre-Stage 훈련용 DataCollator 모듈.

가변 길이 시퀀스를 배치 내 최대 길이로 dynamic padding하고,
labels의 padding 부분은 -100으로 마스킹한다.
"""

import logging
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class PreStageCollator:
    """Pre-Stage 훈련용 DataCollator.

    Dataset에서 반환된 샘플들을 배치로 묶는다.
    배치 내 최대 시퀀스 길이로 right-padding을 수행하며,
    labels의 padding 위치는 -100(loss 무시)으로 처리한다.

    Args:
        tokenizer: pad_token_id를 제공하는 AutoTokenizer.
        max_length: 최대 허용 시퀀스 길이. 이 길이를 초과하는 경우 잘라낸다.
    """

    tokenizer: AutoTokenizer
    max_length: int = 2048

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """샘플 리스트를 패딩된 배치 텐서로 변환한다.

        Args:
            batch: Dataset.__getitem__이 반환한 샘플 딕셔너리의 리스트.
                각 샘플은 "input_ids", "labels", "attention_mask" 키를 가진다.

        Returns:
            dict:
                - input_ids: shape $(B, L_{max})$ — right-padded.
                - labels: shape $(B, L_{max})$ — padding 위치는 -100.
                - attention_mask: shape $(B, L_{max})$ — padding 위치는 0.
        """
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            # Qwen 계열은 eos_token을 pad_token으로 사용하는 경우가 많음
            pad_id = self.tokenizer.eos_token_id

        input_ids_list = [sample["input_ids"] for sample in batch]
        labels_list = [sample["labels"] for sample in batch]
        attention_mask_list = [sample["attention_mask"] for sample in batch]

        # 배치 내 최대 길이 (max_length 초과 방지)
        max_len = min(max(seq.size(0) for seq in input_ids_list), self.max_length)

        padded_input_ids = []
        padded_labels = []
        padded_attention_masks = []

        for input_ids, labels, attention_mask in zip(
            input_ids_list, labels_list, attention_mask_list
        ):
            seq_len = input_ids.size(0)

            # max_length 초과 시 잘라냄
            if seq_len > max_len:
                input_ids = input_ids[:max_len]
                labels = labels[:max_len]
                attention_mask = attention_mask[:max_len]
                seq_len = max_len

            pad_len = max_len - seq_len

            # Right padding
            # input_ids: pad_id로 채움
            padded_input = torch.cat([
                input_ids,
                torch.full((pad_len,), pad_id, dtype=torch.long),
            ])

            # labels: -100으로 채움 (padding 위치는 loss 계산에서 제외)
            padded_label = torch.cat([
                labels,
                torch.full((pad_len,), -100, dtype=torch.long),
            ])

            # attention_mask: 0으로 채움 (padding은 attention 제외)
            padded_mask = torch.cat([
                attention_mask,
                torch.zeros(pad_len, dtype=torch.long),
            ])

            padded_input_ids.append(padded_input)
            padded_labels.append(padded_label)
            padded_attention_masks.append(padded_mask)

        return {
            "input_ids": torch.stack(padded_input_ids),           # (B, L_max)
            "labels": torch.stack(padded_labels),                  # (B, L_max)
            "attention_mask": torch.stack(padded_attention_masks), # (B, L_max)
        }
