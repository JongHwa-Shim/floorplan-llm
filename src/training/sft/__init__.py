"""SFT(Supervised Fine-Tuning) 훈련 모듈 공개 인터페이스.

Pre-Stage에서 워밍업된 로컬 모델에 LoRA를 적용하여 전체 레이어 fine-tuning을 수행한다.
데이터셋 및 DataCollator는 Pre-Stage와 동일한 포맷이므로 PreStageDataset을 재활용한다.
"""

from src.training.sft.model_loader import load_model_and_tokenizer, merge_lora_and_save
from src.training.sft.trainer import build_trainer
from src.training.pre_stage.dataset import PreStageDataset as SFTDataset

__all__ = [
    "load_model_and_tokenizer",
    "merge_lora_and_save",
    "build_trainer",
    "SFTDataset",
]
