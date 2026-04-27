"""Embedding Alignment 훈련 패키지.

새 커스텀 토큰 embedding과 lm_head 행만 훈련하는
워밍업 단계(Embedding Alignment)의 공개 인터페이스를 정의한다.
"""

from src.training.embed_align.collator import EmbedAlignCollator
from src.training.embed_align.dataset import EmbedAlignDataset
from src.training.embed_align.model_loader import load_model_and_tokenizer
from src.training.embed_align.trainer import build_trainer, build_training_arguments

__all__ = [
    "EmbedAlignDataset",
    "EmbedAlignCollator",
    "load_model_and_tokenizer",
    "build_trainer",
    "build_training_arguments",
]
