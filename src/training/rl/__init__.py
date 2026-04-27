"""RL(강화학습) 훈련 모듈 공개 인터페이스.

GDPO + 토큰 수준 신용할당 방식의 GRPO 기반 강화학습 구현.
SFT(frozen) + RL(trainable) 멀티 어댑터 스태킹 구조를 사용한다.
"""

from src.training.rl.model_loader import load_model_and_tokenizer
from src.training.rl.trainer import RLTrainer
from src.training.rl.dataset import RLPromptDataset

__all__ = [
    "load_model_and_tokenizer",
    "RLTrainer",
    "RLPromptDataset",
]
