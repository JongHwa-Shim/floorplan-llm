"""GRPO 훈련 패키지.

GDPO + 토큰 수준 신용할당 강화학습 구현.

공개 API:
    - GDPOTrainer: TRL GRPOTrainer 서브클래스 (GDPO + 토큰 신용할당 + GRPO adapter 저장/로드)
    - GRPOPromptDataset: GRPO 훈련용 프롬프트 Dataset
    - load_model_and_tokenizer: Hub+partial_state + SFT(frozen) + GRPO(trainable) 멀티 어댑터 로드
"""

from src.training.grpo.dataset import GRPOPromptDataset
from src.training.grpo.model_loader import load_model_and_tokenizer
from src.training.grpo.trainer import GDPOTrainer

__all__ = [
    "GDPOTrainer",
    "GRPOPromptDataset",
    "load_model_and_tokenizer",
]
