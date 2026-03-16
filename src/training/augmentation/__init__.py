"""데이터 증강 + 토크나이징 패키지.

Arrow 구조화 데이터에 변형/삭제 증강을 적용하고
커스텀 토큰 ID 시퀀스로 변환한다.

주요 공개 인터페이스:
    - AugmentationPipeline: 증강 + 토크나이징 파이프라인 (DataLoader에서 사용).
    - AugmentationConfig: 증강 파라미터 설정.
    - Vocab / load_vocab: 토큰↔ID 매핑.
    - decode_tokens: 토큰 ID → 가독성 문자열 (검증용).
    - format_origin: raw Arrow 데이터 → 가독성 문자열 (검증용).
    - format_sample_report: 3섹션 리포트 문자열 생성 (검증용).
"""

from src.training.augmentation.decoder import (
    decode_tokens,
    format_origin,
    format_sample_report,
)
from src.training.augmentation.pipeline import (
    AugmentationConfig,
    AugmentationPipeline,
    config_from_omegaconf,
)
from src.training.augmentation.tokenizer import (
    Vocab,
    load_vocab,
    to_row_oriented,
)

__all__ = [
    # 파이프라인
    "AugmentationPipeline",
    "AugmentationConfig",
    "config_from_omegaconf",
    # vocab
    "Vocab",
    "load_vocab",
    "to_row_oriented",
    # 검증용 디코더
    "decode_tokens",
    "format_origin",
    "format_sample_report",
]
