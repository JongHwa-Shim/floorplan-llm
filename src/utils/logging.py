"""훈련 스크립트용 로깅 보조 유틸.

Hydra는 root logger에 file handler를 부착하지만, transformers / datasets / TRL은
자체 logging 모듈(`transformers.utils.logging`)을 통해 메시지를 발행하며 기본적으로
propagation이 비활성화되어 있다. 따라서 Trainer 내부의 train/eval/save 로그가
root logger로 전파되지 않아 Hydra의 file handler까지 도달하지 못하고 stderr로만 나간다.
결과적으로 `outputs/.../run_*.log`에는 훈련 준비 단계까지의 메시지만 남고
본 훈련 루프 로그가 통째로 누락된다.

`setup_library_logging_propagation()`을 main 함수 진입부에서 호출하면 transformers/TRL
의 모든 로그가 root logger로 전파되어 콘솔과 파일 양쪽에 일관되게 기록된다.
"""

from __future__ import annotations

import logging


def setup_library_logging_propagation(level: int = logging.INFO) -> None:
    """외부 라이브러리(transformers, TRL, datasets 등)의 로그를 root logger로 전파한다.

    이 함수는 다음 작업을 수행한다.
        1. `transformers.utils.logging.enable_propagation()` 호출 — root logger 전파 활성화
        2. `transformers.utils.logging.disable_default_handler()` 호출 — 자체 stderr handler
           제거하여 Hydra root logger의 stream handler와의 중복 출력 방지
        3. transformers/datasets/peft/trl/accelerate 로거의 level을 명시적으로 INFO 이상으로
           설정 (라이브러리 기본값이 WARNING인 경우 INFO 메시지가 누락되는 것 방지)

    Args:
        level: 외부 라이브러리 로거에 적용할 logging level. 기본 ``logging.INFO``.

    Raises:
        없음. transformers가 설치되어 있지 않거나 API가 바뀌어도 silently 무시한다
        (훈련 자체를 막지 않기 위함).
    """
    # transformers의 자체 로깅 모듈은 root propagation이 기본 OFF이므로 명시적으로 켠다
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.enable_propagation()
        hf_logging.disable_default_handler()
        hf_logging.set_verbosity(level)
    except Exception:
        # transformers 미설치 또는 API 변경 시에도 훈련은 계속 진행
        pass

    # 일부 라이브러리 로거가 WARNING으로 초기화되어 INFO 메시지가 누락되는 케이스 대응
    for name in ("transformers", "datasets", "peft", "trl", "accelerate"):
        logging.getLogger(name).setLevel(level)
