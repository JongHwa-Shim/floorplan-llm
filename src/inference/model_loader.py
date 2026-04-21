"""추론용 모델 로드 모듈.

Mod Record: 이전 구조에서는 단일 model_dir의 full model(merged)만 지원했음.
새 구조에서는 load_mode 분기를 지원한다:
  - "adapters": Hub NF4 + partial_state.pt + adapter 체인 (stage별 adapter 스태킹)
  - "merged": pre-merged full model 직접 로드 (merge_model.py 유틸로 사전 생성)

adapters 모드에서 중간 adapter는 base에 merge하고 마지막 adapter는 PeftModel로 로드한다.
이로써 모든 stage의 adapter 효과가 올바르게 합성된다.
"""

import logging
from pathlib import Path

import torch
from omegaconf import DictConfig
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.training.pre_stage.model_loader import load_model_with_partial_state

logger = logging.getLogger(__name__)


def load_model_for_inference(
    cfg: DictConfig,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """추론용 모델 + 토크나이저를 로드한다.

    cfg.inference.load_mode에 따라 두 가지 로드 방식 중 하나를 선택한다:
        - "adapters": Hub NF4 + partial_state.pt + adapter 체인
        - "merged": pre-merged full model 직접 bf16 로드

    Args:
        cfg: Hydra DictConfig. cfg.model, cfg.quantization, cfg.inference 섹션을 참조한다.

    Returns:
        tuple:
            - model: 추론 모드(eval)가 적용된 AutoModelForCausalLM 또는 PeftModel
            - tokenizer: 커스텀 토큰이 포함된 AutoTokenizer

    Raises:
        FileNotFoundError: 지정된 경로가 존재하지 않을 경우.
        ValueError: load_mode가 "adapters" 또는 "merged"가 아닌 경우.
    """
    load_mode = cfg.inference.get("load_mode", "merged")

    if load_mode == "adapters":
        model, tokenizer = _load_adapters_mode(cfg)
    elif load_mode == "merged":
        model, tokenizer = _load_merged_mode(cfg)
    else:
        raise ValueError(f"알 수 없는 load_mode: {load_mode!r}. 'adapters' 또는 'merged'여야 합니다.")

    model.eval()
    logger.info(
        "추론 모델 준비 완료 — vocab_size: %d, dtype: %s, load_mode: %s",
        model.config.vocab_size,
        next(model.parameters()).dtype,
        load_mode,
    )

    return model, tokenizer


# ---------------------------------------------------------------------------
# 내부 로드 함수
# ---------------------------------------------------------------------------

def _load_adapters_mode(
    cfg: DictConfig,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Hub NF4 + partial_state.pt + adapter 체인으로 모델을 구성한다.

    adapters 리스트의 순서대로 adapter를 적용한다:
        - 중간 adapter: base에 merge_and_unload()로 병합 (효과를 base에 누적)
        - 마지막 adapter: PeftModel로 로드 (forward 패스에 DoRA 효과 적용)

    Args:
        cfg: Hydra DictConfig. cfg.model.hub_id, cfg.model.pre_stage_dir,
             cfg.inference.adapters 섹션을 참조한다.

    Returns:
        tuple:
            - model: adapter 체인이 적용된 AutoModelForCausalLM 또는 PeftModel
            - tokenizer: 커스텀 토큰이 포함된 AutoTokenizer
    """
    pre_stage_dir = Path(cfg.model.pre_stage_dir)
    partial_state_path = pre_stage_dir / "partial_state.pt"

    if not partial_state_path.exists():
        raise FileNotFoundError(f"partial_state.pt를 찾을 수 없음: {partial_state_path}")

    logger.info(f"Hub 모델 로드 + partial_state.pt 주입: {cfg.model.hub_id}")
    model, tokenizer = load_model_with_partial_state(cfg, partial_state_path)

    adapters = list(cfg.inference.adapters)
    if not adapters:
        logger.warning("inference.adapters가 비어있음. partial_state만 주입된 base model 반환.")
        return model, tokenizer

    for i, adapter_cfg in enumerate(adapters):
        adapter_path = Path(adapter_cfg.path)
        adapter_name = adapter_cfg.get("name", f"adapter_{i}")
        is_last = (i == len(adapters) - 1)

        if not adapter_path.exists():
            raise FileNotFoundError(f"adapter 디렉토리를 찾을 수 없음: {adapter_path}")

        if is_last:
            # 마지막 adapter: PeftModel로 로드하여 forward 패스에 DoRA 효과 유지
            logger.info(f"마지막 adapter 로드 (PeftModel): {adapter_path}")
            model = PeftModel.from_pretrained(
                model,
                str(adapter_path),
                adapter_name=adapter_name,
                is_trainable=False,
            )
        else:
            # 중간 adapter: base에 merge하여 효과를 누적한 뒤 다음 adapter 적용 준비
            logger.info(f"중간 adapter 로드 후 base에 merge: {adapter_path}")
            peft_model = PeftModel.from_pretrained(
                model,
                str(adapter_path),
                adapter_name=adapter_name,
                is_trainable=False,
            )
            model = peft_model.merge_and_unload()
            logger.info(f"adapter '{adapter_name}' merge 완료")

    # Mod Record: PeftModel.from_pretrained으로 로드한 adapter 가중치와
    # attention bias가 float32로 유지될 수 있다. NF4 Params4bit를 제외한
    # 모든 float32 파라미터를 bf16으로 통일해야 generate() forward pass에서
    # "expected BFloat16 but found Float" 오류가 발생하지 않는다.
    try:
        from bitsandbytes.nn import Params4bit as _Params4bit
    except ImportError:
        _Params4bit = None

    cast_count = 0
    for param in model.parameters():
        if _Params4bit is not None and isinstance(param, _Params4bit):
            continue
        if param.data.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)
            cast_count += 1
    logger.info(f"비양자화 float32 파라미터 bf16 캐스팅 완료: {cast_count}개")

    return model, tokenizer


def _load_merged_mode(
    cfg: DictConfig,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """pre-merged full model을 bf16으로 직접 로드한다.

    merge_model.py 유틸로 사전 생성된 병합 모델을 로드한다.
    bf16으로 저장된 모델을 그대로 로드하여 추론에 사용한다.

    Args:
        cfg: Hydra DictConfig. cfg.model.model_dir, cfg.model.tokenizer_dir 섹션을 참조한다.

    Returns:
        tuple:
            - model: bf16 AutoModelForCausalLM
            - tokenizer: 커스텀 토큰이 포함된 AutoTokenizer
    """
    model_dir = Path(cfg.model.model_dir)
    tokenizer_dir = Path(cfg.model.get("tokenizer_dir", str(model_dir)))

    if not model_dir.exists():
        raise FileNotFoundError(f"모델 디렉토리를 찾을 수 없음: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    logger.info(f"토크나이저 로드 완료. vocab size: {len(tokenizer)}")

    logger.info(f"pre-merged bf16 모델 로드 중: {model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    logger.info("모델 로드 완료")

    return model, tokenizer
