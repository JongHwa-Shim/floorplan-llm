"""추론용 모델 로드 모듈.

Mod Record: 이전 구조에서는 단일 model_dir의 full model(merged)만 지원했음.
새 구조에서는 load_mode 분기를 지원한다:
  - "adapters": Hub NF4 + partial_state.pt + adapter 체인 (stage별 adapter 스태킹)
  - "merged": pre-merged full model 직접 로드 (merge_model.py 유틸로 사전 생성)

Mod Record: adapters 모드에서 이전에는 중간 adapter를 base에 merge_and_unload()로 병합했으나,
PEFT의 named adapter API(load_adapter)를 사용해 모든 adapter를 merge 없이 별도 유지하도록 변경.
"""

import logging
from pathlib import Path

import torch
from omegaconf import DictConfig
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _build_bnb_config(quant_cfg: DictConfig) -> BitsAndBytesConfig:
    """DictConfig로부터 BitsAndBytesConfig를 생성한다.

    Args:
        quant_cfg: cfg.quantization 섹션.

    Returns:
        설정된 BitsAndBytesConfig 인스턴스.
    """
    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg.load_in_4bit,
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg.bnb_4bit_compute_dtype),
        bnb_4bit_quant_type=quant_cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=quant_cfg.bnb_4bit_use_double_quant,
    )


def _load_base_with_partial_state(
    cfg: DictConfig,
    partial_state_path: Path,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """HF Hub base model을 NF4로 로드하고 partial_state.pt의 커스텀 토큰 임베딩을 주입한다.

    Pre-Stage에서 PartialEmbedding/PartialLMHead로 훈련된 커스텀 토큰 가중치를
    표준 embed_tokens.weight / lm_head.weight에 직접 복사한다.
    LoRA 없이 순수 base model + 커스텀 토큰만 포함한 상태를 반환한다.
    이후 PeftModel.from_pretrained()이 adapter_config.json을 읽어 LoRA 구조를 복원한다.

    Args:
        cfg: Hydra DictConfig. cfg.model.hub_id, cfg.model.tokenizer_dir,
             cfg.quantization 섹션을 참조한다.
        partial_state_path: partial_state.pt 파일 경로.

    Returns:
        tuple:
            - model: 커스텀 토큰 임베딩이 주입된 AutoModelForCausalLM (NF4)
            - tokenizer: 커스텀 토큰이 포함된 AutoTokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_dir)
    logger.info("토크나이저 로드 완료. vocab size: %d", len(tokenizer))

    bnb_config = _build_bnb_config(cfg.quantization)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.hub_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    logger.info("Hub 모델 로드 및 vocab 확장 완료: %s", cfg.model.hub_id)

    # partial_state.pt: pre-stage에서 PartialEmbedding/PartialLMHead로 훈련된
    # 커스텀 토큰 행만 저장한 파일. new_token_ids 인덱스 위치에 직접 주입한다.
    partial_state = torch.load(partial_state_path, map_location="cpu", weights_only=True)
    new_token_ids = partial_state["new_token_ids"]
    with torch.no_grad():
        embed_w = model.model.embed_tokens.weight
        model.model.embed_tokens.weight.data[new_token_ids] = (
            partial_state["new_embed"].to(device=embed_w.device, dtype=embed_w.dtype)
        )
        lm_w = model.lm_head.weight
        model.lm_head.weight.data[new_token_ids] = (
            partial_state["new_lm_head"].to(device=lm_w.device, dtype=lm_w.dtype)
        )
    logger.info("partial_state.pt 주입 완료 (커스텀 토큰 %d개)", len(new_token_ids))

    return model, tokenizer


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

    adapters 리스트의 순서대로 named adapter를 적재한다.
    모든 adapter는 merge 없이 독립적으로 유지되며 PEFT의 named adapter API로 관리된다:
        - 첫 번째 adapter: PeftModel.from_pretrained()으로 로드
        - 이후 adapter: model.load_adapter()로 추가 로드

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

    logger.info("Hub 모델 로드 + partial_state.pt 주입: %s", cfg.model.hub_id)
    model, tokenizer = _load_base_with_partial_state(cfg, partial_state_path)

    adapters = list(cfg.inference.get("adapters", None) or [])
    if not adapters:
        logger.warning("inference.adapters가 비어있음. partial_state만 주입된 base model 반환.")
        return model, tokenizer

    for i, adapter_entry in enumerate(adapters):
        adapter_path = Path(adapter_entry.path)
        adapter_name = adapter_entry.get("name", f"adapter_{i}")

        if not adapter_path.exists():
            raise FileNotFoundError(f"adapter 디렉토리를 찾을 수 없음: {adapter_path}")

        if i == 0:
            # 첫 번째 adapter: PeftModel 래퍼 생성 (adapter_config.json에서 LoRA 구조 자동 복원)
            logger.info("첫 번째 adapter 로드 (PeftModel.from_pretrained): %s", adapter_path)
            model = PeftModel.from_pretrained(
                model,
                str(adapter_path),
                adapter_name=adapter_name,
                is_trainable=False,
            )
        else:
            # 추가 adapter: 기존 PeftModel에 named adapter로 적재
            logger.info("추가 adapter 로드 (load_adapter): %s", adapter_path)
            model.load_adapter(str(adapter_path), adapter_name=adapter_name)

        logger.info("adapter '%s' 로드 완료", adapter_name)

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
