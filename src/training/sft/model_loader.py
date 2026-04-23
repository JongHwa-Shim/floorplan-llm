"""SFT 모델 로드 및 LoRA 설정 모듈.

HF Hub에서 base model을 로드하고, pre_stage/final의 partial_state.pt로 커스텀 토큰 가중치를
적용한 뒤 LoRA(Low-Rank Adaptation)를 적용한다.

Pre-Stage resume과 유사한 패턴:
  - 모델 로드 출처: HF Hub (cfg.model.hub_id)
  - resize_token_embeddings(): 커스텀 토큰 수만큼 vocab 확장
  - partial_state.pt 적용: embed_tokens / lm_head의 새 토큰 행에 훈련된 가중치 덮어쓰기
  - PartialEmbedding/PartialLMHead 불필요: 가중치를 standard 레이어에 직접 주입 후 freeze
  - 훈련 파라미터: LoRA adapter (attention/MLP 전 레이어)만 학습
"""

import logging
from pathlib import Path
from typing import List

import torch
from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    cfg: DictConfig,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """HF Hub base model을 로드하고 partial_state.pt 적용 후 LoRA를 붙인다.

    Pre-Stage resume과 동일한 패턴으로:
      1. HF Hub에서 base model 로드 (cfg.model.hub_id)
      2. resize_token_embeddings()로 vocab 확장
      3. partial_state.pt의 커스텀 토큰 가중치를 embed_tokens/lm_head에 직접 주입
      4. prepare_model_for_kbit_training + get_peft_model (base params는 PEFT가 자동 freeze)

    Args:
        cfg: Hydra DictConfig. cfg.model, cfg.quantization, cfg.lora 섹션을 참조한다.

    Returns:
        tuple:
            - model: LoRA adapter가 적용된 PeftModelForCausalLM
            - tokenizer: 커스텀 토큰이 포함된 AutoTokenizer

    Raises:
        FileNotFoundError: model_dir 또는 tokenizer_dir 또는 partial_state.pt가 없을 경우.
    """
    model_dir = Path(cfg.model.model_dir)
    tokenizer_dir = Path(cfg.model.tokenizer_dir)
    hub_id: str = cfg.model.hub_id

    if not model_dir.exists():
        raise FileNotFoundError(f"pre_stage/final 디렉토리를 찾을 수 없음: {model_dir}")
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"토크나이저 디렉토리를 찾을 수 없음: {tokenizer_dir}")

    partial_state_path = model_dir / "partial_state.pt"
    if not partial_state_path.exists():
        raise FileNotFoundError(f"partial_state.pt를 찾을 수 없음: {partial_state_path}")

    # 토크나이저 로드 (pre_stage/final에 커스텀 토큰 포함된 tokenizer.json)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    logger.info(f"토크나이저 로드 완료. vocab size: {len(tokenizer)}")

    # 4bit 양자화 설정
    bnb_config = _build_bnb_config(cfg.quantization)

    # HF Hub에서 base model 로드 (Pre-Stage와 동일한 출처)
    logger.info(f"base model 로드 중 (HF Hub): {hub_id}")
    model = AutoModelForCausalLM.from_pretrained(
        hub_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # vocab 크기 확장: 커스텀 토큰을 수용하도록 embed_tokens / lm_head 행 추가
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"vocab 확장 완료. vocab_size: {model.config.vocab_size}")

    # partial_state.pt 로드하여 커스텀 토큰 행에 Pre-Stage 훈련된 가중치 주입
    # embed_tokens / lm_head는 4bit 양자화 대상이 아니므로 (bf16) 직접 인덱싱 가능
    logger.info(f"partial_state.pt 로드 및 커스텀 토큰 가중치 적용 중: {partial_state_path}")
    partial_state = torch.load(partial_state_path, map_location="cpu", weights_only=True)
    new_token_ids: list[int] = partial_state["new_token_ids"]

    with torch.no_grad():
        model.model.embed_tokens.weight.data[new_token_ids] = partial_state["new_embed"].to(
            model.model.embed_tokens.weight.device
        )
        model.lm_head.weight.data[new_token_ids] = partial_state["new_lm_head"].to(
            model.lm_head.weight.device
        )
    logger.info(f"커스텀 토큰 가중치 적용 완료 ({len(new_token_ids)}개 행)")

    # kbit 훈련 준비:
    # - gradient checkpointing 활성화
    # - use_reentrant=False: PyTorch 2.5+ 기본값 변경에 맞춘 명시적 설정
    # - layernorm을 fp32로 업캐스팅 (안정적인 backward 보장)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # LoRA adapter 적용 (get_peft_model이 embed_tokens / lm_head 포함 base params 전체 freeze)
    lora_config = _build_lora_config(cfg.lora)
    model = get_peft_model(model, lora_config)

    # 훈련 가능 파라미터 수 출력
    model.print_trainable_parameters()

    return model, tokenizer


def merge_lora_and_save(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    save_dir: str | Path,
) -> None:
    """LoRA adapter를 base model에 병합하고 표준 HuggingFace 형식으로 저장한다.

    .. deprecated::
        run_sft.py의 기본 최종 저장 흐름에서는 더 이상 호출하지 않는다.
        기본 흐름은 adapter만 저장(PeftModel.save_pretrained)하며, 이 함수는
        PEFT 의존성 없이 standalone 추론 모델이 필요하거나 다음 Stage에서
        full model이 요구될 때 수동으로 호출하기 위해 유지한다.

    저장 후 결과물은 다음 Stage 또는 추론에서 from_pretrained()로 로드 가능하다.

    Args:
        model: LoRA adapter가 적용된 PeftModelForCausalLM.
        tokenizer: 커스텀 토큰이 포함된 AutoTokenizer.
        save_dir: 저장할 디렉토리 경로.

    Returns:
        없음. 디렉토리에 model.safetensors, tokenizer.json 등이 저장된다.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("LoRA adapter를 base model에 병합 중 (merge_and_unload)...")
    merged_model = model.merge_and_unload()

    # Mod Record: transformers 4.51+에서 4bit 양자화 모델에 merge_and_unload() 후
    # save_pretrained()를 호출하면 revert_weight_conversion()이 4bit 역변환을 시도하는데,
    # 해당 역변환(reverse_op)이 미구현 상태라 NotImplementedError가 발생한다.
    # merge_and_unload 이후에도 transformer 레이어는 여전히 NF4 4bit 상태이며,
    # revert_weight_conversion은 이를 원래 dtype으로 복원하려 하지만 실패함.
    # pre_stage의 validate_quantization_for_training 패치와 동일한 방식으로
    # modeling_utils 네임스페이스의 함수를 일시 no-op으로 교체하여 우회한다.
    import transformers.modeling_utils as _modeling_module

    _orig_revert = getattr(_modeling_module, "revert_weight_conversion", None)
    if _orig_revert is not None:
        _modeling_module.revert_weight_conversion = lambda m, sd: sd
    try:
        logger.info(f"병합된 모델 저장 중: {save_dir}")
        merged_model.save_pretrained(str(save_dir))
    finally:
        if _orig_revert is not None:
            _modeling_module.revert_weight_conversion = _orig_revert

    tokenizer.save_pretrained(str(save_dir))
    logger.info("모델 및 토크나이저 저장 완료")


# ---------------------------------------------------------------------------
# 내부 헬퍼 함수
# ---------------------------------------------------------------------------

def _build_bnb_config(quant_cfg: DictConfig) -> BitsAndBytesConfig:
    """BitsAndBytesConfig를 생성한다.

    Args:
        quant_cfg: quantization 설정 DictConfig.

    Returns:
        BitsAndBytesConfig 인스턴스.
    """
    compute_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = compute_dtype_map[quant_cfg.bnb_4bit_compute_dtype]

    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg.load_in_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=quant_cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=quant_cfg.bnb_4bit_use_double_quant,
    )


def _build_lora_config(lora_cfg: DictConfig) -> LoraConfig:
    """LoRA LoraConfig를 생성한다.

    Args:
        lora_cfg: lora 설정 DictConfig (r, lora_alpha, lora_dropout, target_modules, bias).

    Returns:
        LoraConfig 인스턴스.
    """
    return LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=list(lora_cfg.target_modules),
        bias=lora_cfg.bias,
        task_type=TaskType.CAUSAL_LM,
        use_dora=False,
    )
