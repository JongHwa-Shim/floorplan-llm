"""RL 모델 로더 모듈.

Mod Record: 참조 코드(temp/rl_context)에서는 pre_stage.model_loader.load_model_with_partial_state를
호출했으나, 해당 함수가 현재 브랜치에 없음. 대신 sft.model_loader.load_base_model_with_partial_state를
공개 API로 추출하여 재사용한다.

Mod Record: DoRA(use_dora=True)에서 표준 LoRA로 전환. DoRA는 unmerged inference 시
delta_W = lora_B @ lora_A 전체 행렬(O(d²))을 materialization하여 rollout generation이 ~240× 느려짐.

멀티 어댑터 스태킹 구조:
    base(NF4, frozen) + partial_state 주입
        ↓ PeftModel.from_pretrained(sft_adapter, is_trainable=False)
    base + SFT adapter(frozen)
        ↓ model.add_adapter("rl", config)
    base + SFT adapter(frozen) + RL adapter(trainable)
"""

import json
import logging
from pathlib import Path

import torch
from omegaconf import DictConfig
from peft import PeftModel

from src.training.sft.model_loader import load_base_model_with_partial_state, build_lora_config

logger = logging.getLogger(__name__)

__all__ = ["load_model_and_tokenizer", "prepare_vllm_base_model"]


def prepare_vllm_base_model(cfg: DictConfig, model, tokenizer) -> str:
    """vLLM colocate 모드용 확장 vocab base 모델을 로컬에 저장한다.

    Mod Record: vLLM은 model.name_or_path(HF Hub ID)로 초기화하여 원본
    vocab_size=151936을 사용하는데, 훈련 모델의 vocab_size=152232와 달라서
    sync_weights() 시 embed_tokens 크기 불일치 assertion 에러가 발생한다.
    이 함수는 partial_state.pt가 주입된 base 모델(NF4 + 확장 embed_tokens/lm_head)을
    로컬 디렉토리에 1회만 저장하고, 이후 model.config.name_or_path를 그 경로로
    설정하면 vLLM이 올바른 vocab_size=152232로 초기화되어 sync_weights가 정상 동작한다.

    Args:
        cfg: Hydra DictConfig.
        model: SFT+RL 어댑터가 적용된 PeftModel (base HF 모델 추출에 사용).
        tokenizer: 확장된 토크나이저 (vocab_size=152232).

    Returns:
        저장된 base 모델 디렉토리 절대 경로 문자열.
    """
    # tokenizer_dir 기준으로 vllm_base 경로 결정
    # 예: data/models/Qwen2.5-Coder-7B/tokenization → data/models/Qwen2.5-Coder-7B/vllm_base
    tokenizer_dir = Path(cfg.model.tokenizer_dir)
    if not tokenizer_dir.is_absolute():
        tokenizer_dir = Path(__file__).resolve().parents[3] / tokenizer_dir
    vllm_base_dir = tokenizer_dir.parent / "vllm_base"

    # 이미 저장된 경우 재사용 (훈련 재실행 시 재저장 불필요)
    if (vllm_base_dir / "config.json").exists():
        logger.info(f"vLLM base 모델 캐시 재사용: {vllm_base_dir}")
        return str(vllm_base_dir)

    vllm_base_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"vLLM base 모델 최초 저장 중 (이후 캐시 재사용): {vllm_base_dir}\n"
        f"  vocab_size={len(tokenizer)}, NF4 dequantize → bf16 변환 후 저장"
    )

    # Mod Record: save_pretrained()는 NF4(bitsandbytes) 포맷으로 저장하면서
    # (1) 파라미터 이름에 .base_layer. 포함, (2) .absmax/.quant_map 등 NF4 메타데이터 저장.
    # vLLM은 bitsandbytes NF4 포맷을 이해하지 못하므로 dequantize → bf16으로 직접 저장한다.
    base_hf_model = model.get_base_model()

    # 1. config.json 저장 — quantization_config 제거(vLLM이 bf16으로 로드하도록)
    config_dict = base_hf_model.config.to_dict()
    config_dict.pop("quantization_config", None)
    with open(vllm_base_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    # 2. 가중치 추출: NF4 역양자화 + .base_layer. 이름 정리 + PEFT 파라미터 제거
    try:
        from bitsandbytes.nn import Params4bit as _Params4bit
        import bitsandbytes.functional as _bnb_F
    except ImportError:
        _Params4bit = None
        _bnb_F = None

    import safetensors.torch as _safetensors

    state_dict = {}
    for name, param in base_hf_model.named_parameters():
        # LoRA 어댑터 전용 파라미터 제외
        if "lora_" in name:
            continue
        # .base_layer. 제거 → 표준 HF 이름으로 복원
        clean_name = name.replace(".base_layer.", ".")
        # Params4bit(NF4) → bf16 역양자화
        if _Params4bit is not None and isinstance(param, _Params4bit):
            weight = _bnb_F.dequantize_4bit(
                param.data, param.quant_state
            ).to(torch.bfloat16)
        else:
            weight = param.data.to(torch.bfloat16)
        state_dict[clean_name] = weight.contiguous()

    _safetensors.save_file(state_dict, str(vllm_base_dir / "model.safetensors"))
    logger.info(f"  dequantize 완료: {len(state_dict)}개 파라미터 bf16으로 저장")

    # 3. generation_config.json 저장
    if hasattr(base_hf_model, "generation_config") and base_hf_model.generation_config is not None:
        base_hf_model.generation_config.save_pretrained(str(vllm_base_dir))

    # 4. 토크나이저 저장
    tokenizer.save_pretrained(str(vllm_base_dir))

    logger.info(f"vLLM base 모델 저장 완료: {vllm_base_dir}")
    return str(vllm_base_dir)


def load_model_and_tokenizer(cfg: DictConfig) -> tuple:
    """멀티 어댑터 스태킹으로 RL 훈련용 모델을 구성한다.

    구성 순서:
        1. HF Hub NF4 base + partial_state.pt 주입 (커스텀 토큰 복원)
        2. SFT adapter 로드 (frozen, is_trainable=False)
        3. RL adapter 추가 (trainable)

    Args:
        cfg: Hydra DictConfig. cfg.model, cfg.quantization, cfg.lora 섹션을 참조한다.

    Returns:
        tuple:
            - model: SFT(frozen) + RL(trainable) 멀티 어댑터가 적용된 PeftModel
            - tokenizer: 커스텀 토큰이 포함된 AutoTokenizer

    Raises:
        FileNotFoundError: pre_stage_dir/partial_state.pt 또는 sft_adapter_dir가 없을 경우.
    """
    pre_stage_dir = Path(cfg.model.pre_stage_dir)
    sft_adapter_dir = Path(cfg.model.sft_adapter_dir)
    partial_state_path = pre_stage_dir / "partial_state.pt"

    if not partial_state_path.exists():
        raise FileNotFoundError(f"partial_state.pt를 찾을 수 없음: {partial_state_path}")
    if not sft_adapter_dir.exists():
        raise FileNotFoundError(f"SFT adapter 디렉토리를 찾을 수 없음: {sft_adapter_dir}")

    # 1. Hub NF4 base + partial_state.pt 주입 (SFT와 공통 로직 재사용)
    logger.info(f"Hub 모델 로드 + partial_state.pt 주입: {cfg.model.hub_id}")
    base_model, tokenizer = load_base_model_with_partial_state(cfg, partial_state_path)

    # 2. SFT adapter 로드 (frozen)
    # adapter_name="sft"로 네임스페이스 분리, is_trainable=False로 gradient 차단
    logger.info(f"SFT adapter 로드 (frozen): {sft_adapter_dir}")
    model = PeftModel.from_pretrained(
        base_model,
        str(sft_adapter_dir),
        adapter_name="sft",
        is_trainable=False,
    )

    # 3. RL adapter 추가 (trainable)
    # SFT adapter와 동일한 target_modules에 독립적인 LoRA adapter를 추가
    rl_config = build_lora_config(cfg.lora)
    model.add_adapter("rl", rl_config)

    # Mod Record: PeftModel.set_adapter("rl")는 str만 받으며 SFT adapter를 비활성화한다.
    # RL forward에서 base(NF4 + partial_state) + SFT(frozen) + RL(trainable) 전체가
    # 반영되어야 올바른 policy로 rollout이 생성되므로, BaseTuner.set_adapter에 리스트를 전달해
    # 두 어댑터를 모두 활성화한다. BaseTuner는 LoraLayer.forward의 active_adapters 순회를 제어한다.
    model.base_model.set_adapter(["sft", "rl"])
    # BaseTuner.set_adapter는 리스트의 모든 adapter를 trainable로 설정하므로
    # SFT 가중치를 다시 frozen으로 되돌린다.
    for name, param in model.named_parameters():
        if ".sft." in name:
            param.requires_grad_(False)
    logger.info("멀티 어댑터 활성화 완료: sft(frozen) + rl(trainable)")

    # Mod Record: add_adapter()가 새 Linear 모듈을 float32로 초기화하고,
    # PeftModel.from_pretrained으로 로드한 SFT adapter 가중치와 attention bias도
    # float32로 유지될 수 있다. NF4 양자화 파라미터(Params4bit)를 제외한
    # 모든 float32 파라미터를 bf16으로 통일하지 않으면 forward pass에서
    # "expected BFloat16 but found Float" 오류가 발생한다.
    try:
        from bitsandbytes.nn import Params4bit as _Params4bit
    except ImportError:
        _Params4bit = None

    cast_count = 0
    for param in model.parameters():
        if _Params4bit is not None and isinstance(param, _Params4bit):
            continue  # NF4 양자화 파라미터는 건드리지 않음
        if param.data.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)
            cast_count += 1
    logger.info(f"비양자화 float32 파라미터 bf16 캐스팅 완료: {cast_count}개")

    # 훈련 가능 파라미터 확인 (RL adapter만 requires_grad=True여야 함)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"훈련 파라미터: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    return model, tokenizer
