"""GRPO 모델 로더 모듈.

Mod Record: 이전 구조에서는 sft/final/model.safetensors(merged full model)를 로컬에서 로드했음.
새 구조: HF Hub NF4 base + partial_state.pt 주입 + SFT adapter(frozen) + GRPO adapter(trainable)
로 멀티 어댑터 스태킹 방식을 사용한다. 전체 모델 저장 없이 stage별 adapter만 축적한다.

Mod Record: DoRA(use_dora=True)에서 표준 LoRA로 전환. DoRA는 unmerged inference 시
delta_W = lora_B @ lora_A 전체 행렬(O(d²))을 materialization하여 rollout generation이 ~240× 느려짐.

멀티 어댑터 스태킹 구조:
    base(NF4, frozen) + partial_state 주입
        ↓ PeftModel.from_pretrained(sft_adapter, is_trainable=False)
    base + SFT adapter(frozen)
        ↓ model.add_adapter("grpo", config)
    base + SFT adapter(frozen) + GRPO adapter(trainable)
"""

import logging
from pathlib import Path

import torch
from omegaconf import DictConfig
from peft import LoraConfig, PeftModel, TaskType

from src.training.pre_stage.model_loader import load_model_with_partial_state

logger = logging.getLogger(__name__)

__all__ = ["load_model_and_tokenizer"]


def load_model_and_tokenizer(cfg: DictConfig) -> tuple:
    """멀티 어댑터 스태킹으로 GRPO 훈련용 모델을 구성한다.

    구성 순서:
        1. HF Hub NF4 base + partial_state.pt 주입 (커스텀 토큰 복원)
        2. SFT adapter 로드 (frozen, is_trainable=False)
        3. GRPO adapter 추가 (trainable)

    Args:
        cfg: Hydra DictConfig. cfg.model, cfg.quantization, cfg.lora 섹션을 참조한다.

    Returns:
        tuple:
            - model: SFT(frozen) + GRPO(trainable) 멀티 어댑터가 적용된 PeftModel
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

    # 1. Hub NF4 base + partial_state.pt 주입
    logger.info(f"Hub 모델 로드 + partial_state.pt 주입: {cfg.model.hub_id}")
    base_model, tokenizer = load_model_with_partial_state(cfg, partial_state_path)

    # 2. SFT adapter 로드 (frozen)
    # adapter_name="sft"로 네임스페이스 분리, is_trainable=False로 gradient 차단
    logger.info(f"SFT adapter 로드 (frozen): {sft_adapter_dir}")
    model = PeftModel.from_pretrained(
        base_model,
        str(sft_adapter_dir),
        adapter_name="sft",
        is_trainable=False,
    )

    # 3. GRPO adapter 추가 (trainable)
    # SFT adapter와 동일한 target_modules에 독립적인 LoRA adapter를 추가
    grpo_config = _build_lora_config(cfg.lora)
    model.add_adapter("grpo", grpo_config)

    # Mod Record: PeftModel.set_adapter("grpo")는 str만 받으며 SFT adapter를 비활성화한다.
    # GRPO forward에서 base(NF4 + partial_state) + SFT(frozen) + GRPO(trainable) 전체가
    # 반영되어야 올바른 policy로 rollout이 생성되므로, BaseTuner.set_adapter에 리스트를 전달해
    # 두 어댑터를 모두 활성화한다. BaseTuner는 LoraLayer.forward의 active_adapters 순회를 제어한다.
    model.base_model.set_adapter(["sft", "grpo"])
    # BaseTuner.set_adapter는 리스트의 모든 adapter를 trainable로 설정하므로
    # SFT 가중치를 다시 frozen으로 되돌린다.
    for name, param in model.named_parameters():
        if ".sft." in name:
            param.requires_grad_(False)
    logger.info("멀티 어댑터 활성화 완료: sft(frozen) + grpo(trainable)")

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

    # 훈련 가능 파라미터 확인 (GRPO adapter만 requires_grad=True여야 함)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"훈련 파라미터: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    return model, tokenizer


def _build_lora_config(lora_cfg: DictConfig) -> LoraConfig:
    """GRPO 표준 LoRA LoraConfig를 생성한다.

    NF4 base에서 비양자화 레이어가 이미 bf16이므로 adapter도 자동으로 bf16이 된다.

    Args:
        lora_cfg: lora 설정 DictConfig.

    Returns:
        LoraConfig 인스턴스 (표준 LoRA, use_dora 기본값 False).
    """
    # Mod Record: PEFT 0.18.1은 lora_dtype 파라미터 미지원.
    # Mod Record: use_dora=True 제거. DoRA unmerged rollout generation에서
    # delta_W materialization(O(d²))으로 ~1.5 tok/sec → 표준 LoRA(O(r×d))로 전환.
    return LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=list(lora_cfg.target_modules),
        bias=lora_cfg.bias,
        task_type=TaskType.CAUSAL_LM,
    )
