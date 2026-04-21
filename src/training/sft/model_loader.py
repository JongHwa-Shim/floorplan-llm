"""SFT 모델 로드 및 LoRA 설정 모듈.

Mod Record: 이전 구조에서는 pre_stage/final/model.safetensors(full model)를 로컬에서 로드했음.
새 구조에서는 HF Hub에서 NF4 base model을 로드하고 partial_state.pt를 주입하여
전체 모델 저장 비용 없이 커스텀 토큰 가중치를 복원한다.

Mod Record: DoRA(use_dora=True)에서 표준 LoRA로 전환. DoRA는 unmerged inference 시
delta_W = lora_B @ lora_A 전체 행렬(O(d²))을 materialization하여 generation이 ~240× 느려짐.
표준 LoRA는 output += lora_B @ (lora_A @ x) efficient path(O(r×d))를 사용.

Pre-Stage와의 차이점:
  - 모델 로드 출처: 로컬 pre_stage/final → HF Hub + partial_state.pt 주입
  - 훈련 파라미터: 새 토큰 행 일부 → LoRA adapter (attention/MLP 전 레이어)
  - 최종 저장: merge된 전체 모델 → adapter_model.safetensors만 저장
"""

import logging
from pathlib import Path

import torch
from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.training.pre_stage.model_loader import load_model_with_partial_state

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    cfg: DictConfig,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """HF Hub에서 NF4 base model을 로드하고 partial_state.pt를 주입한 뒤 LoRA를 적용한다.

    Mod Record: 이전 구조에서는 pre_stage/final/model.safetensors를 로드하고 NF4 재양자화했음.
    새 구조: load_model_with_partial_state()가 Hub 로드 + partial_state.pt 주입 + bf16 재캐스팅을
    일괄 처리하므로 SFT에서는 LoRA 적용만 담당한다.

    Args:
        cfg: Hydra DictConfig. cfg.model, cfg.quantization, cfg.lora 섹션을 참조한다.

    Returns:
        tuple:
            - model: LoRA adapter가 적용된 PeftModelForCausalLM
            - tokenizer: 커스텀 토큰이 포함된 AutoTokenizer

    Raises:
        FileNotFoundError: pre_stage_dir 또는 partial_state.pt가 없을 경우.
    """
    pre_stage_dir = Path(cfg.model.pre_stage_dir)
    partial_state_path = pre_stage_dir / "partial_state.pt"

    if not pre_stage_dir.exists():
        raise FileNotFoundError(f"pre_stage_dir를 찾을 수 없음: {pre_stage_dir}")
    if not partial_state_path.exists():
        raise FileNotFoundError(f"partial_state.pt를 찾을 수 없음: {partial_state_path}")

    # Hub NF4 base + partial_state.pt 주입 (커스텀 토큰 가중치 복원)
    logger.info(f"Hub 모델 로드 + partial_state.pt 주입: {cfg.model.hub_id}")
    model, tokenizer = load_model_with_partial_state(cfg, partial_state_path)
    logger.info(f"모델 로드 완료. vocab_size: {model.config.vocab_size}")

    # LoRA adapter 적용
    lora_config = _build_lora_config(cfg.lora)
    model = get_peft_model(model, lora_config)

    # 훈련 가능 파라미터 수 출력
    model.print_trainable_parameters()

    return model, tokenizer


def save_adapter_only(
    model: AutoModelForCausalLM,
    save_dir: str | Path,
) -> None:
    """LoRA adapter 가중치만 저장한다 (base model 제외).

    Mod Record: 이전 merge_dora_and_save()는 merge_and_unload()로 전체 모델을 저장했음.
    새 구조에서는 GRPO도 HF Hub에서 base model을 새로 로드하므로 adapter만 저장하면 충분하다.
    intermediate checkpoint와 동일한 구조(adapter_model.safetensors + adapter_config.json)로 저장.

    Args:
        model: LoRA adapter가 적용된 PeftModelForCausalLM.
        save_dir: 저장할 디렉토리 경로.

    Returns:
        없음. adapter_model.safetensors, adapter_config.json이 저장된다.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"LoRA adapter 저장 중 (base model 제외): {save_dir}")
    model.save_pretrained(str(save_dir))
    logger.info("adapter_model.safetensors + adapter_config.json 저장 완료")


# ---------------------------------------------------------------------------
# 내부 헬퍼 함수
# ---------------------------------------------------------------------------

def _build_lora_config(lora_cfg: DictConfig) -> LoraConfig:
    """표준 LoRA LoraConfig를 생성한다.

    NF4 base에서 비양자화 레이어가 이미 bf16이므로 adapter도 자동으로 bf16이 된다.

    Args:
        lora_cfg: lora 설정 DictConfig (r, lora_alpha, lora_dropout, target_modules, bias).

    Returns:
        LoraConfig 인스턴스 (표준 LoRA, use_dora 기본값 False).
    """
    # Mod Record: PEFT 0.18.1은 lora_dtype 파라미터 미지원. NF4 base의 비양자화 레이어가
    # bf16이므로 adapter 가중치도 자동으로 bf16으로 생성된다.
    # Mod Record: use_dora=True 제거. DoRA unmerged inference에서 delta_W materialization(O(d²))으로
    # generation 속도가 ~240× 저하됨. 표준 LoRA(O(r×d) efficient path)로 전환.
    return LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=list(lora_cfg.target_modules),
        bias=lora_cfg.bias,
        task_type=TaskType.CAUSAL_LM,
    )
