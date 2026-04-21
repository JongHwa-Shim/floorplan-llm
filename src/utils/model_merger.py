"""모델 컴포넌트 병합 유틸리티.

Hub NF4 base + partial_state.pt + adapter 체인을 bf16 full model로 병합한다.
병합 결과는 from_pretrained()로 직접 로드 가능한 표준 HF 형식으로 저장된다.

사용 예:
    merger = ModelMerger(cfg)
    merger.merge(output_dir="data/models/Qwen2.5-Coder-7B/merged/sft")
"""

import logging
from pathlib import Path

import torch
from omegaconf import DictConfig
from peft import PeftModel
from transformers import AutoModelForCausalLM

from src.training.pre_stage.model_loader import load_model_with_partial_state

logger = logging.getLogger(__name__)


class ModelMerger:
    """Hub NF4 base + partial_state + adapter 체인을 bf16 full model로 병합한다.

    Attributes:
        cfg: Hydra DictConfig. cfg.model, cfg.quantization 섹션을 참조한다.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """ModelMerger 초기화.

        Args:
            cfg: Hydra DictConfig.
        """
        self.cfg = cfg

    def merge(self, output_dir: str | Path) -> None:
        """adapter 체인을 base에 순차 병합하고 bf16 full model로 저장한다.

        병합 순서:
            1. Hub NF4 + partial_state.pt 주입
            2. adapters 리스트 순서대로 merge_and_unload()
            3. bf16 변환 후 저장

        Args:
            output_dir: 저장 대상 디렉토리 경로.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cfg = self.cfg
        pre_stage_dir = Path(cfg.model.pre_stage_dir)
        partial_state_path = pre_stage_dir / "partial_state.pt"

        if not partial_state_path.exists():
            raise FileNotFoundError(f"partial_state.pt를 찾을 수 없음: {partial_state_path}")

        # 1. Hub NF4 base + partial_state.pt 주입
        logger.info(f"Hub 모델 로드 + partial_state.pt 주입: {cfg.model.hub_id}")
        model, tokenizer = load_model_with_partial_state(cfg, partial_state_path)

        # 2. 각 adapter를 순서대로 base에 merge
        for i, adapter_cfg in enumerate(cfg.merge.adapters):
            adapter_path = Path(adapter_cfg.path)
            adapter_name = adapter_cfg.get("name", f"adapter_{i}")

            if not adapter_path.exists():
                raise FileNotFoundError(f"adapter 디렉토리를 찾을 수 없음: {adapter_path}")

            logger.info(f"adapter '{adapter_name}' 로드 후 merge: {adapter_path}")
            peft_model = PeftModel.from_pretrained(
                model,
                str(adapter_path),
                adapter_name=adapter_name,
                is_trainable=False,
            )
            model = peft_model.merge_and_unload()
            logger.info(f"adapter '{adapter_name}' merge 완료")

        # 3. output_dtype에 따라 dtype 변환
        output_dtype_str = cfg.merge.get("output_dtype", "bfloat16")
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        output_dtype = dtype_map.get(output_dtype_str, torch.bfloat16)

        logger.info(f"dtype 변환 중: → {output_dtype_str}")
        model = model.to(output_dtype)

        # 4. 저장
        logger.info(f"병합 모델 저장 중: {output_dir}")
        model.config.torch_dtype = output_dtype_str
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        logger.info(f"병합 완료: {output_dir}")
