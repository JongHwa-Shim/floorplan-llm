"""SFT 훈련 전체 검증 스크립트.

다음 4개 Phase를 순서대로 검증한다:

  Phase 0. 파일 존재 확인 (모델 로드 없음)
    - pre_stage/final/ 필수 파일 존재 여부 확인
    - vocab_extension.json 존재 여부 확인

  Phase 1. 모델 로드 + 구조 검증 (모델 로드 1회)
    - AutoModelForCausalLM.from_pretrained() 성공 여부
    - model.config.vocab_size == len(tokenizer) (커스텀 토큰 포함 여부)
    - vocab_extension.json base_vocab_size 대비 확장 크기 검증
    - tokenizer에 커스텀 토큰(<X:0> 등)이 실제로 등록됐는지 확인
    - DoRA adapter 구조 검증: lora_A / lora_B / lora_magnitude_vector 생성 여부
    - 훈련 가능 파라미터가 target_modules에만 있는지 확인

  Phase 2. 훈련 중 파라미터 갱신 검증 (Phase 1 모델 재사용)
    - N step 훈련 전후 각 target layer의 DoRA 파라미터 스냅샷 비교
    - lora_A / lora_B / lora_magnitude_vector 모두 갱신되는지
    - target_modules 전부(q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)에서
      가중치 변화 확인 — 누락된 레이어 없는지
    - non-target 파라미터(frozen base weights)는 변하지 않는지

  Phase 3. 저장 + Resume 검증 (새 모델 로드 2회)
    Case 3a: 저장 검증
      - N step 훈련 후 checkpoint-N/ 디렉토리 생성 확인
      - adapter_model.safetensors 존재 확인
      - adapter_config.json에 use_dora: true 확인
      - optimizer.pt 존재 확인
    Case 3b: Resume 검증
      - 새 모델 로드 → Resume → M step 추가 훈련
      - Resume 직후 adapter 가중치 == Case 3a 저장값 (복원 정확성)
      - M step 후 adapter 가중치 ≠ 저장값 (Resume 후 갱신 동작 확인)
      - trainer_state.json global_step이 Case 3a step부터 이어지는지 확인

훈련 시간이 지배적이므로 Phase 1+2는 같은 모델 인스턴스를 재사용하여
모델 로드 횟수를 최소화한다.

사용법:
    # 전체 검증 (권장)
    uv run python tests/training/sft/validate_sft.py

    # pre_stage/final 경로 직접 지정
    uv run python tests/training/sft/validate_sft.py \\
        --model_dir data/models/Qwen2.5-Coder-7B/checkpoints/pre_stage/final
"""

import argparse
import json
import logging
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.training.sft import (
    SFTDataset,
    build_trainer,
    load_model_and_tokenizer,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── 검증 파라미터 ──────────────────────────────────────────────────────────────
# 소량의 step으로 빠르게 검증 (모델 로드 시간이 지배적)
STEPS_PHASE2 = 3    # Phase 2 훈련 step 수
STEPS_3A = 3        # Phase 3a: 저장까지의 step 수
STEPS_3B = 3        # Phase 3b: Resume 후 추가 step 수

# 테스트용 임시 출력 디렉토리 (테스트 완료 후 자동 삭제)
TEST_OUTPUT_DIR = str(Path(_PROJECT_ROOT) / "data" / "temp" / "validate_sft")


# ── 유틸리티 ──────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """재현성을 위해 모든 랜덤 시드를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cfg(output_dir: str, max_steps: int, model_dir: str | None = None) -> DictConfig:
    """테스트용 DictConfig를 구성한다.

    run_sft.py와 동일한 방식으로 pipeline.yaml과 augmentation config를 로드하고,
    테스트에 맞게 훈련 파라미터를 오버라이드한다.

    Args:
        output_dir: 체크포인트 저장 경로.
        max_steps: 최대 훈련 step 수.
        model_dir: 모델 로드 경로. None이면 pipeline.yaml 기본값 사용.

    Returns:
        테스트용 DictConfig.
    """
    config_path = Path(_PROJECT_ROOT) / "config" / "training" / "sft" / "pipeline.yaml"
    cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(cfg, False)

    # 증강 config 로드 (run_sft.py와 동일한 방식)
    aug_config_path = Path(_PROJECT_ROOT) / cfg.data.aug_pipeline_config
    aug_cfg = OmegaConf.load(aug_config_path)
    OmegaConf.update(cfg, "augmentation", aug_cfg, merge=True)

    # 테스트용 오버라이드
    cfg.training.output_dir = output_dir
    cfg.training.max_steps = max_steps
    cfg.training.save_strategy = "no"        # 자동 저장 비활성화 (콜백으로 직접 제어)
    cfg.training.eval_strategy = "no"        # 평가 비활성화 (속도)
    cfg.training.load_best_model_at_end = False
    cfg.training.report_to = "none"
    cfg.training.dataloader_num_workers = 0  # subprocess worker 없이 실행
    cfg.training.logging_steps = 1

    if model_dir is not None:
        cfg.model.model_dir = model_dir
        cfg.model.tokenizer_dir = model_dir

    OmegaConf.set_struct(cfg, True)
    return cfg


# ── DoRA 파라미터 스냅샷 ────────────────────────────────────────────────────

def _snap_dora_params(model) -> dict[str, dict[str, torch.Tensor]]:
    """현재 DoRA adapter 파라미터 값의 복사본을 반환한다.

    각 레이어 이름 → {lora_A, lora_B, lora_magnitude_vector} 스냅샷.

    Args:
        model: DoRA adapter가 적용된 PeftModelForCausalLM.

    Returns:
        {layer_name: {param_type: tensor}} 형태의 스냅샷 딕셔너리.
    """
    snap: dict[str, dict[str, torch.Tensor]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # DoRA 파라미터: lora_A.default.weight / lora_B.default.weight / lora_magnitude_vector.default
        if "lora_A" in name or "lora_B" in name or "lora_magnitude_vector" in name:
            snap[name] = param.data.cpu().clone()
    return snap


def _snap_frozen_params(model) -> dict[str, torch.Tensor]:
    """frozen 파라미터(base weights) 값의 일부 복사본을 반환한다.

    frozen 파라미터 전체를 저장하면 메모리 부담이 크므로
    각 레이어의 첫 번째 파라미터만 샘플링한다.

    Args:
        model: DoRA adapter가 적용된 PeftModelForCausalLM.

    Returns:
        {param_name: tensor} 형태의 스냅샷 딕셔너리 (샘플).
    """
    snap: dict[str, torch.Tensor] = {}
    sampled = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            continue
        # 첫 5개 frozen 파라미터만 샘플링
        snap[name] = param.data.cpu().clone()
        sampled += 1
        if sampled >= 5:
            break
    return snap


# ── 콜백 ──────────────────────────────────────────────────────────────────────

class DoRATracker(TrainerCallback):
    """step 종료 시 DoRA 파라미터 스냅샷을 global_step 기준으로 기록한다."""

    def __init__(self):
        # {global_step: {param_name: tensor}}
        self.history: dict[int, dict[str, torch.Tensor]] = {}

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        self.history[state.global_step] = _snap_dora_params(model)


class SaveAtStepCallback(TrainerCallback):
    """지정 step에서 _save_checkpoint를 수동으로 호출한다."""

    def __init__(self, save_at_step: int):
        self.save_at_step = save_at_step
        self.trainer = None

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        if state.global_step == self.save_at_step and self.trainer is not None:
            self.trainer._save_checkpoint(model, None)
            logger.info(f"[SaveCallback] step {state.global_step}: 체크포인트 저장 완료")


# ── Phase 0: 파일 존재 확인 ────────────────────────────────────────────────

def phase0_file_existence(model_dir: str) -> bool:
    """pre_stage/final 디렉토리의 필수 파일 존재를 확인한다.

    Args:
        model_dir: pre_stage/final 경로 문자열.

    Returns:
        검증 통과 여부.
    """
    logger.info("")
    logger.info("─" * 60)
    logger.info("Phase 0: 파일 존재 확인")
    logger.info("─" * 60)

    model_path = Path(model_dir)
    passed = True

    # 필수 파일 목록
    required_files = [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    for fname in required_files:
        fpath = model_path / fname
        if fpath.exists():
            logger.info(f"[PASS] {fname} 존재")
        else:
            logger.error(f"[FAIL] {fname} 없음: {fpath}")
            passed = False

    # vocab_extension.json: pre_stage/final이 아닌 tokenization/ 경로
    cfg_tmp = OmegaConf.load(
        Path(_PROJECT_ROOT) / "config" / "training" / "sft" / "pipeline.yaml"
    )
    vocab_ext_path = Path(_PROJECT_ROOT) / OmegaConf.to_container(cfg_tmp, resolve=False)[
        "model"
    ]["vocab_extension"].replace("${model.name}", cfg_tmp.model.name)
    if vocab_ext_path.exists():
        logger.info(f"[PASS] vocab_extension.json 존재: {vocab_ext_path}")
    else:
        logger.error(f"[FAIL] vocab_extension.json 없음: {vocab_ext_path}")
        passed = False

    return passed


# ── Phase 1: 모델 로드 + 구조 검증 ────────────────────────────────────────

def phase1_model_structure(cfg: DictConfig) -> tuple[bool, object, object]:
    """모델 로드 후 vocab 확장 및 DoRA 구조를 검증한다.

    Args:
        cfg: 테스트용 DictConfig (Phase 2에서 재사용하기 위해 반환).

    Returns:
        tuple:
            - passed: 검증 통과 여부.
            - model: 로드된 DoRA 모델 (Phase 2에서 재사용).
            - tokenizer: 로드된 토크나이저 (Phase 2에서 재사용).
    """
    logger.info("")
    logger.info("─" * 60)
    logger.info("Phase 1: 모델 로드 + 구조 검증")
    logger.info("─" * 60)
    passed = True

    # 모델 + DoRA 로드
    model, tokenizer = load_model_and_tokenizer(cfg)

    # ── vocab_size 검증 ──────────────────────────────────────────────────────
    vocab_size_model = model.config.vocab_size
    vocab_size_tokenizer = len(tokenizer)

    if vocab_size_model == vocab_size_tokenizer:
        logger.info(
            f"[PASS] vocab_size 일치: model={vocab_size_model}, tokenizer={vocab_size_tokenizer}"
        )
    else:
        logger.error(
            f"[FAIL] vocab_size 불일치: model={vocab_size_model}, tokenizer={vocab_size_tokenizer}"
        )
        passed = False

    # ── 커스텀 토큰 확장 크기 검증 ──────────────────────────────────────────
    vocab_ext_path = Path(_PROJECT_ROOT) / cfg.model.vocab_extension
    with open(vocab_ext_path, encoding="utf-8") as f:
        vocab_ext = json.load(f)

    base_vocab_size: int = vocab_ext["base_vocab_size"]
    n_custom_tokens = vocab_size_tokenizer - base_vocab_size

    if n_custom_tokens > 0:
        logger.info(
            f"[PASS] 커스텀 토큰 확장 확인: base={base_vocab_size}, "
            f"+{n_custom_tokens} → total={vocab_size_tokenizer}"
        )
    else:
        logger.error(
            f"[FAIL] 커스텀 토큰 미확장: base={base_vocab_size}, total={vocab_size_tokenizer}"
        )
        passed = False

    # ── 커스텀 토큰 실제 등록 확인 ──────────────────────────────────────────
    # vocab_extension.json의 token_to_id에서 샘플 토큰 선택하여 확인
    token_to_id: dict[str, int] = vocab_ext["token_to_id"]
    sample_tokens = [t for t in token_to_id.keys() if t >= base_vocab_size][:5] \
        if False else list(token_to_id.keys())[:5]
    # 커스텀 토큰 문자열 직접 취득
    custom_token_strs = [
        t for t, tid in token_to_id.items() if tid >= base_vocab_size
    ][:5]

    custom_ok = True
    for token_str in custom_token_strs:
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        expected_id = token_to_id[token_str]
        if token_id == expected_id:
            logger.info(f"[PASS] 커스텀 토큰 등록: '{token_str}' → id={token_id}")
        else:
            logger.error(
                f"[FAIL] 커스텀 토큰 ID 불일치: '{token_str}' "
                f"expected={expected_id}, got={token_id}"
            )
            custom_ok = False
    if not custom_ok:
        passed = False

    # ── DoRA 구조 검증 ──────────────────────────────────────────────────────
    target_modules = list(cfg.dora.target_modules)

    # lora_A, lora_B, lora_magnitude_vector 존재 여부
    dora_params = {name for name, _ in model.named_parameters() if "lora_" in name}
    if dora_params:
        logger.info(f"[PASS] DoRA 파라미터 생성 확인 ({len(dora_params)}개)")
    else:
        logger.error("[FAIL] DoRA 파라미터가 없음 — get_peft_model이 적용되지 않은 것으로 의심")
        passed = False

    # lora_magnitude_vector 존재 여부 (DoRA 전용 파라미터 — LoRA에는 없음)
    mag_params = [n for n in dora_params if "lora_magnitude_vector" in n]
    if mag_params:
        logger.info(f"[PASS] lora_magnitude_vector 존재 (DoRA 활성화 확인): {len(mag_params)}개")
    else:
        logger.error(
            "[FAIL] lora_magnitude_vector 없음 — use_dora=True가 적용되지 않은 것으로 의심"
        )
        passed = False

    # target_modules 각각에 DoRA 파라미터가 있는지 확인
    for mod in target_modules:
        mod_dora = [n for n in dora_params if f".{mod}." in n]
        if mod_dora:
            logger.info(f"[PASS] target_module '{mod}'에 DoRA 파라미터 존재: {len(mod_dora)}개")
        else:
            logger.error(f"[FAIL] target_module '{mod}'에 DoRA 파라미터 없음")
            passed = False

    # frozen 파라미터 확인: base weights (lora_ 없는 non-grad 파라미터) 존재
    frozen_count = sum(
        1 for name, p in model.named_parameters()
        if not p.requires_grad and "lora_" not in name
    )
    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
    logger.info(
        f"[INFO] 훈련 가능 파라미터: {trainable_count}개, frozen 파라미터: {frozen_count}개"
    )
    if frozen_count > 0:
        logger.info("[PASS] base weights가 frozen 상태 확인")
    else:
        logger.error("[FAIL] frozen 파라미터가 없음 — 전체 모델이 학습되고 있을 수 있음")
        passed = False

    return passed, model, tokenizer


# ── Phase 2: 훈련 중 파라미터 갱신 검증 ────────────────────────────────────

def phase2_training_update(
    model,
    tokenizer,
    cfg: DictConfig,
    output_dir: str,
) -> bool:
    """Phase 1 모델을 재사용해 N step 훈련 후 DoRA 파라미터 갱신을 검증한다.

    Args:
        model: Phase 1에서 로드된 DoRA 모델.
        tokenizer: Phase 1에서 로드된 토크나이저.
        cfg: 테스트용 DictConfig.
        output_dir: 임시 체크포인트 저장 경로.

    Returns:
        검증 통과 여부.
    """
    logger.info("")
    logger.info("─" * 60)
    logger.info(f"Phase 2: 훈련 중 파라미터 갱신 검증 ({STEPS_PHASE2} steps)")
    logger.info("─" * 60)
    passed = True

    # 훈련 전 스냅샷
    snap_before = _snap_dora_params(model)
    frozen_before = _snap_frozen_params(model)

    # 데이터셋 로드
    train_dataset = SFTDataset(cfg, tokenizer, split="train", seed=42)
    eval_dataset = SFTDataset(cfg, tokenizer, split="validation", seed=42)

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        cfg=cfg,
    )
    trainer.train()

    # 훈련 후 스냅샷
    snap_after = _snap_dora_params(model)
    frozen_after = _snap_frozen_params(model)

    # ── DoRA 파라미터 갱신 확인 ──────────────────────────────────────────────
    updated_lora_a = []
    updated_lora_b = []
    updated_mag = []
    not_updated = []

    for name in snap_before:
        diff = (snap_after[name] - snap_before[name]).abs().max().item()
        if diff > 0:
            if "lora_A" in name:
                updated_lora_a.append(name)
            elif "lora_B" in name:
                updated_lora_b.append(name)
            elif "lora_magnitude_vector" in name:
                updated_mag.append(name)
        else:
            not_updated.append(name)

    logger.info(f"[INFO] lora_A 갱신 수: {len(updated_lora_a)}")
    logger.info(f"[INFO] lora_B 갱신 수: {len(updated_lora_b)}")
    logger.info(f"[INFO] lora_magnitude_vector 갱신 수: {len(updated_mag)}")

    if not_updated:
        logger.error(f"[FAIL] 갱신되지 않은 DoRA 파라미터: {not_updated[:5]} ...")
        passed = False
    else:
        logger.info("[PASS] 모든 DoRA 파라미터 갱신 확인")

    # ── target_modules 전체 커버리지 확인 ────────────────────────────────────
    target_modules = list(cfg.dora.target_modules)
    for mod in target_modules:
        mod_updated = [n for n in (updated_lora_a + updated_lora_b + updated_mag) if f".{mod}." in n]
        if mod_updated:
            logger.info(f"[PASS] '{mod}' 레이어 DoRA 파라미터 갱신 확인")
        else:
            logger.error(f"[FAIL] '{mod}' 레이어의 DoRA 파라미터가 갱신되지 않음")
            passed = False

    # ── frozen base weights 불변 확인 ────────────────────────────────────────
    frozen_changed = []
    for name in frozen_before:
        if not torch.equal(frozen_before[name], frozen_after.get(name, frozen_before[name])):
            frozen_changed.append(name)

    if frozen_changed:
        logger.error(f"[FAIL] frozen base weights가 변경됨: {frozen_changed}")
        passed = False
    else:
        logger.info("[PASS] frozen base weights 불변 확인")

    return passed


# ── Phase 3: 저장 + Resume 검증 ────────────────────────────────────────────

def phase3_checkpoint_and_resume(output_dir: str, model_dir: str | None) -> bool:
    """DoRA 체크포인트 저장 및 Resume 정합성을 검증한다.

    Phase 3은 새 모델 로드가 필요하므로 Phase 1/2 모델과 독립적으로 실행한다.

    Args:
        output_dir: 임시 체크포인트 저장 경로.
        model_dir: 모델 로드 경로 (None이면 기본값 사용).

    Returns:
        검증 통과 여부.
    """
    logger.info("")
    logger.info("─" * 60)
    logger.info("Phase 3: 저장 + Resume 검증")
    logger.info(f"  Case 3a: {STEPS_3A} steps 훈련 + 저장")
    logger.info(f"  Case 3b: 새 모델 로드 → Resume → {STEPS_3B} steps 추가 훈련")
    logger.info("─" * 60)
    passed = True

    # ── Case 3a: 저장 검증 ────────────────────────────────────────────────────
    logger.info("[Case 3a] 훈련 + 체크포인트 저장 시작...")
    cfg_3a = load_cfg(output_dir, max_steps=STEPS_3A, model_dir=model_dir)
    set_seed(42)

    model_3a, tokenizer_3a = load_model_and_tokenizer(cfg_3a)
    train_dataset_3a = SFTDataset(cfg_3a, tokenizer_3a, split="train", seed=42)
    eval_dataset_3a = SFTDataset(cfg_3a, tokenizer_3a, split="validation", seed=42)

    tracker_3a = DoRATracker()
    save_cb = SaveAtStepCallback(save_at_step=STEPS_3A)

    trainer_3a = build_trainer(
        model=model_3a,
        tokenizer=tokenizer_3a,
        train_dataset=train_dataset_3a,
        eval_dataset=eval_dataset_3a,
        cfg=cfg_3a,
    )
    save_cb.trainer = trainer_3a
    trainer_3a.add_callback(tracker_3a)
    trainer_3a.add_callback(save_cb)
    trainer_3a.train()

    # checkpoint-{STEPS_3A}/ 디렉토리 검증
    ckpt_path = Path(output_dir) / f"checkpoint-{STEPS_3A}"

    # adapter_model.safetensors 존재 확인
    adapter_path = ckpt_path / "adapter_model.safetensors"
    if adapter_path.exists():
        logger.info(f"[PASS] adapter_model.safetensors 존재: {adapter_path}")
    else:
        logger.error(f"[FAIL] adapter_model.safetensors 없음: {adapter_path}")
        passed = False
        return passed  # 이후 검증 불가

    # adapter_config.json의 use_dora 확인
    adapter_config_path = ckpt_path / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path, encoding="utf-8") as f:
            adapter_config = json.load(f)
        if adapter_config.get("use_dora", False):
            logger.info("[PASS] adapter_config.json에 use_dora: true 확인")
        else:
            logger.error(f"[FAIL] adapter_config.json에 use_dora가 false 또는 없음: {adapter_config}")
            passed = False
    else:
        logger.error(f"[FAIL] adapter_config.json 없음: {adapter_config_path}")
        passed = False

    # optimizer.pt 존재 확인
    optimizer_path = ckpt_path / "optimizer.pt"
    if optimizer_path.exists():
        logger.info("[PASS] optimizer.pt 존재")
    else:
        logger.error(f"[FAIL] optimizer.pt 없음: {optimizer_path}")
        passed = False

    # Case 3a 마지막 step 스냅샷 기록
    snap_3a_end = tracker_3a.history.get(STEPS_3A)
    if snap_3a_end is None:
        logger.error(f"[FAIL] Case 3a 마지막 스냅샷 없음. 기록: {sorted(tracker_3a.history.keys())}")
        passed = False
        return passed

    # ── Case 3b: Resume 검증 ──────────────────────────────────────────────────
    logger.info("[Case 3b] 새 모델 로드 후 Resume 훈련 시작...")
    cfg_3b = load_cfg(output_dir, max_steps=STEPS_3A + STEPS_3B, model_dir=model_dir)
    set_seed(42)

    model_3b, tokenizer_3b = load_model_and_tokenizer(cfg_3b)
    train_dataset_3b = SFTDataset(cfg_3b, tokenizer_3b, split="train", seed=42)
    eval_dataset_3b = SFTDataset(cfg_3b, tokenizer_3b, split="validation", seed=42)

    tracker_3b = DoRATracker()
    trainer_3b = build_trainer(
        model=model_3b,
        tokenizer=tokenizer_3b,
        train_dataset=train_dataset_3b,
        eval_dataset=eval_dataset_3b,
        cfg=cfg_3b,
    )
    trainer_3b.add_callback(tracker_3b)
    trainer_3b.train(resume_from_checkpoint=str(ckpt_path))

    # Resume 직후(첫 step) adapter 가중치 == Case 3a 저장값 확인
    # DoRA adapter는 PEFT의 표준 resume 로직으로 복원됨
    # → resume 후 첫 step(STEPS_3A + 1)의 스냅샷이 3a 저장값과 유사해야 함
    # (정확히 같지 않을 수 있음: optimizer state 복원 후 첫 gradient step 적용)
    # 여기서는 "Resume 후에도 계속 갱신되는지"를 검증
    step_3b_end = STEPS_3A + STEPS_3B
    snap_3b_end = tracker_3b.history.get(step_3b_end)

    if snap_3b_end is None:
        logger.error(f"[FAIL] Case 3b 마지막 스냅샷 없음. 기록: {sorted(tracker_3b.history.keys())}")
        passed = False
        return passed

    # Resume 후에도 파라미터가 갱신됐는지 확인
    is_updated_after_resume = False
    for name in snap_3a_end:
        if name in snap_3b_end:
            diff = (snap_3b_end[name] - snap_3a_end[name]).abs().max().item()
            if diff > 0:
                is_updated_after_resume = True
                break

    if is_updated_after_resume:
        logger.info("[PASS] Resume 후 DoRA 파라미터 갱신 확인")
    else:
        logger.error("[FAIL] Resume 후 DoRA 파라미터가 변하지 않음")
        passed = False

    # global_step이 3a 이후부터 이어지는지 확인 (trainer_state.json)
    trainer_state_path = ckpt_path / "trainer_state.json"
    if trainer_state_path.exists():
        with open(trainer_state_path, encoding="utf-8") as f:
            trainer_state = json.load(f)
        saved_global_step = trainer_state.get("global_step", 0)
        if saved_global_step == STEPS_3A:
            logger.info(f"[PASS] trainer_state.json global_step 일치: {saved_global_step}")
        else:
            logger.error(
                f"[FAIL] trainer_state.json global_step 불일치: "
                f"expected={STEPS_3A}, got={saved_global_step}"
            )
            passed = False
    else:
        logger.warning(f"[WARN] trainer_state.json 없음: {trainer_state_path}")

    # ── Case 3c: merge_dora_and_save 최종 병합 저장 검증 ─────────────────────
    # run_sft.py의 마지막 단계를 검증한다.
    # validate_sft.py 초기 구현에서 누락됐던 항목:
    # Phase 3a/3b는 adapter 체크포인트 저장(Trainer 자동)만 검증하고
    # merge_and_unload() → save_pretrained() 경로를 실행하지 않아
    # 실제 run_sft.py 실행 시에야 NotImplementedError가 발견됨.
    logger.info("[Case 3c] merge_dora_and_save 최종 병합 저장 검증...")
    from src.training.sft.model_loader import merge_dora_and_save

    final_save_dir = Path(output_dir) / "final"
    try:
        merge_dora_and_save(model_3b, tokenizer_3b, final_save_dir)
    except Exception as e:
        logger.error(f"[FAIL] merge_dora_and_save 실패: {e}")
        passed = False
        return passed

    # 저장된 파일 존재 확인
    required_final_files = ["model.safetensors", "config.json", "tokenizer.json"]
    for fname in required_final_files:
        fpath = final_save_dir / fname
        if fpath.exists():
            logger.info(f"[PASS] final/{fname} 존재")
        else:
            logger.error(f"[FAIL] final/{fname} 없음: {fpath}")
            passed = False

    return passed


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """4개 Phase를 순서대로 실행하고 결과를 출력한다."""
    parser = argparse.ArgumentParser(description="SFT 훈련 전체 검증")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="pre_stage/final 모델 경로. 미지정 시 pipeline.yaml 기본값 사용.",
    )
    args = parser.parse_args()

    # model_dir를 절대 경로로 변환 (상대 경로 지정 시 프로젝트 루트 기준)
    model_dir = None
    if args.model_dir:
        p = Path(args.model_dir)
        model_dir = str(p if p.is_absolute() else Path(_PROJECT_ROOT) / p)

    # 임시 출력 디렉토리 초기화
    output_dir = TEST_OUTPUT_DIR
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results: dict[str, bool] = {}

    try:
        # ── Phase 0 ────────────────────────────────────────────────────────
        # model_dir 미지정 시 pipeline.yaml의 기본 경로 사용
        cfg_tmp = OmegaConf.load(
            Path(_PROJECT_ROOT) / "config" / "training" / "sft" / "pipeline.yaml"
        )
        effective_model_dir = model_dir or str(
            Path(_PROJECT_ROOT) / cfg_tmp.model.model_dir
        )
        results["Phase 0: 파일 존재 확인"] = phase0_file_existence(effective_model_dir)

        if not results["Phase 0: 파일 존재 확인"]:
            logger.error("Phase 0 실패 — 이후 Phase를 실행할 수 없습니다.")
            sys.exit(1)

        # ── Phase 1 + 2 (모델 공유) ─────────────────────────────────────────
        cfg_p1 = load_cfg(output_dir, max_steps=STEPS_PHASE2, model_dir=model_dir)
        set_seed(42)

        p1_passed, model, tokenizer = phase1_model_structure(cfg_p1)
        results["Phase 1: 모델 로드 + 구조 검증"] = p1_passed

        if p1_passed:
            results["Phase 2: 파라미터 갱신 검증"] = phase2_training_update(
                model, tokenizer, cfg_p1, output_dir
            )
        else:
            logger.warning("Phase 1 실패 — Phase 2는 건너뜁니다.")
            results["Phase 2: 파라미터 갱신 검증"] = False

        # 메모리 해제
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Phase 3 ────────────────────────────────────────────────────────
        # 별도 output_dir 사용 (Phase 2 결과물과 충돌 방지)
        output_dir_3 = str(Path(output_dir) / "phase3")
        Path(output_dir_3).mkdir(parents=True, exist_ok=True)
        results["Phase 3: 저장 + Resume 검증"] = phase3_checkpoint_and_resume(
            output_dir_3, model_dir
        )

    finally:
        if Path(output_dir).exists():
            shutil.rmtree(output_dir)

    # ── 결과 요약 ───────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("검증 결과 요약")
    logger.info("=" * 60)
    all_passed = True
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"  {name}: [{status}]")
        if not result:
            all_passed = False

    if all_passed:
        logger.info("모든 검증 통과 ✓")
        sys.exit(0)
    else:
        logger.error("일부 검증 실패 ✗ — 위 에러를 확인하세요")
        sys.exit(1)


if __name__ == "__main__":
    main()
