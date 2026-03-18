"""Pre-Stage 체크포인트 저장/로드 정상 동작 검증 스크립트.

두 시나리오를 검증한다:
  Case 1. 연속 훈련: checkpoint 저장 이후에도 new_embed/new_lm_head가 계속 업데이트되는지
  Case 2. Resume 훈련: partial_state.pt 복원 후 new_embed/new_lm_head가 업데이트되는지

기존 버그 (수정 전):
  _save_checkpoint에서 merge_and_restore() → _setup_partial_training() 호출 시
  새 nn.Parameter 객체가 생성되면서 optimizer의 Parameter 참조가 끊어진다.
  이후 optimizer.step()이 소멸된 객체를 업데이트하려 해도 grad=None → no-op.
  결과: checkpoint 저장 이후의 훈련 전체가 무효.

검증 흐름 (run_pre_stage.py와 동일한 모델/데이터/Trainer 구성 사용):
  Case 1:
    1. {STEPS_PHASE1} steps 훈련 → new_embed 스냅샷 기록
    2. _save_checkpoint 수동 호출 (콜백)
    3. {STEPS_PHASE2} steps 추가 훈련 → new_embed 스냅샷 기록
    4. 두 스냅샷이 달라야 PASS (optimizer가 계속 업데이트함)

  Case 2:
    Phase 1: {STEPS_PHASE1} steps 훈련 + checkpoint 저장
    Phase 2: 새 모델 로드 → resume → {STEPS_PHASE2} steps 추가 훈련
    검증:
      a. partial_state.pt 저장값 == Phase1 마지막 new_embed 값
      b. Phase2 훈련 후 new_embed != partial_state.pt 값 (resume 후에도 업데이트됨)

사용법:
    uv run python tests/training/pre_stage/validate_save_and_load.py
"""

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

from src.training.pre_stage import (
    PreStageDataset,
    build_trainer,
    load_model_and_tokenizer,
)
from src.training.pre_stage.model_loader import PartialEmbedding, PartialLMHead

logger = logging.getLogger(__name__)


# ── 검증 파라미터 ──────────────────────────────────────────────────────────────
# 소량의 step으로 빠르게 검증 (모델 로드 시간이 지배적)
STEPS_PHASE1 = 3    # 첫 체크포인트 저장까지의 step 수
STEPS_PHASE2 = 3    # 저장 이후 추가 훈련 step 수

# 테스트용 임시 출력 디렉토리 (테스트 완료 후 자동 삭제)
TEST_OUTPUT_DIR = str(Path(_PROJECT_ROOT) / "data" / "temp" / "validate_save_load")


# ── 유틸리티 ──────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """재현성을 위해 모든 랜덤 시드를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _snap_embed(model) -> torch.Tensor:
    """현재 new_embed 값의 복사본을 CPU로 반환한다."""
    return model.model.embed_tokens.new_embed.data.cpu().clone()


def load_cfg(output_dir: str, max_steps: int) -> DictConfig:
    """테스트용 DictConfig를 구성한다.

    run_pre_stage.py와 동일한 방식으로 pipeline.yaml과 augmentation config를 로드하고,
    테스트에 맞게 훈련 파라미터를 오버라이드한다.

    Args:
        output_dir: 체크포인트 저장 경로.
        max_steps: 최대 훈련 step 수.

    Returns:
        테스트용 DictConfig.
    """
    config_path = Path(_PROJECT_ROOT) / "config" / "training" / "pre_stage" / "pipeline.yaml"
    cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(cfg, False)

    # 증강 config 로드 (run_pre_stage.py와 동일한 방식)
    aug_config_path = Path(_PROJECT_ROOT) / cfg.data.aug_pipeline_config
    aug_cfg = OmegaConf.load(aug_config_path)
    OmegaConf.update(cfg, "augmentation", aug_cfg, merge=True)

    # 테스트용 오버라이드
    cfg.training.output_dir = output_dir
    cfg.training.max_steps = max_steps
    cfg.training.save_strategy = "no"       # 자동 저장 비활성화 (콜백으로 직접 제어)
    cfg.training.eval_strategy = "no"       # 평가 비활성화 (속도)
    cfg.training.load_best_model_at_end = False
    cfg.training.report_to = "none"
    cfg.training.dataloader_num_workers = 0  # 테스트에서 subprocess worker 없이 실행
    cfg.training.logging_steps = 1

    OmegaConf.set_struct(cfg, True)
    return cfg


# ── 콜백 ──────────────────────────────────────────────────────────────────────

class EmbedTracker(TrainerCallback):
    """step 종료 시 new_embed 스냅샷을 global_step 기준으로 기록한다."""

    def __init__(self):
        self.history: dict[int, torch.Tensor] = {}

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        if isinstance(model.model.embed_tokens, PartialEmbedding):
            self.history[state.global_step] = _snap_embed(model)


class SaveAtStepCallback(TrainerCallback):
    """지정 step에서 _save_checkpoint를 수동으로 호출한다.

    EmbedTracker 다음에 add_callback해야 같은 step에서
    스냅샷 기록 → 체크포인트 저장 순서가 보장된다.
    """

    def __init__(self, save_at_step: int):
        self.save_at_step = save_at_step
        self.trainer = None  # build_trainer 후 외부에서 주입

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


# ── Case 1: 연속 훈련 검증 ────────────────────────────────────────────────────

def case1_continuous_training(output_dir: str) -> bool:
    """checkpoint 저장 후에도 new_embed가 계속 업데이트되는지 검증한다.

    Args:
        output_dir: 체크포인트 저장 경로.

    Returns:
        검증 통과 여부.
    """
    logger.info("")
    logger.info("─" * 60)
    logger.info("Case 1: 연속 훈련 검증")
    logger.info(f"  {STEPS_PHASE1} steps → checkpoint 저장 → {STEPS_PHASE2} steps 추가 훈련")
    logger.info("─" * 60)

    cfg = load_cfg(output_dir, max_steps=STEPS_PHASE1 + STEPS_PHASE2)
    set_seed(42)

    model, tokenizer, new_token_ids = load_model_and_tokenizer(cfg)
    train_dataset = PreStageDataset(cfg, tokenizer, split="train", seed=42)
    eval_dataset = PreStageDataset(cfg, tokenizer, split="validation", seed=42)

    tracker = EmbedTracker()
    save_cb = SaveAtStepCallback(save_at_step=STEPS_PHASE1)

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        cfg=cfg,
        new_token_ids=new_token_ids,
    )
    save_cb.trainer = trainer  # 콜백에 trainer 주입
    trainer.add_callback(tracker)
    trainer.add_callback(save_cb)

    trainer.train()

    # 검증: Phase1 끝 vs Phase2 끝 new_embed 비교
    step1 = STEPS_PHASE1
    step2 = STEPS_PHASE1 + STEPS_PHASE2

    s1 = tracker.history.get(step1)
    s2 = tracker.history.get(step2)

    if s1 is None or s2 is None:
        logger.error(f"[FAIL] 스냅샷 누락. 기록된 steps: {sorted(tracker.history.keys())}")
        return False

    is_updated = not torch.equal(s1, s2)
    max_diff = (s2 - s1).abs().max().item()

    if is_updated:
        logger.info(f"[PASS] checkpoint 저장 후에도 new_embed 업데이트 확인 (max_diff={max_diff:.6e})")
    else:
        logger.error(f"[FAIL] checkpoint 저장 후 new_embed가 변하지 않음 (max_diff={max_diff:.6e})")
        logger.error(f"       → optimizer Parameter 참조가 끊어진 것으로 의심됨")

    return is_updated


# ── Case 2: Resume 훈련 검증 ──────────────────────────────────────────────────

def case2_resume(output_dir: str) -> bool:
    """partial_state.pt 복원 후 new_embed가 올바르게 업데이트되는지 검증한다.

    Args:
        output_dir: 체크포인트 저장 경로.

    Returns:
        검증 통과 여부.
    """
    logger.info("")
    logger.info("─" * 60)
    logger.info("Case 2: Resume 훈련 검증")
    logger.info(f"  Phase1({STEPS_PHASE1} steps) → 저장 → 새로 로드 → Phase2({STEPS_PHASE2} steps)")
    logger.info("─" * 60)

    # ── Phase 1: 훈련 + 체크포인트 저장 ──────────────────────────────────────
    logger.info("[Phase 1] 훈련 시작...")
    cfg1 = load_cfg(output_dir, max_steps=STEPS_PHASE1)
    set_seed(42)

    model1, tokenizer1, new_token_ids1 = load_model_and_tokenizer(cfg1)
    train_dataset1 = PreStageDataset(cfg1, tokenizer1, split="train", seed=42)
    eval_dataset1 = PreStageDataset(cfg1, tokenizer1, split="validation", seed=42)

    tracker1 = EmbedTracker()
    save_cb1 = SaveAtStepCallback(save_at_step=STEPS_PHASE1)

    trainer1 = build_trainer(
        model=model1,
        tokenizer=tokenizer1,
        train_dataset=train_dataset1,
        eval_dataset=eval_dataset1,
        cfg=cfg1,
        new_token_ids=new_token_ids1,
    )
    save_cb1.trainer = trainer1
    trainer1.add_callback(tracker1)
    trainer1.add_callback(save_cb1)
    trainer1.train()

    # checkpoint 저장 여부 확인
    ckpt_path = Path(output_dir) / f"checkpoint-{STEPS_PHASE1}"
    partial_state_path = ckpt_path / "partial_state.pt"
    if not partial_state_path.exists():
        logger.error(f"[FAIL] partial_state.pt 없음: {partial_state_path}")
        return False

    # partial_state.pt 저장값 == Phase1 마지막 new_embed 값 확인
    saved = torch.load(partial_state_path, map_location="cpu", weights_only=True)
    snap_phase1_end = tracker1.history.get(STEPS_PHASE1)
    if snap_phase1_end is None:
        logger.error(f"[FAIL] Phase1 마지막 스냅샷 없음. 기록: {sorted(tracker1.history.keys())}")
        return False

    if not torch.equal(snap_phase1_end, saved["new_embed"]):
        diff = (snap_phase1_end - saved["new_embed"]).abs().max().item()
        logger.error(f"[FAIL] partial_state.pt 저장값 ≠ Phase1 마지막 new_embed (max_diff={diff:.6e})")
        return False

    logger.info(f"[OK] partial_state.pt 저장값 = Phase1 마지막 new_embed ✓")

    # ── Phase 2: 새로 로드 후 Resume 훈련 ────────────────────────────────────
    logger.info("[Phase 2] 새 모델 로드 후 Resume 훈련 시작...")
    cfg2 = load_cfg(output_dir, max_steps=STEPS_PHASE1 + STEPS_PHASE2)
    set_seed(42)

    model2, tokenizer2, new_token_ids2 = load_model_and_tokenizer(cfg2)
    train_dataset2 = PreStageDataset(cfg2, tokenizer2, split="train", seed=42)
    eval_dataset2 = PreStageDataset(cfg2, tokenizer2, split="validation", seed=42)

    tracker2 = EmbedTracker()

    trainer2 = build_trainer(
        model=model2,
        tokenizer=tokenizer2,
        train_dataset=train_dataset2,
        eval_dataset=eval_dataset2,
        cfg=cfg2,
        new_token_ids=new_token_ids2,
    )
    trainer2.add_callback(tracker2)
    trainer2.train(resume_from_checkpoint=str(ckpt_path))

    # 검증: Phase2 훈련 후 new_embed != partial_state.pt 값 (resume 후 업데이트됨)
    step_phase2_end = STEPS_PHASE1 + STEPS_PHASE2
    snap_phase2_end = tracker2.history.get(step_phase2_end)

    if snap_phase2_end is None:
        logger.error(f"[FAIL] Phase2 마지막 스냅샷 없음. 기록: {sorted(tracker2.history.keys())}")
        return False

    is_updated = not torch.equal(saved["new_embed"], snap_phase2_end)
    max_diff = (snap_phase2_end - saved["new_embed"]).abs().max().item()

    if is_updated:
        logger.info(f"[PASS] Resume 후 new_embed 업데이트 확인 (max_diff={max_diff:.6e})")
    else:
        logger.error(f"[FAIL] Resume 후 new_embed가 변하지 않음 (max_diff={max_diff:.6e})")
        logger.error(f"       → _load_from_checkpoint 또는 optimizer 복원에 문제가 있을 수 있음")

    return is_updated


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """두 가지 시나리오를 순서대로 검증하고 결과를 출력한다."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = TEST_OUTPUT_DIR
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results: dict[str, bool] = {}

    try:
        # Case 1: 연속 훈련
        results["Case 1: 연속 훈련"] = case1_continuous_training(output_dir)

        # Case 2를 위해 출력 디렉토리 초기화
        shutil.rmtree(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Case 2: Resume 훈련
        results["Case 2: Resume 훈련"] = case2_resume(output_dir)

    finally:
        if Path(output_dir).exists():
            shutil.rmtree(output_dir)

    logger.info("")
    logger.info("=" * 60)
    logger.info("검증 결과 요약")
    logger.info("=" * 60)
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {name}: [{status}]")
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("모든 검증 통과")
        sys.exit(0)
    else:
        logger.error("일부 검증 실패")
        sys.exit(1)


if __name__ == "__main__":
    main()
