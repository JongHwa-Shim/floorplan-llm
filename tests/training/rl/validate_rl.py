"""RL 구현 4단계 검증 스크립트.

Phase 0: 파일 존재 확인
Phase 1: 보상함수 단위 테스트 (GPU 불필요)
Phase 2: 모델 로드 + 구조 검증 (GPU 1개)
Phase 3: 소규모 훈련 루프 (GPU 1개, 3~5 step)

사용법:
    # 전체 phase 실행
    uv run python tests/training/rl/validate_rl.py

    # 특정 phase만
    uv run python tests/training/rl/validate_rl.py --phase 1
    uv run python tests/training/rl/validate_rl.py --phase 2
    uv run python tests/training/rl/validate_rl.py --phase 3
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# 프로젝트 루트를 sys.path에 추가
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _vram_report(label: str) -> None:
    """현재 CUDA VRAM 사용량을 로깅한다.

    Args:
        label: 출력 레이블 (예: "모델 로드 후").
    """
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
        peak = torch.cuda.max_memory_allocated(i) / 1024 ** 3
        logger.info(
            f"[VRAM] {label} | GPU{i}: "
            f"allocated={allocated:.2f} GB, reserved={reserved:.2f} GB, peak={peak:.2f} GB"
        )


# ---------------------------------------------------------------------------
# Phase 0: 파일 존재 확인
# ---------------------------------------------------------------------------

def phase0_file_existence() -> bool:
    """필수 파일 존재 여부를 확인한다.

    Returns:
        모든 파일 존재 시 True.
    """
    logger.info("=== Phase 0: 파일 존재 확인 ===")
    required_files = [
        "config/training/rl/pipeline.yaml",
        "src/training/rl/__init__.py",
        "src/training/rl/trainer.py",
        "src/training/rl/dataset.py",
        "src/training/rl/model_loader.py",
        "src/training/rl/advantage.py",
        "src/training/rl/rewards/__init__.py",
        "src/training/rl/rewards/parser.py",
        "src/training/rl/rewards/format_reward.py",
        "src/training/rl/rewards/count_reward.py",
        "src/training/rl/rewards/geometry_reward.py",
        "src/training/rl/rewards/connectivity_reward.py",
        "src/training/rl/rewards/spatial_reward.py",
        "src/training/rl/rewards/credit_assignment.py",
        "scripts/training/run_rl.py",
    ]

    missing = []
    for rel_path in required_files:
        full_path = Path(_PROJECT_ROOT) / rel_path
        if full_path.exists():
            logger.info(f"  [OK] {rel_path}")
        else:
            logger.error(f"  [MISSING] {rel_path}")
            missing.append(rel_path)

    # partial_state.pt 확인
    partial_state_path = Path(_PROJECT_ROOT) / "data/models/Qwen2.5-Coder-7B/checkpoints/pre_stage/final/partial_state.pt"
    if partial_state_path.exists():
        logger.info(f"  [OK] pre_stage partial_state.pt: {partial_state_path}")
    else:
        logger.warning(f"  [WARNING] partial_state.pt 없음: {partial_state_path} (Phase 2-3 실행 불가)")

    # SFT adapter 확인
    sft_adapter_path = Path(_PROJECT_ROOT) / "data/models/Qwen2.5-Coder-7B/checkpoints/sft/final/adapter_model.safetensors"
    if sft_adapter_path.exists():
        logger.info(f"  [OK] SFT adapter_model.safetensors: {sft_adapter_path}")
    else:
        logger.warning(f"  [WARNING] SFT adapter 없음: {sft_adapter_path} (Phase 2-3 실행 불가)")

    if missing:
        logger.error(f"누락 파일 {len(missing)}개 발견")
        return False

    logger.info("Phase 0 완료: 모든 파일 존재 확인")
    return True


# ---------------------------------------------------------------------------
# Phase 1: 보상함수 단위 테스트
# ---------------------------------------------------------------------------

def phase1_reward_unit_tests() -> bool:
    """보상함수 단위 테스트를 실행한다. GPU 불필요.

    수동 제작한 토큰 시퀀스로 파서 및 보상함수를 검증한다.

    Returns:
        모든 테스트 통과 시 True.
    """
    logger.info("=== Phase 1: 보상함수 단위 테스트 ===")

    errors = []

    # Vocab 로드
    try:
        from src.training.augmentation.tokenizer import load_vocab
        vocab_path = Path(_PROJECT_ROOT) / "data/models/Qwen2.5-Coder-7B/tokenization/vocab_extension.json"
        tokenizer_dir = Path(_PROJECT_ROOT) / "data/models/Qwen2.5-Coder-7B/checkpoints/sft/final"

        if not vocab_path.exists():
            logger.error(f"vocab_extension.json 없음: {vocab_path}")
            return False

        vocab = load_vocab(vocab_path, str(tokenizer_dir) if tokenizer_dir.exists() else None)
        logger.info(f"Vocab 로드 완료: {len(vocab.token_to_id)}개 커스텀 토큰")
    except Exception as e:
        logger.error(f"Vocab 로드 실패: {e}")
        return False

    # 테스트 1: 파서 - 정상 시퀀스
    errors.extend(_test_parser_valid(vocab))

    # 테스트 2: 파서 - 비정상 시퀀스
    errors.extend(_test_parser_invalid(vocab))

    # 테스트 3: R_format 보상함수
    errors.extend(_test_format_reward(vocab))

    # 테스트 4: R_count 보상함수
    errors.extend(_test_count_reward(vocab))

    # 테스트 5: R_orthogonality 보상함수
    errors.extend(_test_orthogonality_reward(vocab))

    # 테스트 6: R_no_overlap 보상함수
    errors.extend(_test_no_overlap_reward(vocab))

    # 테스트 7: GDPO 정규화 수치 검증
    errors.extend(_test_gdpo_normalization())

    # 테스트 8: 토큰 신용할당 수치 검증
    errors.extend(_test_credit_assignment())

    if errors:
        for err in errors:
            logger.error(f"  [FAIL] {err}")
        return False

    logger.info("Phase 1 완료: 모든 단위 테스트 통과")
    return True


def _build_valid_token_sequence(vocab) -> list[int]:
    """직사각형 방 1개를 가진 정상 출력 토큰 시퀀스를 생성한다."""
    def t(s):
        return vocab.token_to_id[s]

    ids = []
    ids.append(t("<OUTPUT>"))
    ids.append(t("<FRONT_DOOR>"))
    ids.append(t("<NO_DOOR>"))
    ids.append(t("<END_DOOR>"))

    # outline 방
    ids.append(t("<ROOM>"))
    ids.append(t("<TYPE:outline>"))
    for x, y in [(0, 0), (100, 0), (100, 100), (0, 100)]:
        ids.append(t(f"<X:{x}>"))
        ids.append(t(f"<Y:{y}>"))
    ids.append(t("<END_ROOM>"))

    # bedroom 방
    ids.append(t("<ROOM>"))
    ids.append(t("<TYPE:bedroom>"))
    for x, y in [(10, 10), (50, 10), (50, 50), (10, 50)]:
        ids.append(t(f"<X:{x}>"))
        ids.append(t(f"<Y:{y}>"))
    ids.append(t("<END_ROOM>"))

    ids.append(t("<END_OUTPUT>"))
    return ids


def _test_parser_valid(vocab) -> list[str]:
    """정상 시퀀스 파싱 테스트."""
    from src.training.rl.rewards.parser import parse_output_tokens
    errors = []

    try:
        ids = _build_valid_token_sequence(vocab)
        parsed = parse_output_tokens(ids, vocab)

        if not parsed.success:
            errors.append(f"파서: 정상 시퀀스 파싱 실패 (level={parsed.level})")
        if len(parsed.rooms) < 2:
            errors.append(f"파서: 방 개수 부족 ({len(parsed.rooms)} < 2)")
        if parsed.rooms[0].room_type != "outline":
            errors.append(f"파서: 첫 번째 방이 outline이 아님 ({parsed.rooms[0].room_type})")
        if len(parsed.error_indices) > 0:
            errors.append(f"파서: 정상 시퀀스에서 오류 인덱스 발생 {parsed.error_indices}")
        if len(parsed.rooms[0].coords) != 4:
            errors.append(f"파서: outline 꼭짓점 개수 오류 ({len(parsed.rooms[0].coords)})")

        if not errors:
            logger.info("  [OK] 파서 - 정상 시퀀스")
    except Exception as e:
        errors.append(f"파서 - 정상 시퀀스 예외: {e}")

    return errors


def _test_parser_invalid(vocab) -> list[str]:
    """비정상 시퀀스 (X/Y 교대 깨짐) 파싱 테스트."""
    from src.training.rl.rewards.parser import parse_output_tokens
    errors = []

    try:
        def t(s):
            return vocab.token_to_id[s]

        # X/Y 교대가 깨진 시퀀스: <X:10> <Y:10> <Y:50> (Y 중복)
        ids = [
            t("<OUTPUT>"),
            t("<FRONT_DOOR>"), t("<NO_DOOR>"), t("<END_DOOR>"),
            t("<ROOM>"), t("<TYPE:outline>"),
            t("<X:0>"), t("<Y:0>"), t("<X:100>"), t("<Y:0>"),
            t("<X:100>"), t("<Y:100>"), t("<X:0>"), t("<Y:100>"),
            t("<END_ROOM>"),
            t("<ROOM>"), t("<TYPE:bedroom>"),
            t("<X:10>"), t("<Y:10>"), t("<Y:50>"),  # Y 중복 → 오류
            t("<END_ROOM>"),
            t("<END_OUTPUT>"),
        ]
        parsed = parse_output_tokens(ids, vocab)

        # 오류가 있어야 함 (level < 3 또는 error_indices 있음)
        if parsed.success and len(parsed.error_indices) == 0:
            errors.append("파서: 비정상 시퀀스에서 오류 미감지")
        else:
            logger.info("  [OK] 파서 - 비정상 시퀀스 (오류 감지)")
    except Exception as e:
        errors.append(f"파서 - 비정상 시퀀스 예외: {e}")

    return errors


def _test_format_reward(vocab) -> list[str]:
    """R_format 보상함수 테스트."""
    from src.training.rl.rewards.parser import parse_output_tokens
    from src.training.rl.rewards.format_reward import compute_format_reward
    errors = []

    try:
        # 정상: reward=1.0
        ids = _build_valid_token_sequence(vocab)
        parsed = parse_output_tokens(ids, vocab)
        reward, err_idx = compute_format_reward(parsed)

        if abs(reward - 1.0) > 1e-6:
            errors.append(f"R_format: 정상 시퀀스 reward={reward} (기대: 1.0)")
        if err_idx:
            errors.append(f"R_format: 정상 시퀀스에서 error_indices 비어있어야 함 ({err_idx})")

        if not errors:
            logger.info("  [OK] R_format - 정상 시퀀스 (reward=1.0)")

        # 비정상: reward=0.0
        def t(s):
            return vocab.token_to_id[s]

        bad_ids = [t("<OUTPUT>"), t("<END_OUTPUT>")]  # 방 없음
        bad_parsed = parse_output_tokens(bad_ids, vocab)
        bad_reward, _ = compute_format_reward(bad_parsed)

        if abs(bad_reward) > 1e-6:
            errors.append(f"R_format: 빈 시퀀스 reward={bad_reward} (기대: 0.0)")
        else:
            logger.info("  [OK] R_format - 빈 시퀀스 (reward=0.0)")

    except Exception as e:
        errors.append(f"R_format 예외: {e}")

    return errors


def _test_count_reward(vocab) -> list[str]:
    """R_count 보상함수 테스트."""
    from src.training.rl.rewards.parser import parse_output_tokens
    from src.training.rl.rewards.count_reward import (
        compute_count_total_reward,
        compute_count_type_reward,
    )
    errors = []

    try:
        ids = _build_valid_token_sequence(vocab)
        parsed = parse_output_tokens(ids, vocab)

        # 총 방 1개 (bedroom 1개, outline 제외)
        metadata_correct = {"total_rooms": 1, "type_counts": {"bedroom": 1}}
        metadata_wrong = {"total_rooms": 3, "type_counts": {"bedroom": 2}}

        r_total_correct = compute_count_total_reward(parsed, metadata_correct)
        r_total_wrong = compute_count_total_reward(parsed, metadata_wrong)

        if abs(r_total_correct - 1.0) > 1e-6:
            errors.append(f"R_count_total: 일치 시 reward={r_total_correct} (기대: 1.0)")
        if abs(r_total_wrong) > 1e-6:
            errors.append(f"R_count_total: 불일치 시 reward={r_total_wrong} (기대: 0.0)")

        r_type_correct = compute_count_type_reward(parsed, metadata_correct)
        if abs(r_type_correct - 1.0) > 1e-6:
            errors.append(f"R_count_type: 정확 시 reward={r_type_correct} (기대: 1.0)")

        if not errors:
            logger.info("  [OK] R_count (total + type)")
    except Exception as e:
        errors.append(f"R_count 예외: {e}")

    return errors


def _test_orthogonality_reward(vocab) -> list[str]:
    """R_orthogonality 보상함수 테스트."""
    from src.training.rl.rewards.parser import parse_output_tokens
    from src.training.rl.rewards.geometry_reward import compute_orthogonality_reward
    errors = []

    try:
        # 직사각형 방 → 모든 꼭짓점이 직각 → reward=1.0
        ids = _build_valid_token_sequence(vocab)
        parsed = parse_output_tokens(ids, vocab)
        reward, err_idx = compute_orthogonality_reward(parsed)

        if abs(reward - 1.0) > 1e-4:
            errors.append(f"R_orthogonality: 직사각형 reward={reward} (기대: ~1.0)")
        if err_idx:
            errors.append(f"R_orthogonality: 직사각형에서 error_indices 있음 {err_idx}")

        if not errors:
            logger.info(f"  [OK] R_orthogonality - 직사각형 (reward={reward:.4f})")
    except Exception as e:
        errors.append(f"R_orthogonality 예외: {e}")

    return errors


def _test_no_overlap_reward(vocab) -> list[str]:
    """R_no_overlap 보상함수 테스트."""
    from src.training.rl.rewards.parser import parse_output_tokens
    from src.training.rl.rewards.geometry_reward import compute_no_overlap_reward
    errors = []

    try:
        # 겹침 없는 방 → reward=1.0
        ids = _build_valid_token_sequence(vocab)
        parsed = parse_output_tokens(ids, vocab)
        reward, err_idx = compute_no_overlap_reward(parsed)

        # 방이 1개뿐이므로 겹침 없음 → 1.0
        if abs(reward - 1.0) > 1e-4:
            errors.append(f"R_no_overlap: 단일 방 reward={reward} (기대: 1.0)")

        if not errors:
            logger.info(f"  [OK] R_no_overlap - 단일 방 (reward={reward:.4f})")
    except Exception as e:
        errors.append(f"R_no_overlap 예외: {e}")

    return errors


def _test_gdpo_normalization() -> list[str]:
    """GDPO 정규화 수치 검증."""
    import torch
    from src.training.rl.advantage import gdpo_group_normalize
    errors = []

    try:
        # 그룹 크기 G=4, 보상 K=2, 2개 프롬프트 (B_total=8)
        # 그룹 2: [4, 4, 4, 4] → mean=4, std=0 → advantage=0
        G = 4
        rewards = torch.tensor([
            [0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0],  # 그룹 1
            [4.0, 1.0], [4.0, 1.0], [4.0, 1.0], [4.0, 1.0],  # 그룹 2 (동일값)
        ])  # (8, 2)

        A_k = gdpo_group_normalize(rewards, num_generations=G, eps=1e-8)

        # 그룹 2는 동일값이므로 advantage=0
        group2_A = A_k[4:8, 0]
        if group2_A.abs().max().item() > 1e-4:
            errors.append(f"GDPO: 동일값 그룹에서 advantage != 0 ({group2_A.tolist()})")

        # 그룹 1 advantage는 표준화되어야 함 (mean≈0)
        group1_A = A_k[:4, 0]
        if abs(group1_A.mean().item()) > 1e-4:
            errors.append(f"GDPO: 그룹 1 mean != 0 ({group1_A.mean().item():.4f})")

        if not errors:
            logger.info("  [OK] GDPO 정규화 수치 검증")
    except Exception as e:
        errors.append(f"GDPO 정규화 예외: {e}")

    return errors


def _test_credit_assignment() -> list[str]:
    """토큰 신용할당 수치 검증."""
    import torch
    from src.training.rl.rewards.credit_assignment import (
        build_error_mask,
        apply_token_credit_assignment,
    )
    errors = []

    try:
        # A=2.0 (양수 advantage), mask[2]=1 (오류 토큰)
        seq_len = 5
        error_mask = build_error_mask(seq_len, error_indices=[2])
        token_A = apply_token_credit_assignment(
            advantage=2.0,
            error_mask=error_mask,
            penalty_scale=1.5,
        )

        # 정상 토큰: A=2.0
        for i in [0, 1, 3, 4]:
            if abs(token_A[i].item() - 2.0) > 1e-5:
                errors.append(f"신용할당: 정상 토큰[{i}]={token_A[i].item()} (기대: 2.0)")

        # 오류 토큰[2]: -|A| × penalty = -2.0 × 1.5 = -3.0
        expected_err = -2.0 * 1.5
        if abs(token_A[2].item() - expected_err) > 1e-5:
            errors.append(f"신용할당: 오류 토큰[2]={token_A[2].item()} (기대: {expected_err})")

        # A=-1.5 (음수 advantage), mask[1]=1
        error_mask2 = build_error_mask(seq_len, error_indices=[1])
        token_A2 = apply_token_credit_assignment(
            advantage=-1.5,
            error_mask=error_mask2,
            penalty_scale=2.0,
        )

        # 오류 토큰[1]: -|-1.5| × 2.0 = -3.0 (A가 음수여도 항상 음수 방향 페널티)
        expected_err2 = -1.5 * 2.0
        if abs(token_A2[1].item() - expected_err2) > 1e-5:
            errors.append(
                f"신용할당: 음수 A, 오류 토큰[1]={token_A2[1].item()} (기대: {expected_err2})"
            )

        if not errors:
            logger.info("  [OK] 토큰 신용할당 수치 검증 (양수/음수 advantage)")
    except Exception as e:
        errors.append(f"신용할당 예외: {e}")

    return errors


# ---------------------------------------------------------------------------
# 더미 SFT adapter 생성 헬퍼 (SFT 훈련 미완료 시 테스트용)
# ---------------------------------------------------------------------------

def _create_dummy_sft_adapter(cfg, output_dir: Path) -> bool:
    """테스트 목적으로 초기화된 SFT adapter를 생성한다.

    실제 SFT 훈련 없이 RL 멀티 어댑터 스태킹 구조 검증을 위해
    Hub + partial_state.pt에서 모델을 로드하여 LoRA adapter를 초기화 후 저장한다.

    Args:
        cfg: RL pipeline DictConfig.
        output_dir: adapter를 저장할 경로.

    Returns:
        생성 성공 시 True.
    """
    try:
        import torch
        from src.training.sft.model_loader import load_base_model_with_partial_state, build_lora_config
        from peft import get_peft_model

        output_dir.mkdir(parents=True, exist_ok=True)

        partial_state_path = Path(_PROJECT_ROOT) / str(cfg.model.pre_stage_dir) / "partial_state.pt"
        logger.info("더미 SFT adapter 생성 중 (Hub 모델 로드)...")
        base_model, tokenizer = load_base_model_with_partial_state(cfg, partial_state_path)

        # SFT와 동일한 LoRA config 사용
        lora_config = build_lora_config(cfg.lora)
        peft_model = get_peft_model(base_model, lora_config)
        peft_model.save_pretrained(str(output_dir))
        logger.info(f"더미 SFT adapter 저장 완료: {output_dir}")
        del peft_model, base_model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    except Exception as e:
        logger.error(f"더미 SFT adapter 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Phase 2: 모델 로드 + 구조 검증
# ---------------------------------------------------------------------------

def phase2_model_load() -> bool:
    """SFT final 모델 로드 및 RLTrainer 생성 검증.

    Returns:
        검증 통과 시 True.
    """
    logger.info("=== Phase 2: 모델 로드 + 구조 검증 ===")

    # Hydra 설정 로드
    try:
        from omegaconf import OmegaConf
        config_path = Path(_PROJECT_ROOT) / "config/training/rl/pipeline.yaml"
        cfg = OmegaConf.load(config_path)
        logger.info("RL pipeline.yaml 로드 완료")
    except Exception as e:
        logger.error(f"설정 로드 실패: {e}")
        return False

    # 필수 경로 존재 확인
    partial_state_path = Path(_PROJECT_ROOT) / str(cfg.model.pre_stage_dir) / "partial_state.pt"
    sft_adapter_dir = Path(_PROJECT_ROOT) / str(cfg.model.sft_adapter_dir)
    if not partial_state_path.exists():
        logger.error(f"partial_state.pt 없음: {partial_state_path}")
        return False

    # SFT adapter가 없으면 테스트용 더미 adapter를 임시 경로에 생성
    dummy_adapter_dir = None
    if not sft_adapter_dir.exists() or not (sft_adapter_dir / "adapter_config.json").exists():
        import tempfile, shutil
        dummy_adapter_dir = Path(tempfile.mkdtemp(prefix="rl_test_sft_"))
        logger.warning(f"SFT adapter 없음 — 테스트용 더미 adapter 생성: {dummy_adapter_dir}")
        OmegaConf.set_struct(cfg, False)
        cfg.model.sft_adapter_dir = str(dummy_adapter_dir)
        OmegaConf.set_struct(cfg, True)
        if not _create_dummy_sft_adapter(cfg, dummy_adapter_dir):
            shutil.rmtree(dummy_adapter_dir, ignore_errors=True)
            return False

    # 모델 로드 (Hub + partial_state + SFT adapter frozen + RL adapter trainable)
    try:
        from src.training.rl.model_loader import load_model_and_tokenizer
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        _vram_report("모델 로드 전")
        logger.info("모델 로드 중 (약 1~2분 소요)...")
        model, tokenizer = load_model_and_tokenizer(cfg)
        _vram_report("모델 로드 후")
        logger.info(f"모델 로드 완료. vocab_size: {model.config.vocab_size}")
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        return False

    # Vocab 로드
    try:
        from src.training.augmentation.tokenizer import load_vocab
        vocab_path = Path(_PROJECT_ROOT) / str(cfg.model.vocab_extension)
        vocab = load_vocab(vocab_path, str(Path(_PROJECT_ROOT) / str(cfg.model.tokenizer_dir)))
        logger.info(f"Vocab 로드 완료: {len(vocab.token_to_id)}개 커스텀 토큰")
    except Exception as e:
        logger.error(f"Vocab 로드 실패: {e}")
        return False

    # RLTrainer 생성 가능 여부 확인
    try:
        from trl import GRPOConfig
        from src.training.rl.trainer import RLTrainer

        grpo_config = GRPOConfig(
            output_dir="/tmp/rl_test",
            num_generations=2,
            generation_batch_size=2,
            max_completion_length=128,
            per_device_train_batch_size=1,
        )

        from datasets import Dataset
        dummy_dataset = Dataset.from_dict({"prompt": ["test prompt"] * 4})

        trainer = RLTrainer(
            model=model,
            args=grpo_config,
            train_dataset=dummy_dataset,
            processing_class=tokenizer,
            reward_cfg=cfg.rewards,
            advantage_cfg=cfg.advantage,
            vocab=vocab,
        )
        logger.info("RLTrainer 생성 완료")
        logger.info(f"활성화된 보상함수: {trainer._reward_names}")
    except Exception as e:
        logger.error(f"RLTrainer 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        if dummy_adapter_dir is not None:
            import shutil
            shutil.rmtree(dummy_adapter_dir, ignore_errors=True)
        return False

    # 멀티 어댑터 구조 검증 (sft frozen, rl trainable)
    try:
        # --- 1단계: 구조적 확인 ---
        active_adapters = model.base_model.active_adapter
        if isinstance(active_adapters, str):
            active_adapters = [active_adapters]
        active_set = set(active_adapters)

        if "sft" in active_set and "rl" in active_set:
            logger.info(f"  [OK] 활성 어댑터 구조 확인: {active_adapters}")
        else:
            logger.error(f"  [FAIL] 활성 어댑터 누락: {active_adapters} (sft+rl 모두 필요)")
            if dummy_adapter_dir is not None:
                import shutil
                shutil.rmtree(dummy_adapter_dir, ignore_errors=True)
            return False

        # --- 2단계: SFT adapter 기여 기능적 확인 (임시 lora_B 주입) ---
        device = next(model.parameters()).device
        dummy_ids = torch.randint(1000, 5000, (1, 8), device=device)

        sft_lora_b_backup = {}
        for name, param in model.named_parameters():
            if ".sft." in name and "lora_B" in name:
                sft_lora_b_backup[name] = param.data.clone()
                param.data.fill_(0.01)  # 임시 비제로 값

        with torch.no_grad():
            output_both = model(dummy_ids).logits.clone()
            model.base_model.set_adapter(["rl"])
            output_rl_only = model(dummy_ids).logits.clone()
            model.base_model.set_adapter(["sft", "rl"])
            for name, param in model.named_parameters():
                if ".sft." in name:
                    param.requires_grad_(False)

        # SFT lora_B 원본값 복원
        for name, param in model.named_parameters():
            if name in sft_lora_b_backup:
                param.data.copy_(sft_lora_b_backup[name])

        max_diff = (output_both - output_rl_only).abs().max().item()
        if max_diff > 1e-5:
            logger.info(f"  [OK] SFT adapter 기여 기능 확인 (max_diff={max_diff:.6f})")
        else:
            logger.error(f"  [FAIL] SFT adapter 비제로 가중치에서도 기여 없음 (max_diff={max_diff:.6f})")
            if dummy_adapter_dir is not None:
                import shutil
                shutil.rmtree(dummy_adapter_dir, ignore_errors=True)
            return False

        # --- 3단계: requires_grad 상태 확인 ---
        sft_grad = any(p.requires_grad for n, p in model.named_parameters() if ".sft." in n)
        rl_grad = any(p.requires_grad for n, p in model.named_parameters() if ".rl." in n)
        if not sft_grad and rl_grad:
            logger.info("  [OK] SFT frozen / RL trainable 상태 확인")
        else:
            logger.error(f"  [FAIL] requires_grad 상태 이상: sft={sft_grad}, rl={rl_grad}")
            if dummy_adapter_dir is not None:
                import shutil
                shutil.rmtree(dummy_adapter_dir, ignore_errors=True)
            return False

    except Exception as e:
        logger.error(f"멀티 어댑터 구조 검증 실패: {e}")
        import traceback
        traceback.print_exc()
        if dummy_adapter_dir is not None:
            import shutil
            shutil.rmtree(dummy_adapter_dir, ignore_errors=True)
        return False
    finally:
        if dummy_adapter_dir is not None:
            import shutil
            shutil.rmtree(dummy_adapter_dir, ignore_errors=True)
            logger.info(f"더미 SFT adapter 임시 디렉토리 삭제: {dummy_adapter_dir}")

    logger.info("Phase 2 완료: 모델 로드 및 구조 검증 통과")
    return True


# ---------------------------------------------------------------------------
# Phase 3: 소규모 훈련 루프
# ---------------------------------------------------------------------------

def phase3_mini_training(use_vllm: bool = False) -> bool:
    """소규모 RL 훈련 루프 검증 (3 step).

    Args:
        use_vllm: True면 vLLM colocate 모드로 rollout 생성. False면 HF generate 사용.

    Returns:
        검증 통과 시 True.
    """
    vllm_label = "vllm=colocate" if use_vllm else "use_vllm=false"
    logger.info(f"=== Phase 3: 소규모 훈련 루프 (3 step, {vllm_label}) ===")

    try:
        from omegaconf import OmegaConf
        config_path = Path(_PROJECT_ROOT) / "config/training/rl/pipeline.yaml"
        cfg = OmegaConf.load(config_path)
        OmegaConf.set_struct(cfg, False)
        cfg.rl.use_vllm = use_vllm
        if use_vllm:
            cfg.rl.vllm_mode = "colocate"
            # 검증 환경(단일 GPU)에서 OOM 방지: 기본 0.45보다 낮게 설정
            cfg.rl.vllm_gpu_memory_utilization = 0.35
        OmegaConf.set_struct(cfg, True)
    except Exception as e:
        logger.error(f"설정 로드 실패: {e}")
        return False

    partial_state_path = Path(_PROJECT_ROOT) / str(cfg.model.pre_stage_dir) / "partial_state.pt"
    sft_adapter_dir = Path(_PROJECT_ROOT) / str(cfg.model.sft_adapter_dir)
    if not partial_state_path.exists():
        logger.error(f"partial_state.pt 없음: {partial_state_path}")
        return False

    dummy_adapter_dir = None
    if not sft_adapter_dir.exists() or not (sft_adapter_dir / "adapter_config.json").exists():
        import tempfile
        dummy_adapter_dir = Path(tempfile.mkdtemp(prefix="rl_test_sft_"))
        logger.warning(f"SFT adapter 없음 — 테스트용 더미 adapter 생성: {dummy_adapter_dir}")
        from omegaconf import OmegaConf as OC
        OC.set_struct(cfg, False)
        cfg.model.sft_adapter_dir = str(dummy_adapter_dir)
        OC.set_struct(cfg, True)
        if not _create_dummy_sft_adapter(cfg, dummy_adapter_dir):
            import shutil
            shutil.rmtree(dummy_adapter_dir, ignore_errors=True)
            return False

    try:
        from src.training.rl.model_loader import load_model_and_tokenizer, prepare_vllm_base_model
        from src.training.augmentation.tokenizer import load_vocab
        from trl import GRPOConfig
        from src.training.rl.trainer import RLTrainer
        from src.training.rl.dataset import RLPromptDataset
        from omegaconf import OmegaConf as OC

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        _vram_report("모델 로드 전")

        logger.info("모델 로드 중...")
        model, tokenizer = load_model_and_tokenizer(cfg)
        _vram_report("모델 로드 후")

        # vLLM colocate 모드: vocab 확장 base 모델 준비 후 name_or_path 교체
        if use_vllm:
            logger.info("vLLM base 모델 준비 중 (vocab 확장 적용)...")
            vllm_base_dir = prepare_vllm_base_model(cfg, model, tokenizer)
            # Mod Record: PeftModel.name_or_path은 nn.Module.__getattribute__가 class property보다
            # instance __dict__를 우선 확인하므로, Qwen2ForCausalLM.__dict__['name_or_path']에
            # 저장된 Hub ID가 반환된다. model.config.name_or_path만 수정하면 TRL의
            # VLLMGeneration(model=model.name_or_path) 호출에 반영되지 않아 vllm이 원본
            # Hub vocab(151936)으로 초기화되고 sync_weights() 시 152232 != 152064 assertion 실패.
            # 따라서 하위 PreTrainedModel의 __dict__ 값도 직접 덮어써야 한다.
            model.base_model.model.name_or_path = vllm_base_dir
            model.config.name_or_path = vllm_base_dir
            logger.info(f"vLLM name_or_path 설정 완료: {vllm_base_dir}")

        vocab_path = Path(_PROJECT_ROOT) / str(cfg.model.vocab_extension)

        # 증강 설정 병합
        aug_config_path = Path(_PROJECT_ROOT) / str(cfg.data.aug_pipeline_config)
        if aug_config_path.exists():
            aug_cfg = OC.load(aug_config_path)
            OC.set_struct(cfg, False)
            OC.update(cfg, "augmentation", aug_cfg, merge=True)
            OC.set_struct(cfg, True)

        vocab = load_vocab(vocab_path, str(Path(_PROJECT_ROOT) / str(cfg.model.tokenizer_dir)))
        train_dataset = RLPromptDataset(cfg, tokenizer, split="train", seed=42)

        # Mod Record: vllm SamplingParams는 HF generate의 eos_token_id 대신
        # stop_token_ids를 사용한다. 모드에 따라 다른 키를 사용해야 한다.
        # Mod Record: vllm은 stop_token_ids에 포함된 토큰을 출력에서 제거(strip)한다.
        # 151643(<|endoftext|>)을 stop_token_ids에 넣으면 vllm이 이를 출력에서 제거하고,
        # TRL의 clipped 감지 로직(ids[-1] not in eos_and_pad)이 항상 True가 되어
        # clipped_ratio=1이 되는 문제가 발생했다. vllm은 config의 eos_token_id(151643)를
        # 자연적 EOS로 처리하여 출력에 포함시키므로, stop_token_ids에는 커스텀 토큰만 넣는다.
        if use_vllm:
            # vllm: config.eos_token_id(151643)는 자동 처리, 커스텀 토큰만 stop으로 등록
            _generation_kwargs = {"stop_token_ids": [vocab.token_to_id["<END_OUTPUT>"]]}
        else:
            # HF generate: 두 토큰 모두 eos_token_id에 명시
            _generation_kwargs = {"eos_token_id": [tokenizer.eos_token_id, vocab.token_to_id["<END_OUTPUT>"]]}
        grpo_config_kwargs = dict(
            output_dir="/tmp/rl_phase3",
            num_generations=4,
            generation_batch_size=4,
            max_completion_length=512,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=3,
            report_to="none",
            logging_steps=1,
            save_strategy="no",
            # <END_OUTPUT>을 추가 EOS로 등록 (run_rl.py 실제 설정과 동일)
            generation_kwargs=_generation_kwargs,
        )
        if use_vllm:
            grpo_config_kwargs.update(dict(
                use_vllm=True,
                vllm_mode="colocate",
                vllm_gpu_memory_utilization=float(cfg.rl.vllm_gpu_memory_utilization),
                vllm_enable_sleep_mode=False,
                vllm_tensor_parallel_size=1,
                vllm_max_model_length=int(cfg.rl.vllm_max_model_len),
            ))
        grpo_config = GRPOConfig(**grpo_config_kwargs)

        trainer = RLTrainer(
            model=model,
            args=grpo_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            reward_cfg=cfg.rewards,
            advantage_cfg=cfg.advantage,
            vocab=vocab,
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        _vram_report("훈련 시작 전")

        logger.info("훈련 시작 (3 step, num_generations=4, batch=2)...")
        train_result = trainer.train()
        _vram_report("훈련 완료 후")
        logger.info(f"훈련 완료: {train_result.metrics}")

        # loss가 NaN/Inf가 아닌지 확인
        final_loss = train_result.metrics.get("train_loss", float("nan"))
        import math
        if math.isnan(final_loss) or math.isinf(final_loss):
            logger.error(f"Loss가 NaN/Inf: {final_loss}")
            return False

        logger.info(f"  최종 loss: {final_loss:.4f}")
    except Exception as e:
        logger.error(f"소규모 훈련 루프 실패: {e}")
        import traceback
        traceback.print_exc()
        if dummy_adapter_dir is not None:
            import shutil
            shutil.rmtree(dummy_adapter_dir, ignore_errors=True)
        return False
    finally:
        if dummy_adapter_dir is not None:
            import shutil
            shutil.rmtree(dummy_adapter_dir, ignore_errors=True)
            logger.info(f"더미 SFT adapter 임시 디렉토리 삭제: {dummy_adapter_dir}")

    logger.info("Phase 3 완료: 소규모 훈련 루프 통과")
    return True


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    """검증 스크립트 메인 함수."""
    parser = argparse.ArgumentParser(description="RL 구현 검증")
    parser.add_argument(
        "--phase",
        type=int,
        choices=[0, 1, 2, 3],
        default=None,
        help="실행할 Phase (미지정 시 전체 실행)",
    )
    parser.add_argument(
        "--vllm",
        action="store_true",
        default=False,
        help="Phase 3을 vLLM colocate 모드로 실행 (기본: HF generate)",
    )
    args = parser.parse_args()

    phase3_func = lambda: phase3_mini_training(use_vllm=args.vllm)  # noqa: E731
    phases = {
        0: ("파일 존재 확인", phase0_file_existence),
        1: ("보상함수 단위 테스트", phase1_reward_unit_tests),
        2: ("모델 로드 + 구조 검증", phase2_model_load),
        3: ("소규모 훈련 루프", phase3_func),
    }

    if args.phase is not None:
        phase_name, phase_func = phases[args.phase]
        success = phase_func()
        status = "PASS" if success else "FAIL"
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Phase {args.phase} ({phase_name}): {status}")
        sys.exit(0 if success else 1)
    else:
        results = {}
        for phase_idx, (phase_name, phase_func) in phases.items():
            try:
                success = phase_func()
            except Exception as e:
                logger.error(f"Phase {phase_idx} 예외 발생: {e}")
                success = False
            results[phase_idx] = (phase_name, success)

        logger.info(f"\n{'=' * 50}")
        logger.info("검증 결과 요약:")
        all_pass = True
        for phase_idx, (phase_name, success) in results.items():
            status = "PASS" if success else "FAIL"
            logger.info(f"  Phase {phase_idx} ({phase_name}): {status}")
            all_pass = all_pass and success

        logger.info(f"\n전체 결과: {'PASS' if all_pass else 'FAIL'}")
        sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
