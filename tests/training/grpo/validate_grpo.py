"""GRPO 구현 4단계 검증 스크립트.

Phase 0: 파일 존재 확인
Phase 1: 보상함수 단위 테스트 (GPU 불필요)
Phase 2: 모델 로드 + 구조 검증 (GPU 1개)
Phase 3: 소규모 훈련 루프 (GPU 1개, 3~5 step)

사용법:
    # 전체 phase 실행
    uv run python tests/training/grpo/validate_grpo.py

    # 특정 phase만
    uv run python tests/training/grpo/validate_grpo.py --phase 1
    uv run python tests/training/grpo/validate_grpo.py --phase 2
    uv run python tests/training/grpo/validate_grpo.py --phase 3
"""

import argparse
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


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
        "config/training/grpo/pipeline.yaml",
        "src/training/grpo/__init__.py",
        "src/training/grpo/trainer.py",
        "src/training/grpo/dataset.py",
        "src/training/grpo/model_loader.py",
        "src/training/grpo/advantage.py",
        "src/training/grpo/rewards/__init__.py",
        "src/training/grpo/rewards/parser.py",
        "src/training/grpo/rewards/format_reward.py",
        "src/training/grpo/rewards/count_reward.py",
        "src/training/grpo/rewards/geometry_reward.py",
        "src/training/grpo/rewards/connectivity_reward.py",
        "src/training/grpo/rewards/spatial_reward.py",
        "src/training/grpo/rewards/credit_assignment.py",
        "scripts/training/run_grpo.py",
    ]

    missing = []
    for rel_path in required_files:
        full_path = Path(_PROJECT_ROOT) / rel_path
        if full_path.exists():
            logger.info(f"  [OK] {rel_path}")
        else:
            logger.error(f"  [MISSING] {rel_path}")
            missing.append(rel_path)

    # partial_state.pt 확인 (새 구조: pre_stage/final에 partial_state.pt만 저장됨)
    partial_state_path = Path(_PROJECT_ROOT) / "data/models/Qwen2.5-Coder-7B/checkpoints/pre_stage/final/partial_state.pt"
    if partial_state_path.exists():
        logger.info(f"  [OK] pre_stage partial_state.pt: {partial_state_path}")
    else:
        logger.warning(f"  [WARNING] partial_state.pt 없음: {partial_state_path} (Phase 2-3 실행 불가)")

    # SFT adapter 확인 (새 구조: sft/final에 adapter_model.safetensors만 저장됨)
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

    # <OUTPUT>
    # <FRONT_DOOR> <NO_DOOR> <END_DOOR>
    # <ROOM> <TYPE:bedroom> <X:10> <Y:10> <X:50> <Y:10> <X:50> <Y:50> <X:10> <Y:50> <END_ROOM>
    # <ROOM> <TYPE:outline> <X:0> <Y:0> <X:100> <Y:0> <X:100> <Y:100> <X:0> <Y:100> <END_ROOM>
    # <END_OUTPUT>
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
    from src.training.grpo.rewards.parser import parse_output_tokens
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
    from src.training.grpo.rewards.parser import parse_output_tokens
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
    from src.training.grpo.rewards.parser import parse_output_tokens
    from src.training.grpo.rewards.format_reward import compute_format_reward
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
    from src.training.grpo.rewards.parser import parse_output_tokens
    from src.training.grpo.rewards.count_reward import (
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
    from src.training.grpo.rewards.parser import parse_output_tokens
    from src.training.grpo.rewards.geometry_reward import compute_orthogonality_reward
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
    from src.training.grpo.rewards.parser import parse_output_tokens
    from src.training.grpo.rewards.geometry_reward import compute_no_overlap_reward
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
    from src.training.grpo.advantage import gdpo_group_normalize
    errors = []

    try:
        # 그룹 크기 G=4, 보상 K=2, 2개 프롬프트 (B_total=8)
        # 그룹 1: [0, 1, 2, 3] → mean=1.5, std=sqrt(5/4)≈1.118
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
            logger.info(f"  [OK] GDPO 정규화 수치 검증")
    except Exception as e:
        errors.append(f"GDPO 정규화 예외: {e}")

    return errors


def _test_credit_assignment() -> list[str]:
    """토큰 신용할당 수치 검증."""
    import torch
    from src.training.grpo.rewards.credit_assignment import (
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

    실제 SFT 훈련 없이 GRPO 멀티 어댑터 스태킹 구조 검증을 위해
    Hub + partial_state.pt에서 모델을 로드하여 DoRA adapter를 초기화 후 저장한다.

    Args:
        cfg: GRPO pipeline DictConfig.
        output_dir: adapter를 저장할 경로.

    Returns:
        생성 성공 시 True.
    """
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from src.training.pre_stage.model_loader import load_model_with_partial_state

        output_dir.mkdir(parents=True, exist_ok=True)

        partial_state_path = Path(_PROJECT_ROOT) / str(cfg.model.pre_stage_dir) / "partial_state.pt"
        logger.info("더미 SFT adapter 생성 중 (Hub 모델 로드)...")
        base_model, tokenizer = load_model_with_partial_state(cfg, partial_state_path)

        # SFT와 동일한 DoRA config 사용
        dora_config = LoraConfig(
            r=cfg.dora.r,
            lora_alpha=cfg.dora.lora_alpha,
            lora_dropout=cfg.dora.lora_dropout,
            target_modules=list(cfg.dora.target_modules),
            bias=cfg.dora.bias,
            task_type=TaskType.CAUSAL_LM,
            use_dora=True,
        )
        peft_model = get_peft_model(base_model, dora_config)
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
    """SFT final 모델 로드 및 GDPOTrainer 생성 검증.

    Returns:
        검증 통과 시 True.
    """
    logger.info("=== Phase 2: 모델 로드 + 구조 검증 ===")

    # Hydra 설정 로드
    try:
        from omegaconf import OmegaConf
        config_path = Path(_PROJECT_ROOT) / "config/training/grpo/pipeline.yaml"
        cfg = OmegaConf.load(config_path)
        logger.info("GRPO pipeline.yaml 로드 완료")
    except Exception as e:
        logger.error(f"설정 로드 실패: {e}")
        return False

    # 필수 경로 존재 확인 (새 구조: partial_state.pt + SFT adapter)
    partial_state_path = Path(_PROJECT_ROOT) / str(cfg.model.pre_stage_dir) / "partial_state.pt"
    sft_adapter_dir = Path(_PROJECT_ROOT) / str(cfg.model.sft_adapter_dir)
    if not partial_state_path.exists():
        logger.error(f"partial_state.pt 없음: {partial_state_path}")
        return False

    # SFT adapter가 없으면 테스트용 더미 adapter를 임시 경로에 생성
    dummy_adapter_dir = None
    if not sft_adapter_dir.exists() or not (sft_adapter_dir / "adapter_config.json").exists():
        import tempfile, shutil
        dummy_adapter_dir = Path(tempfile.mkdtemp(prefix="grpo_test_sft_"))
        logger.warning(f"SFT adapter 없음 — 테스트용 더미 adapter 생성: {dummy_adapter_dir}")
        OmegaConf.set_struct(cfg, False)
        # 상대경로로 저장된 cfg 값을 절대경로로 교체 (OmegaConf string override)
        cfg.model.sft_adapter_dir = str(dummy_adapter_dir)
        OmegaConf.set_struct(cfg, True)
        if not _create_dummy_sft_adapter(cfg, dummy_adapter_dir):
            shutil.rmtree(dummy_adapter_dir, ignore_errors=True)
            return False

    # 모델 로드 (Hub + partial_state + SFT adapter + GRPO adapter 스태킹)
    try:
        from src.training.grpo.model_loader import load_model_and_tokenizer
        logger.info("모델 로드 중 (약 1~2분 소요)...")
        model, tokenizer = load_model_and_tokenizer(cfg)
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

    # GDPOTrainer 생성 가능 여부 확인
    try:
        from trl import GRPOConfig
        from src.training.grpo.trainer import GDPOTrainer

        grpo_config = GRPOConfig(
            output_dir="/tmp/grpo_test",
            num_generations=2,
            generation_batch_size=2,
            max_completion_length=128,
            per_device_train_batch_size=1,
        )

        # TRL이 인스턴스화 시에도 train_dataset을 요구하므로 더미 데이터셋 제공
        from datasets import Dataset
        dummy_dataset = Dataset.from_dict({"prompt": ["test prompt"] * 4})

        trainer = GDPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=dummy_dataset,
            processing_class=tokenizer,
            reward_cfg=cfg.rewards,
            advantage_cfg=cfg.advantage,
            vocab=vocab,
        )
        logger.info("GDPOTrainer 생성 완료")
        logger.info(f"활성화된 보상함수: {trainer._reward_names}")
    except Exception as e:
        logger.error(f"GDPOTrainer 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        if dummy_adapter_dir is not None:
            import shutil
            shutil.rmtree(dummy_adapter_dir, ignore_errors=True)
        return False

    # SFT adapter 기여 검증
    # 1단계: 구조적 확인 — active_adapters 리스트에 "sft", "grpo" 모두 포함되어야 함
    # 2단계: 기능적 확인 — lora_B를 임시 비제로 값으로 교체 후 출력 비교
    #   (더미 adapter는 lora_B=0 초기화이므로, 제로 가중치로는 기여 차이를 관찰 불가)
    try:
        import torch

        # --- 1단계: 구조적 확인 ---
        active_adapters = model.base_model.active_adapter
        if isinstance(active_adapters, str):
            active_adapters = [active_adapters]
        active_set = set(active_adapters)

        if "sft" in active_set and "grpo" in active_set:
            logger.info(f"  [OK] 활성 어댑터 구조 확인: {active_adapters}")
        else:
            logger.error(f"  [FAIL] 활성 어댑터 누락: {active_adapters} (sft+grpo 모두 필요)")
            if dummy_adapter_dir is not None:
                import shutil
                shutil.rmtree(dummy_adapter_dir, ignore_errors=True)
            return False

        # --- 2단계: 기능적 확인 (임시 lora_B 주입) ---
        device = next(model.parameters()).device
        dummy_ids = torch.randint(1000, 5000, (1, 8), device=device)

        # SFT lora_B 파라미터를 임시로 비제로 값으로 교체
        # (원본 값이 0이면 adapter 활성 여부와 무관하게 출력 동일 → 임시 주입으로 확인)
        sft_lora_b_backup = {}
        for name, param in model.named_parameters():
            if ".sft." in name and "lora_B" in name:
                sft_lora_b_backup[name] = param.data.clone()
                param.data.fill_(0.01)  # 임시 비제로 값

        with torch.no_grad():
            # sft + grpo 모두 활성 (현재 상태)
            output_both = model(dummy_ids).logits.clone()

            # SFT만 끄고 grpo만 활성
            model.base_model.set_adapter(["grpo"])
            output_grpo_only = model(dummy_ids).logits.clone()

            # 원래 어댑터 상태 복원
            model.base_model.set_adapter(["sft", "grpo"])
            for name, param in model.named_parameters():
                if ".sft." in name:
                    param.requires_grad_(False)

        # SFT lora_B 원본값 복원
        for name, param in model.named_parameters():
            if name in sft_lora_b_backup:
                param.data.copy_(sft_lora_b_backup[name])

        max_diff = (output_both - output_grpo_only).abs().max().item()
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
        grpo_grad = any(p.requires_grad for n, p in model.named_parameters() if ".grpo." in n)
        if not sft_grad and grpo_grad:
            logger.info("  [OK] SFT frozen / GRPO trainable 상태 확인")
        else:
            logger.error(f"  [FAIL] requires_grad 상태 이상: sft={sft_grad}, grpo={grpo_grad}")
            if dummy_adapter_dir is not None:
                import shutil
                shutil.rmtree(dummy_adapter_dir, ignore_errors=True)
            return False

    except Exception as e:
        logger.error(f"SFT adapter 기여 검증 실패: {e}")
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

def phase3_mini_training() -> bool:
    """소규모 GRPO 훈련 루프 검증 (3~5 step).

    Returns:
        검증 통과 시 True.
    """
    logger.info("=== Phase 3: 소규모 훈련 루프 (3 step) ===")

    try:
        from omegaconf import OmegaConf
        config_path = Path(_PROJECT_ROOT) / "config/training/grpo/pipeline.yaml"
        cfg = OmegaConf.load(config_path)
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
        dummy_adapter_dir = Path(tempfile.mkdtemp(prefix="grpo_test_sft_"))
        logger.warning(f"SFT adapter 없음 — 테스트용 더미 adapter 생성: {dummy_adapter_dir}")
        OmegaConf = __import__("omegaconf", fromlist=["OmegaConf"]).OmegaConf
        OmegaConf.set_struct(cfg, False)
        cfg.model.sft_adapter_dir = str(dummy_adapter_dir)
        OmegaConf.set_struct(cfg, True)
        if not _create_dummy_sft_adapter(cfg, dummy_adapter_dir):
            import shutil
            shutil.rmtree(dummy_adapter_dir, ignore_errors=True)
            return False

    try:
        from src.training.grpo.model_loader import load_model_and_tokenizer
        from src.training.augmentation.tokenizer import load_vocab
        from trl import GRPOConfig
        from src.training.grpo.trainer import GDPOTrainer
        from src.training.grpo.dataset import GRPOPromptDataset

        logger.info("모델 로드 중...")
        model, tokenizer = load_model_and_tokenizer(cfg)

        vocab_path = Path(_PROJECT_ROOT) / str(cfg.model.vocab_extension)

        # 증강 설정 병합
        from omegaconf import OmegaConf as OC
        aug_config_path = Path(_PROJECT_ROOT) / str(cfg.data.aug_pipeline_config)
        if aug_config_path.exists():
            aug_cfg = OC.load(aug_config_path)
            OC.set_struct(cfg, False)
            OC.update(cfg, "augmentation", aug_cfg, merge=True)
            OC.set_struct(cfg, True)

        vocab = load_vocab(vocab_path, str(Path(_PROJECT_ROOT) / str(cfg.model.tokenizer_dir)))
        train_dataset = GRPOPromptDataset(cfg, tokenizer, split="train", seed=42)

        grpo_config = GRPOConfig(
            output_dir="/tmp/grpo_phase3",
            num_generations=2,       # 최소 생성 수
            generation_batch_size=2,
            max_completion_length=512,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=3,             # 3 step만 실행
            report_to="none",        # W&B 비활성화
            logging_steps=1,
            save_strategy="no",
        )

        trainer = GDPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            reward_cfg=cfg.rewards,
            advantage_cfg=cfg.advantage,
            vocab=vocab,
        )

        logger.info("훈련 시작 (3 step)...")
        train_result = trainer.train()
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
    parser = argparse.ArgumentParser(description="GRPO 구현 검증")
    parser.add_argument(
        "--phase",
        type=int,
        choices=[0, 1, 2, 3],
        default=None,
        help="실행할 Phase (미지정 시 전체 실행)",
    )
    args = parser.parse_args()

    phases = {
        0: ("파일 존재 확인", phase0_file_existence),
        1: ("보상함수 단위 테스트", phase1_reward_unit_tests),
        2: ("모델 로드 + 구조 검증", phase2_model_load),
        3: ("소규모 훈련 루프", phase3_mini_training),
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
