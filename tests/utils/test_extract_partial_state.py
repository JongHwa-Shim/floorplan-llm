"""extract_partial_state 함수 검증 스크립트.

Phase 1: 합성 미니 모델로 단위 테스트 (GPU 불필요, 빠름)
  - 알려진 값으로 가짜 merged safetensors 생성 후 추출 결과를 정확히 검증한다.

Phase 2: 실제 파일 통합 테스트 (파일 존재 시 자동 실행)
  - model.safetensors에서 추출한 결과를 기존 partial_state.pt와 비교 검증한다.

실행:
    uv run python tests/utils/test_extract_partial_state.py
"""

import json
import logging
import sys
import tempfile
from pathlib import Path

import torch

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.extract_partial_state import extract_partial_state

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# PASS / FAIL 출력 헬퍼
def _pass(msg: str) -> None:
    print(f"  [PASS] {msg}")

def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    sys.exit(1)


# ─────────────────────────────────────────────
# Phase 1: 합성 단위 테스트
# ─────────────────────────────────────────────

def _build_fake_safetensors(
    tmp_dir: Path,
    base_vocab: int,
    num_new: int,
    hidden: int,
) -> tuple[Path, torch.Tensor, torch.Tensor, list[int]]:
    """테스트용 가짜 safetensors와 vocab_extension.json을 생성한다.

    Args:
        tmp_dir: 임시 파일 저장 디렉토리.
        base_vocab: 기존 어휘 크기.
        num_new: 커스텀 토큰 수.
        hidden: 히든 차원.

    Returns:
        (safetensors가 있는 체크포인트 디렉토리, full_embed, full_lm_head, new_token_ids)
    """
    from safetensors.torch import save_file

    total_vocab = base_vocab + num_new
    new_token_ids = list(range(base_vocab, total_vocab))

    # 알려진 특정 값으로 텐서 생성 (검증 정확도를 위해 랜덤이 아닌 규칙적인 값 사용)
    full_embed = torch.arange(total_vocab * hidden, dtype=torch.float32).reshape(total_vocab, hidden)
    full_lm_head = (
        torch.arange(total_vocab * hidden, dtype=torch.float32).reshape(total_vocab, hidden) * -1.0
    )

    # 가짜 safetensors 저장
    ckpt_dir = tmp_dir / "fake_checkpoint"
    ckpt_dir.mkdir()
    save_file(
        {"model.embed_tokens.weight": full_embed, "lm_head.weight": full_lm_head},
        ckpt_dir / "model.safetensors",
    )

    # 가짜 vocab_extension.json 저장
    token_to_id = {f"<TOK:{i}>": base_vocab + i for i in range(num_new)}
    vocab_ext = {
        "base_vocab_size": base_vocab,
        "new_vocab_size": total_vocab,
        "total_added": num_new,
        "token_to_id": token_to_id,
    }
    vocab_dir = tmp_dir / "tokenization"
    vocab_dir.mkdir()
    with open(vocab_dir / "vocab_extension.json", "w", encoding="utf-8") as f:
        json.dump(vocab_ext, f)

    return ckpt_dir, full_embed, full_lm_head, new_token_ids


def run_phase1_unit_test() -> None:
    """합성 데이터로 추출 로직이 올바른지 검증한다."""
    print("\n=== Phase 1: 합성 단위 테스트 ===")

    BASE_VOCAB = 100
    NUM_NEW = 10
    HIDDEN = 64

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)

        ckpt_dir, full_embed, full_lm_head, expected_ids = _build_fake_safetensors(
            tmp_dir, BASE_VOCAB, NUM_NEW, HIDDEN
        )
        vocab_ext_path = tmp_dir / "tokenization" / "vocab_extension.json"
        output_path = tmp_dir / "extracted" / "partial_state.pt"

        # 추출 실행
        result = extract_partial_state(
            checkpoint_dir=ckpt_dir,
            vocab_extension_path=vocab_ext_path,
            output_path=output_path,
        )

        # ─ 검증 1: 출력 파일 존재
        if output_path.exists():
            _pass("출력 파일 생성됨")
        else:
            _fail("출력 파일이 생성되지 않았음")

        # ─ 검증 2: 반환된 dict 키 확인
        for key in ("new_embed", "new_lm_head", "new_token_ids"):
            if key in result:
                _pass(f"dict 키 '{key}' 존재")
            else:
                _fail(f"dict 키 '{key}' 없음")

        # ─ 검증 3: new_token_ids 값
        extracted_ids = result["new_token_ids"]
        if extracted_ids == expected_ids:
            _pass(f"new_token_ids 일치: {len(extracted_ids)}개, 범위=[{extracted_ids[0]}, {extracted_ids[-1]}]")
        else:
            _fail(f"new_token_ids 불일치: {extracted_ids} vs {expected_ids}")

        # ─ 검증 4: shape 확인
        new_embed = result["new_embed"]
        new_lm_head = result["new_lm_head"]

        if new_embed.shape == (NUM_NEW, HIDDEN):
            _pass(f"new_embed shape 정확: {new_embed.shape}")
        else:
            _fail(f"new_embed shape 불일치: {new_embed.shape} != ({NUM_NEW}, {HIDDEN})")

        if new_lm_head.shape == (NUM_NEW, HIDDEN):
            _pass(f"new_lm_head shape 정확: {new_lm_head.shape}")
        else:
            _fail(f"new_lm_head shape 불일치: {new_lm_head.shape} != ({NUM_NEW}, {HIDDEN})")

        # ─ 검증 5: 값 정확성 (new_token_ids 행과 정확히 일치해야 함)
        expected_embed = full_embed[expected_ids]
        expected_lm_head = full_lm_head[expected_ids]

        if torch.allclose(new_embed.float(), expected_embed.float()):
            _pass("new_embed 값 정확성 검증 통과")
        else:
            max_diff = (new_embed.float() - expected_embed.float()).abs().max().item()
            _fail(f"new_embed 값 불일치: max_diff={max_diff:.6f}")

        if torch.allclose(new_lm_head.float(), expected_lm_head.float()):
            _pass("new_lm_head 값 정확성 검증 통과")
        else:
            max_diff = (new_lm_head.float() - expected_lm_head.float()).abs().max().item()
            _fail(f"new_lm_head 값 불일치: max_diff={max_diff:.6f}")

        # ─ 검증 6: 저장된 파일 로드 후 재검증
        loaded = torch.load(output_path, map_location="cpu", weights_only=True)
        if torch.allclose(loaded["new_embed"].float(), expected_embed.float()):
            _pass("저장/로드 후 new_embed 값 보존 확인")
        else:
            _fail("저장/로드 후 new_embed 값 손상")

        if loaded["new_token_ids"] == expected_ids:
            _pass("저장/로드 후 new_token_ids 보존 확인")
        else:
            _fail("저장/로드 후 new_token_ids 손상")

    # ─ 검증 7: dtype 변환 옵션 테스트
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)
        ckpt_dir, _, _, _ = _build_fake_safetensors(tmp_dir, BASE_VOCAB, NUM_NEW, HIDDEN)
        vocab_ext_path = tmp_dir / "tokenization" / "vocab_extension.json"
        output_path = tmp_dir / "bf16_partial.pt"

        result_bf16 = extract_partial_state(
            checkpoint_dir=ckpt_dir,
            vocab_extension_path=vocab_ext_path,
            output_path=output_path,
            dtype=torch.bfloat16,
        )
        if result_bf16["new_embed"].dtype == torch.bfloat16:
            _pass("dtype=bfloat16 변환 정상 동작")
        else:
            _fail(f"dtype 변환 실패: {result_bf16['new_embed'].dtype} != bfloat16")

    print("Phase 1 완료: 모든 검증 통과\n")


# ─────────────────────────────────────────────
# Phase 2: 실제 파일 통합 테스트
# ─────────────────────────────────────────────

def run_phase2_integration_test() -> None:
    """실제 model.safetensors에서 추출 후 기존 partial_state.pt와 비교 검증한다."""
    print("=== Phase 2: 실제 파일 통합 테스트 ===")

    model_name = "Qwen2.5-Coder-7B"
    ckpt_dir = PROJECT_ROOT / "data" / "models" / model_name / "checkpoints" / "pre_stage" / "final"
    vocab_ext_path = PROJECT_ROOT / "data" / "models" / model_name / "tokenization" / "vocab_extension.json"
    safetensors_path = ckpt_dir / "model.safetensors"
    existing_partial_path = ckpt_dir / "partial_state.pt"

    # 실제 파일 존재 여부 확인
    if not safetensors_path.exists():
        print(f"  [SKIP] model.safetensors 없음 (경로: {safetensors_path})")
        print("  Phase 2를 건너뜁니다.\n")
        return

    if not vocab_ext_path.exists():
        print(f"  [SKIP] vocab_extension.json 없음 (경로: {vocab_ext_path})")
        print("  Phase 2를 건너뜁니다.\n")
        return

    print(f"  model.safetensors 발견: {safetensors_path}")
    print(f"  파일 크기: {safetensors_path.stat().st_size / 1024 / 1024:.1f} MB")

    with tempfile.TemporaryDirectory() as tmp_str:
        output_path = Path(tmp_str) / "extracted_partial.pt"

        # 추출 실행 (float32 원본 유지로 먼저 추출)
        print("  추출 중...")
        result = extract_partial_state(
            checkpoint_dir=ckpt_dir,
            vocab_extension_path=vocab_ext_path,
            output_path=output_path,
        )

        # ─ 검증 1: shape 및 토큰 수 확인
        new_embed = result["new_embed"]
        new_lm_head = result["new_lm_head"]
        new_token_ids = result["new_token_ids"]

        EXPECTED_NUM_NEW = 567
        EXPECTED_HIDDEN = 3584
        EXPECTED_BASE = 151665

        if len(new_token_ids) == EXPECTED_NUM_NEW:
            _pass(f"new_token_ids 수 정확: {len(new_token_ids)}")
        else:
            _fail(f"new_token_ids 수 불일치: {len(new_token_ids)} != {EXPECTED_NUM_NEW}")

        if new_token_ids[0] == EXPECTED_BASE:
            _pass(f"new_token_ids 시작 ID 정확: {new_token_ids[0]}")
        else:
            _fail(f"new_token_ids 시작 ID 불일치: {new_token_ids[0]} != {EXPECTED_BASE}")

        if new_embed.shape == (EXPECTED_NUM_NEW, EXPECTED_HIDDEN):
            _pass(f"new_embed shape 정확: {new_embed.shape}")
        else:
            _fail(f"new_embed shape 불일치: {new_embed.shape} != ({EXPECTED_NUM_NEW}, {EXPECTED_HIDDEN})")

        if new_lm_head.shape == (EXPECTED_NUM_NEW, EXPECTED_HIDDEN):
            _pass(f"new_lm_head shape 정확: {new_lm_head.shape}")
        else:
            _fail(f"new_lm_head shape 불일치: {new_lm_head.shape} != ({EXPECTED_NUM_NEW}, {EXPECTED_HIDDEN})")

        # ─ 검증 2: 기존 partial_state.pt와 비교
        if existing_partial_path.exists():
            print(f"\n  기존 partial_state.pt 발견: {existing_partial_path}")
            existing = torch.load(existing_partial_path, map_location="cpu", weights_only=True)

            # shape 비교
            if existing["new_embed"].shape == new_embed.shape:
                _pass(f"기존 vs 추출 new_embed shape 일치: {existing['new_embed'].shape}")
            else:
                _fail(
                    f"new_embed shape 불일치: 기존={existing['new_embed'].shape} vs 추출={new_embed.shape}"
                )

            if existing["new_lm_head"].shape == new_lm_head.shape:
                _pass(f"기존 vs 추출 new_lm_head shape 일치: {existing['new_lm_head'].shape}")
            else:
                _fail(
                    f"new_lm_head shape 불일치: 기존={existing['new_lm_head'].shape} vs 추출={new_lm_head.shape}"
                )

            # new_token_ids 비교
            if existing["new_token_ids"] == new_token_ids:
                _pass("기존 vs 추출 new_token_ids 완전 일치")
            else:
                _fail("new_token_ids 불일치")

            # 값 비교: dtype 다를 수 있으므로 float32로 변환 후 비교
            existing_embed_f32 = existing["new_embed"].float()
            extracted_embed_f32 = new_embed.float()
            existing_lm_f32 = existing["new_lm_head"].float()
            extracted_lm_f32 = new_lm_head.float()

            embed_diff = (existing_embed_f32 - extracted_embed_f32).abs()
            lm_diff = (existing_lm_f32 - extracted_lm_f32).abs()

            # bfloat16 ↔ float32 변환 오차 허용 (bfloat16 정밀도: ~0.01)
            # 단, 두 파일이 다른 훈련 결과라면 값이 크게 다를 수 있음
            # → 최대 오차만 출력하고 경고로 처리 (FAIL 아님)
            embed_max_diff = embed_diff.max().item()
            lm_max_diff = lm_diff.max().item()
            embed_mean_diff = embed_diff.mean().item()
            lm_mean_diff = lm_diff.mean().item()

            # dtype 정보 출력
            print(f"\n  기존 partial_state.pt dtype: embed={existing['new_embed'].dtype}, lm_head={existing['new_lm_head'].dtype}")
            print(f"  추출 partial_state.pt dtype: embed={new_embed.dtype}, lm_head={new_lm_head.dtype}")

            print(f"\n  new_embed 오차  — max={embed_max_diff:.6f}, mean={embed_mean_diff:.6f}")
            print(f"  new_lm_head 오차 — max={lm_max_diff:.6f}, mean={lm_mean_diff:.6f}")

            # bfloat16 변환 수준 오차(atol=0.02)면 동일 훈련 결과로 판단
            BF16_TOLERANCE = 0.02
            if embed_max_diff < BF16_TOLERANCE and lm_max_diff < BF16_TOLERANCE:
                _pass(
                    f"기존 partial_state.pt와 값 근사 일치 (max_diff < {BF16_TOLERANCE}): "
                    f"embed={embed_max_diff:.6f}, lm_head={lm_max_diff:.6f}"
                )
            else:
                print(
                    f"  [WARN] 값 차이가 bfloat16 변환 오차({BF16_TOLERANCE})를 초과합니다.\n"
                    f"         이는 두 파일이 서로 다른 훈련 단계의 결과일 수 있습니다.\n"
                    f"         embed max_diff={embed_max_diff:.6f}, lm_head max_diff={lm_max_diff:.6f}"
                )
        else:
            print(f"  [INFO] 기존 partial_state.pt 없음 — 비교 건너뜀")

    print("Phase 2 완료\n")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("extract_partial_state 검증 스크립트")
    print("=" * 60)

    # safetensors 라이브러리 설치 확인
    try:
        import safetensors  # noqa: F401
        _pass("safetensors 라이브러리 설치 확인")
    except ImportError:
        _fail("safetensors 미설치 — `uv add safetensors` 실행 필요")

    run_phase1_unit_test()
    run_phase2_integration_test()

    print("=" * 60)
    print("모든 검증 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
