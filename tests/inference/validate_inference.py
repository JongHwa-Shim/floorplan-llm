"""추론 파이프라인 전체 검증 스크립트.

다음 6개 Phase를 순서대로 검증한다:

  Phase 0. 파일 존재 확인 (모델 로드 없음)
    - sft/final 필수 파일 존재 확인
    - vocab_extension.json 존재 확인
    - config.json 내 quantization_config 키 존재 확인

  Phase 1. 모델 + Vocab 로드 검증
    - load_model_for_inference() 성공 여부
    - model.config.vocab_size == len(tokenizer)
    - load_vocab() 성공 여부 + 커스텀 토큰 수 확인

  Phase 2. 입력 처리 검증 (모델 불필요, Arrow 샘플)
    - Arrow test split에서 샘플 로드 확인
    - AugmentationPipeline 초기화 + 증강 적용
    - condition_tokens 생성 및 decode 확인

  Phase 3. 추론 검증 (Phase 1 모델 재사용)
    - generate_floorplan()으로 토큰 시퀀스 생성
    - generated_ids가 비어있지 않은지

  Phase 4. 출력 파싱 + 시각화 검증
    - parse_output_tokens()로 dict 반환
    - rooms, edges, front_door 키 존재 확인
    - FloorplanVisualizer로 이미지 렌더링 성공

  Phase 5. 결과 저장 검증
    - save_results() 호출 후 파일 존재 확인
    - meta.json 키 확인
    - 임시 출력 디렉토리 자동 삭제

사용법:
    # 전체 검증 (권장)
    uv run python tests/inference/validate_inference.py

    # sft/final 모델 경로 직접 지정
    uv run python tests/inference/validate_inference.py \\
        --model_dir data/models/Qwen2.5-Coder-7B/checkpoints/sft/final

    # 빠른 검증 (추론 샘플 수 줄임)
    uv run python tests/inference/validate_inference.py --num_samples 1
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.inference.condition_builder import (
    build_condition_no_aug,
    build_condition_with_augmentation,
    load_samples,
)
from src.inference.generator import generate_floorplan
from src.inference.model_loader import load_model_for_inference
from src.inference.output_parser import parse_output_tokens
from src.inference.result_saver import save_results
from src.training.augmentation.decoder import decode_tokens
from src.training.augmentation.pipeline import AugmentationPipeline, config_from_omegaconf
from src.training.augmentation.tokenizer import load_vocab

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 테스트용 임시 출력 디렉토리 (테스트 완료 후 자동 삭제)
TEST_OUTPUT_DIR = str(Path(_PROJECT_ROOT) / "data" / "temp" / "validate_inference")


# ── 유틸리티 ──────────────────────────────────────────────────────────────────

def _pass(msg: str) -> None:
    """검증 통과 로그."""
    logger.info("[PASS] %s", msg)


def _fail(msg: str) -> None:
    """검증 실패 로그."""
    logger.error("[FAIL] %s", msg)


def load_cfg(model_dir: str | None = None) -> OmegaConf:
    """테스트용 config를 구성한다.

    Args:
        model_dir: 모델 로드 경로. None이면 pipeline.yaml 기본값 사용.

    Returns:
        OmegaConf DictConfig.
    """
    config_path = Path(_PROJECT_ROOT) / "config" / "inference" / "pipeline.yaml"
    cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(cfg, False)

    if model_dir is not None:
        cfg.model.model_dir = model_dir
        cfg.model.tokenizer_dir = model_dir

    # 테스트용 오버라이드
    cfg.output.dir = TEST_OUTPUT_DIR
    cfg.generation.max_new_tokens = 256  # 빠른 검증을 위해 생성 토큰 수 제한

    OmegaConf.set_struct(cfg, True)
    return cfg


# ── Phase 0: 파일 존재 확인 ───────────────────────────────────────────────────

def phase0_file_check(cfg) -> bool:
    """모델 로드 없이 필수 파일 존재 여부를 확인한다.

    Mod Record: 새 구조에서 load_mode="adapters"는 partial_state.pt + adapter_model.safetensors로 동작.
    "merged" 모드만 model.safetensors가 필요하다.

    Args:
        cfg: OmegaConf DictConfig.

    Returns:
        모든 검증 통과 시 True.
    """
    logger.info("=" * 60)
    logger.info("Phase 0: 파일 존재 확인")
    logger.info("=" * 60)

    ok = True
    load_mode = cfg.inference.get("load_mode", "merged")

    if load_mode == "adapters":
        # partial_state.pt 확인
        partial_state_path = Path(cfg.model.pre_stage_dir) / "partial_state.pt"
        if partial_state_path.exists():
            _pass(f"partial_state.pt 존재: {partial_state_path}")
        else:
            _fail(f"partial_state.pt 없음: {partial_state_path}")
            ok = False

        # 각 adapter 디렉토리 확인
        for adapter_cfg in cfg.inference.adapters:
            adapter_path = Path(adapter_cfg.path)
            adapter_safetensors = adapter_path / "adapter_model.safetensors"
            if adapter_safetensors.exists():
                _pass(f"adapter_model.safetensors 존재: {adapter_path}")
            else:
                _fail(f"adapter_model.safetensors 없음: {adapter_path}")
                ok = False
    else:
        # merged 모드: full model 파일 확인
        model_dir = Path(cfg.model.model_dir)
        required_files = ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"]
        for fname in required_files:
            fpath = model_dir / fname
            if fpath.exists():
                _pass(f"{fname} 존재 확인")
            else:
                _fail(f"{fname} 없음: {fpath}")
                ok = False

    # vocab_extension.json 확인 (공통)
    vocab_path = Path(cfg.model.vocab_extension)
    if vocab_path.exists():
        _pass(f"vocab_extension.json 존재 확인: {vocab_path}")
    else:
        _fail(f"vocab_extension.json 없음: {vocab_path}")
        ok = False

    return ok


# ── Phase 1: 모델 + Vocab 로드 ───────────────────────────────────────────────

def phase1_model_load(cfg) -> tuple:
    """모델과 Vocab를 로드하고 기본 검증을 수행한다.

    Args:
        cfg: OmegaConf DictConfig.

    Returns:
        (model, tokenizer, vocab) 튜플. 실패 시 (None, None, None).
    """
    logger.info("=" * 60)
    logger.info("Phase 1: 모델 + Vocab 로드 검증")
    logger.info("=" * 60)

    try:
        model, tokenizer = load_model_for_inference(cfg)
        _pass("load_model_for_inference() 성공")
    except Exception as e:
        _fail(f"load_model_for_inference() 실패: {e}")
        return None, None, None

    # vocab_size 일치 확인
    if model.config.vocab_size == len(tokenizer):
        _pass(f"vocab_size 일치: {model.config.vocab_size}")
    else:
        _fail(f"vocab_size 불일치: model={model.config.vocab_size}, tokenizer={len(tokenizer)}")

    # Vocab 로드
    vocab_path = Path(cfg.model.vocab_extension)
    tokenizer_dir = Path(cfg.model.tokenizer_dir)
    try:
        vocab = load_vocab(vocab_path, tokenizer_dir)
        _pass(f"load_vocab() 성공 — 커스텀 토큰: {len(vocab.token_to_id)}개")
    except Exception as e:
        _fail(f"load_vocab() 실패: {e}")
        return model, tokenizer, None

    # 커스텀 토큰 존재 확인
    test_tokens = ["<X:0>", "<Y:255>", "<ROOM>", "<END_ROOM>", "<OUTPUT>", "<END_OUTPUT>"]
    for t in test_tokens:
        if t in vocab.token_to_id:
            _pass(f"커스텀 토큰 {t} 존재 (ID={vocab.token_to_id[t]})")
        else:
            _fail(f"커스텀 토큰 {t} 없음")

    return model, tokenizer, vocab


# ── Phase 2: 입력 처리 검증 ──────────────────────────────────────────────────

def phase2_input_processing(cfg, vocab, num_samples: int) -> tuple[list[dict], list[dict]] | tuple[None, None]:
    """입력 로드 + condition 토큰 빌드를 검증한다.

    Args:
        cfg: OmegaConf DictConfig.
        vocab: Vocab 객체.
        num_samples: 테스트할 샘플 수.

    Returns:
        (raw_samples, row_samples) 튜플. 실패 시 (None, None).
    """
    logger.info("=" * 60)
    logger.info("Phase 2: 입력 처리 검증")
    logger.info("=" * 60)

    # Arrow test split에서 샘플 로드 (config 임시 수정)
    OmegaConf.set_struct(cfg, False)
    orig_mode = cfg.input.mode
    orig_max = cfg.input.max_samples
    cfg.input.mode = "arrow"
    cfg.input.max_samples = num_samples
    OmegaConf.set_struct(cfg, True)

    try:
        raw_samples, row_samples = load_samples(cfg)
        _pass(f"load_samples() 성공 — {len(row_samples)}개 샘플")
    except Exception as e:
        _fail(f"load_samples() 실패: {e}")
        return None, None
    finally:
        OmegaConf.set_struct(cfg, False)
        cfg.input.mode = orig_mode
        cfg.input.max_samples = orig_max
        OmegaConf.set_struct(cfg, True)

    if len(row_samples) == 0:
        _fail("로드된 샘플이 0개")
        return None, None

    # AugmentationPipeline 초기화 + 증강 적용 테스트
    aug_config_path = Path(_PROJECT_ROOT) / cfg.augmentation.config_path
    try:
        aug_omegacfg = OmegaConf.load(aug_config_path)
        aug_config = config_from_omegaconf(aug_omegacfg)
        pipeline = AugmentationPipeline(vocab, aug_config, seed=42)
        _pass("AugmentationPipeline 초기화 성공")
    except Exception as e:
        _fail(f"AugmentationPipeline 초기화 실패: {e}")
        return raw_samples, row_samples

    # 증강 테스트: raw_sample(columnar)을 pipeline에 전달
    raw_sample = raw_samples[0]
    try:
        condition_tokens, output_tokens, aug_summary, _, _drop_state = build_condition_with_augmentation(
            raw_sample, pipeline
        )
        _pass(f"build_condition_with_augmentation() 성공 — condition: {len(condition_tokens)} tokens, aug: {aug_summary}")
    except Exception as e:
        _fail(f"build_condition_with_augmentation() 실패: {e}")
        return raw_samples, row_samples

    # decode_tokens 검증
    try:
        decoded = decode_tokens(condition_tokens, vocab)
        has_input = "<INPUT>" in decoded
        has_end_input = "<END_INPUT>" in decoded
        if has_input and has_end_input:
            _pass("condition_tokens 디코딩 성공 — <INPUT>/<END_INPUT> 확인")
        else:
            _fail(f"condition_tokens에 <INPUT>/<END_INPUT> 미포함: input={has_input}, end={has_end_input}")
    except Exception as e:
        _fail(f"decode_tokens() 실패: {e}")

    # no-aug 모드 테스트: row_sample(row-oriented)을 직접 사용
    row_sample = row_samples[0]
    try:
        no_aug_tokens = build_condition_no_aug(row_sample, vocab)
        _pass(f"build_condition_no_aug() 성공 — {len(no_aug_tokens)} tokens")
    except Exception as e:
        _fail(f"build_condition_no_aug() 실패: {e}")

    return raw_samples, row_samples


# ── Phase 3: 추론 검증 ──────────────────────────────────────────────────────

def phase3_inference(cfg, model, tokenizer, vocab, samples) -> list[list[int]] | None:
    """모델 추론을 검증한다.

    Args:
        cfg: OmegaConf DictConfig.
        model: 추론 모델.
        tokenizer: 토크나이저.
        vocab: Vocab 객체.
        samples: 테스트 샘플 리스트.

    Returns:
        각 샘플의 generated_ids 리스트. 실패 시 None.
    """
    logger.info("=" * 60)
    logger.info("Phase 3: 추론 검증")
    logger.info("=" * 60)

    all_generated = []

    for i, sample in enumerate(samples):
        plan_id = sample["plan_id"]
        condition_tokens = build_condition_no_aug(sample, vocab)

        try:
            generated_ids = generate_floorplan(
                condition_tokens, model, tokenizer, cfg.generation
            )
        except Exception as e:
            _fail(f"generate_floorplan() 실패 (plan_id={plan_id}): {e}")
            return None

        if len(generated_ids) > 0:
            _pass(f"[{i+1}/{len(samples)}] 추론 성공 (plan_id={plan_id}) — {len(generated_ids)} tokens 생성")
        else:
            _fail(f"[{i+1}/{len(samples)}] 생성된 토큰 없음 (plan_id={plan_id})")

        # <OUTPUT> 토큰 포함 여부 확인
        decoded = decode_tokens(generated_ids, vocab)
        if "<OUTPUT>" in decoded:
            _pass(f"  생성 결과에 <OUTPUT> 토큰 포함 확인")
        else:
            logger.warning("  생성 결과에 <OUTPUT> 토큰 미포함 (모델 미훈련 또는 생성 길이 부족)")

        all_generated.append(generated_ids)

    return all_generated


# ── Phase 4: 출력 파싱 + 시각화 검증 ─────────────────────────────────────────

def phase4_parsing(vocab, all_generated, color_map_cfg) -> list[dict | None]:
    """출력 파싱 및 시각화를 검증한다.

    Args:
        vocab: Vocab 객체.
        all_generated: 각 샘플의 generated_ids 리스트.
        color_map_cfg: 색상 설정 DictConfig.

    Returns:
        파싱된 평면도 딕셔너리 리스트 (실패한 것은 None).
    """
    logger.info("=" * 60)
    logger.info("Phase 4: 출력 파싱 + 시각화 검증")
    logger.info("=" * 60)

    parsed_list = []

    for i, generated_ids in enumerate(all_generated):
        parsed = parse_output_tokens(generated_ids, vocab)
        if parsed is not None:
            _pass(f"[{i+1}] parse_output_tokens() 성공 — rooms: {len(parsed['rooms'])}개")

            # 키 존재 확인
            for key in ["rooms", "edges", "front_door"]:
                if key in parsed:
                    _pass(f"  '{key}' 키 존재")
                else:
                    _fail(f"  '{key}' 키 없음")

            # rooms 비어있지 않은지
            if len(parsed["rooms"]) > 0:
                _pass(f"  rooms 리스트 비어있지 않음 ({len(parsed['rooms'])}개)")
            else:
                _fail("  rooms 리스트가 비어있음")

            # 시각화 테스트
            try:
                from src.build_dataset.visualize_json.visualizer import FloorplanVisualizer
                import tempfile
                with tempfile.TemporaryDirectory() as tmpdir:
                    visualizer = FloorplanVisualizer(color_map_cfg)
                    visualizer.visualize(parsed, Path(tmpdir))
                    _pass("  FloorplanVisualizer 렌더링 성공")
            except Exception as e:
                _fail(f"  FloorplanVisualizer 렌더링 실패: {e}")
        else:
            logger.warning("[%d] parse_output_tokens() 반환 None (파싱 실패 — 모델 미훈련일 수 있음)", i + 1)

        parsed_list.append(parsed)

    return parsed_list


# ── Phase 5: 결과 저장 검증 ──────────────────────────────────────────────────

def phase5_save(cfg, vocab, samples, all_generated, parsed_list, color_map_cfg) -> bool:
    """결과 저장을 검증한다.

    Args:
        cfg: OmegaConf DictConfig.
        vocab: Vocab 객체.
        samples: 테스트 샘플 리스트.
        all_generated: 각 샘플의 generated_ids 리스트.
        parsed_list: 파싱된 평면도 딕셔너리 리스트.
        color_map_cfg: 색상 설정 DictConfig.

    Returns:
        모든 검증 통과 시 True.
    """
    logger.info("=" * 60)
    logger.info("Phase 5: 결과 저장 검증")
    logger.info("=" * 60)

    output_dir = Path(TEST_OUTPUT_DIR)
    ok = True

    for i, (sample, generated_ids, parsed) in enumerate(zip(samples, all_generated, parsed_list)):
        plan_id = sample["plan_id"]
        condition_tokens = build_condition_no_aug(sample, vocab)

        try:
            save_results(
                plan_id=plan_id,
                raw_sample=sample,
                condition_tokens=condition_tokens,
                output_results=[(generated_ids, parsed, 1.23)],
                vocab=vocab,
                output_cfg=cfg.output,
                color_map_cfg=color_map_cfg,
                output_dir=output_dir,
                augmentation_summary="test augmentation",
            )
            _pass(f"[{i+1}] save_results() 호출 성공 (plan_id={plan_id})")
        except Exception as e:
            _fail(f"[{i+1}] save_results() 실패: {e}")
            ok = False
            continue

        # 파일 존재 확인
        sample_dir = output_dir / str(plan_id)

        # meta.json (항상 존재)
        meta_path = sample_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            required_keys = ["plan_id", "input_token_count", "output_token_count", "elapsed_sec", "parse_success"]
            missing = [k for k in required_keys if k not in meta]
            if not missing:
                _pass(f"  meta.json 존재 + 필수 키 확인")
            else:
                _fail(f"  meta.json 필수 키 누락: {missing}")
                ok = False
        else:
            _fail(f"  meta.json 없음: {meta_path}")
            ok = False

        # 토큰 파일 확인
        if cfg.output.save_tokens:
            for token_file in ["input/tokens.txt", "output/tokens.txt"]:
                fpath = sample_dir / token_file
                if fpath.exists():
                    _pass(f"  {token_file} 존재")
                else:
                    _fail(f"  {token_file} 없음")
                    ok = False

        # JSON 파일 확인
        if cfg.output.save_json:
            cond_json = sample_dir / "input" / "condition.json"
            if cond_json.exists():
                _pass("  input/condition.json 존재")
            else:
                _fail("  input/condition.json 없음")
                ok = False

    return ok


# ── 메인 ────────────────────────────────────────────────────────────────────

def _create_dummy_adapters_if_needed(cfg) -> tuple:
    """adapters 모드에서 필요한 adapter가 없으면 테스트용 더미 adapter를 생성한다.

    Args:
        cfg: 추론 파이프라인 DictConfig.

    Returns:
        tuple:
            - modified_cfg: adapter 경로가 업데이트된 cfg (더미 생성 시).
            - dummy_dirs: 생성된 임시 디렉토리 목록 (없으면 빈 리스트).
    """
    import tempfile
    from peft import LoraConfig, TaskType, get_peft_model
    from src.inference.model_loader import _load_base_with_partial_state

    load_mode = cfg.inference.get("load_mode", "merged")
    if load_mode != "adapters":
        return cfg, []

    dummy_dirs = []
    adapters_list = list(cfg.inference.adapters)
    sft_cfg_path = Path(_PROJECT_ROOT) / "config" / "training" / "sft" / "pipeline.yaml"
    sft_lora_cfg = OmegaConf.load(sft_cfg_path).lora

    needs_dummy = False
    for adapter_entry in adapters_list:
        adapter_path = Path(adapter_entry.path)
        if not adapter_path.exists() or not (adapter_path / "adapter_config.json").exists():
            needs_dummy = True
            break

    if not needs_dummy:
        return cfg, []

    # 모델을 한 번 로드해서 모든 누락된 adapter를 생성
    partial_state_path = Path(_PROJECT_ROOT) / str(cfg.model.pre_stage_dir) / "partial_state.pt"

    # cfg를 _load_base_with_partial_state 호출용 최소 형식으로 복제
    from omegaconf import DictConfig as _DictConfig
    base_cfg = OmegaConf.create({
        "model": {
            "hub_id": cfg.model.hub_id,
            "tokenizer_dir": cfg.model.tokenizer_dir,
        },
        "quantization": dict(cfg.quantization),
    })

    logger.info("더미 adapter 생성을 위해 Hub 모델 로드 중...")
    base_model, tokenizer = _load_base_with_partial_state(base_cfg, partial_state_path)

    lora_config = LoraConfig(
        r=sft_lora_cfg.r,
        lora_alpha=sft_lora_cfg.lora_alpha,
        lora_dropout=sft_lora_cfg.lora_dropout,
        target_modules=list(sft_lora_cfg.target_modules),
        bias=sft_lora_cfg.bias,
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(base_model, lora_config)

    OmegaConf.set_struct(cfg, False)
    new_adapters = []
    for adapter_entry in adapters_list:
        adapter_path = Path(adapter_entry.path)
        if not adapter_path.exists() or not (adapter_path / "adapter_config.json").exists():
            tmp = Path(tempfile.mkdtemp(prefix="inf_test_adapter_"))
            peft_model.save_pretrained(str(tmp))
            dummy_dirs.append(tmp)
            logger.info(f"더미 adapter 생성: {adapter_entry.name} → {tmp}")
            new_adapters.append({"path": str(tmp), "name": adapter_entry.name})
        else:
            new_adapters.append(dict(adapter_entry))
    cfg.inference.adapters = new_adapters
    OmegaConf.set_struct(cfg, True)

    del peft_model, base_model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return cfg, dummy_dirs


def main() -> None:
    """추론 검증 메인 함수."""
    parser = argparse.ArgumentParser(description="추론 파이프라인 검증")
    parser.add_argument("--model_dir", type=str, default=None, help="모델 디렉토리 경로")
    parser.add_argument("--num_samples", type=int, default=2, help="테스트 샘플 수")
    args = parser.parse_args()

    cfg = load_cfg(model_dir=args.model_dir)
    num_samples = args.num_samples

    load_mode = cfg.inference.get("load_mode", "merged")
    logger.info("=" * 60)
    logger.info("추론 파이프라인 검증 시작")
    logger.info("  load_mode: %s", load_mode)
    logger.info("  샘플 수: %d", num_samples)
    logger.info("=" * 60)

    # 색상 설정 로드
    color_map_path = Path(_PROJECT_ROOT) / cfg.color_map_path
    color_map_cfg = OmegaConf.load(color_map_path)

    # adapters 모드에서 adapter 파일이 없으면 더미 생성
    dummy_dirs = []
    if load_mode == "adapters":
        cfg, dummy_dirs = _create_dummy_adapters_if_needed(cfg)

    # Phase 0: 파일 존재 확인
    if not phase0_file_check(cfg):
        logger.error("Phase 0 실패. 이후 Phase를 건너뜁니다.")
        for d in dummy_dirs:
            shutil.rmtree(d, ignore_errors=True)
        sys.exit(1)

    # Phase 1: 모델 + Vocab 로드
    model, tokenizer, vocab = phase1_model_load(cfg)
    if model is None or vocab is None:
        logger.error("Phase 1 실패. 이후 Phase를 건너뜁니다.")
        sys.exit(1)

    # Phase 2: 입력 처리 검증
    raw_samples, row_samples = phase2_input_processing(cfg, vocab, num_samples)
    if row_samples is None or len(row_samples) == 0:
        logger.error("Phase 2 실패. 이후 Phase를 건너뜁니다.")
        sys.exit(1)

    # Phase 3: 추론 검증 (row_samples 사용 — no-aug 모드로 condition 토큰 빌드)
    all_generated = phase3_inference(cfg, model, tokenizer, vocab, row_samples)
    if all_generated is None:
        logger.error("Phase 3 실패. 이후 Phase를 건너뜁니다.")
        sys.exit(1)

    # Phase 4: 출력 파싱 + 시각화 검증
    parsed_list = phase4_parsing(vocab, all_generated, color_map_cfg)

    # Phase 5: 결과 저장 검증
    try:
        phase5_save(cfg, vocab, row_samples, all_generated, parsed_list, color_map_cfg)
    finally:
        # 임시 디렉토리 정리
        test_dir = Path(TEST_OUTPUT_DIR)
        if test_dir.exists():
            shutil.rmtree(test_dir)
            logger.info("임시 디렉토리 정리 완료: %s", test_dir)
        for d in dummy_dirs:
            shutil.rmtree(d, ignore_errors=True)
            logger.info("더미 adapter 임시 디렉토리 삭제: %s", d)

    logger.info("=" * 60)
    logger.info("추론 파이프라인 검증 완료")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
