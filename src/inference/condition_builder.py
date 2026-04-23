"""입력 조건 로더 및 토큰 빌더 모듈.

다양한 입력 소스(JSONL 파일, JSONL 디렉토리, Arrow, 텍스트 파일)에서 데이터를 읽고,
AugmentationPipeline을 통해 훈련과 동일한 방식으로 condition 토큰 시퀀스를 생성한다.

반환 데이터 형식:
    load_samples()는 (raw_samples, row_samples) 튜플을 반환한다.
    - raw_samples: AugmentationPipeline에 전달할 원본 데이터 (Arrow columnar 또는 JSONL row-oriented)
    - row_samples: row-oriented 딕셔너리 (결과 저장, 시각화, no-aug 토큰 빌드 등에 사용)

    txt_dir 모드:
    - raw_samples: {"plan_id": str, "token_text": str} 리스트 (토큰 텍스트 포함)
    - row_samples: parse_input_tokens()로 역변환한 구조화 dict 리스트 (시각화/JSON 저장용)

    AugmentationPipeline.__call__()은 내부에서 to_row_oriented()를 호출하므로
    Arrow columnar 데이터를 미리 변환하면 이중 변환 에러가 발생한다.
    반면 JSONL 데이터는 이미 row-oriented이므로 to_row_oriented()를 적용해도
    pipeline 내부에서 동일 결과를 반환한다.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import datasets
from omegaconf import DictConfig

from transformers import AutoTokenizer

from src.build_dataset.visualize_json.loader import FloorplanLoader
from src.inference.output_parser import parse_input_tokens
from src.training.augmentation.pipeline import AugmentationPipeline
from src.training.augmentation.strategies import DropState
from src.training.augmentation.tokenizer import (
    Vocab,
    build_condition_tokens,
    to_row_oriented,
)

logger = logging.getLogger(__name__)


def _jsonl_to_columnar(sample: dict) -> dict:
    """JSONL row-oriented 샘플을 Arrow columnar 포맷으로 변환한다.

    AugmentationPipeline 내부의 to_row_oriented()는 Arrow columnar 포맷만 처리한다.
    JSONL에서 로드한 row-oriented 데이터를 파이프라인에 전달하기 전에 변환한다.

    포맷 차이:
        JSONL rooms: [{"rid", "type", "coords"}, ...]
        JSONL edges: [{"pair", "doors": [{x,y,w,h}]}, ...] — "doors" 복수형 키
        JSONL front_door: {"x", "y", "w", "h"} 단일 dict
        JSONL spatial: [[rid_a, rid_b, direction], ...]

        Arrow rooms: {"rid": [...], "type": [...], "coords": [...]}
        Arrow edges: {"pair": [...], "door": [{"x":[...], ...}]} — "door" 단수형 키, 값이 list
        Arrow front_door: {"x": [val], "y": [val], "w": [val], "h": [val]}
        Arrow spatial: {"rid_a": [...], "rid_b": [...], "direction": [...]}

    Args:
        sample: JSONL FloorplanLoader가 반환한 row-oriented 딕셔너리.

    Returns:
        to_row_oriented()가 처리 가능한 Arrow columnar 포맷 딕셔너리.
    """
    rooms = sample["rooms"]
    edges = sample["edges"]
    spatial = sample.get("spatial") or []
    front_door = sample.get("front_door")

    rooms_col = {
        "rid": [r["rid"] for r in rooms],
        "type": [r["type"] for r in rooms],
        "coords": [r["coords"] for r in rooms],
    }

    # edges: "doors"(JSONL) → "door"(Arrow), 각 도어 목록을 dict-of-lists로 변환
    edge_pairs = [e["pair"] for e in edges]
    edge_doors = []
    for e in edges:
        doors = e.get("doors") or e.get("door") or []
        if doors:
            edge_doors.append({
                "x": [d["x"] for d in doors],
                "y": [d["y"] for d in doors],
                "w": [d["w"] for d in doors],
                "h": [d["h"] for d in doors],
            })
        else:
            edge_doors.append({"x": [], "y": [], "w": [], "h": []})
    edges_col = {"pair": edge_pairs, "door": edge_doors}

    # front_door: 단일 dict → 단일 원소 list-of-values dict (또는 빈 dict)
    if front_door is not None:
        front_door_col = {
            "x": [front_door["x"]],
            "y": [front_door["y"]],
            "w": [front_door["w"]],
            "h": [front_door["h"]],
        }
    else:
        front_door_col = {"x": [], "y": [], "w": [], "h": []}

    # spatial: list-of-lists → columnar dict (비어있으면 빈 list 유지)
    if spatial:
        spatial_col = {
            "rid_a": [s[0] if isinstance(s, (list, tuple)) else s["rid_a"] for s in spatial],
            "rid_b": [s[1] if isinstance(s, (list, tuple)) else s["rid_b"] for s in spatial],
            "direction": [s[2] if isinstance(s, (list, tuple)) else s["direction"] for s in spatial],
        }
    else:
        spatial_col = []

    return {
        "plan_id": sample["plan_id"],
        "rooms": rooms_col,
        "edges": edges_col,
        "front_door": front_door_col,
        "spatial": spatial_col,
    }


def load_samples(
    cfg: DictConfig,
    tokenizer: AutoTokenizer | None = None,
    vocab: Vocab | None = None,
) -> tuple[list[dict], list[dict]]:
    """config의 input 설정에 따라 평면도 샘플을 로드한다.

    Args:
        cfg: Hydra DictConfig. cfg.input 섹션을 참조한다.
        tokenizer: txt_dir 모드 전용. 텍스트 → 토큰 ID 변환에 사용. 다른 모드에서는 무시.
        vocab: txt_dir 모드 전용. INPUT 토큰 파싱(parse_input_tokens)에 사용. 다른 모드에서는 무시.

    Returns:
        (raw_samples, row_samples) 튜플:
            - raw_samples: AugmentationPipeline에 전달할 원본 데이터.
                Arrow 모드: columnar 포맷 (pipeline 내부에서 to_row_oriented 적용)
                JSONL 모드: row-oriented 포맷 (pipeline 내부 to_row_oriented가 안전하게 통과)
                txt_dir 모드: {"plan_id": str, "token_text": str} 딕셔너리
            - row_samples: row-oriented 딕셔너리 리스트.
                결과 저장, 시각화, no-aug 토큰 빌드 등에 사용.
                txt_dir 모드: parse_input_tokens()로 역변환한 구조화 dict

    Raises:
        ValueError: 지원하지 않는 input.mode일 때.
        FileNotFoundError: 입력 경로가 존재하지 않을 때.
    """
    mode = cfg.input.mode
    plan_ids = cfg.input.get("plan_ids")
    max_samples = cfg.input.get("max_samples")

    if mode == "jsonl_file":
        jsonl_file = cfg.input.jsonl_file
        if jsonl_file is None:
            raise ValueError("input.mode='jsonl_file'이지만 input.jsonl_file이 설정되지 않았습니다.")
        loader = FloorplanLoader([Path(jsonl_file)])
        row_samples = loader.load_all()
        # Mod Record: JSONL은 row-oriented이지만 pipeline 내부 to_row_oriented()가
        # Arrow columnar 포맷만 처리하므로 파이프라인 전달 전 columnar 변환 필요
        raw_samples = [_jsonl_to_columnar(s) for s in row_samples]

    elif mode == "jsonl_dir":
        jsonl_dir = Path(cfg.input.jsonl_dir)
        pattern = cfg.input.get("jsonl_pattern", "*.jsonl")
        loader = FloorplanLoader.from_directory(jsonl_dir, pattern)
        row_samples = loader.load_all()
        raw_samples = [_jsonl_to_columnar(s) for s in row_samples]

    elif mode == "arrow":
        arrow_dir = cfg.input.arrow_dir
        split = cfg.input.get("arrow_split", "test")
        ds = datasets.load_from_disk(arrow_dir)
        if isinstance(ds, datasets.DatasetDict):
            if split not in ds:
                raise KeyError(
                    f"split '{split}'이 DatasetDict에 없습니다. "
                    f"사용 가능: {list(ds.keys())}"
                )
            ds = ds[split]
        # Arrow: raw(columnar)와 row-oriented를 각각 보관
        raw_samples = [ds[i] for i in range(len(ds))]
        row_samples = [to_row_oriented(s) for s in raw_samples]

    elif mode == "txt_dir":
        if tokenizer is None or vocab is None:
            raise ValueError("txt_dir 모드에서는 tokenizer와 vocab이 필요합니다.")
        txt_dir_path = Path(cfg.input.get("txt_dir", "data/inference/input_txt"))
        pattern = cfg.input.get("txt_pattern", "*.txt")
        txt_files = sorted(txt_dir_path.glob(pattern))
        if not txt_files:
            raise FileNotFoundError(f"txt_dir 내 텍스트 파일 없음: {txt_dir_path}/{pattern}")

        raw_samples = []
        row_samples = []
        for f in txt_files:
            token_text = f.read_text(encoding="utf-8").strip()
            plan_id = f.stem
            raw_samples.append({"plan_id": plan_id, "token_text": token_text})
            # parse_input_tokens()로 구조화 dict 생성 (condition.json, floorplan.png 저장용)
            token_ids = tokenizer.encode(token_text, add_special_tokens=False)
            parsed = parse_input_tokens(token_ids, vocab, plan_id=plan_id)
            if parsed is None:
                logger.warning("INPUT 파싱 실패: %s — 빈 구조로 대체", f.name)
                parsed = {"plan_id": plan_id, "rooms": [], "edges": [], "front_door": None, "spatial": []}
            row_samples.append(parsed)

    else:
        raise ValueError(f"지원하지 않는 input.mode: {mode}")

    logger.info("입력 샘플 로드 완료: %d개 (mode=%s)", len(row_samples), mode)

    # plan_ids 필터링
    if plan_ids is not None:
        plan_id_set = set(str(pid) for pid in plan_ids)
        filtered = [
            (raw, row) for raw, row in zip(raw_samples, row_samples)
            if str(row["plan_id"]) in plan_id_set
        ]
        if filtered:
            raw_samples, row_samples = zip(*filtered)
            raw_samples, row_samples = list(raw_samples), list(row_samples)
        else:
            raw_samples, row_samples = [], []
        logger.info("plan_ids 필터 적용 후: %d개", len(row_samples))

    # max_samples 제한
    if max_samples is not None and len(row_samples) > max_samples:
        raw_samples = raw_samples[:max_samples]
        row_samples = row_samples[:max_samples]
        logger.info("max_samples 제한 적용: %d개", max_samples)

    return raw_samples, row_samples


def build_condition_with_augmentation(
    raw_sample: dict,
    pipeline: AugmentationPipeline,
) -> tuple[list[int], list[int], str, dict, DropState]:
    """AugmentationPipeline으로 증강 적용 후 condition/output 토큰을 반환한다.

    훈련 시와 동일한 방식: 변형 증강(Shuffle, Flip 등) + 삭제 증강(Drop*)이 모두 적용된다.

    Args:
        raw_sample: 원본 평면도 딕셔너리 (Arrow columnar 또는 JSONL row-oriented).
            pipeline 내부에서 to_row_oriented()가 적용된다.
        pipeline: 초기화된 AugmentationPipeline 인스턴스.

    Returns:
        tuple:
            - condition_tokens: 증강이 적용된 입력 토큰 시퀀스
            - output_tokens: 증강 적용된 full 정답 토큰 시퀀스 (검증 비교용)
            - augmentation_summary: 적용된 증강 요약 문자열
            - augmented_sample: 변형 증강이 적용된 row-oriented 샘플 (입력 시각화용)
            - drop_state: 삭제 증강 상태 (시각화 시 drop된 요소 필터링용)
    """
    condition_tokens, output_tokens = pipeline(raw_sample)
    aug_summary = pipeline.augmented_summary()
    # Mod Record: AugmentationPipeline이 last_augmented_sample을 저장하지 않으므로
    # pipeline 호출 후 raw_sample(columnar)을 row-oriented로 변환해 시각화용 샘플로 사용.
    # 셔플/플립 등 변형 증강은 condition_tokens에 반영되어 있으나 시각화 샘플에는 미반영됨.
    augmented_sample = getattr(pipeline, "last_augmented_sample", None) or to_row_oriented(raw_sample)
    drop_state = pipeline.last_drop_state
    return condition_tokens, output_tokens, aug_summary, augmented_sample, drop_state


def build_condition_no_aug(
    row_sample: dict,
    vocab: Vocab,
) -> list[int]:
    """증강 없이 full condition 토큰 시퀀스를 생성한다.

    augmentation.enabled=false 시 사용.
    DropState의 모든 필드가 비어있는 상태로 build_condition_tokens()를 직접 호출한다.

    Args:
        row_sample: row-oriented 평면도 딕셔너리.
        vocab: Vocab 객체.

    Returns:
        condition 토큰 ID 리스트.
    """
    no_aug_state = DropState()
    return build_condition_tokens(row_sample, no_aug_state, vocab)
