"""추론 결과 저장 모듈.

입력 조건과 추론 결과를 각각 텍스트(토큰), JSON, 이미지 형태로 저장하고,
메타데이터 파일을 함께 기록한다.

저장 디렉토리 구조 (num_outputs=1):
    {output_dir}/{plan_id}/
    ├── input/
    │   ├── tokens.txt          (save_tokens)
    │   ├── condition.json      (save_json)
    │   └── floorplan.png       (save_image)
    ├── output/
    │   ├── tokens.txt          (save_tokens)
    │   ├── floorplan.json      (save_json)
    │   └── floorplan.png       (save_image)
    └── meta.json               (항상 저장)

저장 디렉토리 구조 (num_outputs>1):
    {output_dir}/{plan_id}/
    ├── input/
    │   └── ...                 (동일)
    ├── output_0/
    │   └── ...
    ├── output_1/
    │   └── ...
    └── meta.json               (항상 저장, outputs 리스트 포함)
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import DictConfig

from src.build_dataset.visualize_json.visualizer import FloorplanVisualizer
from src.training.augmentation.decoder import decode_tokens
from src.training.augmentation.strategies import DropState
from src.training.augmentation.tokenizer import Vocab

logger = logging.getLogger(__name__)


def save_results(
    plan_id: str,
    raw_sample: dict,
    condition_tokens: list[int],
    output_results: list[tuple[list[int], dict | None, float]],
    vocab: Vocab,
    output_cfg: DictConfig,
    color_map_cfg: DictConfig,
    output_dir: Path,
    augmentation_summary: str,
    drop_state: DropState | None = None,
) -> None:
    """추론 결과를 파일로 저장한다.

    입력 조건과 N개의 출력 결과를 저장하며, 메타데이터 파일을 항상 생성한다.

    Args:
        plan_id: 평면도 고유 ID.
        raw_sample: 입력 조건 딕셔너리 (입력 JSON/이미지 저장용).
            txt_dir 모드: parse_input_tokens()로 역변환한 구조화 dict.
            jsonl/arrow 모드: 증강이 적용된 row-oriented dict.
        condition_tokens: 입력 condition 토큰 ID 리스트.
        output_results: (generated_ids, parsed_floorplan, elapsed_sec) 튜플 리스트.
            - generated_ids: 모델이 생성한 출력 토큰 ID 리스트.
            - parsed_floorplan: 역변환된 출력 평면도 딕셔너리 (파싱 실패 시 None).
            - elapsed_sec: 해당 출력의 추론 소요 시간 (초).
        vocab: Vocab 객체.
        output_cfg: 출력 설정 DictConfig (save_tokens, save_json, save_image).
        color_map_cfg: 시각화 색상 설정 DictConfig.
        output_dir: 결과 저장 루트 디렉토리 ({output.dir}/{model.name}/{training_stage}).
        augmentation_summary: 적용된 증강 요약 문자열.
        drop_state: 삭제 증강 상태 (시각화 시 drop된 요소 필터링용, None이면 필터링 없음).
    """
    sample_dir = output_dir / str(plan_id)
    input_dir = sample_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    # output 서브디렉토리 이름 결정: N=1이면 "output", N>1이면 "output_0", "output_1", ...
    n = len(output_results)
    output_subdirs = ["output"] if n == 1 else [f"output_{i}" for i in range(n)]
    for subdir_name in output_subdirs:
        (sample_dir / subdir_name).mkdir(parents=True, exist_ok=True)

    # --- 입력 저장 ---
    if output_cfg.save_tokens:
        _save_text(
            input_dir / "tokens.txt",
            decode_tokens(condition_tokens, vocab),
        )

    if output_cfg.save_json:
        _save_json(input_dir / "condition.json", raw_sample)

    # 이미지 저장용 visualizer: save_image=true이면 입력/출력 모두 사용하므로 한 번만 생성
    visualizer = FloorplanVisualizer(color_map_cfg) if output_cfg.save_image else None

    if visualizer is not None:
        try:
            vis_sample = _prepare_sample_for_visualization(raw_sample, drop_state)
            visualizer.visualize(vis_sample, input_dir)
        except Exception as e:
            logger.warning("입력 이미지 저장 실패 (plan_id=%s): %s", plan_id, e)

    # --- N개 출력 저장 ---
    for subdir_name, (generated_ids, parsed_floorplan, elapsed) in zip(output_subdirs, output_results):
        out_dir = sample_dir / subdir_name

        if output_cfg.save_tokens:
            _save_text(
                out_dir / "tokens.txt",
                decode_tokens(generated_ids, vocab),
            )

        if output_cfg.save_json and parsed_floorplan is not None:
            _save_json(out_dir / "floorplan.json", parsed_floorplan)

        if visualizer is not None and parsed_floorplan is not None:
            try:
                visualizer.visualize(parsed_floorplan, out_dir)
            except Exception as e:
                logger.warning("출력 이미지 저장 실패 (plan_id=%s, dir=%s): %s", plan_id, subdir_name, e)

    # --- 메타데이터 저장 (항상) ---
    if n == 1:
        generated_ids, parsed_floorplan, elapsed = output_results[0]
        meta = {
            "plan_id": str(plan_id),
            "augmentation": augmentation_summary,
            "input_token_count": len(condition_tokens),
            "output_token_count": len(generated_ids),
            "elapsed_sec": round(elapsed, 3),
            "parse_success": parsed_floorplan is not None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    else:
        meta = {
            "plan_id": str(plan_id),
            "augmentation": augmentation_summary,
            "input_token_count": len(condition_tokens),
            "num_outputs": n,
            "outputs": [
                {
                    "index": i,
                    "output_token_count": len(gids),
                    "elapsed_sec": round(elapsed, 3),
                    "parse_success": parsed is not None,
                }
                for i, (gids, parsed, elapsed) in enumerate(output_results)
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    _save_json(sample_dir / "meta.json", meta)

    logger.debug("결과 저장 완료: %s", sample_dir)


def _save_text(path: Path, content: str) -> None:
    """텍스트 파일을 저장한다.

    Args:
        path: 저장 경로.
        content: 저장할 문자열.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _save_json(path: Path, data: dict) -> None:
    """JSON 파일을 indent=2 pretty-print로 저장한다.

    Args:
        path: 저장 경로.
        data: 저장할 딕셔너리.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _prepare_sample_for_visualization(
    sample: dict,
    drop_state: DropState | None,
) -> dict:
    """시각화용으로 샘플을 가공한다.

    두 가지 처리를 수행한다:
    1. DropState 반영: drop된 방/엣지/front_door를 제거하여
       모델이 실제로 받은 입력 조건과 일치하는 샘플을 생성한다.
    2. "door" → "doors" 키 복원: _normalize_row_oriented()가 파이프라인 호환을 위해
       "doors" → "door"로 변환하지만, FloorplanVisualizer는 "doors" 키를 기대한다.

    Args:
        sample: 변형 증강이 적용된 row-oriented 평면도 딕셔너리.
        drop_state: 삭제 증강 상태 (None이면 필터링 없음).

    Returns:
        시각화용으로 가공된 딕셔너리 (원본 미변경).
    """
    vis = copy.deepcopy(sample)

    # DropState 기반 필터링
    if drop_state is not None:
        # 좌표 노이즈 적용: noise_room_coords에 해당하는 rid의 좌표를 노이즈 좌표로 교체
        # Mod Record: 노이즈는 build_condition_tokens() 내부에서 토큰 생성 시에만 적용되고
        # sample dict은 수정되지 않는다. 시각화가 모델이 실제로 받은 입력과 일치하려면
        # drop_state.noise_room_coords를 직접 반영해야 한다.
        if drop_state.noise_room_coords:
            for room in vis.get("rooms", []):
                rid = room.get("rid")
                if rid in drop_state.noise_room_coords:
                    room["coords"] = drop_state.noise_room_coords[rid]

        # 방 필터링: drop_block에 해당하는 rid의 방 제거
        if drop_state.drop_block:
            vis["rooms"] = [
                r for r in vis.get("rooms", [])
                if r.get("rid") not in drop_state.drop_block
            ]

        # 방 타입 제거: drop_type에 해당하는 rid의 타입을 "unknown"으로 표시
        # 좌표는 남아있으므로 floorplan.png에서는 회색으로 렌더링되고
        # unknown_type.png에 별도로 저장된다.
        if drop_state.drop_type:
            for room in vis.get("rooms", []):
                if room.get("rid") in drop_state.drop_type:
                    room["type"] = "unknown"

        # 방 좌표 제거: drop_coords에 해당하는 rid의 좌표를 빈 리스트로 대체
        if drop_state.drop_coords:
            for room in vis.get("rooms", []):
                if room.get("rid") in drop_state.drop_coords:
                    room["coords"] = []

        # 엣지 필터링: drop_edge에 해당하는 인덱스의 엣지 제거
        if drop_state.drop_edge:
            edges = vis.get("edges", [])
            vis["edges"] = [
                e for i, e in enumerate(edges)
                if i not in drop_state.drop_edge
            ]

        # 문 정보 필터링: drop_door에 해당하는 엣지의 문 정보 제거
        if drop_state.drop_door:
            edges = vis.get("edges", [])
            for idx, mode in drop_state.drop_door.items():
                if idx < len(edges) and mode == "all":
                    # 문 정보 전체 삭제 — "door" 또는 "doors" 키 모두 처리
                    edges[idx].pop("door", None)
                    edges[idx].pop("doors", None)

        # front_door 필터링
        # drop_front_door: 블록 전체 삭제 → 시각화에서 제거
        # drop_front_door_coords: 좌표만 삭제 (<FRONT_DOOR> <END_DOOR>) → 좌표 없으므로 시각화에서 제거
        if drop_state.drop_front_door or drop_state.drop_front_door_coords:
            vis["front_door"] = None

        # spatial 필터링: drop_spatial에 해당하는 인덱스 제거
        if drop_state.drop_spatial:
            spatials = vis.get("spatial", [])
            vis["spatial"] = [
                s for i, s in enumerate(spatials)
                if i not in drop_state.drop_spatial
            ]

    # Mod Record: _normalize_row_oriented()가 "doors" → "door"로 변환하여
    # AugmentationPipeline과 호환되게 했지만, FloorplanVisualizer._save_doors_image()는
    # edge.get("doors", [])를 사용한다. 시각화 전에 "door" → "doors"로 복원한다.
    for edge in vis.get("edges", []):
        if "door" in edge and "doors" not in edge:
            edge["doors"] = edge.pop("door")

    return vis
