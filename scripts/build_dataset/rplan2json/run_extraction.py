"""평면도 정보 추출 파이프라인 실행 스크립트.

RPLAN PNG 파일들에서 방, 문, Edge, Spatial 정보를 추출하여 JSONL로 저장한다.
Hydra 기반 설정 관리로 하이퍼파라미터를 CLI에서 오버라이드 가능.

Usage:
    # 전체 배치 처리
    python scripts/build_dataset/rplan2json/run_extraction.py

    # 단일 파일 디버깅
    python scripts/build_dataset/rplan2json/run_extraction.py mode=single target_file=0.png

    # 워커 수 오버라이드
    python scripts/build_dataset/rplan2json/run_extraction.py batch.num_workers=16
"""

from __future__ import annotations

import logging
import os
import sys
from glob import glob
from multiprocessing import Pool
from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

# 프로젝트 루트를 sys.path에 추가
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.build_dataset.rplan2json.channel_parser import load_bgra_image, parse_channels
from src.build_dataset.rplan2json.door_extractor import extract_front_door, extract_interior_doors
from src.build_dataset.rplan2json.edge_builder import build_edges
from src.build_dataset.rplan2json.room_extractor import (
    extract_outline,
    extract_room_instances,
    load_type_merge_config,
)
from src.build_dataset.rplan2json.serializer import (
    append_to_jsonl,
    build_plan_record,
    sort_rooms_raster_order,
)
from src.build_dataset.rplan2json.spatial_calculator import build_spatial_relations

log = logging.getLogger(__name__)


def process_single_png(args: tuple[str, DictConfig]) -> dict | None:
    """단일 PNG 파일에 대해 10단계 파이프라인 실행.

    Args:
        args: (png_path, cfg) 튜플. multiprocessing.Pool 호환을 위해 튜플로 전달.

    Returns:
        JSONL 레코드 dict. 실패 시 None.
    """
    png_path, cfg = args
    plan_id = Path(png_path).stem

    try:
        # Step 0: PNG 로드 + 채널 분리
        image = load_bgra_image(png_path)
        channels = parse_channels(image)

        # Step 1~3: 방 인스턴스 분리 + 꼭지점 추출
        merge_config = load_type_merge_config(cfg.data.room_type_merge_config)
        room_instances = extract_room_instances(
            space_type=channels.space_type,
            merge_config=merge_config,
            room_type_ids=cfg.space_type.room_types,
            min_room_area=cfg.processing.min_room_area,
            connectivity=cfg.processing.room_connectivity,
        )

        # Step 4: 외곽선(outline) 추출 - external_area(G==13) 차집합 사용
        exterior_wall_coords = extract_outline(
            space_type=channels.space_type,
        )

        # Step 5: 현관문 추출
        front_door = extract_front_door(
            space_type=channels.space_type,
            connectivity=cfg.processing.door_connectivity,
        )

        # Step 6: 인테리어 문 추출
        interior_doors = extract_interior_doors(
            space_type=channels.space_type,
            connectivity=cfg.processing.door_connectivity,
            min_door_pixels=cfg.processing.min_door_pixels,
        )

        # raster scan 순서로 정렬 + rid 재배정
        room_instances = sort_rooms_raster_order(room_instances)

        # Step 8: Edge 구성
        edges = build_edges(
            room_instances=room_instances,
            door_instances=interior_doors,
            door_dilation_kernel=cfg.processing.door_dilation_kernel,
        )

        # Step 9: Spatial 관계 계산
        spatial = build_spatial_relations(room_instances)

        # Step 10: JSONL 레코드 조립
        record = build_plan_record(
            plan_id=plan_id,
            exterior_wall_coords=exterior_wall_coords,
            room_instances=room_instances,
            edges=edges,
            front_door=front_door,
            spatial_relations=spatial,
        )

        return record

    except Exception as e:
        log.warning("처리 실패 [%s]: %s", plan_id, e)
        return None


@hydra.main(
    version_base=None,
    config_path=os.path.join(_PROJECT_ROOT, "config", "build_dataset", "rplan2json"),
    config_name="pipeline",
)
def main(cfg: DictConfig) -> None:
    """파이프라인 메인 진입점.

    Args:
        cfg: Hydra DictConfig.
    """
    raw_dir = os.path.join(_PROJECT_ROOT, cfg.data.raw_dataset_dir)
    output_dir = os.path.join(_PROJECT_ROOT, cfg.data.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if cfg.mode == "single":
        # 단일 파일 모드
        if cfg.target_file is None:
            log.error("single 모드에서는 target_file을 지정해야 합니다.")
            return

        png_path = os.path.join(raw_dir, cfg.target_file)
        record = process_single_png((png_path, cfg))
        if record is not None:
            import orjson
            # 사람이 읽기 쉽도록 표준 json으로 출력
            import json
            print(json.dumps(record, indent=2, ensure_ascii=False))
            output_path = os.path.join(output_dir, "single_output.jsonl")
            # 단일 모드는 매 실행마다 덮어씀
            import orjson
            with open(output_path, "wb") as f:
                f.write(orjson.dumps(record))
                f.write(b"\n")
            log.info("저장 완료: %s", output_path)
        else:
            log.error("처리 실패: %s", cfg.target_file)

    else:
        # 배치 모드
        png_paths = sorted(glob(os.path.join(raw_dir, "*.png")))
        total = len(png_paths)
        log.info("총 %d개 PNG 파일 발견. 배치 처리 시작.", total)

        # multiprocessing 인자 준비
        task_args = [(p, cfg) for p in png_paths]

        success_count = 0
        fail_count = 0
        shard_idx = 0
        shard_records: list[dict] = []

        num_workers = cfg.batch.num_workers
        shard_size = cfg.batch.output_shard_size

        with Pool(processes=num_workers) as pool:
            results = pool.imap(process_single_png, task_args, chunksize=100)

            for record in tqdm(results, total=total, desc="추출 진행"):
                if record is None:
                    fail_count += 1
                    continue

                success_count += 1
                shard_records.append(record)

                # 샤드 크기에 도달하면 파일로 저장
                if len(shard_records) >= shard_size:
                    shard_path = os.path.join(
                        output_dir, f"floorplans_{shard_idx:04d}.jsonl"
                    )
                    _write_shard(shard_records, shard_path)
                    log.info("샤드 저장: %s (%d건)", shard_path, len(shard_records))
                    shard_records = []
                    shard_idx += 1

        # 남은 레코드 저장
        if shard_records:
            shard_path = os.path.join(
                output_dir, f"floorplans_{shard_idx:04d}.jsonl"
            )
            _write_shard(shard_records, shard_path)
            log.info("샤드 저장: %s (%d건)", shard_path, len(shard_records))

        log.info(
            "배치 처리 완료. 성공: %d, 실패: %d, 총: %d",
            success_count,
            fail_count,
            total,
        )


def _write_shard(records: list[dict], path: str) -> None:
    """레코드 리스트를 JSONL 샤드 파일로 저장.

    Args:
        records: JSONL 레코드 리스트.
        path: 출력 파일 경로.
    """
    import orjson

    with open(path, "wb") as f:
        for record in records:
            f.write(orjson.dumps(record))
            f.write(b"\n")


if __name__ == "__main__":
    main()
