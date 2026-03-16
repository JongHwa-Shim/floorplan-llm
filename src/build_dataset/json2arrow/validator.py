"""Arrow 변환 결과 검증 모듈.

변환된 Arrow 데이터셋에서 랜덤 샘플을 추출하여 원본 JSONL과 필드별로 비교한다.
데이터 손실이나 변형이 없는지 확인하는 것이 목적이다.

검증 항목:
- plan_id 일치
- rooms 개수, type, coords 값 일치
- edges 개수, pair 일치
- front_door null/값 일치
- spatial 관계 수 일치
"""

from __future__ import annotations

import logging
import random

import datasets
import orjson

log = logging.getLogger(__name__)


def _load_jsonl_by_plan_id(
    jsonl_paths: list[str],
    target_ids: set[str],
) -> dict[str, dict]:
    """JSONL 파일들에서 지정된 plan_id 레코드를 수집.

    Args:
        jsonl_paths: JSONL 파일 경로 리스트.
        target_ids: 검색할 plan_id 집합.

    Returns:
        dict[str, dict]: plan_id → 원본 레코드 매핑.
    """
    found: dict[str, dict] = {}
    for path in jsonl_paths:
        if len(found) == len(target_ids):
            break
        with open(path, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = orjson.loads(line)
                    pid = str(record.get("plan_id", ""))
                    if pid in target_ids:
                        found[pid] = record
                except Exception:  # noqa: BLE001
                    continue
    return found


def validate_conversion(
    arrow_dir: str,
    jsonl_paths: list[str],
    num_samples: int = 10,
    seed: int = 42,
) -> bool:
    """Arrow 데이터셋과 원본 JSONL을 비교하여 변환 정확성을 검증.

    랜덤으로 num_samples개의 Arrow 레코드를 추출하고,
    동일 plan_id를 가진 원본 JSONL 레코드와 필드별로 비교한다.

    Args:
        arrow_dir: 검증할 Arrow 데이터셋 경로.
        jsonl_paths: 원본 JSONL 파일 경로 리스트.
        num_samples: 검증할 랜덤 샘플 수.
        seed: 랜덤 시드 (재현성 보장).

    Returns:
        bool: 모든 샘플 검증 통과 시 True, 하나라도 실패 시 False.
    """
    log.info("=== Arrow 변환 검증 시작 (샘플 %d개) ===", num_samples)

    dataset = datasets.load_from_disk(arrow_dir)
    total = len(dataset)
    log.info("Arrow 레코드 수: %d", total)

    rng = random.Random(seed)
    sample_indices = rng.sample(range(total), min(num_samples, total))

    # Arrow 샘플 추출
    arrow_samples = {
        dataset[i]["plan_id"]: dataset[i] for i in sample_indices
    }
    target_ids = set(arrow_samples.keys())

    # JSONL에서 동일 plan_id 레코드 수집
    jsonl_records = _load_jsonl_by_plan_id(jsonl_paths, target_ids)

    missing = target_ids - set(jsonl_records.keys())
    if missing:
        log.error("JSONL에서 찾을 수 없는 plan_id: %s", missing)

    all_passed = True
    for plan_id, arrow_rec in arrow_samples.items():
        orig = jsonl_records.get(plan_id)
        if orig is None:
            log.warning("[%s] 원본 레코드 없음 — 검증 건너뜀", plan_id)
            continue

        passed = _compare_record(plan_id, arrow_rec, orig)
        if not passed:
            all_passed = False

    if all_passed:
        log.info("=== 검증 통과: 모든 샘플 일치 ===")
    else:
        log.error("=== 검증 실패: 일부 샘플 불일치 ===")

    return all_passed


def _compare_record(
    plan_id: str,
    arrow_rec: dict,
    orig: dict,
) -> bool:
    """단일 레코드의 Arrow vs 원본 JSONL 필드별 비교.

    Args:
        plan_id: 로그 출력용 식별자.
        arrow_rec: Arrow 데이터셋에서 읽은 레코드.
        orig: 원본 JSONL 레코드.

    Returns:
        bool: 모든 필드 일치 시 True.
    """
    ok = True

    # plan_id 확인
    if arrow_rec["plan_id"] != str(orig["plan_id"]):
        log.error("[%s] plan_id 불일치: %s vs %s", plan_id, arrow_rec["plan_id"], orig["plan_id"])
        ok = False

    # Arrow는 Sequence(struct)를 columnar dict(dict of lists)로 반환하므로
    # 개수는 첫 번째 컬럼의 길이로 확인해야 한다.
    arrow_rooms_count = len(arrow_rec["rooms"]["rid"])
    orig_rooms_count = len(orig["rooms"])

    # rooms 수 확인
    if arrow_rooms_count != orig_rooms_count:
        log.error("[%s] rooms 수 불일치: %d vs %d", plan_id, arrow_rooms_count, orig_rooms_count)
        ok = False
    else:
        # 각 방의 type, coords 확인 (columnar 접근)
        for i, o_room in enumerate(orig["rooms"]):
            a_type = arrow_rec["rooms"]["type"][i]
            a_coords = list(arrow_rec["rooms"]["coords"][i])
            if a_type != o_room["type"]:
                log.error("[%s] rooms[%d].type 불일치: %s vs %s", plan_id, i, a_type, o_room["type"])
                ok = False
            if a_coords != list(o_room["coords"]):
                log.error("[%s] rooms[%d].coords 불일치", plan_id, i)
                ok = False

    # edges 수 확인 (columnar: edges["pair"] 리스트 길이)
    arrow_edges_count = len(arrow_rec["edges"]["pair"])
    orig_edges_count = len(orig["edges"])
    if arrow_edges_count != orig_edges_count:
        log.error("[%s] edges 수 불일치: %d vs %d", plan_id, arrow_edges_count, orig_edges_count)
        ok = False
    else:
        for i, o_edge in enumerate(orig["edges"]):
            a_pair = list(arrow_rec["edges"]["pair"][i])
            if a_pair != list(o_edge["pair"]):
                log.error("[%s] edges[%d].pair 불일치: %s vs %s", plan_id, i, a_pair, o_edge["pair"])
                ok = False

    # front_door null/비null 일치 확인 (columnar: front_door["x"] 리스트 길이로 판별)
    orig_fd = orig.get("front_door")
    arrow_fd_is_null = len(arrow_rec["front_door"]["x"]) == 0
    orig_fd_is_null = orig_fd is None
    if arrow_fd_is_null != orig_fd_is_null:
        log.error(
            "[%s] front_door null 불일치: arrow=%s, orig=%s",
            plan_id,
            "null" if arrow_fd_is_null else "있음",
            "null" if orig_fd_is_null else "있음",
        )
        ok = False

    # spatial 수 확인 (columnar: spatial["rid_a"] 리스트 길이)
    arrow_spatial_count = len(arrow_rec["spatial"]["rid_a"])
    orig_spatial_count = len(orig["spatial"])
    if arrow_spatial_count != orig_spatial_count:
        log.error(
            "[%s] spatial 수 불일치: %d vs %d",
            plan_id,
            arrow_spatial_count,
            orig_spatial_count,
        )
        ok = False

    if ok:
        log.info("[%s] 검증 통과", plan_id)

    return ok
