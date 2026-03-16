"""출력 JSONL 검증 스크립트.

생성된 JSONL 파일의 스키마 및 데이터 무결성을 검증한다.

Usage:
    python scripts/rplan2json/validate_jsonl.py data/dataset/processed_dataset/rplan/jsonl/floorplans_0000.jsonl
    python scripts/rplan2json/validate_jsonl.py data/dataset/processed_dataset/rplan/jsonl/floorplans_0000.jsonl -o report.txt
    python scripts/rplan2json/validate_jsonl.py data/dataset/processed_dataset/rplan/jsonl/floorplans_0000.jsonl --no-output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import orjson


def validate_record(record: dict, line_num: int) -> list[str]:
    """단일 JSONL 레코드의 스키마 및 무결성 검증.

    Args:
        record: 파싱된 JSONL 레코드.
        line_num: 줄 번호 (에러 메시지용).

    Returns:
        에러 메시지 리스트 (빈 리스트면 통과).
    """
    errors: list[str] = []
    prefix = f"[line {line_num}]"

    # 필수 필드 존재 확인
    for field in ("plan_id", "rooms", "edges", "spatial"):
        if field not in record:
            errors.append(f"{prefix} 필수 필드 누락: {field}")

    if "rooms" not in record:
        return errors

    rooms = record["rooms"]

    # rooms 검증
    rids = set()
    for i, room in enumerate(rooms):
        rid = room.get("rid")
        if rid is None:
            errors.append(f"{prefix} rooms[{i}] rid 누락")
            continue

        if rid in rids:
            errors.append(f"{prefix} 중복 rid: {rid}")
        rids.add(rid)

        if "type" not in room:
            errors.append(f"{prefix} rooms[{i}] type 누락")

        coords = room.get("coords", [])
        if len(coords) % 2 != 0:
            errors.append(f"{prefix} rooms[{i}] coords 길이가 홀수: {len(coords)}")
        if len(coords) < 6:
            errors.append(f"{prefix} rooms[{i}] coords 꼭지점 부족: {len(coords)//2}개")

    # edges 검증
    for i, edge in enumerate(record.get("edges", [])):
        pair = edge.get("pair", [])
        if len(pair) != 2:
            errors.append(f"{prefix} edges[{i}] pair 길이 != 2")
            continue

        for rid in pair:
            if rid not in rids:
                errors.append(f"{prefix} edges[{i}] pair에 존재하지 않는 rid: {rid}")

    # spatial 검증
    for i, rel in enumerate(record.get("spatial", [])):
        if len(rel) != 3:
            errors.append(f"{prefix} spatial[{i}] 길이 != 3")
            continue

        rid_a, rid_b, direction = rel
        for rid in (rid_a, rid_b):
            if rid not in rids:
                errors.append(f"{prefix} spatial[{i}] 존재하지 않는 rid: {rid}")

        valid_dirs = {
            "right", "left", "above", "below",
            "right-above", "right-below", "left-above", "left-below",
        }
        if direction not in valid_dirs:
            errors.append(f"{prefix} spatial[{i}] 잘못된 방위: {direction}")

    # front_door 검증 (존재 시)
    fd = record.get("front_door")
    if fd is not None:
        for key in ("x", "y", "w", "h"):
            if key not in fd:
                errors.append(f"{prefix} front_door에 {key} 누락")

    return errors


def validate_file(jsonl_path: str, output_path: str | None = None) -> None:
    """JSONL 파일 전체 검증.

    Args:
        jsonl_path: JSONL 파일 경로.
        output_path: 결과 저장 경로. None이면 터미널만 출력.
    """
    path = Path(jsonl_path)
    if not path.exists():
        print(f"파일이 존재하지 않음: {jsonl_path}")
        sys.exit(1)

    out_file = open(output_path, "w", encoding="utf-8") if output_path else None

    def _print(msg: str = "") -> None:
        """터미널과 파일에 동시 출력."""
        print(msg)
        if out_file is not None:
            out_file.write(msg + "\n")

    try:
        total = 0
        error_count = 0
        all_errors: list[str] = []

        with open(path, "rb") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                total += 1
                try:
                    record = orjson.loads(line)
                except orjson.JSONDecodeError as e:
                    all_errors.append(f"[line {line_num}] JSON 파싱 에러: {e}")
                    error_count += 1
                    continue

                errors = validate_record(record, line_num)
                if errors:
                    all_errors.extend(errors)
                    error_count += 1

        # 결과 출력
        _print(f"검증 완료: {jsonl_path}")
        _print(f"  총 레코드: {total}")
        _print(f"  에러 레코드: {error_count}")
        _print(f"  통과율: {(total - error_count) / max(total, 1) * 100:.1f}%")

        if all_errors:
            _print(f"\n에러 상세 (최대 20개):")
            for err in all_errors[:20]:
                _print(f"  {err}")
            if len(all_errors) > 20:
                _print(f"  ... 외 {len(all_errors) - 20}개")

        if output_path:
            print(f"\n결과 저장됨: {output_path}")
    finally:
        if out_file is not None:
            out_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSONL 출력 검증")
    parser.add_argument("jsonl_path", help="검증할 JSONL 파일 경로")
    parser.add_argument("-o", "--output", help="결과 저장 경로 (기본: <입력파일명>_validation.txt)", default=None)
    parser.add_argument("--no-output", action="store_true", help="파일 저장 없이 터미널만 출력")
    args = parser.parse_args()

    if args.no_output:
        output_path = None
    elif args.output:
        output_path = args.output
    else:
        # 기본 경로: 입력 파일 기준 ../validation_result/<stem>_validation.txt
        p = Path(args.jsonl_path)
        validation_dir = p.parent.parent / "validation_result"
        validation_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(validation_dir / f"{p.stem}_validation.txt")

    validate_file(args.jsonl_path, output_path)
