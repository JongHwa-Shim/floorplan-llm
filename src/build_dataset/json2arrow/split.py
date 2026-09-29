"""방 개수 제외 분할과 학습 데이터의 프로토콜 검증."""

from __future__ import annotations

import json
from pathlib import Path


def room_count(rooms) -> int:
    """외곽선을 제외한 방 개수를 반환한다.

    Args:
        rooms: Arrow 열 단위 표현 또는 방 딕셔너리 목록.
    Returns:
        실제 방 개수.
    Raises:
        KeyError: 방 종류 필드가 없을 때.
    """
    types = rooms["type"] if isinstance(rooms, dict) else [r["type"] for r in rooms]
    return sum(kind != "outline" for kind in types)


def validate_held_out_count(value) -> None:
    """제외할 방 개수 설정을 검사한다.

    Args:
        value: None 또는 양의 정수.
    Returns:
        없음.
    Raises:
        ValueError: 유효한 방 개수 설정이 아닐 때.
    """
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError("held_out_room_count는 null 또는 양의 정수여야 합니다.")


def holdout_pools(dataset, held_out_room_count: int, test_size: int, val_size: int, seed: int):
    """시험 방 개수 전체를 학습·검증 후보에서 제외한다.

    Args:
        dataset: 분할 전 Arrow Dataset.
        held_out_room_count: 시험할 방 개수.
        test_size: 시험 표본 수.
        val_size: 추후 분리할 검증 표본 수.
        seed: 시험 표본 선택 시드.
    Returns:
        시험 Dataset과 학습·검증용 Dataset.
    Raises:
        ValueError: 분할에 필요한 표본이 부족하거나 설정이 잘못되었을 때.
    """
    validate_held_out_count(held_out_room_count)
    counts = [room_count(rooms) for rooms in dataset["rooms"]]
    test_indices = [i for i, count in enumerate(counts) if count == held_out_room_count]
    remaining = [i for i, count in enumerate(counts) if count != held_out_room_count]
    if len(test_indices) < test_size or len(remaining) <= val_size:
        raise ValueError("방 개수 제외 분할에 필요한 시험 또는 학습·검증 표본이 부족합니다.")
    test = dataset.select(test_indices).shuffle(seed=seed).select(range(test_size))
    return test, dataset.select(remaining)


def validate_training_split(dataset, split: str, held_out_room_count, arrow_dir) -> None:
    """각 학습 단계에서 분할 설정과 실제 방 개수 제외 여부를 확인한다.

    Args:
        dataset: 현재 단계에서 사용할 Dataset.
        split: train, validation 또는 test.
        held_out_room_count: 설정에서 선언한 시험 방 개수 또는 None.
        arrow_dir: 분할 기록이 저장된 Arrow 경로.
    Returns:
        없음.
    Raises:
        ValueError: 선언과 저장된 분할이 다르거나 제외 조건을 위반할 때.
    """
    validate_held_out_count(held_out_room_count)
    manifest = Path(arrow_dir) / "split_manifest.json"
    if manifest.exists():
        recorded = json.loads(manifest.read_text(encoding="utf-8"))
        if recorded.get("held_out_room_count") != held_out_room_count:
            raise ValueError("데이터 분할과 data.held_out_room_count 설정이 다릅니다.")
    if held_out_room_count is None:
        return
    if len(dataset) == 0:
        raise ValueError("사용할 데이터 분할이 비어 있습니다.")
    for rooms in dataset["rooms"]:
        is_held_out = room_count(rooms) == held_out_room_count
        if (split in ("train", "validation") and is_held_out) or (split == "test" and not is_held_out):
            raise ValueError(f"{split} 데이터가 방 개수 제외 프로토콜을 위반합니다.")
