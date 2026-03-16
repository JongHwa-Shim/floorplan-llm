"""커스텀 토큰 목록 정의 모듈.

평면도 생성 LLM에 추가할 커스텀 토큰을 카테고리별로 정의한다.
토큰 순서는 재현성 보장을 위해 고정한다.
BOS/EOS/PAD는 기존 LLM 토큰을 재활용하므로 목록에서 제외한다.
"""

import json
from pathlib import Path


# 기본값 (config 미지정 시 fallback)
_DEFAULT_MAX_COORD_X = 255
_DEFAULT_MAX_COORD_Y = 255
_DEFAULT_MAX_RID = 15

# 8방위 공간 관계
SPATIAL_DIRECTIONS = [
    "right",
    "right-below",
    "below",
    "left-below",
    "left",
    "left-above",
    "above",
    "right-above",
]


def _load_final_room_types(merge_config_path: Path) -> list[str]:
    """room_type_merge.json을 읽어 merge 후 최종 room type 집합을 반환한다.

    Args:
        merge_config_path: room_type_merge.json 파일 경로.

    Returns:
        중복 제거 후 정렬된 최종 room type 문자열 목록.
    """
    with open(merge_config_path, encoding="utf-8") as f:
        config = json.load(f)

    type_name_map: dict[str, str] = config["type_name_map"]   # G값 → 원본 타입명
    merge_rules: dict[str, str] = config["merge_rules"]        # 원본 → 병합 타입명

    final_types: set[str] = set()
    for raw_type in type_name_map.values():
        # merge 규칙이 있으면 병합 타입으로, 없으면 원본 타입 그대로 사용
        merged = merge_rules.get(raw_type, raw_type)
        final_types.add(merged)

    # outline은 방 인스턴스가 아니므로 제외 (별도 구조 토큰으로 처리)
    final_types.discard("outline")

    return sorted(final_types)  # 알파벳 정렬로 순서 고정


def build_token_list(
    merge_config_path: Path,
    max_rid: int = _DEFAULT_MAX_RID,
    max_coord_x: int = _DEFAULT_MAX_COORD_X,
    max_coord_y: int = _DEFAULT_MAX_COORD_Y,
) -> dict[str, list[str]]:
    """카테고리별 커스텀 토큰 목록을 구성하여 반환한다.

    토큰 등록 순서가 ID 배정 순서이므로, 카테고리 순서와 카테고리 내 순서를
    절대 변경하지 않는다.

    Args:
        merge_config_path: room_type_merge.json 파일 경로.
            방 타입 병합 규칙을 읽어 <TYPE:?> 토큰 목록을 동적으로 생성한다.
        max_rid: 방 인스턴스 ID 최대값. <RID:0>~<RID:max_rid> 생성.
        max_coord_x: X 좌표 최대값. <X:0>~<X:max_coord_x> 생성.
        max_coord_y: Y 좌표 최대값. <Y:0>~<Y:max_coord_y> 생성.

    Returns:
        카테고리명 → 토큰 문자열 목록 딕셔너리.
        키 순서: coord_x → coord_y → room_id → room_type →
                 spatial_rel → structure_special → structure

    Raises:
        FileNotFoundError: merge_config_path 파일이 존재하지 않을 때.
    """
    # --- 좌표 토큰 ---
    coord_x = [f"<X:{v}>" for v in range(max_coord_x + 1)]
    coord_y = [f"<Y:{v}>" for v in range(max_coord_y + 1)]

    # --- 방 인스턴스 ID 토큰 ---
    room_id = [f"<RID:{i}>" for i in range(max_rid + 1)]

    # --- 방 타입 토큰 (merge 규칙 적용 후 최종 타입) ---
    final_types = _load_final_room_types(merge_config_path)
    room_type = [f"<TYPE:{t}>" for t in final_types]

    # outline은 구조 토큰으로 별도 처리
    room_type_with_outline = room_type + ["<TYPE:outline>"]

    # --- 공간 관계 토큰 (8방위) ---
    spatial_rel = [f"<REL:{d}>" for d in SPATIAL_DIRECTIONS]  # 8개

    # --- 구조 토큰: special token 계열 (PAD만 추가, BOS/EOS는 기존 재활용) ---
    # PAD는 배치 패딩에 필요하므로 special token으로 등록
    structure_special = ["<PAD>"]

    # --- 구조 토큰: 시퀀스 구역 구분 ---
    structure = [
        # 입력/출력 영역 구분
        "<INPUT>",
        "<END_INPUT>",
        "<OUTPUT>",
        "<END_OUTPUT>",
        # 방 블록 (outline 포함 — <TYPE:outline>으로 구분)
        # 좌표는 <ROOM>...<END_ROOM> 내에 <X:n> <Y:n> 토큰으로 직접 나열
        "<ROOM>",
        "<END_ROOM>",
        # 엣지/문 블록 (<EDGES>/<END_EDGES>는 제거됨 — 개별 <EDGE>로만 구분)
        "<EDGE>",
        "<END_EDGE>",
        "<DOOR>",
        "<SEP_DOOR>",   # 문 위치(cx,cy)와 크기(w,h) 구분자
        "<END_DOOR>",
        "<NO_DOOR>",
        # 현관문 (<DOOR>/<END_DOOR> 재활용, <NO_FRONT_DOOR> 삭제됨)
        "<FRONT_DOOR>",
        # 문 방향 (더 이상 토크나이징에 사용하지 않으나 vocab 호환성 유지)
        "<DOOR_H>",
        "<DOOR_V>",
        # 공간 관계 블록 (<SPATIAL>/<END_SPATIAL> 제거 — 개별 <SP>로만 구분)
        "<SP>",      # 개별 spatial 관계 항목 시작
        "<END_SP>",  # 개별 spatial 관계 항목 끝
        # 방 요약 블록 (INPUT 전용 조건 정보)
        "<ROOM_SUMMARY>",      # 방 요약 블록 시작
        "<END_ROOM_SUMMARY>",  # 방 요약 블록 끝
        "<TOTAL>",   # 전체 방 개수 레이블 (수는 <X:N> 토큰 재활용)
        "<COUNT>",   # 방 종류별 개수 레이블 (수는 <X:N> 토큰 재활용)
    ]

    return {
        "coord_x": coord_x,
        "coord_y": coord_y,
        "room_id": room_id,
        "room_type": room_type_with_outline,
        "spatial_rel": spatial_rel,
        "structure_special": structure_special,
        "structure": structure,
    }


def flatten_token_list(categories: dict[str, list[str]]) -> list[str]:
    """카테고리별 토큰 목록을 단일 순서 고정 리스트로 평탄화한다.

    Args:
        categories: build_token_list()의 반환값.

    Returns:
        순서 고정된 전체 커스텀 토큰 문자열 목록.
    """
    result: list[str] = []
    for tokens in categories.values():
        result.extend(tokens)
    return result
