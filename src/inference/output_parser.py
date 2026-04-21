"""OUTPUT / INPUT 토큰 시퀀스 역변환 모듈.

모델이 생성한 OUTPUT 토큰 시퀀스를 파싱하여 JSONL 호환 딕셔너리로 역변환하고,
텍스트 파일로 저장된 INPUT 토큰 시퀀스를 파싱하여 시각화용 딕셔너리로 역변환한다.
결과 딕셔너리는 기존 FloorplanVisualizer에 바로 전달할 수 있다.

OUTPUT 토큰 형식:
    <OUTPUT>
      <FRONT_DOOR> <X:cx> <Y:cy> <SEP_DOOR> <X:w> <Y:h> <END_DOOR>
      <ROOM> <TYPE:outline> <X:x1> <Y:y1> ... <END_ROOM>
      <ROOM> <TYPE:livingroom> <X:x1> <Y:y1> ... <END_ROOM>
      ...
      <DOOR> <X:cx> <Y:cy> <SEP_DOOR> <X:w> <Y:h> <END_DOOR>
      ...
    <END_OUTPUT>

INPUT 토큰 형식:
    <INPUT>
      <ROOM_SUMMARY> ... <END_ROOM_SUMMARY>         (무시)
      <FRONT_DOOR> [<X:cx> <Y:cy> <SEP_DOOR> <X:w> <Y:h>] <END_DOOR>
      <ROOM> <RID:n> [<TYPE:t>] [<X:x1> <Y:y1> ...] <END_ROOM>
      ...
      <EDGE> <RID:n> [<DOOR> <X:cx> <Y:cy> <SEP_DOOR> <X:w> <Y:h> <END_DOOR>] <END_EDGE>
      ...
      <SP> <RID:a> <RID:b> <REL:dir> <END_SP>
      ...
    <END_INPUT>
"""

from __future__ import annotations

import logging
import re
from enum import Enum, auto

from src.training.augmentation.tokenizer import Vocab

logger = logging.getLogger(__name__)


class _State(Enum):
    """파서 상태 머신 상태 정의."""
    IDLE = auto()
    FRONT_DOOR = auto()
    ROOM = auto()
    DOOR = auto()
    EDGE = auto()       # INPUT 파서 전용
    SPATIAL = auto()    # INPUT 파서 전용
    ROOM_SUMMARY = auto()  # INPUT 파서 전용 (내용 무시)


# 토큰 패턴 (정규식으로 토큰 문자열에서 값 추출)
_X_PATTERN = re.compile(r"^<X:(\d+)>$")
_Y_PATTERN = re.compile(r"^<Y:(\d+)>$")
_TYPE_PATTERN = re.compile(r"^<TYPE:(.+)>$")
_RID_PATTERN = re.compile(r"^<RID:(\d+)>$")
_REL_PATTERN = re.compile(r"^<REL:(.+)>$")


def _extract_x(token: str) -> int | None:
    """<X:n> 토큰에서 좌표값을 추출한다."""
    m = _X_PATTERN.match(token)
    return int(m.group(1)) if m else None


def _extract_y(token: str) -> int | None:
    """<Y:n> 토큰에서 좌표값을 추출한다."""
    m = _Y_PATTERN.match(token)
    return int(m.group(1)) if m else None


def _extract_type(token: str) -> str | None:
    """<TYPE:t> 토큰에서 타입 문자열을 추출한다."""
    m = _TYPE_PATTERN.match(token)
    return m.group(1) if m else None


def _extract_rid(token: str) -> int | None:
    """<RID:n> 토큰에서 rid 값을 추출한다."""
    m = _RID_PATTERN.match(token)
    return int(m.group(1)) if m else None


def _extract_rel(token: str) -> str | None:
    """<REL:dir> 토큰에서 방향 문자열을 추출한다."""
    m = _REL_PATTERN.match(token)
    return m.group(1) if m else None


def _clamp(value: int, lo: int = 0, hi: int = 255) -> int:
    """좌표값을 유효 범위로 클램핑한다."""
    return max(lo, min(hi, value))


def parse_output_tokens(
    token_ids: list[int],
    vocab: Vocab,
) -> dict | None:
    """OUTPUT 토큰 시퀀스를 평면도 딕셔너리로 역변환한다.

    상태 머신 기반으로 토큰을 순차 파싱하여 rooms, edges(doors), front_door를 구성한다.

    Args:
        token_ids: 모델이 생성한 토큰 ID 리스트.
        vocab: Vocab 객체.

    Returns:
        평면도 딕셔너리:
            - rooms: list[dict] (rid는 파싱 순서대로 자동 부여)
            - edges: list[dict] (독립 DOOR들을 수집, pair는 [0,0] 더미값)
            - front_door: dict | None
            - spatial: [] (OUTPUT에 spatial 없음)
        파싱 실패 시 None.
    """
    # 토큰 ID → 문자열 변환
    tokens = [vocab.id_to_token.get(tid, f"<UNK:{tid}>") for tid in token_ids]

    state = _State.IDLE
    started = False  # <OUTPUT> 토큰을 만났는지

    # 결과 컨테이너
    rooms: list[dict] = []
    doors: list[dict] = []
    front_door: dict | None = None

    # 현재 파싱 중인 블록의 임시 데이터
    current_room_type: str | None = None
    current_room_coords: list[int] = []
    current_door_values: list[int] = []  # [cx, cy, w, h] 순서로 수집
    current_door_sep_seen: bool = False
    is_front_door: bool = False
    front_door_no_door: bool = False

    rid_counter = 0  # OUTPUT에 RID가 없으므로 순서대로 자동 부여

    try:
        for token in tokens:
            # <OUTPUT> 시작 탐지
            if token == "<OUTPUT>":
                started = True
                continue

            # <OUTPUT> 이전의 토큰은 무시
            if not started:
                continue

            # <END_OUTPUT> 또는 EOS → 종료
            if token in ("<END_OUTPUT>", "<EOS>"):
                break

            # --- 상태별 파싱 ---

            if state == _State.IDLE:
                if token == "<FRONT_DOOR>":
                    state = _State.FRONT_DOOR
                    is_front_door = True
                    front_door_no_door = False
                    current_door_values = []
                    current_door_sep_seen = False
                elif token == "<ROOM>":
                    state = _State.ROOM
                    current_room_type = None
                    current_room_coords = []
                elif token == "<DOOR>":
                    state = _State.DOOR
                    is_front_door = False
                    current_door_values = []
                    current_door_sep_seen = False
                # 그 외 토큰은 무시 (EOS, 숫자 토큰 등)

            elif state == _State.FRONT_DOOR:
                if token == "<NO_DOOR>":
                    front_door_no_door = True
                elif token == "<SEP_DOOR>":
                    current_door_sep_seen = True
                elif token == "<END_DOOR>":
                    if front_door_no_door:
                        front_door = None
                    else:
                        front_door = _build_door_dict(current_door_values)
                    state = _State.IDLE
                else:
                    # <X:n> 또는 <Y:n> 좌표 수집
                    val = _extract_x(token)
                    if val is not None:
                        current_door_values.append(_clamp(val))
                    else:
                        val = _extract_y(token)
                        if val is not None:
                            current_door_values.append(_clamp(val))

            elif state == _State.ROOM:
                if token == "<END_ROOM>":
                    room = {
                        "rid": rid_counter,
                        "type": current_room_type or "unknown",
                        "coords": current_room_coords,
                    }
                    rooms.append(room)
                    rid_counter += 1
                    state = _State.IDLE
                else:
                    # <TYPE:t> 체크
                    rtype = _extract_type(token)
                    if rtype is not None:
                        current_room_type = rtype
                    else:
                        # <X:n> 또는 <Y:n> 좌표 수집
                        val = _extract_x(token)
                        if val is not None:
                            current_room_coords.append(_clamp(val))
                        else:
                            val = _extract_y(token)
                            if val is not None:
                                current_room_coords.append(_clamp(val))

            elif state == _State.DOOR:
                if token == "<SEP_DOOR>":
                    current_door_sep_seen = True
                elif token == "<END_DOOR>":
                    door = _build_door_dict(current_door_values)
                    if door is not None:
                        doors.append(door)
                    state = _State.IDLE
                else:
                    val = _extract_x(token)
                    if val is not None:
                        current_door_values.append(_clamp(val))
                    else:
                        val = _extract_y(token)
                        if val is not None:
                            current_door_values.append(_clamp(val))

    except Exception as e:
        logger.warning("OUTPUT 토큰 파싱 중 오류 발생: %s", e)
        return None

    if not started:
        logger.warning("OUTPUT 토큰에서 <OUTPUT> 시작 토큰을 찾지 못함")
        return None

    if not rooms:
        logger.warning("OUTPUT 파싱 결과에 방(room)이 없음")
        return None

    # edges 구성: 독립 DOOR들을 리스트로 수집 (pair는 [0,0] 더미값)
    edges = [
        {"pair": [0, 0], "doors": [door]}
        for door in doors
    ]

    return {
        "rooms": rooms,
        "edges": edges,
        "front_door": front_door,
        "spatial": [],
    }


def parse_input_tokens(
    token_ids: list[int],
    vocab: Vocab,
    plan_id: str = "unknown",
) -> dict | None:
    """INPUT 토큰 시퀀스를 평면도 딕셔너리로 역변환한다.

    텍스트 파일 입력 모드(txt_dir)에서 토큰화된 INPUT 시퀀스를 파싱하여
    FloorplanVisualizer 호환 딕셔너리를 생성한다.

    INPUT 형식에서:
    - <ROOM_SUMMARY> 블록은 파싱 후 무시 (시각화 불필요)
    - <EDGE> 블록은 rid + doors로 구성 (edges 리스트)
    - <SP> 블록은 spatial 리스트로 구성
    - EDGE의 doors는 "doors" 키 사용 (FloorplanVisualizer 호환)

    Args:
        token_ids: 입력 조건 토큰 ID 리스트.
        vocab: Vocab 객체.
        plan_id: 결과 딕셔너리에 포함할 plan_id.

    Returns:
        평면도 딕셔너리:
            - plan_id: str
            - rooms: list[dict] (rid, type, coords)
            - edges: list[dict] (rid, doors)
            - front_door: dict | None
            - spatial: list[dict] (rid_a, rid_b, direction)
        파싱 실패 시 None.
    """
    tokens = [vocab.id_to_token.get(tid, f"<UNK:{tid}>") for tid in token_ids]

    state = _State.IDLE
    started = False

    rooms: list[dict] = []
    edges: list[dict] = []
    spatial: list[dict] = []
    front_door: dict | None = None

    # 현재 파싱 중인 블록 임시 데이터
    current_room_rid: int | None = None
    current_room_type: str | None = None
    current_room_coords: list[int] = []

    current_edge_rid: int | None = None
    current_edge_doors: list[dict] = []
    # EDGE 내부 DOOR 파싱용
    current_door_values: list[int] = []
    current_door_sep_seen: bool = False
    in_edge_door: bool = False  # EDGE 안에 있는 DOOR 파싱 중 여부

    current_sp_rids: list[int] = []
    current_sp_rel: str | None = None

    # FRONT_DOOR 임시
    current_fd_values: list[int] = []
    current_fd_sep_seen: bool = False
    fd_no_door: bool = False

    try:
        for token in tokens:
            if token == "<INPUT>":
                started = True
                continue
            if not started:
                continue
            if token == "<END_INPUT>":
                break

            if state == _State.IDLE:
                if token == "<ROOM_SUMMARY>":
                    state = _State.ROOM_SUMMARY
                elif token == "<FRONT_DOOR>":
                    state = _State.FRONT_DOOR
                    current_fd_values = []
                    current_fd_sep_seen = False
                    fd_no_door = False
                elif token == "<ROOM>":
                    state = _State.ROOM
                    current_room_rid = None
                    current_room_type = None
                    current_room_coords = []
                elif token == "<EDGE>":
                    state = _State.EDGE
                    current_edge_rid = None
                    current_edge_doors = []
                    in_edge_door = False
                elif token == "<SP>":
                    state = _State.SPATIAL
                    current_sp_rids = []
                    current_sp_rel = None

            elif state == _State.ROOM_SUMMARY:
                # 내용 전체 무시, END_ROOM_SUMMARY 만나면 IDLE 복귀
                if token == "<END_ROOM_SUMMARY>":
                    state = _State.IDLE

            elif state == _State.FRONT_DOOR:
                if token == "<NO_DOOR>":
                    fd_no_door = True
                elif token == "<SEP_DOOR>":
                    current_fd_sep_seen = True
                elif token == "<END_DOOR>":
                    front_door = None if fd_no_door else _build_door_dict(current_fd_values)
                    state = _State.IDLE
                else:
                    val = _extract_x(token)
                    if val is not None:
                        current_fd_values.append(_clamp(val))
                    else:
                        val = _extract_y(token)
                        if val is not None:
                            current_fd_values.append(_clamp(val))

            elif state == _State.ROOM:
                if token == "<END_ROOM>":
                    rooms.append({
                        "rid": current_room_rid if current_room_rid is not None else len(rooms),
                        "type": current_room_type or "unknown",
                        "coords": current_room_coords,
                    })
                    state = _State.IDLE
                else:
                    rid = _extract_rid(token)
                    if rid is not None:
                        current_room_rid = rid
                    else:
                        rtype = _extract_type(token)
                        if rtype is not None:
                            current_room_type = rtype
                        else:
                            val = _extract_x(token)
                            if val is not None:
                                current_room_coords.append(_clamp(val))
                            else:
                                val = _extract_y(token)
                                if val is not None:
                                    current_room_coords.append(_clamp(val))

            elif state == _State.EDGE:
                if token == "<END_EDGE>":
                    edges.append({
                        "rid": current_edge_rid,
                        "doors": current_edge_doors,
                    })
                    state = _State.IDLE
                elif token == "<DOOR>" and not in_edge_door:
                    # EDGE 내부 DOOR 시작
                    in_edge_door = True
                    current_door_values = []
                    current_door_sep_seen = False
                elif in_edge_door:
                    if token == "<SEP_DOOR>":
                        current_door_sep_seen = True
                    elif token == "<END_DOOR>":
                        door = _build_door_dict(current_door_values)
                        if door is not None:
                            current_edge_doors.append(door)
                        in_edge_door = False
                    else:
                        val = _extract_x(token)
                        if val is not None:
                            current_door_values.append(_clamp(val))
                        else:
                            val = _extract_y(token)
                            if val is not None:
                                current_door_values.append(_clamp(val))
                else:
                    # EDGE의 RID
                    rid = _extract_rid(token)
                    if rid is not None:
                        current_edge_rid = rid

            elif state == _State.SPATIAL:
                if token == "<END_SP>":
                    if len(current_sp_rids) >= 2 and current_sp_rel is not None:
                        spatial.append({
                            "rid_a": current_sp_rids[0],
                            "rid_b": current_sp_rids[1],
                            "direction": current_sp_rel,
                        })
                    state = _State.IDLE
                else:
                    rid = _extract_rid(token)
                    if rid is not None:
                        current_sp_rids.append(rid)
                    else:
                        rel = _extract_rel(token)
                        if rel is not None:
                            current_sp_rel = rel

    except Exception as e:
        logger.warning("INPUT 토큰 파싱 중 오류 발생: %s", e)
        return None

    if not started:
        logger.warning("INPUT 토큰에서 <INPUT> 시작 토큰을 찾지 못함")
        return None

    return {
        "plan_id": plan_id,
        "rooms": rooms,
        "edges": edges,
        "front_door": front_door,
        "spatial": spatial,
    }


def _build_door_dict(values: list[int]) -> dict | None:
    """수집된 좌표값으로 door 딕셔너리를 구성한다.

    기대 순서: [cx, cy, w, h] (SEP_DOOR 전에 cx,cy / 후에 w,h)

    Args:
        values: 수집된 좌표값 리스트.

    Returns:
        {"x": cx, "y": cy, "w": w, "h": h} 딕셔너리.
        값이 부족하면 None.
    """
    if len(values) < 4:
        logger.debug("door 좌표 부족: %d/4개", len(values))
        return None
    return {
        "x": values[0],
        "y": values[1],
        "w": values[2],
        "h": values[3],
    }
