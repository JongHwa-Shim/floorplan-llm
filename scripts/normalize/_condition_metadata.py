"""모델에 실제 제공된 INPUT 토큰에서 평가용 조건을 복원한다."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"<[^>]+>|\d+")
_RID = re.compile(r"<RID:(\d+)>")
_TYPE = re.compile(r"<TYPE:([^>]+)>")
_X = re.compile(r"<X:(\d+)>")
_Y = re.compile(r"<Y:(\d+)>")
_REL = re.compile(r"<REL:([^>]+)>")


def _blocks(tokens: list[str], opening: str, closing: str):
    """시작·종료 태그 사이의 토큰을 순서대로 반환한다."""
    index = 0
    while index < len(tokens):
        if tokens[index] != opening:
            index += 1
            continue
        try:
            end = tokens.index(closing, index + 1)
        except ValueError as exc:
            raise ValueError(f"입력 조건 블록이 닫히지 않았습니다: {opening}") from exc
        yield tokens[index + 1:end]
        index = end + 1


def _read_number(tokens: list[str], index: int) -> tuple[int, int]:
    """기본 어휘의 연속 숫자 토큰을 정수 하나로 읽는다."""
    digits = []
    while index < len(tokens) and tokens[index].isdigit():
        digits.append(tokens[index])
        index += 1
    if not digits:
        raise ValueError("ROOM_SUMMARY의 숫자 토큰이 없습니다.")
    return int("".join(digits)), index


def parse_condition_metadata(text: str) -> dict:
    """보이는 방·연결·공간 관계·개수만 학습 보상의 metadata 형태로 만든다.

    추론의 condition.json에는 drop 이전 좌표가 남을 수 있으므로 실제 INPUT 토큰을
    단일 진실원으로 사용한다. 숫자 카운트는 일반 LLM 토큰이며 여러 조각일 수도 있다.
    """
    tokens = _TOKEN.findall(text)
    try:
        begin = tokens.index("<INPUT>") + 1
        end = tokens.index("<END_INPUT>", begin)
    except ValueError as exc:
        raise ValueError("완전한 INPUT 토큰 시퀀스가 없습니다.") from exc
    tokens = tokens[begin:end]

    total_rooms = None
    type_counts: dict[str, int] = {}
    for summary in _blocks(tokens, "<ROOM_SUMMARY>", "<END_ROOM_SUMMARY>"):
        index = 0
        while index < len(summary):
            token = summary[index]
            if token == "<TOTAL>":
                total_rooms, index = _read_number(summary, index + 1)
            elif (match := _TYPE.fullmatch(token)) is not None:
                if index + 1 >= len(summary) or summary[index + 1] != "<COUNT>":
                    raise ValueError("방 종류 뒤에 COUNT 토큰이 없습니다.")
                type_counts[match.group(1)], index = _read_number(summary, index + 2)
            else:
                index += 1

    rooms = []
    for block in _blocks(tokens, "<ROOM>", "<END_ROOM>"):
        rid = next((int(m.group(1)) for token in block if (m := _RID.fullmatch(token))), None)
        room_type = next((m.group(1) for token in block if (m := _TYPE.fullmatch(token))), "")
        coords = []
        for token in block:
            if (match := _X.fullmatch(token)) or (match := _Y.fullmatch(token)):
                coords.append(int(match.group(1)))
        rooms.append({"rid": rid, "type": room_type, "coords": coords})

    edges = []
    for block in _blocks(tokens, "<EDGE>", "<END_EDGE>"):
        pair = [int(m.group(1)) for token in block if (m := _RID.fullmatch(token))]
        has_door = "<DOOR>" in block
        edges.append({"pair": pair, "door": [], "has_door": has_door})

    spatial = []
    for block in _blocks(tokens, "<SP>", "<END_SP>"):
        pair = [int(m.group(1)) for token in block if (m := _RID.fullmatch(token))]
        direction = next((m.group(1) for token in block if (m := _REL.fullmatch(token))), None)
        if len(pair) == 2 and direction is not None:
            spatial.append({"rid_a": pair[0], "rid_b": pair[1], "direction": direction})

    return {
        "total_rooms": total_rooms,
        "type_counts": type_counts,
        "rooms": rooms,
        "edges": edges,
        "spatial": spatial,
        "front_door": None,
    }
