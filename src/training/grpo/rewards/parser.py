"""출력 토큰 시퀀스 파서 모듈.

모델이 생성한 completion 토큰 ID 시퀀스를 구조화된 ParsedFloorplan으로 변환한다.
모든 보상함수의 공통 입력을 제공하며, 파싱 과정에서 오류 토큰 위치를 기록한다.

출력 토큰 포맷 (tokenizer.py build_output_tokens 역변환):
    <OUTPUT>
      <FRONT_DOOR> <X:cx> <Y:cy> <SEP_DOOR> <X:w> <Y:h> <END_DOOR>
      <ROOM> <TYPE:outline> <X:x1> <Y:y1> ... <END_ROOM>
      <ROOM> <TYPE:t> <X:x1> <Y:y1> ... <END_ROOM>
      ...
      <DOOR> <X:cx> <Y:cy> <SEP_DOOR> <X:w> <Y:h> <END_DOOR>  (또는 <NO_DOOR>)
      ...
    <END_OUTPUT> [EOS]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.training.augmentation.tokenizer import Vocab

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 파싱 결과 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class ParsedRoom:
    """파싱된 방 블록.

    Attributes:
        room_type: 방 타입 문자열 (예: "bedroom", "outline").
        coords: 꼭짓점 좌표 리스트. [(x1,y1), (x2,y2), ...].
        coord_token_indices: 각 꼭짓점 좌표쌍의 첫 토큰 인덱스.
            coord_token_indices[i] = completion 내 (X:xi) 토큰의 인덱스.
            (Y:yi) 토큰은 해당 인덱스 + 1.
        block_start: completion 내 <ROOM> 토큰 인덱스.
        block_end: completion 내 <END_ROOM> 토큰 인덱스 (포함).
    """

    room_type: str
    coords: list[tuple[int, int]]
    coord_token_indices: list[int]
    block_start: int
    block_end: int


@dataclass
class ParsedDoor:
    """파싱된 문 블록.

    Attributes:
        cx: 문 중심 X 좌표.
        cy: 문 중심 Y 좌표.
        w: 문 너비.
        h: 문 높이.
        is_valid: 좌표가 완전히 파싱되었으면 True.
    """

    cx: int
    cy: int
    w: int
    h: int
    is_valid: bool = True


@dataclass
class ParsedFloorplan:
    """출력 토큰 파싱 결과.

    파싱 레벨:
        0: <OUTPUT> 토큰 없음 (완전 실패).
        1: <OUTPUT> 존재하나 구조 파싱 실패.
        2: 방 블록은 파싱됐으나 일부 오류 존재.
        3: 완전 성공 (front_door + rooms + doors 모두 정상).

    Attributes:
        success: 파싱 완전 성공 여부 (level == 3).
        level: 파싱 단계. 0~3.
        front_door: 현관문 정보. 없으면 None.
        rooms: 파싱된 방 블록 리스트 (outline 포함, 첫 번째가 outline).
        doors: 파싱된 문 블록 리스트 (방 블록 이후에 등장하는 DOOR).
        error_indices: 포맷 오류 토큰 인덱스 리스트 (신용할당용).
        error_spans: 오류 유형 → 오류 토큰 인덱스 리스트.
    """

    success: bool
    level: int
    front_door: dict | None  # {cx, cy, w, h}
    rooms: list[ParsedRoom]
    doors: list[ParsedDoor]
    error_indices: list[int]
    error_spans: dict[str, list[int]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def parse_output_tokens(
    token_ids: list[int],
    vocab: "Vocab",
) -> ParsedFloorplan:
    """completion 토큰 ID 리스트를 ParsedFloorplan으로 변환한다.

    tokenizer.py의 build_output_tokens()의 역변환.
    파싱 과정에서 발견된 오류 토큰 인덱스를 error_indices에 기록한다.

    Args:
        token_ids: completion 토큰 ID 리스트 (prompt 제외, <OUTPUT>부터 시작).
        vocab: Vocab 객체. token_to_id 및 id_to_token 포함.

    Returns:
        ParsedFloorplan 인스턴스.
    """
    parser = _OutputParser(token_ids, vocab)
    return parser.parse()


# ---------------------------------------------------------------------------
# 내부 파서
# ---------------------------------------------------------------------------

class _OutputParser:
    """출력 토큰 시퀀스를 순차적으로 파싱하는 내부 파서.

    Args:
        token_ids: 파싱할 토큰 ID 리스트.
        vocab: Vocab 객체.
    """

    def __init__(self, token_ids: list[int], vocab: "Vocab") -> None:
        self.ids = token_ids
        self.vocab = vocab
        self.n = len(token_ids)
        self.pos = 0  # 현재 파싱 위치

        # 자주 쓰는 토큰 ID를 미리 캐싱
        self._id = vocab.token_to_id
        self.OUTPUT = self._id.get("<OUTPUT>", -1)
        self.END_OUTPUT = self._id.get("<END_OUTPUT>", -1)
        self.FRONT_DOOR = self._id.get("<FRONT_DOOR>", -1)
        self.NO_DOOR = self._id.get("<NO_DOOR>", -1)
        self.SEP_DOOR = self._id.get("<SEP_DOOR>", -1)
        self.END_DOOR = self._id.get("<END_DOOR>", -1)
        self.ROOM = self._id.get("<ROOM>", -1)
        self.END_ROOM = self._id.get("<END_ROOM>", -1)
        self.DOOR = self._id.get("<DOOR>", -1)

        # id → 토큰 문자열 (X/Y 파싱용)
        self.id_to_token = vocab.id_to_token

        self.error_indices: list[int] = []

    def parse(self) -> ParsedFloorplan:
        """메인 파싱 진입점.

        Returns:
            ParsedFloorplan 인스턴스.
        """
        # Level 0: <OUTPUT> 토큰 확인
        output_pos = self._find_token(self.OUTPUT)
        if output_pos < 0:
            return ParsedFloorplan(
                success=False, level=0,
                front_door=None, rooms=[], doors=[],
                error_indices=list(range(len(self.ids))),
            )
        self.pos = output_pos + 1

        # Level 1: FRONT_DOOR 파싱
        front_door = self._parse_front_door()

        # Level 2: ROOM 블록 파싱
        rooms = self._parse_rooms()

        # <END_OUTPUT> 이전까지 DOOR 블록 파싱
        doors = self._parse_doors()

        # 파싱 완전 성공 여부 판단
        success = (
            len(rooms) > 0          # 최소 outline 존재
            and len(self.error_indices) == 0  # 포맷 오류 없음
        )
        level = 3 if success else (2 if len(rooms) > 0 else 1)

        error_spans: dict[str, list[int]] = {}
        if self.error_indices:
            error_spans["format"] = list(self.error_indices)

        return ParsedFloorplan(
            success=success,
            level=level,
            front_door=front_door,
            rooms=rooms,
            doors=doors,
            error_indices=list(self.error_indices),
            error_spans=error_spans,
        )

    # -------------------------------------------------------------------
    # 블록별 파싱 메서드
    # -------------------------------------------------------------------

    def _parse_front_door(self) -> dict | None:
        """FRONT_DOOR 블록을 파싱한다.

        형식:
            <FRONT_DOOR> <X:cx> <Y:cy> <SEP_DOOR> <X:w> <Y:h> <END_DOOR>
            <FRONT_DOOR> <NO_DOOR> <END_DOOR>

        Returns:
            {cx, cy, w, h} 딕셔너리. 없거나 파싱 실패 시 None.
        """
        if not self._peek(self.FRONT_DOOR):
            return None
        self.pos += 1  # <FRONT_DOOR> 소비

        # <NO_DOOR> 케이스
        if self._peek(self.NO_DOOR):
            self.pos += 1  # <NO_DOOR>
            if self._peek(self.END_DOOR):
                self.pos += 1
            return None

        # 좌표 파싱: cx, cy, SEP_DOOR, w, h
        cx = self._parse_x()
        cy = self._parse_y()
        if not self._peek(self.SEP_DOOR):
            # SEP_DOOR 누락 → 오류 기록 후 복구 시도
            self.error_indices.append(self.pos)
        else:
            self.pos += 1

        w = self._parse_x()
        h = self._parse_y()

        if not self._peek(self.END_DOOR):
            self.error_indices.append(self.pos)
        else:
            self.pos += 1

        if cx is None or cy is None or w is None or h is None:
            return None

        return {"cx": cx, "cy": cy, "w": w, "h": h}

    def _parse_rooms(self) -> list[ParsedRoom]:
        """연속된 ROOM 블록을 파싱한다.

        형식: <ROOM> <TYPE:t> <X:x1> <Y:y1> ... <END_ROOM>

        Returns:
            ParsedRoom 리스트.
        """
        rooms: list[ParsedRoom] = []
        while self.pos < self.n and self._peek(self.ROOM):
            room = self._parse_single_room()
            if room is not None:
                rooms.append(room)
        return rooms

    def _parse_single_room(self) -> ParsedRoom | None:
        """단일 ROOM 블록을 파싱한다.

        Returns:
            ParsedRoom 인스턴스. 파싱 완전 실패 시 None.
        """
        block_start = self.pos
        self.pos += 1  # <ROOM> 소비

        # <TYPE:t> 파싱
        room_type = self._parse_type()
        if room_type is None:
            # 타입 없음 → <END_ROOM>까지 스킵
            self._skip_to(self.END_ROOM)
            if self._peek(self.END_ROOM):
                block_end = self.pos
                self.pos += 1
            else:
                block_end = self.pos
            self.error_indices.append(block_start)
            return None

        # 좌표 쌍 파싱 (X/Y 교대)
        coords: list[tuple[int, int]] = []
        coord_token_indices: list[int] = []

        while self.pos < self.n and not self._peek(self.END_ROOM):
            # 다음 블록 토큰이면 중단
            if self._is_block_start_token(self.ids[self.pos]):
                # <END_ROOM> 없이 다른 블록 시작 → 오류
                self.error_indices.append(self.pos)
                break

            x_pos = self.pos
            x = self._parse_x()
            if x is None:
                # X 토큰이 아닌 다른 토큰 → 오류 기록 후 한 칸 스킵
                self.error_indices.append(self.pos)
                self.pos += 1
                continue

            y = self._parse_y()
            if y is None:
                # Y 토큰 없음 (X 이후 비정상) → 오류
                self.error_indices.append(self.pos - 1)  # X 토큰 위치
                continue

            coords.append((x, y))
            coord_token_indices.append(x_pos)  # X 토큰 인덱스 저장

        block_end = self.pos
        if self._peek(self.END_ROOM):
            block_end = self.pos
            self.pos += 1
        else:
            self.error_indices.append(self.pos)

        # 최소 4쌍 (직사각형 최소 꼭짓점)
        if len(coords) < 4:
            self.error_indices.extend(range(block_start, self.pos))

        return ParsedRoom(
            room_type=room_type,
            coords=coords,
            coord_token_indices=coord_token_indices,
            block_start=block_start,
            block_end=block_end,
        )

    def _parse_doors(self) -> list[ParsedDoor]:
        """방 블록 이후 DOOR 블록을 파싱한다.

        형식:
            <DOOR> <X:cx> <Y:cy> <SEP_DOOR> <X:w> <Y:h> <END_DOOR>
            <NO_DOOR>

        Returns:
            ParsedDoor 리스트.
        """
        doors: list[ParsedDoor] = []
        while self.pos < self.n:
            if self._peek(self.END_OUTPUT):
                self.pos += 1
                break
            if self._peek(self.DOOR):
                door = self._parse_single_door()
                if door is not None:
                    doors.append(door)
            elif self._peek(self.NO_DOOR):
                self.pos += 1  # <NO_DOOR>: 문 없음 표시, 건너뜀
            elif self.ids[self.pos] == (self.vocab.eos_token_id or -9999):
                break
            else:
                # 예기치 않은 토큰 → 오류 기록 후 스킵
                self.error_indices.append(self.pos)
                self.pos += 1
        return doors

    def _parse_single_door(self) -> ParsedDoor | None:
        """단일 DOOR 블록을 파싱한다.

        Returns:
            ParsedDoor 인스턴스. 좌표 파싱 실패 시 is_valid=False로 반환.
        """
        self.pos += 1  # <DOOR> 소비

        cx = self._parse_x()
        cy = self._parse_y()
        if not self._peek(self.SEP_DOOR):
            self.error_indices.append(self.pos)
        else:
            self.pos += 1
        w = self._parse_x()
        h = self._parse_y()
        if not self._peek(self.END_DOOR):
            self.error_indices.append(self.pos)
        else:
            self.pos += 1

        if cx is None or cy is None or w is None or h is None:
            return ParsedDoor(cx=0, cy=0, w=0, h=0, is_valid=False)
        return ParsedDoor(cx=cx, cy=cy, w=w, h=h, is_valid=True)

    # -------------------------------------------------------------------
    # 토큰 파싱 유틸
    # -------------------------------------------------------------------

    def _parse_x(self) -> int | None:
        """현재 위치의 <X:n> 토큰을 파싱하여 n을 반환한다.

        Returns:
            정수 좌표값. <X:n> 토큰이 아니면 None (pos 불변).
        """
        if self.pos >= self.n:
            return None
        tok = self.id_to_token.get(self.ids[self.pos], "")
        if tok.startswith("<X:") and tok.endswith(">"):
            try:
                val = int(tok[3:-1])
                self.pos += 1
                return val
            except ValueError:
                return None
        return None

    def _parse_y(self) -> int | None:
        """현재 위치의 <Y:m> 토큰을 파싱하여 m을 반환한다.

        Returns:
            정수 좌표값. <Y:m> 토큰이 아니면 None (pos 불변).
        """
        if self.pos >= self.n:
            return None
        tok = self.id_to_token.get(self.ids[self.pos], "")
        if tok.startswith("<Y:") and tok.endswith(">"):
            try:
                val = int(tok[3:-1])
                self.pos += 1
                return val
            except ValueError:
                return None
        return None

    def _parse_type(self) -> str | None:
        """현재 위치의 <TYPE:t> 토큰을 파싱하여 t를 반환한다.

        Returns:
            타입 문자열. <TYPE:t> 토큰이 아니면 None (pos 불변).
        """
        if self.pos >= self.n:
            return None
        tok = self.id_to_token.get(self.ids[self.pos], "")
        if tok.startswith("<TYPE:") and tok.endswith(">"):
            type_str = tok[6:-1]
            self.pos += 1
            return type_str
        return None

    def _peek(self, token_id: int) -> bool:
        """현재 위치 토큰이 token_id인지 확인한다 (pos 불변).

        Args:
            token_id: 확인할 토큰 ID.

        Returns:
            일치하면 True.
        """
        return self.pos < self.n and self.ids[self.pos] == token_id

    def _find_token(self, token_id: int) -> int:
        """시퀀스 내 token_id의 첫 번째 위치를 반환한다.

        Args:
            token_id: 찾을 토큰 ID.

        Returns:
            인덱스. 없으면 -1.
        """
        for i, tid in enumerate(self.ids):
            if tid == token_id:
                return i
        return -1

    def _skip_to(self, target_id: int) -> None:
        """target_id 토큰 위치까지 pos를 이동한다.

        Args:
            target_id: 목표 토큰 ID.
        """
        while self.pos < self.n and self.ids[self.pos] != target_id:
            self.pos += 1

    def _is_block_start_token(self, token_id: int) -> bool:
        """token_id가 블록 시작 토큰인지 확인한다.

        ROOM/DOOR/FRONT_DOOR/OUTPUT 관련 구조 토큰이면 True.

        Args:
            token_id: 확인할 토큰 ID.

        Returns:
            블록 시작 토큰이면 True.
        """
        return token_id in (
            self.ROOM, self.END_ROOM,
            self.DOOR, self.END_DOOR,
            self.NO_DOOR, self.END_OUTPUT,
            self.FRONT_DOOR,
        )
