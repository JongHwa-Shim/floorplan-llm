"""구조화 데이터 → 토큰 ID 시퀀스 변환 모듈.

Arrow에서 읽어온 row-oriented 딕셔너리를 token_definitions.py에 정의된
커스텀 토큰 ID 시퀀스로 변환한다.

토큰 시퀀스 형식:
    입력: [BOS] <INPUT>
              <FRONT_DOOR> <X:cx> <Y:cy> <X:w> <Y:h> <END_DOOR>  (현관문 있음)
              <FRONT_DOOR> <NO_DOOR> <END_DOOR>                    (현관문 없음)
              <ROOM> <RID:0> <TYPE:outline> <X:x1> <Y:y1> ... <END_ROOM>
              <ROOM> <RID:n> <TYPE:t> <X:x1> <Y:y1> ... <END_ROOM>
              <EDGES>
                  <EDGE> <RID:a> <RID:b> <DOOR> <X:cx> <Y:cy> <SEP_DOOR> <X:w> <Y:h> <END_DOOR> <END_EDGE>
                  <EDGE> <RID:a> <RID:b> <NO_DOOR> <END_EDGE>
              <SP> <RID:a> <RID:b> <REL:dir> <END_SP>
          <END_INPUT>
    출력: <OUTPUT>
              <FRONT_DOOR> <X:cx> <Y:cy> <SEP_DOOR> <X:w> <Y:h> <END_DOOR>
              <ROOM> <TYPE:t> <X:x1> <Y:y1> ... <END_ROOM>
              <DOOR> <X:cx> <Y:cy> <SEP_DOOR> <X:w> <Y:h> <END_DOOR>
          <END_OUTPUT> <EOS>

주의사항:
    - outline(rid=0)은 입력 ROOM 블록에 포함 (출력에는 미포함 — 모델의 생성 대상이 아님).
    - outline도 drop_block / drop_coords 증강 대상이나 drop_type 대상에서는 제외.
    입력 ROOM_SUMMARY 블록 형식 (방 블록 앞에 위치):
        <ROOM_SUMMARY> <TOTAL> 7 <TYPE:t1> <COUNT> 2 ... <END_ROOM_SUMMARY>
        - <TOTAL>/<COUNT> 뒤 숫자: LLM 기본 숫자 토큰 사용 (vocab.number_to_ids)
        - tokenizer 미제공 시 KeyError 발생 (숫자 토큰은 반드시 tokenizer 필요)
        - drop_room_summary_total: True이면 <TOTAL> + 숫자 토큰 쌍 생략
        - drop_room_summary_types: 해당 타입의 <TYPE:t> <COUNT> + 숫자 토큰 쌍 생략
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.training.augmentation.strategies import DropState


# ---------------------------------------------------------------------------
# Vocab 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class Vocab:
    """vocab_extension.json 기반 토큰↔ID 매핑 컨테이너.

    Args:
        token_to_id: 토큰 문자열 → ID 딕셔너리 (커스텀 토큰 전용).
        id_to_token: ID → 표시 문자열 딕셔너리 (커스텀 + 숫자 토큰 포함).
        bos_token_id: 기존 LLM BOS 토큰 ID (없으면 None).
        eos_token_id: 기존 LLM EOS 토큰 ID (없으면 None).
        number_to_ids: 정수 → LLM 기본 숫자 토큰 ID 리스트 (tokenizer 제공 시 populated).
            ROOM_SUMMARY의 <TOTAL>/<COUNT> 뒤 숫자 토큰에 사용한다.
    """

    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    number_to_ids: dict[int, list[int]] = field(default_factory=dict)

    def get(self, token: str) -> int:
        """토큰 문자열로 ID를 조회한다. 없으면 KeyError 발생.

        Args:
            token: 조회할 토큰 문자열.

        Returns:
            해당 토큰의 ID.

        Raises:
            KeyError: token이 vocab에 없을 때.
        """
        if token not in self.token_to_id:
            raise KeyError(f"토큰을 vocab에서 찾을 수 없음: {token!r}")
        return self.token_to_id[token]


# BOS 토큰이 없는 모델을 위한 후보 목록.
# 반드시 added_tokens(special token)으로 등록된 토큰만 포함해야 한다.
# 일반 BPE 토큰(<s> 등)은 unk_token_id가 None인 tokenizer에서 오탐될 수 있으므로 제외.
# Qwen2/Qwen2.5는 BOS 없이 사전훈련 → fallback 없이 None 반환.
# LLaMA/Mistral/Gemma 등은 bos_token_id가 명시되어 1순위에서 해결됨.
_BOS_FALLBACK_CANDIDATES: list[str] = [
    "<|begin_of_text|>",  # LLaMA-3 (special token으로 등록됨)
]


def _resolve_bos_token_id(tok) -> int | None:
    """tokenizer에서 BOS 토큰 ID를 동적으로 추출한다.

    표준 bos_token_id가 없는 모델(Qwen2 등)은 chat template 시작 토큰을
    후보 목록에서 순서대로 탐색하여 BOS 대용으로 사용한다.

    Args:
        tok: HuggingFace PreTrainedTokenizer 인스턴스.

    Returns:
        BOS 토큰 ID. 후보가 모두 없으면 None.
    """
    # 1순위: 표준 속성 (LLaMA, Gemma, Mistral 등)
    if tok.bos_token_id is not None:
        return tok.bos_token_id

    # 2순위: 후보 토큰을 순서대로 탐색
    # added_tokens(special token)으로 등록된 토큰만 BOS 대용으로 허용한다.
    # 일반 BPE 토큰은 unk_token_id가 None인 tokenizer에서 오탐되므로 제외.
    special_token_ids: set[int] = set(tok.added_tokens_decoder.keys())
    for candidate in _BOS_FALLBACK_CANDIDATES:
        tid = tok.convert_tokens_to_ids(candidate)
        if tid is not None and tid in special_token_ids:
            return tid

    return None


def load_vocab(
    vocab_extension_path: Path,
    tokenizer_dir: Path | None = None,
) -> Vocab:
    """vocab_extension.json을 로드하여 Vocab 객체를 생성한다.

    BOS/EOS ID는 tokenizer_dir이 주어지면 해당 tokenizer에서 추출하고,
    없으면 None으로 설정한다.

    Args:
        vocab_extension_path: vocab_extension.json 파일 경로.
        tokenizer_dir: 확장된 tokenizer가 저장된 디렉토리 (선택).
            주어지면 BOS/EOS ID를 tokenizer에서 추출한다.

    Returns:
        Vocab 객체.

    Raises:
        FileNotFoundError: vocab_extension_path가 존재하지 않을 때.
    """
    with open(vocab_extension_path, encoding="utf-8") as f:
        ext = json.load(f)

    token_to_id: dict[str, int] = ext["token_to_id"]
    # JSON key는 str이지만 id_to_token은 int key로 변환하여 사용
    id_to_token: dict[int, str] = {int(k): v for k, v in ext["id_to_token"].items()}

    bos_token_id: int | None = None
    eos_token_id: int | None = None
    number_to_ids: dict[int, list[int]] = {}

    if tokenizer_dir is not None:
        # BOS/EOS는 기존 LLM 토큰 재활용이므로 tokenizer에서 추출
        from transformers import AutoTokenizer  # 지연 임포트

        tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=True)
        bos_token_id = _resolve_bos_token_id(tok)
        eos_token_id = tok.eos_token_id

        # 숫자 → LLM 기본 토큰 ID 매핑 구성 (0~255 범위)
        # ROOM_SUMMARY의 <TOTAL>/<COUNT> 뒤 숫자 토큰에 사용
        # Qwen2.5는 숫자를 단일 토큰으로 처리하지만 안전하게 list[int]로 저장
        for n in range(256):
            ids: list[int] = tok.encode(str(n), add_special_tokens=False)
            number_to_ids[n] = ids
            # decoder에서 표시하기 위해 id_to_token에도 추가
            # Ġ/▁ 같은 subword prefix를 제거하여 숫자 문자열로만 저장
            for tid in ids:
                if tid not in id_to_token:
                    raw = tok.convert_ids_to_tokens([tid])[0]
                    id_to_token[tid] = raw.lstrip("Ġ▁Ċ")

    return Vocab(
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        number_to_ids=number_to_ids,
    )


# ---------------------------------------------------------------------------
# Arrow columnar → row-oriented 변환
# ---------------------------------------------------------------------------

def to_row_oriented(sample: dict) -> dict:
    """Arrow에서 읽어온 columnar 포맷 샘플을 row-oriented 딕셔너리로 변환한다.

    HuggingFace datasets는 Sequence(struct)를 struct-of-lists(columnar)로 저장한다.
    증강 및 토크나이징 편의를 위해 list-of-structs(row-oriented)로 변환한다.

    Args:
        sample: datasets[i]로 읽어온 raw 딕셔너리.

    Returns:
        row-oriented 딕셔너리:
            - plan_id: str
            - rooms: list of {"rid": int, "type": str, "coords": list[int]}
            - edges: list of {"pair": list[int], "door": list[{"x","y","w","h": float}]}
            - front_door: {"x","y","w","h": float} or None
            - spatial: list of {"rid_a": int, "rid_b": int, "direction": str}
    """
    # --- rooms ---
    rooms_raw = sample["rooms"]
    n_rooms = len(rooms_raw["rid"])
    rooms = [
        {
            "rid": int(rooms_raw["rid"][i]),
            "type": str(rooms_raw["type"][i]),
            "coords": [int(c) for c in rooms_raw["coords"][i]],
        }
        for i in range(n_rooms)
    ]

    # --- edges ---
    edges_raw = sample["edges"]
    n_edges = len(edges_raw["pair"])
    edges = []
    for i in range(n_edges):
        doors = _parse_door_sequence(edges_raw["door"][i])
        edges.append(
            {
                "pair": [int(r) for r in edges_raw["pair"][i]],
                "door": doors,
            }
        )

    # --- front_door ---
    fd_parsed = _parse_door_sequence(sample["front_door"])
    front_door_dict = fd_parsed[0] if fd_parsed else None

    # --- spatial ---
    sp_raw = sample["spatial"]
    if isinstance(sp_raw, dict):
        n_sp = len(sp_raw["rid_a"])
        spatial = [
            {
                "rid_a": int(sp_raw["rid_a"][j]),
                "rid_b": int(sp_raw["rid_b"][j]),
                "direction": str(sp_raw["direction"][j]),
            }
            for j in range(n_sp)
        ]
    else:
        spatial = [
            {
                "rid_a": int(s["rid_a"]),
                "rid_b": int(s["rid_b"]),
                "direction": str(s["direction"]),
            }
            for s in sp_raw
        ]

    return {
        "plan_id": sample["plan_id"],
        "rooms": rooms,
        "edges": edges,
        "front_door": front_door_dict,
        "spatial": spatial,
    }


def _parse_door_sequence(raw) -> list[dict]:
    """columnar 또는 row-oriented 문 시퀀스를 list-of-dicts로 변환한다.

    Args:
        raw: dict-of-lists (columnar) 또는 list-of-dicts (row-oriented).

    Returns:
        list of {"x": float, "y": float, "w": float, "h": float}.
    """
    if isinstance(raw, dict):
        n = len(raw.get("x", []))
        return [
            {
                "x": float(raw["x"][i]),
                "y": float(raw["y"][i]),
                "w": float(raw["w"][i]),
                "h": float(raw["h"][i]),
            }
            for i in range(n)
        ]
    return [
        {"x": float(d["x"]), "y": float(d["y"]), "w": float(d["w"]), "h": float(d["h"])}
        for d in raw
    ]


# ---------------------------------------------------------------------------
# 내부 토크나이징 유틸
# ---------------------------------------------------------------------------


def _coord_tokens(coords: list[int], vocab: Vocab) -> list[int]:
    """flat 좌표 리스트 [x1,y1,x2,y2,...] → [<X:x1>, <Y:y1>, ...] 토큰 ID 리스트.

    Args:
        coords: flat 정수 좌표 리스트.
        vocab: Vocab 객체.

    Returns:
        좌표 토큰 ID 리스트.
    """
    tokens: list[int] = []
    for i in range(0, len(coords), 2):
        tokens.append(vocab.get(f"<X:{coords[i]}>"))
        tokens.append(vocab.get(f"<Y:{coords[i + 1]}>"))
    return tokens


# ---------------------------------------------------------------------------
# 블록별 토크나이징 함수
# ---------------------------------------------------------------------------

def tokenize_front_door(
    front_door: dict | None,
    drop_front_door: bool,
    drop_front_door_coords: bool,
    vocab: Vocab,
) -> list[int]:
    """현관문 블록을 토큰 ID 시퀀스로 변환한다.

    형식:
        - 일반:      ``<FRONT_DOOR> <X:cx> <Y:cy> <SEP_DOOR> <X:w> <Y:h> <END_DOOR>``
        - 현관문 없음: ``<FRONT_DOOR> <NO_DOOR> <END_DOOR>``
        - DropBlock:  블록 전체 생략 (빈 리스트 반환)
        - DropCoords: ``<FRONT_DOOR> <END_DOOR>``

    Args:
        front_door: 현관문 딕셔너리 (없으면 None).
        drop_front_door: True면 블록 전체를 생략한다 (DropBlock 증강).
        drop_front_door_coords: True면 좌표를 생략한다 (DropCoords 증강).
        vocab: Vocab 객체.

    Returns:
        토큰 ID 리스트.
    """
    # DropBlock: 현관문 블록 전체 생략
    if drop_front_door:
        return []

    # DropCoords: 좌표 없이 블록 껍데기만 출력
    if drop_front_door_coords:
        return [vocab.get("<FRONT_DOOR>"), vocab.get("<END_DOOR>")]

    # 현관문 데이터 없음
    if front_door is None:
        return [
            vocab.get("<FRONT_DOOR>"),
            vocab.get("<NO_DOOR>"),
            vocab.get("<END_DOOR>"),
        ]

    cx = round(front_door["x"])
    cy = round(front_door["y"])
    w = round(front_door["w"])
    h = round(front_door["h"])
    return [
        vocab.get("<FRONT_DOOR>"),
        vocab.get(f"<X:{cx}>"),
        vocab.get(f"<Y:{cy}>"),
        vocab.get("<SEP_DOOR>"),   # 위치(cx,cy)와 크기(w,h) 구분자
        vocab.get(f"<X:{w}>"),
        vocab.get(f"<Y:{h}>"),
        vocab.get("<END_DOOR>"),
    ]


def tokenize_room_block(
    room: dict,
    drop_type: bool,
    drop_coords: bool,
    vocab: Vocab,
    drop_rid: bool = False,
    noisy_coords: list[int] | None = None,
) -> list[int]:
    """방(Room) 블록을 토큰 ID 시퀀스로 변환한다.

    증강 플래그에 따라 RID, 타입 또는 좌표 블록이 생략된다.

    Args:
        room: {"rid": int, "type": str, "coords": list[int]} 딕셔너리.
        drop_type: True면 <TYPE:t> 토큰 생략.
        drop_coords: True면 좌표 토큰 생략.
        vocab: Vocab 객체.
        drop_rid: True면 <RID:n> 토큰 생략 (OUTPUT 전용).
        noisy_coords: 노이즈가 추가된 좌표 리스트 (INPUT 전용).
            None이면 room["coords"] 원본 사용.

    Returns:
        토큰 ID 리스트.
            - 풀:       ``<ROOM> <RID:n> <TYPE:t> <X:x1> <Y:y1> ... <END_ROOM>``
            - DropRID:  ``<ROOM> <TYPE:t> <X:x1> <Y:y1> ... <END_ROOM>``
            - DropType: ``<ROOM> <RID:n> <X:x1> <Y:y1> ... <END_ROOM>``
            - DropCoords: ``<ROOM> <RID:n> <TYPE:t> <END_ROOM>``
    """
    tokens = [vocab.get("<ROOM>")]

    if not drop_rid:
        tokens.append(vocab.get(f"<RID:{room['rid']}>"))

    if not drop_type:
        tokens.append(vocab.get(f"<TYPE:{room['type']}>"))

    if not drop_coords:
        # 노이즈 좌표 우선 사용 (INPUT 전용), 없으면 원본 좌표
        coords = noisy_coords if noisy_coords is not None else room["coords"]
        tokens.extend(_coord_tokens(coords, vocab))

    tokens.append(vocab.get("<END_ROOM>"))
    return tokens


def tokenize_edge_block(
    edge: dict,
    drop_pair_mode: str | tuple[str, int] | None,
    drop_door_mode: str | None,
    vocab: Vocab,
    drop_edge_wrapper: bool = False,
) -> list[int]:
    """엣지(Edge) 블록을 토큰 ID 시퀀스로 변환한다.

    Args:
        edge: {"pair": list[int], "door": list[dict]} 딕셔너리.
        drop_pair_mode: None | "both" | ("one", kept_rid).
            "both"         → 두 RID 모두 생략.
            ("one", rid)   → 지정된 1개 RID만 출력.
        drop_door_mode: None | "position" | "orientation" | "all".
            None          → 전체 출력: ``<DOOR> <X:cx> <Y:cy> <SEP_DOOR> <X:w> <Y:h> <END_DOOR>``
            "position"    → 위치(cx,cy) 삭제, 크기(w,h) 유지: ``<DOOR> <SEP_DOOR> <X:w> <Y:h> <END_DOOR>``
            "orientation" → 크기(w,h) 삭제, 위치(cx,cy) 유지: ``<DOOR> <X:cx> <Y:cy> <SEP_DOOR> <END_DOOR>``
            "all"         → 위치+크기 모두 삭제: ``<DOOR> <SEP_DOOR> <END_DOOR>``
        vocab: Vocab 객체.
        drop_edge_wrapper: True면 <EDGE>/<END_EDGE> 래퍼 토큰 생략 (OUTPUT 전용).

    Returns:
        토큰 ID 리스트.
        형식: ``<EDGE> [RID tokens] <DOOR> <X:cx> <Y:cy> <SEP_DOOR> <X:w> <Y:h> <END_DOOR> <END_EDGE>``
    """
    tokens = [] if drop_edge_wrapper else [vocab.get("<EDGE>")]

    # --- RID 쌍 ---
    if drop_pair_mode == "both":
        pass  # 둘 다 생략
    elif isinstance(drop_pair_mode, tuple) and drop_pair_mode[0] == "one":
        # ("one", kept_rid) — 지정된 1개 RID만 출력
        tokens.append(vocab.get(f"<RID:{drop_pair_mode[1]}>"))
    else:
        for rid in edge["pair"]:
            tokens.append(vocab.get(f"<RID:{rid}>"))

    # --- 문 정보 ---
    # 형식: <DOOR> <X:cx> <Y:cy> <SEP_DOOR> <X:w> <Y:h> <END_DOOR>
    # <SEP_DOOR>은 항상 출력 — 위치(cx,cy)와 크기(w,h)의 경계를 명시
    # drop_door_mode에 따라 <SEP_DOOR> 앞/뒤 좌표를 선택적으로 생략
    if not edge["door"]:
        tokens.append(vocab.get("<NO_DOOR>"))
    else:
        for door in edge["door"]:
            tokens.append(vocab.get("<DOOR>"))
            cx = round(door["x"])
            cy = round(door["y"])
            w = round(door["w"])
            h = round(door["h"])

            if drop_door_mode == "all":
                # 위치+크기 모두 삭제 → <DOOR> <SEP_DOOR> <END_DOOR>
                pass
            elif drop_door_mode == "position":
                # 위치(cx,cy) 삭제, 크기(w,h) 유지 → <DOOR> <SEP_DOOR> <X:w> <Y:h> <END_DOOR>
                tokens.append(vocab.get("<SEP_DOOR>"))
                tokens.append(vocab.get(f"<X:{w}>"))
                tokens.append(vocab.get(f"<Y:{h}>"))
                tokens.append(vocab.get("<END_DOOR>"))
                continue  # SEP_DOOR 이미 삽입했으므로 아래 공통 처리 건너뜀
            elif drop_door_mode == "orientation":
                # 크기(w,h) 삭제, 위치(cx,cy) 유지 → <DOOR> <X:cx> <Y:cy> <SEP_DOOR> <END_DOOR>
                tokens.append(vocab.get(f"<X:{cx}>"))
                tokens.append(vocab.get(f"<Y:{cy}>"))
            else:
                # 정상 출력 → <DOOR> <X:cx> <Y:cy> <SEP_DOOR> <X:w> <Y:h> <END_DOOR>
                tokens.append(vocab.get(f"<X:{cx}>"))
                tokens.append(vocab.get(f"<Y:{cy}>"))

            tokens.append(vocab.get("<SEP_DOOR>"))   # 위치/크기 경계 구분자
            if drop_door_mode not in ("orientation", "all"):
                tokens.append(vocab.get(f"<X:{w}>"))
                tokens.append(vocab.get(f"<Y:{h}>"))
            tokens.append(vocab.get("<END_DOOR>"))

    if not drop_edge_wrapper:
        tokens.append(vocab.get("<END_EDGE>"))
    return tokens


def tokenize_spatial_item(spatial: dict, vocab: Vocab) -> list[int]:
    """단일 공간 관계 항목을 토큰 ID 시퀀스로 변환한다.

    형식: ``<SP> <RID:a> <RID:b> <REL:dir> <END_SP>``

    Args:
        spatial: {"rid_a": int, "rid_b": int, "direction": str} 딕셔너리.
        vocab: Vocab 객체.

    Returns:
        토큰 ID 리스트.
    """
    return [
        vocab.get("<SP>"),
        vocab.get(f"<RID:{spatial['rid_a']}>"),
        vocab.get(f"<RID:{spatial['rid_b']}>"),
        vocab.get(f"<REL:{spatial['direction']}>"),
        vocab.get("<END_SP>"),
    ]


# ---------------------------------------------------------------------------
# ROOM_SUMMARY 블록 토크나이징
# ---------------------------------------------------------------------------

def build_room_summary_tokens(
    sample: dict,
    drop_state: "DropState",
    vocab: Vocab,
) -> list[int]:
    """ROOM_SUMMARY 블록을 토큰 ID 시퀀스로 변환한다 (INPUT 전용).

    outline을 제외한 방 타입별 개수를 집계하여 요약 블록을 구성한다.
    drop_state에 따라 <TOTAL>N 쌍 또는 개별 <TYPE:t><COUNT>M 쌍이 생략된다.
    숫자(N, M)는 LLM 기본 숫자 토큰을 사용한다 (vocab.number_to_ids).

    Args:
        sample: row-oriented 평면도 딕셔너리.
        drop_state: 삭제 증강 상태 기록.
            - drop_room_summary_total: True이면 <TOTAL> + 숫자 토큰 쌍 생략.
            - drop_room_summary_types: 해당 타입의 <TYPE:t> <COUNT> + 숫자 토큰 쌍 생략.
        vocab: Vocab 객체. number_to_ids가 populated되어 있어야 한다.

    Returns:
        토큰 ID 리스트.
        형식: ``<ROOM_SUMMARY> [<TOTAL> N] [<TYPE:t> <COUNT> M ...] <END_ROOM_SUMMARY>``
            (N, M은 LLM 기본 숫자 토큰 ID)

    Raises:
        KeyError: vocab.number_to_ids에 해당 숫자가 없을 때
            (tokenizer 없이 load_vocab()을 호출한 경우).
    """
    tokens: list[int] = [vocab.get("<ROOM_SUMMARY>")]

    # outline을 제외한 방만 집계
    non_outline_rooms = [r for r in sample["rooms"] if r["type"] != "outline"]
    total_count = len(non_outline_rooms)

    def _append_number(n: int) -> None:
        """LLM 기본 숫자 토큰 ID를 tokens에 추가한다 (멀티토큰 숫자 대응)."""
        if n not in vocab.number_to_ids:
            raise KeyError(
                f"숫자 {n}에 해당하는 LLM 토큰이 없습니다. "
                "tokenizer_dir을 지정하여 load_vocab()을 호출하세요."
            )
        tokens.extend(vocab.number_to_ids[n])

    # <TOTAL> + 숫자 쌍 (drop_room_summary_total이면 생략)
    if not drop_state.drop_room_summary_total:
        tokens.append(vocab.get("<TOTAL>"))
        _append_number(total_count)

    # 방 타입별 개수 집계 (타입 이름 알파벳 정렬로 순서 고정)
    type_counts: dict[str, int] = {}
    for room in non_outline_rooms:
        room_type = room["type"]
        type_counts[room_type] = type_counts.get(room_type, 0) + 1

    for room_type in sorted(type_counts):
        # drop_room_summary_types에 포함된 타입은 쌍 전체 생략
        if room_type in drop_state.drop_room_summary_types:
            continue
        tokens.append(vocab.get(f"<TYPE:{room_type}>"))
        tokens.append(vocab.get("<COUNT>"))
        _append_number(type_counts[room_type])

    tokens.append(vocab.get("<END_ROOM_SUMMARY>"))
    return tokens


# ---------------------------------------------------------------------------
# canonical 순서 정렬 헬퍼
# ---------------------------------------------------------------------------

def _room_centroid(room: dict) -> tuple[float, float]:
    """방 좌표에서 centroid (cx, cy)를 계산한다.

    Args:
        room: {"coords": list[int]} 딕셔너리.

    Returns:
        (cx, cy) 튜플.
    """
    coords = room["coords"]
    xs = coords[0::2]
    ys = coords[1::2]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def canonical_room_order(rooms: list[dict]) -> list[dict]:
    """outline을 제외한 방 목록을 raster scan 순서로 정렬한다.

    정렬 기준: centroid y 오름차순 → centroid x 오름차순.

    Args:
        rooms: row-oriented 방 딕셔너리 리스트.

    Returns:
        outline 제외 + raster scan 정렬된 방 리스트.
    """
    non_outline = [r for r in rooms if r["type"] != "outline"]
    return sorted(non_outline, key=lambda r: (_room_centroid(r)[1], _room_centroid(r)[0]))


def canonical_edge_order(edges: list[dict]) -> list[dict]:
    """엣지 목록을 RID 쌍 오름차순으로 정렬한다.

    정렬 기준: (min(rid_a, rid_b), max(rid_a, rid_b)).

    Args:
        edges: row-oriented 엣지 딕셔너리 리스트.

    Returns:
        정렬된 엣지 리스트.
    """
    return sorted(edges, key=lambda e: (min(e["pair"]), max(e["pair"])))


# ---------------------------------------------------------------------------
# 메인 토크나이징 함수
# ---------------------------------------------------------------------------

def build_condition_tokens(
    sample: dict,
    drop_state: "DropState",
    vocab: Vocab,
) -> list[int]:
    """조건(입력) 토큰 ID 시퀀스를 생성한다.

    DropState를 참조하여 증강된 입력을 구성한다.

    Args:
        sample: row-oriented 평면도 딕셔너리 (변형 증강 이미 적용된 상태).
        drop_state: 삭제 증강 상태 기록.
        vocab: Vocab 객체.

    Returns:
        조건 토큰 ID 리스트.
        구조: [BOS] <INPUT> [room_summary] [front_door] [outline+rooms] [edges] [spatial] <END_INPUT>
    """
    tokens: list[int] = []

    if vocab.bos_token_id is not None:
        tokens.append(vocab.bos_token_id)
    tokens.append(vocab.get("<INPUT>"))

    # (0) ROOM_SUMMARY 블록 (INPUT 전용 조건 정보 — 출력에는 미포함)
    tokens.extend(build_room_summary_tokens(sample, drop_state, vocab))

    # (1) 현관문
    tokens.extend(
        tokenize_front_door(
            sample["front_door"],
            drop_front_door=drop_state.drop_front_door,
            drop_front_door_coords=drop_state.drop_front_door_coords,
            vocab=vocab,
        )
    )

    # (2) 방 블록 (outline 포함, ShuffleRoomOrder가 이미 sample["rooms"] 순서를 바꿔놓음)
    # outline은 항상 첫 번째(shuffle_room_order 보장), drop_block/drop_coords 적용 가능
    for room in sample["rooms"]:
        rid = room["rid"]
        if rid in drop_state.drop_block:
            continue
        tokens.extend(
            tokenize_room_block(
                room,
                drop_type=(rid in drop_state.drop_type),
                drop_coords=(rid in drop_state.drop_coords),
                vocab=vocab,
                noisy_coords=drop_state.noise_room_coords.get(rid),
            )
        )

    # (3) 엣지 블록 (<EDGES>/<END_EDGES> 래퍼 없이 개별 <EDGE> 블록만 출력)
    for idx, edge in enumerate(sample["edges"]):
        if idx in drop_state.drop_edge:
            continue

        drop_pair_mode = drop_state.drop_pair.get(idx)

        tokens.extend(
            tokenize_edge_block(
                edge,
                drop_pair_mode=drop_pair_mode,
                drop_door_mode=drop_state.drop_door.get(idx),
                vocab=vocab,
            )
        )

    # (4) Spatial 블록 (<SPATIAL>/<END_SPATIAL> 래퍼 없이 개별 <SP> 항목만 출력)
    for sp_idx, sp in enumerate(sample["spatial"]):
        if sp_idx in drop_state.drop_spatial:
            continue
        tokens.extend(tokenize_spatial_item(sp, vocab))

    tokens.append(vocab.get("<END_INPUT>"))
    return tokens


def build_output_tokens(sample: dict, vocab: Vocab) -> list[int]:
    """정답(출력) 토큰 ID 시퀀스를 생성한다.

    삭제 증강과 무관하게 항상 full information (outline + 모든 방 + 전체 엣지).

    Args:
        sample: row-oriented 평면도 딕셔너리 (변형 증강 이미 적용된 상태).
        vocab: Vocab 객체.

    Returns:
        출력 토큰 ID 리스트.
        구조: <OUTPUT> [front_door] [outline] [rooms raster scan] [edges RID pair 오름차순] <END_OUTPUT> [EOS]
    """
    tokens: list[int] = [vocab.get("<OUTPUT>")]

    # front_door: 항상 full information 출력 (drop 증강 미적용)
    tokens.extend(
        tokenize_front_door(
            sample["front_door"],
            drop_front_door=False,
            drop_front_door_coords=False,
            vocab=vocab,
        )
    )

    # outline: 항상 첫 번째로 full information 출력 (RID 생략)
    outline_room = next((r for r in sample["rooms"] if r["type"] == "outline"), None)
    if outline_room is not None:
        tokens.extend(
            tokenize_room_block(outline_room, drop_type=False, drop_coords=False, vocab=vocab, drop_rid=True)
        )

    # ROOMS: outline 제외, raster scan canonical 순서, 항상 full information (RID 생략)
    for room in canonical_room_order(sample["rooms"]):
        tokens.extend(
            tokenize_room_block(room, drop_type=False, drop_coords=False, vocab=vocab, drop_rid=True)
        )

    # EDGES: RID 쌍 오름차순 canonical 순서, 항상 full information
    # OUTPUT에서는 <EDGE>/<END_EDGE> 래퍼와 RID 쌍 모두 생략 → <DOOR>...<END_DOOR>만 출력
    for edge in canonical_edge_order(sample["edges"]):
        tokens.extend(
            tokenize_edge_block(
                edge,
                drop_pair_mode="both",        # RID 쌍 생략
                drop_door_mode=None,
                vocab=vocab,
                drop_edge_wrapper=True,       # <EDGE>/<END_EDGE> 생략
            )
        )

    tokens.append(vocab.get("<END_OUTPUT>"))

    return tokens
