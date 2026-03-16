"""토큰 ID 시퀀스 → 가독성 높은 문자열 변환 모듈.

토크나이징/증강 결과 검증을 위해 token ID 시퀀스를 구조에 맞는
들여쓰기와 줄바꿈이 적용된 문자열로 포맷팅한다.

Arrow raw 데이터도 사람이 읽기 좋은 형태로 포맷팅하여
origin / input / output 3섹션을 하나의 파일에 출력할 수 있도록 한다.
"""

from __future__ import annotations

from src.training.augmentation.tokenizer import Vocab

# ---------------------------------------------------------------------------
# 포맷팅 상수
# ---------------------------------------------------------------------------

_INDENT = "  "  # 들여쓰기 단위 (2칸)

# 새 줄에서 시작하는 토큰 집합
# <ROOM>, <EDGE>, <SP>, <DOOR>, <FRONT_DOOR>는 인라인 출력이므로 제외
_NEWLINE_BEFORE: frozenset[str] = frozenset(
    {
        "<INPUT>", "<END_INPUT>",
        "<OUTPUT>", "<END_OUTPUT>",
        "<ROOM>", "<EDGE>", "<SP>", "<FRONT_DOOR>", "<DOOR>",
    }
)

# 이 토큰 출력 후 들여쓰기 증가
_INDENT_AFTER: frozenset[str] = frozenset(
    {
        "<INPUT>", "<OUTPUT>",
    }
)

# 이 토큰 출력 전 들여쓰기 감소
_DEDENT_BEFORE: frozenset[str] = frozenset(
    {
        "<END_INPUT>", "<END_OUTPUT>",
    }
)

# 인라인 블록: 시작 토큰부터 종료 토큰까지 한 줄로 출력
# (ROOM, EDGE, SP, DOOR, FRONT_DOOR 모두 해당)
_INLINE_OPEN: frozenset[str] = frozenset(
    {"<ROOM>", "<EDGE>", "<SP>", "<DOOR>", "<FRONT_DOOR>"}
)
# 각 인라인 블록의 종료 토큰 매핑
_INLINE_CLOSE_MAP: dict[str, str] = {
    "<ROOM>":       "<END_ROOM>",
    "<EDGE>":       "<END_EDGE>",
    "<SP>":         "<END_SP>",
    "<DOOR>":       "<END_DOOR>",
    "<FRONT_DOOR>": "<END_DOOR>",
}


# ---------------------------------------------------------------------------
# 토큰 시퀀스 포맷터
# ---------------------------------------------------------------------------

def decode_tokens(token_ids: list[int], vocab: Vocab) -> str:
    """토큰 ID 시퀀스를 들여쓰기/줄바꿈이 적용된 문자열로 변환한다.

    BOS/EOS는 "[BOS]" / "[EOS]" 레이블로 표시한다.

    Args:
        token_ids: 토큰 ID 리스트.
        vocab: Vocab 객체 (id_to_token 매핑 포함).

    Returns:
        구조화된 가독성 높은 토큰 문자열.
    """
    lines: list[str] = []
    current_line: list[str] = []
    indent_level = 0
    inline_close: str | None = None  # 현재 인라인 블록의 종료 토큰 (None이면 블록 밖)

    def flush_line() -> None:
        """현재 줄 버퍼를 lines에 추가한다."""
        if current_line:
            lines.append(_INDENT * indent_level + " ".join(current_line))
            current_line.clear()

    for tid in token_ids:
        # BOS / EOS 특수 처리
        if vocab.bos_token_id is not None and tid == vocab.bos_token_id:
            flush_line()
            lines.append("<BOS>")
            continue
        if vocab.eos_token_id is not None and tid == vocab.eos_token_id:
            flush_line()
            lines.append("<EOS>")
            continue

        token = vocab.id_to_token.get(tid, f"<UNK:{tid}>")

        # 인라인 블록 내부: 같은 줄에 이어붙이고 종료 토큰에서만 블록 탈출
        if inline_close is not None:
            current_line.append(token)
            if token == inline_close:
                inline_close = None
                flush_line()
            continue

        # 새 줄 시작 토큰
        if token in _NEWLINE_BEFORE:
            flush_line()

        # 들여쓰기 감소 (출력 전)
        if token in _DEDENT_BEFORE:
            indent_level = max(0, indent_level - 1)

        current_line.append(token)

        # 인라인 블록 진입: 대응하는 종료 토큰 기록
        if token in _INLINE_OPEN:
            inline_close = _INLINE_CLOSE_MAP[token]
            continue

        # 들여쓰기 증가 (출력 후)
        if token in _INDENT_AFTER:
            flush_line()
            indent_level += 1

    flush_line()
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# origin(raw Arrow 데이터) 포맷터
# ---------------------------------------------------------------------------

def format_origin(sample: dict) -> str:
    """row-oriented 평면도 딕셔너리를 사람이 읽기 좋은 문자열로 포맷팅한다.

    Arrow에서 읽어온 raw 구조화 데이터를 그대로 보여준다.

    Args:
        sample: to_row_oriented()로 변환된 row-oriented 딕셔너리.

    Returns:
        포맷팅된 문자열.

    Example::

        rooms:
          rid=0  type=outline       coords=[80,30, 80,220, 210,220, 210,30]
          rid=1  type=livingroom    coords=[100,200, 100,300, 200,300, 200,200]
        edges:
          [1,2]  door: x=200 y=250 H(w=2,h=10)
          [1,3]  no_door
        front_door: x=128 y=32 H(w=8,h=2)
        spatial:
          1→2  right
          1→3  below
    """
    lines: list[str] = []

    # --- rooms ---
    lines.append("rooms:")
    for room in sample["rooms"]:
        coords = room["coords"]
        # flat 좌표를 (x,y) 쌍으로 묶어서 출력
        pairs = [f"({coords[i]},{coords[i+1]})" for i in range(0, len(coords), 2)]
        coords_str = " ".join(pairs)
        lines.append(
            f"  rid={room['rid']:<3}  type={room['type']:<15}  coords=[{coords_str}]"
        )

    # --- edges ---
    lines.append("edges:")
    for edge in sample["edges"]:
        pair_str = f"[{edge['pair'][0]},{edge['pair'][1]}]"
        if not edge["door"]:
            lines.append(f"  {pair_str}  no_door")
        else:
            door_parts = []
            for d in edge["door"]:
                orient = "H" if d["w"] >= d["h"] else "V"
                door_parts.append(
                    f"x={d['x']:.0f} y={d['y']:.0f} {orient}(w={d['w']:.0f},h={d['h']:.0f})"
                )
            lines.append(f"  {pair_str}  door: {' | '.join(door_parts)}")

    # --- front_door ---
    fd = sample["front_door"]
    if fd is None:
        lines.append("front_door: None")
    else:
        orient = "H" if fd["w"] >= fd["h"] else "V"
        lines.append(
            f"front_door: x={fd['x']:.0f} y={fd['y']:.0f}"
            f" {orient}(w={fd['w']:.0f},h={fd['h']:.0f})"
        )

    # --- spatial ---
    lines.append("spatial:")
    for sp in sample["spatial"]:
        lines.append(f"  {sp['rid_a']}→{sp['rid_b']}  {sp['direction']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 샘플 리포트 생성
# ---------------------------------------------------------------------------

_SEP_THICK = "═" * 60
_SEP_THIN  = "─" * 60
_SEP_MID   = "·" * 60


def format_sample_report(
    plan_id: str,
    sample_idx: int,
    augmentation_summary: str,
    origin_str: str,
    condition_str: str,
    output_str: str,
) -> str:
    """한 샘플의 origin / input / output 3섹션 리포트 문자열을 생성한다.

    Args:
        plan_id: 평면도 고유 ID.
        sample_idx: 샘플 순번 (1-based).
        augmentation_summary: 적용된 증강 요약 문자열 (pipeline.augmented_summary() 반환값).
        origin_str: format_origin() 반환값.
        condition_str: decode_tokens()로 변환한 입력 토큰 문자열.
        output_str: decode_tokens()로 변환한 출력 토큰 문자열.

    Returns:
        포맷팅된 리포트 문자열.
    """
    lines: list[str] = [
        _SEP_THICK,
        f"SAMPLE {sample_idx:03d}  |  plan_id: {plan_id}",
        f"Applied: {augmentation_summary}",
        _SEP_THICK,
        "",
        "[ORIGIN]",
        origin_str,
        "",
        _SEP_THIN,
        "[INPUT]",
        condition_str,
        "",
        _SEP_THIN,
        "[OUTPUT]",
        output_str,
        "",
    ]
    return "\n".join(lines)
