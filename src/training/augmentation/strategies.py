"""데이터 증강 전략 모듈.

변형 기반 증강(Shuffle 계열)과 삭제 기반 증강(Drop 계열) 상태 계산을 담당한다.

증강 적용 순서 (pipeline.py에서 호출 시):
    1. 변형 기반: ShuffleRID → ShuffleVertexOrder → ShuffleRoomOrder
                  → ShuffleEdgeOrder → ShuffleSpatialOrder → ReverseSpatialRelation
    2. 삭제 기반: compute_drop_state() — sample에 기록하지 않고 DropState에만 상태 저장

설계 원칙:
    - 변형 증강: sample 딕셔너리의 데이터 자체를 수정 (in-place on deep copy).
    - 삭제 증강: sample을 수정하지 않고 DropState에 "무엇을 삭제할지" 상태만 기록.
      실제 삭제는 tokenizer.py의 build_condition_tokens()가 수행.
    - DropEdgePair "one" 모드: drop_pair에 ("one", kept_rid) tuple 저장.
      sample["pair"]는 절대 수정하지 않아 OUTPUT 토큰에 영향을 주지 않는다.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 공간 관계 방향 역전 매핑
# ---------------------------------------------------------------------------

_REVERSE_DIRECTION: dict[str, str] = {
    "right":        "left",
    "left":         "right",
    "above":        "below",
    "below":        "above",
    "right-above":  "left-below",
    "left-below":   "right-above",
    "right-below":  "left-above",
    "left-above":   "right-below",
}

# 수평 뒤집기(x축 반전)시 방향 매핑
_FLIP_H_DIRECTION: dict[str, str] = {
    "right":        "left",
    "left":         "right",
    "right-above":  "left-above",
    "left-above":   "right-above",
    "right-below":  "left-below",
    "left-below":   "right-below",
    "above":        "above",
    "below":        "below",
}

# 수직 뒤집기(y축 반전)시 방향 매핑
_FLIP_V_DIRECTION: dict[str, str] = {
    "above":        "below",
    "below":        "above",
    "right-above":  "right-below",
    "right-below":  "right-above",
    "left-above":   "left-below",
    "left-below":   "left-above",
    "right":        "right",
    "left":         "left",
}

# 256×256 그리드 최대 좌표
_MAX_COORD: int = 255


# ---------------------------------------------------------------------------
# DropState 데이터 클래스
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 기하학적 변형 증강 헬퍼 (내부 유틸리티)
# ---------------------------------------------------------------------------

def _compute_outline_bbox(sample: dict) -> tuple[int, int, int, int]:
    """outline 방 좌표의 축-정렬 bbox를 반환한다.

    Args:
        sample: row-oriented 평면도 딕셔너리.

    Returns:
        (x_min, y_min, x_max, y_max) 정수 튜플.
    """
    coords = next(r["coords"] for r in sample["rooms"] if r["type"] == "outline")
    xs = coords[0::2]
    ys = coords[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def _apply_coord_transform(
    sample: dict,
    tx,
    ty,
    scale_wh_x: float = 1.0,
    scale_wh_y: float = 1.0,
) -> None:
    """sample의 모든 좌표에 변환 함수를 in-place 적용한다.

    방 꼭지점은 정수로 반올림, 문 좌표(float)는 그대로 유지하여
    tokenizer에서 round() 처리에 맡긴다.

    Args:
        sample: row-oriented 평면도 딕셔너리.
        tx: x 좌표 변환 함수 (number → number).
        ty: y 좌표 변환 함수 (number → number).
        scale_wh_x: 문 너비(w) 스케일 배율.
        scale_wh_y: 문 높이(h) 스케일 배율.
    """
    # 방 꼭지점 (정수 반올림)
    for room in sample["rooms"]:
        c = room["coords"]
        room["coords"] = [
            round(tx(c[i])) if i % 2 == 0 else round(ty(c[i]))
            for i in range(len(c))
        ]

    # 현관문 center 좌표 (float 유지)
    # Mod Record: scale/zoom의 안전 범위가 outline bbox(정수) 기준이라 문 좌표(float)가
    # 미세하게 _MAX_COORD를 초과할 수 있음 → tokenizer의 round() 후 256이 되어 KeyError 발생.
    # 변환 후 [0, _MAX_COORD] 클램프로 방어.
    if sample["front_door"] is not None:
        fd = sample["front_door"]
        fd["x"] = max(0.0, min(tx(fd["x"]), float(_MAX_COORD)))
        fd["y"] = max(0.0, min(ty(fd["y"]), float(_MAX_COORD)))
        fd["w"] = fd["w"] * scale_wh_x
        fd["h"] = fd["h"] * scale_wh_y

    # 엣지 문 center 좌표 (float 유지, 동일 클램프 적용)
    for edge in sample["edges"]:
        for door in edge["door"]:
            door["x"] = max(0.0, min(tx(door["x"]), float(_MAX_COORD)))
            door["y"] = max(0.0, min(ty(door["y"]), float(_MAX_COORD)))
            door["w"] = door["w"] * scale_wh_x
            door["h"] = door["h"] * scale_wh_y


@dataclass
class DropState:
    """삭제 기반 증강 상태 컨테이너.

    tokenizer.py의 build_condition_tokens()가 이 상태를 참조하여
    삭제 증강을 반영한 입력 시퀀스를 생성한다.

    Attributes:
        drop_block: DropBlock 대상 RID 집합 (방 블록 전체 삭제).
        drop_type: DropType 대상 RID 집합 (방 종류 토큰 삭제).
        drop_coords: DropCoords 대상 RID 집합 (방 좌표 블록 삭제).
        drop_edge: DropEdge 대상 엣지 인덱스 집합 (엣지 전체 삭제).
        drop_pair: DropEdgePair 대상 {엣지 인덱스: "both" | ("one", kept_rid)}.
        drop_door: DropEdgeDoor 대상 {엣지 인덱스: "position" | "orientation" | "all"}.
        drop_spatial: DropSpatial 대상 spatial 인덱스 집합.
        drop_front_door: DropFrontDoor 적용 여부.
    """

    noise_room_coords: dict[int, list[int]] = field(default_factory=dict)  # rid → 노이즈 좌표
    drop_block: set[int] = field(default_factory=set)
    drop_type: set[int] = field(default_factory=set)
    drop_coords: set[int] = field(default_factory=set)
    drop_edge: set[int] = field(default_factory=set)
    drop_pair: dict[int, str | tuple[str, int]] = field(default_factory=dict)   # idx → "both" | ("one", kept_rid)
    drop_door: dict[int, str] = field(default_factory=dict)   # idx → "position"|"orientation"|"all"
    drop_spatial: set[int] = field(default_factory=set)
    drop_front_door: bool = False           # DropBlock: <FRONT_DOOR>...<END_DOOR> 전체 생략
    drop_front_door_coords: bool = False    # DropCoords: 좌표 생략 → <FRONT_DOOR> <END_DOOR>
    drop_room_summary_total: bool = False           # DropRoomSummaryTotal: <TOTAL>N 쌍 삭제
    drop_room_summary_types: set[str] = field(default_factory=set)  # DropRoomSummaryType: 삭제할 방 타입 집합

    def summary(self) -> str:
        """적용된 삭제 증강 요약 문자열을 반환한다 (검증용).

        Returns:
            증강 내역 문자열. 아무것도 없으면 "없음".
        """
        parts: list[str] = []
        if self.noise_room_coords:
            parts.append(f"CoordNoise(rids={sorted(self.noise_room_coords)})")
        if self.drop_block:
            parts.append(f"DropBlock(rids={sorted(self.drop_block)})")
        if self.drop_type:
            parts.append(f"DropType(rids={sorted(self.drop_type)})")
        if self.drop_coords:
            parts.append(f"DropCoords(rids={sorted(self.drop_coords)})")
        if self.drop_edge:
            parts.append(f"DropEdge(idxs={sorted(self.drop_edge)})")
        if self.drop_pair:
            parts.append(f"DropEdgePair({self.drop_pair})")
        if self.drop_door:
            parts.append(f"DropEdgeDoor({self.drop_door})")
        if self.drop_spatial:
            parts.append(f"DropSpatial(idxs={sorted(self.drop_spatial)})")
        if self.drop_front_door:
            parts.append("DropFrontDoor")
        if self.drop_front_door_coords:
            parts.append("DropFrontDoorCoords")
        if self.drop_room_summary_total:
            parts.append("DropRoomSummaryTotal")
        if self.drop_room_summary_types:
            parts.append(f"DropRoomSummaryType(types={sorted(self.drop_room_summary_types)})")
        return ", ".join(parts) if parts else "없음"


# ---------------------------------------------------------------------------
# 변형 기반 증강 함수
# ---------------------------------------------------------------------------

def shuffle_rid(sample: dict, rng: random.Random) -> dict:
    """방 ID(RID)를 랜덤하게 재배정한다.

    0~15 중 방 개수만큼 비복원 추출하여 새 RID를 부여한다.
    rooms, edges, spatial 전체에서 old→new 매핑을 일관되게 적용한다.
    outline(type="outline")은 rid=0으로 고정하고 재배정 대상에서 제외한다.

    Args:
        sample: row-oriented 평면도 딕셔너리 (deep copy 권장).
        rng: 재현성을 위한 Random 인스턴스.

    Returns:
        RID가 재배정된 딕셔너리 (in-place 수정 후 반환).
    """
    # outline을 제외한 방 목록만 재배정
    non_outline_rooms = [r for r in sample["rooms"] if r["type"] != "outline"]
    old_rids = [r["rid"] for r in non_outline_rooms]

    # 0~15 중 outline(0)을 제외한 번호풀에서 비복원 추출
    available = list(range(1, 16))  # outline은 항상 rid=0 고정
    new_rids = rng.sample(available, len(old_rids))
    rid_map: dict[int, int] = dict(zip(old_rids, new_rids))

    # rooms 재배정
    for room in sample["rooms"]:
        if room["type"] != "outline":
            room["rid"] = rid_map[room["rid"]]

    # edges 재배정
    for edge in sample["edges"]:
        edge["pair"] = [rid_map.get(r, r) for r in edge["pair"]]

    # spatial 재배정
    for sp in sample["spatial"]:
        sp["rid_a"] = rid_map.get(sp["rid_a"], sp["rid_a"])
        sp["rid_b"] = rid_map.get(sp["rid_b"], sp["rid_b"])

    return sample


def shuffle_vertex_order(sample: dict, rng: random.Random) -> dict:
    """각 방 꼭지점 리스트의 시작점을 랜덤하게 회전한다.

    방향(시계/반시계)은 유지하고 시작 꼭지점만 변경한다.
    outline과 일반 방 모두 적용한다.

    Args:
        sample: row-oriented 평면도 딕셔너리.
        rng: Random 인스턴스.

    Returns:
        꼭지점 순서가 변경된 딕셔너리 (in-place 수정 후 반환).
    """
    for room in sample["rooms"]:
        coords = room["coords"]
        n_vertices = len(coords) // 2
        if n_vertices < 2:
            continue
        # 시작 꼭지점 인덱스를 랜덤 선택 후 회전
        start = rng.randint(0, n_vertices - 1)
        # flat coords를 vertex 쌍 리스트로 변환 후 회전 후 다시 flat으로
        pairs = [(coords[i * 2], coords[i * 2 + 1]) for i in range(n_vertices)]
        rotated = pairs[start:] + pairs[:start]
        room["coords"] = [v for pair in rotated for v in pair]

    return sample


def shuffle_room_order(sample: dict, rng: random.Random) -> dict:
    """입력에서 방 나열 순서를 랜덤 셔플한다.

    outline은 항상 첫 번째로 유지하고, 나머지 방들만 셔플한다.
    출력 canonical 순서(raster scan)는 tokenizer.py에서 별도 처리한다.

    Args:
        sample: row-oriented 평면도 딕셔너리.
        rng: Random 인스턴스.

    Returns:
        방 순서가 셔플된 딕셔너리 (in-place 수정 후 반환).
    """
    outline = [r for r in sample["rooms"] if r["type"] == "outline"]
    non_outline = [r for r in sample["rooms"] if r["type"] != "outline"]
    rng.shuffle(non_outline)
    sample["rooms"] = outline + non_outline
    return sample


def shuffle_edge_order(sample: dict, rng: random.Random) -> dict:
    """입력에서 엣지 나열 순서를 랜덤 셔플한다.

    출력 canonical 순서(RID 쌍 오름차순)는 tokenizer.py에서 별도 처리한다.

    Args:
        sample: row-oriented 평면도 딕셔너리.
        rng: Random 인스턴스.

    Returns:
        엣지 순서가 셔플된 딕셔너리 (in-place 수정 후 반환).
    """
    rng.shuffle(sample["edges"])
    return sample


def shuffle_spatial_order(sample: dict, rng: random.Random) -> dict:
    """입력에서 spatial 관계 나열 순서를 랜덤 셔플한다.

    출력에는 spatial이 포함되지 않으므로 입력 토큰 구성에만 영향을 준다.

    Args:
        sample: row-oriented 평면도 딕셔너리.
        rng: Random 인스턴스.

    Returns:
        spatial 순서가 셔플된 딕셔너리 (in-place 수정 후 반환).
    """
    rng.shuffle(sample["spatial"])
    return sample


def reverse_spatial_relation(sample: dict, rng: random.Random) -> dict:
    """각 spatial 관계를 50% 확률로 방향 역전한다.

    예: [rid_a=1, rid_b=2, "right"] → [rid_a=2, rid_b=1, "left"]

    출력에는 spatial이 포함되지 않으므로 입력 토큰 구성에만 영향을 준다.

    Args:
        sample: row-oriented 평면도 딕셔너리.
        rng: Random 인스턴스.

    Returns:
        일부 spatial 관계가 역전된 딕셔너리 (in-place 수정 후 반환).
    """
    for sp in sample["spatial"]:
        if rng.random() < 0.5:
            sp["rid_a"], sp["rid_b"] = sp["rid_b"], sp["rid_a"]
            sp["direction"] = _REVERSE_DIRECTION.get(sp["direction"], sp["direction"])
    return sample


def translate(sample: dict, rng: random.Random) -> dict:
    """평면도 전체를 x/y 방향으로 랜덤 평행이동한다.

    outline bbox 기준으로 이동 가능 범위를 계산하여 256×256 경계를 보장한다.

    Args:
        sample: row-oriented 평면도 딕셔너리.
        rng: Random 인스턴스.

    Returns:
        평행이동된 딕셔너리 (in-place 수정 후 반환).
    """
    x1, y1, x2, y2 = _compute_outline_bbox(sample)
    dx = rng.randint(-x1, _MAX_COORD - x2)
    dy = rng.randint(-y1, _MAX_COORD - y2)
    if dx == 0 and dy == 0:
        return sample
    _apply_coord_transform(sample, lambda x: x + dx, lambda y: y + dy)
    return sample


def flip(sample: dict, rng: random.Random) -> dict:
    """평면도를 수평(H), 수직(V), 또는 양방향(HV)으로 뒤집는다.

    뒤집기 후 256×256 경계 내에서 항상 유효하다.
    공간 관계(spatial) 방향도 뒤집기에 맞게 갱신한다.

    Args:
        sample: row-oriented 평면도 딕셔너리.
        rng: Random 인스턴스.

    Returns:
        뒤집어진 딕셔너리 (in-place 수정 후 반환).
    """
    mode = rng.choice(("H", "V", "HV"))
    do_h = "H" in mode
    do_v = "V" in mode

    tx = (lambda x: _MAX_COORD - x) if do_h else (lambda x: x)
    ty = (lambda y: _MAX_COORD - y) if do_v else (lambda y: y)
    _apply_coord_transform(sample, tx, ty)

    # spatial 방향 갱신
    for sp in sample["spatial"]:
        d = sp["direction"]
        if do_h:
            d = _FLIP_H_DIRECTION.get(d, d)
        if do_v:
            d = _FLIP_V_DIRECTION.get(d, d)
        sp["direction"] = d

    return sample


def scale_aspect(
    sample: dict,
    rng: random.Random,
    scale_min: float = 0.7,
    scale_max: float = 1.4,
) -> dict:
    """평면도의 종횡비를 변경한다 (x/y 독립 스케일링).

    outline bbox의 좌상단을 기준점으로 스케일하여 256×256 경계를 보장한다.
    문의 너비/높이도 동일 배율로 조정한다.

    Args:
        sample: row-oriented 평면도 딕셔너리.
        rng: Random 인스턴스.
        scale_min: 스케일 최솟값.
        scale_max: 스케일 상한값 (실제 max는 256 경계 기반 자동 계산값과 min으로 결정).

    Returns:
        종횡비가 변경된 딕셔너리 (in-place 수정 후 반환).
    """
    x1, y1, x2, y2 = _compute_outline_bbox(sample)
    w, h = x2 - x1, y2 - y1

    # 최대 스케일: bbox가 256×256을 넘지 않도록
    max_sx = (_MAX_COORD - x1) / w if w > 0 else 1.0
    max_sy = (_MAX_COORD - y1) / h if h > 0 else 1.0

    sx = rng.uniform(scale_min, min(scale_max, max_sx))
    sy = rng.uniform(scale_min, min(scale_max, max_sy))

    # 좌상단(x1, y1) 고정 기준 스케일
    _apply_coord_transform(
        sample,
        lambda x: x1 + (x - x1) * sx,
        lambda y: y1 + (y - y1) * sy,
        scale_wh_x=sx,
        scale_wh_y=sy,
    )
    return sample


def zoom(
    sample: dict,
    rng: random.Random,
    zoom_min: float = 0.7,
    zoom_max: float = 1.4,
) -> dict:
    """평면도를 균일하게 확대/축소한다.

    bbox 중심 기준으로 스케일 후 256×256 경계를 초과하면 자동으로 평행이동한다.
    문의 너비/높이도 동일 배율로 조정한다.

    Args:
        sample: row-oriented 평면도 딕셔너리.
        rng: Random 인스턴스.
        zoom_min: 줌 스케일 최솟값.
        zoom_max: 줌 스케일 상한값 (실제 max는 256 경계 기반 자동 계산값과 min으로 결정).

    Returns:
        확대/축소된 딕셔너리 (in-place 수정 후 반환).
    """
    x1, y1, x2, y2 = _compute_outline_bbox(sample)
    w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # 최대 스케일: 스케일 후 bbox가 256×256에 맞도록
    max_s = min(_MAX_COORD / w, _MAX_COORD / h) if (w > 0 and h > 0) else 1.0
    s = rng.uniform(zoom_min, min(zoom_max, max_s))

    # 중심 기준 스케일 후 예상 bbox
    new_x1 = cx + (x1 - cx) * s
    new_x2 = cx + (x2 - cx) * s
    new_y1 = cy + (y1 - cy) * s
    new_y2 = cy + (y2 - cy) * s

    # 경계 초과 시 평행이동 오프셋 계산
    off_x = max(0.0, -new_x1) if new_x1 < 0 else min(0.0, _MAX_COORD - new_x2)
    off_y = max(0.0, -new_y1) if new_y1 < 0 else min(0.0, _MAX_COORD - new_y2)

    _apply_coord_transform(
        sample,
        lambda x: cx + (x - cx) * s + off_x,
        lambda y: cy + (y - cy) * s + off_y,
        scale_wh_x=s,
        scale_wh_y=s,
    )
    return sample


def compute_noise_state(
    sample: dict,
    params: dict,
    rng: random.Random,
) -> dict[int, list[int]]:
    """방 좌표 노이즈 증강 상태를 계산한다 (INPUT 전용).

    p_noise 확률로 전체 샘플의 방 꼭지점 좌표에 가우시안 노이즈를 추가한다.
    노이즈가 적용된 rid → noisy_coords 딕셔너리를 반환하며,
    sample 자체는 수정하지 않는다.

    Args:
        sample: row-oriented 평면도 딕셔너리.
        params: 노이즈 파라미터 딕셔너리. 키:
            - p_noise: 노이즈 적용 확률 (샘플 단위).
            - noise_sigma: 가우시안 표준편차 (픽셀).
        rng: Random 인스턴스.

    Returns:
        rid → 노이즈 좌표 리스트 딕셔너리. 노이즈 미적용 시 빈 dict.
    """
    if rng.random() >= params.get("p_noise", 0.0):
        return {}

    sigma = params.get("noise_sigma", 3.0)
    noise_room_coords: dict[int, list[int]] = {}

    for room in sample["rooms"]:
        coords = room["coords"]
        noisy = [
            max(0, min(_MAX_COORD, v + round(rng.gauss(0.0, sigma))))
            for v in coords
        ]
        noise_room_coords[room["rid"]] = noisy

    return noise_room_coords


# ---------------------------------------------------------------------------
# 삭제 기반 증강 상태 계산
# ---------------------------------------------------------------------------

def compute_drop_state(
    sample: dict,
    params: dict,
    rng: random.Random,
) -> DropState:
    """삭제 기반 증강 상태를 확률 샘플링하여 계산한다.

    sample을 직접 수정하지 않고 DropState에 상태만 기록한다.
    DropBlock으로 삭제된 RID를 참조하는 edge/spatial은 즉시 orphan 처리된다.

    Args:
        sample: row-oriented 평면도 딕셔너리 (변형 증강 이미 적용된 상태).
        params: 증강 확률 파라미터 딕셔너리. 키:
            - p_drop_block: 방 블록 전체 삭제 확률
            - p_drop_type: 방 종류 삭제 확률
            - p_drop_coords: 방 좌표 삭제 확률
            - p_drop_edge: 엣지 전체 삭제 확률
            - p_drop_pair: 엣지 RID 쌍 삭제 확률
            - p_drop_door: 엣지 문 정보 삭제 확률
            - p_drop_spatial: spatial 항목 삭제 확률
            - p_drop_front_door: 현관문 삭제 확률
        rng: Random 인스턴스.

    Returns:
        DropState 객체.
    """
    state = DropState()

    p_block  = params.get("p_drop_block",  0.0)
    p_type   = params.get("p_drop_type",   0.0)
    p_coords = params.get("p_drop_coords", 0.0)
    p_edge   = params.get("p_drop_edge",   0.0)
    p_pair   = params.get("p_drop_pair",   0.0)
    p_door   = params.get("p_drop_door",   0.0)
    p_sp     = params.get("p_drop_spatial",    0.0)
    p_fd     = params.get("p_drop_front_door", 0.0)

    # --- 방 블록 (outline 포함, mutually exclusive) ---
    # outline(rid=0)은 drop_block/drop_coords 적용 가능; drop_type은 불가 (항상 "outline")
    # outline은 edge/spatial에 등장하지 않으므로 orphan 처리 불필요
    for room in sample["rooms"]:
        rid = room["rid"]
        roll = rng.random()
        if roll < p_block:
            state.drop_block.add(rid)
            # orphan 처리: outline은 edge/spatial에 없으므로 non-outline만 처리
            if room["type"] != "outline":
                for e_idx, edge in enumerate(sample["edges"]):
                    if rid in edge["pair"]:
                        state.drop_edge.add(e_idx)
                for sp_idx, sp in enumerate(sample["spatial"]):
                    if sp["rid_a"] == rid or sp["rid_b"] == rid:
                        state.drop_spatial.add(sp_idx)
        elif roll < p_block + p_type and room["type"] != "outline":
            # drop_type: outline은 type이 항상 "outline"이므로 의미 없어 제외
            state.drop_type.add(rid)
        elif roll < p_block + p_type + p_coords:
            state.drop_coords.add(rid)
        # else: 유지

    # --- 엣지 (orphan으로 이미 drop_edge에 있는 것 제외, mutually exclusive) ---
    for e_idx, edge in enumerate(sample["edges"]):
        if e_idx in state.drop_edge:
            continue
        roll = rng.random()
        if roll < p_edge:
            state.drop_edge.add(e_idx)
        elif roll < p_edge + p_pair:
            # "both" 또는 "one" 랜덤 선택
            mode = rng.choice(["one", "both"])
            if mode == "one":
                # 남길 RID를 tuple로 기록 — sample은 수정하지 않음
                keep_idx = rng.randint(0, len(edge["pair"]) - 1)
                state.drop_pair[e_idx] = ("one", edge["pair"][keep_idx])
            else:
                state.drop_pair[e_idx] = "both"
        elif roll < p_edge + p_pair + p_door:
            # NO_DOOR 엣지에는 DropEdgeDoor 적용 불가
            if edge["door"]:
                mode = rng.choice(["position", "orientation", "all"])
                state.drop_door[e_idx] = mode
        # else: 유지

    # --- Spatial (orphan 제외, 항목별 독립 확률) ---
    for sp_idx in range(len(sample["spatial"])):
        if sp_idx in state.drop_spatial:
            continue
        if rng.random() < p_sp:
            state.drop_spatial.add(sp_idx)

    # --- 현관문 (drop_block / drop_coords, mutually exclusive) ---
    p_fd_coords = params.get("p_drop_front_door_coords", 0.0)
    roll = rng.random()
    if roll < p_fd:
        state.drop_front_door = True          # 블록 전체 생략
    elif roll < p_fd + p_fd_coords:
        state.drop_front_door_coords = True   # 좌표만 생략

    # --- ROOM_SUMMARY (total 쌍 / 방 타입별 count 쌍, 독립 확률) ---
    p_rs_total = params.get("p_drop_room_summary_total", 0.0)
    p_rs_type  = params.get("p_drop_room_summary_type",  0.0)

    # <TOTAL>N 쌍 삭제 여부 (샘플 단위 1회 결정)
    if rng.random() < p_rs_total:
        state.drop_room_summary_total = True

    # 방 타입별 <TYPE:t><COUNT>N 쌍 삭제 여부 (타입별 독립 확률)
    non_outline_types = {r["type"] for r in sample["rooms"] if r["type"] != "outline"}
    for room_type in non_outline_types:
        if rng.random() < p_rs_type:
            state.drop_room_summary_types.add(room_type)

    return state
