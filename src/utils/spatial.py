"""전처리·증강·보상이 공유하는 폴리곤 중심점과 공간 방향 계산."""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Polygon

DIRECTIONS = ("right", "right-below", "below", "left-below",
              "left", "left-above", "above", "right-above")


def polygon_centroid(coords) -> tuple[float, float]:
    """꼭짓점 표현과 순서에 무관한 폴리곤 중심점을 반환한다.

    Mod Record: 픽셀 평균과 꼭짓점 평균을 폴리곤 면적 중심점으로 통일한다.
    입력 노이즈로 퇴화한 도형도 동일한 Shapely 중심점 규칙을 사용하며,
    도형의 유효성을 보정하거나 원래 좌표를 변경하지 않는다.

    Args:
        coords: 평탄화된 좌표 또는 좌표쌍 목록.
    Returns:
        중심점의 X/Y 좌표.
    Raises:
        ValueError: 좌표가 부족하거나 유한한 중심점을 계산할 수 없을 때.
    """
    vertices = np.asarray(coords, dtype=float).reshape(-1, 2)
    if len(vertices) < 3 or not np.isfinite(vertices).all():
        raise ValueError("중심점 계산에는 유한한 꼭짓점 세 개 이상이 필요합니다.")
    center = Polygon(vertices).centroid
    if center.is_empty or not all(math.isfinite(v) for v in (center.x, center.y)):
        raise ValueError("폴리곤 중심점을 계산할 수 없습니다.")
    return float(center.x), float(center.y)


def vector_to_direction(dx: float, dy: float) -> str:
    """이미지 좌표계 벡터를 반열린 구간의 8방위로 분류한다.

    Args:
        dx: X 좌표 차이.
        dy: 아래 방향이 양수인 Y 좌표 차이.
    Returns:
        8방위 문자열. 영벡터는 기존 전처리 규약에 따라 right.
    Raises:
        ValueError: 벡터가 유한하지 않을 때.
    """
    if not math.isfinite(dx) or not math.isfinite(dy):
        raise ValueError("방향 벡터는 유한해야 합니다.")
    angle = math.degrees(math.atan2(dy, dx)) % 360.0
    return DIRECTIONS[int((angle + 22.5) % 360.0 // 45.0)]


def refresh_spatial_relations(sample: dict) -> None:
    """깨끗한 방 도형으로 기존 공간 관계 라벨을 갱신한다.

    Args:
        sample: rooms와 spatial을 포함하는 행 단위 평면도.
    Returns:
        없음. 기존 관계의 순서와 RID 쌍을 유지하며 라벨만 갱신한다.
    Raises:
        ValueError: 참조하는 방이 없거나 중심점을 계산할 수 없을 때.
    """
    relations = sample.get("spatial", [])
    required = {sp[key] for sp in relations for key in ("rid_a", "rid_b")}
    centers = {r["rid"]: polygon_centroid(r["coords"])
               for r in sample["rooms"] if r["rid"] in required}
    if required - centers.keys():
        raise ValueError("공간 관계가 존재하지 않는 방을 참조합니다.")
    for sp in relations:
        a, b = centers[sp["rid_a"]], centers[sp["rid_b"]]
        sp["direction"] = vector_to_direction(b[0] - a[0], b[1] - a[1])
