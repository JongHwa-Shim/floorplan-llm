"""공통 중심점과 일대일 방 대응으로 공간 방향을 평가한다."""

from __future__ import annotations

from src.utils.spatial import polygon_centroid, vector_to_direction
from src.training.rl.rewards.room_assignment import has_consistent_assignment
from src.training.rl.rewards.connectivity_reward import _hungarian_match, _assignment_candidates


def compute_spatial_reward(parsed, metadata: dict) -> float:
    """모든 방향 조건을 동시에 만족하는 방 대응의 존재 여부를 반환한다.

    Args:
        parsed: 파싱된 생성 평면도.
        metadata: 입력에 노출된 방과 공간 관계.
    Returns:
        모든 방향 조건을 일관된 일대일 대응으로 충족하면 1.
    Raises:
        없음. 중심점을 계산할 수 없는 방 쌍은 만족 후보에서 제외한다.
    """
    if not parsed.rooms:
        return 0.0
    spatial = metadata.get("spatial", [])
    if not spatial:
        return 1.0
    output_rooms = [r for r in parsed.rooms if r.room_type != "outline"]
    centers = {}
    for index, room in enumerate(output_rooms):
        try:
            centers[index] = polygon_centroid(room.coords)
        except (ValueError, TypeError):
            continue
    # Mod Record: 기하는 한 번만 계산하고 모든 방향 제약에 같은 RID 대응을 사용한다.
    by_direction = {}
    for a, first in centers.items():
        for b, second in centers.items():
            dx, dy = second[0] - first[0], second[1] - first[1]
            if a == b or (abs(dx) < 1e-9 and abs(dy) < 1e-9):
                continue
            by_direction.setdefault(vector_to_direction(dx, dy), set()).add((a, b))
    constraints = [(sp["rid_a"], sp["rid_b"], by_direction.get(sp.get("direction"), set()))
                   for sp in spatial if sp.get("rid_a") is not None and sp.get("rid_b") is not None]
    if not constraints:
        return 1.0
    anchors = _hungarian_match(parsed, metadata)
    candidates = _assignment_candidates(constraints, metadata.get("rooms", []), output_rooms, anchors)
    return float(has_consistent_assignment(candidates, constraints))


def _vector_to_direction(dx: float, dy: float) -> str:
    """공통 8방위 계산을 기존 호출 경로에도 제공한다.

    Args:
        dx: X 좌표 차이.
        dy: Y 좌표 차이.
    Returns:
        이미지 좌표계의 8방위 문자열.
    Raises:
        ValueError: 벡터가 유한하지 않을 때.
    """
    return vector_to_direction(dx, dy)
