"""Spatial 관계 계산 모듈 (Step 9).

모든 방 쌍(exterior_wall 제외)에 대해 centroid 기반 8방위 관계를 계산한다.
$\\theta = \\text{atan2}(\\Delta y, \\Delta x)$로 각도를 구하고, ±22.5도 threshold로 8방위를 결정.

이미지 좌표계: y축 아래 방향이 양수.
- $[a, b, \\text{"right"}]$ = b가 a의 오른쪽.
- 역방향은 저장하지 않음 (한 방향만 저장).
"""

from __future__ import annotations

from itertools import combinations

from src.build_dataset.rplan2json.room_extractor import RoomInstance

from src.utils.spatial import polygon_centroid, vector_to_direction


def build_spatial_relations(
    room_instances: list[RoomInstance],
) -> list[list]:
    """Step 9: 모든 방 쌍의 공간 관계 계산.

    각 방 쌍(exterior_wall 제외)에 대해 centroid 기반 8방위를 결정한다.
    $[a, b, \\text{direction}]$ — b가 a의 direction 방향에 위치.

    Args:
        room_instances: 방 인스턴스 리스트 (rid 배정 완료).

    Returns:
        [[rid_a, rid_b, direction], ...] 리스트. rid_a < rid_b.
    """
    rooms = [r for r in room_instances if r.type_name != "outline"]
    if len(rooms) < 2:
        return []

    spatial: list[list] = []

    for room_a, room_b in combinations(rooms, 2):
        # rid_a < rid_b 순서 보장
        if room_a.rid < room_b.rid:
            a, b = room_a, room_b
        else:
            a, b = room_b, room_a

        # Mod Record: 저장된 픽셀 중심점 대신 출력과 같은 폴리곤 중심점을 사용한다.
        direction = _compute_direction(polygon_centroid(a.coords), polygon_centroid(b.coords))
        spatial.append([a.rid, b.rid, direction])

    spatial.sort(key=lambda s: (s[0], s[1]))
    return spatial


def _compute_direction(
    centroid_a: tuple[float, float],
    centroid_b: tuple[float, float],
) -> str:
    """두 centroid 간 8방위 방향 계산.

    a 기준 b가 어느 방향에 있는지 계산.
    $\\theta = \\text{atan2}(b_y - a_y, b_x - a_x)$.

    Args:
        centroid_a: 방 A 중심점 (cx, cy).
        centroid_b: 방 B 중심점 (cx, cy).

    Returns:
        8방위 문자열 (예: "right", "below", "left-above").
    """
    return vector_to_direction(centroid_b[0] - centroid_a[0], centroid_b[1] - centroid_a[1])
