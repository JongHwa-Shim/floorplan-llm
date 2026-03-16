"""방 인스턴스 분리 및 꼭지점 좌표 추출 모듈.

Step 처리:
    1. 방 타입 병합 설정 로드/적용
    2. 방 인스턴스 분리 (타입별 Connected Component, 4-connectivity)
    3. 꼭지점 좌표 추출 (findContours + approxPolyDP)
    4. 외곽선(outline) 폴리곤 추출 - external_area(G==13) 차집합 사용
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class TypeMergeConfig:
    """방 타입 병합 설정.

    Attributes:
        type_name_map: G채널 값 → 원본 타입 문자열 매핑 (예: {"0": "livingroom"}).
        merge_rules: 원본 타입 → 병합 타입 매핑 (예: {"masterroom": "bedroom"}).
    """

    type_name_map: dict[str, str]
    merge_rules: dict[str, str]

    def get_type_name(self, g_value: int) -> str | None:
        """G채널 값을 병합 적용된 최종 타입 문자열로 변환.

        Args:
            g_value: G채널 공간 타입 값 (0~12).

        Returns:
            병합 적용된 타입 문자열. 해당 G값이 매핑에 없으면 None.
        """
        raw_name = self.type_name_map.get(str(g_value))
        if raw_name is None:
            return None
        return self.merge_rules.get(raw_name, raw_name)


@dataclass
class RoomInstance:
    """추출된 방 인스턴스 정보.

    Attributes:
        rid: 방 고유 ID (평면도 내).
        type_name: 병합 적용된 방 종류 문자열.
        mask: 방 영역 바이너리 마스크.
        coords: 꼭지점 좌표 flat 리스트 [x1,y1,x2,y2,...].
        centroid: 방 중심점 (cx, cy).

    Shape:
        mask: $(H, W)$, dtype uint8, 0 또는 255.
    """

    rid: int
    type_name: str
    mask: np.ndarray
    coords: list[int] = field(default_factory=list)
    centroid: tuple[float, float] = (0.0, 0.0)


def load_type_merge_config(config_path: str) -> TypeMergeConfig:
    """room_type_merge.json 로드.

    Args:
        config_path: JSON 설정 파일 경로.

    Returns:
        TypeMergeConfig 인스턴스.

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return TypeMergeConfig(
        type_name_map=data["type_name_map"],
        merge_rules=data.get("merge_rules", {}),
    )


def extract_room_instances(
    space_type: np.ndarray,
    merge_config: TypeMergeConfig,
    room_type_ids: list[int],
    min_room_area: int,
    connectivity: int,
) -> list[RoomInstance]:
    """타입별 Connected Component로 방 인스턴스 분리 + 꼭지점 추출.

    각 방 타입(G=0~12)에 대해 바이너리 마스크를 생성하고,
    Connected Component Labeling으로 개별 인스턴스를 분리한다.
    min_room_area 미만 컴포넌트는 노이즈로 간주하여 제거.

    Args:
        space_type: G채널 배열.
        merge_config: 타입 병합 설정.
        room_type_ids: 방 타입으로 간주할 G값 리스트 (예: [0,1,...,12]).
        min_room_area: 최소 방 면적 (픽셀 수).
        connectivity: CCL 연결성 (4 또는 8).

    Returns:
        추출된 RoomInstance 리스트 (rid 미배정, 임시 0).
    """
    instances: list[RoomInstance] = []

    for g_val in room_type_ids:
        type_name = merge_config.get_type_name(g_val)
        if type_name is None:
            continue

        # 해당 타입의 바이너리 마스크
        type_mask = (space_type == g_val).astype(np.uint8)
        if type_mask.sum() == 0:
            continue

        # Connected Component Labeling
        num_labels, labeled = cv2.connectedComponents(type_mask, connectivity=connectivity)

        for label_id in range(1, num_labels):
            component_mask = (labeled == label_id).astype(np.uint8)  # (H, W)
            pixel_count = int(component_mask.sum())
            if pixel_count < min_room_area:
                continue

            # 꼭지점 좌표 추출
            coords = extract_polygon_coords(component_mask)
            if len(coords) < 6:  # 최소 3개 꼭지점 필요
                continue

            # centroid 계산
            ys, xs = np.where(component_mask > 0)
            cx, cy = float(xs.mean()), float(ys.mean())

            instances.append(
                RoomInstance(
                    rid=0,  # 나중에 raster scan 순서로 재배정
                    type_name=type_name,
                    mask=component_mask * 255,
                    coords=coords,
                    centroid=(cx, cy),
                )
            )

    return instances


def extract_polygon_coords(binary_mask: np.ndarray) -> list[int]:
    """바이너리 마스크에서 직교 폴리곤 꼭지점 좌표 추출.

    approxPolyDP 대신 직교 전용 알고리즘을 사용하여,
    수평/수직 에지만으로 구성된 완벽한 직각 폴리곤을 보장한다.
    canonical 순서(CW, top-left 시작점)로 정렬.

    Args:
        binary_mask: 방 영역 바이너리 마스크 (0 또는 1/255).

    Returns:
        [x1,y1,x2,y2,...] 형태의 flat integer 리스트.
    """
    # 마스크가 0/1이면 255로 스케일
    mask = binary_mask.copy()
    if mask.max() == 1:
        mask = mask * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # 가장 큰 컨투어 선택 후 (N, 2) 배열로 변환
    largest = max(contours, key=cv2.contourArea)
    raw_points = largest.reshape(-1, 2)

    # 직교 꼭지점 추출 (approxPolyDP 대체)
    points = _extract_orthogonal_corners(raw_points)
    if len(points) < 4:
        return []

    # Canonical 꼭지점 순서: CW 방향 보장 + top-left 시작점
    points = _normalize_vertex_order(points)

    # flat 리스트로 변환
    return points.flatten().tolist()


def _extract_orthogonal_corners(points: np.ndarray) -> np.ndarray:
    """컨투어 포인트에서 직교 폴리곤 꼭지점만 추출하고 좌표를 스냅한다.

    RPLAN처럼 수평/수직 에지만 가지는 직교 형태(rectilinear polygon)에 특화된
    알고리즘으로, approxPolyDP의 대각선 근사 문제를 근본적으로 해결한다.

    알고리즘:
        1. 각 포인트에서 진입/진출 방향을 비교하여 H→V 또는 V→H 전환점만 코너로 인식.
        2. 인접 코너 간 세그먼트를 H/V로 분류하여 공유 좌표를 스냅:
           - H 세그먼트: 양 끝점의 y값을 평균으로 맞춤 (수평 보장)
           - V 세그먼트: 양 끝점의 x값을 평균으로 맞춤 (수직 보장)

    Args:
        points: CHAIN_APPROX_SIMPLE 컨투어 포인트 배열.

    Returns:
        직교 정렬된 꼭지점 배열.

    Shape:
        입력: $(N, 2)$, 출력: $(M, 2)$ where $M \\leq N$.
    """
    n = len(points)
    if n < 4:
        return points

    # Step 1: 방향 전환점(코너)만 추출
    # 진입 방향과 진출 방향의 주축(H/V)이 다른 점이 코너
    corners = []
    for i in range(n):
        p_prev = points[(i - 1) % n]
        p_curr = points[i]
        p_next = points[(i + 1) % n]

        d_in = p_curr - p_prev    # 진입 방향 벡터
        d_out = p_next - p_curr   # 진출 방향 벡터

        # 주축 판별: |dx| >= |dy| → 수평(H), 아니면 수직(V)
        in_horiz = abs(int(d_in[0])) >= abs(int(d_in[1]))
        out_horiz = abs(int(d_out[0])) >= abs(int(d_out[1]))

        # 주축이 전환되는 점 = 코너
        if in_horiz != out_horiz:
            corners.append(p_curr.copy())

    if len(corners) < 4:
        return points

    corners = np.array(corners, dtype=np.int32)
    m = len(corners)

    # Step 2: 세그먼트별 좌표 스냅
    # H 세그먼트: 양 끝점의 y를 동일하게 스냅 (수평 에지 보장)
    # V 세그먼트: 양 끝점의 x를 동일하게 스냅 (수직 에지 보장)
    # 각 코너는 H 세그먼트와 V 세그먼트 각각에 정확히 한 번씩 참여하므로 충돌 없음
    for i in range(m):
        j = (i + 1) % m
        d = corners[j].astype(int) - corners[i].astype(int)

        if abs(d[0]) >= abs(d[1]):
            # H 세그먼트: y 좌표 스냅
            avg_y = int(round((int(corners[i][1]) + int(corners[j][1])) / 2))
            corners[i][1] = avg_y
            corners[j][1] = avg_y
        else:
            # V 세그먼트: x 좌표 스냅
            avg_x = int(round((int(corners[i][0]) + int(corners[j][0])) / 2))
            corners[i][0] = avg_x
            corners[j][0] = avg_x

    return corners


def _normalize_vertex_order(points: np.ndarray) -> np.ndarray:
    """꼭지점 순서를 canonical CW + top-left 시작점으로 정규화.

    Args:
        points: 꼭지점 배열.

    Returns:
        정규화된 꼭지점 배열.

    Shape:
        입력/출력: $(N, 2)$.
    """
    # CW 방향 보장: cross product 부호로 판별
    # OpenCV findContours는 이미지 좌표계에서 CW를 반환하므로,
    # signed area가 양수이면 CW (이미지 좌표계에서)
    signed_area = 0.0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        signed_area += points[i][0] * points[j][1]
        signed_area -= points[j][0] * points[i][1]

    if signed_area > 0:
        # 반시계(CCW) → 뒤집어서 CW로
        points = points[::-1].copy()

    # 시작점: x+y가 가장 작은 점 (좌상단에 가장 가까운 점)
    sums = points[:, 0] + points[:, 1]
    start_idx = int(np.argmin(sums))

    # 시작점 기준으로 회전
    points = np.roll(points, -start_idx, axis=0)

    return points


def extract_outline(space_type: np.ndarray) -> list[int]:
    """G채널==13 (external_area) 차집합으로 건물 외곽선(outline) 폴리곤 추출.

    external_area(G==13)는 건물 이외의 외부 영역을 표시하므로,
    전체 이미지 영역에서 external_area를 제외한 차집합이 건물 점유 영역이 된다.
    해당 영역의 가장 큰 connected component의 contour를 직교 폴리곤으로 추출.

    Args:
        space_type: G채널 배열.

    Returns:
        [x1,y1,x2,y2,...] 형태의 flat 좌표 리스트.
    """
    # external_area 마스크 (G==13): 건물 이외의 외부 공간
    external_area_mask = (space_type == 13).astype(np.uint8)

    # 건물 점유 영역 = 전체 이미지 - external_area (차집합 연산)
    full_mask = np.ones_like(external_area_mask, dtype=np.uint8)
    building_mask = full_mask & ~(external_area_mask > 0)
    building_mask = building_mask.astype(np.uint8)

    if building_mask.sum() == 0:
        return []

    # 가장 큰 컴포넌트만 사용 (건물 본체)
    num_labels, labeled = cv2.connectedComponents(building_mask, connectivity=4)
    if num_labels <= 1:
        return []

    max_area = 0
    max_label = 1
    for label_id in range(1, num_labels):
        area = int((labeled == label_id).sum())
        if area > max_area:
            max_area = area
            max_label = label_id

    largest_mask = (labeled == max_label).astype(np.uint8)
    return extract_polygon_coords(largest_mask)
