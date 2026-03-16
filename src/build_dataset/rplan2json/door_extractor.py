"""현관문 및 인테리어 문 추출 모듈.

Step 6: 현관문(Front Door) 정보 추출 (G==15)
Step 7: 인테리어 문(Interior Door) 인스턴스 추출 (G==17)
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DoorInstance:
    """추출된 인테리어 문 인스턴스.

    Attributes:
        door_id: 문 고유 ID.
        mask: 문 영역 바이너리 마스크.
        bbox: 중심점 좌표 + 크기 {"x", "y", "w", "h"}.

    Shape:
        mask: $(H, W)$, dtype uint8, 0 또는 255.
    """

    door_id: int
    mask: np.ndarray
    bbox: dict[str, int]


def extract_front_door(
    space_type: np.ndarray,
    connectivity: int = 8,
) -> dict[str, int] | None:
    """Step 6: G채널==15 영역에서 현관문 정보 추출.

    가장 큰 connected component를 현관문으로 간주하여
    중심점 좌표(x,y) + 폭(w) + 높이(h)를 반환한다.

    Args:
        space_type: G채널 배열.
        connectivity: CCL 연결성 (기본 8).

    Returns:
        {"x": int, "y": int, "w": int, "h": int} 또는 None.
    """
    front_door_mask = (space_type == 15).astype(np.uint8)
    if front_door_mask.sum() == 0:
        return None

    num_labels, labeled = cv2.connectedComponents(
        front_door_mask, connectivity=connectivity
    )
    if num_labels <= 1:
        return None

    # 가장 큰 컴포넌트 선택
    max_area = 0
    max_label = 1
    for label_id in range(1, num_labels):
        area = int((labeled == label_id).sum())
        if area > max_area:
            max_area = area
            max_label = label_id

    component = (labeled == max_label)
    return _compute_bbox(component)


def extract_interior_doors(
    space_type: np.ndarray,
    connectivity: int = 8,
    min_door_pixels: int = 5,
) -> list[DoorInstance]:
    """Step 7: G채널==17 영역에서 인테리어 문 인스턴스 추출.

    CCL로 문 클러스터를 분리하고, min_door_pixels 미만 컴포넌트는 노이즈로 제거한다.
    직각으로 인접한 두 문이 L자형으로 병합된 경우 자동으로 분할한다.

    Args:
        space_type: G채널 배열.
        connectivity: CCL 연결성 (기본 4, 직각 인접 문 분리를 위해 4 권장).
        min_door_pixels: 최소 문 픽셀 수.

    Returns:
        DoorInstance 리스트.
    """
    door_mask = (space_type == 17).astype(np.uint8)
    if door_mask.sum() == 0:
        return []

    num_labels, labeled = cv2.connectedComponents(
        door_mask, connectivity=connectivity
    )

    doors: list[DoorInstance] = []
    door_id = 0

    for label_id in range(1, num_labels):
        component = (labeled == label_id).astype(np.uint8) * 255
        pixel_count = int(component.sum() // 255)
        if pixel_count < min_door_pixels:
            continue

        # 복합 병합 컴포넌트를 개별 직사각형으로 재귀 분해
        sub_components = decompose_door_component(component, min_door_pixels)

        for sub in sub_components:
            if int(sub.sum() // 255) < min_door_pixels:
                continue
            doors.append(
                DoorInstance(
                    door_id=door_id,
                    mask=sub,
                    bbox=_compute_bbox(sub),
                )
            )
            door_id += 1

    return doors


def _is_rectangular(component: np.ndarray) -> bool:
    """컴포넌트가 완벽한 직사각형인지 판별한다.

    픽셀 수 == bbox 면적이면 직사각형 (정수 연산이므로 float 오차 없음).

    Args:
        component: uint8 마스크 (0 또는 255).

    Returns:
        직사각형이면 True.
    """
    ys, xs = np.where(component > 0)
    if len(ys) == 0:
        return True
    area = len(ys)
    bbox_area = (int(xs.max()) - int(xs.min()) + 1) * (int(ys.max()) - int(ys.min()) + 1)
    return area == bbox_area


def _peel_arm(
    component: np.ndarray,
    use_rows: bool,
) -> tuple[np.ndarray | None, np.ndarray]:
    """가장 픽셀 수가 많은 연속 행(또는 열)을 팔(arm)로 박리한다.

    박리된 팔은 arm_mask로, 나머지는 remainder로 반환한다.
    비연속 max_count 행/열이 있을 경우 첫 번째 연속 그룹만 추출한다.

    Args:
        component: uint8 마스크 (0 또는 255).
        use_rows: True이면 행 방향(수평 팔) 박리, False이면 열 방향(수직 팔) 박리.

    Returns:
        (arm_mask, remainder) 튜플. 박리 실패 시 arm_mask=None.
    """
    ys, xs = np.where(component > 0)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())

    if use_rows:
        # 각 행의 픽셀 수
        counts = np.array(
            [(component[y, :] > 0).sum() for y in range(y_min, y_max + 1)],
            dtype=np.int32,
        )
        max_count = int(counts.max())
        indices = np.where(counts == max_count)[0]  # y_min 기준 상대 인덱스

        # 연속 그룹 중 첫 번째 그룹만 사용
        groups = _consecutive_groups(indices)
        if not groups:
            return None, component

        first_group = groups[0]
        arm_y_min = int(first_group[0]) + y_min
        arm_y_max = int(first_group[-1]) + y_min

        arm_mask = np.zeros_like(component)
        arm_mask[arm_y_min:arm_y_max + 1, :] = component[arm_y_min:arm_y_max + 1, :]
    else:
        # 각 열의 픽셀 수
        counts = np.array(
            [(component[:, x] > 0).sum() for x in range(x_min, x_max + 1)],
            dtype=np.int32,
        )
        max_count = int(counts.max())
        indices = np.where(counts == max_count)[0]  # x_min 기준 상대 인덱스

        groups = _consecutive_groups(indices)
        if not groups:
            return None, component

        first_group = groups[0]
        arm_x_min = int(first_group[0]) + x_min
        arm_x_max = int(first_group[-1]) + x_min

        arm_mask = np.zeros_like(component)
        arm_mask[:, arm_x_min:arm_x_max + 1] = component[:, arm_x_min:arm_x_max + 1]

    remainder = component.copy()
    remainder[arm_mask > 0] = 0
    return arm_mask, remainder


def _consecutive_groups(indices: np.ndarray) -> list[np.ndarray]:
    """인덱스 배열을 연속 그룹으로 분할한다.

    Args:
        indices: 정렬된 정수 인덱스 배열.

    Returns:
        연속 인덱스 그룹 리스트.
    """
    if len(indices) == 0:
        return []
    groups = []
    current = [indices[0]]
    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            current.append(indices[i])
        else:
            groups.append(np.array(current))
            current = [indices[i]]
    groups.append(np.array(current))
    return groups


def _to_bbox_rect(component: np.ndarray) -> np.ndarray:
    """컴포넌트의 bbox 범위를 완전히 채운 직사각형 마스크를 반환한다.

    삐뚤빼뚤한 문 영역을 직사각형으로 보정할 때 사용한다.

    Args:
        component: uint8 마스크 (0 또는 255).

    Returns:
        bbox를 꽉 채운 직사각형 마스크.
    """
    ys, xs = np.where(component > 0)
    rect = np.zeros_like(component)
    rect[int(ys.min()):int(ys.max()) + 1, int(xs.min()):int(xs.max()) + 1] = 255
    return rect


def _find_peak_valley_groups(
    counts: np.ndarray,
    threshold_ratio: float = 0.6,
) -> list[tuple[bool, int, int]]:
    """투영 프로파일을 peak/valley 구간으로 분류한다.

    max_count의 threshold_ratio 미만인 연속 구간을 valley,
    이상인 구간을 peak로 판별한다.

    Args:
        counts: 각 행(또는 열)의 픽셀 수 배열.
        threshold_ratio: valley 판별 임계값 비율.

    Returns:
        (is_peak, start_idx, end_idx) 튜플 리스트.
        is_peak=True이면 peak 구간, False이면 valley 구간.
    """
    max_count = int(counts.max())
    if max_count == 0:
        return []
    threshold = max_count * threshold_ratio

    groups: list[tuple[bool, int, int]] = []
    is_peak = bool(counts[0] >= threshold)
    start = 0
    for i in range(1, len(counts)):
        now_peak = bool(counts[i] >= threshold)
        if now_peak != is_peak:
            groups.append((is_peak, start, i - 1))
            start = i
            is_peak = now_peak
    groups.append((is_peak, start, len(counts) - 1))
    return groups


def _try_valley_split(
    component: np.ndarray,
    min_door_pixels: int,
    max_depth: int,
    _depth: int,
    threshold_ratio: float = 0.6,
) -> list[np.ndarray] | None:
    """투영 valley 기반으로 컴포넌트 분해를 시도한다.

    수평/수직 투영 프로파일에서 valley(픽셀 수 급감 구간)를 탐지하여
    peak 구간은 직사각형 bbox로 보정, valley 구간은 재귀 처리한다.

    Args:
        component: uint8 마스크 (0 또는 255).
        min_door_pixels: 최소 문 픽셀 수.
        max_depth: 최대 재귀 깊이.
        _depth: 현재 재귀 깊이.
        threshold_ratio: valley 판별 임계값 비율 (max의 이 비율 미만 → valley).

    Returns:
        분해된 마스크 리스트, 또는 분해 불가 시 None.
    """
    ys, xs = np.where(component > 0)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())

    row_counts = np.array(
        [(component[y, :] > 0).sum() for y in range(y_min, y_max + 1)],
        dtype=np.int32,
    )
    col_counts = np.array(
        [(component[:, x] > 0).sum() for x in range(x_min, x_max + 1)],
        dtype=np.int32,
    )

    for use_rows, counts, base in [
        (True, row_counts, y_min),
        (False, col_counts, x_min),
    ]:
        groups = _find_peak_valley_groups(counts, threshold_ratio)
        peaks = [g for g in groups if g[0]]
        valleys = [g for g in groups if not g[0]]

        # peak가 1개 이상이고 valley가 있어야 분리 가능
        # peak 1개 + valley 1개: L자형 (수직 arm + 수평 arm) 처리 포함
        if not peaks or not valleys:
            continue

        results: list[np.ndarray] = []
        for is_peak, start, end in groups:
            abs_start = start + base
            abs_end = end + base

            # 해당 구간의 픽셀 slice
            slice_mask = np.zeros_like(component)
            if use_rows:
                slice_mask[abs_start:abs_end + 1, :] = component[abs_start:abs_end + 1, :]
            else:
                slice_mask[:, abs_start:abs_end + 1] = component[:, abs_start:abs_end + 1]

            if slice_mask.sum() == 0:
                continue

            # CCL로 분리 (4-connectivity)
            num_labels, labeled = cv2.connectedComponents(slice_mask, connectivity=4)
            for label_id in range(1, num_labels):
                sub = (labeled == label_id).astype(np.uint8) * 255
                if int(sub.sum() // 255) < min_door_pixels:
                    continue

                if is_peak:
                    # peak 구간: 직사각형 bbox로 보정 (겹침 오버랩 포함)
                    results.append(_to_bbox_rect(sub))
                else:
                    # valley 구간: 재귀 분해
                    results.extend(
                        decompose_door_component(sub, min_door_pixels, max_depth, _depth + 1)
                    )

        if results:
            return results

    return None


def decompose_door_component(
    component: np.ndarray,
    min_door_pixels: int = 5,
    max_depth: int = 8,
    _depth: int = 0,
) -> list[np.ndarray]:
    """병합된 문 컴포넌트를 개별 직사각형 마스크로 재귀 분해한다.

    1단계: 투영 프로파일 valley 기반 분해 우선 시도.
      - 수평/수직 투영에서 픽셀 수 급감 구간(valley)을 경계로 분리.
      - peak 구간은 직사각형 bbox로 보정, valley 구간은 재귀 처리.
    2단계: valley가 없으면 최대 스트라이프 arm 박리 fallback.

    Args:
        component: uint8 마스크 (0 또는 255).
        min_door_pixels: 최소 문 픽셀 수. 미만 조각은 버림.
        max_depth: 최대 재귀 깊이 (무한루프 방지).
        _depth: 현재 재귀 깊이 (내부 사용).

    Returns:
        분해된 직사각형 마스크 리스트.
    """
    # 종료 조건 1: 이미 직사각형
    if _is_rectangular(component):
        return [component]

    # 종료 조건 2: 최대 재귀 깊이 초과 → bbox로 보정
    if _depth >= max_depth:
        return [_to_bbox_rect(component)]

    # 1단계: 투영 valley 기반 분해 시도
    valley_result = _try_valley_split(component, min_door_pixels, max_depth, _depth)
    if valley_result is not None:
        return valley_result

    # 2단계: fallback - 최대 스트라이프 arm 박리
    ys, xs = np.where(component > 0)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())

    row_counts = np.array(
        [(component[y, :] > 0).sum() for y in range(y_min, y_max + 1)],
        dtype=np.int32,
    )
    col_counts = np.array(
        [(component[:, x] > 0).sum() for x in range(x_min, x_max + 1)],
        dtype=np.int32,
    )
    use_rows = int(row_counts.max()) >= int(col_counts.max())

    arm_mask, remainder = _peel_arm(component, use_rows)
    if arm_mask is None or arm_mask.sum() == 0:
        return [_to_bbox_rect(component)]

    results = [arm_mask]

    # 나머지를 연결 컴포넌트로 분리 후 재귀
    if remainder.sum() > 0:
        num_labels, labeled = cv2.connectedComponents(remainder, connectivity=4)
        for label_id in range(1, num_labels):
            sub = (labeled == label_id).astype(np.uint8) * 255
            if int(sub.sum() // 255) < min_door_pixels:
                continue
            results.extend(
                decompose_door_component(sub, min_door_pixels, max_depth, _depth + 1)
            )

    return results


def _compute_bbox(component: np.ndarray) -> dict[str, int]:
    """바이너리 컴포넌트에서 중심점 + 크기 bbox 계산.

    Args:
        component: bool 또는 uint8 마스크.

    Returns:
        {"x": 중심x, "y": 중심y, "w": 폭, "h": 높이}.
    """
    ys, xs = np.where(component > 0)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    return {
        "x": (x_min + x_max) // 2,
        "y": (y_min + y_max) // 2,
        "w": x_max - x_min + 1,
        "h": y_max - y_min + 1,
    }
