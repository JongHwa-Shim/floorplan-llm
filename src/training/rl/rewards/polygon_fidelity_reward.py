"""좌표가 주어진 입력 방의 꼭짓점 충실도를 평가하는 이진 보상."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from src.training.rl.rewards.parser import ParsedFloorplan


def compute_polygon_fidelity_reward(
    parsed: "ParsedFloorplan", metadata: dict, tolerance: float = 15.0,
) -> tuple[float, list[int]]:
    """Hungarian 방 매칭 후 입력 꼭짓점의 거리 허용치를 검사한다.

    Mod Record: 중심점만 같고 모양은 다른 방을 통과시키던 연속 보상을 대체한다.
    방 쌍의 비용은 양방향 최근접 꼭짓점 거리의 평균이다. 꼭짓점 순서와 시작점에
    무관하며, 입력에 없는 방은 생성할 수 있다. 출력 꼭짓점 중 입력 꼭짓점과
    대응하지 않는 것만 마스킹하고, 없는 출력 꼭짓점의 책임 위치를 추측하지 않는다.

    Args:
        parsed: 파싱된 생성 평면도.
        metadata: 모델에 노출된 방의 type과 평탄화된 coords.
        tolerance: 입력/출력 꼭짓점 사이의 최대 유클리드 거리(px).

    Returns:
        모든 지정 꼭짓점 재현 여부와 위반 출력 좌표 토큰 인덱스.
        좌표 조건이 없으면 보상 1과 빈 마스크를 반환한다.

    Raises:
        ValueError: tolerance가 음수이거나 유한하지 않을 때.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance는 유한한 0 이상의 값이어야 합니다.")
    if not parsed.success:
        return 0.0, []
    inputs = [room for room in metadata.get("rooms", []) if room.get("coords")]
    if not inputs:
        return 1.0, []
    outputs = parsed.rooms
    if not outputs:
        return 0.0, []

    # 최대 16개 방의 작은 할당 문제이며, 좌표 거리 계산만 NumPy로 벡터화한다.
    costs = np.full((len(inputs), len(outputs)), 1e12)
    distances = {}
    for i, room in enumerate(inputs):
        coords = np.asarray(room["coords"], dtype=float)
        if coords.size % 2 or not np.isfinite(coords).all():
            return 0.0, []
        vertices = coords.reshape(-1, 2)  # (V_in, 2)
        room_type = room.get("type", "")
        for j, output in enumerate(outputs):
            if (room_type and room_type != output.room_type) or (
                not room_type and output.room_type == "outline"
            ) or not output.coords:
                continue
            generated = np.asarray(output.coords, dtype=float)
            delta = vertices[:, None, :] - generated[None, :, :]  # (V_in, V_out, 2)
            distance = np.sqrt(np.sum(delta * delta, axis=-1))  # (V_in, V_out)
            input_nearest = distance.min(axis=1)
            output_nearest = distance.min(axis=0)
            costs[i, j] = (input_nearest.mean() + output_nearest.mean()) / 2
            distances[i, j] = (input_nearest, output_nearest)

    rows, cols = linear_sum_assignment(costs)
    satisfied = len(rows) == len(inputs)
    errors = []
    for i, j in zip(rows, cols):
        if (i, j) not in distances:
            satisfied = False
            continue
        input_nearest, output_nearest = distances[i, j]
        satisfied = satisfied and bool(np.all(input_nearest <= tolerance))
        for vertex in np.flatnonzero(output_nearest > tolerance):
            indices = outputs[j].coord_token_indices
            if vertex < len(indices):
                errors.extend((indices[vertex], indices[vertex] + 1))
    return float(satisfied), sorted(set(errors))