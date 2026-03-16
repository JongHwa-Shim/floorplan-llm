"""
평면도 요소 렌더링 모듈.

방, 문, 현관문을 OpenCV를 이용해 256x256 고정 해상도 이미지에 그린다.
좌표계는 0-255 범위로 이미지 픽셀에 1:1 대응한다.
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import Optional


class RoomRenderer:
    """방 및 문 요소를 256x256 이미지에 렌더링하는 클래스.

    좌표는 0-255 범위를 가정하며 픽셀에 직접 매핑된다.
    폴리곤 채우기(알파 블렌딩), 테두리, 레이블 텍스트를 지원한다.

    Attributes:
        image_size: 고정 출력 해상도 (정사각형).
        border_thickness: 테두리 두께.
        font_scale: 레이블 폰트 크기.
        alpha: 채우기 투명도 (0.0-1.0).
        bg_color: 배경색 RGB 튜플.
    """

    def __init__(
        self,
        image_size: int = 256,
        border_thickness: int = 2,
        font_scale: float = 0.35,
        alpha: float = 0.6,
        bg_color: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        """초기화.

        Args:
            image_size: 출력 이미지 크기 (정사각형, 픽셀).
            border_thickness: 테두리 두께.
            font_scale: 폰트 크기 배율.
            alpha: 폴리곤 채우기 투명도.
            bg_color: 배경색 (R, G, B).
        """
        self.image_size = image_size
        self.border_thickness = border_thickness
        self.font_scale = font_scale
        self.alpha = alpha
        self.bg_color = bg_color

    def create_canvas(self) -> np.ndarray:
        """256x256 빈 캔버스를 생성한다.

        Args:
            없음.

        Returns:
            배경색으로 채워진 uint8 BGR 이미지 배열.
            Shape: $(256, 256, 3)$
        """
        # BGR 순서로 배경 채우기
        canvas = np.full(
            (self.image_size, self.image_size, 3),
            self.bg_color[::-1],
            dtype=np.uint8,
        )
        return canvas

    def coords_to_points(self, coords: list[int]) -> np.ndarray:
        """평탄한 좌표 리스트를 OpenCV polygon 형식 배열로 변환한다.

        좌표는 [x0, y0, x1, y1, ...] 형식의 평탄 리스트이며,
        0-255 범위를 256x256 픽셀에 직접 매핑한다.

        Args:
            coords: [x0, y0, x1, y1, ...] 형식의 정수 리스트.

        Returns:
            픽셀 좌표 배열. Shape: $(N, 1, 2)$ (OpenCV polygon 형식)
        """
        points = []
        for i in range(0, len(coords), 2):
            points.append([int(coords[i]), int(coords[i + 1])])
        return np.array(points, dtype=np.int32).reshape((-1, 1, 2))

    def draw_room_polygon(
        self,
        canvas: np.ndarray,
        coords: list[int],
        fill_color: tuple[int, int, int],
        border_color: tuple[int, int, int],
        label: Optional[str] = None,
    ) -> np.ndarray:
        """방 폴리곤을 캔버스에 그린다.

        alpha 블렌딩으로 채우기 색상을 적용하고, 테두리와 선택적 레이블을 그린다.

        Args:
            canvas: 그릴 대상 BGR 이미지 배열. Shape: $(256, 256, 3)$
            coords: [x0, y0, x1, y1, ...] 좌표 리스트 (0-255 범위).
            fill_color: 채우기 색상 (R, G, B).
            border_color: 테두리 색상 (R, G, B).
            label: 폴리곤 중심에 표시할 텍스트 (None이면 표시 안 함).

        Returns:
            폴리곤이 그려진 BGR 이미지 배열. Shape: $(256, 256, 3)$
        """
        pts = self.coords_to_points(coords)

        # 채우기: alpha 블렌딩으로 반투명 효과
        overlay = canvas.copy()
        fill_bgr = (fill_color[2], fill_color[1], fill_color[0])  # RGB → BGR
        cv2.fillPoly(overlay, [pts], fill_bgr)
        cv2.addWeighted(overlay, self.alpha, canvas, 1 - self.alpha, 0, canvas)

        # 테두리 그리기
        border_bgr = (border_color[2], border_color[1], border_color[0])
        cv2.polylines(canvas, [pts], isClosed=True, color=border_bgr, thickness=self.border_thickness)

        # 레이블 텍스트
        if label:
            cx = int(pts[:, 0, 0].mean())
            cy = int(pts[:, 0, 1].mean())
            cv2.putText(
                canvas,
                label,
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return canvas

    def draw_door_rect(
        self,
        canvas: np.ndarray,
        door: dict,
        color: tuple[int, int, int],
        border_color: tuple[int, int, int],
        label: Optional[str] = None,
    ) -> np.ndarray:
        """문(사각형)을 캔버스에 그린다.

        door 딕셔너리의 x, y, w, h 값을 픽셀에 직접 매핑한다.

        Args:
            canvas: 그릴 대상 BGR 이미지 배열. Shape: $(256, 256, 3)$
            door: {'x': int, 'y': int, 'w': int, 'h': int} 딕셔너리.
            color: 채우기 색상 (R, G, B).
            border_color: 테두리 색상 (R, G, B).
            label: 사각형 위에 표시할 텍스트 (None이면 표시 안 함).

        Returns:
            문이 그려진 BGR 이미지 배열. Shape: $(256, 256, 3)$
        """
        # door의 x,y는 중심 좌표이므로 top-left로 변환
        pt1 = (int(door["x"] - door["w"] // 2), int(door["y"] - door["h"] // 2))
        pt2 = (int(door["x"] + door["w"] // 2), int(door["y"] + door["h"] // 2))

        fill_bgr = (color[2], color[1], color[0])
        border_bgr = (border_color[2], border_color[1], border_color[0])

        # 채우기
        overlay = canvas.copy()
        cv2.rectangle(overlay, pt1, pt2, fill_bgr, -1)
        cv2.addWeighted(overlay, self.alpha, canvas, 1 - self.alpha, 0, canvas)

        # 테두리
        cv2.rectangle(canvas, pt1, pt2, border_bgr, self.border_thickness)

        if label:
            cx = (pt1[0] + pt2[0]) // 2
            cy = (pt1[1] + pt2[1]) // 2
            cv2.putText(
                canvas,
                label,
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return canvas
