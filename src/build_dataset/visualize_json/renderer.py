"""
평면도 요소 렌더링 모듈.

방, 문, 현관문을 OpenCV로 N×N 이미지에 그린다.
좌표는 0-255 범위로 ``image_size`` 픽셀에 직접 매핑된다.

Mod Record:
    2026-05-15a:
        - 텍스트 라벨 렌더링을 draw_room_polygon / draw_door_rect 에서 분리해 별도
          draw_label_at / draw_room_label / draw_door_label 메서드로 옮겼다. z-order 문제
          (방 A 텍스트가 방 B 의 alpha 채우기에 가려짐) 해결.
        - 라벨 위치: polygon 중심 → ``min(xs)`` + ``mean(ys)`` (가장 왼쪽에서 시작).
    2026-05-15c:
        - 텍스트도 ``cv2.LINE_8`` 로 통일 (사용자 선호). 도형·텍스트 모두 픽셀 단위 단색.
          작은 폰트 에서 글자 가장자리가 다소 거칠지만 sharp 톤이 일관됨.

    2026-05-15b:
        - 직교 도형(폴리곤·사각형·테두리) 의 픽셀 깨짐 해결.
          원인: 첫 변경 (z-order 분리 시) 에 ``cv2.polylines(..., lineType=cv2.LINE_AA)`` 를
          넣은 것. LINE_AA 는 가장자리에 부분 투명 픽셀을 만들어 alpha 블렌딩과 결합되면
          색이 흐림. 옛 코드는 lineType 미지정 (= 기본 LINE_8) 이었고 alpha 0.6 블렌딩만으로도
          픽셀 단위 단색이 나와 깨끗했다.
        - 해결: 도형(polylines / rectangle / fillPoly) 은 ``LINE_8`` 로 통일.
          텍스트(``cv2.putText``) 는 작은 폰트 가독성을 위해 ``LINE_AA`` 유지.
        - SSAA (super-sampling anti-aliasing) 옵션 유지하되 기본 비활성화 (supersample=1).
          1024 → 256 INTER_AREA 다운샘플 평균이 좌표 정렬에 따라 가장자리에 부분 평균을
          남기므로, "픽셀 깨짐 X" 최우선 시 supersample=1 이 정답.
          ``supersample > 1`` 은 텍스트 가독성을 더 부드럽게 만들고 싶을 때 선택지로 남김.
        - 좌표·두께·폰트 크기 곱셈은 모두 renderer 내부에서 자동 처리되어 visualizer 측
          호출부는 base 좌표 그대로 전달하면 된다.
"""

from __future__ import annotations

import cv2
import numpy as np


class RoomRenderer:
    """방·문 요소를 N×N 이미지에 렌더링하는 클래스 (SSAA 지원).

    내부 캔버스는 ``image_size * supersample`` 크기이며 ``finalize_canvas`` 가 호출되어야
    최종 ``image_size`` 로 다운샘플된다. 모든 draw_* 메서드는 base 좌표(0-255)를 받고
    내부에서 supersample 배율을 자동 곱셈한다.

    Attributes:
        image_size: 최종 출력 해상도 (정사각형).
        supersample: 내부 SSAA 배율 (1 = SSAA 비활성).
        internal_size: 실제 내부 캔버스 크기 (= image_size × supersample).
        border_thickness: 사용자가 명시한 base 테두리 두께. 내부에서 supersample 곱셈됨.
        font_scale: 사용자가 명시한 base 폰트 크기. 내부에서 supersample 곱셈됨.
        alpha: 채우기 투명도 (0.0-1.0).
        bg_color: 배경색 RGB 튜플.
        label_color: 레이블 텍스트 색 RGB 튜플.
        label_thickness: 텍스트 stroke 두께. 내부에서 supersample 곱셈됨.
    """

    def __init__(
        self,
        image_size: int = 256,
        supersample: int = 1,
        border_thickness: int = 1,
        font_scale: float = 0.35,
        alpha: float = 0.6,
        bg_color: tuple[int, int, int] = (255, 255, 255),
        label_color: tuple[int, int, int] = (20, 20, 20),
        label_thickness: int = 1,
    ) -> None:
        """초기화.

        Args:
            image_size: 최종 출력 이미지 크기 (정사각형, 픽셀).
            supersample: SSAA 배율 (1 ~ 8). 4 권장.
            border_thickness: base 테두리 두께. 내부에서 supersample 배 곱셈.
            font_scale: base 폰트 크기. 내부에서 supersample 배 곱셈.
            alpha: 폴리곤 채우기 투명도.
            bg_color: 배경색 (R, G, B).
            label_color: 레이블 색 (R, G, B).
            label_thickness: 텍스트 stroke 두께.

        Raises:
            ValueError: ``supersample`` < 1.
        """
        if supersample < 1:
            raise ValueError(f"supersample 은 1 이상이어야 합니다 (받음: {supersample})")
        self.image_size = image_size
        self.supersample = supersample
        self.internal_size = image_size * supersample
        self.border_thickness = border_thickness
        self.font_scale = font_scale
        self.alpha = alpha
        self.bg_color = bg_color
        self.label_color = label_color
        self.label_thickness = label_thickness

    # ------------------------------------------------------------------
    # 캔버스 라이프사이클
    # ------------------------------------------------------------------

    def create_canvas(self) -> np.ndarray:
        """내부 SSAA 캔버스를 생성한다.

        Returns:
            배경색으로 채워진 uint8 BGR 이미지 배열.
            Shape: $(image\\_size \\times supersample, image\\_size \\times supersample, 3)$
        """
        canvas = np.full(
            (self.internal_size, self.internal_size, 3),
            self.bg_color[::-1],
            dtype=np.uint8,
        )
        return canvas

    def finalize_canvas(self, canvas: np.ndarray) -> np.ndarray:
        """SSAA 캔버스를 최종 ``image_size`` 로 다운샘플한다.

        INTER_AREA 는 영역 평균으로 다운샘플하므로 super-sampled 픽셀들이 자연스럽게 평균되어
        테두리·텍스트가 부드럽게 정렬된다.

        Args:
            canvas: 내부 SSAA 캔버스.

        Returns:
            ``(image_size, image_size, 3)`` uint8 BGR 이미지.
        """
        if self.supersample == 1:
            return canvas
        return cv2.resize(
            canvas, (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )

    # ------------------------------------------------------------------
    # 내부 좌표 변환
    # ------------------------------------------------------------------

    def coords_to_points(self, coords: list[int]) -> np.ndarray:
        """평탄 좌표를 OpenCV polygon 형식으로 변환 (supersample 배율 자동 적용).

        Args:
            coords: [x0, y0, x1, y1, ...] base 좌표 (0-255).

        Returns:
            Shape: $(N, 1, 2)$ int32 픽셀 좌표 배열.
        """
        s = self.supersample
        points = []
        # range 상한을 len-1 로 두어 홀수 길이 coords(파싱 오류/garbage 출력)에서 마지막 미쌍
        # 좌표를 안전하게 무시한다 (IndexError 방지).
        for i in range(0, len(coords) - 1, 2):
            points.append([int(coords[i]) * s, int(coords[i + 1]) * s])
        return np.array(points, dtype=np.int32).reshape((-1, 1, 2))

    def points_from_xy(self, xy: list[tuple[float, float]]) -> np.ndarray:
        """(x, y) 튜플 리스트 → OpenCV polygon 형식 (supersample 적용).

        겹침 영역 (shapely intersection) 의 exterior 좌표를 채울 때 사용한다.
        """
        s = self.supersample
        pts = [[int(round(x)) * s, int(round(y)) * s] for x, y in xy]
        return np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

    # ------------------------------------------------------------------
    # 도형 그리기
    # ------------------------------------------------------------------

    def fill_polygon_solid(
        self,
        canvas: np.ndarray,
        coords: list[int],
        fill_color: tuple[int, int, int],
    ) -> np.ndarray:
        """방 폴리곤을 **solid (불투명)** 로 채운다. alpha 블렌딩 없음 → 색 왜곡 없음."""
        pts = self.coords_to_points(coords)
        fill_bgr = (fill_color[2], fill_color[1], fill_color[0])
        cv2.fillPoly(canvas, [pts], fill_bgr, lineType=cv2.LINE_8)
        return canvas

    def fill_region_solid(
        self,
        canvas: np.ndarray,
        xy: list[tuple[float, float]],
        fill_color: tuple[int, int, int],
    ) -> np.ndarray:
        """임의 (x, y) 폴리곤 영역을 solid 로 채운다 (겹침 블렌딩 색 도포용)."""
        if len(xy) < 3:
            return canvas
        pts = self.points_from_xy(xy)
        fill_bgr = (fill_color[2], fill_color[1], fill_color[0])
        cv2.fillPoly(canvas, [pts], fill_bgr, lineType=cv2.LINE_8)
        return canvas

    def draw_polygon_border(
        self,
        canvas: np.ndarray,
        coords: list[int],
        border_color: tuple[int, int, int],
    ) -> np.ndarray:
        """방 폴리곤 **테두리만** 그린다 (최상단 재도색용)."""
        pts = self.coords_to_points(coords)
        border_bgr = (border_color[2], border_color[1], border_color[0])
        thickness = self.border_thickness * self.supersample
        cv2.polylines(canvas, [pts], isClosed=True, color=border_bgr,
                      thickness=thickness, lineType=cv2.LINE_8)
        return canvas

    def draw_room_polygon(
        self,
        canvas: np.ndarray,
        coords: list[int],
        fill_color: tuple[int, int, int],
        border_color: tuple[int, int, int],
    ) -> np.ndarray:
        """방 폴리곤(채우기 + 테두리) 만 그린다.

        Args:
            canvas: SSAA 캔버스.
            coords: base 좌표 [x0, y0, ...] (0-255).
            fill_color: 채우기 색상 (R, G, B).
            border_color: 테두리 색상 (R, G, B).

        Returns:
            그려진 canvas (in-place 수정 + 반환).
        """
        pts = self.coords_to_points(coords)

        # 채우기 (alpha blending)
        overlay = canvas.copy()
        fill_bgr = (fill_color[2], fill_color[1], fill_color[0])
        cv2.fillPoly(overlay, [pts], fill_bgr, lineType=cv2.LINE_8)
        cv2.addWeighted(overlay, self.alpha, canvas, 1 - self.alpha, 0, canvas)

        # 테두리 — LINE_8 (anti-alias 없음, SSAA 가 대체)
        border_bgr = (border_color[2], border_color[1], border_color[0])
        thickness = self.border_thickness * self.supersample
        cv2.polylines(
            canvas, [pts], isClosed=True,
            color=border_bgr, thickness=thickness,
            lineType=cv2.LINE_8,
        )
        return canvas

    def draw_door_rect(
        self,
        canvas: np.ndarray,
        door: dict,
        color: tuple[int, int, int],
        border_color: tuple[int, int, int],
    ) -> np.ndarray:
        """문(사각형 채우기 + 테두리) 만 그린다 (라벨 X)."""
        s = self.supersample
        pt1 = (int(door["x"] - door["w"] // 2) * s, int(door["y"] - door["h"] // 2) * s)
        pt2 = (int(door["x"] + door["w"] // 2) * s, int(door["y"] + door["h"] // 2) * s)

        fill_bgr = (color[2], color[1], color[0])
        border_bgr = (border_color[2], border_color[1], border_color[0])

        overlay = canvas.copy()
        cv2.rectangle(overlay, pt1, pt2, fill_bgr, -1, lineType=cv2.LINE_8)
        cv2.addWeighted(overlay, self.alpha, canvas, 1 - self.alpha, 0, canvas)

        thickness = self.border_thickness * self.supersample
        cv2.rectangle(canvas, pt1, pt2, border_bgr, thickness, lineType=cv2.LINE_8)
        return canvas

    # ------------------------------------------------------------------
    # 라벨 (z-order 분리)
    # ------------------------------------------------------------------

    def draw_label_at(
        self,
        canvas: np.ndarray,
        text: str,
        org: tuple[int, int],
    ) -> np.ndarray:
        """SSAA 캔버스에 텍스트를 그린다. org 는 base 좌표.

        SSAA 캔버스에 큰 폰트로 그린 뒤 다운샘플되므로 텍스트가 자연스럽게 다듬어진다.
        cv2.putText 의 thickness/font_scale 도 supersample 배 적용.
        """
        text_bgr = (self.label_color[2], self.label_color[1], self.label_color[0])
        s = self.supersample
        org_internal = (org[0] * s, org[1] * s)
        # 텍스트도 도형과 동일하게 LINE_8 — 픽셀 단위 단색 보장. 사용자 선호 (2026-05-15c).
        # 작은 폰트(scale=0.35) 에서 글자 가장자리가 다소 거칠지만 sharp 톤이 일관됨.
        cv2.putText(
            canvas, text, org_internal,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale * s,
            text_bgr,
            max(1, self.label_thickness * s),
            cv2.LINE_8,
        )
        return canvas

    def draw_room_label(
        self,
        canvas: np.ndarray,
        coords: list[int],
        label: str,
        x_offset: int = 2,
    ) -> np.ndarray:
        """방 폴리곤의 가장 왼쪽 x + 평균 y 위치에 라벨을 그린다.

        Args:
            canvas: SSAA 캔버스.
            coords: base 좌표.
            label: 표시 문자열.
            x_offset: ``min(xs)`` 에서 안쪽으로 들여쓸 base 픽셀.
        """
        # 좌표는 base 단위에서 계산 후 draw_label_at 가 supersample 배 변환
        xs = [int(coords[i]) for i in range(0, len(coords), 2)]
        ys = [int(coords[i]) for i in range(1, len(coords), 2)]
        x_min = min(xs)
        y_mean = int(sum(ys) / len(ys))
        return self.draw_label_at(canvas, label, (x_min + x_offset, y_mean))

    def draw_door_label(
        self,
        canvas: np.ndarray,
        door: dict,
        label: str,
    ) -> np.ndarray:
        """문 사각형의 가장 왼쪽 x + 중심 y 위치에 라벨을 그린다."""
        x_left = int(door["x"] - door["w"] // 2)
        y_center = int(door["y"])
        return self.draw_label_at(canvas, label, (x_left + 1, y_center))
