"""RPLAN PNG 채널 파싱 모듈.

RPLAN 평면도 PNG 파일을 4채널로 로드하고,
각 채널을 의미별 2D 배열로 분리한다.

PNG는 RGBA 순서로 저장되며, cv2.IMREAD_UNCHANGED로 읽으면 채널 순서 그대로 유지:
    - channel 0 (R): 방 인스턴스 ID (0=non-room, 1~N=각 방)
    - channel 1 (G): 공간 타입 레이블 (0~17)
    - channel 2 (B): 구조 레이블 (Exterior wall=127, Front door=255, Other=0)
    - channel 3 (A): 영역 구분 (Exterior=0, Interior=255)

노션 문서에서 "BGRA"는 의미적 레이블(B=구조, G=타입, R=인스턴스, A=영역)을 뜻하며,
실제 PNG 저장 순서는 R,G,B,A이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ChannelData:
    """채널을 분리한 결과 컨테이너.

    Attributes:
        structure: 구조 레이블 (Exterior wall=127, Front door=255).
        space_type: 공간 타입 (0~17).
        room_instance: 방 인스턴스 ID (0=non-room, 1~N).
        area_mask: 영역 구분 (Exterior=0, Interior=255).
        height: 이미지 높이 (픽셀).
        width: 이미지 너비 (픽셀).

    Shape:
        각 채널: $(H, W)$, dtype uint8.
    """

    structure: np.ndarray
    space_type: np.ndarray
    room_instance: np.ndarray
    area_mask: np.ndarray
    height: int
    width: int


def load_bgra_image(png_path: str) -> np.ndarray:
    """PNG 파일을 4채널 numpy 배열로 로드.

    Args:
        png_path: PNG 파일 절대/상대 경로.

    Returns:
        4채널 이미지 배열.

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때.
        ValueError: 이미지가 4채널이 아닐 때.

    Shape:
        출력: $(H, W, 4)$, dtype uint8.
    """
    image = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"PNG 파일을 읽을 수 없음: {png_path}")
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(
            f"4채널 이미지가 아님: shape={image.shape}, path={png_path}"
        )
    return image


def parse_channels(image: np.ndarray) -> ChannelData:
    """4채널 이미지를 의미별 2D 배열로 분리.

    PNG 저장 순서(R,G,B,A)에 맞춰 채널을 올바르게 매핑:
    - ch0 (R) → room_instance
    - ch1 (G) → space_type
    - ch2 (B) → structure
    - ch3 (A) → area_mask

    Args:
        image: 4채널 이미지 배열.

    Returns:
        ChannelData — 각 채널이 의미별로 분리된 결과.

    Shape:
        입력: $(H, W, 4)$.
        출력 각 채널: $(H, W)$.
    """
    h, w = image.shape[:2]
    return ChannelData(
        room_instance=image[:, :, 0],  # ch0 = R = 방 인스턴스 ID
        space_type=image[:, :, 1],     # ch1 = G = 공간 타입
        structure=image[:, :, 2],      # ch2 = B = 구조 레이블
        area_mask=image[:, :, 3],      # ch3 = A = 영역 구분
        height=h,
        width=w,
    )
