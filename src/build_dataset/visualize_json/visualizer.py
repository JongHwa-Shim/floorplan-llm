"""
평면도 전체 시각화 오케스트레이터 모듈.

FloorplanLoader, RoomRenderer를 조합하여
방별 이미지, 문 통합 이미지, 전체 평면도 이미지를 생성하고 저장한다.

출력 해상도는 256x256 고정이며, 좌표는 0-255 픽셀에 직접 매핑된다.
floorplan.png 생성 시 outline이 가장 먼저 그려져 다른 요소의 배경이 된다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
from omegaconf import DictConfig

from src.build_dataset.visualize_json.renderer import RoomRenderer


class FloorplanVisualizer:
    """평면도 JSON 데이터를 256x256 이미지로 변환하는 시각화 클래스.

    설정 파일의 색상 매핑을 이용해 방 타입별로 다른 색상을 적용하며,
    방별 개별 이미지, door.png, floorplan.png를 생성한다.

    Attributes:
        cfg: OmegaConf 설정 객체 (color_map.yaml 로드 결과).
        renderer: RoomRenderer 인스턴스.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """초기화.

        Args:
            cfg: color_map.yaml에서 로드한 OmegaConf 설정 객체.
        """
        self.cfg = cfg
        vis = cfg.vis_settings
        self.renderer = RoomRenderer(
            image_size=vis.image_size,
            border_thickness=vis.border_thickness,
            font_scale=vis.font_scale,
            alpha=vis.alpha,
            bg_color=tuple(vis.background_color),
        )

    def _get_fill_color(self, room_type: str) -> tuple[int, int, int]:
        """방 타입에 맞는 채우기 색상을 반환한다.

        Args:
            room_type: 방 타입 문자열.

        Returns:
            RGB 색상 튜플.
        """
        colors = self.cfg.room_colors
        if room_type in colors:
            c = colors[room_type]
            return (c[0], c[1], c[2])
        c = self.cfg.default_fill_color
        return (c[0], c[1], c[2])

    def _get_border_color(self, room_type: str) -> tuple[int, int, int]:
        """방 타입에 맞는 테두리 색상을 반환한다.

        Args:
            room_type: 방 타입 문자열.

        Returns:
            RGB 색상 튜플.
        """
        colors = self.cfg.border_colors
        if room_type in colors:
            c = colors[room_type]
            return (c[0], c[1], c[2])
        c = self.cfg.default_border_color
        return (c[0], c[1], c[2])

    def visualize(self, floorplan: dict, output_dir: Path) -> None:
        """평면도 데이터를 시각화하여 output_dir에 저장한다.

        생성 파일:
        - {rid}_{type}.png: 방별 개별 이미지 (256x256)
        - front_door_front_door.png: 현관문 개별 이미지
        - door.png: 모든 문 통합 이미지 (256x256)
        - floorplan.png: 전체 평면도 이미지 (256x256, outline 먼저 렌더링)

        Args:
            floorplan: 평면도 JSON 딕셔너리.
            output_dir: 이미지를 저장할 디렉토리 경로.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rooms = floorplan["rooms"]

        # 1) 방별 개별 이미지 저장
        for room in rooms:
            self._save_room_image(room, output_dir)

        # 2) front_door 개별 이미지 저장
        front_door = floorplan.get("front_door")
        if front_door:
            self._save_front_door_image(front_door, output_dir)

        # 3) 모든 문 통합 이미지 저장 (door.png)
        self._save_doors_image(floorplan, output_dir)

        # 4) 전체 평면도 이미지 저장 (floorplan.png) - outline 먼저
        self._save_floorplan_image(floorplan, output_dir)

    def _save_room_image(self, room: dict, output_dir: Path) -> None:
        """방 하나를 256x256 개별 이미지로 저장한다.

        Args:
            room: {'rid', 'type', 'coords'} 딕셔너리.
            output_dir: 저장 디렉토리.
        """
        canvas = self.renderer.create_canvas()
        room_type = room["type"]

        self.renderer.draw_room_polygon(
            canvas,
            room["coords"],
            self._get_fill_color(room_type),
            self._get_border_color(room_type),
            label=room_type,
        )

        cv2.imwrite(str(output_dir / f"{room['rid']}_{room_type}.png"), canvas)

    def _save_front_door_image(self, front_door: dict, output_dir: Path) -> None:
        """front_door를 방으로 취급하여 256x256 개별 이미지로 저장한다.

        Args:
            front_door: {'x', 'y', 'w', 'h'} 딕셔너리.
            output_dir: 저장 디렉토리.
        """
        canvas = self.renderer.create_canvas()

        self.renderer.draw_door_rect(
            canvas,
            front_door,
            tuple(self.cfg.front_door_color),
            tuple(self.cfg.door_border_color),
            label="front_door",
        )

        cv2.imwrite(str(output_dir / "front_door_front_door.png"), canvas)

    def _save_doors_image(self, floorplan: dict, output_dir: Path) -> None:
        """모든 문(edges의 door들만, front door 제외)을 256x256 이미지 하나에 저장한다.

        Args:
            floorplan: 평면도 JSON 딕셔너리.
            output_dir: 저장 디렉토리.
        """
        canvas = self.renderer.create_canvas()
        door_color = tuple(self.cfg.door_color)
        door_border = tuple(self.cfg.door_border_color)

        for edge in floorplan.get("edges", []):
            for door in edge.get("doors", []):
                self.renderer.draw_door_rect(canvas, door, door_color, door_border)

        # front_door = floorplan.get("front_door")
        # if front_door:
        #     self.renderer.draw_door_rect(
        #         canvas,
        #         front_door,
        #         tuple(self.cfg.front_door_color),
        #         door_border,
        #         label="front_door",
        #     )

        cv2.imwrite(str(output_dir / "door.png"), canvas)

    def _save_floorplan_image(self, floorplan: dict, output_dir: Path) -> None:
        """모든 방과 문을 256x256 이미지 하나로 합쳐 저장한다.

        렌더링 순서:
        1. outline (배경 역할, 가장 먼저 그려 다른 요소의 밑에 깔림)
        2. outline을 제외한 나머지 방들
        3. edges의 문들
        4. front_door

        Args:
            floorplan: 평면도 JSON 딕셔너리.
            output_dir: 저장 디렉토리.
        """
        canvas = self.renderer.create_canvas()
        rooms = floorplan["rooms"]

        # outline을 먼저 찾아 그리기 (배경 역할)
        outline_rooms = [r for r in rooms if r["type"] == "outline"]
        other_rooms = [r for r in rooms if r["type"] != "outline"]

        for room in outline_rooms:
            self.renderer.draw_room_polygon(
                canvas,
                room["coords"],
                self._get_fill_color(room["type"]),
                self._get_border_color(room["type"]),
                label=room["type"],
            )

        # 나머지 방 그리기 (outline 위에 overlap)
        for room in other_rooms:
            room_type = room["type"]
            self.renderer.draw_room_polygon(
                canvas,
                room["coords"],
                self._get_fill_color(room_type),
                self._get_border_color(room_type),
                label=room_type,
            )

        # 문 그리기
        door_color = tuple(self.cfg.door_color)
        door_border = tuple(self.cfg.door_border_color)
        for edge in floorplan.get("edges", []):
            for door in edge.get("doors", []):
                self.renderer.draw_door_rect(canvas, door, door_color, door_border)

        # front_door 그리기
        front_door = floorplan.get("front_door")
        if front_door:
            self.renderer.draw_door_rect(
                canvas,
                front_door,
                tuple(self.cfg.front_door_color),
                door_border,
                label="front_door",
            )

        cv2.imwrite(str(output_dir / "floorplan.png"), canvas)
