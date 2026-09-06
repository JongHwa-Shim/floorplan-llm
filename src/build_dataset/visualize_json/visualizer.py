"""
평면도 전체 시각화 오케스트레이터 모듈.

FloorplanLoader, RoomRenderer를 조합하여
방별 이미지, 문 통합 이미지, 전체 평면도 이미지를 생성하고 저장한다.

출력 해상도는 256x256 고정이며, 좌표는 0-255 픽셀에 직접 매핑된다.
floorplan.png 생성 시 outline 이 가장 먼저 그려져 다른 요소의 배경이 된다.

Mod Record (2026-05-15):
    - show_labels 옵션 신설 (생성자 인자 + cfg.vis_settings.show_labels 모두 지원).
      추론 결과 시각화에서는 라벨이 불필요하므로 result_saver 가 show_labels=False 로 호출 가능.
    - z-order 분리: 도형(채우기 + 테두리) 전체를 먼저 그리고, 마지막에 모든 라벨을
      한꺼번에 그린다. 이전 구조에서는 방 A 의 라벨이 방 B 의 alpha 채우기에 가려졌다.
    - 라벨 위치: polygon 중심 → polygon 의 ``min(xs)`` + ``mean(ys)``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
from omegaconf import DictConfig

from src.build_dataset.visualize_json.renderer import RoomRenderer


class FloorplanVisualizer:
    """평면도 JSON 데이터를 256x256 이미지로 변환하는 시각화 클래스.

    Attributes:
        cfg: OmegaConf 설정 객체 (color_map.yaml 로드 결과).
        renderer: RoomRenderer 인스턴스.
        show_labels: True 면 마지막 단계에서 라벨을 그린다. False 면 도형만.
    """

    def __init__(self, cfg: DictConfig, show_labels: bool | None = None,
                 skip_interior_doors: bool = False) -> None:
        """초기화.

        Args:
            cfg: color_map.yaml 에서 로드한 OmegaConf 설정 객체.
            show_labels: 라벨 그리기 여부. None 이면 ``cfg.vis_settings.show_labels`` 를
                조회 (없으면 기본 True).
            skip_interior_doors: True 면 interior door 사각형을 그리지 않는다.
                (front_door 는 영향 안 받음.) DS2D 처럼 door 를 생성하지 않는 baseline 과의
                fair FID 비교를 위해 GT 측에서도 door 표현을 빼야 할 때 사용.
        """
        self.cfg = cfg
        self.skip_interior_doors = skip_interior_doors
        vis = cfg.vis_settings

        renderer_kwargs = dict(
            image_size=vis.image_size,
            border_thickness=vis.border_thickness,
            font_scale=vis.font_scale,
            alpha=vis.alpha,
            bg_color=tuple(vis.background_color),
        )
        # 선택 옵션: SSAA / label color / label thickness
        if "supersample" in vis:
            renderer_kwargs["supersample"] = int(vis.supersample)
        if "label_color" in vis:
            renderer_kwargs["label_color"] = tuple(vis.label_color)
        if "label_thickness" in vis:
            renderer_kwargs["label_thickness"] = int(vis.label_thickness)
        self.renderer = RoomRenderer(**renderer_kwargs)

        if show_labels is None:
            show_labels = bool(vis.get("show_labels", True))
        self.show_labels = show_labels

    # ------------------------------------------------------------------
    # color helpers
    # ------------------------------------------------------------------

    def _get_fill_color(self, room_type: str) -> tuple[int, int, int]:
        colors = self.cfg.room_colors
        if room_type in colors:
            c = colors[room_type]
            return (c[0], c[1], c[2])
        c = self.cfg.default_fill_color
        return (c[0], c[1], c[2])

    def _get_border_color(self, room_type: str) -> tuple[int, int, int]:
        colors = self.cfg.border_colors
        if room_type in colors:
            c = colors[room_type]
            return (c[0], c[1], c[2])
        c = self.cfg.default_border_color
        return (c[0], c[1], c[2])

    # ------------------------------------------------------------------
    # 내부 헬퍼: 단일 canvas 에 전체 floorplan 을 합성
    # ------------------------------------------------------------------

    def _render_floorplan_canvas(self, floorplan: dict):
        """모든 도형 → 라벨 순서로 SSAA 캔버스에 그린 뒤 base 해상도로 다운샘플한다.

        Returns:
            합성 + 다운샘플 완료된 BGR canvas (uint8, image_size × image_size).
        """
        canvas = self.renderer.create_canvas()
        rooms = floorplan["rooms"]

        outline_rooms = [r for r in rooms if r["type"] == "outline"]
        other_rooms = [r for r in rooms if r["type"] != "outline"]

        # 1) 도형 렌더링 (라벨 없이) — z-order 보존
        for room in outline_rooms:
            self.renderer.draw_room_polygon(
                canvas, room["coords"],
                self._get_fill_color(room["type"]),
                self._get_border_color(room["type"]),
            )
        for room in other_rooms:
            self.renderer.draw_room_polygon(
                canvas, room["coords"],
                self._get_fill_color(room["type"]),
                self._get_border_color(room["type"]),
            )

        door_color = tuple(self.cfg.door_color)
        door_border = tuple(self.cfg.door_border_color)
        if not self.skip_interior_doors:
            for edge in floorplan.get("edges", []) or []:
                for door in edge.get("doors", []) or []:
                    self.renderer.draw_door_rect(canvas, door, door_color, door_border)

        front_door = floorplan.get("front_door")
        if front_door:
            self.renderer.draw_door_rect(
                canvas, front_door,
                tuple(self.cfg.front_door_color),
                door_border,
            )

        # 2) 라벨은 모든 도형 위에 (z-order top)
        if self.show_labels:
            for room in outline_rooms:
                self.renderer.draw_room_label(canvas, room["coords"], room["type"])
            for room in other_rooms:
                self.renderer.draw_room_label(canvas, room["coords"], room["type"])
            if front_door:
                self.renderer.draw_door_label(canvas, front_door, "front_door")

        # 3) SSAA → base 해상도 다운샘플
        return self.renderer.finalize_canvas(canvas)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def visualize(self, floorplan: dict, output_dir: Path) -> None:
        """평면도 데이터를 시각화하여 output_dir 에 저장한다.

        생성 파일:
            - {rid}_{type}.png: 방별 개별 이미지 (256x256)
            - front_door_front_door.png: 현관문 개별 이미지
            - door.png: 모든 문 통합 이미지
            - floorplan.png: 전체 평면도 이미지
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rooms = floorplan["rooms"]
        for room in rooms:
            self._save_room_image(room, output_dir)

        front_door = floorplan.get("front_door")
        if front_door:
            self._save_front_door_image(front_door, output_dir)

        self._save_doors_image(floorplan, output_dir)

        canvas = self._render_floorplan_canvas(floorplan)
        cv2.imwrite(str(output_dir / "floorplan.png"), canvas)

    def render_floorplan_to_path(self, floorplan: dict, out_path: Path) -> None:
        """전체 평면도 PNG 한 장만 임의 경로에 저장 (실험 파이프라인 통일 렌더용)."""
        canvas = self._render_floorplan_canvas(floorplan)
        cv2.imwrite(str(out_path), canvas)

    # ------------------------------------------------------------------
    # 개별 PNG 들 (방별 / front_door / doors) — show_labels 동일 적용
    # ------------------------------------------------------------------

    def _save_room_image(self, room: dict, output_dir: Path) -> None:
        canvas = self.renderer.create_canvas()
        room_type = room["type"]
        self.renderer.draw_room_polygon(
            canvas, room["coords"],
            self._get_fill_color(room_type),
            self._get_border_color(room_type),
        )
        if self.show_labels:
            self.renderer.draw_room_label(canvas, room["coords"], room_type)
        canvas = self.renderer.finalize_canvas(canvas)
        cv2.imwrite(str(output_dir / f"{room['rid']}_{room_type}.png"), canvas)

    def _save_front_door_image(self, front_door: dict, output_dir: Path) -> None:
        canvas = self.renderer.create_canvas()
        self.renderer.draw_door_rect(
            canvas, front_door,
            tuple(self.cfg.front_door_color),
            tuple(self.cfg.door_border_color),
        )
        if self.show_labels:
            self.renderer.draw_door_label(canvas, front_door, "front_door")
        canvas = self.renderer.finalize_canvas(canvas)
        cv2.imwrite(str(output_dir / "front_door_front_door.png"), canvas)

    def _save_doors_image(self, floorplan: dict, output_dir: Path) -> None:
        canvas = self.renderer.create_canvas()
        door_color = tuple(self.cfg.door_color)
        door_border = tuple(self.cfg.door_border_color)
        for edge in floorplan.get("edges", []) or []:
            for door in edge.get("doors", []) or []:
                self.renderer.draw_door_rect(canvas, door, door_color, door_border)
        canvas = self.renderer.finalize_canvas(canvas)
        cv2.imwrite(str(output_dir / "door.png"), canvas)
