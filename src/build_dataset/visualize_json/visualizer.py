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

    def _collect_fill_items(self, floorplan: dict) -> tuple[list[dict], list[dict]]:
        """outline 방 리스트 + 채우기 요소(방 non-outline + door) 리스트를 구성한다.

        raster (`_render_floorplan_canvas`) 와 vector (`render_floorplan_to_vector`) 가
        동일한 색·요소 구성을 공유하도록 분리한 공용 헬퍼.

        door (interior + front) 를 방과 동일한 "채우기 요소" 로 통합한다 (2026-07-06 정책):
        door 도 방처럼 solid 채우기 + 테두리 최상단 재도색 대상.
        단 **겹침 색상 블렌딩에서는 door 를 제외**한다 (2026-07-14): door 는 방 위에 solid 색으로
        선명하게 보여야 하므로, 각 fill_item 에 ``is_door`` 플래그를 달아 `_compute_blend_regions`
        가 door 를 건너뛴다. (방-방 겹침만 블렌딩.)

        Args:
            floorplan: {"rooms", "edges", "front_door"} row-oriented dict.

        Returns:
            (outline_rooms, fill_items):
                - outline_rooms: type=="outline" 인 원본 room dict 리스트.
                - fill_items: [{"coords", "fill", "border", "is_door"}] — 방(outline 제외) + door.
        """
        rooms = floorplan["rooms"]
        outline_rooms = [r for r in rooms if r["type"] == "outline"]
        other_rooms = [r for r in rooms if r["type"] != "outline"]

        door_fill = tuple(self.cfg.door_color)
        fd_fill = tuple(self.cfg.front_door_color)
        door_border = tuple(self.cfg.door_border_color)
        door_items: list[dict] = []
        if not self.skip_interior_doors:
            for edge in floorplan.get("edges", []) or []:
                for door in edge.get("doors", []) or []:
                    door_items.append({"coords": self._door_to_coords(door),
                                       "fill": door_fill, "border": door_border, "is_door": True})
        front_door = floorplan.get("front_door")
        if front_door:
            door_items.append({"coords": self._door_to_coords(front_door),
                               "fill": fd_fill, "border": door_border, "is_door": True})

        fill_items = [
            {"coords": r["coords"], "fill": self._get_fill_color(r["type"]),
             "border": self._get_border_color(r["type"]), "is_door": False}
            for r in other_rooms
        ] + door_items
        return outline_rooms, fill_items

    def _render_floorplan_canvas(self, floorplan: dict):
        """모든 도형 → 라벨 순서로 SSAA 캔버스에 그린 뒤 base 해상도로 다운샘플한다.

        Returns:
            합성 + 다운샘플 완료된 BGR canvas (uint8, image_size × image_size).
        """
        canvas = self.renderer.create_canvas()
        outline_rooms, fill_items = self._collect_fill_items(floorplan)
        room_items = [it for it in fill_items if not it.get("is_door")]
        door_items = [it for it in fill_items if it.get("is_door")]
        other_rooms = [r for r in floorplan["rooms"] if r["type"] != "outline"]
        front_door = floorplan.get("front_door")

        # 1) solid 채우기 — outline → 방 (door 제외).
        for room in outline_rooms:
            self.renderer.fill_polygon_solid(
                canvas, room["coords"], self._get_fill_color(room["type"]))
        for it in room_items:
            self.renderer.fill_polygon_solid(canvas, it["coords"], it["fill"])

        # 2) 방-방 겹침만 색 평균 블렌딩 (door 는 제외 — _compute_blend_regions 가 is_door skip).
        self._blend_overlap_regions(canvas, room_items)

        # 3) 방·outline 테두리 재도색.
        for room in outline_rooms:
            self.renderer.draw_polygon_border(
                canvas, room["coords"], self._get_border_color(room["type"]))
        for it in room_items:
            self.renderer.draw_polygon_border(canvas, it["coords"], it["border"])

        # 4) 현관문·interior door 를 **맨 위에** solid(불투명)로 채우고 테두리 그린다 (2026-07-16):
        #    방·블렌딩·방테두리 등 아래 요소를 전부 100% 덮는다. door 는 블렌딩 예외라 원색 유지.
        for it in door_items:
            self.renderer.fill_polygon_solid(canvas, it["coords"], it["fill"])
        for it in door_items:
            self.renderer.draw_polygon_border(canvas, it["coords"], it["border"])

        # 5) 라벨은 모든 도형 위에 (z-order top)
        if self.show_labels:
            for room in outline_rooms:
                self.renderer.draw_room_label(canvas, room["coords"], room["type"])
            for room in other_rooms:
                self.renderer.draw_room_label(canvas, room["coords"], room["type"])
            if front_door:
                self.renderer.draw_door_label(canvas, front_door, "front_door")

        # 5) SSAA → base 해상도 다운샘플
        return self.renderer.finalize_canvas(canvas)

    def render_floorplan_to_vector(self, floorplan: dict, out_paths) -> None:
        """평면도를 벡터(SVG/PDF 등)로 렌더링해 out_paths 각 경로에 저장한다.

        OpenCV raster (`_render_floorplan_canvas`) 와 **동일한 색·겹침 블렌딩·테두리 로직**을
        matplotlib patch 로 재현하되, 벡터 출력이라 어떤 배율로 확대해도 픽셀이 깨지지 않는다.
        FID 등 정량 지표는 raster 를 쓰므로 이 메서드는 사람이 보는 figure 전용(병행 출력).

        좌표계는 원본 0-based y-down(이미지) 이므로 y 축을 반전(invert)해 raster 와 동일 방향·
        프레이밍(0~image_size 정사각 프레임) 을 맞춘다. 포맷은 각 out_path 의 확장자로
        결정된다(.svg / .pdf 등, matplotlib backend 자동). figure 는 1회만 생성해 여러 포맷에 저장.

        Args:
            floorplan: {"rooms", "edges", "front_door"} row-oriented dict (raster 와 동일 입력).
            out_paths: 저장할 경로 리스트(또는 단일 Path). 확장자로 포맷 결정.

        Shape:
            내부 좌표계 $(0, image\\_size) \\times (0, image\\_size)$, y 반전.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon

        if isinstance(out_paths, (str, Path)):
            out_paths = [out_paths]

        size = self.renderer.image_size

        def _rgb01(c):
            return (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)

        def _xy(coords):
            return [(coords[i], coords[i + 1]) for i in range(0, len(coords) - 1, 2)]

        outline_rooms, fill_items = self._collect_fill_items(floorplan)
        room_items = [it for it in fill_items if not it.get("is_door")]
        door_items = [it for it in fill_items if it.get("is_door")]
        other_rooms = [r for r in floorplan["rooms"] if r["type"] != "outline"]
        front_door = floorplan.get("front_door")
        bg = _rgb01(tuple(self.cfg.vis_settings.background_color))
        # 테두리 두께 — 벡터는 point 단위. base border_thickness(px) 를 근사 point 로 사용.
        line_width = max(0.6, float(self.renderer.border_thickness))

        # 여백 없이 캔버스를 꽉 채워(add_axes[0,0,1,1]) raster 프레이밍과 일치시킨다.
        fig = plt.figure(figsize=(size / 100.0, size / 100.0), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.invert_yaxis()                      # 이미지 좌표계(y-down) 재현
        ax.set_aspect("equal")
        ax.set_axis_off()

        # 1) solid 채우기 (테두리 없이) — outline → 방 (door 제외)
        for room in outline_rooms:
            ax.add_patch(MplPolygon(_xy(room["coords"]), closed=True,
                                     facecolor=_rgb01(self._get_fill_color(room["type"])),
                                     edgecolor="none", antialiased=True))
        for it in room_items:
            ax.add_patch(MplPolygon(_xy(it["coords"]), closed=True,
                                     facecolor=_rgb01(it["fill"]),
                                     edgecolor="none", antialiased=True))

        # 2) 방-방 겹침만 평균색 블렌딩 (door 제외)
        for ext, blend in self._compute_blend_regions(room_items):
            ax.add_patch(MplPolygon(ext, closed=True, facecolor=_rgb01(blend),
                                     edgecolor="none", antialiased=True))

        # 3) 방·outline 테두리 재도색
        for room in outline_rooms:
            ax.add_patch(MplPolygon(_xy(room["coords"]), closed=True, fill=False,
                                     edgecolor=_rgb01(self._get_border_color(room["type"])),
                                     linewidth=line_width, joinstyle="miter"))
        for it in room_items:
            ax.add_patch(MplPolygon(_xy(it["coords"]), closed=True, fill=False,
                                     edgecolor=_rgb01(it["border"]),
                                     linewidth=line_width, joinstyle="miter"))

        # 4) 현관문·interior door 를 맨 위에 solid(불투명) 채우기 + 테두리 — 아래 요소 전부 덮음, 블렌딩 예외
        for it in door_items:
            ax.add_patch(MplPolygon(_xy(it["coords"]), closed=True,
                                     facecolor=_rgb01(it["fill"]), edgecolor="none", antialiased=True))
        for it in door_items:
            ax.add_patch(MplPolygon(_xy(it["coords"]), closed=True, fill=False,
                                     edgecolor=_rgb01(it["border"]),
                                     linewidth=line_width, joinstyle="miter"))

        # 5) 라벨 (raster 와 동일 위치: min(xs)+offset, mean(ys))
        if self.show_labels:
            lbl = _rgb01(tuple(self.renderer.label_color))
            for room in outline_rooms + other_rooms:
                coords = room["coords"]
                xs = [coords[i] for i in range(0, len(coords), 2)]
                ys = [coords[i] for i in range(1, len(coords), 2)]
                if xs and ys:
                    ax.text(min(xs) + 2, sum(ys) / len(ys), room["type"], color=lbl,
                            fontsize=6, ha="left", va="center")
            if front_door:
                ax.text(front_door["x"] - front_door["w"] / 2 + 1, front_door["y"],
                        "front_door", color=lbl, fontsize=6, ha="left", va="center")

        for p in out_paths:
            p = Path(p)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(p), facecolor=bg)
        plt.close(fig)

    @staticmethod
    def _door_to_coords(door: dict) -> list[int]:
        """door {x, y, w, h} (중심+크기) → 사각형 4 코너 flat 좌표 [x0,y0,...]."""
        x, y, w, h = float(door["x"]), float(door["y"]), float(door["w"]), float(door["h"])
        x0, y0, x1, y1 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        return [round(x0), round(y0), round(x1), round(y0),
                round(x1), round(y1), round(x0), round(y1)]

    def _compute_blend_regions(self, fill_items: list[dict]) -> list[tuple]:
        """**방(non-outline)** 쌍의 겹침 영역과 평균색을 계산한다 (순수 함수).

        raster (`_blend_overlap_regions`) 와 vector (`render_floorplan_to_vector`) 가
        동일한 겹침 영역·블렌딩 색을 공유하도록 분리했다. 캔버스에 그리지 않고 기하만 반환.

        Mod Record (2026-07-14): 현관문·interior door(``is_door=True``)는 겹침 블렌딩에서 제외한다.
        door 는 방 위에 solid 색으로 선명하게 보여야 하므로 방-방 겹침만 블렌딩한다.

        Args:
            fill_items: [{"coords": flat 좌표, "fill": (R,G,B), "is_door": bool, ...}, ...]

        Returns:
            [(exterior_xy, blend_rgb), ...]:
                - exterior_xy: 겹침 폴리곤 외곽 (x, y) 튜플 리스트 (close 좌표 제거).
                - blend_rgb: 두 방 fill 색 평균 (R, G, B) int 튜플.
        """
        try:
            from shapely.geometry import Polygon
        except ImportError:
            return []
        polys: list[tuple] = []
        for it in fill_items:
            if it.get("is_door"):
                continue  # 현관문·interior door 는 겹침 블렌딩 대상 제외 (solid 색 유지)
            coords = it.get("coords") or []
            if len(coords) < 6:
                continue
            xy = [(coords[i], coords[i + 1]) for i in range(0, len(coords) - 1, 2)]
            if len(xy) < 3:
                continue
            try:
                p = Polygon(xy)
                if not p.is_valid:
                    p = p.buffer(0)
                if p.is_empty or p.area < 1:
                    continue
            except Exception:
                continue
            polys.append((p, it["fill"]))

        regions: list[tuple] = []
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                pi, ci = polys[i]
                pj, cj = polys[j]
                try:
                    inter = pi.intersection(pj)
                except Exception:
                    continue
                if inter.is_empty or inter.area < 1:
                    continue
                blend = (round((ci[0] + cj[0]) / 2),
                         round((ci[1] + cj[1]) / 2),
                         round((ci[2] + cj[2]) / 2))
                geoms = list(inter.geoms) if inter.geom_type == "MultiPolygon" else [inter]
                for g in geoms:
                    if g.geom_type != "Polygon" or g.is_empty:
                        continue
                    regions.append((list(g.exterior.coords)[:-1], blend))  # close 좌표 제거
        return regions

    def _blend_overlap_regions(self, canvas, fill_items: list[dict]) -> None:
        """채우기 요소 (방 + door) 쌍의 겹침 영역만 두 fill_color 평균으로 재도색 (raster in-place).

        단독 영역은 solid 원색 그대로, 겹치는 부분만 섞인 색으로 칠해져 겹침이 색으로 드러난다.
        draw order 무관. 겹침 영역·색 계산은 `_compute_blend_regions` 에 위임 (vector 와 공유).
        """
        for ext, blend in self._compute_blend_regions(fill_items):
            self.renderer.fill_region_solid(canvas, ext, blend)

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
