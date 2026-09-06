"""통일 raster renderer.

공통 스키마 JSON 디렉토리 (또는 Arrow GT split)를 받아 256×256 PNG로 일괄 렌더링한다.
FID 같은 픽셀 기반 메트릭이 모든 모델·GT를 동일한 렌더링 protocol로 비교하도록 보장한다.

Usage:
    # 공통 스키마 디렉토리 (본 연구 / baseline 정규화 결과)
    uv run python scripts/render/render_unified.py \
        --input experiments/generations/ours \
        --output experiments/renders/ours

    # Arrow GT split → PNG
    uv run python scripts/render/render_unified.py \
        --gt_arrow data/dataset/processed_dataset/rplan/arrow/eval_pool \
        --plan_ids_file experiments/testset_unified.json \
        --output experiments/renders/gt_test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts" / "normalize"))

from _schema import load_common_json, to_parsed_floorplan_dict  # noqa: E402
from src.build_dataset.visualize_json.visualizer import FloorplanVisualizer  # noqa: E402

logger = logging.getLogger(__name__)


def _load_visualizer(color_map_path: Path, show_labels: bool | None = None,
                      skip_interior_doors: bool = False) -> FloorplanVisualizer:
    """color_map.yaml 로 visualizer 생성.

    Args:
        color_map_path: yaml 경로.
        show_labels: None 이면 yaml 의 ``vis_settings.show_labels`` 기본값 사용. True/False 이면
            argparse override.
        skip_interior_doors: True 면 interior door 사각형을 그리지 않는다 (DS2D fair FID 측정용).
    """
    cfg = OmegaConf.load(color_map_path)
    return FloorplanVisualizer(cfg, show_labels=show_labels,
                                skip_interior_doors=skip_interior_doors)


def _arrow_row_to_floorplan(row: dict) -> dict:
    """Arrow columnar row → visualizer 가 받는 row-oriented floorplan dict."""
    rooms_col = row["rooms"]
    rooms = [
        {"rid": rooms_col["rid"][i], "type": rooms_col["type"][i], "coords": rooms_col["coords"][i]}
        for i in range(len(rooms_col["rid"]))
    ]
    edges_col = row["edges"]
    edges = []
    for i, pair in enumerate(edges_col["pair"]):
        door_dict = edges_col["door"][i] if edges_col["door"] else {"x": [], "y": [], "w": [], "h": []}
        doors = []
        for j in range(len(door_dict["x"])):
            doors.append({
                "x": door_dict["x"][j], "y": door_dict["y"][j],
                "w": door_dict["w"][j], "h": door_dict["h"][j],
            })
        edges.append({"pair": pair, "doors": doors})
    fd_col = row.get("front_door") or {"x": [], "y": [], "w": [], "h": []}
    if fd_col.get("x"):
        front_door = {
            "x": fd_col["x"][0], "y": fd_col["y"][0],
            "w": fd_col["w"][0], "h": fd_col["h"][0],
        }
    else:
        front_door = None
    return {
        "plan_id": row.get("plan_id"),
        "rooms": rooms,
        "edges": edges,
        "front_door": front_door,
    }


def _render_common_dir(in_dir: Path, out_dir: Path, viz: FloorplanVisualizer) -> int:
    """공통 스키마 JSON 디렉토리를 PNG로 렌더링."""
    files = sorted(in_dir.glob("*.json"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in tqdm(files, desc=f"render {in_dir.name}"):
        common = load_common_json(f)
        parsed = to_parsed_floorplan_dict(common)
        viz.render_floorplan_to_path(parsed, out_dir / f"{f.stem}.png")
    return len(files)


def _render_gt(gt_arrow: Path, plan_ids: set[str] | None, out_dir: Path, viz: FloorplanVisualizer) -> int:
    """Arrow GT split → PNG."""
    from datasets import load_from_disk

    ds = load_from_disk(str(gt_arrow))
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for row in tqdm(ds, desc=f"render GT {gt_arrow.name}"):
        if plan_ids is not None and str(row["plan_id"]) not in plan_ids:
            continue
        floorplan = _arrow_row_to_floorplan(row)
        viz.render_floorplan_to_path(floorplan, out_dir / f"{row['plan_id']}.png")
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=None,
        help="공통 스키마 JSON 디렉토리 (본 연구·baseline 정규화 결과)",
    )
    parser.add_argument(
        "--gt_arrow", type=Path, default=None,
        help="Arrow GT split 경로 (GT raster 렌더링 모드)",
    )
    parser.add_argument(
        "--plan_ids_file", type=Path, default=None,
        help="GT 모드에서 사용할 plan_id 리스트 JSON 경로 (testset_unified.json 호환)",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="PNG 출력 디렉토리",
    )
    parser.add_argument(
        "--color_map",
        type=Path,
        default=_PROJECT_ROOT / "config/build_dataset/visualize_json/color_map.yaml",
        help="visualizer color_map yaml 경로",
    )
    parser.add_argument(
        "--show_labels",
        type=lambda v: str(v).lower() in {"true", "1", "yes", "on"},
        default=False,
        help="방 이름 라벨 그리기 여부 (기본 False — 논문 figure 표준). True 면 디버그·검증용.",
    )
    parser.add_argument(
        "--skip_interior_doors",
        type=lambda v: str(v).lower() in {"true", "1", "yes", "on"},
        default=False,
        help="interior door 사각형 그리기 생략 (DS2D 처럼 door 를 생성하지 않는 baseline 과 "
             "fair FID 비교를 위해 GT 측에서도 사용). front_door 는 영향 X.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    viz = _load_visualizer(args.color_map, show_labels=args.show_labels,
                            skip_interior_doors=args.skip_interior_doors)

    if args.input is not None:
        n = _render_common_dir(args.input, args.output, viz)
        logger.info("[render_unified] %d files rendered → %s", n, args.output)
    elif args.gt_arrow is not None:
        plan_ids = None
        if args.plan_ids_file:
            data = json.loads(args.plan_ids_file.read_text())
            if isinstance(data, dict):
                if "all" in data:
                    plan_ids = set(map(str, data["all"]))
                else:
                    plan_ids = {
                        str(pid)
                        for k, v in data.items()
                        if not k.startswith("_") and isinstance(v, list)
                        for pid in v
                    }
            else:
                plan_ids = set(map(str, data))
        n = _render_gt(args.gt_arrow, plan_ids, args.output, viz)
        logger.info("[render_unified] %d GT plans rendered → %s", n, args.output)
    else:
        parser.error("--input 또는 --gt_arrow 중 하나는 필수")


if __name__ == "__main__":
    main()
