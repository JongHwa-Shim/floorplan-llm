"""
평면도 JSONL 시각화 스크립트.

지정한 JSONL 파일(들)에서 특정 plan_id 또는 전체 평면도를 읽어
이미지로 시각화한다.

사용 예시:
    # 특정 plan_id 하나 시각화
    uv run python tests/build_dataset/rplan2json/visualize_jsonl.py --plan_id 0

    # 여러 plan_id 시각화
    uv run python tests/build_dataset/rplan2json/visualize_jsonl.py --plan_id 0 1 5

    # 전체 평면도 시각화
    uv run python tests/build_dataset/rplan2json/visualize_jsonl.py --all

    # 특정 JSONL 파일 지정
    uv run python tests/build_dataset/rplan2json/visualize_jsonl.py --plan_id 0 \\
        --jsonl_dir data/dataset/processed_dataset/rplan/jsonl \\
        --jsonl_pattern "floorplans_*.jsonl"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf
from tqdm import tqdm

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.build_dataset.visualize_json.loader import FloorplanLoader
from src.build_dataset.visualize_json.visualizer import FloorplanVisualizer


# 기본 경로 상수
DEFAULT_JSONL_DIR = PROJECT_ROOT / "data" / "dataset" / "processed_dataset" / "rplan" / "jsonl"
DEFAULT_JSONL_PATTERN = "*.jsonl"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "visualize_json" / "color_map.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "visualize_json"


def parse_args() -> argparse.Namespace:
    """커맨드라인 인자를 파싱한다.

    Args:
        없음.

    Returns:
        파싱된 Namespace 객체.
    """
    parser = argparse.ArgumentParser(
        description="평면도 JSONL 데이터를 이미지로 시각화합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 대상 선택 (--plan_id 또는 --all 중 하나 필수)
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--plan_id",
        nargs="+",
        metavar="ID",
        help="시각화할 plan_id 목록 (공백으로 구분, 예: --plan_id 0 1 5)",
    )
    target_group.add_argument(
        "--all",
        action="store_true",
        help="모든 평면도를 시각화합니다.",
    )

    # 경로 옵션
    parser.add_argument(
        "--jsonl_dir",
        type=Path,
        default=DEFAULT_JSONL_DIR,
        help=f"JSONL 파일이 있는 디렉토리 (기본값: {DEFAULT_JSONL_DIR})",
    )
    parser.add_argument(
        "--jsonl_pattern",
        type=str,
        default=DEFAULT_JSONL_PATTERN,
        help=f"JSONL 파일 검색 패턴 (기본값: {DEFAULT_JSONL_PATTERN})",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"색상 맵 설정 파일 경로 (기본값: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"출력 디렉토리 (기본값: {DEFAULT_OUTPUT_DIR})",
    )

    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    """시각화 파이프라인을 실행한다.

    Args:
        args: 파싱된 커맨드라인 인자.

    Raises:
        FileNotFoundError: 설정 파일이나 JSONL 디렉토리가 없을 때.
        ValueError: 지정한 plan_id를 찾지 못할 때.
    """
    # 설정 로드
    if not args.config.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {args.config}")
    cfg = OmegaConf.load(args.config)

    # 로더 초기화
    loader = FloorplanLoader.from_directory(args.jsonl_dir, pattern=args.jsonl_pattern)
    if not loader.jsonl_paths:
        raise FileNotFoundError(
            f"'{args.jsonl_dir}'에서 '{args.jsonl_pattern}' 패턴의 파일을 찾지 못했습니다."
        )
    print(f"[INFO] JSONL 파일 {len(loader.jsonl_paths)}개 로드됨: {args.jsonl_dir}")

    # 시각화 대상 수집
    if args.all:
        floorplans = loader.load_all()
        print(f"[INFO] 전체 평면도 {len(floorplans)}개를 시각화합니다.")
    else:
        floorplans = []
        for pid in args.plan_id:
            fp = loader.load_by_plan_id(pid)
            if fp is None:
                print(f"[WARN] plan_id '{pid}'를 찾지 못했습니다. 건너뜁니다.")
            else:
                floorplans.append(fp)

        if not floorplans:
            raise ValueError("유효한 plan_id가 없습니다.")

    # 시각화 실행
    visualizer = FloorplanVisualizer(cfg)
    for fp in tqdm(floorplans, desc="시각화 중"):
        plan_id = fp["plan_id"]
        output_path = args.output_dir / str(plan_id)
        visualizer.visualize(fp, output_path)
        print(f"[INFO] plan_id={plan_id} 저장 완료: {output_path}")

    print(f"\n[완료] 출력 디렉토리: {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    run(args)
