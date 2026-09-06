"""본 연구 추론 출력 → 공통 JSON 스키마 정규화.

추론 결과 디렉토리(`outputs/inference/{model.name}/{stage}/{date}/{time}/`)를 스캔하여
각 `{plan_id}/output*/floorplan.json` 을 공통 스키마로 변환·저장한다.

출력 디렉토리 구조:
    out_root/
    ├── {plan_id}_0.json        # output_0 (num_outputs > 1)
    ├── {plan_id}_1.json
    └── ...
    또는 num_outputs=1 인 경우:
    ├── {plan_id}.json

Usage:
    uv run python scripts/normalize/normalize_ours.py \
        --run_dir outputs/inference/Qwen2.5-Coder-7B/rl/2026-05-15/12-00-00 \
        --out_dir experiments/generations/ours \
        --model_name ours
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from _schema import from_ours_floorplan_json, write_common_json  # noqa: E402


logger = logging.getLogger(__name__)


_OUTPUT_RE = re.compile(r"^output(?:_(\d+))?$")


def _parse_run_dir(run_dir: Path, out_dir: Path, model_name: str) -> tuple[int, int]:
    """run_dir 하위의 모든 {plan_id}/output*/floorplan.json 을 변환.

    Returns:
        (success_count, failure_count).
    """
    success = 0
    failure = 0
    for plan_dir in sorted(run_dir.iterdir()):
        if not plan_dir.is_dir():
            continue
        # .hydra 같은 시스템 디렉토리는 plan_id 가 아님
        if plan_dir.name.startswith("."):
            continue
        plan_id = plan_dir.name
        for output_dir in sorted(plan_dir.iterdir()):
            if not output_dir.is_dir():
                continue
            m = _OUTPUT_RE.match(output_dir.name)
            if m is None:
                continue
            fp_json = output_dir / "floorplan.json"
            if not fp_json.exists():
                failure += 1
                continue
            try:
                raw = json.loads(fp_json.read_text())
            except Exception as e:
                logger.warning("JSON 파싱 실패 %s: %s", fp_json, e)
                failure += 1
                continue
            common = from_ours_floorplan_json(raw, plan_id=plan_id, model=model_name)
            # 파일명: output_0 → {plan_id}_0.json, output → {plan_id}.json
            idx = m.group(1)
            stem = f"{plan_id}_{idx}" if idx is not None else plan_id
            write_common_json(common, out_dir / f"{stem}.json")
            success += 1
    return success, failure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="본 연구 추론 출력 루트 (예: outputs/inference/Qwen2.5-Coder-7B/rl/2026-05-15/12-00-00)",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("experiments/generations/ours"),
        help="공통 스키마 JSON 출력 디렉토리",
    )
    parser.add_argument(
        "--model_name",
        default="ours",
        help="공통 스키마 'model' 필드에 기록할 식별자 (예: ours, ours_no_ea)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    success, failure = _parse_run_dir(args.run_dir, args.out_dir, args.model_name)
    logger.info(
        "[normalize_ours] %s → %s: success=%d, failure=%d",
        args.run_dir, args.out_dir, success, failure,
    )


if __name__ == "__main__":
    main()
