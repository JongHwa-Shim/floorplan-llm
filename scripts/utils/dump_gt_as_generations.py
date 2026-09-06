"""GT 를 공통 스키마 generation 디렉토리에 떨군다 (smoke-test/sanity-check 용).

unified test set 의 GT plan 을 ``experiments/generations/gt/{plan_id}.json`` 으로 저장하여
모든 metric 스크립트가 동일 입력으로 작동하는지 확인할 수 있다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "scripts" / "metrics"))
sys.path.insert(0, str(_HERE.parents[1] / "scripts" / "normalize"))

from _common import load_gt_pool, load_unified_plan_ids  # noqa: E402
from _schema import write_common_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_pool", type=Path, default=Path(
        "data/dataset/processed_dataset/rplan/arrow/eval_pool"
    ))
    parser.add_argument(
        "--plan_ids_file",
        type=Path,
        default=Path("experiments/testset_unified.json"),
    )
    parser.add_argument(
        "--out_dir", type=Path, default=Path("experiments/generations/gt")
    )
    args = parser.parse_args()

    gt_pool = load_gt_pool(args.gt_pool)
    target_ids = load_unified_plan_ids(args.plan_ids_file)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for pid in tqdm(target_ids, desc="dump GT"):
        if pid not in gt_pool:
            continue
        plan = dict(gt_pool[pid])
        plan["model"] = "gt"
        write_common_json(plan, args.out_dir / f"{pid}.json")
        n += 1
    print(f"[dump_gt_as_generations] {n} GT plans → {args.out_dir}")


if __name__ == "__main__":
    main()
