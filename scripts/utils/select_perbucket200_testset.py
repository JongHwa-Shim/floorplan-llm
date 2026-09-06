"""Per-bucket ≥200 plan testset 생성 — train hold-out 포함.

eval_pool (val + test 합본, 485 plan) 만으로는 bucket 5 가 N=29 라 per-bucket FID 통계 약함.
사용자 결정: data leak 허용, train 에서 부족분만큼 hold-out.

산출:
    experiments/testset_perbucket200.json
        {
          "5": [plan_id, ...],  # 200 (eval_pool 29 + train 171)
          "6": [plan_id, ...],  # 200 (eval_pool 173 + train 27)
          "7": [plan_id, ...],  # 200 (eval_pool 181 + train 19)
          "8": [plan_id, ...],  # 200 (eval_pool 100 + train 100)
          "all": [...],          # 800
          "_metadata": {
            "data_leak": True,
            "leak_note": "Train hold-out included for statistical stability of per-bucket FID.",
            "from_train": {"5": 171, "6": 27, "7": 19, "8": 100},
            ...
          }
        }

또한 train hold-out plan_id 의 행만 추출한 별도 arrow 디렉토리도 생성:
    data/dataset/processed_dataset/rplan/arrow/train_holdout_perbucket200/

(이 arrow + eval_pool 두 곳에 plan_id 가 분산되므로 추론 시 두 번 호출 또는 concat 사용.)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, load_from_disk

logger = logging.getLogger(__name__)

TARGET_PER_BUCKET = 200
SEED = 42


def _bucket_of(row: dict) -> int:
    return sum(1 for t in row["rooms"]["type"] if t != "outline")


def _collect_bucket_plan_ids(arrow_dir: Path) -> dict[int, list[str]]:
    ds = load_from_disk(str(arrow_dir))
    by_bucket: dict[int, list[str]] = defaultdict(list)
    for row in ds:
        by_bucket[_bucket_of(row)].append(str(row["plan_id"]))
    return dict(by_bucket)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_pool", type=Path,
                        default=Path("data/dataset/processed_dataset/rplan/arrow/eval_pool"))
    parser.add_argument("--train", type=Path,
                        default=Path("data/dataset/processed_dataset/rplan/arrow/train"))
    parser.add_argument("--output_testset", type=Path,
                        default=Path("experiments/testset_perbucket200.json"))
    parser.add_argument("--output_arrow", type=Path,
                        default=Path("data/dataset/processed_dataset/rplan/arrow/train_holdout_perbucket200"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    eval_buckets = _collect_bucket_plan_ids(args.eval_pool)
    train_buckets = _collect_bucket_plan_ids(args.train)
    logger.info("eval_pool buckets: %s", {b: len(v) for b, v in sorted(eval_buckets.items())})
    logger.info("train buckets: %s", {b: len(v) for b, v in sorted(train_buckets.items())})

    rng = random.Random(SEED)
    selected: dict[str, list[str]] = {}
    from_train_counts: dict[str, int] = {}
    train_holdout_plan_ids: set[str] = set()

    for b in [5, 6, 7, 8]:
        eval_ids = sorted(eval_buckets.get(b, []))
        train_ids = sorted(train_buckets.get(b, []))
        n_eval = len(eval_ids)

        if n_eval >= TARGET_PER_BUCKET:
            picked_eval = rng.sample(eval_ids, TARGET_PER_BUCKET)
            picked_train: list[str] = []
        else:
            picked_eval = eval_ids
            needed = TARGET_PER_BUCKET - n_eval
            picked_train = rng.sample(train_ids, needed)
            train_holdout_plan_ids.update(picked_train)

        combined = picked_eval + picked_train
        selected[str(b)] = combined
        from_train_counts[str(b)] = len(picked_train)
        logger.info("bucket %d: total=%d (eval=%d, train_holdout=%d)",
                    b, len(combined), len(picked_eval), len(picked_train))

    all_ids = [pid for b in ["5", "6", "7", "8"] for pid in selected[b]]
    out_payload = {
        **selected,
        "all": all_ids,
        "_metadata": {
            "data_leak": True,
            "leak_note": "Train hold-out plan_ids are included for statistical stability of "
                         "per-bucket FID (e.g., bucket 5 has only 29 in eval_pool). "
                         "These plans were SEEN during SFT/RL training but with diverse "
                         "augmentation variants — not exact reconstruction targets. "
                         "Footnote required in paper.",
            "from_eval_pool": {b: len(selected[b]) - from_train_counts[b] for b in ["5","6","7","8"]},
            "from_train": from_train_counts,
            "total": len(all_ids),
            "target_per_bucket": TARGET_PER_BUCKET,
            "seed": SEED,
        },
    }
    args.output_testset.parent.mkdir(parents=True, exist_ok=True)
    args.output_testset.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2))
    logger.info("[testset_perbucket200] %d plans → %s", len(all_ids), args.output_testset)

    # train hold-out arrow 추출
    if train_holdout_plan_ids:
        logger.info("[arrow] train hold-out 추출 시작 (%d plan_id)", len(train_holdout_plan_ids))
        ds = load_from_disk(str(args.train))
        holdout_ds = ds.filter(lambda r: str(r["plan_id"]) in train_holdout_plan_ids,
                                desc="filter train hold-out")
        logger.info("[arrow] filter 결과: %d row (target %d)",
                    len(holdout_ds), len(train_holdout_plan_ids))
        args.output_arrow.parent.mkdir(parents=True, exist_ok=True)
        if args.output_arrow.exists():
            import shutil
            shutil.rmtree(args.output_arrow)
        holdout_ds.save_to_disk(str(args.output_arrow))
        logger.info("[arrow] saved → %s", args.output_arrow)


if __name__ == "__main__":
    main()
