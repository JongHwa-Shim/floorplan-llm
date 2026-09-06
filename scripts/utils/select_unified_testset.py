"""통일 test set 선정 + 평가 풀 Arrow split 생성.

방 개수(5/6/7/8)별로 균등 추출하여 모든 baseline 비교에서 동일한 plan_id 집합을 사용하도록
`experiments/testset_unified.json` 을 생성한다. 동시에 본 연구 추론에서 한 번의 호출로 처리할 수
있도록 validation + test split을 concat 한 `data/.../arrow/eval_pool/` 을 함께 저장한다.

데이터 풀: validation + test split 합집합 (학습에 사용되지 않은 모든 plan)

Args (env로 override 가능):
    POOL_SPLITS: 사용할 split 이름들 (콤마 구분, 기본 "validation,test")
    PER_BUCKET: 방 개수당 최대 추출 수 (기본 100)
    SEED: 랜덤 시드 (기본 42)
    OUT_PATH: 출력 JSON 경로 (기본 experiments/testset_unified.json)
    POOL_OUT: eval_pool Arrow split 저장 경로
              (기본 data/dataset/processed_dataset/rplan/arrow/eval_pool)

Outputs:
    experiments/testset_unified.json:
        {"5": [...], "6": [...], "7": [...], "8": [...],
         "all": [...],   # 4 bucket 합집합 (추론 input.plan_ids 직접 주입용)
         "_metadata": {...}}
    data/dataset/processed_dataset/rplan/arrow/eval_pool/:
        validation + test 가 concat 된 단일 Arrow Dataset (DatasetDict 가 아닌 single)
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path

from datasets import concatenate_datasets, load_from_disk


def main() -> None:
    pool_splits = os.environ.get("POOL_SPLITS", "validation,test").split(",")
    per_bucket = int(os.environ.get("PER_BUCKET", "100"))
    seed = int(os.environ.get("SEED", "42"))
    out_path = Path(os.environ.get("OUT_PATH", "experiments/testset_unified.json"))
    pool_out = Path(
        os.environ.get(
            "POOL_OUT", "data/dataset/processed_dataset/rplan/arrow/eval_pool"
        )
    )

    arrow_root = Path("data/dataset/processed_dataset/rplan/arrow")

    # 풀 구성: 각 plan_id의 방 개수 (outline 제외) 산출 + Arrow split concat
    by_count: dict[int, list[str]] = defaultdict(list)
    loaded = []
    for split in pool_splits:
        split = split.strip()
        if not split:
            continue
        ds = load_from_disk(str(arrow_root / split))
        loaded.append(ds)
        for r in ds:
            types = r["rooms"]["type"]
            n_non_outline = sum(1 for t in types if t != "outline")
            by_count[n_non_outline].append(r["plan_id"])
    pool_size = sum(len(d) for d in loaded)

    rng = random.Random(seed)
    selected: dict[str, list[str]] = {}
    for n in [5, 6, 7, 8]:
        available = by_count.get(n, [])
        take = min(per_bucket, len(available))
        selected[str(n)] = sorted(rng.sample(available, take))

    all_ids = sorted({pid for ids in selected.values() for pid in ids})
    selected["all"] = all_ids
    total = len(all_ids)
    selected["_metadata"] = {
        "pool_splits": pool_splits,
        "pool_size": pool_size,
        "per_bucket_target": per_bucket,
        "seed": seed,
        "buckets": {n: len(by_count.get(int(n), [])) for n in ["5", "6", "7", "8"]},
        "selected_counts": {n: len(selected[n]) for n in ["5", "6", "7", "8"]},
        "total": total,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(selected, indent=2, ensure_ascii=False))
    print(f"[select_unified_testset] saved {out_path}")
    print(f"  pool size: {pool_size}, total selected: {total}")
    print(f"  per-bucket: {selected['_metadata']['selected_counts']}")

    # eval_pool Arrow split 저장 (concat)
    pool_ds = concatenate_datasets(loaded)
    pool_out.parent.mkdir(parents=True, exist_ok=True)
    pool_ds.save_to_disk(str(pool_out))
    print(f"[select_unified_testset] eval_pool saved → {pool_out} ({len(pool_ds)} rows)")


if __name__ == "__main__":
    main()
