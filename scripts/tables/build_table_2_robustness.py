"""Table 2 (Robustness under Diverse Input Conditions, 가이드 §6) 빌더.

본 연구 + DStruct2Design (입력 호환 가능한 경우) 만 비교. 4 조건 세트 (full / bubble_only /
partial / sparse) × 5 metric (polygon_fidelity / count_total / count_type / spatial / connectivity).

각 조건은 본 연구에서 별도 run 으로 추론 후 ``experiments/generations/ours_{cond}`` 디렉토리에
정규화 결과를 떨군 다음 ``compute_novel_metrics.py`` 로 메트릭을 측정해 둔다.

입력 CSV (모두 ``compute_novel_metrics.py`` 의 출력 포맷):
    experiments/metrics/exp2_novel_{model}_{cond}.csv

Usage:
    uv run python scripts/tables/build_table_2_robustness.py \
        --metrics_dir experiments/metrics \
        --output experiments/tables_figures/table_2.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

_CONDS = ["full", "bubble_only", "partial", "sparse"]
_METRICS = ["polygon_fidelity", "count_total", "count_type", "spatial", "connectivity"]


def _scan(metrics_dir: Path) -> dict[tuple[str, str], dict[str, float]]:
    """파일명 패턴 ``exp2_novel_{model}_{cond}.csv`` 매칭 → {(model, cond): {metric: mean}}."""
    out: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    pat = re.compile(r"^exp2_novel_(?P<model>[a-zA-Z0-9_]+?)_(?P<cond>full|bubble_only|partial|sparse)\.csv$")
    for f in sorted(metrics_dir.glob("exp2_novel_*.csv")):
        m = pat.match(f.name)
        if not m:
            continue
        model, cond = m.group("model"), m.group("cond")
        with f.open() as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                for mt in _METRICS:
                    if row.get(mt):
                        try:
                            out[(model, cond)][mt].append(float(row[mt]))
                        except ValueError:
                            continue
    return {
        k: {mt: round(sum(v) / len(v), 4) for mt, v in d.items()}
        for k, d in out.items()
    }


def _fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else "—"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data = _scan(args.metrics_dir)
    models = sorted({k[0] for k in data})

    fieldnames = ["model", "condition"] + _METRICS
    rows = []
    for model in models:
        for cond in _CONDS:
            d = data.get((model, cond), {})
            rows.append({
                "model": model,
                "condition": cond,
                **{mt: _fmt(d.get(mt)) for mt in _METRICS},
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("[build_table_2] %d rows → %s", len(rows), args.output)


if __name__ == "__main__":
    main()
