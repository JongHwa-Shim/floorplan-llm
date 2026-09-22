"""Ablation 결과 표 빌더 (Tables 4 / 5 / 6 / 7 공용).

``compute_novel_metrics.py`` 가 떨군 ``exp{N}_novel_{variant}.csv`` 파일들을 모두 모아 한 표로
정리한다.

Usage:
    # Table 4 (Stage Ablation)
    uv run python scripts/tables/build_ablation_table.py \
        --metrics_dir experiments/metrics \
        --pattern 'exp4_novel_*.csv' \
        --output experiments/tables_figures/table_4.csv

    # Table 5 (Reward Ablation)
    uv run python scripts/tables/build_ablation_table.py \
        --metrics_dir experiments/metrics \
        --pattern 'exp5_novel_*.csv' \
        --output experiments/tables_figures/table_5.csv

    # Tables 6, 7 도 동일 패턴.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_METRICS = [
    "format", "orthogonality", "no_overlap",
    "room_in_outline", "coverage",
    "count_total", "count_type",
    "connectivity", "spatial", "polygon_fidelity",
]


def _scan(metrics_dir: Path, pattern: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    name_re = re.compile(r"^exp\d+_novel_(?P<v>[a-zA-Z0-9_]+)\.csv$")
    for f in sorted(metrics_dir.glob(pattern)):
        m = name_re.match(f.name)
        if not m:
            continue
        variant = m.group("v")
        with f.open() as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                for k, v in row.items():
                    if k in {"model", "plan_id", "room_count", "n_generations"}:
                        continue
                    try:
                        fv = float(v)
                    except (TypeError, ValueError):
                        continue
                    if fv != fv:  # NaN
                        continue
                    out[variant][k].append(fv)
    return {
        v: {k: round(sum(vals) / len(vals), 4) for k, vals in d.items()}
        for v, d in out.items()
    }


def _fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else "—"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", type=Path, required=True)
    parser.add_argument("--pattern", required=True, help="예: 'exp4_novel_*.csv'")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metrics", nargs="*", default=_DEFAULT_METRICS,
        help="표 컬럼 순서 (기본 10개 논문 보상)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data = _scan(args.metrics_dir, args.pattern)
    fieldnames = ["variant"] + args.metrics
    rows = []
    for variant, d in sorted(data.items()):
        rows.append({"variant": variant, **{m: _fmt(d.get(m)) for m in args.metrics}})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("[build_ablation_table] %d rows → %s", len(rows), args.output)


if __name__ == "__main__":
    main()
