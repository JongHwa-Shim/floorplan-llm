"""Table 3 (Geometric Quality, 가이드 §7) 빌더.

본 연구 + 3 baseline (HouseDiffusion, GSDiff, DS2D) 의 기하학적 metric 통합.
``exp3_novel_{model}.csv`` 파일들이 ``compute_novel_metrics.py`` 의 출력으로 존재한다고 가정한다.

산출 컬럼:
    orthogonality, no_overlap, room_in_outline, coverage
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

_METRICS = [
    "orthogonality", "no_overlap",
    "room_in_outline", "coverage",
]


def _scan(metrics_dir: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    pat = re.compile(r"^exp3_novel_(?P<model>[a-zA-Z0-9_]+)\.csv$")
    for f in sorted(metrics_dir.glob("exp3_novel_*.csv")):
        m = pat.match(f.name)
        if not m:
            continue
        model = m.group("model")
        with f.open() as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                for mt in _METRICS:
                    val = row.get(mt)
                    if val:
                        try:
                            out[model][mt].append(float(val))
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
    fieldnames = ["model"] + _METRICS
    rows = []
    for model, d in sorted(data.items()):
        rows.append({"model": model, **{mt: _fmt(d.get(mt)) for mt in _METRICS}})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("[build_table_3] %d rows → %s", len(rows), args.output)


if __name__ == "__main__":
    main()
