"""figure_selections.yaml 자동 추천.

ours 메트릭 CSV 를 분석해 cherry-picking 회피 기준으로 후보 plan_id 를 자동 선정한다.

선정 기준 (가이드 §9.5 의 '대표성 있으면서 cherry-picked 가 아닌' 케이스):
    Figure 6 (5-way 비교):
        - 방 개수 (5/6/7/8) 별로 GED 가 **중간값(median)** 인 plan_id 1 개씩 + 자유 추가 1 개 = 5 개
        - "GED 중간값 = 평균적 case = 대표성 ✓"

    Figure 7 (Diversity):
        - ours best-of-10 의 출력간 분산 (예: 방 면적 합의 std) 이 큰 plan_id 3 개
        - 본 연구가 다양한 출력을 만든다는 점이 잘 드러나는 case

    Figure 8 (Robustness):
        - 4 조건 추론 (full/bubble/partial/sparse) 이 아직 안 됐으므로 plan_id 후보만 제시
        - GED 가 평균보다 약간 어려운 (75 percentile) 케이스 권장

    Figure 11 (Failure):
        - 명시적 실패: no_overlap < 1.0 OR room_in_outline < 0.5 OR format < 1.0
        - 또는 방 8 개 plan 중 GED 가 가장 큰 case (8 rooms 의 sparse 조건은 본 연구도 어려운 case 임을 보여줌)

Usage:
    uv run python scripts/utils/suggest_figure_selections.py \
        --novel experiments/metrics/exp3_novel_ours.csv \
        --ged experiments/metrics/exp1_compatibility_ours.csv \
        --output experiments/figure_selections.suggested.yaml

사용자가 결과 yaml 을 검토 후 `experiments/figure_selections.yaml` 로 복사·수정.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _median_plan_id(df: pd.DataFrame, col: str, group_col: str | None = None) -> dict:
    """col 의 중간값에 가장 가까운 plan_id (그룹별 또는 전체)."""
    if group_col is None:
        med = df[col].median()
        idx = (df[col] - med).abs().idxmin()
        return {None: str(df.loc[idx, "plan_id"])}
    out = {}
    for g, sub in df.groupby(group_col):
        med = sub[col].median()
        idx = (sub[col] - med).abs().idxmin()
        out[int(g)] = str(sub.loc[idx, "plan_id"])
    return out


def _quantile_plan_id(df: pd.DataFrame, col: str, q: float) -> str:
    target = df[col].quantile(q)
    idx = (df[col] - target).abs().idxmin()
    return str(df.loc[idx, "plan_id"])


def _diversity_top(novel_df: pd.DataFrame, n: int = 3) -> list[str]:
    """공간 다양성이 큰 plan_id — coverage 가 중간 + room_in_outline 이 높은 안정적 case 우선.

    Heuristic: 공간 reward 가 낮을수록 (spatial < 0.5) 출력 간 차이가 다양할 가능성. 또 best-of-K
    의 std 가 작은 케이스는 모드 collapse 가능성 — 다양성 show 에 부적합. 본 추론에서 다양성
    raw 통계가 별도로 없으므로 단순 heuristic: GED 가 평균 근처 + spatial 0.2~0.5 의 8-room case.
    """
    candidates = novel_df[(novel_df["room_count"] == 8) & (novel_df.get("spatial", 1) < 0.5)]
    if len(candidates) < n:
        candidates = novel_df[novel_df["room_count"] == 8]
    return [str(p) for p in candidates.head(n)["plan_id"].tolist()]


def _failure_cases(novel_df: pd.DataFrame, n: int = 4) -> list[dict]:
    """failure type 별 후보."""
    out: list[dict] = []

    # 1) overlap violation
    cand = novel_df[novel_df["no_overlap"] < 1.0].head(1)
    if not cand.empty:
        out.append({"plan_id": str(cand.iloc[0]["plan_id"]), "failure_type": "overlap_violation"})

    # 2) outline violation (room_in_outline)
    cand = novel_df[novel_df["room_in_outline"] < 0.5].head(1)
    if not cand.empty:
        out.append({"plan_id": str(cand.iloc[0]["plan_id"]), "failure_type": "outline_violation"})

    # 3) connectivity low
    cand = novel_df[novel_df["connectivity"] < 0.5].head(1)
    if not cand.empty:
        out.append({"plan_id": str(cand.iloc[0]["plan_id"]), "failure_type": "connectivity_low"})

    # 4) 8-room hard case (GED 가 큰 8-room)
    cand_8 = novel_df[novel_df["room_count"] == 8]
    if not cand_8.empty and "ged_mean" in cand_8.columns:
        idx = cand_8["ged_mean"].idxmax()
        out.append({"plan_id": str(cand_8.loc[idx, "plan_id"]), "failure_type": "hard_8room"})

    # 부족하면 패딩
    while len(out) < n:
        out.append({"plan_id": None, "failure_type": "TODO"})
    return out[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--novel", type=Path, required=True,
                        help="exp3_novel_ours.csv 경로")
    parser.add_argument("--ged", type=Path, required=True,
                        help="exp1_compatibility_ours.csv 경로 (per-plan)")
    parser.add_argument("--output", type=Path, default=Path("experiments/figure_selections.suggested.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    novel = pd.read_csv(args.novel)
    ged = pd.read_csv(args.ged)
    novel["plan_id"] = novel["plan_id"].astype(str)
    ged["plan_id"] = ged["plan_id"].astype(str)

    # GED 평균을 novel 에 join (failure 분류·hard case 용)
    ged_mean = ged.groupby("plan_id")["ged"].mean().rename("ged_mean")
    novel = novel.merge(ged_mean, on="plan_id", how="left")

    # === Figure 6 — 5-way 비교: room_count 별 median GED plan_id ===
    f6_per_bucket = _median_plan_id(novel, "ged_mean", "room_count")
    figure_6 = [f6_per_bucket.get(n) for n in [5, 6, 7, 8]]
    # 추가 1: 전체에서 GED 평균 plan_id 1 개 더 (5 개 채움)
    figure_6.append(_quantile_plan_id(novel, "ged_mean", 0.5))

    # === Figure 7 — Diversity ===
    figure_7 = _diversity_top(novel, n=3)

    # === Figure 8 — Robustness (조건별 추론 안 됐으므로 평균보다 어려운 case 권장) ===
    figure_8 = [
        _quantile_plan_id(novel, "ged_mean", 0.75),  # 어려운 case 1
        _quantile_plan_id(novel, "ged_mean", 0.50),  # 평균 case 1
    ]

    # === Figure 11 — Failure ===
    figure_11 = _failure_cases(novel, n=4)

    selections = {
        "figure_6": figure_6,
        "figure_7": figure_7,
        "figure_8": figure_8,
        "figure_11": figure_11,
        "_notes": {
            "criteria": {
                "figure_6": "room_count 별 GED median (대표적·중립적 case)",
                "figure_7": "8-room + spatial<0.5 (다양성 후보)",
                "figure_8": "GED 75percentile + 50percentile (Robustness 비교용 hard·typical)",
                "figure_11": "no_overlap/room_in_outline/connectivity 임계값 위반 + 8-room hard case",
            },
            "note": "본 yaml 은 자동 추천. cherry-picking 회피를 위해 사용자가 검토 후 figure_selections.yaml 로 복사 권장.",
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(selections, allow_unicode=True, sort_keys=False))
    logger.info("[suggest_figure_selections] → %s", args.output)
    logger.info("figure_6 (5-way 비교): %s", figure_6)
    logger.info("figure_7 (Diversity): %s", figure_7)
    logger.info("figure_8 (Robustness): %s", figure_8)
    logger.info("figure_11 (Failure):  %s", figure_11)


if __name__ == "__main__":
    main()
