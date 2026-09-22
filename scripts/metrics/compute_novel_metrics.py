"""추론 당시 INPUT과 생성 토큰으로 논문의 10개 이진 보상 충족률을 계산한다.

normalize_ours.py의 _evaluation 기록은 파싱 실패 출력도 포함한다. 원본 토큰이
없는 GT/타 모델의 공통 JSON은 기하 보상 네 개의 참조값에만 사용한다.
형식 하드 게이트는 훈련 전용이며, 평가는 각 보상을 독립적으로 계산한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts/metrics"))

from _common import get_room_count_bucket, load_common_dir, load_gt_pool, load_unified_plan_ids  # noqa: E402
from src.training.augmentation.tokenizer import load_vocab  # noqa: E402
from src.training.rl.rewards import REWARD_NAMES, compute_all_rewards  # noqa: E402
from src.training.rl.rewards.coverage_reward import compute_coverage_reward  # noqa: E402
from src.training.rl.rewards.geometry_reward import compute_no_overlap_reward, compute_orthogonality_reward  # noqa: E402
from src.training.rl.rewards.parser import ParsedDoor, ParsedFloorplan, ParsedRoom  # noqa: E402
from src.training.rl.rewards.room_in_outline_reward import compute_room_in_outline_reward  # noqa: E402

logger = logging.getLogger(__name__)
ALL_METRICS = tuple(REWARD_NAMES)
GEOMETRIC = frozenset(("orthogonality", "no_overlap", "room_in_outline", "coverage"))


def _load_records(gen_dir: Path) -> dict[str, list[dict]]:
    """입력 조건·출력 토큰을 ID별 생성 순서대로 불러온다."""
    records: dict[str, list[dict]] = defaultdict(list)
    for path in sorted((gen_dir / "_evaluation").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if "condition_metadata" not in record or "generated_token_ids" not in record:
            raise ValueError(f"보상 평가 기록이 불완전합니다: {path}")
        records[str(record["plan_id"])].append(record)
    return {key: sorted(value, key=lambda item: item["output_index"])
            for key, value in records.items()}


def _score_tokens(record: dict, vocab, reward_cfg) -> dict[str, float]:
    """공통 파서·보상 함수를 사용하되 훈련용 형식 게이트 없이 평가한다.

    Args:
        record: 생성 토큰 ID와 실제 입력 조건을 담은 평가 기록.
        vocab: 평면도 어휘.
        reward_cfg: 보상별 설정. 훈련용 설정값을 변경하지 않는다.

    Returns:
        복원된 데이터에 각 함수를 독립적으로 적용한 10개 이진 보상.

    Raises:
        ValueError: 활성 보상이 10개가 아니거나 이진값이 아닐 때.
    """
    rewards = compute_all_rewards(
        record["generated_token_ids"], vocab, record["condition_metadata"], reward_cfg,
        apply_format_gate=False,
    )["rewards"]
    if set(rewards) != set(REWARD_NAMES) or any(v not in (0.0, 1.0) for v in rewards.values()):
        raise ValueError("평가 보상이 활성 10개 이진 보상과 일치하지 않습니다.")
    return rewards


def _common_to_parsed(plan: dict) -> ParsedFloorplan:
    """GT의 기하 참조값에 한해 공통 JSON을 보상 입력으로 변환한다."""
    rooms = []
    for room in plan.get("rooms", []):
        rooms.append(ParsedRoom(
            room_type=room.get("type", "unknown"),
            coords=[(float(x), float(y)) for x, y in room.get("polygon", [])],
            coord_token_indices=[], block_start=0, block_end=0,
        ))
    doors = [ParsedDoor(cx=float(d["x"]), cy=float(d["y"]), w=float(d["w"]),
                        h=float(d["h"]), is_valid=True) for d in plan.get("doors", []) or []]
    fd = plan.get("front_door")
    front_door = ({"cx": float(fd["x"]), "cy": float(fd["y"]),
                   "w": float(fd["w"]), "h": float(fd["h"])} if fd else None)
    return ParsedFloorplan(
        success=bool(rooms and rooms[0].room_type == "outline"), level=3,
        front_door=front_door, rooms=rooms, doors=doors,
        error_indices=[], error_spans={}, front_door_token_indices=[],
    )


def _score_common_geometry(plan: dict, metrics: list[str], reward_cfg) -> dict[str, float]:
    """원본 토큰이 없는 실평면도의 기하 보상 네 개만 평가한다."""
    parsed = _common_to_parsed(plan)
    functions = {
        "orthogonality": lambda: compute_orthogonality_reward(parsed)[0],
        "no_overlap": lambda: compute_no_overlap_reward(parsed)[0],
        "room_in_outline": lambda: compute_room_in_outline_reward(parsed)[0],
        "coverage": lambda: compute_coverage_reward(
            parsed, threshold=float(reward_cfg.coverage.threshold)),
    }
    return {name: float(functions[name]()) for name in metrics}


def _reduce(values: list[float], strategy: str) -> float:
    """기본값은 모든 생성 결과의 평균이며 최고값은 명시 요청 시만 사용한다."""
    if not values:
        raise ValueError("평가할 생성 출력이 없습니다.")
    if strategy == "mean":
        return sum(values) / len(values)
    if strategy == "max":
        return max(values)
    return values[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", type=Path, required=True)
    parser.add_argument("--gt_pool", type=Path, required=True)
    parser.add_argument("--plan_ids_file", type=Path, default=None)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", nargs="+", default=list(ALL_METRICS), choices=ALL_METRICS)
    parser.add_argument("--best_of_strategy", choices=("mean", "max", "first"), default="mean")
    parser.add_argument("--tokenizer_dir", type=Path,
                        default=_ROOT / "data/models/Qwen2.5-Coder-7B/tokenization")
    parser.add_argument("--reward_config", type=Path,
                        default=_ROOT / "config/training/rl/pipeline.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    reward_cfg = OmegaConf.load(args.reward_config).rewards
    gt_pool = load_gt_pool(args.gt_pool)
    common_groups = load_common_dir(args.gen)
    records = _load_records(args.gen)
    target_ids = (set(load_unified_plan_ids(args.plan_ids_file)) if args.plan_ids_file
                  else set(common_groups) | set(records))
    vocab = (load_vocab(args.tokenizer_dir / "vocab_extension.json", args.tokenizer_dir)
             if records else None)

    rows = []
    for plan_id in tqdm(sorted(target_ids), desc=f"rewards {args.model_name}"):
        if plan_id in records:
            scores = [_score_tokens(record, vocab, reward_cfg) for record in records[plan_id]]
        elif plan_id in common_groups and set(args.metrics) <= GEOMETRIC:
            scores = [_score_common_geometry(plan, args.metrics, reward_cfg)
                      for plan in common_groups[plan_id]]
        elif plan_id in common_groups:
            raise ValueError(
                f"{plan_id}: 실제 INPUT·생성 토큰 기록이 없습니다. "
                "normalize_ours.py로 평가 기록을 만들거나 기하 보상만 선택하세요."
            )
        else:
            logger.warning("생성 결과가 없는 plan ID를 건너뜀: %s", plan_id)
            continue
        source = gt_pool.get(plan_id) or (common_groups.get(plan_id) or [None])[0]
        if source is None:
            raise ValueError(f"{plan_id}: 방 개수 그룹을 확인할 GT/정규화 출력이 없습니다.")
        rows.append({
            "model": args.model_name, "plan_id": plan_id,
            "room_count": get_room_count_bucket(source),
            **{name: _reduce([score[name] for score in scores], args.best_of_strategy)
               for name in args.metrics},
            "n_generations": len(scores),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "plan_id", "room_count", *args.metrics, "n_generations"]
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if rows:
        logger.info("[rewards] %s: N=%d, means=%s", args.model_name, len(rows),
                    {name: round(sum(row[name] for row in rows) / len(rows), 4)
                     for name in args.metrics})


if __name__ == "__main__":
    main()
