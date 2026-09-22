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
_PROJECT_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_HERE))
from _schema import from_ours_floorplan_json, write_common_json  # noqa: E402
from _condition_metadata import parse_condition_metadata  # noqa: E402
from src.training.augmentation.tokenizer import Vocab, load_vocab  # noqa: E402
from src.training.augmentation.decoder import decode_tokens  # noqa: E402


logger = logging.getLogger(__name__)


_OUTPUT_RE = re.compile(r"^output(?:_(\d+))?$")
_UNKNOWN_ID = re.compile(r"^<UNK:(\d+)>$")


def _saved_token_ids(directory: Path, vocab: Vocab) -> list[int] | None:
    """원본 ID 기록을 읽거나 과거 실행의 가독성 토큰 기록에서 복원한다."""
    id_path = directory / "token_ids.json"
    if id_path.exists():
        return [int(token_id) for token_id in json.loads(id_path.read_text(encoding="utf-8"))["ids"]]
    text_path = directory / "tokens.txt"
    if not text_path.exists():
        return None
    result = []
    for token in text_path.read_text(encoding="utf-8").split():
        if token in vocab.token_to_id:
            result.append(vocab.token_to_id[token])
        elif token == "<BOS>" and vocab.bos_token_id is not None:
            result.append(vocab.bos_token_id)
        elif token == "<EOS>" and vocab.eos_token_id is not None:
            result.append(vocab.eos_token_id)
        elif (match := _UNKNOWN_ID.fullmatch(token)) is not None:
            result.append(int(match.group(1)))
        elif token.isdigit() and int(token) in vocab.number_to_ids:
            result.extend(vocab.number_to_ids[int(token)])
        else:
            raise ValueError(f"복원할 수 없는 토큰: {text_path}: {token}")
    return result


def _parse_run_dir(run_dir: Path, out_dir: Path, model_name: str, vocab: Vocab) -> tuple[int, int]:
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
        condition_path = plan_dir / "input" / "tokens.txt"
        condition_ids = _saved_token_ids(plan_dir / "input", vocab) if not condition_path.exists() else None
        condition_metadata = (
            parse_condition_metadata(condition_path.read_text(encoding="utf-8"))
            if condition_path.exists() else
            parse_condition_metadata(decode_tokens(condition_ids, vocab))
            if condition_ids is not None else None
        )
        if condition_metadata is None:
            logger.warning("입력 토큰 기록이 없어 조건 평가를 준비할 수 없음: %s", plan_dir)
        for output_dir in sorted(plan_dir.iterdir()):
            if not output_dir.is_dir():
                continue
            m = _OUTPUT_RE.match(output_dir.name)
            if m is None:
                continue
            idx = m.group(1)
            stem = f"{plan_id}_{idx}" if idx is not None else plan_id
            token_ids = _saved_token_ids(output_dir, vocab)
            if condition_metadata is not None and token_ids is not None:
                write_common_json({
                    "plan_id": plan_id,
                    "output_index": int(idx) if idx is not None else 0,
                    "condition_metadata": condition_metadata,
                    "generated_token_ids": token_ids,
                }, out_dir / "_evaluation" / f"{stem}.json")
            else:
                logger.warning("보상 평가용 토큰 기록이 없음: %s", output_dir)
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
    parser.add_argument(
        "--tokenizer_dir", type=Path,
        default=_PROJECT_ROOT / "data/models/Qwen2.5-Coder-7B/tokenization",
        help="토큰 기록을 복원할 때 사용하는 추론 tokenizer 경로",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab(args.tokenizer_dir / "vocab_extension.json", args.tokenizer_dir)
    success, failure = _parse_run_dir(args.run_dir, args.out_dir, args.model_name, vocab)
    logger.info(
        "[normalize_ours] %s → %s: success=%d, failure=%d",
        args.run_dir, args.out_dir, success, failure,
    )


if __name__ == "__main__":
    main()
