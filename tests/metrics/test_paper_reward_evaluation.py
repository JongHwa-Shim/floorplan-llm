"""평가 입력 조건, 파싱 실패 포함, 이진 보상 평균을 검증한다."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from scripts.metrics.compute_novel_metrics import _load_records, _reduce, _score_tokens
from scripts.normalize._condition_metadata import parse_condition_metadata
from scripts.normalize.normalize_ours import _parse_run_dir, _saved_token_ids
from scripts.tables.build_ablation_table import _DEFAULT_METRICS
from src.inference.result_saver import save_results
from src.training.augmentation.decoder import decode_tokens
from src.training.rl.rewards import REWARD_NAMES, compute_all_rewards
from tests.training.rl.verification._common import (
    DoorSpec, FrontDoorSpec, RoomSpec, build_output_token_ids, get_vocab,
)


ROOT = Path(__file__).resolve().parents[2]
CONDITION = (
    "<INPUT><ROOM_SUMMARY><TYPE:bedroom><COUNT>2<END_ROOM_SUMMARY>"
    "<ROOM><RID:1><TYPE:bedroom><X:10><Y:20><X:40><Y:20>"
    "<X:40><Y:50><X:10><Y:50><END_ROOM>"
    "<EDGE><RID:1><RID:2><DOOR><SEP_DOOR><END_DOOR><END_EDGE>"
    "<SP><RID:1><RID:2><REL:right><END_SP><END_INPUT>"
)


class PaperRewardEvaluationTests(unittest.TestCase):
    """원고의 평가 단위를 회귀 검사한다."""

    def test_visible_condition_only(self):
        """드롭된 총개수·문 좌표를 GT에서 복원하지 않는다."""
        metadata = parse_condition_metadata(CONDITION)
        self.assertIsNone(metadata["total_rooms"])
        self.assertEqual(metadata["type_counts"], {"bedroom": 2})
        self.assertEqual(metadata["rooms"][0]["coords"][:4], [10, 20, 40, 20])
        self.assertEqual(metadata["edges"], [
            {"pair": [1, 2], "door": [], "has_door": True},
        ])
        self.assertEqual(metadata["spatial"][0]["direction"], "right")

    def test_evaluation_scores_independently_but_training_keeps_format_gate(self):
        """종료 토큰 누락은 평가의 Format에만 영향을 주고 훈련은 게이트를 유지한다.

        Args:
            없음.
        Returns:
            없음.
        Raises:
            AssertionError: 평가 독립성 또는 훈련 게이트가 유지되지 않을 때.
        """
        vocab = get_vocab()
        rooms = [
            RoomSpec("outline", [(0, 0), (100, 0), (100, 100), (0, 100)]),
            RoomSpec("bedroom", [(0, 0), (50, 0), (50, 100), (0, 100)]),
            RoomSpec("kitchen", [(50, 0), (100, 0), (100, 100), (50, 100)]),
        ]
        valid, _ = build_output_token_ids(
            rooms, doors=[DoorSpec(50, 50, 2, 4)],
            front_door=FrontDoorSpec(50, 5, 4, 2), vocab=vocab,
        )
        metadata = {
            "total_rooms": 2, "type_counts": {"bedroom": 1, "kitchen": 1},
            "rooms": [{"rid": i, "type": room.room_type,
                       "coords": [value for point in room.coords for value in point]}
                      for i, room in enumerate(rooms)],
            "edges": [{"pair": [1, 2], "has_door": True, "door": []}],
            "spatial": [{"rid_a": 1, "rid_b": 2, "direction": "right"}],
        }
        cfg = OmegaConf.load(ROOT / "config/training/rl/pipeline.yaml").rewards
        original_cfg = OmegaConf.to_container(cfg, resolve=True)
        valid_scores = _score_tokens({"generated_token_ids": valid,
                                      "condition_metadata": metadata}, vocab, cfg)
        self.assertEqual(valid_scores, dict.fromkeys(REWARD_NAMES, 1.0))
        self.assertEqual(set(_DEFAULT_METRICS), set(REWARD_NAMES))
        for marker in ("<END_OUTPUT>", "<END_ROOM>"):
            with self.subTest(missing_marker=marker):
                invalid = valid.copy()
                invalid.remove(vocab.get(marker))
                invalid_scores = _score_tokens({"generated_token_ids": invalid,
                                                "condition_metadata": metadata}, vocab, cfg)
                self.assertEqual(invalid_scores, {**valid_scores, "format": 0.0})
                self.assertEqual(_reduce([valid_scores["format"], invalid_scores["format"]], "mean"), 0.5)
                self.assertEqual(_reduce([valid_scores["orthogonality"], invalid_scores["orthogonality"]], "mean"), 1.0)
                training = compute_all_rewards(invalid, vocab, metadata, cfg)
                self.assertEqual(training["rewards"], dict.fromkeys(REWARD_NAMES, 0.0))
                self.assertFalse(training["hard_gate_pass"])
        self.assertEqual(OmegaConf.to_container(cfg, resolve=True), original_cfg)

    def test_format_failure_does_not_hide_actual_geometry_failures(self):
        """형식 오류와 겹침·비직교 오류가 함께 있어도 항목별로 판정한다.

        Args:
            없음.
        Returns:
            없음.
        Raises:
            AssertionError: 기하 위반이 무시되거나 다른 보상으로 전파될 때.
        """
        vocab = get_vocab()
        cfg = OmegaConf.load(ROOT / "config/training/rl/pipeline.yaml").rewards
        for right_edge, expected_orthogonality in ((60, 1.0), (70, 0.0)):
            with self.subTest(right_edge=right_edge):
                rooms = [
                    RoomSpec("outline", [(0, 0), (100, 0), (100, 100), (0, 100)]),
                    RoomSpec("bedroom", [(0, 0), (60, 0), (right_edge, 100), (0, 100)]),
                    RoomSpec("kitchen", [(50, 0), (100, 0), (100, 100), (50, 100)]),
                ]
                tokens, _ = build_output_token_ids(rooms, vocab=vocab, omit_end_output=True)
                scores = _score_tokens({"generated_token_ids": tokens,
                                       "condition_metadata": {"total_rooms": 2}}, vocab, cfg)
                self.assertEqual(scores["format"], 0.0)
                self.assertEqual(scores["orthogonality"], expected_orthogonality)
                self.assertEqual(scores["no_overlap"], 0.0)
                self.assertEqual(scores["coverage"], 1.0)
                self.assertEqual(scores["room_in_outline"], 1.0)
                self.assertEqual(scores["count_total"], 1.0)

    def test_unrecoverable_outputs_still_score_zero(self):
        """방을 복원할 수 없는 출력은 모든 평가 항목에서 0점을 받는다.

        Args:
            없음.
        Returns:
            없음.
        Raises:
            AssertionError: 빈 출력이나 완전 파싱 실패를 통과시킬 때.
        """
        vocab = get_vocab()
        cfg = OmegaConf.load(ROOT / "config/training/rl/pipeline.yaml").rewards
        for tokens in ([], [vocab.get("<END_OUTPUT>")],
                       [vocab.get("<OUTPUT>"), vocab.get("<END_OUTPUT>")]):
            with self.subTest(tokens=tokens):
                scores = _score_tokens({"generated_token_ids": tokens,
                                       "condition_metadata": {}}, vocab, cfg)
                self.assertEqual(scores, dict.fromkeys(REWARD_NAMES, 0.0))

    def test_normalizer_keeps_failed_generation_in_evaluation(self):
        """floorplan.json이 없는 출력도 평가 분모에 남는다."""
        vocab = get_vocab()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            plan = base / "run" / "planA"
            (plan / "input").mkdir(parents=True)
            (plan / "input" / "tokens.txt").write_text(CONDITION, encoding="utf-8")
            for index in (0, 1):
                output = plan / f"output_{index}"
                output.mkdir()
                (output / "token_ids.json").write_text(
                    json.dumps({"ids": [1, 2, 3]}), encoding="utf-8",
                )
            (plan / "output_0" / "floorplan.json").write_text(
                json.dumps({"rooms": [{"rid": 0, "type": "outline",
                                       "coords": [0, 0, 10, 0, 10, 10, 0, 10]}],
                            "edges": [], "front_door": None}), encoding="utf-8",
            )
            success, failure = _parse_run_dir(base / "run", base / "normalized", "ours", vocab)
            self.assertEqual((success, failure), (1, 1))
            self.assertEqual(len(_load_records(base / "normalized")["planA"]), 2)
            self.assertTrue((base / "normalized" / "planA_0.json").exists())
            self.assertFalse((base / "normalized" / "planA_1.json").exists())

    def test_inference_saves_failed_output_token_ids(self):
        """JSON 파싱 실패여도 새 추론은 평가용 원본 ID를 남긴다."""
        with tempfile.TemporaryDirectory() as tmp:
            save_results(
                plan_id="planA", raw_sample={"rooms": []},
                condition_tokens=[10, 11], output_results=[([20, 21], None, 0.1)],
                vocab=get_vocab(),
                output_cfg=OmegaConf.create({"save_tokens": False, "save_json": True,
                                             "save_image": False}),
                color_map_cfg=OmegaConf.create({}), output_dir=Path(tmp),
                augmentation_summary="test",
            )
            self.assertEqual(json.loads((Path(tmp) / "planA/input/token_ids.json").read_text())["ids"],
                             [10, 11])
            self.assertEqual(json.loads((Path(tmp) / "planA/output/token_ids.json").read_text())["ids"],
                             [20, 21])

    def test_old_readable_output_can_restore_token_ids(self):
        """과거 추론의 tokens.txt도 보상 파서의 원본 ID 순서로 복원한다."""
        vocab = get_vocab()
        rooms = [RoomSpec("outline", [(0, 0), (100, 0), (100, 100), (0, 100)]),
                 RoomSpec("bedroom", [(10, 10), (40, 10), (40, 40), (10, 40)])]
        ids, _ = build_output_token_ids(rooms, front_door=FrontDoorSpec(50, 5, 4, 2), vocab=vocab)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "tokens.txt").write_text(decode_tokens(ids, vocab), encoding="utf-8")
            self.assertEqual(_saved_token_ids(Path(tmp), vocab), ids)


if __name__ == "__main__":
    unittest.main()
