"""논문 정의에 대한 CPU 회귀 검증. 모델/데이터 다운로드 없이 실행한다."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import Dataset
from omegaconf import OmegaConf
from transformers import get_scheduler

from src.build_model.tokenization.token_definitions import build_token_list, flatten_token_list
from src.build_model.tokenization.vocab_builder import build_vocab
from src.training.augmentation.tokenizer import Vocab, build_condition_tokens
from src.training.augmentation.pipeline import config_from_omegaconf
from src.training.augmentation.strategies import DropState, compute_drop_state
from src.training.rl.dataset import _extract_metadata
from src.training.rl.rewards import compute_all_rewards, REWARD_NAMES, TOKEN_CREDIT_REWARDS
from src.training.rl.rewards.parser import parse_output_tokens
from src.training.rl.rewards.credit_assignment import apply_token_credit_assignment
from src.training.rl.rewards.polygon_fidelity_reward import compute_polygon_fidelity_reward
from src.training.rl.advantage import compute_token_advantages, gdpo_group_normalize, _batch_normalize
from scripts.build_dataset.json2arrow.run_conversion import split_and_save
from tests.training.rl.verification._common import (
    RoomSpec, DoorSpec, FrontDoorSpec, build_output_token_ids,
)

ROOT = Path(__file__).resolve().parents[3]
CFG = OmegaConf.load(ROOT / "config/training/rl/pipeline.yaml")
TOKENS = flatten_token_list(build_token_list(ROOT / "config/build_dataset/rplan2json/room_type_merge.json"))
MAPPING = {token: i + 100 for i, token in enumerate(TOKENS)}
VOCAB = Vocab(MAPPING, {value: key for key, value in MAPPING.items()},
              eos_token_id=1, number_to_ids={n: [10000 + n] for n in range(100)})


def rectangle(kind, x, y, width, height):
    """직사각형 fixture를 만든다.

    Args:
        kind: 방 종류.
        x: 좌상단 X.
        y: 좌상단 Y.
        width: 너비.
        height: 높이.

    Returns:
        RoomSpec.

    Raises:
        없음.
    """
    return RoomSpec(kind, [(x, y), (x + width, y), (x + width, y + height), (x, y + height)])


def fixture(rooms=None, metadata=None, doors=None, front_door=None, cfg=None, **kwargs):
    """완전한 토큰 파이프라인을 통해 보상을 계산한다.

    Args:
        rooms: 외곽선을 포함한 방 목록.
        metadata: 입력 조건.
        doors: 내부 문 목록.
        front_door: 현관문.
        cfg: 보상 설정.
        **kwargs: 형식 오류를 주입하는 빌더 옵션.

    Returns:
        보상 결과, 토큰 ID, 토큰 위치 맵.

    Raises:
        없음.
    """
    rooms = rooms or [rectangle("outline", 0, 0, 100, 100), rectangle("bedroom", 0, 0, 100, 100)]
    ids, indices = build_output_token_ids(
        rooms, doors=doors or [], front_door=front_door or FrontDoorSpec(5, 5, 4, 2),
        vocab=VOCAB, **kwargs,
    )
    return compute_all_rewards(ids, VOCAB, metadata or {}, cfg or CFG.rewards), ids, indices


def distributed_worker(rank, init_file):
    """두 CPU 프로세스에서 실제 gather 결과를 검증한다.

    Args:
        rank: 프로세스 순번.
        init_file: Gloo 초기화 파일.

    Returns:
        없음.

    Raises:
        AssertionError: 단일 배치 결과와 다를 때.
    """
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        values = torch.tensor([[2., 4., 999.], [6., 8., 10.], [-3., 1., 999.], [0., 0., 0.]])
        lengths = [2, 3, 2, 0]

        def gather(value):
            """대표값을 rank 순서로 모은다.

            Args:
                value: 로컬 대표값.

            Returns:
                전체 대표값.

            Raises:
                없음.
            """
            buffers = [torch.empty_like(value) for _ in range(2)]
            dist.all_gather(buffers, value)
            return torch.cat(buffers, dim=0)  # (B_total, 2)

        part = slice(rank * 2, rank * 2 + 2)
        actual = _batch_normalize(values[part], lengths[part], 1e-8, gather_fn=gather)
        expected = _batch_normalize(values, lengths, 1e-8)[part]
        torch.testing.assert_close(actual, expected)
    finally:
        dist.destroy_process_group()


class ManuscriptAlignmentTests(unittest.TestCase):
    """보상 의미, 토큰 마스크, 학습 설정, 분산 통계를 검증한다."""

    def test_ten_binary_rewards_and_zero_masks(self):
        """정상 출력의 10개 보상과 성공 마스크를 검사한다."""
        result, ids, _ = fixture()
        self.assertEqual(set(result["rewards"]), set(REWARD_NAMES))
        self.assertEqual(set(result["rewards"].values()), {1.0})
        self.assertEqual(set(result["error_masks"]), TOKEN_CREDIT_REWARDS)
        for mask in result["error_masks"].values():
            self.assertEqual(mask.shape, (len(ids),))
            self.assertEqual(mask.sum(), 0)

    def test_format_gate(self):
        """형식 실패 시 모든 점수가 0이 되는지 확인한다."""
        result, _, _ = fixture(omit_output_wrapper=True)
        self.assertEqual(set(result["rewards"].values()), {0.0})
        self.assertGreater(result["error_masks"]["format"].sum(), 0)

    def test_format_boundaries_front_door_and_vertex_parity(self):
        """누락된 경계·현관문 및 홀수 꼭짓점을 거부한다."""
        for options in (
            {"omit_end_output": True},
            {"front_door": FrontDoorSpec(omit=True)},
            {"rooms": [rectangle("outline", 0, 0, 100, 100),
                       RoomSpec("bedroom", [(0, 0), (50, 0), (100, 0), (100, 100), (0, 100)])]},
        ):
            with self.subTest(options=options):
                result, _, _ = fixture(**options)
                self.assertEqual(result["rewards"]["format"], 0.0)

    def test_format_ablation_disables_gate(self):
        """format 제거 실험에서 다른 보상을 강제 0으로 만들지 않는다."""
        cfg = copy.deepcopy(CFG.rewards)
        cfg.format.enabled = False
        result, _, _ = fixture(rooms=[rectangle("outline", 0, 0, 100, 100)], cfg=cfg)
        self.assertNotIn("format", result["rewards"])
        self.assertEqual(result["rewards"]["orthogonality"], 1.0)

    def test_counts_all_conditions_and_drops(self):
        """개수 하나만 틀려도 실패하고 생략 조건은 제외한다."""
        for metadata, total, kinds in (
            ({"total_rooms": 2, "type_counts": {"bedroom": 1, "kitchen": 1}}, 0, 0),
            ({"total_rooms": None, "type_counts": {}}, 1, 1),
            ({"total_rooms": 1, "type_counts": {"kitchen": 0}}, 1, 1),
            ({"type_counts": {"bedroom": 0}}, 1, 0),
        ):
            with self.subTest(metadata=metadata):
                result, _, _ = fixture(metadata=metadata)
                self.assertEqual(result["rewards"]["count_total"], total)
                self.assertEqual(result["rewards"]["count_type"], kinds)

    def test_orthogonality_violation_and_zero_edge(self):
        """비직각과 길이 0인 변에 이진 실패와 좌표 마스크를 준다."""
        for coords in (
            [(0, 0), (100, 0), (90, 100), (0, 100)],
            [(0, 0), (100, 0), (100, 0), (0, 100)],
        ):
            result, _, _ = fixture(rooms=[rectangle("outline", 0, 0, 100, 100), RoomSpec("bedroom", coords)])
            self.assertEqual(result["rewards"]["format"], 1)
            self.assertEqual(result["rewards"]["orthogonality"], 0)
            self.assertGreater(result["error_masks"]["orthogonality"].sum(), 0)

    def test_overlap_shared_boundaries_and_crossing(self):
        """공유 경계는 허용하고 꼭짓점 침범 없는 십자 겹침도 거부한다."""
        outline = rectangle("outline", 0, 0, 100, 100)
        cases = (
            ([rectangle("bedroom", 0, 0, 50, 100), rectangle("kitchen", 50, 0, 50, 100)], 1),
            ([rectangle("bedroom", 0, 0, 60, 60), rectangle("kitchen", 40, 40, 60, 60)], 0),
            ([rectangle("bedroom", 0, 40, 100, 20), rectangle("kitchen", 40, 0, 20, 100)], 0),
        )
        for rooms, expected in cases:
            with self.subTest(expected=expected, rooms=rooms):
                result, _, _ = fixture(rooms=[outline] + rooms)
                self.assertEqual(result["rewards"]["no_overlap"], expected)
                if expected:
                    self.assertEqual(result["error_masks"]["no_overlap"].sum(), 0)

    def test_invalid_geometry_not_silently_repaired(self):
        """자기교차 폴리곤을 기하 보상에서 만점 처리하지 않는다."""
        result, _, _ = fixture(rooms=[
            rectangle("outline", 0, 0, 100, 100),
            RoomSpec("bedroom", [(0, 0), (100, 100), (0, 100), (100, 0)]),
        ])
        self.assertEqual(result["rewards"]["format"], 1)
        for name in ("no_overlap", "room_in_outline", "coverage"):
            self.assertEqual(result["rewards"][name], 0)

    def test_concave_outline_full_polygon_containment(self):
        """꼭짓점이 모두 내부여도 변이 외곽선을 벗어나면 실패한다."""
        outline = RoomSpec("outline", [(0, 0), (100, 0), (100, 100), (60, 100),
                                       (60, 40), (40, 40), (40, 100), (0, 100)])
        result, _, _ = fixture(rooms=[outline, rectangle("bedroom", 20, 20, 60, 60)])
        self.assertEqual(result["rewards"]["room_in_outline"], 0)
        self.assertEqual(result["error_masks"]["room_in_outline"].sum(), 0)

    def test_outside_room_and_front_door_masks(self):
        """방의 외부 꼭짓점과 현관문의 범위 초과 크기 토큰을 표시한다."""
        result, _, indices = fixture(
            rooms=[rectangle("outline", 0, 0, 100, 100), rectangle("bedroom", 90, 0, 20, 100)],
            front_door=FrontDoorSpec(99, 5, 4, 2),
        )
        mask = result["error_masks"]["room_in_outline"]
        self.assertEqual(result["rewards"]["room_in_outline"], 0)
        for index in indices.front_door_indices[2:]:
            self.assertEqual(mask[index], 1)
        for index in indices.front_door_indices[:2]:
            self.assertEqual(mask[index], 0)

    def test_front_door_center_size_containment(self):
        """현관문 중심과 크기를 사용해 양쪽 경계의 포함 여부를 판정한다."""
        rooms = [rectangle("outline", 0, 0, 100, 100),
                 rectangle("bedroom", 20, 20, 60, 60)]
        for door, expected in ((FrontDoorSpec(2, 50, 10, 4), 0),
                               (FrontDoorSpec(95, 50, 8, 4), 1),
                               (FrontDoorSpec(5, 5, 10, 10), 1)):
            with self.subTest(door=door):
                result, _, indices = fixture(rooms=rooms, front_door=door)
                self.assertEqual(result["rewards"]["room_in_outline"], expected)
                mask = result["error_masks"]["room_in_outline"]
                for index in indices.front_door_indices[2:]:
                    self.assertEqual(mask[index], 1 - expected)

    def test_coverage_threshold_union_and_clipping(self):
        """임계값 경계·중복 면적·외곽선 바깥 면적을 검증한다."""
        outline = rectangle("outline", 0, 0, 100, 100)
        for rooms, expected in (
            ([rectangle("bedroom", 0, 0, 86, 90)], 1),  # 7740 / 10000
            ([rectangle("bedroom", 0, 0, 85, 90)], 0),
            ([rectangle("bedroom", 0, 0, 50, 100), rectangle("kitchen", 0, 0, 50, 100)], 0),
            ([rectangle("bedroom", 50, 0, 100, 100)], 0),
        ):
            with self.subTest(rooms=rooms):
                result, _, _ = fixture(rooms=[outline] + rooms)
                self.assertEqual(result["rewards"]["coverage"], expected)

    def test_polygon_fidelity_tolerance_masks_and_order(self):
        """꼭짓점 순서 독립성과 15px 경계, 위반 좌표 마스크를 확인한다."""
        metadata = {"rooms": [{"rid": 1, "type": "bedroom", "coords": [20, 20, 60, 20, 60, 60, 20, 60]}]}
        for shift, expected in ((0, 1), (15, 1), (16, 0)):
            room = rectangle("bedroom", 20 + shift, 20, 40, 40)
            room.coords = list(reversed(room.coords[1:] + room.coords[:1]))
            result, _, _ = fixture(rooms=[rectangle("outline", 0, 0, 100, 100), room], metadata=metadata)
            self.assertEqual(result["rewards"]["polygon_fidelity"], expected)
            self.assertEqual(result["error_masks"]["polygon_fidelity"].sum(), 0 if expected else 8)

    def test_polygon_same_centroid_different_shape(self):
        """중심점만 같은 다른 모양을 거부한다."""
        metadata = {"rooms": [{"rid": 1, "type": "bedroom", "coords": [10, 10, 90, 10, 90, 90, 10, 90]}]}
        result, _, _ = fixture(rooms=[rectangle("outline", 0, 0, 100, 100),
                                     rectangle("bedroom", 30, 30, 40, 40)], metadata=metadata)
        self.assertEqual(result["rewards"]["polygon_fidelity"], 0)
        self.assertEqual(result["error_masks"]["polygon_fidelity"].sum(), 8)

    def test_polygon_assignment_missing_types_and_rooms(self):
        """종류 생략, 방 순서, 일대일 매칭, 부족한 출력 방을 검증한다."""
        metadata = {"rooms": [
            {"rid": 1, "type": "", "coords": [50, 0, 100, 0, 100, 100, 50, 100]},
            {"rid": 2, "type": "bedroom", "coords": [0, 0, 50, 0, 50, 100, 0, 100]},
        ]}
        rooms = [rectangle("outline", 0, 0, 100, 100), rectangle("kitchen", 50, 0, 50, 100),
                 rectangle("bedroom", 0, 0, 50, 100)]
        result, _, _ = fixture(rooms=rooms, metadata=metadata)
        self.assertEqual(result["rewards"]["polygon_fidelity"], 1)
        result, _, _ = fixture(rooms=rooms[:2], metadata=metadata)
        self.assertEqual(result["rewards"]["polygon_fidelity"], 0)

    def test_polygon_missing_input_vertex_has_no_invented_mask(self):
        """사라진 입력 꼭짓점은 점수를 낮추되 존재하는 정상 출력에 책임을 만들지 않는다."""
        result, _, _ = fixture()
        parsed = result["parsed"]
        coords = [0, 0, 100, 0, 100, 100, 0, 100, 50, 50]
        score, errors = compute_polygon_fidelity_reward(parsed, {"rooms": [{"type": "bedroom", "coords": coords}]})
        self.assertEqual(score, 0)
        self.assertEqual(errors, [])

    def test_connections_and_spatial_all_conditions(self):
        """두 조건 중 하나만 만족하는 결과를 부분 점수 없이 실패시킨다."""
        rooms = [rectangle("outline", 0, 0, 200, 100),
                 rectangle("bedroom", 0, 0, 50, 100),
                 rectangle("kitchen", 50, 0, 50, 100),
                 rectangle("bathroom", 150, 0, 50, 100)]
        metadata = {"rooms": [{"rid": i, "type": room.room_type, "coords": []} for i, room in enumerate(rooms)],
                    "edges": [{"pair": [1, 2], "has_door": True, "door": []},
                              {"pair": [1, 3], "has_door": True, "door": []}],
                    "spatial": [{"rid_a": 1, "rid_b": 2, "direction": "right"},
                                {"rid_a": 1, "rid_b": 3, "direction": "left"}]}
        result, _, _ = fixture(rooms=rooms, doors=[DoorSpec(50, 50, 2, 4)], metadata=metadata)
        self.assertEqual(result["rewards"]["connectivity"], 0)
        self.assertEqual(result["rewards"]["spatial"], 0)
        metadata["edges"] = metadata["edges"][:1]
        metadata["spatial"] = metadata["spatial"][:1]
        result, _, _ = fixture(rooms=rooms, doors=[DoorSpec(50, 50, 2, 4)], metadata=metadata)
        self.assertEqual(result["rewards"]["connectivity"], 1)
        self.assertEqual(result["rewards"]["spatial"], 1)

    def test_dropped_door_geometry_preserves_connection(self):
        """입력의 DOOR 태그가 남으면 문 좌표 생략 후에도 연결을 검사한다."""
        sample = {"rooms": [{"rid": 1, "type": "bedroom", "coords": []},
                            {"rid": 2, "type": "kitchen", "coords": []}],
                  "edges": [{"pair": [1, 2], "door": [{"x": 50, "y": 50, "w": 2, "h": 4}]}]}
        metadata = _extract_metadata(sample, DropState(drop_door={0: "all"}))
        self.assertTrue(metadata["edges"][0]["has_door"])
        self.assertEqual(metadata["edges"][0]["door"], [])
        rooms = [rectangle("outline", 0, 0, 100, 100), rectangle("bedroom", 0, 0, 50, 100),
                 rectangle("kitchen", 50, 0, 50, 100)]
        result, _, _ = fixture(rooms=rooms, metadata=metadata)
        self.assertEqual(result["rewards"]["connectivity"], 0)
        result, _, _ = fixture(rooms=rooms, doors=[DoorSpec(50, 50, 2, 4)], metadata=metadata)
        self.assertEqual(result["rewards"]["connectivity"], 1)

    def test_credit_equation_signs(self):
        """正·負·0 어드밴티지와 단일/다중 위반을 손계산과 비교한다."""
        mask = torch.tensor([0., 1., 1.])
        for scalar, expected in ((2., [2.6, -0.9, -0.9]), (-2., [-1.4, -4.9, -4.9]), (0., [0., -1.5, -1.5])):
            torch.testing.assert_close(apply_token_credit_assignment(scalar, mask, .3, .7, 1.5), torch.tensor(expected))

    def test_input_noise_metadata_matches_visible_tokens(self):
        """보상은 깨끗한 정답이 아니라 입력에 노출된 잡음 좌표를 사용한다."""
        clean = [20, 20, 60, 20, 60, 60, 20, 60]
        noisy = [25, 23, 65, 23, 65, 63, 25, 63]
        sample = {"rooms": [{"rid": 1, "type": "bedroom", "coords": clean}]}
        state = DropState(noise_room_coords={1: noisy})
        metadata = _extract_metadata(sample, state)
        self.assertEqual(metadata["rooms"][0]["coords"], noisy)
        self.assertEqual(sample["rooms"][0]["coords"], clean)
        state.drop_coords.add(1)
        self.assertEqual(_extract_metadata(sample, state)["rooms"][0]["coords"], [])

    def test_credit_zero_mask_alpha_and_off(self):
        """위반 없는 토큰의 α와 credit OFF를 독립적인 기대값으로 검증한다."""
        cfg = [{"enabled": True, "weight": 1., "credit_assignment": True,
                "nominal_gain": .3, "faulty_attenuation": .7, "penalty_offset": 1.5}]
        values = torch.tensor([[2.], [-2.], [0.]])
        masks = [{"format": torch.tensor([0., 0.])},
                 {"format": torch.tensor([0., 1.])}, {"format": torch.tensor([1.])}]
        lengths = [2, 2, 1]
        raw = torch.tensor([[2.6, 2.6], [-1.4, -4.9], [-1.5, 0.]])
        actual = compute_token_advantages(values, ["format"], cfg, masks, lengths, 2)
        torch.testing.assert_close(actual, _batch_normalize(raw, lengths, 1e-8))
        off = compute_token_advantages(values, ["format"], cfg, masks, lengths, 2, use_token_credit_assignment=False)
        torch.testing.assert_close(off, _batch_normalize(torch.tensor([[2., 2.], [-2., -2.], [0., 0.]]), lengths, 1e-8))
        self.assertEqual(actual[2, 1], 0)

    def test_group_normalization(self):
        """두 그룹의 독립 정규화와 상수 보상을 검사한다."""
        values = torch.tensor([[0., 1.], [1., 1.], [1., 0.], [1., 1.]])
        expected = torch.tensor([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]])
        torch.testing.assert_close(gdpo_group_normalize(values, 2), expected)

    def test_real_two_process_normalization(self):
        """실제 두 CPU 프로세스 gather와 단일 배치 결과를 대조한다."""
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_worker, args=(str(Path(directory) / "gloo"),), nprocs=2, join=True)

    def test_trainer_registry_and_gather_integration(self):
        """실제 Trainer의 등록·정규화 호출 경로를 모델 없이 실행한다."""
        from src.training.rl.trainer import RLTrainer
        trainer = object.__new__(RLTrainer)
        trainer.reward_cfg = CFG.rewards
        trainer.advantage_cfg = CFG.advantage
        trainer._reward_names, trainer._reward_cfgs_list = [], []
        self.assertEqual(len(trainer._build_reward_funcs()), 10)
        trainer.num_generations = 2
        trainer._cached_rewards_per_func = torch.tensor([[0.] * 10, [1.] * 10])
        trainer._error_masks_buffer = [{}, {}]
        calls = []
        trainer.accelerator = SimpleNamespace(process_index=0, gather=lambda value: calls.append(value) or value)
        output = {"completion_ids": torch.ones((2, 3), dtype=torch.long),
                  "completion_mask": torch.tensor([[1, 1, 0], [1, 1, 1]])}
        output = trainer._apply_token_credit_assignment(output)
        self.assertEqual(len(calls), 1)
        self.assertEqual(output["advantages"].shape, (2, 3))
        self.assertEqual(output["advantages"][0, 2], 0)

    def test_exact_split_and_disjointness(self):
        """80,788개의 합성 ID로 정확한 분할 크기·중복 부재를 검증한다."""
        dataset = Dataset.from_dict({"plan_id": list(range(80788))})
        with tempfile.TemporaryDirectory() as directory:
            splits = split_and_save(dataset, directory, 1000, 1000, 42)
            self.assertEqual([len(splits[key]) for key in ("train", "validation", "test")], [78788, 1000, 1000])
            sets = [set(splits[key]["plan_id"]) for key in ("train", "validation", "test")]
            self.assertEqual(len(set.union(*sets)), 80788)
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_split_rejects_fraction_or_empty_train(self):
        """분할 비율을 개수로 오해하거나 훈련 집합을 비우지 못하게 한다."""
        dataset = Dataset.from_dict({"id": list(range(10))})
        for val, test in ((.1, 1), (5, 5), (0, 1)):
            with self.subTest(val=val, test=test), self.assertRaises(ValueError):
                split_and_save(dataset, "unused", val, test, 42)

    def test_scheduler_and_paper_hyperparameters(self):
        """실제 scheduler의 2,500-step 선형 warm-up과 이후 상수를 검증한다."""
        self.assertEqual(CFG.training.max_steps, 50000)
        self.assertEqual(CFG.training.learning_rate, 2e-5)
        self.assertEqual(CFG.rl.num_generations, 16)
        self.assertEqual(CFG.rl.temperature, .7)
        expected = [1., .5, 1., 1.5, 2., 1.5, 1.5, 1., .5, 1.5]
        self.assertEqual([CFG.rewards[name].weight for name in REWARD_NAMES], expected)
        parameter = torch.nn.Parameter(torch.tensor(1.))
        optimizer = torch.optim.AdamW([parameter], lr=CFG.training.learning_rate)
        scheduler = get_scheduler(CFG.training.lr_scheduler_type, optimizer,
                                  num_warmup_steps=int(50000 * CFG.training.warmup_ratio),
                                  num_training_steps=50000)
        self.assertEqual(scheduler.lr_lambdas[0](0), 0)
        self.assertEqual(scheduler.lr_lambdas[0](1250), .5)
        for step in (2500, 2501, 50000):
            self.assertEqual(scheduler.lr_lambdas[0](step), 1)

    def test_vocabulary_builder_preserves_existing_checkpoint_mapping(self):
        """다른 어휘로 기존 ID 파일을 덮어쓰지 않는지 확인한다."""
        with tempfile.TemporaryDirectory() as directory:
            saved = Path(directory) / "vocab_extension.json"
            original = b'{"token_to_id": {"<DOOR_H>": 151650}}'
            saved.write_bytes(original)
            tokenizer = MagicMock()
            tokenizer.__len__.return_value = 151646
            with patch("src.build_model.tokenization.vocab_builder.AutoTokenizer.from_pretrained", return_value=tokenizer):
                with self.assertRaisesRegex(ValueError, "output.dir"):
                    build_vocab("unused", ROOT / "config/build_dataset/rplan2json/room_type_merge.json", Path(directory))
            self.assertEqual(saved.read_bytes(), original)
            tokenizer.add_tokens.assert_not_called()
            tokenizer.save_pretrained.assert_not_called()

    def test_new_vocabulary_and_bubble_input(self):
        """564개 도메인 토큰과 비교 입력에서 공간 관계가 제거되는지 확인한다."""
        self.assertEqual(len(TOKENS) - 1, 564)
        self.assertIn("<PAD>", TOKENS)
        self.assertNotIn("<DOOR_H>", TOKENS)
        self.assertNotIn("<DOOR_V>", TOKENS)
        cfg = config_from_omegaconf(OmegaConf.load(ROOT / "config/training/augmentation/ours_bubble.yaml"))
        sample = {"rooms": [{"rid": 1, "type": "bedroom", "coords": [0, 0, 50, 0, 50, 50, 0, 50]},
                            {"rid": 2, "type": "kitchen", "coords": [50, 0, 100, 0, 100, 50, 50, 50]}],
                  "edges": [{"pair": [1, 2], "door": [{"x": 50, "y": 20, "w": 2, "h": 4}]}],
                  "spatial": [{"rid_a": 1, "rid_b": 2, "direction": "right"}], "front_door": None}
        import random
        state = compute_drop_state(sample, cfg.to_drop_params(), rng=random.Random(42))
        metadata = _extract_metadata(sample, state)
        ids = build_condition_tokens(sample, state, VOCAB)
        self.assertNotIn(VOCAB.get("<SP>"), ids)
        self.assertIn(VOCAB.get("<EDGE>"), ids)
        self.assertIn(VOCAB.get("<TOTAL>"), ids)
        self.assertEqual(metadata["spatial"], [])
        self.assertEqual(metadata["total_rooms"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
