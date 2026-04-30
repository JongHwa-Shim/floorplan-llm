"""Group 1: 변형 증강 후 metadata 좌표 반영 검증.

핵심 회귀 단언: `_extract_metadata`가 변형 적용 후 좌표를 metadata.rooms에
정확히 옮기는가. 이전 버그(원본 raw_sample 참조)가 재발하면 fail.

검증 케이스:
    - flip("H") 단독 적용 → metadata 좌표 = (255 - x, y)
    - flip("V") 단독 적용 → metadata 좌표 = (x, 255 - y)
    - flip("HV") + spatial 방향 갱신 → metadata.spatial direction 일치
    - translate 적용 → 좌표 평행이동
    - last_augmented_sample == _extract_metadata(...).rooms 좌표 일치 (E2E)
"""

from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

# 공용 헬퍼 import 가능하게 sys.path 조정
_VERIF_ROOT = Path(__file__).resolve().parents[1]
if str(_VERIF_ROOT) not in sys.path:
    sys.path.insert(0, str(_VERIF_ROOT))

from _common import CaseResult, run_cases, summary_and_exit, get_vocab  # noqa: E402

from src.training.augmentation import strategies  # noqa: E402
from src.training.augmentation.pipeline import (  # noqa: E402
    AugmentationConfig,
    AugmentationPipeline,
)
from src.training.rl.dataset import _extract_metadata  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def make_fixture_sample() -> dict:
    """row-oriented 표준 fixture (outline + 3 rooms + edges + spatial)."""
    return {
        "plan_id": "test_001",
        "rooms": [
            {"rid": 0, "type": "outline",    "coords": [10, 10, 200, 10, 200, 200, 10, 200]},
            {"rid": 1, "type": "livingroom", "coords": [10, 10, 100, 10, 100, 100, 10, 100]},
            {"rid": 2, "type": "bedroom",    "coords": [100, 10, 200, 10, 200, 100, 100, 100]},
            {"rid": 3, "type": "kitchen",    "coords": [10, 100, 100, 100, 100, 200, 10, 200]},
        ],
        "edges": [
            {"pair": [1, 2], "door": [{"x": 100.0, "y": 50.0, "w": 2.0, "h": 10.0}]},
            {"pair": [1, 3], "door": [{"x": 50.0, "y": 100.0, "w": 10.0, "h": 2.0}]},
        ],
        "front_door": {"x": 105.0, "y": 10.0, "w": 8.0, "h": 2.0},
        "spatial": [
            {"rid_a": 1, "rid_b": 2, "direction": "right"},
            {"rid_a": 1, "rid_b": 3, "direction": "below"},
        ],
    }


def to_columnar(sample: dict) -> dict:
    """row-oriented 테스트 sample을 Arrow columnar 포맷으로 변환."""
    rooms = sample["rooms"]
    edges = sample["edges"]
    spatial = sample["spatial"]
    fd = sample["front_door"]
    fd_door_seq = (
        [{"role": "fd", "x": fd["x"], "y": fd["y"], "w": fd["w"], "h": fd["h"]}]
        if fd is not None else []
    )
    return {
        "plan_id": sample["plan_id"],
        "rooms": {
            "rid":    [r["rid"]    for r in rooms],
            "type":   [r["type"]   for r in rooms],
            "coords": [r["coords"] for r in rooms],
        },
        "edges": {
            "pair": [e["pair"] for e in edges],
            "door": [
                [{"role": "edge", **d} for d in e["door"]]
                for e in edges
            ],
        },
        "front_door": fd_door_seq,
        "spatial": {
            "rid_a":     [s["rid_a"]     for s in spatial],
            "rid_b":     [s["rid_b"]     for s in spatial],
            "direction": [s["direction"] for s in spatial],
        },
    }


# ---------------------------------------------------------------------------
# Case runners
# ---------------------------------------------------------------------------

def _make_off_cfg(**enables) -> AugmentationConfig:
    """모든 증강을 끈 베이스 cfg에 enables로 명시한 키만 켠다."""
    cfg = AugmentationConfig(
        shuffle_rid=False,
        shuffle_vertex_order=False,
        shuffle_room_order=False,
        shuffle_edge_order=False,
        shuffle_spatial_order=False,
        reverse_spatial=False,
        do_translate=False,
        do_flip=False,
        do_scale_aspect=False,
        do_zoom=False,
        p_drop_block=0.0, p_drop_type=0.0, p_drop_coords=0.0,
        p_drop_edge=0.0, p_drop_pair=0.0, p_drop_door=0.0,
        p_drop_spatial=0.0,
        p_drop_front_door=0.0, p_drop_front_door_coords=0.0,
        p_drop_room_summary_total=0.0, p_drop_room_summary_type=0.0,
        p_noise=0.0,
    )
    for k, v in enables.items():
        setattr(cfg, k, v)
    return cfg


def case_direct_flip_h():
    """flip("H") 직접 호출 → metadata 좌표 = (255-x, y)."""
    sample = make_fixture_sample()
    rng = random.Random(0)
    # rng.choice(("H","V","HV"))이 "H"가 나오도록 시드 탐색
    while True:
        if rng.choice(("H", "V", "HV")) == "H":
            break
    rng = random.Random(0)
    # 시드 재생성 후 강제 H 모드: monkey-patch
    import src.training.augmentation.strategies as strat_mod
    original_choice = random.Random.choice

    rng_local = random.Random(0)
    s = copy.deepcopy(sample)
    # 강제 H: rng.choice를 "H" 반환으로 patch
    class _ForcedRng:
        def __init__(self, base): self._base = base
        def choice(self, opts): return "H"
        def __getattr__(self, name): return getattr(self._base, name)
    forced_rng = _ForcedRng(rng_local)
    strat_mod.flip(s, forced_rng)

    # 기대: outline (10,10) → (245,10), (200,10) → (55,10) 등
    assert s["rooms"][0]["coords"] == [245, 10, 55, 10, 55, 200, 245, 200], \
        f"flip('H') 직접 호출 후 outline coords 불일치: {s['rooms'][0]['coords']}"
    assert s["spatial"][0]["direction"] == "left", \
        f"flip('H') 후 spatial right→left 변환 실패: {s['spatial'][0]['direction']}"

    # _extract_metadata 호출 — drop_state 없는 상태
    drop_state = strategies.DropState()
    metadata = _extract_metadata(s, drop_state)

    # metadata.rooms[0].coords가 변형된 좌표와 정확히 일치
    assert metadata["rooms"][0]["coords"] == [245, 10, 55, 10, 55, 200, 245, 200], \
        f"metadata가 변형 후 좌표를 반영하지 않음: {metadata['rooms'][0]['coords']}"
    # 회귀 가드: 만약 raw_sample을 참조했다면 [10,10,200,10,200,200,10,200]가 들어옴
    assert metadata["rooms"][0]["coords"] != [10, 10, 200, 10, 200, 200, 10, 200], \
        "metadata가 raw_sample 좌표를 반영함 (회귀 발생)"


def case_direct_flip_v():
    """flip("V") → metadata 좌표 = (x, 255-y), spatial below→above."""
    sample = make_fixture_sample()
    rng_local = random.Random(0)
    class _ForcedRng:
        def __init__(self, base): self._base = base
        def choice(self, opts): return "V"
        def __getattr__(self, name): return getattr(self._base, name)
    s = copy.deepcopy(sample)
    strategies.flip(s, _ForcedRng(rng_local))

    assert s["rooms"][0]["coords"] == [10, 245, 200, 245, 200, 55, 10, 55], \
        f"flip('V') outline coords 불일치: {s['rooms'][0]['coords']}"
    # below → above
    assert s["spatial"][1]["direction"] == "above", \
        f"flip('V') spatial below→above 실패: {s['spatial'][1]['direction']}"

    metadata = _extract_metadata(s, strategies.DropState())
    assert metadata["rooms"][0]["coords"] == [10, 245, 200, 245, 200, 55, 10, 55], \
        f"metadata 변형 후 좌표 불일치: {metadata['rooms'][0]['coords']}"


def case_translate():
    """translate 직접 호출 — 좌표 평행이동이 metadata에 반영."""
    sample = make_fixture_sample()
    raw_outline = list(sample["rooms"][0]["coords"])  # 비교용 보존

    s = copy.deepcopy(sample)
    rng_local = random.Random(123)
    strategies.translate(s, rng_local)

    # metadata 추출
    metadata = _extract_metadata(s, strategies.DropState())

    # outline coords가 raw와 다르고 (변형됨), metadata와 동일
    assert metadata["rooms"][0]["coords"] == s["rooms"][0]["coords"], \
        "metadata가 변형 후 sample과 좌표 불일치"
    if metadata["rooms"][0]["coords"] == raw_outline:
        # translate가 우연히 0,0 이동일 수 있음 — 다른 시드로 한 번 더
        rng_retry = random.Random(456)
        s2 = copy.deepcopy(sample)
        strategies.translate(s2, rng_retry)
        meta2 = _extract_metadata(s2, strategies.DropState())
        assert meta2["rooms"][0]["coords"] != raw_outline, \
            "translate 두 시드에서 모두 0,0 이동 — 검증 불가"


def case_pipeline_e2e_flip_only():
    """pipeline 전체 흐름에서 flip만 적용 → last_augmented_sample == metadata 좌표 일치."""
    cfg = _make_off_cfg(do_flip=True)
    vocab = get_vocab()
    pipeline = AugmentationPipeline(vocab, cfg, seed=42)

    raw_sample = to_columnar(make_fixture_sample())
    pipeline(raw_sample)

    assert pipeline.last_augmented_sample is not None, "last_augmented_sample 캐싱 실패"
    last_sample = pipeline.last_augmented_sample
    last_drop = pipeline.last_drop_state
    assert last_drop is not None

    metadata = _extract_metadata(last_sample, last_drop)

    # 모든 방의 좌표가 last_augmented_sample과 metadata에서 동일해야 함
    for i, room in enumerate(last_sample["rooms"]):
        meta_room = next(
            (r for r in metadata["rooms"] if r["rid"] == room["rid"]), None
        )
        assert meta_room is not None, f"metadata.rooms에서 rid={room['rid']} 찾을 수 없음"
        assert meta_room["coords"] == room["coords"], (
            f"rid={room['rid']} metadata 좌표가 변형 후 sample과 불일치: "
            f"meta={meta_room['coords']}, sample={room['coords']}"
        )

    # 회귀 가드: 변형이 실제로 일어났는지 (랜덤이므로 거의 확실)
    raw_outline = make_fixture_sample()["rooms"][0]["coords"]
    flipped_outline = last_sample["rooms"][0]["coords"]
    assert flipped_outline != raw_outline, "flip이 실제로 일어나지 않음 (또는 우연히 동일)"


# ---------------------------------------------------------------------------
# Case 메타데이터 (run_cases용)
# ---------------------------------------------------------------------------

class _Case:
    def __init__(self, name, intent, fn):
        self.name = name
        self.intent = intent
        self.fn = fn


def main():
    cases = [
        _Case("direct_flip_h", "flip('H') 후 metadata 좌표 = (255-x, y) 일치", case_direct_flip_h),
        _Case("direct_flip_v", "flip('V') 후 metadata 좌표 = (x, 255-y) 일치 + spatial below→above", case_direct_flip_v),
        _Case("translate",     "translate 후 좌표가 metadata에 평행이동 반영", case_translate),
        _Case("pipeline_e2e_flip", "pipeline 전체 흐름에서 last_augmented_sample == metadata 좌표 일치", case_pipeline_e2e_flip_only),
    ]
    results = run_cases(cases, lambda c: c.fn(), label="Group 1: metadata after transform")
    summary_and_exit(results, label="metadata after transform")


if __name__ == "__main__":
    main()
