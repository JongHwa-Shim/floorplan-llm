"""데이터 증강 파이프라인 모듈.

Arrow에서 읽어온 샘플에 변형 증강 → 삭제 증강 → 토크나이징을 순서대로 적용하여
(condition_tokens, output_tokens) 쌍을 반환한다.

DataLoader의 __getitem__에서 호출하는 것을 가정하며, 매 호출마다
rng.seed(...)를 설정하지 않으면 랜덤하게 다른 증강이 적용된다.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

from src.training.augmentation.strategies import (
    DropState,
    compute_drop_state,
    compute_noise_state,
    flip,
    reverse_spatial_relation,
    scale_aspect,
    shuffle_edge_order,
    shuffle_rid,
    shuffle_room_order,
    shuffle_spatial_order,
    shuffle_vertex_order,
    translate,
    zoom,
)
from src.training.augmentation.tokenizer import (
    Vocab,
    build_condition_tokens,
    build_output_tokens,
    to_row_oriented,
)


# ---------------------------------------------------------------------------
# 증강 설정 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class AugmentationConfig:
    """증강 파이프라인 설정.

    Attributes:
        shuffle_rid: ShuffleRID 적용 여부.
        shuffle_vertex_order: ShuffleVertexOrder 적용 여부.
        shuffle_room_order: ShuffleRoomOrder 적용 여부.
        shuffle_edge_order: ShuffleEdgeOrder 적용 여부.
        shuffle_spatial_order: ShuffleSpatialOrder 적용 여부.
        reverse_spatial: ReverseSpatialRelation 적용 여부.
        p_drop_block: 방 블록 전체 삭제 확률.
        p_drop_type: 방 종류 토큰 삭제 확률.
        p_drop_coords: 방 좌표 블록 삭제 확률.
        p_drop_edge: 엣지 전체 삭제 확률.
        p_drop_pair: 엣지 RID 쌍 삭제 확률.
        p_drop_door: 엣지 문 정보 삭제 확률.
        p_drop_spatial: spatial 항목 삭제 확률.
        p_drop_front_door: 현관문 블록 전체 삭제 확률.
        p_drop_front_door_coords: 현관문 좌표만 삭제 확률.
        p_drop_room_summary_total: ROOM_SUMMARY 내 <TOTAL>N 쌍 삭제 확률.
        p_drop_room_summary_type: ROOM_SUMMARY 내 개별 <TYPE:t><COUNT>M 쌍 삭제 확률.
    """

    # 변형 기반 증강 플래그 (Shuffle 계열)
    shuffle_rid: bool = True
    shuffle_vertex_order: bool = True
    shuffle_room_order: bool = True
    shuffle_edge_order: bool = True
    shuffle_spatial_order: bool = True
    reverse_spatial: bool = True

    # 기하학적 변형 증강 플래그 (INPUT + OUTPUT 모두 반영)
    do_translate: bool = True
    do_flip: bool = True
    do_scale_aspect: bool = True
    scale_aspect_min: float = 0.7   # ScaleAspect 스케일 최솟값
    scale_aspect_max: float = 1.4   # ScaleAspect 스케일 상한값
    do_zoom: bool = True
    zoom_min: float = 0.7           # Zoom 스케일 최솟값
    zoom_max: float = 1.4           # Zoom 스케일 상한값

    # 삭제 기반 증강 확률
    p_drop_block: float = 0.20
    p_drop_type: float = 0.15
    p_drop_coords: float = 0.25
    p_drop_edge: float = 0.20
    p_drop_pair: float = 0.10
    p_drop_door: float = 0.10
    p_drop_spatial: float = 0.20
    p_drop_front_door: float = 0.15
    p_drop_front_door_coords: float = 0.10

    # ROOM_SUMMARY 삭제 증강 (INPUT 전용)
    p_drop_room_summary_total: float = 0.20
    p_drop_room_summary_type: float = 0.30

    # 좌표 노이즈 증강 (INPUT 전용)
    p_noise: float = 0.30
    noise_sigma: float = 3.0

    def to_drop_params(self) -> dict:
        """삭제 증강 확률을 compute_drop_state()용 딕셔너리로 변환한다.

        Returns:
            확률 파라미터 딕셔너리.
        """
        return {
            "p_drop_block":              self.p_drop_block,
            "p_drop_type":               self.p_drop_type,
            "p_drop_coords":             self.p_drop_coords,
            "p_drop_edge":               self.p_drop_edge,
            "p_drop_pair":               self.p_drop_pair,
            "p_drop_door":               self.p_drop_door,
            "p_drop_spatial":            self.p_drop_spatial,
            "p_drop_front_door":             self.p_drop_front_door,
            "p_drop_front_door_coords":      self.p_drop_front_door_coords,
            "p_drop_room_summary_total":     self.p_drop_room_summary_total,
            "p_drop_room_summary_type":      self.p_drop_room_summary_type,
        }

    def to_noise_params(self) -> dict:
        """노이즈 증강 파라미터를 compute_noise_state()용 딕셔너리로 변환한다.

        Returns:
            노이즈 파라미터 딕셔너리.
        """
        return {
            "p_noise":     self.p_noise,
            "noise_sigma": self.noise_sigma,
        }


def config_from_omegaconf(cfg) -> AugmentationConfig:
    """OmegaConf DictConfig에서 AugmentationConfig를 생성한다.

    Args:
        cfg: Hydra/OmegaConf DictConfig (augmentation 섹션).

    Returns:
        AugmentationConfig 객체.
    """
    transform_cfg = cfg.get("transform", {})
    noise_cfg = cfg.get("noise", {})
    return AugmentationConfig(
        shuffle_rid=cfg.shuffle.get("rid", True),
        shuffle_vertex_order=cfg.shuffle.get("vertex_order", True),
        shuffle_room_order=cfg.shuffle.get("room_order", True),
        shuffle_edge_order=cfg.shuffle.get("edge_order", True),
        shuffle_spatial_order=cfg.shuffle.get("spatial_order", True),
        reverse_spatial=cfg.shuffle.get("reverse_spatial", True),
        do_translate=transform_cfg.get("translate", True),
        do_flip=transform_cfg.get("flip", True),
        do_scale_aspect=transform_cfg.get("scale_aspect", True),
        scale_aspect_min=transform_cfg.get("scale_aspect_min", 0.7),
        scale_aspect_max=transform_cfg.get("scale_aspect_max", 1.4),
        do_zoom=transform_cfg.get("zoom", True),
        zoom_min=transform_cfg.get("zoom_min", 0.7),
        zoom_max=transform_cfg.get("zoom_max", 1.4),
        p_drop_block=cfg.drop.get("p_drop_block", 0.20),
        p_drop_type=cfg.drop.get("p_drop_type", 0.15),
        p_drop_coords=cfg.drop.get("p_drop_coords", 0.25),
        p_drop_edge=cfg.drop.get("p_drop_edge", 0.20),
        p_drop_pair=cfg.drop.get("p_drop_pair", 0.10),
        p_drop_door=cfg.drop.get("p_drop_door", 0.10),
        p_drop_spatial=cfg.drop.get("p_drop_spatial", 0.20),
        p_drop_front_door=cfg.drop.get("p_drop_front_door", 0.15),
        p_drop_front_door_coords=cfg.drop.get("p_drop_front_door_coords", 0.10),
        p_drop_room_summary_total=cfg.get("room_summary", {}).get("p_drop_total", 0.20),
        p_drop_room_summary_type=cfg.get("room_summary", {}).get("p_drop_type", 0.30),
        p_noise=noise_cfg.get("p_noise", 0.30),
        noise_sigma=noise_cfg.get("noise_sigma", 3.0),
    )


# ---------------------------------------------------------------------------
# 증강 파이프라인
# ---------------------------------------------------------------------------

class AugmentationPipeline:
    """증강 파이프라인.

    Arrow에서 읽어온 샘플을 받아 다음 6단계를 수행한다:
        1. deep copy + row-oriented 변환
        2. 변형 기반 증강 (Shuffle 계열)
        3. 삭제 기반 증강 상태 계산 (DropState)
        4. 조건(입력) 토큰 시퀀스 생성
        5. 정답(출력) 토큰 시퀀스 생성
        6. (condition_tokens, output_tokens) 반환

    DataLoader의 __getitem__에서 per-sample로 호출한다.
    증강 내역은 last_drop_state, last_applied_shuffles에 저장되므로
    검증 스크립트에서 바로 조회 가능하다.

    Args:
        vocab: Vocab 객체.
        cfg: AugmentationConfig 객체.
        seed: 고정 시드 (None이면 랜덤 — DataLoader 사용 시 권장).
    """

    def __init__(
        self,
        vocab: Vocab,
        cfg: AugmentationConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.vocab = vocab
        self.cfg = cfg or AugmentationConfig()
        self._seed = seed
        self._rng = random.Random(seed)

        # 마지막 호출의 증강 내역 (검증용)
        self.last_drop_state: DropState | None = None
        self.last_applied_shuffles: list[str] = []

    def __call__(self, raw_sample: dict) -> tuple[list[int], list[int]]:
        """샘플에 증강을 적용하고 토큰 ID 시퀀스를 반환한다.

        Args:
            raw_sample: Arrow에서 읽어온 raw 딕셔너리
                (columnar 포맷 또는 이미 row-oriented).

        Returns:
            (condition_tokens, output_tokens) 튜플.
                - condition_tokens: 조건(입력) 토큰 ID 리스트.
                - output_tokens: 정답(출력) 토큰 ID 리스트.
        """
        # 1단계: deep copy + row-oriented 변환
        sample = to_row_oriented(raw_sample)
        sample = copy.deepcopy(sample)

        # 2단계: 변형 기반 증강
        # 순서: Shuffle 계열 → 기하학적 변형 (translate/flip/scale_aspect/zoom)
        applied_shuffles: list[str] = []
        cfg = self.cfg
        rng = self._rng

        if cfg.shuffle_rid:
            shuffle_rid(sample, rng)
            applied_shuffles.append("ShuffleRID")

        if cfg.shuffle_vertex_order:
            shuffle_vertex_order(sample, rng)
            applied_shuffles.append("ShuffleVertexOrder")

        if cfg.shuffle_room_order:
            shuffle_room_order(sample, rng)
            applied_shuffles.append("ShuffleRoomOrder")

        if cfg.shuffle_edge_order:
            shuffle_edge_order(sample, rng)
            applied_shuffles.append("ShuffleEdgeOrder")

        if cfg.shuffle_spatial_order:
            shuffle_spatial_order(sample, rng)
            applied_shuffles.append("ShuffleSpatialOrder")

        if cfg.reverse_spatial:
            reverse_spatial_relation(sample, rng)
            applied_shuffles.append("ReverseSpatialRelation")

        # 기하학적 변형 (INPUT + OUTPUT 모두 반영)
        if cfg.do_translate:
            translate(sample, rng)
            applied_shuffles.append("Translate")

        if cfg.do_flip:
            flip(sample, rng)
            applied_shuffles.append("Flip")

        if cfg.do_scale_aspect:
            scale_aspect(sample, rng, scale_min=cfg.scale_aspect_min, scale_max=cfg.scale_aspect_max)
            applied_shuffles.append("ScaleAspect")

        if cfg.do_zoom:
            zoom(sample, rng, zoom_min=cfg.zoom_min, zoom_max=cfg.zoom_max)
            applied_shuffles.append("Zoom")

        # 3단계: 삭제 기반 증강 상태 계산
        drop_state = compute_drop_state(sample, cfg.to_drop_params(), rng)

        # 좌표 노이즈 상태 계산 (INPUT 전용, sample 수정 없음)
        drop_state.noise_room_coords = compute_noise_state(sample, cfg.to_noise_params(), rng)

        # 마지막 호출 내역 저장 (검증용)
        self.last_drop_state = drop_state
        self.last_applied_shuffles = applied_shuffles

        # 4단계: 조건(입력) 토큰 시퀀스 생성
        condition_tokens = build_condition_tokens(sample, drop_state, self.vocab)

        # 5단계: 정답(출력) 토큰 시퀀스 생성 (항상 full information)
        output_tokens = build_output_tokens(sample, self.vocab)

        # 6단계: 반환
        return condition_tokens, output_tokens

    def augmented_summary(self) -> str:
        """마지막 호출에서 적용된 증강 전략 요약 문자열을 반환한다 (검증용).

        Returns:
            "ShuffleRID, ShuffleVertexOrder, DropBlock(rids=[2])" 형태의 문자열.
            호출 전이면 "없음".
        """
        parts = list(self.last_applied_shuffles)
        if self.last_drop_state:
            drop_summary = self.last_drop_state.summary()
            if drop_summary != "없음":
                parts.append(drop_summary)
        return ", ".join(parts) if parts else "없음"

    def reset_rng(self, seed: int | None = None) -> None:
        """RNG를 재초기화한다.

        Args:
            seed: 새 시드 (None이면 랜덤).
        """
        self._seed = seed
        self._rng = random.Random(seed)
