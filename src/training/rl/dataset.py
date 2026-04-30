"""RL 훈련용 프롬프트 Dataset 모듈.

SFT Dataset과 달리 assistant 응답(output_tokens)을 생성하지 않는다.
모델이 RL 훈련 중 직접 completion을 생성하므로,
이 Dataset은 프롬프트(조건) + 메타데이터만 반환한다.

메타데이터는 보상 계산에 필요한 정보를 담고 있으며,
TRL reward function에 추가 컬럼으로 자동 전달된다.
"""

import logging
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.training.augmentation.pipeline import AugmentationPipeline, config_from_omegaconf
from src.training.augmentation.tokenizer import Vocab, load_vocab

logger = logging.getLogger(__name__)

# 평면도 생성기 시스템 프롬프트 (SFT와 동일)
SYSTEM_PROMPT = (
    "You are a floor plan generator. "
    "Given room conditions, generate complete floorplan coordinates."
)


class RLPromptDataset(Dataset):
    """RL 훈련용 프롬프트 Dataset.

    Arrow 포맷 데이터셋에서 샘플을 로드하고 증강을 적용한 뒤,
    chat template으로 감싼 프롬프트 문자열과 보상 계산용 메타데이터를 반환한다.

    출력 레이블은 생성하지 않는다 (모델이 RL 훈련 중 직접 생성).
    증강은 입력 조건에만 적용하며, 메타데이터는 증강(변형) 적용 후 샘플 + drop_state로 구성한다.

    Args:
        cfg: Hydra DictConfig. cfg.data, cfg.augmentation, cfg.model 섹션 참조.
        tokenizer: 커스텀 토큰이 포함된 AutoTokenizer.
        split: 사용할 데이터셋 split ("train", "validation", "test").
        seed: 증강 파이프라인의 랜덤 시드.
    """

    def __init__(
        self,
        cfg: DictConfig,
        tokenizer: AutoTokenizer,
        split: str = "train",
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_prompt_length: int = cfg.data.max_prompt_length
        self.split = split

        # Arrow 데이터셋 로드
        arrow_dir = Path(cfg.data.arrow_dir)
        if not arrow_dir.exists():
            raise FileNotFoundError(f"Arrow 데이터셋 경로를 찾을 수 없음: {arrow_dir}")

        dataset_dict = load_from_disk(str(arrow_dir))
        if split not in dataset_dict:
            raise ValueError(
                f"split '{split}'이 데이터셋에 없음. "
                f"가능한 split: {list(dataset_dict.keys())}"
            )
        self.dataset = dataset_dict[split]
        logger.info(f"[{split}] Arrow 데이터셋 로드 완료: {len(self.dataset)}개 샘플")

        # Vocab 로드 (파서에서 필요)
        self.vocab: Vocab = load_vocab(cfg.model.vocab_extension, cfg.model.tokenizer_dir)

        # 증강 파이프라인 초기화 (입력 조건 증강용)
        aug_config = config_from_omegaconf(cfg.augmentation)
        self.pipeline = AugmentationPipeline(self.vocab, aug_config, seed=seed)

    def __len__(self) -> int:
        """데이터셋 샘플 수를 반환한다.

        Returns:
            전체 샘플 수.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """단일 샘플을 로드하고 증강 + chat template 적용 후 반환한다.

        Mod Record: 이전 구조에서는 증강 전 원본 샘플로 metadata를 추출했다.
        그러나 모델이 보는 입력은 변형 증강(flip/scale/translate/zoom)이 적용되고
        삭제 증강(drop_block/drop_type/drop_coords/drop_edge 등)으로 일부 정보가
        가려진 상태이다. 보상은 모델이 본 입력 프롬프트의 제약 만족도로 평가되어야
        하므로, metadata는 변형 적용 + drop 반영을 모두 거친 "모델 시점" 정보만 담는다
        (drop된 항목은 metadata에서 제거되거나 마스킹됨). 따라서 reward 함수는
        metadata만 보고 자연스럽게 visible 정보 기준으로 채점한다.

        TRL GRPOTrainer는 이 dict의 모든 키를 reward function에 전달한다.

        Args:
            idx: 샘플 인덱스.

        Returns:
            dict:
                - prompt (list[dict]): chat template 형식의 프롬프트.
                    TRL이 내부적으로 chat template을 적용한다.
                - metadata (dict): 모델 시점 보상 계산용 메타데이터
                    (자세한 필드 의미는 _extract_metadata 도크스트링 참조).
        """
        raw_sample: dict[str, Any] = self.dataset[idx]

        # 증강 파이프라인 적용 (condition_tokens만 생성, output_tokens 불필요)
        # 호출 후 self.pipeline.last_augmented_sample / last_drop_state가 갱신됨
        condition_tokens, _ = self.pipeline(raw_sample)

        # 변형 증강 완료 + 삭제 미적용 상태의 ground-truth 샘플
        augmented_sample = self.pipeline.last_augmented_sample
        drop_state = self.pipeline.last_drop_state

        # 메타데이터 추출 (변형 후 좌표 + drop 상태)
        metadata = _extract_metadata(augmented_sample, drop_state)

        # 토큰 ID → 문자열 디코딩
        decoded_condition = self.tokenizer.decode(
            condition_tokens, skip_special_tokens=False
        )

        # TRL 형식의 conversational prompt 반환
        # TRL이 내부적으로 apply_chat_template()을 처리함
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": decoded_condition},
        ]

        return {
            "prompt": prompt,
            "metadata": metadata,
        }


def _extract_metadata(augmented_sample: dict, drop_state: Any) -> dict:
    """변형 증강 완료 샘플 + drop 상태에서 "모델 시점 metadata"를 구성한다.

    drop_state를 metadata 본체에 직접 반영하여, 모델이 입력 프롬프트에서 본 정보만
    metadata에 남긴다. 따라서 reward 함수는 drop_state를 별도로 참조할 필요 없이
    metadata만 보고 자연스럽게 visible 정보 기준으로 채점한다.

    drop 반영 규칙:
        - drop_block(rid): metadata.rooms에서 해당 방 제거
        - drop_type(rid): 해당 방의 type=""로 마스킹 (좌표는 보존, 모델이 봄)
        - drop_coords(rid): 해당 방의 coords=[]로 마스킹 (type은 보존)
        - drop_edge(idx): metadata.edges에서 제거
        - drop_pair(idx, mode): edge.pair를 [kept_rid](one) 또는 [](both)로 마스킹
        - drop_door(idx, mode): edge.door를 mode에 따라 부분 마스킹
        - drop_spatial(idx): metadata.spatial에서 제거
        - drop_front_door: metadata.front_door=None
        - drop_front_door_coords: front_door의 x,y만 None으로 마스킹
        - drop_room_summary_total: total_rooms=None (count_total 채점 비활성 신호)
        - drop_room_summary_types(t): type_counts에서 해당 타입 키 제거

    설계상 비대칭:
        total_rooms는 ROOM_SUMMARY로 모델에 노출된 N(=drop_block 포함 ground truth 카운트).
        반면 metadata.rooms는 drop_block 제외 visible 방만 담는다.
        len(metadata.rooms) ≠ metadata.total_rooms가 의도된 동작이다.

    Args:
        augmented_sample: AugmentationPipeline.last_augmented_sample.
        drop_state: AugmentationPipeline.last_drop_state (DropState 인스턴스).

    Returns:
        모델 시점 메타데이터 딕셔너리:
            - total_rooms (int|None): ROOM_SUMMARY로 노출된 전체 방 개수, drop 시 None.
            - type_counts (dict[str, int]): ROOM_SUMMARY로 노출된 타입별 개수.
            - rooms (list[dict]): visible 방 (drop_block 제외, drop_type/coords는 부분 마스킹).
            - edges (list[dict]): visible 엣지 (drop_edge 제외, drop_pair/drop_door 부분 마스킹).
            - spatial (list[dict]): visible 공간 관계 (drop_spatial 제외).
            - front_door (dict|None): drop 시 None, drop_coords 시 좌표만 None.
    """
    rooms_raw = augmented_sample.get("rooms", [])
    edges_raw = augmented_sample.get("edges", [])
    spatial_raw = augmented_sample.get("spatial", [])
    front_door_raw = augmented_sample.get("front_door")

    drop_block       = set(drop_state.drop_block)
    drop_type_rids   = set(drop_state.drop_type)
    drop_coords_rids = set(drop_state.drop_coords)
    drop_edge_idxs   = set(drop_state.drop_edge)
    drop_pair_map    = dict(drop_state.drop_pair)
    drop_door_map    = dict(drop_state.drop_door)
    drop_spatial_idxs = set(drop_state.drop_spatial)

    # --- rooms: drop_block 제거 + drop_type/coords 마스킹 ---
    rooms_visible: list[dict] = []
    for room in rooms_raw:
        rid = room.get("rid")
        if rid in drop_block:
            continue
        rooms_visible.append({
            "rid":    rid,
            "type":   "" if rid in drop_type_rids else room.get("type", ""),
            "coords": [] if rid in drop_coords_rids else list(room.get("coords", [])),
        })

    # --- ROOM_SUMMARY: total / type별 개수 ---
    # 전체 ground-truth 카운트 (drop_block 포함) — ROOM_SUMMARY로 노출된 값
    non_outline_full = [r for r in rooms_raw if r.get("type") != "outline"]
    full_total = len(non_outline_full)
    full_type_counts: dict[str, int] = {}
    for r in non_outline_full:
        t = r.get("type", "")
        full_type_counts[t] = full_type_counts.get(t, 0) + 1

    total_rooms = None if drop_state.drop_room_summary_total else full_total
    type_counts = {
        t: c for t, c in full_type_counts.items()
        if t not in drop_state.drop_room_summary_types
    }

    # --- edges: drop_edge 제거 + drop_pair/drop_door 마스킹 ---
    edges_visible: list[dict] = []
    for e_idx, edge in enumerate(edges_raw):
        if e_idx in drop_edge_idxs:
            continue

        new_edge = {
            "pair":  list(edge.get("pair", [])),
            "door":  list(edge.get("door", [])),
        }

        # drop_pair: edge의 RID 쌍 일부/전체 마스킹
        if e_idx in drop_pair_map:
            spec = drop_pair_map[e_idx]
            if isinstance(spec, tuple) and len(spec) == 2 and spec[0] == "one":
                new_edge["pair"] = [int(spec[1])]
            else:  # "both"
                new_edge["pair"] = []

        # drop_door: door 정보 부분 마스킹
        if e_idx in drop_door_map:
            mode = drop_door_map[e_idx]
            if mode == "all":
                new_edge["door"] = []
            elif mode == "position":
                new_edge["door"] = [
                    {"x": None, "y": None, "w": d.get("w"), "h": d.get("h")}
                    for d in edge.get("door", [])
                ]
            elif mode == "orientation":
                new_edge["door"] = [
                    {"x": d.get("x"), "y": d.get("y"), "w": None, "h": None}
                    for d in edge.get("door", [])
                ]

        edges_visible.append(new_edge)

    # --- spatial: drop_spatial 제거 ---
    spatial_visible = [
        dict(sp) for sp_idx, sp in enumerate(spatial_raw)
        if sp_idx not in drop_spatial_idxs
    ]

    # --- front_door: drop_front_door / drop_front_door_coords ---
    if drop_state.drop_front_door or front_door_raw is None:
        front_door = None
    elif drop_state.drop_front_door_coords:
        front_door = {
            "x": None, "y": None,
            "w": front_door_raw.get("w"), "h": front_door_raw.get("h"),
        }
    else:
        front_door = dict(front_door_raw)

    return {
        "total_rooms": total_rooms,
        "type_counts": type_counts,
        "rooms":       rooms_visible,
        "edges":       edges_visible,
        "spatial":     spatial_visible,
        "front_door":  front_door,
    }
