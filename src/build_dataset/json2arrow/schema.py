"""Arrow 데이터셋 스키마 정의 모듈.

JSONL 원본 데이터를 Arrow로 변환할 때 사용할 명시적 스키마(datasets.Features)를 정의한다.
중첩 list/dict 구조의 자동 타입 추론 오류를 방지하기 위해 스키마를 명시한다.

주요 설계 결정:
- front_door: nullable dict → Sequence(길이 0 또는 1)로 정규화
- spatial: 이형 타입 리스트([[int, int, str], ...]) → Sequence(dict)로 변환
- edges[].door: nullable → Sequence(길이 0 이상)로 정규화
"""

from __future__ import annotations

import datasets


def get_floorplan_features() -> datasets.Features:
    """평면도 Arrow 데이터셋의 스키마(Features) 반환.

    Returns:
        datasets.Features: 평면도 레코드의 명시적 Arrow 스키마.

    Notes:
        - front_door는 null 가능성 때문에 Sequence로 wrapping (길이 0 = 없음, 1 = 있음).
        - spatial의 각 항목은 [rid_a(int), rid_b(int), direction(str)] 이형 타입이므로
          {"rid_a", "rid_b", "direction"} 구조체로 변환하여 Arrow의 동형 타입 제약을 해결.
        - edges[].door도 동일하게 Sequence로 정규화.
        - Sequence 내부의 struct는 datasets.Features가 아닌 일반 dict로 정의해야 한다.
    """
    # 문 하나를 나타내는 구조체 (x, y, w, h)
    door_struct = {
        "x": datasets.Value("float32"),
        "y": datasets.Value("float32"),
        "w": datasets.Value("float32"),
        "h": datasets.Value("float32"),
    }

    return datasets.Features(
        {
            # 평면도 고유 식별자
            "plan_id": datasets.Value("string"),

            # 방 정보 리스트 (rid, type, coords)
            "rooms": datasets.Sequence(
                {
                    "rid": datasets.Value("int32"),
                    "type": datasets.Value("string"),
                    # flat integer array: [x1,y1,x2,y2,...]
                    "coords": datasets.Sequence(datasets.Value("int32")),
                }
            ),

            # 방 간 연결관계 + 문 정보
            # door는 Sequence로 정규화 (null → 빈 리스트)
            "edges": datasets.Sequence(
                {
                    "pair": datasets.Sequence(datasets.Value("int32")),
                    "door": datasets.Sequence(door_struct),
                }
            ),

            # 현관문 정보 (없으면 빈 리스트, 있으면 길이 1 리스트)
            "front_door": datasets.Sequence(door_struct),

            # 방 간 위치관계
            # 원본: [[rid_a, rid_b, direction], ...] → dict로 변환
            "spatial": datasets.Sequence(
                {
                    "rid_a": datasets.Value("int32"),
                    "rid_b": datasets.Value("int32"),
                    "direction": datasets.Value("string"),
                }
            ),
        }
    )
