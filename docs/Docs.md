# 프로젝트 설계 문서 (Docs)

좌표 기반 평면도 생성 LLM — 상세 설계 및 구현 명세

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [LLM이 평면도를 표현하는 방식](#2-llm이-평면도를-표현하는-방식)
3. [커스텀 토큰 체계](#3-커스텀-토큰-체계)
4. [데이터 형식 정의](#4-데이터-형식-정의)
5. [전체 파이프라인](#5-전체-파이프라인)
6. [Step 1: 평면도 PNG → JSONL 추출](#6-step-1-평면도-png--jsonl-추출)
7. [Step 2: Vocabulary 빌드](#7-step-2-vocabulary-빌드)
8. [Step 3: JSONL → Arrow 변환](#8-step-3-jsonl--arrow-변환)
9. [Step 4: 데이터 증강 + 토크나이징](#9-step-4-데이터-증강--토크나이징)
10. [Embedding Alignment: 새 토큰 Embedding 워밍업](#10-embedding-alignment-새-토큰-embedding-워밍업)
11. [Step 5: LLM 학습](#11-step-5-llm-학습)
12. [Step 6: 추론 및 시각화](#12-step-6-추론-및-시각화)

---

## 1. 프로젝트 개요

### 목표

사용자가 자연어 또는 구조화된 조건을 입력하면, LLM이 각 방의 꼭지점 좌표를 포함한 완전한 평면도를 생성하는 모델을 구축한다.

### 핵심 설계 원칙

| 원칙 | 내용 |
|------|------|
| **평면도 = 토큰 시퀀스** | 이미지가 아닌 구조화된 커스텀 토큰 시퀀스로 평면도를 표현 |
| **GPT 스타일 조건부 생성** | MASK 방식이 아닌, 조건을 보고 전체 출력을 순차 생성 |
| **2-Layer 저장 구조** | 사람이 읽을 수 있는 JSONL 원본 + 학습용 Arrow 바이너리 |
| **증강 원칙 (변형 → 출력 → 삭제)** | 변형 증강을 먼저 적용 후 출력 확정, 이후 삭제 증강을 입력에만 적용 |
| **독립적 증강 파이프라인** | DataLoader와 완전 분리된 독립 증강 클래스 설계 |

### 전체 파이프라인 요약

```
① 데이터셋 구축
   평면도 PNG → 정보 추출 → JSONL → Vocab 빌드 → Arrow 변환 → 증강 파이프라인

② LLM 훈련
   Pretrained LLM + 커스텀 토큰 → Embedding Alignment → 2-Stage Fine-tune (SFT → GRPO) → 평면도 생성 모델

③ 추론 + 시각화
   조건 입력 → 모델 추론 → 토큰 시퀀스 → 좌표 복원 → 평면도 시각화
```

---

## 2. LLM이 평면도를 표현하는 방식

### 평면도 = 토큰 시퀀스

평면도의 각 구성 요소를 커스텀 토큰으로 표현한다. "거실이 (100,200)~(200,300) 영역에 있다"는 정보는 다음과 같이 토큰화된다.

```
<ROOM> <RID:1> <TYPE:livingroom> <X:100> <Y:200> <X:200> <Y:200> <X:200> <Y:300> <X:100> <Y:300> <END_ROOM>
```

좌표는 `<ROOM>` ~ `<END_ROOM>` 사이에 별도 래퍼(`<COORDS>`) 없이 직접 나열된다. 좌표값 `100`을 텍스트로 쓰면 LLM 토크나이저에 의해 여러 서브워드로 쪼개지지만, `<X:100>`이라는 단일 전용 토큰을 사용하면 시퀀스 길이가 대폭 감소한다.

### 조건부 생성 방식

모델은 조건(입력)을 받아 전체 평면도(출력)를 자동회귀적으로 생성한다.

**입력으로 줄 수 있는 조건:**
- 방 종류와 개수 (`<ROOM_SUMMARY>`)
- 일부 방의 좌표 (`<ROOM>` ~ `<END_ROOM>` 내 좌표 토큰)
- 방 간 연결관계와 문 위치 (`<EDGE>` ~ `<END_EDGE>`)
- 방 간 위치관계 (`<SP>` ~ `<END_SP>`)
- 현관문 위치 (`<FRONT_DOOR>` ~ `<END_DOOR>`)

**모델이 출력하는 것:**
- 현관문 정보 (`<FRONT_DOOR>`)
- 모든 방의 종류 + 꼭지점 좌표 리스트 (ROOMS)
- 전체 인테리어 문 정보 (`<DOOR>` ~ `<END_DOOR>` 목록)

조건이 많을수록 거의 재구성에 가깝고, 조건이 적을수록 모델이 자율적으로 판단해야 하는 범위가 넓어진다. 데이터 증강의 "삭제" 전략이 이 다양한 난이도를 학습 데이터로 제공한다.

---

## 3. 커스텀 토큰 체계

기존 LLM의 Vocabulary에 평면도 도메인 전용 토큰 ~1000개를 추가한다.

### 토큰 카테고리

| 카테고리 | 토큰 예시 | 개수 | 초기화 |
|---------|---------|------|--------|
| 좌표 X | `<X:0>` ~ `<X:255>` | 256 | Sinusoidal (연속성 반영) |
| 좌표 Y | `<Y:0>` ~ `<Y:255>` | 256 | Sinusoidal (연속성 반영) |
| 구조 | `<INPUT>` `<END_INPUT>` `<OUTPUT>` `<END_OUTPUT>` `<ROOM>` `<END_ROOM>` `<EDGE>` `<END_EDGE>` `<SP>` `<END_SP>` `<ROOM_SUMMARY>` `<END_ROOM_SUMMARY>` `<TOTAL>` `<COUNT>` `<SEP_DOOR>` `<DOOR>` `<END_DOOR>` `<NO_DOOR>` `<FRONT_DOOR>` 등 | ~19 | 랜덤 |
| 방 종류 | `<TYPE:livingroom>` `<TYPE:bedroom>` `<TYPE:kitchen>` `<TYPE:bathroom>` `<TYPE:entrance>` 등 | ~8 | 랜덤 |
| 방 ID | `<RID:0>` ~ `<RID:15>` | 16 | 랜덤 |
| 위치관계 | `<REL:right>` `<REL:left>` `<REL:above>` `<REL:below>` `<REL:right-above>` `<REL:right-below>` `<REL:left-above>` `<REL:left-below>` | 8 | 랜덤 |

> **숫자 토큰 (`<TOTAL>` / `<COUNT>` 뒤):** `<TOTAL>` 및 `<COUNT>` 레이블 토큰 뒤에 오는 숫자(개수)는 별도의 커스텀 토큰(`<TOTAL:N>` 형태)을 만들지 않고, **LLM 기본 어휘에 이미 존재하는 숫자 토큰**을 그대로 사용한다. 예: `<TOTAL> 7` (7은 LLM 기본 토큰). 이 덕분에 vocab 크기를 줄이고 숫자 표현에 대한 LLM의 기존 이해를 그대로 활용할 수 있다.

> **좌표 토큰 초기화:** 숫자 간의 연속적 관계(100과 101은 가깝다)를 반영하기 위해 Sinusoidal 위치 인코딩 등을 활용한 초기화를 사용한다. 나머지 토큰은 랜덤 초기화 후 학습 중에 의미를 잡아간다.

### 구현 유의사항: LLM별 Tokenizer 호환

1. **ID를 직접 매기지 말 것.** `base_vocab_size=32000` 같은 하드코딩은 LLM마다 vocab 크기가 달라 충돌이 발생한다 (LLaMA2=32K, LLaMA3=128K, Qwen2.5=152K 등). 반드시 `tokenizer.add_tokens()`로 위임한다.
2. **토큰 문자열 목록만 정의하고 ID 매핑은 tokenizer에서 추출한다.** `tokenizer.convert_tokens_to_ids(token)`으로 매핑 구성.
3. **확장된 tokenizer를 `tokenizer.save_pretrained()`로 저장한다.**
4. **`model.resize_token_embeddings(len(tokenizer))` 필수 호출.** Embedding/lm_head 행렬 크기를 맞춰야 한다.
5. **토큰 목록의 순서를 고정한다.** 순서가 바뀌면 동일 토큰에 다른 ID가 부여될 수 있다.

---

## 4. 데이터 형식 정의

### 4.1 Layer 1: JSONL 원본 데이터

하나의 평면도 = 하나의 JSON 라인.

| 필드 | 타입 | 설명 |
|------|------|------|
| `plan_id` | string | 평면도 고유 식별자 |
| `rooms` | array | 모든 방 정보 목록 |
| `rooms[].rid` | int | 방 고유 ID (평면도 내에서) |
| `rooms[].type` | string | 방 종류 (`livingroom`, `bedroom`, `outline` 등) |
| `rooms[].coords` | int[] | 꼭지점 좌표 flat array `[x1,y1,x2,y2,...]` |
| `edges` | array | 인접한 방 쌍 + 문 정보 |
| `edges[].pair` | [int,int] | 연결된 방 ID 쌍 |
| `edges[].doors` | array | 문 정보 목록 (없으면 빈 리스트 `[]`) |
| `edges[].doors[].x` | int | 문 중심 x 좌표 |
| `edges[].doors[].y` | int | 문 중심 y 좌표 |
| `edges[].doors[].w` | int | 문 폭 |
| `edges[].doors[].h` | int | 문 높이 |
| `front_door` | object\|null | 현관문 정보 (`{x, y, w, h}` 또는 `null`) |
| `spatial` | array | 방 간 위치관계 `[[rid_a, rid_b, "direction"], ...]` |

**JSONL 예시:**

```json
{
  "plan_id": "fp_00123",
  "rooms": [
    {"rid": 0, "type": "outline",     "coords": [80,30, 80,220, 210,220, 210,30]},
    {"rid": 1, "type": "livingroom",  "coords": [100,200, 100,300, 200,300, 200,200]},
    {"rid": 2, "type": "bedroom",     "coords": [200,200, 200,300, 300,300, 300,200]},
    {"rid": 3, "type": "kitchen",     "coords": [100,300, 100,400, 200,400, 200,300]}
  ],
  "edges": [
    {"pair": [1,2], "doors": [{"x": 200, "y": 250, "w": 2, "h": 10}]},
    {"pair": [2,3], "doors": [{"x": 250, "y": 300, "w": 10, "h": 2}]},
    {"pair": [1,3], "doors": []}
  ],
  "front_door": {"x": 128, "y": 32, "w": 8, "h": 2},
  "spatial": [[1,2,"right"], [1,3,"below"], [2,3,"right-below"]]
}
```

**설계 결정:**
- **coords 저장 규칙:** `[x1,y1,x2,y2,...]` flat array. `coords[0::2]`=x좌표, `coords[1::2]`=y좌표로 즉시 분리 가능.
- **Edge 정의:** 픽셀이 맞닿아 있는(인접한) 방 쌍 전체. 문 유무와 무관하게 경계 픽셀을 공유하는 모든 방 쌍이 edge로 등록된다.
- **Front Door 독립 필드:** 현관문(G=15)은 방과 방을 연결하지 않으므로 `edges`가 아닌 별도 `front_door` 필드로 관리.

### 4.2 Layer 2: Arrow 데이터 (구조화 저장)

JSONL을 파싱하여 HuggingFace `datasets` 라이브러리의 Arrow 포맷으로 저장. 토크나이징은 이 단계에서 수행하지 않고 증강 단계에서 동적으로 적용한다.

**Arrow가 JSONL보다 나은 이유:**
- Memory-mapped I/O → RAM보다 큰 데이터셋도 처리 가능
- Columnar format → 특정 필드만 선택 로드 가능
- HuggingFace `datasets` 생태계와 완벽 호환
- 매 iteration마다 JSON 파싱을 반복하지 않아 학습 속도 향상

**스키마 정규화:**
- `front_door: null` → `[]` (길이 0 리스트로 정규화, Arrow 스키마 일관성 유지)
- `spatial: [[int,int,str]]` → `[{"rid_a":int, "rid_b":int, "direction":str}]`

**토크나이징을 Arrow 단계에서 하지 않는 이유:** 조건부 생성 (입력/출력 분리) + 삭제 방식 증강을 반영하면, 증강 단계에서 동적으로 토큰 시퀀스를 구성해야 한다. Arrow에는 구조화된 데이터를 저장하고 토크나이징은 증강 후에 수행한다. 이 덕분에 동일한 Arrow 데이터셋을 어떤 LLM 토크나이저에도 적용 가능하다.

### 4.3 입출력 시퀀스 구조

#### ROOM_SUMMARY 형식

```
<ROOM_SUMMARY> <TOTAL> 전체수 <TYPE:종류1> <COUNT> 종류1수 <TYPE:종류2> <COUNT> 종류2수 ... <END_ROOM_SUMMARY>
```

- `<TOTAL>` + 숫자: `<TOTAL>` 레이블 토큰 뒤에 LLM 기본 숫자 토큰으로 전체 방 개수 표기.
- `<TYPE:t> <COUNT>` + 숫자: 해당 종류의 방 개수. 증강 시 `<TOTAL>` 쌍 또는 개별 타입 쌍을 독립적으로 삭제 가능.

증강 예시 — 총 5개 방 중 침실 정보만 유지:
```
<ROOM_SUMMARY> <TOTAL> 5 <TYPE:bedroom> <COUNT> 2 <END_ROOM_SUMMARY>
```
→ 모델은 "총 5개 방인데 침실 2개만 알려줬으니, 나머지 3개는 직접 판단해야 한다"는 상황을 이해한다.

#### 전체 정보 입출력 (증강 미적용)

`<INPUT>` ~ `<END_INPUT>`, `<OUTPUT>` ~ `<END_OUTPUT>` 사이의 토큰은 줄바꿈·공백 없이 연속으로 이어붙인다. 아래 예시는 가독성을 위해 줄바꿈을 표기한 것이다.

**입력 (조건):**
```
<INPUT>
  <ROOM_SUMMARY> <TOTAL> 3 <TYPE:kitchen> <COUNT> 1 <TYPE:livingroom> <COUNT> 1 <TYPE:bedroom> <COUNT> 1 <END_ROOM_SUMMARY>
  <FRONT_DOOR> <X:128> <Y:32> <SEP_DOOR> <X:8> <Y:2> <END_DOOR>
  <ROOM> <RID:1> <TYPE:livingroom> <X:100> <Y:200> <X:100> <Y:300> <X:200> <Y:300> <X:200> <Y:200> <END_ROOM>
  <ROOM> <RID:2> <TYPE:bedroom> <X:200> <Y:200> <X:200> <Y:300> <X:300> <Y:300> <X:300> <Y:200> <END_ROOM>
  <ROOM> <RID:3> <TYPE:kitchen> <X:100> <Y:300> <X:100> <Y:400> <X:200> <Y:400> <X:200> <Y:300> <END_ROOM>
  <EDGE> <RID:1> <RID:2> <DOOR> <X:200> <Y:250> <SEP_DOOR> <X:2> <Y:10> <END_DOOR> <END_EDGE>
  <EDGE> <RID:2> <RID:3> <DOOR> <X:250> <Y:300> <SEP_DOOR> <X:10> <Y:2> <END_DOOR> <END_EDGE>
  <EDGE> <RID:1> <RID:3> <NO_DOOR> <END_EDGE>
  <SP> <RID:1> <RID:2> <REL:right> <END_SP>
  <SP> <RID:1> <RID:3> <REL:below> <END_SP>
  <SP> <RID:2> <RID:3> <REL:right-below> <END_SP>
<END_INPUT>
```

**출력 (정답 — 항상 전체 정보):**
```
<OUTPUT>
  <FRONT_DOOR> <X:128> <Y:32> <SEP_DOOR> <X:8> <Y:2> <END_DOOR>
  <ROOM> <TYPE:livingroom> <X:100> <Y:200> <X:100> <Y:300> <X:200> <Y:300> <X:200> <Y:200> <END_ROOM>
  <ROOM> <TYPE:bedroom> <X:200> <Y:200> <X:200> <Y:300> <X:300> <Y:300> <X:300> <Y:200> <END_ROOM>
  <ROOM> <TYPE:kitchen> <X:100> <Y:300> <X:100> <Y:400> <X:200> <Y:400> <X:200> <Y:300> <END_ROOM>
  <DOOR> <X:200> <Y:250> <SEP_DOOR> <X:2> <Y:10> <END_DOOR>
  <DOOR> <X:250> <Y:300> <SEP_DOOR> <X:10> <Y:2> <END_DOOR>
<END_OUTPUT>
```

**출력 구조 핵심 사항:**
- 출력의 방(`<ROOM>`)에는 `<RID:N>`이 없다. `<TYPE:xxx>`와 좌표만 포함한다.
- 문(`<DOOR>`)은 `<EDGE>` 블록 없이 독립적으로 나열된다. `<SEP_DOOR>`가 중심 좌표(cx, cy)와 크기(w, h)를 구분한다.
- 현관문(`<FRONT_DOOR>`)도 동일한 형식 (`<SEP_DOOR>`로 중심좌표/크기 구분).

**Chat Template (Qwen2.5 형식 기준, LLM 로드 및 훈련 파트에서 구현):**
```
<|im_start|>system
You are a floor plan generator. Given room conditions, generate complete floorplan coordinates.<|im_end|>
<|im_start|>user
<INPUT>...<END_INPUT><|im_end|>
<|im_start|>assistant
<OUTPUT>...<END_OUTPUT><|im_end|>
```

#### 학습 시 Loss 처리

Chat template으로 구성된 전체 시퀀스에서 **system + user 턴(입력) 부분의 loss는 무시 (-100으로 마스킹)**, assistant 턴(`<OUTPUT>` ~ `<END_OUTPUT>`) 부분만 학습한다.

---

## 5. 전체 파이프라인

```
평면도 PNG (RPLAN 데이터셋)
        │
        ▼
┌─────────────────────────┐
│  Step 1: 정보 추출       │  PNG → BGRA 채널 분리 → 방/문/Edge/Spatial 추출
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│  JSONL 원본 데이터       │     │  Step 2: Vocab 빌드      │
│  (사람이 읽는 형태)       │     │  토큰 목록 + 토크나이저   │
└──────────┬──────────────┘     └──────────┬──────────────┘
           │                               │
           ▼                               │
┌─────────────────────────┐                │
│  Step 3: Arrow 변환      │                │
│  JSONL → 구조화 데이터    │                │
└──────────┬──────────────┘                │
           │                               │
           ▼                               ▼
┌──────────────────────────────────────────────────────┐
│  Step 4: 데이터 증강 + 토크나이징                      │
│  변형 증강 → 삭제 증강 → 토큰 ID 시퀀스 생성           │
└──────────────────────────┬───────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────┐
│  Embedding Alignment: 새 토큰 Embedding 워밍업                   │
│  새 커스텀 토큰 embed + lm_head 행만 훈련              │
└──────────────────────────┬───────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────┐
│  Step 5: LLM Fine-tuning: SFT → GRPO(GDPO)             │
│  condition + output 토큰 → 평면도 생성 모델            │
└──────────────────────────┬───────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────┐
│  Step 6: 추론 + 시각화                                  │
│  조건 입력 → 토큰 시퀀스 → 좌표 복원 → 평면도 이미지    │
└──────────────────────────────────────────────────────┘
```

| Step | 단계 | 입력 | 출력 |
|------|------|------|------|
| 1 | 평면도 정보 추출 | RPLAN PNG (BGRA) | JSONL 샤드 |
| 2 | Vocabulary 빌드 | Pretrained 토크나이저 | vocab_extension.json + 확장 토크나이저 |
| 3 | Arrow 변환 | JSONL | Arrow DatasetDict (train/val/test) |
| 4 | 증강 + 토크나이징 | Arrow + 증강 설정 | (condition_tokens, output_tokens) |
| Embedding Alignment | 새 토큰 Embedding 워밍업 | 토큰 시퀀스 배치 | 워밍업된 embed_tokens + lm_head |
| SFT | LoRA Fine-tuning | HF Hub base model + `partial_state.pt` + 토큰 시퀀스 배치 | LoRA adapter Fine-tuned 모델 |
| GRPO | GDPO 강화학습 | HF Hub base + `partial_state.pt` + SFT adapter + 프롬프트 배치 | RL LoRA adapter |
| 6 | 추론 + 시각화 | 조건 입력 (JSONL/Arrow/txt) | 평면도 JSON + 토큰 텍스트 + 이미지 |

---

## 6. Step 1: 평면도 PNG → JSONL 추출

### RPLAN 데이터셋 채널 구조

RPLAN PNG는 BGRA 4채널 이미지 (256×256px)이며 각 채널이 다른 정보를 담는다.

| 채널 | 내용 | 주요 값 |
|------|------|--------|
| B (ch0) | 구조 레이블 | 벽=127, 현관문=255 |
| G (ch1) | 공간 타입 | 0~12: 방 종류, 13: 외벽 외부, 14: exterior_wall, 15: 현관문, 17: 인테리어 문 |
| R (ch2) | 방 인스턴스 ID | 방마다 고유한 픽셀값 |
| A (ch3) | 영역 구분 | 외부=0, 내부=255 |

### 10단계 처리 파이프라인

```
0. PNG 로드 + BGRA 채널 분리           (channel_parser.py)
1. 방 타입별 CCL (Connected Component Labeling)  (room_extractor.py)
2. 노이즈 제거 (min_room_area < 30px 제외)
3. 직교 폴리곤 근사 → 꼭지점 좌표 추출
4. 외곽선(outline) 추출 (외벽 외부 차집합)
5. 현관문 추출 (G==15, 가장 큰 컴포넌트)  (door_extractor.py)
6. 인테리어 문 추출 (G==17, L자-형 분해)
7. Raster scan 순서로 정렬 (centroid y→x)   (serializer.py)
8. Edge 구성 (직접 인접 + 문 연결)          (edge_builder.py)
9. Spatial 관계 계산 (8방위)               (spatial_calculator.py)
10. JSONL 직렬화 + 샤드 저장
```

### 핵심 알고리즘: 직교 폴리곤 추출

평면도의 방은 항상 직각으로 이루어진 다각형이다. OpenCV `approxPolyDP`는 일반 근사를 수행하므로 직각 보장이 안 된다. 대신 전용 직교 폴리곤 추출 알고리즘을 구현한다.

```
1. 방 픽셀 마스크에서 외곽선(contour) 추출
2. 방향 전환점(코너)만 추출: 수평 → 수직 또는 수직 → 수평으로 방향이 바뀌는 점
3. Canonical 순서 정규화: 시계방향(CW), top-left 코너 시작
```

### 핵심 알고리즘: L자형 문 분해

인테리어 문(G=17)이 두 방의 경계에서 병합되어 L자형으로 나타날 때 자동으로 두 개의 직사각형 문으로 분해한다.

```
1. 문 컴포넌트의 투영 프로파일 계산 (수평/수직 방향)
2. 프로파일의 Valley(최솟값 구간) 탐지
3. Valley 기준으로 재귀 분해
4. Peak 구간은 직사각형 bbox로 보정
```

### 방 타입 병합 (room_type_merge.json)

RPLAN의 상세 방 타입을 더 일반적인 타입으로 병합한다.

| 원본 타입 | 병합 후 타입 |
|---------|------------|
| `masterroom`, `childroom` | `bedroom` |
| `diningroom` | `livingroom` |
| `walkin` | `storage` |

### 주요 모듈

| 파일 | 주요 클래스/함수 | 역할 |
|------|----------------|------|
| `channel_parser.py` | `ChannelData`, `load_bgra_image()`, `parse_channels()` | BGRA 채널 분리 |
| `room_extractor.py` | `RoomInstance`, `extract_room_instances()`, `extract_polygon_coords()` | 방 분리 + 직교 폴리곤 |
| `door_extractor.py` | `DoorInstance`, `extract_front_door()`, `extract_interior_doors()`, `decompose_door_component()` | 문 추출 + L자 분해 |
| `edge_builder.py` | `EdgeRecord`, `build_edges()` | Edge 구성 |
| `spatial_calculator.py` | `build_spatial_relations()`, `_compute_direction()` | 8방위 공간관계 |
| `serializer.py` | `sort_rooms_raster_order()`, `build_plan_record()`, `append_to_jsonl()` | JSONL 직렬화 |

---

## 7. Step 2: Vocabulary 빌드

### 목적

Pretrained LLM의 토크나이저에 평면도 전용 커스텀 토큰을 추가하고, 추후 학습 및 추론에서 동일하게 사용할 수 있도록 저장한다.

### 처리 흐름

```
1. config의 model.hub_id로 Pretrained 토크나이저 로드 (예: Qwen/Qwen2.5-Coder-7B)
2. token_definitions.py에서 커스텀 토큰 목록 생성
3. tokenizer.add_tokens(custom_tokens) 호출
4. vocab_extension.json 저장 (토큰 → ID 매핑)
5. 확장된 tokenizer 저장 (save_pretrained)
```

**저장 경로:** `data/models/{model.name}/tokenization/`
- 모델명별로 독립된 디렉토리에 저장되어 여러 베이스 모델을 동시에 관리 가능
- `model.name`은 `model.hub_id`의 slash 뒤 부분 (예: `Qwen/Qwen2.5-Coder-7B` → `Qwen2.5-Coder-7B`)

### vocab_extension.json 구조

```json
{
  "coord_x": {"<X:0>": 152000, "<X:1>": 152001, ...},
  "coord_y": {"<Y:0>": 152256, "<Y:1>": 152257, ...},
  "room_id": {"<RID:0>": 152512, ...},
  "room_type": {"<TYPE:livingroom>": ..., ...},
  "spatial_rel": {"<REL:right>": ..., ...},
  "structure": {"<INPUT>": ..., "<END_INPUT>": ..., "<OUTPUT>": ..., "<END_OUTPUT>": ..., ...}
}
```

### 주요 모듈

| 파일 | 역할 |
|------|------|
| `token_definitions.py` | 카테고리별 커스텀 토큰 목록 정의 (순서 고정) |
| `vocab_builder.py` | HuggingFace 토크나이저 확장 + vocab_extension.json 저장 |

---

## 8. Step 3: JSONL → Arrow 변환

### 목적

사람이 읽는 JSONL 원본을 학습용 고속 Arrow 바이너리로 변환하고, train/val/test로 분리한다.

### 처리 흐름

```
1. JSONL 샤드 파일 목록 수집 (floorplans_*.jsonl)
2. 각 레코드를 Arrow 스키마에 맞게 정규화
   - front_door: null → []
   - spatial: [[a,b,"dir"]] → [{"rid_a":a, "rid_b":b, "direction":"dir"}]
   - doors: 단일 dict → list로 정규화
3. HuggingFace Dataset 생성
4. train/validation/test split 적용
5. 각 split을 Arrow 포맷으로 저장
6. 샘플 단위 검증 수행 (Arrow ↔ JSONL 원본 비교)
```

### 출력 구조

```
data/dataset/processed_dataset/rplan/arrow/
├── train/       (기본 99.4%)
├── validation/  (기본 0.1%)
└── test/        (기본 0.5%)
```

### 주요 모듈

| 파일 | 역할 |
|------|------|
| `schema.py` | 명시적 Arrow 스키마 정의 (`get_floorplan_features()`) |
| `converter.py` | JSONL 파싱 + 스키마 정규화 + Arrow Dataset 생성 |
| `validator.py` | Arrow ↔ JSONL 원본 비교 검증 |

---

## 9. Step 4: 데이터 증강 + 토크나이징

### 설계 원칙

**증강 적용 순서:** 변형 증강 → 출력 확정 → 삭제 증강 (입력에만)

이 순서 덕분에 출력 정답은 항상 완전한 full information을 유지하면서도, 변형 증강의 효과(예: 재배정된 RID)가 입출력 양쪽에 일관되게 반영된다.

### 증강 전략 목록

#### 표현형 변형 (Shuffle) — 입력만 영향

| 전략 | 대상 | 학습 효과 |
|------|------|---------|
| `ShuffleRID` | 방 ID 번호 재배정 | ID 번호가 아닌 구조와 관계에 집중 |
| `ShuffleVertexOrder` | 꼭지점 리스트 시작점 회전 | 시작점 무관하게 동일한 도형 인식 |
| `ShuffleRoomOrder` | 방 나열 순서 (입력만) | 입력 순서 무관하게 동일한 평면도 생성 |
| `ShuffleEdgeOrder` | 엣지 나열 순서 (입력만) | 엣지 순서 무관하게 연결 구조 인식 |
| `ShuffleSpatialOrder` | Spatial 나열 순서 (입력만) | Spatial 순서 무관하게 위치관계 인식 |
| `ReverseSpatialRelation` | Spatial 방향 반전 | 대칭 위치관계 인식 |

#### 기하학적 변형 (Transform) — 입력 + 출력 모두 반영

| 전략 | 내용 |
|------|------|
| `Translate` | 평행이동 (256×256 경계 보장) |
| `Flip` | 수평/수직/양방향 뒤집기 |
| `ScaleAspect` | 종횡비 변경 (x/y 독립 스케일, 기본 0.7~1.3) |
| `Zoom` | 균일 확대/축소 (기본 0.7~1.3) |

#### 노이즈 — 입력에만 적용

| 전략 | 내용 |
|------|------|
| `GaussianNoise` | σ=3.0px 가우시안 노이즈 (확률 30%) |

#### 삭제 (Drop) — 입력 조건에만 적용

| 전략 | 기본 확률 | 학습 효과 |
|------|---------|---------|
| `DropBlock` | 0.5 | 방 전체 삭제 → 빈 공간에 방 배치 능력 학습 |
| `DropType` | 0.2 | 방 타입만 삭제 → 다른 조건으로 타입 추론 |
| `DropCoords` | 0.2 | 방 좌표만 삭제 → 종류와 관계만으로 좌표 생성 |
| `DropEdge` | 0.5 | 엣지 전체 삭제 → 불완전한 연결관계에서 생성 |
| `DropEdgePair` | 0.2 | 특정 RID 쌍 엣지 삭제 |
| `DropEdgeDoor` | 0.2 | 문 정보만 삭제 (인접 관계는 유지) |
| `DropSpatial` | 0.8 | 개별 Spatial 관계 삭제 |
| `DropFrontDoor` | 0.5 | 현관문 전체 삭제 |
| `DropFrontDoorCoords` | 0.4 | 현관문 좌표만 삭제 |
| `DropRoomSummaryTotal` | 0.5 | `<TOTAL>` + 숫자 쌍 삭제 (샘플 단위) |
| `DropRoomSummaryType` | 0.6 | 개별 `<TYPE:t> <COUNT>` + 숫자 쌍 삭제 (타입별 독립) |

### 토크나이징

증강이 완료된 구조화 데이터를 토큰 ID 시퀀스로 변환한다.

```python
# 조건(입력) 토큰 시퀀스
condition_tokens = build_condition_tokens(augmented_sample, vocab)
# → [<INPUT>, <ROOM_SUMMARY>, ..., <END_INPUT>]

# 정답(출력) 토큰 시퀀스 — 항상 완전한 정보
output_tokens = build_output_tokens(augmented_sample, vocab)
# → [<OUTPUT>, <FRONT_DOOR>, ..., <ROOM>, <TYPE:xxx>, ..., <DOOR>, ..., <END_OUTPUT>]
```

### 주요 모듈

| 파일 | 역할 |
|------|------|
| `pipeline.py` | 증강 파이프라인 오케스트레이터 (`AugmentationPipeline`). 호출 후 `last_augmented_sample`(기하학적 변형 완료 row-oriented 샘플), `last_drop_state`, `last_applied_shuffles`를 속성으로 저장 |
| `strategies.py` | 15+ 증강 전략 순수 함수 구현 |
| `tokenizer.py` | `Vocab` 클래스, `build_condition_tokens()`, `build_output_tokens()` |
| `decoder.py` | `decode_tokens()`, `format_sample_report()` (디버깅용 역변환) |

---

## 10. Embedding Alignment: 새 토큰 Embedding 워밍업

### 목적

새로 추가된 커스텀 토큰 (~1000개)과 기존 Pretrained 파라미터 간의 **gradient 스케일 차이를 완화**하기 위한 준비 단계. SFT(Stage 1) 이전에 먼저 수행한다.

새 토큰의 embedding 벡터와 lm_head 행이 기존 Pretrained embedding space에 대략적으로 자리잡으면, 이후 SFT 훈련이 더 안정적으로 진행된다.

### 훈련 설정

| 설정 | 값 |
|------|---|
| Freeze 대상 | Transformer 레이어 전체 (attention, FFN, layernorm 등) |
| Train 대상 | `embed_tokens.weight[new_token_ids]` + `lm_head.weight[new_token_ids]` |
| Learning rate | 5e-4 (공격적) |
| Epoch | 2 (기본) |
| 양자화 | 4bit (NF4, Double Quant은 안) — embed_tokens/lm_head는 bfloat16 유지 |
| 혼합 정밀도 | bf16 AMP (forward/backward bf16, optimizer state fp32) |
| 분산 학습 | DDP 지원 (`distributed.nproc_per_node` 설정으로 활성화) |

### 파라미터 동결 전략 (PartialEmbedding / PartialLMHead)

전체 파라미터를 `requires_grad=False`로 동결한 뒤, `embed_tokens`와 `lm_head`를 커스텀 모듈로 교체하여 **새 토큰 567행만 `nn.Parameter`로 분리**한다.

```
수정된 모델
├── embed_tokens: PartialEmbedding
│   ├── base_embed:  nn.Embedding(152232, 3584)  [frozen buffer]
│   └── new_embed:   nn.Parameter(567, 3584)     [훈련 대상]
└── lm_head: PartialLMHead
    ├── base_lm_head: nn.Linear(3584, 152232)    [frozen buffer]
    └── new_lm_head:  nn.Parameter(567, 3584)    [훈련 대상]
```

**forward 시 동작:**
- `PartialEmbedding`: frozen base로 전체 조회 후, 새 토큰 위치에 `index_put`으로 `new_embed` 값 교체 (gradient 흐름 유지)
- `PartialLMHead`: frozen base로 전체 logits 계산 후, 새 토큰 위치에 `scatter`로 `new_lm_head` 재계산 값 교체

**Gradient Hook 방식 대비 이점:**

| | Gradient Hook (기존) | PartialEmbedding (현재) |
|---|---|---|
| optimizer state | ~8.8GB (전체 152232행) | ~16MB (567행만) |
| gradient 계산 | 전체 행 계산 후 마스킹 | 필요한 행만 계산 |

**저장 시:** 훈련이 완료된 `new_embed` / `new_lm_head` 가중치를 `partial_state.pt`로 저장한다. SFT 단계에서 HF Hub base model에 이 가중치를 `embed_tokens.weight.data[new_token_ids]` / `lm_head.weight.data[new_token_ids]`로 직접 주입하여 재사용한다.

PEFT 어댑터(LoRA)는 이 단계에서 사용하지 않는다.

### 계속 훈련 (Resume)

`resume.enabled=true`로 중단된 훈련을 재개할 수 있다.

**체크포인트 저장 구조 (`EmbedAlignTrainer._save_checkpoint` 오버라이드):**

```
data/models/{model.name}/checkpoints/embed_align/{run_name}/
└── checkpoint-{step}/
    ├── partial_state.pt      ← new_embed / new_lm_head 가중치만 별도 저장 (model.safetensors 없음)
    ├── optimizer.pt          ← AdamW state (new_embed, new_lm_head 두 파라미터만, ~16MB)
    └── trainer_state.json    ← step, epoch, best_model_checkpoint 등
```

> **model.safetensors를 저장하지 않는 이유:** Transformer 레이어는 항상 HuggingFace에서 새로 로드하므로 저장할 필요가 없다. 기존 방식(중간 체크포인트에서 `merge_and_restore()` 호출)은 `PartialEmbedding`의 `nn.Parameter` 객체를 소멸시키고 `_setup_partial_training()`이 새 객체를 생성하면서 optimizer의 Parameter 참조가 끊어지는 버그가 있었다. 이후 `optimizer.step()`이 소멸된 객체를 업데이트하려 해도 `grad=None`이므로 no-op이 되어 체크포인트 저장 이후의 훈련이 완전히 무효가 된다. `merge_and_restore`는 더 이상 어디서도 호출하지 않는다. 최종 저장도 `partial_state.pt` 방식으로 통일되어 있다.

체크포인트 저장 흐름:
1. `partial_state.pt` 저장 — 현재 훈련된 new_embed/new_lm_head 값 보존 (DDP 환경에서는 `is_world_process_zero()` 가드로 rank 0만 저장)
2. `self.save_model`을 일시 no-op으로 교체 후 `super()._save_checkpoint()` 호출 → optimizer + trainer_state만 저장 (model.safetensors 건너뜀)

**Resume 로드 흐름:**
- 항상 HuggingFace에서 기본 모델을 새로 로드 (`load_model_and_tokenizer`)
- `_setup_partial_training()` — PartialEmbedding/PartialLMHead 구조 적용
- `trainer.train(resume_from_checkpoint=...)` → `_load_from_checkpoint` 오버라이드 호출
  1. `partial_state.pt` 로드 — new_embed/new_lm_head를 훈련된 값으로 직접 복원 (`super()` 호출 없음 — 중간 체크포인트에 `model.safetensors` 없음)
- optimizer.pt에서 AdamW state / step 복원 (Trainer 내부 `_load_optimizer_and_scheduler`가 자동 처리)

**`_load_best_model` 오버라이드:**
`load_best_model_at_end=true` 시 표준 모델 재로드 대신 `partial_state.pt`에서 new_embed/new_lm_head만 직접 복사한다 (key mismatch 방지).

### 증강 설정 관리 (Hydra config group)

증강 파라미터는 훈련 단계마다 독립된 파일로 관리되며, Hydra **config group** 방식으로 각 파이프라인 yaml에 합성된다.

```
config/training/augmentation/
├── embed_align.yaml    ← Embedding Alignment용 (완료)
└── sft.yaml          ← SFT용 (완료, embed_align.yaml과 동일한 증강 전략)
```

`config/training/embed_align/pipeline.yaml`의 `defaults` 선언:
```yaml
defaults:
  - training/augmentation: embed_align   # cfg.augmentation으로 병합
  - _self_                             # pipeline.yaml 값이 최우선
```

- config 루트(`config/`)가 탐색 기준이므로 `training/augmentation: embed_align` →
  `config/training/augmentation/embed_align.yaml` 탐색
- `embed_align.yaml` 내부는 `augmentation:` 래퍼 없이 내용만 작성 (group 이름이 키를 자동 생성)
- SFT, GRPO 파이프라인도 동일한 패턴으로 증강 설정 재사용/오버라이드 가능

### 데이터 구성 및 Chat Template

`AugmentationPipeline`에서 `(condition_tokens, output_tokens)` 쌍을 생성하고, 이를 디코딩하여 Qwen2.5의 Chat Template으로 감싼다.

```
<|im_start|>system
You are a floor plan generator. Given room conditions, generate complete floorplan coordinates.<|im_end|>
<|im_start|>user
<INPUT>...<END_INPUT><|im_end|>
<|im_start|>assistant
<OUTPUT>...<END_OUTPUT><|im_end|>
```

- **Loss 마스킹:** system + user 턴은 `labels=-100`, assistant 턴만 loss 계산
- **Dynamic Padding:** 배치 내 최대 길이로 right-padding (pad 위치도 `labels=-100`)

### 주요 모듈

| 파일 | 역할 |
|------|------|
| `src/training/embed_align/model_loader.py` | 4bit 로드 + `prepare_model_for_kbit_training` + `PartialEmbedding`/`PartialLMHead` 교체 + `merge_and_restore` (기본 흐름에서 미사용) |
| `src/training/embed_align/dataset.py` | Arrow 로드 → 증강 → Chat Template 적용 → `{input_ids, labels, attention_mask}` |
| `src/training/embed_align/collator.py` | Dynamic padding + label 마스킹 |
| `src/training/embed_align/trainer.py` | `TrainingArguments` 구성 + `EmbedAlignTrainer` 빌드 (`_save_checkpoint`, `_load_from_checkpoint`, `_load_best_model` 오버라이드 포함) |
| `scripts/training/run_embed_align.py` | Hydra 진입점, seed 고정, Resume 분기, 훈련 후 `partial_state.pt` + optimizer 저장 |
| `config/training/embed_align/pipeline.yaml` | 모델, 양자화, 데이터, 훈련 하이퍼파라미터, resume 설정 |
| `config/training/augmentation/embed_align.yaml` | Embedding Alignment용 증강 파라미터 (Hydra config group, `cfg.augmentation`으로 병합) |
| `tests/training/embed_align/validate_resume.py` | Resume 체크포인트 검증 스크립트 (partial_state.pt 존재/형태/복원 확인) |
| `tests/training/embed_align/validate_save_and_load.py` | 저장/로드 후 optimizer 업데이트 정상 동작 검증 (체크포인트 저장 후 훈련이 계속 진행되는지 2-case 검증) |
| `src/utils/extract_partial_state.py` | 구 포맷 `model.safetensors`에서 커스텀 토큰 가중치만 추출하는 핵심 로직 |
| `scripts/utils/extract_partial_state.py` | 위 추출 로직의 argparse CLI 진입점 |
| `tests/utils/test_extract_partial_state.py` | 합성 단위 + 실제 파일 통합 검증 (2-Phase) |

### 레거시 체크포인트에서 partial_state.pt 추출

Embedding Alignment 저장 방식 변경 이전에는 훈련 완료 후 `merge_and_restore()` → `save_pretrained()`를 호출하여 frozen base 가중치와 훈련된 커스텀 토큰 가중치를 단일 `model.safetensors`로 병합 저장했다.

이 파일에서 새 토큰 행만 분리하면 현재 코드와 완전히 호환되는 `partial_state.pt`를 복원할 수 있다.

**추출 원리:**

```
model.safetensors
├── model.embed_tokens.weight  (new_vocab_size, hidden)  ← 전체 vocab 포함
├── lm_head.weight             (new_vocab_size, hidden)  ← 전체 vocab 포함
└── (transformer layers — quantized, 불필요)

↓ new_token_ids(= base_vocab_size 이상인 ID) 행만 슬라이싱

partial_state.pt
├── "new_embed"       (num_new, hidden)  = embed_tokens.weight[new_token_ids]
├── "new_lm_head"     (num_new, hidden)  = lm_head.weight[new_token_ids]
└── "new_token_ids"   list[int]
```

`new_token_ids`는 `vocab_extension.json`의 `base_vocab_size`를 기준으로 결정한다 (`token_id >= base_vocab_size`인 ID 정렬). 이 로직은 Embedding Alignment `model_loader.py`의 `_load_new_token_ids()`와 동일하다.

**safetensors 로드 방식:** 전체 파일을 메모리에 올리지 않고 `safetensors.safe_open()`으로 `embed_tokens`, `lm_head` 두 텐서만 읽는다. sharded 포맷(`model.safetensors.index.json`)도 지원한다.

### 체크포인트 및 출력

```
outputs/training/embed_align/
└── YYYY-MM-DD/HH-MM-SS/       # Hydra 실행 로그 + 설정 스냅샷

data/models/{model.name}/
└── checkpoints/embed_align/
    └── {run_name}/             # run_name별 독립 저장 (기본: floorplan-embed-align)
        ├── checkpoint-{step}/  # 에폭별 자동 저장 (save_total_limit 초과 시 오래된 것 삭제)
        │   ├── partial_state.pt    # new_embed / new_lm_head 가중치 (model.safetensors 없음)
        │   ├── optimizer.pt        # AdamW state (~16MB)
        │   └── trainer_state.json
        └── final/              # 훈련 run 최종 체크포인트 (중간 체크포인트와 동일 구조)
            ├── partial_state.pt    # new_embed / new_lm_head 가중치
            ├── optimizer.pt
            ├── scheduler.pt
            ├── trainer_state.json
            └── tokenizer.json 등
```

---

## 11. Step 5: LLM 학습

### 2-Stage Fine-tuning 전략

Embedding Alignment에서 워밍업된 커스텀 토큰 가중치(`partial_state.pt`)를 HF Hub base model에 주입한 뒤 2단계 fine-tuning을 수행한다. LLM 학습 시 QLoRA(Quantized LoRA)를 사용한다. 혼합 정밀도(bf16 AMP)를 적용한다.

### Stage 1: SFT (Supervised Fine-tuning) — 완료

#### 목적

Embedding Alignment 워밍업 이후, LoRA(Low-Rank Adaptation)를 통해 Transformer 전체 레이어를 fine-tuning하여 모델이 평면도 생성 태스크에 적응하도록 한다.

#### Embedding Alignment와의 차이점

| 항목 | Embedding Alignment | SFT |
|------|-----------|-----|
| 모델 로드 출처 | HF Hub | HF Hub (+ `partial_state.pt` 커스텀 토큰 가중치 주입) |
| 훈련 범위 | new_embed/lm_head 행 567개 | LoRA adapter (attention/MLP 전 레이어) |
| 특수 모듈 | PartialEmbedding / PartialLMHead | 불필요 (`partial_state.pt`로 직접 가중치 주입) |
| resize_token_embeddings | 필요 | 필요 (HF Hub 로드 후 커스텀 토큰 수만큼 확장) |
| 체크포인트 포맷 | `partial_state.pt` (커스텀) | `adapter_model.safetensors` (표준 PEFT) |
| Resume 처리 | 커스텀 `_load_from_checkpoint` | 표준 PEFT Resume |

#### LoRA (Low-Rank Adaptation)

`LoraConfig(use_dora=False)`로 설정. weight matrix를 low-rank 행렬 쌍(lora_A, lora_B)으로 분해하여 adapter 파라미터만 학습한다.

| 설정 | 값 |
|------|---|
| Train 대상 | LoRA adapter (lora_A, lora_B) |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| rank (r) | 32 |
| lora_alpha | 64 (실효 스케일 = alpha/r = 2.0) |
| lora_dropout | 0.05 |
| 학습률 | 2e-4 |
| Warmup ratio | 0.03 |
| Weight decay | 0.01 |
| 양자화 | 4bit NF4 |
| 분산 학습 | DDP 지원 (`distributed.nproc_per_node` 설정으로 활성화) |

**LoRA 파라미터 수 계산 (Qwen2.5-Coder-7B 기준):**
- 28 Transformer 레이어 × 7 target_modules × 2 텐서(lora_A, lora_B) = 392개 파라미터 텐서
- scalar 훈련 가능 파라미터: 약 41,760,768개 (~42M)

#### 모델 로드 흐름

```
1. AutoTokenizer.from_pretrained(tokenizer_dir)
   → data/models/{model.name}/tokenization/ 의 토크나이저 로드 (커스텀 토큰 포함)

2. AutoModelForCausalLM.from_pretrained(
       hub_id,               # HF Hub (예: Qwen/Qwen2.5-Coder-7B)
       quantization_config,  # 4bit NF4
       dtype=torch.bfloat16,
       # device_map="auto" 미사용: DDP와 호환되지 않음 (model parallelism vs data parallelism 충돌)
   )
   → resize_token_embeddings(len(tokenizer), mean_resizing=False)으로 vocab 확장
   → mean_resizing=False: transformers 5.x의 multivariate normal 초기화(GPU에 fp32 임시 텐서 ~2GB×2 생성)를 비활성화. partial_state.pt로 새 토큰 행을 곧바로 덮어쓰므로 초기화 방식 무관.

3. partial_state.pt 로드 → embed_tokens / lm_head의 new_token_ids 행에 훈련된 가중치 직접 주입
   → 4bit 양자화 대상이 아닌 bf16 레이어이므로 torch.no_grad()로 직접 index 접근 가능

4. prepare_model_for_kbit_training(model, ...)
   → gradient checkpointing 활성화

5. LoraConfig(..., use_dora=False)
6. get_peft_model(model, lora_config)
   → attention/MLP 레이어에 LoRA adapter 주입
   → base params (embed/lm_head 포함) 전체 freeze 자동 처리
```

#### merge_lora_and_save (deprecated — 기본 흐름에서 미사용)

LoRA adapter를 base model에 병합하여 standalone 표준 HuggingFace 형식으로 저장하는 유틸리티 함수.

기본 훈련 흐름(`run_sft.py`)에서는 더 이상 호출하지 않는다. adapter만 저장하는 방식(`PeftModel.save_pretrained`)으로 변경되었으며, 이 함수는 PEFT 의존성 없이 standalone 추론 모델이 필요하거나 다음 Stage에서 full model이 요구될 때 수동으로 호출하기 위해 유지한다.

`model.merge_and_unload()` 이후 `save_pretrained()` 호출 시, transformers 4.51+에서 `revert_weight_conversion()`이 NF4 역변환을 시도하다 `NotImplementedError`를 발생시키는 버그가 있다. 이를 `transformers.modeling_utils.revert_weight_conversion`을 일시적으로 no-op으로 패치하여 우회한다 (Embedding Alignment의 `validate_quantization_for_training` 패치와 동일한 방식).

#### 체크포인트 및 출력

```
outputs/training/sft/
└── YYYY-MM-DD/HH-MM-SS/       # Hydra 실행 로그 + 설정 스냅샷

data/models/{model.name}/checkpoints/sft/{run_name}/
├── checkpoint-{step}/
│   ├── adapter_model.safetensors  # LoRA adapter 가중치
│   ├── adapter_config.json        # use_dora: false 포함
│   ├── optimizer.pt               # AdamW state
│   └── trainer_state.json
└── final/                         # 훈련 run 최종 체크포인트 (중간 체크포인트와 동일 구조)
    ├── adapter_model.safetensors
    ├── adapter_config.json
    ├── optimizer.pt
    ├── scheduler.pt
    ├── trainer_state.json
    └── tokenizer.json 등
```

#### 주요 모듈

| 파일 | 역할 |
|------|------|
| `src/training/sft/model_loader.py` | HF Hub base model 로드 + `partial_state.pt` 커스텀 토큰 가중치 주입 + LoRA 적용. `load_base_model_with_partial_state()`, `build_lora_config()` 공개 API 제공 (RL에서 재사용) |
| `src/training/sft/trainer.py` | `TrainingArguments` + 표준 `Trainer` 빌드. `_parse_save_total_limit()`: OmegaConf가 YAML `null`을 문자열 `"null"`로 전달하는 케이스를 방어적으로 처리 |
| `scripts/training/run_sft.py` | Hydra 진입점, seed 고정, Resume 분기, 훈련 후 adapter + optimizer 저장 |
| `config/training/sft/pipeline.yaml` | LoRA, 학습률, model_dir 등 SFT 전체 설정 |
| `config/training/augmentation/sft.yaml` | SFT용 증강 파라미터 (embed_align.yaml과 동일) |
| `tests/training/sft/validate_sft.py` | 로드·LoRA구조·훈련·저장·Resume 통합 검증 |

#### DDP (Data Parallel) 지원

Embedding Alignment와 SFT 모두 DDP를 지원한다. `distributed.nproc_per_node` 값이 2 이상이면 `main(cfg)` 진입 직후 `os.execvp`로 torchrun 프로세스를 자동으로 띄운다.

**4bit 양자화 + DDP 호환성:**
- `device_map="auto"`(model parallelism, DDP와 충돌)는 제거됨
- 4bit 양자화(frozen 가중치)는 `requires_grad=False`이므로 DDP all-reduce 대상 제외 → 호환됨
- LoRA adapter(bf16, `requires_grad=True`)만 all-reduce됨

**Embedding Alignment DDP 주의사항:**
- `_save_checkpoint`: `is_world_process_zero()` 가드로 rank 0만 `partial_state.pt` 저장
- `_save_checkpoint` / `_load_from_checkpoint` / `_load_best_model`: DDP 래퍼(`DistributedDataParallel`) 내부 실제 모델에 `.module`으로 접근
- 최종 저장 시: `trainer.accelerator.unwrap_model(trainer.model)`로 언래핑 후 `partial_state.pt` / adapter 저장 (`is_main_process` 가드)

**환경 호환성 패치 (PyTorch 2.10 + cu128 + bitsandbytes 0.49 + WSL2):**

RL 단계 도입으로 vLLM 의존성에 끌려 PyTorch가 2.6(cu124) → 2.10(cu128)로, transformers가 4.x → 5.x로 자동 업그레이드되면서 SFT/Embedding Alignment에서 다음 회귀가 발생한다. `scripts/training/run_*.py` 진입부와 `model_loader.py` / `trainer.py`에 우회 패치를 적용했다.

| 증상 | 원인 | 패치 |
|------|------|------|
| DDP 초기화 `_verify_param_shape_across_processes`에서 `ncclUnhandledCudaError "out of memory"` | NCCL 2.27.5 + WSL2에서 P2P/SHM 통신 경로 회귀 버그. `/dev/shm` 크기와 무관하게 SHM 채널이 깨짐 | `run_*.py` 상단에서 `NCCL_P2P_DISABLE=1` / `NCCL_SHM_DISABLE=1` / `NCCL_IB_DISABLE=1` 환경변수 설정 → SOCKET 통신 강제. 단일 머신 2-GPU에서 throughput 손실 거의 없음 |
| 학습 step 도중 OOM (예: 19.45GB 할당 + 1.79GB unallocated reserved + 3.74GB 신규 요청 → fail) | PyTorch 2.10의 cudaMalloc 파편화 누적 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 환경변수 설정 (가상 주소 공간 점진 확장) |
| `resize_token_embeddings` 호출 시 GPU에 ~2GB 임시 텐서 생성 (embed_tokens + lm_head 합산 ~4GB) | transformers 5.x에서 `mean_resizing=True`(기본값)가 기존 embedding matrix를 fp32로 GPU 복사하여 multivariate normal 공분산 계산 | `model_loader.py`의 `resize_token_embeddings(..., mean_resizing=False)` + 직후 `torch.cuda.empty_cache()`. partial_state.pt로 새 토큰 행을 어차피 덮어쓰므로 초기화 방식 무관 |
| DDP 초기화 시 NF4 quantized buffer까지 NCCL로 rank간 브로드캐스트하여 추가 메모리 점유 | `broadcast_buffers=True`(기본값)가 frozen NF4 buffer까지 동기화 | `trainer.py`에 `ddp_broadcast_buffers=False` 추가 (각 rank가 동일 weight를 독립 로드하므로 동기화 불필요) |

> 환경변수는 `os.environ.setdefault()`로 박혀 외부에서 override 가능하다. 향후 NCCL/PyTorch 호환 버전이 안정화되면 제거 검토.

**실행:**
```bash
# DDP 2-GPU (config에 저장하거나 override로 지정)
uv run python scripts/training/run_embed_align.py distributed.nproc_per_node=2
uv run python scripts/training/run_sft.py distributed.nproc_per_node=2

# 단일 GPU (기본값, 동일 명령어)
uv run python scripts/training/run_embed_align.py
uv run python scripts/training/run_sft.py
```

---

### Stage 2: GRPO (GDPO) — 완료

#### 목적

RLVR(Reinforcement Learning from Verifiable Rewards) 기반 강화학습으로 규칙 기반 보상함수 11개를 적용한다. SFT로 평면도 생성 형식을 학습한 모델이 직교성·겹침 없음·outline 포함·연결성·입력 조건 일관성 등 기하학적 정확도를 스스로 높이도록 RL fine-tuning한다.

TRL의 `GRPOTrainer`를 서브클래싱한 `RLTrainer`가 GDPO(보상별 독립 정규화) + 토큰 수준 신용할당을 구현한다.

#### 멀티어댑터 모델 구조

```
HF Hub NF4 base + partial_state.pt 주입
    ↓
PeftModel.from_pretrained(sft_adapter_dir, adapter_name="sft", is_trainable=False)
    ↓
model.add_adapter("rl", lora_config)           # trainable
    ↓
model.base_model.set_adapter(["sft", "rl"])    # 두 어댑터 동시 활성화
SFT params: requires_grad=False (재동결)
RL params: requires_grad=True
```

- SFT adapter는 frozen base model 역할. 파라미터 갱신 없음.
- RL adapter만 gradient 흐름 (lora_A, lora_B).
- 두 adapter를 동시에 활성화하여 SFT 품질을 유지하면서 RL 정책을 학습.

#### GDPO 알고리즘

표준 GRPO와 달리 보상함수별로 독립 정규화(z-score)를 수행한 뒤 가중합으로 결합한다.

**1. 그룹별 보상 정규화 (프롬프트당 G개 completion 기준)**

$$A_k^{(i)} = \frac{r_k^{(i)} - \mathbb{E}[r_k]}{\sqrt{\text{Var}(r_k)} + \epsilon}$$

**2. 가중합 결합 (K=11개 보상)**

$$A^{(i)} = \sum_{k=1}^{K} w_k \cdot A_k^{(i)}$$

**3. 하드 게이트 (R_format=0이면 전체 보상 0)**

포맷 파싱 실패 시 geometry/connectivity 보상이 의미 없으므로 강제 0.

**4. 토큰 수준 신용할당 (적용 대상: format, orthogonality, no_overlap, room_in_outline, outline_in_room)**

$$a_t = A \cdot (1 - m_t) - |A| \cdot \lambda \cdot m_t$$

- $m_t$: 오류 토큰 마스크 (파싱 실패 위치, 직각 위반 꼭지점, 겹침 발생 방 토큰, outline 벗어난 방/front door 토큰)
- 정상 토큰: 어드밴티지 $A$ 그대로. 오류 토큰: 방향 페널티 추가.

**5. 배치 정규화 (시퀀스 대표값 기반)**

#### 11개 보상함수

| 이름 | 산출 방식 | 토큰 신용할당 | 가중치 | 하드 게이트 |
|------|---------|------------|--------|-----------|
| `R_format` | 파싱 성공 여부 이진값 | ✅ (오류 위치 마스킹) | 1.0 | ✅ (0이면 모두 0) |
| `R_count_total` | 방 전체 개수 일치 이진값 (drop_room_summary_total 시 채점 비활성) | ❌ | 0.5 | - |
| `R_count_type` | ROOM_SUMMARY에 노출된 타입별 개수 정확도 연속값 | ❌ | 1.0 | - |
| `R_orthogonality` | 꼭지점 직각 비율 | ✅ (위반 꼭지점 마스킹) | 1.5 | - |
| `R_no_overlap` | 겹침 없음 (Shapely) | ✅ (겹친 방 토큰 마스킹) | 2.0 | - |
| `R_room_in_outline` | 비-outline 방 + front door의 outline 내 포함 비율 평균 (Shapely). front door는 {cx,cy,w,h}에서 직사각형 4개 꼭짓점 구성 | ✅ (outline 벗어난 방·front door 토큰 마스킹) | 1.5 | - |
| `R_outline_in_room` | outline 꼭짓점이 방 내부에 포함되는지 (케이스 B 검출). 방 꼭짓점이 모두 outline 안이나 방 edge가 outline 오목부를 가로지르는 경우 처리 | ✅ (해당 outline 꼭짓점 토큰 마스킹) | 1.0 | - |
| `R_coverage` | outline 내 빈공간 보수값 $1 - \text{area}(O \setminus \bigcup R_i) / \text{area}(O)$ (Shapely `unary_union`) | ❌ | 1.5 | - |
| `R_connectivity` | 문 연결관계 (헝가리안 + 후보 기반 satisfiability) | ❌ | 1.0 | - |
| `R_spatial` | 8방위 공간관계 정확도 (헝가리안 + 후보 기반 satisfiability) | ❌ | 0.5 | - |
| `R_input_consistency` | 입력 좌표 명시 방(앵커+drop_type) 무게중심 일관성 (선형 거리 점수) | ❌ | 1.5 | - |

**`R_room_in_outline` 산출식:**

비-outline 방 집합 $\mathcal{R}$ 과 front door $f$ 에 대해 (front door 존재 시)

$$R_{\text{room\_in\_outline}} = \frac{1}{|\mathcal{R}| + \mathbf{1}[f \neq \emptyset]} \left( \sum_{r \in \mathcal{R}} \frac{\text{area}(r \cap \text{outline})}{\text{area}(r)} + \mathbf{1}[f \neq \emptyset] \cdot \frac{\text{area}(f \cap \text{outline})}{\text{area}(f)} \right)$$

- front door 폴리곤: 토큰 $\{cx, cy, w, h\}$ 에서 $(cx, cy)$를 left-top, $(cx+w, cy+h)$를 right-bottom으로 하는 직사각형 4개 꼭짓점 구성
- `containment_ratio < 1 - 10^{-4}` 인 방의 outline 밖 꼭짓점을 신용할당 오류로 마킹
- front door는 left-top $(cx, cy)$와 right-bottom $(cx+w, cy+h)$ 두 대표 꼭짓점을 검사하여 각각 $\{cx, cy\}$ 토큰 쌍, $\{w, h\}$ 토큰 쌍을 마킹
- outline 자체는 평가 대상에서 제외 (rooms[0]에서 분리)

**`R_outline_in_room` 산출식 (케이스 B 검출):**

$R_{\text{room\_in\_outline}}$ 은 방 꼭짓점이 outline 밖으로 나간 케이스 A만 검출한다. 케이스 B — 방 꼭짓점은 모두 outline 내부이지만 방의 edge가 outline의 오목부(concavity)를 가로지르는 경우 — 는 $R_{\text{room\_in\_outline}}$으로 검출되지 않는다. $R_{\text{outline\_in\_room}}$이 이를 담당한다.

outline의 각 꼭짓점 $v$ 를 순회하며, $v$ 가 비-outline 방들의 합집합 외부에 있으면 해당 꼭짓점의 좌표 토큰을 신용할당 오류로 마킹한다.

$$R_{\text{outline\_in\_room}} = \frac{1}{|V_{\text{outline}}|} \sum_{v \in V_{\text{outline}}} \mathbf{1}\bigl[v \in \bigcup_{r \in \mathcal{R}} r\bigr]$$

- 신용할당 ON: outline 꼭짓점 토큰 마킹

**`R_coverage` 산출식 (R_room_in_outline의 쌍대):**

outline 폴리곤 $O$ 와 비-outline 방 집합 $\mathcal{R}$ 에 대해 빈공간 비율의 보수값으로 정의한다.

$$R_{\text{coverage}} = 1 - \frac{\text{area}\bigl(O \setminus \bigcup_{r \in \mathcal{R}} r\bigr)}{\text{area}(O)}$$

- shapely `unary_union`으로 모든 방의 합집합을 한 번에 계산한 뒤 `outline.difference(union)`으로 빈공간 면적 산출
- $R_{\text{room\_in\_outline}}$(방 → outline)과 $R_{\text{coverage}}$(outline → 방들 합집합)는 서로 반대 방향 측정의 쌍대 관계. **두 보상 모두 1.0이어야** 비로소 "$O = \bigsqcup_i r_i$"라는 평면도 본질 제약이 강제되며, 단독 사용 시 reward hacking 여지가 남는다 (예: 작은 방 1~2개로 outline 일부만 채우는 경우).
- **신용할당 OFF (sequence-level only):** 빈공간 발생 책임 소재가 본질적으로 모호하다(좌표 잘못 vs 방 개수 부족). 또한 입력 좌표에 노이즈 증강이 들어가 있어 "어느 좌표가 정답인지" 자체가 모호하므로 토큰 단위 페널티는 잘못된 시그널을 줄 수 있다. GDPO의 G개 completion 간 z-score 정규화가 sequence-level 시그널만으로도 충분한 상대적 학습 신호를 제공한다.

**`R_input_consistency` 산출식:**

평가 대상 방 두 종류:
- **앵커 방** $\mathcal{A}$: type+coords 모두 visible (drop_block / drop_type / drop_coords 미적용). 타입 그룹별 헝가리안으로 결정 매핑.
- **drop_type 방** $\mathcal{D}$: coords visible + type="" (drop_type 적용). 앵커 매핑 후 잔여 출력 방 대상으로 타입 무관 헝가리안으로 매핑.

$$s_r = \max\!\left(0, 1 - \frac{d_r}{\tau}\right) \quad,\quad R_{\text{input\_consistency}} = \frac{1}{|\mathcal{A}| + |\mathcal{D}|} \sum_{r \in \mathcal{A} \cup \mathcal{D}} s_r$$

- $\tau$ 기본값: 15px (좌표 노이즈 3σ=9px + 모델 오차 마진. transform 증강은 입력/출력에 동일 적용되어 상대 오차 없음)
- 앵커와 drop_type 방이 모두 없으면 채점 비활성 (1.0 반환)
- 미매칭 앵커는 점수 0; 앵커는 타입 불일치도 점수 0
- drop_type 방은 타입 검증 없이 거리 점수만 산출 (타입 정보 자체가 없으므로)
- 신용할당 OFF: 출력에 RID가 없어 특정 토큰을 오류로 마킹하기 어려운 구조이며, 거리 기반 연속 점수가 자체적으로 그래디언트 신호를 제공한다.

#### 모델 시점 metadata + 후보 기반 satisfiability 채점

연결성/공간관계/입력 일관성 보상은 입력 프롬프트의 **모델 시점**(증강 후 + drop 반영) metadata를 기준으로 채점된다. 이 설계는 다음 두 가지 부당 채점을 동시에 해소한다:

1. **변형 증강에 따른 좌표계 불일치:** 이전에는 metadata가 원본 좌표를, 출력은 변형 후 좌표를 가져 무게중심 비교가 무의미했다.
2. **삭제 증강된 정보로 모델을 채점:** 이전에는 metadata에 full GT가 있어 drop된 edge/spatial까지 강제 채점되었다.

**metadata 구성 규칙 (모델 시점):**

| 입력 정보 손실 | metadata 반영 |
|---------------|--------------|
| `drop_block(rid)` | metadata.rooms에서 해당 방 제거 |
| `drop_type(rid)` | 해당 방의 `type=""`로 마스킹 (좌표는 유지) |
| `drop_coords(rid)` | 해당 방의 `coords=[]`로 마스킹 (type은 유지) |
| `drop_edge(idx)` | metadata.edges에서 제거 |
| `drop_pair(idx, mode)` | edge.pair를 `[kept_rid]` 또는 `[]`로 마스킹 |
| `drop_door(idx, mode)` | door 정보 부분 마스킹 (position/orientation/all) |
| `drop_spatial(idx)` | metadata.spatial에서 제거 |
| `drop_front_door[_coords]` | front_door 또는 그 좌표만 None |
| `drop_room_summary_total` | `total_rooms=None` (count_total 채점 비활성 신호) |
| `drop_room_summary_types(t)` | type_counts에서 해당 타입 제거 |

**의도적 비대칭:** `metadata.total_rooms` ≠ `len(metadata.rooms)`. 전자는 ROOM_SUMMARY로 노출된 GT 카운트(drop_block 방 포함), 후자는 모델이 본 visible 방만 (drop_block 제외). count 보상은 GT 카운트로 채점, 매칭은 visible 방에 한해 수행.

**2-tier 매칭 (connectivity / spatial):**

입력 RID는 다음 4가지로 분류되어 각각 다른 매칭 전략으로 처리된다:

| 분류 | 조건 | 후보 출력 방 |
|------|-----|------------|
| 앵커 | type+coords 모두 visible | 헝가리안으로 결정 매핑된 단일 인덱스 |
| 자유(drop_coords) | type만 visible | 같은 type 출력 방 모두 |
| 자유(drop_type) | coords만 visible | 무게중심 거리 ≤ 30px 출력 방 |
| drop_block | metadata에 없음 | 빈 리스트 (호출 자체 발생 안 함) |

각 제약(edge / spatial)은 두 RID의 후보 집합 모든 조합 $(a, b)$ 에 대해 만족 여부를 검사하고, **하나라도 만족하면 통과**로 채점한다 (satisfiability-based). 모델이 식별 불가능한 정보(예: drop_coords된 같은 type 방의 RID 구분)에 대해 부당 페널티가 부과되지 않는다.

`R_input_consistency`는 앵커 방(결정 매핑)과 drop_type 방(타입 무관 헝가리안, 잔여 출력 방 대상)만 해당하므로 satisfiability 후보 확장은 사용하지 않는다.

#### vLLM Colocate 통합 (현재 비활성, NF4 환경 부적합)

**현 상태:** `rl.use_vllm=false`가 기본값. 우리 환경(NF4 base + LoRA + 200~300 토큰 시퀀스)에서는 vLLM colocate가 수치적으로 발산하므로 HF generate를 사용한다. vLLM은 bf16 base + 긴 시퀀스(≥1024) + 멀티 GPU 환경에서만 의미 있다.

**왜 NF4에서 발산하는가 (검증 결과 2026-05-05):**

[peft/tuners/lora/bnb.py:393-422](../.venv/lib/python3.11/site-packages/peft/tuners/lora/bnb.py#L393-L422)의 `Linear4bit` LoRA merge 구현은 다음 round-trip을 수행한다:

```
merge:   NF4(W) → bf16 dequant → bf16 + B·A → bf16 → NF4 재양자화 → base_layer.weight 덮어쓰기
unmerge: NF4(W_merged) → bf16 dequant → bf16 - B·A → bf16 → NF4 재양자화 → 원본 복원 시도
```

PEFT 자체가 line 397, 446에서 *"may get different generations due to rounding errors"* 라고 경고하는 부분이다. NF4는 lookup table 16개 값으로 양자화하므로 4bit 재양자화 손실이 매우 크다. 매 step `sync_weights()`에서 두 번씩 round-trip이 일어나 (a) vLLM이 보는 가중치와 훈련 forward가 사용하는 가중치가 어긋나고 (b) 훈련 base 자체가 step마다 부식된다.

**관측된 발산 패턴 (단일 GPU, num_generations=2, max_steps=20 기준):**

| step | sampling_logp_difference/mean | importance_sampling_ratio/mean | KL |
|------|------------------------------|-------------------------------|----|
| 1 | 0.22 | 2.86e-08 | 0.027 (정상) |
| 3 | 0.47 | ~e-10 | 0.33 |
| 5 | — | 0 | 2.48e+7 |
| 10 | — | 0 | 4.04e+16 |
| 15+ | 1.36~2.52 | 0 | inf/nan |

→ IS ratio 붕괴(보정 한계 초과) + K3 KL 추정자의 `exp` 항이 outlier 토큰에서 폭주 → loss/grad inf/nan.

**같은 조건의 HF generate 모드는 안정 (train_loss=0.019 정상 종료).** rollout 생성 모델 = logp 계산 모델 = 훈련 모델이라 round-trip 자체가 발생하지 않는다. 시간도 단일 GPU 200~300 토큰 시퀀스에서는 HF가 약 1.6× 빠르다(454.2s vs 280.6s, 20 step 기준) — vLLM `merge_adapter` + `sync_weights` 오버헤드가 throughput 이점을 상쇄.

**vLLM colocate가 의미 있는 환경:**

| 조건 | 이유 |
|------|------|
| bf16 base + LoRA | bf16 merge는 lossless. round-trip 손실 없음. |
| fp16/bf16 full FT | merge 자체가 없음 (LoRA가 없으니). |
| 긴 시퀀스 (≥1024) + 멀티 GPU | vLLM PagedAttention 처리량이 sync 오버헤드를 압도. |

**(참고) 원래 구상했던 아키텍처:**

```
GPU 0 (rank 0)                        GPU 1 (rank 1)
┌─────────────────────┐               ┌─────────────────────┐
│ 훈련 모델 (NF4+LoRA)  │               │ 훈련 모델 (NF4+LoRA)  │
│ vLLM 인스턴스 (NF4)   │               │ vLLM 인스턴스 (NF4)   │
│   → local batch      │               │   → local batch      │
│      rollout 생성    │               │      rollout 생성    │
└─────────────────────┘               └─────────────────────┘
     ↕ DDP gradient sync
```

`rl.use_vllm=true`로 활성화하면 위 구조로 동작은 가능하지만 NF4 환경에서는 학습이 발산한다. bf16 base로 전환할 때만 유효한 옵션.

#### 구현 노트 (핵심 버그 이력)

1. **`PeftModel.name_or_path` 우회:** TRL이 `model.name_or_path`로 vLLM을 초기화하는데, PeftModel에서 `nn.Module.__getattribute__`가 instance `__dict__`를 우선하여 Hub ID를 반환한다. `model.config.name_or_path` 설정만으로는 반영 안 됨. `model.base_model.model.name_or_path = vllm_base_dir`도 함께 설정해야 함.

2. **vLLM `stop_token_ids` vs HF `eos_token_id`:** vLLM `SamplingParams`는 `stop_token_ids` 키를 사용. 단, 151643(`<|endoftext|>`)은 `vllm_base/config.json`의 `eos_token_id`로 자동 처리되므로 `stop_token_ids`에 포함 금지 — 포함 시 vLLM이 출력에서 해당 토큰을 제거하여 TRL의 `clipped_ratio=1` 오진단 발생. 커스텀 종료 토큰(152214)만 등록.

3. **vllm_base NF4 역양자화:** `save_pretrained()`는 bitsandbytes NF4 포맷으로 저장 → vLLM이 로드 불가. `prepare_vllm_base_model()`에서 Params4bit → bf16 역양자화 후 safetensors로 직접 저장. `.base_layer.` 이름 제거, `lora_` 파라미터 제외.

4. **metadata 모델 시점 재구성 (drop 반영):** 이전에는 `_extract_metadata()`가 증강 전 원본 sample을 받아 full GT를 metadata에 담았다. 결과적으로 (a) flip/scale/translate/zoom 증강 시 metadata 좌표계와 출력 좌표계 불일치로 헝가리안 무게중심 매칭이 무의미해졌고, (b) drop_edge / drop_spatial 등 모델이 보지 못한 제약까지 강제 채점되었다. 현재는 `pipeline()` 호출 후 `last_augmented_sample`(변형 적용된 sample)과 `last_drop_state`를 참조하여 drop을 데이터에 직접 반영한 모델 시점 metadata를 구성한다. drop_state 필드는 metadata에 노출하지 않으며, reward 함수는 metadata만 보고 자연스럽게 visible 정보로만 채점된다.

5. **count 보상의 None / 부분 drop 처리:** `total_rooms=None` (drop_room_summary_total) 시 `R_count_total`은 1.0 반환 (채점 비활성). `R_count_type`은 `expected_counts.keys()`만 순회하여 drop된 타입(drop_room_summary_types)에 대한 부당 페널티 회피 — 이전 구현은 출력 타입까지 합집합으로 순회해 drop된 타입을 모델이 출력하면 0점을 부여했었다.

6. **헝가리안 매칭의 자유 방 제외:** drop_coords 방은 `coords=[]`로 metadata에 남으므로 무게중심 계산 시 `(0, 0)`이 되어 잘못된 매칭이 발생했다. 또한 drop_type 방은 `type=""`로 격리되어 자동 미매칭이지만 비명시적이었다. 현재는 `_hungarian_match()`가 두 경우 모두 매칭 후보에서 명시적으로 제외한다. 자유 방의 RID는 connectivity / spatial에서 후보 확장(`_get_candidate_output_indices()`)으로 satisfiability 채점된다.

7. **`R_input_consistency` drop_type 방 확장:** 초기 구현은 앵커(type+coords 모두 visible)에만 채점했다. drop_type 방(coords visible, type="")도 좌표 힌트가 있으므로 "해당 위치에 방을 그렸는가"를 평가할 수 있다. 타입 그룹 분류가 불가하므로 앵커 헝가리안 완료 후 남은 출력 방들을 대상으로 타입 무관 헝가리안(`_match_drop_type_rooms()`)을 별도로 수행한다. 동시에 τ를 30px → 15px로 축소했다 (transform 증강은 입력/출력 좌표에 동일 적용되어 상대 오차가 없으므로 변형 잔차가 실질적 오차 요인이 아님을 확인; 실질 오차는 노이즈 3σ=9px + 모델 오차 마진).

8. **`outline_in_room` 보상이 `RLTrainer.reward_order`에서 누락:** [`src/training/rl/trainer.py`](../src/training/rl/trainer.py)의 `_build_reward_funcs()`가 순회하는 `reward_order` 리스트에 `outline_in_room`이 빠져 있어 callable이 만들어지지 않았다. 결과적으로 `compute_all_rewards()`는 보상을 계산하지만 TRL `rewards_per_func` 행렬(K)에 반영되지 않고, `compute_token_advantages()` 가중합 루프도 `_reward_names`만 순회하므로 신용할당 mask까지 무시되어 새 보상이 학습 신호에서 weight=0으로 적용됐다. 11개 보상 모두 활성화되도록 `room_in_outline` 다음에 `outline_in_room`을 추가했다. 실제 1 micro-step에서 `_cached_rewards_per_func.shape == (B_total, 11)`로 확장되며 W&B에 `rewards/reward_outline_in_room/mean` 메트릭이 신규 등장한다.

9. **`R_format`이 outline 부재/잘못된 위치를 통과시키는 결함:** parser는 `<ROOM>` 토큰을 만나는 순서대로 파싱할 뿐 첫 번째 방이 outline이라는 보장을 하지 않는다. 기존 `R_format`은 `len(parsed.rooms) < 2`만 검사했기 때문에 모델이 outline 없이 일반 방 두 개만 출력해도 1.0이 부여되어 hard gate가 우회됐다. 현재는 `parsed.rooms[0].room_type == "outline"`이고 `rooms[1:]`에 outline이 추가로 등장하지 않을 때만 1.0을 반환한다. 위반 시 모든 방 블록 토큰을 error로 마킹하여 신용할당까지 작동한다.

10. **parser의 Y 토큰 누락 시 강제 진행 (무한 루프 가드):** `_parse_single_room()`은 `<X:n> <Y:m>` 쌍을 순서대로 파싱한다. X가 성공한 직후 Y가 아닌 토큰이 오면 이전 구현은 `continue`만 하여 `self.pos`가 이동하지 않았고, 그 자리가 또 X 토큰이면 다음 iter에서 같은 분기에 빠져 X 한 개씩 미아가 되며 진행하는 비정상 동작이 발생할 수 있었다. 현재는 `y is None` 분기에서 `self.pos += 1`로 한 토큰 강제 진행한다.

11. **parser EOS 비교의 falsy 단락 버그:** `_parse_doors()`의 EOS 검사가 `self.ids[self.pos] == (self.vocab.eos_token_id or -9999)` 형태였다. Python `or`는 첫 truthy 값을 반환하므로 `eos_token_id`가 정수 0인 LLM에서는 `0 or -9999`가 `-9999`로 평가되어 EOS 비교가 영구적으로 false가 되는 잠재 결함이 있었다. `is not None` 명시 체크로 교체했다 (Qwen2.5는 EOS=151643이라 영향 없으나 다른 LLM에서 회귀 방지).

12. **`R_connectivity`의 door 내부 위치 false positive 제거:** 이전 `_has_door_between()`은 자체 작성한 `_min_distance_to_polygon`으로 door 중심에서 폴리곤 경계까지의 최소 거리를 계산했는데, 점이 폴리곤 **내부**에 있어도 경계까지 거리만 반환하므로 door가 한 방 내부 깊숙이 있어도 dist가 작게 나와 통과되는 문제가 있었다. door는 두 방의 공유 벽 근처에 있어야 한다는 의도와 어긋남. shapely `polygon.boundary.distance(point)`로 교체하여 점이 내부에 있을수록 boundary까지 거리가 커져 자동으로 걸러진다. 미사용 헬퍼(`_min_distance_to_polygon`, `_point_to_segment_distance`)는 제거.

13. **`R_spatial` 8방위 분기의 부등호 일관성:** `_vector_to_direction()`의 분기가 대부분 `low <= x < high` 형태인데 left 분기만 `angle >= 157.5 or angle < -157.5`로 `>=`를 사용해 정확히 ±157.5° 입력에서 분류가 비대칭이었다. atan2 결과를 `[0, 360)`으로 정규화한 뒤 모든 분기를 반열림 구간(`<`)으로 통일하고, wrap-around은 right만 (0°/360° 경계)에서 처리한다.

14. **검증 도구 분리 (verification/):** 위 결함들을 의도 격리 단위로 검출/방어하기 위해 [`tests/training/rl/verification/`](../tests/training/rl/verification/) 아래에 challenging fixture 기반 verifier 17개를 신설했다. 기존 `validate_rl.py`(통합 4-phase)는 그대로 두고 보완. Group 1(전처리), Group 2(보상별 11개), Group 3(어드밴티지·손실 mock + 실제 모델 1 micro-step) 구성. 모든 verifier가 회귀 가드로 작동하여 위 13개 항목의 의도가 미래 코드 변경 시에도 유지됨을 보장한다.

#### 체크포인트 및 출력

```
data/models/{model.name}/checkpoints/rl/{run_name}/
├── checkpoint-{step}/
│   ├── adapter_model.safetensors  # RL LoRA adapter 가중치
│   ├── adapter_config.json        # use_dora: false
│   ├── optimizer.pt
│   └── trainer_state.json
└── final/
    ├── adapter_model.safetensors
    ├── adapter_config.json
    ├── optimizer.pt
    ├── scheduler.pt
    └── trainer_state.json
```

#### 주요 모듈

| 파일 | 역할 |
|------|------|
| `src/training/rl/model_loader.py` | HF Hub NF4 + partial_state.pt + SFT(frozen)+RL(trainable) 멀티어댑터 구성 + vllm_base bf16 저장 |
| `src/training/rl/trainer.py` | `RLTrainer` (GRPOTrainer 서브클래스) — GDPO + 토큰 신용할당 |
| `src/training/rl/advantage.py` | `gdpo_group_normalize()`, `compute_token_advantages()`, `_batch_normalize()` |
| `src/training/rl/dataset.py` | `RLPromptDataset` — 프롬프트 + 모델 시점 metadata 로드 (drop 데이터에 반영, 출력 label 없음) |
| `src/training/rl/rewards/__init__.py` | `compute_all_rewards()` 공개 API |
| `src/training/rl/rewards/*.py` | 11개 규칙 기반 보상함수 (parser, format, geometry, room_in_outline, outline_in_room, coverage, connectivity, count, spatial, input_consistency, credit_assignment) |
| `src/training/rl/rewards/parser.py` | 생성 토큰 파싱. `ParsedFloorplan.front_door_token_indices` ([cx_idx, cy_idx, w_idx, h_idx]) 포함 |
| `src/training/rl/rewards/room_in_outline_reward.py` | 비-outline 방 + front door의 outline 포함 검증 (케이스 A). front door는 {cx,cy,w,h}로 직사각형 구성 |
| `src/training/rl/rewards/outline_in_room_reward.py` | outline 꼭짓점이 방 내부 포함 여부 검증 (케이스 B) |
| `src/training/rl/rewards/connectivity_reward.py` | 헝가리안 매칭(앵커) + `_get_candidate_output_indices()` 후보 확장 + satisfiability 기반 채점 |
| `src/training/rl/rewards/spatial_reward.py` | 동일 후보 헬퍼 재사용 + 8방위 satisfiability |
| `src/training/rl/rewards/coverage_reward.py` | shapely `unary_union` + `difference`로 outline 내 빈공간 비율 산출 (sequence-level) |
| `src/training/rl/rewards/input_consistency_reward.py` | 좌표 명시 방(앵커+drop_type) 무게중심 선형 거리 점수 (threshold=15px 기본) |
| `scripts/training/run_rl.py` | Hydra 진입점, seed 고정, DDP 자동 전환, vllm_base 준비 |
| `config/training/rl/pipeline.yaml` | GDPO, 보상함수, vLLM colocate, DDP 전체 설정 |
| `tests/training/rl/validate_rl.py` | 4단계 통합 검증 (파일 존재·어댑터 구조·훈련 갱신·보상+생성) |
| `tests/training/rl/verification/_common.py` | 토큰 fixture 빌더(의도된 violation 지원), metadata/reward_cfg 빌더, assert 헬퍼 |
| `tests/training/rl/verification/group1_preprocessing/` | 변형 후 metadata 좌표 추적 + 8가지 drop 마스킹 격리 검증 |
| `tests/training/rl/verification/group2_rewards/` | 11개 보상함수 의도 격리 검증 (각 보상별 challenging 엣지케이스 + 회귀 가드) |
| `tests/training/rl/verification/group3_advantage/` | GDPO·token credit·batch_norm mock 검증 + 실제 모델 1 micro-step E2E |
| `tests/training/rl/verification/run_all.py` | verification 일괄 실행 오케스트레이터 (`--skip-microstep`, `--only group2` 지원) |
| `tests/training/rl/verification/findings.md` | 트랙 A(스크립트 실행) + 트랙 B(직접 코드 정독) 발견 사항 통합 보고서 |

### 학습 데이터 구성 (공통)

- **입력:** condition_tokens (삭제 증강이 적용된 부분 정보)
- **출력:** output_tokens (모든 방 + 모든 Edge의 완전한 정보)
- **Loss 마스킹:** 입력 토큰 구간은 loss 무시 (ignore_index=-100), 출력 토큰 구간만 학습

---

## 12. Step 6: 추론 및 시각화

### 추론 흐름

```
1. 입력 소스에서 평면도 샘플 로드 (JSONL / Arrow / txt_dir)
2. AugmentationPipeline으로 condition_tokens 생성 (훈련과 동일한 증강)
3. condition_tokens → Chat Template 적용 → input_ids
4. model.generate()로 output 토큰 시퀀스 생성
5. 생성된 토큰 → output_parser.py로 구조화 딕셔너리 역변환
6. result_saver.py로 JSON + 텍스트 토큰 + 이미지 저장
```

### 모델 로드 모드

| 모드 | 방식 | 용도 |
|------|------|------|
| `adapters` (권장) | HF Hub NF4 + `partial_state.pt` 주입 + PEFT named adapter 스태킹 | adapter 파일만으로 추론 |
| `merged` | 사전 병합된 standalone bf16 full model 직접 로드 | `merge_model.py` 유틸로 사전 생성 필요 |

**adapters 모드 로드 흐름:**

```
1. AutoTokenizer.from_pretrained(tokenizer_dir)
2. AutoModelForCausalLM.from_pretrained(hub_id, quantization_config=NF4, dtype=bfloat16)
   → resize_token_embeddings(len(tokenizer))
3. partial_state.pt 로드 → embed_tokens/lm_head의 new_token_ids 행에 직접 주입
   (embed_tokens/lm_head는 NF4 양자화 대상이 아닌 bf16 레이어)
4. PeftModel.from_pretrained(model, adapter_path, adapter_name=name)  # 첫 번째 adapter
5. model.load_adapter(adapter_path, adapter_name=name)                # 이후 adapter들
   (adapter_config.json 자동 파싱 — LoRA/DoRA 투명하게 처리)
6. float32 파라미터 bf16 일괄 캐스팅 (PEFT가 attention bias를 float32로 유지하는 문제 대응)
```

> **PEFT named adapter 방식:** 중간 adapter를 `merge_and_unload()` 없이 모두 named adapter로 독립 유지한다. `adapter_config.json`을 읽어 LoRA/DoRA 구조를 자동 복원하므로 코드 변경 없이 두 방식을 모두 지원한다.

### 입력 소스

| 모드 | 설명 | 포맷 변환 |
|------|------|---------|
| `jsonl_file` | 단일 JSONL 파일 | `_jsonl_to_columnar()` → Arrow columnar로 변환 후 AugmentationPipeline 전달 |
| `jsonl_dir` | JSONL 디렉토리 전체 | 동일 |
| `arrow` | HuggingFace Arrow 데이터셋 특정 split | 변환 없이 직접 전달 |
| `txt_dir` | 사전 증강된 토큰 텍스트 파일 (파일 1개=입력 1개) | `parse_input_tokens()`로 구조화 dict 생성, 증강 미적용 |

> **JSONL ↔ Arrow 포맷 차이:** JSONL의 `rooms`는 list-of-dicts이지만 `AugmentationPipeline` 내부의 `to_row_oriented()`는 Arrow columnar 포맷(dict-of-lists)만 처리한다. `_jsonl_to_columnar()`가 추론 코드 내부에서 변환을 수행하며 훈련 코드(`src/training/`)는 수정하지 않는다.

### 생성 설정

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `max_new_tokens` | `2048` | 최대 생성 토큰 수 |
| `do_sample` | `true` | 샘플링 여부 (false=greedy) |
| `temperature` | `1.0` | 샘플링 온도 |
| `top_p` | `0.95` | nucleus sampling |
| `num_beams` | `1` | beam search 너비 (1=greedy/sampling) |
| `repetition_penalty` | `1.0` | 반복 억제 |
| `num_outputs` | `2` | 동일 조건에 대해 생성할 출력 수 |

**EOS 처리:** `<|im_end|>` + `<|endoftext|>` + `<END_OUTPUT>` 세 토큰을 모두 EOS로 등록한다. Qwen2.5 Chat Template의 assistant 턴 종료 토큰(`<|im_end|>`)과 커스텀 평면도 종료 토큰(`<END_OUTPUT>`)이 다르기 때문이다.

### 추론 성능 (Qwen2.5-Coder-7B, NF4, 단일 GPU)

| 구성 | 생성 속도 | 비고 |
|------|---------|------|
| Embedding Alignment base (adapter 없음) | ~30 tok/s | NF4 base만 사용 |
| SFT DoRA adapter | ~3.5 tok/s | DoRA의 컬럼-노름 재계산 오버헤드 (~8.6× 느림) |
| SFT LoRA adapter (예상) | ~30 tok/s | LoRA는 forward 중 행렬 추가 연산만 발생 |

> **DoRA 속도 저하 원인:** DoRA는 forward 패스마다 적응된 전체 가중치 행렬 `(W + lora_B @ lora_A × scale)`을 구체화하고 컬럼 노름을 계산한다. LoRA에 비해 추론 비용이 크게 증가한다. PEFT가 `adapter_config.json`을 읽어 DoRA/LoRA를 투명하게 처리하므로 코드 레벨에서는 차이가 없다.

### 결과 저장 구조

Hydra `run.dir`이 날짜/시간 경로로 설정되어 있어 Hydra 로그·설정 스냅샷과 추론 결과가 동일 폴더에 저장된다.

```
outputs/inference/{model.name}/{training_stage}/{YYYY-MM-DD}/{HH-MM-SS}/
├── .hydra/             # Hydra 설정 스냅샷 (config.yaml, overrides.yaml 등)
├── run_inference.log   # 실행 로그
└── {plan_id}/
    ├── input/
    │   ├── tokens.txt          # 증강이 적용된 조건 토큰 텍스트
    │   ├── condition.json      # 조건 구조화 JSON
    │   └── floorplan.png       # 입력 조건 시각화 (기하학적 변형 + drop된 요소 모두 반영)
    ├── output/                 # num_outputs=1
    │   ├── tokens.txt          # 생성 토큰 텍스트
    │   ├── floorplan.json      # 역변환된 평면도 JSON
    │   └── floorplan.png       # 생성 결과 시각화
    └── meta.json               # plan_id, 토큰 수, 소요 시간, 파싱 성공 여부
```

> `num_outputs>1`이면 `output_0/`, `output_1/`, … 형태로 인덱스별 저장.

### 주요 모듈

| 파일 | 역할 |
|------|------|
| `src/inference/model_loader.py` | adapters/merged 모드 분기, NF4 + partial_state.pt 주입, PEFT named adapter 스태킹 |
| `src/inference/condition_builder.py` | 입력 소스별 샘플 로드, `_jsonl_to_columnar()` 변환, AugmentationPipeline 적용. 파이프라인 호출 후 `pipeline.last_augmented_sample`(기하학적 변형 완료 샘플)을 시각화용으로 사용 |
| `src/inference/generator.py` | Chat Template 구성, `model.generate()` 호출, EOS 후처리 |
| `src/inference/output_parser.py` | 생성 토큰 ID → 구조화 평면도 딕셔너리 역변환 |
| `src/inference/result_saver.py` | JSON / 토큰 텍스트 / PNG 이미지 저장, DropState 기반 입력 시각화 필터링 |
| `scripts/inference/run_inference.py` | Hydra 진입점, 배치 추론, seed 고정 |
| `config/inference/pipeline.yaml` | 모델 로드 모드, 입력 소스, 생성 파라미터, 출력 설정 |
| `tests/inference/validate_inference.py` | import·모델 로드·토큰 생성·파싱 통합 검증 |
